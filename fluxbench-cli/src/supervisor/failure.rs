//! Failure classification: exit status to failure kind, and failure messages.

use fluxbench_ipc::FailureKind;
use std::process::ExitStatus;
use std::time::Duration;

/// Name of a signal number: `SIGSEGV` for known signals, else `signal N`.
#[cfg(unix)]
pub(super) fn signal_name(signal: i32) -> String {
    let name = match signal {
        libc::SIGSEGV => "SIGSEGV",
        libc::SIGABRT => "SIGABRT",
        libc::SIGKILL => "SIGKILL",
        libc::SIGBUS => "SIGBUS",
        libc::SIGILL => "SIGILL",
        libc::SIGFPE => "SIGFPE",
        libc::SIGTERM => "SIGTERM",
        libc::SIGINT => "SIGINT",
        other => return format!("signal {other}"),
    };
    name.to_string()
}

/// Failure kind and a short cause for how a worker ended.
///
/// Unix reads the terminating signal. Other platforms read the exit code only.
pub(super) fn classify_exit(status: ExitStatus) -> (FailureKind, String) {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            let name = signal_name(signal);
            let cause = format!("worker killed by {name}");
            return (FailureKind::Signal(name), cause);
        }
    }
    match status.code() {
        Some(code) => (
            FailureKind::Crashed,
            format!("worker exited with code {code}"),
        ),
        None => (
            FailureKind::Crashed,
            "worker exited with unknown status".to_string(),
        ),
    }
}

/// Message for a worker that ended before the benchmark finished.
pub(super) fn exit_message(cause: &str, partial: usize) -> String {
    format!(
        "{cause} before the benchmark finished; {partial} partial samples received. Run it with --isolated=false to debug."
    )
}

/// Message for a benchmark that exceeded the worker timeout.
pub(super) fn timeout_message(timeout: Duration, partial: usize) -> String {
    format!(
        "benchmark exceeded the {}s worker timeout; {partial} partial samples received and not used for statistics. Raise --worker-timeout or flux.toml runner.timeout, or shorten the benchmark.",
        timeout.as_secs_f64()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn known_signals_have_names() {
        let cases = [
            (libc::SIGSEGV, "SIGSEGV"),
            (libc::SIGABRT, "SIGABRT"),
            (libc::SIGKILL, "SIGKILL"),
            (libc::SIGBUS, "SIGBUS"),
            (libc::SIGILL, "SIGILL"),
            (libc::SIGFPE, "SIGFPE"),
            (libc::SIGTERM, "SIGTERM"),
            (libc::SIGINT, "SIGINT"),
        ];
        for (signal, name) in cases {
            assert_eq!(signal_name(signal), name);
        }
    }

    #[cfg(unix)]
    #[test]
    fn unknown_signal_falls_back_to_number() {
        assert_eq!(
            signal_name(libc::SIGUSR1),
            format!("signal {}", libc::SIGUSR1)
        );
    }

    #[cfg(unix)]
    #[test]
    fn signal_exit_maps_to_signal_kind() {
        use std::os::unix::process::ExitStatusExt;
        let status = ExitStatus::from_raw(libc::SIGSEGV);
        let (kind, cause) = classify_exit(status);
        assert_eq!(kind, FailureKind::Signal("SIGSEGV".to_string()));
        assert_eq!(cause, "worker killed by SIGSEGV");
    }

    #[cfg(unix)]
    #[test]
    fn code_exit_maps_to_crashed() {
        use std::os::unix::process::ExitStatusExt;
        // Wait status for exit code 3: the code sits in the second byte.
        let status = ExitStatus::from_raw(3 << 8);
        let (kind, cause) = classify_exit(status);
        assert_eq!(kind, FailureKind::Crashed);
        assert_eq!(cause, "worker exited with code 3");
    }

    #[test]
    fn messages_name_partial_samples() {
        assert!(timeout_message(Duration::from_secs(3), 7).contains("3s worker timeout"));
        assert!(timeout_message(Duration::from_secs(3), 7).contains("7 partial samples"));
        assert!(
            exit_message("worker killed by SIGABRT", 0).starts_with("worker killed by SIGABRT")
        );
    }
}
