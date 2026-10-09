//! Worker process control: graceful stop, kill, and reap.
//!
//! Every path that ends a worker reaps it, so no zombie stays behind.

use std::process::{Child, ExitStatus};
use std::time::{Duration, Instant};

/// Wait this long for a worker to exit on its own before SIGKILL.
pub(super) const EXIT_GRACE: Duration = Duration::from_secs(1);

/// Step between exit checks while waiting.
const REAP_STEP: Duration = Duration::from_millis(5);

/// Send SIGTERM to a process. Returns `Err` when the signal was not delivered.
#[cfg(unix)]
pub(super) fn send_sigterm(pid: u32) -> Result<(), std::io::Error> {
    let pid = libc::pid_t::try_from(pid).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("pid {pid} does not fit pid_t"),
        )
    })?;
    // SAFETY: `kill` takes plain integers and has no memory effects.
    let ret = unsafe { libc::kill(pid, libc::SIGTERM) };
    if ret == -1 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

/// `true` while the child runs. A finished child is reaped by this call.
pub(super) fn is_running(child: &mut Child) -> bool {
    matches!(child.try_wait(), Ok(None))
}

/// Wait up to `grace` for the child to exit, then SIGKILL it. Always reaps.
///
/// Returns the exit status, or `None` when the OS reports no status.
pub(super) fn reap(child: &mut Child, grace: Duration) -> Option<ExitStatus> {
    let deadline = Instant::now() + grace;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Some(status),
            Ok(None) => {}
            Err(_) => break,
        }
        if Instant::now() >= deadline {
            break;
        }
        std::thread::sleep(REAP_STEP);
    }
    let _ = child.kill();
    child.wait().ok()
}

/// SIGKILL the child at once and reap it.
pub(super) fn kill_and_reap(child: &mut Child) -> Option<ExitStatus> {
    reap(child, Duration::ZERO)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::process::Command;

    #[test]
    fn reap_returns_status_of_exited_child() {
        let mut child = Command::new("true").spawn().expect("spawn true");
        let status = reap(&mut child, Duration::from_secs(5)).expect("status");
        assert!(status.success());
    }

    #[test]
    fn reap_kills_child_past_grace() {
        use std::os::unix::process::ExitStatusExt;
        let mut child = Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("spawn sleep");
        let status = reap(&mut child, Duration::from_millis(20)).expect("status");
        assert_eq!(status.signal(), Some(libc::SIGKILL));
        assert!(!is_running(&mut child));
    }
}
