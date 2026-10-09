//! Low-level IPC helpers: pipes, `poll(2)`, and the readiness wait loop.

use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::unix::io::RawFd;

/// Longest single poll. Between polls the caller checks the deadline and liveness.
pub(super) const POLL_SLICE: Duration = Duration::from_millis(100);

/// Result of one poll for data.
#[derive(Debug)]
#[cfg_attr(not(unix), allow(dead_code))]
pub(super) enum PollResult {
    DataAvailable,
    Timeout,
    PipeClosed,
    Error(std::io::Error),
}

/// Outcome of waiting for worker data.
#[derive(Debug)]
pub(super) enum Readiness {
    /// Data is ready to read.
    Ready,
    /// The deadline passed with no data.
    Expired,
    /// The pipe closed or the worker exited.
    Closed,
}

/// Wait until data is ready, the deadline passes, or the worker is gone.
///
/// `poll(slice)` waits up to `slice` for data.
/// `alive()` reports whether the worker still runs.
/// A worker that exits while a grandchild holds the pipe open still reads as `Closed`.
pub(super) fn wait_ready(
    deadline: Instant,
    mut poll: impl FnMut(Duration) -> PollResult,
    mut alive: impl FnMut() -> bool,
) -> Readiness {
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Readiness::Expired;
        }
        match poll(remaining.min(POLL_SLICE)) {
            PollResult::DataAvailable => return Readiness::Ready,
            PollResult::PipeClosed => return Readiness::Closed,
            PollResult::Error(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            PollResult::Error(_) => return Readiness::Closed,
            PollResult::Timeout => {
                if !alive() {
                    return Readiness::Closed;
                }
            }
        }
    }
}

/// Wait up to `timeout` for data on a raw fd (`poll(2)`).
#[cfg(unix)]
pub(super) fn wait_for_data_fd(fd: RawFd, timeout: Duration) -> PollResult {
    let mut pollfd = libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    };
    let timeout_ms = i32::try_from(timeout.as_millis()).unwrap_or(i32::MAX);

    // SAFETY: `pollfd` is a valid, initialized struct for one fd.
    let result = unsafe { libc::poll(&mut pollfd, 1, timeout_ms) };

    if result < 0 {
        PollResult::Error(std::io::Error::last_os_error())
    } else if result == 0 {
        PollResult::Timeout
    } else if pollfd.revents & libc::POLLIN != 0 {
        PollResult::DataAvailable
    } else if pollfd.revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 {
        PollResult::PipeClosed
    } else {
        PollResult::Timeout
    }
}

/// Create a pipe with both ends close-on-exec.
#[cfg(unix)]
pub(super) fn create_pipe() -> Result<(RawFd, RawFd), std::io::Error> {
    let mut fds = [0 as RawFd; 2];
    // SAFETY: `fds` has room for the two fds `pipe` writes.
    let ret = unsafe { libc::pipe(fds.as_mut_ptr()) };
    if ret != 0 {
        return Err(std::io::Error::last_os_error());
    }
    for &fd in &fds {
        // SAFETY: `fd` is a valid fd from `pipe` above.
        unsafe {
            let flags = libc::fcntl(fd, libc::F_GETFD);
            libc::fcntl(fd, libc::F_SETFD, flags | libc::FD_CLOEXEC);
        }
    }
    Ok((fds[0], fds[1]))
}

/// Close a raw fd this process owns.
#[cfg(unix)]
pub(super) fn close_fd(fd: RawFd) {
    // SAFETY: callers pass an fd they own and never use again.
    unsafe {
        libc::close(fd);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wait_ready_returns_ready_on_data() {
        let deadline = Instant::now() + Duration::from_secs(5);
        let outcome = wait_ready(deadline, |_| PollResult::DataAvailable, || true);
        assert!(matches!(outcome, Readiness::Ready));
    }

    #[test]
    fn wait_ready_expires_at_deadline() {
        let deadline = Instant::now() + Duration::from_millis(30);
        let outcome = wait_ready(
            deadline,
            |slice| {
                std::thread::sleep(slice);
                PollResult::Timeout
            },
            || true,
        );
        assert!(matches!(outcome, Readiness::Expired));
        assert!(Instant::now() >= deadline);
    }

    #[test]
    fn wait_ready_reports_dead_worker() {
        let deadline = Instant::now() + Duration::from_secs(5);
        let outcome = wait_ready(deadline, |_| PollResult::Timeout, || false);
        assert!(matches!(outcome, Readiness::Closed));
    }

    #[test]
    fn wait_ready_retries_after_interrupt() {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut calls = 0;
        let outcome = wait_ready(
            deadline,
            |_| {
                calls += 1;
                if calls == 1 {
                    PollResult::Error(std::io::Error::from(std::io::ErrorKind::Interrupted))
                } else {
                    PollResult::DataAvailable
                }
            },
            || true,
        );
        assert!(matches!(outcome, Readiness::Ready));
        assert_eq!(calls, 2);
    }

    #[test]
    fn wait_ready_reports_closed_pipe() {
        let deadline = Instant::now() + Duration::from_secs(5);
        let outcome = wait_ready(deadline, |_| PollResult::PipeClosed, || true);
        assert!(matches!(outcome, Readiness::Closed));
    }
}
