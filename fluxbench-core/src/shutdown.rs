//! Graceful stop flag.
//!
//! In a worker, a SIGTERM handler sets the flag.
//! The measurement engine checks it between batches and stops early.
//! In-process runs never install the handler, so the flag stays clear.

use std::sync::atomic::{AtomicBool, Ordering};

/// Set by the SIGTERM handler to request a graceful stop.
static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);

/// `true` once a graceful stop is requested via SIGTERM.
pub(crate) fn shutdown_requested() -> bool {
    SHUTDOWN_REQUESTED.load(Ordering::Relaxed)
}

/// Install a SIGTERM handler that sets the stop flag.
///
/// The handler only stores an atomic, so it is async-signal-safe.
#[cfg(unix)]
pub(crate) fn install_sigterm_handler() {
    // SAFETY: `sigaction` gets a zeroed, fully initialized struct.
    // The handler only touches an atomic.
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = sigterm_handler as *const () as usize;
        sa.sa_flags = libc::SA_RESTART;
        libc::sigemptyset(&mut sa.sa_mask);
        libc::sigaction(libc::SIGTERM, &sa, std::ptr::null_mut());
    }
}

#[cfg(unix)]
extern "C" fn sigterm_handler(_sig: libc::c_int) {
    SHUTDOWN_REQUESTED.store(true, Ordering::Relaxed);
}

/// No-op on non-Unix: there is no SIGTERM equivalent.
#[cfg(not(unix))]
pub(crate) fn install_sigterm_handler() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flag_starts_clear() {
        assert!(!shutdown_requested());
    }
}
