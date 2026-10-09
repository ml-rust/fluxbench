//! Background flusher: sends due sample batches while the benchmark body runs.
//!
//! A body can hang inside one call. The measurement thread then never pushes again.
//! This thread still sends pending samples on the heartbeat and on SIGTERM.

use super::stream::SharedStream;
use crate::shutdown::shutdown_requested;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

/// Wake interval. It bounds the delay between SIGTERM and the last flush.
const FLUSH_TICK: Duration = Duration::from_millis(20);

/// Flusher thread handle. Drop stops and joins the thread.
pub(super) struct Flusher {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Flusher {
    /// Start the flusher thread.
    ///
    /// Start it before CPU pinning, so it does not share the pinned core.
    /// If the thread does not start, the worker still flushes on each push.
    pub(super) fn start(stream: SharedStream) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let handle = std::thread::Builder::new()
            .name("fluxbench-flusher".to_string())
            .spawn(move || {
                while !thread_stop.load(Ordering::Relaxed) {
                    std::thread::park_timeout(FLUSH_TICK);
                    stream.flush_due(shutdown_requested());
                }
            });
        let handle = match handle {
            Ok(handle) => Some(handle),
            Err(e) => {
                eprintln!(
                    "fluxbench: warning: flusher thread did not start: {e}. A hung benchmark loses its unsent samples."
                );
                None
            }
        };
        Self { stop, handle }
    }
}

impl Drop for Flusher {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            handle.thread().unpark();
            let _ = handle.join();
        }
    }
}
