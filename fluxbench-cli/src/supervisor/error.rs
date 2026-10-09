//! Supervisor error type.

use fluxbench_ipc::FrameError;
use std::time::Duration;
use thiserror::Error;

/// Errors that can occur during supervisor operations
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SupervisorError {
    /// Failed to spawn worker process.
    #[error("worker spawn failed: {0}. Check that the bench binary exists and runs.")]
    SpawnFailed(#[from] std::io::Error),

    /// IPC communication error.
    #[error("IPC error: {0}")]
    IpcError(String),

    /// Worker process crashed or exited unexpectedly.
    #[error("worker crashed: {0}")]
    WorkerCrashed(String),

    /// Worker did not respond within timeout.
    #[error("timeout waiting for worker")]
    Timeout,

    /// Worker sent no `Hello` within the deadline. The supervisor killed it.
    #[error(
        "worker sent no Hello within {0:?} and was killed. Check that the bench binary calls fluxbench::run() early in main."
    )]
    HelloTimeout(Duration),

    /// Benchmark with the given ID not found.
    #[error("benchmark not found: {0}")]
    BenchmarkNotFound(String),

    /// Worker protocol version or message type mismatch.
    #[error(
        "worker protocol error: expected {expected}, got {got}. Rebuild the bench binary against this fluxbench version."
    )]
    ProtocolError {
        /// Expected message or protocol version.
        expected: String,
        /// Actual message or protocol version received.
        got: String,
    },
}

impl From<FrameError> for SupervisorError {
    fn from(e: FrameError) -> Self {
        SupervisorError::IpcError(e.to_string())
    }
}
