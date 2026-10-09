//! Supervisor process: runs benchmarks in worker processes and collects results over IPC.
//!
//! Every benchmark yields one result, whatever happens to its worker.
//! Panic, signal, timeout, and IPC errors each map to a failure kind.

mod error;
mod failure;
mod handle;
mod hello;
mod ipc_io;
mod process;
mod result;
mod run;
mod spawn;

pub use error::SupervisorError;
pub use fluxbench_ipc::FailureKind;
pub use handle::WorkerHandle;
pub use hello::WORKER_HELLO_TIMEOUT;
pub use result::{IpcBenchmarkResult, IpcBenchmarkStatus};
pub use run::Supervisor;
