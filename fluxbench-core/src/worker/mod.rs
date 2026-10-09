//! Worker process side of the supervisor-worker architecture.

mod flusher;
mod main_loop;
mod panic_report;
mod stream;
mod transport;

pub use main_loop::WorkerMain;
