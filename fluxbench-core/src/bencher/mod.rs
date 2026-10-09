//! Bencher - The Benchmark Iteration API
//!
//! Criterion-style batched sampling: one `iter*` call runs warmup and measurement.
//! Each sample holds per-iteration means over a batch of calls.

mod async_body;
mod async_iter;
mod config;
mod engine;
mod iter;
mod run;
mod state;

pub use async_body::drive_async_body;
pub use config::{
    BenchmarkResult, DEFAULT_SAMPLE_COUNT, IterationMode, MIN_SAMPLE_COUNT, MeasureConfig,
    MeasureError,
};
pub use iter::SETUP_CHUNK;
pub use run::run_benchmark_loop;
pub use state::Bencher;
