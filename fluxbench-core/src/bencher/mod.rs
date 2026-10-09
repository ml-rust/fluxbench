//! Bencher - The Benchmark Iteration API
//!
//! Criterion-style batched sampling: one `iter*` call runs warmup and measurement.
//! Each sample holds per-iteration means over a batch of calls.

#[cfg(feature = "async")]
mod async_body;
#[cfg(feature = "async")]
mod async_iter;
mod batch_size;
mod batched;
mod config;
mod custom;
mod engine;
mod iter;
mod run;
mod sink;
mod state;
#[cfg(test)]
mod test_support;

#[cfg(feature = "async")]
pub use async_body::drive_async_body;
pub use batch_size::{BatchSize, LARGE_INPUT_CHUNK, SETUP_CHUNK};
pub use config::{
    BenchmarkResult, DEFAULT_SAMPLE_COUNT, IterationMode, MIN_SAMPLE_COUNT, MeasureConfig,
    MeasureError,
};
pub use run::run_benchmark_loop;
pub use sink::SampleSink;
pub use state::Bencher;
