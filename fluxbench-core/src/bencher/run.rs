//! Shared entry point for in-process and isolated execution.

use super::config::{BenchmarkResult, MeasureError};
use super::state::Bencher;

/// Run one benchmark body and return its result.
///
/// `runner_fn` runs exactly once.
/// The body's single `iter*` call runs warmup and measurement.
/// The bencher's `MeasureConfig` sets every time budget and limit.
///
/// # Errors
/// - `MeasureError::NoIterCall`: the body never called an `iter*` method.
/// - `MeasureError::MultipleIterCalls`: the body called a second `iter*` method.
/// - `MeasureError::NoSamples`: measurement collected no samples.
/// - `MeasureError::AsyncRuntime`: the async runtime did not build, or `iter_async` ran inside a runtime context.
pub fn run_benchmark_loop<F>(
    mut bencher: Bencher,
    runner_fn: F,
) -> Result<BenchmarkResult, MeasureError>
where
    F: FnOnce(&mut Bencher),
{
    runner_fn(&mut bencher);
    bencher.finish()
}
