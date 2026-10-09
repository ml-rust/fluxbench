//! Bencher - The Benchmark Iteration API
//!
//! Provides the user-facing API for defining what to measure.
//! Uses Criterion-style batched sampling: iterations are grouped into samples,
//! with each sample being the average of many iterations.

use crate::allocator::{current_allocation, reset_allocation_counter};
use crate::measure::Timer;
use fluxbench_ipc::Sample;

/// Default number of samples to collect (matches Criterion)
pub const DEFAULT_SAMPLE_COUNT: usize = fluxbench_ipc::DEFAULT_SAMPLE_COUNT as usize;

/// Minimum sample count. Two samples give a defined variance.
pub const MIN_SAMPLE_COUNT: usize = 2;

/// Mode of iteration for the benchmark
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IterationMode {
    /// Simple iteration - measure entire closure
    Simple,
    /// Iteration with setup - separate setup from measurement
    WithSetup,
    /// Iteration with teardown
    WithTeardown,
}

/// Result of a single benchmark run
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// All collected samples (each sample = average of many iterations)
    pub samples: Vec<Sample>,
    /// Total iterations performed across all samples
    pub iterations: u64,
    /// Total time spent in measurement (excluding warmup)
    pub total_time_ns: u64,
}

/// The Bencher provides iteration control for benchmarks.
///
/// Uses Criterion-style batched sampling:
/// - Warmup phase estimates iteration time
/// - Measurement phase batches iterations into samples
/// - Each sample = average of many iterations (reduces noise)
pub struct Bencher {
    // === Current sample accumulation ===
    current_sample_time_ns: u64,
    current_sample_cycles: u64,
    current_sample_iters: u64,
    current_sample_alloc_bytes: u64,
    current_sample_alloc_count: u64,

    // === Completed samples ===
    samples: Vec<Sample>,

    // === Configuration ===
    target_samples: usize,
    iters_per_sample: u64,
    fixed_iters_per_sample: Option<u64>,
    track_allocations: bool,

    // === State ===
    total_iterations: u64,
    is_warmup: bool,
    warmup_times: Vec<u64>, // Raw timings during warmup for estimation

    // === Cached runtime for iter_async fallback ===
    cached_runtime: Option<tokio::runtime::Runtime>,
}

impl Bencher {
    /// Create a new Bencher
    pub fn new(track_allocations: bool) -> Self {
        Self::with_sample_count(track_allocations, DEFAULT_SAMPLE_COUNT)
    }

    /// Create a Bencher that targets `sample_count` samples.
    ///
    /// The count has no upper cap. Values below `MIN_SAMPLE_COUNT` rise to it.
    pub fn with_sample_count(track_allocations: bool, sample_count: usize) -> Self {
        let target_samples = sample_count.max(MIN_SAMPLE_COUNT);
        Self {
            current_sample_time_ns: 0,
            current_sample_cycles: 0,
            current_sample_iters: 0,
            current_sample_alloc_bytes: 0,
            current_sample_alloc_count: 0,
            samples: Vec::with_capacity(target_samples),
            target_samples,
            iters_per_sample: 1, // Will be set after warmup
            fixed_iters_per_sample: None,
            track_allocations,
            total_iterations: 0,
            is_warmup: true,
            warmup_times: Vec::with_capacity(1000),
            cached_runtime: None,
        }
    }

    /// Create a Bencher from run config values.
    ///
    /// `Some(n)` iterations per sample selects fixed mode. `None` selects time-based mode.
    pub fn for_config(
        track_allocations: bool,
        sample_count: usize,
        iters_per_sample: Option<u64>,
    ) -> Self {
        let bencher = Self::with_sample_count(track_allocations, sample_count);
        match iters_per_sample {
            Some(iters) => bencher.with_fixed_iters_per_sample(iters),
            None => bencher,
        }
    }

    /// Fix the iterations per sample. This selects fixed mode: no warmup, no estimation.
    ///
    /// A value of 0 rises to 1.
    pub fn with_fixed_iters_per_sample(mut self, iters: u64) -> Self {
        let iters = iters.max(1);
        self.fixed_iters_per_sample = Some(iters);
        self.iters_per_sample = iters;
        self
    }

    /// Fixed iterations per sample, if fixed mode is on
    pub fn fixed_iters_per_sample(&self) -> Option<u64> {
        self.fixed_iters_per_sample
    }

    /// Set iterations per sample (called after warmup estimation)
    pub fn set_iters_per_sample(&mut self, iters: u64) {
        self.iters_per_sample = iters.max(1);
    }

    /// Get estimated iteration time from warmup (in nanoseconds)
    pub fn estimated_iter_time_ns(&self) -> Option<u64> {
        if self.warmup_times.is_empty() {
            return None;
        }
        let sum: u64 = self.warmup_times.iter().sum();
        Some(sum / self.warmup_times.len() as u64)
    }

    /// Transition from warmup to measurement phase
    pub fn start_measurement(&mut self, measurement_time_ns: u64) {
        self.is_warmup = false;

        // Calculate iterations per sample based on warmup estimate
        if let Some(iters) = self.fixed_iters_per_sample {
            self.iters_per_sample = iters;
        } else if let Some(iter_time) = self.estimated_iter_time_ns() {
            let time_per_sample = measurement_time_ns / self.target_samples as u64;
            self.iters_per_sample = (time_per_sample / iter_time).max(1);
        } else {
            self.iters_per_sample = 1;
        }

        // Clear warmup data
        self.warmup_times.clear();
        self.warmup_times.shrink_to_fit();

        // Reset accumulation state
        self.current_sample_time_ns = 0;
        self.current_sample_cycles = 0;
        self.current_sample_iters = 0;
        self.current_sample_alloc_bytes = 0;
        self.current_sample_alloc_count = 0;
    }

    /// Accumulate a single iteration's measurements into the current sample
    ///
    /// During warmup: records timing for iteration time estimation.
    /// During measurement: accumulates into batched samples, flushing when complete.
    ///
    /// # Arguments
    /// * `duration_nanos` - Iteration duration in nanoseconds
    /// * `cpu_cycles` - CPU cycles consumed (x86_64 only, 0 otherwise)
    /// * `alloc_bytes` - Bytes allocated during iteration
    /// * `alloc_count` - Number of allocations during iteration
    #[inline]
    fn accumulate_sample(
        &mut self,
        duration_nanos: u64,
        cpu_cycles: u64,
        alloc_bytes: u64,
        alloc_count: u64,
    ) {
        self.total_iterations += 1;

        if self.is_warmup {
            // During warmup: collect raw timings for estimation
            self.warmup_times.push(duration_nanos);
        } else {
            // During measurement: accumulate into current sample
            self.current_sample_time_ns += duration_nanos;
            self.current_sample_cycles += cpu_cycles;
            self.current_sample_iters += 1;
            self.current_sample_alloc_bytes += alloc_bytes;
            self.current_sample_alloc_count += alloc_count;

            // Check if we've completed this sample batch
            if self.current_sample_iters >= self.iters_per_sample {
                self.flush_sample();
            }
        }
    }

    /// Run the benchmark closure for one iteration.
    ///
    /// During warmup: records individual timings for estimation.
    /// During measurement: accumulates into batched samples.
    #[inline]
    pub fn iter<T, F>(&mut self, mut f: F)
    where
        F: FnMut() -> T,
    {
        // Reset allocation tracking
        if self.track_allocations {
            reset_allocation_counter();
        }

        // Start timing
        let timer = Timer::start();

        // Run the benchmark
        let _ = std::hint::black_box(f());

        // Stop timing
        let (duration_nanos, cpu_cycles) = timer.stop();

        // Collect allocation data
        let (alloc_bytes, alloc_count) = if self.track_allocations {
            current_allocation()
        } else {
            (0, 0)
        };

        self.accumulate_sample(duration_nanos, cpu_cycles, alloc_bytes, alloc_count);
    }

    /// Run the benchmark with separate setup phase
    #[inline]
    pub fn iter_with_setup<T, S, F, R>(&mut self, mut setup: S, mut routine: F)
    where
        S: FnMut() -> T,
        F: FnMut(T) -> R,
    {
        // Run setup (not timed)
        let input = setup();

        // Reset allocation tracking after setup
        if self.track_allocations {
            reset_allocation_counter();
        }

        // Start timing
        let timer = Timer::start();

        // Run the benchmark
        let _ = std::hint::black_box(routine(input));

        // Stop timing
        let (duration_nanos, cpu_cycles) = timer.stop();

        // Collect allocation data
        let (alloc_bytes, alloc_count) = if self.track_allocations {
            current_allocation()
        } else {
            (0, 0)
        };

        self.accumulate_sample(duration_nanos, cpu_cycles, alloc_bytes, alloc_count);
    }

    /// Run benchmark with batched iterations (user-specified batch size)
    ///
    /// A `batch_size` of 0 is clamped to 1.
    #[inline]
    pub fn iter_batched<T, S, F, R>(&mut self, batch_size: u64, mut setup: S, mut routine: F)
    where
        S: FnMut() -> T,
        F: FnMut(&T) -> R,
    {
        let batch_size = batch_size.max(1);

        // Run setup
        let input = setup();

        // Reset allocation tracking
        if self.track_allocations {
            reset_allocation_counter();
        }

        // Start timing
        let timer = Timer::start();

        // Run batched iterations
        for _ in 0..batch_size {
            let _ = std::hint::black_box(routine(std::hint::black_box(&input)));
        }

        // Stop timing
        let (total_nanos, total_cycles) = timer.stop();

        // Per-iteration values (use f64 to avoid integer truncation for fast ops)
        let per_iter_nanos = ((total_nanos as f64) / (batch_size as f64)).round() as u64;
        let per_iter_cycles = ((total_cycles as f64) / (batch_size as f64)).round() as u64;

        // Collect allocation data (total for batch, then average)
        let (alloc_bytes, alloc_count) = if self.track_allocations {
            let (bytes, count) = current_allocation();
            (bytes / batch_size, count / batch_size)
        } else {
            (0, 0)
        };

        // For batched iterations, we count the batch as batch_size iterations
        // but accumulate as a single sample point
        self.total_iterations += batch_size - 1; // -1 because accumulate_sample adds 1
        self.accumulate_sample(per_iter_nanos, per_iter_cycles, alloc_bytes, alloc_count);
    }

    /// Run an async benchmark closure (standalone - creates its own runtime)
    #[inline]
    pub fn iter_async_standalone<T, F, Fut>(&mut self, mut f: F)
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = T>,
    {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("Failed to create tokio runtime");

        if self.track_allocations {
            reset_allocation_counter();
        }

        let timer = Timer::start();
        let _ = std::hint::black_box(rt.block_on(f()));
        let (duration_nanos, cpu_cycles) = timer.stop();

        let (alloc_bytes, alloc_count) = if self.track_allocations {
            current_allocation()
        } else {
            (0, 0)
        };

        self.accumulate_sample(duration_nanos, cpu_cycles, alloc_bytes, alloc_count);
    }

    /// Run an async benchmark closure within an existing tokio runtime
    #[inline]
    pub fn iter_async<T, F, Fut>(&mut self, mut f: F)
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = T>,
    {
        if self.track_allocations {
            reset_allocation_counter();
        }

        let handle = tokio::runtime::Handle::try_current();

        let (duration_nanos, cpu_cycles, alloc_bytes, alloc_count) = if let Ok(handle) = handle {
            tokio::task::block_in_place(|| {
                let timer = Timer::start();
                let _ = std::hint::black_box(handle.block_on(f()));
                let (duration_nanos, cpu_cycles) = timer.stop();

                let (alloc_bytes, alloc_count) = if self.track_allocations {
                    current_allocation()
                } else {
                    (0, 0)
                };

                (duration_nanos, cpu_cycles, alloc_bytes, alloc_count)
            })
        } else {
            // Cache runtime across iterations to avoid per-iteration construction overhead
            let rt = self.cached_runtime.get_or_insert_with(|| {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("Failed to create tokio runtime")
            });

            let timer = Timer::start();
            let _ = std::hint::black_box(rt.block_on(f()));
            let (duration_nanos, cpu_cycles) = timer.stop();

            let (alloc_bytes, alloc_count) = if self.track_allocations {
                current_allocation()
            } else {
                (0, 0)
            };

            (duration_nanos, cpu_cycles, alloc_bytes, alloc_count)
        };

        self.accumulate_sample(duration_nanos, cpu_cycles, alloc_bytes, alloc_count);
    }

    /// Flush current accumulated iterations as a single sample
    fn flush_sample(&mut self) {
        if self.current_sample_iters == 0 || self.samples.len() >= self.target_samples {
            return;
        }

        let n = self.current_sample_iters;
        let n_f64 = n as f64;

        // Per-iteration means for this sample
        self.samples.push(Sample::new(
            self.current_sample_time_ns as f64 / n_f64,
            self.current_sample_cycles as f64 / n_f64,
            self.current_sample_alloc_bytes as f64 / n_f64,
            self.current_sample_alloc_count as f64 / n_f64,
            n,
        ));

        // Reset for next sample
        self.current_sample_time_ns = 0;
        self.current_sample_cycles = 0;
        self.current_sample_iters = 0;
        self.current_sample_alloc_bytes = 0;
        self.current_sample_alloc_count = 0;
    }

    /// Check if we've collected enough samples
    pub fn has_enough_samples(&self) -> bool {
        self.samples.len() >= self.target_samples
    }

    /// Get collected samples
    pub fn samples(&self) -> &[Sample] {
        &self.samples
    }

    /// Take ownership of collected samples (clears warmup data)
    pub fn take_samples(&mut self) -> Vec<Sample> {
        self.warmup_times.clear();
        std::mem::take(&mut self.samples)
    }

    /// Get total iteration count
    pub fn iteration_count(&self) -> u64 {
        self.total_iterations
    }

    /// Get target sample count
    pub fn target_samples(&self) -> usize {
        self.target_samples
    }

    /// Finalize and return results
    pub fn finish(mut self) -> BenchmarkResult {
        // Flush any remaining accumulated iterations
        self.flush_sample();

        let total_time_ns = self
            .samples
            .iter()
            .map(|s| s.duration_nanos * s.iterations as f64)
            .sum::<f64>()
            .round() as u64;

        BenchmarkResult {
            samples: self.samples,
            iterations: self.total_iterations,
            total_time_ns,
        }
    }
}

/// Run the full benchmark loop: warmup → measurement → finish
///
/// This is the shared implementation used by both in-process and isolated execution.
/// Extracts common logic to avoid duplication between `worker.rs` and `executor.rs`.
///
/// # Arguments
/// * `bencher` - The Bencher instance (takes ownership)
/// * `runner_fn` - Function that calls the benchmark under test
/// * `warmup_time_ns` - How long to run warmup phase (nanoseconds)
/// * `measurement_time_ns` - How long to run measurement phase (nanoseconds)
/// * `min_iterations` - Optional minimum total measurement iterations before completion
/// * `max_iterations` - Optional cap on total measurement iterations
///
/// In fixed mode (`Bencher::with_fixed_iters_per_sample`) the loop skips warmup.
/// It then runs exactly `sample_count * iters_per_sample` iterations.
/// The iteration and time arguments are ignored in this mode.
pub fn run_benchmark_loop<F>(
    mut bencher: Bencher,
    mut runner_fn: F,
    warmup_time_ns: u64,
    measurement_time_ns: u64,
    min_iterations: Option<u64>,
    max_iterations: Option<u64>,
) -> BenchmarkResult
where
    F: FnMut(&mut Bencher),
{
    use crate::Instant;

    // Fixed mode: no warmup, and exactly sample_count * iters_per_sample iterations
    let (warmup_time_ns, min_iterations, max_iterations) = match bencher.fixed_iters_per_sample() {
        Some(iters) => {
            let total = (bencher.target_samples() as u64).saturating_mul(iters);
            (0, Some(total), Some(total))
        }
        None => (warmup_time_ns, min_iterations, max_iterations),
    };

    // Warmup phase - Bencher starts in warmup mode
    // This collects timing data to estimate iterations per sample
    let warmup_start = Instant::now();
    while warmup_start.elapsed().as_nanos() < warmup_time_ns as u128 {
        runner_fn(&mut bencher);
    }

    // Transition to measurement phase
    // This calculates iters_per_sample based on warmup timings
    bencher.start_measurement(measurement_time_ns);

    // Measurement phase - run until we have enough samples or time runs out
    let measure_start = Instant::now();
    let measurement_start_iterations = bencher.iteration_count();
    let min_iterations = min_iterations.unwrap_or(0);
    let max_iterations = max_iterations.unwrap_or(u64::MAX).max(min_iterations);

    loop {
        let measurement_iterations = bencher
            .iteration_count()
            .saturating_sub(measurement_start_iterations);
        let min_iterations_met = measurement_iterations >= min_iterations;
        let max_iterations_reached = measurement_iterations >= max_iterations;
        let has_enough_samples = bencher.has_enough_samples();
        let time_limit_reached = measure_start.elapsed().as_nanos() >= measurement_time_ns as u128;

        if max_iterations_reached {
            break;
        }

        // Respect both quality controls:
        // - stop when sample target reached AND minimum iterations satisfied, or
        // - stop on time budget only after minimum iterations are satisfied.
        if (has_enough_samples || time_limit_reached) && min_iterations_met {
            break;
        }

        runner_fn(&mut bencher);
    }

    bencher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iter_batched_zero_batch_size() {
        let mut bencher = Bencher::new(false);
        let mut calls = 0u64;

        bencher.iter_batched(
            0,
            || 1u64,
            |x| {
                calls += 1;
                *x
            },
        );

        assert_eq!(calls, 1);
        assert_eq!(bencher.iteration_count(), 1);
    }

    #[test]
    fn test_basic_iter() {
        let mut bencher = Bencher::new(false);

        // Simulate warmup
        for _ in 0..100 {
            bencher.iter(|| {
                let mut sum = 0u64;
                for i in 0..1000 {
                    sum += i;
                }
                sum
            });
        }

        // Transition to measurement
        bencher.start_measurement(1_000_000_000); // 1 second

        // Simulate measurement
        for _ in 0..1000 {
            bencher.iter(|| {
                let mut sum = 0u64;
                for i in 0..1000 {
                    sum += i;
                }
                sum
            });
        }

        let result = bencher.finish();
        assert!(!result.samples.is_empty());
        assert!(result.samples.len() <= DEFAULT_SAMPLE_COUNT);
    }

    #[test]
    fn test_iter_with_setup() {
        let mut bencher = Bencher::new(false);

        for _ in 0..5 {
            bencher.iter_with_setup(
                || vec![1, 2, 3, 4, 5],    // Setup: create vec
                |v| v.iter().sum::<i32>(), // Measure: sum
            );
        }

        // During warmup, samples aren't recorded
        assert_eq!(bencher.samples().len(), 0);
        assert_eq!(bencher.warmup_times.len(), 5);
    }

    #[test]
    fn test_sample_batching() {
        let mut bencher = Bencher::with_sample_count(false, 10); // 10 target samples

        // Skip warmup, go directly to measurement
        bencher.is_warmup = false;
        bencher.iters_per_sample = 5; // 5 iterations per sample

        // Run 50 iterations -> should produce 10 samples
        for _ in 0..50 {
            bencher.iter(|| 42);
        }

        let result = bencher.finish();
        assert_eq!(result.samples.len(), 10);
        assert_eq!(result.iterations, 50);
    }

    #[test]
    fn test_run_loop_respects_min_iterations() {
        let bencher = Bencher::with_sample_count(false, 10);
        let result = run_benchmark_loop(bencher, |b| b.iter(|| 42_u64), 0, 0, Some(100), Some(100));

        assert_eq!(result.iterations, 100);
    }

    #[test]
    fn test_run_loop_clamps_min_to_max() {
        let bencher = Bencher::with_sample_count(false, 10);
        let result = run_benchmark_loop(bencher, |b| b.iter(|| 7_u64), 0, 0, Some(200), Some(50));

        assert_eq!(result.iterations, 200);
    }

    #[test]
    fn test_sample_count_above_100_is_honoured() {
        let bencher = Bencher::with_sample_count(false, 250);
        assert_eq!(bencher.target_samples(), 250);

        // min = max = 250 with 1 iteration per sample (no warmup estimate)
        let result = run_benchmark_loop(bencher, |b| b.iter(|| 1_u64), 0, 1, Some(250), Some(250));

        assert_eq!(result.samples.len(), 250);
        assert!(result.samples.iter().all(|s| s.iterations == 1));
    }

    #[test]
    fn test_sample_count_floor_is_two() {
        assert_eq!(Bencher::with_sample_count(false, 0).target_samples(), 2);
        assert_eq!(Bencher::with_sample_count(false, 1).target_samples(), 2);
        assert_eq!(Bencher::with_sample_count(false, 2).target_samples(), 2);
    }

    #[test]
    fn test_fixed_mode_500_samples_of_one_iteration() {
        let bencher = Bencher::with_sample_count(false, 500).with_fixed_iters_per_sample(1);
        // Time and iteration arguments are ignored in fixed mode
        let result = run_benchmark_loop(
            bencher,
            |b| b.iter(|| 3_u64),
            1_000_000_000,
            1,
            Some(7),
            Some(7),
        );

        assert_eq!(result.samples.len(), 500);
        assert_eq!(result.iterations, 500);
        assert!(result.samples.iter().all(|s| s.iterations == 1));
    }

    #[test]
    fn test_fixed_mode_groups_iterations_per_sample() {
        let bencher = Bencher::with_sample_count(false, 10).with_fixed_iters_per_sample(3);
        let result = run_benchmark_loop(bencher, |b| b.iter(|| 5_u64), 0, 1, None, None);

        assert_eq!(result.samples.len(), 10);
        assert_eq!(result.iterations, 30);
        assert!(result.samples.iter().all(|s| s.iterations == 3));
    }

    #[test]
    fn test_flushed_sample_holds_means() {
        let mut bencher = Bencher::with_sample_count(false, 2);
        bencher.is_warmup = false;
        bencher.iters_per_sample = 4;
        for (time, cycles) in [(10, 30), (20, 60), (30, 90), (41, 121)] {
            bencher.accumulate_sample(time, cycles, 8, 1);
        }

        let result = bencher.finish();
        assert_eq!(result.samples.len(), 1);
        let sample = result.samples[0];
        assert_eq!(sample.duration_nanos, 25.25);
        assert_eq!(sample.cpu_cycles, 75.25);
        assert_eq!(sample.alloc_bytes, 8.0);
        assert_eq!(sample.alloc_count, 1.0);
        assert_eq!(sample.iterations, 4);
        assert_eq!(result.total_time_ns, 101);
    }
}
