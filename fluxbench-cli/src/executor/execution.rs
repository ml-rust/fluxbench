//! Benchmark Execution
//!
//! Execution config, result type, and in-process execution.
//! `isolated.rs` holds the IPC-based executor.
//!
//! ## Execution Modes
//!
//! - **In-process (`Executor`)**: Runs benchmarks in the same process. Fast but
//!   a panic in one benchmark will crash the entire run. Best for development.
//!
//! - **Isolated (`IsolatedExecutor`)**: Spawns worker processes via IPC. Provides
//!   crash isolation - a panic in one benchmark won't affect others. Recommended
//!   for production runs.
//!
//! ## Data Flow
//!
//! ```text
//! BenchmarkDef (from inventory)
//!        │
//!        ▼
//!   ExecutionConfig
//!        │
//!        ▼
//! ┌──────────────────┐
//! │  Executor/       │  Warmup → Measurement → Sample Collection
//! │  IsolatedExecutor│
//! └────────┬─────────┘
//!          │
//!          ▼
//!  BenchExecutionResult (samples, status, allocations, cycles)
//! ```

use super::group_configs::GroupConfigs;
use super::throughput::resolve_throughput;
use fluxbench_core::{
    Bencher, BenchmarkDef, GroupConfig, MeasureConfig, Throughput, cycles_overhead,
    run_benchmark_loop, timer_overhead_ns,
};
use fluxbench_ipc::{BenchmarkConfig, Sample};
use fluxbench_report::BenchmarkStatus;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::{Duration, Instant};

/// Progress bar for `len` benchmarks.
pub(super) fn progress_bar(len: usize) -> ProgressBar {
    let pb = ProgressBar::new(len as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("#>-"),
    );
    pb
}

/// Mean per-iteration allocation bytes and count across samples, unrounded.
///
/// Each sample already holds a per-iteration mean. Returns `(0.0, 0.0)` for no samples.
pub(super) fn exact_allocations(samples: &[Sample]) -> (f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }
    let n = samples.len() as f64;
    let bytes: f64 = samples.iter().map(|s| s.alloc_bytes).sum();
    let count: f64 = samples.iter().map(|s| s.alloc_count).sum();
    (bytes / n, count / n)
}

/// Mean per-iteration allocation bytes and count across samples, rounded.
///
/// Returns `(0, 0)` for no samples.
pub(super) fn mean_allocations(samples: &[Sample]) -> (u64, u64) {
    let (bytes, count) = exact_allocations(samples);
    (bytes.round() as u64, count.round() as u64)
}

/// Configuration for benchmark execution
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Warmup time in nanoseconds
    pub warmup_time_ns: u64,
    /// Measurement time in nanoseconds
    pub measurement_time_ns: u64,
    /// Minimum total measurement iterations
    pub min_iterations: Option<u64>,
    /// Maximum total measurement iterations
    pub max_iterations: Option<u64>,
    /// Target number of samples
    pub sample_count: usize,
    /// Fixed iterations per sample. `Some` selects fixed mode with no warmup.
    pub iters_per_sample: Option<u64>,
    /// Track allocations
    pub track_allocations: bool,
    /// Number of bootstrap iterations for statistics
    pub bootstrap_iterations: usize,
    /// Confidence level for intervals
    pub confidence_level: f64,
}

impl ExecutionConfig {
    /// Merge per-benchmark and group overrides with this config.
    ///
    /// Precedence per key: `#[flux::bench]`, then `group`, then this config, which holds
    /// CLI flags over flux.toml over defaults. `group` is the merged `#[flux::group]` chain.
    ///
    /// - Count: bench `sample_count`, bench `samples`, group, this config.
    /// - Iterations per sample: bench `iters_per_sample`, bench `samples` (1), group, this config.
    ///   `Some` selects fixed mode.
    /// - Warmup, measurement, min and max iterations: bench, group, this config.
    ///
    /// Fixed mode sets warmup to 0 and clears min/max iterations.
    pub fn resolve_for_benchmark(
        &self,
        bench: &BenchmarkDef,
        group: &GroupConfig,
    ) -> ExecutionConfig {
        let sample_count = bench
            .sample_count
            .or(bench.samples)
            .or(group.sample_count)
            .map_or(self.sample_count, |n| n as usize);
        let iters_per_sample = bench
            .iters_per_sample
            .or(bench.samples.map(|_| 1))
            .or(group.iters_per_sample)
            .or(self.iters_per_sample);
        let measurement_time_ns = bench
            .measurement_ns
            .or(group.measurement_ns)
            .unwrap_or(self.measurement_time_ns);

        if iters_per_sample.is_some() {
            return ExecutionConfig {
                warmup_time_ns: 0,
                measurement_time_ns,
                min_iterations: None,
                max_iterations: None,
                sample_count,
                iters_per_sample,
                ..self.clone()
            };
        }

        ExecutionConfig {
            warmup_time_ns: bench
                .warmup_ns
                .or(group.warmup_ns)
                .unwrap_or(self.warmup_time_ns),
            measurement_time_ns,
            min_iterations: bench
                .min_iterations
                .or(group.min_iterations)
                .or(self.min_iterations),
            max_iterations: bench
                .max_iterations
                .or(group.max_iterations)
                .or(self.max_iterations),
            sample_count,
            iters_per_sample: None,
            ..self.clone()
        }
    }

    /// Build the measurement config for one in-process run.
    pub fn to_measure_config(&self) -> MeasureConfig {
        MeasureConfig {
            warmup_ns: self.warmup_time_ns,
            measurement_ns: self.measurement_time_ns,
            sample_count: self.sample_count,
            fixed_iters_per_sample: self.iters_per_sample,
            min_iterations: self.min_iterations,
            max_iterations: self.max_iterations,
            track_allocations: self.track_allocations,
        }
    }

    /// Build the IPC config for one worker run.
    pub fn to_ipc_config(&self, timeout: Duration) -> BenchmarkConfig {
        BenchmarkConfig {
            warmup_time_ns: self.warmup_time_ns,
            measurement_time_ns: self.measurement_time_ns,
            min_iterations: self.min_iterations,
            max_iterations: self.max_iterations,
            sample_count: u32::try_from(self.sample_count).unwrap_or(u32::MAX),
            iters_per_sample: self.iters_per_sample,
            track_allocations: self.track_allocations,
            timeout_ns: u64::try_from(timeout.as_nanos()).unwrap_or(u64::MAX),
        }
    }
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            warmup_time_ns: 3_000_000_000,      // 3 seconds
            measurement_time_ns: 5_000_000_000, // 5 seconds
            min_iterations: Some(100),
            max_iterations: None,
            sample_count: fluxbench_core::DEFAULT_SAMPLE_COUNT,
            iters_per_sample: None,
            track_allocations: true,
            bootstrap_iterations: 100_000, // Matches Criterion default
            confidence_level: 0.95,
        }
    }
}

/// Result from executing a single benchmark
#[derive(Debug)]
pub struct BenchExecutionResult {
    pub benchmark_id: String,
    pub benchmark_name: String,
    pub group: String,
    pub file: String,
    pub line: u32,
    pub status: BenchmarkStatus,
    pub samples: Vec<f64>,
    /// Mean CPU cycles per iteration for each sample (parallel with samples)
    pub cpu_cycles: Vec<f64>,
    /// Mean bytes allocated per iteration across samples
    pub alloc_bytes: u64,
    /// Mean allocation count per iteration across samples
    pub alloc_count: u64,
    /// Unrounded mean bytes per iteration. Allocation limits use it.
    pub alloc_bytes_exact: f64,
    /// Unrounded mean allocation count per iteration. Allocation limits use it.
    pub alloc_count_exact: f64,
    pub duration_ns: u64,
    pub error_message: Option<String>,
    /// Report string of the failure kind, e.g. `panic`, `timeout`, `signal`
    pub failure_kind: Option<String>,
    /// Signal name when a signal killed the worker, e.g. `SIGSEGV`
    pub signal: Option<String>,
    pub backtrace: Option<String>,
    pub severity: fluxbench_core::Severity,
    /// Per-benchmark regression threshold (0.0 = use global)
    pub threshold: f64,
    /// Work per call after precedence: body, then `#[flux::bench]`, then group. `None` on failure.
    pub throughput: Option<Throughput>,
}

/// Execute benchmarks and produce results (in-process mode)
pub struct Executor {
    config: ExecutionConfig,
    groups: GroupConfigs,
    results: Vec<BenchExecutionResult>,
}

impl Executor {
    /// Create a new in-process executor with the given configuration
    pub fn new(config: ExecutionConfig) -> Self {
        // Calibrate timer overhead before any benchmark runs
        let _ = timer_overhead_ns();
        let _ = cycles_overhead();
        Self {
            config,
            groups: GroupConfigs::default(),
            results: Vec::new(),
        }
    }

    /// Apply `#[flux::group]` run config. Without it, no group config applies.
    pub fn with_group_configs(mut self, groups: GroupConfigs) -> Self {
        self.groups = groups;
        self
    }

    /// Execute all provided benchmarks
    pub fn execute(&mut self, benchmarks: &[&BenchmarkDef]) -> Vec<BenchExecutionResult> {
        let pb = progress_bar(benchmarks.len());

        for bench in benchmarks {
            pb.set_message(bench.id.to_string());
            let result = self.execute_single(bench);
            self.results.push(result);
            pb.inc(1);
        }

        pb.finish_with_message("Complete");
        std::mem::take(&mut self.results)
    }

    /// Execute a single benchmark
    fn execute_single(&self, bench: &BenchmarkDef) -> BenchExecutionResult {
        let start = Instant::now();
        let group = self.groups.get(bench.group);
        let cfg = self.config.resolve_for_benchmark(bench, &group);

        // Run with panic catching
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let bencher = Bencher::with_config(cfg.to_measure_config());
            run_benchmark_loop(bencher, |b| (bench.runner_fn)(b))
        }));

        let duration_ns = start.elapsed().as_nanos() as u64;

        match result {
            Ok(Ok(bench_result)) => {
                if bench_result.truncated {
                    eprintln!(
                        "fluxbench: warning: {}: deadline reached, {} of {} samples kept",
                        bench.id,
                        bench_result.samples.len(),
                        cfg.sample_count
                    );
                }

                // Extract timing samples as f64 for statistics
                let samples: Vec<f64> = bench_result
                    .samples
                    .iter()
                    .map(|s| s.duration_nanos)
                    .collect();

                // Extract CPU cycles (parallel array with samples)
                let cpu_cycles: Vec<f64> =
                    bench_result.samples.iter().map(|s| s.cpu_cycles).collect();

                let (alloc_bytes, alloc_count) = mean_allocations(&bench_result.samples);
                let (alloc_bytes_exact, alloc_count_exact) =
                    exact_allocations(&bench_result.samples);

                BenchExecutionResult {
                    benchmark_id: bench.id.to_string(),
                    benchmark_name: bench.name.to_string(),
                    group: bench.group.to_string(),
                    file: bench.file.to_string(),
                    line: bench.line,
                    status: BenchmarkStatus::Passed,
                    samples,
                    cpu_cycles,
                    alloc_bytes,
                    alloc_count,
                    alloc_bytes_exact,
                    alloc_count_exact,
                    duration_ns,
                    error_message: None,
                    failure_kind: None,
                    signal: None,
                    backtrace: None,
                    severity: bench.severity,
                    threshold: bench.threshold,
                    throughput: resolve_throughput(bench_result.throughput, bench, &group),
                }
            }
            Ok(Err(error)) => error_result(
                bench,
                BenchmarkStatus::Failed,
                error.to_string(),
                "unknown",
                duration_ns,
            ),
            Err(panic) => {
                let message = if let Some(s) = panic.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "Unknown panic".to_string()
                };
                error_result(
                    bench,
                    BenchmarkStatus::Crashed,
                    message,
                    "panic",
                    duration_ns,
                )
            }
        }
    }
}

/// Result with no samples for a benchmark that did not finish.
///
/// `kind` matches the isolated-mode failure kind names.
pub(super) fn error_result(
    bench: &BenchmarkDef,
    status: BenchmarkStatus,
    message: String,
    kind: &str,
    duration_ns: u64,
) -> BenchExecutionResult {
    BenchExecutionResult {
        benchmark_id: bench.id.to_string(),
        benchmark_name: bench.name.to_string(),
        group: bench.group.to_string(),
        file: bench.file.to_string(),
        line: bench.line,
        status,
        samples: Vec::new(),
        cpu_cycles: Vec::new(),
        alloc_bytes: 0,
        alloc_count: 0,
        alloc_bytes_exact: 0.0,
        alloc_count_exact: 0.0,
        duration_ns,
        error_message: Some(message),
        failure_kind: Some(kind.to_string()),
        signal: None,
        backtrace: None,
        severity: bench.severity,
        threshold: bench.threshold,
        throughput: None,
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// Minimal benchmark definition for executor tests.
    pub(in crate::executor) fn bench_def() -> BenchmarkDef {
        crate::test_support::bench_def("b")
    }

    /// No `#[flux::group]` config.
    const NO_GROUP: &GroupConfig = &GroupConfig::NONE;

    #[test]
    fn exact_allocations_keep_fractions() {
        let samples = [
            Sample::new(10.0, 0.0, 0.0, 0.0, 4),
            Sample::new(10.0, 0.0, 1.0, 0.25, 4),
        ];
        assert_eq!(exact_allocations(&samples), (0.5, 0.125));
        assert_eq!(mean_allocations(&samples), (1, 0));
    }

    #[test]
    fn resolve_keeps_global_sample_count() {
        let global = ExecutionConfig {
            sample_count: 300,
            ..Default::default()
        };
        let cfg = global.resolve_for_benchmark(&bench_def(), NO_GROUP);
        assert_eq!(cfg.sample_count, 300);
        assert_eq!(cfg.iters_per_sample, None);
        assert_eq!(cfg.warmup_time_ns, global.warmup_time_ns);
    }

    #[test]
    fn resolve_bench_sample_count_overrides_only_the_count() {
        let global = ExecutionConfig {
            sample_count: 300,
            ..Default::default()
        };
        let bench = BenchmarkDef {
            sample_count: Some(40),
            ..bench_def()
        };
        let cfg = global.resolve_for_benchmark(&bench, NO_GROUP);
        assert_eq!(cfg.sample_count, 40);
        assert_eq!(cfg.iters_per_sample, None);
        assert_eq!(cfg.warmup_time_ns, global.warmup_time_ns);
        assert_eq!(cfg.min_iterations, global.min_iterations);
    }

    #[test]
    fn resolve_bench_samples_selects_fixed_mode() {
        let bench = BenchmarkDef {
            samples: Some(500),
            ..bench_def()
        };
        let cfg = ExecutionConfig::default().resolve_for_benchmark(&bench, NO_GROUP);
        assert_eq!(cfg.sample_count, 500);
        assert_eq!(cfg.iters_per_sample, Some(1));
        assert_eq!(cfg.warmup_time_ns, 0);
        assert_eq!(cfg.min_iterations, None);
        assert_eq!(cfg.max_iterations, None);
    }

    #[test]
    fn resolve_bench_sample_count_beats_bench_samples() {
        let bench = BenchmarkDef {
            samples: Some(500),
            sample_count: Some(20),
            ..bench_def()
        };
        let cfg = ExecutionConfig::default().resolve_for_benchmark(&bench, NO_GROUP);
        assert_eq!(cfg.sample_count, 20);
        assert_eq!(cfg.iters_per_sample, Some(1));
    }

    #[test]
    fn resolve_bench_beats_global_fixed_mode() {
        let global = ExecutionConfig {
            sample_count: 7,
            iters_per_sample: Some(1),
            warmup_time_ns: 0,
            ..Default::default()
        };
        let bench = BenchmarkDef {
            sample_count: Some(250),
            iters_per_sample: Some(4),
            ..bench_def()
        };
        let cfg = global.resolve_for_benchmark(&bench, NO_GROUP);
        assert_eq!(cfg.sample_count, 250);
        assert_eq!(cfg.iters_per_sample, Some(4));
    }

    #[test]
    fn resolve_group_beats_global() {
        let global = ExecutionConfig {
            sample_count: 300,
            ..Default::default()
        };
        let group = GroupConfig {
            sample_count: Some(30),
            warmup_ns: Some(500_000_000),
            measurement_ns: Some(2_000_000_000),
            min_iterations: Some(7),
            max_iterations: Some(70),
            ..GroupConfig::NONE
        };
        let cfg = global.resolve_for_benchmark(&bench_def(), &group);
        assert_eq!(cfg.sample_count, 30);
        assert_eq!(cfg.warmup_time_ns, 500_000_000);
        assert_eq!(cfg.measurement_time_ns, 2_000_000_000);
        assert_eq!(cfg.min_iterations, Some(7));
        assert_eq!(cfg.max_iterations, Some(70));
        assert_eq!(cfg.iters_per_sample, None);
    }

    #[test]
    fn resolve_bench_beats_group() {
        let group = GroupConfig {
            sample_count: Some(30),
            warmup_ns: Some(500_000_000),
            measurement_ns: Some(2_000_000_000),
            ..GroupConfig::NONE
        };
        let bench = BenchmarkDef {
            sample_count: Some(12),
            measurement_ns: Some(1_000_000_000),
            ..bench_def()
        };
        let cfg = ExecutionConfig::default().resolve_for_benchmark(&bench, &group);
        assert_eq!(cfg.sample_count, 12);
        assert_eq!(cfg.measurement_time_ns, 1_000_000_000);
        // The bench leaves warmup unset, so the group value applies.
        assert_eq!(cfg.warmup_time_ns, 500_000_000);
    }

    #[test]
    fn resolve_bench_samples_beats_group_count_and_mode() {
        let group = GroupConfig {
            sample_count: Some(30),
            iters_per_sample: Some(8),
            ..GroupConfig::NONE
        };
        let bench = BenchmarkDef {
            samples: Some(500),
            ..bench_def()
        };
        let cfg = ExecutionConfig::default().resolve_for_benchmark(&bench, &group);
        assert_eq!(cfg.sample_count, 500);
        assert_eq!(cfg.iters_per_sample, Some(1));
    }

    #[test]
    fn resolve_group_fixed_mode_drops_warmup() {
        let group = GroupConfig {
            iters_per_sample: Some(8),
            warmup_ns: Some(500_000_000),
            min_iterations: Some(7),
            ..GroupConfig::NONE
        };
        let cfg = ExecutionConfig::default().resolve_for_benchmark(&bench_def(), &group);
        assert_eq!(cfg.iters_per_sample, Some(8));
        assert_eq!(cfg.warmup_time_ns, 0);
        assert_eq!(cfg.min_iterations, None);
    }

    #[test]
    fn executor_applies_group_config_and_throughput() {
        let bench = BenchmarkDef {
            group: "io",
            runner_fn: |b| b.iter(|| 1_u64),
            ..bench_def()
        };
        let mut groups = GroupConfigs::default();
        groups.insert(
            "io",
            GroupConfig {
                sample_count: Some(4),
                iters_per_sample: Some(2),
                throughput: Some(Throughput::Elements(5)),
                ..GroupConfig::NONE
            },
        );
        let executor = Executor::new(ExecutionConfig {
            bootstrap_iterations: 0,
            ..Default::default()
        })
        .with_group_configs(groups);
        let result = executor.execute_single(&bench);
        assert_eq!(result.status, BenchmarkStatus::Passed);
        assert_eq!(result.samples.len(), 4);
        assert_eq!(result.throughput, Some(Throughput::Elements(5)));
    }

    #[test]
    fn body_throughput_beats_bench_attribute_in_process() {
        let bench = BenchmarkDef {
            iters_per_sample: Some(1),
            throughput: Some(Throughput::Bytes(64)),
            runner_fn: |b| {
                b.throughput(Throughput::Bytes(128));
                b.iter(|| 1_u64);
            },
            ..bench_def()
        };
        let result = Executor::new(ExecutionConfig::default()).execute_single(&bench);
        assert_eq!(result.throughput, Some(Throughput::Bytes(128)));
    }

    #[test]
    fn ipc_config_carries_sample_fields() {
        let cfg = ExecutionConfig {
            sample_count: 250,
            iters_per_sample: Some(2),
            warmup_time_ns: 0,
            ..Default::default()
        };
        let ipc = cfg.to_ipc_config(Duration::from_secs(60));
        assert_eq!(ipc.sample_count, 250);
        assert_eq!(ipc.iters_per_sample, Some(2));
        assert!(ipc.validate().is_ok());
    }

    #[test]
    fn measure_config_carries_execution_fields() {
        let cfg = ExecutionConfig {
            warmup_time_ns: 11,
            measurement_time_ns: 22,
            min_iterations: Some(3),
            max_iterations: Some(4),
            sample_count: 250,
            iters_per_sample: Some(5),
            track_allocations: false,
            ..Default::default()
        };
        let measure = cfg.to_measure_config();
        assert_eq!(measure.warmup_ns, 11);
        assert_eq!(measure.measurement_ns, 22);
        assert_eq!(measure.min_iterations, Some(3));
        assert_eq!(measure.max_iterations, Some(4));
        assert_eq!(measure.sample_count, 250);
        assert_eq!(measure.fixed_iters_per_sample, Some(5));
        assert!(!measure.track_allocations);
    }

    #[test]
    fn body_without_iter_fails_in_process() {
        let executor = Executor::new(ExecutionConfig {
            iters_per_sample: Some(1),
            warmup_time_ns: 0,
            ..Default::default()
        });
        let result = executor.execute_single(&bench_def());
        assert_eq!(result.status, BenchmarkStatus::Failed);
        assert_eq!(
            result.error_message.as_deref(),
            Some("benchmark body never called b.iter*")
        );
        assert_eq!(result.failure_kind.as_deref(), Some("unknown"));
        assert!(result.samples.is_empty());
    }

    #[test]
    fn mean_allocations_empty_is_zero() {
        assert_eq!(mean_allocations(&[]), (0, 0));
    }

    #[test]
    fn mean_allocations_single_sample() {
        let samples = [Sample::new(10.0, 0.0, 512.0, 5.0, 1)];
        assert_eq!(mean_allocations(&samples), (512, 5));
    }

    #[test]
    fn mean_allocations_rounds_fractional_means() {
        let samples = [
            Sample::new(10.0, 0.0, 100.25, 0.5, 4),
            Sample::new(10.0, 0.0, 100.75, 1.0, 4),
        ];
        assert_eq!(mean_allocations(&samples), (101, 1));
    }

    #[test]
    fn mean_allocations_is_mean_not_sum() {
        let samples = [
            Sample::new(10.0, 0.0, 100.0, 2.0, 1),
            Sample::new(10.0, 0.0, 200.0, 4.0, 1),
            Sample::new(10.0, 0.0, 300.0, 6.0, 1),
        ];
        assert_eq!(mean_allocations(&samples), (200, 4));
    }
}
