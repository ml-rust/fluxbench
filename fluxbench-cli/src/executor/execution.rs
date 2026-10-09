//! Benchmark Execution
//!
//! Core execution logic for running benchmarks, including both in-process
//! and isolated (IPC-based) execution modes.
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

use crate::supervisor::{IpcBenchmarkResult, IpcBenchmarkStatus, Supervisor};
use fluxbench_core::{Bencher, BenchmarkDef, run_benchmark_loop};
use fluxbench_ipc::{BenchmarkConfig, Sample};
use fluxbench_report::BenchmarkStatus;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::{Duration, Instant};

/// Mean per-iteration allocation bytes and count across samples, rounded.
///
/// Each sample already holds a per-iteration mean. Returns `(0, 0)` for no samples.
fn mean_allocations(samples: &[Sample]) -> (u64, u64) {
    if samples.is_empty() {
        return (0, 0);
    }
    let n = samples.len() as f64;
    let bytes: f64 = samples.iter().map(|s| s.alloc_bytes).sum();
    let count: f64 = samples.iter().map(|s| s.alloc_count).sum();
    ((bytes / n).round() as u64, (count / n).round() as u64)
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
    /// Merge per-benchmark configuration overrides with global defaults.
    ///
    /// Priority:
    /// 1. Per-benchmark `samples = N`: fixed mode, N samples of one iteration, no warmup
    /// 2. Per-benchmark `sample_count`: overrides only the count
    /// 3. Per-benchmark `iters_per_sample`: selects fixed mode with that many iterations
    /// 4. Per-benchmark `warmup_ns`/`measurement_ns`/`min/max_iterations`: override global values
    /// 5. Falls back to global config for anything not overridden
    ///
    /// Fixed mode sets warmup to 0 and clears min/max iterations.
    pub fn resolve_for_benchmark(&self, bench: &BenchmarkDef) -> ExecutionConfig {
        let sample_count = bench
            .sample_count
            .or(bench.samples)
            .map_or(self.sample_count, |n| n as usize);
        let iters_per_sample = bench
            .iters_per_sample
            .or(bench.samples.map(|_| 1))
            .or(self.iters_per_sample);

        if iters_per_sample.is_some() {
            return ExecutionConfig {
                warmup_time_ns: 0,
                measurement_time_ns: bench.measurement_ns.unwrap_or(self.measurement_time_ns),
                min_iterations: None,
                max_iterations: None,
                sample_count,
                iters_per_sample,
                ..self.clone()
            };
        }

        ExecutionConfig {
            warmup_time_ns: bench.warmup_ns.unwrap_or(self.warmup_time_ns),
            measurement_time_ns: bench.measurement_ns.unwrap_or(self.measurement_time_ns),
            min_iterations: bench.min_iterations.or(self.min_iterations),
            max_iterations: bench.max_iterations.or(self.max_iterations),
            sample_count,
            iters_per_sample: None,
            ..self.clone()
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
            fail_on_allocation: false,
            timeout_ns: timeout.as_nanos() as u64,
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
    pub duration_ns: u64,
    pub error_message: Option<String>,
    pub failure_kind: Option<String>,
    pub backtrace: Option<String>,
    pub severity: fluxbench_core::Severity,
    /// Per-benchmark regression threshold (0.0 = use global)
    pub threshold: f64,
}

/// Execute benchmarks and produce results (in-process mode)
pub struct Executor {
    config: ExecutionConfig,
    results: Vec<BenchExecutionResult>,
}

impl Executor {
    /// Create a new in-process executor with the given configuration
    pub fn new(config: ExecutionConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
        }
    }

    /// Execute all provided benchmarks
    pub fn execute(&mut self, benchmarks: &[&BenchmarkDef]) -> Vec<BenchExecutionResult> {
        let pb = ProgressBar::new(benchmarks.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
                )
                .unwrap_or_else(|_| ProgressStyle::default_bar())
                .progress_chars("#>-"),
        );

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
        let cfg = self.config.resolve_for_benchmark(bench);

        // Run with panic catching
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let bencher = Bencher::for_config(
                cfg.track_allocations,
                cfg.sample_count,
                cfg.iters_per_sample,
            );

            run_benchmark_loop(
                bencher,
                |b| (bench.runner_fn)(b),
                cfg.warmup_time_ns,
                cfg.measurement_time_ns,
                cfg.min_iterations,
                cfg.max_iterations,
            )
        }));

        let duration_ns = start.elapsed().as_nanos() as u64;

        match result {
            Ok(bench_result) => {
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
                    duration_ns,
                    error_message: None,
                    failure_kind: None,
                    backtrace: None,
                    severity: bench.severity,
                    threshold: bench.threshold,
                }
            }
            Err(panic) => {
                let message = if let Some(s) = panic.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "Unknown panic".to_string()
                };

                BenchExecutionResult {
                    benchmark_id: bench.id.to_string(),
                    benchmark_name: bench.name.to_string(),
                    group: bench.group.to_string(),
                    file: bench.file.to_string(),
                    line: bench.line,
                    status: BenchmarkStatus::Crashed,
                    samples: Vec::new(),
                    cpu_cycles: Vec::new(),
                    alloc_bytes: 0,
                    alloc_count: 0,
                    duration_ns,
                    error_message: Some(message),
                    failure_kind: Some("panic".to_string()),
                    backtrace: None,
                    severity: bench.severity,
                    threshold: bench.threshold,
                }
            }
        }
    }
}

/// Executor that runs benchmarks in isolated worker processes via IPC
///
/// This provides crash isolation - if a benchmark panics or crashes,
/// it won't take down the supervisor process.
pub struct IsolatedExecutor {
    config: ExecutionConfig,
    timeout: Duration,
    reuse_workers: bool,
    num_workers: usize,
}

impl IsolatedExecutor {
    /// Create a new isolated executor
    pub fn new(
        config: ExecutionConfig,
        timeout: Duration,
        reuse_workers: bool,
        num_workers: usize,
    ) -> Self {
        Self {
            config,
            timeout,
            reuse_workers,
            num_workers: num_workers.max(1),
        }
    }

    /// Execute all provided benchmarks in isolated worker processes
    pub fn execute(&self, benchmarks: &[&BenchmarkDef]) -> Vec<BenchExecutionResult> {
        let pb = ProgressBar::new(benchmarks.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
                )
                .unwrap_or_else(|_| ProgressStyle::default_bar())
                .progress_chars("#>-"),
        );
        pb.set_message("Starting isolated workers...");

        // Build per-benchmark IPC configs
        let ipc_configs: Vec<BenchmarkConfig> = benchmarks
            .iter()
            .map(|bench| {
                self.config
                    .resolve_for_benchmark(bench)
                    .to_ipc_config(self.timeout)
            })
            .collect();

        // Use the first config as default for the supervisor
        let default_config = ipc_configs
            .first()
            .cloned()
            .unwrap_or_else(|| self.config.to_ipc_config(self.timeout));

        let supervisor = Supervisor::new(default_config, self.timeout, self.num_workers);

        // Run benchmarks via IPC with per-benchmark configs
        let ipc_results = if self.reuse_workers {
            supervisor.run_with_reuse_configs(benchmarks, &ipc_configs)
        } else {
            supervisor.run_all_configs(benchmarks, &ipc_configs)
        };

        // Convert IPC results to BenchExecutionResult
        let mut results = Vec::with_capacity(benchmarks.len());

        match ipc_results {
            Ok(ipc_results) => {
                for (ipc_result, bench) in ipc_results.into_iter().zip(benchmarks.iter()) {
                    pb.set_message(bench.id.to_string());
                    results.push(self.convert_ipc_result(ipc_result, bench));
                    pb.inc(1);
                }
            }
            Err(e) => {
                // Supervisor-level failure - mark all as crashed
                for bench in benchmarks {
                    results.push(BenchExecutionResult {
                        benchmark_id: bench.id.to_string(),
                        benchmark_name: bench.name.to_string(),
                        group: bench.group.to_string(),
                        file: bench.file.to_string(),
                        line: bench.line,
                        status: BenchmarkStatus::Crashed,
                        samples: Vec::new(),
                        cpu_cycles: Vec::new(),
                        alloc_bytes: 0,
                        alloc_count: 0,
                        duration_ns: 0,
                        error_message: Some(format!("Supervisor error: {}", e)),
                        failure_kind: Some("crashed".to_string()),
                        backtrace: None,
                        severity: bench.severity,
                        threshold: bench.threshold,
                    });
                    pb.inc(1);
                }
            }
        }

        pb.finish_with_message("Complete (isolated)");
        results
    }

    /// Convert an IPC result to a BenchExecutionResult
    fn convert_ipc_result(
        &self,
        ipc_result: IpcBenchmarkResult,
        bench: &BenchmarkDef,
    ) -> BenchExecutionResult {
        let (status, error_message, failure_kind, backtrace) = match ipc_result.status {
            IpcBenchmarkStatus::Success => (BenchmarkStatus::Passed, None, None, None),
            IpcBenchmarkStatus::Failed {
                message,
                kind,
                backtrace,
            } => (
                BenchmarkStatus::Failed,
                Some(message),
                Some(kind),
                backtrace,
            ),
            IpcBenchmarkStatus::Crashed {
                message,
                kind,
                backtrace,
            } => (
                BenchmarkStatus::Crashed,
                Some(message),
                Some(kind),
                backtrace,
            ),
        };

        // Extract timing samples as f64 for statistics
        let samples: Vec<f64> = ipc_result
            .samples
            .iter()
            .map(|s| s.duration_nanos)
            .collect();

        // Extract CPU cycles
        let cpu_cycles: Vec<f64> = ipc_result.samples.iter().map(|s| s.cpu_cycles).collect();

        let (alloc_bytes, alloc_count) = mean_allocations(&ipc_result.samples);

        BenchExecutionResult {
            benchmark_id: bench.id.to_string(),
            benchmark_name: bench.name.to_string(),
            group: bench.group.to_string(),
            file: bench.file.to_string(),
            line: bench.line,
            status,
            samples,
            cpu_cycles,
            alloc_bytes,
            alloc_count,
            duration_ns: ipc_result.total_duration_nanos,
            error_message,
            failure_kind,
            backtrace,
            severity: bench.severity,
            threshold: bench.threshold,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxbench_core::Severity;

    fn bench_def() -> BenchmarkDef {
        BenchmarkDef {
            id: "b",
            name: "b",
            group: "default",
            severity: Severity::Info,
            threshold: 0.0,
            budget_ns: None,
            tags: &[],
            runner_fn: |_| {},
            file: "",
            line: 0,
            module_path: "",
            warmup_ns: None,
            measurement_ns: None,
            samples: None,
            sample_count: None,
            iters_per_sample: None,
            min_iterations: None,
            max_iterations: None,
            depends_on: &[],
        }
    }

    #[test]
    fn resolve_keeps_global_sample_count() {
        let global = ExecutionConfig {
            sample_count: 300,
            ..Default::default()
        };
        let cfg = global.resolve_for_benchmark(&bench_def());
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
        let cfg = global.resolve_for_benchmark(&bench);
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
        let cfg = ExecutionConfig::default().resolve_for_benchmark(&bench);
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
        let cfg = ExecutionConfig::default().resolve_for_benchmark(&bench);
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
        let cfg = global.resolve_for_benchmark(&bench);
        assert_eq!(cfg.sample_count, 250);
        assert_eq!(cfg.iters_per_sample, Some(4));
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
