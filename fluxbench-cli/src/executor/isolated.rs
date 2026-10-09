//! Isolated execution: run benchmarks in worker processes via IPC.
//!
//! A panic, signal, timeout, or lost worker fails only its own benchmark.

use super::execution::{
    BenchExecutionResult, ExecutionConfig, error_result, exact_allocations, mean_allocations,
    progress_bar,
};
use super::group_configs::GroupConfigs;
use super::throughput::resolve_throughput;
use crate::supervisor::{IpcBenchmarkResult, IpcBenchmarkStatus, Supervisor};
use fluxbench_core::{BenchmarkDef, GroupConfig};
use fluxbench_ipc::{BenchmarkConfig, FailureKind, Sample};
use fluxbench_report::BenchmarkStatus;
use std::time::Duration;

/// Executor that runs benchmarks in isolated worker processes via IPC
///
/// This provides crash isolation - if a benchmark panics or crashes,
/// it won't take down the supervisor process.
pub struct IsolatedExecutor {
    config: ExecutionConfig,
    timeout: Duration,
    reuse_workers: bool,
    num_workers: usize,
    cpus: Vec<Option<usize>>,
    groups: GroupConfigs,
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
            cpus: Vec::new(),
            groups: GroupConfigs::default(),
        }
    }

    /// Set the CPU each worker pins to. An empty list leaves workers unpinned.
    pub fn with_cpus(mut self, cpus: Vec<Option<usize>>) -> Self {
        self.cpus = cpus;
        self
    }

    /// Apply `#[flux::group]` run config. Without it, no group config applies.
    pub fn with_group_configs(mut self, groups: GroupConfigs) -> Self {
        self.groups = groups;
        self
    }

    /// Execute all provided benchmarks in isolated worker processes
    pub fn execute(&self, benchmarks: &[&BenchmarkDef]) -> Vec<BenchExecutionResult> {
        let pb = progress_bar(benchmarks.len());
        pb.set_message("Starting isolated workers...");

        // Build per-benchmark IPC configs
        let ipc_configs: Vec<BenchmarkConfig> = benchmarks
            .iter()
            .map(|bench| {
                self.config
                    .resolve_for_benchmark(bench, &self.groups.get(bench.group))
                    .to_ipc_config(self.timeout)
            })
            .collect();

        // Use the first config as default for the supervisor
        let default_config = ipc_configs
            .first()
            .cloned()
            .unwrap_or_else(|| self.config.to_ipc_config(self.timeout));

        let supervisor = Supervisor::new(default_config, self.timeout, self.num_workers)
            .with_cpus(self.cpus.clone());

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
                    let group = self.groups.get(bench.group);
                    results.push(convert_ipc_result(ipc_result, bench, &group));
                    pb.inc(1);
                }
            }
            Err(e) => {
                // The supervisor returns per-benchmark results and does not reach this arm today.
                // Keep every benchmark in the report if it ever does.
                for bench in benchmarks {
                    results.push(error_result(
                        bench,
                        BenchmarkStatus::Crashed,
                        format!("supervisor error: {e}. Run again with --isolated=false to debug."),
                        FailureKind::Crashed.as_str(),
                        0,
                    ));
                    pb.inc(1);
                }
            }
        }

        pb.finish_with_message("Complete (isolated)");
        results
    }
}

/// Convert an IPC result to a BenchExecutionResult.
///
/// Only a successful run feeds statistics.
/// A failed run's partial samples are dropped here. Its message states their count.
/// `group` is the merged group config. It is the last source of throughput.
fn convert_ipc_result(
    ipc_result: IpcBenchmarkResult,
    bench: &BenchmarkDef,
    group: &GroupConfig,
) -> BenchExecutionResult {
    let (status, error_message, kind, backtrace) = match ipc_result.status {
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
    let failure_kind = kind.as_ref().map(|k| k.as_str().to_string());
    let signal = kind
        .as_ref()
        .and_then(|k| k.signal_name())
        .map(str::to_string);

    let measured: &[Sample] = if status == BenchmarkStatus::Passed {
        &ipc_result.samples
    } else {
        &[]
    };

    // Extract timing samples as f64 for statistics
    let samples: Vec<f64> = measured.iter().map(|s| s.duration_nanos).collect();

    // Extract CPU cycles
    let cpu_cycles: Vec<f64> = measured.iter().map(|s| s.cpu_cycles).collect();

    let (alloc_bytes, alloc_count) = mean_allocations(measured);
    let (alloc_bytes_exact, alloc_count_exact) = exact_allocations(measured);
    let throughput = if status == BenchmarkStatus::Passed {
        resolve_throughput(ipc_result.throughput, bench, group)
    } else {
        None
    };

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
        alloc_bytes_exact,
        alloc_count_exact,
        duration_ns: ipc_result.total_duration_nanos,
        error_message,
        failure_kind,
        signal,
        backtrace,
        severity: bench.severity,
        threshold: bench.threshold,
        throughput,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::execution::tests::bench_def;
    use fluxbench_core::Throughput;

    fn crashed(kind: FailureKind, message: &str, samples: Vec<Sample>) -> IpcBenchmarkResult {
        IpcBenchmarkResult {
            bench_id: "b".to_string(),
            samples,
            total_iterations: 0,
            total_duration_nanos: 0,
            throughput: None,
            status: IpcBenchmarkStatus::Crashed {
                message: message.to_string(),
                kind,
                backtrace: None,
            },
        }
    }

    fn success(throughput: Option<Throughput>) -> IpcBenchmarkResult {
        IpcBenchmarkResult {
            bench_id: "b".to_string(),
            samples: vec![Sample::timing_only(5.0); 4],
            total_iterations: 4,
            total_duration_nanos: 20,
            throughput,
            status: IpcBenchmarkStatus::Success,
        }
    }

    #[test]
    fn timeout_with_partial_samples_is_not_a_pass() {
        let ipc = crashed(
            FailureKind::Timeout,
            "benchmark exceeded the 3s worker timeout; 3 partial samples received",
            vec![Sample::timing_only(5.0); 3],
        );
        let result = convert_ipc_result(ipc, &bench_def(), &GroupConfig::NONE);
        assert_eq!(result.status, BenchmarkStatus::Crashed);
        assert_eq!(result.failure_kind.as_deref(), Some("timeout"));
        assert!(result.samples.is_empty());
        assert!(result.cpu_cycles.is_empty());
        assert!(
            result
                .error_message
                .as_deref()
                .is_some_and(|m| m.contains("3 partial samples"))
        );
    }

    #[test]
    fn signal_kind_carries_signal_name() {
        let ipc = crashed(
            FailureKind::Signal("SIGSEGV".to_string()),
            "worker killed by SIGSEGV",
            Vec::new(),
        );
        let result = convert_ipc_result(ipc, &bench_def(), &GroupConfig::NONE);
        assert_eq!(result.failure_kind.as_deref(), Some("signal"));
        assert_eq!(result.signal.as_deref(), Some("SIGSEGV"));
    }

    #[test]
    fn success_keeps_samples() {
        let result = convert_ipc_result(success(None), &bench_def(), &GroupConfig::NONE);
        assert_eq!(result.status, BenchmarkStatus::Passed);
        assert_eq!(result.samples.len(), 4);
        assert_eq!(result.failure_kind, None);
        assert_eq!(result.throughput, None);
    }

    #[test]
    fn worker_throughput_beats_bench_and_group() {
        let bench = BenchmarkDef {
            throughput: Some(Throughput::Bytes(2)),
            ..bench_def()
        };
        let group = GroupConfig {
            throughput: Some(Throughput::Bytes(1)),
            ..GroupConfig::NONE
        };
        let from_worker = Some(Throughput::Elements(3));
        let result = convert_ipc_result(success(from_worker), &bench, &group);
        assert_eq!(result.throughput, from_worker);
        let result = convert_ipc_result(success(None), &bench, &group);
        assert_eq!(result.throughput, Some(Throughput::Bytes(2)));
        let result = convert_ipc_result(success(None), &bench_def(), &group);
        assert_eq!(result.throughput, Some(Throughput::Bytes(1)));
    }

    #[test]
    fn failed_run_has_no_throughput() {
        let bench = BenchmarkDef {
            throughput: Some(Throughput::Bytes(2)),
            ..bench_def()
        };
        let ipc = crashed(FailureKind::Panic, "boom", Vec::new());
        let result = convert_ipc_result(ipc, &bench, &GroupConfig::NONE);
        assert_eq!(result.throughput, None);
    }
}
