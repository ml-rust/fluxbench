//! Per-benchmark result of an isolated run.

use fluxbench_ipc::{FailureKind, Sample};

/// Result from a benchmark run via IPC
#[derive(Debug)]
pub struct IpcBenchmarkResult {
    /// Benchmark identifier.
    pub bench_id: String,
    /// Samples the worker sent.
    /// A failed run keeps the partial samples it sent before the failure.
    pub samples: Vec<Sample>,
    /// Measurement-phase routine calls. Warmup calls are excluded. 0 on failure.
    pub total_iterations: u64,
    /// Total benchmark duration in nanoseconds. 0 on failure.
    pub total_duration_nanos: u64,
    /// Status of the benchmark execution.
    pub status: IpcBenchmarkStatus,
}

/// Status of a benchmark execution
#[derive(Debug, Clone)]
pub enum IpcBenchmarkStatus {
    /// Benchmark completed successfully.
    Success,
    /// Benchmark failed with an error, not a crash.
    Failed {
        /// Error message.
        message: String,
        /// Error kind. `FailureKind::as_str` gives the report string.
        kind: FailureKind,
        /// Optional backtrace from the error.
        backtrace: Option<String>,
    },
    /// Benchmark crashed: panic, signal, timeout, or lost worker.
    Crashed {
        /// Error message.
        message: String,
        /// Crash kind. `FailureKind::as_str` gives the report string.
        kind: FailureKind,
        /// Backtrace captured at panic time, when there was a panic.
        backtrace: Option<String>,
    },
}

impl IpcBenchmarkStatus {
    /// Failure kind, or `None` on success.
    pub fn kind(&self) -> Option<&FailureKind> {
        match self {
            IpcBenchmarkStatus::Success => None,
            IpcBenchmarkStatus::Failed { kind, .. } | IpcBenchmarkStatus::Crashed { kind, .. } => {
                Some(kind)
            }
        }
    }
}

impl IpcBenchmarkResult {
    /// Failed result for `bench_id`.
    ///
    /// `kind.is_crash()` selects `Crashed`, else `Failed`.
    /// `samples` holds what the worker sent before the failure.
    pub(crate) fn failure(
        bench_id: &str,
        kind: FailureKind,
        message: String,
        backtrace: Option<String>,
        samples: Vec<Sample>,
    ) -> Self {
        let status = if kind.is_crash() {
            IpcBenchmarkStatus::Crashed {
                message,
                kind,
                backtrace,
            }
        } else {
            IpcBenchmarkStatus::Failed {
                message,
                kind,
                backtrace,
            }
        };
        Self {
            bench_id: bench_id.to_string(),
            samples,
            total_iterations: 0,
            total_duration_nanos: 0,
            status,
        }
    }

    /// Crashed result with no samples, for a run that never produced data.
    pub(crate) fn crashed(bench_id: &str, message: String) -> Self {
        Self::failure(bench_id, FailureKind::Crashed, message, None, Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crash_kinds_map_to_crashed() {
        for kind in [
            FailureKind::Panic,
            FailureKind::Timeout,
            FailureKind::Signal("SIGSEGV".to_string()),
            FailureKind::Crashed,
        ] {
            let result = IpcBenchmarkResult::failure("b", kind.clone(), "m".into(), None, vec![]);
            assert!(matches!(result.status, IpcBenchmarkStatus::Crashed { .. }));
            assert_eq!(result.status.kind(), Some(&kind));
        }
    }

    #[test]
    fn other_kinds_map_to_failed() {
        let result =
            IpcBenchmarkResult::failure("b", FailureKind::Unknown, "m".into(), None, vec![]);
        assert!(matches!(result.status, IpcBenchmarkStatus::Failed { .. }));
    }

    #[test]
    fn failure_keeps_partial_samples() {
        let samples = vec![Sample::timing_only(1.0), Sample::timing_only(2.0)];
        let result =
            IpcBenchmarkResult::failure("b", FailureKind::Timeout, "m".into(), None, samples);
        assert_eq!(result.samples.len(), 2);
        assert_eq!(result.total_iterations, 0);
    }
}
