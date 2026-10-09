//! IPC Message Types
//!
//! All messages are serialized with rkyv for zero-copy deserialization.

use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};

/// One measurement sample (40 bytes, 8-byte aligned).
///
/// Each field except `iterations` holds a per-iteration mean over the sample.
/// `iterations` counts the routine calls in the sample.
///
/// # Layout history
/// - Protocol v1: 32 bytes of integer per-iteration means.
/// - Protocol v2: 40 bytes of `f64` means plus the `iterations` count.
#[derive(Debug, Clone, Copy, PartialEq, Archive, RkyvSerialize, RkyvDeserialize)]
#[repr(C, align(8))]
pub struct Sample {
    /// Mean duration per iteration in nanoseconds
    pub duration_nanos: f64,
    /// Mean CPU cycles per iteration (0 without a cycle counter)
    pub cpu_cycles: f64,
    /// Mean bytes allocated per iteration
    pub alloc_bytes: f64,
    /// Mean allocation count per iteration
    pub alloc_count: f64,
    /// Routine calls in this sample
    pub iterations: u64,
}

impl Sample {
    /// Create a sample from per-iteration means and the call count
    #[inline]
    pub fn new(
        duration_nanos: f64,
        cpu_cycles: f64,
        alloc_bytes: f64,
        alloc_count: f64,
        iterations: u64,
    ) -> Self {
        Self {
            duration_nanos,
            cpu_cycles,
            alloc_bytes,
            alloc_count,
            iterations,
        }
    }

    /// Create a timing-only sample of one call (no cycles, no allocations)
    #[inline]
    pub fn timing_only(duration_nanos: f64) -> Self {
        Self {
            duration_nanos,
            cpu_cycles: 0.0,
            alloc_bytes: 0.0,
            alloc_count: 0.0,
            iterations: 1,
        }
    }
}

/// Reason for flushing a sample batch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Archive, RkyvSerialize, RkyvDeserialize)]
pub enum FlushReason {
    /// Batch size limit reached (10K samples)
    BatchFull,
    /// Byte size limit reached (64KB)
    ByteLimitReached,
    /// Heartbeat timeout (100ms with no activity)
    HeartbeatTimeout,
    /// Benchmark completed normally
    BenchmarkComplete,
    /// Worker shutting down
    Shutdown,
}

/// A batch of samples sent from worker to supervisor
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct SampleBatch {
    /// Hash of the benchmark ID (for fast lookup without string comparison)
    pub bench_id_hash: u64,
    /// Sequence number for ordering and detecting dropped batches
    pub batch_sequence: u32,
    /// First iteration index in this batch
    pub start_iteration: u64,
    /// The actual samples
    pub samples: Vec<Sample>,
    /// Why this batch was flushed
    pub flush_reason: FlushReason,
}

impl SampleBatch {
    /// Estimated size in bytes for this batch
    pub fn estimated_size(&self) -> usize {
        // Header fields + samples
        std::mem::size_of::<u64>() * 2  // bench_id_hash, start_iteration
            + std::mem::size_of::<u32>()  // batch_sequence
            + std::mem::size_of::<FlushReason>()
            + self.samples.len() * std::mem::size_of::<Sample>()
    }
}

/// Worker capabilities advertised during handshake
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct WorkerCapabilities {
    /// Protocol version for compatibility
    pub protocol_version: u32,
    /// Whether RDTSC is available for high-precision timing
    pub has_rdtsc: bool,
    /// Whether allocation tracking is enabled
    pub has_alloc_tracking: bool,
    /// Whether CPU performance counters are available
    pub has_perf_counters: bool,
    /// Number of logical CPUs available
    pub cpu_count: u32,
    /// CPU model string (for reports)
    pub cpu_model: String,
}

impl Default for WorkerCapabilities {
    fn default() -> Self {
        Self {
            protocol_version: crate::PROTOCOL_VERSION,
            has_rdtsc: cfg!(target_arch = "x86_64"),
            has_alloc_tracking: true,
            has_perf_counters: false,
            cpu_count: num_cpus(),
            cpu_model: cpu_model_string(),
        }
    }
}

/// Messages sent from Worker to Supervisor
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize)]
pub enum WorkerMessage {
    /// Initial handshake with worker capabilities
    Hello(WorkerCapabilities),

    /// A batch of measurement samples
    SampleBatch(SampleBatch),

    /// Warmup phase completed, ready for measurement
    WarmupComplete {
        /// Number of warmup iterations performed
        iterations: u64,
        /// Total warmup time in nanoseconds
        duration_nanos: u64,
    },

    /// Benchmark completed successfully
    Complete {
        /// Measurement-phase routine calls. Warmup calls are excluded.
        total_iterations: u64,
        /// Sum of each sample's mean times its iterations, in nanoseconds
        total_duration_nanos: u64,
    },

    /// Benchmark failed with an error
    Failure {
        /// Error category
        kind: FailureKind,
        /// Human-readable error message
        message: String,
        /// Optional backtrace
        backtrace: Option<String>,
    },

    /// Progress update for long-running benchmarks
    Progress {
        /// Iterations completed so far
        completed: u64,
        /// Estimated total iterations
        estimated_total: u64,
    },
}

/// Categories of benchmark failures
///
/// `as_str` gives the report string for each kind.
/// The worker sends `Panic` and `Unknown`.
/// The supervisor sets `Timeout`, `Signal` and `Crashed` from what it sees.
#[derive(Debug, Clone, PartialEq, Eq, Archive, RkyvSerialize, RkyvDeserialize)]
pub enum FailureKind {
    /// Rust panic (caught). Report string: `panic`.
    Panic,
    /// The worker exceeded the worker timeout. Report string: `timeout`.
    Timeout,
    /// Assertion failure in verification. Report string: `assertion`.
    Assertion,
    /// Allocation limit exceeded. Report string: `allocation_limit`.
    AllocationLimit,
    /// A signal killed the worker. Holds the name, e.g. `SIGSEGV` or `signal 40`.
    /// Report string: `signal`.
    Signal(String),
    /// The worker exited, broke the IPC stream, or never started. Report string: `crashed`.
    Crashed,
    /// Unknown error. Report string: `unknown`.
    Unknown,
}

impl FailureKind {
    /// Report string for this kind. Strings for older kinds never change.
    pub fn as_str(&self) -> &'static str {
        match self {
            FailureKind::Panic => "panic",
            FailureKind::Timeout => "timeout",
            FailureKind::Assertion => "assertion",
            FailureKind::AllocationLimit => "allocation_limit",
            FailureKind::Signal(_) => "signal",
            FailureKind::Crashed => "crashed",
            FailureKind::Unknown => "unknown",
        }
    }

    /// Signal name for `Signal`, else `None`.
    pub fn signal_name(&self) -> Option<&str> {
        match self {
            FailureKind::Signal(name) => Some(name),
            _ => None,
        }
    }

    /// `true` when the run ended abnormally: panic, timeout, signal, or crash.
    ///
    /// A crash fails CI. Other kinds report a failed benchmark.
    pub fn is_crash(&self) -> bool {
        matches!(
            self,
            FailureKind::Panic
                | FailureKind::Timeout
                | FailureKind::Signal(_)
                | FailureKind::Crashed
        )
    }
}

/// Commands sent from Supervisor to Worker
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize)]
pub enum SupervisorCommand {
    /// Run a specific benchmark
    Run {
        /// Benchmark identifier
        bench_id: String,
        /// Configuration for this run
        config: BenchmarkConfig,
    },

    /// Abort the current benchmark
    Abort,

    /// Request graceful shutdown
    Shutdown,

    /// Ping for health check
    Ping,
}

/// Configuration for a benchmark run
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct BenchmarkConfig {
    /// Warmup time in nanoseconds
    pub warmup_time_ns: u64,
    /// Measurement time in nanoseconds
    pub measurement_time_ns: u64,
    /// Minimum total measurement iterations (overrides time-based)
    pub min_iterations: Option<u64>,
    /// Maximum total measurement iterations (cap)
    pub max_iterations: Option<u64>,
    /// Target number of samples
    pub sample_count: u32,
    /// Fixed iterations per sample. `Some` selects fixed mode with no warmup.
    pub iters_per_sample: Option<u64>,
    /// Whether to track allocations
    pub track_allocations: bool,
    /// Whether to fail on any allocation (for hot path testing)
    pub fail_on_allocation: bool,
    /// Timeout in nanoseconds (0 = no timeout)
    pub timeout_ns: u64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_time_ns: 3_000_000_000,      // 3 seconds
            measurement_time_ns: 5_000_000_000, // 5 seconds
            min_iterations: Some(100),
            max_iterations: None,
            sample_count: DEFAULT_SAMPLE_COUNT,
            iters_per_sample: None,
            track_allocations: true,
            fail_on_allocation: false,
            timeout_ns: 60_000_000_000, // 60 seconds
        }
    }
}

/// Default target sample count (matches Criterion)
pub const DEFAULT_SAMPLE_COUNT: u32 = 100;

impl BenchmarkConfig {
    /// Validate configuration values, returning a description of the first error found.
    pub fn validate(&self) -> Result<(), String> {
        if self.warmup_time_ns == 0 && self.iters_per_sample.is_none() {
            return Err("warmup_time_ns must be > 0 unless iters_per_sample is set".to_string());
        }
        if self.sample_count < 2 {
            return Err(format!(
                "sample_count must be >= 2, got {}",
                self.sample_count
            ));
        }
        if self.iters_per_sample == Some(0) {
            return Err("iters_per_sample must be > 0".to_string());
        }
        if self.measurement_time_ns == 0 {
            return Err("measurement_time_ns must be > 0".to_string());
        }
        if self.timeout_ns > 0 && self.timeout_ns < self.measurement_time_ns {
            return Err(format!(
                "timeout_ns ({}) must be >= measurement_time_ns ({})",
                self.timeout_ns, self.measurement_time_ns
            ));
        }
        if let (Some(min), Some(max)) = (self.min_iterations, self.max_iterations) {
            if max < min {
                return Err(format!(
                    "max_iterations ({}) must be >= min_iterations ({})",
                    max, min
                ));
            }
        }
        Ok(())
    }
}

// Helper functions

fn num_cpus() -> u32 {
    std::thread::available_parallelism()
        .map(|p| p.get() as u32)
        .unwrap_or(1)
}

fn cpu_model_string() -> String {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|content| {
                content
                    .lines()
                    .find(|line| line.starts_with("model name"))
                    .and_then(|line| line.split(':').nth(1))
                    .map(|s| s.trim().to_string())
            })
            .unwrap_or_else(|| "Unknown CPU".to_string())
    }

    #[cfg(not(target_os = "linux"))]
    {
        "Unknown CPU".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rkyv::rancor;

    #[test]
    fn test_sample_creation() {
        let sample = Sample::new(1000.5, 4000.25, 512.0, 5.5, 7);
        assert_eq!(sample.duration_nanos, 1000.5);
        assert_eq!(sample.cpu_cycles, 4000.25);
        assert_eq!(sample.alloc_bytes, 512.0);
        assert_eq!(sample.alloc_count, 5.5);
        assert_eq!(sample.iterations, 7);
    }

    #[test]
    fn test_sample_timing_only() {
        let sample = Sample::timing_only(5000.0);
        assert_eq!(sample.duration_nanos, 5000.0);
        assert_eq!(sample.alloc_bytes, 0.0);
        assert_eq!(sample.alloc_count, 0.0);
        assert_eq!(sample.cpu_cycles, 0.0);
        assert_eq!(sample.iterations, 1);
    }

    #[test]
    fn test_sample_size_is_40_bytes() {
        assert_eq!(std::mem::size_of::<Sample>(), 40);
    }

    #[test]
    fn test_sample_rkyv_roundtrip() {
        let original = Sample::new(12.75, 48.5, 1024.0, 2.25, 4096);
        let bytes = rkyv::to_bytes::<rancor::Error>(&original).unwrap();
        let decoded = rkyv::from_bytes::<Sample, rancor::Error>(&bytes).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_benchmark_config_rkyv_roundtrip() {
        let original = BenchmarkConfig {
            sample_count: 250,
            iters_per_sample: Some(3),
            warmup_time_ns: 0,
            ..Default::default()
        };
        let bytes = rkyv::to_bytes::<rancor::Error>(&original).unwrap();
        let decoded = rkyv::from_bytes::<BenchmarkConfig, rancor::Error>(&bytes).unwrap();
        assert_eq!(decoded.sample_count, 250);
        assert_eq!(decoded.iters_per_sample, Some(3));
        assert_eq!(decoded.warmup_time_ns, 0);
        assert_eq!(decoded.measurement_time_ns, original.measurement_time_ns);
    }

    #[test]
    fn test_benchmark_config_validate_default() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.sample_count, DEFAULT_SAMPLE_COUNT);
        assert_eq!(config.iters_per_sample, None);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_benchmark_config_validate_zero_warmup() {
        let config = BenchmarkConfig {
            warmup_time_ns: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_benchmark_config_validate_fixed_mode_zero_warmup() {
        let config = BenchmarkConfig {
            warmup_time_ns: 0,
            iters_per_sample: Some(1),
            sample_count: 500,
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_benchmark_config_validate_zero_iters_per_sample() {
        let config = BenchmarkConfig {
            iters_per_sample: Some(0),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_benchmark_config_validate_sample_count_floor() {
        for sample_count in [0, 1] {
            let config = BenchmarkConfig {
                sample_count,
                ..Default::default()
            };
            assert!(config.validate().is_err());
        }
        let config = BenchmarkConfig {
            sample_count: 2,
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_benchmark_config_validate_timeout_lt_measurement() {
        let config = BenchmarkConfig {
            timeout_ns: 1_000_000,              // 1ms
            measurement_time_ns: 5_000_000_000, // 5s
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_benchmark_config_validate_max_lt_min() {
        let config = BenchmarkConfig {
            min_iterations: Some(200),
            max_iterations: Some(50),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn failure_kind_strings_are_stable() {
        let cases = [
            (FailureKind::Panic, "panic"),
            (FailureKind::Timeout, "timeout"),
            (FailureKind::Assertion, "assertion"),
            (FailureKind::AllocationLimit, "allocation_limit"),
            (FailureKind::Signal("SIGSEGV".to_string()), "signal"),
            (FailureKind::Crashed, "crashed"),
            (FailureKind::Unknown, "unknown"),
        ];
        for (kind, text) in cases {
            assert_eq!(kind.as_str(), text);
        }
    }

    #[test]
    fn failure_kind_signal_name_and_crash_class() {
        let segv = FailureKind::Signal("SIGSEGV".to_string());
        assert_eq!(segv.signal_name(), Some("SIGSEGV"));
        assert_eq!(FailureKind::Timeout.signal_name(), None);
        assert!(segv.is_crash());
        assert!(FailureKind::Panic.is_crash());
        assert!(FailureKind::Timeout.is_crash());
        assert!(FailureKind::Crashed.is_crash());
        assert!(!FailureKind::Unknown.is_crash());
        assert!(!FailureKind::Assertion.is_crash());
    }

    #[test]
    fn failure_kind_signal_rkyv_roundtrip() {
        let original = FailureKind::Signal("SIGABRT".to_string());
        let bytes = rkyv::to_bytes::<rancor::Error>(&original).unwrap();
        let decoded = rkyv::from_bytes::<FailureKind, rancor::Error>(&bytes).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_worker_capabilities_default() {
        let caps = WorkerCapabilities::default();
        assert_eq!(caps.protocol_version, crate::PROTOCOL_VERSION);
        assert!(caps.cpu_count >= 1);
    }
}
