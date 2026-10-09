//! Measurement config, errors, and results.

use fluxbench_ipc::{BenchmarkConfig, Sample};

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

/// Config for one benchmark measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeasureConfig {
    /// Warmup budget in nanoseconds. Fixed mode skips warmup.
    pub warmup_ns: u64,
    /// Measurement budget in nanoseconds. It sets the iterations per sample.
    pub measurement_ns: u64,
    /// Samples to collect. Values below `MIN_SAMPLE_COUNT` rise to it.
    pub sample_count: usize,
    /// Fixed iterations per sample. `Some` selects fixed mode: no warmup, no estimate.
    pub fixed_iters_per_sample: Option<u64>,
    /// Minimum total measurement iterations. Ignored in fixed mode.
    pub min_iterations: Option<u64>,
    /// Maximum total measurement iterations. Ignored in fixed mode.
    pub max_iterations: Option<u64>,
    /// Read the allocation counters around each timed region.
    pub track_allocations: bool,
}

impl Default for MeasureConfig {
    fn default() -> Self {
        Self::from(&BenchmarkConfig::default())
    }
}

impl From<&BenchmarkConfig> for MeasureConfig {
    fn from(config: &BenchmarkConfig) -> Self {
        Self {
            warmup_ns: config.warmup_time_ns,
            measurement_ns: config.measurement_time_ns,
            sample_count: config.sample_count as usize,
            fixed_iters_per_sample: config.iters_per_sample,
            min_iterations: config.min_iterations,
            max_iterations: config.max_iterations,
            track_allocations: config.track_allocations,
        }
    }
}

impl MeasureConfig {
    /// Raise the sample count to `MIN_SAMPLE_COUNT` and a fixed count of 0 to 1.
    pub fn normalized(mut self) -> Self {
        self.sample_count = self.sample_count.max(MIN_SAMPLE_COUNT);
        self.fixed_iters_per_sample = self.fixed_iters_per_sample.map(|n| n.max(1));
        self
    }
}

/// Error from one benchmark measurement.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MeasureError {
    /// The benchmark body returned without calling an `iter*` method.
    #[error("benchmark body never called b.iter*")]
    NoIterCall,
    /// The benchmark body called a second `iter*` method. The second call does not run.
    #[error("benchmark body called more than one b.iter* method")]
    MultipleIterCalls,
    /// The measurement collected no samples.
    #[error("benchmark produced zero samples")]
    NoSamples,
    /// The async fallback runtime did not build.
    #[error("async runtime build error: {0}")]
    AsyncRuntime(String),
}

/// Result of a single benchmark run
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Collected samples. Each holds per-iteration means over `iters_per_sample` calls.
    pub samples: Vec<Sample>,
    /// Routine calls in the measurement phase. Warmup calls are excluded.
    pub iterations: u64,
    /// Routine calls in the warmup phase
    pub warmup_iterations: u64,
    /// Sum of each sample's mean times its iterations, in nanoseconds
    pub total_time_ns: u64,
    /// Routine calls per sample
    pub iters_per_sample: u64,
    /// `true` when the hard deadline stopped measurement before `sample_count` samples
    pub truncated: bool,
}
