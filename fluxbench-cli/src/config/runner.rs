//! `[runner]`: execution settings.

use super::duration::parse_config_duration;
use serde::{Deserialize, Serialize};

/// Isolation mode for benchmark execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum IsolationMode {
    /// Run each benchmark in a separate worker process (default)
    #[default]
    Process,
    /// Run benchmarks in the supervisor process (no isolation, useful for debugging)
    InProcess,
}

impl IsolationMode {
    /// Whether this mode provides process isolation
    pub fn is_isolated(self) -> bool {
        matches!(self, IsolationMode::Process)
    }
}

/// Runner configuration for benchmark execution
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunnerConfig {
    /// Default worker timeout per benchmark (e.g., "60s", "5m"). `--worker-timeout` overrides it.
    #[serde(default = "default_timeout")]
    pub timeout: String,
    /// Isolation mode: "process" or "in-process". `--isolated` overrides it.
    #[serde(default)]
    pub isolation: IsolationMode,
    /// Warmup duration before measurement (e.g., "3s")
    #[serde(default = "default_warmup")]
    pub warmup_time: String,
    /// Measurement duration (e.g., "5s")
    #[serde(default = "default_measurement")]
    pub measurement_time: String,
    /// Fixed mode: N samples of one iteration each, no warmup
    #[serde(default)]
    pub samples: Option<u64>,
    /// Target sample count (minimum 2, no upper cap)
    #[serde(default)]
    pub sample_count: Option<usize>,
    /// Minimum total measurement iterations
    #[serde(default)]
    pub min_iterations: Option<u64>,
    /// Maximum total measurement iterations
    #[serde(default)]
    pub max_iterations: Option<u64>,
    /// Number of bootstrap iterations for statistics
    #[serde(default = "default_bootstrap_iterations")]
    pub bootstrap_iterations: usize,
    /// Confidence level (e.g., 0.95 for 95%)
    #[serde(default = "default_confidence_level")]
    pub confidence_level: f64,
    /// Number of parallel isolated workers
    #[serde(default)]
    pub jobs: Option<usize>,
    /// Pin each isolated worker to its own CPU (default: true)
    #[serde(default)]
    pub pin_cpus: Option<bool>,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            timeout: default_timeout(),
            isolation: IsolationMode::default(),
            warmup_time: default_warmup(),
            measurement_time: default_measurement(),
            samples: None,
            sample_count: None,
            min_iterations: None,
            max_iterations: None,
            bootstrap_iterations: default_bootstrap_iterations(),
            confidence_level: default_confidence_level(),
            jobs: None,
            pin_cpus: None,
        }
    }
}

impl RunnerConfig {
    /// `warmup_time` in nanoseconds
    pub fn warmup_time_ns(&self) -> anyhow::Result<u64> {
        parse_config_duration("runner.warmup_time", &self.warmup_time)
    }

    /// `measurement_time` in nanoseconds
    pub fn measurement_time_ns(&self) -> anyhow::Result<u64> {
        parse_config_duration("runner.measurement_time", &self.measurement_time)
    }

    /// `timeout` in nanoseconds
    pub fn timeout_ns(&self) -> anyhow::Result<u64> {
        parse_config_duration("runner.timeout", &self.timeout)
    }

    /// Check every value. The error names the key, the value, and the fix.
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        self.warmup_time_ns()?;
        if self.measurement_time_ns()? == 0 {
            anyhow::bail!("runner.measurement_time must be above 0. Use a value such as \"5s\".");
        }
        if self.timeout_ns()? == 0 {
            anyhow::bail!("runner.timeout must be above 0. Use a value such as \"60s\".");
        }
        if let Some(n) = self.samples.filter(|&n| n < 2) {
            anyhow::bail!("runner.samples = {n} is below the minimum of 2. Use 2 or more.");
        }
        if let Some(n) = self.sample_count.filter(|&n| n < 2) {
            anyhow::bail!("runner.sample_count = {n} is below the minimum of 2. Use 2 or more.");
        }
        if let (Some(min), Some(max)) = (self.min_iterations, self.max_iterations) {
            if max < min {
                anyhow::bail!(
                    "runner.max_iterations = {max} is below runner.min_iterations = {min}. Raise max_iterations or lower min_iterations."
                );
            }
        }
        if !(self.confidence_level > 0.0 && self.confidence_level < 1.0) {
            anyhow::bail!(
                "runner.confidence_level = {} is outside (0, 1). Use a value such as 0.95.",
                self.confidence_level
            );
        }
        if self.jobs == Some(0) {
            anyhow::bail!("runner.jobs = 0 is invalid. Use 1 or more, or remove the key.");
        }
        Ok(())
    }
}

fn default_timeout() -> String {
    "60s".to_string()
}
fn default_warmup() -> String {
    "3s".to_string()
}
fn default_measurement() -> String {
    "5s".to_string()
}
fn default_bootstrap_iterations() -> usize {
    10_000
}
fn default_confidence_level() -> f64 {
    0.95
}
