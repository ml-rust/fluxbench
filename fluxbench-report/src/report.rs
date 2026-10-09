//! Report Data Structures

use chrono::{DateTime, Utc};
use fluxbench_stats::SummaryStatistics;
use serde::{Deserialize, Serialize};

/// Complete benchmark report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    /// Report metadata and configuration
    pub meta: ReportMeta,
    /// Individual benchmark results
    pub results: Vec<BenchmarkReportResult>,
    /// Comparison groups between benchmarks
    pub comparisons: Vec<ComparisonResult>,
    /// Multi-point comparison series (e.g., batch size scaling)
    pub comparison_series: Vec<ComparisonSeries>,
    /// Synthetic measurement results
    pub synthetics: Vec<fluxbench_logic::SyntheticResult>,
    /// Verification results
    pub verifications: Vec<fluxbench_logic::VerificationResult>,
    /// Overall report summary statistics
    pub summary: ReportSummary,
    /// Baseline report metadata (populated when comparing against a baseline)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_meta: Option<ReportMeta>,
}

/// Result of a comparison group
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    /// Comparison identifier
    pub id: String,
    /// Human-readable title
    pub title: String,
    /// Baseline benchmark ID
    pub baseline: String,
    /// Metric used for comparison
    pub metric: String,
    /// Individual benchmark entries in the comparison
    pub entries: Vec<ComparisonEntry>,
}

/// Single entry in a comparison table
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonEntry {
    /// Benchmark ID
    pub benchmark_id: String,
    /// Metric value (e.g., mean time in ns)
    pub value: f64,
    /// Speedup vs baseline (1.0 = same, >1.0 = faster, <1.0 = slower)
    pub speedup: f64,
    /// Whether this is the baseline
    pub is_baseline: bool,
}

/// Grouped comparison series for multi-point charts (e.g., batch size scaling)
/// Multiple #[compare(group = "...", x = "...")] combine into one series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonSeries {
    /// Group identifier
    pub group: String,
    /// Chart title (from first comparison's title)
    pub title: String,
    /// X-axis values in order (e.g., ["1", "10", "100", "1000"])
    pub x_values: Vec<String>,
    /// Competitor/series names (benchmark IDs)
    pub series_names: Vec<String>,
    /// Data points: `series_data[series_idx][x_idx] = value`
    pub series_data: Vec<Vec<f64>>,
    /// Metric used
    pub metric: String,
}

/// Report schema version this crate writes.
///
/// - 1: no raw samples
/// - 2: saved baselines carry `results[].samples`. Comparisons carry the threshold and difference CI.
///
/// Readers accept every version up to this one.
pub const REPORT_SCHEMA_VERSION: u32 = 2;

/// Report metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMeta {
    /// Schema version number. See [`REPORT_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Report version string
    pub version: String,
    /// Report generation timestamp
    pub timestamp: DateTime<Utc>,
    /// Git commit hash (if available)
    pub git_commit: Option<String>,
    /// Git branch name (if available)
    pub git_branch: Option<String>,
    /// System information where benchmarks ran
    pub system: SystemInfo,
    /// Benchmark execution configuration
    pub config: ReportConfig,
}

/// Execution configuration captured in report metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportConfig {
    /// Warmup duration in nanoseconds
    pub warmup_time_ns: u64,
    /// Measurement duration in nanoseconds
    pub measurement_time_ns: u64,
    /// Minimum total measurement iterations (if specified)
    pub min_iterations: Option<u64>,
    /// Maximum total measurement iterations (if specified)
    pub max_iterations: Option<u64>,
    /// Target sample count. Reports without this field read as 100.
    #[serde(default = "default_sample_count")]
    pub sample_count: usize,
    /// Fixed iterations per sample (`None` = time-based mode)
    #[serde(default)]
    pub iters_per_sample: Option<u64>,
    /// Bootstrap iterations for confidence intervals
    pub bootstrap_iterations: usize,
    /// Confidence level for intervals (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// Whether memory allocations were tracked
    pub track_allocations: bool,
    /// Median cost of one empty timer pair in nanoseconds.
    /// The engine subtracts it once per timed region.
    /// Reports without this field read as 0.
    #[serde(default)]
    pub timer_overhead_ns: f64,
}

/// Sample count for reports written before `sample_count` existed
fn default_sample_count() -> usize {
    100
}

/// System information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    /// Operating system name
    pub os: String,
    /// Operating system version
    pub os_version: String,
    /// CPU model name
    pub cpu: String,
    /// Number of CPU cores
    pub cpu_cores: u32,
    /// Total system memory in gigabytes
    pub memory_gb: f64,
}

/// Individual benchmark result in the report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReportResult {
    /// Unique benchmark identifier
    pub id: String,
    /// Human-readable benchmark name
    pub name: String,
    /// Benchmark group/category
    pub group: String,
    /// Execution status (passed, failed, etc.)
    pub status: BenchmarkStatus,
    /// Severity level if failed
    pub severity: fluxbench_core::Severity,
    /// Source file path
    pub file: String,
    /// Source line number
    pub line: u32,
    /// Timing and statistical metrics (if successful)
    pub metrics: Option<BenchmarkMetrics>,
    /// Per-benchmark regression threshold percentage (0.0 = use global)
    #[serde(default)]
    pub threshold: f64,
    /// Comparison results against baseline (if applicable)
    pub comparison: Option<Comparison>,
    /// Failure details (if failed)
    pub failure: Option<FailureInfo>,
    /// Raw per-sample timings in nanoseconds: each is the mean time of one iteration in that sample.
    /// These are the values behind `metrics.mean_ns` and its CI.
    /// Saved baselines carry them. Other reports and schema 1 baselines omit them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub samples: Option<Vec<f64>>,
}

/// Benchmark execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BenchmarkStatus {
    /// Benchmark completed successfully
    Passed,
    /// Benchmark ran but assertion failed
    Failed,
    /// Benchmark crashed or panicked
    Crashed,
    /// Benchmark was skipped
    Skipped,
}

/// Benchmark timing metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    /// Number of samples collected
    pub samples: usize,
    /// Mean execution time in nanoseconds over all samples, outliers included.
    /// The bootstrap CI estimates this value.
    pub mean_ns: f64,
    /// Mean execution time in nanoseconds with detected outliers removed.
    /// Reports without this field read as `None`.
    #[serde(default)]
    pub trimmed_mean_ns: Option<f64>,
    /// Median execution time in nanoseconds
    pub median_ns: f64,
    /// Standard deviation in nanoseconds
    pub std_dev_ns: f64,
    /// Minimum execution time in nanoseconds
    pub min_ns: f64,
    /// Maximum execution time in nanoseconds
    pub max_ns: f64,
    /// 50th percentile (P50) in nanoseconds
    pub p50_ns: f64,
    /// 90th percentile (P90) in nanoseconds
    pub p90_ns: f64,
    /// 95th percentile (P95) in nanoseconds
    pub p95_ns: f64,
    /// 99th percentile (P99) in nanoseconds
    pub p99_ns: f64,
    /// 99.9th percentile (P99.9) in nanoseconds
    pub p999_ns: f64,
    /// Distribution skewness
    pub skewness: f64,
    /// Distribution kurtosis
    pub kurtosis: f64,
    /// Confidence interval lower bound in nanoseconds (`None` when the bootstrap is skipped)
    #[serde(default)]
    pub ci_lower_ns: Option<f64>,
    /// Confidence interval upper bound in nanoseconds (`None` when the bootstrap is skipped)
    #[serde(default)]
    pub ci_upper_ns: Option<f64>,
    /// Confidence level used for interval (e.g., 0.95)
    pub ci_level: f64,
    /// Throughput in operations per second (if applicable)
    pub throughput_ops_sec: Option<f64>,
    /// Mean bytes allocated per iteration
    pub alloc_bytes: u64,
    /// Mean allocation count per iteration
    pub alloc_count: u64,
    /// Mean CPU cycles (x86_64 only, 0 on other platforms)
    pub mean_cycles: f64,
    /// Median CPU cycles (x86_64 only, 0 on other platforms)
    pub median_cycles: f64,
    /// Minimum CPU cycles (x86_64 only, 0 on other platforms)
    pub min_cycles: u64,
    /// Maximum CPU cycles (x86_64 only, 0 on other platforms)
    pub max_cycles: u64,
    /// CPU cycles per nanosecond
    pub cycles_per_ns: f64,
}

impl From<&SummaryStatistics> for BenchmarkMetrics {
    fn from(stats: &SummaryStatistics) -> Self {
        Self {
            samples: stats.sample_count,
            mean_ns: stats.mean,
            trimmed_mean_ns: Some(stats.trimmed_mean),
            median_ns: stats.median,
            std_dev_ns: stats.std_dev,
            min_ns: stats.min,
            max_ns: stats.max,
            p50_ns: stats.p50,
            p90_ns: stats.p90,
            p95_ns: stats.p95,
            p99_ns: stats.p99,
            p999_ns: stats.p999,
            skewness: stats.skewness,
            kurtosis: stats.kurtosis,
            ci_lower_ns: None, // Filled by bootstrap
            ci_upper_ns: None,
            ci_level: fluxbench_stats::DEFAULT_CONFIDENCE_LEVEL,
            throughput_ops_sec: None,
            alloc_bytes: 0,
            alloc_count: 0,
            // Cycles filled separately via CyclesStatistics
            mean_cycles: 0.0,
            median_cycles: 0.0,
            min_cycles: 0,
            max_cycles: 0,
            cycles_per_ns: 0.0,
        }
    }
}

/// Comparison against baseline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Comparison {
    /// Baseline mean time in nanoseconds
    pub baseline_mean_ns: f64,
    /// Absolute change from baseline in nanoseconds
    pub absolute_change_ns: f64,
    /// Relative change from baseline (percentage)
    pub relative_change: f64,
    /// Probability that the current mean is slower than the baseline mean (0.0 to 1.0)
    pub probability_regression: f64,
    /// The change is statistically significant and its size exceeds `threshold`.
    /// A positive `relative_change` makes it a regression, a negative one an improvement.
    pub is_significant: bool,
    /// Effect size (Cohen's d, pooled standard deviation)
    pub effect_size: f64,
    /// Regression threshold percentage applied to this benchmark.
    /// Reports without this field read as 0.
    #[serde(default)]
    pub threshold: f64,
    /// Lower bound of the CI of (current mean - baseline mean) in nanoseconds
    #[serde(default)]
    pub difference_ci_lower_ns: Option<f64>,
    /// Upper bound of the CI of (current mean - baseline mean) in nanoseconds
    #[serde(default)]
    pub difference_ci_upper_ns: Option<f64>,
    /// Confidence level of the difference CI and the significance check.
    /// Reports without this field read as 0.95.
    #[serde(default = "default_confidence_level")]
    pub confidence_level: f64,
    /// Statistical method behind the probability and the difference CI
    #[serde(default)]
    pub method: ComparisonMethod,
}

impl Comparison {
    /// A significant slowdown beyond the threshold
    pub fn is_regression(&self) -> bool {
        self.is_significant && self.relative_change > 0.0
    }

    /// A significant speedup beyond the threshold
    pub fn is_improvement(&self) -> bool {
        self.is_significant && self.relative_change < 0.0
    }
}

/// Statistical method of a baseline comparison
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ComparisonMethod {
    /// Bootstrap of raw samples from both runs
    Bootstrap,
    /// Welch-style z test on mean, std dev, and sample count.
    /// Used when either side has no raw samples or the bootstrap is off.
    #[default]
    Summary,
}

/// Confidence level for comparisons written before `confidence_level` existed
fn default_confidence_level() -> f64 {
    fluxbench_stats::DEFAULT_CONFIDENCE_LEVEL
}

/// Failure information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureInfo {
    /// Failure kind. One of:
    /// - `panic`: the benchmark panicked
    /// - `timeout`: the worker exceeded `--worker-timeout`
    /// - `signal`: a signal killed the worker. `signal` holds its name.
    /// - `crashed`: the worker exited, broke the IPC stream, or never started
    /// - `assertion`, `allocation_limit`, `unknown`: other errors
    pub kind: String,
    /// Error message
    pub message: String,
    /// Stack backtrace (if available)
    pub backtrace: Option<String>,
    /// Signal name for kind `signal`, e.g. `SIGSEGV`. Absent for other kinds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signal: Option<String>,
}

/// Report summary
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReportSummary {
    /// Total number of benchmarks
    pub total_benchmarks: usize,
    /// Number of passed benchmarks
    pub passed: usize,
    /// Number of failed benchmarks
    pub failed: usize,
    /// Number of crashed benchmarks
    pub crashed: usize,
    /// Number of skipped benchmarks
    pub skipped: usize,
    /// Number of performance regressions detected
    pub regressions: usize,
    /// Number of performance improvements detected
    pub improvements: usize,
    /// Number of critical failures
    pub critical_failures: usize,
    /// Number of warnings
    pub warnings: usize,
    /// Total execution duration in milliseconds
    pub total_duration_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_config_reads_json_without_sample_fields() {
        let json = r#"{
            "warmup_time_ns": 3000000000,
            "measurement_time_ns": 5000000000,
            "min_iterations": null,
            "max_iterations": null,
            "bootstrap_iterations": 10000,
            "confidence_level": 0.95,
            "track_allocations": true
        }"#;
        let config: ReportConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.sample_count, 100);
        assert_eq!(config.iters_per_sample, None);
    }

    #[test]
    fn failure_info_omits_absent_signal_and_reads_old_json() {
        let info = FailureInfo {
            kind: "panic".to_string(),
            message: "boom".to_string(),
            backtrace: None,
            signal: None,
        };
        let json = serde_json::to_string(&info).unwrap();
        assert_eq!(
            json,
            r#"{"kind":"panic","message":"boom","backtrace":null}"#
        );
        let back: FailureInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(back.signal, None);
    }

    #[test]
    fn failure_info_carries_signal_name() {
        let info = FailureInfo {
            kind: "signal".to_string(),
            message: "worker killed by SIGSEGV".to_string(),
            backtrace: None,
            signal: Some("SIGSEGV".to_string()),
        };
        let value: serde_json::Value = serde_json::to_value(&info).unwrap();
        assert_eq!(value["kind"], "signal");
        assert_eq!(value["signal"], "SIGSEGV");
    }

    /// A schema 1 result: no `samples`, and a comparison without the schema 2 fields.
    const SCHEMA_1_RESULT: &str = r#"{
        "id": "b",
        "name": "b",
        "group": "g",
        "status": "passed",
        "severity": "Critical",
        "file": "b.rs",
        "line": 1,
        "metrics": null,
        "threshold": 0.0,
        "comparison": {
            "baseline_mean_ns": 100.0,
            "absolute_change_ns": 8.0,
            "relative_change": 8.0,
            "probability_regression": 0.99,
            "is_significant": true,
            "effect_size": 2.0
        },
        "failure": null
    }"#;

    #[test]
    fn schema_1_result_still_parses() {
        let result: BenchmarkReportResult = serde_json::from_str(SCHEMA_1_RESULT).unwrap();
        assert_eq!(result.samples, None);
        let cmp = result.comparison.unwrap();
        assert!(cmp.is_regression());
        assert!(!cmp.is_improvement());
        assert_eq!(cmp.threshold, 0.0);
        assert_eq!(cmp.difference_ci_lower_ns, None);
        assert_eq!(cmp.confidence_level, 0.95);
        assert_eq!(cmp.method, ComparisonMethod::Summary);
    }

    #[test]
    fn result_without_samples_omits_the_field() {
        let mut result: BenchmarkReportResult = serde_json::from_str(SCHEMA_1_RESULT).unwrap();
        result.samples = None;
        let value = serde_json::to_value(&result).unwrap();
        assert!(value.get("samples").is_none());
    }

    #[test]
    fn result_samples_roundtrip() {
        let mut result: BenchmarkReportResult = serde_json::from_str(SCHEMA_1_RESULT).unwrap();
        result.samples = Some(vec![1.5, 2.0, 2.5]);
        let json = serde_json::to_string(&result).unwrap();
        let back: BenchmarkReportResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.samples, Some(vec![1.5, 2.0, 2.5]));
    }

    #[test]
    fn comparison_schema_2_fields_roundtrip() {
        let cmp = Comparison {
            baseline_mean_ns: 100.0,
            absolute_change_ns: -10.0,
            relative_change: -10.0,
            probability_regression: 0.001,
            is_significant: true,
            effect_size: -3.0,
            threshold: 5.0,
            difference_ci_lower_ns: Some(-12.0),
            difference_ci_upper_ns: Some(-8.0),
            confidence_level: 0.99,
            method: ComparisonMethod::Bootstrap,
        };
        let value = serde_json::to_value(&cmp).unwrap();
        assert_eq!(value["method"], "bootstrap");
        let back: Comparison = serde_json::from_value(value).unwrap();
        assert!(back.is_improvement());
        assert_eq!(back.threshold, 5.0);
        assert_eq!(back.difference_ci_lower_ns, Some(-12.0));
        assert_eq!(back.difference_ci_upper_ns, Some(-8.0));
        assert_eq!(back.confidence_level, 0.99);
        assert_eq!(back.method, ComparisonMethod::Bootstrap);
    }

    #[test]
    fn report_config_roundtrips_sample_fields() {
        let json = r#"{
            "warmup_time_ns": 0,
            "measurement_time_ns": 5000000000,
            "min_iterations": null,
            "max_iterations": null,
            "sample_count": 500,
            "iters_per_sample": 1,
            "bootstrap_iterations": 10000,
            "confidence_level": 0.95,
            "track_allocations": false
        }"#;
        let config: ReportConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.sample_count, 500);
        assert_eq!(config.iters_per_sample, Some(1));
        let back: ReportConfig =
            serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
        assert_eq!(back.sample_count, 500);
        assert_eq!(back.iters_per_sample, Some(1));
    }
}
