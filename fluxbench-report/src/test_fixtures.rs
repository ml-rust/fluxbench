//! Shared report fixtures for the format tests.

use crate::report::*;
use chrono::DateTime;

/// Metrics with fixed values derived from `mean_ns`
pub(crate) fn metrics(mean_ns: f64) -> BenchmarkMetrics {
    BenchmarkMetrics {
        samples: 10,
        mean_ns,
        trimmed_mean_ns: Some(mean_ns - 1.0),
        median_ns: mean_ns - 2.0,
        std_dev_ns: 2.0,
        min_ns: mean_ns - 10.0,
        max_ns: mean_ns + 10.0,
        p50_ns: mean_ns - 2.0,
        p90_ns: mean_ns + 5.0,
        p95_ns: mean_ns + 7.0,
        p99_ns: mean_ns + 9.0,
        p999_ns: mean_ns + 10.0,
        skewness: 0.0,
        kurtosis: 3.0,
        ci_lower_ns: Some(mean_ns - 3.0),
        ci_upper_ns: Some(mean_ns + 3.0),
        ci_level: 0.95,
        throughput_ops_sec: Some(2000.5),
        throughput_bytes_per_sec: None,
        throughput_elements_per_sec: None,
        alloc_bytes: 64,
        alloc_count: 2,
        mean_cycles: 300.4,
        median_cycles: 299.6,
        min_cycles: 290,
        max_cycles: 310,
        cycles_per_ns: 3.0,
    }
}

/// A result with no comparison and no raw samples
pub(crate) fn result(
    id: &str,
    group: &str,
    status: BenchmarkStatus,
    metrics: Option<BenchmarkMetrics>,
    failure: Option<FailureInfo>,
) -> BenchmarkReportResult {
    BenchmarkReportResult {
        id: id.to_string(),
        name: id.to_string(),
        group: group.to_string(),
        status,
        severity: fluxbench_core::Severity::Warning,
        file: "benches/b.rs".to_string(),
        line: 1,
        metrics,
        threshold: 0.0,
        comparison: None,
        failure,
        samples: None,
    }
}

/// A failure with no backtrace and no signal
pub(crate) fn failure(kind: &str, message: &str) -> FailureInfo {
    FailureInfo {
        kind: kind.to_string(),
        message: message.to_string(),
        backtrace: None,
        signal: None,
    }
}

/// A report at a fixed timestamp whose summary counts follow `results`
pub(crate) fn report(results: Vec<BenchmarkReportResult>) -> Report {
    let count = |status| results.iter().filter(|r| r.status == status).count();
    let summary = ReportSummary {
        total_benchmarks: results.len(),
        passed: count(BenchmarkStatus::Passed),
        failed: count(BenchmarkStatus::Failed),
        crashed: count(BenchmarkStatus::Crashed),
        skipped: count(BenchmarkStatus::Skipped),
        ..Default::default()
    };
    Report {
        meta: ReportMeta {
            schema_version: REPORT_SCHEMA_VERSION,
            version: "0.1.0".to_string(),
            timestamp: DateTime::from_timestamp(0, 0).unwrap_or_default(),
            git_commit: Some("abc1234def".to_string()),
            git_branch: Some("main".to_string()),
            system: SystemInfo {
                os: "linux".to_string(),
                os_version: "6.0".to_string(),
                cpu: "test".to_string(),
                cpu_cores: 8,
                memory_gb: 16.0,
            },
            config: ReportConfig {
                warmup_time_ns: 0,
                measurement_time_ns: 0,
                min_iterations: None,
                max_iterations: None,
                sample_count: 10,
                iters_per_sample: None,
                bootstrap_iterations: 0,
                confidence_level: 0.95,
                track_allocations: false,
                timer_overhead_ns: 0.0,
            },
        },
        results,
        comparisons: vec![],
        comparison_series: vec![],
        synthetics: vec![],
        verifications: vec![],
        summary,
        baseline_meta: None,
    }
}
