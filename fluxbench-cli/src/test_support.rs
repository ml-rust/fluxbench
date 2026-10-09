//! Report fixtures for unit tests.

use fluxbench_core::BenchmarkDef;
use fluxbench_report::{
    BenchmarkMetrics, BenchmarkReportResult, BenchmarkStatus, Report, ReportConfig, ReportMeta,
    ReportSummary, SystemInfo,
};

pub(crate) fn dummy_meta() -> ReportMeta {
    ReportMeta {
        schema_version: 1,
        version: "0.1.0".to_string(),
        timestamp: chrono::Utc::now(),
        git_commit: None,
        git_branch: None,
        system: SystemInfo {
            os: "linux".to_string(),
            os_version: "6.0".to_string(),
            cpu: "test".to_string(),
            cpu_cores: 1,
            memory_gb: 1.0,
        },
        config: ReportConfig {
            warmup_time_ns: 0,
            measurement_time_ns: 0,
            min_iterations: None,
            max_iterations: None,
            sample_count: 100,
            iters_per_sample: None,
            bootstrap_iterations: 0,
            confidence_level: 0.95,
            track_allocations: false,
            timer_overhead_ns: 0.0,
        },
    }
}

/// Metrics with 100 samples and a std dev of 1% of `mean`.
pub(crate) fn dummy_metrics(mean: f64) -> BenchmarkMetrics {
    BenchmarkMetrics {
        samples: 100,
        mean_ns: mean,
        trimmed_mean_ns: Some(mean),
        median_ns: mean,
        std_dev_ns: mean * 0.01,
        min_ns: mean * 0.9,
        max_ns: mean * 1.1,
        p50_ns: mean,
        p90_ns: mean * 1.05,
        p95_ns: mean * 1.07,
        p99_ns: mean * 1.09,
        p999_ns: mean * 1.1,
        skewness: 0.0,
        kurtosis: 3.0,
        ci_lower_ns: Some(mean * 0.98),
        ci_upper_ns: Some(mean * 1.02),
        ci_level: 0.95,
        throughput_ops_sec: None,
        throughput_bytes_per_sec: None,
        throughput_elements_per_sec: None,
        alloc_bytes: 0,
        alloc_count: 0,
        mean_cycles: 0.0,
        median_cycles: 0.0,
        min_cycles: 0,
        max_cycles: 0,
        cycles_per_ns: 0.0,
    }
}

/// A passed result with severity `Warning` and no samples.
pub(crate) fn dummy_result(id: &str, mean: f64, threshold: f64) -> BenchmarkReportResult {
    BenchmarkReportResult {
        id: id.to_string(),
        name: id.to_string(),
        group: "test".to_string(),
        status: BenchmarkStatus::Passed,
        severity: fluxbench_core::Severity::Warning,
        file: "test.rs".to_string(),
        line: 1,
        metrics: Some(dummy_metrics(mean)),
        threshold,
        comparison: None,
        failure: None,
        samples: None,
    }
}

/// Benchmark definition with every option unset, in group `default`.
pub(crate) fn bench_def(id: &'static str) -> BenchmarkDef {
    BenchmarkDef {
        id,
        name: id,
        group: "default",
        severity: fluxbench_core::Severity::Info,
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
        throughput: None,
        depends_on: &[],
    }
}

pub(crate) fn dummy_report(results: Vec<BenchmarkReportResult>) -> Report {
    let total = results.len();
    Report {
        meta: dummy_meta(),
        results,
        comparisons: vec![],
        comparison_series: vec![],
        synthetics: vec![],
        verifications: vec![],
        summary: ReportSummary {
            total_benchmarks: total,
            passed: total,
            ..Default::default()
        },
        baseline_meta: None,
    }
}
