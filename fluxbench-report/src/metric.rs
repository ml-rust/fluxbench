//! Named report metrics that charts can plot.

use crate::rate::{format_byte_rate, format_element_rate};
use crate::{BenchmarkMetrics, format_duration, scale_throughput};

/// Metric names a chart can plot
pub const CHART_METRICS: &[&str] = &[
    "mean",
    "median",
    "min",
    "max",
    "p50",
    "p90",
    "p95",
    "p99",
    "p999",
    "std_dev",
    "throughput",
    "throughput_bytes",
    "throughput_elements",
    "alloc_bytes",
    "alloc_count",
    "mean_cycles",
];

/// Value of the named metric, or `None` for an unknown name or a missing throughput.
///
/// `throughput_bytes` and `throughput_elements` are `None` unless the benchmark has that kind.
pub fn metric_value(metrics: &BenchmarkMetrics, name: &str) -> Option<f64> {
    match name {
        "mean" => Some(metrics.mean_ns),
        "median" => Some(metrics.median_ns),
        "min" => Some(metrics.min_ns),
        "max" => Some(metrics.max_ns),
        "p50" => Some(metrics.p50_ns),
        "p90" => Some(metrics.p90_ns),
        "p95" => Some(metrics.p95_ns),
        "p99" => Some(metrics.p99_ns),
        "p999" => Some(metrics.p999_ns),
        "std_dev" => Some(metrics.std_dev_ns),
        "throughput" => metrics.throughput_ops_sec,
        "throughput_bytes" => metrics.throughput_bytes_per_sec,
        "throughput_elements" => metrics.throughput_elements_per_sec,
        "alloc_bytes" => Some(metrics.alloc_bytes as f64),
        "alloc_count" => Some(metrics.alloc_count as f64),
        "mean_cycles" => Some(metrics.mean_cycles),
        _ => None,
    }
}

/// Format a value of the named metric with its unit.
pub fn format_metric(name: &str, value: f64) -> String {
    match name {
        "throughput" => {
            let (scaled, unit) = scale_throughput(value);
            format!("{scaled:.2} {unit}")
        }
        "throughput_bytes" => format_byte_rate(value),
        "throughput_elements" => format_element_rate(value),
        "alloc_bytes" => format!("{value:.0} B"),
        "alloc_count" => format!("{value:.0}"),
        "mean_cycles" => format!("{value:.0} cycles"),
        _ => format_duration(value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_listed_metric_resolves() {
        let metrics: BenchmarkMetrics = serde_json::from_value(serde_json::json!({
            "samples": 2, "mean_ns": 10.0, "median_ns": 9.0, "std_dev_ns": 1.0,
            "min_ns": 8.0, "max_ns": 12.0, "p50_ns": 9.0, "p90_ns": 11.0, "p95_ns": 11.5,
            "p99_ns": 12.0, "p999_ns": 12.0, "skewness": 0.0, "kurtosis": 3.0,
            "ci_level": 0.95, "throughput_ops_sec": 1.0e8,
            "throughput_bytes_per_sec": 2048.0, "throughput_elements_per_sec": 3.0e6,
            "alloc_bytes": 64,
            "alloc_count": 2, "mean_cycles": 30.0, "median_cycles": 30.0,
            "min_cycles": 25, "max_cycles": 40, "cycles_per_ns": 3.0
        }))
        .unwrap();
        for name in CHART_METRICS {
            assert!(metric_value(&metrics, name).is_some(), "{name}");
        }
        assert_eq!(metric_value(&metrics, "fastest"), None);
        assert_eq!(format_metric("alloc_bytes", 64.0), "64 B");
        assert_eq!(format_metric("throughput_bytes", 2048.0), "2.00 KiB/s");
        assert_eq!(format_metric("throughput_elements", 3.0e6), "3.00 Melem/s");
    }

    #[test]
    fn missing_work_throughput_has_no_value() {
        let metrics: BenchmarkMetrics = serde_json::from_value(serde_json::json!({
            "samples": 2, "mean_ns": 10.0, "median_ns": 9.0, "std_dev_ns": 1.0,
            "min_ns": 8.0, "max_ns": 12.0, "p50_ns": 9.0, "p90_ns": 11.0, "p95_ns": 11.5,
            "p99_ns": 12.0, "p999_ns": 12.0, "skewness": 0.0, "kurtosis": 3.0,
            "ci_level": 0.95, "throughput_ops_sec": 1.0e8, "alloc_bytes": 64,
            "alloc_count": 2, "mean_cycles": 30.0, "median_cycles": 30.0,
            "min_cycles": 25, "max_cycles": 40, "cycles_per_ns": 3.0
        }))
        .unwrap();
        assert_eq!(metric_value(&metrics, "throughput_bytes"), None);
        assert_eq!(metric_value(&metrics, "throughput_elements"), None);
        assert!(metric_value(&metrics, "throughput").is_some());
    }
}
