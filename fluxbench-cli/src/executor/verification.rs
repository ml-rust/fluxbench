//! Verification and Comparison Execution
//!
//! Processes benchmark comparisons, synthetic metrics, and performance
//! verifications against computed statistics.
//!
//! ## Data Flow
//!
//! ```text
//! BenchExecutionResult + SummaryStatistics
//!           │
//!           ▼
//!    ┌──────────────────┐
//!    │  MetricContext   │ ◄── Collects all benchmark metrics
//!    └────────┬─────────┘     (mean, median, percentiles, allocations)
//!             │
//!    ┌────────┴────────┬─────────────────┐
//!    ▼                 ▼                 ▼
//! Comparisons     Synthetics      Verifications
//! (speedup vs     (computed        (pass/fail
//!  baseline)       metrics)         thresholds)
//! ```
//!
//! ## Key Components
//!
//! - **Comparisons**: Compare benchmarks against baselines (e.g., speedup tables)
//! - **ComparisonSeries**: Group comparisons for multi-point charts (scaling analysis)
//! - **Synthetics**: Computed metrics from expressions (e.g., `ops_per_byte = 1e9 / mean`)
//! - **Verifications**: Pass/fail checks with configurable severity levels

use super::execution::BenchExecutionResult;
use super::throughput::work_rates;
use fluxbench_core::CompareDef;
use fluxbench_logic::{
    MetricContext, SyntheticDef, SyntheticResult, Verification, VerificationContext,
    VerificationResult, VerifyDef, compute_synthetics, run_verifications,
};
use fluxbench_report::{ComparisonEntry, ComparisonResult, ComparisonSeries};
use fluxbench_stats::SummaryStatistics;
use rustc_hash::FxHashSet;

/// Run comparisons, synthetics, and verifications against computed metrics
///
/// # Arguments
/// * `results` - Benchmark execution results
/// * `stats` - Pre-computed summary statistics
///
/// # Returns
/// Tuple of (comparisons, comparison_series, synthetics, verifications)
pub fn execute_verifications(
    results: &[BenchExecutionResult],
    stats: &[(String, Option<SummaryStatistics>)],
) -> (
    Vec<ComparisonResult>,
    Vec<ComparisonSeries>,
    Vec<SyntheticResult>,
    Vec<VerificationResult>,
) {
    // Build metric context with benchmark results
    let (mut context, unavailable) = metric_context(results, stats);

    // Build stats lookup for comparison processing
    let stats_lookup: std::collections::HashMap<_, _> = stats
        .iter()
        .filter_map(|(id, s)| s.as_ref().map(|s| (id.as_str(), s)))
        .collect();

    // Process comparison groups
    let mut comparison_results: Vec<ComparisonResult> = Vec::new();
    let mut grouped_comparisons: std::collections::BTreeMap<
        String,
        Vec<(&CompareDef, Vec<ComparisonEntry>)>,
    > = std::collections::BTreeMap::new();

    for cmp in inventory::iter::<CompareDef> {
        // Get baseline (first benchmark if not specified)
        let baseline_id = cmp
            .baseline
            .unwrap_or_else(|| cmp.benchmarks.first().copied().unwrap_or(""));
        let baseline_stats = match stats_lookup.get(baseline_id) {
            Some(s) => s,
            None => continue,
        };
        // `#[compare]` checks the metric name at compile time.
        let Some(baseline_value) = get_metric_value(baseline_stats, cmp.metric) else {
            continue;
        };

        // Build entries for all benchmarks
        let entries: Vec<ComparisonEntry> = cmp
            .benchmarks
            .iter()
            .filter_map(|bench_id| {
                let bench_stats = stats_lookup.get(bench_id)?;
                let value = get_metric_value(bench_stats, cmp.metric)?;
                let speedup = if value > 0.0 {
                    baseline_value / value
                } else {
                    0.0
                };

                Some(ComparisonEntry {
                    benchmark_id: bench_id.to_string(),
                    value,
                    speedup,
                    is_baseline: *bench_id == baseline_id,
                })
            })
            .collect();

        // Only include comparison if we have at least 2 entries
        if entries.len() >= 2 {
            // If grouped, collect for chart generation
            if let Some(group) = cmp.group {
                grouped_comparisons
                    .entry(group.to_string())
                    .or_default()
                    .push((cmp, entries.clone()));
            }

            // Always add to individual comparisons (for non-grouped display)
            if cmp.group.is_none() {
                comparison_results.push(ComparisonResult {
                    id: cmp.id.to_string(),
                    title: cmp.title.to_string(),
                    baseline: baseline_id.to_string(),
                    metric: cmp.metric.to_string(),
                    entries,
                });
            }
        }
    }

    // Build comparison series from grouped comparisons
    let comparison_series: Vec<ComparisonSeries> = grouped_comparisons
        .into_iter()
        .filter_map(|(group, comparisons)| {
            if comparisons.is_empty() {
                return None;
            }

            // Get title and metric from first comparison
            let (first_cmp, _) = &comparisons[0];
            let title = first_cmp.title.to_string();
            let metric = first_cmp.metric.to_string();

            // Get series names: use series labels if provided, otherwise benchmark IDs
            let series_names: Vec<String> = if let Some(labels) = first_cmp.series {
                labels.iter().map(|s| s.to_string()).collect()
            } else {
                first_cmp.benchmarks.iter().map(|s| s.to_string()).collect()
            };

            // Sort comparisons by x value
            let mut sorted_comparisons = comparisons;
            sorted_comparisons.sort_by(|(a, _), (b, _)| {
                let ax = a.x.unwrap_or("0");
                let bx = b.x.unwrap_or("0");
                ax.parse::<f64>()
                    .unwrap_or(0.0)
                    .partial_cmp(&bx.parse::<f64>().unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Build x_values
            let x_values: Vec<String> = sorted_comparisons
                .iter()
                .map(|(cmp, _)| cmp.x.unwrap_or("").to_string())
                .collect();

            // Build series_data[series_idx][x_idx]
            // When series labels are provided, match by position index
            let mut series_data: Vec<Vec<f64>> =
                vec![vec![0.0; x_values.len()]; series_names.len()];

            for (x_idx, (cmp, entries)) in sorted_comparisons.iter().enumerate() {
                // Build a map from benchmark_id to value for this x point
                let entry_map: std::collections::HashMap<&str, f64> = entries
                    .iter()
                    .map(|e| (e.benchmark_id.as_str(), e.value))
                    .collect();

                // Match by position in the benchmarks array
                for (series_idx, bench_id) in cmp.benchmarks.iter().enumerate() {
                    if series_idx < series_names.len() {
                        if let Some(&value) = entry_map.get(bench_id) {
                            series_data[series_idx][x_idx] = value;
                        }
                    }
                }
            }

            Some(ComparisonSeries {
                group,
                title,
                x_values,
                series_names,
                series_data,
                metric,
            })
        })
        .collect();

    // Collect and compute synthetic metrics
    let synthetic_defs: Vec<SyntheticDef> = inventory::iter::<SyntheticDef>
        .into_iter()
        .cloned()
        .collect();

    let mut synthetic_results = Vec::new();
    if !synthetic_defs.is_empty() {
        let computed = compute_synthetics(&synthetic_defs, &context);
        for result in computed {
            match result {
                Ok(sr) => {
                    // Add synthetic metric to context for verifications to use
                    context.set(&sr.id, sr.value);
                    synthetic_results.push(sr);
                }
                Err(_) => {
                    // Synthetic couldn't be computed (missing dependencies)
                }
            }
        }
    }

    // Collect all registered verifications
    let verifications: Vec<Verification> = inventory::iter::<VerifyDef>
        .into_iter()
        .map(|v| Verification {
            id: v.id.to_string(),
            expression: v.expression.to_string(),
            severity: v.severity,
            margin: v.margin,
        })
        .collect();

    let verification_results = if verifications.is_empty() {
        Vec::new()
    } else {
        // Run verifications
        let mut verification_context = VerificationContext::new(&context, unavailable);
        // Propagate unit info from synthetics
        for sr in &synthetic_results {
            if let Some(ref unit) = sr.unit {
                verification_context.set_unit(&sr.id, unit);
            }
        }
        run_verifications(&verifications, &verification_context)
    };

    (
        comparison_results,
        comparison_series,
        synthetic_results,
        verification_results,
    )
}

/// Get metric value from stats based on metric name. Matches `COMPARE_METRICS` in fluxbench-macros.
fn get_metric_value(stats: &SummaryStatistics, metric: &str) -> Option<f64> {
    match metric {
        "mean" => Some(stats.mean),
        "median" => Some(stats.median),
        "min" => Some(stats.min),
        "max" => Some(stats.max),
        "p50" => Some(stats.p50),
        "p90" => Some(stats.p90),
        "p95" => Some(stats.p95),
        "p99" => Some(stats.p99),
        "p999" => Some(stats.p999),
        _ => None,
    }
}

/// Metrics `#[verify]` and `#[synthetic]` can name, plus the IDs of benchmarks without stats.
fn metric_context(
    results: &[BenchExecutionResult],
    stats: &[(String, Option<SummaryStatistics>)],
) -> (MetricContext, FxHashSet<String>) {
    let mut context = MetricContext::new();
    let mut unavailable = FxHashSet::default();

    // Allocation data and throughput come from results, not stats
    let result_lookup: std::collections::HashMap<_, _> = results
        .iter()
        .map(|r| (r.benchmark_id.as_str(), r))
        .collect();

    for (bench_id, stats_opt) in stats {
        let Some(stats) = stats_opt else {
            unavailable.insert(bench_id.clone());
            continue;
        };
        // Add mean as the primary metric for each benchmark
        context.set(bench_id, stats.mean);

        // Central tendency
        context.set(format!("{}_mean", bench_id), stats.mean);
        context.set(format!("{}_median", bench_id), stats.median);
        context.set(format!("{}_std_dev", bench_id), stats.std_dev);

        // Extremes
        context.set(format!("{}_min", bench_id), stats.min);
        context.set(format!("{}_max", bench_id), stats.max);

        // Percentiles
        context.set(format!("{}_p50", bench_id), stats.p50);
        context.set(format!("{}_p90", bench_id), stats.p90);
        context.set(format!("{}_p95", bench_id), stats.p95);
        context.set(format!("{}_p99", bench_id), stats.p99);
        context.set(format!("{}_p999", bench_id), stats.p999);

        // Distribution shape
        context.set(format!("{}_skewness", bench_id), stats.skewness);
        context.set(format!("{}_kurtosis", bench_id), stats.kurtosis);

        // Operations per second, when the mean is above zero
        if stats.mean > 0.0 {
            context.set(format!("{}_throughput", bench_id), 1e9 / stats.mean);
        }

        // Sample info
        context.set(format!("{}_samples", bench_id), stats.sample_count as f64);

        if let Some(result) = result_lookup.get(bench_id.as_str()) {
            context.set(
                format!("{}_alloc_bytes", bench_id),
                result.alloc_bytes as f64,
            );
            context.set(
                format!("{}_alloc_count", bench_id),
                result.alloc_count as f64,
            );

            // Bytes or elements per second, for a benchmark with throughput
            let (bytes, elements) = work_rates(result.throughput, stats.mean);
            if let Some(rate) = bytes {
                context.set(format!("{}_throughput_bytes", bench_id), rate);
            }
            if let Some(rate) = elements {
                context.set(format!("{}_throughput_elements", bench_id), rate);
            }
        }
    }
    (context, unavailable)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::execution::error_result;
    use fluxbench_core::Throughput;
    use fluxbench_report::BenchmarkStatus;
    use fluxbench_stats::{OutlierMethod, compute_summary};

    /// A passed result with `throughput` and 100 ns samples.
    fn passed(id: &'static str, throughput: Option<Throughput>) -> BenchExecutionResult {
        let mut result = error_result(
            &crate::test_support::bench_def(id),
            BenchmarkStatus::Passed,
            String::new(),
            "unknown",
            0,
        );
        result.error_message = None;
        result.failure_kind = None;
        result.samples = vec![100.0; 4];
        result.throughput = throughput;
        result
    }

    fn stats_of(results: &[BenchExecutionResult]) -> Vec<(String, Option<SummaryStatistics>)> {
        results
            .iter()
            .map(|r| {
                let stats = compute_summary(&r.samples, OutlierMethod::Iqr { k: 3 });
                (r.benchmark_id.clone(), Some(stats))
            })
            .collect()
    }

    #[test]
    fn throughput_rates_reach_the_metric_context() {
        let results = [
            passed("copy", Some(Throughput::Bytes(1024))),
            passed("parse", Some(Throughput::Elements(10))),
            passed("plain", None),
        ];
        let (context, unavailable) = metric_context(&results, &stats_of(&results));
        assert!(unavailable.is_empty());
        // 1024 bytes per 100 ns call.
        assert_eq!(context.get("copy_throughput_bytes"), Some(10_240_000_000.0));
        assert_eq!(context.get("copy_throughput_elements"), None);
        assert_eq!(
            context.get("parse_throughput_elements"),
            Some(100_000_000.0)
        );
        assert_eq!(context.get("parse_throughput_bytes"), None);
        assert_eq!(context.get("plain_throughput_bytes"), None);
        assert_eq!(context.get("plain_throughput"), Some(10_000_000.0));
    }

    #[test]
    fn missing_stats_mark_the_benchmark_unavailable() {
        let results = [passed("gone", None)];
        let stats = vec![("gone".to_string(), None)];
        let (context, unavailable) = metric_context(&results, &stats);
        assert!(unavailable.contains("gone"));
        assert_eq!(context.get("gone"), None);
    }
}
