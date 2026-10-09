//! Baseline comparison of a whole report.

use super::estimate::{EstimateInput, estimate};
use super::verdict::{Verdict, classify, is_significant};
use fluxbench_report::{BenchmarkReportResult, Comparison, Report};
use std::collections::HashMap;

/// Settings for one baseline comparison
#[derive(Debug, Clone, Copy)]
pub(crate) struct ComparisonSettings {
    /// Global regression threshold in percent. A per-benchmark threshold above 0 replaces it.
    pub threshold: f64,
    /// Confidence level for the difference CI and the significance check
    pub confidence_level: f64,
    /// Bootstrap iterations. 0 selects the summary test.
    pub bootstrap_iterations: usize,
}

/// Compare `report` against `baseline` and fill each result's `comparison`.
///
/// `current_samples` maps benchmark IDs to this run's raw samples.
/// It adds the counted regressions and improvements to `report.summary`.
/// Returns one warning per benchmark that has a baseline entry but no usable comparison.
pub(crate) fn apply_baseline_comparison(
    report: &mut Report,
    baseline: &Report,
    current_samples: &HashMap<&str, &[f64]>,
    settings: &ComparisonSettings,
) -> Vec<String> {
    report.baseline_meta = Some(baseline.meta.clone());

    let baseline_map: HashMap<&str, &BenchmarkReportResult> = baseline
        .results
        .iter()
        .filter(|r| r.metrics.is_some())
        .map(|r| (r.id.as_str(), r))
        .collect();

    let mut warnings = Vec::new();
    for result in &mut report.results {
        let Some(base) = baseline_map.get(result.id.as_str()) else {
            continue;
        };
        match compare_one(result, base, current_samples, settings) {
            Ok(Some((comparison, verdict))) => {
                match verdict {
                    Verdict::Regression => report.summary.regressions += 1,
                    Verdict::Improvement => report.summary.improvements += 1,
                    Verdict::NoChange => {}
                }
                result.comparison = Some(comparison);
            }
            Ok(None) => {}
            Err(message) => warnings.push(message),
        }
    }
    warnings
}

/// Compare one result. `Ok(None)` means the current run has no metrics.
fn compare_one(
    result: &BenchmarkReportResult,
    base: &BenchmarkReportResult,
    current_samples: &HashMap<&str, &[f64]>,
    settings: &ComparisonSettings,
) -> Result<Option<(Comparison, Verdict)>, String> {
    let (Some(metrics), Some(base_metrics)) = (&result.metrics, &base.metrics) else {
        return Ok(None);
    };

    let threshold = if result.threshold > 0.0 {
        result.threshold
    } else {
        settings.threshold
    };

    let est = estimate(&EstimateInput {
        baseline: base_metrics,
        baseline_samples: base.samples.as_deref(),
        current: metrics,
        current_samples: current_samples.get(result.id.as_str()).copied(),
        confidence_level: settings.confidence_level,
        bootstrap_iterations: settings.bootstrap_iterations,
    })
    .map_err(|e| {
        format!(
            "Warning: baseline comparison skipped for {}: {e}. Re-save the baseline with --save-baseline.",
            result.id
        )
    })?;

    let significant = is_significant(
        Some(est.difference_ci),
        est.relative_change,
        est.probability_regression,
        settings.confidence_level,
    );
    let verdict = classify(est.relative_change, threshold, significant);

    let comparison = Comparison {
        baseline_mean_ns: base_metrics.mean_ns,
        absolute_change_ns: est.absolute_change_ns,
        relative_change: est.relative_change,
        probability_regression: est.probability_regression,
        is_significant: verdict != Verdict::NoChange,
        effect_size: est.effect_size,
        threshold,
        difference_ci_lower_ns: Some(est.difference_ci.0),
        difference_ci_upper_ns: Some(est.difference_ci.1),
        confidence_level: settings.confidence_level,
        method: est.method,
    };
    Ok(Some((comparison, verdict)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{dummy_report, dummy_result};
    use fluxbench_report::ComparisonMethod;

    fn settings(threshold: f64) -> ComparisonSettings {
        ComparisonSettings {
            threshold,
            confidence_level: 0.95,
            bootstrap_iterations: 2_000,
        }
    }

    fn no_samples() -> HashMap<&'static str, &'static [f64]> {
        HashMap::new()
    }

    fn apply(report: &mut Report, baseline: &Report, threshold: f64) -> Vec<String> {
        apply_baseline_comparison(report, baseline, &no_samples(), &settings(threshold))
    }

    #[test]
    fn per_bench_threshold_overrides_global() {
        // Baseline: 100ns. Current: 108ns → 8% regression.
        // Global threshold: 25%. Per-bench threshold: 5%.
        // Should detect regression via per-bench threshold but not global.
        let mut report = dummy_report(vec![dummy_result("fast_bench", 108.0, 5.0)]);
        let baseline = dummy_report(vec![dummy_result("fast_bench", 100.0, 5.0)]);

        apply(&mut report, &baseline, 25.0);

        assert_eq!(
            report.summary.regressions, 1,
            "per-bench 5% should catch 8% regression"
        );
        let cmp = report.results[0].comparison.as_ref().unwrap();
        assert!(cmp.is_significant);
        assert_eq!(cmp.threshold, 5.0);
    }

    #[test]
    fn zero_threshold_falls_back_to_global() {
        // Baseline: 100ns. Current: 108ns → 8% regression.
        // Global threshold: 25%. Per-bench threshold: 0.0 (use global).
        // 8% < 25%, so no regression.
        let mut report = dummy_report(vec![dummy_result("normal_bench", 108.0, 0.0)]);
        let baseline = dummy_report(vec![dummy_result("normal_bench", 100.0, 0.0)]);

        apply(&mut report, &baseline, 25.0);

        assert_eq!(
            report.summary.regressions, 0,
            "8% under 25% global should not regress"
        );
        let cmp = report.results[0].comparison.as_ref().unwrap();
        assert!(!cmp.is_significant);
        assert_eq!(cmp.threshold, 25.0);
    }

    #[test]
    fn mixed_thresholds_independent() {
        // Two benchmarks: one with tight per-bench threshold, one using global.
        // Both regress by 8%.
        let mut report = dummy_report(vec![
            dummy_result("tight", 108.0, 5.0), // per-bench 5% → should regress
            dummy_result("loose", 108.0, 0.0), // global 25% → should not
        ]);
        let baseline = dummy_report(vec![
            dummy_result("tight", 100.0, 5.0),
            dummy_result("loose", 100.0, 0.0),
        ]);

        apply(&mut report, &baseline, 25.0);

        assert_eq!(report.summary.regressions, 1);
        assert!(
            report.results[0]
                .comparison
                .as_ref()
                .unwrap()
                .is_significant
        );
        assert!(
            !report.results[1]
                .comparison
                .as_ref()
                .unwrap()
                .is_significant
        );
    }

    #[test]
    fn per_bench_threshold_detects_improvement() {
        // Baseline: 100ns. Current: 90ns → -10% improvement.
        // Per-bench threshold: 5%.
        let mut report = dummy_report(vec![dummy_result("improving", 90.0, 5.0)]);
        let baseline = dummy_report(vec![dummy_result("improving", 100.0, 5.0)]);

        apply(&mut report, &baseline, 25.0);

        assert_eq!(report.summary.improvements, 1);
        assert_eq!(report.summary.regressions, 0);
        assert!(
            report.results[0]
                .comparison
                .as_ref()
                .unwrap()
                .is_improvement()
        );
    }

    #[test]
    fn large_but_noisy_change_is_not_a_regression() {
        // +10% over a 5% threshold, but a std dev of 100% of the mean hides it.
        let mut current = dummy_result("noisy", 110.0, 0.0);
        let mut base = dummy_result("noisy", 100.0, 0.0);
        for r in [&mut current, &mut base] {
            let m = r.metrics.as_mut().unwrap();
            m.std_dev_ns = m.mean_ns;
            m.samples = 10;
        }
        let mut report = dummy_report(vec![current]);
        let baseline = dummy_report(vec![base]);

        apply(&mut report, &baseline, 5.0);

        assert_eq!(report.summary.regressions, 0);
        let cmp = report.results[0].comparison.as_ref().unwrap();
        assert!(!cmp.is_significant);
        assert!(cmp.relative_change > 5.0);
        let (lo, hi) = (
            cmp.difference_ci_lower_ns.unwrap(),
            cmp.difference_ci_upper_ns.unwrap(),
        );
        assert!(lo < 0.0 && hi > 0.0, "CI [{lo}, {hi}] should span zero");
        assert!(cmp.probability_regression > 0.5 && cmp.probability_regression < 0.95);
    }

    #[test]
    fn significant_change_inside_threshold_is_not_counted() {
        // +3% is significant at 1% spread and 100 samples, but under 5%.
        let mut report = dummy_report(vec![dummy_result("small", 103.0, 0.0)]);
        let baseline = dummy_report(vec![dummy_result("small", 100.0, 0.0)]);

        apply(&mut report, &baseline, 5.0);

        assert_eq!(report.summary.regressions, 0);
        let cmp = report.results[0].comparison.as_ref().unwrap();
        assert!(!cmp.is_significant);
        assert!(cmp.difference_ci_lower_ns.unwrap() > 0.0);
    }

    #[test]
    fn summary_path_reports_method_and_probability() {
        let mut report = dummy_report(vec![dummy_result("b", 108.0, 0.0)]);
        let baseline = dummy_report(vec![dummy_result("b", 100.0, 0.0)]);

        apply(&mut report, &baseline, 5.0);

        let cmp = report.results[0].comparison.as_ref().unwrap();
        assert_eq!(cmp.method, ComparisonMethod::Summary);
        assert!(cmp.probability_regression > 0.999);
        assert!((cmp.relative_change - 8.0).abs() < 1e-9);
        assert_eq!(cmp.confidence_level, 0.95);
    }

    #[test]
    fn samples_on_both_sides_select_bootstrap() {
        let base_samples: Vec<f64> = (0..50).map(|i| 100.0 + (i % 5) as f64).collect();
        let cur_samples: Vec<f64> = (0..50).map(|i| 120.0 + (i % 5) as f64).collect();

        let mut base = dummy_result("b", 102.0, 0.0);
        base.samples = Some(base_samples);
        let mut report = dummy_report(vec![dummy_result("b", 122.0, 0.0)]);
        let baseline = dummy_report(vec![base]);
        let mut current = HashMap::new();
        current.insert("b", cur_samples.as_slice());

        apply_baseline_comparison(&mut report, &baseline, &current, &settings(5.0));

        let cmp = report.results[0].comparison.as_ref().unwrap();
        assert_eq!(cmp.method, ComparisonMethod::Bootstrap);
        assert!(cmp.is_regression());
        assert!(cmp.probability_regression > 0.99);
        assert!(cmp.difference_ci_lower_ns.unwrap() > 0.0);
        assert!(cmp.effect_size > 0.8);
        assert_eq!(report.summary.regressions, 1);
    }

    #[test]
    fn baseline_without_samples_falls_back_to_summary() {
        let cur_samples: Vec<f64> = (0..50).map(|i| 120.0 + (i % 5) as f64).collect();
        let mut report = dummy_report(vec![dummy_result("b", 122.0, 0.0)]);
        let baseline = dummy_report(vec![dummy_result("b", 102.0, 0.0)]);
        let mut current = HashMap::new();
        current.insert("b", cur_samples.as_slice());

        apply_baseline_comparison(&mut report, &baseline, &current, &settings(5.0));

        let cmp = report.results[0].comparison.as_ref().unwrap();
        assert_eq!(cmp.method, ComparisonMethod::Summary);
        assert!(cmp.is_regression());
    }

    #[test]
    fn zero_bootstrap_iterations_select_summary() {
        let samples: Vec<f64> = (0..50).map(|i| 100.0 + (i % 5) as f64).collect();
        let mut base = dummy_result("b", 102.0, 0.0);
        base.samples = Some(samples.clone());
        let mut report = dummy_report(vec![dummy_result("b", 102.0, 0.0)]);
        let baseline = dummy_report(vec![base]);
        let mut current = HashMap::new();
        current.insert("b", samples.as_slice());
        let off = ComparisonSettings {
            bootstrap_iterations: 0,
            ..settings(5.0)
        };

        apply_baseline_comparison(&mut report, &baseline, &current, &off);

        let cmp = report.results[0].comparison.as_ref().unwrap();
        assert_eq!(cmp.method, ComparisonMethod::Summary);
        assert!(!cmp.is_significant);
    }

    #[test]
    fn unusable_summary_yields_warning_and_no_comparison() {
        let mut current = dummy_result("one_sample", 108.0, 0.0);
        current.metrics.as_mut().unwrap().samples = 1;
        let mut report = dummy_report(vec![current]);
        let baseline = dummy_report(vec![dummy_result("one_sample", 100.0, 0.0)]);

        let warnings = apply(&mut report, &baseline, 5.0);

        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("one_sample"), "{}", warnings[0]);
        assert!(report.results[0].comparison.is_none());
        assert_eq!(report.summary.regressions, 0);
    }

    #[test]
    fn benchmark_missing_from_baseline_is_skipped() {
        let mut report = dummy_report(vec![dummy_result("new_bench", 108.0, 0.0)]);
        let baseline = dummy_report(vec![dummy_result("old_bench", 100.0, 0.0)]);

        let warnings = apply(&mut report, &baseline, 5.0);

        assert!(warnings.is_empty());
        assert!(report.results[0].comparison.is_none());
        assert!(report.baseline_meta.is_some());
    }
}
