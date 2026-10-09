//! Statistical estimate of the change between a baseline and a current run.
//!
//! With raw samples on both sides and the bootstrap on, it bootstraps the difference of means.
//! Otherwise it falls back to a Welch-style z test on mean, std dev, and sample count.
//! Schema 1 baselines carry no samples, so they always take the fallback.

use fluxbench_report::{BenchmarkMetrics, ComparisonMethod};
use fluxbench_stats::{
    ComparisonConfig, ComparisonError, OutlierMethod, SampleSummary, compare_distributions,
    compare_summaries,
};

/// Change of the current run relative to the baseline
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Estimate {
    /// Current mean minus baseline mean, in nanoseconds
    pub absolute_change_ns: f64,
    /// Change in percent of the baseline mean
    pub relative_change: f64,
    /// Probability that the current mean is larger
    pub probability_regression: f64,
    /// CI of the difference of means, in nanoseconds
    pub difference_ci: (f64, f64),
    /// Cohen's d with the pooled standard deviation
    pub effect_size: f64,
    /// Method behind the probability and the CI
    pub method: ComparisonMethod,
}

/// Inputs for one estimate
pub(crate) struct EstimateInput<'a> {
    pub baseline: &'a BenchmarkMetrics,
    pub baseline_samples: Option<&'a [f64]>,
    pub current: &'a BenchmarkMetrics,
    pub current_samples: Option<&'a [f64]>,
    pub confidence_level: f64,
    pub bootstrap_iterations: usize,
}

/// Estimate the change. The bootstrap runs when both sides have samples and iterations are set.
///
/// A bootstrap error falls back to the summary test. A summary error returns the error.
pub(crate) fn estimate(input: &EstimateInput<'_>) -> Result<Estimate, ComparisonError> {
    let config = ComparisonConfig {
        bootstrap_iterations: input.bootstrap_iterations,
        confidence_level: input.confidence_level,
        // Thresholds apply later, in `verdict::classify`.
        significance_threshold: 0.0,
        outlier_method: OutlierMethod::default(),
    };

    if let (Some(baseline), Some(current)) = (input.baseline_samples, input.current_samples) {
        if input.bootstrap_iterations > 0 {
            if let Ok(r) = compare_distributions(baseline, current, &config) {
                return Ok(Estimate {
                    absolute_change_ns: r.absolute_change,
                    relative_change: r.relative_change,
                    probability_regression: r.probability_regression,
                    difference_ci: (r.difference_ci_lower, r.difference_ci_upper),
                    effect_size: r.effect_size,
                    method: ComparisonMethod::Bootstrap,
                });
            }
        }
    }

    let r = compare_summaries(
        &summary_of(input.baseline),
        &summary_of(input.current),
        &config,
    )?;
    Ok(Estimate {
        absolute_change_ns: r.absolute_change,
        relative_change: r.relative_change,
        probability_regression: r.probability_regression,
        difference_ci: (r.difference_ci_lower, r.difference_ci_upper),
        effect_size: r.effect_size,
        method: ComparisonMethod::Summary,
    })
}

/// Mean, deviation, and count of a report entry. All three cover every sample, outliers included.
fn summary_of(metrics: &BenchmarkMetrics) -> SampleSummary {
    SampleSummary {
        mean: metrics.mean_ns,
        std_dev: metrics.std_dev_ns,
        count: metrics.samples,
    }
}
