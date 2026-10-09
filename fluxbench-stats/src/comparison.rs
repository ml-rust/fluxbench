//! A/B Comparison Statistics
//!
//! Compares a baseline and a candidate distribution.
//! `compare_distributions` bootstraps raw samples.
//! `compare_summaries` uses a Welch-style z test when only mean, deviation, and count exist.

use crate::bootstrap::{normal_cdf, normal_quantile};
use crate::outliers::OutlierMethod;
use crate::summary::{SummaryStatistics, compute_summary};
use rand::prelude::*;
use rayon::prelude::*;

/// Result of comparing two distributions
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// Summary statistics for baseline
    pub baseline_stats: SummaryStatistics,
    /// Summary statistics for candidate
    pub candidate_stats: SummaryStatistics,
    /// Relative change: (candidate - baseline) / baseline
    pub relative_change: f64,
    /// Absolute change in nanoseconds
    pub absolute_change: f64,
    /// Probability that candidate is slower than baseline (0.0 to 1.0)
    pub probability_regression: f64,
    /// Confidence interval of the difference
    pub difference_ci_lower: f64,
    /// Confidence interval of the difference
    pub difference_ci_upper: f64,
    /// Whether the difference is statistically significant
    pub is_significant: bool,
    /// Effect size (Cohen's d)
    pub effect_size: f64,
    /// Effect size interpretation
    pub effect_interpretation: EffectInterpretation,
}

/// Interpretation of effect size magnitude
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectInterpretation {
    /// |d| < 0.2 - negligible difference
    Negligible,
    /// 0.2 <= |d| < 0.5 - small difference
    Small,
    /// 0.5 <= |d| < 0.8 - medium difference
    Medium,
    /// |d| >= 0.8 - large difference
    Large,
}

impl std::fmt::Display for EffectInterpretation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EffectInterpretation::Negligible => write!(f, "negligible"),
            EffectInterpretation::Small => write!(f, "small"),
            EffectInterpretation::Medium => write!(f, "medium"),
            EffectInterpretation::Large => write!(f, "large"),
        }
    }
}

/// Configuration for comparison
#[derive(Debug, Clone)]
pub struct ComparisonConfig {
    /// Number of bootstrap iterations
    pub bootstrap_iterations: usize,
    /// Confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// Threshold for significance (relative change %)
    pub significance_threshold: f64,
    /// Outlier detection method
    pub outlier_method: OutlierMethod,
}

impl Default for ComparisonConfig {
    fn default() -> Self {
        Self {
            bootstrap_iterations: 10_000,
            confidence_level: 0.95,
            significance_threshold: 5.0, // 5% change threshold
            outlier_method: OutlierMethod::default(),
        }
    }
}

/// Compare two distributions using bootstrap resampling
///
/// Returns the probability that `candidate` is slower than `baseline`,
/// along with effect size and confidence intervals for the difference.
///
/// # Examples
///
/// ```ignore
/// # use fluxbench_stats::{compare_distributions, ComparisonConfig};
/// let baseline = vec![100.0, 102.0, 98.0, 101.0, 99.0];
/// let candidate = vec![105.0, 107.0, 103.0, 106.0, 104.0];
/// let config = ComparisonConfig::default();
/// let result = compare_distributions(&baseline, &candidate, &config).unwrap();
/// println!("Relative change: {:.2}%", result.relative_change);
/// println!("P(regression): {:.2}", result.probability_regression);
/// println!("Effect size: {:.2}", result.effect_size);
/// ```
pub fn compare_distributions(
    baseline: &[f64],
    candidate: &[f64],
    config: &ComparisonConfig,
) -> Result<ComparisonResult, ComparisonError> {
    // Validate inputs
    if baseline.is_empty() {
        return Err(ComparisonError::EmptyBaseline);
    }
    if candidate.is_empty() {
        return Err(ComparisonError::EmptyCandidate);
    }
    if baseline.len() < 2 {
        return Err(ComparisonError::InsufficientBaseline);
    }
    if candidate.len() < 2 {
        return Err(ComparisonError::InsufficientCandidate);
    }
    if config.bootstrap_iterations == 0 {
        return Err(ComparisonError::NoIterations);
    }

    // Compute summary statistics for both
    let baseline_stats = compute_summary(baseline, config.outlier_method);
    let candidate_stats = compute_summary(candidate, config.outlier_method);

    // Summary means are raw-sample means: the same estimator the bootstrap below resamples.
    let observed_diff = candidate_stats.mean - baseline_stats.mean;
    let relative_change = if baseline_stats.mean > 0.0 {
        (observed_diff / baseline_stats.mean) * 100.0
    } else {
        0.0
    };

    // Bootstrap the difference of means
    let bootstrap_diffs: Vec<f64> = (0..config.bootstrap_iterations)
        .into_par_iter()
        .map_init(rand::rng, |rng, _| {
            // Resample baseline
            let baseline_mean: f64 = (0..baseline.len())
                .map(|_| baseline[rng.random_range(0..baseline.len())])
                .sum::<f64>()
                / baseline.len() as f64;

            // Resample candidate
            let candidate_mean: f64 = (0..candidate.len())
                .map(|_| candidate[rng.random_range(0..candidate.len())])
                .sum::<f64>()
                / candidate.len() as f64;

            candidate_mean - baseline_mean
        })
        .collect();

    // Probability of regression (candidate slower = positive difference)
    let regressions = bootstrap_diffs.iter().filter(|&&d| d > 0.0).count();
    let probability_regression = regressions as f64 / config.bootstrap_iterations as f64;

    // Confidence interval of the difference
    let mut sorted_diffs = bootstrap_diffs.clone();
    sorted_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = 1.0 - config.confidence_level;
    let lower_idx = (alpha / 2.0 * config.bootstrap_iterations as f64) as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * config.bootstrap_iterations as f64) as usize;
    let last_idx = sorted_diffs.len() - 1;
    let difference_ci_lower = sorted_diffs[lower_idx.min(last_idx)];
    let difference_ci_upper = sorted_diffs[upper_idx.min(last_idx)];

    let effect_size = cohens_d(
        observed_diff,
        baseline.len(),
        baseline_stats.std_dev,
        candidate.len(),
        candidate_stats.std_dev,
    );
    let effect_interpretation = interpret_effect_size(effect_size);

    // Significance: CI doesn't include zero AND change exceeds threshold
    let ci_excludes_zero = (difference_ci_lower > 0.0) || (difference_ci_upper < 0.0);
    let exceeds_threshold = relative_change.abs() >= config.significance_threshold;
    let is_significant = ci_excludes_zero && exceeds_threshold;

    Ok(ComparisonResult {
        baseline_stats,
        candidate_stats,
        relative_change,
        absolute_change: observed_diff,
        probability_regression,
        difference_ci_lower,
        difference_ci_upper,
        is_significant,
        effect_size,
        effect_interpretation,
    })
}

/// Cohen's d: `difference` divided by the pooled standard deviation.
///
/// `n1`/`std_dev1` describe the baseline, `n2`/`std_dev2` the candidate.
/// Returns 0.0 when the pooled deviation is zero or the result is not finite.
pub fn cohens_d(difference: f64, n1: usize, std_dev1: f64, n2: usize, std_dev2: f64) -> f64 {
    let (n1, n2) = (n1 as f64, n2 as f64);
    let dof = n1 + n2 - 2.0;
    if dof <= 0.0 {
        return 0.0;
    }
    let pooled_var = ((n1 - 1.0) * std_dev1.powi(2) + (n2 - 1.0) * std_dev2.powi(2)) / dof;
    let pooled_std = pooled_var.sqrt();
    if pooled_std.is_nan() || pooled_std <= 0.0 {
        return 0.0;
    }
    let d = difference / pooled_std;
    if d.is_finite() { d } else { 0.0 }
}

/// Mean, standard deviation, and sample count of one distribution.
///
/// `std_dev` is the sample standard deviation (n - 1 denominator).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleSummary {
    /// Arithmetic mean
    pub mean: f64,
    /// Sample standard deviation
    pub std_dev: f64,
    /// Number of samples
    pub count: usize,
}

/// Result of comparing two distributions from summary statistics alone
#[derive(Debug, Clone)]
pub struct SummaryComparison {
    /// Relative change in percent: (candidate - baseline) / baseline * 100
    pub relative_change: f64,
    /// Candidate mean minus baseline mean
    pub absolute_change: f64,
    /// Normal-approximation probability that the candidate mean is larger
    pub probability_regression: f64,
    /// Lower bound of the difference CI at the configured confidence level
    pub difference_ci_lower: f64,
    /// Upper bound of the difference CI at the configured confidence level
    pub difference_ci_upper: f64,
    /// Welch z statistic: difference over its standard error
    pub z_score: f64,
    /// CI excludes zero and the change reaches `significance_threshold`
    pub is_significant: bool,
    /// Effect size (Cohen's d, pooled standard deviation)
    pub effect_size: f64,
    /// Effect size interpretation
    pub effect_interpretation: EffectInterpretation,
}

/// Compare two distributions from mean, standard deviation, and count.
///
/// This is the fallback when raw samples are not available.
/// It uses a Welch-style z test: the difference of means over
/// `sqrt(s1²/n1 + s2²/n2)`, read against the standard normal.
/// The normal approximation holds for large counts. Benchmarks collect 100 samples by default.
/// `config.bootstrap_iterations` and `config.outlier_method` are not used.
pub fn compare_summaries(
    baseline: &SampleSummary,
    candidate: &SampleSummary,
    config: &ComparisonConfig,
) -> Result<SummaryComparison, ComparisonError> {
    if baseline.count < 2 {
        return Err(ComparisonError::InsufficientBaseline);
    }
    if candidate.count < 2 {
        return Err(ComparisonError::InsufficientCandidate);
    }
    if !summary_is_valid(baseline) {
        return Err(ComparisonError::InvalidBaselineSummary);
    }
    if !summary_is_valid(candidate) {
        return Err(ComparisonError::InvalidCandidateSummary);
    }
    let level = config.confidence_level;
    if level.is_nan() || level <= 0.0 || level >= 1.0 {
        return Err(ComparisonError::InvalidConfidenceLevel(level));
    }

    let difference = candidate.mean - baseline.mean;
    let relative_change = if baseline.mean > 0.0 {
        difference / baseline.mean * 100.0
    } else {
        0.0
    };

    let standard_error = (baseline.std_dev.powi(2) / baseline.count as f64
        + candidate.std_dev.powi(2) / candidate.count as f64)
        .sqrt();
    let critical = normal_quantile(1.0 - (1.0 - level) / 2.0);

    // Zero spread on both sides: any nonzero difference is exact.
    let (z_score, probability_regression, half_width) = if standard_error > 0.0 {
        let z = difference / standard_error;
        (z, normal_cdf(z), critical * standard_error)
    } else if difference > 0.0 {
        (f64::INFINITY, 1.0, 0.0)
    } else if difference < 0.0 {
        (f64::NEG_INFINITY, 0.0, 0.0)
    } else {
        (0.0, 0.5, 0.0)
    };

    let difference_ci_lower = difference - half_width;
    let difference_ci_upper = difference + half_width;
    let ci_excludes_zero = difference_ci_lower > 0.0 || difference_ci_upper < 0.0;
    let is_significant = ci_excludes_zero && relative_change.abs() >= config.significance_threshold;

    let effect_size = cohens_d(
        difference,
        baseline.count,
        baseline.std_dev,
        candidate.count,
        candidate.std_dev,
    );

    Ok(SummaryComparison {
        relative_change,
        absolute_change: difference,
        probability_regression,
        difference_ci_lower,
        difference_ci_upper,
        z_score,
        is_significant,
        effect_size,
        effect_interpretation: interpret_effect_size(effect_size),
    })
}

/// A summary is usable when its mean is finite and its deviation is finite and non-negative.
fn summary_is_valid(summary: &SampleSummary) -> bool {
    summary.mean.is_finite() && summary.std_dev.is_finite() && summary.std_dev >= 0.0
}

/// Interpret effect size magnitude using Cohen's conventions
pub fn interpret_effect_size(d: f64) -> EffectInterpretation {
    let abs_d = d.abs();
    if abs_d < 0.2 {
        EffectInterpretation::Negligible
    } else if abs_d < 0.5 {
        EffectInterpretation::Small
    } else if abs_d < 0.8 {
        EffectInterpretation::Medium
    } else {
        EffectInterpretation::Large
    }
}

/// Errors from comparison operations
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum ComparisonError {
    /// Baseline samples are empty
    #[error("Baseline samples are empty")]
    EmptyBaseline,
    /// Candidate samples are empty
    #[error("Candidate samples are empty")]
    EmptyCandidate,
    /// Baseline needs at least 2 samples
    #[error("Baseline needs at least 2 samples")]
    InsufficientBaseline,
    /// Candidate needs at least 2 samples
    #[error("Candidate needs at least 2 samples")]
    InsufficientCandidate,
    /// Bootstrap iterations is 0
    #[error("Bootstrap iterations is 0: set a count of at least 1")]
    NoIterations,
    /// Baseline summary has a non-finite mean or an invalid deviation
    #[error("Baseline summary is invalid: mean and std dev must be finite, std dev non-negative")]
    InvalidBaselineSummary,
    /// Candidate summary has a non-finite mean or an invalid deviation
    #[error("Candidate summary is invalid: mean and std dev must be finite, std dev non-negative")]
    InvalidCandidateSummary,
    /// Confidence level is outside the open interval (0, 1)
    #[error("Confidence level {0} is outside (0, 1): set a value such as 0.95")]
    InvalidConfidenceLevel(f64),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare_identical() {
        let samples = vec![100.0, 102.0, 98.0, 101.0, 99.0, 100.0, 101.0, 99.0];
        let config = ComparisonConfig {
            bootstrap_iterations: 1000,
            ..Default::default()
        };

        let result = compare_distributions(&samples, &samples, &config).unwrap();

        // Should have ~50% probability of regression (random noise)
        assert!(result.probability_regression > 0.3 && result.probability_regression < 0.7);
        assert!(result.relative_change.abs() < 1.0);
        assert!(!result.is_significant);
        assert_eq!(
            result.effect_interpretation,
            EffectInterpretation::Negligible
        );
    }

    #[test]
    fn test_compare_clear_regression() {
        let baseline = vec![100.0, 102.0, 98.0, 101.0, 99.0, 100.0, 101.0, 99.0];
        let candidate = vec![200.0, 202.0, 198.0, 201.0, 199.0, 200.0, 201.0, 199.0];
        let config = ComparisonConfig {
            bootstrap_iterations: 1000,
            ..Default::default()
        };

        let result = compare_distributions(&baseline, &candidate, &config).unwrap();

        // Clear regression - probability should be very high
        assert!(result.probability_regression > 0.95);
        assert!(result.relative_change > 90.0); // ~100% regression
        assert!(result.is_significant);
        assert_eq!(result.effect_interpretation, EffectInterpretation::Large);
    }

    #[test]
    fn test_compare_clear_improvement() {
        let baseline = vec![200.0, 202.0, 198.0, 201.0, 199.0, 200.0, 201.0, 199.0];
        let candidate = vec![100.0, 102.0, 98.0, 101.0, 99.0, 100.0, 101.0, 99.0];
        let config = ComparisonConfig {
            bootstrap_iterations: 1000,
            ..Default::default()
        };

        let result = compare_distributions(&baseline, &candidate, &config).unwrap();

        // Clear improvement - probability should be very low
        assert!(result.probability_regression < 0.05);
        assert!(result.relative_change < -40.0); // ~50% improvement
        assert!(result.is_significant);
        assert_eq!(result.effect_interpretation, EffectInterpretation::Large);
    }

    #[test]
    fn test_comparison_uses_one_estimator() {
        // The baseline spike is an IQR outlier. Every field must still use raw means.
        let baseline = vec![
            100.0, 101.0, 99.0, 100.0, 102.0, 98.0, 100.0, 101.0, 99.0, 400.0,
        ];
        let candidate = vec![
            130.0, 131.0, 129.0, 130.0, 132.0, 128.0, 130.0, 131.0, 129.0, 130.0,
        ];
        let config = ComparisonConfig {
            bootstrap_iterations: 4000,
            ..Default::default()
        };

        let result = compare_distributions(&baseline, &candidate, &config).unwrap();
        assert!(result.baseline_stats.outlier_count > 0);

        let raw_baseline = baseline.iter().sum::<f64>() / baseline.len() as f64;
        let raw_candidate = candidate.iter().sum::<f64>() / candidate.len() as f64;
        let raw_diff = raw_candidate - raw_baseline;
        assert!((result.absolute_change - raw_diff).abs() < 1e-9);
        assert!((result.relative_change - raw_diff / raw_baseline * 100.0).abs() < 1e-9);
        assert!(
            result.difference_ci_lower <= raw_diff && raw_diff <= result.difference_ci_upper,
            "diff {raw_diff} outside [{}, {}]",
            result.difference_ci_lower,
            result.difference_ci_upper
        );

        // Raw means are both 130.0. The trimmed baseline mean of 100.0 must not leak in.
        assert!(result.relative_change.abs() < 1.0);
        assert!(!result.is_significant);
    }

    #[test]
    fn test_effect_size_interpretation() {
        assert_eq!(interpret_effect_size(0.1), EffectInterpretation::Negligible);
        assert_eq!(interpret_effect_size(0.3), EffectInterpretation::Small);
        assert_eq!(interpret_effect_size(0.6), EffectInterpretation::Medium);
        assert_eq!(interpret_effect_size(1.0), EffectInterpretation::Large);
        assert_eq!(interpret_effect_size(-0.5), EffectInterpretation::Medium);
    }

    #[test]
    fn test_empty_samples() {
        let config = ComparisonConfig::default();

        assert!(matches!(
            compare_distributions(&[], &[1.0, 2.0], &config),
            Err(ComparisonError::EmptyBaseline)
        ));
        assert!(matches!(
            compare_distributions(&[1.0, 2.0], &[], &config),
            Err(ComparisonError::EmptyCandidate)
        ));
    }

    #[test]
    fn test_both_empty() {
        let config = ComparisonConfig::default();

        assert!(matches!(
            compare_distributions(&[], &[], &config),
            Err(ComparisonError::EmptyBaseline)
        ));
    }

    #[test]
    fn test_zero_iterations() {
        let config = ComparisonConfig {
            bootstrap_iterations: 0,
            ..Default::default()
        };

        assert!(matches!(
            compare_distributions(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], &config),
            Err(ComparisonError::NoIterations)
        ));
    }

    fn summary(mean: f64, std_dev: f64, count: usize) -> SampleSummary {
        SampleSummary {
            mean,
            std_dev,
            count,
        }
    }

    fn summary_config(threshold: f64) -> ComparisonConfig {
        ComparisonConfig {
            significance_threshold: threshold,
            ..Default::default()
        }
    }

    #[test]
    fn compare_summaries_known_regression() {
        let r = compare_summaries(
            &summary(100.0, 10.0, 100),
            &summary(103.0, 10.0, 100),
            &summary_config(1.0),
        )
        .unwrap();
        assert!((r.relative_change - 3.0).abs() < 1e-12);
        assert!((r.absolute_change - 3.0).abs() < 1e-12);
        assert!((r.z_score - 2.1213203435596424).abs() < 1e-12);
        // normal_cdf uses an erf approximation accurate to about 1.5e-7.
        assert!((r.probability_regression - 0.9830525732376554).abs() < 1e-6);
        assert!((r.difference_ci_lower - 0.22819235130064497).abs() < 1e-9);
        assert!((r.difference_ci_upper - 5.771807648699355).abs() < 1e-9);
        assert!((r.effect_size - 0.3).abs() < 1e-12);
        assert_eq!(r.effect_interpretation, EffectInterpretation::Small);
        assert!(r.is_significant);
    }

    #[test]
    fn compare_summaries_small_change_is_not_significant() {
        let r = compare_summaries(
            &summary(100.0, 10.0, 100),
            &summary(101.0, 10.0, 100),
            &summary_config(0.5),
        )
        .unwrap();
        assert!((r.probability_regression - 0.7602499389065233).abs() < 1e-6);
        assert!((r.difference_ci_lower + 1.771807648699355).abs() < 1e-9);
        assert!((r.difference_ci_upper - 3.771807648699355).abs() < 1e-9);
        assert!(!r.is_significant);
    }

    #[test]
    fn compare_summaries_known_improvement() {
        let r = compare_summaries(
            &summary(100.0, 10.0, 100),
            &summary(97.0, 10.0, 100),
            &summary_config(1.0),
        )
        .unwrap();
        assert!((r.relative_change + 3.0).abs() < 1e-12);
        assert!((r.probability_regression - 0.016947426762344675).abs() < 1e-6);
        assert!(r.difference_ci_upper < 0.0);
        assert!((r.effect_size + 0.3).abs() < 1e-12);
        assert!(r.is_significant);
    }

    #[test]
    fn compare_summaries_unequal_variance_and_count() {
        let r = compare_summaries(
            &summary(200.0, 20.0, 50),
            &summary(210.0, 5.0, 200),
            &summary_config(1.0),
        )
        .unwrap();
        assert!((r.z_score - 3.5082320772281173).abs() < 1e-9);
        assert!((r.probability_regression - 0.9997744523261929).abs() < 1e-6);
        assert!((r.difference_ci_lower - 4.413243076870111).abs() < 1e-9);
        assert!((r.difference_ci_upper - 15.58675692312989).abs() < 1e-9);
        assert!((r.effect_size - 1.0045673924539302).abs() < 1e-9);
        assert_eq!(r.effect_interpretation, EffectInterpretation::Large);
    }

    #[test]
    fn compare_summaries_threshold_gates_significance() {
        // The CI excludes zero, but 3% is under the 5% threshold.
        let r = compare_summaries(
            &summary(100.0, 10.0, 100),
            &summary(103.0, 10.0, 100),
            &summary_config(5.0),
        )
        .unwrap();
        assert!(r.difference_ci_lower > 0.0);
        assert!(!r.is_significant);
    }

    #[test]
    fn compare_summaries_zero_spread() {
        let config = summary_config(1.0);
        let up =
            compare_summaries(&summary(100.0, 0.0, 10), &summary(105.0, 0.0, 10), &config).unwrap();
        assert_eq!(up.probability_regression, 1.0);
        assert_eq!(up.difference_ci_lower, 5.0);
        assert_eq!(up.difference_ci_upper, 5.0);
        assert_eq!(up.effect_size, 0.0);
        assert!(up.is_significant);

        let same =
            compare_summaries(&summary(100.0, 0.0, 10), &summary(100.0, 0.0, 10), &config).unwrap();
        assert_eq!(same.probability_regression, 0.5);
        assert!(!same.is_significant);
    }

    #[test]
    fn compare_summaries_rejects_bad_input() {
        let config = ComparisonConfig::default();
        let good = summary(100.0, 1.0, 10);
        assert!(matches!(
            compare_summaries(&summary(100.0, 1.0, 1), &good, &config),
            Err(ComparisonError::InsufficientBaseline)
        ));
        assert!(matches!(
            compare_summaries(&good, &summary(100.0, 1.0, 1), &config),
            Err(ComparisonError::InsufficientCandidate)
        ));
        assert!(matches!(
            compare_summaries(&summary(f64::NAN, 1.0, 10), &good, &config),
            Err(ComparisonError::InvalidBaselineSummary)
        ));
        assert!(matches!(
            compare_summaries(&good, &summary(100.0, -1.0, 10), &config),
            Err(ComparisonError::InvalidCandidateSummary)
        ));
        let bad_level = ComparisonConfig {
            confidence_level: 1.0,
            ..Default::default()
        };
        assert!(matches!(
            compare_summaries(&good, &good, &bad_level),
            Err(ComparisonError::InvalidConfidenceLevel(_))
        ));
    }

    #[test]
    fn cohens_d_uses_pooled_deviation() {
        assert!((cohens_d(5.0, 10, 10.0, 10, 10.0) - 0.5).abs() < 1e-12);
        // Pooled variance: (9 * 4 + 9 * 16) / 18 = 10.
        assert!((cohens_d(10.0f64.sqrt(), 10, 2.0, 10, 4.0) - 1.0).abs() < 1e-12);
        assert_eq!(cohens_d(5.0, 10, 0.0, 10, 0.0), 0.0);
        assert_eq!(cohens_d(5.0, 1, 1.0, 1, 1.0), 0.0);
    }

    #[test]
    fn test_insufficient_samples() {
        let config = ComparisonConfig::default();

        assert!(matches!(
            compare_distributions(&[1.0], &[1.0, 2.0], &config),
            Err(ComparisonError::InsufficientBaseline)
        ));
        assert!(matches!(
            compare_distributions(&[1.0, 2.0], &[1.0], &config),
            Err(ComparisonError::InsufficientCandidate)
        ));
    }
}
