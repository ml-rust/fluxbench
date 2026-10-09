//! Bootstrap Resampling
//!
//! Implements both percentile and BCa (Bias-Corrected and Accelerated) bootstrap
//! methods for computing confidence intervals.

use crate::{BCA_THRESHOLD, DEFAULT_BOOTSTRAP_ITERATIONS, DEFAULT_CONFIDENCE_LEVEL};
use rand::{Rng, RngExt};
use rayon::prelude::*;
use thiserror::Error;

/// Bootstrap configuration
#[derive(Debug, Clone)]
pub struct BootstrapConfig {
    /// Number of bootstrap iterations (default: 100,000)
    pub iterations: usize,
    /// Confidence level (default: 0.95 for 95% CI)
    pub confidence_level: f64,
    /// Whether to use parallel computation
    pub parallel: bool,
    /// Force BCa method even for large samples
    pub force_bca: bool,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            iterations: DEFAULT_BOOTSTRAP_ITERATIONS,
            confidence_level: DEFAULT_CONFIDENCE_LEVEL,
            parallel: true,
            force_bca: false,
        }
    }
}

/// Which bootstrap method was used
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootstrapMethod {
    /// Standard percentile method (for N >= 100)
    Percentile,
    /// BCa method (for small samples or when forced)
    Bca,
}

/// Confidence interval bounds
#[derive(Debug, Clone, Copy)]
pub struct ConfidenceInterval {
    /// Lower bound of the confidence interval
    pub lower: f64,
    /// Upper bound of the confidence interval
    pub upper: f64,
    /// Confidence level (e.g., 0.95 for 95% CI)
    pub level: f64,
}

/// Result of bootstrap analysis
#[derive(Debug, Clone)]
pub struct BootstrapResult {
    /// Point estimate (sample mean)
    pub point_estimate: f64,
    /// Confidence interval
    pub confidence_interval: ConfidenceInterval,
    /// Standard error of the mean
    pub standard_error: f64,
    /// Which method was used
    pub method: BootstrapMethod,
    /// Warning message if any
    pub warning: Option<String>,
}

/// Errors that can occur during bootstrap
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum BootstrapError {
    #[error("Not enough samples: got {got}, need at least {min}")]
    NotEnoughSamples { got: usize, min: usize },

    #[error("Invalid confidence level: {0} (must be between 0 and 1)")]
    InvalidConfidenceLevel(f64),

    #[error("All samples have the same value")]
    NoVariance,

    #[error("Bootstrap iterations is 0: set a count of at least 1, or skip the bootstrap")]
    NoIterations,
}

/// Compute bootstrap confidence interval for the mean
///
/// Automatically selects BCa method for small samples (N < 100).
///
/// # Examples
///
/// ```ignore
/// # use fluxbench_stats::{compute_bootstrap, BootstrapConfig};
/// let samples = vec![100.0, 102.0, 98.0, 101.0, 99.0];
/// let config = BootstrapConfig::default();
/// let result = compute_bootstrap(&samples, &config).unwrap();
/// println!("Mean: {}", result.point_estimate);
/// println!("95% CI: [{}, {}]",
///     result.confidence_interval.lower,
///     result.confidence_interval.upper);
/// ```
pub fn compute_bootstrap(
    samples: &[f64],
    config: &BootstrapConfig,
) -> Result<BootstrapResult, BootstrapError> {
    // Validate inputs
    if samples.len() < 3 {
        return Err(BootstrapError::NotEnoughSamples {
            got: samples.len(),
            min: 3,
        });
    }

    if config.confidence_level <= 0.0 || config.confidence_level >= 1.0 {
        return Err(BootstrapError::InvalidConfidenceLevel(
            config.confidence_level,
        ));
    }

    if config.iterations == 0 {
        return Err(BootstrapError::NoIterations);
    }

    let n = samples.len();
    let point_estimate = mean(samples);

    // Check for zero variance
    let variance = samples
        .iter()
        .map(|x| (x - point_estimate).powi(2))
        .sum::<f64>()
        / n as f64;
    if variance == 0.0 {
        return Ok(BootstrapResult {
            point_estimate,
            confidence_interval: ConfidenceInterval {
                lower: point_estimate,
                upper: point_estimate,
                level: config.confidence_level,
            },
            standard_error: 0.0,
            method: BootstrapMethod::Percentile,
            warning: Some("All samples have identical values".to_string()),
        });
    }

    // Select method based on sample size
    let use_bca = config.force_bca || n < BCA_THRESHOLD;

    // Generate bootstrap distribution
    let bootstrap_means = if config.parallel {
        generate_bootstrap_means_parallel(samples, config.iterations)
    } else {
        generate_bootstrap_means_serial(samples, config.iterations, &mut rand::rng())
    };

    // Sort bootstrap means once (shared by both CI methods and avoids per-call allocation)
    let mut sorted_means = bootstrap_means.clone();
    sorted_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Compute confidence interval
    let (ci, method) = if use_bca {
        let ci = bca_interval(samples, &sorted_means, config.confidence_level);
        (ci, BootstrapMethod::Bca)
    } else {
        let ci = percentile_interval_sorted(&sorted_means, config.confidence_level);
        (ci, BootstrapMethod::Percentile)
    };

    // Compute standard error from bootstrap distribution.
    // By the bootstrap property, mean(bootstrap_means) == point_estimate,
    // so we reuse it directly to avoid a redundant O(N) pass.
    let se = (bootstrap_means
        .iter()
        .map(|x| (x - point_estimate).powi(2))
        .sum::<f64>()
        / bootstrap_means.len() as f64)
        .sqrt();

    let warning = if n < 10 {
        Some("Very small sample size may lead to unreliable estimates".to_string())
    } else {
        None
    };

    Ok(BootstrapResult {
        point_estimate,
        confidence_interval: ConfidenceInterval {
            lower: ci.0,
            upper: ci.1,
            level: config.confidence_level,
        },
        standard_error: se,
        method,
        warning,
    })
}

/// Generate bootstrap means using parallel iteration (Rayon).
///
/// # Panics
///
/// Panics if `samples` is empty. Callers must validate this before calling.
fn generate_bootstrap_means_parallel(samples: &[f64], iterations: usize) -> Vec<f64> {
    assert!(!samples.is_empty(), "samples must not be empty");
    let n = samples.len();
    (0..iterations)
        .into_par_iter()
        .map_init(rand::rng, |rng, _| {
            let mut sum = 0.0;
            for _ in 0..n {
                // SAFETY: index is always in bounds because random_range(0..n) < n
                sum += samples[rng.random_range(0..n)];
            }
            sum / n as f64
        })
        .collect()
}

/// Generate bootstrap means serially from the given generator.
///
/// A seeded generator gives reproducible means.
///
/// # Panics
///
/// Panics if `samples` is empty. Callers must validate this before calling.
fn generate_bootstrap_means_serial<R: Rng + ?Sized>(
    samples: &[f64],
    iterations: usize,
    rng: &mut R,
) -> Vec<f64> {
    assert!(!samples.is_empty(), "samples must not be empty");
    let n = samples.len();
    (0..iterations)
        .map(|_| {
            let mut sum = 0.0;
            for _ in 0..n {
                sum += samples[rng.random_range(0..n)];
            }
            sum / n as f64
        })
        .collect()
}

/// Standard percentile interval from pre-sorted bootstrap means.
///
/// # Arguments
/// * `sorted` — Bootstrap means, **must be pre-sorted in ascending order**.
/// * `confidence` — Confidence level (e.g. 0.95).
///
/// `sorted` must not be empty. `compute_bootstrap` checks this.
fn percentile_interval_sorted(sorted: &[f64], confidence: f64) -> (f64, f64) {
    let n = sorted.len();
    let alpha = (1.0 - confidence) / 2.0;

    let lower_idx = ((alpha * n as f64).floor() as usize).min(n - 1);
    let upper_idx = (((1.0 - alpha) * n as f64).floor() as usize).min(n - 1);

    (sorted[lower_idx], sorted[upper_idx])
}

/// BCa (Bias-Corrected and Accelerated) interval.
///
/// More accurate for small samples and skewed distributions.
///
/// # Arguments
/// * `samples` — Original sample data (unsorted).
/// * `sorted` — Bootstrap means, **must be pre-sorted in ascending order**.
/// * `confidence` — Confidence level (e.g. 0.95).
///
/// `samples` needs at least 2 values and `sorted` must not be empty.
/// `compute_bootstrap` checks both.
fn bca_interval(samples: &[f64], sorted: &[f64], confidence: f64) -> (f64, f64) {
    let n = samples.len();
    let b = sorted.len();

    let theta_hat = mean(samples);

    // Bias correction factor (z0)
    let count_below = sorted.iter().filter(|&&x| x < theta_hat).count();
    let prop = count_below as f64 / b as f64;
    let z0 = normal_quantile(prop.clamp(0.0001, 0.9999));

    // Acceleration factor (a) via jackknife
    let jackknife_means: Vec<f64> = (0..n)
        .map(|i| {
            let sum: f64 = samples
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, &v)| v)
                .sum();
            sum / (n - 1) as f64
        })
        .collect();

    let jack_mean = mean(&jackknife_means);
    let numerator: f64 = jackknife_means
        .iter()
        .map(|x| (jack_mean - x).powi(3))
        .sum();
    let denominator: f64 = jackknife_means
        .iter()
        .map(|x| (jack_mean - x).powi(2))
        .sum();

    let a = if denominator.abs() < 1e-10 {
        0.0
    } else {
        numerator / (6.0 * denominator.powf(1.5))
    };

    // Adjusted percentiles
    let alpha = (1.0 - confidence) / 2.0;
    let z_alpha = normal_quantile(alpha);
    let z_1_alpha = normal_quantile(1.0 - alpha);

    let alpha1 = normal_cdf(z0 + (z0 + z_alpha) / (1.0 - a * (z0 + z_alpha)));
    let alpha2 = normal_cdf(z0 + (z0 + z_1_alpha) / (1.0 - a * (z0 + z_1_alpha)));

    let lower_idx = ((alpha1 * b as f64).floor() as usize).clamp(0, b - 1);
    let upper_idx = ((alpha2 * b as f64).floor() as usize).clamp(0, b - 1);

    (sorted[lower_idx], sorted[upper_idx])
}

/// Compute mean of samples
fn mean(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.iter().sum::<f64>() / samples.len() as f64
}

/// Standard normal quantile (inverse CDF).
///
/// Uses Wichura's algorithm AS241 (PPND16), accurate to about 1e-16.
/// Returns negative infinity for `p <= 0` and positive infinity for `p >= 1`.
// The coefficients are the published AS241 values, so they keep every digit.
#[allow(clippy::excessive_precision)]
fn normal_quantile(p: f64) -> f64 {
    if p.is_nan() {
        return f64::NAN;
    }
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
        let num = (((((((2.509_080_928_730_122_672_7e3 * r + 3.343_057_558_358_812_810_5e4)
            * r
            + 6.726_577_092_700_870_085_3e4)
            * r
            + 4.592_195_393_154_987_145_7e4)
            * r
            + 1.373_169_376_550_946_112_5e4)
            * r
            + 1.971_590_950_306_551_442_7e3)
            * r
            + 1.331_416_678_917_843_774_5e2)
            * r
            + 3.387_132_872_796_366_608_0e0)
            * q;
        let den = ((((((5.226_495_278_852_854_561_0e3 * r + 2.872_908_573_572_194_267_4e4) * r
            + 3.930_789_580_009_271_061_0e4)
            * r
            + 2.121_379_430_158_659_586_7e4)
            * r
            + 5.394_196_021_424_751_107_7e3)
            * r
            + 6.871_870_074_920_579_083_0e2)
            * r
            + 4.231_333_070_160_091_125_2e1)
            * r
            + 1.0;
        return num / den;
    }

    let tail = if q < 0.0 { p } else { 1.0 - p };
    let r = (-tail.ln()).sqrt();
    let x = if r <= 5.0 {
        let r = r - 1.6;
        let num = ((((((7.745_450_142_783_414_076_40e-4 * r + 2.272_384_498_926_918_458_33e-2)
            * r
            + 2.417_807_251_774_506_117_70e-1)
            * r
            + 1.270_458_252_452_368_382_58e0)
            * r
            + 3.647_848_324_763_204_605_04e0)
            * r
            + 5.769_497_221_460_691_405_50e0)
            * r
            + 4.630_337_846_156_545_295_90e0)
            * r
            + 1.423_437_110_749_683_577_34e0;
        let den = ((((((1.050_750_071_644_416_843_24e-9 * r + 5.475_938_084_995_344_946_00e-4)
            * r
            + 1.519_866_656_361_645_719_66e-2)
            * r
            + 1.481_039_764_274_800_745_90e-1)
            * r
            + 6.897_673_349_851_000_045_50e-1)
            * r
            + 1.676_384_830_183_803_849_40e0)
            * r
            + 2.053_191_626_637_758_821_87e0)
            * r
            + 1.0;
        num / den
    } else {
        let r = r - 5.0;
        let num = ((((((2.010_334_399_292_288_132_65e-7 * r + 2.711_555_568_743_487_578_15e-5)
            * r
            + 1.242_660_947_388_078_438_60e-3)
            * r
            + 2.653_218_952_657_612_309_30e-2)
            * r
            + 2.965_605_718_285_048_912_30e-1)
            * r
            + 1.784_826_539_917_291_335_80e0)
            * r
            + 5.463_784_911_164_114_369_90e0)
            * r
            + 6.657_904_643_501_103_777_20e0;
        let den =
            ((((((2.044_263_103_389_939_785_64e-15 * r + 1.421_511_758_316_445_888_70e-7) * r
                + 1.846_318_317_510_054_681_80e-5)
                * r
                + 7.868_691_311_456_132_591_00e-4)
                * r
                + 1.487_536_129_085_061_485_25e-2)
                * r
                + 1.369_298_809_227_358_053_10e-1)
                * r
                + 5.998_322_065_558_879_376_90e-1)
                * r
                + 1.0;
        num / den
    };

    if q < 0.0 { -x } else { x }
}

/// Standard normal CDF
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation (7.1.26)
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_basic_bootstrap() {
        let samples: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let config = BootstrapConfig {
            iterations: 1000,
            ..Default::default()
        };

        let result = compute_bootstrap(&samples, &config).unwrap();

        // Mean should be around 49.5
        assert!((result.point_estimate - 49.5).abs() < 0.1);

        // CI should contain the mean
        assert!(result.confidence_interval.lower < result.point_estimate);
        assert!(result.confidence_interval.upper > result.point_estimate);
    }

    #[test]
    fn test_bca_for_small_samples() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = BootstrapConfig {
            iterations: 1000,
            ..Default::default()
        };

        let result = compute_bootstrap(&samples, &config).unwrap();

        // Should use BCa for small sample
        assert_eq!(result.method, BootstrapMethod::Bca);
    }

    #[test]
    fn test_percentile_for_large_samples() {
        let samples: Vec<f64> = (0..200).map(|x| x as f64).collect();
        let config = BootstrapConfig {
            iterations: 1000,
            force_bca: false,
            ..Default::default()
        };

        let result = compute_bootstrap(&samples, &config).unwrap();

        // Should use percentile for large sample
        assert_eq!(result.method, BootstrapMethod::Percentile);
    }

    #[test]
    fn test_force_bca() {
        let samples: Vec<f64> = (0..200).map(|x| x as f64).collect();
        let config = BootstrapConfig {
            iterations: 1000,
            force_bca: true,
            ..Default::default()
        };

        let result = compute_bootstrap(&samples, &config).unwrap();

        // Should use BCa when forced
        assert_eq!(result.method, BootstrapMethod::Bca);
    }

    #[test]
    fn test_not_enough_samples() {
        let samples = vec![1.0, 2.0];
        let config = BootstrapConfig::default();

        let result = compute_bootstrap(&samples, &config);
        assert!(matches!(
            result,
            Err(BootstrapError::NotEnoughSamples { .. })
        ));
    }

    #[test]
    fn test_zero_iterations() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = BootstrapConfig {
            iterations: 0,
            ..Default::default()
        };

        assert!(matches!(
            compute_bootstrap(&samples, &config),
            Err(BootstrapError::NoIterations)
        ));
    }

    #[test]
    fn test_zero_iterations_identical_samples() {
        let samples = vec![2.0, 2.0, 2.0, 2.0];
        let config = BootstrapConfig {
            iterations: 0,
            ..Default::default()
        };

        assert!(matches!(
            compute_bootstrap(&samples, &config),
            Err(BootstrapError::NoIterations)
        ));
    }

    #[test]
    fn test_single_sample() {
        let config = BootstrapConfig {
            iterations: 100,
            ..Default::default()
        };

        assert!(matches!(
            compute_bootstrap(&[1.0], &config),
            Err(BootstrapError::NotEnoughSamples { got: 1, min: 3 })
        ));
    }

    #[test]
    fn test_normal_quantile_known_values() {
        assert_eq!(normal_quantile(0.5), 0.0);
        assert!((normal_quantile(0.975) - 1.959963984540054).abs() < 1e-9);
        assert!((normal_quantile(0.025) + 1.959963984540054).abs() < 1e-9);
        assert!((normal_quantile(0.8413447460685429) - 1.0).abs() < 1e-9);
        assert!((normal_quantile(0.001) + 3.090232306167813).abs() < 1e-9);
    }

    #[test]
    fn test_normal_quantile_symmetry() {
        for p in [0.001, 0.01, 0.025, 0.1, 0.25, 0.3, 0.45] {
            let lower = normal_quantile(p);
            let upper = normal_quantile(1.0 - p);
            assert!((lower + upper).abs() < 1e-9, "p={p}: {lower} vs {upper}");
            assert!(lower < 0.0, "p={p}: {lower}");
        }
    }

    #[test]
    fn test_normal_quantile_extreme_tails() {
        let low = normal_quantile(1e-10);
        assert!((low + 6.361340902404056).abs() < 1e-9, "q(1e-10)={low}");
        let high = normal_quantile(1.0 - 1e-10);
        assert!((high - 6.361340889697421).abs() < 1e-9, "q(1-1e-10)={high}");
        assert!(normal_quantile(1e-300) < -37.0);
        assert!(low < normal_quantile(1e-9));
        assert_eq!(normal_quantile(0.0), f64::NEG_INFINITY);
        assert_eq!(normal_quantile(1.0), f64::INFINITY);
        assert!(normal_quantile(f64::NAN).is_nan());
    }

    /// Standard normal draw via the Box-Muller transform.
    fn standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
        let u1: f64 = 1.0 - rng.random::<f64>();
        let u2: f64 = rng.random::<f64>();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Seeded synthetic sample sets: normal, skewed, uniform, spiky, small.
    fn synthetic_sets() -> Vec<(&'static str, Vec<f64>)> {
        let mut rng = StdRng::seed_from_u64(0x5eed);
        let normal: Vec<f64> = (0..200)
            .map(|_| 100.0 + 5.0 * standard_normal(&mut rng))
            .collect();
        let skewed: Vec<f64> = (0..150)
            .map(|_| 50.0 - 20.0 * (1.0 - rng.random::<f64>()).ln())
            .collect();
        let uniform: Vec<f64> = (0..120).map(|_| rng.random_range(10.0..20.0)).collect();
        let spiky: Vec<f64> = (0..80)
            .map(|i| {
                let base = 1000.0 + 10.0 * standard_normal(&mut rng);
                if i % 17 == 0 { base * 5.0 } else { base }
            })
            .collect();
        let small: Vec<f64> = (0..12).map(|_| 7.0 + standard_normal(&mut rng)).collect();
        vec![
            ("normal", normal),
            ("skewed", skewed),
            ("uniform", uniform),
            ("spiky", spiky),
            ("small", small),
        ]
    }

    #[test]
    fn test_ci_contains_point_estimate() {
        for (name, samples) in synthetic_sets() {
            for force_bca in [false, true] {
                let config = BootstrapConfig {
                    iterations: 2000,
                    force_bca,
                    ..Default::default()
                };
                let result = compute_bootstrap(&samples, &config).unwrap();
                let ci = result.confidence_interval;
                assert!(
                    ci.lower <= result.point_estimate && result.point_estimate <= ci.upper,
                    "{name} ({:?}): mean {} outside [{}, {}]",
                    result.method,
                    result.point_estimate,
                    ci.lower,
                    ci.upper
                );
                assert!((result.point_estimate - mean(&samples)).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_seeded_percentile_and_bca_contain_mean() {
        let mut rng = StdRng::seed_from_u64(7);
        for (name, samples) in synthetic_sets() {
            let mut sorted = generate_bootstrap_means_serial(&samples, 2000, &mut rng);
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let m = mean(&samples);
            let (lo, hi) = percentile_interval_sorted(&sorted, 0.95);
            assert!(
                lo <= m && m <= hi,
                "{name} percentile: {m} outside [{lo}, {hi}]"
            );
            let (lo, hi) = bca_interval(&samples, &sorted, 0.95);
            assert!(lo <= m && m <= hi, "{name} BCa: {m} outside [{lo}, {hi}]");
        }
    }

    #[test]
    fn test_ci_coverage_on_normal_data() {
        const DATASETS: usize = 200;
        const N: usize = 50;
        const TRUE_MEAN: f64 = 100.0;
        let mut data_rng = StdRng::seed_from_u64(42);
        let mut boot_rng = StdRng::seed_from_u64(4242);
        let mut covered_bca = 0;
        let mut covered_percentile = 0;
        for _ in 0..DATASETS {
            let samples: Vec<f64> = (0..N)
                .map(|_| TRUE_MEAN + 10.0 * standard_normal(&mut data_rng))
                .collect();
            let mut sorted = generate_bootstrap_means_serial(&samples, 500, &mut boot_rng);
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let (lo, hi) = bca_interval(&samples, &sorted, 0.95);
            if lo <= TRUE_MEAN && TRUE_MEAN <= hi {
                covered_bca += 1;
            }
            let (lo, hi) = percentile_interval_sorted(&sorted, 0.95);
            if lo <= TRUE_MEAN && TRUE_MEAN <= hi {
                covered_percentile += 1;
            }
        }
        for (method, covered) in [("BCa", covered_bca), ("percentile", covered_percentile)] {
            let rate = covered as f64 / DATASETS as f64;
            assert!(
                (0.90..=0.99).contains(&rate),
                "{method} coverage {rate} outside [0.90, 0.99]"
            );
        }
    }

    #[test]
    fn test_normal_cdf() {
        // Test known values
        assert!((normal_cdf(0.0) - 0.5).abs() < 0.01);
        assert!((normal_cdf(1.96) - 0.975).abs() < 0.01);
        assert!((normal_cdf(-1.96) - 0.025).abs() < 0.01);
    }
}
