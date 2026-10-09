//! Summary Statistics
//!
//! Every statistic comes from the raw samples, outliers included.
//! The bootstrap CI estimates the raw mean, so mean and CI share one estimator.
//! Outlier detection still runs. It reports counts, fences, and a separate trimmed mean.

use crate::outliers::{OutlierAnalysis, OutlierMethod, detect_outliers};
use crate::percentiles::compute_percentile;

/// Comprehensive summary statistics
#[derive(Debug, Clone)]
pub struct SummaryStatistics {
    /// Mean of all samples, outliers included
    pub mean: f64,
    /// Median of all samples, outliers included
    pub median: f64,
    /// Sample standard deviation of all samples, outliers included
    pub std_dev: f64,
    /// Mean with detected outliers removed. Equals `mean` when no outlier is found.
    pub trimmed_mean: f64,

    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,

    /// 50th percentile, equal to `median`
    pub p50: f64,
    /// 90th percentile
    pub p90: f64,
    /// 95th percentile
    pub p95: f64,
    /// 99th percentile
    pub p99: f64,
    /// 99.9th percentile
    pub p999: f64,

    /// Skewness of all samples
    pub skewness: f64,
    /// Excess kurtosis of all samples
    pub kurtosis: f64,

    /// Total number of samples (before outlier removal)
    pub sample_count: usize,
    /// Number of outliers detected
    pub outlier_count: usize,
    /// Complete outlier analysis including detection bounds and indices
    pub outlier_analysis: OutlierAnalysis,
}

/// CPU cycles statistics (computed alongside time stats)
#[derive(Debug, Clone, Default)]
pub struct CyclesStatistics {
    /// Mean CPU cycles per iteration
    pub mean_cycles: f64,
    /// Median CPU cycles per iteration
    pub median_cycles: f64,
    /// Standard deviation of cycles
    pub std_dev_cycles: f64,
    /// Minimum cycles observed
    pub min_cycles: u64,
    /// Maximum cycles observed
    pub max_cycles: u64,
    /// Cycles per nanosecond (approximates CPU frequency in GHz)
    pub cycles_per_ns: f64,
}

/// Compute summary statistics from the raw samples.
///
/// Mean, median, stddev, percentiles, extremes, and shape use every sample.
/// They match the estimator that `compute_bootstrap` uses on the same slice.
/// `outlier_method` drives outlier counts, fences, and `trimmed_mean` only.
///
/// # Examples
///
/// ```ignore
/// # use fluxbench_stats::{compute_summary, OutlierMethod};
/// let samples = vec![100.0, 102.0, 98.0, 101.0, 99.0];
/// let stats = compute_summary(&samples, OutlierMethod::default());
/// println!("Mean: {:.2}", stats.mean);
/// println!("P99: {:.2}", stats.p99);
/// println!("Outliers detected: {}", stats.outlier_count);
/// ```
pub fn compute_summary(samples: &[f64], outlier_method: OutlierMethod) -> SummaryStatistics {
    if samples.is_empty() {
        return SummaryStatistics {
            mean: 0.0,
            median: 0.0,
            std_dev: 0.0,
            trimmed_mean: 0.0,
            min: 0.0,
            max: 0.0,
            p50: 0.0,
            p90: 0.0,
            p95: 0.0,
            p99: 0.0,
            p999: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
            sample_count: 0,
            outlier_count: 0,
            outlier_analysis: detect_outliers(samples, outlier_method),
        };
    }

    let analysis = detect_outliers(samples, outlier_method);
    let all = &analysis.all_samples;
    let cleaned = &analysis.cleaned_samples;

    let mean = all.iter().sum::<f64>() / all.len() as f64;
    let median = compute_percentile(all, 50.0);

    let std_dev = if all.len() < 2 {
        0.0
    } else {
        let variance = all.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (all.len() - 1) as f64;
        variance.sqrt()
    };

    // Detection keeps at least one sample in practice. Fall back to the raw mean if not.
    let trimmed_mean = if cleaned.is_empty() {
        mean
    } else {
        cleaned.iter().sum::<f64>() / cleaned.len() as f64
    };

    let min = all
        .iter()
        .cloned()
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0);
    let max = all
        .iter()
        .cloned()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0);

    // Outliers stay in the tails: they are the tail signal.
    let p50 = median;
    let p90 = compute_percentile(all, 90.0);
    let p95 = compute_percentile(all, 95.0);
    let p99 = compute_percentile(all, 99.0);
    let p999 = compute_percentile(all, 99.9);

    // Fisher's definitions over all samples
    let (skewness, kurtosis) = if all.len() < 3 || std_dev < f64::EPSILON {
        (0.0, 0.0)
    } else {
        let n = all.len() as f64;
        let mut m3 = 0.0;
        let mut m4 = 0.0;
        for &x in all.iter() {
            let d = (x - mean) / std_dev;
            let d2 = d * d;
            m3 += d2 * d;
            m4 += d2 * d2;
        }
        (m3 / n, m4 / n - 3.0)
    };

    SummaryStatistics {
        mean,
        median,
        std_dev,
        trimmed_mean,
        min,
        max,
        p50,
        p90,
        p95,
        p99,
        p999,
        skewness,
        kurtosis,
        sample_count: all.len(),
        outlier_count: analysis.outlier_indices.len(),
        outlier_analysis: analysis,
    }
}

impl SummaryStatistics {
    /// Coefficient of variation (relative stddev)
    pub fn coefficient_of_variation(&self) -> f64 {
        if self.mean == 0.0 {
            0.0
        } else {
            (self.std_dev / self.mean) * 100.0
        }
    }

    /// Interquartile range (from all data)
    pub fn iqr(&self) -> f64 {
        let q1 = compute_percentile(&self.outlier_analysis.all_samples, 25.0);
        let q3 = compute_percentile(&self.outlier_analysis.all_samples, 75.0);
        q3 - q1
    }

    /// Check if distribution appears stable (low CV)
    pub fn is_stable(&self, cv_threshold: f64) -> bool {
        self.coefficient_of_variation() < cv_threshold
    }

    /// Classify the distribution shape based on skewness and kurtosis.
    ///
    /// Returns a string like "symmetric, normal-tailed" or "right-skewed, heavy-tailed".
    pub fn distribution_shape(&self) -> String {
        let skew_label = if self.skewness.abs() < 0.5 {
            "symmetric"
        } else if self.skewness > 0.0 {
            "right-skewed"
        } else {
            "left-skewed"
        };

        let kurt_label = if self.kurtosis.abs() < 1.0 {
            "normal-tailed"
        } else if self.kurtosis > 0.0 {
            "heavy-tailed"
        } else {
            "light-tailed"
        };

        format!("{}, {}", skew_label, kurt_label)
    }
}

/// Compute CPU cycles statistics from per-sample mean cycle counts
///
/// Takes parallel arrays of cycles and nanos to compute cycles_per_ns ratio.
/// Min and max round to the nearest whole cycle.
pub fn compute_cycles_stats(cycles: &[f64], nanos: &[f64]) -> CyclesStatistics {
    if cycles.is_empty() {
        return CyclesStatistics::default();
    }

    // Mean cycles
    let mean_cycles = cycles.iter().sum::<f64>() / cycles.len() as f64;

    // Median cycles
    let mut sorted = cycles.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_cycles = if sorted.len() % 2 == 0 {
        let mid = sorted.len() / 2;
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };

    // Std dev
    let std_dev_cycles = if cycles.len() < 2 {
        0.0
    } else {
        let variance = cycles
            .iter()
            .map(|x| (x - mean_cycles).powi(2))
            .sum::<f64>()
            / (cycles.len() - 1) as f64;
        variance.sqrt()
    };

    // Min/max
    let min_cycles = sorted.first().map_or(0, |c| c.round() as u64);
    let max_cycles = sorted.last().map_or(0, |c| c.round() as u64);

    // Cycles per nanosecond (CPU frequency approximation)
    let cycles_per_ns = if !nanos.is_empty() {
        let total_nanos: f64 = nanos.iter().sum();
        let total_cycles: f64 = cycles.iter().sum();
        if total_nanos > 0.0 {
            total_cycles / total_nanos
        } else {
            0.0
        }
    } else {
        0.0
    };

    CyclesStatistics {
        mean_cycles,
        median_cycles,
        std_dev_cycles,
        min_cycles,
        max_cycles,
        cycles_per_ns,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_summary() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let summary = compute_summary(&samples, OutlierMethod::default());

        assert!((summary.mean - 3.0).abs() < 0.01);
        assert!((summary.median - 3.0).abs() < 0.01);
        assert_eq!(summary.min, 1.0);
        assert_eq!(summary.max, 5.0);
        assert_eq!(summary.sample_count, 5);
    }

    #[test]
    fn test_outlier_handling() {
        // 100.0 is an outlier
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let summary = compute_summary(&samples, OutlierMethod::default());

        // Mean includes the outlier: (1+2+3+4+5+100)/6 = 19.1667
        assert!((summary.mean - 115.0 / 6.0).abs() < 1e-9);
        // Trimmed mean drops it
        assert!((summary.trimmed_mean - 3.0).abs() < 1e-9);
        // Median and p50 come from the same samples
        assert_eq!(summary.median, summary.p50);

        // Max should be from ALL data (include 100.0)
        assert_eq!(summary.max, 100.0);

        // p99 should include the outlier
        assert!(summary.p99 > 50.0);

        assert_eq!(summary.outlier_count, 1);
    }

    #[test]
    fn test_trimmed_mean_equals_mean_without_outliers() {
        let samples = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let summary = compute_summary(&samples, OutlierMethod::default());
        assert_eq!(summary.outlier_count, 0);
        assert!((summary.trimmed_mean - summary.mean).abs() < 1e-12);
    }

    #[test]
    fn test_mean_matches_bootstrap_estimator() {
        use crate::bootstrap::{BootstrapConfig, compute_bootstrap};

        let samples = vec![10.0, 11.0, 9.0, 10.5, 9.5, 10.0, 11.0, 9.0, 80.0, 10.0];
        let summary = compute_summary(&samples, OutlierMethod::default());
        assert!(summary.outlier_count > 0);

        let config = BootstrapConfig {
            iterations: 2000,
            ..Default::default()
        };
        let boot = compute_bootstrap(&samples, &config).unwrap();
        assert!((boot.point_estimate - summary.mean).abs() < 1e-9);
        let ci = boot.confidence_interval;
        assert!(ci.lower <= summary.mean && summary.mean <= ci.upper);
    }

    #[test]
    fn test_coefficient_of_variation() {
        let samples = vec![100.0, 100.0, 100.0, 100.0, 100.0];
        let summary = compute_summary(&samples, OutlierMethod::None);

        // Zero variance = zero CV
        assert!((summary.coefficient_of_variation() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_samples() {
        let samples: Vec<f64> = Vec::new();
        let summary = compute_summary(&samples, OutlierMethod::default());

        assert_eq!(summary.sample_count, 0);
        assert!((summary.mean - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_skewness_symmetric() {
        // Symmetric data should have skewness ≈ 0
        let samples: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let summary = compute_summary(&samples, OutlierMethod::None);
        assert!(
            summary.skewness.abs() < 0.1,
            "skewness={}",
            summary.skewness
        );
        // Uniform distribution has excess kurtosis ≈ -1.2 (light-tailed)
        assert!(summary.kurtosis < 0.0, "kurtosis={}", summary.kurtosis);
        assert!(summary.distribution_shape().contains("symmetric"));
    }

    #[test]
    fn test_skewness_right_skewed() {
        // Right-skewed data (exponential-like)
        let samples: Vec<f64> = (0..200).map(|x| (x as f64 * 0.05).exp()).collect();
        let summary = compute_summary(&samples, OutlierMethod::None);
        assert!(summary.skewness > 0.5, "skewness={}", summary.skewness);
        assert!(summary.distribution_shape().contains("right-skewed"));
    }

    #[test]
    fn test_skewness_left_skewed() {
        // Left-skewed data (mirror of exponential)
        let samples: Vec<f64> = (0..200).map(|x| 200.0 - (x as f64 * 0.05).exp()).collect();
        let summary = compute_summary(&samples, OutlierMethod::None);
        assert!(summary.skewness < -0.5, "skewness={}", summary.skewness);
        assert!(summary.distribution_shape().contains("left-skewed"));
    }

    #[test]
    fn test_skewness_constant_values() {
        let samples = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let summary = compute_summary(&samples, OutlierMethod::None);
        assert_eq!(summary.skewness, 0.0);
        assert_eq!(summary.kurtosis, 0.0);
    }

    #[test]
    fn test_skewness_two_samples() {
        let samples = vec![1.0, 2.0];
        let summary = compute_summary(&samples, OutlierMethod::None);
        // < 3 samples: returns (0.0, 0.0)
        assert_eq!(summary.skewness, 0.0);
        assert_eq!(summary.kurtosis, 0.0);
    }

    #[test]
    fn test_kurtosis_normal() {
        // Uniform distribution has negative excess kurtosis (light-tailed)
        let samples: Vec<f64> = (0..1000).map(|x| x as f64).collect();
        let summary = compute_summary(&samples, OutlierMethod::None);
        // Uniform: kurtosis = -1.2
        assert!(summary.kurtosis < 0.0, "kurtosis={}", summary.kurtosis);
    }

    #[test]
    fn test_cycles_stats() {
        let cycles = vec![3000.0, 3100.4, 2899.6, 3050.0, 2950.0];
        let nanos = vec![1000.0, 1033.0, 967.0, 1017.0, 983.0];
        let stats = compute_cycles_stats(&cycles, &nanos);

        // Mean should be ~3000
        assert!((stats.mean_cycles - 3000.0).abs() < 50.0);
        assert_eq!(stats.min_cycles, 2900);
        assert_eq!(stats.max_cycles, 3100);
        // ~3 cycles per ns
        assert!((stats.cycles_per_ns - 3.0).abs() < 0.5);
    }

    #[test]
    fn test_cycles_stats_empty() {
        let cycles: Vec<f64> = vec![];
        let nanos: Vec<f64> = vec![];
        let stats = compute_cycles_stats(&cycles, &nanos);

        assert!((stats.mean_cycles - 0.0).abs() < f64::EPSILON);
        assert_eq!(stats.min_cycles, 0);
        assert_eq!(stats.max_cycles, 0);
    }
}
