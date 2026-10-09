//! Regression and improvement rules.
//!
//! A change counts only when it is statistically significant and larger than the threshold.

/// Outcome of one benchmark against its baseline
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Verdict {
    /// Significant slowdown beyond the threshold
    Regression,
    /// Significant speedup beyond the threshold
    Improvement,
    /// Not significant, or within the threshold
    NoChange,
}

/// Check statistical significance of a change at `confidence`.
///
/// A change is significant when the difference CI excludes zero.
/// It is also significant when the probability of its direction reaches `confidence`.
/// A zero change is never significant.
pub(crate) fn is_significant(
    difference_ci: Option<(f64, f64)>,
    relative_change: f64,
    probability_regression: f64,
    confidence: f64,
) -> bool {
    let ci_excludes_zero = difference_ci.is_some_and(|(lower, upper)| lower > 0.0 || upper < 0.0);
    let direction_probability = if relative_change > 0.0 {
        probability_regression
    } else if relative_change < 0.0 {
        1.0 - probability_regression
    } else {
        return false;
    };
    ci_excludes_zero || direction_probability >= confidence
}

/// Classify a change in percent against `threshold` percent.
///
/// The change must be strictly larger than the threshold.
pub(crate) fn classify(relative_change: f64, threshold: f64, significant: bool) -> Verdict {
    if !significant {
        Verdict::NoChange
    } else if relative_change > threshold {
        Verdict::Regression
    } else if relative_change < -threshold {
        Verdict::Improvement
    } else {
        Verdict::NoChange
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ci_excluding_zero_is_significant() {
        assert!(is_significant(Some((0.5, 3.0)), 2.0, 0.6, 0.95));
        assert!(is_significant(Some((-3.0, -0.5)), -2.0, 0.4, 0.95));
    }

    #[test]
    fn direction_probability_alone_is_significant() {
        assert!(is_significant(Some((-0.1, 3.0)), 2.0, 0.96, 0.95));
        assert!(is_significant(None, -2.0, 0.03, 0.95));
    }

    #[test]
    fn ci_spanning_zero_with_low_probability_is_not_significant() {
        assert!(!is_significant(Some((-1.0, 3.0)), 2.0, 0.80, 0.95));
        assert!(!is_significant(Some((-3.0, 1.0)), -2.0, 0.20, 0.95));
        assert!(!is_significant(None, 2.0, 0.5, 0.95));
    }

    #[test]
    fn zero_change_is_never_significant() {
        assert!(!is_significant(Some((0.1, 0.2)), 0.0, 1.0, 0.95));
    }

    #[test]
    fn insignificant_change_is_no_change_at_any_size() {
        assert_eq!(classify(50.0, 5.0, false), Verdict::NoChange);
        assert_eq!(classify(-50.0, 5.0, false), Verdict::NoChange);
    }

    #[test]
    fn significant_change_must_exceed_threshold() {
        assert_eq!(classify(5.0, 5.0, true), Verdict::NoChange);
        assert_eq!(classify(5.01, 5.0, true), Verdict::Regression);
        assert_eq!(classify(4.99, 5.0, true), Verdict::NoChange);
    }

    #[test]
    fn improvement_mirrors_regression() {
        assert_eq!(classify(-5.0, 5.0, true), Verdict::NoChange);
        assert_eq!(classify(-5.01, 5.0, true), Verdict::Improvement);
    }

    #[test]
    fn zero_threshold_counts_any_significant_change() {
        assert_eq!(classify(0.1, 0.0, true), Verdict::Regression);
        assert_eq!(classify(-0.1, 0.0, true), Verdict::Improvement);
    }
}
