//! Human-readable baseline comparison.

use crate::executor::ci_label;
use fluxbench_core::Severity;
use fluxbench_report::{BenchmarkStatus, Comparison, ComparisonMethod, Report, format_duration};
use fluxbench_stats::interpret_effect_size;

/// Format the baseline comparison section for terminal display.
///
/// Each compared benchmark shows its change, verdict, threshold, probability,
/// difference CI, and effect size.
pub(crate) fn format_comparison_output(report: &Report, baseline: &Report) -> String {
    let mut output = String::new();

    output.push('\n');
    output.push_str("FluxBench Comparison Results\n");
    output.push_str(&"=".repeat(60));
    output.push_str("\n\n");

    output.push_str(&format!(
        "Baseline: {} ({})\n",
        baseline.meta.git_commit.as_deref().unwrap_or("unknown"),
        baseline.meta.timestamp.format("%Y-%m-%d %H:%M:%S")
    ));
    output.push_str(&format!(
        "Current:  {} ({})\n\n",
        report.meta.git_commit.as_deref().unwrap_or("unknown"),
        report.meta.timestamp.format("%Y-%m-%d %H:%M:%S")
    ));

    let mut compared = 0usize;
    for result in &report.results {
        output.push_str(&format!(
            "{} {}{}\n",
            status_icon(result.status),
            result.id,
            severity_tag(result.severity)
        ));
        match (&result.metrics, &result.comparison) {
            (Some(metrics), Some(cmp)) => {
                compared += 1;
                push_comparison(&mut output, metrics.mean_ns, cmp);
            }
            _ => output.push_str("    no baseline comparison\n"),
        }
        output.push('\n');
    }

    output.push_str("Summary\n");
    output.push_str(&"-".repeat(60));
    output.push('\n');
    output.push_str(&format!(
        "  Regressions: {}  Improvements: {}  No Change: {}\n",
        report.summary.regressions,
        report.summary.improvements,
        compared.saturating_sub(report.summary.regressions + report.summary.improvements)
    ));

    output
}

/// Append the detail lines of one comparison.
fn push_comparison(output: &mut String, current_mean_ns: f64, cmp: &Comparison) {
    output.push_str(&format!(
        "    baseline: {} → current: {}\n",
        format_duration(cmp.baseline_mean_ns),
        format_duration(current_mean_ns),
    ));
    output.push_str(&format!(
        "    change: {:+.2}% ({}) {} (threshold {}%)\n",
        cmp.relative_change,
        signed_duration(cmp.absolute_change_ns),
        verdict_label(cmp),
        cmp.threshold,
    ));

    let ci = match (cmp.difference_ci_lower_ns, cmp.difference_ci_upper_ns) {
        (Some(lower), Some(upper)) => format!(
            "  {} of change: [{}, {}]",
            ci_label(cmp.confidence_level),
            signed_duration(lower),
            signed_duration(upper)
        ),
        _ => String::new(),
    };
    output.push_str(&format!(
        "    P(regression): {:.1}%{}\n",
        cmp.probability_regression * 100.0,
        ci,
    ));
    output.push_str(&format!(
        "    effect size: {:.2} ({})  method: {}\n",
        cmp.effect_size,
        interpret_effect_size(cmp.effect_size),
        method_label(cmp.method),
    ));
}

fn verdict_label(cmp: &Comparison) -> &'static str {
    if cmp.is_regression() {
        "📈 REGRESSION"
    } else if cmp.is_improvement() {
        "📉 improvement"
    } else {
        "≈ no significant change"
    }
}

fn method_label(method: ComparisonMethod) -> &'static str {
    match method {
        ComparisonMethod::Bootstrap => "bootstrap",
        ComparisonMethod::Summary => "summary z-test",
    }
}

fn status_icon(status: BenchmarkStatus) -> &'static str {
    match status {
        BenchmarkStatus::Passed => "✓",
        BenchmarkStatus::Failed => "✗",
        BenchmarkStatus::Crashed => "💥",
        BenchmarkStatus::Skipped => "⊘",
    }
}

fn severity_tag(severity: Severity) -> &'static str {
    match severity {
        Severity::Critical => " [critical]",
        Severity::Warning => " [warning]",
        Severity::Info => "",
    }
}

/// Duration with an explicit sign, e.g. `+1.20 µs` or `-350.00 ns`.
fn signed_duration(ns: f64) -> String {
    let sign = if ns < 0.0 { '-' } else { '+' };
    format!("{sign}{}", format_duration(ns.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{dummy_report, dummy_result};

    fn comparison(relative_change: f64, threshold: f64, significant: bool) -> Comparison {
        Comparison {
            baseline_mean_ns: 100.0,
            absolute_change_ns: relative_change,
            relative_change,
            probability_regression: 0.999,
            is_significant: significant,
            effect_size: 2.5,
            threshold,
            difference_ci_lower_ns: Some(relative_change - 1.0),
            difference_ci_upper_ns: Some(relative_change + 1.0),
            confidence_level: 0.95,
            method: ComparisonMethod::Bootstrap,
        }
    }

    fn render(cmp: Comparison) -> String {
        let mut result = dummy_result("b", 100.0 + cmp.relative_change, 0.0);
        result.comparison = Some(cmp);
        let report = dummy_report(vec![result]);
        let baseline = dummy_report(vec![]);
        format_comparison_output(&report, &baseline)
    }

    #[test]
    fn shows_actual_threshold_and_statistics() {
        let text = render(comparison(12.0, 10.0, true));
        assert!(text.contains("📈 REGRESSION (threshold 10%)"), "{text}");
        assert!(text.contains("P(regression): 99.9%"), "{text}");
        assert!(text.contains("95% CI of change: [+"), "{text}");
        assert!(text.contains("effect size: 2.50 (large)"), "{text}");
        assert!(text.contains("method: bootstrap"), "{text}");
    }

    #[test]
    fn change_above_five_percent_inside_threshold_is_no_change() {
        let text = render(comparison(8.0, 10.0, false));
        assert!(
            text.contains("≈ no significant change (threshold 10%)"),
            "{text}"
        );
        assert!(!text.contains("REGRESSION"), "{text}");
    }

    #[test]
    fn improvement_is_labelled() {
        let text = render(comparison(-12.0, 10.0, true));
        assert!(text.contains("📉 improvement"), "{text}");
    }

    #[test]
    fn signed_duration_keeps_sign() {
        assert!(signed_duration(-5.0).starts_with('-'));
        assert!(signed_duration(5.0).starts_with('+'));
    }
}
