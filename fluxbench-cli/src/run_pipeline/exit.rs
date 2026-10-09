//! CI exit decision.
//!
//! - A crash always fails the run.
//! - A critical verification failure fails it when `ci.fail_on_critical` is on.
//! - A regression or `failed` status on a `severity = "critical"` benchmark fails it when `ci.fail_on_critical` is on.
//! - Other regressions and failed benchmarks print warnings and do not fail.

use fluxbench_core::Severity;
use fluxbench_logic::VerificationSummary;
use fluxbench_report::{BenchmarkStatus, Report, format_duration};

/// Exit code and the messages behind it
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CiOutcome {
    /// Process exit code: 0 or 1
    pub exit_code: i32,
    /// Reasons for a non-zero exit code
    pub errors: Vec<String>,
    /// Findings that do not fail the run
    pub warnings: Vec<String>,
}

/// Decide the exit code for a finished run.
pub(crate) fn ci_outcome(
    report: &Report,
    verification: &VerificationSummary,
    fail_on_critical: bool,
) -> CiOutcome {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    let crashed = report
        .results
        .iter()
        .filter(|r| r.status == BenchmarkStatus::Crashed)
        .count();
    if crashed > 0 {
        errors.push(format!("{crashed} benchmark(s) crashed during execution"));
    }

    for result in report
        .results
        .iter()
        .filter(|r| r.status == BenchmarkStatus::Failed)
    {
        let reason = result
            .failure
            .as_ref()
            .map(|f| format!("{}: {}", f.kind, f.message))
            .unwrap_or_else(|| "no failure details".to_string());
        let detail = format!("benchmark {} failed ({reason})", result.id);
        match (result.severity, fail_on_critical) {
            (Severity::Critical, true) => errors.push(format!("critical {detail}")),
            (Severity::Critical, false) => {
                warnings.push(format!("critical {detail}; ci.fail_on_critical is off"))
            }
            (Severity::Warning, _) | (Severity::Info, _) => {
                warnings.push(format!("{detail}; not critical, so the run does not fail"))
            }
        }
    }

    if verification.should_fail_ci() {
        let count = verification.critical_failures + verification.critical_errors;
        let message = format!("{count} critical verification failure(s)");
        if fail_on_critical {
            errors.push(message);
        } else {
            warnings.push(format!("{message}; ci.fail_on_critical is off"));
        }
    }

    for result in &report.results {
        let (Some(cmp), Some(metrics)) = (&result.comparison, &result.metrics) else {
            continue;
        };
        if !cmp.is_regression() {
            continue;
        }
        let detail = format!(
            "{} regressed {:+.2}% ({} → {}), threshold {}%",
            result.id,
            cmp.relative_change,
            format_duration(cmp.baseline_mean_ns),
            format_duration(metrics.mean_ns),
            cmp.threshold
        );
        match (result.severity, fail_on_critical) {
            (Severity::Critical, true) => errors.push(format!("critical benchmark {detail}")),
            (Severity::Critical, false) => warnings.push(format!(
                "critical benchmark {detail}; ci.fail_on_critical is off"
            )),
            (Severity::Warning, _) | (Severity::Info, _) => warnings.push(format!(
                "benchmark {detail}; not critical, so the run does not fail"
            )),
        }
    }

    CiOutcome {
        exit_code: if errors.is_empty() { 0 } else { 1 },
        errors,
        warnings,
    }
}

/// Print warnings and errors on stderr.
pub(crate) fn print_outcome(outcome: &CiOutcome) {
    for warning in &outcome.warnings {
        eprintln!("Warning: {warning}");
    }
    for error in &outcome.errors {
        eprintln!("\nError: {error}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{dummy_report, dummy_result};
    use fluxbench_report::{Comparison, ComparisonMethod};

    fn regression(relative_change: f64, significant: bool) -> Comparison {
        Comparison {
            baseline_mean_ns: 100.0,
            absolute_change_ns: relative_change,
            relative_change,
            probability_regression: 0.999,
            is_significant: significant,
            effect_size: 3.0,
            threshold: 5.0,
            difference_ci_lower_ns: Some(1.0),
            difference_ci_upper_ns: Some(20.0),
            confidence_level: 0.95,
            method: ComparisonMethod::Summary,
        }
    }

    fn report_with(severity: Severity, cmp: Option<Comparison>) -> Report {
        let mut result = dummy_result("b", 110.0, 0.0);
        result.severity = severity;
        result.comparison = cmp;
        dummy_report(vec![result])
    }

    fn clean() -> VerificationSummary {
        VerificationSummary::default()
    }

    #[test]
    fn critical_regression_fails_when_enabled() {
        let report = report_with(Severity::Critical, Some(regression(10.0, true)));
        let outcome = ci_outcome(&report, &clean(), true);
        assert_eq!(outcome.exit_code, 1);
        assert!(outcome.errors[0].contains("critical benchmark b regressed"));
        assert!(outcome.errors[0].contains("threshold 5%"));
    }

    #[test]
    fn critical_regression_warns_when_disabled() {
        let report = report_with(Severity::Critical, Some(regression(10.0, true)));
        let outcome = ci_outcome(&report, &clean(), false);
        assert_eq!(outcome.exit_code, 0);
        assert_eq!(outcome.warnings.len(), 1);
    }

    #[test]
    fn non_critical_regression_only_warns() {
        for severity in [Severity::Warning, Severity::Info] {
            let report = report_with(severity, Some(regression(10.0, true)));
            let outcome = ci_outcome(&report, &clean(), true);
            assert_eq!(outcome.exit_code, 0, "{severity:?}");
            assert_eq!(outcome.warnings.len(), 1, "{severity:?}");
            assert!(outcome.errors.is_empty());
        }
    }

    #[test]
    fn insignificant_change_and_improvement_pass() {
        let noise = report_with(Severity::Critical, Some(regression(10.0, false)));
        assert_eq!(ci_outcome(&noise, &clean(), true).exit_code, 0);

        let faster = report_with(Severity::Critical, Some(regression(-10.0, true)));
        let outcome = ci_outcome(&faster, &clean(), true);
        assert_eq!(outcome.exit_code, 0);
        assert!(outcome.warnings.is_empty());

        let none = report_with(Severity::Critical, None);
        assert_eq!(ci_outcome(&none, &clean(), true).exit_code, 0);
    }

    #[test]
    fn crash_always_fails() {
        let mut report = report_with(Severity::Info, None);
        report.results[0].status = BenchmarkStatus::Crashed;
        let outcome = ci_outcome(&report, &clean(), false);
        assert_eq!(outcome.exit_code, 1);
        assert!(outcome.errors[0].contains("crashed"));
    }

    #[test]
    fn failed_benchmark_follows_severity() {
        use fluxbench_report::FailureInfo;
        let failed = |severity: Severity| {
            let mut report = report_with(severity, None);
            report.results[0].status = BenchmarkStatus::Failed;
            report.results[0].failure = Some(FailureInfo {
                kind: "budget".to_string(),
                message: "mean 2 us exceeds the budget of 1 us.".to_string(),
                backtrace: None,
                signal: None,
            });
            report
        };
        let outcome = ci_outcome(&failed(Severity::Critical), &clean(), true);
        assert_eq!(outcome.exit_code, 1);
        assert!(outcome.errors[0].contains("critical benchmark b failed (budget:"));

        let outcome = ci_outcome(&failed(Severity::Critical), &clean(), false);
        assert_eq!(outcome.exit_code, 0);
        assert_eq!(outcome.warnings.len(), 1);

        let outcome = ci_outcome(&failed(Severity::Warning), &clean(), true);
        assert_eq!(outcome.exit_code, 0);
        assert!(outcome.warnings[0].contains("not critical"));
    }

    #[test]
    fn critical_verification_follows_fail_on_critical() {
        let report = report_with(Severity::Info, None);
        let failing = VerificationSummary {
            critical_failures: 1,
            ..Default::default()
        };
        assert_eq!(ci_outcome(&report, &failing, true).exit_code, 1);
        let outcome = ci_outcome(&report, &failing, false);
        assert_eq!(outcome.exit_code, 0);
        assert_eq!(outcome.warnings.len(), 1);
    }
}
