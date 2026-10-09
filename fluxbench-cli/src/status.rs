//! Status lines that never belong in a report document.
//!
//! The report goes to stdout or `--output`. Banners and annotations go to stderr.
//! Each writer takes `&mut impl Write` so tests can capture the text.

use fluxbench_report::{Report, format_duration};
use std::io::{self, Write};

/// Write the "Running N benchmarks" banner.
pub(crate) fn write_run_banner(
    out: &mut impl Write,
    count: usize,
    mode: &str,
    threads: &str,
    jobs: usize,
) -> io::Result<()> {
    writeln!(
        out,
        "Running {} benchmarks{}, {} threads, {} worker(s)...\n",
        count, mode, threads, jobs
    )
}

/// Write the warnings that parallel workers need.
///
/// `usable_cpus` is the pin pool size, or `None` when workers are not pinned.
pub(crate) fn write_parallel_warnings(
    out: &mut impl Write,
    jobs: usize,
    usable_cpus: Option<usize>,
) -> io::Result<()> {
    if jobs > 1 {
        writeln!(
            out,
            "Warning: --jobs {jobs} runs benchmarks in parallel. Parallel jobs perturb measurements."
        )?;
    }
    if let Some(cpus) = usable_cpus.filter(|&cpus| jobs > cpus) {
        writeln!(
            out,
            "Warning: --jobs {jobs} exceeds the {cpus} usable CPU(s). Workers share cores."
        )?;
    }
    Ok(())
}

/// Write `::error::` and `::warning::` workflow commands for GitHub Actions.
///
/// The caller passes stderr. Whether the runner reads workflow commands
/// from stderr is unconfirmed. Check this in CI before relying on inline annotations.
pub(crate) fn write_github_annotations(out: &mut impl Write, report: &Report) -> io::Result<()> {
    for result in &report.results {
        match result.status {
            fluxbench_report::BenchmarkStatus::Crashed => {
                let msg = result
                    .failure
                    .as_ref()
                    .map(|f| f.message.as_str())
                    .unwrap_or("benchmark crashed");
                writeln!(
                    out,
                    "::error file={},line={}::{}: {}",
                    result.file, result.line, result.id, msg
                )?;
            }
            fluxbench_report::BenchmarkStatus::Failed => {
                let msg = result
                    .failure
                    .as_ref()
                    .map(|f| f.message.as_str())
                    .unwrap_or("benchmark failed");
                writeln!(
                    out,
                    "::error file={},line={}::{}: {}",
                    result.file, result.line, result.id, msg
                )?;
            }
            _ => {}
        }

        if let Some(cmp) = result.comparison.as_ref().filter(|c| c.is_regression()) {
            // Only a critical regression can fail CI, so only it is an error.
            let level = match result.severity {
                fluxbench_core::Severity::Critical => "error",
                _ => "warning",
            };
            writeln!(
                out,
                "::{} file={},line={}::{}: regression {:+.1}% over {}% threshold ({} → {})",
                level,
                result.file,
                result.line,
                result.id,
                cmp.relative_change,
                cmp.threshold,
                format_duration(cmp.baseline_mean_ns),
                result
                    .metrics
                    .as_ref()
                    .map(|m| format_duration(m.mean_ns))
                    .unwrap_or_default(),
            )?;
        }
    }

    for v in &report.verifications {
        match &v.status {
            fluxbench_logic::VerificationStatus::Failed => {
                let level = match v.severity {
                    fluxbench_core::Severity::Critical => "error",
                    _ => "warning",
                };
                writeln!(out, "::{}::{}: {}", level, v.id, v.message)?;
            }
            fluxbench_logic::VerificationStatus::Error { message } => {
                writeln!(out, "::error::{}: evaluation error: {}", v.id, message)?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// Emit the banner on stderr. A write error is dropped.
pub(crate) fn print_run_banner(count: usize, mode: &str, threads: &str, jobs: usize) {
    let _ = write_run_banner(&mut io::stderr().lock(), count, mode, threads, jobs);
}

/// Emit the parallel-worker warnings on stderr. A write error is dropped.
pub(crate) fn print_parallel_warnings(jobs: usize, usable_cpus: Option<usize>) {
    let _ = write_parallel_warnings(&mut io::stderr().lock(), jobs, usable_cpus);
}

/// Emit annotations on stderr. A write error is dropped.
pub(crate) fn print_github_annotations(report: &Report) {
    let _ = write_github_annotations(&mut io::stderr().lock(), report);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn warnings(jobs: usize, cpus: Option<usize>) -> String {
        let mut buf = Vec::new();
        write_parallel_warnings(&mut buf, jobs, cpus).unwrap();
        String::from_utf8(buf).unwrap()
    }

    #[test]
    fn one_job_prints_no_warning() {
        assert_eq!(warnings(1, Some(4)), "");
        assert_eq!(warnings(1, None), "");
    }

    #[test]
    fn parallel_jobs_warn_about_perturbation() {
        let text = warnings(2, Some(4));
        assert!(text.contains("perturb"));
        assert!(!text.contains("share cores"));
        assert!(!warnings(2, None).contains("share cores"));
    }

    #[test]
    fn more_jobs_than_cpus_warns_about_shared_cores() {
        let text = warnings(4, Some(2));
        assert!(text.contains("perturb"));
        assert!(text.contains("share cores"));
    }

    #[test]
    fn regression_annotation_level_follows_severity() {
        use crate::test_support::{dummy_report, dummy_result};
        use fluxbench_core::Severity;
        use fluxbench_report::{Comparison, ComparisonMethod};

        let annotate = |severity: Severity| {
            let mut result = dummy_result("b", 110.0, 0.0);
            result.severity = severity;
            result.comparison = Some(Comparison {
                baseline_mean_ns: 100.0,
                absolute_change_ns: 10.0,
                relative_change: 10.0,
                probability_regression: 0.999,
                is_significant: true,
                effect_size: 3.0,
                threshold: 5.0,
                difference_ci_lower_ns: Some(8.0),
                difference_ci_upper_ns: Some(12.0),
                confidence_level: 0.95,
                method: ComparisonMethod::Summary,
            });
            let mut buf = Vec::new();
            write_github_annotations(&mut buf, &dummy_report(vec![result])).unwrap();
            String::from_utf8(buf).unwrap()
        };

        let critical = annotate(Severity::Critical);
        let expected = "::error file=test.rs,line=1::b: regression +10.0% over 5% threshold";
        assert!(critical.starts_with(expected), "{critical}");
        let warning = annotate(Severity::Warning);
        assert!(warning.starts_with("::warning file="), "{warning}");
    }

    #[test]
    fn banner_has_counts_and_trailing_blank_line() {
        let mut buf = Vec::new();
        write_run_banner(&mut buf, 3, " (in-process)", "all", 1).unwrap();
        let text = String::from_utf8(buf).unwrap();
        assert_eq!(
            text,
            "Running 3 benchmarks (in-process), all threads, 1 worker(s)...\n\n"
        );
    }
}
