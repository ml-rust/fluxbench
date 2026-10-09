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

        if let Some(cmp) = &result.comparison {
            if cmp.is_significant && cmp.relative_change > 0.0 {
                writeln!(
                    out,
                    "::error file={},line={}::{}: regression {:+.1}% ({} → {})",
                    result.file,
                    result.line,
                    result.id,
                    cmp.relative_change,
                    format_duration(cmp.baseline_mean_ns),
                    result
                        .metrics
                        .as_ref()
                        .map(|m| format_duration(m.mean_ns))
                        .unwrap_or_default(),
                )?;
            }
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

/// Emit annotations on stderr. A write error is dropped.
pub(crate) fn print_github_annotations(report: &Report) {
    let _ = write_github_annotations(&mut io::stderr().lock(), report);
}

#[cfg(test)]
mod tests {
    use super::*;

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
