//! Report rendering and writing, for full runs and empty selections.

use crate::baseline::format_comparison_output;
use crate::{Cli, FluxConfig, build_report, format_human_output, run_config};
use fluxbench_report::{
    HtmlOptions, OutputFormat, Report, generate_csv_report, generate_github_action_benchmark,
    generate_github_summary, generate_html_report_with, generate_json_report,
};
use std::io::Write;

/// Render `report` in `format`. Human output appends the baseline comparison.
pub(crate) fn render_report(
    format: OutputFormat,
    report: &Report,
    baseline: Option<&Report>,
    html: &HtmlOptions,
) -> anyhow::Result<String> {
    Ok(match format {
        OutputFormat::Json => generate_json_report(report)?,
        OutputFormat::GithubSummary => generate_github_summary(report),
        OutputFormat::GithubActionBenchmark => generate_github_action_benchmark(report),
        OutputFormat::Html => generate_html_report_with(report, html),
        OutputFormat::Csv => generate_csv_report(report),
        OutputFormat::Human => {
            let mut text = format_human_output(report);
            if let Some(baseline) = baseline {
                text.push_str(&format_comparison_output(report, baseline));
            }
            text
        }
    })
}

/// Write `text` to `--output`, else to stdout.
///
/// A bare `--output` file name goes under `output.directory` only when that key is set.
/// Otherwise the path stays as written. Missing parent directories are created.
pub(crate) fn write_report(cli: &Cli, config: &FluxConfig, text: &str) -> anyhow::Result<()> {
    let Some(requested) = cli.output.as_deref() else {
        print!("{text}");
        let _ = std::io::stdout().flush();
        return Ok(());
    };
    let path = config.output.resolve_report(requested);
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| {
            anyhow::anyhow!(
                "cannot create output directory {}: {e}. Check permissions or pass another --output path.",
                parent.display()
            )
        })?;
    }
    std::fs::write(&path, text).map_err(|e| {
        anyhow::anyhow!(
            "cannot write output file {}: {e}. Check the --output path.",
            path.display()
        )
    })?;
    eprintln!("Report written to: {}", path.display());
    Ok(())
}

/// Handle a selection with no benchmarks.
///
/// Human format prints only the stderr notice.
/// Other formats also write a valid report with no results, so pipelines get a parseable document.
pub(crate) fn emit_empty_report(
    cli: &Cli,
    config: &FluxConfig,
    format: OutputFormat,
    html: &HtmlOptions,
) -> anyhow::Result<()> {
    eprintln!("No benchmarks found.");
    if format == OutputFormat::Human {
        return Ok(());
    }
    let exec_config = run_config::build_execution_config(cli, config)?;
    let report = build_report(&[], &[], &exec_config, 0.0);
    write_report(cli, config, &render_report(format, &report, None, html)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_report() -> Report {
        build_report(&[], &[], &crate::ExecutionConfig::default(), 0.0)
    }

    fn render(format: OutputFormat) -> String {
        render_report(format, &empty_report(), None, &HtmlOptions::default()).unwrap()
    }

    #[test]
    fn empty_json_report_parses() {
        let value: serde_json::Value = serde_json::from_str(&render(OutputFormat::Json)).unwrap();
        assert_eq!(value["results"].as_array().map(Vec::len), Some(0));
        assert_eq!(value["summary"]["total_benchmarks"], 0);
    }

    #[test]
    fn empty_csv_report_is_header_only() {
        let csv = render(OutputFormat::Csv);
        assert_eq!(csv.lines().count(), 1);
        assert!(csv.starts_with("id,name,group,status"));
    }

    #[test]
    fn empty_github_action_report_is_an_empty_json_array() {
        let value: serde_json::Value =
            serde_json::from_str(&render(OutputFormat::GithubActionBenchmark)).unwrap();
        assert_eq!(value.as_array().map(Vec::len), Some(0));
    }

    #[test]
    fn empty_html_report_is_a_page() {
        let html = render(OutputFormat::Html);
        assert!(html.starts_with("<!DOCTYPE html>"));
        assert!(html.contains("No benchmarks ran."));
    }
}
