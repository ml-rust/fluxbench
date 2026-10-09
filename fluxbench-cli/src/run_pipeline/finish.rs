//! Shared tail of a run: report, baseline comparison, output, baseline save, and exit status.

use super::execute::Execution;
use super::exit::{ci_outcome, print_outcome};
use crate::baseline::{
    ComparisonSettings, apply_baseline_comparison, format_comparison_output,
    save_baseline_if_needed,
};
use crate::{
    Cli, FluxConfig, build_report, compute_statistics, execute_verifications, format_human_output,
    status,
};
use fluxbench_logic::aggregate_verifications;
use fluxbench_report::{
    OutputFormat, Report, generate_csv_report, generate_github_action_benchmark,
    generate_github_summary, generate_html_report, generate_json_report,
};
use std::collections::HashMap;
use std::io::Write;

/// Build the report, compare it to `baseline`, write output, and exit non-zero on a CI error.
///
/// It calls `std::process::exit(1)` when [`ci_outcome`] fails the run.
pub(crate) fn finish(
    cli: &Cli,
    config: &FluxConfig,
    format: OutputFormat,
    execution: Execution,
    baseline: Option<&Report>,
) -> anyhow::Result<()> {
    let Execution {
        results,
        exec_config,
        started,
    } = execution;

    let stats = compute_statistics(&results, &exec_config);
    let total_duration_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut report = build_report(&results, &stats, &exec_config, total_duration_ms);

    if let Some(baseline) = baseline {
        let current_samples: HashMap<&str, &[f64]> = results
            .iter()
            .map(|r| (r.benchmark_id.as_str(), r.samples.as_slice()))
            .collect();
        let settings = ComparisonSettings {
            threshold: cli.threshold.unwrap_or(config.ci.regression_threshold),
            confidence_level: exec_config.confidence_level,
            bootstrap_iterations: exec_config.bootstrap_iterations,
        };
        for warning in apply_baseline_comparison(&mut report, baseline, &current_samples, &settings)
        {
            eprintln!("{warning}");
        }
    }

    let (comparisons, comparison_series, synthetics, verifications) =
        execute_verifications(&results, &stats);
    let verification_summary = aggregate_verifications(&verifications);
    report.comparisons = comparisons;
    report.comparison_series = comparison_series;
    report.synthetics = synthetics;
    report.verifications = verifications;
    report.summary.critical_failures = verification_summary.critical_failures;
    report.summary.warnings = verification_summary.failed - verification_summary.critical_failures;

    if config.ci.github_annotations {
        status::print_github_annotations(&report);
    }

    let output = match format {
        OutputFormat::Json => generate_json_report(&report)?,
        OutputFormat::GithubSummary => generate_github_summary(&report),
        OutputFormat::GithubActionBenchmark => generate_github_action_benchmark(&report),
        OutputFormat::Html => generate_html_report(&report),
        OutputFormat::Csv => generate_csv_report(&report),
        OutputFormat::Human => {
            let mut text = format_human_output(&report);
            if let Some(baseline) = baseline {
                text.push_str(&format_comparison_output(&report, baseline));
            }
            text
        }
    };

    if let Some(ref path) = cli.output {
        let mut file = std::fs::File::create(path).map_err(|e| {
            anyhow::anyhow!(
                "cannot create output file {}: {e}. Check the --output path.",
                path.display()
            )
        })?;
        file.write_all(output.as_bytes())?;
        eprintln!("Report written to: {}", path.display());
    } else {
        print!("{}", output);
    }

    save_baseline_if_needed(cli, config, &report, &results)?;

    let outcome = ci_outcome(&report, &verification_summary, config.ci.fail_on_critical);
    print_outcome(&outcome);
    if outcome.exit_code != 0 {
        let _ = std::io::stdout().flush();
        std::process::exit(outcome.exit_code);
    }
    Ok(())
}
