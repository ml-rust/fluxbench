//! Shared tail of a run: checks, report, baseline comparison, output, baseline save, and exit status.

use super::checks::apply_checks;
use super::execute::Execution;
use super::exit::{ci_outcome, print_outcome};
use super::output::{render_report, write_report};
use crate::baseline::{ComparisonSettings, apply_baseline_comparison, save_baseline_if_needed};
use crate::{Cli, FluxConfig, build_report, compute_statistics, execute_verifications, status};
use fluxbench_logic::aggregate_verifications;
use fluxbench_report::{HtmlOptions, OutputFormat, Report};
use std::collections::HashMap;
use std::io::Write;

/// Build the report, compare it to `baseline`, write output, and exit non-zero on a CI error.
///
/// It calls `std::process::exit(1)` when [`ci_outcome`] fails the run.
pub(crate) fn finish(
    cli: &Cli,
    config: &FluxConfig,
    format: OutputFormat,
    html: &HtmlOptions,
    execution: Execution,
    baseline: Option<&Report>,
) -> anyhow::Result<()> {
    let Execution {
        mut results,
        exec_config,
        started,
        budgets,
    } = execution;

    let stats = compute_statistics(&results, &exec_config);
    for warning in apply_checks(
        &mut results,
        &stats,
        &budgets,
        &config.allocator,
        exec_config.track_allocations,
    ) {
        eprintln!("{warning}");
    }
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

    write_report(
        cli,
        config,
        &render_report(format, &report, baseline, html)?,
    )?;

    save_baseline_if_needed(cli, config, &report, &results)?;

    let outcome = ci_outcome(&report, &verification_summary, config.ci.fail_on_critical);
    print_outcome(&outcome);
    if outcome.exit_code != 0 {
        let _ = std::io::stdout().flush();
        std::process::exit(outcome.exit_code);
    }
    Ok(())
}
