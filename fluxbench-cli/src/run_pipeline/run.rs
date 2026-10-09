//! Default run: execute the selected benchmarks, with an optional baseline comparison.

use super::execute::execute;
use super::finish::finish;
use super::output::emit_empty_report;
use crate::baseline::{load_baseline, resolve_baseline_path};
use crate::selection::Selection;
use crate::{Cli, FluxConfig};
use fluxbench_report::{HtmlOptions, OutputFormat, Report};

/// Run benchmarks. `--baseline` adds a comparison.
///
/// A missing baseline file prints a warning, so a first CI run passes.
/// An unreadable or invalid baseline file is an error.
/// An empty selection writes an empty report in non-human formats and exits 0.
pub(crate) fn run_benchmarks(
    cli: &Cli,
    config: &FluxConfig,
    format: OutputFormat,
    jobs: usize,
    selection: &Selection,
    html: &HtmlOptions,
) -> anyhow::Result<()> {
    if selection.selected.is_empty() {
        return emit_empty_report(cli, config, format, html);
    }

    let baseline = optional_baseline(cli, config)?;
    let execution = execute(cli, config, &selection.selected, jobs)?;
    finish(cli, config, format, html, execution, baseline.as_ref())
}

/// Load the `--baseline` file when the flag is set and the file exists.
fn optional_baseline(cli: &Cli, config: &FluxConfig) -> anyhow::Result<Option<Report>> {
    let Some(path) = resolve_baseline_path(&cli.baseline, config) else {
        return Ok(None);
    };
    if !path.exists() {
        eprintln!(
            "Warning: baseline file not found: {}. No comparison runs. Save one with --save-baseline.",
            path.display()
        );
        return Ok(None);
    }
    eprintln!("Comparing against baseline: {}", path.display());
    load_baseline(&path).map(Some)
}
