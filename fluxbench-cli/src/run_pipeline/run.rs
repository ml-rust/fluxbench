//! Default run: execute the selected benchmarks, with an optional baseline comparison.

use super::execute::execute;
use super::finish::finish;
use crate::baseline::{load_baseline, resolve_baseline_path};
use crate::selection::{report_empty_selection, select_benchmarks};
use crate::{Cli, FluxConfig};
use fluxbench_report::{OutputFormat, Report};

/// Run benchmarks. `--baseline` adds a comparison.
///
/// A missing baseline file prints a warning, so a first CI run passes.
/// An unreadable or invalid baseline file is an error.
pub(crate) fn run_benchmarks(
    cli: &Cli,
    config: &FluxConfig,
    format: OutputFormat,
    jobs: usize,
) -> anyhow::Result<()> {
    let (all, benchmarks) = select_benchmarks(cli);
    if benchmarks.is_empty() {
        report_empty_selection(cli, &all);
        return Ok(());
    }

    let baseline = optional_baseline(cli, config)?;
    let execution = execute(cli, config, &benchmarks, jobs);
    finish(cli, config, format, execution, baseline.as_ref())
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
