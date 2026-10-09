//! `compare <REF>`: run benchmarks against a baseline saved at a git ref.

use super::execute::execute;
use super::finish::finish;
use crate::baseline::{
    check_baseline_commit, configured_or_default, load_baseline, resolve_git_ref,
};
use crate::selection::{report_empty_selection, select_benchmarks};
use crate::{Cli, FluxConfig};
use fluxbench_report::OutputFormat;

/// Compare the current tree against the baseline saved at `git_ref`.
///
/// The baseline path comes from `--baseline`, then flux.toml, then the default.
/// The baseline must exist and its commit must match `git_ref`.
pub(crate) fn compare_benchmarks(
    cli: &Cli,
    config: &FluxConfig,
    git_ref: &str,
    format: OutputFormat,
    jobs: usize,
) -> anyhow::Result<()> {
    let path = cli
        .baseline
        .clone()
        .flatten()
        .unwrap_or_else(|| configured_or_default(config));

    if !path.exists() {
        anyhow::bail!(
            "baseline file not found: {}. Save one on {git_ref} with --save-baseline, or pass --baseline <path>.",
            path.display()
        );
    }

    let baseline = load_baseline(&path)?;
    let resolved = resolve_git_ref(git_ref)?;
    check_baseline_commit(&baseline, git_ref, &resolved)?;

    eprintln!("Comparing against baseline: {}", path.display());
    eprintln!("Git ref: {} ({})\n", git_ref, resolved);

    let (all, benchmarks) = select_benchmarks(cli);
    if benchmarks.is_empty() {
        report_empty_selection(cli, &all);
        return Ok(());
    }

    let execution = execute(cli, config, &benchmarks, jobs);
    finish(cli, config, format, execution, Some(&baseline))
}
