//! `compare <REF>`: run benchmarks against a baseline saved at a git ref.

use super::execute::execute;
use super::finish::finish;
use super::output::emit_empty_report;
use crate::baseline::{check_baseline_commit, load_baseline, resolve_git_ref};
use crate::selection::Selection;
use crate::{Cli, FluxConfig};
use fluxbench_report::{HtmlOptions, OutputFormat};

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
    selection: &Selection,
    html: &HtmlOptions,
) -> anyhow::Result<()> {
    let path = match cli.baseline.clone().flatten() {
        Some(path) => config.output.resolve_baseline(&path),
        None => config.output.baseline_file(),
    };

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

    if selection.selected.is_empty() {
        return emit_empty_report(cli, config, format, html);
    }

    let groups = selection.groups.configs();
    let execution = execute(cli, config, &selection.selected, &groups, jobs)?;
    finish(cli, config, format, html, execution, Some(&baseline))
}
