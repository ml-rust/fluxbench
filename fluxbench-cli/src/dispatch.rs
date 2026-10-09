//! Entry points: parse arguments, set up logging, and dispatch the subcommand.

use crate::run_pipeline::{compare_benchmarks, run_benchmarks};
use crate::selection::list_benchmarks;
use crate::{Cli, Commands, FluxConfig, run_config};
use clap::Parser;
use fluxbench_core::WorkerMain;
use fluxbench_report::OutputFormat;

/// Run the FluxBench CLI with the given arguments.
/// This is the main entry point for benchmark binaries.
///
/// # Returns
/// Returns `Ok(())` on success, or an error if something goes wrong.
pub fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    run_with_cli(cli)
}

/// Run the FluxBench CLI with pre-parsed arguments.
pub fn run_with_cli(cli: Cli) -> anyhow::Result<()> {
    // Worker mode comes before any other initialization.
    if cli.flux_worker {
        return run_worker_mode();
    }

    let filter = if cli.verbose {
        "fluxbench=debug"
    } else {
        "fluxbench=info"
    };
    tracing_subscriber::fmt().with_env_filter(filter).init();

    // flux.toml configuration. CLI flags override it.
    let config = FluxConfig::discover().unwrap_or_default();
    let format: OutputFormat = cli.format.parse().unwrap_or(OutputFormat::Human);
    let jobs = run_config::resolve_jobs(&cli, &config);

    match cli.command {
        Some(Commands::List) => list_benchmarks(&cli),
        Some(Commands::Run) => run_benchmarks(&cli, &config, format, jobs)?,
        Some(Commands::Compare { ref git_ref }) => {
            compare_benchmarks(&cli, &config, git_ref, format, jobs)?
        }
        None if cli.dry_run => list_benchmarks(&cli),
        None => run_benchmarks(&cli, &config, format, jobs)?,
    }

    Ok(())
}

/// Run as a worker process (IPC mode)
fn run_worker_mode() -> anyhow::Result<()> {
    let mut worker = WorkerMain::new();
    worker
        .run()
        .map_err(|e| anyhow::anyhow!("worker error: {e}"))
}
