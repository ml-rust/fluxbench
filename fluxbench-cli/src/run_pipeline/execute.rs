//! Executor selection and benchmark execution.
//!
//! The default run and `compare` share it: isolation mode, `--jobs`, and CPU pinning apply to both.

use crate::executor::BenchExecutionResult;
use crate::{
    Cli, ExecutionConfig, Executor, FluxConfig, IsolatedExecutor, cpu_pin, run_config, status,
};
use fluxbench_core::BenchmarkDef;
use rayon::ThreadPoolBuilder;
use std::time::{Duration, Instant};

/// Raw results of one run
pub(crate) struct Execution {
    pub results: Vec<BenchExecutionResult>,
    pub exec_config: ExecutionConfig,
    /// Start of the run. The report duration covers execution and statistics.
    pub started: Instant,
}

/// Run `benchmarks` with the executor that the CLI and flux.toml select.
pub(crate) fn execute(
    cli: &Cli,
    config: &FluxConfig,
    benchmarks: &[&'static BenchmarkDef],
    jobs: usize,
) -> Execution {
    let jobs = jobs.max(1);

    if cli.threads > 0 {
        // A second call in one process keeps the first pool. That is harmless here.
        let _ = ThreadPoolBuilder::new()
            .num_threads(cli.threads)
            .build_global();
    }

    // flux.toml can force in-process mode. Otherwise the CLI flag decides.
    let isolated = config.runner.isolation.is_isolated() && cli.isolated;

    let threads = if cli.threads == 0 {
        "all".to_string()
    } else {
        cli.threads.to_string()
    };
    let mode = match (isolated, cli.one_shot) {
        (true, true) => " (isolated, one-shot)",
        (true, false) => " (isolated, persistent)",
        (false, _) => " (in-process)",
    };
    status::print_run_banner(benchmarks.len(), mode, &threads, jobs);

    let started = Instant::now();
    let exec_config = run_config::build_execution_config(cli, config);

    if exec_config.bootstrap_iterations > 0 && exec_config.bootstrap_iterations < 100 {
        eprintln!(
            "Warning: bootstrap_iterations={} is very low; confidence intervals will be unreliable. \
             Use >= 1000 for meaningful results, or 0 to skip bootstrap.",
            exec_config.bootstrap_iterations
        );
    }

    let results = if isolated {
        let timeout = Duration::from_secs(cli.worker_timeout);
        let allowed = cpu_pin::allowed_cpus();
        let pinned = run_config::resolve_pin(cli, config) && !allowed.is_empty();
        status::print_parallel_warnings(jobs, pinned.then(|| cpu_pin::usable_cpus(&allowed).len()));
        let cpus = if pinned {
            cpu_pin::assign_cpus(&allowed, jobs)
        } else {
            Vec::new()
        };
        IsolatedExecutor::new(exec_config.clone(), timeout, !cli.one_shot, jobs)
            .with_cpus(cpus)
            .execute(benchmarks)
    } else {
        if jobs > 1 {
            eprintln!(
                "Warning: --jobs currently applies only to isolated mode; running in-process serially."
            );
        }
        Executor::new(exec_config.clone()).execute(benchmarks)
    };

    warn_if_allocations_missing(&exec_config, &results);

    Execution {
        results,
        exec_config,
        started,
    }
}

/// Warn when allocation tracking is on but no benchmark recorded an allocation.
fn warn_if_allocations_missing(exec_config: &ExecutionConfig, results: &[BenchExecutionResult]) {
    if exec_config.track_allocations
        && !results.is_empty()
        && results
            .iter()
            .all(|r| r.alloc_bytes == 0 && r.alloc_count == 0)
    {
        eprintln!(
            "Warning: allocation tracking enabled but all benchmarks reported 0 bytes allocated.\n\
             Ensure TrackingAllocator is set as #[global_allocator] in your benchmark binary."
        );
    }
}
