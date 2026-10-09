//! Executor selection and benchmark execution.
//!
//! The default run and `compare` share it: isolation mode, `--jobs`, and CPU pinning apply to both.

use crate::executor::BenchExecutionResult;
use crate::{
    Cli, ExecutionConfig, Executor, FluxConfig, GroupConfigs, IsolatedExecutor, cpu_pin,
    run_config, status,
};
use fluxbench_core::BenchmarkDef;
use fluxbench_report::format_duration;
use rayon::ThreadPoolBuilder;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Raw results of one run
pub(crate) struct Execution {
    pub results: Vec<BenchExecutionResult>,
    pub exec_config: ExecutionConfig,
    /// Start of the run. The report duration covers execution and statistics.
    pub started: Instant,
    /// `#[bench(budget)]` in nanoseconds, by benchmark ID
    pub budgets: HashMap<String, u64>,
}

/// Run `benchmarks` with the executor that the CLI and flux.toml select.
///
/// `groups` holds the merged `#[flux::group]` run config. It beats the CLI and flux.toml.
/// Errors before any benchmark runs: a bad flux.toml duration, or a benchmark whose
/// resolved run config fails [`fluxbench_ipc::BenchmarkConfig::validate`].
pub(crate) fn execute(
    cli: &Cli,
    config: &FluxConfig,
    benchmarks: &[&'static BenchmarkDef],
    groups: &GroupConfigs,
    jobs: usize,
) -> anyhow::Result<Execution> {
    let isolated = run_config::resolve_isolated(cli, config);
    let exec_config = run_config::build_execution_config(cli, config)?;
    let timeout_ns = if isolated {
        worker_timeout_ns(cli, config, &exec_config, benchmarks, groups)?
    } else {
        0
    };
    check_run_configs(&exec_config, benchmarks, groups, timeout_ns)?;

    if cli.threads > 0 {
        // A second call in one process keeps the first pool. That is harmless here.
        let _ = ThreadPoolBuilder::new()
            .num_threads(cli.threads)
            .build_global();
    }

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

    if exec_config.bootstrap_iterations > 0 && exec_config.bootstrap_iterations < 100 {
        eprintln!(
            "Warning: bootstrap_iterations={} is very low; confidence intervals will be unreliable. \
             Use >= 1000 for meaningful results, or 0 to skip bootstrap.",
            exec_config.bootstrap_iterations
        );
    }

    let started = Instant::now();
    let results = if isolated {
        let timeout = Duration::from_nanos(timeout_ns);
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
            .with_group_configs(groups.clone())
            .execute(benchmarks)
    } else {
        if jobs > 1 {
            eprintln!(
                "Warning: --jobs currently applies only to isolated mode; running in-process serially."
            );
        }
        Executor::new(exec_config.clone())
            .with_group_configs(groups.clone())
            .execute(benchmarks)
    };

    let budgets = benchmarks
        .iter()
        .filter_map(|b| b.budget_ns.map(|ns| (b.id.to_string(), ns)))
        .collect();

    Ok(Execution {
        results,
        exec_config,
        started,
        budgets,
    })
}

/// Worker timeout after the clamp. A raised value prints one warning on stderr.
fn worker_timeout_ns(
    cli: &Cli,
    config: &FluxConfig,
    exec_config: &ExecutionConfig,
    benchmarks: &[&'static BenchmarkDef],
    groups: &GroupConfigs,
) -> anyhow::Result<u64> {
    let configured = run_config::resolve_worker_timeout_ns(cli, config)?;
    let needs = benchmarks.iter().map(|b| {
        let cfg = exec_config.resolve_for_benchmark(b, &groups.get(b.group));
        (b.id, run_config::required_timeout_ns(&cfg))
    });
    let timeout = run_config::clamp_worker_timeout(configured, needs);
    if let Some((configured, ref id)) = timeout.raised {
        eprintln!(
            "Warning: worker timeout {} is below warmup + measurement + 2 s for benchmark `{id}`. Using {}. Raise --worker-timeout or runner.timeout to silence this.",
            format_duration(configured as f64),
            format_duration(timeout.effective_ns as f64)
        );
    }
    Ok(timeout.effective_ns)
}

/// Check each benchmark's resolved run config before any benchmark runs.
///
/// `timeout_ns` 0 means no worker timeout (in-process mode).
fn check_run_configs(
    exec_config: &ExecutionConfig,
    benchmarks: &[&'static BenchmarkDef],
    groups: &GroupConfigs,
    timeout_ns: u64,
) -> anyhow::Result<()> {
    for bench in benchmarks {
        exec_config
            .resolve_for_benchmark(bench, &groups.get(bench.group))
            .to_ipc_config(Duration::from_nanos(timeout_ns))
            .validate()
            .map_err(|e| {
                anyhow::anyhow!(
                    "benchmark `{}` has an invalid run config: {e}. Fix the CLI flag, flux.toml [runner] key, #[bench] attribute, or #[group] attribute that sets it.",
                    bench.id
                )
            })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::bench_def;

    fn leak(def: BenchmarkDef) -> &'static BenchmarkDef {
        Box::leak(Box::new(def))
    }

    fn no_groups() -> GroupConfigs {
        GroupConfigs::default()
    }

    #[test]
    fn valid_configs_pass() {
        let benches = [leak(bench_def("a"))];
        let cfg = ExecutionConfig::default();
        assert!(check_run_configs(&cfg, &benches, &no_groups(), 60_000_000_000).is_ok());
        assert!(check_run_configs(&cfg, &benches, &no_groups(), 0).is_ok());
    }

    #[test]
    fn zero_warmup_outside_fixed_mode_names_the_benchmark() {
        let benches = [leak(bench_def("needs_warmup"))];
        let cfg = ExecutionConfig {
            warmup_time_ns: 0,
            ..Default::default()
        };
        let err = check_run_configs(&cfg, &benches, &no_groups(), 0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("benchmark `needs_warmup`"), "{err}");
        assert!(err.contains("warmup_time_ns"), "{err}");
    }

    #[test]
    fn group_config_is_checked_too() {
        let benches = [leak(BenchmarkDef {
            group: "g",
            ..bench_def("grouped")
        })];
        let mut groups = no_groups();
        groups.insert(
            "g",
            fluxbench_core::GroupConfig {
                sample_count: Some(1),
                ..fluxbench_core::GroupConfig::NONE
            },
        );
        let err = check_run_configs(&ExecutionConfig::default(), &benches, &groups, 0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("benchmark `grouped`"), "{err}");
        assert!(err.contains("sample_count"), "{err}");
    }

    #[test]
    fn timeout_below_measurement_is_an_error_without_the_clamp() {
        let benches = [leak(bench_def("long"))];
        let err = check_run_configs(
            &ExecutionConfig::default(),
            &benches,
            &no_groups(),
            1_000_000,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("timeout_ns"), "{err}");
    }
}
