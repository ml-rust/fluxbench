//! Run config resolution
//!
//! Layers run settings in this order: macro attribute > CLI flag > flux.toml > default.
//! This module covers the CLI and flux.toml layers.
//! `ExecutionConfig::resolve_for_benchmark` applies the macro layer per benchmark.

use crate::{Cli, ExecutionConfig, FluxConfig};

/// Slack the worker timeout keeps above warmup + measurement
pub(crate) const TIMEOUT_SLACK_NS: u64 = 2_000_000_000;

/// Build the global `ExecutionConfig` from CLI flags and flux.toml.
///
/// An explicit CLI flag beats flux.toml. A malformed flux.toml duration is an error.
/// `samples = N` (CLI, then flux.toml) selects fixed mode: N samples of one iteration, no warmup.
/// `sample_count` overrides only the count at the same layer.
pub(crate) fn build_execution_config(
    cli: &Cli,
    config: &FluxConfig,
) -> anyhow::Result<ExecutionConfig> {
    let defaults = ExecutionConfig::default();
    let runner = &config.runner;

    let warmup_time_ns = match cli.warmup {
        Some(ns) => ns,
        None => runner.warmup_time_ns()?,
    };
    let measurement_time_ns = match cli.measurement {
        Some(ns) => ns,
        None => runner.measurement_time_ns()?,
    };

    let sample_count = cli
        .sample_count
        .or(cli.samples)
        .map(|n| n as usize)
        .or(runner.sample_count)
        .or(runner.samples.map(|n| n as usize))
        .unwrap_or(defaults.sample_count);

    let base = ExecutionConfig {
        sample_count,
        track_allocations: config.allocator.track,
        bootstrap_iterations: runner.bootstrap_iterations,
        confidence_level: runner.confidence_level,
        ..defaults
    };

    if cli.samples.or(runner.samples).is_some() {
        return Ok(ExecutionConfig {
            warmup_time_ns: 0,
            measurement_time_ns,
            min_iterations: None,
            max_iterations: None,
            iters_per_sample: Some(1),
            ..base
        });
    }

    Ok(ExecutionConfig {
        warmup_time_ns,
        measurement_time_ns,
        min_iterations: cli.min_iterations.or(runner.min_iterations),
        max_iterations: cli.max_iterations.or(runner.max_iterations),
        iters_per_sample: None,
        ..base
    })
}

/// Resolve the isolated worker count: CLI flag, then flux.toml, then 1.
pub(crate) fn resolve_jobs(cli: &Cli, config: &FluxConfig) -> anyhow::Result<usize> {
    match cli.jobs.or(config.runner.jobs).unwrap_or(1) {
        0 => anyhow::bail!("--jobs 0 is invalid. Use 1 or more."),
        jobs => Ok(jobs),
    }
}

/// Resolve isolation: `--isolated`, then flux.toml `runner.isolation`, then process isolation.
pub(crate) fn resolve_isolated(cli: &Cli, config: &FluxConfig) -> bool {
    cli.isolated
        .unwrap_or_else(|| config.runner.isolation.is_isolated())
}

/// Resolve the configured worker timeout: `--worker-timeout`, then `runner.timeout`, then 60 s.
pub(crate) fn resolve_worker_timeout_ns(cli: &Cli, config: &FluxConfig) -> anyhow::Result<u64> {
    match cli.worker_timeout {
        Some(ns) => Ok(ns),
        None => config.runner.timeout_ns(),
    }
}

/// Timeout one benchmark needs: warmup + measurement + [`TIMEOUT_SLACK_NS`]
pub(crate) fn required_timeout_ns(cfg: &ExecutionConfig) -> u64 {
    cfg.warmup_time_ns
        .saturating_add(cfg.measurement_time_ns)
        .saturating_add(TIMEOUT_SLACK_NS)
}

/// Worker timeout after the clamp
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WorkerTimeout {
    /// Timeout every worker uses, in nanoseconds
    pub effective_ns: u64,
    /// Set when the clamp raised the configured value: (configured ns, benchmark ID that needs more)
    pub raised: Option<(u64, String)>,
}

/// Raise `configured_ns` to the largest timeout any benchmark needs.
///
/// `needs` holds (benchmark ID, required ns) pairs.
pub(crate) fn clamp_worker_timeout<'a>(
    configured_ns: u64,
    needs: impl IntoIterator<Item = (&'a str, u64)>,
) -> WorkerTimeout {
    let longest = needs.into_iter().max_by_key(|(_, ns)| *ns);
    match longest {
        Some((id, required)) if required > configured_ns => WorkerTimeout {
            effective_ns: required,
            raised: Some((configured_ns, id.to_string())),
        },
        _ => WorkerTimeout {
            effective_ns: configured_ns,
            raised: None,
        },
    }
}

/// Resolve worker pinning: `--no-pin`, then `--pin`, then flux.toml `pin_cpus`, then on.
pub(crate) fn resolve_pin(cli: &Cli, config: &FluxConfig) -> bool {
    if cli.no_pin {
        return false;
    }
    cli.pin || config.runner.pin_cpus.unwrap_or(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn cli(args: &[&str]) -> Cli {
        let mut argv = vec!["bench"];
        argv.extend_from_slice(args);
        Cli::try_parse_from(argv).unwrap()
    }

    fn toml_config(text: &str) -> FluxConfig {
        toml::from_str(text).unwrap()
    }

    fn build_execution_config(cli: &Cli, config: &FluxConfig) -> ExecutionConfig {
        super::build_execution_config(cli, config).unwrap()
    }

    #[test]
    fn worker_timeout_precedence_cli_then_toml_then_default() {
        let config = toml_config("[runner]\ntimeout = \"90s\"\n");
        assert_eq!(
            resolve_worker_timeout_ns(&cli(&["--worker-timeout", "2m"]), &config).unwrap(),
            120_000_000_000
        );
        assert_eq!(
            resolve_worker_timeout_ns(&cli(&["--worker-timeout", "30"]), &config).unwrap(),
            30_000_000_000
        );
        assert_eq!(
            resolve_worker_timeout_ns(&cli(&[]), &config).unwrap(),
            90_000_000_000
        );
        assert_eq!(
            resolve_worker_timeout_ns(&cli(&[]), &FluxConfig::default()).unwrap(),
            60_000_000_000
        );
    }

    #[test]
    fn bad_worker_timeout_flag_is_rejected() {
        assert!(Cli::try_parse_from(["bench", "--worker-timeout", "later"]).is_err());
    }

    #[test]
    fn timeout_clamp_raises_to_longest_need() {
        let cfg = ExecutionConfig {
            warmup_time_ns: 3_000_000_000,
            measurement_time_ns: 5_000_000_000,
            ..Default::default()
        };
        assert_eq!(required_timeout_ns(&cfg), 10_000_000_000);

        let raised =
            clamp_worker_timeout(4_000_000_000, [("a", 6_000_000_000), ("b", 10_000_000_000)]);
        assert_eq!(raised.effective_ns, 10_000_000_000);
        assert_eq!(raised.raised, Some((4_000_000_000, "b".to_string())));

        let kept = clamp_worker_timeout(60_000_000_000, [("a", 10_000_000_000)]);
        assert_eq!(kept.effective_ns, 60_000_000_000);
        assert_eq!(kept.raised, None);

        let empty = clamp_worker_timeout(1, std::iter::empty());
        assert_eq!(empty.effective_ns, 1);
    }

    #[test]
    fn isolation_precedence() {
        let in_process = toml_config("[runner]\nisolation = \"in-process\"\n");
        assert!(!resolve_isolated(&cli(&[]), &in_process));
        assert!(resolve_isolated(&cli(&["--isolated=true"]), &in_process));
        assert!(!resolve_isolated(
            &cli(&["--isolated=false"]),
            &FluxConfig::default()
        ));
        assert!(resolve_isolated(&cli(&[]), &FluxConfig::default()));
    }

    #[test]
    fn zero_jobs_is_rejected() {
        assert!(resolve_jobs(&cli(&["--jobs", "0"]), &FluxConfig::default()).is_err());
    }

    #[test]
    fn defaults_without_cli_or_toml() {
        let cfg = build_execution_config(&cli(&[]), &FluxConfig::default());
        assert_eq!(cfg.sample_count, 100);
        assert_eq!(cfg.iters_per_sample, None);
        assert_eq!(cfg.warmup_time_ns, 3_000_000_000);
        assert_eq!(cfg.measurement_time_ns, 5_000_000_000);
    }

    #[test]
    fn explicit_warmup_overrides_toml() {
        let config = toml_config("[runner]\nwarmup_time = \"1s\"\nmeasurement_time = \"2s\"\n");

        // `--warmup 3` equals the old clap default and still wins
        let cfg = build_execution_config(&cli(&["--warmup", "3"]), &config);
        assert_eq!(cfg.warmup_time_ns, 3_000_000_000);
        assert_eq!(cfg.measurement_time_ns, 2_000_000_000);

        let cfg = build_execution_config(&cli(&[]), &config);
        assert_eq!(cfg.warmup_time_ns, 1_000_000_000);
    }

    #[test]
    fn warmup_and_measurement_accept_duration_strings() {
        let args = ["--warmup", "500ms", "--measurement", "5s"];
        let cfg = build_execution_config(&cli(&args), &FluxConfig::default());
        assert_eq!(cfg.warmup_time_ns, 500_000_000);
        assert_eq!(cfg.measurement_time_ns, 5_000_000_000);
    }

    #[test]
    fn bad_warmup_string_is_rejected() {
        assert!(Cli::try_parse_from(["bench", "--warmup", "soon"]).is_err());
    }

    #[test]
    fn sample_count_flag_sets_count_only() {
        let cfg = build_execution_config(&cli(&["--sample-count", "250"]), &FluxConfig::default());
        assert_eq!(cfg.sample_count, 250);
        assert_eq!(cfg.iters_per_sample, None);
        assert_eq!(cfg.warmup_time_ns, 3_000_000_000);
    }

    #[test]
    fn sample_count_below_two_is_rejected() {
        assert!(Cli::try_parse_from(["bench", "--sample-count", "1"]).is_err());
        assert!(Cli::try_parse_from(["bench", "--samples", "1"]).is_err());
    }

    #[test]
    fn samples_flag_selects_uncapped_fixed_mode() {
        let cfg = build_execution_config(&cli(&["--samples", "500"]), &FluxConfig::default());
        assert_eq!(cfg.sample_count, 500);
        assert_eq!(cfg.iters_per_sample, Some(1));
        assert_eq!(cfg.warmup_time_ns, 0);
        assert_eq!(cfg.min_iterations, None);
        assert_eq!(cfg.max_iterations, None);
    }

    #[test]
    fn sample_count_flag_overrides_samples_count() {
        let args = ["-n", "500", "--sample-count", "50"];
        let cfg = build_execution_config(&cli(&args), &FluxConfig::default());
        assert_eq!(cfg.sample_count, 50);
        assert_eq!(cfg.iters_per_sample, Some(1));
        assert_eq!(cfg.warmup_time_ns, 0);
    }

    #[test]
    fn toml_sample_count_applies_without_cli() {
        let config = toml_config("[runner]\nsample_count = 300\n");
        let cfg = build_execution_config(&cli(&[]), &config);
        assert_eq!(cfg.sample_count, 300);
        assert_eq!(cfg.iters_per_sample, None);
    }

    #[test]
    fn cli_sample_count_beats_toml() {
        let config = toml_config("[runner]\nsample_count = 300\nsamples = 40\n");

        let cfg = build_execution_config(&cli(&["--sample-count", "120"]), &config);
        assert_eq!(cfg.sample_count, 120);
        assert_eq!(cfg.iters_per_sample, Some(1));

        let cfg = build_execution_config(&cli(&["--samples", "60"]), &config);
        assert_eq!(cfg.sample_count, 60);

        let cfg = build_execution_config(&cli(&[]), &config);
        assert_eq!(cfg.sample_count, 300);
        assert_eq!(cfg.iters_per_sample, Some(1));
    }

    #[test]
    fn explicit_jobs_overrides_toml() {
        let config = toml_config("[runner]\njobs = 4\n");
        assert_eq!(resolve_jobs(&cli(&["--jobs", "1"]), &config).unwrap(), 1);
        assert_eq!(resolve_jobs(&cli(&[]), &config).unwrap(), 4);
        assert_eq!(resolve_jobs(&cli(&[]), &FluxConfig::default()).unwrap(), 1);
    }

    #[test]
    fn pin_defaults_on() {
        assert!(resolve_pin(&cli(&[]), &FluxConfig::default()));
    }

    #[test]
    fn toml_can_disable_pin() {
        let config = toml_config("[runner]\npin_cpus = false\n");
        assert!(!resolve_pin(&cli(&[]), &config));
    }

    #[test]
    fn cli_pin_flags_beat_toml() {
        let off = toml_config("[runner]\npin_cpus = false\n");
        let on = toml_config("[runner]\npin_cpus = true\n");
        assert!(resolve_pin(&cli(&["--pin"]), &off));
        assert!(!resolve_pin(&cli(&["--no-pin"]), &on));
        assert!(!resolve_pin(&cli(&["--no-pin"]), &FluxConfig::default()));
    }

    #[test]
    fn pin_flags_conflict() {
        assert!(Cli::try_parse_from(["bench", "--pin", "--no-pin"]).is_err());
    }
}
