//! Run config resolution
//!
//! Layers run settings in this order: macro attribute > CLI flag > flux.toml > default.
//! This module covers the CLI and flux.toml layers.
//! `ExecutionConfig::resolve_for_benchmark` applies the macro layer per benchmark.

use crate::{Cli, ExecutionConfig, FluxConfig};

/// Build the global `ExecutionConfig` from CLI flags and flux.toml.
///
/// An explicit CLI flag beats flux.toml. A malformed flux.toml duration falls back to the default.
/// `samples = N` (CLI, then flux.toml) selects fixed mode: N samples of one iteration, no warmup.
/// `sample_count` overrides only the count at the same layer.
pub(crate) fn build_execution_config(cli: &Cli, config: &FluxConfig) -> ExecutionConfig {
    let defaults = ExecutionConfig::default();
    let runner = &config.runner;

    let warmup_time_ns = cli
        .warmup
        .or_else(|| FluxConfig::parse_duration(&runner.warmup_time).ok())
        .unwrap_or(defaults.warmup_time_ns);
    let measurement_time_ns = cli
        .measurement
        .or_else(|| FluxConfig::parse_duration(&runner.measurement_time).ok())
        .unwrap_or(defaults.measurement_time_ns);

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
        return ExecutionConfig {
            warmup_time_ns: 0,
            measurement_time_ns,
            min_iterations: None,
            max_iterations: None,
            iters_per_sample: Some(1),
            ..base
        };
    }

    ExecutionConfig {
        warmup_time_ns,
        measurement_time_ns,
        min_iterations: cli.min_iterations.or(runner.min_iterations),
        max_iterations: cli.max_iterations.or(runner.max_iterations),
        iters_per_sample: None,
        ..base
    }
}

/// Resolve the isolated worker count: CLI flag, then flux.toml, then 1.
pub(crate) fn resolve_jobs(cli: &Cli, config: &FluxConfig) -> usize {
    cli.jobs.or(config.runner.jobs).unwrap_or(1)
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
        assert_eq!(resolve_jobs(&cli(&["--jobs", "1"]), &config), 1);
        assert_eq!(resolve_jobs(&cli(&[]), &config), 4);
        assert_eq!(resolve_jobs(&cli(&[]), &FluxConfig::default()), 1);
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
