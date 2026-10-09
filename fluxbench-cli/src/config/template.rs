//! Default flux.toml template. It lists every key. Commented keys show optional settings.

pub(crate) const DEFAULT_TOML: &str = r#"# FluxBench Configuration
# https://github.com/ml-rust/fluxbench
#
# Priority: macro attribute > CLI flag > flux.toml > built-in default.
# Unknown keys are an error.

[runner]
# Warmup before measurement
warmup_time = "3s"
# Measurement duration
measurement_time = "5s"
# Worker timeout per benchmark. --worker-timeout overrides it.
# It rises to warmup + measurement + 2s when smaller, with a warning.
timeout = "60s"
# Isolation mode: "process" or "in-process". --isolated overrides it.
isolation = "process"
# Bootstrap resamples for confidence intervals. 0 skips the bootstrap.
bootstrap_iterations = 10000
# Confidence level, between 0 and 1
confidence_level = 0.95
# Target sample count, minimum 2, no upper cap (default: 100)
# sample_count = 100
# Fixed mode: N samples of one iteration each, no warmup
# samples = 5
# Minimum total measurement iterations
# min_iterations = 100
# Maximum total measurement iterations
# max_iterations = 1000000
# Parallel isolated workers (default: 1)
# jobs = 4
# Pin each isolated worker to its own CPU, Linux only (default: true)
# pin_cpus = false

[allocator]
# Track allocations during measurement
track = true
# Fail a benchmark whose mean allocation count per iteration is above 0
fail_on_allocation = false
# Fail a benchmark whose mean bytes per iteration is above this (default: unlimited)
# max_bytes_per_iter = 1024

[output]
# Report format when --format is absent: human, json, github-summary, github-action, csv, html
format = "human"
# Directory for bare file names given to --output, --baseline, --save-baseline, and baseline_path.
# Unset: bare --output names stay in the current directory, and bare baseline names go under target/fluxbench.
# directory = "target/fluxbench"
# Save a JSON baseline after each run
save_baseline = false
# Baseline file. A bare file name goes under the baseline directory. (default: baseline.json)
# baseline_path = "baseline.json"

[visuals]
# HTML report theme: "light" or "dark"
theme = "light"
# #[flux::chart] canvas size in pixels, at least 200 x 150
width = 1280
height = 720

[ci]
# A significant slowdown above this percentage is a regression
regression_threshold = 5.0
# Print GitHub Actions ::warning and ::error annotations on stderr
github_annotations = false
# Exit 1 on a critical verification failure, or a critical benchmark's regression or failure
fail_on_critical = true
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FluxConfig, IsolationMode, ReportFormat, Theme};
    use std::collections::BTreeSet;
    use std::path::Path;

    /// Uncomment every `# key = value` line so optional keys parse too.
    fn uncomment_keys(text: &str) -> String {
        text.lines()
            .map(|line| {
                let rest = line.strip_prefix("# ").unwrap_or(line);
                let is_key = rest.split_once(" = ").is_some_and(|(k, _)| {
                    !k.is_empty() && k.chars().all(|c| c.is_ascii_lowercase() || c == '_')
                });
                if line.starts_with("# ") && is_key {
                    rest
                } else {
                    line
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// `table.key` for every key in a TOML document
    fn key_paths(table: &toml::Table) -> BTreeSet<String> {
        let mut keys = BTreeSet::new();
        for (name, value) in table {
            match value.as_table() {
                Some(inner) => keys.extend(inner.keys().map(|k| format!("{name}.{k}"))),
                None => {
                    keys.insert(name.clone());
                }
            }
        }
        keys
    }

    /// Every config field, from a config with every optional field set
    fn struct_key_paths() -> BTreeSet<String> {
        let mut config = FluxConfig::default();
        let runner = &mut config.runner;
        runner.samples = Some(5);
        runner.sample_count = Some(100);
        runner.min_iterations = Some(1);
        runner.max_iterations = Some(2);
        runner.jobs = Some(1);
        runner.pin_cpus = Some(true);
        runner.isolation = IsolationMode::Process;
        config.allocator.max_bytes_per_iter = Some(1);
        config.output.baseline_path = Some("b.json".to_string());
        config.output.directory = Some("out".to_string());
        config.output.format = ReportFormat::Json;
        config.visuals.theme = Theme::Dark;
        let text = toml::to_string(&config).unwrap();
        key_paths(&toml::from_str(&text).unwrap())
    }

    fn check_document(text: &str) {
        FluxConfig::parse(text, Path::new("doc.toml")).unwrap();
        let full = uncomment_keys(text);
        let parsed = FluxConfig::parse(&full, Path::new("doc.toml"));
        assert!(
            parsed.is_ok(),
            "uncommented keys must be real: {parsed:?}\n{full}"
        );
        let table: toml::Table = toml::from_str(&full).unwrap();
        assert_eq!(key_paths(&table), struct_key_paths());
    }

    #[test]
    fn template_keys_match_struct_fields() {
        check_document(DEFAULT_TOML);
    }

    #[test]
    fn template_values_are_the_defaults() {
        let config = FluxConfig::parse(DEFAULT_TOML, Path::new("doc.toml")).unwrap();
        let defaults = FluxConfig::default();
        assert_eq!(config.runner.timeout, defaults.runner.timeout);
        assert_eq!(config.runner.warmup_time, defaults.runner.warmup_time);
        assert_eq!(
            config.runner.measurement_time,
            defaults.runner.measurement_time
        );
        assert_eq!(
            config.runner.bootstrap_iterations,
            defaults.runner.bootstrap_iterations
        );
        assert_eq!(config.output.directory, None);
        assert_eq!(config.output.directory, defaults.output.directory);
        assert_eq!(config.visuals.width, defaults.visuals.width);
        assert_eq!(
            config.ci.regression_threshold,
            defaults.ci.regression_threshold
        );
    }

    #[test]
    fn repo_flux_toml_keys_match_struct_fields() {
        check_document(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../flux.toml"
        )));
    }

    #[test]
    fn default_toml_matches_template() {
        assert_eq!(FluxConfig::default_toml(), DEFAULT_TOML);
    }
}
