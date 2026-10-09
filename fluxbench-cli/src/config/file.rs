//! `FluxConfig`: the whole flux.toml, its discovery, and its checks.

use super::duration::parse_duration_ns;
use super::output::OutputConfig;
use super::runner::RunnerConfig;
use super::sections::{AllocatorConfig, CiConfig, VisualsConfig};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Config file name that discovery looks for
pub const CONFIG_FILE_NAME: &str = "flux.toml";

/// FluxBench configuration. Every table rejects unknown keys.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct FluxConfig {
    /// Runner configuration
    #[serde(default)]
    pub runner: RunnerConfig,
    /// HTML report visuals
    #[serde(default)]
    pub visuals: VisualsConfig,
    /// Allocator tracking configuration
    #[serde(default)]
    pub allocator: AllocatorConfig,
    /// Output configuration
    #[serde(default)]
    pub output: OutputConfig,
    /// CI/CD configuration
    #[serde(default)]
    pub ci: CiConfig,
}

impl FluxConfig {
    /// Read, parse, and check a config file.
    ///
    /// The error names the file. A parse error also names the line, column, and key.
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).map_err(|e| {
            anyhow::anyhow!(
                "cannot read config file {}: {e}. Fix its permissions, or remove it to use defaults.",
                path.display()
            )
        })?;
        Self::parse(&content, path)
    }

    /// Parse and check config text. `origin` names the file in errors.
    pub fn parse(content: &str, origin: &Path) -> anyhow::Result<Self> {
        let config: Self = toml::from_str(content).map_err(|e| {
            anyhow::anyhow!(
                "invalid config file {}: {}Fix the key or value it names. The README lists every key.",
                origin.display(),
                with_trailing_newline(e.to_string())
            )
        })?;
        config
            .validate()
            .map_err(|e| anyhow::anyhow!("invalid config file {}: {e}", origin.display()))?;
        Ok(config)
    }

    /// Check every value. The error names the key, the value, and the fix.
    pub fn validate(&self) -> anyhow::Result<()> {
        self.runner.validate()?;
        self.visuals.validate()?;
        self.output.validate()?;
        self.ci.validate()
    }

    /// Find flux.toml in the current directory or a parent, then load it.
    ///
    /// `Ok(None)` means no file exists. A file that fails to load is an error.
    pub fn discover() -> anyhow::Result<Option<Self>> {
        let cwd = std::env::current_dir().map_err(|e| {
            anyhow::anyhow!(
                "cannot read the current directory: {e}. Run from an existing directory."
            )
        })?;
        Self::discover_from(&cwd)
    }

    /// Find flux.toml in `start` or a parent, then load it.
    pub fn discover_from(start: &Path) -> anyhow::Result<Option<Self>> {
        match find_config_file(start) {
            Some(path) => Self::load(&path).map(Some),
            None => Ok(None),
        }
    }

    /// Generate a default configuration as TOML string
    pub fn default_toml() -> String {
        super::template::DEFAULT_TOML.to_string()
    }

    /// Parse duration string (e.g., "3s", "500ms", "2m") to nanoseconds
    pub fn parse_duration(s: &str) -> anyhow::Result<u64> {
        parse_duration_ns(s)
    }
}

/// Path of flux.toml in `start` or its nearest parent that has one
fn find_config_file(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .map(|dir| dir.join(CONFIG_FILE_NAME))
        .find(|path| path.exists())
}

fn with_trailing_newline(mut text: String) -> String {
    if !text.ends_with('\n') {
        text.push('\n');
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ReportFormat;

    fn parse(text: &str) -> anyhow::Result<FluxConfig> {
        FluxConfig::parse(text, Path::new("/proj/flux.toml"))
    }

    fn temp_dir(label: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("fluxbench-config-{}-{label}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn defaults() {
        let config = FluxConfig::default();
        assert_eq!(config.runner.warmup_time, "3s");
        assert_eq!(config.runner.measurement_time, "5s");
        assert_eq!(config.runner.timeout_ns().unwrap(), 60_000_000_000);
        assert!(config.allocator.track);
        assert!(!config.allocator.fail_on_allocation);
        assert_eq!(config.output.format, ReportFormat::Human);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn partial_toml_keeps_defaults() {
        let config = parse(
            "[runner]\nwarmup_time = \"1s\"\nmeasurement_time = \"2s\"\n\n[allocator]\ntrack = false\n",
        )
        .unwrap();
        assert_eq!(config.runner.warmup_time, "1s");
        assert!(!config.allocator.track);
        assert_eq!(config.output.format, ReportFormat::Human);
    }

    #[test]
    fn unknown_key_is_rejected_with_name_path_and_valid_keys() {
        let err = parse("[runner]\nwarmup = \"1s\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("/proj/flux.toml"), "{err}");
        assert!(err.contains("unknown field `warmup`"), "{err}");
        assert!(err.contains("warmup_time"), "{err}");
    }

    #[test]
    fn unknown_table_is_rejected() {
        let err = parse("[runnr]\njobs = 2\n").unwrap_err().to_string();
        assert!(err.contains("unknown field `runnr`"), "{err}");
        assert!(err.contains("runner"), "{err}");
    }

    #[test]
    fn unknown_key_in_every_table_is_rejected() {
        for table in ["runner", "allocator", "output", "visuals", "ci"] {
            let err = parse(&format!("[{table}]\nbogus_key = 1\n"))
                .unwrap_err()
                .to_string();
            assert!(err.contains("unknown field `bogus_key`"), "{table}: {err}");
        }
    }

    #[test]
    fn malformed_toml_names_path_and_location() {
        let err = parse("[runner\nwarmup_time = \"1s\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("invalid config file /proj/flux.toml"), "{err}");
        assert!(err.contains("line 1"), "{err}");
    }

    #[test]
    fn bad_values_are_rejected_with_key() {
        for (text, needle) in [
            (
                "[runner]\ntimeout = \"soon\"\n",
                "runner.timeout = \"soon\"",
            ),
            ("[runner]\nwarmup_time = \"-1s\"\n", "runner.warmup_time"),
            (
                "[runner]\nmeasurement_time = \"0s\"\n",
                "runner.measurement_time",
            ),
            ("[runner]\nsample_count = 1\n", "runner.sample_count = 1"),
            ("[runner]\nsamples = 1\n", "runner.samples = 1"),
            (
                "[runner]\nconfidence_level = 1.5\n",
                "runner.confidence_level",
            ),
            ("[runner]\njobs = 0\n", "runner.jobs = 0"),
            (
                "[runner]\nmin_iterations = 10\nmax_iterations = 5\n",
                "runner.max_iterations",
            ),
            (
                "[runner]\nisolation = \"thread\"\n",
                "unknown variant `thread`",
            ),
            ("[output]\nformat = \"yaml\"\n", "unknown variant `yaml`"),
            ("[output]\ndirectory = \"\"\n", "output.directory"),
            ("[visuals]\ntheme = \"blue\"\n", "unknown variant `blue`"),
            ("[visuals]\nwidth = 10\n", "visuals.width"),
            (
                "[ci]\nregression_threshold = -1.0\n",
                "ci.regression_threshold",
            ),
        ] {
            let err = parse(text).unwrap_err().to_string();
            assert!(err.contains(needle), "{text}: {err}");
            assert!(err.contains("/proj/flux.toml"), "{text}: {err}");
        }
    }

    #[test]
    fn load_missing_file_is_an_error_naming_it() {
        let err = FluxConfig::load("/nonexistent/fluxbench/flux.toml")
            .unwrap_err()
            .to_string();
        assert!(err.contains("cannot read config file /nonexistent/fluxbench/flux.toml"));
    }

    #[test]
    fn discover_without_file_is_none() {
        let dir = temp_dir("none");
        let nested = dir.join("a/b");
        std::fs::create_dir_all(&nested).unwrap();
        // A flux.toml above the temp dir is outside this test's control, so check the walk directly.
        assert!(
            find_config_file(&nested).is_none_or(|p| !p.starts_with(&dir)),
            "no flux.toml exists under the temp dir"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn discover_finds_parent_file_and_fails_on_malformed() {
        let dir = temp_dir("parent");
        let nested = dir.join("x/y");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(dir.join("flux.toml"), "[runner]\njobs = 3\n").unwrap();
        let config = FluxConfig::discover_from(&nested).unwrap().unwrap();
        assert_eq!(config.runner.jobs, Some(3));

        std::fs::write(dir.join("flux.toml"), "[runner]\njobs = \"three\"\n").unwrap();
        let err = FluxConfig::discover_from(&nested).unwrap_err().to_string();
        assert!(
            err.contains(&dir.join("flux.toml").display().to_string()),
            "{err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
