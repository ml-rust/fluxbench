//! `[output]`: report format, output directory, and baseline files.

use crate::ReportFormat;
use serde::{Deserialize, Serialize};
use std::path::{Component, Path, PathBuf};

/// Baseline file name when nothing names one
pub(crate) const DEFAULT_BASELINE_FILE: &str = "baseline.json";

/// Baseline directory when `output.directory` is unset
pub(crate) const DEFAULT_BASELINE_DIR: &str = "target/fluxbench";

/// Output configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputConfig {
    /// Report format when `--format` is absent
    #[serde(default)]
    pub format: ReportFormat,
    /// Directory for bare file names given to `--output`, `--baseline`, `--save-baseline`,
    /// and `baseline_path`. Unset: `--output` names stay as written, and baselines use `target/fluxbench`.
    #[serde(default)]
    pub directory: Option<String>,
    /// Save JSON baseline after each run
    #[serde(default)]
    pub save_baseline: bool,
    /// Baseline file. A bare file name goes under the baseline directory.
    #[serde(default)]
    pub baseline_path: Option<String>,
}

impl OutputConfig {
    /// Resolve a `--output` report path.
    ///
    /// A bare file name goes under `directory` only when `directory` is set.
    /// Any other path, or any path with `directory` unset, stays as written.
    pub fn resolve_report(&self, path: &Path) -> PathBuf {
        match self.directory.as_deref() {
            Some(dir) if is_bare_file_name(path) => Path::new(dir).join(path),
            _ => path.to_path_buf(),
        }
    }

    /// Resolve a baseline path from `--baseline`, `--save-baseline`, or `baseline_path`.
    ///
    /// A bare file name goes under `directory`, else under `target/fluxbench`.
    /// Any other path stays as written.
    pub fn resolve_baseline(&self, path: &Path) -> PathBuf {
        if is_bare_file_name(path) {
            Path::new(self.directory.as_deref().unwrap_or(DEFAULT_BASELINE_DIR)).join(path)
        } else {
            path.to_path_buf()
        }
    }

    /// Baseline file: `baseline_path`, else `baseline.json`, resolved by [`Self::resolve_baseline`]
    pub fn baseline_file(&self) -> PathBuf {
        let name = self
            .baseline_path
            .as_deref()
            .unwrap_or(DEFAULT_BASELINE_FILE);
        self.resolve_baseline(Path::new(name))
    }

    /// Check every value. The error names the key, the value, and the fix.
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        if self
            .directory
            .as_deref()
            .is_some_and(|d| d.trim().is_empty())
        {
            anyhow::bail!(
                "output.directory is empty. Use a directory such as \"target/fluxbench\", or remove the key."
            );
        }
        if self
            .baseline_path
            .as_deref()
            .is_some_and(|p| p.trim().is_empty())
        {
            anyhow::bail!(
                "output.baseline_path is empty. Use a file such as \"baseline.json\", or remove the key."
            );
        }
        Ok(())
    }
}

/// `true` for a single relative file name with no directory part
fn is_bare_file_name(path: &Path) -> bool {
    let mut components = path.components();
    matches!(
        (components.next(), components.next()),
        (Some(Component::Normal(_)), None)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn output(directory: &str) -> OutputConfig {
        OutputConfig {
            directory: Some(directory.to_string()),
            ..Default::default()
        }
    }

    #[test]
    fn unset_directory_leaves_bare_report_name_unchanged() {
        assert_eq!(
            OutputConfig::default().resolve_report(Path::new("r.html")),
            PathBuf::from("r.html")
        );
    }

    #[test]
    fn set_directory_redirects_bare_report_name() {
        assert_eq!(
            output("results").resolve_report(Path::new("r.html")),
            PathBuf::from("results/r.html")
        );
    }

    #[test]
    fn paths_with_a_directory_part_stay() {
        let out = output("results");
        for path in ["./r.json", "a/r.json", "/tmp/r.json"] {
            assert_eq!(out.resolve_report(Path::new(path)), PathBuf::from(path));
            assert_eq!(out.resolve_baseline(Path::new(path)), PathBuf::from(path));
        }
    }

    #[test]
    fn baseline_bare_names_default_to_target_fluxbench() {
        let unset = OutputConfig::default();
        assert_eq!(
            unset.baseline_file(),
            PathBuf::from("target/fluxbench/baseline.json")
        );
        assert_eq!(
            unset.resolve_baseline(Path::new("main.json")),
            PathBuf::from("target/fluxbench/main.json")
        );
    }

    #[test]
    fn baseline_bare_names_follow_set_directory() {
        let mut out = output("ci");
        assert_eq!(
            out.resolve_baseline(Path::new("x.json")),
            PathBuf::from("ci/x.json")
        );
        out.baseline_path = Some("main.json".to_string());
        assert_eq!(out.baseline_file(), PathBuf::from("ci/main.json"));
        out.baseline_path = Some("./main.json".to_string());
        assert_eq!(out.baseline_file(), PathBuf::from("./main.json"));
    }

    #[test]
    fn empty_directory_is_rejected() {
        assert!(output(" ").validate().is_err());
        assert!(OutputConfig::default().validate().is_ok());
    }
}
