//! `[allocator]`, `[ci]`, and `[visuals]` sections.

use serde::{Deserialize, Serialize};

/// Allocator tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AllocatorConfig {
    /// Enable allocation tracking
    #[serde(default = "default_track")]
    pub track: bool,
    /// Fail a benchmark whose mean allocation count per iteration is above 0
    #[serde(default)]
    pub fail_on_allocation: bool,
    /// Fail a benchmark whose mean bytes per iteration is above this (None = unlimited)
    #[serde(default)]
    pub max_bytes_per_iter: Option<u64>,
}

impl Default for AllocatorConfig {
    fn default() -> Self {
        Self {
            track: default_track(),
            fail_on_allocation: false,
            max_bytes_per_iter: None,
        }
    }
}

impl AllocatorConfig {
    /// Whether any allocation limit is set
    pub fn has_limits(&self) -> bool {
        self.fail_on_allocation || self.max_bytes_per_iter.is_some()
    }
}

fn default_track() -> bool {
    true
}

/// CI/CD configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CiConfig {
    /// Regression threshold percentage.
    /// A statistically significant slowdown above it counts as a regression.
    #[serde(default = "default_threshold")]
    pub regression_threshold: f64,
    /// Enable GitHub Actions annotations
    #[serde(default)]
    pub github_annotations: bool,
    /// Exit non-zero on a critical verification failure,
    /// or on a regression or failure of a `severity = "critical"` benchmark
    #[serde(default = "default_fail_on_critical")]
    pub fail_on_critical: bool,
}

impl Default for CiConfig {
    fn default() -> Self {
        Self {
            regression_threshold: default_threshold(),
            github_annotations: false,
            fail_on_critical: default_fail_on_critical(),
        }
    }
}

impl CiConfig {
    /// Check every value. The error names the key, the value, and the fix.
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        if !self.regression_threshold.is_finite() || self.regression_threshold < 0.0 {
            anyhow::bail!(
                "ci.regression_threshold = {} is invalid. Use a finite percentage >= 0, such as 5.0.",
                self.regression_threshold
            );
        }
        Ok(())
    }
}

fn default_threshold() -> f64 {
    5.0
}
fn default_fail_on_critical() -> bool {
    true
}

/// HTML report color theme
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Theme {
    /// Dark text on a light background
    #[default]
    Light,
    /// Light text on a dark background
    Dark,
}

/// HTML report visuals
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VisualsConfig {
    /// HTML report color theme: "light" or "dark"
    #[serde(default)]
    pub theme: Theme,
    /// `#[flux::chart]` canvas width in pixels
    #[serde(default = "default_width")]
    pub width: u32,
    /// `#[flux::chart]` canvas height in pixels
    #[serde(default = "default_height")]
    pub height: u32,
}

impl Default for VisualsConfig {
    fn default() -> Self {
        Self {
            theme: Theme::default(),
            width: default_width(),
            height: default_height(),
        }
    }
}

impl VisualsConfig {
    /// Check every value. The error names the key, the value, and the fix.
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        if self.width < 200 || self.height < 150 {
            anyhow::bail!(
                "visuals.width = {} and visuals.height = {} are too small. Use at least 200 x 150.",
                self.width,
                self.height
            );
        }
        Ok(())
    }
}

fn default_width() -> u32 {
    1280
}
fn default_height() -> u32 {
    720
}
