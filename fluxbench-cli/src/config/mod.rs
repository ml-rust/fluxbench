//! flux.toml config: discovery, parsing, checks, and the default template.
//!
//! FluxBench finds `flux.toml` by walking up from the current directory.
//! Unknown keys, bad values, and malformed files are errors.

mod duration;
mod file;
mod output;
mod runner;
mod sections;
mod template;

pub(crate) use duration::parse_duration_arg;
pub use file::{CONFIG_FILE_NAME, FluxConfig};
pub use output::OutputConfig;
pub use runner::{IsolationMode, RunnerConfig};
pub use sections::{AllocatorConfig, CiConfig, Theme, VisualsConfig};
