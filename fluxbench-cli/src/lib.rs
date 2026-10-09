#![warn(missing_docs)]
//! FluxBench CLI Library
//!
//! This module provides the CLI infrastructure for benchmark binaries.
//! Use `fluxbench::run()` (or `fluxbench_cli::run()`) in your main function to get the full
//! fluxbench CLI experience with your registered benchmarks.
//!
//! # Example
//!
//! ```ignore
//! use fluxbench::prelude::*;
//!
//! #[bench]
//! fn my_benchmark(b: &mut Bencher) {
//!     b.iter(|| expensive_operation());
//! }
//!
//! fn main() {
//!     fluxbench_cli::run();
//! }
//! ```

mod baseline;
mod cli_args;
mod config;
mod cpu_pin;
mod definitions;
mod dispatch;
mod executor;
mod groups;
mod planner;
mod report_format;
mod run_config;
mod run_pipeline;
mod selection;
mod status;
mod supervisor;
#[cfg(test)]
mod test_support;

pub use cli_args::{Cli, Commands};
pub use config::*;
pub use dispatch::{run, run_with_cli};
pub use executor::{
    ExecutionConfig, Executor, GroupConfigs, IsolatedExecutor, build_report, compute_statistics,
    execute_verifications, format_human_output,
};
pub use report_format::ReportFormat;
pub use supervisor::*;
