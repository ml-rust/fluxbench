//! Run pipeline shared by the default run and the `compare` subcommand.
//!
//! - `execute`: executor selection, run config checks, and execution
//! - `checks`: budget and allocation checks after measurement
//! - `finish`: report, baseline comparison, output, baseline save
//! - `output`: report rendering and writing, empty reports included
//! - `exit`: CI exit decision
//! - `run`, `compare`: the two entry points

mod checks;
mod compare;
mod execute;
mod exit;
mod finish;
mod output;
mod run;

pub(crate) use compare::compare_benchmarks;
pub(crate) use run::run_benchmarks;
