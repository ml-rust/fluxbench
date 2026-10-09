//! Run pipeline shared by the default run and the `compare` subcommand.
//!
//! - `execute`: executor selection and execution
//! - `finish`: report, baseline comparison, output, baseline save
//! - `exit`: CI exit decision
//! - `run`, `compare`: the two entry points

mod compare;
mod execute;
mod exit;
mod finish;
mod run;

pub(crate) use compare::compare_benchmarks;
pub(crate) use run::run_benchmarks;
