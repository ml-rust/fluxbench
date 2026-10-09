//! Baseline files and regression detection.
//!
//! - `store`: baseline path, load, and save
//! - `estimate`: bootstrap or summary statistics of one change
//! - `verdict`: significance and threshold rules
//! - `compare`: comparison of a whole report
//! - `format`: human-readable comparison
//! - `git_ref`: git ref checks for `compare`

mod compare;
mod estimate;
mod format;
mod git_ref;
mod store;
mod verdict;

pub(crate) use compare::{ComparisonSettings, apply_baseline_comparison};
pub(crate) use format::format_comparison_output;
pub(crate) use git_ref::{check_baseline_commit, resolve_git_ref};
pub(crate) use store::{
    configured_or_default, load_baseline, resolve_baseline_path, save_baseline_if_needed,
};
