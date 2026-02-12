//! FluxBench Examples
//!
//! Runnable demonstrations of every FluxBench feature. This crate is not
//! published — it exists solely to host examples that depend on `fluxbench`.
//!
//! Run any example with:
//! ```sh
//! cargo run --example <name> -p fluxbench-examples --release
//! ```
//!
//! ## Feature Examples
//!
//! | Example | Feature |
//! |---------|---------|
//! | `feature_iteration` | `iter()`, `iter_with_setup()`, `iter_batched()` |
//! | `feature_async` | Tokio runtimes, spawn, channels, mutexes, semaphores |
//! | `feature_params` | `args = [...]` parameterized scaling tests |
//! | `feature_verify` | `#[verify]` assertions and `#[synthetic]` metrics |
//! | `feature_compare` | `#[compare]` tables and series charts |
//! | `feature_allocations` | `TrackingAllocator` and zero-alloc verification |
//! | `feature_panic` | Crash isolation, stack overflow, edge cases |
//!
//! ## Use-Case Examples
//!
//! | Example | Scenario |
//! |---------|----------|
//! | `library_bench` | Library maintainer: groups, regression guards, comparisons |
//! | `ci_regression` | CI pipeline: severity levels, baselines, thresholds |
