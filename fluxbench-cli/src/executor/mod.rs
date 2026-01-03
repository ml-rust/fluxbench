//! Benchmark Executor
//!
//! Runs benchmarks and collects results. Supports both in-process execution
//! (for development) and out-of-process execution via supervisor-worker IPC.
//!
//! ## Pipeline Overview
//!
//! ```text
//! BenchmarkDef (registered via #[bench])
//!       │
//!       ▼
//! ┌─────────────┐
//! │  execution  │  Run benchmarks, collect samples
//! └──────┬──────┘
//!        │
//!        ▼
//! ┌─────────────┐
//! │ statistics  │  Compute summary stats (parallel)
//! └──────┬──────┘
//!        │
//!        ▼
//! ┌─────────────┐
//! │   report    │  Build Report with bootstrap CIs
//! └──────┬──────┘
//!        │
//!        ▼
//! ┌─────────────┐
//! │verification │  Comparisons, synthetics, pass/fail
//! └──────┬──────┘
//!        │
//!        ▼
//! ┌─────────────┐
//! │ formatting  │  Human-readable output
//! └─────────────┘
//! ```
//!
//! ## Modules
//!
//! - [`execution`] - Core benchmark execution logic (in-process and isolated)
//! - [`statistics`] - Parallel statistics computation
//! - [`report`] - Report building with bootstrap CIs
//! - [`verification`] - Comparison, synthetic, and verification processing
//! - [`formatting`] - Human-readable output formatting
//! - [`metadata`] - System metadata collection

mod execution;
mod formatting;
mod metadata;
mod report;
mod statistics;
mod verification;

// Re-export public API
pub use execution::{ExecutionConfig, Executor, IsolatedExecutor};
pub use formatting::format_human_output;
pub use report::build_report;
pub use statistics::compute_statistics;
pub use verification::execute_verifications;
