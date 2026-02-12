#![warn(missing_docs)]
//! # FluxBench
//!
//! Benchmarking framework for Rust with crash isolation, statistical rigor, and CI integration.
//!
//! FluxBench provides a next-generation benchmarking platform:
//! - **Process Isolation**: Crash-resilient "Fail-Late" architecture; panicking benchmarks don't crash the suite
//! - **Zero-Copy IPC**: Efficient supervisor-worker communication using rkyv serialization
//! - **Statistical Rigor**: Bootstrap resampling with BCa (bias-corrected and accelerated) confidence intervals
//! - **CI Integration**: Severity levels (critical/warning/info), GitHub Actions summaries, baseline comparison
//! - **Algebraic Verification**: Performance assertions directly in code with mathematical expressions
//! - **Synthetic Metrics**: Compute derived metrics from benchmark results
//! - **Multi-Way Comparisons**: Generate comparison tables and series charts
//! - **Allocation Tracking**: `TrackingAllocator` measures heap usage per iteration
//! - **High-Precision Timing**: RDTSC cycle counting on x86_64 with Instant fallback
//!
//! ## Quick Start
//!
//! ```ignore
//! use fluxbench::prelude::*;
//!
//! #[flux::bench]
//! fn my_benchmark(b: &mut Bencher) {
//!     b.iter(|| {
//!         // Code to benchmark
//!         expensive_operation()
//!     });
//! }
//! ```
//!
//! ## Async Benchmarks
//!
//! ```ignore
//! #[flux::bench(runtime = "multi_thread", worker_threads = 4)]
//! async fn async_benchmark(b: &mut Bencher) {
//!     b.iter(|| async {
//!         tokio::time::sleep(Duration::from_millis(1)).await;
//!     });
//! }
//! ```
//!
//! ## Performance Assertions
//!
//! ```ignore
//! #[flux::verify(expr = "(raw - overhead) < 50000000", severity = "critical")]
//! struct NetTimeCheck;
//! ```

// Re-export core types
pub use fluxbench_core::{
    Bencher, BenchmarkDef, BenchmarkResult, ChartDef, ChartType, CompareDef, GroupDef,
    IterationMode, ReportDef, Severity, TrackingAllocator, current_allocation,
    reset_allocation_counter,
};

// Re-export macros
pub use fluxbench_macros::{bench, compare, group, report, synthetic, verify};

// Re-export logic types
pub use fluxbench_logic::{
    MetricContext, SyntheticDef, Verification, VerificationResult, VerificationStatus, VerifyDef,
};

// Re-export stats
pub use fluxbench_stats::{
    BootstrapConfig, BootstrapResult, SummaryStatistics, compute_bootstrap, compute_summary,
};

/// Internal re-exports for macro use
#[doc(hidden)]
pub mod internal {
    pub use inventory;
    pub use tokio;
}

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::{
        Bencher, CompareDef, GroupDef, ReportDef, Severity, bench, compare, group, report,
        synthetic, verify,
    };
}

/// Attribute namespace for flux macros
pub mod flux {
    pub use fluxbench_macros::{bench, compare, group, report, synthetic, verify};
}

/// Run the FluxBench CLI harness.
///
/// Call this from your benchmark binary's `main()`:
/// ```ignore
/// fn main() {
///     fluxbench::run().unwrap();
/// }
/// ```
pub use fluxbench_cli::run;
