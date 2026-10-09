#![warn(missing_docs)]
//! # FluxBench
//!
//! Benchmarking framework for Rust with crash isolation, statistical rigor, and CI integration.
//!
//! FluxBench provides a next-generation benchmarking platform:
//! - **Process Isolation**: Crash-resilient "Fail-Late" architecture; panicking benchmarks don't crash the suite
//! - **Binary IPC**: Supervisor-worker messages use length-prefixed rkyv frames
//! - **Statistical Rigor**: Bootstrap resampling with BCa (bias-corrected and accelerated) confidence intervals
//! - **CI Integration**: Severity levels (critical/warning/info), GitHub Actions summaries, baseline comparison
//! - **Algebraic Verification**: Performance assertions directly in code with mathematical expressions
//! - **Synthetic Metrics**: Compute derived metrics from benchmark results
//! - **Multi-Way Comparisons**: Generate comparison tables and series charts
//! - **Allocation Tracking**: `TrackingAllocator` measures heap usage per iteration
//! - **Timing**: `Instant` per timed region with timer overhead subtracted. Cycle counts use RDTSCP (x86_64) or CNTVCT_EL0 (aarch64), else 0
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
//! ## Setup and Batching
//!
//! `iter_batched` builds inputs untimed and hands each one to the routine by value.
//! `iter_batched_ref` hands a `&mut` input instead. `BatchSize` sets the inputs per timed batch.
//!
//! ```ignore
//! #[flux::bench]
//! fn sort_fresh(b: &mut Bencher) {
//!     b.iter_batched(
//!         || (0..1_000u32).rev().collect::<Vec<_>>(),
//!         |mut v| { v.sort(); v },
//!         BatchSize::SmallInput,
//!     );
//! }
//!
//! #[flux::bench]
//! fn clear_in_place(b: &mut Bencher) {
//!     b.iter_batched_ref(|| vec![0u8; 4096], |v| v.fill(1), BatchSize::LargeInput);
//! }
//! ```
//!
//! `iter_custom` lets the body time `iters` calls itself:
//!
//! ```ignore
//! #[flux::bench]
//! fn self_timed(b: &mut Bencher) {
//!     b.iter_custom(|iters| {
//!         let start = std::time::Instant::now();
//!         for _ in 0..iters {
//!             std::hint::black_box(2 + 2);
//!         }
//!         start.elapsed()
//!     });
//! }
//! ```
//!
//! ## Throughput
//!
//! Set the work per call. Reports show bytes per second (binary units) or elements per second.
//!
//! ```ignore
//! #[flux::bench(throughput_bytes = 4096)]
//! fn checksum_4k(b: &mut Bencher) {
//!     let data = vec![1u8; 4096];
//!     b.iter(|| data.iter().map(|&x| x as u64).sum::<u64>());
//! }
//!
//! #[flux::bench]
//! fn parse_lines(b: &mut Bencher) {
//!     let lines: Vec<String> = (0..100).map(|i| i.to_string()).collect();
//!     b.throughput(Throughput::Elements(lines.len() as u64));
//!     b.iter(|| lines.iter().filter_map(|l| l.parse::<u32>().ok()).sum::<u32>());
//! }
//! ```
//!
//! ## Group Config
//!
//! A group sets run config for its benchmarks and child groups.
//! A `#[flux::bench]` attribute beats it. CLI flags and flux.toml lose to it.
//!
//! ```ignore
//! #[flux::group(id = "io", sample_count = 30, measurement = "2s", warmup = "500ms")]
//! struct Io;
//! ```
//!
//! ## Async Benchmarks
//!
//! The `async` feature (default) enables them.
//! `iter_async` times creating and awaiting each future.
//! The runtime builds once per benchmark. `current_thread` (the default) and `multi_thread` both work.
//!
//! ```ignore
//! #[flux::bench(runtime = "multi_thread", worker_threads = 4)]
//! async fn async_benchmark(b: &mut Bencher) {
//!     b.iter_async(|| async {
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

mod async_bench;

// Re-export core types
pub use fluxbench_core::{
    BatchSize, Bencher, BenchmarkDef, BenchmarkResult, ChartDef, ChartType, CompareDef,
    GroupConfig, GroupDef, IterationMode, MeasureConfig, MeasureError, ReportDef, Severity,
    Throughput, TrackingAllocator, current_allocation, reset_allocation_counter,
};

// Re-export macros
pub use fluxbench_macros::{bench, chart, compare, group, report, synthetic, verify};

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
    #[cfg(feature = "async")]
    pub use fluxbench_core::drive_async_body;
    pub use inventory;
    #[cfg(feature = "async")]
    pub use tokio;
}

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::{
        BatchSize, Bencher, ChartDef, ChartType, CompareDef, GroupDef, ReportDef, Severity,
        Throughput, bench, chart, compare, group, report, synthetic, verify,
    };
}

/// Attribute namespace for flux macros
pub mod flux {
    pub use fluxbench_macros::{bench, chart, compare, group, report, synthetic, verify};
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
