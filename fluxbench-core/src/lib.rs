#![warn(missing_docs)]
//! FluxBench Core - Worker Runtime
//!
//! This crate provides the execution environment for benchmarks:
//! - `Bencher` struct for iteration-based benchmarking
//! - Timing: `std::time::Instant` per timed region, with the empty-timer overhead subtracted.
//!   Cycle counts come from RDTSCP (x86_64) or CNTVCT_EL0 (aarch64), and are 0 elsewhere
//! - Global allocator interceptor for memory tracking
//! - CPU affinity pinning for stable cycle-counter readings
//!
//! The `async` feature (default) adds `iter_async` and `drive_async_body` on tokio.
//! Without it, the crate has no tokio dependency.

mod allocator;
mod bencher;
mod group_config;
mod measure;
mod shutdown;
mod worker;

pub use allocator::{TrackingAllocator, current_allocation, reset_allocation_counter};
#[cfg(feature = "async")]
pub use bencher::drive_async_body;
pub use bencher::{
    BatchSize, Bencher, BenchmarkResult, DEFAULT_SAMPLE_COUNT, IterationMode, LARGE_INPUT_CHUNK,
    MIN_SAMPLE_COUNT, MeasureConfig, MeasureError, SETUP_CHUNK, SampleSink, run_benchmark_loop,
};
pub use fluxbench_ipc::Throughput;
pub use group_config::GroupConfig;
/// Whether this platform provides hardware cycle counters (x86_64 RDTSCP or AArch64 CNTVCT_EL0).
/// When `false`, cycle counts are reported as 0 and only wall-clock nanoseconds are available.
pub use measure::HAS_CYCLE_COUNTER;
pub use measure::Instant;
pub use measure::Timer;
pub use measure::{cycles_overhead, timer_overhead_ns};
pub use worker::WorkerMain;

/// Benchmark definition registered via `#[flux::bench]`
#[derive(Debug, Clone)]
pub struct BenchmarkDef {
    /// Unique identifier
    pub id: &'static str,
    /// Human-readable name
    pub name: &'static str,
    /// Group this benchmark belongs to
    pub group: &'static str,
    /// Severity level for CI reporting
    pub severity: Severity,
    /// Per-benchmark regression threshold percentage (0.0 = use global threshold)
    pub threshold: f64,
    /// Time budget for the mean per-iteration time, in nanoseconds.
    /// A mean above it fails the benchmark.
    pub budget_ns: Option<u64>,
    /// Tags for filtering
    pub tags: &'static [&'static str],
    /// Function pointer to the wrapper
    pub runner_fn: fn(&mut Bencher),
    /// Source file path
    pub file: &'static str,
    /// Source line number
    pub line: u32,
    /// Module path
    pub module_path: &'static str,
    /// Per-benchmark warmup override (nanoseconds)
    pub warmup_ns: Option<u64>,
    /// Per-benchmark measurement override (nanoseconds)
    pub measurement_ns: Option<u64>,
    /// Per-benchmark fixed mode: N samples of one iteration each, no warmup
    pub samples: Option<u64>,
    /// Per-benchmark target sample count. It overrides only the count.
    pub sample_count: Option<u64>,
    /// Per-benchmark fixed iterations per sample. `Some` selects fixed mode.
    pub iters_per_sample: Option<u64>,
    /// Per-benchmark minimum total measurement iterations
    pub min_iterations: Option<u64>,
    /// Per-benchmark maximum total measurement iterations
    pub max_iterations: Option<u64>,
    /// Per-benchmark work per call for throughput rates.
    /// `Bencher::throughput` in the body overrides it.
    pub throughput: Option<Throughput>,
    /// Benchmark IDs that must run before this one
    pub depends_on: &'static [&'static str],
}

/// Severity levels for CI integration
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Severity {
    /// Critical benchmark - regression fails the build
    Critical,
    /// Warning level - logged but doesn't fail
    Warning,
    /// Informational only
    Info,
}

/// Group definition registered via `#[flux::group]`.
///
/// `id` matches the `group` of `#[bench]`.
/// The HTML report shows `description` and `tags` in the group's section.
#[derive(Debug, Clone)]
pub struct GroupDef {
    /// Group identifier, as used by `#[bench(group = "...")]`
    pub id: &'static str,
    /// Description shown in the HTML report
    pub description: &'static str,
    /// Tags every benchmark in this group and its child groups carries for `--tag` and `--skip-tag`
    pub tags: &'static [&'static str],
    /// Parent group. `--group <parent>` also selects this group.
    pub parent: Option<&'static str>,
    /// Run config for every benchmark in this group and its child groups
    pub config: GroupConfig,
}

/// Chart kind for `#[flux::chart]`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChartType {
    /// One bar per benchmark
    Bar,
    /// One point per benchmark, joined in the listed order
    Line,
}

/// Chart definition registered via `#[flux::chart]`.
///
/// The HTML report draws it as an inline SVG.
#[derive(Debug, Clone)]
pub struct ChartDef {
    /// Chart title
    pub title: &'static str,
    /// Chart kind
    pub chart_type: ChartType,
    /// Report metric to plot, e.g. `mean` or `p99`
    pub metric: &'static str,
    /// Benchmark IDs to plot, in order
    pub benchmarks: &'static [&'static str],
    /// Optional horizontal target line, in the metric's unit
    pub target_line: Option<f64>,
    /// Optional 1-based grid cell (row, col) in the `#[flux::report]` layout
    pub position: Option<(u32, u32)>,
}

/// Report definition registered via `#[flux::report]`.
///
/// The HTML report takes its title and description from it.
/// Charts sit in a `layout` grid of (rows, cols).
#[derive(Debug, Clone)]
pub struct ReportDef {
    /// Report title
    pub title: &'static str,
    /// Report description shown under the title
    pub description: &'static str,
    /// Chart grid layout (rows, cols)
    pub layout: (u32, u32),
}

/// Comparison group - groups multiple benchmarks for side-by-side comparison
#[derive(Debug, Clone)]
pub struct CompareDef {
    /// Comparison identifier
    pub id: &'static str,
    /// Human-readable title
    pub title: &'static str,
    /// Benchmark IDs to compare
    pub benchmarks: &'static [&'static str],
    /// Optional baseline benchmark (first one if not specified)
    pub baseline: Option<&'static str>,
    /// Metric to compare (default: mean)
    pub metric: &'static str,
    /// Group for chart generation (comparisons with same group form a chart)
    pub group: Option<&'static str>,
    /// X-axis value for this comparison point (e.g., "1", "10", "100")
    pub x: Option<&'static str>,
    /// Series labels (display names) - must match benchmarks array length
    /// If not specified, benchmark IDs are used as labels
    pub series: Option<&'static [&'static str]>,
}

// Collect all registered benchmarks
inventory::collect!(BenchmarkDef);
inventory::collect!(GroupDef);
inventory::collect!(ReportDef);
inventory::collect!(ChartDef);
inventory::collect!(CompareDef);

/// Anchor to prevent LTO from stripping inventory entries
#[used]
#[doc(hidden)]
pub static REGISTRY_ANCHOR: fn() = || {
    for _ in inventory::iter::<BenchmarkDef> {}
    for _ in inventory::iter::<GroupDef> {}
    for _ in inventory::iter::<ReportDef> {}
    for _ in inventory::iter::<ChartDef> {}
    for _ in inventory::iter::<CompareDef> {}
};
