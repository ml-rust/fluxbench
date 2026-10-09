//! FluxBench Macros
//!
//! Procedural macros for benchmark registration and async wrapping.
//!
//! ## Macros
//!
//! - `#[flux::bench]` - Register a benchmark function
//! - `#[flux::verify]` - Define a performance assertion
//! - `#[flux::synthetic]` - Define a computed metric
//! - `#[flux::compare]` - Define a comparison table or series
//! - `#[flux::group]` - Describe a benchmark group
//! - `#[flux::report]` - Set the HTML report title, description, and chart grid
//! - `#[flux::chart]` - Add an SVG chart to the HTML report

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use syn::{ItemFn, ItemStruct, parse_macro_input};

mod attr;
mod bench;
mod compare;
mod duration;
mod group;
mod layout;
mod throughput;
mod verify;

/// Register a benchmark function
///
/// # Example
///
/// ```ignore
/// #[flux::bench]
/// fn my_benchmark(b: &mut Bencher) {
///     b.iter(|| expensive_operation());
/// }
///
/// // With configuration
/// #[flux::bench(
///     id = "custom_id",
///     group = "parsing",
///     severity = "critical",
///     threshold = 5.0,
///     budget = "100ms"
/// )]
/// fn critical_benchmark(b: &mut Bencher) { ... }
///
/// // 250 samples of exactly 10 iterations each, no warmup
/// #[flux::bench(sample_count = 250, iters_per_sample = 10)]
/// fn fixed_benchmark(b: &mut Bencher) { ... }
///
/// // 4 KiB of work per call: reports show bytes per second
/// #[flux::bench(throughput_bytes = 4096)]
/// fn checksum_4k(b: &mut Bencher) { ... }
///
/// // Async with multi-threaded runtime. Needs the `async` feature of fluxbench (default).
/// #[flux::bench(runtime = "multi_thread", worker_threads = 4)]
/// async fn async_benchmark(b: &mut Bencher) { ... }
/// ```
///
/// `args = [..]` runs one benchmark per value. The function takes the value as a second argument.
/// Each ID is `id@value`.
/// `depends_on = [..]` lists benchmark IDs that run first.
/// `enable_time` and `enable_io` (default `true`) set the tokio drivers of the async runtime.
///
/// `budget`, `warmup`, and `measurement` take a number with a unit: `ns`, `us`, `ms`, or `s`.
/// A mean per-iteration time above `budget` fails the benchmark.
/// `throughput_bytes` and `throughput_elements` take an integer above 0. Set at most one.
/// `Bencher::throughput` in the body overrides them.
/// An `async fn` without the `async` feature is a compile error that names the feature.
#[proc_macro_attribute]
pub fn bench(args: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    finish(bench::bench_impl(TokenStream2::from(args), func))
}

/// Define a performance verification
///
/// `margin = N` widens a single top-level comparison by N percent of its right side.
#[proc_macro_attribute]
pub fn verify(args: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemStruct);
    finish(verify::verify_impl(TokenStream2::from(args), input))
}

/// Define a synthetic (computed) metric
#[proc_macro_attribute]
pub fn synthetic(args: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemStruct);
    finish(verify::synthetic_impl(TokenStream2::from(args), input))
}

/// Describe a benchmark group
///
/// # Example
///
/// ```ignore
/// #[flux::group(
///     id = "parsing",
///     description = "Parser performance tests",
///     tags = "hot-path, critical"
/// )]
/// struct ParsingGroup;
///
/// // Nested groups with run config for every benchmark below them
/// #[flux::group(
///     id = "json_parsing",
///     parent = "parsing",
///     description = "JSON-specific parsing benchmarks",
///     sample_count = 30,
///     measurement = "2s",
///     warmup = "500ms",
///     throughput_bytes = 65536
/// )]
/// struct JsonParsingGroup;
/// ```
///
/// Run keys: `warmup`, `measurement`, `sample_count`, `iters_per_sample`, `min_iterations`,
/// `max_iterations`, and one of `throughput_bytes` or `throughput_elements`.
/// Precedence per key: `#[flux::bench]`, then the nearest group, then parent groups,
/// then CLI flags and flux.toml, then the default.
#[proc_macro_attribute]
pub fn group(args: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemStruct);
    finish(group::group_impl(TokenStream2::from(args), input))
}

/// Set the HTML report title, description, and chart grid
///
/// # Example
///
/// ```ignore
/// #[flux::report(
///     title = "Performance Dashboard",
///     description = "Nightly run",
///     layout = "grid(2, 2)"
/// )]
/// struct MyDashboard;
/// ```
#[proc_macro_attribute]
pub fn report(args: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemStruct);
    finish(layout::report_impl(TokenStream2::from(args), input))
}

/// Add an SVG chart of one metric per benchmark to the HTML report
///
/// # Example
///
/// ```ignore
/// #[flux::chart(
///     title = "Sort latency",
///     kind = "bar",
///     metric = "p99",
///     benchmarks = ["sort_small", "sort_large"],
///     target = 5000,
///     row = 1,
///     col = 1
/// )]
/// struct SortChart;
/// ```
#[proc_macro_attribute]
pub fn chart(args: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemStruct);
    finish(layout::chart_impl(TokenStream2::from(args), input))
}

/// Define a comparison group for multiple benchmarks
///
/// # Example
///
/// ```ignore
/// #[flux::compare(
///     id = "parser_showdown",
///     title = "JSON Parser Comparison",
///     benchmarks = ["bench_serde", "bench_simd_json"],
///     baseline = "bench_serde",
///     metric = "mean"
/// )]
/// struct ParserComparison;
/// ```
#[proc_macro_attribute]
pub fn compare(args: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemStruct);
    finish(compare::compare_impl(TokenStream2::from(args), input))
}

/// Turn a macro result into tokens, or into a compile error.
fn finish(result: Result<TokenStream2, syn::Error>) -> TokenStream {
    result.unwrap_or_else(|e| e.to_compile_error()).into()
}
