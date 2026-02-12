//! Comparisons & Series Charts
//!
//! `#[compare]` creates side-by-side tables with speedup ratios. Add `group`,
//! `x`, and `series` to build multi-point scaling charts where each x value
//! is a separate comparison.
//!
//! Run with: cargo run --example feature_compare -p fluxbench --release

use fluxbench::prelude::*;
use fluxbench::{bench, compare};
use std::hint::black_box;

// ============================================================================
// Basic comparison — two implementations of the same task
// ============================================================================

/// Build a string via repeated push_str + format!
#[bench(group = "string_build")]
fn string_format(b: &mut Bencher) {
    b.iter(|| {
        let mut s = String::new();
        for i in 0..200 {
            s.push_str(&format!("{i},"));
        }
        black_box(s)
    });
}

/// Build a string via write! macro (avoids intermediate String from format!)
#[bench(group = "string_build")]
fn string_write(b: &mut Bencher) {
    use std::fmt::Write;
    b.iter(|| {
        let mut s = String::with_capacity(1024);
        for i in 0..200 {
            write!(s, "{i},").unwrap();
        }
        black_box(s)
    });
}

/// Build a string by collecting into a joined String
#[bench(group = "string_build")]
fn string_collect_join(b: &mut Bencher) {
    b.iter(|| {
        let s: String = (0..200)
            .map(|i: u32| i.to_string())
            .collect::<Vec<_>>()
            .join(",");
        black_box(s)
    });
}

#[compare(
    id = "string_build_cmp",
    title = "String Building Strategies",
    benchmarks = ["string_format", "string_write", "string_collect_join"],
    baseline = "string_format",
    metric = "mean"
)]
#[allow(dead_code)]
struct StringBuildComparison;

// ============================================================================
// Series chart — sum() vs fold() at increasing vector sizes
// ============================================================================
// Each size needs its own benchmark functions because `#[compare]` references
// benchmarks by name at compile time — there is no way to generate them from
// a loop or macro parameter list. This is an inherent limitation of the
// inventory-based registration model. Use `args = [...]` (see feature_params)
// when you only need one implementation at multiple sizes; use explicit
// functions + `#[compare]` when you need to compare DIFFERENT implementations
// at each size.
//
// Each #[compare] with the same `group` combines into one scaling chart.

// --- Size 100 ---
#[bench(group = "reduce")]
fn reduce_sum_100(b: &mut Bencher) {
    let v: Vec<i64> = (0..100).collect();
    b.iter(|| black_box(v.iter().sum::<i64>()));
}

#[bench(group = "reduce")]
#[allow(clippy::unnecessary_fold)]
fn reduce_fold_100(b: &mut Bencher) {
    let v: Vec<i64> = (0..100).collect();
    b.iter(|| black_box(v.iter().fold(0i64, |a, x| a + x)));
}

#[compare(
    id = "reduce_100",
    title = "Reduce Scaling",
    benchmarks = ["reduce_sum_100", "reduce_fold_100"],
    group = "vec_reduce",
    x = "100",
    series = ["sum()", "fold()"]
)]
#[allow(dead_code)]
struct Reduce100;

// --- Size 1 000 ---
#[bench(group = "reduce")]
fn reduce_sum_1k(b: &mut Bencher) {
    let v: Vec<i64> = (0..1_000).collect();
    b.iter(|| black_box(v.iter().sum::<i64>()));
}

#[bench(group = "reduce")]
#[allow(clippy::unnecessary_fold)]
fn reduce_fold_1k(b: &mut Bencher) {
    let v: Vec<i64> = (0..1_000).collect();
    b.iter(|| black_box(v.iter().fold(0i64, |a, x| a + x)));
}

#[compare(
    id = "reduce_1k",
    title = "Reduce Scaling",
    benchmarks = ["reduce_sum_1k", "reduce_fold_1k"],
    group = "vec_reduce",
    x = "1000",
    series = ["sum()", "fold()"]
)]
#[allow(dead_code)]
struct Reduce1k;

// --- Size 10 000 ---
#[bench(group = "reduce")]
fn reduce_sum_10k(b: &mut Bencher) {
    let v: Vec<i64> = (0..10_000).collect();
    b.iter(|| black_box(v.iter().sum::<i64>()));
}

#[bench(group = "reduce")]
#[allow(clippy::unnecessary_fold)]
fn reduce_fold_10k(b: &mut Bencher) {
    let v: Vec<i64> = (0..10_000).collect();
    b.iter(|| black_box(v.iter().fold(0i64, |a, x| a + x)));
}

#[compare(
    id = "reduce_10k",
    title = "Reduce Scaling",
    benchmarks = ["reduce_sum_10k", "reduce_fold_10k"],
    group = "vec_reduce",
    x = "10000",
    series = ["sum()", "fold()"]
)]
#[allow(dead_code)]
struct Reduce10k;

// --- Size 100 000 ---
#[bench(group = "reduce")]
fn reduce_sum_100k(b: &mut Bencher) {
    let v: Vec<i64> = (0..100_000).collect();
    b.iter(|| black_box(v.iter().sum::<i64>()));
}

#[bench(group = "reduce")]
#[allow(clippy::unnecessary_fold)]
fn reduce_fold_100k(b: &mut Bencher) {
    let v: Vec<i64> = (0..100_000).collect();
    b.iter(|| black_box(v.iter().fold(0i64, |a, x| a + x)));
}

#[compare(
    id = "reduce_100k",
    title = "Reduce Scaling",
    benchmarks = ["reduce_sum_100k", "reduce_fold_100k"],
    group = "vec_reduce",
    x = "100000",
    series = ["sum()", "fold()"]
)]
#[allow(dead_code)]
struct Reduce100k;

fn main() {
    if let Err(e) = fluxbench::run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
