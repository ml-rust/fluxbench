//! Allocation Tracking — Measure and verify heap usage
//!
//! Install `TrackingAllocator` as the global allocator to capture per-iteration
//! allocation bytes and counts. Use `reset_allocation_counter()` before the
//! measured section and `current_allocation()` after to inspect results.
//!
//! Run with: cargo run --example feature_allocations -p fluxbench --release

use fluxbench::prelude::*;
use fluxbench::{TrackingAllocator, current_allocation, reset_allocation_counter};
use fluxbench::{bench, compare, verify};
use std::hint::black_box;

// Install the tracking allocator for the entire process.
#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

// ---------------------------------------------------------------------------
// Benchmark: allocation-heavy path
// ---------------------------------------------------------------------------

/// Each iteration allocates three fresh Vec buffers on the heap.
#[bench(id = "alloc_heavy", group = "memory")]
fn alloc_heavy(b: &mut Bencher) {
    b.iter(|| {
        reset_allocation_counter();
        let v1 = vec![0u8; 1024];
        let v2 = vec![0u8; 2048];
        let v3 = vec![0u8; 512];
        let result = v1.len() + v2.len() + v3.len();
        let (bytes, count) = current_allocation();
        black_box((result, bytes, count))
    });
}

// ---------------------------------------------------------------------------
// Benchmark: pre-allocated (zero-alloc hot path)
// ---------------------------------------------------------------------------

/// Re-uses a pre-allocated buffer — hot path should not allocate.
#[bench(id = "alloc_zero", group = "memory")]
fn alloc_zero(b: &mut Bencher) {
    let mut buf = vec![0u8; 4096];
    b.iter(|| {
        reset_allocation_counter();
        // Write into existing buffer — no heap allocation
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = (i & 0xFF) as u8;
        }
        let (bytes, count) = current_allocation();
        black_box((buf.len(), bytes, count))
    });
}

// ---------------------------------------------------------------------------
// Benchmark: String building with vs without capacity
// ---------------------------------------------------------------------------

/// Build a string without pre-allocation — triggers multiple reallocs.
#[bench(id = "string_no_cap", group = "memory")]
fn string_no_cap(b: &mut Bencher) {
    b.iter(|| {
        reset_allocation_counter();
        let mut s = String::new();
        for i in 0u32..100 {
            s.push_str(&i.to_string());
        }
        let (_bytes, count) = current_allocation();
        black_box((s.len(), count))
    });
}

/// Build a string with pre-allocated capacity — fewer reallocs.
#[bench(id = "string_with_cap", group = "memory")]
fn string_with_cap(b: &mut Bencher) {
    b.iter(|| {
        reset_allocation_counter();
        let mut s = String::with_capacity(512);
        for i in 0u32..100 {
            s.push_str(&i.to_string());
        }
        let (_bytes, count) = current_allocation();
        black_box((s.len(), count))
    });
}

// ---------------------------------------------------------------------------
// Comparisons
// ---------------------------------------------------------------------------

#[compare(
    id = "alloc_strategy",
    title = "Allocation vs Pre-allocation",
    benchmarks = ["alloc_heavy", "alloc_zero"],
    baseline = "alloc_heavy",
    metric = "mean"
)]
#[allow(dead_code)]
struct AllocComparison;

#[compare(
    id = "string_alloc",
    title = "String Capacity Pre-allocation",
    benchmarks = ["string_no_cap", "string_with_cap"],
    baseline = "string_no_cap",
    metric = "mean"
)]
#[allow(dead_code)]
struct StringAllocComparison;

// ---------------------------------------------------------------------------
// Verification — ensure pre-allocated path is faster
// ---------------------------------------------------------------------------

#[verify(expr = "alloc_zero < alloc_heavy", severity = "critical")]
#[allow(dead_code)]
struct PreallocFaster;

#[verify(expr = "string_with_cap < string_no_cap", severity = "warning")]
#[allow(dead_code)]
struct CapacityFaster;

fn main() {
    if let Err(e) = fluxbench::run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
