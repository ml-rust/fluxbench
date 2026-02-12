//! Verification & Synthetic Metrics
//!
//! `#[verify]` asserts relationships between benchmark results.
//! `#[synthetic]` computes derived metrics (throughput, speedup, cost).
//! Together they turn a benchmark suite into a regression gate.
//!
//! Run with: cargo run --example feature_verify -p fluxbench --release

use fluxbench::prelude::*;
use fluxbench::{bench, synthetic, verify};
use std::hint::black_box;

// ---------------------------------------------------------------------------
// Benchmarks under test
// ---------------------------------------------------------------------------

/// Naive O(2^n) recursive Fibonacci.
#[bench(id = "fib_naive", group = "compute")]
fn fib_naive(b: &mut Bencher) {
    fn fib(n: u32) -> u64 {
        if n <= 1 {
            n as u64
        } else {
            fib(n - 1) + fib(n - 2)
        }
    }
    b.iter(|| black_box(fib(20)));
}

/// O(n) iterative Fibonacci.
#[bench(id = "fib_iter", group = "compute")]
fn fib_iter(b: &mut Bencher) {
    fn fib(n: u32) -> u64 {
        let (mut a, mut b) = (0u64, 1u64);
        for _ in 0..n {
            let t = a;
            a = b;
            b = b.wrapping_add(t);
        }
        a
    }
    b.iter(|| black_box(fib(20)));
}

/// Process 10 000 items — baseline for throughput metrics.
#[bench(id = "process_10k", group = "throughput")]
fn process_10k(b: &mut Bencher) {
    let items: Vec<u64> = (0..10_000).collect();
    b.iter(|| {
        let s: u64 = items
            .iter()
            .map(|x| x.wrapping_mul(6364136223846793005))
            .sum();
        black_box(s)
    });
}

/// Copy 1 MiB — baseline for bandwidth metric.
#[bench(id = "memcpy_1m", group = "throughput")]
fn memcpy_1m(b: &mut Bencher) {
    let src = vec![0xABu8; 1024 * 1024];
    let mut dst = vec![0u8; 1024 * 1024];
    b.iter(|| {
        dst.copy_from_slice(&src);
        black_box(dst.len())
    });
}

// ---------------------------------------------------------------------------
// Verifications — severity = "critical" fails CI, "warning" prints, "info" logs
// ---------------------------------------------------------------------------

/// Iterative must be faster than naive — absolute ordering.
#[verify(expr = "fib_iter < fib_naive", severity = "critical")]
#[allow(dead_code)]
struct IterFasterThanNaive;

/// Iterative must be at least 100x faster.
#[verify(expr = "fib_naive / fib_iter > 100", severity = "warning")]
#[allow(dead_code)]
struct IterMuchFaster;

/// p99 latency of iterative Fibonacci stays under 1 ms (1 000 000 ns).
#[verify(expr = "fib_iter_p99 < 1000000", severity = "info")]
#[allow(dead_code)]
struct IterP99Under1ms;

/// The computed speedup synthetic must exceed 50x.
#[verify(expr = "fib_speedup > 50", severity = "warning")]
#[allow(dead_code)]
struct SpeedupSignificant;

/// Throughput must exceed 1 billion items/s.
#[verify(expr = "items_per_sec > 1000000000", severity = "warning")]
#[allow(dead_code)]
struct ThroughputCheck;

// ---------------------------------------------------------------------------
// Synthetic metrics — derived from benchmark means (in nanoseconds)
// ---------------------------------------------------------------------------

/// How many times faster is iterative vs naive?
#[synthetic(id = "fib_speedup", formula = "fib_naive / fib_iter", unit = "x")]
#[allow(dead_code)]
struct FibSpeedup;

/// Items processed per second (mean is in ns, 10 000 items per call).
#[synthetic(
    id = "items_per_sec",
    formula = "10000 / process_10k * 1000000000",
    unit = "items/s"
)]
#[allow(dead_code)]
struct ItemsPerSec;

/// Nanoseconds per item.
#[synthetic(id = "ns_per_item", formula = "process_10k / 10000", unit = "ns/item")]
#[allow(dead_code)]
struct NsPerItem;

/// Memory bandwidth in MiB/s (1 MiB copied, mean in ns).
#[synthetic(
    id = "memcpy_bandwidth",
    formula = "1048576 / memcpy_1m * 1000000000",
    unit = "B/s"
)]
#[allow(dead_code)]
struct MemcpyBandwidth;

fn main() {
    if let Err(e) = fluxbench::run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
