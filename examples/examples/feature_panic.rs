//! Crash Isolation & Failure Handling
//!
//! FluxBench runs each benchmark in a separate child process by default
//! ("Fail-Late" architecture). A panicking benchmark is reported as crashed
//! but does NOT take down the suite — subsequent benchmarks continue.
//!
//! Run with:
//!   cargo run --example feature_panic -p fluxbench --release            # Isolated (default)
//!   cargo run --example feature_panic -p fluxbench --release -- --isolated=false  # In-process (will abort on panic)
//!
//! Expected output (isolated mode):
//!   - `panics_after_warmup` crashes, reported as CRASH
//!   - `panics_immediately` crashes, reported as CRASH
//!   - All other benchmarks pass normally
//!   - Suite exits with non-zero status (crashed benchmarks present)

use fluxbench::bench;
use fluxbench::prelude::*;
use std::hint::black_box;

// ---------------------------------------------------------------------------
// Normal benchmarks — should run fine even after crashes
// ---------------------------------------------------------------------------

/// Runs before any crash — baseline to show the suite starts healthy.
#[bench(group = "stable")]
fn before_crash(b: &mut Bencher) {
    b.iter(|| black_box(42u64.wrapping_mul(17)));
}

/// Runs after crashes — proves isolation kept the suite alive.
#[bench(group = "stable")]
fn after_crash(b: &mut Bencher) {
    b.iter(|| black_box(99u64.wrapping_add(1)));
}

/// Another post-crash benchmark to confirm the suite is still functional.
#[bench(group = "stable")]
fn also_after_crash(b: &mut Bencher) {
    let data: Vec<u64> = (0..100).collect();
    b.iter(|| black_box(data.iter().sum::<u64>()));
}

// ---------------------------------------------------------------------------
// Crashing benchmarks — demonstrate isolation
// ---------------------------------------------------------------------------

/// Panics after 5 iterations — the warmup phase succeeds, then
/// the measurement phase hits the panic.  In isolated mode, only this
/// child process dies; the supervisor marks it as CRASH and moves on.
#[bench(group = "crash")]
fn panics_after_warmup(b: &mut Bencher) {
    static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    b.iter(|| {
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if n >= 5 {
            panic!("Intentional panic after {n} iterations!");
        }
        black_box(n)
    });
}

/// Panics on the very first iteration — tests that FluxBench handles
/// immediate failures gracefully (no partial results, clean error report).
#[bench(group = "crash")]
fn panics_immediately(b: &mut Bencher) {
    b.iter(|| -> u64 {
        panic!("Immediate panic — no iterations complete!");
    });
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

/// Stack overflow via deep recursion — tests that the child process
/// segfault / stack overflow is caught as a crash, not a hang.
#[bench(group = "crash")]
fn stack_overflow(b: &mut Bencher) {
    #[allow(unconditional_recursion)]
    fn recurse() -> u64 {
        recurse() + black_box(1)
    }
    b.iter(|| black_box(recurse()));
}

/// Benchmark that succeeds but is very fast — verifies that sub-nanosecond
/// benchmarks don't produce NaN or negative timing.
#[bench(group = "edge")]
fn near_zero_time(b: &mut Bencher) {
    b.iter(|| black_box(1u64));
}

/// Benchmark with intentionally heavy allocation per iteration —
/// should not OOM thanks to per-sample cleanup.
#[bench(group = "edge")]
fn heavy_alloc_per_iter(b: &mut Bencher) {
    b.iter(|| {
        let v = vec![0u8; 1024 * 1024]; // 1 MiB per iteration
        black_box(v.len())
    });
}

fn main() {
    if let Err(e) = fluxbench::run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
