//! Process-isolation fixtures for `examples/tests/isolation.rs`.
//!
//! Each benchmark ends its worker in a different way.
//! The integration test runs this binary and checks each benchmark's report entry.
//!
//! Run by hand:
//!   cargo run --example test_isolation -p fluxbench-examples -- \
//!     --warmup 10ms --measurement 50ms --worker-timeout 3 --format json

use fluxbench::bench;
use fluxbench::prelude::*;
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Passes quickly.
#[bench(group = "isolation")]
fn ok_fast(b: &mut Bencher) {
    b.iter(|| black_box(21u64).wrapping_mul(2));
}

/// Panics on the first call.
#[bench(group = "isolation")]
fn panics(b: &mut Bencher) {
    b.iter(|| -> u64 { panic!("isolation fixture panic") });
}

/// Aborts the worker: SIGABRT on Unix.
#[bench(group = "isolation")]
fn aborts(b: &mut Bencher) {
    b.iter(|| -> u64 { std::process::abort() });
}

/// Kills the worker with SIGSEGV.
///
/// The default action is restored first.
/// The Rust runtime's stack-overflow handler would otherwise swallow one raised SIGSEGV.
#[cfg(unix)]
#[bench(group = "isolation")]
fn segv(b: &mut Bencher) {
    b.iter(|| {
        // SAFETY: restoring the default action and raising a signal have no memory effects.
        unsafe {
            libc::signal(libc::SIGSEGV, libc::SIG_DFL);
            libc::raise(libc::SIGSEGV);
        }
    });
}

/// Records some samples, then hangs past any short worker timeout.
#[bench(group = "isolation")]
fn sleeps_past_timeout(b: &mut Bencher) {
    static CALLS: AtomicU64 = AtomicU64::new(0);
    b.iter(|| {
        if CALLS.fetch_add(1, Ordering::Relaxed) < 30 {
            std::thread::sleep(Duration::from_millis(1));
        } else {
            std::thread::sleep(Duration::from_secs(600));
        }
    });
}

/// Writes to stdout from the body. The supervisor's JSON report must stay intact.
#[bench(group = "isolation")]
fn prints_to_stdout(b: &mut Bencher) {
    b.iter(|| {
        println!("}}{{ noise from a benchmark body");
        black_box(1u64)
    });
}

/// Runs after every failing fixture. It proves the suite keeps going.
#[bench(
    group = "isolation",
    depends_on = ["aborts", "panics", "segv", "sleeps_past_timeout", "prints_to_stdout"]
)]
fn ok_after(b: &mut Bencher) {
    b.iter(|| black_box(99u64).wrapping_add(1));
}

fn main() {
    if let Err(e) = fluxbench::run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
