//! Baseline regression fixtures for `examples/tests/baseline.rs`.
//!
//! `FLUX_TEST_SLOWDOWN` adds that many spin iterations to `critical_spin`.
//! Unset or unparsable means 0. Isolated workers inherit the variable.
//!
//! Run by hand:
//!   cargo run --example test_baseline -p fluxbench-examples -- \
//!     --warmup 10ms --measurement 100ms --save-baseline /tmp/base.json
//!   FLUX_TEST_SLOWDOWN=200000 cargo run --example test_baseline -p fluxbench-examples -- \
//!     --warmup 10ms --measurement 100ms --baseline /tmp/base.json

use fluxbench::bench;
use fluxbench::prelude::*;
use std::hint::black_box;
use std::sync::OnceLock;

/// Spin iterations every run performs
const BASE_SPIN: u64 = 1_000;

/// Extra spin iterations from `FLUX_TEST_SLOWDOWN`
fn slowdown() -> u64 {
    static EXTRA: OnceLock<u64> = OnceLock::new();
    *EXTRA.get_or_init(|| {
        std::env::var("FLUX_TEST_SLOWDOWN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

fn spin(iterations: u64) -> u64 {
    let mut acc = 0u64;
    for i in 0..iterations {
        acc = black_box(acc.wrapping_add(i));
    }
    acc
}

/// Critical benchmark. Its regression fails CI.
/// The wide threshold keeps run-to-run noise from failing an unchanged run.
#[bench(group = "baseline", severity = "critical", threshold = 50.0)]
fn critical_spin(b: &mut Bencher) {
    let iterations = BASE_SPIN + slowdown();
    b.iter(|| spin(black_box(iterations)));
}

/// Warning benchmark. It never changes, and a regression here never fails CI.
#[bench(group = "baseline", severity = "warning")]
fn stable_spin(b: &mut Bencher) {
    b.iter(|| spin(black_box(BASE_SPIN)));
}

fn main() {
    if let Err(e) = fluxbench::run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
