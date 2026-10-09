//! Throughput — Report bytes or elements per second
//!
//! Set the work one call processes. Reports then show a rate next to the time:
//! bytes in binary units (KiB/s, MiB/s, GiB/s) or elements per second.
//!
//! Three ways to set it, highest precedence first:
//! 1. `b.throughput(...)` in the body
//! 2. `#[bench(throughput_bytes = N)]` or `#[bench(throughput_elements = N)]`
//! 3. `#[group(throughput_bytes = N)]` on the benchmark's group or a parent group
//!
//! `#[verify]` can name `<id>_throughput_bytes` and `<id>_throughput_elements`.
//!
//! Run with: cargo run -p fluxbench-examples --example feature_throughput --release

use fluxbench::prelude::*;
use fluxbench::{bench, group, verify};
use std::hint::black_box;

// ---------------------------------------------------------------------------
// Attribute: a fixed amount of work per call
// ---------------------------------------------------------------------------

/// Sum 64 KiB per call. Reports show bytes per second.
#[bench(group = "memory", throughput_bytes = 65536)]
fn sum_64k(b: &mut Bencher) {
    let data = vec![1u8; 64 * 1024];
    b.iter(|| black_box(&data).iter().map(|&x| x as u64).sum::<u64>());
}

/// Parse 1000 integers per call. Reports show elements per second.
#[bench(group = "parsing", throughput_elements = 1000)]
fn parse_ints(b: &mut Bencher) {
    let lines: Vec<String> = (0..1000).map(|i| i.to_string()).collect();
    b.iter(|| {
        black_box(&lines)
            .iter()
            .filter_map(|l| l.parse::<u32>().ok())
            .sum::<u32>()
    });
}

// ---------------------------------------------------------------------------
// Body: work that depends on the input
// ---------------------------------------------------------------------------

/// The body knows the size only after it builds the input.
#[bench(group = "memory")]
fn copy_dynamic(b: &mut Bencher) {
    let src: Vec<u8> = (0..300_000u32).map(|i| i as u8).collect();
    b.throughput(Throughput::Bytes(src.len() as u64));
    b.iter(|| black_box(src.clone()));
}

/// `b.throughput` overrides the attribute: this reports 1 MiB per call, not 1 byte.
#[bench(group = "memory", throughput_bytes = 1)]
fn fill_override(b: &mut Bencher) {
    b.throughput(Throughput::Bytes(1024 * 1024));
    b.iter_batched_ref(
        || vec![0u8; 1024 * 1024],
        |buf| buf.fill(7),
        BatchSize::LargeInput,
    );
}

// ---------------------------------------------------------------------------
// Group: one setting for every benchmark below it
// ---------------------------------------------------------------------------

/// Every benchmark in `checksum` hashes 4 KiB per call and runs 30 samples.
#[group(
    id = "checksum",
    description = "4 KiB checksums",
    throughput_bytes = 4096,
    sample_count = 30
)]
#[allow(dead_code)]
struct Checksum;

#[bench(group = "checksum")]
fn checksum_add(b: &mut Bencher) {
    let block = vec![0xA5u8; 4096];
    b.iter(|| {
        black_box(&block)
            .iter()
            .fold(0u32, |acc, &x| acc.wrapping_add(x as u32))
    });
}

#[bench(group = "checksum")]
fn checksum_fletcher(b: &mut Bencher) {
    let block = vec![0xA5u8; 4096];
    b.iter(|| {
        let (mut a, mut c) = (0u32, 0u32);
        for &x in black_box(&block) {
            a = (a + x as u32) % 65_535;
            c = (c + a) % 65_535;
        }
        (c << 16) | a
    });
}

// ---------------------------------------------------------------------------
// Verify: assert a minimum rate
// ---------------------------------------------------------------------------

/// Summing memory runs above 100 MiB/s on any machine this example targets.
#[verify(expr = "sum_64k_throughput_bytes > 104857600", severity = "warning")]
#[allow(dead_code)]
struct SumAbove100MiB;

fn main() {
    if let Err(e) = fluxbench::run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
