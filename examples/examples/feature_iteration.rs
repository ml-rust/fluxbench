//! Iteration Methods — All ways to drive a benchmark loop
//!
//! - `iter`: pure computation on input built once.
//! - `iter_with_setup` / `iter_batched`: fresh input per call, taken by value.
//! - `iter_batched_ref`: fresh input per call, borrowed mutably.
//! - `iter_batched_shared`: one shared input per batch of calls.
//! - `iter_custom`: the body times its own calls.
//!
//! Run with: cargo run -p fluxbench-examples --example feature_iteration --release

use fluxbench::bench;
use fluxbench::prelude::*;
use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;

// ---------------------------------------------------------------------------
// b.iter() — Simple closure, measured end-to-end
// ---------------------------------------------------------------------------
// Best for: pure computation where the input can be prepared once and reused.

/// Summing a pre-built vector — no per-iteration allocation.
#[bench(group = "iter")]
fn iter_vector_sum(b: &mut Bencher) {
    let data: Vec<i64> = (0..10_000).collect();
    b.iter(|| black_box(data.iter().sum::<i64>()));
}

/// Sorting inside iter() means we re-randomize outside the loop,
/// but the *same* shuffled vec is sorted every iteration — after the
/// first, it's already sorted. This is a common mistake; see
/// `setup_sort_fresh` below for the correct approach.
#[bench(group = "iter")]
fn iter_sort_already_sorted(b: &mut Bencher) {
    let mut data: Vec<u32> = (0..1_000).rev().collect();
    b.iter(|| {
        data.sort();
        black_box(data.len())
    });
}

// ---------------------------------------------------------------------------
// b.iter_with_setup() — Per-iteration setup (not measured)
// ---------------------------------------------------------------------------
// Best for: benchmarks that consume or mutate their input. The setup closure
// runs before each iteration but its time is excluded from the measurement.

/// Each iteration gets a freshly-shuffled vector, so we always measure
/// worst-case sort — not the already-sorted fast path.
#[bench(group = "setup")]
fn setup_sort_fresh(b: &mut Bencher) {
    use rand::prelude::*;
    b.iter_with_setup(
        || {
            let mut rng = rand::rng();
            let mut v: Vec<u32> = (0..1_000).collect();
            v.shuffle(&mut rng);
            v
        },
        |mut data| {
            data.sort();
            black_box(data)
        },
    );
}

/// HashMap benchmarks often need a fresh map per iteration because
/// insertion order affects bucket layout and performance.
#[bench(group = "setup")]
fn setup_hashmap_drain(b: &mut Bencher) {
    b.iter_with_setup(
        || {
            let mut map = HashMap::with_capacity(256);
            for i in 0u64..256 {
                map.insert(i, i.wrapping_mul(0x517cc1b727220a95));
            }
            map
        },
        |mut map| {
            // Measure drain — consumes the map
            let sum: u64 = map.drain().map(|(_, v)| v).sum();
            black_box(sum)
        },
    );
}

// ---------------------------------------------------------------------------
// b.iter_batched() — Fresh input per call, with a chosen batch size
// ---------------------------------------------------------------------------
// Setup runs once per call, untimed. `BatchSize` sets how many inputs one
// timed batch holds: more inputs spread the timer cost, fewer save memory.

/// Small inputs: up to 1024 per batch. Same as `iter_with_setup`.
#[bench(group = "batched")]
fn batched_small_input(b: &mut Bencher) {
    b.iter_batched(
        || (0..256u32).rev().collect::<Vec<_>>(),
        |mut v| {
            v.sort_unstable();
            v
        },
        BatchSize::SmallInput,
    );
}

/// Large inputs: up to 16 per batch keeps at most 16 MiB of buffers alive.
#[bench(group = "batched")]
fn batched_large_input(b: &mut Bencher) {
    b.iter_batched(
        || vec![1u8; 1024 * 1024],
        |buf| buf.iter().map(|&x| x as u64).sum::<u64>(),
        BatchSize::LargeInput,
    );
}

/// One input per batch: each call gets its own timer.
#[bench(group = "batched")]
fn batched_per_iteration(b: &mut Bencher) {
    b.iter_batched(
        || String::from("hello world"),
        |s| s.to_uppercase(),
        BatchSize::PerIteration,
    );
}

/// Fixed batch layout: 8 batches per sample, or 32 inputs per batch.
#[bench(group = "batched")]
fn batched_fixed_layout(b: &mut Bencher) {
    b.iter_batched(
        || vec![3u64; 64],
        |v| v.into_iter().product::<u64>(),
        BatchSize::NumBatches(8),
    );
}

#[bench(group = "batched")]
fn batched_fixed_size(b: &mut Bencher) {
    b.iter_batched(
        || vec![3u64; 64],
        |v| v.into_iter().max(),
        BatchSize::NumIterations(32),
    );
}

// ---------------------------------------------------------------------------
// b.iter_batched_ref() — Fresh input per call, borrowed mutably
// ---------------------------------------------------------------------------
// Best for: routines that mutate their input in place. The input and the
// output both drop after the timer stops, so drop cost stays out of the time.

/// Clear a filled map in place. The emptied map drops outside the timer.
#[bench(group = "batched")]
fn batched_ref_clear_map(b: &mut Bencher) {
    b.iter_batched_ref(
        || (0u64..512).map(|i| (i, i)).collect::<HashMap<_, _>>(),
        |map| {
            map.clear();
            map.capacity()
        },
        BatchSize::SmallInput,
    );
}

// ---------------------------------------------------------------------------
// b.iter_batched_shared() — Amortized setup over N iterations
// ---------------------------------------------------------------------------
// Best for: when setup is expensive relative to the measured work. The setup
// runs once per *batch* of `batch_size` iterations, and every call in the
// batch borrows the same input.

/// Allocating a large buffer is expensive; amortize it over 64 iterations.
#[bench(group = "shared")]
fn shared_buffer_reuse(b: &mut Bencher) {
    b.iter_batched_shared(
        64,
        || vec![0u8; 64 * 1024], // 64 KiB buffer, allocated once per batch
        |buf| {
            // Simulate a hot-path read over the shared buffer
            let sum: u8 = buf.iter().fold(0u8, |a, &x| a.wrapping_add(x));
            black_box(sum)
        },
    );
}

/// Regex compilation is slow; compile once, match 128 times per batch.
#[bench(group = "shared")]
fn shared_regex_match(b: &mut Bencher) {
    let pattern = "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$";
    let inputs = [
        "user@example.com",
        "bad@@email",
        "hello.world+tag@sub.domain.org",
        "nope",
    ];
    b.iter_batched_shared(
        128,
        || regex::Regex::new(pattern).unwrap(),
        |re| {
            let hits: usize = inputs.iter().filter(|s| re.is_match(s)).count();
            black_box(hits)
        },
    );
}

// ---------------------------------------------------------------------------
// b.iter_custom() — The body times its own calls
// ---------------------------------------------------------------------------
// Best for: work the engine cannot time directly, such as a region inside a
// larger loop. The closure runs `iters` calls and returns their total time.

/// Time only the lookups, not the per-batch map build.
#[bench(group = "custom")]
fn custom_timed_lookups(b: &mut Bencher) {
    b.iter_custom(|iters| {
        let map: HashMap<u64, u64> = (0..1_024).map(|i| (i, i * 2)).collect();
        let start = Instant::now();
        for i in 0..iters {
            black_box(map.get(&(i % 1_024)));
        }
        start.elapsed()
    });
}

fn main() {
    if let Err(e) = fluxbench::run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
