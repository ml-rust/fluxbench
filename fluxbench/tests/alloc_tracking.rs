//! Allocation tracking through the measurement engine.
//!
//! This binary installs `TrackingAllocator` as its global allocator.
//! The counters are process-wide, so this file holds a single test.

use fluxbench::{Bencher, MeasureConfig, TrackingAllocator};

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

#[test]
fn allocation_per_iteration_mean_is_exact() {
    let mut bencher = Bencher::with_config(MeasureConfig {
        warmup_ns: 0,
        measurement_ns: 10_000_000,
        sample_count: 10,
        fixed_iters_per_sample: Some(64),
        min_iterations: None,
        max_iterations: None,
        track_allocations: true,
    });

    // One allocation of 100 * 8 bytes per call.
    bencher.iter(|| Vec::<u64>::with_capacity(std::hint::black_box(100)));

    let result = bencher.finish().expect("measurement succeeds");
    assert_eq!(result.samples.len(), 10);
    for sample in &result.samples {
        assert_eq!(sample.iterations, 64);
        assert_eq!(sample.alloc_bytes, 800.0, "sample: {sample:?}");
        assert_eq!(sample.alloc_count, 1.0, "sample: {sample:?}");
    }
}
