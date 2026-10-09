//! Measurement configs shared by the bencher unit tests.

use super::config::MeasureConfig;

/// Short time-based config: 2 ms warmup, 10 ms measurement.
pub(super) fn quick(sample_count: usize) -> MeasureConfig {
    MeasureConfig {
        warmup_ns: 2_000_000,
        measurement_ns: 10_000_000,
        sample_count,
        fixed_iters_per_sample: None,
        min_iterations: None,
        max_iterations: None,
        track_allocations: false,
    }
}

/// Fixed-mode config: `samples` samples of `ips` calls, no warmup.
pub(super) fn fixed(samples: usize, ips: u64) -> MeasureConfig {
    MeasureConfig {
        fixed_iters_per_sample: Some(ips),
        ..quick(samples)
    }
}
