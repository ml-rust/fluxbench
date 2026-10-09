//! Measurement engine: warmup, iterations per sample, and sample math.
//!
//! Each `iter*` method supplies a batch function.
//! A batch runs the routine `n` times and returns raw totals.
//! The engine turns totals into per-iteration means.

use super::config::MeasureConfig;
use super::state::Bencher;
use crate::allocator::{current_allocation, reset_allocation_counter};
use crate::measure::{cycles_overhead, timer_overhead_ns};
use fluxbench_ipc::Sample;
use std::time::{Duration, Instant};

/// Lower bound on the hard deadline.
const MIN_DEADLINE: Duration = Duration::from_secs(1);

/// Lower bound on the warmup estimate per call, in nanoseconds.
///
/// It keeps the iterations-per-sample division finite for a zero estimate.
const MIN_ESTIMATE_NS: f64 = 0.01;

/// Raw totals for one batch of `n` routine calls.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct SampleRaw {
    /// Total timed nanoseconds
    pub ns: u64,
    /// Total timed cycles
    pub cycles: u64,
    /// Total bytes allocated inside timed regions
    pub alloc_b: u64,
    /// Total allocations inside timed regions
    pub alloc_c: u64,
    /// Timed regions in the batch. Each region costs one timer overhead.
    pub timers: u64,
}

impl SampleRaw {
    /// Add one timed region's totals.
    pub fn add(&mut self, ns: u64, cycles: u64, alloc: (u64, u64)) {
        self.ns = self.ns.saturating_add(ns);
        self.cycles = self.cycles.saturating_add(cycles);
        self.alloc_b = self.alloc_b.saturating_add(alloc.0);
        self.alloc_c = self.alloc_c.saturating_add(alloc.1);
        self.timers += 1;
    }

    /// Per-iteration means for `n` calls, with timer overhead removed per timed region.
    pub fn to_sample(self, n: u64, overhead_ns: f64, overhead_cycles: f64) -> Sample {
        let n_f = n.max(1) as f64;
        let timers = self.timers as f64;
        let net = |total: u64, overhead: f64| (total as f64 - overhead * timers).max(0.0) / n_f;
        Sample::new(
            net(self.ns, overhead_ns),
            net(self.cycles, overhead_cycles),
            self.alloc_b as f64 / n_f,
            self.alloc_c as f64 / n_f,
            n,
        )
    }
}

/// Iterations per sample for a warmup estimate of `est_ns` per call.
///
/// Order: time budget, then raise to the minimum, then cap at the maximum.
/// A maximum below the minimum rises to the minimum.
fn iters_per_sample(config: &MeasureConfig, est_ns: f64) -> u64 {
    if let Some(fixed) = config.fixed_iters_per_sample {
        return fixed.max(1);
    }
    let samples = config.sample_count.max(1) as u64;
    let per_sample_ns = config.measurement_ns as f64 / samples as f64;
    let ratio = (per_sample_ns / est_ns.max(MIN_ESTIMATE_NS)).ceil();
    // `as` saturates for values above u64::MAX.
    let mut ips = if ratio.is_finite() && ratio >= 1.0 {
        ratio as u64
    } else {
        1
    };

    let min_ips = config.min_iterations.map(|m| m.div_ceil(samples));
    if let Some(min) = min_ips {
        ips = ips.max(min);
    }
    if let Some(max) = config.max_iterations {
        let max_ips = (max / samples).max(1).max(min_ips.unwrap_or(0));
        ips = ips.min(max_ips);
    }
    ips.max(1)
}

/// Hard deadline for the measurement phase: `max(3 * measurement_ns, 1s)`.
fn deadline(config: &MeasureConfig) -> Duration {
    Duration::from_nanos(config.measurement_ns.saturating_mul(3)).max(MIN_DEADLINE)
}

impl Bencher {
    /// Reset the allocation counters when tracking is on.
    #[inline(always)]
    pub(super) fn alloc_reset(&self) {
        if self.config.track_allocations {
            reset_allocation_counter();
        }
    }

    /// Read the allocation counters when tracking is on. Returns zeros otherwise.
    #[inline(always)]
    pub(super) fn alloc_read(&self) -> (u64, u64) {
        if self.config.track_allocations {
            current_allocation()
        } else {
            (0, 0)
        }
    }

    /// Run warmup and measurement through `batch`.
    ///
    /// `batch(self, n)` runs the routine `n` times and returns the totals.
    pub(super) fn run_loop<B>(&mut self, mut batch: B)
    where
        B: FnMut(&mut Self, u64) -> SampleRaw,
    {
        let overhead_ns = timer_overhead_ns();
        let overhead_cycles = cycles_overhead();

        if self.config.fixed_iters_per_sample.is_none() {
            let est_ns = self.warmup(&mut batch);
            self.iters_per_sample = iters_per_sample(&self.config, est_ns);
        } else {
            self.iters_per_sample = iters_per_sample(&self.config, 0.0);
        }

        let ips = self.iters_per_sample;
        let limit = deadline(&self.config);
        let start = Instant::now();
        for _ in 0..self.config.sample_count {
            if !self.samples.is_empty() && start.elapsed() > limit {
                self.truncated = true;
                break;
            }
            let raw = batch(self, ips);
            self.samples
                .push(raw.to_sample(ips, overhead_ns, overhead_cycles));
        }
    }

    /// Run doubling batches until `warmup_ns` of wall time elapses.
    ///
    /// Returns wall-clock nanoseconds per call, untimed setup included.
    /// Sample sizing uses this value, so setup-heavy bodies still fit the measurement budget.
    /// Only running sums persist, so memory stays constant.
    /// A zero budget runs one batch of one call.
    fn warmup<B>(&mut self, batch: &mut B) -> f64
    where
        B: FnMut(&mut Self, u64) -> SampleRaw,
    {
        let budget = Duration::from_nanos(self.config.warmup_ns);
        let start = Instant::now();
        let mut n = 1u64;
        let mut sum_wall_ns = 0u64;
        let mut sum_iters = 0u64;
        loop {
            let batch_start = Instant::now();
            let raw = batch(self, n);
            // Wall time covers setup and the timed region. It is never below the timed total.
            let wall_ns = u64::try_from(batch_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
            sum_wall_ns = sum_wall_ns.saturating_add(wall_ns.max(raw.ns));
            sum_iters = sum_iters.saturating_add(n);
            if start.elapsed() >= budget {
                break;
            }
            n = n.saturating_mul(2);
        }
        self.warmup_iterations = sum_iters;
        sum_wall_ns as f64 / sum_iters.max(1) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(sample_count: usize, measurement_ns: u64) -> MeasureConfig {
        MeasureConfig {
            warmup_ns: 0,
            measurement_ns,
            sample_count,
            fixed_iters_per_sample: None,
            min_iterations: None,
            max_iterations: None,
            track_allocations: false,
        }
    }

    #[test]
    fn min_max_iterations_adjust_ips() {
        // 10 samples of 1000 ns at 1 ns per call: 1000 calls per sample.
        let base = config(10, 10_000);
        assert_eq!(iters_per_sample(&base, 1.0), 1000);
        // Fractional ratios round up.
        assert_eq!(iters_per_sample(&base, 3.0), 334);

        let min = MeasureConfig {
            min_iterations: Some(50_000),
            ..base
        };
        assert_eq!(iters_per_sample(&min, 1.0), 5000);
        // Minimum per sample rounds up.
        let min_odd = MeasureConfig {
            min_iterations: Some(50_001),
            ..base
        };
        assert_eq!(iters_per_sample(&min_odd, 1.0), 5001);

        let max = MeasureConfig {
            max_iterations: Some(2_000),
            ..base
        };
        assert_eq!(iters_per_sample(&max, 1.0), 200);
        // A maximum below one call per sample still allows one.
        let tiny_max = MeasureConfig {
            max_iterations: Some(3),
            ..base
        };
        assert_eq!(iters_per_sample(&tiny_max, 1.0), 1);

        // A maximum below the minimum rises to the minimum.
        let crossed = MeasureConfig {
            min_iterations: Some(200),
            max_iterations: Some(50),
            ..base
        };
        assert_eq!(iters_per_sample(&crossed, 1_000_000.0), 20);
    }

    #[test]
    fn ips_floor_is_one_and_zero_estimate_is_finite() {
        let base = config(10, 10_000);
        assert_eq!(iters_per_sample(&base, 1e12), 1);
        let ips = iters_per_sample(&base, 0.0);
        assert!(ips >= 1);
        assert!(ips < u64::MAX);
    }

    #[test]
    fn fixed_mode_ignores_limits() {
        let fixed = MeasureConfig {
            fixed_iters_per_sample: Some(7),
            min_iterations: Some(1_000),
            max_iterations: Some(10),
            ..config(10, 10_000)
        };
        assert_eq!(iters_per_sample(&fixed, 1.0), 7);
    }

    #[test]
    fn deadline_has_one_second_floor() {
        assert_eq!(deadline(&config(10, 1)), Duration::from_secs(1));
        assert_eq!(deadline(&config(10, 2_000_000_000)), Duration::from_secs(6));
    }

    #[test]
    fn sample_math_removes_overhead_per_timer() {
        let raw = SampleRaw {
            ns: 1_020,
            cycles: 4_100,
            alloc_b: 800,
            alloc_c: 10,
            timers: 2,
        };
        let sample = raw.to_sample(10, 10.0, 50.0);
        assert_eq!(sample.duration_nanos, 100.0);
        assert_eq!(sample.cpu_cycles, 400.0);
        assert_eq!(sample.alloc_bytes, 80.0);
        assert_eq!(sample.alloc_count, 1.0);
        assert_eq!(sample.iterations, 10);
    }

    #[test]
    fn sample_math_keeps_subnanosecond_values_and_clamps_at_zero() {
        let raw = SampleRaw {
            ns: 35,
            timers: 1,
            ..SampleRaw::default()
        };
        assert_eq!(raw.to_sample(100, 20.0, 0.0).duration_nanos, 0.15);
        assert_eq!(raw.to_sample(100, 50.0, 0.0).duration_nanos, 0.0);
    }
}
