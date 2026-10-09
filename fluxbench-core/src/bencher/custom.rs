//! `iter_custom`: the body times its own calls.

use super::engine::SampleRaw;
use super::state::Bencher;
use std::time::Duration;

impl Bencher {
    /// Measure a routine that times itself.
    ///
    /// `routine(iters)` runs `iters` calls and returns the time they took.
    /// The engine runs warmup, sizes the samples, and calls it once per batch.
    /// The returned time is the sample total. No engine timer overhead is removed from it.
    /// Allocation counters cover the whole `routine` call. Cycle counts are 0.
    #[inline]
    pub fn iter_custom<F>(&mut self, mut routine: F)
    where
        F: FnMut(u64) -> Duration,
    {
        if !self.begin_iter() {
            return;
        }
        self.run_loop(|b, n| {
            b.alloc_reset();
            let elapsed = routine(n);
            let (alloc_b, alloc_c) = b.alloc_read();
            SampleRaw {
                ns: u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX),
                alloc_b,
                alloc_c,
                ..SampleRaw::default()
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use crate::bencher::test_support::{fixed, quick};
    use crate::bencher::{Bencher, run_benchmark_loop};
    use std::cell::Cell;
    use std::time::Duration;

    #[test]
    fn reported_time_becomes_the_sample_mean() {
        let result = run_benchmark_loop(Bencher::with_config(fixed(4, 25)), |b| {
            b.iter_custom(|iters| Duration::from_nanos(iters * 100));
        })
        .expect("measurement succeeds");
        assert_eq!(result.samples.len(), 4);
        assert!(result.samples.iter().all(|s| s.duration_nanos == 100.0));
        assert!(result.samples.iter().all(|s| s.iterations == 25));
        assert_eq!(result.iterations, 100);
        assert_eq!(result.total_time_ns, 100 * 100);
    }

    #[test]
    fn every_requested_iteration_is_accounted() {
        let requested = Cell::new(0_u64);
        let result = run_benchmark_loop(Bencher::with_config(quick(10)), |b| {
            b.iter_custom(|iters| {
                requested.set(requested.get() + iters);
                Duration::from_nanos(iters * 10)
            });
        })
        .expect("measurement succeeds");
        assert_eq!(result.samples.len(), 10);
        assert!(result.warmup_iterations > 0);
        assert_eq!(
            requested.get(),
            result.iterations + result.warmup_iterations
        );
        assert_eq!(result.total_time_ns, result.iterations * 10);
    }

    #[test]
    fn warmup_sizes_samples_from_the_wall_clock() {
        // The routine reports 1 ns per call but sleeps 1 ms per call.
        let result = run_benchmark_loop(Bencher::with_config(quick(5)), |b| {
            b.iter_custom(|iters| {
                std::thread::sleep(Duration::from_millis(iters));
                Duration::from_nanos(iters)
            });
        })
        .expect("measurement succeeds");
        // 2 ms per 10 ms / 5 samples: about 2 calls per sample, not millions.
        assert!(
            result.iters_per_sample <= 10,
            "ips: {}",
            result.iters_per_sample
        );
    }
}
