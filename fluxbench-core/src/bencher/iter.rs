//! Synchronous `iter*` methods.

use super::engine::SampleRaw;
use super::state::Bencher;
use crate::measure::Timer;
use std::hint::black_box;

/// Most inputs `iter_with_setup` holds at once.
///
/// Peak memory is at most this many live inputs plus this many live outputs.
pub const SETUP_CHUNK: u64 = 1024;

impl Bencher {
    /// Measure `f`. This call runs warmup and every sample.
    ///
    /// One timer pair covers a whole batch of calls.
    /// Each sample holds the per-iteration mean of that batch.
    #[inline]
    pub fn iter<T, F>(&mut self, mut f: F)
    where
        F: FnMut() -> T,
    {
        if !self.begin_iter() {
            return;
        }
        self.run_loop(|b, n| {
            let mut raw = SampleRaw::default();
            b.alloc_reset();
            let timer = Timer::start();
            for _ in 0..n {
                black_box(f());
            }
            let (ns, cycles) = timer.stop();
            raw.add(ns, cycles, b.alloc_read());
            raw
        });
    }

    /// Measure `routine` on fresh inputs from `setup`. Setup time is excluded.
    ///
    /// Setup builds inputs untimed in chunks of at most `SETUP_CHUNK`.
    /// Outputs drop after the timer stops.
    /// Peak memory is `SETUP_CHUNK` inputs plus `SETUP_CHUNK` outputs.
    #[inline]
    pub fn iter_with_setup<T, S, F, R>(&mut self, mut setup: S, mut routine: F)
    where
        S: FnMut() -> T,
        F: FnMut(T) -> R,
    {
        if !self.begin_iter() {
            return;
        }
        let mut inputs: Vec<T> = Vec::new();
        let mut outputs: Vec<R> = Vec::new();
        self.run_loop(|b, n| {
            let mut raw = SampleRaw::default();
            let mut remaining = n;
            while remaining > 0 {
                let chunk = remaining.min(SETUP_CHUNK);
                // `chunk` is at most SETUP_CHUNK, so the cast is lossless.
                inputs.reserve(chunk as usize);
                outputs.reserve(chunk as usize);
                inputs.extend((0..chunk).map(|_| setup()));

                b.alloc_reset();
                let timer = Timer::start();
                for input in inputs.drain(..) {
                    outputs.push(black_box(routine(input)));
                }
                let (ns, cycles) = timer.stop();
                raw.add(ns, cycles, b.alloc_read());

                outputs.clear();
                remaining -= chunk;
            }
            raw
        });
    }

    /// Measure `routine` on a shared input that `setup` rebuilds every `batch_size` calls.
    ///
    /// `batch_size` sets only the setup cadence. A value of 0 rises to 1.
    /// Each sample calls `setup` `ceil(iters_per_sample / batch_size)` times, untimed.
    #[inline]
    pub fn iter_batched<T, S, F, R>(&mut self, batch_size: u64, mut setup: S, mut routine: F)
    where
        S: FnMut() -> T,
        F: FnMut(&T) -> R,
    {
        if !self.begin_iter() {
            return;
        }
        let batch_size = batch_size.max(1);
        self.run_loop(|b, n| {
            let mut raw = SampleRaw::default();
            let mut remaining = n;
            while remaining > 0 {
                let chunk = remaining.min(batch_size);
                let input = setup();

                b.alloc_reset();
                let timer = Timer::start();
                for _ in 0..chunk {
                    black_box(routine(black_box(&input)));
                }
                let (ns, cycles) = timer.stop();
                raw.add(ns, cycles, b.alloc_read());

                drop(input);
                remaining -= chunk;
            }
            raw
        });
    }
}

#[cfg(test)]
mod tests {
    use crate::bencher::{Bencher, MeasureConfig, MeasureError, run_benchmark_loop};
    use std::cell::Cell;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// Short time-based config: 2 ms warmup, 10 ms measurement.
    fn quick(sample_count: usize) -> MeasureConfig {
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

    /// Fixed-mode config: `samples` samples of `ips` calls.
    fn fixed(samples: usize, ips: u64) -> MeasureConfig {
        MeasureConfig {
            fixed_iters_per_sample: Some(ips),
            ..quick(samples)
        }
    }

    #[test]
    fn runner_called_once() {
        let calls = AtomicUsize::new(0);
        let result = run_benchmark_loop(Bencher::with_config(quick(10)), |b| {
            calls.fetch_add(1, Ordering::Relaxed);
            b.iter(|| 1_u64);
        });
        assert!(result.is_ok());
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn iter_calls_f_many_times_per_sample() {
        let count = Cell::new(0_u64);
        let result = run_benchmark_loop(Bencher::with_config(quick(10)), |b| {
            b.iter(|| count.set(count.get() + 1));
        })
        .expect("measurement succeeds");

        assert_eq!(result.samples.len(), 10);
        assert!(result.iters_per_sample > 1);
        assert!(count.get() > 100 * result.samples.len() as u64);
        assert!(
            result
                .samples
                .iter()
                .all(|s| s.iterations == result.iters_per_sample)
        );
        assert_eq!(count.get(), result.iterations + result.warmup_iterations);
    }

    #[test]
    fn sample_count_honoured_above_100() {
        let result = run_benchmark_loop(Bencher::with_config(quick(250)), |b| b.iter(|| 1_u64))
            .expect("measurement succeeds");
        assert_eq!(result.samples.len(), 250);
        assert!(!result.truncated);
    }

    #[test]
    fn fixed_mode_samples_n() {
        let count = Cell::new(0_u64);
        let result = run_benchmark_loop(Bencher::with_config(fixed(10, 3)), |b| {
            b.iter(|| count.set(count.get() + 1));
        })
        .expect("measurement succeeds");

        assert_eq!(result.samples.len(), 10);
        assert!(result.samples.iter().all(|s| s.iterations == 3));
        assert_eq!(result.iterations, 30);
        assert_eq!(result.warmup_iterations, 0);
        assert_eq!(count.get(), 30);
    }

    #[test]
    fn fixed_mode_500_samples_of_one_iteration() {
        let result = run_benchmark_loop(Bencher::with_config(fixed(500, 1)), |b| b.iter(|| 3_u64))
            .expect("measurement succeeds");
        assert_eq!(result.samples.len(), 500);
        assert_eq!(result.iterations, 500);
    }

    #[test]
    fn subnanosecond_resolution() {
        let result = run_benchmark_loop(Bencher::with_config(fixed(20, 10_000)), |b| {
            b.iter(|| 1_u64);
        })
        .expect("measurement succeeds");

        // A trivial call costs under 1 ns or a fraction of one.
        // Integer division returns whole numbers only.
        assert!(
            result
                .samples
                .iter()
                .any(|s| s.duration_nanos < 1.0 || s.duration_nanos.fract() != 0.0),
            "samples: {:?}",
            result.samples
        );
    }

    #[test]
    fn warmup_uses_constant_memory() {
        let config = MeasureConfig {
            warmup_ns: 20_000_000,
            ..quick(10)
        };
        let mut bencher = Bencher::with_config(config);
        bencher.iter(|| 1_u64);

        // Warmup ran many calls, yet the only buffer holds `sample_count` samples.
        assert!(bencher.warmup_iterations > 10_000);
        assert_eq!(bencher.samples.len(), 10);
        assert!(bencher.samples.capacity() <= 2 * 10);
    }

    #[test]
    fn run_loop_respects_min_iterations() {
        let config = MeasureConfig {
            min_iterations: Some(100),
            max_iterations: Some(100),
            measurement_ns: 1,
            ..quick(10)
        };
        let result = run_benchmark_loop(Bencher::with_config(config), |b| b.iter(|| 42_u64))
            .expect("measurement succeeds");
        assert_eq!(result.iterations, 100);
    }

    #[test]
    fn run_loop_raises_max_to_min() {
        let config = MeasureConfig {
            min_iterations: Some(200),
            max_iterations: Some(50),
            measurement_ns: 1,
            ..quick(10)
        };
        let result = run_benchmark_loop(Bencher::with_config(config), |b| b.iter(|| 7_u64))
            .expect("measurement succeeds");
        assert_eq!(result.iterations, 200);
    }

    #[test]
    fn second_iter_call_returns_error() {
        let second = Cell::new(0_u64);
        let result = run_benchmark_loop(Bencher::with_config(fixed(2, 1)), |b| {
            b.iter(|| 1_u64);
            b.iter(|| second.set(second.get() + 1));
        });
        let err = result.err();
        assert_eq!(err, Some(MeasureError::MultipleIterCalls));
        assert_eq!(
            err.map(|e| e.to_string()).as_deref(),
            Some("benchmark body called more than one b.iter* method")
        );
        assert_eq!(second.get(), 0);
    }

    #[test]
    fn no_iter_call_returns_error() {
        let result = run_benchmark_loop(Bencher::with_config(fixed(2, 1)), |_| {});
        let err = result.err();
        assert_eq!(err, Some(MeasureError::NoIterCall));
        assert_eq!(
            err.map(|e| e.to_string()).as_deref(),
            Some("benchmark body never called b.iter*")
        );
    }

    #[test]
    fn iter_batched_setup_cadence() {
        let setups = Cell::new(0_u64);
        let calls = Cell::new(0_u64);
        let result = run_benchmark_loop(Bencher::with_config(fixed(4, 10)), |b| {
            b.iter_batched(
                3,
                || setups.set(setups.get() + 1),
                |_| calls.set(calls.get() + 1),
            );
        })
        .expect("measurement succeeds");

        assert_eq!(result.iterations, 40);
        assert_eq!(calls.get(), 40);
        // ceil(10 / 3) = 4 setups per sample.
        assert_eq!(setups.get(), 4 * 10_u64.div_ceil(3));
    }

    #[test]
    fn iter_batched_zero_batch_size() {
        let setups = Cell::new(0_u64);
        let calls = Cell::new(0_u64);
        let result = run_benchmark_loop(Bencher::with_config(fixed(2, 5)), |b| {
            b.iter_batched(
                0,
                || setups.set(setups.get() + 1),
                |_| calls.set(calls.get() + 1),
            );
        })
        .expect("measurement succeeds");

        assert_eq!(result.iterations, 10);
        assert_eq!(calls.get(), 10);
        assert_eq!(setups.get(), 10);
    }

    #[test]
    fn iter_with_setup_excludes_setup_time() {
        let setups = Cell::new(0_u64);
        let result = run_benchmark_loop(Bencher::with_config(fixed(2, 5)), |b| {
            b.iter_with_setup(
                || {
                    setups.set(setups.get() + 1);
                    std::thread::sleep(Duration::from_millis(1));
                    vec![1_u64, 2, 3]
                },
                |v| v.iter().sum::<u64>(),
            );
        })
        .expect("measurement succeeds");

        assert_eq!(setups.get(), 10);
        assert_eq!(result.iterations, 10);
        // Each setup sleeps 1 ms. The routine takes far less.
        assert!(
            result.samples.iter().all(|s| s.duration_nanos < 500_000.0),
            "samples: {:?}",
            result.samples
        );
    }

    #[test]
    fn slow_setup_keeps_every_sample() {
        let config = MeasureConfig {
            warmup_ns: 5_000_000,
            measurement_ns: 50_000_000,
            ..quick(10)
        };
        let result = run_benchmark_loop(Bencher::with_config(config), |b| {
            b.iter_with_setup(|| std::thread::sleep(Duration::from_millis(1)), |()| 1_u64);
        })
        .expect("measurement succeeds");

        // Wall-clock sizing counts the 1 ms setup: about 5 calls per 5 ms sample.
        assert_eq!(result.samples.len(), 10);
        assert!(!result.truncated);
        assert!(
            result.iters_per_sample <= 10,
            "ips: {}",
            result.iters_per_sample
        );
    }

    #[test]
    fn iter_with_setup_chunks_large_batches() {
        let setups = Cell::new(0_u64);
        let calls = Cell::new(0_u64);
        let result = run_benchmark_loop(Bencher::with_config(fixed(2, 2_500)), |b| {
            b.iter_with_setup(
                || setups.set(setups.get() + 1),
                |()| calls.set(calls.get() + 1),
            );
        })
        .expect("measurement succeeds");

        assert_eq!(result.iterations, 5_000);
        assert_eq!(setups.get(), 5_000);
        assert_eq!(calls.get(), 5_000);
    }
}
