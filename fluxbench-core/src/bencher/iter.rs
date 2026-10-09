//! `iter`: measure a closure with no per-call setup.

use super::engine::SampleRaw;
use super::state::Bencher;
use crate::measure::Timer;
use std::hint::black_box;

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
}

#[cfg(test)]
mod tests {
    use crate::bencher::test_support::{fixed, quick};
    use crate::bencher::{Bencher, MeasureConfig, MeasureError, run_benchmark_loop};
    use std::cell::Cell;
    use std::sync::atomic::{AtomicUsize, Ordering};

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
}
