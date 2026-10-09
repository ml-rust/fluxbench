//! Batched `iter*` methods: untimed setup builds inputs, one timer covers each batch.

use super::batch_size::BatchSize;
use super::engine::SampleRaw;
use super::state::Bencher;
use crate::measure::Timer;
use std::hint::black_box;

impl Bencher {
    /// Measure `routine` on fresh inputs from `setup`. Setup time is excluded.
    ///
    /// Same as `iter_batched(setup, routine, BatchSize::SmallInput)`.
    #[inline]
    pub fn iter_with_setup<T, S, F, R>(&mut self, setup: S, routine: F)
    where
        S: FnMut() -> T,
        F: FnMut(T) -> R,
    {
        self.iter_batched(setup, routine, BatchSize::SmallInput);
    }

    /// Measure `routine` on inputs it takes by value. Setup time is excluded.
    ///
    /// `setup` runs once per call. It builds each batch of inputs untimed.
    /// `size` sets the inputs per batch. One timer covers the routine over a batch.
    /// Outputs drop after the timer stops.
    #[inline]
    pub fn iter_batched<T, S, F, R>(&mut self, setup: S, mut routine: F, size: BatchSize)
    where
        S: FnMut() -> T,
        F: FnMut(T) -> R,
    {
        self.run_batched(size, setup, |inputs: &mut Vec<T>, outputs: &mut Vec<R>| {
            for input in inputs.drain(..) {
                outputs.push(black_box(routine(input)));
            }
        });
    }

    /// Measure `routine` on inputs it borrows mutably. Setup time is excluded.
    ///
    /// Same batching as `iter_batched`.
    /// Inputs and outputs drop after the timer stops.
    #[inline]
    pub fn iter_batched_ref<T, S, F, R>(&mut self, setup: S, mut routine: F, size: BatchSize)
    where
        S: FnMut() -> T,
        F: FnMut(&mut T) -> R,
    {
        self.run_batched(size, setup, |inputs: &mut Vec<T>, outputs: &mut Vec<R>| {
            for input in inputs.iter_mut() {
                outputs.push(black_box(routine(input)));
            }
        });
    }

    /// Measure `routine` on a shared input that `setup` rebuilds every `batch_size` calls.
    ///
    /// `batch_size` sets only the setup cadence. A value of 0 rises to 1.
    /// Each sample calls `setup` `ceil(iters_per_sample / batch_size)` times, untimed.
    #[inline]
    pub fn iter_batched_shared<T, S, F, R>(&mut self, batch_size: u64, mut setup: S, mut routine: F)
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

    /// Run warmup and measurement in batches of `size` inputs.
    ///
    /// Per batch: `setup` fills `inputs` untimed, `timed` runs under one timer, then both buffers clear.
    /// `timed` consumes or borrows the inputs and pushes one output per input.
    /// The buffers live across batches, so their capacity is reused.
    fn run_batched<T, R, S, B>(&mut self, size: BatchSize, mut setup: S, mut timed: B)
    where
        S: FnMut() -> T,
        B: FnMut(&mut Vec<T>, &mut Vec<R>),
    {
        if !self.begin_iter() {
            return;
        }
        let mut inputs: Vec<T> = Vec::new();
        let mut outputs: Vec<R> = Vec::new();
        self.run_loop(|b, n| {
            let mut raw = SampleRaw::default();
            let batch_len = size.batch_len(n);
            let mut remaining = n;
            while remaining > 0 {
                let chunk = remaining.min(batch_len);
                // `chunk` always fits `usize` on 64-bit targets. Elsewhere it saturates.
                let len = usize::try_from(chunk).unwrap_or(usize::MAX);
                inputs.reserve(len);
                outputs.reserve(len);
                inputs.extend((0..chunk).map(|_| setup()));

                b.alloc_reset();
                let timer = Timer::start();
                timed(&mut inputs, &mut outputs);
                let (ns, cycles) = timer.stop();
                raw.add(ns, cycles, b.alloc_read());

                outputs.clear();
                inputs.clear();
                remaining -= chunk;
            }
            raw
        });
    }
}

#[cfg(test)]
mod tests {
    use crate::bencher::test_support::{fixed, quick};
    use crate::bencher::{BatchSize, Bencher, MeasureConfig, run_benchmark_loop};
    use std::cell::Cell;
    use std::time::Duration;

    /// Counts setup calls and batches. A batch starts at a setup that follows a routine call.
    #[derive(Default)]
    struct Trace {
        setups: Cell<u64>,
        calls: Cell<u64>,
        batches: Cell<u64>,
        in_setup: Cell<bool>,
    }

    impl Trace {
        fn setup(&self) {
            if !self.in_setup.replace(true) {
                self.batches.set(self.batches.get() + 1);
            }
            self.setups.set(self.setups.get() + 1);
        }

        fn call(&self) {
            self.in_setup.set(false);
            self.calls.set(self.calls.get() + 1);
        }
    }

    /// Run `iter_batched` with `size` over 2 samples of 40 calls. Returns the trace.
    fn trace_by_value(size: BatchSize) -> Trace {
        let trace = Trace::default();
        let result = run_benchmark_loop(Bencher::with_config(fixed(2, 40)), |b| {
            b.iter_batched(|| trace.setup(), |()| trace.call(), size);
        })
        .expect("measurement succeeds");
        assert_eq!(result.iterations, 80);
        trace
    }

    #[test]
    fn setup_runs_once_per_call_for_every_batch_size() {
        for size in [
            BatchSize::SmallInput,
            BatchSize::LargeInput,
            BatchSize::PerIteration,
            BatchSize::NumBatches(4),
            BatchSize::NumIterations(10),
        ] {
            let trace = trace_by_value(size);
            assert_eq!(trace.setups.get(), 80, "{size:?}");
            assert_eq!(trace.calls.get(), 80, "{size:?}");
        }
    }

    #[test]
    fn batch_count_follows_batch_size() {
        // Two samples of 40 calls each.
        let cases = [
            (BatchSize::SmallInput, 2),
            (BatchSize::LargeInput, 2 * 3),
            (BatchSize::PerIteration, 80),
            (BatchSize::NumBatches(4), 2 * 4),
            (BatchSize::NumIterations(10), 2 * 4),
            (BatchSize::NumIterations(15), 2 * 3),
        ];
        for (size, batches) in cases {
            assert_eq!(trace_by_value(size).batches.get(), batches, "{size:?}");
        }
    }

    #[test]
    fn iter_batched_ref_uses_the_same_batches() {
        let trace = Trace::default();
        run_benchmark_loop(Bencher::with_config(fixed(2, 40)), |b| {
            b.iter_batched_ref(|| trace.setup(), |_| trace.call(), BatchSize::LargeInput);
        })
        .expect("measurement succeeds");
        assert_eq!(trace.setups.get(), 80);
        assert_eq!(trace.batches.get(), 6);
    }

    #[test]
    fn iter_batched_excludes_setup_time() {
        let result = run_benchmark_loop(Bencher::with_config(fixed(2, 5)), |b| {
            b.iter_batched(
                || {
                    std::thread::sleep(Duration::from_millis(1));
                    vec![1_u64, 2, 3]
                },
                |v| v.iter().sum::<u64>(),
                BatchSize::SmallInput,
            );
        })
        .expect("measurement succeeds");
        assert!(
            result.samples.iter().all(|s| s.duration_nanos < 500_000.0),
            "samples: {:?}",
            result.samples
        );
    }

    /// Sleeps 1 ms on drop. A drop inside the timer adds about 1 ms per call.
    struct SlowDrop;

    impl Drop for SlowDrop {
        fn drop(&mut self) {
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn iter_batched_drops_outputs_outside_the_timer() {
        let result = run_benchmark_loop(Bencher::with_config(fixed(2, 5)), |b| {
            b.iter_batched(|| (), |()| SlowDrop, BatchSize::SmallInput);
        })
        .expect("measurement succeeds");
        assert!(
            result.samples.iter().all(|s| s.duration_nanos < 500_000.0),
            "samples: {:?}",
            result.samples
        );
    }

    #[test]
    fn iter_batched_ref_drops_inputs_and_outputs_outside_the_timer() {
        let result = run_benchmark_loop(Bencher::with_config(fixed(2, 5)), |b| {
            b.iter_batched_ref(|| SlowDrop, |_| SlowDrop, BatchSize::PerIteration);
        })
        .expect("measurement succeeds");
        assert!(
            result.samples.iter().all(|s| s.duration_nanos < 500_000.0),
            "samples: {:?}",
            result.samples
        );
    }

    /// Adds its final `value` to `total` on drop.
    struct Probe<'a> {
        value: u64,
        total: &'a Cell<u64>,
    }

    impl Drop for Probe<'_> {
        fn drop(&mut self) {
            self.total.set(self.total.get() + self.value);
        }
    }

    #[test]
    fn iter_batched_ref_mutation_is_visible() {
        let total = Cell::new(0_u64);
        let result = run_benchmark_loop(Bencher::with_config(fixed(3, 7)), |b| {
            b.iter_batched_ref(
                || Probe {
                    value: 0,
                    total: &total,
                },
                |probe| {
                    probe.value += 2;
                    probe.value
                },
                BatchSize::NumBatches(2),
            );
        })
        .expect("measurement succeeds");
        assert_eq!(result.iterations, 21);
        // Each input starts at 0. The routine adds 2 before the input drops.
        assert_eq!(total.get(), 42);
    }

    #[test]
    fn iter_batched_shared_setup_cadence() {
        let setups = Cell::new(0_u64);
        let calls = Cell::new(0_u64);
        let result = run_benchmark_loop(Bencher::with_config(fixed(4, 10)), |b| {
            b.iter_batched_shared(
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
    fn iter_batched_shared_zero_batch_size() {
        let setups = Cell::new(0_u64);
        let calls = Cell::new(0_u64);
        let result = run_benchmark_loop(Bencher::with_config(fixed(2, 5)), |b| {
            b.iter_batched_shared(
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
        let trace = Trace::default();
        let result = run_benchmark_loop(Bencher::with_config(fixed(2, 2_500)), |b| {
            b.iter_with_setup(|| trace.setup(), |()| trace.call());
        })
        .expect("measurement succeeds");

        assert_eq!(result.iterations, 5_000);
        assert_eq!(trace.setups.get(), 5_000);
        assert_eq!(trace.calls.get(), 5_000);
        // ceil(2500 / 1024) = 3 chunks per sample.
        assert_eq!(trace.batches.get(), 6);
    }
}
