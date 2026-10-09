//! Async `iter*` methods.
//!
//! Each batch runs one `block_on` over all its calls.

use super::config::MeasureError;
use super::engine::SampleRaw;
use super::state::Bencher;
use crate::measure::Timer;
use std::future::Future;
use std::hint::black_box;
use tokio::runtime::{Builder, Handle, Runtime};

/// Build a current-thread runtime with every driver on.
fn current_thread_runtime() -> Result<Runtime, MeasureError> {
    Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| MeasureError::AsyncRuntime(e.to_string()))
}

impl Bencher {
    /// Measure an async closure on a new current-thread runtime.
    ///
    /// The runtime builds once, outside the timer.
    /// A runtime build error is recorded and `finish` returns it.
    #[inline]
    pub fn iter_async_standalone<T, F, Fut>(&mut self, f: F)
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = T>,
    {
        if !self.begin_iter() {
            return;
        }
        match current_thread_runtime() {
            Ok(rt) => self.run_on_runtime(&rt, f),
            Err(error) => self.record_error(error),
        }
    }

    /// Measure an async closure on the current tokio runtime.
    ///
    /// Inside a runtime, each batch runs in `block_in_place` on the current handle.
    /// That needs a multi-thread runtime.
    /// Outside a runtime, it builds a current-thread runtime once.
    #[inline]
    pub fn iter_async<T, F, Fut>(&mut self, mut f: F)
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = T>,
    {
        if !self.begin_iter() {
            return;
        }
        match Handle::try_current() {
            Ok(handle) => self.run_loop(|b, n| {
                tokio::task::block_in_place(|| {
                    let mut raw = SampleRaw::default();
                    b.alloc_reset();
                    let timer = Timer::start();
                    handle.block_on(async {
                        for _ in 0..n {
                            black_box(f().await);
                        }
                    });
                    let (ns, cycles) = timer.stop();
                    raw.add(ns, cycles, b.alloc_read());
                    raw
                })
            }),
            Err(_) => match current_thread_runtime() {
                Ok(rt) => self.run_on_runtime(&rt, f),
                Err(error) => self.record_error(error),
            },
        }
    }

    /// Drive warmup and measurement with one `rt.block_on` per batch.
    fn run_on_runtime<T, F, Fut>(&mut self, rt: &Runtime, mut f: F)
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = T>,
    {
        self.run_loop(|b, n| {
            let mut raw = SampleRaw::default();
            b.alloc_reset();
            let timer = Timer::start();
            rt.block_on(async {
                for _ in 0..n {
                    black_box(f().await);
                }
            });
            let (ns, cycles) = timer.stop();
            raw.add(ns, cycles, b.alloc_read());
            raw
        });
    }
}

#[cfg(test)]
mod tests {
    use crate::bencher::{Bencher, MeasureConfig, run_benchmark_loop};
    use std::cell::Cell;

    fn fixed(samples: usize, ips: u64) -> MeasureConfig {
        MeasureConfig {
            warmup_ns: 0,
            measurement_ns: 10_000_000,
            sample_count: samples,
            fixed_iters_per_sample: Some(ips),
            min_iterations: None,
            max_iterations: None,
            track_allocations: false,
        }
    }

    #[test]
    fn iter_async_outside_runtime_batches_calls() {
        let calls = Cell::new(0_u64);
        let result = run_benchmark_loop(Bencher::with_config(fixed(4, 8)), |b| {
            b.iter_async(|| {
                calls.set(calls.get() + 1);
                async { 1_u64 }
            });
        })
        .expect("measurement succeeds");
        assert_eq!(result.iterations, 32);
        assert_eq!(calls.get(), 32);
    }

    #[test]
    fn iter_async_inside_multi_thread_runtime() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("runtime builds");
        let result = rt.block_on(async {
            run_benchmark_loop(Bencher::with_config(fixed(3, 5)), |b| {
                b.iter_async(|| async { tokio::task::yield_now().await });
            })
        });
        let result = result.expect("measurement succeeds");
        assert_eq!(result.iterations, 15);
    }

    #[test]
    fn iter_async_standalone_runs_fixed_samples() {
        let result = run_benchmark_loop(Bencher::with_config(fixed(2, 4)), |b| {
            b.iter_async_standalone(|| async { 2_u64 });
        })
        .expect("measurement succeeds");
        assert_eq!(result.samples.len(), 2);
        assert_eq!(result.iterations, 8);
    }
}
