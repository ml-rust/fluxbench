//! Async `iter*` methods.
//!
//! Each batch runs one `Runtime::block_on` over all its calls.

use super::async_body::runtime_context_error;
use super::config::MeasureError;
use super::engine::SampleRaw;
use super::state::Bencher;
use crate::measure::Timer;
use std::future::Future;
use std::hint::black_box;
use std::sync::Arc;
use tokio::runtime::{Builder, Runtime};

impl Bencher {
    /// Attach the runtime that `iter_async` runs on.
    ///
    /// The `#[flux::bench]` wrapper attaches the runtime it builds.
    pub fn attach_runtime(&mut self, rt: Arc<Runtime>) {
        self.runtime = Some(rt);
    }

    /// Record a runtime build error. `finish` returns it as `MeasureError::AsyncRuntime`.
    pub fn attach_runtime_error(&mut self, message: impl Into<String>) {
        self.record_error(MeasureError::AsyncRuntime(message.into()));
    }

    /// Measure an async closure. Each batch awaits `n` calls inside one `block_on`.
    ///
    /// The timer covers creating and awaiting each future.
    /// It runs on the attached runtime.
    /// Without one, it builds a current-thread runtime once and keeps it.
    /// Call it from sync code or from a body that `drive_async_body` runs.
    /// A call inside a runtime context records `MeasureError::AsyncRuntime` and runs nothing.
    #[inline]
    pub fn iter_async<T, F, Fut>(&mut self, f: F)
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = T>,
    {
        if !self.begin_iter() {
            return;
        }
        if let Some(error) = runtime_context_error("iter_async") {
            self.record_error(error);
            return;
        }
        match self.runtime_or_build() {
            Ok(rt) => self.run_on_runtime(&rt, f),
            Err(error) => self.record_error(error),
        }
    }

    /// Measure an async closure. Same as `iter_async`.
    #[deprecated(note = "use `iter_async`")]
    #[inline]
    pub fn iter_async_standalone<T, F, Fut>(&mut self, f: F)
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = T>,
    {
        self.iter_async(f);
    }

    /// Return the attached runtime, or build a current-thread runtime and attach it.
    fn runtime_or_build(&mut self) -> Result<Arc<Runtime>, MeasureError> {
        if let Some(rt) = &self.runtime {
            return Ok(Arc::clone(rt));
        }
        let rt = Builder::new_current_thread()
            .enable_all()
            .build()
            .map(Arc::new)
            .map_err(|e| MeasureError::AsyncRuntime(e.to_string()))?;
        self.runtime = Some(Arc::clone(&rt));
        Ok(rt)
    }

    /// Drive warmup and measurement with one `rt.block_on` per batch.
    fn run_on_runtime<T, F, Fut>(&mut self, rt: &Runtime, mut f: F)
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = T>,
    {
        self.run_loop(|b, n| {
            let b = &*b;
            let (ns, cycles, alloc) = rt.block_on(async {
                b.alloc_reset();
                let timer = Timer::start();
                for _ in 0..n {
                    black_box(f().await);
                }
                let (ns, cycles) = timer.stop();
                (ns, cycles, b.alloc_read())
            });
            let mut raw = SampleRaw::default();
            raw.add(ns, cycles, alloc);
            raw
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bencher::async_body::drive_async_body;
    use crate::bencher::{MeasureConfig, run_benchmark_loop};
    use std::cell::Cell;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

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

    fn current_thread() -> Arc<Runtime> {
        Arc::new(
            Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime builds"),
        )
    }

    #[test]
    fn iter_async_without_runtime_batches_calls() {
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
    fn iter_async_without_runtime_builds_one_lazily() {
        let mut bencher = Bencher::with_config(fixed(3, 2));
        assert!(bencher.runtime.is_none());
        bencher.iter_async(|| async { tokio::time::sleep(Duration::from_micros(1)).await });
        let rt = bencher.runtime.clone().expect("lazy runtime attached");
        // The bencher and this clone hold the only references.
        assert_eq!(Arc::strong_count(&rt), 2);
        let result = bencher.finish().expect("measurement succeeds");
        assert_eq!(result.iterations, 6);
    }

    #[test]
    fn iter_async_current_thread() {
        let rt = current_thread();
        let mut bencher = Bencher::with_config(fixed(3, 4));
        bencher.attach_runtime(Arc::clone(&rt));
        drive_async_body(&rt, async {
            bencher.iter_async(|| tokio::time::sleep(Duration::from_micros(1)));
        })
        .expect("body runs");
        let result = bencher.finish().expect("measurement succeeds");
        assert_eq!(result.iterations, 12);
    }

    #[test]
    fn iter_async_multi_thread() {
        let rt = Arc::new(
            Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .expect("runtime builds"),
        );
        let mut bencher = Bencher::with_config(fixed(3, 5));
        bencher.attach_runtime(Arc::clone(&rt));
        drive_async_body(&rt, async {
            bencher.iter_async(|| async { tokio::task::yield_now().await });
        })
        .expect("body runs");
        let result = bencher.finish().expect("measurement succeeds");
        assert_eq!(result.iterations, 15);
    }

    #[test]
    fn runtime_built_once() {
        let started = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&started);
        let rt = Arc::new(
            Builder::new_multi_thread()
                .worker_threads(1)
                .enable_all()
                .on_thread_start(move || {
                    counter.fetch_add(1, Ordering::SeqCst);
                })
                .build()
                .expect("runtime builds"),
        );
        let mut bencher = Bencher::with_config(fixed(4, 4));
        bencher.attach_runtime(Arc::clone(&rt));
        drive_async_body(&rt, async {
            // The worker thread starts before it runs this task.
            tokio::spawn(async {}).await.expect("task joins");
            bencher.iter_async(|| async { tokio::task::yield_now().await });
        })
        .expect("body runs");
        let held = bencher.runtime.clone().expect("runtime attached");
        assert!(Arc::ptr_eq(&held, &rt));
        drop(held);
        bencher.finish().expect("measurement succeeds");
        // One worker thread. No blocking-pool thread runs in this test.
        assert_eq!(started.load(Ordering::SeqCst), 1);
        assert_eq!(Arc::strong_count(&rt), 1);
    }

    #[test]
    fn body_setup_awaits_io_on_current_thread() {
        let rt = current_thread();
        let mut bencher = Bencher::with_config(fixed(2, 3));
        bencher.attach_runtime(Arc::clone(&rt));
        drive_async_body(&rt, async {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("listener binds");
            let addr = listener.local_addr().expect("listener has an address");
            tokio::time::sleep(Duration::from_micros(1)).await;
            bencher.iter_async(|| async move { black_box(addr.port()) });
        })
        .expect("body runs");
        let result = bencher.finish().expect("measurement succeeds");
        assert_eq!(result.iterations, 6);
    }

    #[test]
    fn spawn_in_body_runs_during_iter_async() {
        let rt = current_thread();
        let mut bencher = Bencher::with_config(fixed(2, 4));
        bencher.attach_runtime(Arc::clone(&rt));
        drive_async_body(&rt, async {
            let (tx, mut rx) = tokio::sync::mpsc::channel::<u64>(1);
            let echo = tokio::spawn(async move {
                let mut count = 0_u64;
                while rx.recv().await.is_some() {
                    count += 1;
                }
                count
            });
            bencher.iter_async(|| {
                let tx = tx.clone();
                async move { tx.send(1).await.expect("echo task receives") }
            });
            drop(tx);
            let received = echo.await.expect("echo task joins");
            assert_eq!(received, 8);
        })
        .expect("body runs");
        bencher.finish().expect("measurement succeeds");
    }

    #[test]
    fn iter_async_inside_runtime_context_records_error() {
        let outer = current_thread();
        let result = outer.block_on(async {
            run_benchmark_loop(Bencher::with_config(fixed(2, 2)), |b| {
                b.iter_async(|| async {});
            })
        });
        assert!(matches!(result, Err(MeasureError::AsyncRuntime(_))));
    }

    #[test]
    fn iter_async_inside_task_records_error() {
        let outer = current_thread();
        let result = outer.block_on(async {
            tokio::spawn(async {
                let mut bencher = Bencher::with_config(fixed(2, 2));
                bencher.iter_async(|| async {});
                bencher.finish().err()
            })
            .await
            .expect("task joins")
        });
        assert!(matches!(result, Some(MeasureError::AsyncRuntime(_))));
    }

    #[test]
    fn attach_runtime_error_is_reported() {
        let mut bencher = Bencher::with_config(fixed(2, 2));
        bencher.attach_runtime_error("no runtime");
        let err = bencher.finish().err();
        assert_eq!(
            err,
            Some(MeasureError::AsyncRuntime("no runtime".to_string()))
        );
    }

    #[test]
    #[allow(deprecated)]
    fn iter_async_standalone_runs_fixed_samples() {
        let result = run_benchmark_loop(Bencher::with_config(fixed(2, 4)), |b| {
            b.iter_async_standalone(|| async { 2_u64 });
        })
        .expect("measurement succeeds");
        assert_eq!(result.samples.len(), 2);
        assert_eq!(result.iterations, 8);
    }
}
