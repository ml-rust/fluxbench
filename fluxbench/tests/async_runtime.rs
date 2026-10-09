//! Async benchmark runtime tests.
//!
//! `#[flux::bench]` registers into the harness, so these tests call the wrapper steps directly.
//! `run_like_macro` mirrors the code the macro generates for an async body.

use fluxbench::internal::drive_async_body;
use fluxbench::{Bencher, BenchmarkResult, MeasureConfig, MeasureError};
use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use tokio::runtime::{Builder, Runtime};

/// Fixed-mode config: `samples` samples of `ips` calls, no warmup.
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

/// Attach the runtime, drive the body, and finish, as the macro wrapper does.
///
/// The body is an `async move` block with `$b: &mut Bencher` in scope.
macro_rules! run_like_macro {
    ($rt:expr, $config:expr, |$b:ident| $body:block) => {{
        let rt = Arc::new($rt);
        let mut bencher = Bencher::with_config($config);
        bencher.attach_runtime(Arc::clone(&rt));
        let $b = &mut bencher;
        if let Err(error) = drive_async_body(&rt, async move $body) {
            bencher.record_error(error);
        }
        let result: Result<BenchmarkResult, MeasureError> = bencher.finish();
        drop(rt);
        result
    }};
}

fn current_thread() -> Runtime {
    Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime builds")
}

#[test]
fn iter_async_current_thread() {
    let result = run_like_macro!(current_thread(), fixed(3, 4), |b| {
        b.iter_async(|| tokio::time::sleep(Duration::from_micros(1)));
    })
    .expect("measurement succeeds");
    assert_eq!(result.iterations, 12);
    assert!(result.samples.iter().all(|s| s.duration_nanos >= 1_000.0));
}

#[test]
fn iter_async_multi_thread() {
    let rt = Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("runtime builds");
    let result = run_like_macro!(rt, fixed(3, 5), |b| {
        b.iter_async(|| async {
            tokio::spawn(async { black_box(1_u64) })
                .await
                .expect("task joins")
        });
    })
    .expect("measurement succeeds");
    assert_eq!(result.iterations, 15);
}

#[test]
fn runtime_built_once() {
    let started = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&started);
    let rt = Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .on_thread_start(move || {
            counter.fetch_add(1, Ordering::SeqCst);
        })
        .build()
        .expect("runtime builds");
    let result = run_like_macro!(rt, fixed(4, 4), |b| {
        tokio::spawn(async {}).await.expect("task joins");
        b.iter_async(|| async { tokio::task::yield_now().await });
    })
    .expect("measurement succeeds");
    assert_eq!(result.iterations, 16);
    // The runtime dropped, so every thread it started has run `on_thread_start`.
    // Two workers from one runtime. A second runtime starts more threads.
    assert_eq!(started.load(Ordering::SeqCst), 2);
}

#[test]
fn async_body_setup_await_on_current_thread() {
    let result = run_like_macro!(current_thread(), fixed(2, 3), |b| {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener binds");
        let addr = listener.local_addr().expect("listener has an address");
        tokio::time::sleep(Duration::from_micros(1)).await;
        b.iter_async(|| async move { black_box(addr.port()) });
    })
    .expect("measurement succeeds");
    assert_eq!(result.iterations, 6);
}

#[test]
fn spawn_inside_body_works() {
    let received = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&received);
    let result = run_like_macro!(current_thread(), fixed(2, 4), |b| {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<u64>(1);
        let consumer = tokio::spawn(async move {
            while rx.recv().await.is_some() {
                seen.fetch_add(1, Ordering::SeqCst);
            }
        });
        b.iter_async(|| {
            let tx = tx.clone();
            async move { tx.send(1).await.expect("consumer receives") }
        });
        drop(tx);
        consumer.await.expect("consumer joins");
    })
    .expect("measurement succeeds");
    assert_eq!(result.iterations, 8);
    assert_eq!(received.load(Ordering::SeqCst), 8);
}

#[test]
fn iter_async_without_attached_runtime_builds_lazy_runtime() {
    let calls = AtomicUsize::new(0);
    let mut bencher = Bencher::with_config(fixed(3, 2));
    bencher.iter_async(|| {
        calls.fetch_add(1, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_micros(1))
    });
    let result = bencher.finish().expect("measurement succeeds");
    assert_eq!(result.iterations, 6);
    assert_eq!(calls.load(Ordering::SeqCst), 6);
}

#[test]
fn iter_async_inside_runtime_context_records_error() {
    let outer = current_thread();
    let error = outer.block_on(async {
        let mut bencher = Bencher::with_config(fixed(2, 2));
        bencher.iter_async(|| async {});
        bencher.finish().err()
    });
    assert!(matches!(error, Some(MeasureError::AsyncRuntime(_))));
}

#[test]
fn drive_async_body_inside_runtime_context_records_error() {
    // A runtime must drop outside any runtime context, so `inner` outlives `block_on`.
    let outer = current_thread();
    let inner = current_thread();
    let ran = AtomicUsize::new(0);
    let result = outer.block_on(async {
        drive_async_body(&inner, async {
            ran.fetch_add(1, Ordering::SeqCst);
        })
    });
    assert!(matches!(result, Err(MeasureError::AsyncRuntime(_))));
    assert_eq!(ran.load(Ordering::SeqCst), 0);
}

#[test]
fn attach_runtime_error_is_reported() {
    let mut bencher = Bencher::with_config(fixed(2, 2));
    bencher.attach_runtime_error("build error");
    assert_eq!(
        bencher.finish().err(),
        Some(MeasureError::AsyncRuntime("build error".to_string()))
    );
}
