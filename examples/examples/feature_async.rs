//! Async Benchmarks — Tokio runtime integration
//!
//! FluxBench supports async benchmark functions. Annotate with `runtime`,
//! `worker_threads`, `enable_io`, and `enable_time` to configure the
//! Tokio runtime that drives each benchmark.
//!
//! Run with: cargo run --example feature_async -p fluxbench --release

use fluxbench::bench;
use fluxbench::prelude::*;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Basic async — single-threaded runtime
// ---------------------------------------------------------------------------

/// Minimal async benchmark — measures task spawn overhead alone.
#[bench(runtime = "multi_thread", worker_threads = 1, group = "async_basic")]
async fn async_noop(b: &mut Bencher) {
    b.iter_async(|| async { black_box(42) });
}

/// Timer resolution benchmark — measures tokio::time overhead.
#[bench(
    runtime = "multi_thread",
    worker_threads = 1,
    group = "async_basic",
    enable_time = true
)]
async fn async_timer_resolution(b: &mut Bencher) {
    b.iter_async(|| async {
        tokio::time::sleep(Duration::from_micros(1)).await;
    });
}

// ---------------------------------------------------------------------------
// Concurrency patterns — multi-threaded runtime
// ---------------------------------------------------------------------------

/// Spawn N tasks, join all — measures task spawn + schedule overhead.
#[bench(runtime = "multi_thread", worker_threads = 4, group = "concurrency")]
async fn spawn_join_fanout(b: &mut Bencher) {
    b.iter_async(|| async {
        let mut handles = Vec::with_capacity(16);
        for i in 0u64..16 {
            handles.push(tokio::spawn(async move { black_box(i * i) }));
        }
        let mut sum = 0u64;
        for h in handles {
            sum += h.await.unwrap();
        }
        black_box(sum)
    });
}

/// MPSC channel throughput — producer/consumer pattern.
#[bench(runtime = "multi_thread", worker_threads = 2, group = "concurrency")]
async fn mpsc_channel_throughput(b: &mut Bencher) {
    b.iter_async(|| async {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<u64>(64);
        let producer = tokio::spawn(async move {
            for i in 0u64..64 {
                tx.send(i).await.ok();
            }
        });
        let mut sum = 0u64;
        while let Some(v) = rx.recv().await {
            sum += v;
        }
        producer.await.unwrap();
        black_box(sum)
    });
}

/// Mutex contention — 4 tasks competing for a single lock.
#[bench(runtime = "multi_thread", worker_threads = 4, group = "concurrency")]
async fn mutex_contention(b: &mut Bencher) {
    let counter = Arc::new(tokio::sync::Mutex::new(0u64));
    b.iter_async(|| {
        let counter = counter.clone();
        async move {
            let mut handles = Vec::with_capacity(4);
            for _ in 0..4 {
                let c = counter.clone();
                handles.push(tokio::spawn(async move {
                    let mut guard = c.lock().await;
                    *guard += 1;
                }));
            }
            for h in handles {
                h.await.unwrap();
            }
            let val = *counter.lock().await;
            black_box(val)
        }
    });
}

/// Semaphore-gated concurrency — limit parallelism to 2 out of 8 tasks.
#[bench(runtime = "multi_thread", worker_threads = 4, group = "concurrency")]
async fn semaphore_gated(b: &mut Bencher) {
    let sem = Arc::new(tokio::sync::Semaphore::new(2));
    b.iter_async(|| {
        let sem = sem.clone();
        async move {
            let mut handles = Vec::with_capacity(8);
            for i in 0u64..8 {
                let s = sem.clone();
                handles.push(tokio::spawn(async move {
                    let _permit = s.acquire().await.unwrap();
                    black_box(i * i)
                }));
            }
            let mut sum = 0u64;
            for h in handles {
                sum += h.await.unwrap();
            }
            black_box(sum)
        }
    });
}

// ---------------------------------------------------------------------------
// I/O simulation — enable_io = true
// ---------------------------------------------------------------------------

/// Async file I/O benchmark (reads /dev/null as a cheap I/O syscall).
#[bench(
    runtime = "multi_thread",
    worker_threads = 1,
    enable_io = true,
    group = "async_io"
)]
async fn async_read_dev_null(b: &mut Bencher) {
    b.iter_async(|| async {
        let data = tokio::fs::read("/dev/null").await.unwrap();
        black_box(data.len())
    });
}

fn main() {
    if let Err(e) = fluxbench::run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
