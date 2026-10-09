//! Supervisor: shard benchmarks across worker threads and collect one result per benchmark.
//!
//! A failed benchmark yields its own failed result. The rest keep running.
//! A worker that crashes, times out, or breaks the IPC stream is discarded.
//! The next benchmark then gets a fresh worker.

use super::error::SupervisorError;
use super::handle::WorkerHandle;
use super::result::IpcBenchmarkResult;
use fluxbench_core::BenchmarkDef;
use fluxbench_ipc::BenchmarkConfig;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use std::time::Duration;

/// One benchmark with its position in the input and its config.
type Job<'a> = (usize, &'a BenchmarkDef, &'a BenchmarkConfig);

/// Supervisor that manages worker pool and distributes benchmarks
pub struct Supervisor {
    config: BenchmarkConfig,
    timeout: Duration,
    num_workers: usize,
    /// CPU for worker slot `i`. A missing or `None` entry leaves that worker unpinned.
    cpus: Vec<Option<usize>>,
}

impl Supervisor {
    /// Create a new supervisor
    pub fn new(config: BenchmarkConfig, timeout: Duration, num_workers: usize) -> Self {
        Self {
            config,
            timeout,
            num_workers: num_workers.max(1),
            cpus: Vec::new(),
        }
    }

    /// Set the CPU each worker slot pins to. Slot `i` takes entry `i`.
    pub fn with_cpus(mut self, cpus: Vec<Option<usize>>) -> Self {
        self.cpus = cpus;
        self
    }

    /// Run all benchmarks in one-shot mode with the default config.
    ///
    /// For per-benchmark configuration, use [`run_all_configs`](Self::run_all_configs).
    pub fn run_all(
        &self,
        benchmarks: &[&BenchmarkDef],
    ) -> Result<Vec<IpcBenchmarkResult>, SupervisorError> {
        let configs: Vec<_> = benchmarks.iter().map(|_| self.config.clone()).collect();
        self.run_all_configs(benchmarks, &configs)
    }

    /// Run each benchmark in a fresh worker (one-shot mode).
    ///
    /// Returns one result per benchmark, in input order. It never returns `Err`.
    /// The `Result` stays for API compatibility.
    pub fn run_all_configs(
        &self,
        benchmarks: &[&BenchmarkDef],
        configs: &[BenchmarkConfig],
    ) -> Result<Vec<IpcBenchmarkResult>, SupervisorError> {
        Ok(self.run_sharded(benchmarks, configs, false))
    }

    /// Run benchmarks with worker reuse using the default config.
    ///
    /// For per-benchmark configuration, use [`run_with_reuse_configs`](Self::run_with_reuse_configs).
    pub fn run_with_reuse(
        &self,
        benchmarks: &[&BenchmarkDef],
    ) -> Result<Vec<IpcBenchmarkResult>, SupervisorError> {
        let configs: Vec<_> = benchmarks.iter().map(|_| self.config.clone()).collect();
        self.run_with_reuse_configs(benchmarks, &configs)
    }

    /// Run benchmarks with worker reuse (persistent mode) and per-benchmark configs.
    ///
    /// Returns one result per benchmark, in input order. It never returns `Err`.
    /// The `Result` stays for API compatibility.
    pub fn run_with_reuse_configs(
        &self,
        benchmarks: &[&BenchmarkDef],
        configs: &[BenchmarkConfig],
    ) -> Result<Vec<IpcBenchmarkResult>, SupervisorError> {
        Ok(self.run_sharded(benchmarks, configs, true))
    }

    /// Split benchmarks round-robin over `num_workers` threads. Each thread runs one shard.
    ///
    /// A benchmark with no config gets the default config.
    fn run_sharded(
        &self,
        benchmarks: &[&BenchmarkDef],
        configs: &[BenchmarkConfig],
        reuse: bool,
    ) -> Vec<IpcBenchmarkResult> {
        if benchmarks.is_empty() {
            return Vec::new();
        }

        let jobs: Vec<Job<'_>> = benchmarks
            .iter()
            .enumerate()
            .map(|(index, bench)| (index, *bench, configs.get(index).unwrap_or(&self.config)))
            .collect();

        let worker_count = self.num_workers.min(jobs.len());
        let indexed = if worker_count == 1 {
            self.run_shard(&jobs, reuse, self.cpu_for(0))
        } else {
            let mut shards: Vec<Vec<Job<'_>>> = vec![Vec::new(); worker_count];
            for (position, job) in jobs.iter().enumerate() {
                shards[position % worker_count].push(*job);
            }
            match ThreadPoolBuilder::new().num_threads(worker_count).build() {
                Ok(pool) => pool.install(|| {
                    shards
                        .par_iter()
                        .enumerate()
                        .flat_map_iter(|(slot, shard)| {
                            self.run_shard(shard, reuse, self.cpu_for(slot))
                        })
                        .collect()
                }),
                Err(e) => {
                    eprintln!(
                        "fluxbench: warning: worker thread pool build failed: {e}. Running {} benchmarks serially.",
                        jobs.len()
                    );
                    self.run_shard(&jobs, reuse, self.cpu_for(0))
                }
            }
        };

        let mut slots: Vec<Option<IpcBenchmarkResult>> = std::iter::repeat_with(|| None)
            .take(benchmarks.len())
            .collect();
        for (index, result) in indexed {
            if let Some(slot) = slots.get_mut(index) {
                *slot = Some(result);
            }
        }
        slots
            .into_iter()
            .zip(benchmarks)
            .map(|(slot, bench)| {
                slot.unwrap_or_else(|| {
                    IpcBenchmarkResult::crashed(
                        bench.id,
                        "internal error: the supervisor produced no result for this benchmark. Report a bug."
                            .to_string(),
                    )
                })
            })
            .collect()
    }

    /// CPU for worker slot `slot`, if one is assigned.
    fn cpu_for(&self, slot: usize) -> Option<usize> {
        self.cpus.get(slot).copied().flatten()
    }

    /// Run one shard in order on one worker slot, pinned to `cpu`.
    ///
    /// `reuse == false` retires the worker after each benchmark (one-shot mode).
    fn run_shard(
        &self,
        shard: &[Job<'_>],
        reuse: bool,
        cpu: Option<usize>,
    ) -> Vec<(usize, IpcBenchmarkResult)> {
        let mut slot: Option<WorkerHandle> = None;
        let mut results = Vec::with_capacity(shard.len());
        for &(index, bench, config) in shard {
            results.push((index, self.run_one(&mut slot, bench, config, cpu)));
            if !reuse {
                retire(slot.take());
            }
        }
        retire(slot.take());
        results
    }

    /// Run one benchmark on the worker in `slot`. Spawn a worker when the slot is empty.
    ///
    /// Always returns a result. A send error on a reused worker retries once on a fresh one.
    /// After an IPC error or a dead worker, the slot is emptied.
    fn run_one(
        &self,
        slot: &mut Option<WorkerHandle>,
        bench: &BenchmarkDef,
        config: &BenchmarkConfig,
        cpu: Option<usize>,
    ) -> IpcBenchmarkResult {
        let mut last_send_error = None;
        for _ in 0..2 {
            let worker = match self.ready_worker(slot, cpu) {
                Ok(worker) => worker,
                Err(e) => return IpcBenchmarkResult::crashed(bench.id, e.to_string()),
            };
            if let Err(e) = worker.send_run(bench.id, config) {
                // A reused worker can die between benchmarks. Retry on a fresh one.
                discard(slot.take());
                last_send_error = Some(e);
                continue;
            }
            let outcome = worker.collect(bench.id);
            return match outcome {
                Ok(result) => {
                    if !worker.is_alive() {
                        discard(slot.take());
                    }
                    result
                }
                Err(e) => {
                    discard(slot.take());
                    IpcBenchmarkResult::crashed(bench.id, e.to_string())
                }
            };
        }
        let message = match last_send_error {
            Some(e) => format!("sending the run command to the worker failed twice: {e}"),
            None => "sending the run command to the worker failed twice".to_string(),
        };
        IpcBenchmarkResult::crashed(bench.id, message)
    }

    /// Return the worker in `slot`, replacing a dead or missing one with a fresh spawn.
    fn ready_worker<'s>(
        &self,
        slot: &'s mut Option<WorkerHandle>,
        cpu: Option<usize>,
    ) -> Result<&'s mut WorkerHandle, SupervisorError> {
        if slot.as_mut().is_some_and(|w| !w.is_alive()) {
            discard(slot.take());
        }
        if slot.is_none() {
            *slot = Some(WorkerHandle::spawn(self.timeout, cpu)?);
        }
        slot.as_mut().ok_or_else(|| {
            SupervisorError::IpcError("worker slot is empty after spawn. Report a bug.".to_string())
        })
    }
}

/// Shut a worker down gracefully. It is reaped either way.
fn retire(worker: Option<WorkerHandle>) {
    if let Some(worker) = worker {
        let _ = worker.shutdown();
    }
}

/// Kill and reap a worker whose state is unknown.
fn discard(worker: Option<WorkerHandle>) {
    if let Some(mut worker) = worker {
        let _ = worker.kill();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_count_has_floor_of_one() {
        let supervisor = Supervisor::new(BenchmarkConfig::default(), Duration::from_secs(30), 0);
        assert_eq!(supervisor.num_workers, 1);
    }

    #[test]
    fn empty_input_yields_empty_output() {
        let supervisor = Supervisor::new(BenchmarkConfig::default(), Duration::from_secs(30), 2);
        let results = supervisor
            .run_with_reuse_configs(&[], &[])
            .expect("never errs");
        assert!(results.is_empty());
        let results = supervisor.run_all_configs(&[], &[]).expect("never errs");
        assert!(results.is_empty());
    }
}
