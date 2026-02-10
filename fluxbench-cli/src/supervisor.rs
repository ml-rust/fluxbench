//! Supervisor Process
//!
//! Manages worker processes and aggregates results via IPC.

use fluxbench_core::BenchmarkDef;
use fluxbench_ipc::{
    BenchmarkConfig, FrameError, FrameReader, FrameWriter, Sample, SupervisorCommand,
    WorkerCapabilities, WorkerMessage,
};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use std::env;
use std::os::unix::io::AsRawFd;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SupervisorError {
    #[error("Failed to spawn worker: {0}")]
    SpawnFailed(#[from] std::io::Error),

    #[error("IPC error: {0}")]
    IpcError(String),

    #[error("Worker crashed: {0}")]
    WorkerCrashed(String),

    #[error("Timeout waiting for worker")]
    Timeout,

    #[error("Benchmark not found: {0}")]
    BenchmarkNotFound(String),

    #[error("Worker protocol error: expected {expected}, got {got}")]
    ProtocolError { expected: String, got: String },
}

impl From<FrameError> for SupervisorError {
    fn from(e: FrameError) -> Self {
        SupervisorError::IpcError(e.to_string())
    }
}

/// Result from a benchmark run via IPC
#[derive(Debug)]
pub struct IpcBenchmarkResult {
    pub bench_id: String,
    pub samples: Vec<Sample>,
    pub total_iterations: u64,
    pub total_duration_nanos: u64,
    pub status: IpcBenchmarkStatus,
}

#[derive(Debug, Clone)]
pub enum IpcBenchmarkStatus {
    Success,
    Failed { message: String },
    Crashed { message: String },
}

/// Result of polling for data
#[derive(Debug)]
enum PollResult {
    DataAvailable,
    Timeout,
    PipeClosed,
    Error(std::io::Error),
}

/// Wait for data to be available on a file descriptor with timeout
fn wait_for_data(fd: i32, timeout_ms: i32) -> PollResult {
    let mut pollfd = libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    };

    let result = unsafe { libc::poll(&mut pollfd, 1, timeout_ms) };

    if result < 0 {
        PollResult::Error(std::io::Error::last_os_error())
    } else if result == 0 {
        PollResult::Timeout
    } else {
        // Check if data is available (even if pipe is closing, there might be data)
        if pollfd.revents & libc::POLLIN != 0 {
            PollResult::DataAvailable
        } else if pollfd.revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 {
            // No data, but pipe error/hangup
            PollResult::PipeClosed
        } else {
            // Spurious wakeup, treat as timeout
            PollResult::Timeout
        }
    }
}

/// Worker process handle
pub struct WorkerHandle {
    child: Child,
    reader: FrameReader<std::process::ChildStdout>,
    writer: FrameWriter<std::process::ChildStdin>,
    capabilities: Option<WorkerCapabilities>,
    timeout: Duration,
    stdout_fd: i32, // Cache the fd for polling
}

impl WorkerHandle {
    /// Spawn a new worker process
    pub fn spawn(timeout: Duration) -> Result<Self, SupervisorError> {
        // Get current executable path
        let binary = env::current_exe().map_err(SupervisorError::SpawnFailed)?;

        let mut child = Command::new(&binary)
            .arg("--flux-worker")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;

        let stdin = child.stdin.take().expect("stdin should be available");
        let stdout = child.stdout.take().expect("stdout should be available");
        let stdout_fd = stdout.as_raw_fd();

        let mut handle = Self {
            child,
            reader: FrameReader::new(stdout),
            writer: FrameWriter::new(stdin),
            capabilities: None,
            timeout,
            stdout_fd,
        };

        // Wait for Hello message
        handle.wait_for_hello()?;

        Ok(handle)
    }

    /// Spawn a worker for a specific binary (for testing)
    pub fn spawn_binary(binary: &str, timeout: Duration) -> Result<Self, SupervisorError> {
        let mut child = Command::new(binary)
            .arg("--flux-worker")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;

        let stdin = child.stdin.take().expect("stdin should be available");
        let stdout = child.stdout.take().expect("stdout should be available");
        let stdout_fd = stdout.as_raw_fd();

        let mut handle = Self {
            child,
            reader: FrameReader::new(stdout),
            writer: FrameWriter::new(stdin),
            capabilities: None,
            timeout,
            stdout_fd,
        };

        handle.wait_for_hello()?;

        Ok(handle)
    }

    /// Wait for Hello message from worker
    fn wait_for_hello(&mut self) -> Result<(), SupervisorError> {
        let msg: WorkerMessage = self.reader.read()?;

        match msg {
            WorkerMessage::Hello(caps) => {
                self.capabilities = Some(caps);
                Ok(())
            }
            other => Err(SupervisorError::ProtocolError {
                expected: "Hello".to_string(),
                got: format!("{:?}", other),
            }),
        }
    }

    /// Get worker capabilities
    pub fn capabilities(&self) -> Option<&WorkerCapabilities> {
        self.capabilities.as_ref()
    }

    /// Run a benchmark on this worker
    pub fn run_benchmark(
        &mut self,
        bench_id: &str,
        config: &BenchmarkConfig,
    ) -> Result<IpcBenchmarkResult, SupervisorError> {
        // Send run command
        self.writer.write(&SupervisorCommand::Run {
            bench_id: bench_id.to_string(),
            config: config.clone(),
        })?;

        // Collect all sample batches
        let mut all_samples = Vec::new();
        let start = Instant::now();

        loop {
            // Check timeout
            let remaining = self.timeout.saturating_sub(start.elapsed());
            if remaining.is_zero() {
                return Err(SupervisorError::Timeout);
            }

            // Check if there's buffered data, or poll for new data
            if self.reader.has_buffered_data() {
                // Have buffered data - but still check if worker is alive before blocking read
                // (the buffered data might be an incomplete frame)
                if !self.is_alive() {
                    return Err(SupervisorError::WorkerCrashed(
                        "Worker process crashed with partial data buffered".to_string(),
                    ));
                }
            } else {
                // Poll for data with 100ms intervals (allows checking worker status)
                let poll_timeout = remaining.min(Duration::from_millis(100));
                let poll_result = wait_for_data(self.stdout_fd, poll_timeout.as_millis() as i32);

                match poll_result {
                    PollResult::DataAvailable => {
                        // Data available - check if worker is still alive before blocking read
                        if !self.is_alive() {
                            return Err(SupervisorError::WorkerCrashed(
                                "Worker process crashed with data in pipe".to_string(),
                            ));
                        }
                    }
                    PollResult::Timeout => {
                        // No data - check if worker is still alive
                        if !self.is_alive() {
                            return Err(SupervisorError::WorkerCrashed(
                                "Worker process exited unexpectedly".to_string(),
                            ));
                        }
                        continue; // Poll again
                    }
                    PollResult::PipeClosed => {
                        return Err(SupervisorError::WorkerCrashed(
                            "Worker pipe closed unexpectedly".to_string(),
                        ));
                    }
                    PollResult::Error(e) => {
                        return Err(SupervisorError::WorkerCrashed(format!("Pipe error: {}", e)));
                    }
                }
            }

            // Read next message (blocking, but poll confirmed data is available)
            let msg: WorkerMessage = match self.reader.read() {
                Ok(msg) => msg,
                Err(FrameError::EndOfStream) => {
                    return Err(SupervisorError::WorkerCrashed(
                        "Worker closed connection unexpectedly".to_string(),
                    ));
                }
                Err(e) => {
                    if !self.is_alive() {
                        return Err(SupervisorError::WorkerCrashed(
                            "Worker crashed during read".to_string(),
                        ));
                    }
                    return Err(SupervisorError::IpcError(e.to_string()));
                }
            };

            match msg {
                WorkerMessage::SampleBatch(batch) => {
                    all_samples.extend(batch.samples);
                }
                WorkerMessage::WarmupComplete { .. } => {
                    // Warmup done, measurement phase starting
                    continue;
                }
                WorkerMessage::Progress { .. } => {
                    // Progress update, continue
                    continue;
                }
                WorkerMessage::Complete {
                    total_iterations,
                    total_duration_nanos,
                } => {
                    return Ok(IpcBenchmarkResult {
                        bench_id: bench_id.to_string(),
                        samples: all_samples,
                        total_iterations,
                        total_duration_nanos,
                        status: IpcBenchmarkStatus::Success,
                    });
                }
                WorkerMessage::Failure {
                    kind,
                    message,
                    backtrace: _,
                } => {
                    return Ok(IpcBenchmarkResult {
                        bench_id: bench_id.to_string(),
                        samples: all_samples,
                        total_iterations: 0,
                        total_duration_nanos: 0,
                        status: match kind {
                            fluxbench_ipc::FailureKind::Panic => {
                                IpcBenchmarkStatus::Crashed { message }
                            }
                            _ => IpcBenchmarkStatus::Failed { message },
                        },
                    });
                }
                WorkerMessage::Hello(_) => {
                    return Err(SupervisorError::ProtocolError {
                        expected: "SampleBatch/Complete/Failure".to_string(),
                        got: "Hello".to_string(),
                    });
                }
            }
        }
    }

    /// Ping the worker to check if it's alive
    pub fn ping(&mut self) -> Result<bool, SupervisorError> {
        self.writer.write(&SupervisorCommand::Ping)?;
        // Worker doesn't respond to ping, just continues
        Ok(true)
    }

    /// Abort the current benchmark
    pub fn abort(&mut self) -> Result<(), SupervisorError> {
        self.writer.write(&SupervisorCommand::Abort)?;
        Ok(())
    }

    /// Shutdown the worker gracefully
    pub fn shutdown(mut self) -> Result<(), SupervisorError> {
        self.writer.write(&SupervisorCommand::Shutdown)?;
        // Wait for process to exit
        let _ = self.child.wait();
        Ok(())
    }

    /// Check if worker process is still running
    pub fn is_alive(&mut self) -> bool {
        match self.child.try_wait() {
            Ok(Some(_)) => false, // Exited
            Ok(None) => true,     // Still running
            Err(_) => false,      // Error, assume dead
        }
    }

    /// Kill the worker process forcefully
    pub fn kill(&mut self) -> Result<(), SupervisorError> {
        self.child.kill().map_err(SupervisorError::SpawnFailed)?;
        let _ = self.child.wait();
        Ok(())
    }
}

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        // Try to kill the worker if it's still running
        if self.is_alive() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

/// Supervisor that manages worker pool and distributes benchmarks
pub struct Supervisor {
    config: BenchmarkConfig,
    timeout: Duration,
    num_workers: usize,
}

impl Supervisor {
    /// Create a new supervisor
    pub fn new(config: BenchmarkConfig, timeout: Duration, num_workers: usize) -> Self {
        Self {
            config,
            timeout,
            num_workers: num_workers.max(1),
        }
    }

    /// Run all benchmarks with process isolation
    pub fn run_all(
        &self,
        benchmarks: &[&BenchmarkDef],
    ) -> Result<Vec<IpcBenchmarkResult>, SupervisorError> {
        if benchmarks.is_empty() {
            return Ok(Vec::new());
        }

        // Fast path for single-worker or single-benchmark execution.
        if self.num_workers == 1 || benchmarks.len() == 1 {
            let mut results = Vec::with_capacity(benchmarks.len());
            for bench in benchmarks {
                results.push(self.run_isolated(bench)?);
            }
            return Ok(results);
        }

        let worker_count = self.num_workers.min(benchmarks.len());
        let pool = ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .build()
            .map_err(|e| {
                SupervisorError::IpcError(format!("Failed to build worker pool: {}", e))
            })?;

        let outcomes: Vec<Result<IpcBenchmarkResult, SupervisorError>> = pool.install(|| {
            benchmarks
                .par_iter()
                .map(|bench| self.run_isolated(bench))
                .collect()
        });

        let mut results = Vec::with_capacity(outcomes.len());
        for outcome in outcomes {
            results.push(outcome?);
        }
        Ok(results)
    }

    /// Run a single benchmark in an isolated worker process
    fn run_isolated(&self, bench: &BenchmarkDef) -> Result<IpcBenchmarkResult, SupervisorError> {
        // Spawn a fresh worker for this benchmark
        let mut worker = WorkerHandle::spawn(self.timeout)?;

        // Run the benchmark
        let result = worker.run_benchmark(bench.id, &self.config);

        // Always try to shutdown cleanly
        let _ = worker.shutdown();

        result
    }

    fn crashed_result(bench: &BenchmarkDef, message: String) -> IpcBenchmarkResult {
        IpcBenchmarkResult {
            bench_id: bench.id.to_string(),
            samples: Vec::new(),
            total_iterations: 0,
            total_duration_nanos: 0,
            status: IpcBenchmarkStatus::Crashed { message },
        }
    }

    fn run_with_reuse_indexed(
        &self,
        benchmarks: &[(usize, &BenchmarkDef)],
    ) -> Vec<(usize, IpcBenchmarkResult)> {
        let mut results = Vec::with_capacity(benchmarks.len());
        if benchmarks.is_empty() {
            return results;
        }

        let mut worker = match WorkerHandle::spawn(self.timeout) {
            Ok(worker) => Some(worker),
            Err(e) => {
                let message = e.to_string();
                for &(index, bench) in benchmarks {
                    results.push((index, Self::crashed_result(bench, message.clone())));
                }
                return results;
            }
        };

        for &(index, bench) in benchmarks {
            // Re-spawn worker lazily if a previous benchmark crashed it.
            if worker.is_none() {
                match WorkerHandle::spawn(self.timeout) {
                    Ok(new_worker) => worker = Some(new_worker),
                    Err(e) => {
                        results.push((index, Self::crashed_result(bench, e.to_string())));
                        continue;
                    }
                }
            }

            let run_result = match worker.as_mut() {
                Some(worker) => worker.run_benchmark(bench.id, &self.config),
                None => unreachable!("worker should exist after spawn check"),
            };

            match run_result {
                Ok(result) => results.push((index, result)),
                Err(e) => {
                    let worker_is_alive = worker.as_mut().map(|w| w.is_alive()).unwrap_or(false);
                    if !worker_is_alive {
                        if let Some(mut dead_worker) = worker.take() {
                            let _ = dead_worker.kill();
                        }
                    }
                    results.push((index, Self::crashed_result(bench, e.to_string())));
                }
            }
        }

        if let Some(worker) = worker {
            let _ = worker.shutdown();
        }

        results
    }

    /// Run benchmarks with worker reuse (less isolation but faster)
    pub fn run_with_reuse(
        &self,
        benchmarks: &[&BenchmarkDef],
    ) -> Result<Vec<IpcBenchmarkResult>, SupervisorError> {
        if benchmarks.is_empty() {
            return Ok(Vec::new());
        }

        let indexed_benchmarks: Vec<(usize, &BenchmarkDef)> = benchmarks
            .iter()
            .enumerate()
            .map(|(index, bench)| (index, *bench))
            .collect();

        let mut indexed_results = if self.num_workers == 1 || benchmarks.len() == 1 {
            self.run_with_reuse_indexed(&indexed_benchmarks)
        } else {
            let worker_count = self.num_workers.min(indexed_benchmarks.len());
            let mut shards: Vec<Vec<(usize, &BenchmarkDef)>> = vec![Vec::new(); worker_count];
            for (position, entry) in indexed_benchmarks.into_iter().enumerate() {
                shards[position % worker_count].push(entry);
            }

            let pool = ThreadPoolBuilder::new()
                .num_threads(worker_count)
                .build()
                .map_err(|e| {
                    SupervisorError::IpcError(format!("Failed to build worker pool: {}", e))
                })?;

            let shard_results: Vec<Vec<(usize, IpcBenchmarkResult)>> = pool.install(|| {
                shards
                    .into_par_iter()
                    .map(|shard| self.run_with_reuse_indexed(&shard))
                    .collect()
            });

            shard_results.into_iter().flatten().collect()
        };

        indexed_results.sort_by_key(|(index, _)| *index);
        if indexed_results.len() != benchmarks.len() {
            return Err(SupervisorError::IpcError(format!(
                "Internal error: expected {} results, got {}",
                benchmarks.len(),
                indexed_results.len()
            )));
        }

        Ok(indexed_results
            .into_iter()
            .map(|(_, result)| result)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require a built flux-examples binary
    // They are integration tests that should be run manually

    #[test]
    #[ignore] // Requires built binary
    fn test_supervisor_spawn() {
        let timeout = Duration::from_secs(30);
        let config = BenchmarkConfig::default();
        let supervisor = Supervisor::new(config, timeout, 1);
        // Just verify it compiles
        assert_eq!(supervisor.num_workers, 1);
    }
}
