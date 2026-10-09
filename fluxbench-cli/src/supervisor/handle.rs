//! Worker handle: run one benchmark over IPC and classify how it ended.
//!
//! Timeout on Unix: SIGTERM, drain samples for 500 ms, then SIGKILL.
//! Timeout elsewhere: kill at once. Samples already received stay.

use super::error::SupervisorError;
use super::failure::{classify_exit, exit_message, timeout_message};
use super::ipc_io::{self, PollResult, Readiness};
use super::process::{EXIT_GRACE, is_running, kill_and_reap, reap};
use super::result::{IpcBenchmarkResult, IpcBenchmarkStatus};
use fluxbench_ipc::{
    BenchmarkConfig, FailureKind, FrameError, FrameReader, FrameWriter, Sample, SupervisorCommand,
    WorkerCapabilities, WorkerMessage,
};
use std::process::Child;
use std::time::{Duration, Instant};

#[cfg(unix)]
use super::ipc_io::{close_fd, wait_for_data_fd};
#[cfg(unix)]
use super::process::send_sigterm;

/// Reader for worker messages.
pub(super) type IpcReader = FrameReader<Box<dyn std::io::Read + Send>>;
/// Writer for supervisor commands.
pub(super) type IpcWriter = FrameWriter<Box<dyn std::io::Write + Send>>;

/// Drain window after SIGTERM on timeout.
#[cfg(unix)]
const TIMEOUT_DRAIN: Duration = Duration::from_millis(500);

/// Grace before SIGKILL when a handle drops with a live worker.
const DROP_GRACE: Duration = Duration::from_millis(50);

/// Worker process handle.
///
/// The IPC reader and writer are `Send` trait objects.
/// So Unix (File-backed) and other (pipe-backed) paths share one struct.
/// The supervisor moves handles across its `rayon` threads.
pub struct WorkerHandle {
    child: Child,
    reader: IpcReader,
    writer: IpcWriter,
    capabilities: Option<WorkerCapabilities>,
    timeout: Duration,
    /// Duplicate of the read fd for `poll(2)` on Unix. -1 elsewhere.
    poll_fd: i32,
}

impl WorkerHandle {
    /// Build a handle after a successful handshake.
    pub(super) fn from_parts(
        child: Child,
        reader: IpcReader,
        writer: IpcWriter,
        capabilities: WorkerCapabilities,
        timeout: Duration,
        poll_fd: i32,
    ) -> Self {
        Self {
            child,
            reader,
            writer,
            capabilities: Some(capabilities),
            timeout,
            poll_fd,
        }
    }

    /// Get worker capabilities
    pub fn capabilities(&self) -> Option<&WorkerCapabilities> {
        self.capabilities.as_ref()
    }

    /// Run a benchmark on this worker.
    ///
    /// `Ok` carries every outcome of the benchmark: success, failure, panic, timeout, or crash.
    /// `Err` means the IPC stream is out of step. Discard the worker after it.
    pub fn run_benchmark(
        &mut self,
        bench_id: &str,
        config: &BenchmarkConfig,
    ) -> Result<IpcBenchmarkResult, SupervisorError> {
        self.send_run(bench_id, config)?;
        self.collect(bench_id)
    }

    /// Send the `Run` command.
    pub fn send_run(
        &mut self,
        bench_id: &str,
        config: &BenchmarkConfig,
    ) -> Result<(), SupervisorError> {
        self.writer.write(&SupervisorCommand::Run {
            bench_id: bench_id.to_string(),
            config: config.clone(),
        })?;
        Ok(())
    }

    /// Collect samples and the outcome of the running benchmark.
    ///
    /// Same contract as `run_benchmark`.
    pub fn collect(&mut self, bench_id: &str) -> Result<IpcBenchmarkResult, SupervisorError> {
        let mut samples = Vec::new();
        let deadline = Instant::now() + self.timeout;

        loop {
            if Instant::now() >= deadline {
                return Ok(self.timeout_result(bench_id, samples));
            }
            // Buffered bytes hold at least part of the next frame. Read it without polling.
            if !self.reader.has_buffered_data() {
                match self.wait_ready(deadline) {
                    Readiness::Ready => {}
                    Readiness::Expired => return Ok(self.timeout_result(bench_id, samples)),
                    Readiness::Closed => return Ok(self.exit_result(bench_id, samples)),
                }
            }

            let msg = match self.reader.read::<WorkerMessage>() {
                Ok(msg) => msg,
                Err(FrameError::EndOfStream | FrameError::Io(_)) => {
                    return Ok(self.exit_result(bench_id, samples));
                }
                Err(e) => {
                    return Err(SupervisorError::IpcError(format!(
                        "decode of a worker message for {bench_id} failed: {e}. The worker is out of step; discard it."
                    )));
                }
            };

            match msg {
                WorkerMessage::SampleBatch(batch) => samples.extend(batch.samples),
                WorkerMessage::WarmupComplete { .. } | WorkerMessage::Progress { .. } => {}
                WorkerMessage::Complete {
                    total_iterations,
                    total_duration_nanos,
                } => {
                    return Ok(IpcBenchmarkResult {
                        bench_id: bench_id.to_string(),
                        samples,
                        total_iterations,
                        total_duration_nanos,
                        status: IpcBenchmarkStatus::Success,
                    });
                }
                WorkerMessage::Failure {
                    kind,
                    message,
                    backtrace,
                } => {
                    return Ok(IpcBenchmarkResult::failure(
                        bench_id, kind, message, backtrace, samples,
                    ));
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

    /// Wait for worker data until `deadline`.
    #[cfg(unix)]
    fn wait_ready(&mut self, deadline: Instant) -> Readiness {
        let fd = self.poll_fd;
        let child = &mut self.child;
        ipc_io::wait_ready(
            deadline,
            |slice| wait_for_data_fd(fd, slice),
            || is_running(child),
        )
    }

    /// Wait for worker data until `deadline`.
    ///
    /// Without `poll(2)`, this sleeps briefly and checks liveness.
    /// The read after it blocks until data arrives.
    #[cfg(not(unix))]
    fn wait_ready(&mut self, deadline: Instant) -> Readiness {
        let child = &mut self.child;
        ipc_io::wait_ready(
            deadline,
            |slice| {
                std::thread::sleep(slice.min(Duration::from_millis(10)));
                if is_running(child) {
                    PollResult::DataAvailable
                } else {
                    PollResult::PipeClosed
                }
            },
            || true,
        )
    }

    /// Read sample batches that are ready now. Stop at the first other message or error.
    fn drain_ready(&mut self, samples: &mut Vec<Sample>) {
        loop {
            if !self.reader.has_buffered_data() && !self.data_ready_now() {
                return;
            }
            match self.reader.read::<WorkerMessage>() {
                Ok(WorkerMessage::SampleBatch(batch)) => samples.extend(batch.samples),
                Ok(WorkerMessage::WarmupComplete { .. } | WorkerMessage::Progress { .. }) => {}
                Ok(_) | Err(_) => return,
            }
        }
    }

    /// `true` when the pipe has data to read without blocking.
    #[cfg(unix)]
    fn data_ready_now(&self) -> bool {
        matches!(
            wait_for_data_fd(self.poll_fd, Duration::ZERO),
            PollResult::DataAvailable
        )
    }

    /// Without `poll(2)` there is no non-blocking check.
    #[cfg(not(unix))]
    fn data_ready_now(&self) -> bool {
        false
    }

    /// Result for a worker that ended mid-run. Reaps it and reads its exit status.
    fn exit_result(&mut self, bench_id: &str, mut samples: Vec<Sample>) -> IpcBenchmarkResult {
        self.drain_ready(&mut samples);
        let (kind, cause) = match reap(&mut self.child, EXIT_GRACE) {
            Some(status) => classify_exit(status),
            None => (
                FailureKind::Crashed,
                "worker exited with unknown status".to_string(),
            ),
        };
        let message = exit_message(&cause, samples.len());
        IpcBenchmarkResult::failure(bench_id, kind, message, None, samples)
    }

    /// Result for a benchmark past the worker timeout.
    ///
    /// Unix: SIGTERM, drain samples for 500 ms, then SIGKILL and reap.
    #[cfg(unix)]
    fn timeout_result(&mut self, bench_id: &str, mut samples: Vec<Sample>) -> IpcBenchmarkResult {
        // The worker may already be gone. Ignore a failed signal.
        let _ = send_sigterm(self.child.id());

        let drain_deadline = Instant::now() + TIMEOUT_DRAIN;
        loop {
            if !self.reader.has_buffered_data() {
                match self.wait_ready(drain_deadline) {
                    Readiness::Ready => {}
                    Readiness::Expired | Readiness::Closed => break,
                }
            }
            match self.reader.read::<WorkerMessage>() {
                Ok(WorkerMessage::SampleBatch(batch)) => samples.extend(batch.samples),
                Ok(WorkerMessage::WarmupComplete { .. } | WorkerMessage::Progress { .. }) => {}
                Ok(_) | Err(_) => break,
            }
        }
        self.drain_ready(&mut samples);

        let _ = kill_and_reap(&mut self.child);
        self.timed_out(bench_id, samples)
    }

    /// Result for a benchmark past the worker timeout.
    ///
    /// Without SIGTERM, kill at once. Samples already received stay.
    #[cfg(not(unix))]
    fn timeout_result(&mut self, bench_id: &str, samples: Vec<Sample>) -> IpcBenchmarkResult {
        let _ = kill_and_reap(&mut self.child);
        self.timed_out(bench_id, samples)
    }

    fn timed_out(&self, bench_id: &str, samples: Vec<Sample>) -> IpcBenchmarkResult {
        let message = timeout_message(self.timeout, samples.len());
        IpcBenchmarkResult::failure(bench_id, FailureKind::Timeout, message, None, samples)
    }

    /// Ping the worker to check if it's alive
    pub fn ping(&mut self) -> Result<bool, SupervisorError> {
        self.writer.write(&SupervisorCommand::Ping)?;
        Ok(true)
    }

    /// Abort the current benchmark
    pub fn abort(&mut self) -> Result<(), SupervisorError> {
        self.writer.write(&SupervisorCommand::Abort)?;
        Ok(())
    }

    /// Shut the worker down. Waits up to 1 s, then kills it. Always reaps.
    pub fn shutdown(mut self) -> Result<(), SupervisorError> {
        let sent = self.writer.write(&SupervisorCommand::Shutdown);
        let _ = reap(&mut self.child, EXIT_GRACE);
        sent?;
        Ok(())
    }

    /// Check if worker process is still running. A finished worker is reaped.
    pub fn is_alive(&mut self) -> bool {
        is_running(&mut self.child)
    }

    /// Kill the worker process at once and reap it.
    pub fn kill(&mut self) -> Result<(), SupervisorError> {
        let _ = kill_and_reap(&mut self.child);
        Ok(())
    }
}

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        // Graceful first: SIGTERM, a short wait, then SIGKILL. Always reaps.
        #[cfg(unix)]
        if is_running(&mut self.child) {
            let _ = send_sigterm(self.child.id());
        }
        let _ = reap(&mut self.child, DROP_GRACE);

        #[cfg(unix)]
        if self.poll_fd >= 0 {
            close_fd(self.poll_fd);
        }
    }
}
