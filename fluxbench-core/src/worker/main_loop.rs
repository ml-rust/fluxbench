//! Worker main loop.
//!
//! On Unix, the IPC fds come from `FLUX_IPC_FD`, and a SIGTERM handler requests a graceful stop.
//! On non-Unix, IPC falls back to stdin/stdout and there is no signal handling.
//!
//! Samples stream to the supervisor during measurement.
//! So a timeout or crash keeps the samples sent before it.

use super::flusher::Flusher;
use super::panic_report;
use super::stream::{SharedStream, StreamSink};
use super::transport::{self, IpcReader};
use crate::measure::{cycles_overhead, pin_to_cpu, timer_overhead_ns};
use crate::shutdown::{install_sigterm_handler, shutdown_requested};
use crate::{Bencher, BenchmarkDef, MeasureConfig, run_benchmark_loop};
use fluxbench_ipc::{
    BenchmarkConfig, FailureKind, SupervisorCommand, WorkerCapabilities, WorkerMessage,
};

/// Worker main loop
pub struct WorkerMain {
    reader: IpcReader,
    stream: SharedStream,
}

impl WorkerMain {
    /// Create a worker on the fds in `FLUX_IPC_FD`, else on stdin/stdout.
    pub fn new() -> Self {
        let (reader, writer) = transport::open();
        Self {
            reader,
            stream: SharedStream::new(writer),
        }
    }

    /// Run the worker main loop
    pub fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        install_sigterm_handler();
        panic_report::install();

        self.stream
            .send(&WorkerMessage::Hello(WorkerCapabilities::default()))?;

        // Start before pinning, so the flusher does not share the pinned core.
        let _flusher = Flusher::start(self.stream.clone());

        // Pin to CPU 0 for stable TSC
        let _ = pin_to_cpu(0);

        // Calibrate timer overhead on the pinned core before any benchmark runs
        let _ = timer_overhead_ns();
        let _ = cycles_overhead();

        loop {
            if shutdown_requested() {
                break;
            }

            let command: SupervisorCommand = self.reader.read()?;

            match command {
                SupervisorCommand::Run { bench_id, config } => {
                    self.run_benchmark(&bench_id, &config)?;
                    if shutdown_requested() {
                        break;
                    }
                }
                SupervisorCommand::Abort | SupervisorCommand::Shutdown => break,
                SupervisorCommand::Ping => {}
            }
        }

        Ok(())
    }

    /// Run one benchmark and send its samples and outcome.
    fn run_benchmark(
        &mut self,
        bench_id: &str,
        config: &BenchmarkConfig,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let bench = inventory::iter::<BenchmarkDef>
            .into_iter()
            .find(|b| b.id == bench_id);

        let Some(bench) = bench else {
            self.stream.send(&WorkerMessage::Failure {
                kind: FailureKind::Unknown,
                message: format!(
                    "benchmark {bench_id:?} not found in the worker binary. Rebuild the bench binary and run again."
                ),
                backtrace: None,
            })?;
            return Ok(());
        };

        self.stream.begin(bench_id);
        panic_report::clear();

        let sink = StreamSink(self.stream.clone());
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let bencher = Bencher::with_config(MeasureConfig::from(config)).with_sample_sink(sink);
            run_benchmark_loop(bencher, |b| (bench.runner_fn)(b))
        }));

        let last = match result {
            Ok(Ok(bench_result)) => {
                if bench_result.truncated {
                    let reason = if shutdown_requested() {
                        "stop requested"
                    } else {
                        "deadline reached"
                    };
                    eprintln!(
                        "fluxbench: warning: {bench_id}: {reason}, {} of {} samples kept",
                        bench_result.samples.len(),
                        config.sample_count
                    );
                }
                // Samples already went out through the sink.
                WorkerMessage::Complete {
                    total_iterations: bench_result.iterations,
                    total_duration_nanos: bench_result.total_time_ns,
                }
            }
            Ok(Err(error)) => WorkerMessage::Failure {
                kind: FailureKind::Unknown,
                message: error.to_string(),
                backtrace: None,
            },
            Err(payload) => match panic_report::take() {
                Some(report) => WorkerMessage::Failure {
                    kind: FailureKind::Panic,
                    message: report.message_with_location(),
                    backtrace: Some(report.backtrace),
                },
                None => WorkerMessage::Failure {
                    kind: FailureKind::Panic,
                    message: panic_report::payload_message(payload.as_ref()),
                    backtrace: None,
                },
            },
        };

        self.stream.end(&last)?;
        Ok(())
    }
}

impl Default for WorkerMain {
    fn default() -> Self {
        Self::new()
    }
}
