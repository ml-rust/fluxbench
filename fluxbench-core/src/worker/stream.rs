//! Shared IPC writer with a sample ring buffer.
//!
//! The measurement thread pushes samples through `StreamSink`.
//! The flusher thread sends batches that are due.
//! Every frame goes out under one lock, so frames never interleave.

use crate::bencher::SampleSink;
use fluxbench_ipc::{FrameError, FrameWriter, Sample, SampleRingBuffer, WorkerMessage};
use std::io::Write;
use std::sync::{Arc, Mutex, MutexGuard};

/// Writer for worker messages.
pub(super) type IpcWriter = FrameWriter<Box<dyn Write + Send>>;

/// Writer state behind the lock.
struct Stream {
    writer: IpcWriter,
    /// Pending samples of the running benchmark. `None` between runs.
    ring: Option<SampleRingBuffer>,
    /// First write error during streaming. `end` returns it.
    error: Option<FrameError>,
}

impl Stream {
    /// Write one message. After a write error, skip writes: the stream is broken.
    fn write(&mut self, message: &WorkerMessage) {
        if self.error.is_some() {
            return;
        }
        if let Err(e) = self.writer.write(message) {
            self.error = Some(e);
        }
    }
}

/// Cloneable handle to the shared writer.
#[derive(Clone)]
pub(super) struct SharedStream(Arc<Mutex<Stream>>);

impl SharedStream {
    pub(super) fn new(writer: IpcWriter) -> Self {
        Self(Arc::new(Mutex::new(Stream {
            writer,
            ring: None,
            error: None,
        })))
    }

    /// Lock the stream. A panic never happens under this lock, but poison still recovers.
    fn lock(&self) -> MutexGuard<'_, Stream> {
        match self.0.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    /// Send one message now.
    pub(super) fn send(&self, message: &WorkerMessage) -> Result<(), FrameError> {
        let mut stream = self.lock();
        if let Some(e) = stream.error.take() {
            return Err(e);
        }
        stream.writer.write(message)
    }

    /// Start buffering samples for `bench_id`.
    pub(super) fn begin(&self, bench_id: &str) {
        let mut stream = self.lock();
        stream.ring = Some(SampleRingBuffer::new(bench_id));
        stream.error = None;
    }

    /// Buffer one sample. Send a batch when it is full or the heartbeat is due.
    pub(super) fn push(&self, sample: Sample) {
        let mut stream = self.lock();
        let Some(ring) = stream.ring.as_mut() else {
            return;
        };
        let batch = ring.push(sample).or_else(|| ring.check_timeout());
        if let Some(batch) = batch {
            stream.write(&WorkerMessage::SampleBatch(batch));
        }
    }

    /// Send pending samples when the heartbeat is due, or at once on a stop request.
    pub(super) fn flush_due(&self, stop_requested: bool) {
        let mut stream = self.lock();
        let Some(ring) = stream.ring.as_mut() else {
            return;
        };
        let batch = if stop_requested {
            ring.flush_shutdown()
        } else {
            ring.check_timeout()
        };
        if let Some(batch) = batch {
            stream.write(&WorkerMessage::SampleBatch(batch));
        }
    }

    /// Send the pending samples, then `last`, and stop buffering.
    ///
    /// Returns the first write error of this run.
    pub(super) fn end(&self, last: &WorkerMessage) -> Result<(), FrameError> {
        let mut stream = self.lock();
        if let Some(batch) = stream.ring.take().and_then(|mut ring| ring.flush_final()) {
            stream.write(&WorkerMessage::SampleBatch(batch));
        }
        stream.write(last);
        match stream.error.take() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

/// Sample sink that streams into a `SharedStream`.
pub(super) struct StreamSink(pub(super) SharedStream);

impl SampleSink for StreamSink {
    fn on_sample(&mut self, sample: &Sample) {
        self.0.push(*sample);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxbench_ipc::{FlushReason, FrameReader};
    use std::io::Cursor;

    /// Writer that appends into a shared buffer.
    #[derive(Clone, Default)]
    struct Shared(Arc<Mutex<Vec<u8>>>);

    impl Write for Shared {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().expect("lock").extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn stream() -> (SharedStream, Shared) {
        let buffer = Shared::default();
        let writer = FrameWriter::new(Box::new(buffer.clone()) as Box<dyn Write + Send>);
        (SharedStream::new(writer), buffer)
    }

    fn messages(buffer: &Shared) -> Vec<WorkerMessage> {
        let bytes = buffer.0.lock().expect("lock").clone();
        let mut reader = FrameReader::new(Cursor::new(bytes));
        let mut out = Vec::new();
        while let Ok(msg) = reader.read::<WorkerMessage>() {
            out.push(msg);
        }
        out
    }

    #[test]
    fn end_sends_pending_samples_then_last_message() {
        let (stream, buffer) = stream();
        stream.begin("b");
        for i in 0..5 {
            stream.push(Sample::timing_only(i as f64));
        }
        stream
            .end(&WorkerMessage::Complete {
                total_iterations: 5,
                total_duration_nanos: 10,
            })
            .expect("end succeeds");
        let msgs = messages(&buffer);
        let sampled: usize = msgs
            .iter()
            .map(|m| match m {
                WorkerMessage::SampleBatch(b) => b.samples.len(),
                _ => 0,
            })
            .sum();
        assert_eq!(sampled, 5);
        assert!(matches!(msgs.last(), Some(WorkerMessage::Complete { .. })));
    }

    #[test]
    fn stop_request_flushes_pending_samples() {
        let (stream, buffer) = stream();
        stream.begin("b");
        stream.push(Sample::timing_only(1.0));
        stream.push(Sample::timing_only(2.0));
        stream.flush_due(true);
        let msgs = messages(&buffer);
        match msgs.as_slice() {
            [WorkerMessage::SampleBatch(batch)] => {
                assert_eq!(batch.samples.len(), 2);
                assert_eq!(batch.flush_reason, FlushReason::Shutdown);
            }
            other => panic!("expected one batch, got {other:?}"),
        }
    }

    #[test]
    fn push_outside_a_run_sends_nothing() {
        let (stream, buffer) = stream();
        stream.push(Sample::timing_only(1.0));
        stream.flush_due(true);
        assert!(messages(&buffer).is_empty());
    }
}
