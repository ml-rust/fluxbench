#![warn(missing_docs)]
//! FluxBench IPC Protocol
//!
//! Zero-copy serialization protocol for supervisor-worker communication.
//! Uses rkyv for efficient binary serialization with no parsing overhead.
//! Handles benchmark configuration, sample collection, and heartbeat coordination.

mod framing;
mod messages;
mod ring_buffer;
mod throughput;

pub use framing::{FrameError, FrameReader, FrameWriter, MAX_FRAME_SIZE, read_frame, write_frame};
pub use messages::{
    BenchmarkConfig, DEFAULT_SAMPLE_COUNT, FailureKind, FlushReason, Sample, SampleBatch,
    SupervisorCommand, WorkerCapabilities, WorkerMessage,
};
pub use ring_buffer::SampleRingBuffer;
pub use throughput::Throughput;

/// Protocol version for compatibility checking.
///
/// Version 2 carries `f64` sample means and the per-sample iteration count.
/// Version 3 adds `FailureKind::Signal(name)` and `FailureKind::Crashed`.
/// Version 4 drops `BenchmarkConfig::fail_on_allocation`. The supervisor checks allocations.
/// The worker streams sample batches during measurement.
/// Version 5 adds `WorkerMessage::Complete::throughput`, the value `Bencher::throughput` set.
pub const PROTOCOL_VERSION: u32 = 5;

/// Env var the supervisor sets to the CPU index a worker pins to.
///
/// Absent means the worker does not pin.
pub const PIN_CPU_ENV: &str = "FLUX_PIN_CPU";

/// Maximum samples per batch (prevents unbounded memory growth)
pub const MAX_BATCH_SIZE: usize = 10_000;

/// Maximum batch size in bytes (64KB)
pub const MAX_BATCH_BYTES: usize = 64 * 1024;

/// Heartbeat interval in nanoseconds (100ms)
pub const HEARTBEAT_INTERVAL_NS: u64 = 100_000_000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_size() {
        // Four f64 means plus a u64 iteration count
        assert_eq!(std::mem::size_of::<Sample>(), 40);
    }

    #[test]
    fn test_sample_alignment() {
        // Sample must be 8-byte aligned
        assert_eq!(std::mem::align_of::<Sample>(), 8);
    }
}
