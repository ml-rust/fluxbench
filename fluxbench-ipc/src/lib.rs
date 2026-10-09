#![warn(missing_docs)]
//! FluxBench IPC Protocol
//!
//! Zero-copy serialization protocol for supervisor-worker communication.
//! Uses rkyv for efficient binary serialization with no parsing overhead.
//! Handles benchmark configuration, sample collection, and heartbeat coordination.

mod framing;
mod messages;
mod ring_buffer;

pub use framing::{FrameError, FrameReader, FrameWriter, MAX_FRAME_SIZE, read_frame, write_frame};
pub use messages::{
    BenchmarkConfig, DEFAULT_SAMPLE_COUNT, FailureKind, FlushReason, Sample, SampleBatch,
    SupervisorCommand, WorkerCapabilities, WorkerMessage,
};
pub use ring_buffer::SampleRingBuffer;

/// Protocol version for compatibility checking.
///
/// Version 2 carries `f64` sample means and the per-sample iteration count.
/// Version 3 adds `FailureKind::Signal(name)` and `FailureKind::Crashed`.
/// The worker streams sample batches during measurement.
pub const PROTOCOL_VERSION: u32 = 3;

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
