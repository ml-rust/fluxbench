//! Worker handshake: read and check the `Hello` message within a deadline.

use super::error::SupervisorError;
use fluxbench_ipc::{FrameReader, WorkerCapabilities, WorkerMessage};
use std::io::Read;
use std::sync::mpsc;
use std::time::Duration;

/// Deadline for a new worker to send `Hello`.
pub const WORKER_HELLO_TIMEOUT: Duration = Duration::from_secs(10);

/// Check the first worker message: it must be `Hello` with this protocol version.
pub(super) fn check_hello(msg: WorkerMessage) -> Result<WorkerCapabilities, SupervisorError> {
    match msg {
        WorkerMessage::Hello(caps) => {
            if caps.protocol_version != fluxbench_ipc::PROTOCOL_VERSION {
                return Err(SupervisorError::ProtocolError {
                    expected: format!("protocol version {}", fluxbench_ipc::PROTOCOL_VERSION),
                    got: format!("protocol version {}", caps.protocol_version),
                });
            }
            Ok(caps)
        }
        other => Err(SupervisorError::ProtocolError {
            expected: "Hello".to_string(),
            got: format!("{:?}", other),
        }),
    }
}

/// Read and check `Hello` within `limit`. Returns the reader on success.
///
/// A helper thread does the blocking read, so a silent worker cannot hang the caller.
/// On a timeout the thread keeps the reader.
/// The caller then kills the worker, which ends the read and the thread.
pub(super) fn read_hello_within<R>(
    mut reader: FrameReader<R>,
    limit: Duration,
) -> Result<(FrameReader<R>, WorkerCapabilities), SupervisorError>
where
    R: Read + Send + 'static,
{
    let (tx, rx) = mpsc::channel();
    std::thread::Builder::new()
        .name("fluxbench-hello".to_string())
        .spawn(move || {
            let msg = reader.read::<WorkerMessage>();
            let _ = tx.send((reader, msg));
        })
        .map_err(SupervisorError::SpawnFailed)?;

    match rx.recv_timeout(limit) {
        Ok((reader, Ok(msg))) => Ok((reader, check_hello(msg)?)),
        Ok((_, Err(e))) => Err(SupervisorError::IpcError(format!(
            "worker handshake read failed: {e}. Check the worker stderr above."
        ))),
        Err(mpsc::RecvTimeoutError::Timeout) => Err(SupervisorError::HelloTimeout(limit)),
        Err(mpsc::RecvTimeoutError::Disconnected) => Err(SupervisorError::IpcError(
            "worker handshake thread ended without a result. Run again.".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxbench_ipc::FrameWriter;
    use std::io::Cursor;
    use std::time::Instant;

    /// Reader that blocks until its sender drops.
    struct Blocking(mpsc::Receiver<()>);

    impl Read for Blocking {
        fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
            let _ = self.0.recv();
            Ok(0)
        }
    }

    fn frames(msg: &WorkerMessage) -> Vec<u8> {
        let mut buffer = Vec::new();
        FrameWriter::new(&mut buffer).write(msg).expect("write");
        buffer
    }

    #[test]
    fn check_hello_accepts_current_protocol() {
        let caps = WorkerCapabilities::default();
        assert!(check_hello(WorkerMessage::Hello(caps)).is_ok());
    }

    #[test]
    fn check_hello_rejects_v1_worker() {
        let caps = WorkerCapabilities {
            protocol_version: 1,
            ..WorkerCapabilities::default()
        };
        let err = check_hello(WorkerMessage::Hello(caps));
        assert!(matches!(
            err,
            Err(SupervisorError::ProtocolError { ref got, .. }) if got == "protocol version 1"
        ));
    }

    #[test]
    fn check_hello_rejects_non_hello() {
        let msg = WorkerMessage::Progress {
            completed: 0,
            estimated_total: 0,
        };
        assert!(matches!(
            check_hello(msg),
            Err(SupervisorError::ProtocolError { .. })
        ));
    }

    #[test]
    fn hello_within_deadline_succeeds() {
        let bytes = frames(&WorkerMessage::Hello(WorkerCapabilities::default()));
        let reader = FrameReader::new(Cursor::new(bytes));
        let result = read_hello_within(reader, Duration::from_secs(5));
        assert!(result.is_ok());
    }

    #[test]
    fn silent_worker_times_out() {
        let (keep, rx) = mpsc::channel::<()>();
        let reader = FrameReader::new(Blocking(rx));
        let start = Instant::now();
        let result = read_hello_within(reader, Duration::from_millis(50));
        assert!(matches!(result, Err(SupervisorError::HelloTimeout(_))));
        assert!(start.elapsed() < Duration::from_secs(5));
        // Unblock the helper thread.
        drop(keep);
    }

    #[test]
    fn closed_stream_is_an_ipc_error() {
        let reader = FrameReader::new(Cursor::new(Vec::new()));
        let result = read_hello_within(reader, Duration::from_secs(5));
        assert!(matches!(result, Err(SupervisorError::IpcError(_))));
    }
}
