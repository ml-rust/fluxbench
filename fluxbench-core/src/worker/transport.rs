//! IPC transport: an inherited fd pair, or stdin/stdout as a fallback.

use super::stream::IpcWriter;
use fluxbench_ipc::{FrameReader, FrameWriter};
use std::io::{Read, Write};

#[cfg(unix)]
use std::os::unix::io::FromRawFd;

/// Reader for supervisor commands.
pub(super) type IpcReader = FrameReader<Box<dyn Read>>;

/// IPC transport the supervisor chose.
enum IpcTransport {
    #[cfg(unix)]
    Fds {
        read_fd: i32,
        write_fd: i32,
    },
    Stdio,
}

/// Read `FLUX_IPC_FD` (`<read_fd>,<write_fd>`). Fall back to stdio when absent or malformed.
fn detect_transport() -> IpcTransport {
    #[cfg(unix)]
    if let Ok(val) = std::env::var("FLUX_IPC_FD") {
        let parts: Vec<&str> = val.split(',').collect();
        if parts.len() == 2 {
            if let (Ok(r), Ok(w)) = (parts[0].parse::<i32>(), parts[1].parse::<i32>()) {
                return IpcTransport::Fds {
                    read_fd: r,
                    write_fd: w,
                };
            }
        }
        eprintln!(
            "fluxbench: warning: invalid FLUX_IPC_FD={val:?} (expected format: <read_fd>,<write_fd>), falling back to stdio"
        );
    }
    IpcTransport::Stdio
}

/// Open the command reader and the message writer.
pub(super) fn open() -> (IpcReader, IpcWriter) {
    match detect_transport() {
        #[cfg(unix)]
        IpcTransport::Fds { read_fd, write_fd } => {
            // SAFETY: the supervisor passes these fds open and owned by this process.
            let read_file = unsafe { std::fs::File::from_raw_fd(read_fd) };
            // SAFETY: as above.
            let write_file = unsafe { std::fs::File::from_raw_fd(write_fd) };
            (
                FrameReader::new(Box::new(read_file) as Box<dyn Read>),
                FrameWriter::new(Box::new(write_file) as Box<dyn Write + Send>),
            )
        }
        IpcTransport::Stdio => (
            FrameReader::new(Box::new(std::io::stdin()) as Box<dyn Read>),
            FrameWriter::new(Box::new(std::io::stdout()) as Box<dyn Write + Send>),
        ),
    }
}
