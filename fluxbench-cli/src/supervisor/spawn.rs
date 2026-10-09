//! Worker spawn.
//!
//! On Unix, the supervisor creates two pipes and passes their fd numbers in `FLUX_IPC_FD`.
//! This leaves stdout and stderr free for user benchmark code.
//! On other platforms, IPC falls back to stdin/stdout pipes.
//! There, user `println!` can corrupt the protocol stream.

use super::error::SupervisorError;
use super::handle::{IpcReader, IpcWriter, WorkerHandle};
use super::hello::{WORKER_HELLO_TIMEOUT, read_hello_within};
use super::process::kill_and_reap;
use fluxbench_ipc::{FrameReader, FrameWriter, PIN_CPU_ENV};
use std::env;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

#[cfg(unix)]
use super::ipc_io::{close_fd, create_pipe};
#[cfg(unix)]
use std::os::unix::io::FromRawFd;
#[cfg(unix)]
use std::os::unix::process::CommandExt;

impl WorkerHandle {
    /// Spawn a worker from the current executable.
    ///
    /// `cpu` is the core the worker pins to. `None` leaves the worker unpinned.
    pub fn spawn(timeout: Duration, cpu: Option<usize>) -> Result<Self, SupervisorError> {
        let binary = env::current_exe().map_err(SupervisorError::SpawnFailed)?;
        spawn_impl(&binary, timeout, cpu)
    }

    /// Spawn a worker from a specific binary (for testing).
    pub fn spawn_binary(binary: &str, timeout: Duration) -> Result<Self, SupervisorError> {
        spawn_impl(binary.as_ref(), timeout, None)
    }
}

/// Wait for `Hello`, then build the handle.
///
/// On any handshake error, kill and reap the child and close `poll_fd`.
fn handshake(
    mut child: Child,
    reader: IpcReader,
    writer: IpcWriter,
    poll_fd: i32,
    timeout: Duration,
) -> Result<WorkerHandle, SupervisorError> {
    match read_hello_within(reader, WORKER_HELLO_TIMEOUT) {
        Ok((reader, caps)) => Ok(WorkerHandle::from_parts(
            child, reader, writer, caps, timeout, poll_fd,
        )),
        Err(e) => {
            let status = kill_and_reap(&mut child);
            #[cfg(unix)]
            if poll_fd >= 0 {
                close_fd(poll_fd);
            }
            Err(match (e, status) {
                (SupervisorError::IpcError(msg), Some(status)) => {
                    let (_, cause) = super::failure::classify_exit(status);
                    SupervisorError::WorkerCrashed(format!("{cause} during handshake: {msg}"))
                }
                (e, _) => e,
            })
        }
    }
}

#[cfg(unix)]
fn spawn_impl(
    binary: &Path,
    timeout: Duration,
    cpu: Option<usize>,
) -> Result<WorkerHandle, SupervisorError> {
    // cmd pipe: supervisor writes commands, worker reads from cmd_read.
    let (cmd_read, cmd_write) = create_pipe()?;
    // msg pipe: worker writes to msg_write, supervisor reads from msg_read.
    let (msg_read, msg_write) = match create_pipe() {
        Ok(fds) => fds,
        Err(e) => {
            close_fd(cmd_read);
            close_fd(cmd_write);
            return Err(SupervisorError::SpawnFailed(e));
        }
    };

    // Pass the real fd numbers. Fixed fds 3/4 can collide with library static initializers.
    let mut command = Command::new(binary);
    command
        .arg("--flux-worker")
        .env("FLUX_IPC_FD", format!("{},{}", cmd_read, msg_write))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit());
    if let Some(cpu) = cpu {
        command.env(PIN_CPU_ENV, cpu.to_string());
    }

    // In the child: keep the child-side fds across exec and close the parent-side fds.
    // SAFETY: the closure only calls async-signal-safe `fcntl` and `close`.
    unsafe {
        command.pre_exec(move || {
            for fd in [cmd_read, msg_write] {
                let flags = libc::fcntl(fd, libc::F_GETFD);
                if flags == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::fcntl(fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC) == -1 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            libc::close(cmd_write);
            libc::close(msg_read);
            Ok(())
        });
    }

    let child = match command.spawn() {
        Ok(c) => c,
        Err(e) => {
            for fd in [cmd_read, cmd_write, msg_read, msg_write] {
                close_fd(fd);
            }
            return Err(SupervisorError::SpawnFailed(e));
        }
    };

    // Close the child-side ends in the parent.
    close_fd(cmd_read);
    close_fd(msg_write);

    // SAFETY: msg_read and cmd_write are open fds this process owns.
    // `from_raw_fd` takes ownership and closes them on drop.
    let reader_file = unsafe { std::fs::File::from_raw_fd(msg_read) };
    // SAFETY: as above.
    let writer_file = unsafe { std::fs::File::from_raw_fd(cmd_write) };

    // Duplicate the read fd for poll(2), so it never aliases the File-owned fd.
    // WorkerHandle::drop closes it.
    // SAFETY: msg_read is open; `dup` returns a new fd or -1.
    let poll_fd = unsafe { libc::dup(msg_read) };
    if poll_fd < 0 {
        let error = std::io::Error::last_os_error();
        let mut child = child;
        let _ = kill_and_reap(&mut child);
        return Err(SupervisorError::SpawnFailed(error));
    }

    handshake(
        child,
        FrameReader::new(Box::new(reader_file) as Box<dyn std::io::Read + Send>),
        FrameWriter::new(Box::new(writer_file) as Box<dyn std::io::Write + Send>),
        poll_fd,
        timeout,
    )
}

#[cfg(not(unix))]
fn spawn_impl(
    binary: &Path,
    timeout: Duration,
    cpu: Option<usize>,
) -> Result<WorkerHandle, SupervisorError> {
    let mut command = Command::new(binary);
    command
        .arg("--flux-worker")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    if let Some(cpu) = cpu {
        command.env(PIN_CPU_ENV, cpu.to_string());
    }
    let mut child = command.spawn()?;

    let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
        let _ = kill_and_reap(&mut child);
        return Err(SupervisorError::IpcError(
            "worker stdin or stdout pipe missing after spawn. Run again.".to_string(),
        ));
    };

    handshake(
        child,
        FrameReader::new(Box::new(stdout) as Box<dyn std::io::Read + Send>),
        FrameWriter::new(Box::new(stdin) as Box<dyn std::io::Write + Send>),
        -1,
        timeout,
    )
}
