//! Panic hook that records message, location and backtrace at panic time.
//!
//! A backtrace taken after `catch_unwind` returns shows the catch site, not the panic.
//! So the hook captures it while the panicking frames still exist.

use std::any::Any;
use std::backtrace::Backtrace;
use std::cell::RefCell;
use std::sync::Once;

/// What the hook saw at panic time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PanicReport {
    /// Panic payload text
    pub message: String,
    /// `file:line:column` of the panic, when known
    pub location: Option<String>,
    /// Backtrace of the panicking thread
    pub backtrace: String,
}

impl PanicReport {
    /// Message with the location appended, e.g. `boom at src/lib.rs:3:5`.
    pub(super) fn message_with_location(&self) -> String {
        match &self.location {
            Some(location) => format!("{} at {}", self.message, location),
            None => self.message.clone(),
        }
    }
}

thread_local! {
    /// First panic on this thread since the last `clear`.
    static LAST_PANIC: RefCell<Option<PanicReport>> = const { RefCell::new(None) };
}

static INSTALL: Once = Once::new();

/// Install the recording hook once. It runs the previous hook after recording.
pub(super) fn install() {
    INSTALL.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let report = PanicReport {
                message: payload_message(info.payload()),
                location: info
                    .location()
                    .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column())),
                backtrace: Backtrace::force_capture().to_string(),
            };
            record(report);
            previous(info);
        }));
    });
}

/// Keep `report` unless this thread already holds one. A nested panic keeps the first.
fn record(report: PanicReport) {
    let _ = LAST_PANIC.try_with(|slot| {
        if let Ok(mut slot) = slot.try_borrow_mut() {
            if slot.is_none() {
                *slot = Some(report);
            }
        }
    });
}

/// Forget any recorded panic on this thread.
pub(super) fn clear() {
    let _ = LAST_PANIC.try_with(|slot| {
        if let Ok(mut slot) = slot.try_borrow_mut() {
            *slot = None;
        }
    });
}

/// Take the recorded panic of this thread.
pub(super) fn take() -> Option<PanicReport> {
    LAST_PANIC
        .try_with(|slot| slot.try_borrow_mut().ok().and_then(|mut s| s.take()))
        .ok()
        .flatten()
}

/// Text of a panic payload: `&str`, `String`, or a fixed fallback.
pub(super) fn payload_message(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "Unknown panic".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_message_reads_str_and_string() {
        assert_eq!(payload_message(&"boom"), "boom");
        assert_eq!(payload_message(&"bang".to_string()), "bang");
        assert_eq!(payload_message(&42_u32), "Unknown panic");
    }

    #[test]
    fn hook_records_message_location_and_backtrace() {
        install();
        clear();
        let caught: std::thread::Result<()> = std::panic::catch_unwind(|| panic!("recorded panic"));
        assert!(caught.is_err());
        let report = take().expect("hook records the panic");
        assert_eq!(report.message, "recorded panic");
        let location = report.location.clone().expect("location known");
        assert!(location.contains("panic_report.rs"));
        assert!(!report.backtrace.is_empty());
        assert!(
            report
                .message_with_location()
                .starts_with("recorded panic at ")
        );
        assert!(take().is_none());
    }

    #[test]
    fn nested_record_keeps_first() {
        clear();
        let first = PanicReport {
            message: "first".to_string(),
            location: None,
            backtrace: String::new(),
        };
        let second = PanicReport {
            message: "second".to_string(),
            ..first.clone()
        };
        record(first);
        record(second);
        assert_eq!(take().map(|r| r.message), Some("first".to_string()));
    }
}
