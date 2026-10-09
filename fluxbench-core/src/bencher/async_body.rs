//! Drive an async benchmark body without nesting `block_on`.
//!
//! The body runs on the caller thread, outside any `block_on`.
//! So `iter_async` can call `Runtime::block_on` once per batch.

use super::config::MeasureError;
use std::cell::Cell;
use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Wake, Waker};
use tokio::runtime::{Handle, Runtime};
use tokio::sync::Notify;

thread_local! {
    /// `true` while `drive_async_body` polls a body on this thread.
    static POLLING_BODY: Cell<bool> = const { Cell::new(false) };
}

/// Sets `POLLING_BODY` for one poll and restores the old value on drop.
struct PollingBody(bool);

impl PollingBody {
    fn set() -> Self {
        Self(POLLING_BODY.with(|flag| flag.replace(true)))
    }
}

impl Drop for PollingBody {
    fn drop(&mut self) {
        POLLING_BODY.with(|flag| flag.set(self.0));
    }
}

/// Waker that stores a wake as a `Notify` permit.
///
/// A wake before the pump starts is kept, so no wake is lost.
struct BodyWaker(Notify);

impl Wake for BodyWaker {
    fn wake(self: Arc<Self>) {
        self.0.notify_one();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.notify_one();
    }
}

/// Return an error when this thread cannot call `Runtime::block_on`.
///
/// Tokio panics on `block_on` inside a task or inside another `block_on`.
/// Tokio exposes no check for that state.
/// A tokio task id, or a current handle outside `drive_async_body`, counts as that state.
/// A bare `Runtime::enter` guard outside `drive_async_body` counts too.
/// `caller` names the function in the error message.
pub(super) fn runtime_context_error(caller: &str) -> Option<MeasureError> {
    let in_task = tokio::task::try_id().is_some();
    let in_body = POLLING_BODY.with(Cell::get);
    let has_handle = Handle::try_current().is_ok();
    if in_task || (has_handle && !in_body) {
        return Some(MeasureError::AsyncRuntime(format!(
            "{caller} called inside a runtime context. \
             Call it from a thread outside any runtime, or inside a drive_async_body body."
        )));
    }
    None
}

/// Run an async benchmark body to completion on `rt`. Its output is dropped.
///
/// Each poll holds `rt.enter()`, so `tokio::spawn` and `tokio::time` work in the body.
/// The body is never polled inside `block_on`.
/// So `Bencher::iter_async` can call `rt.block_on` once per batch.
/// On `Pending`, `rt.block_on` waits for the body's waker.
/// That wait drives IO, timers, and spawned tasks on a current-thread runtime.
///
/// # Errors
/// `MeasureError::AsyncRuntime` when the caller sits inside a runtime context.
/// The body does not run in that case.
pub fn drive_async_body<F>(rt: &Runtime, body: F) -> Result<(), MeasureError>
where
    F: Future,
{
    if let Some(error) = runtime_context_error("drive_async_body") {
        return Err(error);
    }
    let mut body = pin!(body);
    let signal = Arc::new(BodyWaker(Notify::new()));
    let waker = Waker::from(Arc::clone(&signal));
    let mut cx = Context::from_waker(&waker);
    loop {
        let poll = {
            let _context = rt.enter();
            let _polling = PollingBody::set();
            body.as_mut().poll(&mut cx)
        };
        if poll.is_ready() {
            return Ok(());
        }
        rt.block_on(signal.0.notified());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn current_thread() -> Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime builds")
    }

    #[test]
    fn body_awaits_timer_on_current_thread() {
        let rt = current_thread();
        let done = Cell::new(false);
        drive_async_body(&rt, async {
            tokio::time::sleep(Duration::from_micros(1)).await;
            done.set(true);
        })
        .expect("body runs");
        assert!(done.get());
    }

    #[test]
    fn body_spawns_task_on_current_thread() {
        let rt = current_thread();
        let value = Cell::new(0_u64);
        drive_async_body(&rt, async {
            let task = tokio::spawn(async { 7_u64 });
            value.set(task.await.expect("task joins"));
        })
        .expect("body runs");
        assert_eq!(value.get(), 7);
    }

    #[test]
    fn polling_flag_resets_after_body() {
        let rt = current_thread();
        drive_async_body(&rt, async {
            assert!(POLLING_BODY.with(Cell::get));
        })
        .expect("body runs");
        assert!(!POLLING_BODY.with(Cell::get));
    }

    #[test]
    fn inside_block_on_returns_error() {
        let outer = current_thread();
        let inner = current_thread();
        let result = outer.block_on(async { drive_async_body(&inner, async {}) });
        assert!(matches!(result, Err(MeasureError::AsyncRuntime(_))));
    }
}
