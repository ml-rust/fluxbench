//! Wrapper body for async `#[flux::bench]` functions.
//!
//! `#[flux::bench]` expands an async body into one `__flux_async_bench!` call.
//! With the `async` feature, the call builds the runtime and drives the body.
//! Without it, the call is a compile error that names the feature.

/// Build the runtime, attach it to the bencher, and drive the body.
///
/// Input: `bencher ident, [Builder constructor and config calls], body future`.
/// A runtime build error or a context error goes to the bencher. Nothing panics.
#[cfg(feature = "async")]
#[doc(hidden)]
#[macro_export]
macro_rules! __flux_async_bench {
    ($bencher:ident, [$($builder:tt)*], $body:expr) => {
        let __flux_rt = match $crate::internal::tokio::runtime::Builder::$($builder)*.build() {
            ::std::result::Result::Ok(__flux_rt) => ::std::sync::Arc::new(__flux_rt),
            ::std::result::Result::Err(__flux_error) => {
                $bencher.attach_runtime_error(__flux_error.to_string());
                return;
            }
        };
        $bencher.attach_runtime(::std::sync::Arc::clone(&__flux_rt));
        if let ::std::result::Result::Err(__flux_error) =
            $crate::internal::drive_async_body(&__flux_rt, $body)
        {
            $bencher.record_error(__flux_error);
        }
    };
}

/// Async benchmarks need tokio. Without the `async` feature, report that at compile time.
#[cfg(not(feature = "async"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __flux_async_bench {
    ($($tokens:tt)*) => {
        ::core::compile_error!(
            "async #[flux::bench] needs tokio: enable the `async` feature of fluxbench"
        );
    };
}
