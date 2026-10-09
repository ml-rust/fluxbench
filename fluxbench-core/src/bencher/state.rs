//! The `Bencher` struct: construction, accessors, and result assembly.

use super::config::{BenchmarkResult, DEFAULT_SAMPLE_COUNT, MeasureConfig, MeasureError};
use super::sink::SampleSink;
use fluxbench_ipc::{Sample, Throughput};
#[cfg(feature = "async")]
use std::sync::Arc;
use std::sync::Mutex;
#[cfg(feature = "async")]
use tokio::runtime::Runtime;

/// The Bencher provides iteration control for benchmarks.
///
/// The benchmark body calls exactly one `iter*` method once.
/// That method runs warmup and measurement in full before it returns.
/// Each sample holds per-iteration means over one batch of calls.
pub struct Bencher {
    /// Normalized measurement config
    pub(super) config: MeasureConfig,
    /// Completed samples. Capacity is `sample_count`. Warmup stores no per-call data.
    pub(super) samples: Vec<Sample>,
    /// Routine calls in the warmup phase
    pub(super) warmup_iterations: u64,
    /// Routine calls per measurement sample
    pub(super) iters_per_sample: u64,
    /// `true` when the hard deadline or a stop request cut measurement short
    pub(super) truncated: bool,
    /// `true` once the body called an `iter*` method
    pub(super) iter_called: bool,
    /// First error the body caused. `finish` reports it.
    pub(super) error: Option<MeasureError>,
    /// Runtime for `iter_async`. `None` until attached or built.
    #[cfg(feature = "async")]
    pub(super) runtime: Option<Arc<Runtime>>,
    /// Receives each sample when it is recorded. The mutex keeps `Bencher: Sync`.
    /// The engine reaches it through `&mut self`, so it never locks.
    pub(super) sink: Option<Mutex<Box<dyn SampleSink>>>,
    /// Work per call the body set with `throughput`. The last call wins.
    pub(super) throughput: Option<Throughput>,
}

impl Bencher {
    /// Create a Bencher with the default config and the given allocation tracking.
    pub fn new(track_allocations: bool) -> Self {
        Self::with_sample_count(track_allocations, DEFAULT_SAMPLE_COUNT)
    }

    /// Create a Bencher from a measurement config.
    ///
    /// The config is normalized first: see `MeasureConfig::normalized`.
    pub fn with_config(config: MeasureConfig) -> Self {
        let config = config.normalized();
        Self {
            samples: Vec::with_capacity(config.sample_count),
            iters_per_sample: config.fixed_iters_per_sample.unwrap_or(0),
            config,
            warmup_iterations: 0,
            truncated: false,
            iter_called: false,
            error: None,
            #[cfg(feature = "async")]
            runtime: None,
            sink: None,
            throughput: None,
        }
    }

    /// Set the work one routine call processes. Reports turn it into a rate per second.
    ///
    /// Call it anywhere in the body. The last call wins.
    /// It overrides `throughput_bytes` or `throughput_elements` on `#[flux::bench]` and `#[flux::group]`.
    pub fn throughput(&mut self, throughput: Throughput) {
        self.throughput = Some(throughput);
    }

    /// Send each recorded sample to `sink` during measurement.
    ///
    /// The sink runs between samples, outside every timed region.
    /// Samples still land in the result as before.
    pub fn with_sample_sink(mut self, sink: impl SampleSink + 'static) -> Self {
        self.sink = Some(Mutex::new(Box::new(sink)));
        self
    }

    /// Record one sample and pass it to the sink.
    pub(super) fn record_sample(&mut self, sample: Sample) {
        self.samples.push(sample);
        if let Some(sink) = self.sink.as_mut() {
            let sink = match sink.get_mut() {
                Ok(sink) => sink,
                Err(poisoned) => poisoned.into_inner(),
            };
            sink.on_sample(&sample);
        }
    }

    /// Create a Bencher that targets `sample_count` samples. Other values are defaults.
    ///
    /// The count has no upper cap. Values below `MIN_SAMPLE_COUNT` rise to it.
    pub fn with_sample_count(track_allocations: bool, sample_count: usize) -> Self {
        Self::with_config(MeasureConfig {
            sample_count,
            track_allocations,
            ..MeasureConfig::default()
        })
    }

    /// Create a Bencher from sample values. Times and iteration limits are defaults.
    ///
    /// `Some(n)` iterations per sample selects fixed mode. `None` selects time-based mode.
    pub fn for_config(
        track_allocations: bool,
        sample_count: usize,
        iters_per_sample: Option<u64>,
    ) -> Self {
        Self::with_config(MeasureConfig {
            sample_count,
            fixed_iters_per_sample: iters_per_sample,
            track_allocations,
            ..MeasureConfig::default()
        })
    }

    /// Fix the iterations per sample. This selects fixed mode: no warmup, no estimation.
    ///
    /// A value of 0 rises to 1.
    pub fn with_fixed_iters_per_sample(mut self, iters: u64) -> Self {
        let iters = iters.max(1);
        self.config.fixed_iters_per_sample = Some(iters);
        self.iters_per_sample = iters;
        self
    }

    /// The normalized measurement config
    pub fn config(&self) -> &MeasureConfig {
        &self.config
    }

    /// Fixed iterations per sample, if fixed mode is on
    pub fn fixed_iters_per_sample(&self) -> Option<u64> {
        self.config.fixed_iters_per_sample
    }

    /// Target sample count
    pub fn target_samples(&self) -> usize {
        self.config.sample_count
    }

    /// Collected samples
    pub fn samples(&self) -> &[Sample] {
        &self.samples
    }

    /// Routine calls in the measurement phase so far. Warmup calls are excluded.
    pub fn iteration_count(&self) -> u64 {
        self.samples.iter().map(|s| s.iterations).sum()
    }

    /// Mark the start of an `iter*` call. Returns `false` when the call must not run.
    ///
    /// A second call records `MeasureError::MultipleIterCalls`.
    pub(super) fn begin_iter(&mut self) -> bool {
        if self.iter_called {
            self.record_error(MeasureError::MultipleIterCalls);
            return false;
        }
        self.iter_called = true;
        true
    }

    /// Record an error for `finish` to return. The first error wins.
    pub fn record_error(&mut self, error: MeasureError) {
        if self.error.is_none() {
            self.error = Some(error);
        }
    }

    /// Assemble the result, or return the first error the body caused.
    pub fn finish(self) -> Result<BenchmarkResult, MeasureError> {
        if let Some(error) = self.error {
            return Err(error);
        }
        if !self.iter_called {
            return Err(MeasureError::NoIterCall);
        }
        if self.samples.is_empty() {
            return Err(MeasureError::NoSamples);
        }

        let iterations = self.samples.iter().map(|s| s.iterations).sum();
        let total_time_ns = self
            .samples
            .iter()
            .map(|s| s.duration_nanos * s.iterations as f64)
            .sum::<f64>()
            .round() as u64;

        Ok(BenchmarkResult {
            samples: self.samples,
            iterations,
            warmup_iterations: self.warmup_iterations,
            total_time_ns,
            iters_per_sample: self.iters_per_sample,
            truncated: self.truncated,
            throughput: self.throughput,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bencher::config::MIN_SAMPLE_COUNT;
    use std::sync::Arc;

    #[test]
    fn throughput_is_none_without_a_call() {
        let result =
            crate::bencher::run_benchmark_loop(Bencher::for_config(false, 2, Some(1)), |b| {
                b.iter(|| 1_u64);
            })
            .expect("run succeeds");
        assert_eq!(result.throughput, None);
    }

    #[test]
    fn throughput_before_or_after_iter_reaches_the_result() {
        let before =
            crate::bencher::run_benchmark_loop(Bencher::for_config(false, 2, Some(1)), |b| {
                b.throughput(Throughput::Bytes(64));
                b.iter(|| 1_u64);
            })
            .expect("run succeeds");
        assert_eq!(before.throughput, Some(Throughput::Bytes(64)));

        let after =
            crate::bencher::run_benchmark_loop(Bencher::for_config(false, 2, Some(1)), |b| {
                b.iter(|| 1_u64);
                b.throughput(Throughput::Elements(3));
            })
            .expect("run succeeds");
        assert_eq!(after.throughput, Some(Throughput::Elements(3)));
    }

    #[test]
    fn last_throughput_call_wins() {
        let result =
            crate::bencher::run_benchmark_loop(Bencher::for_config(false, 2, Some(1)), |b| {
                b.throughput(Throughput::Bytes(64));
                b.throughput(Throughput::Elements(8));
                b.iter(|| 1_u64);
            })
            .expect("run succeeds");
        assert_eq!(result.throughput, Some(Throughput::Elements(8)));
    }

    #[test]
    fn sample_count_floor_is_two() {
        assert_eq!(
            Bencher::with_sample_count(false, 0).target_samples(),
            MIN_SAMPLE_COUNT
        );
        assert_eq!(Bencher::with_sample_count(false, 1).target_samples(), 2);
        assert_eq!(Bencher::with_sample_count(false, 2).target_samples(), 2);
    }

    #[test]
    fn fixed_zero_rises_to_one() {
        let bencher = Bencher::for_config(false, 10, Some(0));
        assert_eq!(bencher.fixed_iters_per_sample(), Some(1));
        let bencher = Bencher::new(false).with_fixed_iters_per_sample(0);
        assert_eq!(bencher.fixed_iters_per_sample(), Some(1));
    }

    #[test]
    fn finish_without_iter_is_error() {
        let err = Bencher::new(false).finish().err();
        assert_eq!(err, Some(MeasureError::NoIterCall));
    }

    #[test]
    fn sink_receives_every_sample_in_order() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink_seen = Arc::clone(&seen);
        let bencher =
            Bencher::for_config(false, 5, Some(3)).with_sample_sink(move |sample: &Sample| {
                sink_seen.lock().expect("lock").push(*sample);
            });
        let result = crate::bencher::run_benchmark_loop(bencher, |b| {
            b.iter(|| std::hint::black_box(1u64 + 1));
        })
        .expect("run succeeds");
        let seen = seen.lock().expect("lock");
        assert_eq!(seen.len(), 5);
        assert_eq!(seen.as_slice(), result.samples.as_slice());
    }

    #[test]
    fn finish_reports_means_and_totals() {
        let mut bencher = Bencher::with_sample_count(false, 2);
        bencher.iter_called = true;
        bencher.iters_per_sample = 4;
        bencher.samples.push(Sample::new(25.25, 75.25, 8.0, 1.0, 4));
        bencher.samples.push(Sample::new(10.0, 30.0, 0.0, 0.0, 4));
        let result = bencher.finish().expect("finish succeeds");
        assert_eq!(result.iterations, 8);
        assert_eq!(result.total_time_ns, 141);
        assert_eq!(result.iters_per_sample, 4);
        assert!(!result.truncated);
    }
}
