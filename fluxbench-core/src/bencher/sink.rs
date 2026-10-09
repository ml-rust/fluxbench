//! Sample sink: receives each sample when the engine records it.

use fluxbench_ipc::Sample;

/// Receives each measurement sample when the engine records it.
///
/// The engine calls the sink between samples, outside every timed region.
/// The worker uses a sink to stream samples to the supervisor during measurement.
/// So a timeout or crash keeps the samples taken before it.
pub trait SampleSink: Send {
    /// Take one recorded sample.
    fn on_sample(&mut self, sample: &Sample);
}

impl<F> SampleSink for F
where
    F: FnMut(&Sample) + Send,
{
    fn on_sample(&mut self, sample: &Sample) {
        self(sample)
    }
}
