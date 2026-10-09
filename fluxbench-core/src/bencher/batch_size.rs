//! `BatchSize`: how many inputs `iter_batched` builds before one timed batch.

/// Most inputs a `BatchSize::SmallInput` batch holds. `iter_with_setup` uses it too.
///
/// Peak memory is at most this many live inputs plus this many live outputs.
pub const SETUP_CHUNK: u64 = 1024;

/// Most inputs a `BatchSize::LargeInput` batch holds.
pub const LARGE_INPUT_CHUNK: u64 = 16;

/// Inputs per timed batch for `iter_batched` and `iter_batched_ref`.
///
/// Setup builds a whole batch untimed. Then one timer covers the routine over that batch.
/// Larger batches spread the timer cost over more calls. Smaller batches hold less memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchSize {
    /// Up to `SETUP_CHUNK` (1024) inputs per batch. Fits inputs that are cheap to hold.
    SmallInput,
    /// Up to `LARGE_INPUT_CHUNK` (16) inputs per batch. Fits inputs that hold a lot of memory.
    LargeInput,
    /// One input per batch. Each call gets its own timer. Overhead per call is highest.
    PerIteration,
    /// Split each sample's calls into this many batches. 0 counts as 1.
    NumBatches(u64),
    /// Exactly this many inputs per batch, capped at the sample's calls. 0 counts as 1.
    NumIterations(u64),
}

impl BatchSize {
    /// Inputs per batch for a sample of `iters` calls. Always in `1..=max(iters, 1)`.
    pub(super) fn batch_len(self, iters: u64) -> u64 {
        let len = match self {
            BatchSize::SmallInput => SETUP_CHUNK,
            BatchSize::LargeInput => LARGE_INPUT_CHUNK,
            BatchSize::PerIteration => 1,
            BatchSize::NumBatches(batches) => iters.div_ceil(batches.max(1)),
            BatchSize::NumIterations(size) => size,
        };
        len.clamp(1, iters.max(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_sizes_cap_at_the_sample() {
        assert_eq!(BatchSize::SmallInput.batch_len(5_000), 1024);
        assert_eq!(BatchSize::SmallInput.batch_len(10), 10);
        assert_eq!(BatchSize::LargeInput.batch_len(5_000), 16);
        assert_eq!(BatchSize::LargeInput.batch_len(3), 3);
        assert_eq!(BatchSize::PerIteration.batch_len(5_000), 1);
    }

    #[test]
    fn num_batches_rounds_up() {
        assert_eq!(BatchSize::NumBatches(4).batch_len(10), 3);
        assert_eq!(BatchSize::NumBatches(1).batch_len(10), 10);
        assert_eq!(BatchSize::NumBatches(0).batch_len(10), 10);
        assert_eq!(BatchSize::NumBatches(100).batch_len(10), 1);
    }

    #[test]
    fn num_iterations_clamps() {
        assert_eq!(BatchSize::NumIterations(7).batch_len(10), 7);
        assert_eq!(BatchSize::NumIterations(70).batch_len(10), 10);
        assert_eq!(BatchSize::NumIterations(0).batch_len(10), 1);
    }

    #[test]
    fn zero_iters_still_gives_one() {
        assert_eq!(BatchSize::SmallInput.batch_len(0), 1);
    }
}
