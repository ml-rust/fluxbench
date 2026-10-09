//! Run config a `#[flux::group]` sets for its benchmarks.

use fluxbench_ipc::Throughput;

/// Run config a `#[flux::group]` sets for every benchmark in it and in its child groups.
///
/// Precedence per field: `#[flux::bench]` attribute, then the nearest group, then parent groups,
/// then CLI flags and flux.toml, then the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GroupConfig {
    /// Warmup time in nanoseconds
    pub warmup_ns: Option<u64>,
    /// Measurement time in nanoseconds
    pub measurement_ns: Option<u64>,
    /// Target sample count
    pub sample_count: Option<u64>,
    /// Fixed iterations per sample. `Some` selects fixed mode.
    pub iters_per_sample: Option<u64>,
    /// Minimum total measurement iterations
    pub min_iterations: Option<u64>,
    /// Maximum total measurement iterations
    pub max_iterations: Option<u64>,
    /// Work per call for throughput rates
    pub throughput: Option<Throughput>,
}

impl GroupConfig {
    /// A config that sets nothing.
    pub const NONE: GroupConfig = GroupConfig {
        warmup_ns: None,
        measurement_ns: None,
        sample_count: None,
        iters_per_sample: None,
        min_iterations: None,
        max_iterations: None,
        throughput: None,
    };

    /// Keep each field set here. Take each unset field from `parent`.
    pub fn or(self, parent: &GroupConfig) -> GroupConfig {
        GroupConfig {
            warmup_ns: self.warmup_ns.or(parent.warmup_ns),
            measurement_ns: self.measurement_ns.or(parent.measurement_ns),
            sample_count: self.sample_count.or(parent.sample_count),
            iters_per_sample: self.iters_per_sample.or(parent.iters_per_sample),
            min_iterations: self.min_iterations.or(parent.min_iterations),
            max_iterations: self.max_iterations.or(parent.max_iterations),
            throughput: self.throughput.or(parent.throughput),
        }
    }

    /// Merge a group chain ordered nearest first. The nearest group wins per field.
    pub fn merge_chain<'a>(chain: impl IntoIterator<Item = &'a GroupConfig>) -> GroupConfig {
        chain
            .into_iter()
            .fold(GroupConfig::NONE, |merged, parent| merged.or(parent))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_group_wins_per_field() {
        let child = GroupConfig {
            sample_count: Some(20),
            ..GroupConfig::NONE
        };
        let parent = GroupConfig {
            sample_count: Some(50),
            warmup_ns: Some(7),
            throughput: Some(Throughput::Bytes(1)),
            ..GroupConfig::NONE
        };
        let root = GroupConfig {
            warmup_ns: Some(9),
            measurement_ns: Some(11),
            throughput: Some(Throughput::Elements(2)),
            ..GroupConfig::NONE
        };
        let merged = GroupConfig::merge_chain([&child, &parent, &root]);
        assert_eq!(merged.sample_count, Some(20));
        assert_eq!(merged.warmup_ns, Some(7));
        assert_eq!(merged.measurement_ns, Some(11));
        assert_eq!(merged.throughput, Some(Throughput::Bytes(1)));
        assert_eq!(merged.iters_per_sample, None);
    }

    #[test]
    fn empty_chain_sets_nothing() {
        assert_eq!(GroupConfig::merge_chain([]), GroupConfig::NONE);
        assert_eq!(GroupConfig::default(), GroupConfig::NONE);
    }
}
