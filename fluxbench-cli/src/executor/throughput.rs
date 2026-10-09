//! Throughput resolution and work rates.

use fluxbench_core::{BenchmarkDef, GroupConfig, Throughput};

/// Throughput of one benchmark.
///
/// Precedence: `Bencher::throughput` in the body, then `#[flux::bench]`, then the merged group config.
pub(crate) fn resolve_throughput(
    from_body: Option<Throughput>,
    bench: &BenchmarkDef,
    group: &GroupConfig,
) -> Option<Throughput> {
    from_body.or(bench.throughput).or(group.throughput)
}

/// Work rates at `mean_ns` per call: (bytes per second, elements per second).
///
/// Each is `None` unless the throughput has that kind and the mean is above 0.
pub(crate) fn work_rates(
    throughput: Option<Throughput>,
    mean_ns: f64,
) -> (Option<f64>, Option<f64>) {
    match throughput {
        Some(t @ Throughput::Bytes(_)) => (t.per_second(mean_ns), None),
        Some(t @ Throughput::Elements(_)) => (None, t.per_second(mean_ns)),
        None => (None, None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::bench_def;

    #[test]
    fn body_beats_bench_beats_group() {
        let group = GroupConfig {
            throughput: Some(Throughput::Bytes(1)),
            ..GroupConfig::NONE
        };
        let bench = BenchmarkDef {
            throughput: Some(Throughput::Bytes(2)),
            ..bench_def("b")
        };
        let body = Some(Throughput::Elements(3));

        assert_eq!(resolve_throughput(body, &bench, &group), body);
        assert_eq!(
            resolve_throughput(None, &bench, &group),
            Some(Throughput::Bytes(2))
        );
        assert_eq!(
            resolve_throughput(None, &bench_def("b"), &group),
            Some(Throughput::Bytes(1))
        );
        assert_eq!(
            resolve_throughput(None, &bench_def("b"), &GroupConfig::NONE),
            None
        );
    }

    #[test]
    fn rates_follow_the_kind() {
        assert_eq!(
            work_rates(Some(Throughput::Bytes(1024)), 1_000.0),
            (Some(1_024_000_000.0), None)
        );
        assert_eq!(
            work_rates(Some(Throughput::Elements(10)), 10.0),
            (None, Some(1e9))
        );
        assert_eq!(work_rates(None, 10.0), (None, None));
        assert_eq!(work_rates(Some(Throughput::Bytes(8)), 0.0), (None, None));
    }
}
