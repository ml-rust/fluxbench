//! Checks after measurement: `#[bench(budget)]` and the `[allocator]` limits.
//!
//! A passed benchmark that breaks a check becomes `failed`. Its metrics stay in the report.

use crate::AllocatorConfig;
use crate::executor::BenchExecutionResult;
use fluxbench_ipc::FailureKind;
use fluxbench_report::{BenchmarkStatus, format_duration};
use fluxbench_stats::SummaryStatistics;
use std::collections::HashMap;

/// Report string of `failure.kind` for a mean above `#[bench(budget)]`
pub(crate) const BUDGET_KIND: &str = "budget";

/// Failure message when `mean_ns` exceeds `budget_ns`, else `None`.
pub(crate) fn budget_verdict(budget_ns: u64, mean_ns: f64) -> Option<String> {
    (mean_ns > budget_ns as f64).then(|| {
        format!(
            "mean {} exceeds the budget of {}. Speed up the benchmark or raise #[bench(budget)].",
            format_duration(mean_ns),
            format_duration(budget_ns as f64)
        )
    })
}

/// Failure message when per-iteration allocations break a limit, else `None`.
///
/// `bytes` and `count` are unrounded means per iteration.
pub(crate) fn allocation_verdict(
    limits: &AllocatorConfig,
    bytes: f64,
    count: f64,
) -> Option<String> {
    let mut reasons = Vec::new();
    if limits.fail_on_allocation && count > 0.0 {
        reasons.push(format!(
            "mean {count:.3} allocations per iteration, but allocator.fail_on_allocation allows none. Remove the allocations or turn the key off."
        ));
    }
    if let Some(max) = limits.max_bytes_per_iter.filter(|&max| bytes > max as f64) {
        reasons.push(format!(
            "mean {bytes:.1} bytes per iteration exceeds allocator.max_bytes_per_iter = {max}. Allocate less or raise the limit."
        ));
    }
    (!reasons.is_empty()).then(|| reasons.join(" "))
}

/// Why the allocation limits cannot run, or `None` when they can.
///
/// They need tracking on and at least one result with nonzero allocations.
/// All-zero totals mean either no allocations or no `TrackingAllocator`, so the check cannot tell.
pub(crate) fn allocation_check_blocker(
    limits: &AllocatorConfig,
    track: bool,
    results: &[BenchExecutionResult],
) -> Option<String> {
    let measured: Vec<&BenchExecutionResult> =
        results.iter().filter(|r| !r.samples.is_empty()).collect();
    let all_zero = measured
        .iter()
        .all(|r| r.alloc_bytes_exact == 0.0 && r.alloc_count_exact == 0.0);
    let limits_note = if limits.has_limits() {
        " The allocator.fail_on_allocation and allocator.max_bytes_per_iter checks do not run."
    } else {
        ""
    };

    if !track {
        return limits.has_limits().then(|| {
            "Warning: allocator.fail_on_allocation and allocator.max_bytes_per_iter need allocator.track = true. The allocation checks do not run.".to_string()
        });
    }
    (!measured.is_empty() && all_zero).then(|| {
        format!(
            "Warning: allocation tracking is on, but every benchmark reported 0 bytes allocated.\n\
             Set TrackingAllocator as #[global_allocator] in the benchmark binary.{limits_note}"
        )
    })
}

/// Apply the budget and allocation checks to passed results.
///
/// Returns warnings for stderr. `budgets` maps benchmark ID to its budget in nanoseconds.
pub(crate) fn apply_checks(
    results: &mut [BenchExecutionResult],
    stats: &[(String, Option<SummaryStatistics>)],
    budgets: &HashMap<String, u64>,
    limits: &AllocatorConfig,
    track: bool,
) -> Vec<String> {
    let blocker = allocation_check_blocker(limits, track, results);
    let check_allocations = limits.has_limits() && blocker.is_none();
    let means: HashMap<&str, f64> = stats
        .iter()
        .filter_map(|(id, s)| s.as_ref().map(|s| (id.as_str(), s.mean)))
        .collect();

    for result in results.iter_mut() {
        if result.status != BenchmarkStatus::Passed {
            continue;
        }
        let mut failures: Vec<(&str, String)> = Vec::new();
        if let (Some(&budget), Some(&mean)) = (
            budgets.get(&result.benchmark_id),
            means.get(result.benchmark_id.as_str()),
        ) {
            if let Some(message) = budget_verdict(budget, mean) {
                failures.push((BUDGET_KIND, message));
            }
        }
        if check_allocations {
            if let Some(message) =
                allocation_verdict(limits, result.alloc_bytes_exact, result.alloc_count_exact)
            {
                failures.push((FailureKind::AllocationLimit.as_str(), message));
            }
        }
        if let Some((kind, _)) = failures.first() {
            result.status = BenchmarkStatus::Failed;
            result.failure_kind = Some(kind.to_string());
            result.error_message = Some(
                failures
                    .iter()
                    .map(|(_, m)| m.as_str())
                    .collect::<Vec<_>>()
                    .join(" "),
            );
        }
    }

    blocker.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits(fail_on_allocation: bool, max_bytes_per_iter: Option<u64>) -> AllocatorConfig {
        AllocatorConfig {
            track: true,
            fail_on_allocation,
            max_bytes_per_iter,
        }
    }

    fn passed(id: &str, bytes: f64, count: f64) -> BenchExecutionResult {
        BenchExecutionResult {
            benchmark_id: id.to_string(),
            benchmark_name: id.to_string(),
            group: "g".to_string(),
            file: String::new(),
            line: 0,
            status: BenchmarkStatus::Passed,
            samples: vec![10.0, 12.0],
            cpu_cycles: vec![0.0, 0.0],
            alloc_bytes: bytes.round() as u64,
            alloc_count: count.round() as u64,
            alloc_bytes_exact: bytes,
            alloc_count_exact: count,
            duration_ns: 0,
            error_message: None,
            failure_kind: None,
            signal: None,
            backtrace: None,
            severity: fluxbench_core::Severity::Info,
            threshold: 0.0,
            throughput: None,
        }
    }

    #[test]
    fn budget_decision() {
        assert_eq!(budget_verdict(1_000, 999.0), None);
        assert_eq!(budget_verdict(1_000, 1_000.0), None);
        let message = budget_verdict(1_000, 1_500.0).unwrap();
        assert!(
            message.contains("mean 1.5 us exceeds the budget of 1.0 us"),
            "{message}"
        );
    }

    #[test]
    fn fail_on_allocation_decision() {
        let on = limits(true, None);
        assert_eq!(allocation_verdict(&on, 0.0, 0.0), None);
        let message = allocation_verdict(&on, 8.0, 0.25).unwrap();
        assert!(
            message.contains("0.250 allocations per iteration"),
            "{message}"
        );
        assert_eq!(allocation_verdict(&limits(false, None), 8.0, 1.0), None);
    }

    #[test]
    fn max_bytes_per_iter_decision() {
        let max = limits(false, Some(1024));
        assert_eq!(allocation_verdict(&max, 1024.0, 3.0), None);
        let message = allocation_verdict(&max, 1024.5, 3.0).unwrap();
        assert!(message.contains("max_bytes_per_iter = 1024"), "{message}");
    }

    #[test]
    fn apply_checks_fails_only_breaking_results() {
        let mut results = vec![passed("fast", 0.0, 0.0), passed("slow", 64.0, 1.0)];
        let stats: Vec<(String, Option<SummaryStatistics>)> = vec![];
        let budgets = HashMap::new();
        let warnings = apply_checks(&mut results, &stats, &budgets, &limits(true, None), true);
        assert!(warnings.is_empty());
        assert_eq!(results[0].status, BenchmarkStatus::Passed);
        assert_eq!(results[1].status, BenchmarkStatus::Failed);
        assert_eq!(results[1].failure_kind.as_deref(), Some("allocation_limit"));
        assert!(!results[1].samples.is_empty(), "metrics stay");
    }

    #[test]
    fn allocation_checks_skip_with_one_warning_when_totals_are_zero() {
        let mut results = vec![passed("a", 0.0, 0.0), passed("b", 0.0, 0.0)];
        let warnings = apply_checks(
            &mut results,
            &[],
            &HashMap::new(),
            &limits(true, None),
            true,
        );
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("checks do not run"), "{}", warnings[0]);
    }

    #[test]
    fn allocation_checks_need_tracking() {
        let mut results = vec![passed("a", 64.0, 1.0)];
        let warnings = apply_checks(
            &mut results,
            &[],
            &HashMap::new(),
            &limits(true, None),
            false,
        );
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("need allocator.track = true"));
        assert_eq!(results[0].status, BenchmarkStatus::Passed);
    }
}
