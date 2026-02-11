//! Benchmark Planner
//!
//! Builds execution plan by filtering and ordering benchmarks.
//!
//! Filtering options:
//! - Regex pattern matching on benchmark ID
//! - Group filtering
//! - Tag inclusion/exclusion
//!
//! Ordering:
//! 1. Dependency-based topological sort (benchmarks with `depends_on` run after their deps)
//! 2. Alphabetical by ID within the same dependency level

use fluxbench_core::BenchmarkDef;
use std::collections::{HashMap, HashSet, VecDeque};

/// Execution plan for benchmarks
pub struct ExecutionPlan {
    /// Ordered list of benchmarks to run
    pub benchmarks: Vec<&'static BenchmarkDef>,
}

/// Build execution plan from discovered benchmarks
///
/// Filters benchmarks based on CLI options and returns them in dependency-respecting order.
pub fn build_plan(
    benchmarks: impl IntoIterator<Item = &'static BenchmarkDef>,
    filter: Option<&regex::Regex>,
    group: Option<&str>,
    tag: Option<&str>,
    skip_tag: Option<&str>,
) -> ExecutionPlan {
    let mut selected: Vec<_> = benchmarks
        .into_iter()
        .filter(|b| {
            if let Some(re) = filter {
                if !re.is_match(b.id) {
                    return false;
                }
            }
            if let Some(g) = group {
                if b.group != g {
                    return false;
                }
            }
            if let Some(t) = tag {
                if !b.tags.contains(&t) {
                    return false;
                }
            }
            if let Some(st) = skip_tag {
                if b.tags.contains(&st) {
                    return false;
                }
            }
            true
        })
        .collect();

    // Sort alphabetically first (stable base order)
    selected.sort_by_key(|b| b.id);

    // Apply topological sort if any benchmark has dependencies
    let has_deps = selected.iter().any(|b| !b.depends_on.is_empty());
    if has_deps {
        selected = topological_sort(selected);
    }

    ExecutionPlan {
        benchmarks: selected,
    }
}

/// Topological sort using Kahn's algorithm.
/// Preserves alphabetical order among benchmarks at the same dependency level.
/// Warns about missing dependencies and detects cycles.
fn topological_sort(benchmarks: Vec<&'static BenchmarkDef>) -> Vec<&'static BenchmarkDef> {
    let id_set: HashSet<&str> = benchmarks.iter().map(|b| b.id).collect();
    let index_map: HashMap<&str, usize> = benchmarks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id, i))
        .collect();

    // Build adjacency: in_degree[i] = number of deps that i waits on
    let n = benchmarks.len();
    let mut in_degree = vec![0usize; n];
    // dependents[i] = list of benchmarks that depend on i
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (i, b) in benchmarks.iter().enumerate() {
        for &dep_id in b.depends_on {
            if !id_set.contains(dep_id) {
                eprintln!(
                    "Warning: benchmark '{}' depends on '{}' which is not in the filtered set; skipping dependency",
                    b.id, dep_id
                );
                continue;
            }
            let dep_idx = index_map[dep_id];
            in_degree[i] += 1;
            dependents[dep_idx].push(i);
        }
    }

    // Kahn's algorithm (VecDeque for O(1) pop_front)
    let mut queue: VecDeque<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut result = Vec::with_capacity(n);

    while let Some(idx) = queue.pop_front() {
        result.push(benchmarks[idx]);

        let mut newly_ready = Vec::new();
        for &dep_idx in &dependents[idx] {
            in_degree[dep_idx] -= 1;
            if in_degree[dep_idx] == 0 {
                newly_ready.push(dep_idx);
            }
        }
        // Sort newly ready by ID for deterministic ordering
        newly_ready.sort_by_key(|&i| benchmarks[i].id);
        queue.extend(newly_ready);
    }

    if result.len() != n {
        // Cycle detected - emit error and return original order
        let in_cycle: Vec<&str> = (0..n)
            .filter(|&i| in_degree[i] > 0)
            .map(|i| benchmarks[i].id)
            .collect();
        eprintln!(
            "Error: dependency cycle detected among benchmarks: {}. Running in alphabetical order.",
            in_cycle.join(", ")
        );
        return benchmarks;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxbench_core::Severity;

    fn make_bench(
        id: &'static str,
        group: &'static str,
        tags: &'static [&'static str],
    ) -> BenchmarkDef {
        make_bench_with_deps(id, group, tags, &[])
    }

    fn make_bench_with_deps(
        id: &'static str,
        group: &'static str,
        tags: &'static [&'static str],
        depends_on: &'static [&'static str],
    ) -> BenchmarkDef {
        BenchmarkDef {
            id,
            name: id,
            group,
            severity: Severity::Info,
            threshold: 0.0,
            budget_ns: None,
            tags,
            runner_fn: |_| {},
            file: "",
            line: 0,
            module_path: "",
            warmup_ns: None,
            measurement_ns: None,
            samples: None,
            min_iterations: None,
            max_iterations: None,
            depends_on,
        }
    }

    #[test]
    fn test_no_filter() {
        let benches = [
            make_bench("c_bench", "default", &[]),
            make_bench("a_bench", "default", &[]),
            make_bench("b_bench", "default", &[]),
        ];
        let refs: Vec<_> = benches
            .iter()
            .map(|b| unsafe { &*(b as *const _) })
            .collect();

        let plan = build_plan(refs, None, None, None, None);

        // Should be sorted alphabetically
        assert_eq!(plan.benchmarks.len(), 3);
        assert_eq!(plan.benchmarks[0].id, "a_bench");
        assert_eq!(plan.benchmarks[1].id, "b_bench");
        assert_eq!(plan.benchmarks[2].id, "c_bench");
    }

    #[test]
    fn test_group_filter() {
        let benches = [
            make_bench("bench1", "group_a", &[]),
            make_bench("bench2", "group_b", &[]),
            make_bench("bench3", "group_a", &[]),
        ];
        let refs: Vec<_> = benches
            .iter()
            .map(|b| unsafe { &*(b as *const _) })
            .collect();

        let plan = build_plan(refs, None, Some("group_a"), None, None);

        assert_eq!(plan.benchmarks.len(), 2);
        assert!(plan.benchmarks.iter().all(|b| b.group == "group_a"));
    }

    #[test]
    fn test_tag_filter() {
        let benches = [
            make_bench("bench1", "default", &["fast"]),
            make_bench("bench2", "default", &["slow"]),
            make_bench("bench3", "default", &["fast", "important"]),
        ];
        let refs: Vec<_> = benches
            .iter()
            .map(|b| unsafe { &*(b as *const _) })
            .collect();

        let plan = build_plan(refs, None, None, Some("fast"), None);

        assert_eq!(plan.benchmarks.len(), 2);
        assert!(plan.benchmarks.iter().all(|b| b.tags.contains(&"fast")));
    }

    #[test]
    fn test_dependency_ordering() {
        let benches = [
            make_bench_with_deps("c_bench", "default", &[], &["a_bench"]),
            make_bench("a_bench", "default", &[]),
            make_bench_with_deps("b_bench", "default", &[], &["a_bench"]),
        ];
        let refs: Vec<_> = benches
            .iter()
            .map(|b| unsafe { &*(b as *const _) })
            .collect();

        let plan = build_plan(refs, None, None, None, None);

        assert_eq!(plan.benchmarks.len(), 3);
        // a_bench must come first (no deps)
        assert_eq!(plan.benchmarks[0].id, "a_bench");
        // b_bench and c_bench depend on a_bench, so they come after
        let remaining: Vec<&str> = plan.benchmarks[1..].iter().map(|b| b.id).collect();
        assert!(remaining.contains(&"b_bench"));
        assert!(remaining.contains(&"c_bench"));
    }

    #[test]
    fn test_dependency_chain() {
        let benches = [
            make_bench_with_deps("c_bench", "default", &[], &["b_bench"]),
            make_bench("a_bench", "default", &[]),
            make_bench_with_deps("b_bench", "default", &[], &["a_bench"]),
        ];
        let refs: Vec<_> = benches
            .iter()
            .map(|b| unsafe { &*(b as *const _) })
            .collect();

        let plan = build_plan(refs, None, None, None, None);

        assert_eq!(plan.benchmarks.len(), 3);
        assert_eq!(plan.benchmarks[0].id, "a_bench");
        assert_eq!(plan.benchmarks[1].id, "b_bench");
        assert_eq!(plan.benchmarks[2].id, "c_bench");
    }

    #[test]
    fn test_dependency_cycle_fallback() {
        let benches = [
            make_bench_with_deps("a_bench", "default", &[], &["c_bench"]),
            make_bench_with_deps("b_bench", "default", &[], &["a_bench"]),
            make_bench_with_deps("c_bench", "default", &[], &["b_bench"]),
        ];
        let refs: Vec<_> = benches
            .iter()
            .map(|b| unsafe { &*(b as *const _) })
            .collect();

        // Cycle: a -> c -> b -> a; should fall back to alphabetical
        let plan = build_plan(refs, None, None, None, None);
        assert_eq!(plan.benchmarks.len(), 3);
        assert_eq!(plan.benchmarks[0].id, "a_bench");
        assert_eq!(plan.benchmarks[1].id, "b_bench");
        assert_eq!(plan.benchmarks[2].id, "c_bench");
    }

    #[test]
    fn test_skip_tag() {
        let benches = [
            make_bench("bench1", "default", &["fast"]),
            make_bench("bench2", "default", &["slow"]),
            make_bench("bench3", "default", &["fast", "skip_ci"]),
        ];
        let refs: Vec<_> = benches
            .iter()
            .map(|b| unsafe { &*(b as *const _) })
            .collect();

        let plan = build_plan(refs, None, None, None, Some("skip_ci"));

        assert_eq!(plan.benchmarks.len(), 2);
        assert!(plan.benchmarks.iter().all(|b| !b.tags.contains(&"skip_ci")));
    }
}
