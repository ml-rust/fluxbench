//! Benchmark discovery, filtering, and listing.

use crate::Cli;
use crate::groups::GroupTree;
use crate::planner::{self, Filters};
use fluxbench_core::{BenchmarkDef, GroupDef};
use regex::Regex;
use std::collections::{BTreeMap, BTreeSet};

/// Every registered benchmark, the group tree, and the benchmarks the CLI filters select
pub(crate) struct Selection {
    /// Every registered benchmark
    pub all: Vec<&'static BenchmarkDef>,
    /// Checked `#[group]` definitions
    pub groups: GroupTree<'static>,
    /// Selected benchmarks in execution order
    pub selected: Vec<&'static BenchmarkDef>,
}

/// Select benchmarks with the CLI regex, group, and tags.
///
/// Errors: an invalid regex, an unknown `--group`, `--tag`, or `--skip-tag`, or bad group definitions.
/// A valid filter that matches nothing gives an empty selection.
pub(crate) fn select_benchmarks(cli: &Cli) -> anyhow::Result<Selection> {
    let all: Vec<&'static BenchmarkDef> = inventory::iter::<BenchmarkDef>.into_iter().collect();
    let bench_groups: BTreeSet<&str> = all.iter().map(|b| b.group).collect();
    let groups = GroupTree::new(inventory::iter::<GroupDef>, &bench_groups)?;
    let selected = filter_benchmarks(cli, &all, &groups)?;
    Ok(Selection {
        all,
        groups,
        selected,
    })
}

/// Filter `benchmarks` with the CLI options. Unknown filter values are errors.
fn filter_benchmarks(
    cli: &Cli,
    benchmarks: &[&'static BenchmarkDef],
    groups: &GroupTree,
) -> anyhow::Result<Vec<&'static BenchmarkDef>> {
    let regex = Regex::new(&cli.filter).map_err(|e| {
        anyhow::anyhow!(
            "invalid filter regex `{}`: {e}\nFix the pattern. Quote it for the shell, or escape special characters such as `(`.",
            cli.filter
        )
    })?;

    if let Some(group) = cli.group.as_deref() {
        let mut known: BTreeSet<&str> = benchmarks.iter().map(|b| b.group).collect();
        known.extend(groups.defined_ids());
        if !known.contains(group) {
            anyhow::bail!(
                "unknown --group `{group}`. Known groups: {}.",
                join_or_none(known)
            );
        }
    }

    let known_tags: BTreeSet<&str> = benchmarks.iter().flat_map(|b| groups.tags_of(b)).collect();
    for (flag, tag) in [("--tag", &cli.tag), ("--skip-tag", &cli.skip_tag)] {
        if let Some(tag) = tag.as_deref().filter(|t| !known_tags.contains(t)) {
            anyhow::bail!(
                "unknown {flag} `{tag}`. Known tags: {}.",
                join_or_none(known_tags.iter().copied())
            );
        }
    }

    let filters = Filters {
        regex: Some(&regex),
        group: cli.group.as_deref(),
        tag: cli.tag.as_deref(),
        skip_tag: cli.skip_tag.as_deref(),
    };
    Ok(planner::build_plan(benchmarks.iter().copied(), &filters, groups).benchmarks)
}

fn join_or_none<'a>(items: impl IntoIterator<Item = &'a str>) -> String {
    let list: Vec<&str> = items.into_iter().collect();
    if list.is_empty() {
        "none".to_string()
    } else {
        list.join(", ")
    }
}

/// Print the benchmark plan grouped by group, then every tag in the suite.
pub(crate) fn list_benchmarks(cli: &Cli) -> anyhow::Result<()> {
    let selection = select_benchmarks(cli)?;
    println!("FluxBench Plan:");

    let mut by_group: BTreeMap<&str, Vec<&BenchmarkDef>> = BTreeMap::new();
    for bench in &selection.selected {
        by_group.entry(bench.group).or_default().push(bench);
    }

    let mut total = 0;
    for (group, benches) in &by_group {
        println!("├── group: {}", group);
        for bench in benches {
            let tags = selection.groups.tags_of(bench);
            let tags = if tags.is_empty() {
                String::new()
            } else {
                format!(" [{}]", tags.into_iter().collect::<Vec<_>>().join(", "))
            };
            println!(
                "│   ├── {}{} ({}:{})",
                bench.id, tags, bench.file, bench.line
            );
            total += 1;
        }
    }

    println!("{} benchmarks found.", total);

    // Tags span the whole suite, not the filtered results, so users can discover filters.
    let mut tag_counts: BTreeMap<&str, usize> = BTreeMap::new();
    for bench in &selection.all {
        for tag in selection.groups.tags_of(bench) {
            *tag_counts.entry(tag).or_default() += 1;
        }
    }
    if !tag_counts.is_empty() {
        let tags_display: Vec<String> = tag_counts
            .iter()
            .map(|(tag, count)| format!("{} ({})", tag, count))
            .collect();
        println!("Tags: {}", tags_display.join(", "));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn cli(args: &[&str]) -> Cli {
        let mut argv = vec!["bench"];
        argv.extend_from_slice(args);
        Cli::try_parse_from(argv).unwrap()
    }

    fn benches() -> Vec<&'static BenchmarkDef> {
        let defs = vec![
            BenchmarkDef {
                group: "sorting",
                tags: &["fast"],
                ..crate::test_support::bench_def("sort_a")
            },
            BenchmarkDef {
                group: "io",
                tags: &["net"],
                ..crate::test_support::bench_def("io_b")
            },
        ];
        defs.into_iter().map(|d| &*Box::leak(Box::new(d))).collect()
    }

    fn err(args: &[&str]) -> String {
        filter_benchmarks(&cli(args), &benches(), &GroupTree::default())
            .unwrap_err()
            .to_string()
    }

    #[test]
    fn invalid_regex_is_an_error_naming_it() {
        let e = err(&["sort_(a"]);
        assert!(e.contains("invalid filter regex `sort_(a`"), "{e}");
        assert!(e.contains("unclosed group"), "{e}");
    }

    #[test]
    fn unknown_group_and_tags_are_errors() {
        assert!(err(&["--group", "nope"]).contains("Known groups: io, sorting"));
        assert!(err(&["--tag", "slow"]).contains("unknown --tag `slow`. Known tags: fast, net"));
        assert!(err(&["--skip-tag", "slow"]).contains("unknown --skip-tag `slow`"));
    }

    #[test]
    fn valid_filter_matching_nothing_is_empty() {
        let selected =
            filter_benchmarks(&cli(&["no_such_bench"]), &benches(), &GroupTree::default()).unwrap();
        assert!(selected.is_empty());
    }

    #[test]
    fn filters_select_by_regex_group_and_tag() {
        let pick = |args: &[&str]| -> Vec<&str> {
            filter_benchmarks(&cli(args), &benches(), &GroupTree::default())
                .unwrap()
                .into_iter()
                .map(|b| b.id)
                .collect()
        };
        assert_eq!(pick(&["sort"]), ["sort_a"]);
        assert_eq!(pick(&["--group", "io"]), ["io_b"]);
        assert_eq!(pick(&["--skip-tag", "net"]), ["sort_a"]);
    }
}
