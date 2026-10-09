//! Benchmark discovery, filtering, and listing.

use crate::{Cli, planner};
use fluxbench_core::BenchmarkDef;
use regex::Regex;
use std::collections::{BTreeMap, BTreeSet};

/// Every registered benchmark, and the ones the CLI filters select.
///
/// The selection is sorted by ID for deterministic execution.
pub(crate) fn select_benchmarks(
    cli: &Cli,
) -> (Vec<&'static BenchmarkDef>, Vec<&'static BenchmarkDef>) {
    let all: Vec<&'static BenchmarkDef> = inventory::iter::<BenchmarkDef>.into_iter().collect();
    let selected = filter_benchmarks(cli, &all);
    (all, selected)
}

/// Filter benchmarks by the CLI regex, group, and tags.
fn filter_benchmarks(
    cli: &Cli,
    benchmarks: &[&'static BenchmarkDef],
) -> Vec<&'static BenchmarkDef> {
    let filter_re = Regex::new(&cli.filter).ok();

    planner::build_plan(
        benchmarks.iter().copied(),
        filter_re.as_ref(),
        cli.group.as_deref(),
        cli.tag.as_deref(),
        cli.skip_tag.as_deref(),
    )
    .benchmarks
}

/// Explain an empty selection on stderr. An unknown `--tag` lists the known tags.
pub(crate) fn report_empty_selection(cli: &Cli, all: &[&'static BenchmarkDef]) {
    if let Some(ref tag) = cli.tag {
        let all_tags: BTreeSet<&str> = all.iter().flat_map(|b| b.tags.iter().copied()).collect();
        if !all_tags.contains(tag.as_str()) {
            let available: Vec<&str> = all_tags.into_iter().collect();
            eprintln!(
                "Warning: tag '{}' not found. Available tags: {}",
                tag,
                available.join(", ")
            );
        }
    }
    eprintln!("No benchmarks found.");
}

/// Print the benchmark plan grouped by group, then every tag in the suite.
pub(crate) fn list_benchmarks(cli: &Cli) {
    println!("FluxBench Plan:");

    let (all_benchmarks, benchmarks) = select_benchmarks(cli);

    let mut groups: BTreeMap<&str, Vec<&BenchmarkDef>> = BTreeMap::new();
    for bench in &benchmarks {
        groups.entry(bench.group).or_default().push(bench);
    }

    let mut total = 0;
    for (group, benches) in &groups {
        println!("├── group: {}", group);
        for bench in benches {
            let tags = if bench.tags.is_empty() {
                String::new()
            } else {
                format!(" [{}]", bench.tags.join(", "))
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
    for bench in &all_benchmarks {
        for tag in bench.tags {
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
}
