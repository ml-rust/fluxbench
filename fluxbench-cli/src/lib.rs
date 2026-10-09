#![warn(missing_docs)]
//! FluxBench CLI Library
//!
//! This module provides the CLI infrastructure for benchmark binaries.
//! Use `fluxbench::run()` (or `fluxbench_cli::run()`) in your main function to get the full
//! fluxbench CLI experience with your registered benchmarks.
//!
//! # Example
//!
//! ```ignore
//! use fluxbench::prelude::*;
//!
//! #[bench]
//! fn my_benchmark(b: &mut Bencher) {
//!     b.iter(|| expensive_operation());
//! }
//!
//! fn main() {
//!     fluxbench_cli::run();
//! }
//! ```

mod config;
mod cpu_pin;
mod executor;
mod planner;
mod run_config;
mod status;
mod supervisor;

pub use config::*;
pub use executor::{
    ExecutionConfig, Executor, IsolatedExecutor, build_report, compute_statistics,
    execute_verifications, format_human_output,
};
pub use supervisor::*;

use clap::{Parser, Subcommand};
use fluxbench_core::{BenchmarkDef, WorkerMain};
use fluxbench_logic::aggregate_verifications;
use fluxbench_report::{
    OutputFormat, format_duration, generate_csv_report, generate_github_action_benchmark,
    generate_github_summary, generate_html_report, generate_json_report,
};
use rayon::ThreadPoolBuilder;
use regex::Regex;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

/// FluxBench CLI arguments
#[derive(Parser, Debug)]
#[command(name = "fluxbench")]
#[command(author, version, about = "FluxBench - benchmarking framework for Rust")]
pub struct Cli {
    /// Optional subcommand (List, Run, Compare); defaults to Run
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Filter benchmarks by regex pattern
    #[arg(default_value = ".*")]
    pub filter: String,

    /// Output format: json, github-summary, github-action, csv, html, human
    #[arg(long, default_value = "human")]
    pub format: String,

    /// Output file (stdout if not specified)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Load baseline for comparison
    /// Optionally specify a path; defaults to config or target/fluxbench/baseline.json
    #[arg(long)]
    pub baseline: Option<Option<PathBuf>>,

    /// Dry run - list benchmarks without executing
    #[arg(long)]
    pub dry_run: bool,

    /// Regression threshold percentage
    #[arg(long)]
    pub threshold: Option<f64>,

    /// Run benchmarks for this group only
    #[arg(long)]
    pub group: Option<String>,

    /// Filter by tag
    #[arg(long)]
    pub tag: Option<String>,

    /// Skip benchmarks with this tag
    #[arg(long)]
    pub skip_tag: Option<String>,

    /// Warmup time, e.g. `3s` or `500ms`. A bare number means seconds. [default: 3s]
    ///
    /// Held in nanoseconds. It overrides flux.toml when set.
    #[arg(long, value_parser = config::parse_duration_arg)]
    pub warmup: Option<u64>,

    /// Measurement time, e.g. `5s` or `500ms`. A bare number means seconds. [default: 5s]
    ///
    /// Held in nanoseconds. It overrides flux.toml when set.
    #[arg(long, value_parser = config::parse_duration_arg)]
    pub measurement: Option<u64>,

    /// Fixed mode: N samples of one iteration each, no warmup.
    /// Ignores warmup and min/max iterations. `--sample-count` overrides N.
    #[arg(long, short = 'n', value_parser = clap::value_parser!(u64).range(2..))]
    pub samples: Option<u64>,

    /// Target sample count (minimum 2, no upper cap) [default: 100]
    #[arg(long, value_parser = clap::value_parser!(u64).range(2..))]
    pub sample_count: Option<u64>,

    /// Minimum total measurement iterations
    #[arg(long)]
    pub min_iterations: Option<u64>,

    /// Maximum total measurement iterations
    #[arg(long)]
    pub max_iterations: Option<u64>,

    /// Verbose output
    #[arg(short, long)]
    pub verbose: bool,

    /// Run benchmarks in isolated worker processes (default: true)
    /// Use --isolated=false to disable and run in-process
    #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
    pub isolated: bool,

    /// Use fresh worker process for each benchmark (One-Shot mode)
    /// Default is Persistent mode: reuse worker for safe Rust code
    #[arg(long)]
    pub one_shot: bool,

    /// Worker timeout in seconds
    #[arg(long, default_value = "60")]
    pub worker_timeout: u64,

    /// Number of parallel isolated workers [default: 1]
    ///
    /// It overrides flux.toml when set.
    #[arg(long)]
    pub jobs: Option<usize>,

    /// Pin each isolated worker to its own CPU (default)
    ///
    /// It overrides flux.toml `runner.pin_cpus`.
    #[arg(long, conflicts_with = "no_pin")]
    pub pin: bool,

    /// Do not pin isolated workers to CPUs
    ///
    /// It overrides flux.toml `runner.pin_cpus`.
    #[arg(long)]
    pub no_pin: bool,

    /// Number of threads for parallel statistics computation
    /// 0 = use all available cores (default), 1 = single-threaded
    #[arg(long, short = 'j', default_value = "0")]
    pub threads: usize,

    /// Internal: Run as worker process (used by supervisor)
    #[arg(long, hide = true)]
    pub flux_worker: bool,

    /// Save benchmark results as baseline JSON
    /// Optionally specify a path; defaults to config or target/fluxbench/baseline.json
    #[arg(long)]
    pub save_baseline: Option<Option<PathBuf>>,

    /// Internal: Absorb cargo bench's --bench flag
    #[arg(long, hide = true)]
    pub bench: bool,
}

/// CLI subcommands
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// List all discovered benchmarks
    List,
    /// Run benchmarks (default)
    Run,
    /// Compare against a git ref
    Compare {
        /// Git ref to compare against (e.g., origin/main)
        #[arg(name = "REF")]
        git_ref: String,
    },
}

/// Run the FluxBench CLI with the given arguments.
/// This is the main entry point for benchmark binaries.
///
/// # Returns
/// Returns `Ok(())` on success, or an error if something goes wrong.
pub fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    run_with_cli(cli)
}

/// Run the FluxBench CLI with pre-parsed arguments.
pub fn run_with_cli(cli: Cli) -> anyhow::Result<()> {
    // Handle worker mode first (before any other initialization)
    if cli.flux_worker {
        return run_worker_mode();
    }

    // Initialize logging
    if cli.verbose {
        tracing_subscriber::fmt()
            .with_env_filter("fluxbench=debug")
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_env_filter("fluxbench=info")
            .init();
    }

    // Discover flux.toml configuration (CLI flags override)
    let config = FluxConfig::discover().unwrap_or_default();

    // Parse output format
    let format: OutputFormat = cli.format.parse().unwrap_or(OutputFormat::Human);

    let jobs = run_config::resolve_jobs(&cli, &config);

    match cli.command {
        Some(Commands::List) => {
            list_benchmarks(&cli)?;
        }
        Some(Commands::Run) => {
            run_benchmarks(&cli, &config, format, jobs)?;
        }
        Some(Commands::Compare { ref git_ref }) => {
            compare_benchmarks(&cli, &config, git_ref, format)?;
        }
        None => {
            // Default: run benchmarks
            if cli.dry_run {
                list_benchmarks(&cli)?;
            } else {
                run_benchmarks(&cli, &config, format, jobs)?;
            }
        }
    }

    Ok(())
}

/// Run as a worker process (IPC mode)
fn run_worker_mode() -> anyhow::Result<()> {
    let mut worker = WorkerMain::new();
    worker
        .run()
        .map_err(|e| anyhow::anyhow!("Worker error: {}", e))
}

/// Filter benchmarks based on CLI options using the planner module.
///
/// Returns benchmarks sorted alphabetically by ID for deterministic execution.
fn filter_benchmarks(
    cli: &Cli,
    benchmarks: &[&'static BenchmarkDef],
) -> Vec<&'static BenchmarkDef> {
    let filter_re = Regex::new(&cli.filter).ok();

    let plan = planner::build_plan(
        benchmarks.iter().copied(),
        filter_re.as_ref(),
        cli.group.as_deref(),
        cli.tag.as_deref(),
        cli.skip_tag.as_deref(),
    );

    plan.benchmarks
}

fn list_benchmarks(cli: &Cli) -> anyhow::Result<()> {
    println!("FluxBench Plan:");

    let all_benchmarks: Vec<_> = inventory::iter::<BenchmarkDef>.into_iter().collect();
    let benchmarks = filter_benchmarks(cli, &all_benchmarks);

    let mut groups: std::collections::BTreeMap<&str, Vec<&BenchmarkDef>> =
        std::collections::BTreeMap::new();

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

    // Show all available tags across the entire suite (not just filtered results)
    // so users can discover what tags they can filter by.
    let mut tag_counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
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

    Ok(())
}

fn run_benchmarks(
    cli: &Cli,
    config: &FluxConfig,
    format: OutputFormat,
    jobs: usize,
) -> anyhow::Result<()> {
    let jobs = jobs.max(1);

    // Configure Rayon thread pool for statistics computation
    if cli.threads > 0 {
        ThreadPoolBuilder::new()
            .num_threads(cli.threads)
            .build_global()
            .ok();
    }

    // Discover benchmarks
    let all_benchmarks: Vec<_> = inventory::iter::<BenchmarkDef>.into_iter().collect();
    let benchmarks = filter_benchmarks(cli, &all_benchmarks);

    if benchmarks.is_empty() {
        // If filtering by tag and no matches, check if the tag exists at all
        if let Some(ref tag) = cli.tag {
            let all_tags: std::collections::BTreeSet<&str> = all_benchmarks
                .iter()
                .flat_map(|b| b.tags.iter().copied())
                .collect();
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
        return Ok(());
    }

    // Determine isolation mode: flux.toml can override CLI default
    let isolated = if config.runner.isolation.is_isolated() {
        cli.isolated
    } else {
        false
    };

    let threads_str = if cli.threads == 0 {
        "all".to_string()
    } else {
        cli.threads.to_string()
    };
    let mode_str = if isolated {
        if cli.one_shot {
            " (isolated, one-shot)"
        } else {
            " (isolated, persistent)"
        }
    } else {
        " (in-process)"
    };
    status::print_run_banner(benchmarks.len(), mode_str, &threads_str, jobs);

    let start_time = Instant::now();

    // Build execution config from flux.toml + CLI overrides
    let exec_config = run_config::build_execution_config(cli, config);

    if exec_config.bootstrap_iterations > 0 && exec_config.bootstrap_iterations < 100 {
        eprintln!(
            "Warning: bootstrap_iterations={} is very low; confidence intervals will be unreliable. \
             Use >= 1000 for meaningful results, or 0 to skip bootstrap.",
            exec_config.bootstrap_iterations
        );
    }

    // Execute benchmarks (isolated by default per TDD)
    let results = if isolated {
        let timeout = std::time::Duration::from_secs(cli.worker_timeout);
        let reuse_workers = !cli.one_shot;
        let allowed = cpu_pin::allowed_cpus();
        let pinned = run_config::resolve_pin(cli, config) && !allowed.is_empty();
        status::print_parallel_warnings(jobs, pinned.then(|| cpu_pin::usable_cpus(&allowed).len()));
        let cpus = if pinned {
            cpu_pin::assign_cpus(&allowed, jobs)
        } else {
            Vec::new()
        };
        let isolated_executor =
            IsolatedExecutor::new(exec_config.clone(), timeout, reuse_workers, jobs)
                .with_cpus(cpus);
        isolated_executor.execute(&benchmarks)
    } else {
        if jobs > 1 {
            eprintln!(
                "Warning: --jobs currently applies only to isolated mode; running in-process serially."
            );
        }
        let mut executor = Executor::new(exec_config.clone());
        executor.execute(&benchmarks)
    };

    // Compute statistics
    let stats = compute_statistics(&results, &exec_config);

    // Warn if allocation tracking is enabled but nothing was recorded
    if exec_config.track_allocations
        && !results.is_empty()
        && results
            .iter()
            .all(|r| r.alloc_bytes == 0 && r.alloc_count == 0)
    {
        eprintln!(
            "Warning: allocation tracking enabled but all benchmarks reported 0 bytes allocated.\n\
             Ensure TrackingAllocator is set as #[global_allocator] in your benchmark binary."
        );
    }

    // Build report
    let total_duration_ms = start_time.elapsed().as_secs_f64() * 1000.0;
    let mut report = build_report(&results, &stats, &exec_config, total_duration_ms);

    // Load and apply baseline comparison if --baseline was passed
    if let Some(baseline_path) = resolve_baseline_path(&cli.baseline, config) {
        if baseline_path.exists() {
            match std::fs::read_to_string(&baseline_path).and_then(|json| {
                serde_json::from_str::<fluxbench_report::Report>(&json)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
            }) {
                Ok(baseline) => {
                    let threshold = cli.threshold.unwrap_or(config.ci.regression_threshold);
                    apply_baseline_comparison(&mut report, &baseline, threshold);
                }
                Err(e) => {
                    eprintln!(
                        "Warning: failed to load baseline {}: {}",
                        baseline_path.display(),
                        e
                    );
                }
            }
        } else {
            eprintln!(
                "Warning: baseline file not found: {}",
                baseline_path.display()
            );
        }
    }

    // Run comparisons, synthetics, and verifications
    let (comparison_results, comparison_series, synthetic_results, verification_results) =
        execute_verifications(&results, &stats);
    let verification_summary = aggregate_verifications(&verification_results);
    report.comparisons = comparison_results;
    report.comparison_series = comparison_series;
    report.synthetics = synthetic_results;
    report.verifications = verification_results;

    // Update summary with verification info
    report.summary.critical_failures = verification_summary.critical_failures;
    report.summary.warnings = verification_summary.failed - verification_summary.critical_failures;

    // Emit GitHub Actions annotations if enabled
    if config.ci.github_annotations {
        status::print_github_annotations(&report);
    }

    // Generate output
    let output = match format {
        OutputFormat::Json => generate_json_report(&report)?,
        OutputFormat::GithubSummary => generate_github_summary(&report),
        OutputFormat::GithubActionBenchmark => generate_github_action_benchmark(&report),
        OutputFormat::Html => generate_html_report(&report),
        OutputFormat::Csv => generate_csv_report(&report),
        OutputFormat::Human => format_human_output(&report),
    };

    // Write output
    if let Some(ref path) = cli.output {
        let mut file = std::fs::File::create(path)?;
        file.write_all(output.as_bytes())?;
        eprintln!("Report written to: {}", path.display());
    } else {
        print!("{}", output);
    }

    // Save baseline if requested
    save_baseline_if_needed(cli, config, &report)?;

    // Exit with appropriate code
    let has_crashes = report
        .results
        .iter()
        .any(|r| matches!(r.status, fluxbench_report::BenchmarkStatus::Crashed));

    if verification_summary.should_fail_ci() || has_crashes {
        if has_crashes {
            eprintln!("\nBenchmark(s) crashed during execution");
        }
        if verification_summary.should_fail_ci() {
            eprintln!(
                "\n{} critical verification failure(s)",
                verification_summary.critical_failures + verification_summary.critical_errors
            );
        }
        std::process::exit(1);
    }

    Ok(())
}

fn compare_benchmarks(
    cli: &Cli,
    config: &FluxConfig,
    git_ref: &str,
    format: OutputFormat,
) -> anyhow::Result<()> {
    // Load baseline — resolve path from CLI, config, or default
    let baseline_path = resolve_baseline_path(&cli.baseline, config).ok_or_else(|| {
        anyhow::anyhow!(
            "--baseline required for comparison, or use 'compare' command with a git ref"
        )
    })?;

    if !baseline_path.exists() {
        return Err(anyhow::anyhow!(
            "Baseline file not found: {}",
            baseline_path.display()
        ));
    }

    let baseline_json = std::fs::read_to_string(&baseline_path)?;
    let baseline: fluxbench_report::Report = serde_json::from_str(&baseline_json)?;
    let resolved_git_ref = resolve_git_ref(git_ref)?;

    if let Some(baseline_commit) = baseline.meta.git_commit.as_deref() {
        let matches_ref = baseline_commit == resolved_git_ref
            || baseline_commit.starts_with(&resolved_git_ref)
            || resolved_git_ref.starts_with(baseline_commit);
        if !matches_ref {
            return Err(anyhow::anyhow!(
                "Baseline commit {} does not match git ref {} ({})",
                baseline_commit,
                git_ref,
                resolved_git_ref
            ));
        }
    } else {
        eprintln!(
            "Warning: baseline report has no commit metadata; git ref consistency cannot be verified."
        );
    }

    eprintln!("Comparing against baseline: {}", baseline_path.display());
    eprintln!("Git ref: {} ({})\n", git_ref, resolved_git_ref);

    // Run current benchmarks
    let all_benchmarks: Vec<_> = inventory::iter::<BenchmarkDef>.into_iter().collect();
    let benchmarks = filter_benchmarks(cli, &all_benchmarks);

    if benchmarks.is_empty() {
        eprintln!("No benchmarks found.");
        return Ok(());
    }

    let start_time = Instant::now();

    let exec_config = run_config::build_execution_config(cli, config);

    let mut executor = Executor::new(exec_config.clone());
    let results = executor.execute(&benchmarks);
    let stats = compute_statistics(&results, &exec_config);

    let total_duration_ms = start_time.elapsed().as_secs_f64() * 1000.0;
    let mut report = build_report(&results, &stats, &exec_config, total_duration_ms);

    // Apply baseline comparison data
    let regression_threshold = cli.threshold.unwrap_or(config.ci.regression_threshold);
    apply_baseline_comparison(&mut report, &baseline, regression_threshold);

    // Run comparisons, synthetics, and verifications
    let (comparison_results, comparison_series, synthetic_results, verification_results) =
        execute_verifications(&results, &stats);
    let verification_summary = aggregate_verifications(&verification_results);
    report.comparisons = comparison_results;
    report.comparison_series = comparison_series;
    report.synthetics = synthetic_results;
    report.verifications = verification_results;
    report.summary.critical_failures = verification_summary.critical_failures;
    report.summary.warnings = verification_summary.failed - verification_summary.critical_failures;

    // Emit GitHub Actions annotations if enabled
    if config.ci.github_annotations {
        status::print_github_annotations(&report);
    }

    // Generate output
    let output = match format {
        OutputFormat::Json => generate_json_report(&report)?,
        OutputFormat::GithubSummary => generate_github_summary(&report),
        OutputFormat::GithubActionBenchmark => generate_github_action_benchmark(&report),
        OutputFormat::Html => generate_html_report(&report),
        OutputFormat::Csv => generate_csv_report(&report),
        OutputFormat::Human => format_comparison_output(&report, &baseline),
    };

    if let Some(ref path) = cli.output {
        let mut file = std::fs::File::create(path)?;
        file.write_all(output.as_bytes())?;
        eprintln!("Report written to: {}", path.display());
    } else {
        print!("{}", output);
    }

    // Save baseline if requested
    save_baseline_if_needed(cli, config, &report)?;

    // Exit with error if regressions exceed threshold or verifications fail
    let should_fail = report.summary.regressions > 0 || verification_summary.should_fail_ci();
    if should_fail {
        if report.summary.regressions > 0 {
            eprintln!(
                "\n{} regression(s) detected above {}% threshold",
                report.summary.regressions, regression_threshold
            );
        }
        if verification_summary.should_fail_ci() {
            eprintln!(
                "\n{} critical verification failure(s)",
                verification_summary.critical_failures + verification_summary.critical_errors
            );
        }
        std::process::exit(1);
    }

    Ok(())
}

/// Save the report as a baseline JSON file if configured.
fn save_baseline_if_needed(
    cli: &Cli,
    config: &FluxConfig,
    report: &fluxbench_report::Report,
) -> anyhow::Result<()> {
    // Determine if we should save: CLI --save-baseline flag or config.output.save_baseline
    let should_save = cli.save_baseline.is_some() || config.output.save_baseline;
    if !should_save {
        return Ok(());
    }

    // Resolve path: CLI value > config value > default
    let path = cli
        .save_baseline
        .as_ref()
        .and_then(|opt| opt.clone())
        .or_else(|| config.output.baseline_path.as_ref().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("target/fluxbench/baseline.json"));

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let json = generate_json_report(report)?;
    std::fs::write(&path, json)?;
    eprintln!("Baseline saved to: {}", path.display());

    Ok(())
}

/// Apply baseline comparison data to the report.
///
/// Computes per-benchmark regression/improvement metrics by comparing current
/// results against baseline means, CI overlap, and effect size.
fn apply_baseline_comparison(
    report: &mut fluxbench_report::Report,
    baseline: &fluxbench_report::Report,
    regression_threshold: f64,
) {
    report.baseline_meta = Some(baseline.meta.clone());

    let baseline_map: std::collections::HashMap<_, _> = baseline
        .results
        .iter()
        .filter_map(|r| r.metrics.as_ref().map(|m| (r.id.clone(), m.clone())))
        .collect();

    for result in &mut report.results {
        if let (Some(metrics), Some(baseline_metrics)) =
            (&result.metrics, baseline_map.get(&result.id))
        {
            // Use per-benchmark threshold if set (> 0.0), otherwise global
            let effective_threshold = if result.threshold > 0.0 {
                result.threshold
            } else {
                regression_threshold
            };

            let baseline_mean = baseline_metrics.mean_ns;
            let absolute_change = metrics.mean_ns - baseline_mean;
            let relative_change = if baseline_mean > 0.0 {
                (absolute_change / baseline_mean) * 100.0
            } else {
                0.0
            };

            // Without both intervals, overlap is unknown: treat as overlapping.
            let ci_non_overlap = match (
                metrics.ci_lower_ns,
                metrics.ci_upper_ns,
                baseline_metrics.ci_lower_ns,
                baseline_metrics.ci_upper_ns,
            ) {
                (Some(lo), Some(hi), Some(base_lo), Some(base_hi)) => hi < base_lo || lo > base_hi,
                _ => false,
            };
            let is_significant = relative_change.abs() > effective_threshold && ci_non_overlap;

            if relative_change > effective_threshold {
                report.summary.regressions += 1;
            } else if relative_change < -effective_threshold {
                report.summary.improvements += 1;
            }

            let mut effect_size = if metrics.std_dev_ns > f64::EPSILON {
                absolute_change / metrics.std_dev_ns
            } else {
                0.0
            };
            if !effect_size.is_finite() {
                effect_size = 0.0;
            }

            let probability_regression = if ci_non_overlap {
                if relative_change > 0.0 { 0.99 } else { 0.01 }
            } else if relative_change > 0.0 {
                0.60
            } else {
                0.40
            };

            result.comparison = Some(fluxbench_report::Comparison {
                baseline_mean_ns: baseline_mean,
                absolute_change_ns: absolute_change,
                relative_change,
                probability_regression,
                is_significant,
                effect_size,
            });
        }
    }
}

/// Resolve baseline path from CLI flag, config, or default.
///
/// - `Some(Some(path))` — explicit path from `--baseline /path/to/file`
/// - `Some(None)` — `--baseline` with no value, use config or default
/// - `None` — flag not passed at all
fn resolve_baseline_path(
    cli_baseline: &Option<Option<PathBuf>>,
    config: &FluxConfig,
) -> Option<PathBuf> {
    match cli_baseline {
        Some(Some(path)) => Some(path.clone()),
        Some(None) => {
            // --baseline passed without path: use config or default
            Some(
                config
                    .output
                    .baseline_path
                    .as_ref()
                    .map(PathBuf::from)
                    .unwrap_or_else(|| PathBuf::from("target/fluxbench/baseline.json")),
            )
        }
        None => None,
    }
}

fn resolve_git_ref(git_ref: &str) -> anyhow::Result<String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--verify", git_ref])
        .output()
        .map_err(|e| anyhow::anyhow!("Failed to resolve git ref '{}': {}", git_ref, e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow::anyhow!(
            "Invalid git ref '{}': {}",
            git_ref,
            stderr.trim()
        ));
    }

    let resolved = String::from_utf8(output.stdout)?.trim().to_string();
    if resolved.is_empty() {
        return Err(anyhow::anyhow!(
            "Git ref '{}' resolved to an empty commit hash",
            git_ref
        ));
    }

    Ok(resolved)
}

/// Format comparison output for human display
fn format_comparison_output(
    report: &fluxbench_report::Report,
    baseline: &fluxbench_report::Report,
) -> String {
    let mut output = String::new();

    output.push('\n');
    output.push_str("FluxBench Comparison Results\n");
    output.push_str(&"=".repeat(60));
    output.push_str("\n\n");

    output.push_str(&format!(
        "Baseline: {} ({})\n",
        baseline.meta.git_commit.as_deref().unwrap_or("unknown"),
        baseline.meta.timestamp.format("%Y-%m-%d %H:%M:%S")
    ));
    output.push_str(&format!(
        "Current:  {} ({})\n\n",
        report.meta.git_commit.as_deref().unwrap_or("unknown"),
        report.meta.timestamp.format("%Y-%m-%d %H:%M:%S")
    ));

    for result in &report.results {
        let status_icon = match result.status {
            fluxbench_report::BenchmarkStatus::Passed => "✓",
            fluxbench_report::BenchmarkStatus::Failed => "✗",
            fluxbench_report::BenchmarkStatus::Crashed => "💥",
            fluxbench_report::BenchmarkStatus::Skipped => "⊘",
        };

        output.push_str(&format!("{} {}\n", status_icon, result.id));

        if let (Some(metrics), Some(comparison)) = (&result.metrics, &result.comparison) {
            let change_icon = if comparison.relative_change > 5.0 {
                "📈 REGRESSION"
            } else if comparison.relative_change < -5.0 {
                "📉 improvement"
            } else {
                "≈ no change"
            };

            output.push_str(&format!(
                "    baseline: {} → current: {}\n",
                format_duration(comparison.baseline_mean_ns),
                format_duration(metrics.mean_ns),
            ));
            output.push_str(&format!(
                "    change: {:+.2}% ({}) {}\n",
                comparison.relative_change,
                format_duration(comparison.absolute_change_ns.abs()),
                change_icon,
            ));
        }

        output.push('\n');
    }

    // Summary
    output.push_str("Summary\n");
    output.push_str(&"-".repeat(60));
    output.push('\n');
    output.push_str(&format!(
        "  Regressions: {}  Improvements: {}  No Change: {}\n",
        report.summary.regressions,
        report.summary.improvements,
        report.summary.total_benchmarks - report.summary.regressions - report.summary.improvements
    ));

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxbench_report::{
        BenchmarkMetrics, BenchmarkReportResult, BenchmarkStatus, Report, ReportConfig, ReportMeta,
        ReportSummary, SystemInfo,
    };

    fn dummy_meta() -> ReportMeta {
        ReportMeta {
            schema_version: 1,
            version: "0.1.0".to_string(),
            timestamp: chrono::Utc::now(),
            git_commit: None,
            git_branch: None,
            system: SystemInfo {
                os: "linux".to_string(),
                os_version: "6.0".to_string(),
                cpu: "test".to_string(),
                cpu_cores: 1,
                memory_gb: 1.0,
            },
            config: ReportConfig {
                warmup_time_ns: 0,
                measurement_time_ns: 0,
                min_iterations: None,
                max_iterations: None,
                sample_count: 100,
                iters_per_sample: None,
                bootstrap_iterations: 0,
                confidence_level: 0.95,
                track_allocations: false,
                timer_overhead_ns: 0.0,
            },
        }
    }

    fn dummy_metrics(mean: f64) -> BenchmarkMetrics {
        BenchmarkMetrics {
            samples: 100,
            mean_ns: mean,
            trimmed_mean_ns: Some(mean),
            median_ns: mean,
            std_dev_ns: mean * 0.01,
            min_ns: mean * 0.9,
            max_ns: mean * 1.1,
            p50_ns: mean,
            p90_ns: mean * 1.05,
            p95_ns: mean * 1.07,
            p99_ns: mean * 1.09,
            p999_ns: mean * 1.1,
            skewness: 0.0,
            kurtosis: 3.0,
            ci_lower_ns: Some(mean * 0.98),
            ci_upper_ns: Some(mean * 1.02),
            ci_level: 0.95,
            throughput_ops_sec: None,
            alloc_bytes: 0,
            alloc_count: 0,
            mean_cycles: 0.0,
            median_cycles: 0.0,
            min_cycles: 0,
            max_cycles: 0,
            cycles_per_ns: 0.0,
        }
    }

    fn dummy_result(id: &str, mean: f64, threshold: f64) -> BenchmarkReportResult {
        BenchmarkReportResult {
            id: id.to_string(),
            name: id.to_string(),
            group: "test".to_string(),
            status: BenchmarkStatus::Passed,
            severity: fluxbench_core::Severity::Warning,
            file: "test.rs".to_string(),
            line: 1,
            metrics: Some(dummy_metrics(mean)),
            threshold,
            comparison: None,
            failure: None,
        }
    }

    fn dummy_report(results: Vec<BenchmarkReportResult>) -> Report {
        let total = results.len();
        Report {
            meta: dummy_meta(),
            results,
            comparisons: vec![],
            comparison_series: vec![],
            synthetics: vec![],
            verifications: vec![],
            summary: ReportSummary {
                total_benchmarks: total,
                passed: total,
                ..Default::default()
            },
            baseline_meta: None,
        }
    }

    #[test]
    fn per_bench_threshold_overrides_global() {
        // Baseline: 100ns. Current: 108ns → 8% regression.
        // Global threshold: 25%. Per-bench threshold: 5%.
        // Should detect regression via per-bench threshold but not global.
        let mut report = dummy_report(vec![dummy_result("fast_bench", 108.0, 5.0)]);
        let baseline = dummy_report(vec![dummy_result("fast_bench", 100.0, 5.0)]);

        apply_baseline_comparison(&mut report, &baseline, 25.0);

        assert_eq!(
            report.summary.regressions, 1,
            "per-bench 5% should catch 8% regression"
        );
        let cmp = report.results[0].comparison.as_ref().unwrap();
        assert!(cmp.is_significant);
    }

    #[test]
    fn zero_threshold_falls_back_to_global() {
        // Baseline: 100ns. Current: 108ns → 8% regression.
        // Global threshold: 25%. Per-bench threshold: 0.0 (use global).
        // 8% < 25%, so no regression.
        let mut report = dummy_report(vec![dummy_result("normal_bench", 108.0, 0.0)]);
        let baseline = dummy_report(vec![dummy_result("normal_bench", 100.0, 0.0)]);

        apply_baseline_comparison(&mut report, &baseline, 25.0);

        assert_eq!(
            report.summary.regressions, 0,
            "8% under 25% global should not regress"
        );
        let cmp = report.results[0].comparison.as_ref().unwrap();
        assert!(!cmp.is_significant);
    }

    #[test]
    fn mixed_thresholds_independent() {
        // Two benchmarks: one with tight per-bench threshold, one using global.
        // Both regress by 8%.
        let mut report = dummy_report(vec![
            dummy_result("tight", 108.0, 5.0), // per-bench 5% → should regress
            dummy_result("loose", 108.0, 0.0), // global 25% → should not
        ]);
        let baseline = dummy_report(vec![
            dummy_result("tight", 100.0, 5.0),
            dummy_result("loose", 100.0, 0.0),
        ]);

        apply_baseline_comparison(&mut report, &baseline, 25.0);

        assert_eq!(report.summary.regressions, 1);
        assert!(
            report.results[0]
                .comparison
                .as_ref()
                .unwrap()
                .is_significant
        );
        assert!(
            !report.results[1]
                .comparison
                .as_ref()
                .unwrap()
                .is_significant
        );
    }

    #[test]
    fn per_bench_threshold_detects_improvement() {
        // Baseline: 100ns. Current: 90ns → -10% improvement.
        // Per-bench threshold: 5%.
        let mut report = dummy_report(vec![dummy_result("improving", 90.0, 5.0)]);
        let baseline = dummy_report(vec![dummy_result("improving", 100.0, 5.0)]);

        apply_baseline_comparison(&mut report, &baseline, 25.0);

        assert_eq!(report.summary.improvements, 1);
        assert_eq!(report.summary.regressions, 0);
    }
}
