//! Command-line arguments.

use crate::config;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

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

    /// Regression threshold percentage. It overrides flux.toml `ci.regression_threshold`.
    ///
    /// A change counts only when it is also statistically significant.
    /// A per-benchmark `threshold` replaces it.
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

    /// Save benchmark results as baseline JSON, raw samples included
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
    /// Compare against the baseline saved at a git ref
    ///
    /// The baseline comes from --baseline, flux.toml, or target/fluxbench/baseline.json.
    Compare {
        /// Git ref to compare against (e.g., origin/main)
        #[arg(name = "REF")]
        git_ref: String,
    },
}
