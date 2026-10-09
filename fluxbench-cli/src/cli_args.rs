//! Command-line arguments.

use crate::{ReportFormat, config};
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

    /// Filter benchmarks by regex on the benchmark ID. An invalid regex is an error.
    #[arg(default_value = ".*")]
    pub filter: String,

    /// Report format [default: flux.toml output.format, else human]
    #[arg(long, value_enum)]
    pub format: Option<ReportFormat>,

    /// Report file (stdout if not specified)
    ///
    /// A bare file name goes under flux.toml output.directory when that key is set.
    /// Otherwise the path stays as written.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Load baseline for comparison
    ///
    /// Without a path: flux.toml output.baseline_path, else baseline.json.
    /// A bare file name goes under flux.toml output.directory, else target/fluxbench.
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

    /// Run benchmarks of this group and its #[flux::group] child groups. An unknown group is an error.
    #[arg(long)]
    pub group: Option<String>,

    /// Run benchmarks with this tag, from #[bench] or #[flux::group]. An unknown tag is an error.
    #[arg(long)]
    pub tag: Option<String>,

    /// Skip benchmarks with this tag, from #[bench] or #[flux::group]. An unknown tag is an error.
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

    /// Run benchmarks in isolated worker processes [default: flux.toml runner.isolation, else true]
    ///
    /// Use --isolated=false to run in-process.
    #[arg(long, action = clap::ArgAction::Set)]
    pub isolated: Option<bool>,

    /// Use fresh worker process for each benchmark (One-Shot mode)
    /// Default is Persistent mode: reuse worker for safe Rust code
    #[arg(long)]
    pub one_shot: bool,

    /// Worker timeout per benchmark, e.g. `120` or `2m`. A bare number means seconds.
    /// [default: flux.toml runner.timeout, else 60s]
    ///
    /// Held in nanoseconds. A value below warmup + measurement + 2s rises to that, with a warning.
    #[arg(long, value_parser = config::parse_duration_arg)]
    pub worker_timeout: Option<u64>,

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
    ///
    /// Without a path: flux.toml output.baseline_path, else baseline.json.
    /// A bare file name goes under flux.toml output.directory, else target/fluxbench.
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
    /// The baseline comes from --baseline, flux.toml output.baseline_path, or baseline.json under output.directory (default target/fluxbench).
    Compare {
        /// Git ref to compare against (e.g., origin/main)
        #[arg(name = "REF")]
        git_ref: String,
    },
}
