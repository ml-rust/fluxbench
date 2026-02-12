# FluxBench

[![Crates.io](https://img.shields.io/crates/v/fluxbench)](https://crates.io/crates/fluxbench) [![docs.rs](https://img.shields.io/docsrs/fluxbench)](https://docs.rs/fluxbench) [![License](https://img.shields.io/crates/l/fluxbench)](LICENSE) [![MSRV](https://img.shields.io/badge/MSRV-1.85-blue)](https://blog.rust-lang.org/2025/02/20/Rust-1.85.0.html)

Rigorous, configurable, and composable benchmarking framework for Rust.

## The problem

Most Rust benchmarking tools give you timings. That's it. When a benchmark panics, your entire suite dies. When you want to know _"is version B actually faster than version A?"_, you're left eyeballing numbers. And when CI passes despite a 40% regression, you only find out about it from your users.

## What FluxBench does differently

**Start in 5 lines, grow without rewriting.** A benchmark is just a function with `#[bench]`:

```rust
#[bench]
fn my_benchmark(b: &mut Bencher) {
    b.iter(|| expensive_operation());
}
```

Run `cargo bench` and you get bootstrap confidence intervals, percentile stats (p50–p999), outlier detection, and cycle-accurate timing — all automatic.

**Then compose what you need:**

```rust
#[bench(group = "sorting", tags = "alloc")]      // organize and filter
#[verify(expr = "sort_new < sort_old")]          // fail CI if regression
#[synthetic(formula = "sort_old / sort_new")]    // compute speedup ratio
#[compare(baseline = "sort_old")]                // generate comparison tables
```

Each attribute is independent. Use one, use all, add them later — your benchmarks don't need to be restructured.

**Benchmarks that crash don't take down the suite.** Every benchmark runs in its own process. A panic, segfault, or timeout in one is reported as a failure for _that_ benchmark — the rest keep running. Your CI finishes, you see what broke, you fix it.

**Performance rules live next to the code they protect.** Instead of a fragile shell script that parses output and compares numbers, you write `#[verify(expr = "api_latency_p99 < 5000", severity = "critical")]` and FluxBench enforces it on every run. Critical failures exit non-zero. Warnings get reported. Info is logged.

**Multiple output formats.** The same `cargo bench` run can produce terminal output for you, JSON for your pipeline, HTML for your team, CSV for a spreadsheet, or a GitHub Actions summary — just change `--format`.

## Features

| Feature                                                 | Description                                                          |
| ------------------------------------------------------- | -------------------------------------------------------------------- |
| [Crash isolation](#crash-isolation)                     | Supervisor-worker architecture — panics never terminate the suite    |
| [Bootstrap statistics](#custom-bootstrap-configuration) | BCa bootstrap CIs, RDTSC cycle counting, outlier detection, p50–p999 |
| [Verification](#verification-macros)                    | `#[verify(expr = "bench_a < bench_b", severity = "critical")]`       |
| [Synthetic metrics](#synthetic-metrics)                 | `#[synthetic(formula = "bench_a / bench_b", unit = "x")]`            |
| [Comparisons](#comparisons)                             | `#[compare(...)]` — tables and series charts vs baseline             |
| [Output formats](#output-formats)                       | Human, JSON, HTML, CSV, GitHub Actions summary                       |
| [CI integration](#ci-integration)                       | Exit code 1 on critical failures, `flux.toml` severity levels        |
| [Allocation tracking](#allocation-tracking)             | Per-iteration heap bytes and count                                   |
| [Async support](#async-benchmarks)                      | Tokio runtimes via `#[bench(runtime = "multi_thread")]`              |
| [Configuration](#configuration)                         | `flux.toml` with CLI override, macro > CLI > file > default          |

## Quick Start

### 1. Add Dependency

```toml
[dev-dependencies]
fluxbench = "<latest-version>"
```

### 2. Configure Bench Target

```toml
# Cargo.toml
[[bench]]
name = "my_benchmarks"
harness = false
```

### 3. Write Benchmarks

Create `benches/my_benchmarks.rs`:

```rust
use fluxbench::prelude::*;
use std::hint::black_box;

#[bench]
fn addition(b: &mut Bencher) {
    b.iter(|| black_box(42) + black_box(17));
}

#[bench(group = "compute")]
fn fibonacci(b: &mut Bencher) {
    fn fib(n: u32) -> u64 {
        if n <= 1 { n as u64 } else { fib(n - 1) + fib(n - 2) }
    }
    b.iter(|| black_box(fib(20)));
}

fn main() {
    fluxbench::run().unwrap();
}
```

Benchmarks can also live in `examples/` (`cargo run --example name --release`). Both `benches/` and `examples/` are only compiled on demand and never included in your production binary.

### 4. Run Benchmarks

```bash
cargo bench
```

Or with specific CLI options:

```bash
cargo bench -- --group compute --warmup 5 --measurement 10
```

## Defining Benchmarks

### With Setup

```rust
#[bench]
fn with_setup(b: &mut Bencher) {
    b.iter_with_setup(
        || vec![1, 2, 3, 4, 5],  // Setup
        |data| data.iter().sum::<i32>()  // Measured code
    );
}
```

### Grouping Benchmarks

```rust
#[bench(group = "sorting")]
fn sort_small(b: &mut Bencher) {
    let data: Vec<i32> = (0..100).collect();
    b.iter(|| {
        let mut v = data.clone();
        v.sort();
        v
    });
}

#[bench(group = "sorting")]
fn sort_large(b: &mut Bencher) {
    let data: Vec<i32> = (0..100000).collect();
    b.iter(|| {
        let mut v = data.clone();
        v.sort();
        v
    });
}
```

### Tagging for Filtering

```rust
#[bench(group = "io", tags = "network")]
fn http_request(b: &mut Bencher) {
    // ...
}

#[bench(group = "io", tags = "file")]
fn disk_write(b: &mut Bencher) {
    // ...
}
```

Then run with: `cargo bench -- --tag network` or `cargo bench -- --skip-tag network`

### Async Benchmarks

```rust
#[bench(runtime = "multi_thread", worker_threads = 4, group = "async")]
async fn async_operation(b: &mut Bencher) {
    b.iter_async(|| async {
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
    });
}
```

Runtimes: `"multi_thread"` or `"current_thread"`

## Performance Assertions

### Verification Macros

Assert that benchmarks meet performance criteria:

```rust
use fluxbench::verify;

#[verify(
    expr = "fibonacci < 50000",  // Less than 50us
    severity = "critical"
)]
struct FibUnder50us;

#[verify(
    expr = "fibonacci_iter < fibonacci_naive",
    severity = "warning"
)]
struct IterFasterThanNaive;

#[verify(
    expr = "fibonacci_naive_p99 < 1000000",  // p99 latency
    severity = "info"
)]
struct P99Check;
```

**Severity Levels:**

- `critical`: Fails the benchmark suite (exit code 1)
- `warning`: Reported but doesn't fail
- `info`: Informational only

**Available Metrics** (for a benchmark named `bench_name`):

| Suffix                              | Metric                              |
| ----------------------------------- | ----------------------------------- |
| _(none)_                            | Mean time (ns)                      |
| `_median`                           | Median time                         |
| `_min` / `_max`                     | Min / max time                      |
| `_p50` `_p90` `_p95` `_p99` `_p999` | Percentiles                         |
| `_std_dev`                          | Standard deviation                  |
| `_skewness` / `_kurtosis`           | Distribution shape                  |
| `_ci_lower` / `_ci_upper`           | 95% confidence interval bounds      |
| `_throughput`                       | Operations per second (if measured) |

### Synthetic Metrics

Compute derived metrics:

```rust
use fluxbench::synthetic;

#[synthetic(
    id = "speedup",
    formula = "fibonacci_naive / fibonacci_iter",
    unit = "x"
)]
struct FibSpeedup;

#[verify(expr = "speedup > 100", severity = "critical")]
struct SpeedupSignificant;
```

The formula supports:

- Arithmetic: `+`, `-`, `*`, `/`, `%`
- Comparison: `<`, `>`, `<=`, `>=`, `==`, `!=`
- Logical: `&&`, `||`
- Parentheses for grouping

## Comparisons

### Simple Comparison

```rust
use fluxbench::compare;

#[compare(
    id = "string_ops",
    title = "String Operations",
    benchmarks = ["bench_string_concat", "bench_string_parse"],
    baseline = "bench_string_concat",
    metric = "mean"
)]
struct StringComparison;
```

Generates a table showing speedup vs baseline for each benchmark.

### Series Comparison

Create multi-point comparisons for scaling studies:

```rust
#[bench(group = "scaling")]
fn vec_sum_100(b: &mut Bencher) {
    let data: Vec<i64> = (0..100).collect();
    b.iter(|| data.iter().sum::<i64>());
}

#[bench(group = "scaling")]
fn vec_sum_1000(b: &mut Bencher) {
    let data: Vec<i64> = (0..1000).collect();
    b.iter(|| data.iter().sum::<i64>());
}

#[compare(
    id = "scale_100",
    title = "Vector Sum Scaling",
    benchmarks = ["bench_vec_sum_100"],
    group = "vec_scaling",
    x = "100"
)]
struct Scale100;

#[compare(
    id = "scale_1000",
    title = "Vector Sum Scaling",
    benchmarks = ["bench_vec_sum_1000"],
    group = "vec_scaling",
    x = "1000"
)]
struct Scale1000;
```

Multiple `#[compare]` with the same `group` combine into one chart.

## CLI Usage

Run benchmarks with options:

```bash
cargo bench -- [OPTIONS] [FILTER]
```

### Common Options

```bash
# List benchmarks without running
cargo bench -- list

# Run specific benchmark by regex
cargo bench -- bench_fib

# Run only a group
cargo bench -- --group sorting

# Filter by tag
cargo bench -- --tag expensive
cargo bench -- --skip-tag network

# Control execution
cargo bench -- --warmup 10 --measurement 20    # Seconds
cargo bench -- --min-iterations 100
cargo bench -- --isolated=false                # In-process mode
cargo bench -- --worker-timeout 120            # Worker timeout in seconds

# Output formats
cargo bench -- --format json --output results.json
cargo bench -- --format html --output results.html
cargo bench -- --format csv --output results.csv
cargo bench -- --format github-summary         # GitHub Actions summary

# Baseline comparison
cargo bench -- --baseline previous_results.json

# Dry run
cargo bench -- --dry-run
```

Run `cargo bench -- --help` for the full option reference.

## Configuration

FluxBench works out of the box with sensible defaults — no configuration file is needed. For workspace-wide customization, you can optionally create a `flux.toml` in your project root. FluxBench auto-discovers it by walking up from the current directory.

Settings are applied in this priority order: **macro attribute > CLI flag > flux.toml > built-in default**.

### Runner

```toml
[runner]
warmup_time = "500ms"        # Warmup before measurement (default: "3s")
measurement_time = "1s"      # Measurement duration (default: "5s")
timeout = "30s"              # Per-benchmark timeout (default: "60s")
isolation = "process"        # "process", "in-process", or "thread" (default: "process")
bootstrap_iterations = 1000  # Bootstrap resamples for CIs (default: 10000)
confidence_level = 0.95      # Confidence level, 0.0–1.0 (default: 0.95)
# samples = 5               # Fixed sample count — skips warmup, runs exactly N iterations
# min_iterations = 100       # Minimum iterations per sample (default: auto-tuned)
# max_iterations = 1000000   # Maximum iterations per sample (default: auto-tuned)
# jobs = 4                   # Parallel isolated workers (default: sequential)
```

### Allocator

```toml
[allocator]
track = true                 # Track allocations during benchmarks (default: true)
fail_on_allocation = false   # Fail if any allocation occurs during measurement (default: false)
# max_bytes_per_iter = 1024  # Maximum bytes per iteration (default: unlimited)
```

### Output

```toml
[output]
format = "human"                    # "human", "json", "github", "html", "csv" (default: "human")
directory = "target/fluxbench"      # Output directory for reports and baselines (default: "target/fluxbench")
save_baseline = false               # Save a JSON baseline after each run (default: false)
# baseline_path = "baseline.json"   # Compare against a saved baseline (default: unset)
```

### CI Integration

```toml
[ci]
regression_threshold = 5.0   # Fail CI if regression exceeds this percentage (default: 5.0)
github_annotations = true    # Emit ::warning and ::error annotations on PRs (default: false)
fail_on_critical = true      # Exit non-zero on critical verification failures (default: true)
```

## Output Formats

### Human (Default)

Console output with grouped results and statistics:

```
Group: compute
------------------------------------------------------------
  ✓ bench_fibonacci_iter
      mean: 127.42 ns  median: 127.00 ns  stddev: 0.77 ns
      min: 126.00 ns  max: 147.00 ns  samples: 60
      p50: 127.00 ns  p95: 129.00 ns  p99: 136.38 ns
      95% CI: [127.35, 129.12] ns
      throughput: 7847831.87 ops/sec
      cycles: mean 603  median 601  (4.72 GHz)
```

### JSON

Machine-readable format with full metadata:

```json
{
  "meta": {
    "version": "0.1.0",
    "timestamp": "2026-02-10T...",
    "git_commit": "abc123...",
    "system": { "os": "linux", "cpu": "...", "cpu_cores": 24 }
  },
  "results": [
    {
      "id": "bench_fibonacci_iter",
      "group": "compute",
      "status": "passed",
      "metrics": {
        "mean_ns": 127.42,
        "median_ns": 127.0,
        "std_dev_ns": 0.77,
        "p50_ns": 127.0,
        "p95_ns": 129.0,
        "p99_ns": 136.38
      }
    }
  ],
  "verifications": [...],
  "synthetics": [...]
}
```

### CSV

Spreadsheet-compatible format with all metrics:

```
id,name,group,status,mean_ns,median_ns,std_dev_ns,min_ns,max_ns,p50_ns,p95_ns,p99_ns,samples,alloc_bytes,alloc_count,mean_cycles,median_cycles,cycles_per_ns
bench_fibonacci_iter,bench_fibonacci_iter,compute,passed,127.42,...
```

### HTML

Self-contained interactive report with charts and tables.

### GitHub Summary

Renders verification results in GitHub Actions workflow:

```bash
cargo bench -- --format github-summary >> $GITHUB_STEP_SUMMARY
```

## Advanced Usage

### Allocation Tracking

FluxBench can track heap allocations per benchmark iteration. To enable this, install the
`TrackingAllocator` as the global allocator in your benchmark binary:

```rust
use fluxbench::prelude::*;
use fluxbench::TrackingAllocator;

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

#[bench]
fn vec_allocation(b: &mut Bencher) {
    b.iter(|| vec![1, 2, 3, 4, 5]);
}

fn main() { fluxbench::run(); }
```

Results will include allocation metrics for each benchmark:

- **alloc_bytes** — total bytes allocated per iteration
- **alloc_count** — number of allocations per iteration

These appear in JSON, CSV, and human output automatically.

> **Note:** `#[global_allocator]` must be declared in the binary crate (your `benches/*.rs` file),
> not in a library. Rust allows only one global allocator per binary. Without it, `track = true`
> in `flux.toml` will report zero allocations.

You can also query allocation counters manually:

```rust
fluxbench::reset_allocation_counter();
// ... run some code ...
let (alloc_bytes, alloc_count) = fluxbench::current_allocation();
println!("Bytes: {}, Count: {}", alloc_bytes, alloc_count);
```

### In-Process Mode

For debugging, run benchmarks in the same process:

```bash
cargo bench -- --isolated=false
```

Panics will crash immediately, so use this only for development.

### Custom Bootstrap Configuration

Via `flux.toml`:

```toml
[runner]
bootstrap_iterations = 100000
confidence_level = 0.99
```

Higher iterations = more precise intervals, slower reporting.

## Examples

The `examples/` crate contains runnable demos for each feature:

| Example               | What it shows                             |
| --------------------- | ----------------------------------------- |
| `feature_iteration`   | `iter`, `iter_with_setup`, `iter_batched` |
| `feature_async`       | Async benchmarks with tokio runtimes      |
| `feature_params`      | Parameterized benchmarks with `args`      |
| `feature_verify`      | `#[verify]` performance assertions        |
| `feature_compare`     | `#[compare]` baseline tables and series   |
| `feature_allocations` | Heap allocation tracking                  |
| `feature_panic`       | Crash isolation (panicking benchmarks)    |
| `library_bench`       | Benchmarking a library crate              |
| `ci_regression`       | CI regression detection workflow          |

```bash
cargo run -p fluxbench-examples --example feature_iteration --release
cargo run -p fluxbench-examples --example feature_verify --release -- --format json
```

## License

Licensed under the Apache License, Version 2.0.

## Contributing

Contributions welcome. Please ensure benchmarks remain crash-isolated and statistical integrity is maintained.
