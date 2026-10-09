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

Run `cargo bench` and you get bootstrap confidence intervals, percentile stats (p50–p999), outlier detection, and cycle counts — all automatic. Timing reads `Instant` once per timed batch and subtracts the measured empty-timer overhead. Cycle counts come from RDTSCP (x86_64) or CNTVCT_EL0 (aarch64), else 0.

**Then compose what you need:**

```rust
#[bench(group = "sorting", tags = "alloc")]      // organize and filter
#[verify(expr = "sort_new < sort_old")]          // fail CI if regression
#[synthetic(formula = "sort_old / sort_new")]    // compute speedup ratio
#[compare(baseline = "sort_old")]                // generate comparison tables
```

Each attribute is independent. Use one, use all, add them later — your benchmarks don't need to be restructured.

**Benchmarks that crash don't take down the suite.** Every benchmark runs in a worker process. A panic, segfault, abort, or timeout in one is reported as a failure for _that_ benchmark — the rest keep running. Your CI finishes, you see what broke, you fix it.

**Performance rules live next to the code they protect.** Instead of a fragile shell script that parses output and compares numbers, you write `#[verify(expr = "api_latency_p99 < 5000", severity = "critical")]` and FluxBench enforces it on every run. Critical failures exit non-zero. Warnings get reported. Info is logged.

**Multiple output formats.** The same `cargo bench` run can produce terminal output for you, JSON for your pipeline, HTML for your team, CSV for a spreadsheet, or a GitHub Actions summary — just change `--format`.

## Features

| Feature                                                 | Description                                                          |
| ------------------------------------------------------- | -------------------------------------------------------------------- |
| [Crash isolation](#crash-isolation)                     | Supervisor-worker architecture — panics never terminate the suite    |
| [Bootstrap statistics](#custom-bootstrap-configuration) | BCa bootstrap CIs, cycle counts (RDTSCP, CNTVCT_EL0), outlier detection, p50–p999 |
| [Verification](#verification-macros)                    | `#[verify(expr = "bench_a < bench_b", severity = "critical")]`       |
| [Synthetic metrics](#synthetic-metrics)                 | `#[synthetic(formula = "bench_a / bench_b", unit = "x")]`            |
| [Comparisons](#comparisons)                             | `#[compare(...)]` — tables and series charts vs baseline             |
| [Output formats](#output-formats)                       | Human, JSON, HTML, CSV, GitHub summary, github-action-benchmark JSON |
| [CI integration](#ci-integration)                       | Exit code 1 on critical failures, `flux.toml` severity levels        |
| [Allocation tracking](#allocation-tracking)             | Per-iteration heap bytes and count                                   |
| [Throughput](#throughput)                               | Bytes per second (MiB/s, GiB/s) or elements per second               |
| [Batched inputs](#batched-inputs)                       | `iter_batched`, `iter_batched_ref`, `BatchSize`, `iter_custom`       |
| [Async support](#async-benchmarks)                      | Tokio `current_thread` or `multi_thread`, behind the `async` feature |
| [Configuration](#configuration)                         | `flux.toml` with CLI override, bench > group > CLI > file > default  |

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

`iter_with_setup(setup, routine)` equals `iter_batched(setup, routine, BatchSize::SmallInput)`.

### Batched Inputs

`iter_batched` and `iter_batched_ref` follow criterion:

```rust
#[bench]
fn sort_fresh(b: &mut Bencher) {
    b.iter_batched(
        || (0..1_000u32).rev().collect::<Vec<_>>(),  // Setup: once per call, untimed
        |mut v| { v.sort(); v },                      // Routine: takes the input by value
        BatchSize::SmallInput,
    );
}

#[bench]
fn clear_in_place(b: &mut Bencher) {
    b.iter_batched_ref(
        || vec![1u8; 1 << 20],
        |buf| buf.fill(0),                            // Routine: borrows the input mutably
        BatchSize::LargeInput,
    );
}
```

- Setup runs once per call. It builds each batch of inputs before the timer starts.
- One timer covers the routine over a whole batch.
- Outputs drop after the timer stops. `iter_batched_ref` also drops the inputs then.
- `BatchSize` sets the inputs per batch:

| `BatchSize`         | Inputs per batch                                    |
| ------------------- | --------------------------------------------------- |
| `SmallInput`        | Up to 1024. Fits cheap inputs.                      |
| `LargeInput`        | Up to 16. Fits inputs that hold a lot of memory.    |
| `PerIteration`      | 1. Each call gets its own timer.                    |
| `NumBatches(n)`     | Each sample's calls split into `n` batches.         |
| `NumIterations(n)`  | `n`, capped at the sample's calls.                  |

- `NumBatches(0)` and `NumIterations(0)` count as 1.
- Peak memory is one batch of inputs plus one batch of outputs.

`iter_batched_shared(batch_size, setup, routine)` hands one shared `&T` to `batch_size` calls:

```rust
#[bench]
fn regex_match(b: &mut Bencher) {
    b.iter_batched_shared(
        128,                                          // Calls per setup
        || regex::Regex::new("^[a-z]+$").unwrap(),    // Setup: once per 128 calls, untimed
        |re| re.is_match("hello"),                    // Routine: borrows the shared input
    );
}
```

Before this API, `iter_batched` had this signature. Rename old calls to `iter_batched_shared`.

### Custom Timing

`iter_custom` hands the body a call count. The body runs that many calls, times them, and returns the elapsed time:

```rust
#[bench]
fn self_timed(b: &mut Bencher) {
    b.iter_custom(|iters| {
        let start = std::time::Instant::now();
        for _ in 0..iters {
            std::hint::black_box(2 + 2);
        }
        start.elapsed()
    });
}
```

- The engine runs warmup and sizes the samples from the wall time of each call to the closure.
- The returned time is the sample total. No timer overhead is removed from it.
- Allocation counts cover the whole closure. Cycle counts are 0.

### Throughput

Set the work one call processes. Reports then show a rate:

```rust
#[bench(throughput_bytes = 4096)]                    // 4 KiB per call
fn checksum_4k(b: &mut Bencher) {
    let data = vec![1u8; 4096];
    b.iter(|| data.iter().map(|&x| x as u64).sum::<u64>());
}

#[bench(throughput_elements = 1000)]                 // 1000 items per call
fn parse_1000(b: &mut Bencher) { /* ... */ }

#[bench]
fn copy_dynamic(b: &mut Bencher) {
    let src = vec![0u8; 300_000];
    b.throughput(Throughput::Bytes(src.len() as u64)); // Set from the body
    b.iter(|| src.clone());
}
```

- Precedence: `b.throughput(...)` in the body, then `#[bench]`, then the nearest `#[group]`, then parent groups.
- `throughput_bytes` and `throughput_elements` are mutually exclusive. Either one at 0 is a compile error.
- The rate is the work per call over the mean time per call.
  - Bytes use binary units: B/s, KiB/s, MiB/s, GiB/s, TiB/s.
  - Elements use elem/s, Kelem/s, Melem/s, Gelem/s.
- JSON metrics: `throughput_bytes_per_sec`, `throughput_elements_per_sec`. A benchmark without that kind omits the value.
- `throughput_ops_sec` (1e9 / mean) stays for every benchmark.

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

### Bench Attributes

| Attribute                                          | Meaning                                                              |
| -------------------------------------------------- | -------------------------------------------------------------------- |
| `id`                                               | Benchmark ID. Default: the function name.                            |
| `group`                                            | Group name. Default: `default`.                                      |
| `tags`                                             | Tags for `--tag` and `--skip-tag`.                                   |
| `severity`                                         | `critical`, `warning`, or `info`. Default: `info`.                   |
| `threshold`                                        | Regression threshold in percent. 0 uses the global value.            |
| `budget`                                           | Time budget for the mean per iteration. See [Time Budget](#time-budget). |
| `warmup`, `measurement`                            | Per-benchmark durations, e.g. `"500ms"`.                             |
| `samples`                                          | Fixed mode: N samples of one iteration each, no warmup.              |
| `sample_count`, `iters_per_sample`                 | Target sample count. Fixed iterations per sample.                    |
| `min_iterations`, `max_iterations`                 | Bounds on total measurement iterations.                              |
| `throughput_bytes`, `throughput_elements`          | Work per call. See [Throughput](#throughput).                        |
| `depends_on`                                       | Benchmark IDs that run before this one, e.g. `depends_on = ["a"]`.   |
| `args`                                             | Parameter values. See below.                                         |
| `runtime`, `worker_threads`, `enable_time`, `enable_io` | Async runtime. See [Async Benchmarks](#async-benchmarks).       |

`args = [..]` runs one benchmark per value. The function takes the value as a second argument. Each ID is `id@value`:

```rust
#[bench(group = "scaling", args = [100, 1000])]
fn linear_scan(b: &mut Bencher, n: u32) { /* IDs: linear_scan@100, linear_scan@1000 */ }
```

### Describing Groups

`#[group]` describes a group that `#[bench(group = ...)]` names:

```rust
#[group(id = "parsing", description = "Parser performance", tags = "hot-path")]
struct Parsing;

#[group(id = "json_parsing", parent = "parsing", description = "JSON parsers")]
struct JsonParsing;
```

- `tags`: every benchmark in the group and its child groups carries them for `--tag` and `--skip-tag`.
- `parent`: `--group parsing` also selects `json_parsing`. The parent must be a `#[group]` or a benchmark group.
- `description`, `tags`: the HTML report shows them in the group's section.
- A duplicate `id` or a parent cycle is an error at startup.

### Group Run Config

A `#[group]` can set run config for every benchmark in it and in its child groups:

```rust
#[group(id = "io", sample_count = 30, measurement = "2s", warmup = "500ms")]
struct Io;

#[group(id = "disk", parent = "io", throughput_bytes = 65536, warmup = "1s")]
struct Disk;

#[bench(group = "disk", sample_count = 10)]
fn read_64k(b: &mut Bencher) { /* ... */ }
```

- Keys: `warmup`, `measurement`, `sample_count`, `iters_per_sample`, `min_iterations`, `max_iterations`, and one of `throughput_bytes` or `throughput_elements`.
- `warmup` and `measurement` use the `#[bench]` duration syntax. A bad value is a compile error.
- Precedence per key:
  1. `#[bench]` attribute
  2. Nearest `#[group]`
  3. Parent groups, nearest first
  4. CLI flag
  5. `flux.toml`
  6. Built-in default
- Above, `read_64k` runs 10 samples (bench), 1 s warmup (`disk`), 2 s measurement (`io`), and 64 KiB per call (`disk`).
- `#[bench(samples = N)]` beats a group's `sample_count` and `iters_per_sample`.
- A group `iters_per_sample` selects fixed mode: no warmup, no `min_iterations` or `max_iterations`.
- A resolved config that fails the startup check names the benchmark, e.g. a group `sample_count = 1`.

### Time Budget

`budget` fails a benchmark whose mean time per iteration exceeds it:

```rust
#[bench(budget = "50us")]
fn hot_path(b: &mut Bencher) {
    b.iter(|| std::hint::black_box(2 + 2));
}
```

- Units: `ns`, `us` (or `µs`), `ms`, `s`. A missing unit or a bad number is a compile error.
- `warmup` and `measurement` take the same syntax.
- Over budget, the status is `failed` with `failure.kind = "budget"`. The metrics stay in the report.
- A failed `severity = "critical"` benchmark exits 1 while `ci.fail_on_critical` is on.

### Async Benchmarks

```rust
#[bench(runtime = "multi_thread", worker_threads = 4, group = "async")]
async fn async_operation(b: &mut Bencher) {
    b.iter_async(|| async {
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
    });
}

#[bench(runtime = "current_thread", group = "async")]
async fn async_with_setup(b: &mut Bencher) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    b.iter_async(|| async move { std::hint::black_box(addr.port()) });
}
```

- Runtimes: `"current_thread"` (default) or `"multi_thread"`. Any other value is a compile error.
- `enable_time` and `enable_io` (both default `true`) turn the tokio timer and IO drivers on or off.
- `worker_threads` must be at least 1. It needs `runtime = "multi_thread"`, else it is a compile error.
- The runtime builds once per benchmark, outside the timer.
- `iter_async` times creating and awaiting each future. Each batch runs in one `block_on`.
- Setup before `iter_async` can await, use `tokio::time`, and call `tokio::spawn`.
- `iter_async` inside another runtime's `block_on` or task records an error and runs nothing.

#### The `async` feature

Async support needs tokio. The `async` cargo feature of `fluxbench` gates it. It is on by default.

```toml
[dev-dependencies]
# Sync benchmarks only: no tokio in the dependency tree
fluxbench = { version = "<latest-version>", default-features = false }
```

- Off: `iter_async`, `drive_async_body`, and `attach_runtime` do not exist.
- Off: an `async fn` under `#[bench]` is a compile error: "enable the `async` feature of fluxbench".
- `fluxbench-core` has the same `async` feature, on by default. `fluxbench` forwards its own.
- tokio features in use: `rt` and `sync` in `fluxbench-core`. `fluxbench` adds `rt-multi-thread`, `time`, and `net` for the generated runtimes.
- `fluxbench-cli` has no tokio dependency.

CI checks for the gate, run from the workspace root:

```bash
# Builds with no tokio in the tree
cargo build -p fluxbench-core --no-default-features
cargo tree -p fluxbench-core --no-default-features -e normal | grep -c tokio   # prints 0
cargo build -p fluxbench --no-default-features
# Builds with the async API
cargo build -p fluxbench-core
cargo test -p fluxbench --test async_runtime
```

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

- `critical` (default): Fails the benchmark suite (exit code 1) while `ci.fail_on_critical` is on, the default
- `warning`: Reported but doesn't fail
- `info`: Informational only
- Any other value is a compile error.

**Margin:** `margin = N` widens the rule's single top-level comparison by N percent of its right side.

```rust
// Passes while fib_iter <= fib_naive * 1.05
#[verify(expr = "fib_iter < fib_naive", margin = 5)]
struct IterNotSlower;
```

- The tolerance is `|right| * N / 100`.
- `a < b` and `a <= b` pass when `a <= b + tolerance`.
- `a > b` and `a >= b` pass when `a >= b - tolerance`.
- `a == b` passes when `|a - b| <= tolerance`.
- With a margin, the expression needs exactly one comparison outside parentheses, and no top-level `&&`, `||`, or `!=`. Otherwise the rule reports an evaluation error.
- `margin` must be a number >= 0. The default 0 evaluates the expression as written.

**Available Metrics** (for a benchmark named `bench_name`):

| Suffix                              | Metric                                  |
| ----------------------------------- | --------------------------------------- |
| _(none)_ / `_mean`                  | Mean time (ns)                          |
| `_median`                           | Median time                             |
| `_min` / `_max`                     | Min / max time                          |
| `_p50` `_p90` `_p95` `_p99` `_p999` | Percentiles                             |
| `_std_dev`                          | Standard deviation                      |
| `_skewness` / `_kurtosis`           | Distribution shape                      |
| `_throughput`                       | Operations per second (1e9 / mean)      |
| `_throughput_bytes`                 | Bytes per second. Needs byte throughput. |
| `_throughput_elements`              | Elements per second. Needs element throughput. |
| `_samples`                          | Sample count                            |
| `_alloc_bytes` / `_alloc_count`     | Mean bytes / allocations per iteration  |

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
`metric` is one of `mean` (default), `median`, `min`, `max`, `p50`, `p90`, `p95`, `p99`, `p999`, `throughput_bytes`, `throughput_elements`. Any other value is a compile error.
`throughput_bytes` and `throughput_elements` are rates, so a higher value is faster and speedup is value over baseline. A benchmark without that throughput kind has no entry. A baseline without it drops the comparison.

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

#[bench(group = "scaling")]
fn vec_fold_100(b: &mut Bencher) {
    let data: Vec<i64> = (0..100).collect();
    b.iter(|| data.iter().fold(0, |a, x| a + x));
}

#[bench(group = "scaling")]
fn vec_fold_1000(b: &mut Bencher) {
    let data: Vec<i64> = (0..1000).collect();
    b.iter(|| data.iter().fold(0, |a, x| a + x));
}

#[compare(
    id = "scale_100",
    title = "Vector Sum Scaling",
    benchmarks = ["vec_sum_100", "vec_fold_100"],
    group = "vec_scaling",
    x = "100",
    series = ["sum()", "fold()"]
)]
struct Scale100;

#[compare(
    id = "scale_1000",
    title = "Vector Sum Scaling",
    benchmarks = ["vec_sum_1000", "vec_fold_1000"],
    group = "vec_scaling",
    x = "1000",
    series = ["sum()", "fold()"]
)]
struct Scale1000;
```

- Multiple `#[compare]` with the same `group` combine into one chart.
- Each `#[compare]` needs at least 2 benchmarks with data. Otherwise it produces no output.
- `x` places the point on the x axis. The chart sorts points by `x` as a number.
- `series` names the lines by position. Without it, the lines use the benchmark IDs of the first `#[compare]`.
- The HTML report draws each group as an inline SVG under `Comparison Series`. Numeric `x` values draw one line per series. Other `x` values draw grouped bars. A value of 0 shows no point.

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
cargo bench -- --warmup 10 --measurement 20    # Bare numbers are seconds
cargo bench -- --warmup 500ms --measurement 2s # Duration strings
cargo bench -- --sample-count 250              # Target sample count (no cap)
cargo bench -- --samples 500                   # Fixed mode: 500 one-iteration samples
cargo bench -- --min-iterations 100            # Minimum total measurement iterations
cargo bench -- --isolated=false                # In-process mode
cargo bench -- --worker-timeout 2m             # Worker timeout; a bare number means seconds
cargo bench -- --threads 1                     # Statistics threads (0 = all cores, the default)

# Output formats: human, json, github-summary (alias github), github-action, csv, html
cargo bench -- --format json --output results.json   # ./results.json, or <output.directory>/results.json when set
cargo bench -- --format html --output ./results.html # Always ./results.html
cargo bench -- --format csv --output results.csv
cargo bench -- --format github-summary         # GitHub Actions summary

# Baseline comparison
cargo bench -- --save-baseline main.json       # Save results plus raw samples
cargo bench -- --baseline main.json            # Compare against a saved baseline
cargo bench -- --baseline main.json --threshold 10
cargo bench -- compare origin/main             # Baseline must be saved at that commit

# Dry run
cargo bench -- --dry-run
```

Run `cargo bench -- --help` for the full option reference.

- An invalid filter regex is an error that names the regex.
- An unknown `--group`, `--tag`, or `--skip-tag` is an error that lists the known values.
- An unknown `--format` value is a usage error (exit 2).
- A valid filter that matches nothing prints `No benchmarks found.` on stderr and exits 0. Formats other than `human` also write a valid empty report: JSON with no results, CSV with only the header, and so on.
- A bare `--output` file name such as `results.json` goes under `output.directory` when that key is set. Without it, the name stays in the current directory.
- A bare `--baseline` or `--save-baseline` file name goes under `output.directory`, else under `target/fluxbench`.
- Any path with a directory part, such as `./results.json`, stays as written.

## Configuration

FluxBench works out of the box with sensible defaults — no configuration file is needed. For workspace-wide customization, you can optionally create a `flux.toml` in your project root. FluxBench auto-discovers it by walking up from the current directory.

Settings are applied in this priority order: **`#[bench]` attribute > `#[group]` attribute (nearest first) > CLI flag > flux.toml > built-in default**. See [Group Run Config](#group-run-config).

Every key below is real, and every real key is listed. These are errors that stop the run before any benchmark starts:

- An unknown table or key. The message names the file, the key, and the valid keys.
- Malformed TOML. The message names the file, line, and column.
- An unreadable file.
- A bad value, such as `timeout = "soon"` or `confidence_level = 1.5`. The message names the key and the value.

A missing file uses the defaults. Relative paths resolve against the current directory.

### Runner

```toml
[runner]
warmup_time = "500ms"        # Warmup before measurement (default: "3s")
measurement_time = "1s"      # Measurement duration, above 0 (default: "5s")
timeout = "30s"              # Worker timeout per benchmark, above 0 (default: "60s")
isolation = "process"        # "process" or "in-process" (default: "process")
bootstrap_iterations = 1000  # Bootstrap resamples for CIs; 0 skips the bootstrap (default: 10000)
confidence_level = 0.95      # Confidence level, between 0 and 1 exclusive (default: 0.95)
# sample_count = 100         # Target sample count, minimum 2, no upper cap (default: 100)
# samples = 5                # Fixed mode: N samples of one iteration each, no warmup, minimum 2
# min_iterations = 100       # Minimum total measurement iterations (default: unset)
# max_iterations = 1000000   # Maximum total measurement iterations, >= min_iterations (default: unset)
# jobs = 4                   # Parallel isolated workers, minimum 1 (default: 1)
# pin_cpus = false           # Pin each isolated worker to its own CPU (default: true)
```

- Durations take `ns`, `us`, `ms`, `s`, or `m`. A bare number means seconds.
- `timeout`: `--worker-timeout` overrides it. It applies to isolated mode only.
  - A timeout below a benchmark's warmup + measurement + 2 s rises to that value. One stderr warning names both values.
- `isolation`: `--isolated=true` or `--isolated=false` overrides it.
- Before any benchmark runs, each benchmark's resolved settings are checked. A bad combination is an error that names the benchmark, e.g. `--warmup 0` outside fixed mode.

### Allocator

```toml
[allocator]
track = true                 # Track allocations during measurement (default: true)
fail_on_allocation = false   # Fail a benchmark whose mean allocations per iteration are above 0 (default: false)
# max_bytes_per_iter = 1024  # Fail a benchmark whose mean bytes per iteration are above this (default: unlimited)
```

- A benchmark that breaks a limit gets status `failed` and `failure.kind = "allocation_limit"`. Its metrics stay in the report.
- The limits use the unrounded mean per iteration across samples.
- The limits need `track = true` and `TrackingAllocator` as `#[global_allocator]`. Otherwise one stderr warning says the checks do not run.
  - All-zero totals cannot tell "no allocations" from "no `TrackingAllocator`", so the checks do not run then either.

### Output

```toml
[output]
format = "human"                    # Report format when --format is absent (default: "human")
# directory = "target/fluxbench"    # Directory for bare file names (default: unset)
save_baseline = false               # Save a JSON baseline after each run (default: false)
# baseline_path = "baseline.json"   # Baseline file for --baseline and saves (default: "baseline.json")
```

- `format`: `human`, `json`, `github-summary` (alias `github`), `github-action`, `csv`, or `html`. `--format` overrides it.
- `directory`: a bare file name such as `r.json` given to `--output`, `--baseline`, `--save-baseline`, or `baseline_path` goes under it.
  - Unset: a bare `--output` name stays in the current directory. Bare baseline names go under `target/fluxbench`.
  - A path with a directory part, such as `./r.json` or `ci/r.json`, stays as written.
  - Missing directories are created when a report or baseline is written.
- `baseline_path`: the file that `--baseline` and `--save-baseline` use without a value. With no key, it is `baseline.json` under `directory`, else `target/fluxbench/baseline.json`.
- `--output` unset writes the report to stdout.

### Visuals

```toml
[visuals]
theme = "light"              # HTML report theme: "light" or "dark" (default: "light")
width = 1280                 # #[chart] canvas width in pixels, at least 200 (default: 1280)
height = 720                 # #[chart] canvas height in pixels, at least 150 (default: 720)
```

They apply to `--format html` only. A narrow page shrinks a chart and keeps its aspect ratio.

### CI Integration

```toml
[ci]
regression_threshold = 5.0   # A significant slowdown above this percentage is a regression, >= 0 (default: 5.0)
github_annotations = true    # Print ::warning and ::error annotations on stderr (default: false)
fail_on_critical = true      # Exit 1 on critical verification failures and critical benchmarks' regressions or failures (default: true)
```

#### Regression detection

`--baseline` and `compare <REF>` compare each benchmark with its baseline entry.

- A change counts only when both hold:
  - It is statistically significant: the difference CI excludes zero, or the probability of its direction reaches `runner.confidence_level`.
  - Its size exceeds the threshold: `#[bench(threshold = ...)]` if above 0, else `--threshold`, else `regression_threshold`.
- Improvements follow the same rule in the other direction.
- Statistics:
  - Saved baselines (schema 2) carry raw samples. Both runs' samples are bootstrapped for the probability, the difference CI, and Cohen's d.
  - Schema 1 baselines, or `bootstrap_iterations = 0`, fall back to a Welch-style z test on mean, std dev, and sample count.
- Exit status, for the default run and `compare` alike:

| Finding | `fail_on_critical = true` | `fail_on_critical = false` |
| --- | --- | --- |
| Crashed benchmark | exit 1 | exit 1 |
| Critical verification failure | exit 1 | warning |
| Regression on a `severity = "critical"` benchmark | exit 1 | warning |
| Regression on a `warning` or `info` benchmark | warning | warning |
| Failed `severity = "critical"` benchmark, e.g. over `budget` | exit 1 | warning |
| Failed `warning` or `info` benchmark | warning | warning |

A missing `--baseline` file prints a warning and runs without a comparison. An unreadable or invalid baseline file, or one with a newer schema, is an error. `compare <REF>` requires the baseline file and checks that its commit matches `REF`. `compare` uses the same executor as a normal run: isolation, `--jobs`, and CPU pinning apply.

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

A benchmark with throughput adds `bytes: 1.50 GiB/s` or `elements: 3.20 Melem/s` under `throughput`.

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
id,name,group,status,mean_ns,median_ns,std_dev_ns,min_ns,max_ns,p50_ns,p95_ns,p99_ns,samples,alloc_bytes,alloc_count,mean_cycles,median_cycles,cycles_per_ns,p90_ns,p999_ns,ci_lower_ns,ci_upper_ns,trimmed_mean_ns,throughput_ops_sec,failure_kind,failure_message,throughput_bytes_per_sec,throughput_elements_per_sec
bench_fibonacci_iter,bench_fibonacci_iter,compute,passed,127.42,...
```

New columns go at the end, so existing consumers keep working. A cell without a value is empty.

### HTML

One self-contained page with inline CSS and SVG. It has no scripts.

- Header: title and description from `#[report]`, else "FluxBench Report". Then commit, branch, and time.
- Summary: passed, failed, crashed, and skipped counts.
- Charts: one SVG per `#[chart]`, in the `#[report]` layout grid.
- Results: one table per group, nested under parent groups. Each shows the `#[group]` description and tags.
  - A Throughput column shows the byte rate, else the element rate, else `-`.
- `[visuals]` sets the theme and the chart canvas size.

```rust
#[report(title = "Parser Dashboard", description = "Nightly run", layout = "grid(1, 2)")]
struct Dashboard;

#[chart(
    title = "Mean time",
    kind = "bar",                       // "bar" (default) or "line"
    metric = "mean",                    // default "mean"
    benchmarks = ["parse_small", "parse_large"],
    target = 5000,                      // optional line, in the metric's unit (ns for times)
    row = 1, col = 1                    // optional 1-based cell; both or neither
)]
struct MeanChart;
```

- `layout`: `"grid(rows, cols)"` or `"RxC"`, each at least 1. Default `grid(2, 2)`. A bad value is a compile error.
- `metric`: `mean`, `median`, `min`, `max`, `p50`, `p90`, `p95`, `p99`, `p999`, `std_dev`, `throughput`, `throughput_bytes`, `throughput_elements`, `alloc_bytes`, `alloc_count`, `mean_cycles`.
- `throughput_bytes` and `throughput_elements` need a benchmark with that throughput kind. Without it the slot shows `(no data)`.
- A benchmark without data, e.g. filtered out or crashed, shows `(no data)` under its slot.
- Startup errors: more than one `#[report]`, an unknown chart metric or benchmark ID, a cell outside the layout, or two charts in one cell.

### GitHub Summary

Renders Markdown for a GitHub Actions workflow. Sections: Summary, Baseline Comparison (with a baseline), Critical Benchmarks, Throughput, and Verifications.

```bash
cargo bench -- --format github-summary >> $GITHUB_STEP_SUMMARY
```

A Throughput table lists each benchmark with a byte or element rate. It is absent when no benchmark has one.

### GitHub Action

`--format github-action` writes a JSON array for [github-action-benchmark](https://github.com/benchmark-action/github-action-benchmark) with `tool: 'customSmallerIsBetter'`:

```bash
cargo bench -- --format github-action --output output.json
```

- One entry per passed benchmark, with `name` (the benchmark ID), `unit`, `value` (mean time), `range` (`± std dev`), and `extra`.
- `extra` holds median, min, max, p95, p99, samples, and ops per second.
- Failed, crashed, and skipped benchmarks have no entry.

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

fn main() { fluxbench::run().unwrap(); }
```

Results will include allocation metrics for each benchmark:

- **alloc_bytes** — total bytes allocated per iteration
- **alloc_count** — number of allocations per iteration

These appear in JSON, CSV, and human output automatically.
`[allocator]` limits in `flux.toml` can fail a benchmark on them. See [Allocator](#allocator).

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

### Crash Isolation

Benchmarks run in worker processes by default. Every benchmark gets its own report entry, whatever happens to its worker.

- Persistent mode (default): one worker runs many benchmarks. A worker that crashes, times out, or breaks the IPC stream is killed and replaced. The next benchmark gets a fresh worker.
- One-shot mode (`--one-shot`): each benchmark gets a fresh worker.
- `--jobs N`: N workers run benchmarks in parallel. Both modes support it. Parallel jobs perturb measurements.
- `--pin` / `--no-pin`: pin each worker to its own CPU (default: on). They conflict. They override `runner.pin_cpus` in `flux.toml`.
  - Linux only. The supervisor reads the process affinity mask and skips CPU 0 when other CPUs exist.
  - `--jobs N` above the usable CPU count makes workers share cores. A warning prints on stderr.
  - A failed pin prints one stderr warning in the worker. The benchmark still runs.
  - `--no-pin` leaves workers to the OS scheduler.

A failed benchmark reports status `crashed` or `failed` and a `failure.kind`:

| `failure.kind` | Cause                                                                     |
| -------------- | ------------------------------------------------------------------------- |
| `panic`        | The body panicked. `message` holds the panic text and location. `backtrace` comes from the panic site. |
| `signal`       | A signal killed the worker. `failure.signal` holds the name, e.g. `SIGSEGV` or `SIGABRT`. Unix only. |
| `timeout`      | The worker exceeded the worker timeout (`--worker-timeout`, else `runner.timeout`). `message` states the partial sample count. |
| `crashed`      | The worker exited with a code, broke the IPC stream, or did not start.    |
| `unknown`      | The body never called `b.iter*`, or another measurement error occurred.   |
| `budget`       | The mean time per iteration exceeded `#[bench(budget)]`.                  |
| `allocation_limit` | The benchmark broke `allocator.fail_on_allocation` or `allocator.max_bytes_per_iter`. |

`panic`, `signal`, `timeout`, and `crashed` set status `crashed`, so the run exits with code 1.
`unknown`, `budget`, and `allocation_limit` set status `failed`. A failed critical benchmark exits 1 while `ci.fail_on_critical` is on.

Workers stream samples to the supervisor during measurement. On a timeout, the supervisor sends SIGTERM and collects the samples sent so far for 500 ms. Then it sends SIGKILL. The partial samples never count as a pass and never form metrics. On other platforms, a timeout kills the worker at once.

A new worker must send its handshake within 10 s. Otherwise the supervisor kills it and reports `crashed` for that benchmark.

User code can print to stdout: on Unix the worker's stdout goes to `/dev/null`, and IPC uses dedicated pipes. On other platforms, IPC uses the worker's stdin and stdout, so `println!` in a benchmark body corrupts the protocol stream.

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
| `feature_iteration`   | `iter`, `iter_with_setup`, `iter_batched` with `BatchSize`, `iter_batched_ref`, `iter_batched_shared`, `iter_custom` |
| `feature_throughput`  | Bytes and elements per second: attribute, `b.throughput()`, group, `#[verify]` |
| `feature_async`       | Async benchmarks with tokio runtimes      |
| `feature_params`      | Parameterized benchmarks with `args`      |
| `feature_verify`      | `#[verify]` performance assertions        |
| `feature_compare`     | `#[compare]` baseline tables and series   |
| `feature_allocations` | Heap allocation tracking                  |
| `feature_panic`       | Crash isolation (panicking benchmarks)    |
| `library_bench`       | Benchmarking a library crate              |
| `ci_regression`       | CI regression detection workflow          |
| `test_isolation`      | Fixtures for the isolation tests: panic, abort, SIGSEGV, timeout |
| `test_baseline`       | Fixtures for the baseline tests: `FLUX_TEST_SLOWDOWN` slows a critical benchmark |

```bash
cargo run -p fluxbench-examples --example feature_iteration --release
cargo run -p fluxbench-examples --example feature_verify --release -- --format json
```

## License

Licensed under the Apache License, Version 2.0.

## Contributing

Contributions welcome. Please ensure benchmarks remain crash-isolated and statistical integrity is maintained.
