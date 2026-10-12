<div align="center">

# fluxbench

<h3>Benchmarks that survive panics, with statistics you can gate CI on</h3>

<p>
Runs each benchmark in a worker process and reports bootstrap confidence intervals and percentiles.
Critical <code>#[verify]</code> rules exit 1 on a failed check.
</p>

<p>
  <a href="https://docs.rs/fluxbench"><strong>Docs</strong></a>
  ·
  <a href="https://crates.io/crates/fluxbench"><strong>Crate</strong></a>
  ·
  <a href="#defining-benchmarks"><strong>Benchmarks</strong></a>
  ·
  <a href="#verify-and-compare"><strong>Verification</strong></a>
  ·
  <a href="#documentation"><strong>Documentation</strong></a>
  ·
  <a href="#quick-start"><strong>Example</strong></a>
  ·
  <a href="CONTRIBUTING.md"><strong>Contributing</strong></a>
</p>

<p>
  <a href="https://discord.gg/jBhFk9kHPg">
    <img src="https://img.shields.io/discord/1453357769720594543?label=Discord&logo=discord&logoColor=white&color=5865F2" alt="Join the Discord">
  </a>
</p>

<p>
  <a href="https://crates.io/crates/fluxbench">
    <img src="https://img.shields.io/crates/v/fluxbench" alt="crates.io">
  </a>
  <a href="https://docs.rs/fluxbench">
    <img src="https://img.shields.io/docsrs/fluxbench" alt="docs.rs">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/crates/l/fluxbench" alt="License">
  </a>
  <a href="https://github.com/ml-rust/fluxbench/stargazers">
    <img src="https://img.shields.io/github/stars/ml-rust/fluxbench?style=social" alt="GitHub stars">
  </a>
</p>

</div>

FluxBench is a benchmarking framework for Rust. A benchmark is a function with `#[bench]`. `cargo bench` runs it in a worker process, so a panic in one benchmark does not stop the suite. Each run reports bootstrap confidence intervals, percentiles, and cycle counts. Attributes add verification rules, derived metrics, comparison tables, and report output.

## Quick Start

FluxBench requires Rust 1.85 or later.

### 1. Add the dependency

```bash
cargo add --dev fluxbench
```

### 2. Configure the bench target

```toml
# Cargo.toml
[[bench]]
name = "my_benchmarks"
harness = false
```

### 3. Write benchmarks

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

Benchmarks can also live in `examples/`. See [Bench function](docs/macros.md#bench-function).

### 4. Run benchmarks

```bash
cargo bench
cargo bench -- --group compute --warmup 5 --measurement 10
```

Run `cargo bench -- --help` for all flags. See the [CLI reference](docs/cli.md).

## Features

| Feature | Description |
| --- | --- |
| [Crash isolation](docs/isolation.md) | A panic, signal, or timeout fails one benchmark. The suite continues. |
| [Bootstrap statistics](docs/statistics.md) | BCa confidence intervals, p50 to p999 percentiles, cycle counts, outlier detection |
| [Verification](docs/verification.md) | `#[verify(expr = "bench_a < bench_b", severity = "critical")]` |
| [Synthetic metrics and comparisons](docs/verification.md) | `#[synthetic(...)]` derived metrics and `#[compare(...)]` tables and series charts |
| [Output formats](docs/output.md) | Human, JSON, HTML, CSV, GitHub summary, and github-action-benchmark JSON |
| [CI integration](docs/statistics.md#exit-status) | Exit code 1 on critical failures. Severity levels in `flux.toml`. |
| [Allocation tracking](docs/statistics.md#allocation-tracking) | Per-iteration heap bytes and count |
| [Throughput](docs/macros.md#throughput) | Bytes per second (MiB/s, GiB/s) or elements per second |
| [Batched inputs and async](docs/macros.md#batched-inputs) | `iter_batched`, `iter_batched_ref`, `BatchSize`, `iter_custom`, and Tokio runtimes |
| [Config](docs/configuration.md) | `flux.toml` with CLI override. Priority: bench > group > CLI > file > default. |

## Defining Benchmarks

Attributes compose. Add one, several, or all of them. Benchmarks do not need restructuring when you add an attribute.

```rust
#[bench(group = "io", tags = "network")]
fn http_request(b: &mut Bencher) {
    // ...
}
```

Run the tagged benchmarks with `cargo bench -- --tag network`. Use `--skip-tag network` to exclude them.

The [macro reference](docs/macros.md) lists every `#[bench]` key. It covers groups, `#[group]` run config, parameterized benchmarks with `args`, time budgets, and async benchmarks.

## Verify and Compare

`#[verify]` states a performance rule next to the code it checks:

```rust
use fluxbench::verify;

#[verify(expr = "fibonacci < 50000", severity = "critical")]
struct FibUnder50us;
```

FluxBench checks the rule on every run:

- `critical` failures exit 1 while `ci.fail_on_critical` is on, the default.
- `warning` failures are reported. The run does not fail.
- `info` results are logged only.

The [verification reference](docs/verification.md) lists every metric suffix, the margin option, `#[synthetic]`, and `#[compare]`. The [statistics reference](docs/statistics.md#exit-status) lists every exit status.

## Documentation

| Topic | File |
| --- | --- |
| Benchmark macros, attributes, groups, async | [docs/macros.md](docs/macros.md) |
| Verification, synthetic metrics, comparisons | [docs/verification.md](docs/verification.md) |
| CLI commands and flags | [docs/cli.md](docs/cli.md) |
| `flux.toml` keys and priority | [docs/configuration.md](docs/configuration.md) |
| Statistics, regression detection, exit status, allocation tracking | [docs/statistics.md](docs/statistics.md) |
| Crash isolation, workers, `failure.kind` values | [docs/isolation.md](docs/isolation.md) |
| Output formats and HTML charts | [docs/output.md](docs/output.md) |
| Runnable examples | [docs/examples.md](docs/examples.md) |

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for conventions and checks.

Questions and design discussion happen on [Discord](https://discord.gg/jBhFk9kHPg).

## License

Licensed under Apache-2.0. See [LICENSE](LICENSE).
