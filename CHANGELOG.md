# Changelog

All notable changes to fluxbench are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
fluxbench uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The 0.1.4 entry covers everything up to and including 0.1.4. Tags before it predate this file and are folded into it, so nothing in that entry is a delta against an earlier version. Every later entry is a delta against the one below it.

---

## [0.2.0] — 2026-10-09

Criterion-style measurement, hardened process isolation, and real baseline regression detection. Config errors are now hard errors. Baselines saved with 0.1.x are not comparable.

### Breaking

- **Measurement engine** — a benchmark body runs once. `iter*` drives warmup and sampling, and each sample times a batch of calls with timer overhead subtracted. Setup code in the body no longer re-runs per iteration. A body that calls no `iter*` method, or more than one, is a failed benchmark.
- **`Bencher` API** — `with_config` is replaced by `with_sample_count` and `with_config(MeasureConfig)`. `start_measurement`, `set_iters_per_sample`, `estimated_iter_time_ns`, `has_enough_samples`, and `take_samples` are removed. `run_benchmark_loop` takes a `MeasureConfig` and returns `Result<BenchmarkResult, MeasureError>`.
- **`iter_batched` renamed** — the `&T` form is now `iter_batched_shared`. `iter_batched` takes `(setup, routine(T), BatchSize)`.
- **`#[flux::bench(iterations = N)]`** is a compile error. Use `iters_per_sample`.
- **IPC** — `Sample` holds `f64` per-iteration means plus `iterations: u64`. `PROTOCOL_VERSION` is 5. A supervisor and worker from different versions fail at the handshake.
- **Report schema** — `schema_version` is 2. `ci_lower_ns` and `ci_upper_ns` are `Option<f64>`. Reports from 0.1.x still parse.
- **Statistics** — `mean_ns`, `median_ns`, `std_dev_ns`, skewness, and kurtosis come from the raw samples, including outliers. `trimmed_mean_ns` carries the outlier-removed mean. Compare a 0.2.0 run only against a 0.2.0 baseline.
- **Config** — `flux.toml` rejects unknown keys, and a malformed file is an error. `--format`, `--group`, `--tag`, `--skip-tag`, and filter regexes reject bad values. `runner.isolation = "thread"` is removed.
- **Paths** — a bare `--baseline` or `--save-baseline` name resolves under `output.directory`, else `target/fluxbench`. A bare `--output` name resolves under `output.directory` only when the key is set.
- **CLI types** — `--warmup`, `--measurement`, and `--worker-timeout` accept duration strings such as `500ms`. A bare number means seconds. `Cli.warmup`, `Cli.measurement`, and `Cli.jobs` are `Option` types.
- **Removed** — `ChartType::{Violin, Histogram, Scatter}`, `#[verify(bench = …)]`, and `#[synthetic(deps = …)]`. They had no working meaning.
- **`async` feature** — tokio sits behind the default-on `async` feature of `fluxbench` and `fluxbench-core`. `fluxbench-cli` no longer depends on tokio.

### Added

- **Sampling controls** — `--sample-count`, `--samples` with no 100-sample cap, `sample_count` and `iters_per_sample` in `flux.toml`, `#[flux::bench]`, and `#[flux::group]`.
- **Throughput** — `Throughput::{Bytes, Elements}`, `Bencher::throughput`, and `throughput_bytes` / `throughput_elements` attributes. Reports show B/s and elem/s. `#[verify]`, `#[compare]`, and `#[chart]` accept `throughput_bytes` and `throughput_elements`.
- **Batching** — `BatchSize`, by-value `iter_batched`, `iter_batched_ref`, and `iter_custom`.
- **Group run config** — `#[flux::group]` carries warmup, measurement, sample count, iteration limits, and throughput. Precedence is bench, group, CLI, `flux.toml`, default.
- **Isolation** — `timeout` and `signal` failure kinds, partial-sample streaming on timeout, a 10 s worker handshake deadline, and per-benchmark results in `--one-shot` mode. Panics carry a backtrace captured at panic time.
- **CPU pinning** — each parallel worker pins to its own CPU and skips CPU 0. `--pin`, `--no-pin`, and `runner.pin_cpus` control it.
- **Baselines** — saved baselines carry raw samples. Comparison uses `compare_distributions`, or `compare_summaries` for baselines without samples. Reports show P(regression), the difference CI, effect size, and the method. A significant regression on a critical benchmark exits 1.
- **Config keys** — `runner.timeout`, `output.format`, `output.directory`, `allocator.fail_on_allocation`, `allocator.max_bytes_per_iter`, `ci.fail_on_critical`, `[visuals]`, bench `budget`, and `#[verify(margin)]` now take effect.
- **Reports** — `#[flux::chart]` and `ReportDef` render inline SVG charts in HTML. `GroupDef` renders group sections. `#[compare]` series render as line or bar charts. A run with no benchmarks writes a valid empty report.
- **Verify metrics** — `_skewness`, `_kurtosis`, `_throughput`, and the throughput metrics above.
- **Timer overhead** — `timer_overhead_ns` in `ReportConfig`.
- **Examples** — `feature_throughput`, `test_isolation`, `test_baseline`, and `current_thread` async benchmarks.

### Changed

- The CI is bootstrapped from the same samples as the mean. The normal quantile uses Wichura AS241. The CI label reflects `confidence_level`.
- Allocation figures are per-iteration means across samples. They were summed.
- Banner, status, and annotation lines go to stderr. `--format json` on stdout is a clean document.
- `compare` uses the same executor as the run path, including isolation, `--jobs`, and pinning.
- CSV uses RFC 4180 quoting and one column list for header and rows. The GitHub summary escapes table cells.
- `compute_summary` sorts samples once. The verification context builds once per pass.
- Dependencies are updated to their latest releases. `chrono` drops its default `clock` feature.

### Fixed

- `bootstrap_iterations = 0` no longer panics. It skips the bootstrap and omits the CI.
- `iter_batched` with a batch size of 0 no longer panics.
- `iter_async` no longer panics on a `current_thread` runtime. The runtime is built once per benchmark.
- One crash or timeout in `--one-shot` mode no longer marks every benchmark as crashed.
- A worker with a broken IPC stream is killed and replaced, not reused.
- CSV rows for benchmarks without metrics match the header width.
- HTML output escapes ids, groups, branches, and commits.
- `--jobs` workers no longer share CPU 0.

---

## [0.1.4] — 2026-07-27

Benchmarking framework with process isolation and bootstrap statistics. A replacement for Criterion.

### Added

- **Benchmarks** — `#[flux::bench]` registers sync and async benchmarks with `iter`, `iter_with_setup`, `iter_batched`, and `iter_async`. Parameterized benchmarks, groups, tags, severity levels, and `depends_on` ordering.
- **Process isolation** — each benchmark runs in a supervised worker process over rkyv-framed IPC. A panic or crash fails that benchmark and the rest keep running. Persistent and `--one-shot` workers, `--jobs` parallelism, and a worker timeout.
- **Statistics** — bootstrap resampling with percentile and BCa confidence intervals, IQR outlier detection, percentiles, skewness, kurtosis, and CPU cycle counts on x86_64 and aarch64.
- **Verification** — `#[flux::verify]` rules, `#[flux::synthetic]` derived metrics, and `#[flux::compare]` series, evaluated with `evalexpr`.
- **Allocation tracking** — `TrackingAllocator` reports heap bytes and allocation count per benchmark.
- **Baselines** — `--save-baseline` and `--baseline`, `compare <REF>`, a global regression threshold, and per-benchmark thresholds.
- **Reports** — human, JSON, CSV, HTML, GitHub summary, and `github-action-benchmark` JSON, plus GitHub annotations.
- **Config** — `flux.toml` discovery walks up from the current directory, with CLI overrides.
