# Config (flux.toml)

FluxBench runs with built-in defaults. No config file is required. Create a `flux.toml` in your project root to set workspace-wide values.

FluxBench finds the file by walking up from the current directory. A missing file uses the defaults. Relative paths resolve against the current directory.

## Priority

Config values resolve in this order. The first match wins:

1. `#[bench]` attribute
2. `#[group]` attribute, nearest group first
3. CLI flag
4. `flux.toml`
5. Built-in default

See [Group run config](macros.md#group-run-config) for the group rules.

## Full example

Every key below is shown with its default. Keys with a `#` prefix are unset by default.

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

[allocator]
track = true                 # Track allocations during measurement (default: true)
fail_on_allocation = false   # Fail a benchmark whose mean allocations per iteration are above 0 (default: false)
# max_bytes_per_iter = 1024  # Fail a benchmark whose mean bytes per iteration are above this (default: unlimited)

[output]
format = "human"                    # Report format when --format is absent (default: "human")
# directory = "target/fluxbench"    # Directory for bare file names (default: unset)
save_baseline = false               # Save a JSON baseline after each run (default: false)
# baseline_path = "baseline.json"   # Baseline file for --baseline and saves (default: "baseline.json")

[visuals]
theme = "light"              # HTML report theme: "light" or "dark" (default: "light")
width = 1280                 # #[chart] canvas width in pixels, at least 200 (default: 1280)
height = 720                 # #[chart] canvas height in pixels, at least 150 (default: 720)

[ci]
regression_threshold = 5.0   # A significant slowdown above this percentage is a regression, >= 0 (default: 5.0)
github_annotations = true    # Print ::warning and ::error annotations on stderr (default: false)
fail_on_critical = true      # Exit 1 on critical verification failures and critical benchmarks' regressions or failures (default: true)
```

## Runner

- Durations take `ns`, `us`, `ms`, `s`, or `m`. A bare number means seconds.
- `timeout`: `--worker-timeout` overrides it. It applies to isolated mode only.
  - A timeout below a benchmark's warmup + measurement + 2 s rises to that value. One stderr warning names both values.
- `isolation`: `--isolated=true` or `--isolated=false` overrides it.
- Before any benchmark runs, each benchmark's resolved config is checked. A bad combination is an error that names the benchmark. Example: `--warmup 0` outside fixed mode.

## Allocator

- A benchmark that breaks a limit gets status `failed` and `failure.kind = "allocation_limit"`. Its metrics stay in the report.
- The limits use the unrounded mean per iteration across samples.
- The limits need `track = true` and `TrackingAllocator` as `#[global_allocator]`. Otherwise one stderr warning says the checks do not run.
  - All-zero totals cannot tell "no allocations" from "no `TrackingAllocator`". The checks do not run then either.

See [Allocation tracking](statistics.md#allocation-tracking).

## Output

- `format`: `human`, `json`, `github-summary` (alias `github`), `github-action`, `csv`, or `html`. `--format` overrides it.
- `directory`: a bare file name such as `r.json` given to `--output`, `--baseline`, `--save-baseline`, or `baseline_path` goes under it.
  - Unset: a bare `--output` name stays in the current directory. Bare baseline names go under `target/fluxbench`.
  - A path with a directory part, such as `./r.json` or `ci/r.json`, stays as written.
  - Missing directories are created when a report or baseline is written.
- `baseline_path`: the file that `--baseline` and `--save-baseline` use without a value. With no key, it is `baseline.json` under `directory`, else `target/fluxbench/baseline.json`.
- `--output` unset writes the report to stdout.

## Visuals

The `[visuals]` table applies to `--format html` only. A narrow page shrinks a chart and keeps its aspect ratio.

## CI

- `regression_threshold`: the percentage that a significant change must exceed to count as a regression. `--threshold` and `#[bench(threshold)]` override it.
- `github_annotations`: print `::warning` and `::error` lines on stderr.
- `fail_on_critical`: controls the exit codes in [Exit status](statistics.md#exit-status).

## Errors

These errors stop the run before any benchmark starts:

- An unknown table or key. The message names the file, the key, and the valid keys.
- Malformed TOML. The message names the file, line, and column.
- An unreadable file.
- A bad value, such as `timeout = "soon"` or `confidence_level = 1.5`. The message names the key and the value.
