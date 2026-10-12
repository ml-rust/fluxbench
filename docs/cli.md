# CLI Reference

The benchmark binary takes FluxBench flags after `--`:

```bash
cargo bench -- [OPTIONS] [FILTER]
cargo bench -- [OPTIONS] <COMMAND>
```

Run `cargo bench -- --help` for the full flag list.

## Commands

| Command | Meaning |
| --- | --- |
| `run` | Run benchmarks. This is the default when you omit a command. |
| `list` | List all discovered benchmarks without running them. |
| `compare <REF>` | Compare against the baseline saved at a git ref, such as `origin/main`. |

## Flags

### Selection

| Flag | Meaning |
| --- | --- |
| `[FILTER]` | Regex on the benchmark ID. Default `.*`. An invalid regex is an error that names the regex. |
| `--group <NAME>` | Run the group and its child groups. An unknown group is an error that lists the known groups. |
| `--tag <TAG>` | Run benchmarks with this tag. An unknown tag is an error that lists the known tags. |
| `--skip-tag <TAG>` | Skip benchmarks with this tag. An unknown tag is an error that lists the known tags. |
| `--dry-run` | List benchmarks without running them. |

A valid filter that matches nothing prints `No benchmarks found.` on stderr and exits 0. Formats other than `human` also write a valid empty report. JSON has no results, CSV has only the header, and so on.

### Timing

| Flag | Meaning |
| --- | --- |
| `--warmup <DURATION>` | Warmup time. Default `3s`. A bare number means seconds. Overrides `flux.toml`. |
| `--measurement <DURATION>` | Measurement time. Default `5s`. A bare number means seconds. Overrides `flux.toml`. |
| `-n, --samples <N>` | Fixed mode: N samples of one iteration each, no warmup. Ignores warmup and min/max iterations. `--sample-count` overrides N. Minimum 2. |
| `--sample-count <N>` | Target sample count. Default 100. Minimum 2, no upper cap. |
| `--min-iterations <N>` | Minimum total measurement iterations. |
| `--max-iterations <N>` | Maximum total measurement iterations. |

Duration flags take `ns`, `us`, `ms`, `s`, or `m` units. Examples: `500ms`, `2s`, `120`.

### Isolation and workers

| Flag | Meaning |
| --- | --- |
| `--isolated=true\|false` | Run benchmarks in worker processes. Default `true`, or `runner.isolation` in `flux.toml`. Use `--isolated=false` to run in-process. |
| `--one-shot` | Start a fresh worker for each benchmark. Default: one worker runs many benchmarks. |
| `--worker-timeout <DURATION>` | Worker timeout per benchmark. Default `60s`, or `runner.timeout`. A bare number means seconds. |
| `--jobs <N>` | Number of parallel isolated workers. Default 1. `0` is an error. |
| `--pin` | Pin each isolated worker to its own CPU. This is the default. |
| `--no-pin` | Do not pin isolated workers to CPUs. |
| `-j, --threads <N>` | Threads for statistics. `0` uses all cores (default). `1` runs single-threaded. |

`--pin` and `--no-pin` conflict. They override `runner.pin_cpus` in `flux.toml`. See [Crash isolation](isolation.md).

### Reports and baselines

| Flag | Meaning |
| --- | --- |
| `--format <FORMAT>` | Report format: `human`, `json`, `github-summary`, `github-action`, `csv`, or `html`. Default `human`, or `output.format` in `flux.toml`. |
| `-o, --output <PATH>` | Report file. Stdout when omitted. |
| `--baseline [PATH]` | Load a baseline for comparison. Without a path, uses `output.baseline_path`, else `baseline.json`. |
| `--save-baseline [PATH]` | Save results as a baseline JSON file, raw samples included. Path rules match `--baseline`. |
| `--threshold <PERCENT>` | Regression threshold in percent. Overrides `ci.regression_threshold`. A per-benchmark `threshold` replaces it. |
| `-v, --verbose` | Verbose output. |

`--format` accepts the aliases `text` (for `human`) and `github` (for `github-summary`). See [Output formats](output.md).

## Examples

```bash
# List benchmarks without running
cargo bench -- list

# Run benchmarks whose ID matches a regex
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

# Reports
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

## Path rules

- A bare `--output` file name such as `results.json` goes under `output.directory` when that key is set. Without it, the name stays in the current directory.
- A bare `--baseline` or `--save-baseline` file name goes under `output.directory`. Without it, the name goes under `target/fluxbench`.
- A path with a directory part, such as `./results.json`, stays as written.

## Errors

- An unknown `--format` value is a usage error. The exit code is 2.
- `--jobs 0` is an error. Use 1 or more.
- A missing `--baseline` file prints a warning and runs without a comparison.
- An unreadable or invalid baseline file, or one with a newer schema, is an error.
- `compare <REF>` requires the baseline file. It checks that the baseline commit matches `REF`.
- `compare` uses the same executor as a normal run. Isolation, `--jobs`, and CPU pinning apply.
