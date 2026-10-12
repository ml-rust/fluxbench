# Output Formats and HTML Charts

Select a format with `--format` or `output.format` in [flux.toml](configuration.md). The formats are `human`, `json`, `github-summary` (alias `github`), `github-action`, `csv`, and `html`. The alias `text` selects `human`.

| Format | Output | Use |
| --- | --- | --- |
| `human` | Terminal text, grouped by group | Local runs |
| `json` | JSON with full metadata | Pipelines and baselines |
| `csv` | One row per benchmark | Spreadsheets |
| `html` | One self-contained page | Sharing a report |
| `github-summary` | Markdown | `$GITHUB_STEP_SUMMARY` |
| `github-action` | JSON for github-action-benchmark | Tracking results over time |

## Human

Console output with grouped results and statistics. Example output. Values vary by machine:

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

## JSON

Machine-readable format with full metadata:

```json
{
  "meta": {
    "version": "<fluxbench version>",
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

- `meta.version` is the FluxBench version that wrote the file.
- Baseline files use the same schema. Schema 2 adds the raw samples. See [Regression detection](statistics.md#regression-detection).

## CSV

Spreadsheet-compatible format with all metrics. The header row is:

```
id,name,group,status,mean_ns,median_ns,std_dev_ns,min_ns,max_ns,p50_ns,p95_ns,p99_ns,samples,alloc_bytes,alloc_count,mean_cycles,median_cycles,cycles_per_ns,p90_ns,p999_ns,ci_lower_ns,ci_upper_ns,trimmed_mean_ns,throughput_ops_sec,failure_kind,failure_message,throughput_bytes_per_sec,throughput_elements_per_sec
```

- New columns go at the end, so existing consumers keep working.
- A cell without a value is empty.

## HTML

One self-contained page with inline CSS and SVG. The page has no `<script>` tags.

- Header: title and description from `#[report]`, else "FluxBench Report". Then commit, branch, and time.
- Summary: passed, failed, crashed, and skipped counts.
- Charts: one SVG per `#[chart]`, in the `#[report]` layout grid.
- Results: one table per group, nested under parent groups. Each shows the `#[group]` description and tags.
  - A Throughput column shows the byte rate, else the element rate, else `-`.
- `[visuals]` in `flux.toml` sets the theme and the chart canvas size.
- Comparison series draw under `Comparison Series`. See [Comparisons](verification.md#comparisons).

### Report and chart macros

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
- Startup errors stop the run in these cases:
  - More than one `#[report]`.
  - An unknown chart metric or benchmark ID.
  - A cell outside the layout.
  - Two charts in one cell.

### Visuals

The `[visuals]` table sets the theme and chart size for HTML output only:

| Key | Default | Meaning |
| --- | --- | --- |
| `theme` | `"light"` | `"light"` or `"dark"`. |
| `width` | `1280` | `#[chart]` canvas width in pixels. At least 200. |
| `height` | `720` | `#[chart]` canvas height in pixels. At least 150. |

## GitHub Summary

Renders Markdown for a GitHub Actions workflow. Sections: Summary, Baseline Comparison (with a baseline), Critical Benchmarks, Throughput, and Verifications.

```bash
cargo bench -- --format github-summary >> $GITHUB_STEP_SUMMARY
```

The Throughput table lists each benchmark with a byte or element rate. The table is absent when no benchmark has one.

## GitHub Action

`--format github-action` writes a JSON array for [github-action-benchmark](https://github.com/benchmark-action/github-action-benchmark) with `tool: 'customSmallerIsBetter'`:

```bash
cargo bench -- --format github-action --output output.json
```

- One entry per passed benchmark, with `name` (the benchmark ID), `unit`, `value` (mean time), `range` (`± std dev`), and `extra`.
- `extra` holds median, min, max, p95, p99, samples, and ops per second.
- Failed, crashed, and skipped benchmarks have no entry.
