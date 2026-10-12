# Verification, Synthetic Metrics, and Comparisons

Reference for `#[verify]`, `#[synthetic]`, and `#[compare]`.

- [Verification macros](#verification-macros)
- [Available metrics](#available-metrics)
- [Synthetic metrics](#synthetic-metrics)
- [Comparisons](#comparisons)

## Verification macros

`#[verify]` asserts a performance rule. The rule runs on every `cargo bench`:

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

### Severity levels

| `severity` | Result |
| --- | --- |
| `critical` (default) | Fails the run with exit code 1 while `ci.fail_on_critical` is on. |
| `warning` | Reported. Does not fail the run. |
| `info` | Informational only. |

Any other value is a compile error.

### Margin

`margin = N` widens the rule's single top-level comparison by N percent of its right side:

```rust
// Passes while fib_iter <= fib_naive * 1.05
#[verify(expr = "fib_iter < fib_naive", margin = 5)]
struct IterNotSlower;
```

- The tolerance is `|right| * N / 100`.
- `a < b` and `a <= b` pass when `a <= b + tolerance`.
- `a > b` and `a >= b` pass when `a >= b - tolerance`.
- `a == b` passes when `|a - b| <= tolerance`.
- With a margin, the expression needs exactly one comparison outside parentheses. It cannot contain a top-level `&&`, `||`, or `!=`. Otherwise the rule reports an evaluation error.
- `margin` must be a number >= 0. The default 0 evaluates the expression as written.

## Available metrics

Metric names in `expr` take the form `<bench_name><suffix>`. The bare `<bench_name>` is the mean.

| Suffix | Metric |
| --- | --- |
| _(none)_ / `_mean` | Mean time (ns) |
| `_median` | Median time |
| `_min` / `_max` | Min / max time |
| `_p50` `_p90` `_p95` `_p99` `_p999` | Percentiles |
| `_std_dev` | Standard deviation |
| `_skewness` / `_kurtosis` | Distribution shape |
| `_throughput` | Operations per second (1e9 / mean) |
| `_throughput_bytes` | Bytes per second. Needs byte throughput. |
| `_throughput_elements` | Elements per second. Needs element throughput. |
| `_samples` | Sample count |
| `_alloc_bytes` / `_alloc_count` | Mean bytes / allocations per iteration |

## Synthetic metrics

`#[synthetic]` computes a derived metric. A `#[verify]` rule can read it by its `id`:

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

`expr` is accepted as an alias for `formula`.

## Comparisons

### Simple comparison

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

The comparison produces a table that shows the speedup against the baseline for each benchmark.

`metric` takes one of these values. Any other value is a compile error:

| `metric` | Note |
| --- | --- |
| `mean` (default) | |
| `median`, `min`, `max` | |
| `p50`, `p90`, `p95`, `p99`, `p999` | |
| `throughput_bytes`, `throughput_elements` | Rates. A higher value is faster, so speedup is value over baseline. |

- A benchmark without that throughput kind has no entry.
- A baseline without that throughput kind drops the comparison.

### Series comparison

Series comparisons show multi-point results for scaling studies:

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

### Compare keys

| Key | Meaning |
| --- | --- |
| `id` | Comparison ID. |
| `title` | Chart title. |
| `benchmarks` | Benchmark IDs to compare. |
| `baseline` | Benchmark ID the speedup is measured against. |
| `metric` | Metric from the table above. Default `mean`. |
| `group` | Shared chart group for series comparisons. |
| `x` | Point position on the x axis. |
| `series` | Line names, by position. |
