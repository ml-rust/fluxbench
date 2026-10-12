# Statistics, Regression Detection, and Exit Status

## Measured values

Each benchmark reports these values:

- Bootstrap confidence intervals with the BCa method.
- Percentiles p50, p90, p95, p99, and p999.
- Outlier detection.
- Cycle counts from RDTSCP on x86_64 or CNTVCT_EL0 on aarch64. Other targets report 0.
- Allocation counts and bytes, when [allocation tracking](#allocation-tracking) is on.

Timing reads `Instant` once per timed batch. The run subtracts the measured empty-timer overhead.

### Custom bootstrap config

Set the resample count and confidence level in `flux.toml`:

```toml
[runner]
bootstrap_iterations = 100000
confidence_level = 0.99
```

More iterations give more precise intervals and slower reporting.

## Regression detection

`--baseline` and `compare <REF>` compare each benchmark with its baseline entry.

A change counts as a regression only when both hold:

- It is statistically significant. The difference CI excludes zero, or the probability of its direction reaches `runner.confidence_level`.
- Its size exceeds the threshold. The threshold comes from the first match: `#[bench(threshold = ...)]` if above 0, else `--threshold`, else `ci.regression_threshold`.

Improvements follow the same rule in the other direction.

### Statistical method

- Saved baselines (schema 2) carry raw samples. Both runs' samples are bootstrapped for the probability, the difference CI, and Cohen's d.
- Schema 1 baselines, or `bootstrap_iterations = 0`, fall back to a Welch-style z test on mean, std dev, and sample count.

## Exit status

The table shows the exit code for the default run and for `compare`:

| Finding | `fail_on_critical = true` | `fail_on_critical = false` |
| --- | --- | --- |
| Crashed benchmark | exit 1 | exit 1 |
| Critical `#[verify]` rule fails | exit 1 | warning |
| Regression on a `severity = "critical"` benchmark | exit 1 | warning |
| Regression on a `warning` or `info` benchmark | warning | warning |
| Failed `severity = "critical"` benchmark, e.g. over `budget` | exit 1 | warning |
| Failed `warning` or `info` benchmark | warning | warning |

`fail_on_critical` is set under `[ci]`. The default is `true`.

## Allocation tracking

FluxBench tracks heap allocations per benchmark iteration. Install `TrackingAllocator` as the global allocator in your benchmark binary:

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

Results include these metrics for each benchmark:

- **alloc_bytes**: total bytes allocated per iteration.
- **alloc_count**: number of allocations per iteration.

These metrics appear in JSON, CSV, and human output.

Limits in `[allocator]` in `flux.toml` can fail a benchmark on them. See [Config](configuration.md#allocator).

> **Note:** `#[global_allocator]` must be declared in the binary crate, such as your `benches/*.rs` file. It cannot go in a library. Rust allows one global allocator per binary. Without it, `track = true` reports zero allocations.

You can query the allocation counters yourself:

```rust
fluxbench::reset_allocation_counter();
// ... run some code ...
let (alloc_bytes, alloc_count) = fluxbench::current_allocation();
println!("Bytes: {}, Count: {}", alloc_bytes, alloc_count);
```
