# Benchmark Macros

Reference for `#[bench]` and `#[group]`. The README shows the short form. This page lists every key.

- [Bench function](#bench-function)
- [Bench attributes](#bench-attributes)
- [Batched inputs](#batched-inputs)
- [Custom timing](#custom-timing)
- [Throughput](#throughput)
- [Groups](#groups)
- [Tags](#tags)
- [Parameterized benchmarks](#parameterized-benchmarks)
- [Time budget](#time-budget)
- [Async benchmarks](#async-benchmarks)

## Bench function

Add `#[bench]` to a function that takes `&mut Bencher`. Call `b.iter` once in the body.

```rust
use fluxbench::prelude::*;
use std::hint::black_box;

#[bench]
fn addition(b: &mut Bencher) {
    b.iter(|| black_box(42) + black_box(17));
}
```

Benchmarks can also live in `examples/`. Run them with `cargo run --example name --release`. Both `benches/` and `examples/` compile only when you run them. Neither ends up in your production binary.

## Bench attributes

| Attribute | Meaning |
| --- | --- |
| `id` | Benchmark ID. Default: the function name. |
| `group` | Group name. Default: `default`. |
| `tags` | Tags for `--tag` and `--skip-tag`. |
| `severity` | `critical`, `warning`, or `info`. Default: `info`. |
| `threshold` | Regression threshold in percent. 0 uses the global value. |
| `budget` | Time budget for the mean per iteration. See [Time budget](#time-budget). |
| `warmup`, `measurement` | Per-benchmark durations, e.g. `"500ms"`. |
| `samples` | Fixed mode: N samples of one iteration each, no warmup. |
| `sample_count`, `iters_per_sample` | Target sample count. Fixed iterations per sample. |
| `min_iterations`, `max_iterations` | Bounds on total measurement iterations. |
| `throughput_bytes`, `throughput_elements` | Work per call. See [Throughput](#throughput). |
| `depends_on` | Benchmark IDs that run before this one, e.g. `depends_on = ["a"]`. |
| `args` | Parameter values. See [Parameterized benchmarks](#parameterized-benchmarks). |
| `runtime`, `worker_threads`, `enable_time`, `enable_io` | Async runtime. See [Async benchmarks](#async-benchmarks). |

Duration values take the units `ns`, `us` (or `µs`), `ms`, `s`, and `m`. A bare number means seconds.

The `iterations` key is not supported. The macro returns an error that names `iters_per_sample`.

## Batched inputs

`iter_batched` and `iter_batched_ref` follow the criterion API:

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
- `iter_with_setup(setup, routine)` equals `iter_batched(setup, routine, BatchSize::SmallInput)`.

`BatchSize` sets the inputs per batch:

| `BatchSize` | Inputs per batch |
| --- | --- |
| `SmallInput` | Up to 1024. Fits cheap inputs. |
| `LargeInput` | Up to 16. Fits inputs that hold a lot of memory. |
| `PerIteration` | 1. Each call gets its own timer. |
| `NumBatches(n)` | Each sample's calls split into `n` batches. |
| `NumIterations(n)` | `n`, capped at the sample's calls. |

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

Earlier releases passed a `&T` routine to `iter_batched`. Change those calls to `iter_batched_shared`. `iter_batched` now takes a routine that receives `T` by value.

## Custom timing

`iter_custom` passes the body a call count. The body runs that many calls, times them, and returns the elapsed time:

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

## Throughput

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

## Groups

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

### Describing groups

`#[group]` describes a group that `#[bench(group = ...)]` names:

```rust
#[group(id = "parsing", description = "Parser performance", tags = "hot-path")]
struct Parsing;

#[group(id = "json_parsing", parent = "parsing", description = "JSON parsers")]
struct JsonParsing;
```

| `#[group]` key | Meaning |
| --- | --- |
| `id` | Group ID. |
| `description`, `desc` | Text the HTML report shows in the group's section. `desc` is an alias. |
| `tags` | Every benchmark in the group and its child groups carries them. |
| `parent` | The parent group. `--group parsing` also selects `json_parsing`. The parent must be a `#[group]` or a benchmark group. |

- A duplicate `id` or a parent cycle is an error at startup.
- The HTML report shows `description` and `tags` in the group's section.

### Group run config

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
- `read_64k` runs 10 samples (bench) and 1 s warmup (`disk`).
- It also runs with a 2 s measurement (`io`) and 64 KiB per call (`disk`).
- `#[bench(samples = N)]` beats a group's `sample_count` and `iters_per_sample`.
- A group `iters_per_sample` selects fixed mode: no warmup, no `min_iterations` or `max_iterations`.
- A resolved config that fails the startup check names the benchmark. Example: a group `sample_count = 1`.

## Tags

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

Run them with `cargo bench -- --tag network` or `cargo bench -- --skip-tag network`.

An unknown tag is an error that lists the known tags.

## Parameterized benchmarks

`args = [..]` runs one benchmark per value. The function takes the value as a second argument. Each ID is `id@value`:

```rust
#[bench(group = "scaling", args = [100, 1000])]
fn linear_scan(b: &mut Bencher, n: u32) { /* IDs: linear_scan@100, linear_scan@1000 */ }
```

## Time budget

`budget` fails a benchmark whose mean time per iteration exceeds it:

```rust
#[bench(budget = "50us")]
fn hot_path(b: &mut Bencher) {
    b.iter(|| std::hint::black_box(2 + 2));
}
```

- Units: `ns`, `us` (or `µs`), `ms`, `s`. A missing unit or a bad number is a compile error.
- `budget` must be above 0. A value of 0 is a compile error.
- `warmup` and `measurement` take the same syntax.
- Over budget, the status is `failed` with `failure.kind = "budget"`. The metrics stay in the report.
- A failed `severity = "critical"` benchmark exits 1 while `ci.fail_on_critical` is on. See [Exit status](statistics.md#exit-status).

## Async benchmarks

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
- `worker_threads` must be at least 1. It needs `runtime = "multi_thread"`. Otherwise it is a compile error.
- The runtime builds once per benchmark, outside the timer.
- `iter_async` times creating and awaiting each future. Each batch runs in one `block_on`.
- Setup before `iter_async` can await, use `tokio::time`, and call `tokio::spawn`.
- `iter_async` inside another runtime's `block_on` or task records an error and runs nothing.

### The `async` feature

Async support needs tokio. The `async` cargo feature of `fluxbench` gates it. The feature is on by default.

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

Check the gate from the workspace root:

```bash
# Builds with no tokio in the tree
cargo build -p fluxbench-core --no-default-features
cargo tree -p fluxbench-core --no-default-features -e normal | grep -c tokio   # prints 0
cargo build -p fluxbench --no-default-features
# Builds with the async API
cargo build -p fluxbench-core
cargo test -p fluxbench --test async_runtime
```
