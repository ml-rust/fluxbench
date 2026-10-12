# Examples

The `examples/` crate (`fluxbench-examples`) contains runnable demos for each feature:

| Example | What it shows |
| --- | --- |
| `feature_iteration` | `iter`, `iter_with_setup`, `iter_batched` with `BatchSize`, `iter_batched_ref`, `iter_batched_shared`, `iter_custom` |
| `feature_throughput` | Bytes and elements per second: attribute, `b.throughput()`, group, `#[verify]` |
| `feature_async` | Async benchmarks with tokio runtimes |
| `feature_params` | Parameterized benchmarks with `args` |
| `feature_verify` | `#[verify]` performance assertions |
| `feature_compare` | `#[compare]` baseline tables and series |
| `feature_allocations` | Heap allocation tracking |
| `feature_panic` | Crash isolation (panicking benchmarks) |
| `library_bench` | Benchmarking a library crate |
| `ci_regression` | CI regression detection workflow |
| `test_isolation` | Fixtures for the isolation tests: panic, abort, SIGSEGV, timeout |
| `test_baseline` | Fixtures for the baseline tests: `FLUX_TEST_SLOWDOWN` slows a critical benchmark |

Run an example from the workspace root:

```bash
cargo run -p fluxbench-examples --example feature_iteration --release
cargo run -p fluxbench-examples --example feature_verify --release -- --format json
```
