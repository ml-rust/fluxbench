# Crash Isolation and Workers

Benchmarks run in worker processes by default. Every benchmark gets its own report entry, whatever happens to its worker. A panic, segfault, abort, or timeout in one benchmark does not stop the rest of the run.

## Modes

| Mode | Flag | Behavior |
| --- | --- | --- |
| Persistent (default) | none | One worker runs many benchmarks. A worker that crashes, times out, or breaks the IPC stream is killed and replaced. The next benchmark gets a fresh worker. |
| One-shot | `--one-shot` | Each benchmark gets a fresh worker. |
| In-process | `--isolated=false` | Benchmarks run in the supervisor process. Use this for debugging only. |

- `--jobs N`: N workers run benchmarks in parallel. Both persistent and one-shot modes support it. Parallel jobs perturb measurements.
- In-process mode: a panic crashes the whole run at once.

## CPU pinning

`--pin` and `--no-pin` control CPU pinning. They conflict with each other. They override `runner.pin_cpus` in `flux.toml`. The default is on.

- Linux only. The supervisor reads the process affinity mask and skips CPU 0 when other CPUs exist.
- `--jobs N` above the usable CPU count makes workers share cores. A warning prints on stderr.
- A failed pin prints one stderr warning in the worker. The benchmark still runs.
- `--no-pin` leaves workers to the OS scheduler.

## `failure.kind` values

A benchmark that does not pass reports a `status` and a `failure.kind`:

| `failure.kind` | Cause | Status |
| --- | --- | --- |
| `panic` | The body panicked. `message` holds the panic text and location. `backtrace` comes from the panic site. | `crashed` |
| `signal` | A signal killed the worker. `failure.signal` holds the name, e.g. `SIGSEGV` or `SIGABRT`. Unix only. | `crashed` |
| `timeout` | The worker exceeded the worker timeout (`--worker-timeout`, else `runner.timeout`). `message` states the partial sample count. | `crashed` |
| `crashed` | The worker exited with a code, broke the IPC stream, or did not start. | `crashed` |
| `unknown` | The body never called `b.iter*`, or another measurement error occurred. | `failed` |
| `budget` | The mean time per iteration exceeded `#[bench(budget)]`. | `failed` |
| `allocation_limit` | The benchmark broke `allocator.fail_on_allocation` or `allocator.max_bytes_per_iter`. | `failed` |

- `panic`, `signal`, `timeout`, and `crashed` set status `crashed`. The run exits with code 1.
- `unknown`, `budget`, and `allocation_limit` set status `failed`. A failed critical benchmark exits 1 while `ci.fail_on_critical` is on. See [Exit status](statistics.md#exit-status).

## Timeouts

- Workers stream samples to the supervisor during measurement.
- On Unix, a timeout sends SIGTERM. The supervisor collects the samples sent so far for 500 ms. Then it sends SIGKILL.
- The partial samples never count as a pass and never form metrics.
- On other platforms, a timeout kills the worker at once.

## Handshake

A new worker must send its handshake within 10 s. Otherwise the supervisor kills it and reports `crashed` for that benchmark.

## Worker stdout

- On Unix, the worker's stdout goes to `/dev/null`. IPC uses dedicated pipes, so `println!` in a benchmark body does not corrupt the protocol.
- On other platforms, IPC uses the worker's stdin and stdout. A `println!` in a benchmark body corrupts the protocol stream.
