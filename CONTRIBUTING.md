# Contributing to FluxBench

## Before you start

Open an issue before you start on anything non-trivial. Discuss the design on [Discord](https://discord.gg/jBhFk9kHPg) when the change affects the public API or the CLI.

## Build and test

Run these commands from the repository root. They match the `Test` workflow in `.github/workflows/test.yml`:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo doc --workspace --no-deps
cargo test --workspace --doc
cargo test --workspace
```

Changes to async code must keep the no-tokio gate. See [The async feature](docs/macros.md#the-async-feature) for the commands.

## Conventions

- Keep benchmarks crash-isolated. Do not add code that runs benchmark bodies in the supervisor process by default.
- Keep the statistics intact. Do not change the bootstrap, the BCa intervals, or the regression rules without tests that cover the new behavior.
- Put code in the crate that owns it:
  - `fluxbench`: the public facade and prelude.
  - `fluxbench-core`: the `Bencher` and the worker runtime.
  - `fluxbench-macros`: the proc macros.
  - `fluxbench-cli`: the supervisor, config, and CLI.
  - `fluxbench-stats`: bootstrap and percentile statistics.
  - `fluxbench-report`: JSON, CSV, HTML, and GitHub output.
  - `fluxbench-ipc`: the supervisor-worker protocol.
  - `fluxbench-logic`: the `#[verify]` expression engine.
- Add a test next to the code it covers. Use the `#[cfg(test)] mod tests` pattern that the crates already use.

## Pull requests

- Include tests for every behavior change.
- Update the matching `docs/` page for any change to a flag, key, or output field.
- Add a line to `CHANGELOG.md` for every user-visible change.
- Keep one logical change in each pull request.
