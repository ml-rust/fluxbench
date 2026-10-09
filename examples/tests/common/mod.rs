//! Shared helpers for integration tests that run example binaries.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Path of the example binary `name`.
///
/// `cargo test` builds examples next to `deps/`. If it is missing, build it once.
pub fn example_binary(name: &str) -> PathBuf {
    let exe = std::env::current_exe().expect("test binary path");
    let profile_dir = exe
        .parent()
        .and_then(Path::parent)
        .expect("test binary sits in <target>/<profile>/deps");
    let built = profile_dir.join("examples").join(name);
    if built.exists() {
        return built;
    }

    let release = profile_dir.file_name().is_some_and(|n| n == "release");
    let target_dir = profile_dir
        .parent()
        .expect("profile dir has a parent")
        .join(format!("{name}-fixture"));
    let mut cargo = Command::new(env!("CARGO"));
    cargo
        .args(["build", "--example", name, "-p", "fluxbench-examples"])
        .arg("--target-dir")
        .arg(&target_dir);
    if release {
        cargo.arg("--release");
    }
    let status = cargo.status().expect("cargo runs");
    assert!(status.success(), "building the {name} example failed");
    target_dir
        .join(if release { "release" } else { "debug" })
        .join("examples")
        .join(name)
}
