//! Process-isolation integration tests.
//!
//! Each test runs the `test_isolation` example binary as a real supervisor.
//! Its workers panic, abort, segfault, hang, and print to stdout.
//! The JSON report must hold one correct entry per benchmark in every mode.

#![cfg(unix)]

use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

/// Path of the `test_isolation` example binary.
///
/// `cargo test` builds examples next to `deps/`. If it is missing, build it once.
fn fixture_binary() -> &'static Path {
    static BINARY: OnceLock<PathBuf> = OnceLock::new();
    BINARY.get_or_init(|| {
        let exe = std::env::current_exe().expect("test binary path");
        let profile_dir = exe
            .parent()
            .and_then(Path::parent)
            .expect("test binary sits in <target>/<profile>/deps");
        let built = profile_dir.join("examples").join("test_isolation");
        if built.exists() {
            return built;
        }

        let release = profile_dir.file_name().is_some_and(|n| n == "release");
        let target_dir = profile_dir
            .parent()
            .expect("profile dir has a parent")
            .join("isolation-fixture");
        let mut cargo = Command::new(env!("CARGO"));
        cargo
            .args([
                "build",
                "--example",
                "test_isolation",
                "-p",
                "fluxbench-examples",
            ])
            .arg("--target-dir")
            .arg(&target_dir);
        if release {
            cargo.arg("--release");
        }
        let status = cargo.status().expect("cargo runs");
        assert!(
            status.success(),
            "building the test_isolation example failed"
        );
        target_dir
            .join(if release { "release" } else { "debug" })
            .join("examples")
            .join("test_isolation")
    })
}

/// Run the fixture with `extra` flags. Returns the parsed JSON report.
fn run_fixture(label: &str, extra: &[&str]) -> Value {
    let dir = std::env::temp_dir().join(format!(
        "fluxbench-isolation-{}-{label}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    // Skip the bootstrap: these tests check statuses, not intervals.
    std::fs::write(
        dir.join("flux.toml"),
        "[runner]\nbootstrap_iterations = 0\n",
    )
    .expect("write flux.toml");

    let output = Command::new(fixture_binary())
        .args([
            "--warmup",
            "10ms",
            "--measurement",
            "50ms",
            "--worker-timeout",
            "3",
            "--format",
            "json",
        ])
        .args(extra)
        .current_dir(&dir)
        .output()
        .expect("fixture runs");
    let _ = std::fs::remove_dir_all(&dir);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    // Crashed benchmarks make the supervisor exit 1.
    assert_eq!(
        output.status.code(),
        Some(1),
        "unexpected exit status.\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    serde_json::from_str(&stdout).unwrap_or_else(|e| {
        panic!("stdout is not a JSON report: {e}\nstdout:\n{stdout}\nstderr:\n{stderr}")
    })
}

/// Report entry for benchmark `id`.
fn entry<'a>(report: &'a Value, id: &str) -> &'a Value {
    report["results"]
        .as_array()
        .expect("results array")
        .iter()
        .find(|r| r["id"] == id)
        .unwrap_or_else(|| panic!("no result for {id} in {report:#}"))
}

fn assert_passed(report: &Value, id: &str) {
    let r = entry(report, id);
    assert_eq!(r["status"], "passed", "{id}: {r:#}");
    assert!(r["failure"].is_null(), "{id}: {r:#}");
    assert!(
        r["metrics"]["samples"].as_u64().unwrap_or(0) > 0,
        "{id}: {r:#}"
    );
}

fn assert_crashed(report: &Value, id: &str, kind: &str) -> Value {
    let r = entry(report, id);
    assert_eq!(r["status"], "crashed", "{id}: {r:#}");
    assert_eq!(r["failure"]["kind"], kind, "{id}: {r:#}");
    assert!(
        r["metrics"].is_null(),
        "{id}: partial samples must not form metrics: {r:#}"
    );
    r["failure"].clone()
}

fn check_report(report: &Value) {
    assert_eq!(report["results"].as_array().map(Vec::len), Some(7));

    assert_passed(report, "ok_fast");
    assert_passed(report, "prints_to_stdout");
    assert_passed(report, "ok_after");

    let panic = assert_crashed(report, "panics", "panic");
    let message = panic["message"].as_str().unwrap_or_default();
    assert!(message.contains("isolation fixture panic"), "{panic:#}");
    assert!(
        message.contains("test_isolation.rs"),
        "panic location: {panic:#}"
    );
    let backtrace = panic["backtrace"].as_str().unwrap_or_default();
    assert!(!backtrace.is_empty(), "panic backtrace: {panic:#}");

    let abort = assert_crashed(report, "aborts", "signal");
    assert_eq!(abort["signal"], "SIGABRT", "{abort:#}");

    let segv = assert_crashed(report, "segv", "signal");
    assert_eq!(segv["signal"], "SIGSEGV", "{segv:#}");

    let timeout = assert_crashed(report, "sleeps_past_timeout", "timeout");
    let message = timeout["message"].as_str().unwrap_or_default();
    assert!(message.contains("partial samples"), "{timeout:#}");
    assert!(timeout.get("signal").is_none(), "{timeout:#}");

    let summary = &report["summary"];
    assert_eq!(summary["passed"], 3, "{summary:#}");
    assert_eq!(summary["crashed"], 4, "{summary:#}");
}

#[test]
fn persistent_mode_reports_every_benchmark() {
    check_report(&run_fixture("persistent", &[]));
}

#[test]
fn one_shot_mode_reports_every_benchmark() {
    check_report(&run_fixture("one-shot", &["--one-shot"]));
}

#[test]
fn two_jobs_report_every_benchmark() {
    check_report(&run_fixture("jobs2", &["--jobs", "2"]));
}

#[test]
fn two_jobs_one_shot_report_every_benchmark() {
    check_report(&run_fixture(
        "jobs2-one-shot",
        &["--jobs", "2", "--one-shot"],
    ));
}

#[test]
fn hung_benchmark_keeps_streamed_partial_samples() {
    let report = run_fixture("partial", &["sleeps_past_timeout|ok_fast"]);
    let r = entry(&report, "sleeps_past_timeout");
    assert_eq!(r["failure"]["kind"], "timeout", "{r:#}");
    let message = r["failure"]["message"].as_str().unwrap_or_default();
    let count: usize = message
        .split("; ")
        .nth(1)
        .and_then(|tail| tail.split_whitespace().next())
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("no partial count in {message:?}"));
    // The fixture records up to 30 fast samples before it hangs.
    // The flusher sends them while the body sleeps.
    assert!(
        count > 0,
        "expected streamed samples before the hang: {message:?}"
    );
}
