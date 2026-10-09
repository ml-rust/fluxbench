//! Baseline regression detection, end to end.
//!
//! Each step runs the `test_baseline` example binary as a real supervisor.
//! `FLUX_TEST_SLOWDOWN` slows its critical benchmark.

mod common;

use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

/// Path of the `test_baseline` example binary.
fn fixture_binary() -> &'static Path {
    static BINARY: OnceLock<PathBuf> = OnceLock::new();
    BINARY.get_or_init(|| common::example_binary("test_baseline"))
}

/// Run the fixture in `dir` with `extra` flags and an optional slowdown.
fn run_fixture(dir: &Path, slowdown: Option<&str>, extra: &[&str]) -> Output {
    let mut cmd = Command::new(fixture_binary());
    cmd.args([
        "--warmup",
        "10ms",
        "--measurement",
        "100ms",
        "--format",
        "json",
    ])
    .args(extra)
    .current_dir(dir)
    .env_remove("FLUX_TEST_SLOWDOWN");
    if let Some(value) = slowdown {
        cmd.env("FLUX_TEST_SLOWDOWN", value);
    }
    cmd.output().expect("fixture runs")
}

fn report(output: &Output) -> Value {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    serde_json::from_str(&stdout).unwrap_or_else(|e| {
        panic!("stdout is not a JSON report: {e}\nstdout:\n{stdout}\nstderr:\n{stderr}")
    })
}

fn entry<'a>(report: &'a Value, id: &str) -> &'a Value {
    report["results"]
        .as_array()
        .expect("results array")
        .iter()
        .find(|r| r["id"] == id)
        .unwrap_or_else(|| panic!("no result for {id} in {report:#}"))
}

fn describe(output: &Output) -> String {
    format!(
        "status {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

#[test]
fn critical_regression_fails_and_unchanged_run_passes() {
    let dir = std::env::temp_dir().join(format!("fluxbench-baseline-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    std::fs::write(
        dir.join("flux.toml"),
        "[runner]\nbootstrap_iterations = 1000\n",
    )
    .expect("write flux.toml");
    let baseline = dir.join("baseline.json");
    let baseline_arg = baseline.to_str().expect("utf-8 temp path");

    // 1. Save a baseline. Saved files carry raw samples.
    let saved = run_fixture(&dir, None, &["--save-baseline", baseline_arg]);
    assert_eq!(saved.status.code(), Some(0), "{}", describe(&saved));
    let stdout_report = report(&saved);
    assert!(
        entry(&stdout_report, "critical_spin")
            .get("samples")
            .is_none(),
        "normal JSON output must not carry samples"
    );
    let file: Value =
        serde_json::from_str(&std::fs::read_to_string(&baseline).expect("baseline written"))
            .expect("baseline is JSON");
    assert_eq!(file["meta"]["schema_version"], 2);
    let samples = entry(&file, "critical_spin")["samples"]
        .as_array()
        .expect("baseline carries samples");
    assert!(samples.len() >= 2, "{samples:?}");

    // 2. Slow the critical benchmark: the run must fail and record the regression.
    let slow = run_fixture(&dir, Some("200000"), &["--baseline", baseline_arg]);
    assert_eq!(slow.status.code(), Some(1), "{}", describe(&slow));
    let slow_report = report(&slow);
    assert!(
        slow_report["summary"]["regressions"].as_u64().unwrap_or(0) >= 1,
        "{slow_report:#}"
    );
    let cmp = &entry(&slow_report, "critical_spin")["comparison"];
    assert_eq!(cmp["is_significant"], true, "{cmp:#}");
    assert!(
        cmp["relative_change"].as_f64().unwrap_or(0.0) > 50.0,
        "{cmp:#}"
    );
    assert_eq!(cmp["method"], "bootstrap", "{cmp:#}");
    assert_eq!(cmp["threshold"], 50.0, "{cmp:#}");
    let stderr = String::from_utf8_lossy(&slow.stderr);
    assert!(
        stderr.contains("critical benchmark critical_spin regressed"),
        "{stderr}"
    );

    // 3. No slowdown: the run against the same baseline passes.
    let same = run_fixture(&dir, None, &["--baseline", baseline_arg]);
    assert_eq!(same.status.code(), Some(0), "{}", describe(&same));
    let same_report = report(&same);
    let cmp = &entry(&same_report, "critical_spin")["comparison"];
    // Noise can show an improvement. Only a slowdown past the threshold is an error.
    assert!(
        cmp["relative_change"].as_f64().unwrap_or(f64::MAX) <= 50.0,
        "{cmp:#}"
    );
    assert_eq!(
        same_report["summary"]["regressions"].as_u64(),
        Some(0),
        "{same_report:#}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
