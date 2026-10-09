//! Config and selection errors, end to end.
//!
//! Each test runs the `test_baseline` example binary in its own temp directory.
//! No test here runs a benchmark: each one fails at startup or selects nothing.

mod common;

use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

/// Filter that matches no benchmark in the fixture
const NO_MATCH: &str = "^no_such_benchmark$";

fn fixture_binary() -> &'static Path {
    static BINARY: OnceLock<PathBuf> = OnceLock::new();
    BINARY.get_or_init(|| common::example_binary("test_baseline"))
}

/// Fresh temp directory with an optional flux.toml
fn workdir(label: &str, flux_toml: Option<&str>) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "fluxbench-config-it-{}-{label}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    if let Some(text) = flux_toml {
        std::fs::write(dir.join("flux.toml"), text).expect("write flux.toml");
    }
    dir
}

fn run(dir: &Path, args: &[&str]) -> Output {
    Command::new(fixture_binary())
        .args(args)
        .current_dir(dir)
        .output()
        .expect("fixture runs")
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn describe(output: &Output) -> String {
    format!(
        "status: {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        stderr(output)
    )
}

#[test]
fn empty_selection_with_json_prints_an_empty_report() {
    let dir = workdir("empty-json", None);
    let output = run(&dir, &["--format", "json", NO_MATCH]);
    assert_eq!(output.status.code(), Some(0), "{}", describe(&output));
    let report: Value = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|e| panic!("stdout is not JSON: {e}\n{}", describe(&output)));
    assert_eq!(report["results"].as_array().map(Vec::len), Some(0));
    assert!(stderr(&output).contains("No benchmarks found."));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn output_format_from_toml_applies_without_flag() {
    let dir = workdir("toml-format", Some("[output]\nformat = \"csv\"\n"));
    let output = run(&dir, &[NO_MATCH]);
    assert_eq!(output.status.code(), Some(0), "{}", describe(&output));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.starts_with("id,name,group,status"),
        "{}",
        describe(&output)
    );
    assert_eq!(stdout.lines().count(), 1);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn bare_output_file_goes_under_output_directory() {
    let dir = workdir("out-dir", Some("[output]\ndirectory = \"reports\"\n"));
    let output = run(&dir, &["--format", "json", "--output", "r.json", NO_MATCH]);
    assert_eq!(output.status.code(), Some(0), "{}", describe(&output));
    let written = std::fs::read_to_string(dir.join("reports/r.json")).expect("report written");
    let report: Value = serde_json::from_str(&written).expect("report is JSON");
    assert_eq!(report["results"].as_array().map(Vec::len), Some(0));
    assert!(output.stdout.is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn bare_output_file_stays_in_cwd_without_output_directory() {
    let dir = workdir("out-cwd", None);
    let output = run(&dir, &["--format", "json", "--output", "r.json", NO_MATCH]);
    assert_eq!(output.status.code(), Some(0), "{}", describe(&output));
    let written = std::fs::read_to_string(dir.join("r.json")).expect("report written to cwd");
    let report: Value = serde_json::from_str(&written).expect("report is JSON");
    assert_eq!(report["results"].as_array().map(Vec::len), Some(0));
    assert!(!dir.join("target/fluxbench/r.json").exists());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn unknown_toml_key_is_an_error_naming_it() {
    let dir = workdir("unknown-key", Some("[runner]\nwarmup = \"1s\"\n"));
    let output = run(&dir, &[NO_MATCH]);
    assert_eq!(output.status.code(), Some(1), "{}", describe(&output));
    let err = stderr(&output);
    assert!(err.contains("unknown field `warmup`"), "{err}");
    assert!(err.contains("warmup_time"), "{err}");
    assert!(err.contains("flux.toml"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn malformed_toml_is_an_error_with_path_and_line() {
    let dir = workdir("malformed", Some("[runner\njobs = 2\n"));
    let output = run(&dir, &[NO_MATCH]);
    assert_eq!(output.status.code(), Some(1), "{}", describe(&output));
    let err = stderr(&output);
    assert!(
        err.contains(&dir.join("flux.toml").display().to_string()),
        "{err}"
    );
    assert!(err.contains("line 1"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn invalid_filter_regex_is_an_error() {
    let dir = workdir("bad-regex", None);
    let output = run(&dir, &["crit(ical"]);
    assert_eq!(output.status.code(), Some(1), "{}", describe(&output));
    assert!(stderr(&output).contains("invalid filter regex `crit(ical`"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn unknown_format_is_rejected_by_clap() {
    let dir = workdir("bad-format", None);
    let output = run(&dir, &["--format", "yaml", NO_MATCH]);
    assert_eq!(output.status.code(), Some(2), "{}", describe(&output));
    let err = stderr(&output);
    assert!(err.contains("yaml"), "{err}");
    assert!(err.contains("json"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}
