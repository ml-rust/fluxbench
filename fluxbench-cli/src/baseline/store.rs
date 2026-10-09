//! Baseline files: path resolution, loading, and saving.
//!
//! A saved baseline is the JSON report plus each result's raw samples.
//! Other report outputs omit the samples.

use crate::executor::BenchExecutionResult;
use crate::{Cli, FluxConfig};
use fluxbench_report::{REPORT_SCHEMA_VERSION, Report, generate_json_report};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Resolve the baseline path from the CLI flag, then flux.toml, then the default.
///
/// - `Some(Some(path))`: explicit path from `--baseline /path/to/file`
/// - `Some(None)`: `--baseline` with no value. flux.toml `baseline_path` or `baseline.json` applies.
/// - `None`: flag not passed
///
/// A bare file name goes under `output.directory`, else under `target/fluxbench`.
pub(crate) fn resolve_baseline_path(
    cli_baseline: &Option<Option<PathBuf>>,
    config: &FluxConfig,
) -> Option<PathBuf> {
    match cli_baseline {
        Some(Some(path)) => Some(config.output.resolve_baseline(path)),
        Some(None) => Some(config.output.baseline_file()),
        None => None,
    }
}

/// Read and parse a baseline report.
///
/// It accepts every schema version up to [`REPORT_SCHEMA_VERSION`].
/// Schema 1 files carry no samples and take the summary comparison.
pub(crate) fn load_baseline(path: &Path) -> anyhow::Result<Report> {
    let json = std::fs::read_to_string(path).map_err(|e| {
        anyhow::anyhow!(
            "cannot read baseline {}: {e}. Check the path, or save one with --save-baseline.",
            path.display()
        )
    })?;
    parse_baseline(&json).map_err(|e| anyhow::anyhow!("baseline {}: {e}", path.display()))
}

/// Parse baseline JSON and check its schema version.
pub(crate) fn parse_baseline(json: &str) -> anyhow::Result<Report> {
    let value: serde_json::Value = serde_json::from_str(json).map_err(|e| {
        anyhow::anyhow!("invalid JSON: {e}. Re-save the baseline with --save-baseline.")
    })?;
    let version = value
        .get("meta")
        .and_then(|m| m.get("schema_version"))
        .and_then(serde_json::Value::as_u64);
    if let Some(version) = version.filter(|&v| v > u64::from(REPORT_SCHEMA_VERSION)) {
        anyhow::bail!(
            "schema version {version} is newer than supported version {REPORT_SCHEMA_VERSION}. \
             Upgrade fluxbench, or re-save the baseline with this version."
        );
    }
    serde_json::from_value(value).map_err(|e| {
        anyhow::anyhow!("not a fluxbench report: {e}. Re-save the baseline with --save-baseline.")
    })
}

/// Save the report as a baseline when `--save-baseline` or flux.toml `save_baseline` asks.
///
/// The saved file carries each result's raw samples from `results`.
pub(crate) fn save_baseline_if_needed(
    cli: &Cli,
    config: &FluxConfig,
    report: &Report,
    results: &[BenchExecutionResult],
) -> anyhow::Result<()> {
    if cli.save_baseline.is_none() && !config.output.save_baseline {
        return Ok(());
    }

    let path = match cli.save_baseline.clone().flatten() {
        Some(path) => config.output.resolve_baseline(&path),
        None => config.output.baseline_file(),
    };

    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| {
            anyhow::anyhow!(
                "cannot create baseline directory {}: {e}. Check permissions or pass another --save-baseline path.",
                parent.display()
            )
        })?;
    }

    let json = generate_json_report(&with_samples(report, results))?;
    std::fs::write(&path, json).map_err(|e| {
        anyhow::anyhow!(
            "cannot write baseline {}: {e}. Check permissions or pass another --save-baseline path.",
            path.display()
        )
    })?;
    eprintln!("Baseline saved to: {}", path.display());
    Ok(())
}

/// Copy of `report` with each result's raw samples from `results`.
///
/// Results without samples keep `samples: None`.
pub(crate) fn with_samples(report: &Report, results: &[BenchExecutionResult]) -> Report {
    let samples: HashMap<&str, &[f64]> = results
        .iter()
        .filter(|r| !r.samples.is_empty())
        .map(|r| (r.benchmark_id.as_str(), r.samples.as_slice()))
        .collect();
    let mut out = report.clone();
    for result in &mut out.results {
        result.samples = samples.get(result.id.as_str()).map(|s| s.to_vec());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{dummy_report, dummy_result};

    #[test]
    fn baseline_with_samples_roundtrips() {
        let mut result = dummy_result("b", 100.0, 0.0);
        result.samples = Some(vec![99.0, 100.0, 101.0]);
        let mut report = dummy_report(vec![result]);
        report.meta.schema_version = REPORT_SCHEMA_VERSION;

        let json = generate_json_report(&report).unwrap();
        let back = parse_baseline(&json).unwrap();

        assert_eq!(back.results[0].samples, Some(vec![99.0, 100.0, 101.0]));
        assert_eq!(back.meta.schema_version, REPORT_SCHEMA_VERSION);
    }

    #[test]
    fn baseline_without_samples_roundtrips() {
        let report = dummy_report(vec![dummy_result("b", 100.0, 0.0)]);
        let json = generate_json_report(&report).unwrap();
        assert!(!json.contains("\"samples\": ["));

        let back = parse_baseline(&json).unwrap();
        assert_eq!(back.results[0].samples, None);
    }

    #[test]
    fn schema_1_baseline_parses() {
        // Schema 1 has no `samples` and no schema 2 comparison fields.
        let mut value =
            serde_json::to_value(dummy_report(vec![dummy_result("b", 100.0, 0.0)])).unwrap();
        value["meta"]["schema_version"] = 1.into();
        value["results"][0]
            .as_object_mut()
            .unwrap()
            .remove("samples");
        let back = parse_baseline(&value.to_string()).unwrap();
        assert_eq!(back.meta.schema_version, 1);
        assert_eq!(back.results[0].samples, None);
    }

    #[test]
    fn newer_schema_is_rejected_with_next_action() {
        let mut value = serde_json::to_value(dummy_report(vec![])).unwrap();
        value["meta"]["schema_version"] = (REPORT_SCHEMA_VERSION + 1).into();
        let err = parse_baseline(&value.to_string()).unwrap_err().to_string();
        assert!(err.contains("newer than supported"), "{err}");
        assert!(err.contains("Upgrade fluxbench"), "{err}");
    }

    #[test]
    fn invalid_json_is_rejected_with_next_action() {
        let err = parse_baseline("{ not json").unwrap_err().to_string();
        assert!(err.contains("invalid JSON"), "{err}");
        assert!(err.contains("--save-baseline"), "{err}");
    }

    #[test]
    fn resolve_baseline_path_layers() {
        let config = FluxConfig::default();
        assert_eq!(resolve_baseline_path(&None, &config), None);
        assert_eq!(
            resolve_baseline_path(&Some(None), &config),
            Some(PathBuf::from("target/fluxbench/baseline.json"))
        );
        assert_eq!(
            resolve_baseline_path(&Some(Some(PathBuf::from("x.json"))), &config),
            Some(PathBuf::from("target/fluxbench/x.json"))
        );
        assert_eq!(
            resolve_baseline_path(&Some(Some(PathBuf::from("./x.json"))), &config),
            Some(PathBuf::from("./x.json"))
        );

        let mut configured = FluxConfig::default();
        configured.output.baseline_path = Some("main.json".to_string());
        configured.output.directory = Some("ci".to_string());
        assert_eq!(
            resolve_baseline_path(&Some(None), &configured),
            Some(PathBuf::from("ci/main.json"))
        );
    }
}
