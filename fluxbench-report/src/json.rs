//! JSON Output

use crate::report::Report;
use serde::{Deserialize, Serialize};

/// Schema information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSchema {
    /// Schema identifier
    pub schema: String,
    /// Schema version
    pub version: String,
}

/// Generate a prettified JSON report.
///
/// Serializes the benchmark report into machine-readable JSON format.
pub fn generate_json_report(report: &Report) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::{BenchmarkStatus, REPORT_SCHEMA_VERSION};
    use crate::test_fixtures::{failure, metrics, report, result};

    fn fixture() -> Report {
        report(vec![
            result(
                "ok",
                "g",
                BenchmarkStatus::Passed,
                Some(metrics(100.0)),
                None,
            ),
            result(
                "bad",
                "g",
                BenchmarkStatus::Crashed,
                None,
                Some(failure("signal", "worker killed")),
            ),
        ])
    }

    #[test]
    fn json_has_stable_top_level_keys() {
        let json = generate_json_report(&fixture()).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let keys: Vec<&str> = value
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        for key in [
            "meta",
            "results",
            "comparisons",
            "comparison_series",
            "synthetics",
            "verifications",
            "summary",
        ] {
            assert!(keys.contains(&key), "missing {key}");
        }
        assert!(!keys.contains(&"baseline_meta"));
        assert_eq!(value["meta"]["schema_version"], REPORT_SCHEMA_VERSION);
        assert_eq!(value["meta"]["git_branch"], "main");
    }

    #[test]
    fn json_results_keep_status_and_failure_fields() {
        let json = generate_json_report(&fixture()).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["results"][0]["id"], "ok");
        assert_eq!(value["results"][0]["status"], "passed");
        assert_eq!(value["results"][0]["metrics"]["mean_ns"], 100.0);
        assert!(value["results"][0].get("samples").is_none());
        assert_eq!(value["results"][1]["status"], "crashed");
        assert!(value["results"][1]["metrics"].is_null());
        assert_eq!(value["results"][1]["failure"]["kind"], "signal");
        assert_eq!(value["summary"]["crashed"], 1);
    }

    #[test]
    fn json_is_pretty_and_round_trips() {
        let json = generate_json_report(&fixture()).unwrap();
        assert!(json.starts_with("{\n  \"meta\": {"));
        let back: Report = serde_json::from_str(&json).unwrap();
        assert_eq!(back.results.len(), 2);
        assert_eq!(back.results[1].status, BenchmarkStatus::Crashed);
    }
}
