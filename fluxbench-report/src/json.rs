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
