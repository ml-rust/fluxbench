//! CSV Report Generation
//!
//! Generates CSV output for spreadsheet-compatible benchmark results.
//! Includes all timing metrics, allocation data, and CPU cycle information.
//! Fields follow RFC 4180 quoting. Header and rows come from one column list.

use crate::{BenchmarkMetrics, BenchmarkReportResult, BenchmarkStatus, FailureInfo, Report};
use std::borrow::Cow;

/// One CSV column: its header name and the cell for a result
struct Column {
    name: &'static str,
    cell: fn(&BenchmarkReportResult) -> String,
}

/// Lowercase status name
fn status_name(status: BenchmarkStatus) -> &'static str {
    match status {
        BenchmarkStatus::Passed => "passed",
        BenchmarkStatus::Failed => "failed",
        BenchmarkStatus::Crashed => "crashed",
        BenchmarkStatus::Skipped => "skipped",
    }
}

/// Cell from the metrics of a result. Empty when the result has no metrics.
fn from_metrics(result: &BenchmarkReportResult, f: impl Fn(&BenchmarkMetrics) -> String) -> String {
    result.metrics.as_ref().map(f).unwrap_or_default()
}

/// Cell from an optional metric. Empty when the value is absent.
fn optional(value: Option<f64>) -> String {
    value.map(|v| format!("{v:.2}")).unwrap_or_default()
}

/// Cell from the failure info of a result. Empty when the result has no failure.
fn from_failure(result: &BenchmarkReportResult, f: impl Fn(&FailureInfo) -> &str) -> String {
    result
        .failure
        .as_ref()
        .map(|failure| f(failure).to_string())
        .unwrap_or_default()
}

/// Output columns in order. Older columns come first so existing consumers keep working.
const COLUMNS: &[Column] = &[
    Column {
        name: "id",
        cell: |r| r.id.clone(),
    },
    Column {
        name: "name",
        cell: |r| r.name.clone(),
    },
    Column {
        name: "group",
        cell: |r| r.group.clone(),
    },
    Column {
        name: "status",
        cell: |r| status_name(r.status).to_string(),
    },
    Column {
        name: "mean_ns",
        cell: |r| from_metrics(r, |m| format!("{:.2}", m.mean_ns)),
    },
    Column {
        name: "median_ns",
        cell: |r| from_metrics(r, |m| format!("{:.2}", m.median_ns)),
    },
    Column {
        name: "std_dev_ns",
        cell: |r| from_metrics(r, |m| format!("{:.2}", m.std_dev_ns)),
    },
    Column {
        name: "min_ns",
        cell: |r| from_metrics(r, |m| format!("{:.2}", m.min_ns)),
    },
    Column {
        name: "max_ns",
        cell: |r| from_metrics(r, |m| format!("{:.2}", m.max_ns)),
    },
    Column {
        name: "p50_ns",
        cell: |r| from_metrics(r, |m| format!("{:.2}", m.p50_ns)),
    },
    Column {
        name: "p95_ns",
        cell: |r| from_metrics(r, |m| format!("{:.2}", m.p95_ns)),
    },
    Column {
        name: "p99_ns",
        cell: |r| from_metrics(r, |m| format!("{:.2}", m.p99_ns)),
    },
    Column {
        name: "samples",
        cell: |r| from_metrics(r, |m| m.samples.to_string()),
    },
    Column {
        name: "alloc_bytes",
        cell: |r| from_metrics(r, |m| m.alloc_bytes.to_string()),
    },
    Column {
        name: "alloc_count",
        cell: |r| from_metrics(r, |m| m.alloc_count.to_string()),
    },
    Column {
        name: "mean_cycles",
        cell: |r| from_metrics(r, |m| format!("{:.0}", m.mean_cycles)),
    },
    Column {
        name: "median_cycles",
        cell: |r| from_metrics(r, |m| format!("{:.0}", m.median_cycles)),
    },
    Column {
        name: "cycles_per_ns",
        cell: |r| from_metrics(r, |m| format!("{:.2}", m.cycles_per_ns)),
    },
    Column {
        name: "p90_ns",
        cell: |r| from_metrics(r, |m| format!("{:.2}", m.p90_ns)),
    },
    Column {
        name: "p999_ns",
        cell: |r| from_metrics(r, |m| format!("{:.2}", m.p999_ns)),
    },
    Column {
        name: "ci_lower_ns",
        cell: |r| from_metrics(r, |m| optional(m.ci_lower_ns)),
    },
    Column {
        name: "ci_upper_ns",
        cell: |r| from_metrics(r, |m| optional(m.ci_upper_ns)),
    },
    Column {
        name: "trimmed_mean_ns",
        cell: |r| from_metrics(r, |m| optional(m.trimmed_mean_ns)),
    },
    Column {
        name: "throughput_ops_sec",
        cell: |r| from_metrics(r, |m| optional(m.throughput_ops_sec)),
    },
    Column {
        name: "failure_kind",
        cell: |r| from_failure(r, |f| &f.kind),
    },
    Column {
        name: "failure_message",
        cell: |r| from_failure(r, |f| &f.message),
    },
];

/// Quote a field when it holds a comma, a quote, CR, or LF. Inner quotes double.
fn quote(field: &str) -> Cow<'_, str> {
    if field.contains([',', '"', '\r', '\n']) {
        Cow::Owned(format!("\"{}\"", field.replace('"', "\"\"")))
    } else {
        Cow::Borrowed(field)
    }
}

/// Push `fields` as one comma-separated, newline-terminated record.
fn push_record<'a>(csv: &mut String, fields: impl Iterator<Item = Cow<'a, str>>) {
    for (i, field) in fields.enumerate() {
        if i > 0 {
            csv.push(',');
        }
        csv.push_str(&field);
    }
    csv.push('\n');
}

/// Generate a CSV report from benchmark results
///
/// # Arguments
/// * `report` - Complete benchmark report
///
/// # Returns
/// CSV-formatted string with header row and data rows for each benchmark.
/// Every row has one cell per column. Cells without a value are empty.
/// Fields with a comma, quote, CR, or LF are quoted per RFC 4180.
///
/// # CSV Columns
/// - `id` - Benchmark identifier
/// - `name` - Human-readable benchmark name
/// - `group` - Benchmark group
/// - `status` - Execution status (passed/failed/crashed/skipped)
/// - `mean_ns` - Mean execution time in nanoseconds
/// - `median_ns` - Median execution time
/// - `std_dev_ns` - Standard deviation
/// - `min_ns` / `max_ns` - Min/max execution times
/// - `p50_ns` / `p95_ns` / `p99_ns` - Percentiles
/// - `samples` - Number of samples collected
/// - `alloc_bytes` / `alloc_count` - Allocation metrics
/// - `mean_cycles` / `median_cycles` - CPU cycle metrics (x86_64 only)
/// - `cycles_per_ns` - CPU frequency estimate
/// - `p90_ns` / `p999_ns` - Percentiles
/// - `ci_lower_ns` / `ci_upper_ns` - Confidence interval bounds (empty when the bootstrap is skipped)
/// - `trimmed_mean_ns` - Mean without detected outliers (empty when absent)
/// - `throughput_ops_sec` - Operations per second (empty when absent)
/// - `failure_kind` / `failure_message` - Failure details (empty without a failure)
pub fn generate_csv_report(report: &Report) -> String {
    let mut csv = String::new();
    push_record(&mut csv, COLUMNS.iter().map(|c| Cow::Borrowed(c.name)));
    for result in &report.results {
        let cells: Vec<String> = COLUMNS.iter().map(|c| (c.cell)(result)).collect();
        push_record(&mut csv, cells.iter().map(|cell| quote(cell)));
    }
    csv
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::{failure, metrics, report, result};

    /// Parse RFC 4180 text into records. Handles quoted fields with commas, quotes, and newlines.
    fn parse(text: &str) -> Vec<Vec<String>> {
        let mut records = Vec::new();
        let mut record = Vec::new();
        let mut field = String::new();
        let mut in_quotes = false;
        let mut chars = text.chars().peekable();
        while let Some(c) = chars.next() {
            match (in_quotes, c) {
                (true, '"') if chars.peek() == Some(&'"') => {
                    field.push('"');
                    chars.next();
                }
                (true, '"') => in_quotes = false,
                (true, c) => field.push(c),
                (false, '"') => in_quotes = true,
                (false, ',') => record.push(std::mem::take(&mut field)),
                (false, '\n') => {
                    record.push(std::mem::take(&mut field));
                    records.push(std::mem::take(&mut record));
                }
                (false, c) => field.push(c),
            }
        }
        records
    }

    fn all_status_report() -> Report {
        report(vec![
            result(
                "p",
                "g",
                BenchmarkStatus::Passed,
                Some(metrics(100.0)),
                None,
            ),
            result(
                "f",
                "g",
                BenchmarkStatus::Failed,
                None,
                Some(failure("assertion", "left != right")),
            ),
            result(
                "c",
                "g",
                BenchmarkStatus::Crashed,
                None,
                Some(failure("signal", "worker killed")),
            ),
            result("s", "g", BenchmarkStatus::Skipped, None, None),
        ])
    }

    #[test]
    fn every_row_has_the_header_width() {
        let records = parse(&generate_csv_report(&all_status_report()));
        assert_eq!(records.len(), 5);
        assert_eq!(records[0].len(), COLUMNS.len());
        for row in &records {
            assert_eq!(row.len(), records[0].len());
        }
    }

    #[test]
    fn failure_cells_are_filled_for_failed_and_crashed() {
        let records = parse(&generate_csv_report(&all_status_report()));
        let kind = COLUMNS
            .iter()
            .position(|c| c.name == "failure_kind")
            .unwrap();
        assert_eq!(records[1][kind], "assertion");
        assert_eq!(records[1][kind + 1], "left != right");
        assert_eq!(records[2][kind], "signal");
        assert_eq!(records[3][kind], "");
        assert_eq!(records[4][kind], "");
    }

    #[test]
    fn missing_optional_values_are_empty_cells() {
        let mut m = metrics(100.0);
        m.ci_lower_ns = None;
        m.ci_upper_ns = None;
        m.trimmed_mean_ns = None;
        m.throughput_ops_sec = None;
        let rep = report(vec![result(
            "p",
            "g",
            BenchmarkStatus::Passed,
            Some(m),
            None,
        )]);
        let records = parse(&generate_csv_report(&rep));
        for name in [
            "ci_lower_ns",
            "ci_upper_ns",
            "trimmed_mean_ns",
            "throughput_ops_sec",
        ] {
            let i = COLUMNS.iter().position(|c| c.name == name).unwrap();
            assert_eq!(records[1][i], "", "{name}");
        }
        assert_eq!(records[1][4], "100.00");
    }

    #[test]
    fn special_characters_round_trip() {
        let id = "vec![1,2] \"quoted\"";
        let message = "line1\nline2, \"q\"\r\nline3";
        let rep = report(vec![result(
            id,
            "g,h",
            BenchmarkStatus::Failed,
            None,
            Some(failure("panic", message)),
        )]);
        let records = parse(&generate_csv_report(&rep));
        assert_eq!(records.len(), 2);
        assert_eq!(records[1].len(), COLUMNS.len());
        assert_eq!(records[1][0], id);
        assert_eq!(records[1][2], "g,h");
        assert_eq!(records[1][COLUMNS.len() - 1], message);
    }

    #[test]
    fn plain_fields_stay_unquoted() {
        assert_eq!(quote("plain_id@10"), "plain_id@10");
        assert_eq!(quote("a,b"), "\"a,b\"");
        assert_eq!(quote("say \"hi\""), "\"say \"\"hi\"\"\"");
    }

    #[test]
    fn golden_two_result_snapshot() {
        let rep = report(vec![
            result(
                "a",
                "g",
                BenchmarkStatus::Passed,
                Some(metrics(100.0)),
                None,
            ),
            result(
                "vec![1,2]",
                "g",
                BenchmarkStatus::Failed,
                None,
                Some(failure("panic", "say \"hi\"")),
            ),
        ]);
        let failed_row = format!(
            "\"vec![1,2]\",\"vec![1,2]\",g,failed{},panic,\"say \"\"hi\"\"\"",
            ",".repeat(20)
        );
        let expected = format!(
            "{}\n{}\n{}\n",
            "id,name,group,status,mean_ns,median_ns,std_dev_ns,min_ns,max_ns,p50_ns,p95_ns,p99_ns,samples,alloc_bytes,alloc_count,mean_cycles,median_cycles,cycles_per_ns,p90_ns,p999_ns,ci_lower_ns,ci_upper_ns,trimmed_mean_ns,throughput_ops_sec,failure_kind,failure_message",
            "a,a,g,passed,100.00,98.00,2.00,90.00,110.00,98.00,107.00,109.00,10,64,2,300,300,3.00,105.00,110.00,97.00,103.00,99.00,2000.50,,",
            failed_row,
        );
        assert_eq!(generate_csv_report(&rep), expected);
    }
}
