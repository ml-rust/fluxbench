//! Report format choice shared by `--format` and flux.toml `output.format`.

use fluxbench_report::OutputFormat;
use serde::{Deserialize, Serialize};

/// Report format for `--format` and `output.format`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ReportFormat {
    /// Terminal text
    #[default]
    #[value(alias = "text")]
    #[serde(alias = "text")]
    Human,
    /// JSON with full schema
    Json,
    /// Markdown for `$GITHUB_STEP_SUMMARY`
    #[value(alias = "github")]
    #[serde(alias = "github")]
    GithubSummary,
    /// github-action-benchmark JSON (customSmallerIsBetter)
    #[value(alias = "github-action-benchmark")]
    #[serde(alias = "github-action-benchmark")]
    GithubAction,
    /// CSV for spreadsheets
    Csv,
    /// Single-file HTML page
    Html,
}

impl From<ReportFormat> for OutputFormat {
    fn from(format: ReportFormat) -> Self {
        match format {
            ReportFormat::Human => OutputFormat::Human,
            ReportFormat::Json => OutputFormat::Json,
            ReportFormat::GithubSummary => OutputFormat::GithubSummary,
            ReportFormat::GithubAction => OutputFormat::GithubActionBenchmark,
            ReportFormat::Csv => OutputFormat::Csv,
            ReportFormat::Html => OutputFormat::Html,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cli;
    use clap::Parser;

    #[test]
    fn clap_accepts_names_and_aliases() {
        for (arg, expected) in [
            ("json", ReportFormat::Json),
            ("github", ReportFormat::GithubSummary),
            ("github-summary", ReportFormat::GithubSummary),
            ("github-action", ReportFormat::GithubAction),
            ("github-action-benchmark", ReportFormat::GithubAction),
            ("csv", ReportFormat::Csv),
            ("html", ReportFormat::Html),
            ("human", ReportFormat::Human),
            ("text", ReportFormat::Human),
        ] {
            let cli = Cli::try_parse_from(["bench", "--format", arg]).unwrap();
            assert_eq!(cli.format, Some(expected), "{arg}");
        }
    }

    #[test]
    fn clap_rejects_unknown_format() {
        let err = Cli::try_parse_from(["bench", "--format", "jsn"]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::InvalidValue);
        let text = err.to_string();
        assert!(text.contains("jsn"), "{text}");
        assert!(text.contains("json"), "{text}");
    }

    #[test]
    fn format_is_unset_without_flag() {
        assert_eq!(Cli::try_parse_from(["bench"]).unwrap().format, None);
    }

    #[test]
    fn toml_accepts_names_and_aliases() {
        #[derive(Deserialize)]
        struct Holder {
            format: ReportFormat,
        }
        let parse =
            |s: &str| toml::from_str::<Holder>(&format!("format = \"{s}\"")).map(|h| h.format);
        assert_eq!(parse("github").unwrap(), ReportFormat::GithubSummary);
        assert_eq!(parse("github-action").unwrap(), ReportFormat::GithubAction);
        assert_eq!(parse("csv").unwrap(), ReportFormat::Csv);
        assert!(parse("yaml").is_err());
    }
}
