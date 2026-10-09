//! Embedded CSS for the HTML report.

use super::options::HtmlTheme;

/// CSS color variables for `theme`
pub(crate) fn theme_vars(theme: HtmlTheme) -> &'static str {
    match theme {
        HtmlTheme::Light => {
            r#"
    :root {
      --bg-primary: #ffffff;
      --bg-secondary: #f5f5f5;
      --text-primary: #1f2933;
      --text-muted: #616e7c;
      --border: #e5e7eb;
      --accent: #3b82f6;
      --target: #dc2626;
      --passed-bg: #dcfce7; --passed-fg: #166534;
      --failed-bg: #fee2e2; --failed-fg: #991b1b;
      --crashed-bg: #fef3c7; --crashed-fg: #92400e;
      --skipped-bg: #e5e7eb; --skipped-fg: #374151;
    }
"#
        }
        HtmlTheme::Dark => {
            r#"
    :root {
      --bg-primary: #111827;
      --bg-secondary: #1f2937;
      --text-primary: #f3f4f6;
      --text-muted: #9ca3af;
      --border: #374151;
      --accent: #60a5fa;
      --target: #f87171;
      --passed-bg: #14532d; --passed-fg: #bbf7d0;
      --failed-bg: #7f1d1d; --failed-fg: #fecaca;
      --crashed-bg: #78350f; --crashed-fg: #fde68a;
      --skipped-bg: #374151; --skipped-fg: #e5e7eb;
    }
"#
        }
    }
}

/// Layout and component CSS. It reads only the theme variables.
pub(crate) const BASE_CSS: &str = r#"
    body { font-family: system-ui, sans-serif; margin: 0; padding: 20px; background: var(--bg-primary); color: var(--text-primary); }
    header { margin-bottom: 20px; }
    h1 { margin: 0 0 10px 0; }
    .description { margin: 0 0 10px 0; }
    .meta { color: var(--text-muted); font-size: 14px; }
    .meta span { margin-right: 20px; }
    .stats-grid { display: flex; flex-wrap: wrap; gap: 20px; margin: 20px 0; }
    .stat { padding: 20px; border-radius: 8px; font-weight: bold; }
    .stat.passed, .status-passed { background: var(--passed-bg); color: var(--passed-fg); }
    .stat.failed, .status-failed { background: var(--failed-bg); color: var(--failed-fg); }
    .stat.crashed, .status-crashed { background: var(--crashed-bg); color: var(--crashed-fg); }
    .stat.skipped, .status-skipped { background: var(--skipped-bg); color: var(--skipped-fg); }
    .group-desc, .group-tags { color: var(--text-muted); margin: 4px 0; }
    .results-table { width: 100%; border-collapse: collapse; margin-bottom: 24px; }
    .results-table th, .results-table td { padding: 10px; text-align: left; border-bottom: 1px solid var(--border); }
    .results-table th { background: var(--bg-secondary); }
    .table-wrap { overflow-x: auto; }
    .chart-grid { display: grid; gap: 20px; }
    .chart { margin: 0; padding: 12px; background: var(--bg-secondary); border-radius: 8px; min-width: 0; }
    .chart figcaption { font-weight: bold; margin-bottom: 8px; }
    .chart-svg { max-width: 100%; height: auto; display: block; }
    .chart-svg .axis { stroke: var(--text-muted); }
    .chart-svg .grid { stroke: var(--border); }
    .chart-svg .tick, .chart-svg .label, .chart-svg .value { fill: var(--text-primary); font-size: 12px; }
    .chart-svg .bar { fill: var(--accent); }
    .chart-svg .line { fill: none; stroke: var(--accent); stroke-width: 2; }
    .chart-svg .point { fill: var(--accent); }
    .chart-svg .target { stroke: var(--target); stroke-dasharray: 6 4; stroke-width: 2; }
    .chart-svg .target-label { fill: var(--target); font-size: 12px; }
    footer { margin-top: 40px; color: var(--text-muted); font-size: 12px; }
"#;
