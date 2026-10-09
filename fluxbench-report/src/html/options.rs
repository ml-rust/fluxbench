//! Page structure for the HTML report: title, theme, groups, and charts.

use fluxbench_core::ChartType;

/// Page color theme
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HtmlTheme {
    /// Dark text on a light background
    #[default]
    Light,
    /// Light text on a dark background
    Dark,
}

/// Group shown as its own results section
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HtmlGroup {
    /// Group ID, as in `BenchmarkReportResult::group`
    pub id: String,
    /// Description under the section heading
    pub description: String,
    /// Tags listed under the section heading
    pub tags: Vec<String>,
    /// Parent group ID. The section heading shows the full path.
    pub parent: Option<String>,
}

/// Chart drawn as inline SVG
#[derive(Debug, Clone, PartialEq)]
pub struct HtmlChart {
    /// Chart title
    pub title: String,
    /// Bar or line
    pub kind: ChartType,
    /// Metric name from [`crate::CHART_METRICS`]
    pub metric: String,
    /// Benchmark IDs, in plot order
    pub benchmarks: Vec<String>,
    /// Optional horizontal target line, in the metric's unit
    pub target: Option<f64>,
    /// Optional 1-based grid cell (row, col)
    pub position: Option<(u32, u32)>,
}

/// Options for [`crate::generate_html_report_with`]
#[derive(Debug, Clone, PartialEq)]
pub struct HtmlOptions {
    /// Page heading and `<title>`
    pub title: String,
    /// Paragraph under the heading. Empty means none.
    pub description: String,
    /// Color theme
    pub theme: HtmlTheme,
    /// Chart canvas width in pixels. A narrow page shrinks the chart and keeps its aspect ratio.
    pub chart_width: u32,
    /// Chart canvas height in pixels
    pub chart_height: u32,
    /// Chart grid (rows, cols)
    pub grid: (u32, u32),
    /// Group descriptions. Result groups without an entry still get a section.
    pub groups: Vec<HtmlGroup>,
    /// Charts, in display order
    pub charts: Vec<HtmlChart>,
}

impl Default for HtmlOptions {
    fn default() -> Self {
        Self {
            title: "FluxBench Report".to_string(),
            description: String::new(),
            theme: HtmlTheme::Light,
            chart_width: 1280,
            chart_height: 720,
            grid: (2, 2),
            groups: Vec::new(),
            charts: Vec::new(),
        }
    }
}
