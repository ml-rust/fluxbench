//! HTML report layout from `#[flux::report]`, `#[flux::chart]`, `#[flux::group]`, and `[visuals]`.
//!
//! A bad definition is a startup error: two reports, an unknown chart metric or benchmark,
//! or a chart cell outside the layout or shared with another chart.

use crate::groups::GroupTree;
use crate::{FluxConfig, Theme};
use fluxbench_core::{BenchmarkDef, ChartDef, ReportDef};
use fluxbench_report::{CHART_METRICS, HtmlChart, HtmlGroup, HtmlOptions, HtmlTheme};
use std::collections::{BTreeSet, HashMap};

/// HTML options from the registered definitions and flux.toml `[visuals]`.
pub(crate) fn html_options(
    config: &FluxConfig,
    groups: &GroupTree,
    all: &[&'static BenchmarkDef],
) -> anyhow::Result<HtmlOptions> {
    let reports: Vec<&ReportDef> = inventory::iter::<ReportDef>.into_iter().collect();
    let charts: Vec<&ChartDef> = inventory::iter::<ChartDef>.into_iter().collect();
    let known: BTreeSet<&str> = all.iter().map(|b| b.id).collect();
    build_html_options(config, &reports, &charts, groups, &known)
}

fn build_html_options(
    config: &FluxConfig,
    reports: &[&ReportDef],
    charts: &[&ChartDef],
    groups: &GroupTree,
    known: &BTreeSet<&str>,
) -> anyhow::Result<HtmlOptions> {
    let mut options = HtmlOptions {
        theme: match config.visuals.theme {
            Theme::Light => HtmlTheme::Light,
            Theme::Dark => HtmlTheme::Dark,
        },
        chart_width: config.visuals.width,
        chart_height: config.visuals.height,
        ..HtmlOptions::default()
    };

    match reports {
        [] => {}
        [report] => {
            options.title = report.title.to_string();
            options.description = report.description.to_string();
            options.grid = report.layout;
        }
        many => {
            let titles: Vec<&str> = many.iter().map(|r| r.title).collect();
            anyhow::bail!(
                "found {} #[flux::report] definitions: {}. Keep one.",
                many.len(),
                titles.join(", ")
            );
        }
    }

    let mut cells: HashMap<(u32, u32), &str> = HashMap::new();
    for chart in charts {
        check_chart(chart, known, options.grid)?;
        if let Some(cell) = chart.position {
            if let Some(other) = cells.insert(cell, chart.title) {
                anyhow::bail!(
                    "#[flux::chart] `{}` and `{other}` share grid cell (row {}, col {}). Give each chart its own cell.",
                    chart.title,
                    cell.0,
                    cell.1
                );
            }
        }
    }

    let mut ordered: Vec<&&ChartDef> = charts.iter().collect();
    ordered.sort_by_key(|c| (c.position.unwrap_or((u32::MAX, u32::MAX)), c.title));
    options.charts = ordered
        .into_iter()
        .map(|c| HtmlChart {
            title: c.title.to_string(),
            kind: c.chart_type,
            metric: c.metric.to_string(),
            benchmarks: c.benchmarks.iter().map(|b| b.to_string()).collect(),
            target: c.target_line,
            position: c.position,
        })
        .collect();

    let mut group_defs: Vec<HtmlGroup> = groups
        .defs()
        .map(|g| HtmlGroup {
            id: g.id.to_string(),
            description: g.description.to_string(),
            tags: g.tags.iter().map(|t| t.to_string()).collect(),
            parent: g.parent.map(str::to_string),
        })
        .collect();
    group_defs.sort_by(|a, b| a.id.cmp(&b.id));
    options.groups = group_defs;

    Ok(options)
}

fn check_chart(chart: &ChartDef, known: &BTreeSet<&str>, grid: (u32, u32)) -> anyhow::Result<()> {
    if !CHART_METRICS.contains(&chart.metric) {
        anyhow::bail!(
            "#[flux::chart] `{}` has unknown metric `{}`. Use one of: {}.",
            chart.title,
            chart.metric,
            CHART_METRICS.join(", ")
        );
    }
    if let Some(missing) = chart.benchmarks.iter().find(|id| !known.contains(**id)) {
        anyhow::bail!(
            "#[flux::chart] `{}` names unknown benchmark `{missing}`. Use a registered benchmark ID; `list` prints them.",
            chart.title
        );
    }
    if let Some((row, col)) = chart.position {
        if row > grid.0 || col > grid.1 {
            anyhow::bail!(
                "#[flux::chart] `{}` sits at row {row}, col {col}, outside the {}x{} report layout. Move it or enlarge #[flux::report(layout)].",
                chart.title,
                grid.0,
                grid.1
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxbench_core::ChartType;

    fn chart(title: &'static str, metric: &'static str, position: Option<(u32, u32)>) -> ChartDef {
        ChartDef {
            title,
            chart_type: ChartType::Bar,
            metric,
            benchmarks: &["a"],
            target_line: None,
            position,
        }
    }

    fn report(title: &'static str) -> ReportDef {
        ReportDef {
            title,
            description: "desc",
            layout: (1, 2),
        }
    }

    fn build(reports: &[&ReportDef], charts: &[&ChartDef]) -> anyhow::Result<HtmlOptions> {
        let known: BTreeSet<&str> = ["a"].into_iter().collect();
        build_html_options(
            &FluxConfig::default(),
            reports,
            charts,
            &GroupTree::default(),
            &known,
        )
    }

    #[test]
    fn report_sets_title_description_and_grid() {
        let r = report("Nightly");
        let options = build(&[&r], &[]).unwrap();
        assert_eq!(options.title, "Nightly");
        assert_eq!(options.description, "desc");
        assert_eq!(options.grid, (1, 2));
        assert_eq!(options.chart_width, 1280);
    }

    #[test]
    fn two_reports_are_an_error() {
        let (a, b) = (report("A"), report("B"));
        let err = build(&[&a, &b], &[]).unwrap_err().to_string();
        assert!(err.contains("2 #[flux::report] definitions: A, B"), "{err}");
    }

    #[test]
    fn chart_checks() {
        let r = report("R");
        let bad_metric = chart("m", "fastest", None);
        assert!(
            build(&[&r], &[&bad_metric])
                .unwrap_err()
                .to_string()
                .contains("unknown metric `fastest`")
        );

        let mut unknown_bench = chart("u", "mean", None);
        unknown_bench.benchmarks = &["nope"];
        assert!(
            build(&[&r], &[&unknown_bench])
                .unwrap_err()
                .to_string()
                .contains("unknown benchmark `nope`")
        );

        let outside = chart("o", "mean", Some((2, 1)));
        assert!(
            build(&[&r], &[&outside])
                .unwrap_err()
                .to_string()
                .contains("outside the 1x2")
        );

        let (x, y) = (
            chart("x", "mean", Some((1, 1))),
            chart("y", "mean", Some((1, 1))),
        );
        assert!(
            build(&[&r], &[&x, &y])
                .unwrap_err()
                .to_string()
                .contains("share grid cell")
        );
    }

    #[test]
    fn charts_sort_by_cell_then_title() {
        let r = report("R");
        let (late, early, free) = (
            chart("late", "p99", Some((1, 2))),
            chart("early", "mean", Some((1, 1))),
            chart("free", "mean", None),
        );
        let options = build(&[&r], &[&free, &late, &early]).unwrap();
        let titles: Vec<&str> = options.charts.iter().map(|c| c.title.as_str()).collect();
        assert_eq!(titles, ["early", "late", "free"]);
    }

    #[test]
    fn visuals_theme_and_size_apply() {
        let mut config = FluxConfig::default();
        config.visuals.theme = Theme::Dark;
        config.visuals.width = 640;
        config.visuals.height = 360;
        let options =
            build_html_options(&config, &[], &[], &GroupTree::default(), &BTreeSet::new()).unwrap();
        assert_eq!(options.theme, HtmlTheme::Dark);
        assert_eq!((options.chart_width, options.chart_height), (640, 360));
    }
}
