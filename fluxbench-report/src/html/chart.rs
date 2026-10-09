//! Inline SVG bar and line charts of one metric per benchmark or per comparison series.

use super::escape::escape;
use super::options::HtmlChart;
use crate::metric::{format_metric, metric_value};
use crate::{ComparisonSeries, Report};
use fluxbench_core::ChartType;
use std::fmt::Write;

const MARGIN_LEFT: f64 = 90.0;
const MARGIN_RIGHT: f64 = 20.0;
const MARGIN_TOP: f64 = 30.0;
const MARGIN_BOTTOM: f64 = 70.0;
const Y_TICKS: usize = 4;
/// Distinct series colors. The CSS defines `s1` to `s5` after the accent color.
const SERIES_COLORS: usize = 6;
const LEGEND_STEP: f64 = 150.0;

/// One plotted series: a name and one value per x label, `None` when there is no value
struct Series<'a> {
    name: &'a str,
    values: Vec<Option<f64>>,
}

/// Data and style for one SVG chart
struct Plot<'a> {
    title: &'a str,
    metric: &'a str,
    kind: ChartType,
    /// One label per x position
    labels: Vec<&'a str>,
    series: Vec<Series<'a>>,
    target: Option<f64>,
}

/// Render `chart` as an SVG element sized `width` x `height` pixels.
pub(crate) fn render_chart(chart: &HtmlChart, report: &Report, width: u32, height: u32) -> String {
    let values = chart
        .benchmarks
        .iter()
        .map(|id| {
            report
                .results
                .iter()
                .find(|r| &r.id == id)
                .and_then(|r| r.metrics.as_ref())
                .and_then(|m| metric_value(m, &chart.metric))
        })
        .collect();
    let plot = Plot {
        title: &chart.title,
        metric: &chart.metric,
        kind: chart.kind,
        labels: chart.benchmarks.iter().map(String::as_str).collect(),
        series: vec![Series {
            name: &chart.metric,
            values,
        }],
        target: chart.target,
    };
    render_plot(&plot, width, height)
}

/// Render a `#[compare]` group as an SVG element sized `width` x `height` pixels.
///
/// Numeric x values draw a line per series. Other x values draw grouped bars.
/// A value of 0 means no data, as in the terminal table.
pub(crate) fn render_series_chart(series: &ComparisonSeries, width: u32, height: u32) -> String {
    let numeric = !series.x_values.is_empty()
        && series
            .x_values
            .iter()
            .all(|x| x.trim().parse::<f64>().is_ok());
    let plot = Plot {
        title: &series.title,
        metric: &series.metric,
        kind: if numeric {
            ChartType::Line
        } else {
            ChartType::Bar
        },
        labels: series.x_values.iter().map(String::as_str).collect(),
        series: series
            .series_names
            .iter()
            .enumerate()
            .map(|(i, name)| Series {
                name,
                values: (0..series.x_values.len())
                    .map(|x| {
                        series
                            .series_data
                            .get(i)
                            .and_then(|d| d.get(x))
                            .copied()
                            .filter(|v| *v != 0.0)
                    })
                    .collect(),
            })
            .collect(),
        target: None,
    };
    render_plot(&plot, width, height)
}

/// CSS class for series `index`. Series 0 keeps the base class.
fn series_class(base: &str, index: usize) -> String {
    match index % SERIES_COLORS {
        0 => base.to_string(),
        n => format!("{base} s{n}"),
    }
}

/// Tooltip text for one value. A multi-series plot names the series.
fn tooltip(plot: &Plot, series: usize, x: usize) -> String {
    let label = plot.labels.get(x).copied().unwrap_or("");
    match plot.series.get(series) {
        Some(s) if plot.series.len() > 1 => format!("{} @ {label}", s.name),
        _ => label.to_string(),
    }
}

fn render_plot(plot: &Plot, width: u32, height: u32) -> String {
    let (w, h) = (f64::from(width.max(1)), f64::from(height.max(1)));
    let plot_w = (w - MARGIN_LEFT - MARGIN_RIGHT).max(1.0);
    let plot_h = (h - MARGIN_TOP - MARGIN_BOTTOM).max(1.0);

    let values = plot
        .series
        .iter()
        .flat_map(|s| s.values.iter().flatten().copied())
        .chain(plot.target);
    let (min, max) = values.fold((0.0f64, 0.0f64), |(lo, hi), v| (lo.min(v), hi.max(v)));
    let max = if max <= min {
        min + 1.0
    } else {
        max + (max - min) * 0.1
    };
    let y_of = |v: f64| MARGIN_TOP + (max - v) / (max - min) * plot_h;
    let band = plot_w / plot.labels.len().max(1) as f64;
    let x_of = |i: usize| MARGIN_LEFT + band * (i as f64 + 0.5);

    let mut svg = String::new();
    let _ = write!(
        svg,
        "<svg class=\"chart-svg\" role=\"img\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\" xmlns=\"http://www.w3.org/2000/svg\">\n<title>{}</title>\n",
        escape(plot.title)
    );

    if plot.series.len() > 1 {
        for (i, series) in plot.series.iter().enumerate() {
            let x = MARGIN_LEFT + LEGEND_STEP * i as f64;
            let _ = writeln!(
                svg,
                "<rect class=\"{}\" x=\"{x:.1}\" y=\"6\" width=\"10\" height=\"10\"/><text class=\"legend\" x=\"{:.1}\" y=\"15\">{}</text>",
                series_class("legend-swatch bar", i),
                x + 14.0,
                escape(series.name)
            );
        }
    }

    // Y axis with ticks
    let _ = writeln!(
        svg,
        "<line class=\"axis\" x1=\"{MARGIN_LEFT}\" y1=\"{MARGIN_TOP}\" x2=\"{MARGIN_LEFT}\" y2=\"{:.1}\"/>",
        MARGIN_TOP + plot_h
    );
    for tick in 0..=Y_TICKS {
        let v = min + (max - min) * tick as f64 / Y_TICKS as f64;
        let y = y_of(v);
        let _ = writeln!(
            svg,
            "<line class=\"grid\" x1=\"{MARGIN_LEFT}\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\"/><text class=\"tick\" x=\"{:.1}\" y=\"{y:.1}\" text-anchor=\"end\" dominant-baseline=\"middle\">{}</text>",
            MARGIN_LEFT + plot_w,
            MARGIN_LEFT - 6.0,
            escape(&format_metric(plot.metric, v))
        );
    }

    match plot.kind {
        ChartType::Bar => draw_bars(&mut svg, plot, band, &x_of, &y_of),
        ChartType::Line => draw_line(&mut svg, plot, &x_of, &y_of),
    }

    // X labels under the plot
    let label_y = MARGIN_TOP + plot_h + 18.0;
    for (i, label) in plot.labels.iter().enumerate() {
        let x = x_of(i);
        let has_data = plot
            .series
            .iter()
            .any(|s| s.values.get(i).copied().flatten().is_some());
        let note = if has_data { "" } else { " (no data)" };
        let _ = writeln!(
            svg,
            "<text class=\"label\" x=\"{x:.1}\" y=\"{label_y:.1}\" text-anchor=\"end\" transform=\"rotate(-30 {x:.1} {label_y:.1})\">{}{note}</text>",
            escape(label)
        );
    }

    if let Some(target) = plot.target {
        let y = y_of(target);
        let _ = writeln!(
            svg,
            "<line class=\"target\" x1=\"{MARGIN_LEFT}\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\"/><text class=\"target-label\" x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\">target {}</text>",
            MARGIN_LEFT + plot_w,
            MARGIN_LEFT + plot_w,
            y - 4.0,
            escape(&format_metric(plot.metric, target))
        );
    }

    svg.push_str("</svg>\n");
    svg
}

fn draw_bars(
    svg: &mut String,
    plot: &Plot,
    band: f64,
    x_of: &dyn Fn(usize) -> f64,
    y_of: &dyn Fn(f64) -> f64,
) {
    let group_w = band * 0.7;
    let bar_w = group_w / plot.series.len().max(1) as f64;
    let single = plot.series.len() == 1;
    // The value range always spans 0, so bars grow from the zero line.
    let base = y_of(0.0);
    for (si, series) in plot.series.iter().enumerate() {
        for (i, value) in series.values.iter().enumerate() {
            let Some(value) = *value else { continue };
            let y = y_of(value);
            let (top, height) = if y <= base {
                (y, base - y)
            } else {
                (base, y - base)
            };
            let left = x_of(i) - group_w / 2.0 + bar_w * si as f64;
            let label = escape(&format_metric(plot.metric, value));
            let _ = write!(
                svg,
                "<rect class=\"{}\" x=\"{left:.1}\" y=\"{top:.1}\" width=\"{bar_w:.1}\" height=\"{height:.1}\"><title>{}: {label}</title></rect>",
                series_class("bar", si),
                escape(&tooltip(plot, si, i)),
            );
            if single {
                let _ = write!(
                    svg,
                    "<text class=\"value\" x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\">{label}</text>",
                    x_of(i),
                    top - 4.0
                );
            }
            svg.push('\n');
        }
    }
}

fn draw_line(
    svg: &mut String,
    plot: &Plot,
    x_of: &dyn Fn(usize) -> f64,
    y_of: &dyn Fn(f64) -> f64,
) {
    fn flush(run: &mut Vec<String>, class: &str, svg: &mut String) {
        if run.len() > 1 {
            let _ = writeln!(
                svg,
                "<polyline class=\"{class}\" points=\"{}\"/>",
                run.join(" ")
            );
        }
        run.clear();
    }

    let single = plot.series.len() == 1;
    for (si, series) in plot.series.iter().enumerate() {
        let line_class = series_class("line", si);
        // A missing value breaks the line into separate runs.
        let mut run: Vec<String> = Vec::new();
        for (i, value) in series.values.iter().enumerate() {
            let Some(value) = *value else {
                flush(&mut run, &line_class, svg);
                continue;
            };
            let (x, y) = (x_of(i), y_of(value));
            run.push(format!("{x:.1},{y:.1}"));
            let label = escape(&format_metric(plot.metric, value));
            let _ = write!(
                svg,
                "<circle class=\"{}\" cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"4\"><title>{}: {label}</title></circle>",
                series_class("point", si),
                escape(&tooltip(plot, si, i)),
            );
            if single {
                let _ = write!(
                    svg,
                    "<text class=\"value\" x=\"{x:.1}\" y=\"{:.1}\" text-anchor=\"middle\">{label}</text>",
                    y - 8.0
                );
            }
            svg.push('\n');
        }
        flush(&mut run, &line_class, svg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn series(x_values: &[&str], names: &[&str], data: Vec<Vec<f64>>) -> ComparisonSeries {
        ComparisonSeries {
            group: "g".to_string(),
            title: "Scaling".to_string(),
            x_values: x_values.iter().map(|s| s.to_string()).collect(),
            series_names: names.iter().map(|s| s.to_string()).collect(),
            series_data: data,
            metric: "mean".to_string(),
        }
    }

    #[test]
    fn numeric_x_draws_one_point_per_x_and_series() {
        let s = series(
            &["1", "10", "100"],
            &["fast", "slow"],
            vec![vec![10.0, 20.0, 30.0], vec![15.0, 40.0, 90.0]],
        );
        let svg = render_series_chart(&s, 640, 360);
        assert!(svg.starts_with("<svg "));
        assert_eq!(svg.matches("<circle").count(), 6);
        assert_eq!(svg.matches("<polyline").count(), 2);
        assert!(svg.contains("fast @ 10"));
        assert!(svg.contains(">slow</text>"));
    }

    #[test]
    fn single_series_numeric_x_has_one_point_per_x() {
        let s = series(&["1", "10", "100"], &["only"], vec![vec![1.0, 2.0, 3.0]]);
        let svg = render_series_chart(&s, 640, 360);
        assert_eq!(svg.matches("<circle").count(), 3);
        assert_eq!(svg.matches("<polyline").count(), 1);
    }

    #[test]
    fn text_x_draws_grouped_bars() {
        let s = series(
            &["small", "large"],
            &["a", "b"],
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        );
        let svg = render_series_chart(&s, 640, 360);
        assert_eq!(svg.matches("<rect class=\"bar").count(), 4);
        assert!(!svg.contains("<polyline"));
    }

    #[test]
    fn series_text_is_escaped() {
        let evil = "<img src=x onerror=\"a\">&'";
        let mut s = series(&[evil], &[evil, "ok"], vec![vec![1.0], vec![2.0]]);
        s.title = evil.to_string();
        let svg = render_series_chart(&s, 640, 360);
        assert!(!svg.contains("<img"));
        assert!(!svg.contains("onerror=\"a\""));
        assert!(svg.contains("&lt;img src=x onerror=&quot;a&quot;&gt;&amp;&#39;"));
    }

    #[test]
    fn ragged_series_data_does_not_panic() {
        let s = series(&["1", "2"], &["a", "b"], vec![vec![1.0]]);
        let svg = render_series_chart(&s, 640, 360);
        assert_eq!(svg.matches("<circle").count(), 1);
    }
}
