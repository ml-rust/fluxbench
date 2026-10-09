//! Inline SVG bar and line charts of one metric per benchmark.

use super::escape::escape;
use super::options::HtmlChart;
use crate::Report;
use crate::metric::{format_metric, metric_value};
use fluxbench_core::ChartType;
use std::fmt::Write;

const MARGIN_LEFT: f64 = 90.0;
const MARGIN_RIGHT: f64 = 20.0;
const MARGIN_TOP: f64 = 20.0;
const MARGIN_BOTTOM: f64 = 70.0;
const Y_TICKS: usize = 4;

/// One plotted benchmark: its ID and value, or `None` when the report has no value
struct Point<'a> {
    id: &'a str,
    value: Option<f64>,
}

/// Render `chart` as an SVG element sized `width` x `height` pixels.
pub(crate) fn render_chart(chart: &HtmlChart, report: &Report, width: u32, height: u32) -> String {
    let points: Vec<Point> = chart
        .benchmarks
        .iter()
        .map(|id| Point {
            id,
            value: report
                .results
                .iter()
                .find(|r| &r.id == id)
                .and_then(|r| r.metrics.as_ref())
                .and_then(|m| metric_value(m, &chart.metric)),
        })
        .collect();

    let (w, h) = (f64::from(width.max(1)), f64::from(height.max(1)));
    let plot_w = (w - MARGIN_LEFT - MARGIN_RIGHT).max(1.0);
    let plot_h = (h - MARGIN_TOP - MARGIN_BOTTOM).max(1.0);

    let values = points.iter().filter_map(|p| p.value).chain(chart.target);
    let (min, max) = values.fold((0.0f64, 0.0f64), |(lo, hi), v| (lo.min(v), hi.max(v)));
    let max = if max <= min {
        min + 1.0
    } else {
        max + (max - min) * 0.1
    };
    let y_of = |v: f64| MARGIN_TOP + (max - v) / (max - min) * plot_h;
    let band = plot_w / points.len().max(1) as f64;
    let x_of = |i: usize| MARGIN_LEFT + band * (i as f64 + 0.5);

    let mut svg = String::new();
    let _ = write!(
        svg,
        "<svg class=\"chart-svg\" role=\"img\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\" xmlns=\"http://www.w3.org/2000/svg\">\n<title>{}</title>\n",
        escape(&chart.title)
    );

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
            escape(&format_metric(&chart.metric, v))
        );
    }

    match chart.kind {
        ChartType::Bar => draw_bars(&mut svg, &points, &chart.metric, band, &x_of, &y_of),
        ChartType::Line => draw_line(&mut svg, &points, &chart.metric, &x_of, &y_of),
    }

    // Benchmark labels under the plot
    let label_y = MARGIN_TOP + plot_h + 18.0;
    for (i, point) in points.iter().enumerate() {
        let x = x_of(i);
        let note = if point.value.is_none() {
            " (no data)"
        } else {
            ""
        };
        let _ = writeln!(
            svg,
            "<text class=\"label\" x=\"{x:.1}\" y=\"{label_y:.1}\" text-anchor=\"end\" transform=\"rotate(-30 {x:.1} {label_y:.1})\">{}{note}</text>",
            escape(point.id)
        );
    }

    if let Some(target) = chart.target {
        let y = y_of(target);
        let _ = writeln!(
            svg,
            "<line class=\"target\" x1=\"{MARGIN_LEFT}\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\"/><text class=\"target-label\" x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\">target {}</text>",
            MARGIN_LEFT + plot_w,
            MARGIN_LEFT + plot_w,
            y - 4.0,
            escape(&format_metric(&chart.metric, target))
        );
    }

    svg.push_str("</svg>\n");
    svg
}

fn draw_bars(
    svg: &mut String,
    points: &[Point],
    metric: &str,
    band: f64,
    x_of: &dyn Fn(usize) -> f64,
    y_of: &dyn Fn(f64) -> f64,
) {
    let bar_w = band * 0.7;
    // The value range always spans 0, so bars grow from the zero line.
    let base = y_of(0.0);
    for (i, point) in points.iter().enumerate() {
        let Some(value) = point.value else { continue };
        let y = y_of(value);
        let (top, height) = if y <= base {
            (y, base - y)
        } else {
            (base, y - base)
        };
        let label = escape(&format_metric(metric, value));
        let _ = writeln!(
            svg,
            "<rect class=\"bar\" x=\"{:.1}\" y=\"{top:.1}\" width=\"{bar_w:.1}\" height=\"{height:.1}\"><title>{}: {label}</title></rect><text class=\"value\" x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\">{label}</text>",
            x_of(i) - bar_w / 2.0,
            escape(point.id),
            x_of(i),
            top - 4.0
        );
    }
}

fn draw_line(
    svg: &mut String,
    points: &[Point],
    metric: &str,
    x_of: &dyn Fn(usize) -> f64,
    y_of: &dyn Fn(f64) -> f64,
) {
    fn flush(run: &mut Vec<String>, svg: &mut String) {
        if run.len() > 1 {
            let _ = writeln!(
                svg,
                "<polyline class=\"line\" points=\"{}\"/>",
                run.join(" ")
            );
        }
        run.clear();
    }

    // A missing value breaks the line into separate runs.
    let mut run: Vec<String> = Vec::new();
    for (i, point) in points.iter().enumerate() {
        let Some(value) = point.value else {
            flush(&mut run, svg);
            continue;
        };
        let (x, y) = (x_of(i), y_of(value));
        run.push(format!("{x:.1},{y:.1}"));
        let label = escape(&format_metric(metric, value));
        let _ = writeln!(
            svg,
            "<circle class=\"point\" cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"4\"><title>{}: {label}</title></circle><text class=\"value\" x=\"{x:.1}\" y=\"{:.1}\" text-anchor=\"middle\">{label}</text>",
            escape(point.id),
            y - 8.0
        );
    }
    flush(&mut run, svg);
}
