//! HTML report: one self-contained page with inline CSS and SVG charts.
//!
//! - `options`: title, theme, groups, and charts for the page
//! - `page`: page assembly
//! - `chart`: SVG bar and line charts
//! - `style`: embedded CSS
//! - `escape`: HTML escaping

mod chart;
mod escape;
mod options;
mod page;
mod style;

pub use options::{HtmlChart, HtmlGroup, HtmlOptions, HtmlTheme};
pub use page::{generate_html_report, generate_html_report_with};
