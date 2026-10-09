//! Throughput rate formatting: bytes per second in binary units, elements per second in SI units.

use crate::BenchmarkMetrics;

/// Scale bytes per second to B/s, KiB/s, MiB/s, GiB/s, or TiB/s.
pub fn scale_byte_rate(bytes_per_sec: f64) -> (f64, &'static str) {
    const UNITS: [&str; 5] = ["B/s", "KiB/s", "MiB/s", "GiB/s", "TiB/s"];
    let mut value = bytes_per_sec;
    let mut unit = 0;
    while value.abs() >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    (value, UNITS[unit])
}

/// Scale elements per second to elem/s, Kelem/s, Melem/s, or Gelem/s.
pub fn scale_element_rate(per_sec: f64) -> (f64, &'static str) {
    let abs = per_sec.abs();
    if abs >= 1e9 {
        (per_sec / 1e9, "Gelem/s")
    } else if abs >= 1e6 {
        (per_sec / 1e6, "Melem/s")
    } else if abs >= 1e3 {
        (per_sec / 1e3, "Kelem/s")
    } else {
        (per_sec, "elem/s")
    }
}

/// Format bytes per second, e.g. `1.50 GiB/s`.
pub fn format_byte_rate(bytes_per_sec: f64) -> String {
    let (value, unit) = scale_byte_rate(bytes_per_sec);
    format!("{value:.2} {unit}")
}

/// Format elements per second, e.g. `3.20 Melem/s`.
pub fn format_element_rate(per_sec: f64) -> String {
    let (value, unit) = scale_element_rate(per_sec);
    format!("{value:.2} {unit}")
}

/// The byte rate, else the element rate, formatted. `None` when neither is set.
pub fn format_work_rate(metrics: &BenchmarkMetrics) -> Option<String> {
    metrics
        .throughput_bytes_per_sec
        .map(format_byte_rate)
        .or_else(|| metrics.throughput_elements_per_sec.map(format_element_rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_rates_use_binary_units() {
        assert_eq!(format_byte_rate(512.0), "512.00 B/s");
        assert_eq!(format_byte_rate(1024.0), "1.00 KiB/s");
        assert_eq!(format_byte_rate(1.5 * 1024.0 * 1024.0), "1.50 MiB/s");
        assert_eq!(
            format_byte_rate(3.0 * 1024.0 * 1024.0 * 1024.0),
            "3.00 GiB/s"
        );
        assert_eq!(format_byte_rate(2048.0_f64.powi(4)), "16.00 TiB/s");
    }

    #[test]
    fn element_rates_use_si_units() {
        assert_eq!(format_element_rate(12.0), "12.00 elem/s");
        assert_eq!(format_element_rate(2_500.0), "2.50 Kelem/s");
        assert_eq!(format_element_rate(3_200_000.0), "3.20 Melem/s");
        assert_eq!(format_element_rate(4e9), "4.00 Gelem/s");
    }

    #[test]
    fn work_rate_prefers_bytes() {
        let mut m = crate::test_fixtures::metrics(100.0);
        assert_eq!(format_work_rate(&m), None);
        m.throughput_elements_per_sec = Some(2_000.0);
        assert_eq!(format_work_rate(&m).as_deref(), Some("2.00 Kelem/s"));
        m.throughput_bytes_per_sec = Some(1024.0);
        assert_eq!(format_work_rate(&m).as_deref(), Some("1.00 KiB/s"));
    }
}
