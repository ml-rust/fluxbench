//! Duration strings such as `3s`, `500ms`, or `2m`.

/// Parse a duration string to nanoseconds.
///
/// Units: `ns`, `us` or `µs`, `ms`, `s`, `m` or `min`. A bare number means seconds.
/// The value must be finite and >= 0.
pub(crate) fn parse_duration_ns(s: &str) -> anyhow::Result<u64> {
    let s = s.trim();
    if s.is_empty() {
        anyhow::bail!("empty duration");
    }

    // The number ends where the unit begins.
    let (num_part, unit_part) = s
        .char_indices()
        .find(|(_, c)| c.is_alphabetic())
        .map(|(i, _)| s.split_at(i))
        .unwrap_or((s, "s"));

    let value: f64 = num_part
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("`{}` is not a number", num_part.trim()))?;
    if !value.is_finite() || value < 0.0 {
        anyhow::bail!("the value must be finite and >= 0");
    }

    let multiplier: f64 = match unit_part.to_lowercase().as_str() {
        "ns" => 1.0,
        "us" | "µs" => 1_000.0,
        "ms" => 1_000_000.0,
        "s" => 1_000_000_000.0,
        "m" | "min" => 60_000_000_000.0,
        other => anyhow::bail!("unknown unit `{other}`"),
    };

    Ok((value * multiplier).round() as u64)
}

/// Parse a CLI duration such as `5s` or `500ms` into nanoseconds.
///
/// A bare number means seconds.
pub(crate) fn parse_duration_arg(s: &str) -> Result<u64, String> {
    parse_duration_ns(s)
        .map_err(|e| format!("invalid duration `{s}`: {e}. Use a value such as `5s` or `500ms`."))
}

/// Parse the flux.toml duration at `key`. The error names the key and value.
pub(crate) fn parse_config_duration(key: &str, value: &str) -> anyhow::Result<u64> {
    parse_duration_ns(value).map_err(|e| {
        anyhow::anyhow!(
            "{key} = \"{value}\" is not a duration: {e}. Use a value such as \"5s\" or \"500ms\"."
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_units() {
        assert_eq!(parse_duration_ns("3s").unwrap(), 3_000_000_000);
        assert_eq!(parse_duration_ns("500ms").unwrap(), 500_000_000);
        assert_eq!(parse_duration_ns("100us").unwrap(), 100_000);
        assert_eq!(parse_duration_ns("1000ns").unwrap(), 1000);
        assert_eq!(parse_duration_ns("2m").unwrap(), 120_000_000_000);
        assert_eq!(parse_duration_ns("1.5s").unwrap(), 1_500_000_000);
    }

    #[test]
    fn rejects_negative_and_non_finite() {
        assert!(parse_duration_ns("-3s").is_err());
        assert!(parse_duration_ns("inf").is_err());
        assert!(parse_duration_ns("NaN").is_err());
        assert!(parse_duration_ns("3h").is_err());
    }

    #[test]
    fn cli_duration_arg() {
        assert_eq!(parse_duration_arg("3"), Ok(3_000_000_000));
        assert_eq!(parse_duration_arg("500ms"), Ok(500_000_000));
        assert!(parse_duration_arg("fast").is_err());
    }

    #[test]
    fn config_duration_error_names_key() {
        let err = parse_config_duration("runner.timeout", "soon")
            .unwrap_err()
            .to_string();
        assert!(err.contains("runner.timeout = \"soon\""), "{err}");
    }
}
