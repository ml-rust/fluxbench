//! Duration strings in macro attributes, such as `"50us"` or `"1.5s"`.

/// Parse a duration string into nanoseconds.
///
/// Units: `ns`, `us` or `µs`, `ms`, `s`. The value must be finite and >= 0.
pub fn parse_duration(s: &str) -> Result<u64, String> {
    let trimmed = s.trim();
    let units: [(&str, f64); 5] = [
        ("ns", 1.0),
        ("us", 1_000.0),
        ("µs", 1_000.0),
        ("ms", 1_000_000.0),
        ("s", 1_000_000_000.0),
    ];
    let Some((number, scale)) = units
        .iter()
        .find_map(|(unit, scale)| trimmed.strip_suffix(unit).map(|n| (n.trim(), *scale)))
    else {
        return Err(invalid(s, "it has no unit"));
    };
    let value: f64 = number
        .parse()
        .map_err(|_| invalid(s, &format!("`{number}` is not a number")))?;
    if !value.is_finite() || value < 0.0 {
        return Err(invalid(s, "the value must be finite and >= 0"));
    }
    Ok((value * scale).round() as u64)
}

fn invalid(input: &str, reason: &str) -> String {
    format!(
        "invalid duration \"{input}\": {reason}. Use a number with a unit ns, us, ms, or s, such as \"50us\"."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_every_unit() {
        assert_eq!(parse_duration("7ns"), Ok(7));
        assert_eq!(parse_duration("50us"), Ok(50_000));
        assert_eq!(parse_duration("50µs"), Ok(50_000));
        assert_eq!(parse_duration("1.5ms"), Ok(1_500_000));
        assert_eq!(parse_duration("2s"), Ok(2_000_000_000));
        assert_eq!(parse_duration(" 3 ms "), Ok(3_000_000));
    }

    #[test]
    fn rejects_bad_input() {
        for bad in ["", "50", "fast", "-1ms", "1h", "infs", "NaNms"] {
            let err = parse_duration(bad).unwrap_err();
            assert!(err.contains("invalid duration"), "{bad}: {err}");
            assert!(err.contains("\"50us\""), "{bad}: {err}");
        }
    }
}
