//! `#[verify(margin = N)]`: percent tolerance around one top-level comparison.
//!
//! The tolerance is `|rhs| * N / 100`.
//! - `lhs < rhs` and `lhs <= rhs` pass when `lhs <= rhs + tolerance`.
//! - `lhs > rhs` and `lhs >= rhs` pass when `lhs >= rhs - tolerance`.
//! - `lhs == rhs` passes when `|lhs - rhs| <= tolerance`.
//!
//! A margin needs exactly one comparison outside parentheses, and no `&&`, `||`, or `!=` there.

/// Comparison operator at the top level of an expression
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Comparison {
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
    /// `==`
    Eq,
}

/// An expression split at its single top-level comparison
#[derive(Debug, PartialEq)]
pub(crate) struct SplitComparison<'a> {
    pub lhs: &'a str,
    pub op: Comparison,
    pub rhs: &'a str,
}

/// Split `expr` at its single top-level comparison.
///
/// Text inside parentheses or double quotes is skipped.
pub(crate) fn split_comparison(expr: &str) -> Result<SplitComparison<'_>, String> {
    let bytes = expr.as_bytes();
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut found: Option<(usize, usize, Comparison)> = None;
    let mut i = 0;

    while i < bytes.len() {
        let c = bytes[i];
        let next = bytes.get(i + 1).copied();
        if in_string {
            if c == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        match c {
            b'"' => in_string = true,
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth < 0 {
                    return Err("unbalanced `)`".to_string());
                }
            }
            _ if depth == 0 => {
                let (op, width) = match (c, next) {
                    (b'&', Some(b'&')) | (b'|', Some(b'|')) => {
                        return Err("`&&` and `||` are not allowed at the top level".to_string());
                    }
                    (b'!', Some(b'=')) => {
                        return Err("`!=` has no margin meaning".to_string());
                    }
                    (b'<', Some(b'=')) => (Some(Comparison::Le), 2),
                    (b'>', Some(b'=')) => (Some(Comparison::Ge), 2),
                    (b'=', Some(b'=')) => (Some(Comparison::Eq), 2),
                    (b'<', _) => (Some(Comparison::Lt), 1),
                    (b'>', _) => (Some(Comparison::Gt), 1),
                    _ => (None, 1),
                };
                if let Some(op) = op {
                    if found.is_some() {
                        return Err("more than one top-level comparison".to_string());
                    }
                    found = Some((i, i + width, op));
                }
                i += width;
                continue;
            }
            _ => {}
        }
        i += 1;
    }

    if depth != 0 || in_string {
        return Err("unbalanced `(` or `\"`".to_string());
    }
    let (start, end, op) = found.ok_or_else(|| "no top-level comparison".to_string())?;
    let lhs = expr[..start].trim();
    let rhs = expr[end..].trim();
    if lhs.is_empty() || rhs.is_empty() {
        return Err("the comparison needs a left and a right side".to_string());
    }
    Ok(SplitComparison { lhs, op, rhs })
}

/// Whether `lhs op rhs` holds within `margin_pct` percent of `|rhs|`.
pub(crate) fn passes_with_margin(lhs: f64, op: Comparison, rhs: f64, margin_pct: f64) -> bool {
    let tolerance = rhs.abs() * margin_pct / 100.0;
    match op {
        Comparison::Lt | Comparison::Le => lhs <= rhs + tolerance,
        Comparison::Gt | Comparison::Ge => lhs >= rhs - tolerance,
        Comparison::Eq => (lhs - rhs).abs() <= tolerance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_simple_and_nested() {
        let s = split_comparison("a < b").unwrap();
        assert_eq!((s.lhs, s.op, s.rhs), ("a", Comparison::Lt, "b"));

        let s = split_comparison("(raw - min(x, y)) >= 2 * (b < c)").unwrap();
        assert_eq!(s.lhs, "(raw - min(x, y))");
        assert_eq!(s.op, Comparison::Ge);
        assert_eq!(s.rhs, "2 * (b < c)");

        assert_eq!(split_comparison("a<=b").unwrap().op, Comparison::Le);
        assert_eq!(split_comparison("a == 5").unwrap().op, Comparison::Eq);
        assert_eq!(split_comparison("a > 5").unwrap().op, Comparison::Gt);
    }

    #[test]
    fn rejects_ambiguous_expressions() {
        for (expr, needle) in [
            ("a < b && c < d", "`&&`"),
            ("a < b || c", "`&&`"),
            ("a != b", "`!=`"),
            ("a < b < c", "more than one"),
            ("a + b", "no top-level"),
            ("(a < b", "unbalanced"),
            ("a < b)", "unbalanced"),
            ("< b", "left and a right"),
        ] {
            let err = split_comparison(expr).unwrap_err();
            assert!(err.contains(needle), "{expr}: {err}");
        }
    }

    #[test]
    fn less_than_allows_margin_above_rhs() {
        // 104 < 100 fails without margin, passes within 5%
        assert!(!passes_with_margin(104.0, Comparison::Lt, 100.0, 0.0));
        assert!(passes_with_margin(104.0, Comparison::Lt, 100.0, 5.0));
        assert!(passes_with_margin(105.0, Comparison::Lt, 100.0, 5.0));
        assert!(!passes_with_margin(106.0, Comparison::Lt, 100.0, 5.0));
    }

    #[test]
    fn greater_than_allows_margin_below_rhs() {
        assert!(passes_with_margin(96.0, Comparison::Gt, 100.0, 5.0));
        assert!(!passes_with_margin(94.0, Comparison::Ge, 100.0, 5.0));
    }

    #[test]
    fn equality_uses_symmetric_tolerance() {
        assert!(passes_with_margin(103.0, Comparison::Eq, 100.0, 5.0));
        assert!(passes_with_margin(97.0, Comparison::Eq, 100.0, 5.0));
        assert!(!passes_with_margin(110.0, Comparison::Eq, 100.0, 5.0));
    }

    #[test]
    fn negative_rhs_uses_absolute_tolerance() {
        assert!(passes_with_margin(-95.0, Comparison::Lt, -100.0, 5.0));
        assert!(!passes_with_margin(-94.0, Comparison::Lt, -100.0, 5.0));
    }
}
