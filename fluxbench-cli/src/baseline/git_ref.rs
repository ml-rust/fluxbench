//! Git ref checks for the `compare` subcommand.

use fluxbench_report::Report;

/// Resolve `git_ref` to a full commit hash with `git rev-parse --verify`.
pub(crate) fn resolve_git_ref(git_ref: &str) -> anyhow::Result<String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--verify", git_ref])
        .output()
        .map_err(|e| {
            anyhow::anyhow!(
                "cannot run git to resolve ref '{git_ref}': {e}. Check that git is on PATH."
            )
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "invalid git ref '{git_ref}': {}. Pass a branch, tag, or commit.",
            stderr.trim()
        );
    }

    let resolved = String::from_utf8(output.stdout)?.trim().to_string();
    if resolved.is_empty() {
        anyhow::bail!("git ref '{git_ref}' resolved to an empty commit hash. Pass another ref.");
    }
    Ok(resolved)
}

/// Check that `baseline` was recorded at `resolved` (a full or abbreviated hash).
///
/// A baseline without commit metadata passes with a warning.
pub(crate) fn check_baseline_commit(
    baseline: &Report,
    git_ref: &str,
    resolved: &str,
) -> anyhow::Result<()> {
    let Some(commit) = baseline.meta.git_commit.as_deref() else {
        eprintln!(
            "Warning: baseline report has no commit metadata; git ref consistency cannot be checked."
        );
        return Ok(());
    };
    if commits_match(commit, resolved) {
        return Ok(());
    }
    anyhow::bail!(
        "baseline commit {commit} does not match git ref {git_ref} ({resolved}). \
         Save a baseline on {git_ref} with --save-baseline, or pass the matching ref."
    )
}

/// Hashes match when they are equal or one is a prefix of the other.
fn commits_match(a: &str, b: &str) -> bool {
    !a.is_empty() && !b.is_empty() && (a.starts_with(b) || b.starts_with(a))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abbreviated_hashes_match() {
        assert!(commits_match("abc123def", "abc123def"));
        assert!(commits_match("abc123", "abc123def"));
        assert!(commits_match("abc123def", "abc123"));
        assert!(!commits_match("abc124", "abc123def"));
        assert!(!commits_match("", "abc"));
    }
}
