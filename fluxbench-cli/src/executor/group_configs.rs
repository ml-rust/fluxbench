//! Merged `#[flux::group]` run config per benchmark group.

use fluxbench_core::GroupConfig;
use std::collections::HashMap;

/// Merged `#[flux::group]` run config, keyed by group ID.
///
/// Each entry holds the group's own config with its ancestors' configs below it, nearest first.
#[derive(Debug, Clone, Default)]
pub struct GroupConfigs(HashMap<String, GroupConfig>);

impl GroupConfigs {
    /// Set the merged config of `group`.
    pub fn insert(&mut self, group: impl Into<String>, config: GroupConfig) {
        self.0.insert(group.into(), config);
    }

    /// Merged config of `group`. A group with no entry gets `GroupConfig::NONE`.
    pub fn get(&self, group: &str) -> GroupConfig {
        self.0.get(group).copied().unwrap_or(GroupConfig::NONE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_group_sets_nothing() {
        let mut configs = GroupConfigs::default();
        let set = GroupConfig {
            sample_count: Some(9),
            ..GroupConfig::NONE
        };
        configs.insert("g", set);
        assert_eq!(configs.get("g"), set);
        assert_eq!(configs.get("other"), GroupConfig::NONE);
    }
}
