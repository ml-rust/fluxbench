//! `#[flux::group]` hierarchy: parent links and inherited tags.
//!
//! A benchmark carries its own tags plus the tags of its group and every ancestor group.
//! `--group g` selects group `g` and every group below it.

use fluxbench_core::{BenchmarkDef, GroupDef};
use std::collections::{BTreeSet, HashMap};

/// Checked group definitions
#[derive(Debug, Default)]
pub(crate) struct GroupTree<'a> {
    defs: HashMap<&'a str, &'a GroupDef>,
}

impl<'a> GroupTree<'a> {
    /// Check and index `defs`.
    ///
    /// Errors: a duplicate ID, a parent that is neither a `#[flux::group]` nor a benchmark group, or a parent cycle.
    pub(crate) fn new(
        defs: impl IntoIterator<Item = &'a GroupDef>,
        bench_groups: &BTreeSet<&str>,
    ) -> anyhow::Result<Self> {
        let mut index: HashMap<&'a str, &'a GroupDef> = HashMap::new();
        for def in defs {
            if index.insert(def.id, def).is_some() {
                anyhow::bail!(
                    "#[flux::group] id `{}` is defined twice. Give each group one definition.",
                    def.id
                );
            }
        }
        let tree = Self { defs: index };

        for def in tree.defs.values() {
            if let Some(parent) = def.parent {
                if !tree.defs.contains_key(parent) && !bench_groups.contains(parent) {
                    anyhow::bail!(
                        "#[flux::group] `{}` names parent `{parent}`, which is not a known group. Fix the parent or add #[flux::group(id = \"{parent}\")].",
                        def.id
                    );
                }
            }
            tree.check_acyclic(def.id)?;
        }
        Ok(tree)
    }

    fn check_acyclic(&self, start: &str) -> anyhow::Result<()> {
        let mut seen = vec![start];
        let mut current = start;
        while let Some(parent) = self.defs.get(current).and_then(|d| d.parent) {
            if seen.contains(&parent) {
                seen.push(parent);
                anyhow::bail!(
                    "#[flux::group] parents form a cycle: {}. Remove one parent link.",
                    seen.join(" -> ")
                );
            }
            seen.push(parent);
            current = parent;
        }
        Ok(())
    }

    /// `group` followed by its ancestors, nearest first
    pub(crate) fn ancestors<'g>(&'g self, group: &'g str) -> Vec<&'g str> {
        let mut chain = vec![group];
        let mut current = group;
        while let Some(parent) = self.defs.get(current).and_then(|d| d.parent) {
            if chain.contains(&parent) {
                break;
            }
            chain.push(parent);
            current = parent;
        }
        chain
    }

    /// Whether `bench_group` is `filter` or a group below it
    pub(crate) fn in_group(&self, bench_group: &str, filter: &str) -> bool {
        self.ancestors(bench_group).contains(&filter)
    }

    /// The benchmark's own tags plus every tag of its group and ancestor groups
    pub(crate) fn tags_of<'b>(&'b self, bench: &'b BenchmarkDef) -> BTreeSet<&'b str> {
        let mut tags: BTreeSet<&str> = bench.tags.iter().copied().collect();
        for group in self.ancestors(bench.group) {
            if let Some(def) = self.defs.get(group) {
                tags.extend(def.tags.iter().copied());
            }
        }
        tags
    }

    /// IDs of every `#[flux::group]`
    pub(crate) fn defined_ids(&self) -> Vec<&'a str> {
        self.defs.keys().copied().collect()
    }

    /// Every definition
    pub(crate) fn defs(&self) -> impl Iterator<Item = &'a GroupDef> + '_ {
        self.defs.values().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn def(
        id: &'static str,
        parent: Option<&'static str>,
        tags: &'static [&'static str],
    ) -> GroupDef {
        GroupDef {
            id,
            description: "",
            tags,
            parent,
        }
    }

    #[test]
    fn child_inherits_parent_tags_and_group_filter() {
        let defs = [
            def("parsing", None, &["hot"]),
            def("json", Some("parsing"), &["json"]),
        ];
        let tree = GroupTree::new(&defs, &BTreeSet::new()).unwrap();
        let bench = BenchmarkDef {
            group: "json",
            tags: &["own"],
            ..crate::test_support::bench_def("b")
        };
        let tags: Vec<&str> = tree.tags_of(&bench).into_iter().collect();
        assert_eq!(tags, ["hot", "json", "own"]);
        assert!(tree.in_group("json", "parsing"));
        assert!(tree.in_group("json", "json"));
        assert!(!tree.in_group("parsing", "json"));
    }

    #[test]
    fn duplicate_id_is_an_error() {
        let defs = [def("g", None, &[]), def("g", None, &[])];
        let err = GroupTree::new(&defs, &BTreeSet::new())
            .unwrap_err()
            .to_string();
        assert!(err.contains("`g` is defined twice"), "{err}");
    }

    #[test]
    fn unknown_parent_is_an_error_but_bench_group_parent_is_fine() {
        let defs = [def("child", Some("nowhere"), &[])];
        let err = GroupTree::new(&defs, &BTreeSet::new())
            .unwrap_err()
            .to_string();
        assert!(err.contains("parent `nowhere`"), "{err}");
        let bench_groups: BTreeSet<&str> = ["nowhere"].into_iter().collect();
        assert!(GroupTree::new(&defs, &bench_groups).is_ok());
    }

    #[test]
    fn parent_cycle_is_an_error() {
        let defs = [def("a", Some("b"), &[]), def("b", Some("a"), &[])];
        let err = GroupTree::new(&defs, &BTreeSet::new())
            .unwrap_err()
            .to_string();
        assert!(err.contains("cycle"), "{err}");
    }
}
