//! Which code produced a baseline or a self-golden, and whether that is a legitimate "before" for this build.
//!
//! Review finding 1 (docs/reviews/2026-10-13-opus-phase1b-review.md): a "before" baseline recorded from the new tree
//! with kernels switched off is not the pre-change code, and nothing used to record which code it was. Now:
//!
//! * The build stamps itself. `scripts/phase1b-acceptance.sh` computes `scripts/tree-stamp.sh` on a machine that has
//!   the git history and passes the result into `cargo build` as `INTARSIA_BUILD_STAMP` (read with `option_env!`, so
//!   cargo rebuilds when it changes). The remote build boxes get the tree without `.git`, so the binary cannot ask
//!   git itself. A binary built without the variable is "unstamped".
//! * A baseline / self-golden stores that stamp. The gate refuses to compare unless the recorded code is the declared
//!   pre-change commit (`--pre-change SHA`) or an ancestor of the code under test (`--revlist FILE`: `git rev-list
//!   HEAD` of the tree being tested, written on the machine with the history), is not the same code tree as this
//!   build (a self-comparison proves nothing), and is clean, apart from a declared harness overlay.
//! * The harness overlay: the pre-change engine has no baseline or self-golden support, so "before" is built from the
//!   pre-change commit with only the files in [`HARNESS_OVERLAY_PATHS`] copied over it. Its stamp says `dirty=yes
//!   overlay=<paths>`. The gate accepts a dirty baseline only when every dirty path is on that list, because those
//!   files read engine output and never change it.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Set at build time by `scripts/phase1b-acceptance.sh` (`scripts/tree-stamp.sh` output plus `overlay=a,b`).
pub const BUILD_STAMP: Option<&str> = option_env!("INTARSIA_BUILD_STAMP");

/// The only files an "old engine + harness" build may differ from its commit in. They compare and dump logits; they
/// do not run or change the model.
pub const HARNESS_OVERLAY_PATHS: &[&str] = &[
    "crates/intarsia-cli/src/cmd/accuracy.rs",
    "crates/intarsia-cli/src/cmd/accuracy/baseline.rs",
    "crates/intarsia-cli/src/cmd/accuracy/envelope.rs",
    "crates/intarsia-cli/src/cmd/accuracy/provenance.rs",
    "crates/intarsia-cli/src/cmd/accuracy/selfgolden.rs",
];

/// The code a measurement ran on.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
pub struct CodeStamp {
    /// `git rev-parse HEAD` (7 characters or more) of the commit the build was made from; empty = unstamped.
    pub head: String,
    /// `scripts/tree-stamp.sh` `code-tree=`: hash of the working tree's code paths (docs excluded).
    pub code_tree: String,
    /// The code tree differs from `head`.
    pub dirty: bool,
    /// The paths that differ from `head` when `dirty` is the declared harness overlay (empty = plain dirty tree).
    pub overlay: Vec<String>,
}

impl CodeStamp {
    pub fn is_stamped(&self) -> bool {
        self.head.len() >= 7 && !self.code_tree.is_empty()
    }

    /// Parse `head=a7da13b code-tree=1f2e3d4c5b6a dirty=no [overlay=a,b]` (the output of `scripts/tree-stamp.sh`).
    pub fn parse(s: &str) -> CodeStamp {
        let mut c = CodeStamp::default();
        for tok in s.split_whitespace() {
            let Some((k, v)) = tok.split_once('=') else { continue };
            match k {
                "head" => c.head = v.to_string(),
                "code-tree" => c.code_tree = v.to_string(),
                "dirty" => c.dirty = v == "yes",
                "overlay" => c.overlay = v.split(',').filter(|p| !p.is_empty()).map(String::from).collect(),
                _ => {}
            }
        }
        c
    }

    /// The stamp of this binary (unstamped when `INTARSIA_BUILD_STAMP` was not set at build time).
    pub fn of_this_build() -> CodeStamp {
        BUILD_STAMP.map(CodeStamp::parse).unwrap_or_default()
    }

    pub fn describe(&self) -> String {
        if !self.is_stamped() {
            return "UNSTAMPED (built without INTARSIA_BUILD_STAMP)".into();
        }
        format!(
            "head {} code-tree {} {}{}",
            self.head,
            self.code_tree,
            if self.dirty { "dirty" } else { "clean" },
            if self.overlay.is_empty() { String::new() } else { format!(" + harness overlay [{}]", self.overlay.join(", ")) }
        )
    }
}

/// How the run was executed: facts that legitimately differ between "before" and "after" (so they are printed as a
/// diff, never used to refuse a comparison, except the kernel disable list, see `baseline::compatible`).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
pub struct ExecInfo {
    /// The command line (carries `--no-graphs`, `--prefill-chunk`, ...).
    pub cmdline: Vec<String>,
    /// `INTARSIA_CUDA_DISABLE` / `INTARSIA_HIP_DISABLE`, "" when unset.
    pub disable_list: String,
    /// `INTARSIA_CPU_FAULT` / `INTARSIA_CPU_MUTATE` (gate-sensitivity hooks of the cpu backend), "" when unset.
    /// Printed in the diff, never used to refuse a comparison; the noise envelope uses it to see that a perturbation
    /// set this way really differs from the unperturbed run. Absent in files written before it existed.
    #[serde(default)]
    pub fault: String,
    /// Resolved kernel per op (`engine.plan()["ops"]`): op -> "kernel_a+kernel_b".
    pub kernels: BTreeMap<String, String>,
}

impl ExecInfo {
    pub fn capture(plan: &serde_json::Value) -> ExecInfo {
        let disable_list = ["INTARSIA_CUDA_DISABLE", "INTARSIA_HIP_DISABLE"]
            .iter()
            .filter_map(|k| std::env::var(k).ok().filter(|v| !v.trim().is_empty()).map(|v| format!("{k}={v}")))
            .collect::<Vec<_>>()
            .join(" ");
        let fault = ["INTARSIA_CPU_FAULT", "INTARSIA_CPU_MUTATE"]
            .iter()
            .filter_map(|k| std::env::var(k).ok().filter(|v| !v.trim().is_empty()).map(|v| format!("{k}={v}")))
            .collect::<Vec<_>>()
            .join(" ");
        let mut kernels: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for o in plan["ops"].as_array().into_iter().flatten() {
            if let (Some(op), Some(k)) = (o["op"].as_str(), o["kernel"].as_str()) {
                let e = kernels.entry(op.to_string()).or_default();
                if !e.iter().any(|x| x == k) {
                    e.push(k.to_string());
                }
            }
        }
        ExecInfo {
            cmdline: std::env::args().collect(),
            disable_list,
            fault,
            kernels: kernels.into_iter().map(|(k, mut v)| (k, { v.sort(); v.join("+") })).collect(),
        }
    }

    /// Human-readable differences to another run (empty = identical kernels and flags).
    pub fn diff(&self, other: &ExecInfo) -> Vec<String> {
        let mut d = Vec::new();
        let flag = |c: &ExecInfo, f: &str| c.cmdline.iter().any(|a| a == f);
        if flag(self, "--no-graphs") != flag(other, "--no-graphs") {
            d.push(format!("graphs: baseline {}, this run {}", if flag(self, "--no-graphs") { "off" } else { "on/unknown" }, if flag(other, "--no-graphs") { "off" } else { "on" }));
        }
        if self.disable_list != other.disable_list {
            d.push(format!("kernel disable list: baseline '{}', this run '{}'", self.disable_list, other.disable_list));
        }
        if self.fault != other.fault {
            d.push(format!("fault injection: baseline '{}', this run '{}'", self.fault, other.fault));
        }
        for (op, k) in &self.kernels {
            match other.kernels.get(op) {
                Some(o) if o == k => {}
                Some(o) => d.push(format!("{op}: {k} -> {o}")),
                None => d.push(format!("{op}: {k} -> (not in this plan)")),
            }
        }
        for (op, o) in &other.kernels {
            if !self.kernels.contains_key(op) {
                d.push(format!("{op}: (not in the baseline's plan) -> {o}"));
            }
        }
        d
    }
}

/// What the caller says about the comparison.
#[derive(Clone, Debug, Default)]
pub struct Policy {
    /// `--pre-change SHA`: the commit the "before" numbers must come from.
    pub pre_change: Option<String>,
    /// `--revlist FILE`: the ancestors (`git rev-list HEAD`) of the code under test, one sha per line.
    pub revlist: Vec<String>,
    /// Dev escape hatches: compare although a stamp is missing / the trees are equal / this build is dirty.
    pub allow_unstamped: bool,
    pub allow_same_code: bool,
    pub allow_dirty: bool,
}

fn same_commit(a: &str, b: &str) -> bool {
    let (a, b) = (a.trim(), b.trim());
    a.len() >= 7 && b.len() >= 7 && (a.starts_with(b) || b.starts_with(a))
}

/// Judge whether `base` (what recorded the numbers) is a legitimate "before" of `cur` (the code under test).
/// `what` names the artefact in messages ("baseline", "self-golden"). Returns warnings.
pub fn check(what: &str, base: &CodeStamp, cur: &CodeStamp, p: &Policy) -> Result<Vec<String>> {
    let mut warn = Vec::new();
    let mut soft = |msg: String, hard: bool| -> Result<()> {
        if hard {
            bail!("{msg}");
        }
        warn.push(format!("{msg} (--allow-unstamped)"));
        Ok(())
    };
    if !base.is_stamped() {
        soft(
            format!("the {what} does not record the code that produced it (written by a build without INTARSIA_BUILD_STAMP, or by an older tool): record it again from the pre-change build"),
            !p.allow_unstamped,
        )?;
        return Ok(warn);
    }
    if !cur.is_stamped() {
        soft(
            "this binary was built without INTARSIA_BUILD_STAMP, so it cannot say which code it is; build through scripts/phase1b-acceptance.sh (or set the variable to the output of scripts/tree-stamp.sh)".into(),
            !p.allow_unstamped,
        )?;
        return Ok(warn);
    }
    if base.dirty {
        let allowed = !base.overlay.is_empty() && base.overlay.iter().all(|o| HARNESS_OVERLAY_PATHS.contains(&o.as_str()));
        if !allowed {
            bail!(
                "the {what} was recorded from a modified tree ({}): only a build of {} plus the harness files {:?} is a legitimate pre-change build",
                base.describe(),
                p.pre_change.as_deref().unwrap_or("the pre-change commit"),
                HARNESS_OVERLAY_PATHS
            );
        }
    }
    if cur.dirty && !p.allow_dirty {
        bail!("this build is from a modified tree ({}): commit first, or pass --allow-dirty for a development run", cur.describe());
    }
    // ancestry: the declared pre-change commit, or an ancestor of the code under test
    let is_declared = p.pre_change.as_deref().is_some_and(|pc| same_commit(&base.head, pc));
    let is_ancestor = p.revlist.iter().any(|r| same_commit(&base.head, r));
    if !is_declared && !is_ancestor {
        let mut why = format!("the {what} was recorded from {} which is ", base.describe());
        match (&p.pre_change, p.revlist.is_empty()) {
            (Some(pc), true) => why += &format!("not the declared pre-change commit {pc}"),
            (Some(pc), false) => {
                why += &format!("neither the declared pre-change commit {pc} nor an ancestor of this build ({} commits checked)", p.revlist.len())
            }
            (None, false) => why += &format!("not an ancestor of this build ({} commits checked)", p.revlist.len()),
            (None, true) => {
                why += "unverifiable: pass --pre-change SHA and/or --revlist FILE (git rev-list HEAD, written where the history is)"
            }
        }
        bail!("{why}");
    }
    if let Some(pc) = &p.pre_change {
        if !p.revlist.is_empty() && !p.revlist.iter().any(|r| same_commit(r, pc)) {
            bail!("the declared pre-change commit {pc} is not an ancestor of this build ({} commits checked)", p.revlist.len());
        }
    }
    if base.code_tree == cur.code_tree {
        if !p.allow_same_code {
            bail!(
                "the {what} was recorded from the same code tree ({}) as this build: comparing code with itself cannot show a regression \
                 (--allow-same-code to check run-to-run reproducibility)",
                cur.code_tree
            );
        }
        warn.push(format!("the {what} is from the same code tree as this build (--allow-same-code): this checks reproducibility only"));
    }
    Ok(warn)
}

#[cfg(test)]
mod tests {
    use super::*;

    const OLD: &str = "head=4963b43 code-tree=aaaaaaaaaaaa dirty=no";
    const OLD_OVERLAY: &str = "head=4963b43 code-tree=bbbbbbbbbbbb dirty=yes overlay=crates/intarsia-cli/src/cmd/accuracy.rs,crates/intarsia-cli/src/cmd/accuracy/baseline.rs";
    const NEW: &str = "head=88c5285 code-tree=cccccccccccc dirty=no";

    fn pol(pre: Option<&str>, rev: &[&str]) -> Policy {
        Policy { pre_change: pre.map(String::from), revlist: rev.iter().map(|s| s.to_string()).collect(), ..Default::default() }
    }
    fn err(base: &str, cur: &str, p: &Policy) -> String {
        check("baseline", &CodeStamp::parse(base), &CodeStamp::parse(cur), p).unwrap_err().to_string()
    }
    fn ok(base: &str, cur: &str, p: &Policy) -> Vec<String> {
        check("baseline", &CodeStamp::parse(base), &CodeStamp::parse(cur), p).unwrap()
    }

    #[test]
    fn parse_round_trip_and_unstamped() {
        let s = CodeStamp::parse(OLD_OVERLAY);
        assert_eq!((s.head.as_str(), s.dirty, s.overlay.len()), ("4963b43", true, 2));
        assert!(s.is_stamped());
        assert!(!CodeStamp::default().is_stamped());
        assert!(CodeStamp::default().describe().contains("UNSTAMPED"));
    }

    #[test]
    fn the_declared_pre_change_commit_or_an_ancestor_is_accepted() {
        let full = "4963b43aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        ok(OLD, NEW, &pol(Some("4963b43"), &[]));
        ok(OLD, NEW, &pol(Some(full), &[]));
        ok(OLD, NEW, &pol(None, &["88c5285000000000000000000000000000000000", full]));
        ok(OLD, NEW, &pol(Some("4963b43"), &["88c5285000000000000000000000000000000000", full]));
    }

    #[test]
    fn a_baseline_from_somewhere_else_is_refused() {
        // not the declared commit
        let e = err("head=1111111 code-tree=dddddddddddd dirty=no", NEW, &pol(Some("4963b43"), &[]));
        assert!(e.contains("not the declared pre-change commit"), "{e}");
        // not an ancestor
        let e = err(OLD, NEW, &pol(None, &["88c5285000000000000000000000000000000000", "2222222000000000000000000000000000000000"]));
        assert!(e.contains("not an ancestor"), "{e}");
        // nothing to verify against
        let e = err(OLD, NEW, &pol(None, &[]));
        assert!(e.contains("unverifiable"), "{e}");
        // declared commit is not even in the history of this build
        let e = err(OLD, NEW, &pol(Some("4963b43"), &["88c5285000000000000000000000000000000000"]));
        assert!(e.contains("not an ancestor of this build"), "{e}");
    }

    #[test]
    fn unstamped_artefacts_are_refused_unless_allowed() {
        let e = err("", NEW, &pol(Some("4963b43"), &[]));
        assert!(e.contains("does not record the code"), "{e}");
        let e = err(OLD, "", &pol(Some("4963b43"), &[]));
        assert!(e.contains("INTARSIA_BUILD_STAMP"), "{e}");
        let p = Policy { allow_unstamped: true, ..pol(Some("4963b43"), &[]) };
        assert_eq!(check("baseline", &CodeStamp::default(), &CodeStamp::default(), &p).unwrap().len(), 1);
    }

    #[test]
    fn only_the_harness_overlay_may_be_dirty() {
        // old engine + harness overlay is the legitimate pre-change build
        ok(OLD_OVERLAY, NEW, &pol(Some("4963b43"), &[]));
        // dirty without a declared overlay, or touching engine files: refused
        let e = err("head=4963b43 code-tree=bbbbbbbbbbbb dirty=yes", NEW, &pol(Some("4963b43"), &[]));
        assert!(e.contains("modified tree"), "{e}");
        let e = err(
            "head=4963b43 code-tree=bbbbbbbbbbbb dirty=yes overlay=crates/intarsia-cli/src/cmd/accuracy.rs,crates/intarsia-core/src/model.rs",
            NEW,
            &pol(Some("4963b43"), &[]),
        );
        assert!(e.contains("modified tree"), "{e}");
        // the build under test must be clean
        let e = err(OLD, "head=88c5285 code-tree=cccccccccccc dirty=yes", &pol(Some("4963b43"), &[]));
        assert!(e.contains("modified tree"), "{e}");
    }

    #[test]
    fn comparing_code_with_itself_needs_an_explicit_flag() {
        let same = "head=88c5285 code-tree=cccccccccccc dirty=no";
        let e = err(same, same, &pol(None, &["88c5285000000000000000000000000000000000"]));
        assert!(e.contains("same code tree"), "{e}");
        let p = Policy { allow_same_code: true, ..pol(None, &["88c5285000000000000000000000000000000000"]) };
        assert_eq!(ok(same, same, &p).len(), 1);
    }

    #[test]
    fn exec_diff_names_the_changed_kernels_and_flags() {
        let plan = |k: &str| serde_json::json!({"ops": [{"op": "moe_experts", "kernel": k}, {"op": "matmul", "kernel": "mm"}]});
        let mut a = ExecInfo::capture(&plan("moe_experts_bf16"));
        let mut b = ExecInfo::capture(&plan("moe_experts_mma_bf16"));
        a.cmdline = vec!["intarsia".into(), "--no-graphs".into()];
        b.cmdline = vec!["intarsia".into()];
        let d = a.diff(&b);
        assert!(d.iter().any(|l| l.contains("moe_experts: moe_experts_bf16 -> moe_experts_mma_bf16")), "{d:?}");
        assert!(d.iter().any(|l| l.contains("graphs: baseline off, this run on")), "{d:?}");
        assert!(a.diff(&a).is_empty());
    }
}
