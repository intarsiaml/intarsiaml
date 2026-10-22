//! Gate-definition history (review finding 2, 2026-10-22, requirement 3).
//!
//! Several accuracy gates were redefined on 2026-10-21 after a run had failed under the old definition. The new
//! definitions are the user's decisions (`docs/decisions/LOG.md`); this module changes no verdict. It makes the change
//! visible: for every gate whose definition changed in October, the verdict file (`--write-verdict`), the baseline
//! JSON's `modes[].overall` (and with it the qualify verdict JSON, which copies that object) and one stdout line per
//! mode carry the metric under the old definition and under the current one, both verdicts, the date of the change and
//! the LOG row that records it.
//!
//! Covered here:
//! * `kl_floor`: the KL floor of a golden gate. Old: the KL over ALL rows against the static class floor alone
//!   (the 2026-10-20 definition that failed the 35B build on `chat_reuse_4k`, 4.36e-2 against 3.0e-2). Current: the KL
//!   over the SCORED rows (`turns.rs`) against `max(static floor, factor x reference noise)`.
//! * `self_vs_base.*`: `--self-golden --self-vs-ref-noise`. Old: every self-golden budget gated (largest position KL,
//!   top-1 flips, tail mass, worst prompt). Current: only the mean KL (against `factor x noise`) and a per-prompt check
//!   gate; the rest is reported.
//!
//! The Metal decode floor lives in `scripts/metal-speed-gate/gate.py`, which writes the same record shape.
//!
//! Not covered: the 35b-short baseline gate that ADR-0008 made advisory is judged in `scripts/phase1b-acceptance.sh`.
//!
//! The stdout line starts with [`LINE_PREFIX`] and avoids the substrings the scripts parse (`golden gate:`,
//! `floor [`, `... verdict`, `self-golden =>`, `SELF-STATS`).

use super::selfgolden::SelfCheck;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const LINE_PREFIX: &str = "gate-definition:";

pub const PASS: &str = "PASS";
pub const FAIL: &str = "FAIL";
/// A check that is still computed and printed but no longer decides.
pub const NOT_GATED: &str = "NOT_GATED";
/// The check did not exist under that definition.
pub const ABSENT: &str = "ABSENT";

/// The date (UTC / EDT calendar day) both KL-gate changes took effect.
pub const CHANGED_ON: &str = "2026-10-21";
/// `docs/decisions/LOG.md` rows by date, time and lead words (line numbers go stale with every new row).
pub const LOG_SCORED_ROWS: &str = "docs/decisions/LOG.md row 2026-10-21 10:43 \"The accuracy KL gate scores only the rows the model is asked to predict\"";
pub const LOG_NOISE_GATE: &str =
    "docs/decisions/LOG.md row 2026-10-21 11:23 \"Item 3 below (KL gate restated against vLLM's own run-to-run KL)\"";
pub const LOG_SELF_REPORT_ONLY: &str = "docs/decisions/LOG.md row 2026-10-21 13:23 \"--self-golden --self-vs-ref-noise: largest position KL, top-1 flips and tail mass become report-only\"";
pub const LOG_SELF_PER_PROMPT: &str =
    "docs/decisions/LOG.md row 2026-10-21 13:58 \"the worst-prompt hard check becomes a per-prompt one\"";

/// One gate as one definition judged it.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct DefinitionView {
    /// What the gate measured and what it was held to, in words.
    pub definition: String,
    /// The metric this definition judges (`kl21_all_rows`, `kl21_scored_rows`, ...).
    pub metric_name: String,
    pub metric: Option<f64>,
    /// The threshold this definition held the metric to.
    pub limit: Option<f64>,
    /// `PASS`, `FAIL`, `NOT_GATED` or `ABSENT`.
    pub verdict: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// A gate whose definition changed, with the verdict under both definitions.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct DefinitionChange {
    pub gate: String,
    /// `YYYY-MM-DD` the current definition took effect.
    pub changed_on: String,
    /// LOG.md rows that record the change.
    pub log_ref: Vec<String>,
    /// Every number behind the two views (`all_rows_kl`, `scored_rows_kl`, ...), so neither has to be recomputed.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metrics: BTreeMap<String, f64>,
    pub old_def: DefinitionView,
    pub current_def: DefinitionView,
}

fn e(v: Option<f64>) -> String {
    v.map_or("-".to_string(), |v| format!("{v:.3e}"))
}

impl DefinitionView {
    fn short(&self) -> String {
        format!("{} {}", self.verdict, e(self.metric))
    }
}

impl DefinitionChange {
    /// `gate-definition: <gate>: old def: FAIL 4.360e-2 | current def: PASS 1.420e-2 (changed ...; LOG ...)`.
    pub fn line(&self, scope: &str) -> String {
        let refs = self
            .log_ref
            .iter()
            .map(|r| r.split(" \"").next().unwrap_or(r).trim_start_matches("docs/decisions/LOG.md row "))
            .collect::<Vec<_>>()
            .join("; ");
        format!(
            "{LINE_PREFIX} {}{}: old def: {} ({}) | current def: {} ({}) [changed {}; LOG {}]",
            self.gate,
            if scope.is_empty() { String::new() } else { format!(" [{scope}]") },
            self.old_def.short(),
            self.old_def.definition,
            self.current_def.short(),
            self.current_def.definition,
            self.changed_on,
            refs,
        )
    }
}

fn pf(ok: bool) -> &'static str {
    if ok {
        PASS
    } else {
        FAIL
    }
}

/// The KL floor of a golden gate under the 2026-10-20 definition and the current one.
///
/// * `all_rows_kl` / `scored_rows_kl`: the mode's mean KL (`kl_metric`) over all rows / the scored rows.
/// * `static_floor`: the class floor (or `--max-mean-kl`).
/// * `current_pass`, `effective_threshold`, `bound_by`: the verdict the gate actually gave (`refnoise::KlVerdict`).
///
/// The old verdict is the all-rows KL against the static floor alone: the reference-noise term did not exist then, so
/// no hybrid of the two is reported. A NaN never passes.
pub fn kl_floor_change(
    kl_metric: &str,
    all_rows_kl: f64,
    scored_rows_kl: f64,
    static_floor: f64,
    current_pass: bool,
    effective_threshold: f64,
    bound_by: &str,
) -> DefinitionChange {
    DefinitionChange {
        gate: "kl_floor".into(),
        changed_on: CHANGED_ON.into(),
        log_ref: vec![LOG_SCORED_ROWS.into(), LOG_NOISE_GATE.into()],
        metrics: BTreeMap::from([
            ("all_rows_kl".to_string(), all_rows_kl),
            ("scored_rows_kl".to_string(), scored_rows_kl),
        ]),
        old_def: DefinitionView {
            definition: format!("{kl_metric} over all rows <= static floor"),
            metric_name: "all_rows_kl".into(),
            metric: Some(all_rows_kl),
            limit: Some(static_floor),
            verdict: pf(all_rows_kl <= static_floor).into(),
            note: None,
        },
        current_def: DefinitionView {
            definition: format!(
                "{kl_metric} over scored rows <= max(static floor, factor x reference noise), bound by {bound_by}"
            ),
            metric_name: "scored_rows_kl".into(),
            metric: Some(scored_rows_kl),
            limit: Some(effective_threshold),
            verdict: pf(current_pass).into(),
            note: None,
        },
    }
}

fn check_view(c: &SelfCheck) -> DefinitionView {
    DefinitionView {
        definition: format!("{} <= budget, gated", c.name),
        metric_name: c.name.clone(),
        metric: Some(c.value),
        limit: Some(c.limit),
        verdict: pf(c.ok).into(),
        note: None,
    }
}

/// `--self-golden --self-vs-ref-noise`: `before` are the checks as `selfgolden::judge` returned them (every budget
/// gated), `after` the same checks once `refnoise::apply_to_self_checks` has run (same order, one check appended).
/// One record per check whose limit, name or gating changed, plus one for the mode's overall result.
pub fn self_check_changes(before: &[SelfCheck], after: &[SelfCheck]) -> Vec<DefinitionChange> {
    let mut out = Vec::new();
    for (i, a) in after.iter().enumerate() {
        let old = before.get(i);
        let changed = old.is_none_or(|b| b.name != a.name || b.gated != a.gated || b.limit != a.limit);
        if !changed {
            continue;
        }
        let old_def = old.map_or_else(
            || DefinitionView {
                definition: "no such check".into(),
                metric_name: a.name.clone(),
                metric: None,
                limit: None,
                verdict: ABSENT.into(),
                note: None,
            },
            check_view,
        );
        let (verdict, note) = if a.gated {
            (pf(a.ok), None)
        } else {
            (
                NOT_GATED,
                Some(if a.ok {
                    "reported only; inside its budget".to_string()
                } else {
                    "reported only; over its budget".to_string()
                }),
            )
        };
        out.push(DefinitionChange {
            gate: format!("self_vs_base.{}", a.name),
            changed_on: CHANGED_ON.into(),
            log_ref: vec![LOG_NOISE_GATE.into(), LOG_SELF_REPORT_ONLY.into(), LOG_SELF_PER_PROMPT.into()],
            metrics: BTreeMap::from([("value".to_string(), a.value)]),
            old_def,
            current_def: DefinitionView {
                definition: if a.gated {
                    format!("{} <= limit, gated", a.name)
                } else {
                    format!("{}: reported, not gated", a.name)
                },
                metric_name: a.name.clone(),
                metric: Some(a.value),
                limit: Some(a.limit),
                verdict: verdict.into(),
                note,
            },
        });
    }
    if !out.is_empty() {
        let old_ok = before.iter().filter(|c| c.gated).all(|c| c.ok);
        let new_ok = after.iter().filter(|c| c.gated).all(|c| c.ok);
        out.push(DefinitionChange {
            gate: "self_vs_base.overall".into(),
            changed_on: CHANGED_ON.into(),
            log_ref: vec![LOG_NOISE_GATE.into(), LOG_SELF_REPORT_ONLY.into(), LOG_SELF_PER_PROMPT.into()],
            metrics: BTreeMap::new(),
            old_def: DefinitionView {
                definition: "every self-golden budget gated".into(),
                metric_name: "checks_failed".into(),
                metric: Some(before.iter().filter(|c| c.gated && !c.ok).count() as f64),
                limit: Some(0.0),
                verdict: pf(old_ok).into(),
                note: None,
            },
            current_def: DefinitionView {
                definition: "mean KL vs factor x reference noise and per-prompt check gated; other budgets reported"
                    .into(),
                metric_name: "checks_failed".into(),
                metric: Some(after.iter().filter(|c| c.gated && !c.ok).count() as f64),
                limit: Some(0.0),
                verdict: pf(new_ok).into(),
                note: None,
            },
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chk(name: &str, value: f64, limit: f64, gated: bool) -> SelfCheck {
        SelfCheck { name: name.into(), value, limit, ok: value <= limit, gated }
    }

    /// The 35b-long case of 2026-10-20/21: 4.36e-2 on all rows against 3.0e-2 failed; 1.42e-2 on the scored rows passes.
    #[test]
    fn kl_floor_carries_both_verdicts_and_both_metrics() {
        let c = kl_floor_change("21-way KL", 4.36e-2, 1.42e-2, 3.0e-2, true, 3.0e-2, "static floor");
        assert_eq!(c.old_def.verdict, FAIL);
        assert_eq!(c.current_def.verdict, PASS);
        assert_eq!(c.old_def.metric, Some(4.36e-2));
        assert_eq!(c.current_def.metric, Some(1.42e-2));
        assert_eq!(c.metrics["all_rows_kl"], 4.36e-2);
        assert_eq!(c.metrics["scored_rows_kl"], 1.42e-2);
        assert_eq!(c.changed_on, "2026-10-21");
        assert!(c.log_ref.iter().any(|r| r.contains("2026-10-21 10:43")));
        let line = c.line("prefill");
        assert!(line.starts_with("gate-definition: kl_floor [prefill]: old def: FAIL 4.360e-2"), "{line}");
        assert!(line.contains("| current def: PASS 1.420e-2"), "{line}");
        // the strings the scripts parse must not appear
        for bad in ["golden gate:", " floor [", "verdict", "self-golden =>", "SELF-STATS"] {
            assert!(!line.contains(bad), "{bad:?} in {line}");
        }
        let j = serde_json::to_value(&c).unwrap();
        assert_eq!(j["old_def"]["verdict"], "FAIL");
        assert_eq!(j["current_def"]["verdict"], "PASS");
    }

    /// Where no rows are hidden the two definitions see the same KL; a NaN never passes under either.
    #[test]
    fn kl_floor_same_rows_and_nan() {
        let c = kl_floor_change("full-vocab KL", 1e-4, 1e-4, 1e-3, true, 1e-3, "static floor");
        assert_eq!((c.old_def.verdict.as_str(), c.current_def.verdict.as_str()), (PASS, PASS));
        let n = kl_floor_change("21-way KL", f64::NAN, f64::NAN, 1e-3, false, 1e-3, "static floor");
        assert_eq!((n.old_def.verdict.as_str(), n.current_def.verdict.as_str()), (FAIL, FAIL));
    }

    /// A noise-bound pass: the static floor alone fails, the noise term rescues it. Both verdicts are recorded.
    #[test]
    fn kl_floor_noise_bound_pass_is_visible() {
        let c = kl_floor_change("21-way KL", 2.0e-2, 1.6e-2, 1.0e-2, true, 1.7e-2, "ref noise");
        assert_eq!(c.old_def.verdict, FAIL);
        assert_eq!(c.current_def.verdict, PASS);
        assert_eq!(c.current_def.limit, Some(1.7e-2));
    }

    /// `--self-vs-ref-noise`: the ungated max-position KL, flips and tail are recorded with the verdict they had.
    #[test]
    fn self_vs_ref_noise_records_ungated_checks_with_both_verdicts() {
        let before = vec![
            chk("mean KL(21-way)", 5e-3, 1e-4, true),
            chk("max position KL", 0.8, 0.05, true),
            chk("top-1 flips", 12.0, 5.0, true),
            chk("non-finite rows", 0.0, 0.0, true),
        ];
        // what apply_to_self_checks does
        let mut after = before.clone();
        after[0].name = "mean KL(21-way) vs ref noise".into();
        after[0].limit = 1e-2;
        after[0].ok = after[0].value <= 1e-2;
        after[1].gated = false;
        after[2].gated = false;
        after.push(SelfCheck {
            name: "prompt KL / (prompt factor x its noise) [p]".into(),
            value: 0.4,
            limit: 1.0,
            ok: true,
            gated: true,
        });
        let ch = self_check_changes(&before, &after);
        let by = |n: &str| {
            ch.iter().find(|c| c.gate == format!("self_vs_base.{n}")).unwrap_or_else(|| panic!("no record {n}: {ch:?}"))
        };
        let mean = by("mean KL(21-way) vs ref noise");
        assert_eq!((mean.old_def.verdict.as_str(), mean.current_def.verdict.as_str()), (FAIL, PASS));
        let maxp = by("max position KL");
        assert_eq!((maxp.old_def.verdict.as_str(), maxp.current_def.verdict.as_str()), (FAIL, NOT_GATED));
        assert_eq!(maxp.current_def.note.as_deref(), Some("reported only; over its budget"));
        assert_eq!(maxp.old_def.metric, Some(0.8));
        assert_eq!(maxp.current_def.metric, Some(0.8));
        let flips = by("top-1 flips");
        assert_eq!((flips.old_def.verdict.as_str(), flips.current_def.verdict.as_str()), (FAIL, NOT_GATED));
        let pp = by("prompt KL / (prompt factor x its noise) [p]");
        assert_eq!((pp.old_def.verdict.as_str(), pp.current_def.verdict.as_str()), (ABSENT, PASS));
        // an unchanged check has no record
        assert!(ch.iter().all(|c| c.gate != "self_vs_base.non-finite rows"));
        let all = by("overall");
        assert_eq!((all.old_def.verdict.as_str(), all.current_def.verdict.as_str()), (FAIL, PASS));
        assert_eq!(all.changed_on, "2026-10-21");
        assert!(all.log_ref.iter().any(|r| r.contains("2026-10-21 13:23")));
        for c in &ch {
            assert!(!c.old_def.verdict.is_empty() && !c.current_def.verdict.is_empty());
        }
    }

    /// No change, no records (a self-golden run that is not `--self-vs-ref-noise`).
    #[test]
    fn self_vs_base_unchanged_has_no_records() {
        let before = vec![chk("mean KL(21-way)", 5e-5, 1e-4, true)];
        assert!(self_check_changes(&before, &before).is_empty());
    }
}
