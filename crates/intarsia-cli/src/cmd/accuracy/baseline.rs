//! Baseline-relative accuracy gate (`intarsia accuracy --baseline FILE` / `--write-baseline FILE`).
//!
//! The class thresholds (bf16 / fp8 / 4bit) are absolute floors set for a quantisation error against a
//! different reference; for a speed change on the same weights they are far too loose (a bug worth 2e-2 KL
//! passes the 4-bit floor). A baseline records what the code did on a given golden before the change; the gate
//! then says "no worse than that, plus a small margin" on top-1, mean KL, tail-bucket KL, top-5 and the tail
//! mass. Margins (defaults, all flags):
//!
//! | metric | allowed regression |
//! |---|---|
//! | top-1 agreement | -0.3 pt, but at least one position (100/n pt: a single flip must not fail a 160-position mode) |
//! | mean KL, 21-way KL | x1.15, and at least +1e-6 absolute (a baseline near zero is not brittle) |
//! | top-5 overlap | -0.5 pt, but at least five overlap units (100/n pt) |
//! | mean tail excess (ours - reference tail mass) | \|excess\| may grow by 0.002 over the baseline's \|excess\| |
//! | mean \|tail difference\| per position | may grow by 0.002 (the signed mean lets positions that lose and gain tail mass cancel) |
//! | max single-position 21-way KL | x2.5, and at least +1e-3 absolute (a maximum is a noisy statistic) |
//! | each prompt: mean 21-way KL | x1.75, and at least +1e-5 absolute (one prompt doubling must not hide in the mean) |
//! | each prompt: top-1 | -1 pt, but at least one position |
//! | non-finite rows | none |
//!
//! A baseline is only comparable with the same golden file (content hash and size) and the same run
//! configuration (KV dtype, activation mode, prefill chunk, decode positions, mode set, kernel disable list);
//! anything else is refused, so a regenerated golden or a changed flag cannot make a stale baseline pass silently.
//! It also records the code that produced it (`provenance.rs`) and is refused unless that code is the declared
//! pre-change commit or its ancestor: the "before" numbers must come from the pre-change build.
//!
//! Next to this, `selfgolden.rs` compares a build with the *logits of another build on the same weights*; that is the
//! gate that can see a small change, because it has no reference noise in it (review finding 2).

use super::provenance::{CodeStamp, ExecInfo};
use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const FORMAT: &str = "intarsia-accuracy-baseline-v2";

/// Aggregate of one mode (prefill / decode (teacher-forced)) over all prompts.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
pub struct ModeStats {
    pub mode: String,
    pub positions: usize,
    pub top1_pct: f64,
    /// Full-vocabulary KL (full golden) or KL over the reference's renormalised top-k support (top-k golden).
    pub mean_kl: f64,
    /// KL over the 21-way partition with the explicit "other" bucket.
    pub mean_kl21: f64,
    pub top5_pct: f64,
    pub tail_ref: f64,
    pub tail_ours: f64,
    pub max_kl: f64,
    /// Largest single-position 21-way KL.
    pub max_kl21: f64,
    /// Mean over positions of |tail mass ours - tail mass reference| (the signed mean in `tail_excess` cancels).
    pub tail_abs: f64,
    pub nonfinite: usize,
    pub greedy_match_pct: f64,
    /// Did the absolute class floor pass when this baseline was recorded?
    pub floor_pass: bool,
    /// `mean_kl` / `mean_kl21` over the rows the model is asked to predict (outside the system and user turns of chat
    /// prompts; `accuracy/turns.rs`), and how many rows that is. `mean_kl` and `mean_kl21` above stay over ALL rows. The
    /// KL floor gates the scored numbers. `None`: a baseline written before the row mask existed.
    #[serde(default)]
    pub mean_kl_scored: Option<f64>,
    #[serde(default)]
    pub mean_kl21_scored: Option<f64>,
    #[serde(default)]
    pub scored_positions: Option<usize>,
}

impl ModeStats {
    pub fn tail_excess(&self) -> f64 {
        self.tail_ours - self.tail_ref
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
pub struct PromptStats {
    pub name: String,
    pub positions: usize,
    pub top1_pct: f64,
    pub mean_kl: f64,
    pub mean_kl21: f64,
    pub top5_pct: f64,
    pub tail_excess: f64,
    pub max_kl21: f64,
    pub tail_abs: f64,
    /// `mean_kl21` over the rows the model is asked to predict, and how many (`None`: written before the row mask).
    #[serde(default)]
    pub mean_kl21_scored: Option<f64>,
    #[serde(default)]
    pub scored_positions: Option<usize>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ModeBaseline {
    pub overall: ModeStats,
    pub prompts: Vec<PromptStats>,
}

/// Identity of the golden file the baseline was measured against.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct GoldenId {
    pub file: String,
    /// FNV-1a 64 of the file's bytes (a fingerprint, not a security hash).
    pub fnv1a64: String,
    pub size: u64,
    pub format: String,
    pub prompts: usize,
    pub positions: usize,
    pub topk: Option<usize>,
    /// What the golden's metadata says about the reference (empty when the golden predates the metadata).
    pub engine: String,
    pub engine_version: String,
    pub model: String,
    pub model_revision: String,
    pub kv_dtype: String,
}

/// The run configuration that must match for a baseline to be comparable.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct RunConfig {
    pub model: String,
    pub backend: String,
    pub device: String,
    pub kv_dtype: String,
    pub act_dtype: String,
    pub fp8_act: String,
    pub prefill_chunk: usize,
    /// Where the chunk came from (`--prefill-chunk`, `recipe <family>, <backend>`, `default`); absent in older files.
    #[serde(default)]
    pub prefill_chunk_source: String,
    pub decode_positions: usize,
    pub gate_class: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Baseline {
    pub format: String,
    pub created_unix: u64,
    pub tool_version: String,
    /// The code that produced these numbers (`provenance.rs`).
    pub code: CodeStamp,
    /// How it was executed: command line, kernel disable list, resolved kernel per op.
    pub exec: ExecInfo,
    pub golden: GoldenId,
    pub run: RunConfig,
    pub modes: Vec<ModeBaseline>,
}

#[derive(Clone, Copy, Debug)]
pub struct Margins {
    pub top1_pts: f64,
    pub kl_ratio: f64,
    pub kl_eps: f64,
    pub top5_pts: f64,
    pub tail_abs: f64,
    /// Largest single-position 21-way KL: ratio and absolute slack.
    pub max_kl_ratio: f64,
    pub max_kl_eps: f64,
    /// Per prompt: mean 21-way KL ratio and absolute slack, top-1 drop in points (at least one position).
    pub prompt_kl_ratio: f64,
    pub prompt_kl_eps: f64,
    pub prompt_top1_pts: f64,
}

impl Default for Margins {
    fn default() -> Self {
        Margins {
            top1_pts: 0.3,
            kl_ratio: 1.15,
            kl_eps: 1e-6,
            top5_pts: 0.5,
            tail_abs: 0.002,
            max_kl_ratio: 2.5,
            max_kl_eps: 1e-3,
            prompt_kl_ratio: 1.75,
            prompt_kl_eps: 1e-5,
            prompt_top1_pts: 1.0,
        }
    }
}

/// One line of the baseline comparison.
#[derive(Clone, Debug)]
pub struct Check {
    pub name: &'static str,
    pub base: f64,
    pub cur: f64,
    /// The worst acceptable value (a lower bound for `higher_is_better`, an upper bound otherwise).
    pub limit: f64,
    pub higher_is_better: bool,
    pub ok: bool,
}

pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Judge one mode against its baseline.
pub fn judge(base: &ModeStats, cur: &ModeStats, m: &Margins) -> Vec<Check> {
    let n = cur.positions.max(1) as f64;
    let one_position = 100.0 / n;
    let kl_limit = |b: f64| (b * m.kl_ratio).max(b + m.kl_eps);
    let mut v = Vec::new();
    let lim = base.top1_pct - m.top1_pts.max(one_position);
    v.push(Check { name: "top-1 %", base: base.top1_pct, cur: cur.top1_pct, limit: lim, higher_is_better: true, ok: cur.top1_pct >= lim - 1e-9 });
    let lim = kl_limit(base.mean_kl);
    v.push(Check { name: "mean KL", base: base.mean_kl, cur: cur.mean_kl, limit: lim, higher_is_better: false, ok: cur.mean_kl <= lim });
    let lim = kl_limit(base.mean_kl21);
    v.push(Check { name: "tail-bucket KL", base: base.mean_kl21, cur: cur.mean_kl21, limit: lim, higher_is_better: false, ok: cur.mean_kl21 <= lim });
    // like with like: the scored-row KL only against a baseline that recorded it (an older baseline holds all-rows KL
    // only, and judging a scored number against it would pass trivially on a chat golden)
    if let (Some(b), Some(c)) = (base.mean_kl21_scored, cur.mean_kl21_scored) {
        let lim = kl_limit(b);
        v.push(Check { name: "tail-bucket KL scored", base: b, cur: c, limit: lim, higher_is_better: false, ok: c <= lim });
    }
    let lim = base.top5_pct - m.top5_pts.max(one_position);
    v.push(Check { name: "top-5 %", base: base.top5_pct, cur: cur.top5_pct, limit: lim, higher_is_better: true, ok: cur.top5_pct >= lim - 1e-9 });
    let lim = base.tail_excess().abs() + m.tail_abs;
    v.push(Check { name: "|tail excess|", base: base.tail_excess().abs(), cur: cur.tail_excess().abs(), limit: lim, higher_is_better: false, ok: cur.tail_excess().abs() <= lim });
    let lim = base.tail_abs + m.tail_abs;
    v.push(Check { name: "mean |tail diff|", base: base.tail_abs, cur: cur.tail_abs, limit: lim, higher_is_better: false, ok: cur.tail_abs <= lim });
    let lim = base.max_kl21 * m.max_kl_ratio + m.max_kl_eps;
    v.push(Check { name: "max KL(21-way)", base: base.max_kl21, cur: cur.max_kl21, limit: lim, higher_is_better: false, ok: cur.max_kl21 <= lim });
    v.push(Check { name: "non-finite rows", base: base.nonfinite as f64, cur: cur.nonfinite as f64, limit: 0.0, higher_is_better: false, ok: cur.nonfinite == 0 });
    v
}

/// Where a per-prompt limit came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// The fixed margin (`--margin-prompt-kl` ...), no envelope or the envelope allows no more.
    Fixed,
    /// `k` x the prompt's measured noise (`envelope.rs`) was wider than the fixed margin.
    Envelope,
}

impl Source {
    pub fn name(self) -> &'static str {
        match self {
            Source::Fixed => "fixed",
            Source::Envelope => "envelope",
        }
    }
}

/// The measured noise of one prompt in one mode: |difference| of its per-prompt numbers between two runs of the
/// same build (`envelope.rs`), in the units of the checks (top-1 in percentage points, KL and tail in nats / mass).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PromptDelta {
    pub top1_pts: f64,
    pub kl21: f64,
    pub tail_abs: f64,
}

/// Per-prompt limits from a noise envelope: limit = baseline value -/+ max(fixed margin, `k` x the prompt's delta).
pub struct Calibration<'a> {
    pub k: f64,
    pub delta: &'a dyn Fn(&str) -> Option<PromptDelta>,
}

/// One per-prompt check and how its limit was made.
#[derive(Clone, Debug)]
pub struct PromptCheck {
    pub check: Check,
    /// The prompt's measured noise in this metric (None: no envelope).
    pub delta: Option<f64>,
    pub source: Source,
}

impl PromptCheck {
    /// The smallest `k` with which this check would pass (0: it passes on its own, None: no envelope delta, or it
    /// is zero and the check fails on the fixed margin).
    pub fn k_needed(&self) -> Option<f64> {
        let c = &self.check;
        let worse = if c.higher_is_better { c.base - c.cur } else { c.cur - c.base };
        if !worse.is_finite() {
            return None;
        }
        if worse <= 0.0 {
            return Some(0.0);
        }
        match self.delta {
            Some(d) if d > 0.0 => Some(worse / d),
            _ => None,
        }
    }
}

/// All per-prompt checks of one prompt.
#[derive(Clone, Debug)]
pub struct PromptVerdict {
    pub name: String,
    pub checks: Vec<PromptCheck>,
}

impl PromptVerdict {
    pub fn ok(&self) -> bool {
        self.checks.iter().all(|c| c.check.ok)
    }
}

fn upper_limit(fixed: f64, base: f64, delta: Option<f64>, k: f64) -> (f64, Source) {
    match delta {
        Some(d) if base + k * d > fixed => (base + k * d, Source::Envelope),
        _ => (fixed, Source::Fixed),
    }
}

fn lower_limit(fixed: f64, base: f64, delta: Option<f64>, k: f64) -> (f64, Source) {
    match delta {
        Some(d) if base - k * d < fixed => (base - k * d, Source::Envelope),
        _ => (fixed, Source::Fixed),
    }
}

/// Per-prompt checks of one mode, every prompt (not only the failing ones): the mean over a mode hides one prompt
/// that got much worse. Without `cal` the limits are the fixed margins; with it each limit is the baseline value
/// -/+ max(fixed margin, k x the prompt's measured noise). A prompt missing from either side is an error.
pub fn judge_prompts_calibrated(base: &[PromptStats], cur: &[PromptStats], m: &Margins, cal: Option<&Calibration>) -> Result<Vec<PromptVerdict>> {
    let mut out = Vec::new();
    for c in cur {
        let b = base
            .iter()
            .find(|b| b.name == c.name)
            .ok_or_else(|| anyhow!("prompt '{}' is not in the baseline (it has: {})", c.name, base.iter().map(|b| b.name.as_str()).collect::<Vec<_>>().join(", ")))?;
        let d = match cal {
            Some(cal) => Some((cal.k, (cal.delta)(&c.name).ok_or_else(|| anyhow!("prompt '{}' is not in the noise envelope", c.name))?)),
            None => None,
        };
        let (k, d) = d.map_or((0.0, None), |(k, d)| (k, Some(d)));
        let one_position = 100.0 / c.positions.max(1) as f64;
        let mut v = Vec::new();
        let fixed = b.mean_kl21 * m.prompt_kl_ratio + m.prompt_kl_eps;
        let (lim, source) = upper_limit(fixed, b.mean_kl21, d.map(|d| d.kl21), k);
        v.push(PromptCheck {
            check: Check { name: "prompt KL(21-way)", base: b.mean_kl21, cur: c.mean_kl21, limit: lim, higher_is_better: false, ok: c.mean_kl21 <= lim },
            delta: d.map(|d| d.kl21),
            source,
        });
        let fixed = b.top1_pct - m.prompt_top1_pts.max(one_position);
        let (lim, source) = lower_limit(fixed, b.top1_pct, d.map(|d| d.top1_pts), k);
        v.push(PromptCheck {
            check: Check { name: "prompt top-1 %", base: b.top1_pct, cur: c.top1_pct, limit: lim, higher_is_better: true, ok: c.top1_pct >= lim - 1e-9 },
            delta: d.map(|d| d.top1_pts),
            source,
        });
        let fixed = b.tail_abs + m.tail_abs;
        let (lim, source) = upper_limit(fixed, b.tail_abs, d.map(|d| d.tail_abs), k);
        v.push(PromptCheck {
            check: Check { name: "prompt |tail diff|", base: b.tail_abs, cur: c.tail_abs, limit: lim, higher_is_better: false, ok: c.tail_abs <= lim },
            delta: d.map(|d| d.tail_abs),
            source,
        });
        out.push(PromptVerdict { name: c.name.clone(), checks: v });
    }
    if base.len() != cur.len() {
        bail!("the baseline has {} prompts, this run {}", base.len(), cur.len());
    }
    Ok(out)
}

/// Per-prompt checks of one mode against the fixed margins: (prompt name, its checks) for every prompt that fails.
#[cfg(test)]
pub fn judge_prompts(base: &[PromptStats], cur: &[PromptStats], m: &Margins) -> Result<Vec<(String, Vec<Check>)>> {
    Ok(judge_prompts_calibrated(base, cur, m, None)?
        .into_iter()
        .filter(|p| !p.ok())
        .map(|p| (p.name, p.checks.into_iter().map(|c| c.check).collect()))
        .collect())
}

/// Baseline and run sides of a prefill chunk difference, naming where each value came from; when this run's value is a
/// recipe default the hint says how to run at the baseline's chunk instead.
pub fn chunk_texts(base: usize, base_source: &str, run: usize, run_source: &str) -> (String, String) {
    let with = |v: usize, src: &str| if src.is_empty() { v.to_string() } else { format!("{v} ({src})") };
    let mut b = with(run, run_source);
    if run_source.starts_with("recipe ") {
        b.push_str(&format!(" -- the recipe's default chunk; pass --prefill-chunk {base} to run at the baseline's"));
    }
    (with(base, base_source), b)
}

/// Compare identities. The golden must be the same (hash, size, shape); a configuration difference is an error
/// unless `allow_config_mismatch`, in which case the differences come back as warnings.
pub fn compatible(base: &Baseline, golden: &GoldenId, run: &RunConfig, exec: &ExecInfo, allow_config_mismatch: bool) -> Result<Vec<String>> {
    if base.format != FORMAT {
        bail!("baseline format '{}' is not {FORMAT}", base.format);
    }
    let g = &base.golden;
    if g.fnv1a64 != golden.fnv1a64 || g.size != golden.size || g.prompts != golden.prompts || g.positions != golden.positions {
        bail!(
            "the baseline was measured against a different golden file ({} {} bytes hash {}, {} prompts / {} positions) than this run's \
             ({} {} bytes hash {}, {} prompts / {} positions): record a new baseline for the new golden (--write-baseline)",
            g.file, g.size, g.fnv1a64, g.prompts, g.positions, golden.file, golden.size, golden.fnv1a64, golden.prompts, golden.positions
        );
    }
    let r = &base.run;
    let mut diffs = Vec::new();
    let mut d = |what: &str, a: &dyn std::fmt::Display, b: &dyn std::fmt::Display, strict: bool| {
        if a.to_string() != b.to_string() {
            diffs.push((format!("{what}: baseline {a}, this run {b}"), strict));
        }
    };
    d("model", &r.model, &run.model, false);
    d("backend", &r.backend, &run.backend, true);
    d("device", &r.device, &run.device, false);
    d("kv dtype", &r.kv_dtype, &run.kv_dtype, true);
    d("activation dtype", &r.act_dtype, &run.act_dtype, true);
    d("fp8 activation mode", &r.fp8_act, &run.fp8_act, true);
    if r.prefill_chunk != run.prefill_chunk {
        let (a, b) = chunk_texts(r.prefill_chunk, &r.prefill_chunk_source, run.prefill_chunk, &run.prefill_chunk_source);
        d("prefill chunk", &a, &b, true);
    }
    d("decode positions", &r.decode_positions, &run.decode_positions, true);
    // a baseline recorded with kernels switched off is not the pre-change code (review finding 1)
    d("kernel disable list", &base.exec.disable_list, &exec.disable_list, true);
    let strict: Vec<&String> = diffs.iter().filter(|d| d.1).map(|d| &d.0).collect();
    if !strict.is_empty() && !allow_config_mismatch {
        bail!(
            "the baseline was recorded with a different configuration ({}); record a new baseline or pass --allow-config-mismatch",
            strict.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("; ")
        );
    }
    Ok(diffs.into_iter().map(|d| d.0).collect())
}

pub fn write(path: &Path, b: &Baseline) -> Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    std::fs::write(path, serde_json::to_string_pretty(b)? + "\n").with_context(|| format!("writing {}", path.display()))
}

pub fn read(path: &Path) -> Result<Baseline> {
    let t = std::fs::read_to_string(path).with_context(|| format!("reading baseline {}", path.display()))?;
    serde_json::from_str(&t).map_err(|e| anyhow!("{}: not a baseline file: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(n: usize) -> ModeStats {
        ModeStats {
            mode: "prefill".into(),
            positions: n,
            top1_pct: 98.0,
            mean_kl: 6.5e-3,
            mean_kl21: 7.0e-3,
            top5_pct: 93.0,
            tail_ref: 0.050,
            tail_ours: 0.051,
            max_kl: 0.5,
            max_kl21: 0.4,
            tail_abs: 0.004,
            nonfinite: 0,
            greedy_match_pct: 60.0,
            floor_pass: true,
            ..Default::default()
        }
    }
    fn failed(v: &[Check]) -> Vec<&'static str> {
        v.iter().filter(|c| !c.ok).map(|c| c.name).collect()
    }

    #[test]
    fn fnv_known_vectors() {
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
    }

    #[test]
    fn a_run_passes_its_own_baseline_and_small_noise() {
        let b = stats(1441);
        assert!(failed(&judge(&b, &b, &Margins::default())).is_empty());
        let mut c = b.clone();
        c.mean_kl *= 1.10;
        c.mean_kl21 *= 1.10;
        c.top1_pct -= 0.2;
        c.top5_pct -= 0.3;
        assert!(failed(&judge(&b, &c, &Margins::default())).is_empty(), "within the margins");
    }

    #[test]
    fn each_metric_can_fail_the_gate() {
        let b = stats(1441);
        let m = Margins::default();
        let mut c = b.clone();
        c.top1_pct -= 0.5;
        assert_eq!(failed(&judge(&b, &c, &m)), ["top-1 %"]);
        let mut c = b.clone();
        c.mean_kl *= 1.3;
        assert_eq!(failed(&judge(&b, &c, &m)), ["mean KL"]);
        let mut c = b.clone();
        c.mean_kl21 *= 1.3;
        assert_eq!(failed(&judge(&b, &c, &m)), ["tail-bucket KL"]);
        let mut c = b.clone();
        c.top5_pct -= 0.8;
        assert_eq!(failed(&judge(&b, &c, &m)), ["top-5 %"]);
        let mut c = b.clone();
        c.tail_ours += 0.01;
        assert_eq!(failed(&judge(&b, &c, &m)), ["|tail excess|"]);
        let mut c = b.clone();
        c.tail_ours -= 0.01; // less tail mass than the reference is just as wrong
        assert_eq!(failed(&judge(&b, &c, &m)), ["|tail excess|"]);
        let mut c = b.clone();
        c.nonfinite = 1;
        assert_eq!(failed(&judge(&b, &c, &m)), ["non-finite rows"]);
        let mut c = b.clone();
        c.max_kl21 = 0.4 * 2.5 + 1e-3 + 1e-3; // one position got much worse; the mean did not move
        assert_eq!(failed(&judge(&b, &c, &m)), ["max KL(21-way)"]);
        let mut c = b.clone();
        c.tail_abs += 0.003; // positions that lose and gain tail mass cancel in the signed mean, not here
        assert_eq!(failed(&judge(&b, &c, &m)), ["mean |tail diff|"]);
        let mut c = b.clone();
        c.mean_kl = f64::NAN;
        assert_eq!(failed(&judge(&b, &c, &m)), ["mean KL"], "NaN never passes");
    }

    #[test]
    fn one_flipped_position_does_not_fail_a_short_mode() {
        // 160 decode positions: one flip is 0.625 pt, more than the 0.3 pt margin, and must still pass
        let mut b = stats(160);
        b.top1_pct = 98.75;
        let mut c = b.clone();
        c.top1_pct = 98.125;
        assert!(failed(&judge(&b, &c, &Margins::default())).is_empty());
        c.top1_pct = 96.875; // three flips
        assert_eq!(failed(&judge(&b, &c, &Margins::default())), ["top-1 %"]);
    }

    #[test]
    fn scored_row_kl_is_judged_only_against_a_baseline_that_has_it() {
        let mut b = stats(1000);
        let mut c = b.clone();
        c.mean_kl21_scored = Some(1e-2);
        // an older baseline (all-rows KL only): the scored number is not compared with it
        assert!(failed(&judge(&b, &c, &Margins::default())).is_empty());
        assert!(judge(&b, &c, &Margins::default()).iter().all(|k| k.name != "tail-bucket KL scored"));
        // a baseline with it: like with like
        b.mean_kl21_scored = Some(5e-3);
        assert_eq!(failed(&judge(&b, &c, &Margins::default())), ["tail-bucket KL scored"]);
        c.mean_kl21_scored = Some(5.5e-3);
        assert!(failed(&judge(&b, &c, &Margins::default())).is_empty());
    }

    #[test]
    fn an_old_baseline_json_without_scored_fields_still_reads() {
        let json = serde_json::to_string(&stats(10)).unwrap();
        let mut v: serde_json::Value = serde_json::from_str(&json).unwrap();
        for k in ["mean_kl_scored", "mean_kl21_scored", "scored_positions"] {
            v.as_object_mut().unwrap().remove(k);
        }
        let back: ModeStats = serde_json::from_value(v).unwrap();
        assert_eq!(back, stats(10));
    }

    #[test]
    fn a_tiny_baseline_kl_is_not_brittle() {
        let mut b = stats(1000);
        b.mean_kl = 1e-9;
        let mut c = b.clone();
        c.mean_kl = 5e-7; // 500x the baseline but below +1e-6
        assert!(failed(&judge(&b, &c, &Margins::default())).is_empty());
        c.mean_kl = 5e-6;
        assert_eq!(failed(&judge(&b, &c, &Margins::default())), ["mean KL"]);
    }

    fn pstats(name: &str, n: usize, kl21: f64) -> PromptStats {
        PromptStats { name: name.into(), positions: n, top1_pct: 95.0, mean_kl: kl21, mean_kl21: kl21, top5_pct: 90.0, tail_excess: 0.0, max_kl21: 0.1, tail_abs: 0.003, ..Default::default() }
    }

    #[test]
    fn one_prompt_doubling_fails_although_the_overall_mean_passes() {
        // review finding 2: 10 prompts, one doubles; the mean KL moves by ~10% and the x1.15 overall gate passes
        let m = Margins::default();
        let base: Vec<PromptStats> = (0..10).map(|i| pstats(&format!("p{i}"), 100, 6.5e-3)).collect();
        let mut cur = base.clone();
        cur[3].mean_kl21 *= 2.0;
        let mean = |v: &[PromptStats]| v.iter().map(|p| p.mean_kl21).sum::<f64>() / v.len() as f64;
        assert!(mean(&cur) <= mean(&base) * m.kl_ratio, "the aggregate gate cannot see it");
        let f = judge_prompts(&base, &cur, &m).unwrap();
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].0, "p3");
        assert!(f[0].1.iter().any(|c| c.name == "prompt KL(21-way)" && !c.ok));
        // noise on every prompt passes
        let mut noisy = base.clone();
        noisy.iter_mut().for_each(|p| p.mean_kl21 *= 1.1);
        assert!(judge_prompts(&base, &noisy, &m).unwrap().is_empty());
        // a single extra flip on a 16-position prompt is allowed, two are not
        let b16 = vec![pstats("p", 16, 6.5e-3)];
        let mut c16 = b16.clone();
        c16[0].top1_pct -= 6.25;
        assert!(judge_prompts(&b16, &c16, &m).unwrap().is_empty());
        c16[0].top1_pct -= 6.25;
        assert_eq!(judge_prompts(&b16, &c16, &m).unwrap().len(), 1);
        // a prompt that vanished is an error, not a pass
        assert!(judge_prompts(&base, &base[..9], &m).is_err());
        assert!(judge_prompts(&base[..9], &base, &m).is_err());
    }

    fn ids() -> (GoldenId, RunConfig) {
        (
            GoldenId {
                file: "g.safetensors".into(), fnv1a64: "00ff".into(), size: 10, format: "intarsia-golden-topk-v1".into(),
                prompts: 2, positions: 30, topk: Some(20), engine: "vllm".into(), engine_version: "0.1".into(),
                model: "m".into(), model_revision: "r".into(), kv_dtype: "fp8".into(),
            },
            RunConfig {
                model: "qwen".into(), backend: "cuda".into(), device: "GB10".into(), kv_dtype: "bf16".into(),
                act_dtype: "bf16".into(), fp8_act: "static".into(), prefill_chunk: 256, prefill_chunk_source: "default".into(), decode_positions: 16,
                gate_class: "4bit".into(),
            },
        )
    }
    fn baseline() -> Baseline {
        let (g, r) = ids();
        Baseline {
            format: FORMAT.into(),
            created_unix: 1,
            tool_version: "x".into(),
            code: CodeStamp::parse("head=4963b43 code-tree=aaaaaaaaaaaa dirty=no"),
            exec: ExecInfo::default(),
            golden: g,
            run: r,
            modes: vec![],
        }
    }

    #[test]
    fn mismatching_golden_or_config_is_refused() {
        let base = baseline();
        let (g, r) = ids();
        assert!(compatible(&base, &g, &r, &ExecInfo::default(), false).unwrap().is_empty());
        let mut g2 = g.clone();
        g2.fnv1a64 = "beef".into(); // a regenerated golden
        assert!(compatible(&base, &g2, &r, &ExecInfo::default(), false).unwrap_err().to_string().contains("different golden"));
        assert!(compatible(&base, &g2, &r, &ExecInfo::default(), true).is_err(), "the golden must match even with --allow-config-mismatch");
        let mut r2 = r.clone();
        r2.kv_dtype = "fp8_e4m3".into();
        let e = compatible(&base, &g, &r2, &ExecInfo::default(), false).unwrap_err().to_string();
        assert!(e.contains("kv dtype"), "{e}");
        assert_eq!(compatible(&base, &g, &r2, &ExecInfo::default(), true).unwrap().len(), 1);
        let mut r3 = r.clone();
        r3.device = "other GPU".into(); // only a warning
        assert_eq!(compatible(&base, &g, &r3, &ExecInfo::default(), false).unwrap().len(), 1);
        // a run with kernels disabled (or a baseline recorded that way) is not a comparison against the pre-change code
        let disabled = ExecInfo { disable_list: "INTARSIA_CUDA_DISABLE=moe_experts_mma_bf16".into(), ..Default::default() };
        let e = compatible(&base, &g, &r, &disabled, false).unwrap_err().to_string();
        assert!(e.contains("kernel disable list"), "{e}");
        assert_eq!(compatible(&base, &g, &r, &disabled, true).unwrap().len(), 1);
    }

    #[test]
    fn baseline_json_round_trips() {
        let mut b = baseline();
        b.modes.push(ModeBaseline {
            overall: stats(10),
            prompts: vec![PromptStats { name: "p".into(), positions: 10, top1_pct: 100.0, mean_kl: 0.0, mean_kl21: 0.0, top5_pct: 100.0, tail_excess: 0.0, max_kl21: 0.0, tail_abs: 0.0, ..Default::default() }],
        });
        let p = std::env::temp_dir().join(format!("intarsia-baseline-{}.json", std::process::id()));
        write(&p, &b).unwrap();
        assert_eq!(read(&p).unwrap(), b);
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn chunk_mismatch_names_the_recipe_default() {
        let (a, b) = chunk_texts(256, "default", 2048, "recipe qwen3_5_moe, cuda");
        assert_eq!(a, "256 (default)");
        assert!(b.starts_with("2048 (recipe qwen3_5_moe, cuda)") && b.contains("recipe's default chunk") && b.contains("--prefill-chunk 256"), "{b}");
        let (_, b) = chunk_texts(256, "", 2048, "--prefill-chunk");
        assert_eq!(b, "2048 (--prefill-chunk)");
        // a baseline recorded at 256 is refused against a run on the recipe's 2048, naming the recipe default
        let (g, mut r) = ids();
        r.prefill_chunk = 2048;
        r.prefill_chunk_source = "recipe qwen3_5_moe, cuda".into();
        let e = compatible(&baseline(), &g, &r, &ExecInfo::default(), false).unwrap_err().to_string();
        assert!(e.contains("baseline 256 (default), this run 2048 (recipe qwen3_5_moe, cuda)") && e.contains("pass --prefill-chunk 256"), "{e}");
        assert_eq!(compatible(&baseline(), &g, &r, &ExecInfo::default(), true).unwrap().len(), 1);
    }
}
