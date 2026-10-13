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
//! | non-finite rows | none |
//!
//! A baseline is only comparable with the same golden file (content hash and size) and the same run
//! configuration (KV dtype, activation mode, prefill chunk, decode positions, mode set); anything else is refused,
//! so a regenerated golden or a changed flag cannot make a stale baseline pass silently.

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const FORMAT: &str = "intarsia-accuracy-baseline-v1";

/// Aggregate of one mode (prefill / decode (teacher-forced)) over all prompts.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
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
    pub nonfinite: usize,
    pub greedy_match_pct: f64,
    /// Did the absolute class floor pass when this baseline was recorded?
    pub floor_pass: bool,
}

impl ModeStats {
    pub fn tail_excess(&self) -> f64 {
        self.tail_ours - self.tail_ref
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct PromptStats {
    pub name: String,
    pub positions: usize,
    pub top1_pct: f64,
    pub mean_kl: f64,
    pub mean_kl21: f64,
    pub top5_pct: f64,
    pub tail_excess: f64,
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
    pub decode_positions: usize,
    pub gate_class: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Baseline {
    pub format: String,
    pub created_unix: u64,
    pub tool_version: String,
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
}

impl Default for Margins {
    fn default() -> Self {
        Margins { top1_pts: 0.3, kl_ratio: 1.15, kl_eps: 1e-6, top5_pts: 0.5, tail_abs: 0.002 }
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
    let lim = base.top5_pct - m.top5_pts.max(one_position);
    v.push(Check { name: "top-5 %", base: base.top5_pct, cur: cur.top5_pct, limit: lim, higher_is_better: true, ok: cur.top5_pct >= lim - 1e-9 });
    let lim = base.tail_excess().abs() + m.tail_abs;
    v.push(Check { name: "|tail excess|", base: base.tail_excess().abs(), cur: cur.tail_excess().abs(), limit: lim, higher_is_better: false, ok: cur.tail_excess().abs() <= lim });
    v.push(Check { name: "non-finite rows", base: base.nonfinite as f64, cur: cur.nonfinite as f64, limit: 0.0, higher_is_better: false, ok: cur.nonfinite == 0 });
    v
}

/// Compare identities. The golden must be the same (hash, size, shape); a configuration difference is an error
/// unless `allow_config_mismatch`, in which case the differences come back as warnings.
pub fn compatible(base: &Baseline, golden: &GoldenId, run: &RunConfig, allow_config_mismatch: bool) -> Result<Vec<String>> {
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
    d("prefill chunk", &r.prefill_chunk, &run.prefill_chunk, true);
    d("decode positions", &r.decode_positions, &run.decode_positions, true);
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
            nonfinite: 0,
            greedy_match_pct: 60.0,
            floor_pass: true,
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
    fn a_tiny_baseline_kl_is_not_brittle() {
        let mut b = stats(1000);
        b.mean_kl = 1e-9;
        let mut c = b.clone();
        c.mean_kl = 5e-7; // 500x the baseline but below +1e-6
        assert!(failed(&judge(&b, &c, &Margins::default())).is_empty());
        c.mean_kl = 5e-6;
        assert_eq!(failed(&judge(&b, &c, &Margins::default())), ["mean KL"]);
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
                act_dtype: "bf16".into(), fp8_act: "static".into(), prefill_chunk: 256, decode_positions: 16,
                gate_class: "4bit".into(),
            },
        )
    }
    fn baseline() -> Baseline {
        let (g, r) = ids();
        Baseline { format: FORMAT.into(), created_unix: 1, tool_version: "x".into(), golden: g, run: r, modes: vec![] }
    }

    #[test]
    fn mismatching_golden_or_config_is_refused() {
        let base = baseline();
        let (g, r) = ids();
        assert!(compatible(&base, &g, &r, false).unwrap().is_empty());
        let mut g2 = g.clone();
        g2.fnv1a64 = "beef".into(); // a regenerated golden
        assert!(compatible(&base, &g2, &r, false).unwrap_err().to_string().contains("different golden"));
        assert!(compatible(&base, &g2, &r, true).is_err(), "the golden must match even with --allow-config-mismatch");
        let mut r2 = r.clone();
        r2.kv_dtype = "fp8_e4m3".into();
        let e = compatible(&base, &g, &r2, false).unwrap_err().to_string();
        assert!(e.contains("kv dtype"), "{e}");
        assert_eq!(compatible(&base, &g, &r2, true).unwrap().len(), 1);
        let mut r3 = r.clone();
        r3.device = "other GPU".into(); // only a warning
        assert_eq!(compatible(&base, &g, &r3, false).unwrap().len(), 1);
    }

    #[test]
    fn baseline_json_round_trips() {
        let mut b = baseline();
        b.modes.push(ModeBaseline {
            overall: stats(10),
            prompts: vec![PromptStats { name: "p".into(), positions: 10, top1_pct: 100.0, mean_kl: 0.0, mean_kl21: 0.0, top5_pct: 100.0, tail_excess: 0.0 }],
        });
        let p = std::env::temp_dir().join(format!("intarsia-baseline-{}.json", std::process::id()));
        write(&p, &b).unwrap();
        assert_eq!(read(&p).unwrap(), b);
        let _ = std::fs::remove_file(p);
    }
}
