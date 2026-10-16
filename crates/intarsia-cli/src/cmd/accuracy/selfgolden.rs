//! Same-weights regression gate: a golden made from *our own engine* (review finding 2).
//!
//! The vLLM / HF goldens measure our distance to another engine. For a 35B NVFP4 model against vLLM with an fp8 KV cache
//! the mean KL is about 6.5e-3, nearly all of it the reference's own noise, so a gate that only says "no worse than the
//! baseline x1.15" lets about 1e-3 nats of real regression through. When the weights are the same, the right question is
//! "did this change move the logits?", asked against the logits of the build before the change:
//!
//! 1. Build the pre-change commit (plus the harness overlay, `provenance.rs`) and run
//!    `intarsia accuracy --golden G --write-self-golden S.safetensors`. It dumps, for every prompt of G, the top-K
//!    log-probs of our own distribution (full-vocabulary normalised, so the mass outside the K ids is known) of the
//!    **prefill** path (every position) and of the **decode** path (the teacher-forced window: the single-token
//!    kernels, which are what graph replay and the decode MoE kernel changed).
//! 2. Build the new commit and run `intarsia accuracy --self-golden S.safetensors`. Each path is compared with the
//!    same path of the old build (never prefill with decode: that mixes in the difference between the two paths) with
//!    the existing comparator (top-1, KL on the K-way support, KL on the K+1-way partition with the tail bucket).
//!
//! The file is an ordinary `intarsia-golden-topk-v1` file for the prefill rows (`--golden S` works), plus decode
//! tensors `dec_pos_i`, `dec_topk_ids_i`, `dec_topk_logprobs_i`, and metadata `self_golden=1`, the code stamp and the
//! run configuration of the build that wrote it.
//!
//! Budgets (see docs/accuracy.md "Choosing the self-golden budgets" for how they were chosen) are on the mean 21-way
//! KL, the largest single-position KL, the share of positions whose argmax moved, the mean |tail mass difference| and
//! the mean KL of each prompt. They are flags.

use super::baseline::{ModeStats, PromptStats};
use super::provenance::{CodeStamp, ExecInfo};
use super::{lse, top_k, Mode};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

/// Default number of ids kept per position. 64 ids carry all but ~1e-4 of the mass of a normal position, so the
/// tail bucket stays small and the 65-way KL is close to the full-vocabulary KL.
pub const DEFAULT_TOPK: usize = 64;

/// Our top-`k` row of one position: ids (best first) and full-vocabulary log-probs.
pub fn capture_row(ours: &[f32], k: usize) -> (Vec<i32>, Vec<f32>) {
    let l = lse(ours.iter().map(|&v| v as f64));
    let ids = top_k(ours.iter().map(|&v| v as f64), k);
    let lps = ids.iter().map(|&i| (ours[i] as f64 - l) as f32).collect();
    (ids.into_iter().map(|i| i as i32).collect(), lps)
}

/// Rows captured for one prompt: (position, ids, log-probs) per compared position, in call order.
#[derive(Default, Clone)]
pub struct PromptCapture {
    pub prefill: Vec<(usize, Vec<i32>, Vec<f32>)>,
    pub decode: Vec<(usize, Vec<i32>, Vec<f32>)>,
}

/// Everything `--write-self-golden` collects while the modes run.
pub struct SelfCapture {
    pub k: usize,
    pub prompts: Vec<PromptCapture>,
}

impl SelfCapture {
    pub fn new(k: usize, prompts: usize) -> SelfCapture {
        SelfCapture { k, prompts: vec![PromptCapture::default(); prompts] }
    }
    pub fn push(&mut self, prompt: usize, mode: Mode, pos: usize, row: &[f32]) {
        let (ids, lps) = capture_row(row, self.k);
        let p = &mut self.prompts[prompt];
        match mode {
            Mode::Prefill => p.prefill.push((pos, ids, lps)),
            Mode::Decode => p.decode.push((pos, ids, lps)),
        }
    }
}

/// What a self-golden file says about the build and run that wrote it.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
pub struct SelfMeta {
    pub code: CodeStamp,
    pub exec: ExecInfo,
    pub model: String,
    pub backend: String,
    pub device: String,
    pub kv_dtype: String,
    pub act_dtype: String,
    pub fp8_act: String,
    pub prefill_chunk: usize,
    /// Where the chunk came from; absent in older files.
    #[serde(default)]
    pub prefill_chunk_source: String,
    pub decode_positions: usize,
    pub topk: usize,
    /// "prefill", "decode" or "prefill,decode".
    pub modes: String,
    pub source_golden: String,
    pub source_fnv1a64: String,
}

struct Tensor {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn i32s(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Pad a row to `k` entries the way the top-k goldens do (-1 ids, -1e30 log-probs).
fn padded(ids: &[i32], lps: &[f32], k: usize) -> (Vec<i32>, Vec<f32>) {
    let mut i = ids.to_vec();
    let mut l = lps.to_vec();
    i.resize(k, -1);
    l.resize(k, -1e30);
    (i, l)
}

/// Write a safetensors file (no dependency: the format is an 8-byte little-endian header length, a JSON header and the
/// raw tensor data; `golden.py` and `Golden::open` use the same layout).
fn write_safetensors(path: &Path, tensors: &[Tensor], meta: &BTreeMap<String, String>) -> Result<()> {
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".into(), serde_json::to_value(meta)?);
    let mut off = 0usize;
    for t in tensors {
        header.insert(
            t.name.clone(),
            serde_json::json!({"dtype": t.dtype, "shape": t.shape, "data_offsets": [off, off + t.bytes.len()]}),
        );
        off += t.bytes.len();
    }
    let mut hdr = serde_json::to_vec(&serde_json::Value::Object(header))?;
    while (8 + hdr.len()) % 8 != 0 {
        hdr.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + hdr.len() + off);
    out.extend_from_slice(&(hdr.len() as u64).to_le_bytes());
    out.extend_from_slice(&hdr);
    for t in tensors {
        out.extend_from_slice(&t.bytes);
    }
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    std::fs::write(path, out).with_context(|| format!("writing {}", path.display()))
}

/// Write the self-golden. `ids[i]` are the token ids of prompt `i`; `prompts_json` is the `prompts` metadata of the
/// source golden (names, lengths, reuse splits), passed through so `Golden::names` finds the names.
pub fn write(path: &Path, cap: &SelfCapture, ids: &[Vec<u32>], prompts_json: &str, meta: &SelfMeta) -> Result<()> {
    let k = cap.k;
    let mut ts = Vec::new();
    for (i, id) in ids.iter().enumerate() {
        let p = &cap.prompts[i];
        let n = id.len();
        ts.push(Tensor { name: format!("ids_{i}"), dtype: "I32", shape: vec![n], bytes: i32s(&id.iter().map(|&x| x as i32).collect::<Vec<_>>()) });
        // prefill rows: one per position, in order. A decode-only dump writes empty (all padding) rows so the file
        // stays a valid top-k golden.
        let mut by_pos: Vec<Option<&(usize, Vec<i32>, Vec<f32>)>> = vec![None; n];
        for r in &p.prefill {
            if r.0 >= n {
                bail!("prompt {i}: captured position {} >= {n}", r.0);
            }
            by_pos[r.0] = Some(r);
        }
        if !p.prefill.is_empty() && by_pos.iter().any(|r| r.is_none()) {
            bail!("prompt {i}: the prefill capture is missing positions");
        }
        let (mut idv, mut lpv) = (Vec::with_capacity(n * k), Vec::with_capacity(n * k));
        for r in &by_pos {
            let (a, b) = match r {
                Some(r) => padded(&r.1, &r.2, k),
                None => (vec![-1; k], vec![-1e30; k]),
            };
            idv.extend(a);
            lpv.extend(b);
        }
        ts.push(Tensor { name: format!("topk_ids_{i}"), dtype: "I32", shape: vec![n, k], bytes: i32s(&idv) });
        ts.push(Tensor { name: format!("topk_logprobs_{i}"), dtype: "F32", shape: vec![n, k], bytes: f32s(&lpv) });
        if !p.decode.is_empty() {
            let m = p.decode.len();
            let (mut di, mut dl) = (Vec::with_capacity(m * k), Vec::with_capacity(m * k));
            for r in &p.decode {
                let (a, b) = padded(&r.1, &r.2, k);
                di.extend(a);
                dl.extend(b);
            }
            ts.push(Tensor { name: format!("dec_pos_{i}"), dtype: "I32", shape: vec![m], bytes: i32s(&p.decode.iter().map(|r| r.0 as i32).collect::<Vec<_>>()) });
            ts.push(Tensor { name: format!("dec_topk_ids_{i}"), dtype: "I32", shape: vec![m, k], bytes: i32s(&di) });
            ts.push(Tensor { name: format!("dec_topk_logprobs_{i}"), dtype: "F32", shape: vec![m, k], bytes: f32s(&dl) });
        }
    }
    let mut m: BTreeMap<String, String> = BTreeMap::new();
    let mut put = |k: &str, v: String| {
        m.insert(k.to_string(), v);
    };
    put("format", "intarsia-golden-topk-v1".into());
    put("self_golden", "1".into());
    put("self_meta", serde_json::to_string(meta)?);
    put("engine", "intarsia".into());
    put("engine_version", meta.code.describe());
    put("model", meta.model.clone());
    put("kv_dtype", meta.kv_dtype.clone());
    put("topk", k.to_string());
    put("source", "intarsia".into());
    put("prompts", prompts_json.to_string());
    put("built_unix", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0).to_string());
    write_safetensors(path, &ts, &m)
}

// ---------------------------------------------------------------------------
// budgets and the verdict
// ---------------------------------------------------------------------------

/// How far a new build may move from the old build's logits on the same weights. Defaults: docs/accuracy.md.
#[derive(Clone, Copy, Debug)]
pub struct SelfBudget {
    /// Mean K+1-way KL(old || new) over all positions of a mode.
    pub max_mean_kl: f64,
    /// The largest single-position K+1-way KL.
    pub max_pos_kl: f64,
    /// Positions whose argmax differs from the old build's, percent.
    pub max_flip_pct: f64,
    /// Mean per-position |tail mass difference|.
    pub max_tail_abs: f64,
    /// Each prompt's mean KL may be this many times `max_mean_kl` (a prompt has far fewer positions than the mode).
    pub prompt_factor: f64,
}

/// The default budget. Chosen in docs/accuracy.md ("Choosing the self-golden budgets"): about 4x the KL a
/// legitimate bf16 reordering produced in the measured proxies, and about 1/5 of what a bug the size of the review's
/// examples (KV rounded to fp16 where the engine uses bf16 / a dropped rounding point) produced.
pub const DEFAULT_BUDGET: SelfBudget =
    SelfBudget { max_mean_kl: 1e-4, max_pos_kl: 5e-3, max_flip_pct: 0.5, max_tail_abs: 5e-4, prompt_factor: 5.0 };

#[derive(Clone, Debug, PartialEq)]
pub struct SelfCheck {
    pub name: String,
    pub value: f64,
    pub limit: f64,
    pub ok: bool,
}

fn chk(name: &str, value: f64, limit: f64) -> SelfCheck {
    SelfCheck { name: name.into(), value, limit, ok: value <= limit } // NaN never passes
}

/// Judge one mode against the old build. `flips_allowed` is at least one position: one flip must not fail a short mode.
pub fn judge(stats: &ModeStats, prompts: &[PromptStats], b: &SelfBudget) -> Vec<SelfCheck> {
    let n = stats.positions.max(1) as f64;
    let flips = n * (100.0 - stats.top1_pct) / 100.0;
    let flip_limit = (b.max_flip_pct / 100.0 * n).max(1.0);
    let mut v = vec![
        chk("mean KL(21-way)", stats.mean_kl21, b.max_mean_kl),
        chk("max position KL(21-way)", stats.max_kl21, b.max_pos_kl),
        chk("top-1 flips", (flips + 1e-9).floor(), (flip_limit + 1e-9).floor()),
        chk("mean |tail diff|", stats.tail_abs, b.max_tail_abs),
        chk("non-finite rows", stats.nonfinite as f64, 0.0),
    ];
    let worst = prompts.iter().max_by(|a, c| a.mean_kl21.partial_cmp(&c.mean_kl21).unwrap_or(std::cmp::Ordering::Equal));
    if let Some(w) = worst {
        v.push(chk(&format!("worst prompt KL(21-way) [{}]", w.name), w.mean_kl21, b.max_mean_kl * b.prompt_factor));
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(n: usize) -> ModeStats {
        ModeStats {
            mode: "decode".into(),
            positions: n,
            top1_pct: 100.0,
            mean_kl: 1e-6,
            mean_kl21: 1e-6,
            top5_pct: 100.0,
            tail_ref: 0.01,
            tail_ours: 0.01,
            max_kl: 1e-4,
            max_kl21: 1e-4,
            tail_abs: 1e-5,
            nonfinite: 0,
            greedy_match_pct: 100.0,
            floor_pass: true,
        }
    }
    fn prompt(name: &str, kl: f64) -> PromptStats {
        PromptStats { name: name.into(), positions: 16, top1_pct: 100.0, mean_kl: kl, mean_kl21: kl, top5_pct: 100.0, tail_excess: 0.0, max_kl21: kl, tail_abs: 0.0 }
    }
    fn failed(v: &[SelfCheck]) -> Vec<String> {
        v.iter().filter(|c| !c.ok).map(|c| c.name.clone()).collect()
    }

    #[test]
    fn identical_logits_pass_and_each_check_can_fail() {
        let b = DEFAULT_BUDGET;
        let s = stats(1441);
        assert!(failed(&judge(&s, &[prompt("a", 1e-6)], &b)).is_empty());
        let mut c = s.clone();
        c.mean_kl21 = 2e-4;
        assert_eq!(failed(&judge(&c, &[], &b)), ["mean KL(21-way)"]);
        let mut c = s.clone();
        c.max_kl21 = 1e-2;
        assert_eq!(failed(&judge(&c, &[], &b)), ["max position KL(21-way)"]);
        let mut c = s.clone();
        c.top1_pct = 100.0 - 100.0 * 10.0 / 1441.0; // 10 flips of 1441 = 0.69% > 0.5%
        assert_eq!(failed(&judge(&c, &[], &b)), ["top-1 flips"]);
        let mut c = s.clone();
        c.top1_pct = 100.0 - 100.0 * 7.0 / 1441.0; // 7 flips = 0.49%
        assert!(failed(&judge(&c, &[], &b)).is_empty());
        let mut c = s.clone();
        c.tail_abs = 1e-3;
        assert_eq!(failed(&judge(&c, &[], &b)), ["mean |tail diff|"]);
        let mut c = s.clone();
        c.nonfinite = 1;
        assert_eq!(failed(&judge(&c, &[], &b)), ["non-finite rows"]);
        let mut c = s.clone();
        c.mean_kl21 = f64::NAN;
        assert_eq!(failed(&judge(&c, &[], &b)), ["mean KL(21-way)"], "NaN never passes");
        // one prompt much worse than the rest, mean still inside the budget
        let c = judge(&s, &[prompt("a", 1e-6), prompt("b", 6e-4)], &b);
        assert_eq!(failed(&c), ["worst prompt KL(21-way) [b]"]);
    }

    #[test]
    fn one_flip_does_not_fail_a_short_mode() {
        let b = DEFAULT_BUDGET;
        let mut s = stats(160); // 0.5% of 160 is 0.8 positions: the floor of one flip applies
        s.top1_pct = 100.0 - 100.0 / 160.0;
        assert!(failed(&judge(&s, &[], &b)).is_empty());
        s.top1_pct = 100.0 - 200.0 / 160.0;
        assert_eq!(failed(&judge(&s, &[], &b)), ["top-1 flips"]);
    }

    #[test]
    fn captured_rows_are_normalised_log_probs_best_first() {
        let row: Vec<f32> = vec![0.0, 3.0, 1.0, 2.0, -1.0];
        let (ids, lps) = capture_row(&row, 3);
        assert_eq!(ids, [1, 3, 2]);
        let l = (0.0f64.exp() + 3.0f64.exp() + 1.0f64.exp() + 2.0f64.exp() + (-1.0f64).exp()).ln();
        assert!((lps[0] as f64 - (3.0 - l)).abs() < 1e-6);
        assert!(lps[0] > lps[1] && lps[1] > lps[2]);
        // fewer entries than k: ids come back short and `padded` fills them like the top-k goldens do
        let (ids, lps) = capture_row(&row, 8);
        assert_eq!(ids.len(), 5);
        let (pi, pl) = padded(&ids, &lps, 8);
        assert_eq!((pi[5], pl[5]), (-1, -1e30));
    }
}
