//! `intarsia accuracy`: accuracy gates (ARCHITECTURE section 6, docs/accuracy.md).
//!
//! * `--golden FILE` (default: `golden/<model>.safetensors`): compare our logits with a reference
//!   on fixed prompts: top-1 agreement, KL(ref||ours), top-5 overlap, greedy-match length. Two modes:
//!   `prefill` (one forward over the whole prompt, all positions) and `decode` (teacher-forced: prefill
//!   `n-k` tokens, then feed the golden ids one at a time through the single-token decode path).
//!   Reference files are full-vocabulary fp16 log-probs (`tools/golden.py`) or top-k log-probs
//!   (`tools/golden_vllm.py`, `golden.py --topk`).
//! * `--quick`: day-0 gate: perplexity on a fixed English text plus greedy sanity prompts.
//!
//! * KL is reported twice: over the reference's top-k support renormalised (as before), and over the **21-way
//!   partition** {the 20 reference top ids, everything else}: the "other" bucket is `1 - sum(top-k)` for the
//!   reference and our softmax mass outside the same ids for us. That one sees probability moving outside the
//!   top-k (the renormalised KL cannot) and is the gated KL for top-k goldens. The mass each side puts in the
//!   bucket ("tail") is printed and gated.
//! * `--baseline FILE` gates "no worse than a recorded run + margin" (`--write-baseline FILE` records one);
//!   the class thresholds stay as an absolute floor. See `baseline.rs`.
//! * `--mode reuse`: the prefix-reuse path through `Engine::generate` (turn 2 reusing turn 1 vs a cold run).
//! * `--write-self-golden FILE` / `--self-golden FILE`: dump this build's own top-k log-probs, and compare another build
//!   with such a dump on the same weights, with budgets tight enough to see a small change. See `selfgolden.rs`.
//! * Baselines and self-goldens record the code that produced them and are refused unless that is the declared
//!   pre-change commit (`--pre-change`) or its ancestor (`--revlist`). See `provenance.rs`.
//!
//! Any non-finite value in one of our logits rows fails the gate. Exits non-zero when any gate fails.

use crate::common::ModelArgs;
use anyhow::{anyhow, bail, Context, Result};
use intarsia_core::engine::{Engine, GenEvent, GenRequest};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

mod baseline;
mod provenance;
mod selfgolden;
use baseline::{Baseline, Margins, ModeStats, PromptStats};
use provenance::{CodeStamp, ExecInfo};

const PPL_TEXT: &str = include_str!("../../../../tools/prompts/ppl.txt");

#[derive(clap::Args, Debug)]
pub struct AccuracyArgs {
    #[command(flatten)]
    pub model: ModelArgs,
    /// Golden file written by tools/golden.py or tools/golden_vllm.py. Default: golden/<model name>.safetensors.
    #[arg(long)]
    pub golden: Option<PathBuf>,
    /// Day-0 quick gate (perplexity + greedy sanity prompts) instead of the golden comparison
    /// (give both --quick and --golden to run both).
    #[arg(long)]
    pub quick: bool,
    /// Golden comparison mode: `prefill`, `decode` (teacher-forced single-token steps), `both` (prefill and decode),
    /// `reuse` (prefix reuse through `generate`, cold vs reused) or `all`.
    #[arg(long, default_value = "both")]
    pub mode: String,
    /// Decode mode: positions per prompt fed one token at a time (clamped to prompt length - 1).
    #[arg(long, default_value_t = 16)]
    pub decode_positions: usize,
    /// Gate class that picks the default thresholds: `bf16`, `fp8` or `4bit`. Default: from the
    /// checkpoint's quantization metadata and the KV dtype (the lower precision of the two).
    #[arg(long)]
    pub gate_class: Option<String>,
    /// Golden gate: minimum top-1 agreement in percent. Default per class: bf16 99.5; fp8 / 4bit: not gated.
    #[arg(long)]
    pub min_top1: Option<f64>,
    /// Golden gate: maximum mean KL(ref||ours) in nats. Default per class: bf16 1e-3, fp8 5e-3, 4bit 3e-2.
    #[arg(long)]
    pub max_mean_kl: Option<f64>,
    /// Golden gate: maximum |mean(our tail mass - reference tail mass)|, probability points as a fraction (the tail is
    /// everything outside the reference's top-20). Absolute floor next to the KL gate; default 0.02.
    #[arg(long)]
    pub max_tail_excess: Option<f64>,
    /// Record this run as a baseline (JSON, e.g. golden/baselines/<name>.json). Written even when a floor gate fails
    /// (it then says so inside), before the run exits.
    #[arg(long)]
    pub write_baseline: Option<PathBuf>,
    /// Gate "no worse than this recorded baseline + margins" (the class thresholds stay as an absolute floor).
    /// Refuses to compare when the golden file or the run configuration differs from the baseline's.
    #[arg(long)]
    pub baseline: Option<PathBuf>,
    /// Baseline gate: allowed drop of top-1 agreement, percentage points (at least one position's worth).
    #[arg(long, default_value_t = 0.3)]
    pub margin_top1: f64,
    /// Baseline gate: allowed ratio of mean KL and tail-bucket KL to the baseline's (plus 1e-6 absolute).
    #[arg(long, default_value_t = 1.15)]
    pub margin_kl: f64,
    /// Baseline gate: allowed drop of top-5 overlap, percentage points (at least five overlap units).
    #[arg(long, default_value_t = 0.5)]
    pub margin_top5: f64,
    /// Baseline gate: allowed growth of |mean tail excess| over the baseline's, as a fraction of probability.
    #[arg(long, default_value_t = 0.002)]
    pub margin_tail: f64,
    /// Baseline gate: allowed ratio of the largest single-position 21-way KL to the baseline's (plus 1e-3 absolute).
    #[arg(long, default_value_t = 2.5)]
    pub margin_max_kl: f64,
    /// Baseline gate: allowed ratio of each prompt's mean 21-way KL to the baseline's (plus 1e-5 absolute).
    #[arg(long, default_value_t = 1.75)]
    pub margin_prompt_kl: f64,
    /// Compare against a baseline even when the run configuration (KV dtype, fp8 mode, chunk, ...) differs; the
    /// golden file must still be the same.
    #[arg(long)]
    pub allow_config_mismatch: bool,
    /// Dump this build's own distribution as a golden (top-k log-probs per position, prefill and decode paths) for
    /// `--self-golden` of another build. Needs `--golden` (its prompts). The file records the code stamp of this build.
    #[arg(long)]
    pub write_self_golden: Option<PathBuf>,
    /// Ids kept per position in a self-golden.
    #[arg(long, default_value_t = selfgolden::DEFAULT_TOPK)]
    pub self_topk: usize,
    /// Same-weights regression gate: compare this build with the logits another build dumped with
    /// `--write-self-golden` (its prompts; give no `--golden`). Budgets below.
    #[arg(long)]
    pub self_golden: Option<PathBuf>,
    /// Self-golden budget: mean 21-way KL(old || new), nats.
    #[arg(long, default_value_t = selfgolden::DEFAULT_BUDGET.max_mean_kl)]
    pub self_max_kl: f64,
    /// Self-golden budget: largest single-position 21-way KL, nats.
    #[arg(long, default_value_t = selfgolden::DEFAULT_BUDGET.max_pos_kl)]
    pub self_max_pos_kl: f64,
    /// Self-golden budget: positions whose argmax moved, percent (at least one position).
    #[arg(long, default_value_t = selfgolden::DEFAULT_BUDGET.max_flip_pct)]
    pub self_max_flip_pct: f64,
    /// Self-golden budget: mean per-position |tail mass difference|.
    #[arg(long, default_value_t = selfgolden::DEFAULT_BUDGET.max_tail_abs)]
    pub self_max_tail: f64,
    /// Self-golden budget: a prompt's mean KL may be this many times the mean-KL budget.
    #[arg(long, default_value_t = selfgolden::DEFAULT_BUDGET.prompt_factor)]
    pub self_prompt_factor: f64,
    /// The commit the "before" numbers (baseline, self-golden) must come from. The gate refuses when their code
    /// stamp is neither this commit nor (with `--revlist`) an ancestor of this build.
    #[arg(long)]
    pub pre_change: Option<String>,
    /// File with `git rev-list HEAD` of the tree this build was made from (written where the git history is: the build
    /// boxes get the tree without `.git`). Used to check that a baseline's code is an ancestor.
    #[arg(long)]
    pub revlist: Option<PathBuf>,
    /// Development: accept a baseline / self-golden, or a build, that carries no code stamp.
    #[arg(long)]
    pub allow_unstamped: bool,
    /// Check reproducibility: accept a baseline / self-golden from the same code tree as this build.
    #[arg(long)]
    pub allow_same_code: bool,
    /// Development: accept a build from a modified working tree.
    #[arg(long)]
    pub allow_dirty: bool,
    /// Reuse mode: tokens generated per comparison (the cold and the reuse run must agree on them).
    #[arg(long, default_value_t = 32)]
    pub reuse_gen: usize,
    /// Reuse mode: the reuse and cold token streams must agree on at least this many leading tokens
    /// (bf16 rounding can flip a near-tie when chunk boundaries differ; bit-identical paths agree on all).
    #[arg(long, default_value_t = 16)]
    pub reuse_min_match: usize,
    /// Quick gate: maximum perplexity on the fixed text.
    #[arg(long, default_value_t = 40.0)]
    pub max_ppl: f64,
    /// Quick gate: perplexity text file (default: built-in tools/prompts/ppl.txt).
    #[arg(long)]
    pub ppl_file: Option<PathBuf>,
    /// Quick gate: tokens to generate per sanity prompt.
    #[arg(long, default_value_t = 48)]
    pub sanity_tokens: usize,
}

// ---------------------------------------------------------------------------
// safetensors reading (golden file)
// ---------------------------------------------------------------------------

struct Golden {
    data: Vec<u8>,
    data_start: usize,
    tensors: HashMap<String, (String, Vec<usize>, usize, usize)>, // dtype, shape, begin, end
    meta: HashMap<String, String>,
}

/// Reference rows of one prompt.
enum RefRows {
    /// `[n, vocab]` log-probs as fp16 bits.
    Full { lp: Vec<u16>, vocab: usize },
    /// `[n, k]` ids (best first, -1 padded) and their full-vocabulary log-probs.
    TopK { ids: Vec<i32>, lps: Vec<f32>, k: usize },
}

struct PromptRef {
    ids: Vec<u32>,
    rows: RefRows,
}

enum RefRow<'a> {
    Full(&'a [u16]),
    TopK(&'a [i32], &'a [f32]),
}

impl PromptRef {
    fn row(&self, pos: usize) -> RefRow<'_> {
        match &self.rows {
            RefRows::Full { lp, vocab } => RefRow::Full(&lp[pos * vocab..(pos + 1) * vocab]),
            RefRows::TopK { ids, lps, k } => RefRow::TopK(&ids[pos * k..(pos + 1) * k], &lps[pos * k..(pos + 1) * k]),
        }
    }
}

impl Golden {
    fn open(path: &Path) -> Result<Golden> {
        let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        if data.len() < 8 {
            bail!("{}: too short for safetensors", path.display());
        }
        let hlen = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
        let hdr: Value = serde_json::from_slice(data.get(8..8 + hlen).ok_or_else(|| anyhow!("bad header length"))?)?;
        let mut tensors = HashMap::new();
        let mut meta = HashMap::new();
        for (k, v) in hdr.as_object().ok_or_else(|| anyhow!("header is not an object"))? {
            if k == "__metadata__" {
                for (mk, mv) in v.as_object().into_iter().flatten() {
                    if let Some(s) = mv.as_str() {
                        meta.insert(mk.clone(), s.to_string());
                    }
                }
                continue;
            }
            let dtype = v["dtype"].as_str().unwrap_or("").to_string();
            let shape = v["shape"].as_array().ok_or_else(|| anyhow!("{k}: no shape"))?;
            let shape: Vec<usize> = shape.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect();
            let off = v["data_offsets"].as_array().ok_or_else(|| anyhow!("{k}: no offsets"))?;
            let (b, e) = (off[0].as_u64().unwrap_or(0) as usize, off[1].as_u64().unwrap_or(0) as usize);
            tensors.insert(k.clone(), (dtype, shape, b, e));
        }
        Ok(Golden { data, data_start: 8 + hlen, tensors, meta })
    }

    fn raw(&self, name: &str, dtype: &str) -> Result<(&[u8], &[usize])> {
        let (dt, shape, b, e) = self.tensors.get(name).ok_or_else(|| anyhow!("golden file has no tensor `{name}`"))?;
        if dt != dtype {
            bail!("tensor `{name}` is {dt}, expected {dtype}");
        }
        let bytes = self
            .data
            .get(self.data_start + b..self.data_start + e)
            .ok_or_else(|| anyhow!("tensor `{name}` out of bounds"))?;
        Ok((bytes, shape))
    }

    fn is_topk(&self) -> bool {
        self.meta.get("format").is_some_and(|f| f == "intarsia-golden-topk-v1")
            || self.tensors.contains_key("topk_ids_0")
    }

    fn prompt(&self, i: usize) -> Result<PromptRef> {
        let (b, _) = self.raw(&format!("ids_{i}"), "I32")?;
        let ids: Vec<u32> = b.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32).collect();
        let rows = if self.is_topk() {
            let (ib, ishape) = self.raw(&format!("topk_ids_{i}"), "I32")?;
            let (lb, lshape) = self.raw(&format!("topk_logprobs_{i}"), "F32")?;
            if ishape.len() != 2 || ishape != lshape || ishape[0] != ids.len() {
                bail!("prompt {i}: topk shapes {ishape:?} / {lshape:?} for {} ids", ids.len());
            }
            RefRows::TopK {
                ids: ib.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect(),
                lps: lb.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
                k: ishape[1],
            }
        } else {
            let (b, shape) = self.raw(&format!("logprobs_{i}"), "F16")?;
            if shape.len() != 2 || shape[0] != ids.len() {
                bail!("logprobs_{i}: shape {shape:?} for {} ids", ids.len());
            }
            RefRows::Full { lp: b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect(), vocab: shape[1] }
        };
        Ok(PromptRef { ids, rows })
    }

    /// Written by `--write-self-golden`: this engine's own distribution rather than another engine's.
    fn is_self(&self) -> bool {
        self.meta.get("self_golden").is_some_and(|v| v == "1")
    }

    fn self_meta(&self) -> Result<selfgolden::SelfMeta> {
        let m = self.meta.get("self_meta").ok_or_else(|| anyhow!("self-golden without `self_meta` metadata"))?;
        serde_json::from_str(m).map_err(|e| anyhow!("self-golden `self_meta`: {e}"))
    }

    /// Self-golden, decode path: the rows of the teacher-forced window of prompt `i`, as a top-k reference with
    /// padding (-1 ids) everywhere else. `None`: the file has no decode rows for this prompt.
    fn self_decode_prompt(&self, i: usize) -> Result<Option<PromptRef>> {
        if !self.tensors.contains_key(&format!("dec_pos_{i}")) {
            return Ok(None);
        }
        let (b, _) = self.raw(&format!("ids_{i}"), "I32")?;
        let ids: Vec<u32> = b.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32).collect();
        let (pb, _) = self.raw(&format!("dec_pos_{i}"), "I32")?;
        let pos: Vec<usize> = pb.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as usize).collect();
        let (ib, ishape) = self.raw(&format!("dec_topk_ids_{i}"), "I32")?;
        let (lb, lshape) = self.raw(&format!("dec_topk_logprobs_{i}"), "F32")?;
        if ishape.len() != 2 || ishape != lshape || ishape[0] != pos.len() {
            bail!("prompt {i}: decode topk shapes {ishape:?} / {lshape:?} for {} positions", pos.len());
        }
        let k = ishape[1];
        let (n, mut tids, mut lps) = (ids.len(), vec![-1i32; ids.len() * k], vec![-1e30f32; ids.len() * k]);
        for (r, &p) in pos.iter().enumerate() {
            if p >= n {
                bail!("prompt {i}: decode position {p} >= {n}");
            }
            for j in 0..k {
                tids[p * k + j] = i32::from_le_bytes(ib[(r * k + j) * 4..(r * k + j) * 4 + 4].try_into().unwrap());
                lps[p * k + j] = f32::from_le_bytes(lb[(r * k + j) * 4..(r * k + j) * 4 + 4].try_into().unwrap());
            }
        }
        Ok(Some(PromptRef { ids, rows: RefRows::TopK { ids: tids, lps, k } }))
    }

    fn names(&self) -> Vec<String> {
        let parsed: Option<Vec<Value>> = self.meta.get("prompts").and_then(|s| serde_json::from_str(s).ok());
        let n = (0..).take_while(|i| self.tensors.contains_key(&format!("ids_{i}"))).count();
        (0..n)
            .map(|i| {
                parsed
                    .as_ref()
                    .and_then(|p| p.get(i))
                    .and_then(|p| p["name"].as_str())
                    .map(String::from)
                    .unwrap_or_else(|| format!("prompt{i}"))
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// metrics
// ---------------------------------------------------------------------------

fn f16_to_f64(b: u16) -> f64 {
    half::f16::from_bits(b).to_f64()
}

/// Indices of the `k` largest values, ties to the lowest index, ordered best first.
/// NaN entries are skipped (callers reject non-finite rows before ranking).
fn top_k(x: impl Iterator<Item = f64>, k: usize) -> Vec<usize> {
    let mut best: Vec<(f64, usize)> = Vec::with_capacity(k + 1);
    for (i, v) in x.enumerate() {
        if v.is_nan() {
            continue;
        }
        if best.len() == k && v <= best[k - 1].0 {
            continue;
        }
        let pos = best.iter().position(|b| v > b.0).unwrap_or(best.len());
        best.insert(pos, (v, i));
        best.truncate(k);
    }
    best.into_iter().map(|b| b.1).collect()
}

/// Statistics of one compared position.
#[derive(Clone, Debug)]
struct PosStat {
    /// Our argmax equals the reference argmax (strict; ties are reported separately).
    top1: bool,
    /// Not equal, but the reference scores both tokens identically (fp16 resolution).
    tie: bool,
    /// |top5(ref) and top5(ours)| (at most 5, or k when the reference has fewer).
    overlap: usize,
    n_top: usize,
    /// KL(ref||ours): `Full` golden: over the whole vocabulary; `TopK` golden: over the reference's top-k support,
    /// both sides renormalised there (blind to mass moving outside it).
    kl: f64,
    /// KL over the 21-way partition {reference top-`TAIL_K` ids (top-k golden: its k ids), everything else}; the
    /// "other" bucket is `1 - sum(top-k)` for the reference and our softmax mass outside those ids for us. By the
    /// chain rule this is a true lower bound of the full-vocabulary KL and it sees mass that leaves the top-k.
    kl21: f64,
    /// Reference probability mass outside the support S, and ours.
    tail_ref: f64,
    tail_ours: f64,
    /// Our row has a NaN or an infinity: the position fails whatever the other numbers say.
    nonfinite: bool,
}

/// Support size of the tail bucket for full-vocabulary goldens (a top-k golden uses its own k).
const TAIL_K: usize = 20;

#[derive(Default, Clone, Debug)]
struct Agg {
    n: usize,
    top1: usize,
    ties: usize,
    top5: usize, // summed overlap counts
    top5_den: usize,
    kl_sum: f64,
    kl_max: f64,
    kl21_sum: f64,
    kl21_max: f64,
    tail_ref_sum: f64,
    tail_ours_sum: f64,
    /// Sum over positions of |tail ours - tail reference|.
    tail_abs_sum: f64,
    nonfinite: usize,
}

impl Agg {
    fn push(&mut self, p: &PosStat) {
        self.n += 1;
        self.top1 += p.top1 as usize;
        self.ties += p.tie as usize;
        self.top5 += p.overlap;
        self.top5_den += p.n_top;
        self.kl_sum += p.kl;
        self.kl_max = self.kl_max.max(p.kl);
        self.kl21_sum += p.kl21;
        self.kl21_max = self.kl21_max.max(p.kl21);
        self.tail_ref_sum += p.tail_ref;
        self.tail_ours_sum += p.tail_ours;
        self.tail_abs_sum += (p.tail_ours - p.tail_ref).abs();
        self.nonfinite += p.nonfinite as usize;
    }
    fn add(&mut self, o: &Agg) {
        self.n += o.n;
        self.top1 += o.top1;
        self.ties += o.ties;
        self.top5 += o.top5;
        self.top5_den += o.top5_den;
        self.kl_sum += o.kl_sum;
        self.kl_max = self.kl_max.max(o.kl_max);
        self.kl21_sum += o.kl21_sum;
        self.kl21_max = self.kl21_max.max(o.kl21_max);
        self.tail_ref_sum += o.tail_ref_sum;
        self.tail_ours_sum += o.tail_ours_sum;
        self.tail_abs_sum += o.tail_abs_sum;
        self.nonfinite += o.nonfinite;
    }
    fn top1_pct(&self) -> f64 {
        100.0 * self.top1 as f64 / self.n.max(1) as f64
    }
    fn mean_kl(&self) -> f64 {
        self.kl_sum / self.n.max(1) as f64
    }
    fn top5_pct(&self) -> f64 {
        100.0 * self.top5 as f64 / self.top5_den.max(1) as f64
    }
    fn mean_kl21(&self) -> f64 {
        self.kl21_sum / self.n.max(1) as f64
    }
    fn mean_tail_abs(&self) -> f64 {
        self.tail_abs_sum / self.n.max(1) as f64
    }
    fn mean_tail_ref(&self) -> f64 {
        self.tail_ref_sum / self.n.max(1) as f64
    }
    fn mean_tail_ours(&self) -> f64 {
        self.tail_ours_sum / self.n.max(1) as f64
    }
    /// Mean of (our tail mass - reference tail mass): positive when we put more probability outside the top-k.
    fn tail_excess(&self) -> f64 {
        (self.tail_ours_sum - self.tail_ref_sum) / self.n.max(1) as f64
    }
}

fn lse(x: impl Iterator<Item = f64> + Clone) -> f64 {
    let m = x.clone().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    m + x.map(|v| (v - m).exp()).sum::<f64>().ln()
}

fn is_finite_row(row: &[f32]) -> bool {
    row.iter().all(|v| v.is_finite())
}

/// `ours` softmax split by a support: (log-sum-exp of the whole row, probability mass outside `support`, and the
/// probabilities of the support ids in order). The tail is summed directly, not as `1 - sum(support)`.
fn ours_split(ours: &[f32], support: &[usize]) -> (f64, f64, Vec<f64>) {
    let l = lse(ours.iter().map(|&v| v as f64));
    let mut mask = vec![false; ours.len()];
    for &i in support {
        mask[i] = true;
    }
    let tail: f64 = ours.iter().zip(&mask).filter(|(_, &m)| !m).map(|(&v, _)| (v as f64 - l).exp()).sum();
    (l, tail, support.iter().map(|&i| (ours[i] as f64 - l).exp()).collect())
}

/// KL over the partition {support ids, tail}: `p`, `q` are the probabilities of the support ids (in the same order)
/// and `pt`, `qt` the tail masses. A reference tail of zero contributes nothing; a tail of ours of zero against a
/// reference tail above zero is floored at 1e-300 (a huge but finite KL).
fn kl_partition(p: &[f64], q: &[f64], pt: f64, qt: f64) -> f64 {
    let mut kl = 0.0;
    for (&pi, &qi) in p.iter().zip(q) {
        if pi > 0.0 {
            kl += pi * (pi.ln() - qi.max(1e-300).ln());
        }
    }
    if pt > 0.0 {
        kl += pt * (pt.ln() - qt.max(1e-300).ln());
    }
    kl.max(0.0) // rounding can leave -1e-17
}

/// Compare one position. `ours` are raw logits. A non-finite value anywhere in `ours` makes the
/// position fail (KL = infinity, top-1 miss). A NaN in the reference is a broken golden: error.
///
/// * `Full`: the fp16 reference is renormalised (cancels fp16 rounding bias); `kl` over the whole vocab. The tail
///   bucket (`kl21`, masses) uses the reference's top-`TAIL_K` ids as the support, so the numbers are comparable
///   with top-k goldens.
/// * `TopK`: `kl` over the reference's top-k support, both sides renormalised there (blind to mass that moves
///   outside); `kl21` over the 21-way partition with an explicit "other" bucket on both sides, `1 - sum(top-k)` for
///   the reference (read from the full-vocabulary log-probs it reports) and our softmax mass outside the k ids;
///   top-1 is our full-vocabulary argmax against the reference's best id.
fn compare_pos(reference: RefRow<'_>, ours: &[f32]) -> Result<PosStat> {
    let finite = is_finite_row(ours);
    let o = || ours.iter().map(|&v| v as f64);
    let bad = |n_top: usize| PosStat {
        top1: false,
        tie: false,
        overlap: 0,
        n_top,
        kl: f64::INFINITY,
        kl21: f64::INFINITY,
        tail_ref: 0.0,
        tail_ours: 0.0,
        nonfinite: true,
    };
    match reference {
        RefRow::Full(r) => {
            if r.iter().any(|&b| half::f16::from_bits(b).is_nan()) {
                bail!("the golden row contains NaN");
            }
            if r.len() != ours.len() {
                bail!("golden row has {} entries, our logits {}", r.len(), ours.len());
            }
            let rf = || r.iter().map(|&b| f16_to_f64(b));
            let rt = top_k(rf(), 5);
            if !finite {
                return Ok(bad(rt.len()));
            }
            let (rl, ol) = (lse(rf()), lse(o()));
            let mut kl = 0.0;
            for (&rb, &ov) in r.iter().zip(ours) {
                let lr = f16_to_f64(rb) - rl;
                if lr.is_finite() {
                    kl += lr.exp() * (lr - (ov as f64 - ol));
                }
            }
            let kl = kl.max(0.0); // renormalisation rounding can leave -1e-17
            // tail bucket over the reference's top-TAIL_K ids
            let support = top_k(rf(), TAIL_K);
            let (_, q_tail, q) = ours_split(ours, &support);
            let p: Vec<f64> = support.iter().map(|&i| (f16_to_f64(r[i]) - rl).exp()).collect();
            let p_tail = (1.0 - p.iter().sum::<f64>()).max(0.0);
            let kl21 = kl_partition(&p, &q, p_tail, q_tail);
            let ot = top_k(o(), 5);
            let top1 = ot[0] == rt[0];
            let tie = !top1 && f16_to_f64(r[ot[0]]) >= f16_to_f64(r[rt[0]]);
            let overlap = ot.iter().filter(|i| rt.contains(i)).count();
            Ok(PosStat { top1, tie, overlap, n_top: rt.len(), kl, kl21, tail_ref: p_tail, tail_ours: q_tail, nonfinite: false })
        }
        RefRow::TopK(ids, lps) => {
            let valid: Vec<(usize, f64)> =
                ids.iter().zip(lps).filter(|(&i, _)| i >= 0).map(|(&i, &l)| (i as usize, l as f64)).collect();
            if valid.is_empty() {
                bail!("the golden row has no entries");
            }
            if valid.iter().any(|(_, l)| l.is_nan()) {
                bail!("the golden row contains NaN");
            }
            if valid.iter().any(|(i, _)| *i >= ours.len()) {
                bail!(
                    "golden id {} outside our vocabulary of {}",
                    valid.iter().map(|v| v.0).max().unwrap(),
                    ours.len()
                );
            }
            let ref_top = valid.iter().fold(valid[0], |b, &x| if x.1 > b.1 { x } else { b }).0;
            let rt: Vec<usize> = {
                let mut v = valid.clone();
                v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
                v.into_iter().take(5).map(|x| x.0).collect()
            };
            if !finite {
                return Ok(bad(rt.len()));
            }
            // distributions over the support, both renormalised there
            let lr_all = lse(valid.iter().map(|v| v.1));
            let lo: Vec<f64> = valid.iter().map(|v| ours[v.0] as f64).collect();
            let lo_all = lse(lo.iter().copied());
            let mut kl = 0.0;
            for (v, &l) in valid.iter().zip(&lo) {
                let lp = v.1 - lr_all;
                kl += lp.exp() * (lp - (l - lo_all));
            }
            let kl = kl.max(0.0);
            // 21-way partition: the reference's log-probs are full-vocabulary, so its tail is what the k ids leave;
            // rounding of the reported log-probs can push the sum a hair over 1 (then rescale, tail 0)
            let support: Vec<usize> = valid.iter().map(|v| v.0).collect();
            let mut p: Vec<f64> = valid.iter().map(|v| v.1.exp()).collect();
            let ps: f64 = p.iter().sum();
            if ps > 1.0 {
                p.iter_mut().for_each(|x| *x /= ps);
            }
            let p_tail = (1.0 - p.iter().sum::<f64>()).max(0.0);
            let (_, q_tail, q) = ours_split(ours, &support);
            let kl21 = kl_partition(&p, &q, p_tail, q_tail);
            let ot = top_k(o(), 5);
            let overlap = ot.iter().filter(|i| rt.contains(i)).count();
            Ok(PosStat {
                top1: ot[0] == ref_top,
                tie: false,
                overlap,
                n_top: rt.len(),
                kl,
                kl21,
                tail_ref: p_tail,
                tail_ours: q_tail,
                nonfinite: false,
            })
        }
    }
}

/// Number of leading positions (in order) where our argmax equals the reference argmax: how many
/// tokens greedy decoding tracks the reference before the first divergence.
fn greedy_match_len(stats: &[PosStat]) -> usize {
    stats.iter().take_while(|s| s.top1).count()
}

// ---------------------------------------------------------------------------
// gates
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum GateClass {
    /// Lowest precision first so `min` picks the looser class.
    Fp4,
    Fp8,
    Bf16,
}

impl GateClass {
    fn parse(s: &str) -> Result<GateClass> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "bf16" | "fp16" | "f16" | "f32" | "fp32" | "float" => GateClass::Bf16,
            "fp8" => GateClass::Fp8,
            "4bit" | "fp4" | "nvfp4" | "mxfp4" | "int4" => GateClass::Fp4,
            o => bail!("unknown gate class '{o}' (bf16, fp8, 4bit)"),
        })
    }
    fn name(self) -> &'static str {
        match self {
            GateClass::Bf16 => "bf16",
            GateClass::Fp8 => "fp8",
            GateClass::Fp4 => "4bit",
        }
    }
    /// ARCHITECTURE section 6: BF16 vs HF top-1 >= 99.5% and mean KL <= 1e-3; FP8 KL <= 5e-3;
    /// 4-bit KL <= 3e-2. Section 6 sets no top-1 threshold for FP8 / 4-bit, so none is gated there
    /// (top-1 is still reported; `--min-top1` adds one).
    fn defaults(self) -> (Option<f64>, f64) {
        match self {
            GateClass::Bf16 => (Some(99.5), 1e-3),
            GateClass::Fp8 => (None, 5e-3),
            GateClass::Fp4 => (None, 3e-2),
        }
    }
}

fn collect_strings(v: &Value, out: &mut Vec<String>) {
    match v {
        Value::String(s) => out.push(s.to_ascii_lowercase()),
        Value::Number(_) => {}
        Value::Array(a) => a.iter().for_each(|x| collect_strings(x, out)),
        Value::Object(o) => {
            for (k, x) in o {
                out.push(k.to_ascii_lowercase());
                // bit widths count only under the keys that mean "bits" (not group sizes and the like)
                if matches!(k.as_str(), "num_bits" | "bits" | "weight_bits" | "w_bit") {
                    if let Some(n) = x.as_u64() {
                        out.push(format!("bits={n}"));
                    }
                }
                collect_strings(x, out);
            }
        }
        _ => {}
    }
}

/// Weight precision class of a checkpoint from its quantization metadata (`quantization_config` in
/// config.json, `hf_quant_config.json`). No metadata: a dense float checkpoint (bf16 class).
/// Mixed checkpoints take the lowest precision found.
fn weight_class(dir: &Path) -> Result<(GateClass, String)> {
    let mut docs: Vec<Value> = Vec::new();
    if let Ok(t) = std::fs::read_to_string(dir.join("config.json")) {
        if let Ok(c) = serde_json::from_str::<Value>(&t) {
            for v in [c.get("quantization_config"), c.get("text_config").and_then(|t| t.get("quantization_config"))]
                .into_iter()
                .flatten()
            {
                docs.push(v.clone());
            }
        }
    }
    if let Ok(t) = std::fs::read_to_string(dir.join("hf_quant_config.json")) {
        if let Ok(c) = serde_json::from_str::<Value>(&t) {
            docs.push(c);
        }
    }
    if docs.is_empty() {
        return Ok((GateClass::Bf16, "no quantization metadata".into()));
    }
    let mut strs = Vec::new();
    docs.iter().for_each(|d| collect_strings(d, &mut strs));
    let has = |pat: &str| strs.iter().any(|s| s.contains(pat));
    let has_exact = |pat: &str| strs.iter().any(|s| s == pat);
    if has("fp4") || has("int4") || has("4bit") || has("w4a") || has_exact("bits=4") {
        Ok((GateClass::Fp4, "quantization metadata names a 4-bit format".into()))
    } else if has("fp8") || has("float8") || has("e4m3") || has_exact("bits=8") {
        Ok((GateClass::Fp8, "quantization metadata names an 8-bit format".into()))
    } else {
        bail!(
            "{}: unrecognised quantization metadata; say which defaults apply with --gate-class bf16|fp8|4bit",
            dir.display()
        )
    }
}

struct Gates {
    class: GateClass,
    why: String,
    min_top1: Option<f64>,
    /// Maximum mean KL: the 21-way (tail-bucket) KL for top-k goldens, the full-vocabulary KL otherwise.
    max_kl: f64,
    /// Maximum |mean(our tail mass - reference tail mass)|.
    max_tail_excess: f64,
}

/// Default absolute floor on the tail-mass difference (all classes): a bug that moves a few percent of the
/// probability outside the reference's top-k fails, quantisation noise (measured, docs/accuracy.md) does not.
const DEFAULT_MAX_TAIL_EXCESS: f64 = 0.02;

fn resolve_gates(args: &AccuracyArgs, kv_dtype: &str) -> Result<Gates> {
    let (class, why) = match &args.gate_class {
        Some(c) => (GateClass::parse(c)?, "--gate-class".to_string()),
        None => {
            let (wc, wwhy) = weight_class(&args.model.model)?;
            let kc = match kv_dtype {
                "fp8_e4m3" | "fp8_e5m2" => GateClass::Fp8,
                _ => GateClass::Bf16,
            };
            (wc.min(kc), format!("weights: {} ({}), kv cache: {kv_dtype}", wc.name(), wwhy))
        }
    };
    let (t1, kl) = class.defaults();
    Ok(Gates {
        class,
        why,
        min_top1: args.min_top1.or(t1),
        max_kl: args.max_mean_kl.unwrap_or(kl),
        max_tail_excess: args.max_tail_excess.unwrap_or(DEFAULT_MAX_TAIL_EXCESS),
    })
}

// ---------------------------------------------------------------------------
// golden gate
// ---------------------------------------------------------------------------

fn default_golden(model: &Path) -> Option<PathBuf> {
    let full = model.to_string_lossy().to_string();
    let name = match full.split("models--").nth(1) {
        Some(rest) => rest.split('/').next().unwrap_or("").rsplit("--").next().unwrap_or("").to_string(),
        None => model.file_name()?.to_string_lossy().to_string(),
    }
    .to_lowercase();
    let file = format!("{name}.safetensors");
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../golden");
    [PathBuf::from("golden"), repo].into_iter().map(|d| d.join(&file)).find(|p| p.exists())
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    Prefill,
    Decode,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Mode::Prefill => "prefill",
            Mode::Decode => "decode (teacher-forced)",
        }
    }
    /// Key of the mode inside a baseline file.
    fn key(self) -> &'static str {
        match self {
            Mode::Prefill => "prefill",
            Mode::Decode => "decode",
        }
    }
}

struct ModeResult {
    /// Per-prompt leading run of argmax agreement and the number of positions compared.
    matches: Vec<(usize, usize)>,
    /// The absolute (class) floor passed.
    pass: bool,
    stats: ModeStats,
    prompts: Vec<PromptStats>,
}

// ---------------------------------------------------------------------------
// golden metadata
// ---------------------------------------------------------------------------

/// Canonical name of a dtype string (HF `bfloat16`, vLLM `fp8`, our `bf16`, ...), so aliases compare equal.
fn norm_dtype(s: &str) -> String {
    let l = s.trim().to_ascii_lowercase();
    match l.as_str() {
        "bfloat16" | "bf16" => "bf16".into(),
        "float16" | "fp16" | "f16" | "half" => "f16".into(),
        "float32" | "fp32" | "f32" | "float" => "f32".into(),
        "auto" => "auto".into(),
        _ if l.contains("e5m2") => "fp8_e5m2".into(),
        _ if l == "fp8" || l.contains("e4m3") || l == "fp8e4m3" => "fp8_e4m3".into(),
        _ => l,
    }
}

/// Warning text when the reference's KV cache precision differs from ours (or is not recorded), else `None`.
/// A reference that ran an fp8 KV cache against our bf16 one cannot be told apart from model error.
fn kv_dtype_warning(golden_kv: Option<&str>, ours: &str) -> Option<String> {
    let o = norm_dtype(ours);
    match golden_kv.map(norm_dtype).as_deref() {
        None | Some("") | Some("unknown") => Some(format!(
            "WARNING: the golden does not record the reference's KV cache dtype (it predates the metadata, or the server did not say); \
             we run a {o} KV cache, and a KV precision difference (e.g. an fp8 reference cache) would be indistinguishable from model error"
        )),
        // vLLM's `auto` is the model dtype: a 16-bit cache. Fine against any 16/32-bit cache of ours.
        Some("auto") if matches!(o.as_str(), "bf16" | "f16" | "f32") => None,
        Some(g) if g == o => None,
        Some(g) => Some(format!(
            "WARNING: KV dtype mismatch: the reference ran with a {g} KV cache, we run {o}. Part of the KL / top-1 gap is KV \
             quantisation noise of the reference, not model error; rebuild the golden with a matching KV dtype to separate them"
        )),
    }
}

impl Golden {
    /// The `prompts` metadata entry of prompt `i` (name, n_tokens, n_prompt, reuse_split, ...).
    fn prompt_meta(&self, i: usize) -> Option<Value> {
        let parsed: Vec<Value> = serde_json::from_str(self.meta.get("prompts")?).ok()?;
        parsed.into_iter().nth(i)
    }

    fn meta_str(&self, k: &str) -> Option<&str> {
        self.meta.get(k).map(String::as_str).filter(|s| !s.is_empty())
    }

    /// One line per fact the golden's metadata records about its reference.
    fn describe(&self) -> Vec<String> {
        let m = |k: &str| self.meta_str(k).unwrap_or("-");
        let engine = self.meta_str("engine").or(self.meta_str("source")).unwrap_or("hf (older golden)");
        let mut out = vec![format!(
            "reference: engine {} {} | model {} | revision {} | dtype {} | KV dtype {} | template {}",
            engine,
            self.meta_str("engine_version").unwrap_or_else(|| self.meta_str("transformers").unwrap_or("-")),
            m("model"),
            m("model_revision"),
            m("dtype"),
            m("kv_dtype"),
            m("template"),
        )];
        if let Some(c) = self.meta_str("cache_config_info") {
            out.push(format!("reference cache config: {c}"));
        }
        out.push(format!(
            "sampling/prompts: {} | prompt files {} | add_special_tokens {} | max_model_len {} | launch flags: {}",
            m("sampling"),
            m("prompt_files"),
            m("add_special_tokens"),
            m("max_model_len"),
            m("launch_flags"),
        ));
        out
    }
}

// ---------------------------------------------------------------------------
// run one mode
// ---------------------------------------------------------------------------

/// Run one mode over all prompts and judge it against `gates`.
fn run_mode(
    engine: &dyn Engine,
    prompts: &[(String, PromptRef)],
    mode: Mode,
    args: &AccuracyArgs,
    gates: &Gates,
    topk: bool,
    // same-weights comparison with another build's dump: judged with these budgets instead of the class floor
    self_budget: Option<&selfgolden::SelfBudget>,
    // `--write-self-golden`: collects our own top-k rows while the mode runs
    capture: &mut Option<selfgolden::SelfCapture>,
) -> Result<ModeResult> {
    let vocab = engine.info().vocab;
    println!("\n== mode: {} ==", mode.name());
    let kl_name = if topk { "KL(supp)" } else { "KL(full)" };
    let row = |name: &str, n: String, pos: usize, a: &Agg, greedy: String| {
        println!(
            "{:<18} {:>5} {:>5} {:>8.2} {:>10.3e} {:>10.3e} {:>10.3e} {:>8.4} {:>8.4} {:>8.2} {:>9} {:>6}",
            name,
            n,
            pos,
            a.top1_pct(),
            a.mean_kl(),
            a.mean_kl21(),
            a.kl_max,
            a.mean_tail_ref(),
            a.mean_tail_ours(),
            a.top5_pct(),
            greedy,
            a.nonfinite
        );
    };
    println!(
        "{:<18} {:>5} {:>5} {:>8} {:>10} {:>10} {:>10} {:>8} {:>8} {:>8} {:>9} {:>6}",
        "prompt", "n", "pos", "top-1 %", kl_name, "KL(21-way)", "max KL", "tail ref", "tail us", "top-5 %", "greedy", "nonfin"
    );
    let mut total = Agg::default();
    let mut matches = Vec::new();
    let mut per_prompt = Vec::new();
    // development aid: `INTARSIA_ACC_POS_DUMP=<file>` appends one TSV line per position (mode, prompt, pos, token id,
    // top-1 agrees, 21-way KL, tail masses) so a divergence can be located against chunk / tile boundaries
    let mut pos_dump = std::env::var_os("INTARSIA_ACC_POS_DUMP")
        .and_then(|f| std::fs::OpenOptions::new().create(true).append(true).open(f).ok());
    for (pi, (name, p)) in prompts.iter().enumerate() {
        let n = p.ids.len();
        let mut stats: Vec<PosStat> = Vec::new();
        let mut cb = |pos: usize, row: &[f32]| -> intarsia_core::Result<()> {
            if row.len() != vocab {
                return Err(intarsia_core::Error::model(format!("logits row has {} entries, want {vocab}", row.len())));
            }
            if let Some(c) = capture.as_mut() {
                c.push(pi, mode, pos, row);
            }
            let s = compare_pos(p.row(pos), row)
                .map_err(|e| intarsia_core::Error::model(format!("{name} position {pos}: {e}")))?;
            if let Some(f) = pos_dump.as_mut() {
                use std::io::Write as _;
                let _ = writeln!(
                    f,
                    "{}\t{}\t{}\t{}\t{}\t{:.6e}\t{:.6e}\t{:.6e}",
                    mode.name(),
                    name,
                    pos,
                    p.ids[pos],
                    s.top1 as u8,
                    s.kl21,
                    s.tail_ref,
                    s.tail_ours
                );
            }
            stats.push(s);
            Ok(())
        };
        match mode {
            Mode::Prefill => engine.forward_logits_each(&p.ids, &mut cb),
            Mode::Decode => {
                let k = args.decode_positions.min(n - 1).max(1);
                engine.forward_logits_decode(&p.ids, k, &mut cb)
            }
        }
        .map_err(|e| anyhow!("{name}: {e}"))?;
        let want = if mode == Mode::Prefill { n } else { args.decode_positions.min(n - 1).max(1) };
        if stats.len() != want {
            bail!("{name}: engine returned {} positions, want {want}", stats.len());
        }
        let mut agg = Agg::default();
        stats.iter().for_each(|s| agg.push(s));
        let ml = greedy_match_len(&stats);
        row(name, n.to_string(), stats.len(), &agg, format!("{ml}/{}", stats.len()));
        per_prompt.push(PromptStats {
            name: name.clone(),
            positions: agg.n,
            top1_pct: agg.top1_pct(),
            mean_kl: agg.mean_kl(),
            mean_kl21: agg.mean_kl21(),
            top5_pct: agg.top5_pct(),
            tail_excess: agg.tail_excess(),
            max_kl21: agg.kl21_max,
            tail_abs: agg.mean_tail_abs(),
        });
        matches.push((ml, stats.len()));
        total.add(&agg);
    }
    let greedy_pct = 100.0 * matches.iter().map(|m| m.0).sum::<usize>() as f64 / total.n.max(1) as f64;
    row("OVERALL", String::new(), total.n, &total, format!("{greedy_pct:.1}%"));
    let mstats = |pass: bool| ModeStats {
        mode: mode.key().into(),
        positions: total.n,
        top1_pct: total.top1_pct(),
        mean_kl: total.mean_kl(),
        mean_kl21: total.mean_kl21(),
        top5_pct: total.top5_pct(),
        tail_ref: total.mean_tail_ref(),
        tail_ours: total.mean_tail_ours(),
        max_kl: total.kl_max,
        max_kl21: total.kl21_max,
        tail_abs: total.mean_tail_abs(),
        nonfinite: total.nonfinite,
        greedy_match_pct: greedy_pct,
        floor_pass: pass,
    };
    if let Some(b) = self_budget {
        // same weights: the class floors are not the question, the distance to the other build's logits is
        let checks = selfgolden::judge(&mstats(true), &per_prompt, b);
        let pass = checks.iter().all(|c| c.ok);
        println!("{} vs the other build's logits (same weights):", mode.name());
        for c in &checks {
            println!("  {:<44} {:>12.4e}  (limit {:.4e})  {}", c.name, c.value, c.limit, if c.ok { "ok" } else { "FAIL" });
        }
        // one machine-readable line per mode for scripts/phase1b-acceptance.sh (noise calibration of the budgets)
        let flips = total.n - total.top1;
        println!(
            "SELF-STATS mode={} positions={} mean_kl21={:.6e} max_kl21={:.6e} flips={} flip_pct={:.4} tail_abs={:.6e} worst_prompt_kl21={:.6e}",
            mode.key(),
            total.n,
            total.mean_kl21(),
            total.kl21_max,
            flips,
            100.0 * flips as f64 / total.n.max(1) as f64,
            total.mean_tail_abs(),
            per_prompt.iter().map(|p| p.mean_kl21).fold(0.0, f64::max)
        );
        println!("{} self-golden => {}", mode.name(), if pass { "PASS" } else { "FAIL" });
        return Ok(ModeResult { matches, pass, stats: mstats(pass), prompts: per_prompt });
    }
    let ok_nf = total.nonfinite == 0;
    let ok_top1 = gates.min_top1.is_none_or(|m| total.top1_pct() >= m);
    let (gate_kl, gate_kl_name) = if topk { (total.mean_kl21(), "21-way KL") } else { (total.mean_kl(), "full-vocab KL") };
    let ok_kl = gate_kl <= gates.max_kl; // NaN / inf compare false
    let ok_tail = total.tail_excess().abs() <= gates.max_tail_excess;
    let tag = |ok: bool| if ok { "ok" } else { "FAIL" };
    let pass = ok_nf && ok_top1 && ok_kl && ok_tail;
    println!(
        "{} floor [{}]: top-1 {:.2}%{} ({}) {} | {gate_kl_name} {:.3e} (need <= {:.1e}) {} | tail mass ours {:.4} vs ref {:.4}, excess {:+.4} (|.| <= {}) {} | non-finite rows {} {} => {}",
        mode.name(),
        gates.class.name(),
        total.top1_pct(),
        if total.ties > 0 {
            format!(
                " strict ({:.2}% if the {} fp16 reference ties counted as agreement)",
                100.0 * (total.top1 + total.ties) as f64 / total.n.max(1) as f64,
                total.ties
            )
        } else {
            String::new()
        },
        match gates.min_top1 {
            Some(m) => format!("need >= {m}"),
            None => "not gated for this class".into(),
        },
        tag(ok_top1),
        gate_kl,
        gates.max_kl,
        tag(ok_kl),
        total.mean_tail_ours(),
        total.mean_tail_ref(),
        total.tail_excess(),
        gates.max_tail_excess,
        tag(ok_tail),
        total.nonfinite,
        tag(ok_nf),
        if pass { "PASS" } else { "FAIL" },
    );
    let stats = mstats(pass);
    Ok(ModeResult { matches, pass, stats, prompts: per_prompt })
}

/// Print the baseline comparison of one mode; returns whether every check passed.
fn print_baseline_checks(mode: Mode, base: &baseline::ModeBaseline, cur: &ModeResult, m: &Margins) -> bool {
    let checks = baseline::judge(&base.overall, &cur.stats, m);
    println!("\nbaseline gate, {} ({} positions):", mode.name(), cur.stats.positions);
    println!("  {:<16} {:>12} {:>12} {:>12}  {}", "metric", "baseline", "this run", "limit", "");
    for c in &checks {
        println!(
            "  {:<16} {:>12.5e} {:>12.5e} {:>12.5e}  {} {}",
            c.name,
            c.base,
            c.cur,
            c.limit,
            if c.higher_is_better { ">=" } else { "<=" },
            if c.ok { "ok" } else { "FAIL" }
        );
    }
    let mut ok = checks.iter().all(|c| c.ok);
    // every prompt on its own: the mode aggregate hides one prompt that got much worse
    let bad = match baseline::judge_prompts(&base.prompts, &cur.prompts, m) {
        Ok(b) => b,
        Err(e) => {
            println!("  per-prompt gate: FAIL ({e})");
            return false;
        }
    };
    if bad.is_empty() {
        println!("  per-prompt gate ({} prompts: KL x{} +{:.0e}, top-1 -{} pt or one position, |tail diff| +{}): ok", cur.prompts.len(), m.prompt_kl_ratio, m.prompt_kl_eps, m.prompt_top1_pts, m.tail_abs);
    }
    for (name, cs) in &bad {
        ok = false;
        for c in cs.iter().filter(|c| !c.ok) {
            println!(
                "  prompt {:<18} {:<19} baseline {:>11.4e}  this run {:>11.4e}  limit {} {:>11.4e}  FAIL",
                name,
                c.name,
                c.base,
                c.cur,
                if c.higher_is_better { ">=" } else { "<=" },
                c.limit
            );
        }
    }
    ok
}

// ---------------------------------------------------------------------------
// prefix reuse through `generate`
// ---------------------------------------------------------------------------

/// Greedy generation of `max_tokens` tokens (EOS ignored) and the number of prompt tokens served from the cache.
fn gen_tokens(engine: &dyn Engine, prompt: Vec<u32>, max_tokens: usize) -> Result<(Vec<u32>, usize)> {
    let req = GenRequest { prompt, max_tokens, ignore_eos: true, temperature: 0.0, ..Default::default() };
    let (mut toks, mut cached, mut done, mut err) = (Vec::<u32>::new(), 0usize, 0usize, None);
    engine
        .generate(req, &mut |ev| match ev {
            GenEvent::Token { id, .. } => toks.push(id),
            GenEvent::Done { cached_tokens, completion_tokens, .. } => {
                cached = cached_tokens;
                done = completion_tokens;
            }
            GenEvent::Error { message } => err = Some(message),
        })
        .map_err(|e| anyhow!("{e}"))?;
    if let Some(m) = err {
        bail!("generation failed: {m}");
    }
    toks.truncate(done); // a held-back text remainder arrives as one extra event repeating the last id
    Ok((toks, cached))
}

/// Replace whatever the engine holds in its cache by an unrelated short prompt whose first token differs from `first`.
fn flush_cache(engine: &dyn Engine, first: u32, vocab: usize) -> Result<()> {
    let head = ((first as usize + 1) % vocab.max(2)) as u32;
    let prompt: Vec<u32> = std::iter::once(head).chain((1..8).map(|i| (i * 31 % vocab.max(2)) as u32)).collect();
    gen_tokens(engine, prompt, 1)?;
    Ok(())
}

fn first_divergence(a: &[u32], b: &[u32]) -> Option<usize> {
    let n = a.len().min(b.len());
    (0..n).find(|&i| a[i] != b[i]).or((a.len() != b.len()).then_some(n))
}

struct ReuseRow {
    prompt: String,
    case: &'static str,
    split: usize,
    turn2: usize,
    cached: usize,
    cold_cached: usize,
    agree: usize,
    of: usize,
    ok: bool,
    note: String,
}

/// The prefix-reuse path of a real model, compared with a cold run, through `Engine::generate` (what `serve` runs):
///
/// * `continue`: turn 1 = `ids[..s]` and a real reply of the model; turn 2 = turn 1 + reply + the rest of the prompt.
///   The engine must reuse everything it holds (`cached >= s`) and produce the same greedy tokens as a cold run.
/// * `restore`: turn 1 = `ids[..s]` + a different tail; turn 2 = `ids`. The common prefix is `s`, shorter than what
///   is held, so the cache is truncated (attention) or a recurrent snapshot is restored (GatedDeltaNet / Mamba,
///   taken at prefill chunk boundaries): the engine reuses `cached > 0` tokens and again must agree with a cold run.
///
/// The cold run is forced by serving an unrelated prompt in between (`cached == 0` is checked). Agreement is
/// "at least `--reuse-min-match` identical leading tokens": a reuse run chunks the prompt differently from a cold one, so
/// bf16 rounding can flip a near-tie late in the stream; a broken restore diverges at once.
fn reuse_gate(engine: &dyn Engine, g: &Golden, prompts: &[(String, PromptRef)], args: &AccuracyArgs) -> Result<bool> {
    let info = engine.info();
    let (chunk, ctx, vocab, m) = (args.model.prefill_chunk, info.context_len, info.vocab, args.reuse_gen.max(1));
    let reply = 16usize;
    println!("\n== mode: prefix reuse through generate (turn 2 reuses turn 1, vs a cold run) ==");
    println!(
        "{:<18} {:<9} {:>6} {:>6} {:>7} {:>6} {:>9}  {}",
        "prompt", "case", "split", "turn2", "cached", "cold", "agree", "result"
    );
    let mut rows: Vec<ReuseRow> = Vec::new();
    for (i, (name, p)) in prompts.iter().enumerate() {
        let n = p.ids.len();
        let meta = g.prompt_meta(i);
        let n_prompt = meta.as_ref().and_then(|v| v["n_prompt"].as_u64()).map_or(n, |x| x as usize);
        let split = meta.as_ref().and_then(|v| v["reuse_split"].as_u64()).map_or(n_prompt / 2, |x| x as usize).min(n.saturating_sub(2));
        let skip = |case: &'static str, why: String| {
            println!("{name:<18} {case:<9} skipped: {why}");
        };
        if split < chunk {
            skip("both", format!("split {split} is below one prefill chunk ({chunk}): nothing to reuse across chunk boundaries"));
            continue;
        }
        // ---- continue
        {
            let run = || -> Result<ReuseRow> {
                flush_cache(engine, p.ids[0], vocab)?;
                let (reply_toks, _) = gen_tokens(engine, p.ids[..split].to_vec(), reply)?;
                let mut turn2: Vec<u32> = p.ids[..split].to_vec();
                turn2.extend_from_slice(&reply_toks);
                turn2.extend_from_slice(&p.ids[split..]);
                turn2.truncate(ctx.saturating_sub(m + 8));
                let (warm, cached) = gen_tokens(engine, turn2.clone(), m)?;
                flush_cache(engine, p.ids[0], vocab)?;
                let (cold, cold_cached) = gen_tokens(engine, turn2.clone(), m)?;
                let agree = first_divergence(&warm, &cold).unwrap_or(warm.len());
                let ok = cached >= split && cold_cached == 0 && agree >= args.reuse_min_match.min(m);
                let note = if cached < split {
                    format!("reuse covered {cached} < the {split} tokens of turn 1")
                } else if cold_cached != 0 {
                    "the cold run was not cold".into()
                } else if agree < args.reuse_min_match.min(m) {
                    format!("streams diverge at token {agree}")
                } else {
                    String::new()
                };
                Ok(ReuseRow { prompt: name.clone(), case: "continue", split, turn2: turn2.len(), cached, cold_cached, agree, of: m, ok, note })
            };
            rows.push(run()?);
        }
        // ---- restore
        {
            let run = || -> Result<ReuseRow> {
                flush_cache(engine, p.ids[0], vocab)?;
                let mut turn1: Vec<u32> = p.ids[..split].to_vec();
                // a different tail: every token differs from the one at that place of the real prompt
                turn1.extend(p.ids[split..].iter().take(40).map(|&t| ((t as usize + 1) % vocab) as u32));
                gen_tokens(engine, turn1, 4)?;
                let mut turn2 = p.ids.clone();
                turn2.truncate(ctx.saturating_sub(m + 8));
                let (warm, cached) = gen_tokens(engine, turn2.clone(), m)?;
                flush_cache(engine, p.ids[0], vocab)?;
                let (cold, cold_cached) = gen_tokens(engine, turn2.clone(), m)?;
                let agree = first_divergence(&warm, &cold).unwrap_or(warm.len());
                let ok = cached > 0 && cached <= split && cold_cached == 0 && agree >= args.reuse_min_match.min(m);
                let note = if cached == 0 {
                    "nothing was reused (no restorable state at or below the split)".into()
                } else if cached > split {
                    format!("reused {cached} > the shared prefix {split}: stale state")
                } else if cold_cached != 0 {
                    "the cold run was not cold".into()
                } else if agree < args.reuse_min_match.min(m) {
                    format!("streams diverge at token {agree}")
                } else {
                    String::new()
                };
                Ok(ReuseRow { prompt: name.clone(), case: "restore", split, turn2: turn2.len(), cached, cold_cached, agree, of: m, ok, note })
            };
            rows.push(run()?);
        }
        for r in rows.iter().rev().take(2).collect::<Vec<_>>().into_iter().rev() {
            println!(
                "{:<18} {:<9} {:>6} {:>6} {:>7} {:>6} {:>9}  {}{}",
                r.prompt,
                r.case,
                r.split,
                r.turn2,
                r.cached,
                r.cold_cached,
                format!("{}/{}", r.agree, r.of),
                if r.ok { "ok" } else { "FAIL" },
                if r.note.is_empty() { String::new() } else { format!(" ({})", r.note) }
            );
        }
    }
    if rows.is_empty() {
        println!("reuse gate: SKIPPED (no prompt is long enough: it needs a split of at least one prefill chunk, --prefill-chunk {chunk}); not a pass");
        return Ok(false);
    }
    let ok = rows.iter().all(|r| r.ok);
    println!("prefix-reuse gate: {}", if ok { "PASS" } else { "FAIL" });
    Ok(ok)
}

// ---------------------------------------------------------------------------
// golden gate
// ---------------------------------------------------------------------------

/// The provenance policy from the flags (the revlist file is read here).
fn provenance_policy(args: &AccuracyArgs) -> Result<provenance::Policy> {
    let revlist = match &args.revlist {
        Some(p) => std::fs::read_to_string(p)
            .with_context(|| format!("reading --revlist {}", p.display()))?
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect(),
        None => Vec::new(),
    };
    Ok(provenance::Policy {
        pre_change: args.pre_change.clone(),
        revlist,
        allow_unstamped: args.allow_unstamped,
        allow_same_code: args.allow_same_code,
        allow_dirty: args.allow_dirty,
    })
}

/// A self-golden is comparable only with the same weights and the same way of running them.
fn self_compat(sm: &selfgolden::SelfMeta, run: &baseline::RunConfig, allow_mismatch: bool) -> Result<Vec<String>> {
    let mut diffs: Vec<(String, bool)> = Vec::new();
    let mut d = |what: &str, a: &dyn std::fmt::Display, b: &dyn std::fmt::Display, strict: bool| {
        if a.to_string() != b.to_string() {
            diffs.push((format!("{what}: self-golden {a}, this run {b}"), strict));
        }
    };
    d("model", &sm.model, &run.model, true);
    d("backend", &sm.backend, &run.backend, true);
    d("device", &sm.device, &run.device, false);
    d("kv dtype", &sm.kv_dtype, &run.kv_dtype, true);
    d("activation dtype", &sm.act_dtype, &run.act_dtype, true);
    d("fp8 activation mode", &sm.fp8_act, &run.fp8_act, true);
    d("prefill chunk", &sm.prefill_chunk, &run.prefill_chunk, true);
    d("decode positions", &sm.decode_positions, &run.decode_positions, true);
    let strict: Vec<&String> = diffs.iter().filter(|d| d.1).map(|d| &d.0).collect();
    if !strict.is_empty() && !allow_mismatch {
        bail!(
            "the self-golden was written with a different configuration ({}); write a new one or pass --allow-config-mismatch",
            strict.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("; ")
        );
    }
    Ok(diffs.into_iter().map(|d| d.0).collect())
}

fn golden_gate(engine: &dyn Engine, args: &AccuracyArgs) -> Result<bool> {
    if args.golden.is_some() && args.self_golden.is_some() {
        bail!("--golden and --self-golden are separate runs (the self-golden carries its own prompts); run them one after the other");
    }
    let self_mode = args.self_golden.is_some();
    if self_mode && (args.baseline.is_some() || args.write_baseline.is_some() || args.write_self_golden.is_some()) {
        bail!("--self-golden cannot be combined with --baseline, --write-baseline or --write-self-golden");
    }
    if args.write_self_golden.is_some() && args.golden.is_none() {
        bail!("--write-self-golden needs --golden: the dump uses that golden's prompts (and teacher-forced ids)");
    }
    let path = match (&args.self_golden, &args.golden) {
        (Some(p), _) | (None, Some(p)) => p.clone(),
        (None, None) => default_golden(&args.model.model).ok_or_else(|| {
            anyhow!("no golden file found for this model; generate one with tools/golden.py and pass --golden")
        })?,
    };
    let g = Golden::open(&path)?;
    if self_mode && !g.is_self() {
        bail!("{}: not a self-golden (no self_golden metadata); write one with --write-self-golden", path.display());
    }
    let names = g.names();
    if names.is_empty() {
        bail!("{}: no ids_N tensors", path.display());
    }
    let (want_prefill, want_decode, want_reuse) = match args.mode.as_str() {
        "prefill" => (true, false, false),
        "decode" => (false, true, false),
        "both" => (true, true, false),
        "reuse" => (false, false, true),
        "all" => (true, true, true),
        o => bail!("unknown --mode '{o}' (prefill, decode, both, reuse, all)"),
    };
    let mut modes: Vec<Mode> = Vec::new();
    if want_prefill {
        modes.push(Mode::Prefill);
    }
    if want_decode {
        modes.push(Mode::Decode);
    }
    let info = engine.info();
    let gates = resolve_gates(args, &info.kv_dtype)?;
    let self_meta = if self_mode { Some(g.self_meta()?) } else { None };
    if let Some(sm) = &self_meta {
        if want_reuse {
            bail!("--mode reuse needs the source golden's reuse splits; a self-golden covers the prefill and decode paths only");
        }
        for m in &modes {
            if !sm.modes.split(',').any(|x| x == m.key()) {
                bail!("the self-golden has no '{}' rows (it has: {}); write it with --mode {}", m.key(), sm.modes, if modes.len() > 1 { "both" } else { m.key() });
            }
        }
    }
    let mut prompts = Vec::new();
    let mut prompts_dec = Vec::new(); // self-golden: the decode path has its own rows
    for (i, name) in names.iter().enumerate() {
        if self_mode {
            if let Some(d) = g.self_decode_prompt(i)? {
                prompts_dec.push((name.clone(), d));
            }
        }
        let p = g.prompt(i)?;
        if let RefRows::Full { vocab, .. } = &p.rows {
            if *vocab != info.vocab {
                bail!("{name}: golden vocab {vocab} != model vocab {}", info.vocab);
            }
        }
        prompts.push((name.clone(), p));
    }
    let longest = prompts.iter().map(|p| p.1.ids.len()).max().unwrap_or(0);
    let positions: usize = prompts.iter().map(|p| p.1.ids.len()).sum();
    let chunk = args.model.prefill_chunk;
    let multi = prompts.iter().filter(|p| p.1.ids.len() > chunk).count();
    let topk_k = match &prompts[0].1.rows {
        RefRows::TopK { k, .. } => Some(*k),
        _ => None,
    };
    eprintln!(
        "golden: {} ({} prompts, {} positions, {})",
        path.display(),
        names.len(),
        positions,
        match topk_k {
            Some(k) => format!(
                "top-{k} log-probs: KL(supp) over the top-{k} support renormalised, KL(21-way) with an explicit other bucket"
            ),
            None => format!("full-vocabulary log-probs; the 21-way KL / tail use the reference top-{TAIL_K} as support"),
        },
    );
    for l in g.describe() {
        eprintln!("  {l}");
    }
    if !self_mode {
        if let Some(w) = kv_dtype_warning(g.meta_str("kv_dtype"), &info.kv_dtype) {
            eprintln!("{w}");
        }
        eprintln!(
            "gates: class {} ({}); top-1 {}, {} <= {:.1e}, |tail excess| <= {}",
            gates.class.name(),
            gates.why,
            gates.min_top1.map_or("not gated".to_string(), |m| format!(">= {m}%")),
            if topk_k.is_some() { "21-way KL" } else { "KL" },
            gates.max_kl,
            gates.max_tail_excess,
        );
    }
    eprintln!(
        "prefill chunk {chunk}; longest prompt {longest}; prompts spanning more than one chunk: {multi}/{}",
        prompts.len()
    );
    if multi == 0 {
        eprintln!("WARNING: no prompt crosses a prefill chunk boundary, so multi-chunk prefill is untested; use --prefill-chunk {} or a golden with longer prompts", (longest / 3).max(1));
    }
    // Identity of this run, and the baseline (read and checked BEFORE the long runs: a baseline for another golden or
    // another configuration is refused up front).
    let golden_id = baseline::GoldenId {
        file: path.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default(),
        fnv1a64: format!("{:016x}", baseline::fnv1a64(&g.data)),
        size: g.data.len() as u64,
        format: g.meta_str("format").unwrap_or(if topk_k.is_some() { "intarsia-golden-topk-v1" } else { "intarsia-golden-v1" }).to_string(),
        prompts: names.len(),
        positions,
        topk: topk_k,
        engine: g.meta_str("engine").or(g.meta_str("source")).unwrap_or("").to_string(),
        engine_version: g.meta_str("engine_version").unwrap_or("").to_string(),
        model: g.meta_str("model").unwrap_or("").to_string(),
        model_revision: g.meta_str("model_revision").unwrap_or("").to_string(),
        kv_dtype: g.meta_str("kv_dtype").unwrap_or("").to_string(),
    };
    let run_cfg = baseline::RunConfig {
        model: info.id.clone(),
        backend: info.backend.clone(),
        device: info.device.clone(),
        kv_dtype: norm_dtype(&info.kv_dtype),
        act_dtype: norm_dtype(&info.act_dtype),
        fp8_act: args.model.fp8_act.clone().unwrap_or_else(|| "recipe default".into()),
        prefill_chunk: chunk,
        decode_positions: args.decode_positions,
        gate_class: gates.class.name().to_string(),
    };

    // What code this is, and how it ran. Checked before the long runs, like the golden and the configuration.
    let cur_code = CodeStamp::of_this_build();
    let exec = ExecInfo::capture(&engine.plan());
    let policy = provenance_policy(args)?;
    eprintln!("this build: {}", cur_code.describe());
    let show_exec_diff = |what: &str, rec: &ExecInfo| {
        let d = rec.diff(&exec);
        if d.is_empty() {
            eprintln!(
                "WARNING: this run resolved exactly the same kernels and flags as the {what}: a kernel or graph change cannot show up in this comparison"
            );
        } else {
            eprintln!("differences from the {what} (kernels per op, flags): {}", d.join("; "));
        }
    };
    let base_file = match &args.baseline {
        Some(bp) => {
            let base = baseline::read(bp)?;
            for w in baseline::compatible(&base, &golden_id, &run_cfg, &exec, args.allow_config_mismatch)? {
                eprintln!("WARNING: this run differs from the baseline: {w}");
            }
            for w in provenance::check("baseline", &base.code, &cur_code, &policy)? {
                eprintln!("WARNING: {w}");
            }
            eprintln!("baseline recorded from: {}", base.code.describe());
            show_exec_diff("baseline's run", &base.exec);
            Some((bp, base))
        }
        None => None,
    };
    let self_budget = selfgolden::SelfBudget {
        max_mean_kl: args.self_max_kl,
        max_pos_kl: args.self_max_pos_kl,
        max_flip_pct: args.self_max_flip_pct,
        max_tail_abs: args.self_max_tail,
        prompt_factor: args.self_prompt_factor,
    };
    if let Some(sm) = &self_meta {
        for w in self_compat(sm, &run_cfg, args.allow_config_mismatch)? {
            eprintln!("WARNING: this run differs from the self-golden: {w}");
        }
        for w in provenance::check("self-golden", &sm.code, &cur_code, &policy)? {
            eprintln!("WARNING: {w}");
        }
        eprintln!("self-golden written by: {} (top-{}, modes {})", sm.code.describe(), sm.topk, sm.modes);
        show_exec_diff("self-golden's run", &sm.exec);
        eprintln!(
            "self-golden budgets: mean KL(21-way) <= {:.1e}, max position KL <= {:.1e}, top-1 flips <= {}% (>= 1 position), mean |tail diff| <= {:.1e}, prompt KL <= x{}",
            self_budget.max_mean_kl, self_budget.max_pos_kl, self_budget.max_flip_pct, self_budget.max_tail_abs, self_budget.prompt_factor
        );
    }
    if args.write_baseline.is_some() || args.write_self_golden.is_some() {
        if !cur_code.is_stamped() {
            eprintln!("WARNING: this build carries no code stamp, so what it writes will be refused as a 'before': build through scripts/phase1b-acceptance.sh");
        }
        if cur_code.dirty && cur_code.overlay.is_empty() {
            eprintln!("WARNING: this build is from a modified tree and not a declared harness overlay; what it writes will be refused as a 'before'");
        }
    }
    let mut capture = args.write_self_golden.as_ref().map(|_| selfgolden::SelfCapture::new(args.self_topk.max(1), prompts.len()));
    let mut floor_ok = true;
    let mut results = Vec::new();
    for m in modes {
        let set = if self_mode && m == Mode::Decode { &prompts_dec } else { &prompts };
        if set.len() != prompts.len() {
            bail!("the self-golden has decode rows for {} of {} prompts", set.len(), prompts.len());
        }
        let r = run_mode(engine, set, m, args, &gates, topk_k.is_some(), self_mode.then_some(&self_budget), &mut capture)?;
        floor_ok &= r.pass;
        results.push((m, r));
    }
    for (m, r) in &results {
        let (mean, min) = {
            let fr: Vec<f64> = r.matches.iter().map(|&(l, n)| l as f64 / n.max(1) as f64).collect();
            (fr.iter().sum::<f64>() / fr.len().max(1) as f64, fr.iter().cloned().fold(1.0, f64::min))
        };
        println!("greedy-match [{}]: mean {:.1}% of positions before the first argmax divergence, worst prompt {:.1}% (informational)", m.name(), 100.0 * mean, 100.0 * min);
    }

    // baseline-relative gate
    let mut baseline_ok: Option<bool> = None;
    if let Some((bp, base)) = &base_file {
        let margins = Margins {
            top1_pts: args.margin_top1,
            kl_ratio: args.margin_kl,
            kl_eps: 1e-6,
            top5_pts: args.margin_top5,
            tail_abs: args.margin_tail,
            max_kl_ratio: args.margin_max_kl,
            prompt_kl_ratio: args.margin_prompt_kl,
            ..Margins::default()
        };
        println!(
            "\nbaseline: {} (recorded {}, golden {} hash {}); margins: top-1 -{} pt (>= 1 position), KL x{} (+1e-6), top-5 -{} pt, tail +{}",
            bp.display(),
            base.created_unix,
            base.golden.file,
            base.golden.fnv1a64,
            margins.top1_pts,
            margins.kl_ratio,
            margins.top5_pts,
            margins.tail_abs
        );
        let mut ok = true;
        for (m, r) in &results {
            let Some(bm) = base.modes.iter().find(|b| b.overall.mode == m.key()) else {
                bail!("{}: the baseline has no '{}' mode (it has: {}); record one with --write-baseline", bp.display(), m.key(), base.modes.iter().map(|b| b.overall.mode.as_str()).collect::<Vec<_>>().join(", "));
            };
            ok &= print_baseline_checks(*m, bm, r, &margins);
        }
        baseline_ok = Some(ok);
    }
    if let Some(wp) = &args.write_baseline {
        let b = Baseline {
            format: baseline::FORMAT.into(),
            created_unix: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
            tool_version: env!("CARGO_PKG_VERSION").into(),
            code: cur_code.clone(),
            exec: exec.clone(),
            golden: golden_id.clone(),
            run: run_cfg.clone(),
            modes: results.iter().map(|(_, r)| baseline::ModeBaseline { overall: r.stats.clone(), prompts: r.prompts.clone() }).collect(),
        };
        baseline::write(wp, &b)?;
        println!(
            "\nwrote baseline {} ({} mode(s); absolute floor {})",
            wp.display(),
            b.modes.len(),
            if floor_ok { "passed" } else { "FAILED in this run: the baseline records it" }
        );
    }
    if let (Some(wp), Some(cap)) = (&args.write_self_golden, &capture) {
        let ids: Vec<Vec<u32>> = prompts.iter().map(|p| p.1.ids.clone()).collect();
        let prompts_json = g.meta.get("prompts").cloned().unwrap_or_else(|| {
            serde_json::to_string(&names.iter().zip(&ids).map(|(n, i)| serde_json::json!({"name": n, "n_tokens": i.len()})).collect::<Vec<_>>())
                .unwrap_or_default()
        });
        let sm = selfgolden::SelfMeta {
            code: cur_code.clone(),
            exec: exec.clone(),
            model: run_cfg.model.clone(),
            backend: run_cfg.backend.clone(),
            device: run_cfg.device.clone(),
            kv_dtype: run_cfg.kv_dtype.clone(),
            act_dtype: run_cfg.act_dtype.clone(),
            fp8_act: run_cfg.fp8_act.clone(),
            prefill_chunk: chunk,
            decode_positions: args.decode_positions,
            topk: cap.k,
            modes: results.iter().map(|(m, _)| m.key()).collect::<Vec<_>>().join(","),
            source_golden: golden_id.file.clone(),
            source_fnv1a64: golden_id.fnv1a64.clone(),
        };
        selfgolden::write(wp, cap, &ids, &prompts_json, &sm)?;
        println!(
            "\nwrote self-golden {} (top-{} rows of modes {} for {} prompts; written by {})",
            wp.display(),
            cap.k,
            sm.modes,
            ids.len(),
            cur_code.describe()
        );
    }

    let mut reuse_ok = true;
    if want_reuse {
        reuse_ok = reuse_gate(engine, &g, &prompts, args)?;
    }
    let all_ok = floor_ok && baseline_ok.unwrap_or(true) && reuse_ok;
    if !results.is_empty() {
        println!(
            "\n{} verdict ({}): {}{}{}",
            if self_mode { "self-golden" } else { "floor" },
            if self_mode { "same weights, tight budgets, against the other build's logits" } else { "class thresholds, absolute" },
            if floor_ok { "PASS" } else { "FAIL" },
            match baseline_ok {
                Some(b) => format!("   baseline verdict (no worse than the recorded run + margins): {}", if b { "PASS" } else { "FAIL" }),
                None if self_mode => String::new(),
                None => "   baseline verdict: not requested".to_string(),
            },
            if want_reuse { format!("   prefix-reuse verdict: {}", if reuse_ok { "PASS" } else { "FAIL" }) } else { String::new() },
        );
    }
    println!("golden gate: {}", if all_ok { "PASS" } else { "FAIL" });
    Ok(all_ok)
}

// ---------------------------------------------------------------------------
// quick gate
// ---------------------------------------------------------------------------

/// Greedy sanity prompts (ARCHITECTURE section 6.2: five, one answering a known fact). The second field,
/// when set, must appear in the (case-insensitive) output.
const SANITY_PROMPTS: [(&str, Option<&str>); 5] = [
    ("1, 2, 3, 4, 5, 6, 7, 8, 9, 10,", None),
    ("Once upon a time, in a small village at the edge of a great forest, there lived an old clockmaker who", None),
    ("The three primary colors are red, yellow, and", None),
    ("The capital of France is", Some("paris")),
    ("Water is a molecule made of two hydrogen atoms and one", Some("oxygen")),
];

/// The largest number of times any 4-gram of `ids` occurs.
fn max_4gram_repeat(ids: &[u32]) -> usize {
    let mut counts: HashMap<&[u32], usize> = HashMap::new();
    for w in ids.windows(4) {
        *counts.entry(w).or_default() += 1;
    }
    counts.values().copied().max().unwrap_or(0)
}

fn quick_gate(engine: &dyn Engine, args: &AccuracyArgs) -> Result<bool> {
    let text = match &args.ppl_file {
        Some(p) => std::fs::read_to_string(p).with_context(|| p.display().to_string())?,
        None => PPL_TEXT.to_string(),
    };
    let mut ids = engine.tokenize(&text, true).map_err(|e| anyhow!("{e}"))?;
    ids.truncate(engine.info().context_len.min(2048));
    if ids.len() < 2 {
        bail!("perplexity text is too short");
    }
    let mut nll = 0.0f64;
    let mut count = 0usize;
    let mut nonfinite = 0usize;
    engine
        .forward_logits_each(&ids, &mut |pos, row| {
            if !is_finite_row(row) {
                nonfinite += 1;
            }
            if let Some(&next) = ids.get(pos + 1) {
                nll += lse(row.iter().map(|&v| v as f64)) - row[next as usize] as f64;
                count += 1;
            }
            Ok(())
        })
        .map_err(|e| anyhow!("{e}"))?;
    let ppl = (nll / count as f64).exp();
    let ppl_ok = ppl.is_finite() && ppl <= args.max_ppl && nonfinite == 0;
    println!(
        "perplexity: {ppl:.3} over {count} tokens (need <= {}), non-finite rows {nonfinite} {}",
        args.max_ppl,
        if ppl_ok { "ok" } else { "FAIL" }
    );

    let mut all_ok = ppl_ok;
    for (p, expect) in SANITY_PROMPTS {
        let prompt = engine.tokenize(p, true).map_err(|e| anyhow!("{e}"))?;
        let req = GenRequest { prompt, max_tokens: args.sanity_tokens, ..Default::default() };
        let (mut gen, mut err) = (Vec::<u32>::new(), None);
        engine
            .generate(req, &mut |ev| match ev {
                GenEvent::Token { id, .. } => gen.push(id),
                GenEvent::Error { message } => err = Some(message),
                GenEvent::Done { .. } => {}
            })
            .map_err(|e| anyhow!("{e}"))?;
        if let Some(m) = err {
            bail!("generation failed: {m}");
        }
        let out = engine.detokenize(&gen).map_err(|e| anyhow!("{e}"))?;
        let rep = max_4gram_repeat(&gen);
        // fact prompts: the answer must be in the first 16 tokens; a correct answer followed by a list of
        // similar facts is fine, so the repetition check applies to the open-ended prompts only
        let head = engine.detokenize(&gen[..gen.len().min(16)]).map_err(|e| anyhow!("{e}"))?;
        let fact = expect.is_none_or(|e| head.to_lowercase().contains(e));
        let ok = !out.trim().is_empty() && (rep <= 3 || expect.is_some()) && fact;
        all_ok &= ok;
        println!(
            "sanity {}: {} tokens, max 4-gram repeat {rep}: {}{}\n    {:?} -> {:?}",
            if ok { "ok  " } else { "FAIL" },
            gen.len(),
            if out.trim().is_empty() { "EMPTY" } else { "non-empty" },
            match expect {
                Some(e) => format!(", answers the fact ({e:?}): {}", if fact { "yes" } else { "NO" }),
                None => String::new(),
            },
            p,
            out
        );
    }
    println!("quick gate: {}", if all_ok { "PASS" } else { "FAIL" });
    Ok(all_ok)
}

pub fn run(args: AccuracyArgs) -> Result<()> {
    let loaded = args.model.load()?;
    let engine: &dyn Engine = &*loaded.engine;
    let info = engine.info();
    eprintln!(
        "accuracy: {} on {} (vocab {}, weights/act/kv {}/{})",
        info.id, info.device, info.vocab, info.act_dtype, info.kv_dtype
    );
    if args.model.reference {
        eprintln!("--reference: every op runs its ref_* oracle kernel");
    }
    let mut ok = true;
    if args.quick {
        ok &= quick_gate(engine, &args)?;
    }
    if !args.quick || args.golden.is_some() || args.self_golden.is_some() {
        ok &= golden_gate(engine, &args)?;
    }
    if !ok {
        bail!("accuracy gate FAILED");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i * 7) % 13) as f32 * 0.3).collect()
    }
    fn full_ref(l: &[f32]) -> Vec<u16> {
        let z = lse(l.iter().map(|&v| v as f64));
        l.iter().map(|&v| half::f16::from_f64(v as f64 - z).to_bits()).collect()
    }
    fn topk_ref(l: &[f32], k: usize) -> (Vec<i32>, Vec<f32>) {
        let z = lse(l.iter().map(|&v| v as f64));
        let ix = top_k(l.iter().map(|&v| v as f64), k);
        (ix.iter().map(|&i| i as i32).collect(), ix.iter().map(|&i| (l[i] as f64 - z) as f32).collect())
    }

    #[test]
    fn top_k_orders_and_breaks_ties_low() {
        assert_eq!(top_k([1.0, 5.0, 3.0, 5.0, 2.0].into_iter(), 3), vec![1, 3, 2]);
    }

    #[test]
    fn identical_distributions_agree() {
        let l = logits(50);
        let a = compare_pos(RefRow::Full(&full_ref(&l)), &l).unwrap();
        assert!(a.top1 || a.tie);
        assert_eq!(a.overlap, 5);
        assert!(a.kl < 1e-5 && !a.nonfinite, "{}", a.kl);
    }

    #[test]
    fn nan_anywhere_fails_the_position() {
        let l = logits(50);
        let r = full_ref(&l);
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            // a single bad entry, even far from the top tokens, must not be skipped
            let mut x = l.clone();
            x[7] = bad;
            let a = compare_pos(RefRow::Full(&r), &x).unwrap();
            assert!(a.nonfinite && !a.top1 && a.kl.is_infinite(), "{bad}: {a:?}");
            let mut agg = Agg::default();
            agg.push(&a);
            assert_eq!(agg.nonfinite, 1);
            assert!(agg.mean_kl().is_infinite());
        }
        // an all-NaN row used to panic on ot[0]
        let all = vec![f32::NAN; 50];
        let a = compare_pos(RefRow::Full(&r), &all).unwrap();
        assert!(a.nonfinite && a.kl.is_infinite());
        // top-k reference too
        let (ids, lps) = topk_ref(&l, 8);
        let a = compare_pos(RefRow::TopK(&ids, &lps), &all).unwrap();
        assert!(a.nonfinite && !a.top1);
        let mut x = l.clone();
        x[49] = f32::NAN;
        assert!(compare_pos(RefRow::TopK(&ids, &lps), &x).unwrap().nonfinite);
    }

    #[test]
    fn nan_in_the_golden_is_an_error() {
        let l = logits(10);
        let mut r = full_ref(&l);
        r[3] = half::f16::NAN.to_bits();
        assert!(compare_pos(RefRow::Full(&r), &l).is_err());
        let (ids, mut lps) = topk_ref(&l, 4);
        lps[1] = f32::NAN;
        assert!(compare_pos(RefRow::TopK(&ids, &lps), &l).is_err());
    }

    #[test]
    fn gate_fails_on_nan_even_with_perfect_other_rows() {
        let l = logits(40);
        let r = full_ref(&l);
        let mut agg = Agg::default();
        for _ in 0..999 {
            agg.push(&compare_pos(RefRow::Full(&r), &l).unwrap());
        }
        let mut x = l.clone();
        x[3] = f32::NAN;
        agg.push(&compare_pos(RefRow::Full(&r), &x).unwrap());
        assert!(agg.top1_pct() >= 99.5 - 1e-9, "top-1 alone would pass: {}", agg.top1_pct());
        assert_eq!(agg.nonfinite, 1);
        assert!(!(agg.mean_kl() <= 1e-3), "inf KL must fail the KL gate: {}", agg.mean_kl());
    }

    #[test]
    fn strict_top1_and_tie_reporting() {
        // two tokens tie in the (fp16) reference; ours prefers the later one
        let r: Vec<u16> = [-2.0f64, -0.5, -0.5, -3.0].iter().map(|&v| half::f16::from_f64(v).to_bits()).collect();
        let ours = [0.0f32, 1.0, 1.1, -1.0];
        let a = compare_pos(RefRow::Full(&r), &ours).unwrap();
        assert!(!a.top1 && a.tie, "strict top-1 with the tie reported separately: {a:?}");
    }

    #[test]
    fn topk_kl_is_zero_for_equal_and_positive_for_shifted() {
        let l = logits(200);
        let (ids, lps) = topk_ref(&l, 20);
        let a = compare_pos(RefRow::TopK(&ids, &lps), &l).unwrap();
        assert!(a.top1 && a.kl < 1e-6 && a.overlap == 5, "{a:?}");
        let mut x = l.clone();
        for v in x.iter_mut().take(100) {
            *v += 0.5;
        }
        let b = compare_pos(RefRow::TopK(&ids, &lps), &x).unwrap();
        assert!(b.kl > 1e-4, "{}", b.kl);
        // padding entries (-1) are ignored
        let mut pid = ids.clone();
        let mut plp = lps.clone();
        pid.push(-1);
        plp.push(-1e30);
        let c = compare_pos(RefRow::TopK(&pid, &plp), &l).unwrap();
        assert!(c.kl < 1e-6);
    }

    #[test]
    fn greedy_match_length_counts_the_leading_run() {
        let mk = |ok: bool| PosStat { top1: ok, tie: false, overlap: 5, n_top: 5, kl: 0.0, kl21: 0.0, tail_ref: 0.0, tail_ours: 0.0, nonfinite: false };
        let s: Vec<PosStat> = [true, true, true, false, true].into_iter().map(mk).collect();
        assert_eq!(greedy_match_len(&s), 3);
        assert_eq!(greedy_match_len(&[]), 0);
    }

    #[test]
    fn default_gates_by_class() {
        assert_eq!(GateClass::Bf16.defaults(), (Some(99.5), 1e-3));
        assert_eq!(GateClass::Fp8.defaults(), (None, 5e-3));
        assert_eq!(GateClass::Fp4.defaults(), (None, 3e-2));
        assert_eq!(GateClass::Bf16.min(GateClass::Fp8), GateClass::Fp8);
        assert_eq!(GateClass::Fp8.min(GateClass::Fp4), GateClass::Fp4);
        assert_eq!(GateClass::parse("4bit").unwrap(), GateClass::Fp4);
        assert!(GateClass::parse("int3").is_err());
    }

    #[test]
    fn weight_class_from_quantization_metadata() {
        let d = std::env::temp_dir().join(format!("intarsia-acc-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let w = |cfg: &str, hf: Option<&str>| {
            std::fs::write(d.join("config.json"), cfg).unwrap();
            let _ = std::fs::remove_file(d.join("hf_quant_config.json"));
            if let Some(h) = hf {
                std::fs::write(d.join("hf_quant_config.json"), h).unwrap();
            }
            weight_class(&d)
        };
        assert_eq!(w(r#"{"torch_dtype":"bfloat16"}"#, None).unwrap().0, GateClass::Bf16);
        assert_eq!(
            w(r#"{"quantization_config":{"quant_method":"fp8","activation_scheme":"static"}}"#, None).unwrap().0,
            GateClass::Fp8
        );
        assert_eq!(
            w(r#"{"text_config":{"quantization_config":{"quant_method":"modelopt","quant_algo":"NVFP4"}}}"#, None)
                .unwrap()
                .0,
            GateClass::Fp4
        );
        assert_eq!(
            w(r#"{"quantization_config":{"quant_method":"modelopt"}}"#, Some(r#"{"quantization":{"quant_algo":"MIXED_PRECISION","quantized_layers":{"a":{"quant_algo":"FP8"},"b":{"quant_algo":"NVFP4"}}}}"#))
                .unwrap()
                .0,
            GateClass::Fp4
        );
        assert!(w(r#"{"quantization_config":{"quant_method":"mystery"}}"#, None).is_err());
        assert_eq!(
            w(r#"{"quantization_config":{"quant_method":"fp8","group_size":4,"block":[4,128]}}"#, None).unwrap().0,
            GateClass::Fp8,
            "a bare 4 is not a bit width"
        );
        assert_eq!(
            w(r#"{"quantization_config":{"quant_method":"compressed-tensors","num_bits":4}}"#, None).unwrap().0,
            GateClass::Fp4
        );
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn weight_class_of_the_real_checkpoints() {
        let fx = Path::new(env!("CARGO_MANIFEST_DIR")).join("../intarsia-models/tests/fixtures");
        let d = std::env::temp_dir().join(format!("intarsia-acc-real-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        // dense bf16 checkpoint
        std::fs::copy(fx.join("config_qwen3_8b.json"), d.join("config.json")).unwrap();
        assert_eq!(weight_class(&d).unwrap().0, GateClass::Bf16);
        // Qwen3.6-35B-A3B NVFP4 (FP8 attention + NVFP4 experts): the lowest precision wins
        std::fs::copy(fx.join("config_qwen36_35b_a3b_nvfp4.json"), d.join("config.json")).unwrap();
        assert_eq!(weight_class(&d).unwrap().0, GateClass::Fp4);
        std::fs::copy(fx.join("hf_quant_config_qwen36_trimmed.json"), d.join("hf_quant_config.json")).unwrap();
        assert_eq!(weight_class(&d).unwrap().0, GateClass::Fp4);
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn repeat_detection() {
        assert_eq!(max_4gram_repeat(&[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]), 4);
        assert_eq!(max_4gram_repeat(&[1, 2, 3]), 0);
    }

    // -- tail bucket (review finding 1) -----------------------------------------------------------

    /// A vocabulary of `v` tokens: the first 20 hold the probability, the rest sit at logit -10.
    fn peaked(v: usize) -> Vec<f32> {
        (0..v).map(|i| if i < 20 { 5.0 - 0.1 * i as f32 } else { -10.0 }).collect()
    }

    #[test]
    fn mass_leaving_the_top_k_is_invisible_to_the_renormalised_kl_but_not_to_the_tail_bucket() {
        // The scenario of the review: a bug (zeroed block scales in the lm_head) lifts thousands of rare-token
        // logits from about -10 to about 0. Most of our probability moves outside the reference's top-20.
        let reference = peaked(20_000);
        let (ids, lps) = topk_ref(&reference, 20);
        let mut broken = reference.clone();
        for v in broken.iter_mut().skip(20).take(3000) {
            *v = 0.0;
        }
        let a = compare_pos(RefRow::TopK(&ids, &lps), &broken).unwrap();
        assert!(a.top1, "the argmax survives, as in the review");
        assert!(a.kl < 1e-9, "the renormalised KL is blind to it: {}", a.kl);
        assert!(a.tail_ours > 0.6 && a.tail_ref < 0.01, "tail mass ours {} ref {}", a.tail_ours, a.tail_ref);
        assert!(a.kl21 > 0.5, "the 21-way KL sees it: {}", a.kl21);
        // and the gates built on those numbers fail where the old one passed
        let mut agg = Agg::default();
        agg.push(&a);
        assert!(agg.mean_kl() <= 3e-2, "old 4-bit gate (renormalised KL) passes");
        assert!(agg.mean_kl21() > 3e-2, "21-way KL fails the same floor");
        assert!(agg.tail_excess().abs() > DEFAULT_MAX_TAIL_EXCESS, "tail excess {}", agg.tail_excess());
        // the same with a full-vocabulary golden
        let r = full_ref(&reference);
        let b = compare_pos(RefRow::Full(&r), &broken).unwrap();
        assert!(b.kl > 0.5, "full KL sees it too: {}", b.kl);
        assert!(b.kl21 > 0.5 && b.tail_ours > 0.6, "{b:?}");
    }

    #[test]
    fn identical_distributions_have_no_tail_difference() {
        let l = peaked(5000);
        let (ids, lps) = topk_ref(&l, 20);
        let a = compare_pos(RefRow::TopK(&ids, &lps), &l).unwrap();
        assert!(a.kl21 < 1e-6 && (a.tail_ours - a.tail_ref).abs() < 1e-6, "{a:?}");
        assert!(a.tail_ref > 1e-5, "the tail is real: {}", a.tail_ref);
        let b = compare_pos(RefRow::Full(&full_ref(&l)), &l).unwrap();
        assert!(b.kl21 < 1e-3 && (b.tail_ours - b.tail_ref).abs() < 1e-3, "{b:?}"); // fp16 reference rounding
    }

    #[test]
    fn the_21_way_kl_is_a_lower_bound_of_the_full_kl() {
        let l = logits(300);
        let r = full_ref(&l);
        let mut x = l.clone();
        for (i, v) in x.iter_mut().enumerate() {
            *v += ((i * 31) % 17) as f32 * 0.07;
        }
        let a = compare_pos(RefRow::Full(&r), &x).unwrap();
        assert!(a.kl > 1e-3, "{}", a.kl);
        assert!(a.kl21 <= a.kl + 1e-9, "partition KL {} must not exceed the full KL {}", a.kl21, a.kl);
        // a top-k golden of the same reference: its 21-way value is the same partition (K = 20)
        let (ids, lps) = topk_ref(&l, 20);
        let b = compare_pos(RefRow::TopK(&ids, &lps), &x).unwrap();
        assert!((b.kl21 - a.kl21).abs() < 1e-3, "{} vs {}", b.kl21, a.kl21);
    }

    #[test]
    fn reference_logprobs_summing_just_over_one_do_not_break_the_tail() {
        let l = logits(100);
        let (ids, mut lps) = topk_ref(&l, 100); // the whole vocabulary: the tail is exactly empty
        lps.iter_mut().for_each(|x| *x += 1e-6); // reporting rounding: the sum is a hair over 1
        let a = compare_pos(RefRow::TopK(&ids, &lps), &l).unwrap();
        assert!(a.tail_ref < 1e-9 && a.tail_ours < 1e-9 && a.kl21 < 1e-5, "{a:?}");
    }

    #[test]
    fn kv_dtype_aliases_and_warnings() {
        assert_eq!(norm_dtype("bfloat16"), "bf16");
        assert_eq!(norm_dtype("fp8"), "fp8_e4m3");
        assert_eq!(norm_dtype("fp8_e4m3"), "fp8_e4m3");
        assert_eq!(norm_dtype("float8_e4m3fn"), "fp8_e4m3");
        assert_eq!(norm_dtype("fp8_e5m2"), "fp8_e5m2");
        assert!(kv_dtype_warning(Some("bfloat16"), "bf16").is_none());
        assert!(kv_dtype_warning(Some("auto"), "bf16").is_none());
        assert!(kv_dtype_warning(Some("fp8"), "fp8_e4m3").is_none());
        let w = kv_dtype_warning(Some("fp8"), "bf16").unwrap();
        assert!(w.contains("mismatch") && w.contains("fp8_e4m3") && w.contains("bf16"), "{w}");
        assert!(kv_dtype_warning(None, "bf16").unwrap().contains("does not record"));
        assert!(kv_dtype_warning(Some("unknown"), "bf16").unwrap().contains("does not record"));
        assert!(kv_dtype_warning(Some("auto"), "fp8_e4m3").is_some());
    }

    #[test]
    fn first_divergence_of_streams() {
        assert_eq!(first_divergence(&[1, 2, 3], &[1, 2, 3]), None);
        assert_eq!(first_divergence(&[1, 2, 3], &[1, 9, 3]), Some(1));
        assert_eq!(first_divergence(&[1, 2], &[1, 2, 3]), Some(2));
    }
}
