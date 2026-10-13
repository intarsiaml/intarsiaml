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
//! Any non-finite value in one of our logits rows fails the gate. Exits non-zero when any gate fails.

use crate::common::ModelArgs;
use anyhow::{anyhow, bail, Context, Result};
use intarsia_core::engine::{Engine, GenEvent, GenRequest};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

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
    /// Golden comparison mode: `prefill`, `decode` (teacher-forced single-token steps) or `both`.
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
    kl: f64,
    /// Our row has a NaN or an infinity: the position fails whatever the other numbers say.
    nonfinite: bool,
}

#[derive(Default, Clone, Debug)]
struct Agg {
    n: usize,
    top1: usize,
    ties: usize,
    top5: usize, // summed overlap counts
    top5_den: usize,
    kl_sum: f64,
    kl_max: f64,
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

/// Compare one position. `ours` are raw logits. A non-finite value anywhere in `ours` makes the
/// position fail (KL = infinity, top-1 miss). A NaN in the reference is a broken golden: error.
///
/// * `Full`: the fp16 reference is renormalised (cancels fp16 rounding bias); KL over the whole vocab.
/// * `TopK`: KL over the reference's top-k support, both sides renormalised over that support
///   (a lower bound of the full-vocabulary KL); top-1 is our full-vocabulary argmax against the
///   reference's best id.
fn compare_pos(reference: RefRow<'_>, ours: &[f32]) -> Result<PosStat> {
    let finite = is_finite_row(ours);
    let o = || ours.iter().map(|&v| v as f64);
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
                return Ok(PosStat {
                    top1: false,
                    tie: false,
                    overlap: 0,
                    n_top: rt.len(),
                    kl: f64::INFINITY,
                    nonfinite: true,
                });
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
            let ot = top_k(o(), 5);
            let top1 = ot[0] == rt[0];
            let tie = !top1 && f16_to_f64(r[ot[0]]) >= f16_to_f64(r[rt[0]]);
            let overlap = ot.iter().filter(|i| rt.contains(i)).count();
            Ok(PosStat { top1, tie, overlap, n_top: rt.len(), kl, nonfinite: false })
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
                return Ok(PosStat {
                    top1: false,
                    tie: false,
                    overlap: 0,
                    n_top: rt.len(),
                    kl: f64::INFINITY,
                    nonfinite: true,
                });
            }
            // distributions over the support
            let lr_all = lse(valid.iter().map(|v| v.1));
            let lo: Vec<f64> = valid.iter().map(|v| ours[v.0] as f64).collect();
            let lo_all = lse(lo.iter().copied());
            let mut kl = 0.0;
            for (v, &l) in valid.iter().zip(&lo) {
                let lp = v.1 - lr_all;
                kl += lp.exp() * (lp - (l - lo_all));
            }
            let kl = kl.max(0.0);
            let ot = top_k(o(), 5);
            let overlap = ot.iter().filter(|i| rt.contains(i)).count();
            Ok(PosStat { top1: ot[0] == ref_top, tie: false, overlap, n_top: rt.len(), kl, nonfinite: false })
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
    max_kl: f64,
}

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
    Ok(Gates { class, why, min_top1: args.min_top1.or(t1), max_kl: args.max_mean_kl.unwrap_or(kl) })
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
}

struct ModeResult {
    /// Per-prompt leading run of argmax agreement and the number of positions compared.
    matches: Vec<(usize, usize)>,
    pass: bool,
}

/// Run one mode over all prompts and judge it against `gates`.
fn run_mode(
    engine: &dyn Engine,
    prompts: &[(String, PromptRef)],
    mode: Mode,
    args: &AccuracyArgs,
    gates: &Gates,
) -> Result<ModeResult> {
    let vocab = engine.info().vocab;
    println!("\n== mode: {} ==", mode.name());
    println!(
        "{:<18} {:>5} {:>5} {:>8} {:>11} {:>11} {:>8} {:>9} {:>7}",
        "prompt", "n", "pos", "top-1 %", "mean KL", "max KL", "top-5 %", "greedy", "nonfin"
    );
    let mut total = Agg::default();
    let mut matches = Vec::new();
    for (name, p) in prompts {
        let n = p.ids.len();
        let mut stats: Vec<PosStat> = Vec::new();
        let mut cb = |pos: usize, row: &[f32]| -> intarsia_core::Result<()> {
            if row.len() != vocab {
                return Err(intarsia_core::Error::model(format!("logits row has {} entries, want {vocab}", row.len())));
            }
            let s = compare_pos(p.row(pos), row)
                .map_err(|e| intarsia_core::Error::model(format!("{name} position {pos}: {e}")))?;
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
        println!(
            "{:<18} {:>5} {:>5} {:>8.2} {:>11.3e} {:>11.3e} {:>8.2} {:>9} {:>7}",
            name,
            n,
            stats.len(),
            agg.top1_pct(),
            agg.mean_kl(),
            agg.kl_max,
            agg.top5_pct(),
            format!("{ml}/{}", stats.len()),
            agg.nonfinite
        );
        matches.push((ml, stats.len()));
        total.add(&agg);
    }
    println!(
        "{:<18} {:>5} {:>5} {:>8.2} {:>11.3e} {:>11.3e} {:>8.2} {:>9} {:>7}",
        "OVERALL",
        "",
        total.n,
        total.top1_pct(),
        total.mean_kl(),
        total.kl_max,
        total.top5_pct(),
        format!("{:.1}%", 100.0 * matches.iter().map(|m| m.0).sum::<usize>() as f64 / total.n.max(1) as f64),
        total.nonfinite
    );
    let ok_nf = total.nonfinite == 0;
    let ok_top1 = gates.min_top1.is_none_or(|m| total.top1_pct() >= m);
    let ok_kl = total.mean_kl() <= gates.max_kl; // NaN / inf compare false
    let tag = |ok: bool| if ok { "ok" } else { "FAIL" };
    println!(
        "{} gate [{}]: top-1 {:.2}%{} ({}) {} | mean KL {:.3e} (need <= {:.1e}) {} | non-finite rows {} {} => {}{}",
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
        total.mean_kl(),
        gates.max_kl,
        tag(ok_kl),
        total.nonfinite,
        tag(ok_nf),
        if ok_nf && ok_top1 && ok_kl { "PASS" } else { "FAIL" },
        "",
    );
    Ok(ModeResult { matches, pass: ok_nf && ok_top1 && ok_kl })
}

fn golden_gate(engine: &dyn Engine, args: &AccuracyArgs) -> Result<bool> {
    let path = match &args.golden {
        Some(p) => p.clone(),
        None => default_golden(&args.model.model).ok_or_else(|| {
            anyhow!("no golden file found for this model; generate one with tools/golden.py and pass --golden")
        })?,
    };
    let g = Golden::open(&path)?;
    let names = g.names();
    if names.is_empty() {
        bail!("{}: no ids_N tensors", path.display());
    }
    let modes: Vec<Mode> = match args.mode.as_str() {
        "prefill" => vec![Mode::Prefill],
        "decode" => vec![Mode::Decode],
        "both" => vec![Mode::Prefill, Mode::Decode],
        o => bail!("unknown --mode '{o}' (prefill, decode, both)"),
    };
    let info = engine.info();
    let gates = resolve_gates(args, &info.kv_dtype)?;
    let mut prompts = Vec::new();
    for (i, name) in names.iter().enumerate() {
        let p = g.prompt(i)?;
        if let RefRows::Full { vocab, .. } = &p.rows {
            if *vocab != info.vocab {
                bail!("{name}: golden vocab {vocab} != model vocab {}", info.vocab);
            }
        }
        prompts.push((name.clone(), p));
    }
    let longest = prompts.iter().map(|p| p.1.ids.len()).max().unwrap_or(0);
    let chunk = args.model.prefill_chunk;
    let multi = prompts.iter().filter(|p| p.1.ids.len() > chunk).count();
    eprintln!(
        "golden: {} ({} prompts, {}, reference {} {} {}, torch {})",
        path.display(),
        names.len(),
        if g.is_topk() {
            format!(
                "top-{} log-probs, KL over the reference's top-k support renormalised",
                match &prompts[0].1.rows {
                    RefRows::TopK { k, .. } => *k,
                    _ => 0,
                }
            )
        } else {
            "full-vocabulary log-probs".into()
        },
        g.meta.get("source").map(String::as_str).unwrap_or("hf"),
        g.meta.get("device").map(String::as_str).unwrap_or("?"),
        g.meta.get("dtype").map(String::as_str).unwrap_or("?"),
        g.meta.get("torch").map(String::as_str).unwrap_or("?"),
    );
    eprintln!(
        "gates: class {} ({}); top-1 {}, mean KL <= {:.1e}",
        gates.class.name(),
        gates.why,
        gates.min_top1.map_or("not gated".to_string(), |m| format!(">= {m}%")),
        gates.max_kl
    );
    eprintln!(
        "prefill chunk {chunk}; longest prompt {longest}; prompts spanning more than one chunk: {multi}/{}",
        prompts.len()
    );
    if multi == 0 {
        eprintln!("WARNING: no prompt crosses a prefill chunk boundary, so multi-chunk prefill is untested; use --prefill-chunk {} or a golden with longer prompts", (longest / 3).max(1));
    }
    let mut all_ok = true;
    let mut results = Vec::new();
    for m in modes {
        let r = run_mode(engine, &prompts, m, args, &gates)?;
        all_ok &= r.pass;
        results.push((m, r));
    }
    for (m, r) in &results {
        let (mean, min) = {
            let fr: Vec<f64> = r.matches.iter().map(|&(l, n)| l as f64 / n.max(1) as f64).collect();
            (fr.iter().sum::<f64>() / fr.len().max(1) as f64, fr.iter().cloned().fold(1.0, f64::min))
        };
        println!("greedy-match [{}]: mean {:.1}% of positions before the first argmax divergence, worst prompt {:.1}% (informational)", m.name(), 100.0 * mean, 100.0 * min);
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
    if !args.quick || args.golden.is_some() {
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
        let mk = |ok: bool| PosStat { top1: ok, tie: false, overlap: 5, n_top: 5, kl: 0.0, nonfinite: false };
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
}
