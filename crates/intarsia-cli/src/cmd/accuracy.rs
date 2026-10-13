//! `intarsia accuracy`: accuracy gates (ARCHITECTURE section 6, docs/accuracy.md).
//!
//! * `--golden FILE` (default: `golden/<model>.safetensors`): compare our logits with the Hugging Face
//!   bf16 reference on fixed prompts (top-1 agreement, KL(ref||ours), top-5 overlap).
//! * `--quick`: day-0 gate: perplexity on a fixed English text plus greedy sanity prompts.
//!
//! Exits non-zero when any gate fails.

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
    /// Golden logprobs file written by tools/golden.py. Default: golden/<model name>.safetensors.
    #[arg(long)]
    pub golden: Option<PathBuf>,
    /// Day-0 quick gate (perplexity + greedy sanity prompts) instead of the golden comparison
    /// (give both --quick and --golden to run both).
    #[arg(long)]
    pub quick: bool,
    /// Golden gate: minimum top-1 agreement in percent.
    #[arg(long, default_value_t = 99.0)]
    pub min_top1: f64,
    /// Golden gate: maximum mean KL(ref||ours) in nats. Default 1e-3 on cpu, 5e-3 on other backends.
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

    fn ids(&self, i: usize) -> Result<Vec<u32>> {
        let (b, _) = self.raw(&format!("ids_{i}"), "I32")?;
        Ok(b.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32).collect())
    }

    /// Log-probs of prompt `i` as raw f16 bits, `[n, vocab]`.
    fn logprobs(&self, i: usize) -> Result<(Vec<u16>, usize, usize)> {
        let (b, shape) = self.raw(&format!("logprobs_{i}"), "F16")?;
        if shape.len() != 2 {
            bail!("logprobs_{i}: shape {shape:?}");
        }
        Ok((b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect(), shape[0], shape[1]))
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

#[derive(Default, Clone)]
struct Agg {
    n: usize,
    top1: usize,
    top5: usize, // summed overlap counts
    kl_sum: f64,
    kl_max: f64,
}

impl Agg {
    fn add(&mut self, o: &Agg) {
        self.n += o.n;
        self.top1 += o.top1;
        self.top5 += o.top5;
        self.kl_sum += o.kl_sum;
        self.kl_max = self.kl_max.max(o.kl_max);
    }
    fn top1_pct(&self) -> f64 {
        100.0 * self.top1 as f64 / self.n.max(1) as f64
    }
    fn mean_kl(&self) -> f64 {
        self.kl_sum / self.n.max(1) as f64
    }
    fn top5_pct(&self) -> f64 {
        100.0 * self.top5 as f64 / (5 * self.n.max(1)) as f64
    }
}

fn lse(x: impl Iterator<Item = f64> + Clone) -> f64 {
    let m = x.clone().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    m + x.map(|v| (v - m).exp()).sum::<f64>().ln()
}

/// Compare one position. The reference (fp16 log-probs) is renormalised to cancel fp16 rounding bias.
fn compare_pos(reference: &[u16], ours: &[f32]) -> Agg {
    let r = || reference.iter().map(|&b| f64_of(b));
    let o = || ours.iter().map(|&v| v as f64);
    let (rl, ol) = (lse(r()), lse(o()));
    let mut kl = 0.0;
    for (&rb, &ov) in reference.iter().zip(ours) {
        let lr = f16_to_f64(rb) - rl;
        if lr.is_finite() {
            kl += lr.exp() * (lr - (ov as f64 - ol));
        }
    }
    let kl = kl.max(0.0);
    let rt = top_k(r(), 5);
    let ot = top_k(o(), 5);
    let top1 = ot[0] == rt[0] || f16_to_f64(reference[ot[0]]) >= f16_to_f64(reference[rt[0]]);
    let overlap = ot.iter().filter(|i| rt.contains(i)).count();
    Agg { n: 1, top1: top1 as usize, top5: overlap, kl_sum: kl, kl_max: kl }
}

fn f64_of(b: u16) -> f64 {
    f16_to_f64(b)
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
    let vocab = engine.info().vocab;
    let max_kl = args.max_mean_kl.unwrap_or(if args.model.backend == "cpu" { 1e-3 } else { 5e-3 });
    eprintln!(
        "golden: {} ({} prompts, reference {}, torch {}, {})",
        path.display(),
        names.len(),
        g.meta.get("device").map(String::as_str).unwrap_or("?"),
        g.meta.get("torch").map(String::as_str).unwrap_or("?"),
        g.meta.get("dtype").map(String::as_str).unwrap_or("?"),
    );
    println!("{:<18} {:>5} {:>9} {:>11} {:>11} {:>9}", "prompt", "n", "top-1 %", "mean KL", "max KL", "top-5 %");
    let mut total = Agg::default();
    for (i, name) in names.iter().enumerate() {
        let ids = g.ids(i)?;
        let (lp, n, v) = g.logprobs(i)?;
        if n != ids.len() {
            bail!("{name}: {} ids but {n} logprob rows", ids.len());
        }
        if v != vocab {
            bail!("{name}: golden vocab {v} != model vocab {vocab}");
        }
        let mut agg = Agg::default();
        engine
            .forward_logits_each(&ids, &mut |pos, row| {
                if row.len() != v {
                    return Err(intarsia_core::Error::model(format!("logits row has {} entries, want {v}", row.len())));
                }
                agg.add(&compare_pos(&lp[pos * v..(pos + 1) * v], row));
                Ok(())
            })
            .map_err(|e| anyhow!("{e}"))?;
        if agg.n != n {
            bail!("{name}: engine returned {} positions, want {n}", agg.n);
        }
        println!(
            "{:<18} {:>5} {:>9.2} {:>11.3e} {:>11.3e} {:>9.2}",
            name,
            n,
            agg.top1_pct(),
            agg.mean_kl(),
            agg.kl_max,
            agg.top5_pct()
        );
        total.add(&agg);
    }
    let ok_top1 = total.top1_pct() >= args.min_top1;
    let ok_kl = total.mean_kl() <= max_kl;
    println!(
        "{:<18} {:>5} {:>9.2} {:>11.3e} {:>11.3e} {:>9.2}",
        "OVERALL",
        total.n,
        total.top1_pct(),
        total.mean_kl(),
        total.kl_max,
        total.top5_pct()
    );
    println!(
        "golden gate: top-1 {:.2}% (need >= {}) {} | mean KL {:.3e} (need <= {:.1e}) {} => {}",
        total.top1_pct(),
        args.min_top1,
        if ok_top1 { "ok" } else { "FAIL" },
        total.mean_kl(),
        max_kl,
        if ok_kl { "ok" } else { "FAIL" },
        if ok_top1 && ok_kl { "PASS" } else { "FAIL" },
    );
    Ok(ok_top1 && ok_kl)
}

// ---------------------------------------------------------------------------
// quick gate
// ---------------------------------------------------------------------------

const SANITY_PROMPTS: [&str; 3] = [
    "1, 2, 3, 4, 5, 6, 7, 8, 9, 10,",
    "Once upon a time, in a small village at the edge of a great forest, there lived an old clockmaker who",
    "The three primary colors are red, yellow, and",
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
    engine
        .forward_logits_each(&ids, &mut |pos, row| {
            if let Some(&next) = ids.get(pos + 1) {
                nll += lse(row.iter().map(|&v| v as f64)) - row[next as usize] as f64;
                count += 1;
            }
            Ok(())
        })
        .map_err(|e| anyhow!("{e}"))?;
    let ppl = (nll / count as f64).exp();
    let ppl_ok = ppl.is_finite() && ppl <= args.max_ppl;
    println!(
        "perplexity: {ppl:.3} over {count} tokens (need <= {}) {}",
        args.max_ppl,
        if ppl_ok { "ok" } else { "FAIL" }
    );

    let mut all_ok = ppl_ok;
    for p in SANITY_PROMPTS {
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
        let ok = !out.trim().is_empty() && rep <= 3;
        all_ok &= ok;
        println!(
            "sanity {}: {} tokens, max 4-gram repeat {rep}: {}\n    {:?} -> {:?}",
            if ok { "ok  " } else { "FAIL" },
            gen.len(),
            if out.trim().is_empty() { "EMPTY" } else { "non-empty" },
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
    eprintln!("accuracy: {} on {} (vocab {})", info.id, info.device, info.vocab);
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

    #[test]
    fn top_k_orders_and_breaks_ties_low() {
        assert_eq!(top_k([1.0, 5.0, 3.0, 5.0, 2.0].into_iter(), 3), vec![1, 3, 2]);
    }

    #[test]
    fn identical_distributions_agree() {
        let logits: Vec<f32> = (0..50).map(|i| ((i * 7) % 13) as f32 * 0.3).collect();
        let l = lse(logits.iter().map(|&v| v as f64));
        let reference: Vec<u16> = logits.iter().map(|&v| half::f16::from_f64(v as f64 - l).to_bits()).collect();
        let a = compare_pos(&reference, &logits);
        assert_eq!((a.top1, a.top5), (1, 5));
        assert!(a.kl_sum < 1e-5, "{}", a.kl_sum);
    }

    #[test]
    fn repeat_detection() {
        assert_eq!(max_4gram_repeat(&[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]), 4);
        assert_eq!(max_4gram_repeat(&[1, 2, 3]), 0);
    }
}
