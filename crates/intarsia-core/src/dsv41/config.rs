//! The DeepSeek-V4.1 config: parsed from a Hugging Face `config.json` (`text_config` nested, HF key names, the form both
//! the official FP8 base and the Mia EXL3 checkpoint ship) or from the official `inference/config.json` (flat, the field
//! names of `model.py` `ModelArgs`), into one typed struct, then checked for the structural facts the graph relies on.

use super::ops::ScoreFunc;
use crate::error::{Error, Result};
use serde_json::Value;

/// How the checkpoint stores its linear layers (`quantization_config.quant_method`).
#[derive(Clone, Debug, PartialEq)]
pub enum Dsv41Quant {
    /// The official base: FP8 e4m3 linears with E8M0 block scales (`weight_block_size`), FP4 routed experts.
    Fp8 { block: usize, expert_fp4: bool },
    /// ExLlamaV3 trellis (Mia): every linear except norms, router, indexer `weights_proj` and the Engram tables.
    Exl3 { avg_bits: f64, head_bits: f64 },
    /// No quantisation (bf16 everywhere).
    None,
}

/// DSpark / MTP draft blocks (parsed, not executed by lane 1a).
#[derive(Clone, Debug, PartialEq)]
pub struct Dsv41Mtp {
    pub n_layers: usize,
    pub block_size: usize,
    pub noise_token_id: u32,
    pub target_layer_ids: Vec<usize>,
    pub markov_rank: usize,
    pub n_routed_experts: usize,
    pub top_k: usize,
}

/// Engram (n-gram hash table) parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct Dsv41Engram {
    pub layer_ids: Vec<usize>,
    pub num_embeddings: Vec<u64>,
    pub max_ngram: usize,
    pub vocab: u64,
    pub n_heads: usize,
    pub head_dim: usize,
    pub pad_id: u32,
    pub compressed_vocab: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Dsv41Config {
    pub vocab: usize,
    pub dim: usize,
    pub moe_inter: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub swiglu_limit: f32,
    pub norm_eps: f32,
    // rope: base theta for sliding-window-only layers, compress_rope_theta + YaRN for the rest
    pub rope_theta: f32,
    pub compress_rope_theta: f32,
    pub rope_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub original_seq_len: usize,
    // moe
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub top_k: usize,
    pub score_func: ScoreFunc,
    pub norm_topk: bool,
    pub route_scale: f32,
    // sparse attention
    pub window: usize,
    /// One per layer, MTP layers included: 0 sliding window only, `r` compressed `r`-to-1.
    pub compress_ratios: Vec<usize>,
    pub kv_source_layers: Vec<usize>,
    pub index_source_layers: Vec<usize>,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub candidate_source_layer: Option<usize>,
    pub candidate_topk_blocks: usize,
    pub candidate_block_size: usize,
    // hyper-connections
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub engram: Option<Dsv41Engram>,
    pub mtp: Option<Dsv41Mtp>,
    pub quant: Dsv41Quant,
    pub bos: u32,
    pub eos: u32,
}

/// One config value under its HF name or its `ModelArgs` name.
struct Keys<'a> {
    t: &'a Value,
}

impl Keys<'_> {
    fn get(&self, hf: &str, flat: &str) -> Option<&Value> {
        self.t.get(hf).or_else(|| self.t.get(flat)).filter(|v| !v.is_null())
    }
    fn u(&self, hf: &str, flat: &str) -> Result<usize> {
        self.get(hf, flat).and_then(Value::as_u64).map(|v| v as usize).ok_or_else(|| Error::model(format!("deepseek_v41 config: missing integer {hf} / {flat}")))
    }
    fn u_or(&self, hf: &str, flat: &str, d: usize) -> usize {
        self.get(hf, flat).and_then(Value::as_u64).map_or(d, |v| v as usize)
    }
    fn f(&self, hf: &str, flat: &str) -> Result<f32> {
        self.get(hf, flat).and_then(Value::as_f64).map(|v| v as f32).ok_or_else(|| Error::model(format!("deepseek_v41 config: missing number {hf} / {flat}")))
    }
    fn f_or(&self, hf: &str, flat: &str, d: f32) -> f32 {
        self.get(hf, flat).and_then(Value::as_f64).map_or(d, |v| v as f32)
    }
    fn list(&self, hf: &str, flat: &str) -> Vec<u64> {
        self.get(hf, flat).and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_u64).collect()).unwrap_or_default()
    }
    fn ulist(&self, hf: &str, flat: &str) -> Vec<usize> {
        self.list(hf, flat).into_iter().map(|v| v as usize).collect()
    }
}

impl Dsv41Config {
    /// The architecture names this config belongs to (`recipes/deepseek_v41.toml` matches the same list).
    pub const ARCHITECTURES: [&'static str; 1] = ["DeepseekV41ForCausalLM"];

    /// Whether a `config.json` is a DeepSeek-V4.1 one (HF `architectures` / `model_type`, or the official flat form).
    pub fn matches(cfg: &Value) -> bool {
        let arch = cfg.get("architectures").and_then(Value::as_array).is_some_and(|a| a.iter().any(|v| v.as_str().is_some_and(|s| Self::ARCHITECTURES.contains(&s))));
        arch || cfg.get("model_type").and_then(Value::as_str) == Some("deepseek_v41") || (cfg.get("hc_mult").is_some() && cfg.get("compress_ratios").is_some())
    }

    pub fn from_json(cfg: &Value) -> Result<Dsv41Config> {
        let t = cfg.get("text_config").unwrap_or(cfg);
        let k = Keys { t };
        let rs = t.get("rope_scaling");
        let rsf = |key: &str, flat: &str, d: f32| rs.and_then(|r| r.get(key)).and_then(Value::as_f64).map_or_else(|| k.f_or(flat, flat, d), |v| v as f32);
        let score_func = match k.get("scoring_func", "score_func").and_then(Value::as_str).unwrap_or("sqrtsoftplus") {
            "softmax" => ScoreFunc::Softmax,
            "sigmoid" => ScoreFunc::Sigmoid,
            "sqrtsoftplus" => ScoreFunc::SqrtSoftplus,
            other => return Err(Error::model(format!("deepseek_v41 config: score function {other:?}"))),
        };
        let engram_layers = k.ulist("engram_layer_ids", "engram_layer_ids");
        let engram = (!engram_layers.is_empty()).then(|| -> Result<Dsv41Engram> {
            Ok(Dsv41Engram {
                layer_ids: engram_layers.clone(),
                num_embeddings: k.list("engram_num_embeddings", "engram_num_embeddings"),
                max_ngram: k.u("engram_max_ngram_size", "engram_max_ngram_size")?,
                vocab: k.u("engram_vocab_size", "engram_vocab_size")? as u64,
                n_heads: k.u("engram_n_heads", "engram_n_heads")?,
                head_dim: k.u("engram_head_dim", "engram_head_dim")?,
                pad_id: k.u_or("engram_pad_token_id", "engram_pad_id", 2) as u32,
                compressed_vocab: k.u("engram_compressed_vocab_size", "engram_compressed_vocab_size")? as u64,
            })
        });
        let n_mtp = k.u_or("num_nextn_predict_layers", "n_mtp_layers", 0);
        let mtp = (n_mtp > 0 && k.u_or("dspark_block_size", "dspark_block_size", 0) > 0).then(|| Dsv41Mtp {
            n_layers: n_mtp,
            block_size: k.u_or("dspark_block_size", "dspark_block_size", 0),
            noise_token_id: k.u_or("dspark_noise_token_id", "dspark_noise_token_id", 0) as u32,
            target_layer_ids: k.ulist("dspark_target_layer_ids", "dspark_target_layer_ids"),
            markov_rank: k.u_or("dspark_markov_rank", "dspark_markov_rank", 256),
            n_routed_experts: k.u_or("dspark_n_routed_experts", "dspark_n_routed_experts", 0),
            top_k: k.u_or("dspark_num_experts_per_tok", "dspark_n_activated_experts", 0),
        });
        let quant = match cfg.get("quantization_config") {
            Some(q) => match q.get("quant_method").and_then(Value::as_str) {
                Some("exl3") => Dsv41Quant::Exl3 {
                    avg_bits: q.get("avg_bits").and_then(Value::as_f64).unwrap_or(0.0),
                    head_bits: q.get("head_bits").and_then(Value::as_f64).unwrap_or(0.0),
                },
                Some("fp8") => Dsv41Quant::Fp8 {
                    block: q.get("weight_block_size").and_then(|b| b.get(0)).and_then(Value::as_u64).unwrap_or(128) as usize,
                    expert_fp4: q.get("expert_dtype").and_then(Value::as_str) == Some("fp4"),
                },
                other => return Err(Error::Unsupported(format!("deepseek_v41: quant_method {other:?}"))),
            },
            // the official flat config names the dtypes directly
            None => match cfg.get("dtype").and_then(Value::as_str) {
                Some("fp8") => Dsv41Quant::Fp8 { block: 32, expert_fp4: cfg.get("expert_dtype").and_then(Value::as_str) == Some("fp4") },
                _ => Dsv41Quant::None,
            },
        };
        let c = Dsv41Config {
            vocab: k.u("vocab_size", "vocab_size")?,
            dim: k.u("hidden_size", "dim")?,
            moe_inter: k.u("moe_intermediate_size", "moe_inter_dim")?,
            n_layers: k.u("num_hidden_layers", "n_layers")?,
            n_heads: k.u("num_attention_heads", "n_heads")?,
            head_dim: k.u("head_dim", "head_dim")?,
            rope_head_dim: k.u("qk_rope_head_dim", "rope_head_dim")?,
            q_lora_rank: k.u("q_lora_rank", "q_lora_rank")?,
            o_lora_rank: k.u("o_lora_rank", "o_lora_rank")?,
            o_groups: k.u("o_groups", "o_groups")?,
            swiglu_limit: k.f_or("swiglu_limit", "swiglu_limit", 0.0),
            norm_eps: k.f("rms_norm_eps", "norm_eps")?,
            rope_theta: k.f("rope_theta", "rope_theta")?,
            compress_rope_theta: k.f("compress_rope_theta", "compress_rope_theta")?,
            rope_factor: rsf("factor", "rope_factor", 1.0),
            beta_fast: rsf("beta_fast", "beta_fast", 32.0),
            beta_slow: rsf("beta_slow", "beta_slow", 1.0),
            original_seq_len: rsf("original_max_position_embeddings", "original_seq_len", 0.0) as usize,
            n_routed_experts: k.u("n_routed_experts", "n_routed_experts")?,
            n_shared_experts: k.u_or("n_shared_experts", "n_shared_experts", 1),
            top_k: k.u("num_experts_per_tok", "n_activated_experts")?,
            score_func,
            norm_topk: k.get("norm_topk_prob", "norm_topk_prob").and_then(Value::as_bool).unwrap_or(true),
            route_scale: k.f_or("routed_scaling_factor", "route_scale", 1.0),
            window: k.u("sliding_window", "window_size")?,
            compress_ratios: k.ulist("compress_ratios", "compress_ratios"),
            kv_source_layers: k.ulist("kv_source_layer_ids", "kv_source_layers"),
            index_source_layers: k.ulist("index_source_layer_ids", "index_source_layers"),
            index_n_heads: k.u("index_n_heads", "index_n_heads")?,
            index_head_dim: k.u("index_head_dim", "index_head_dim")?,
            index_topk: k.u("index_topk", "index_topk")?,
            candidate_source_layer: k.get("candidate_source_layer_id", "candidate_source_layer").and_then(Value::as_i64).filter(|&v| v >= 0).map(|v| v as usize),
            candidate_topk_blocks: k.u_or("candidate_topk_blocks", "candidate_topk_blocks", 0),
            candidate_block_size: k.u_or("candidate_block_size", "candidate_block_size", 0),
            hc_mult: k.u("hc_mult", "hc_mult")?,
            hc_sinkhorn_iters: k.u("hc_sinkhorn_iters", "hc_sinkhorn_iters")?,
            hc_eps: k.f("hc_eps", "hc_eps")?,
            engram: engram.transpose()?,
            mtp,
            quant,
            bos: cfg.get("bos_token_id").and_then(Value::as_u64).unwrap_or(0) as u32,
            eos: cfg.get("eos_token_id").and_then(Value::as_u64).unwrap_or(1) as u32,
        };
        c.validate()?;
        Ok(c)
    }

    fn validate(&self) -> Result<()> {
        let bad = |m: String| Err(Error::model(format!("deepseek_v41 config: {m}")));
        let n_mtp = self.mtp.as_ref().map_or(0, |m| m.n_layers);
        if self.compress_ratios.len() < self.n_layers {
            return bad(format!("{} compress_ratios for {} layers", self.compress_ratios.len(), self.n_layers));
        }
        if self.compress_ratios.len() != self.n_layers + n_mtp && self.compress_ratios.len() != self.n_layers {
            return bad(format!("{} compress_ratios for {} layers + {n_mtp} MTP layers", self.compress_ratios.len(), self.n_layers));
        }
        if self.head_dim <= self.rope_head_dim || self.rope_head_dim % 2 != 0 || (self.n_heads * self.head_dim) % self.o_groups != 0 {
            return bad(format!("head_dim {} rope_head_dim {} o_groups {}", self.head_dim, self.rope_head_dim, self.o_groups));
        }
        for l in 0..self.n_layers {
            if self.ratio(l) > 0 && (self.kv_source(l).is_none() || self.index_source(l).is_none()) {
                return bad(format!("layer {l} (ratio {}) has no kv / index source at or before it with the same ratio", self.ratio(l)));
            }
        }
        for &s in self.kv_source_layers.iter().chain(&self.index_source_layers) {
            if s >= self.n_layers || self.ratio(s) == 0 {
                return bad(format!("source layer {s} is not a compressed backbone layer"));
            }
        }
        if let Some(c) = self.candidate_source_layer {
            if !self.index_source_layers.contains(&c) || self.candidate_block_size == 0 || self.candidate_topk_blocks == 0 {
                return bad(format!("candidate source {c} must be an index source with a block size and a block count"));
            }
        }
        if let Some(e) = &self.engram {
            if e.layer_ids.len() != e.num_embeddings.len() || e.layer_ids.iter().any(|&l| l >= self.n_layers) {
                return bad(format!("engram layers {:?} / table sizes {:?}", e.layer_ids, e.num_embeddings));
            }
        }
        if self.hc_mult == 0 || self.top_k == 0 || self.top_k > self.n_routed_experts || self.n_shared_experts != 1 {
            return bad(format!("hc_mult {} top_k {} of {} experts, {} shared", self.hc_mult, self.top_k, self.n_routed_experts, self.n_shared_experts));
        }
        Ok(())
    }

    /// `compress_ratios[layer]` (0: sliding window only).
    pub fn ratio(&self, layer: usize) -> usize {
        self.compress_ratios.get(layer).copied().unwrap_or(0)
    }

    /// The layer whose compressed KV (and index keys) `layer` reads: the latest KV source at or before it with the same
    /// ratio (`model.py` `SharedAttentionRuntime.compress_kv` / `index_k` is overwritten by every source in layer order).
    pub fn kv_source(&self, layer: usize) -> Option<usize> {
        let r = self.ratio(layer);
        (r > 0).then(|| self.kv_source_layers.iter().copied().filter(|&s| s <= layer && self.ratio(s) == r).max()).flatten()
    }

    /// The layer whose top-k indices `layer` uses (`SharedAttentionRuntime.topk_idxs`): the latest index source at or
    /// before it with the same ratio.
    pub fn index_source(&self, layer: usize) -> Option<usize> {
        let r = self.ratio(layer);
        (r > 0).then(|| self.index_source_layers.iter().copied().filter(|&s| s <= layer && self.ratio(s) == r).max()).flatten()
    }

    /// Whether `layer`'s indexer masks with the candidate source's blocks (`Indexer.uses_candidates`).
    pub fn uses_candidates(&self, layer: usize) -> bool {
        self.candidate_source_layer.is_some_and(|c| c < layer && self.index_source_layers.contains(&layer))
    }

    /// Rotary frequencies (`precompute_freqs_cis` without the outer product): the base table (theta `rope_theta`, no
    /// YaRN) for sliding-window-only layers, the YaRN table at `compress_rope_theta` for compressed layers.
    pub fn inv_freq(&self, layer: usize) -> Vec<f32> {
        let dim = self.rope_head_dim;
        let (base, orig) = if self.ratio(layer) > 0 { (self.compress_rope_theta, self.original_seq_len) } else { (self.rope_theta, 0) };
        let mut f: Vec<f32> = (0..dim / 2).map(|i| 1.0 / base.powf((2 * i) as f32 / dim as f32)).collect();
        if orig > 0 {
            let corrected = |rot: f32| dim as f64 * (orig as f64 / (rot as f64 * 2.0 * std::f64::consts::PI)).ln() / (2.0 * (base as f64).ln());
            let low = corrected(self.beta_fast).floor().max(0.0);
            let high = corrected(self.beta_slow).ceil().min(dim as f64 - 1.0);
            for (i, v) in f.iter_mut().enumerate() {
                let ramp = ((i as f64 - low) / (high - low).max(1e-3)).clamp(0.0, 1.0) as f32;
                let smooth = 1.0 - ramp;
                *v = *v / self.rope_factor * (1.0 - smooth) + *v * smooth;
            }
        }
        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mia() -> Value {
        serde_json::from_str(include_str!("../../../intarsia-models/tests/fixtures/dsv41/mia-config.json")).unwrap()
    }

    #[test]
    fn hf_and_flat_configs_parse_to_the_same_shapes() {
        let hf = Dsv41Config::from_json(&mia()).unwrap();
        let base = Dsv41Config::from_json(&serde_json::from_str(include_str!("../../../intarsia-models/tests/fixtures/dsv41/base-config.json")).unwrap()).unwrap();
        let flat = Dsv41Config::from_json(&serde_json::from_str(include_str!("../../../intarsia-models/tests/fixtures/dsv41/inference-config.json")).unwrap()).unwrap();
        assert!(Dsv41Config::matches(&mia()));
        assert_eq!((hf.dim, hf.n_layers, hf.n_heads, hf.head_dim, hf.window, hf.hc_mult), (5120, 40, 64, 512, 128, 4));
        assert_eq!(hf.quant, Dsv41Quant::Exl3 { avg_bits: 2.9, head_bits: 6.0 });
        assert_eq!(base.quant, Dsv41Quant::Fp8 { block: 32, expert_fp4: true });
        assert_eq!(flat.quant, Dsv41Quant::Fp8 { block: 32, expert_fp4: true });
        // the three forms describe the same model apart from quantisation
        for c in [&base, &flat] {
            assert_eq!(Dsv41Config { quant: hf.quant.clone(), bos: hf.bos, eos: hf.eos, ..(*c).clone() }, hf);
        }
        assert_eq!(hf.compress_ratios.len(), 43);
        assert_eq!(hf.mtp.as_ref().unwrap().n_layers, 3);
        assert_eq!(hf.engram.as_ref().unwrap().num_embeddings, [384_006_168, 384_016_682]);
    }

    #[test]
    fn shared_sources_follow_layer_order_and_ratio() {
        let c = Dsv41Config::from_json(&mia()).unwrap();
        let kv: Vec<Option<usize>> = (0..40).map(|l| c.kv_source(l)).collect();
        assert_eq!(kv[0..2], [None, None]);
        assert!(kv[2..8].iter().all(|&s| s == Some(2)));
        assert!(kv[14..20].iter().all(|&s| s == Some(14)));
        assert!(kv[20..40].iter().all(|&s| s == Some(20)), "ratio-1 layers all read layer 20's compressed KV and index keys");
        assert_eq!(c.ratio(40), 0, "the MTP blocks are sliding window only");
        let ix: Vec<Option<usize>> = (0..40).map(|l| c.index_source(l)).collect();
        assert_eq!((ix[23], ix[24], ix[27], ix[39]), (Some(20), Some(24), Some(24), Some(36)));
        assert!(c.uses_candidates(24) && !c.uses_candidates(20) && !c.uses_candidates(21));
        // two rope tables: base theta without YaRN (window-only layers), YaRN at compress_rope_theta
        assert_eq!(c.inv_freq(0)[1], 1.0 / 10000f32.powf(2.0 / 64.0));
        assert!(c.inv_freq(2)[31] < 1.0 / 160000f32.powf(62.0 / 64.0), "YaRN divides the long wavelengths");
        assert_eq!(c.inv_freq(2)[0], 1.0, "the shortest wavelengths keep their frequency");
    }
}
