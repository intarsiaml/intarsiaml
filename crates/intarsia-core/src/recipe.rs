//! Recipe types (ARCHITECTURE 3.5). A recipe maps a Hugging Face checkpoint
//! onto library blocks: which blocks make up a layer, and which tensor names
//! feed them. Parsing from TOML lives in `intarsia-models`; the types are here
//! so the generic decoder can walk them without depending on the loader.

use crate::dtype::DType;
use crate::error::{Error, Result};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Deserialize)]
pub struct Recipe {
    pub model: RecipeModel,
    pub layer: LayerSpec,
    pub top: TopSpec,
    #[serde(default)]
    pub defaults: Defaults,
    /// Optional overrides of the `config.json` key names (defaults are the HF names).
    #[serde(default)]
    pub config: ConfigKeys,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RecipeModel {
    #[serde(rename = "match")]
    pub matcher: MatchSpec,
    pub family: String,
    /// "hf": the model's own template via minijinja.
    #[serde(default = "default_hf")]
    pub chat_template: String,
    pub reasoning_parser: Option<String>,
    pub tool_parser: Option<String>,
}

fn default_hf() -> String {
    "hf".into()
}

#[derive(Clone, Debug, Deserialize)]
pub struct MatchSpec {
    pub architectures: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct LayerSpec {
    /// Block names applied in order inside every layer, e.g. `["attn", "ffn"]`.
    /// Each name selects the `[layer.<name>]` table below.
    pub pattern: Vec<String>,
    /// Hybrid models: `config.json` key holding a per-layer type list (parsed, not yet used).
    pub pattern_by_layer: Option<String>,
    pub attn: Option<AttnSpec>,
    pub ffn: Option<FfnSpec>,
    /// `pre_<block>` -> norm tensor prefix; the norm applied before that block.
    #[serde(default)]
    pub norms: BTreeMap<String, String>,
    /// Other block tables (gdn, moe, ...): accepted by the parser, rejected by the builder.
    #[serde(flatten)]
    pub other: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct AttnSpec {
    /// "gqa" (Phase 0). Others (mla, sliding_window, gdn, mamba2) are future block types.
    pub kind: String,
    pub q: String,
    pub k: String,
    pub v: String,
    pub o: String,
    /// Per-head RMSNorm on q / k before rope (Qwen3). Absent = no qk_norm.
    pub q_norm: Option<String>,
    pub k_norm: Option<String>,
    /// "default" = neox/rotate-half with `rope_theta` from config.json.
    #[serde(default = "default_rope")]
    pub rope: String,
}

fn default_rope() -> String {
    "default".into()
}

#[derive(Clone, Debug, Deserialize)]
pub struct FfnSpec {
    /// "dense" (Phase 0).
    pub kind: String,
    /// "silu_glu".
    pub act: String,
    pub gate: String,
    pub up: String,
    pub down: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TopSpec {
    pub embed: String,
    pub final_norm: String,
    pub lm_head: LmHeadSpec,
}

#[derive(Clone, Debug, Deserialize)]
pub struct LmHeadSpec {
    pub tensor: String,
    /// When set and `tensor` is absent from the checkpoint (or `tie_word_embeddings`), reuse this tensor.
    pub tied_to: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Defaults {
    /// KV cache dtype. Phase 0 CPU/CUDA support bf16 / f16 / f32 / fp8_e4m3.
    #[serde(default = "default_kv")]
    pub kv_dtype: String,
    #[serde(default = "default_batch")]
    pub max_batch: usize,
}

fn default_kv() -> String {
    "bf16".into()
}
fn default_batch() -> usize {
    1
}

impl Default for Defaults {
    fn default() -> Self {
        Defaults { kv_dtype: default_kv(), max_batch: 1 }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct ConfigKeys {
    pub hidden_size: String,
    pub num_layers: String,
    pub num_heads: String,
    pub num_kv_heads: String,
    pub head_dim: String,
    pub intermediate_size: String,
    pub vocab_size: String,
    pub rms_eps: String,
    pub rope_theta: String,
    pub tie_embeddings: String,
    pub max_positions: String,
}

impl Default for ConfigKeys {
    fn default() -> Self {
        ConfigKeys {
            hidden_size: "hidden_size".into(),
            num_layers: "num_hidden_layers".into(),
            num_heads: "num_attention_heads".into(),
            num_kv_heads: "num_key_value_heads".into(),
            head_dim: "head_dim".into(),
            intermediate_size: "intermediate_size".into(),
            vocab_size: "vocab_size".into(),
            rms_eps: "rms_norm_eps".into(),
            rope_theta: "rope_theta".into(),
            tie_embeddings: "tie_word_embeddings".into(),
            max_positions: "max_position_embeddings".into(),
        }
    }
}

impl Recipe {
    pub fn kv_dtype(&self) -> Result<DType> {
        DType::parse(&self.defaults.kv_dtype)
    }

    /// Expand `{i}` in a tensor name template.
    pub fn expand(template: &str, layer: usize) -> String {
        template.replace("{i}", &layer.to_string())
    }

    /// Light structural validation (the builder re-checks what it needs).
    pub fn validate(&self) -> Result<()> {
        if self.layer.pattern.is_empty() {
            return Err(Error::model("recipe layer.pattern is empty"));
        }
        for b in &self.layer.pattern {
            let known = match b.as_str() {
                "attn" => self.layer.attn.is_some(),
                "ffn" => self.layer.ffn.is_some(),
                other => self.layer.other.contains_key(other),
            };
            if !known {
                return Err(Error::model(format!(
                    "layer.pattern references block '{b}' but there is no [layer.{b}] table"
                )));
            }
        }
        Ok(())
    }
}

/// Model dimensions resolved from `config.json` through the recipe's key names.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ModelDims {
    pub hidden: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub tie_embeddings: bool,
    pub max_positions: usize,
}

impl ModelDims {
    pub fn from_hf_config(cfg: &serde_json::Value, keys: &ConfigKeys) -> Result<ModelDims> {
        let get_u = |k: &str| cfg.get(k).and_then(|v| v.as_u64()).map(|v| v as usize);
        let need_u = |k: &str| get_u(k).ok_or_else(|| Error::model(format!("config.json missing '{k}'")));
        let hidden = need_u(&keys.hidden_size)?;
        let n_heads = need_u(&keys.num_heads)?;
        let n_kv_heads = get_u(&keys.num_kv_heads).unwrap_or(n_heads);
        // head_dim is explicit for Qwen3 (128 with hidden 1024 / 16 heads) - never assume hidden/heads.
        let head_dim = get_u(&keys.head_dim).unwrap_or(hidden / n_heads);
        let get_f = |k: &str| cfg.get(k).and_then(|v| v.as_f64());
        Ok(ModelDims {
            hidden,
            n_layers: need_u(&keys.num_layers)?,
            n_heads,
            n_kv_heads,
            head_dim,
            intermediate: need_u(&keys.intermediate_size)?,
            vocab: need_u(&keys.vocab_size)?,
            rms_eps: get_f(&keys.rms_eps).unwrap_or(1e-6) as f32,
            rope_theta: get_f(&keys.rope_theta).unwrap_or(10000.0) as f32,
            tie_embeddings: cfg.get(&keys.tie_embeddings).and_then(|v| v.as_bool()).unwrap_or(false),
            max_positions: get_u(&keys.max_positions).unwrap_or(4096),
        })
    }
}
