//! Recipe types (ARCHITECTURE 3.5). A recipe maps a Hugging Face checkpoint
//! onto library blocks: which blocks make up a layer, and which tensor names
//! feed them. Parsing from TOML lives in `intarsia-models`; the types are here
//! so the generic decoder can walk them without depending on the loader.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::rope::RopeScaling;
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recipe {
    pub model: RecipeModel,
    pub layer: LayerSpec,
    pub top: TopSpec,
    #[serde(default)]
    pub defaults: Defaults,
    /// Per-backend performance defaults (`[tuning]`, `[tuning.<backend>]`).
    #[serde(default)]
    pub tuning: Tuning,
    /// Optional overrides of the `config.json` key names (defaults are the HF names).
    #[serde(default)]
    pub config: ConfigKeys,
    /// What `intarsia accuracy` needs to know about the model (`[accuracy]`): no effect on loading or serving.
    #[serde(default)]
    pub accuracy: AccuracySpec,
}

/// `[accuracy]` table of a recipe, read only by `intarsia accuracy`.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AccuracySpec {
    /// Turn format of the model's chat template (`chatml`): which rows of a chat prompt the model is asked to predict
    /// (the KL floor skips the system and user turns). Fallback for a golden whose metadata names no known format.
    pub chat_turns: Option<String>,
    /// Prefix-reuse judge: fixed rounding scale (nats of log-probability over the top-5 tokens) within which a warm and
    /// a cold distribution count as one when their greedy streams diverge. Measured once on a known-good build of this
    /// model, never derived per run. Unset: the tool's built-in constant.
    pub reuse_logit_tol: Option<f64>,
    /// Prefix-reuse judge: largest KL(cold || warm) in nats over the shared rows that still counts as rounding.
    pub reuse_kl_tol: Option<f64>,
    /// Factor on the measured reference noise of the golden (`intarsia accuracy`: KL floor = max(class floor, factor x
    /// noise)). A per-golden `factor` in the noise record wins; unset: 1.0.
    pub ref_noise_factor: Option<f64>,
    /// `intarsia accuracy --self-golden --self-vs-ref-noise`: the worst prompt's mean KL may be this many times the
    /// reference noise. A per-golden `prompt_factor` wins; unset: 3.0.
    pub ref_noise_prompt_factor: Option<f64>,
    /// Absolute top-1 floor in percent, for a model whose bf16 forward cannot reach the class default (99.5%) against the
    /// fp32 golden: the gate uses `min(this, class default)`, never a higher one. Set it from a measurement (the top-1 of
    /// the reference's own bf16 forward against the fp32 golden, minus 0.5 points) and cite it in a comment. `--min-top1`
    /// wins. Unset: the class default. Ignored when the class default is "not gated" (format-reference goldens).
    pub min_top1: Option<f64>,
    /// `intarsia qualify` quick tier (`acc-short`): teacher-forced decode positions per prompt (clamped to the prompt
    /// length - 1). Raise it for a model whose decode top-1 is noisy at the default 16 positions per prompt (160 rows on
    /// the short golden); the gate is unchanged. Unset: 16.
    pub decode_positions: Option<usize>,
    /// `intarsia accuracy --quick` (the `gate` stage of `qualify` / `onboard`): the form of the perplexity text and of the
    /// greedy sanity prompts. `"raw"` (the built-in): `tools/prompts/ppl.txt` as plain text and the five completion
    /// prompts. `"chat"`: for an instruction-only model that has no usable raw-text distribution (Gemma-4: HF bf16 520,
    /// vLLM 402 on the raw text): the fixed conversation `tools/prompts/ppl_chat.json` rendered with the model's own chat
    /// template, perplexity over the assistant tokens only, and the sanity prompts sent as user turns. `--ppl-form`
    /// wins. Unset: `"raw"`.
    pub ppl_form: Option<String>,
    /// Quick gate: the largest perplexity (`--max-ppl` wins). Measured per form and model, with the reference's own
    /// number on the same text cited in a comment next to it. Unset: 40.
    pub max_ppl: Option<f64>,
}

/// The two forms of the quick gate's perplexity text (`[accuracy] ppl_form`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PplForm {
    Raw,
    Chat,
}

impl AccuracySpec {
    /// `ppl_form` as an enum; an unknown name is an error (a typo must not silently keep the raw gate).
    pub fn ppl_form(&self) -> Result<PplForm> {
        match self.ppl_form.as_deref().map(str::trim) {
            None | Some("raw") => Ok(PplForm::Raw),
            Some("chat") => Ok(PplForm::Chat),
            Some(o) => Err(Error::model(format!("[accuracy] ppl_form = {o:?}: expected \"raw\" or \"chat\""))),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeModel {
    #[serde(rename = "match")]
    pub matcher: MatchSpec,
    pub family: String,
    /// "hf": the model's own template via minijinja.
    #[serde(default = "default_hf")]
    pub chat_template: String,
    pub reasoning_parser: Option<String>,
    pub tool_parser: Option<String>,
    /// Zero-centred RMSNorm (Qwen3.5/3.6: `x * (1 + w)`): added to every norm weight in the f32 math of the
    /// pre-block norms, the final norm and the q/k norms (not the gated norm inside GatedDeltaNet).
    #[serde(default)]
    pub norm_offset: f32,
}

fn default_hf() -> String {
    "hf".into()
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatchSpec {
    pub architectures: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct LayerSpec {
    /// Block names applied in order inside every layer, e.g. `["attn", "ffn"]`.
    /// Each name selects the `[layer.<name>]` table below.
    pub pattern: Vec<String>,
    /// Hybrid models: `config.json` key holding a per-layer type list (must equal `[config] layer_types`,
    /// default `"layer_types"`). When set, layer `i` uses the block pattern `by_type[layer_types[i]]`
    /// instead of `pattern`; a layer type without an entry is a load error.
    pub pattern_by_layer: Option<String>,
    /// Block pattern per layer type, e.g. `linear_attention = ["gdn", "ffn"]`.
    #[serde(default)]
    pub by_type: BTreeMap<String, Vec<String>>,
    pub attn: Option<AttnSpec>,
    pub gdn: Option<GdnSpec>,
    pub ffn: Option<FfnSpec>,
    /// `pre_<block>` -> norm tensor prefix; the norm applied before that block.
    #[serde(default)]
    pub norms: BTreeMap<String, String>,
    /// Other block tables (gdn, moe, ...): accepted by the parser, rejected by the builder.
    #[serde(flatten)]
    pub other: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
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
    /// "default" = neox/rotate-half with `rope_theta` from config.json (the first
    /// `partial_rotary_factor * head_dim` values of each head rotate).
    #[serde(default = "default_rope")]
    pub rope: String,
    /// Projection bias tensors (Qwen2: `model.layers.{i}.self_attn.q_proj.bias`, ...), added after the matching
    /// projection, before the norm / rope. Each independently optional; absent = no bias.
    #[serde(default)]
    pub q_bias: Option<String>,
    #[serde(default)]
    pub k_bias: Option<String>,
    #[serde(default)]
    pub v_bias: Option<String>,
    /// `q` projects to `[n_heads, 2 * head_dim]` per token: the query and a per-head output gate
    /// (`attn_out * sigmoid(gate)` before `o`), interleaved per head (HF `attn_output_gate`).
    #[serde(default)]
    pub gated: bool,
}

/// Gated DeltaNet linear-attention block (Qwen3-Next / Qwen3.5 / Qwen3.6): tensor name templates.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GdnSpec {
    /// "gated_delta_net".
    pub kind: String,
    /// Fused q/k/v projection `[2*Hk*dk + Hv*dv, hidden]`.
    pub in_qkv: String,
    /// Output-gate projection `[Hv*dv, hidden]`.
    pub in_z: String,
    /// Beta projection `[Hv, hidden]`.
    pub in_b: String,
    /// Decay projection `[Hv, hidden]`.
    pub in_a: String,
    /// Depthwise causal conv taps `[C, 1, K]` (or `[C, K]`).
    pub conv: String,
    pub a_log: String,
    pub dt_bias: String,
    /// Gated RMSNorm weight `[dv]` (plain weight, never offset).
    pub norm: String,
    pub out: String,
}

fn default_rope() -> String {
    "default".into()
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
pub struct TopSpec {
    pub embed: String,
    pub final_norm: String,
    pub lm_head: LmHeadSpec,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LmHeadSpec {
    pub tensor: String,
    /// When set and `tensor` is absent from the checkpoint (or `tie_word_embeddings`), reuse this tensor.
    pub tied_to: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Defaults {
    /// KV cache dtype. Phase 0 CPU/CUDA support bf16 / f16 / f32 / fp8_e4m3.
    #[serde(default = "default_kv")]
    pub kv_dtype: String,
    #[serde(default = "default_batch")]
    pub max_batch: usize,
    /// FP8 checkpoints: `static` (W8A8, activations quantised with the checkpoint's `input_scale`) or `none`
    /// (W8A16). See `quant::fp8` and docs/quant-moe.md.
    #[serde(default)]
    pub fp8_act: crate::quant::Fp8Act,
    /// The checkpoint declares `NVFP4` (weights AND 4-bit activations, W4A4) but the engine runs the experts weight-only
    /// (W4A16): off = refuse to load it (a deviation from the declared scheme and from vLLM); on = accept, knowingly.
    #[serde(default)]
    pub w4a4_as_w4a16: bool,
}

fn default_kv() -> String {
    "bf16".into()
}
fn default_batch() -> usize {
    1
}

impl Default for Defaults {
    fn default() -> Self {
        Defaults { kv_dtype: default_kv(), max_batch: 1, fp8_act: Default::default(), w4a4_as_w4a16: false }
    }
}

/// Prefill chunk used when neither the flag nor the recipe says otherwise.
pub const DEFAULT_PREFILL_CHUNK: usize = 256;

/// Performance defaults a recipe may carry: `[tuning] prefill_chunk = N` for every backend, `[tuning.<backend>]`
/// (`cpu`, `cuda`, `hip`) to override it for one. A backend without a table falls back to the base value.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct Tuning {
    pub prefill_chunk: Option<usize>,
    #[serde(flatten)]
    pub backends: BTreeMap<String, BackendTuning>,
}

/// A resolved `--max-batch` and where it came from.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct MaxBatch {
    pub value: usize,
    /// `--max-batch` or `recipe <family>` (clamped: `recipe <family>, clamped to N`).
    pub source: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct BackendTuning {
    pub prefill_chunk: Option<usize>,
}

/// A resolved prefill chunk and where it came from.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct PrefillChunk {
    pub value: usize,
    /// `--prefill-chunk`, `recipe <family>, <backend>`, `recipe <family>, base` or `default`.
    pub source: String,
}

impl PrefillChunk {
    /// `2048 (recipe qwen3_5_moe, cuda)`.
    pub fn describe(&self) -> String {
        format!("{} ({})", self.value, self.source)
    }
}

impl Recipe {
    /// Prefill chunk for `backend`: the explicit flag, then the recipe's value for that backend, then the recipe's
    /// base value, then [`DEFAULT_PREFILL_CHUNK`].
    pub fn resolve_prefill_chunk(&self, flag: Option<usize>, backend: &str) -> PrefillChunk {
        let family = &self.model.family;
        if let Some(value) = flag {
            return PrefillChunk { value, source: "--prefill-chunk".into() };
        }
        if let Some(value) = self.tuning.backends.get(backend).and_then(|b| b.prefill_chunk) {
            return PrefillChunk { value, source: format!("recipe {family}, {backend}") };
        }
        if let Some(value) = self.tuning.prefill_chunk {
            return PrefillChunk { value, source: format!("recipe {family}, base") };
        }
        PrefillChunk { value: DEFAULT_PREFILL_CHUNK, source: "default".into() }
    }
}

impl Recipe {
    /// Concurrent sequences to decode together. `supported` is the most the serving engine implements. An explicit
    /// `flag` must be in `1..=min(recipe max_batch, supported)`; without one the recipe's `[defaults] max_batch` is
    /// used, clamped to `supported` (a recipe may declare more than the engine can do yet: the qwen3_5 ones say 32).
    pub fn resolve_max_batch(&self, flag: Option<usize>, supported: usize) -> Result<MaxBatch> {
        let family = &self.model.family;
        let cap = self.defaults.max_batch.max(1);
        match flag {
            Some(0) => Err(Error::invalid("--max-batch must be at least 1")),
            Some(n) if n > supported => Err(Error::Unsupported(format!(
                "--max-batch {n} is not implemented yet (the engine decodes {supported} sequence(s) at a time; \
                 continuous batching is milestone M1 of docs/research/continuous-batching-design.md); use --max-batch {supported}"
            ))),
            Some(n) if n > cap => {
                Err(Error::invalid(format!("--max-batch {n} exceeds the max_batch {cap} of recipe {family}")))
            }
            Some(n) => Ok(MaxBatch { value: n, source: "--max-batch".into() }),
            None if cap > supported => {
                Ok(MaxBatch { value: supported, source: format!("recipe {family}, clamped to {supported}") })
            }
            None => Ok(MaxBatch { value: cap, source: format!("recipe {family}") }),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
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
    /// Per-layer block type list (hybrid models).
    pub layer_types: String,
    /// Linear-attention (GatedDeltaNet) dimensions.
    pub linear_k_heads: String,
    pub linear_v_heads: String,
    pub linear_k_head_dim: String,
    pub linear_v_head_dim: String,
    pub linear_conv_kernel: String,
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
            layer_types: "layer_types".into(),
            linear_k_heads: "linear_num_key_heads".into(),
            linear_v_heads: "linear_num_value_heads".into(),
            linear_k_head_dim: "linear_key_head_dim".into(),
            linear_v_head_dim: "linear_value_head_dim".into(),
            linear_conv_kernel: "linear_conv_kernel_dim".into(),
        }
    }
}

impl LayerSpec {
    /// Block pattern of layer `i`: `pattern`, or `by_type[layer_types[i]]` for hybrid recipes.
    pub fn pattern_for(&self, i: usize, layer_types: &[String]) -> Result<&[String]> {
        if self.pattern_by_layer.is_none() {
            return Ok(&self.pattern);
        }
        let ty = layer_types.get(i).ok_or_else(|| {
            Error::model(format!(
                "recipe maps blocks by layer type but config.json has {} layer types (layer {i} has none)",
                layer_types.len()
            ))
        })?;
        self.by_type.get(ty).map(|v| v.as_slice()).ok_or_else(|| {
            let known: Vec<&str> = self.by_type.keys().map(|k| k.as_str()).collect();
            Error::model(format!("layer {i} has type '{ty}' but the recipe only maps {known:?}"))
        })
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
        let chunks = self.tuning.prefill_chunk.iter().chain(self.tuning.backends.values().filter_map(|b| b.prefill_chunk.as_ref()));
        if chunks.into_iter().any(|&c| c == 0) {
            return Err(Error::model("recipe [tuning] prefill_chunk must be at least 1"));
        }
        self.accuracy.ppl_form()?;
        if self.accuracy.max_ppl.is_some_and(|m| !(m.is_finite() && m > 0.0)) {
            return Err(Error::model("recipe [accuracy] max_ppl must be a positive number"));
        }
        if self.layer.pattern.is_empty() {
            return Err(Error::model("recipe layer.pattern is empty"));
        }
        let all = self.layer.pattern.iter().chain(self.layer.by_type.values().flatten());
        for b in all {
            let known = match b.as_str() {
                "attn" => self.layer.attn.is_some(),
                "gdn" => self.layer.gdn.is_some(),
                "ffn" => self.layer.ffn.is_some(),
                other => self.layer.other.contains_key(other),
            };
            if !known {
                return Err(Error::model(format!(
                    "layer pattern references block '{b}' but there is no [layer.{b}] table"
                )));
            }
        }
        if let Some(key) = &self.layer.pattern_by_layer {
            if *key != self.config.layer_types {
                return Err(Error::model(format!(
                    "layer.pattern_by_layer = \"{key}\" must equal [config] layer_types = \"{}\"",
                    self.config.layer_types
                )));
            }
            if self.layer.by_type.is_empty() || self.layer.by_type.values().any(|p| p.is_empty()) {
                return Err(Error::model("layer.pattern_by_layer needs a non-empty [layer.by_type] pattern per type"));
            }
        } else if !self.layer.by_type.is_empty() {
            return Err(Error::model("[layer.by_type] needs layer.pattern_by_layer"));
        }
        Ok(())
    }
}

fn is_unscaled(r: &RopeScaling) -> bool {
    !r.is_scaled()
}

/// Model dimensions resolved from `config.json` through the recipe's key names.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ModelDims {
    pub hidden: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Dense FFN width; 0 for MoE-only checkpoints (no `intermediate_size` key).
    pub intermediate: usize,
    pub vocab: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    /// Fraction of `head_dim` that rotary embeddings rotate (HF `partial_rotary_factor`, 1.0 when absent).
    /// The generic GQA block rotates the full head and refuses other values.
    pub partial_rotary_factor: f32,
    /// `rope_type` frequency scaling (llama3, yarn) from `rope_parameters` / `rope_scaling`; [`RopeScaling::None`]
    /// for the plain embedding. Not serialised when unscaled, so existing models' dims JSON is unchanged.
    #[serde(skip_serializing_if = "is_unscaled")]
    pub rope_scaling: RopeScaling,
    pub tie_embeddings: bool,
    pub max_positions: usize,
    /// HF `layer_types` (one entry per layer: "full_attention", "linear_attention", ...); empty when absent.
    pub layer_types: Vec<String>,
    /// GatedDeltaNet dimensions when the config carries them.
    pub linear: Option<LinearDims>,
    /// Sparse MoE FFN dimensions when the config carries them.
    pub moe: Option<crate::quant::moe::MoeConfig>,
    /// Gemma 4 text-model dimensions (`model_type = "gemma4_text"`); `None` for every other family.
    pub gemma: Option<GemmaDims>,
}

/// What a Gemma 4 text config adds to [`ModelDims`] (HF `Gemma4TextConfig`): sliding / global attention layers with their
/// own head sizes and RoPE, `attention_k_eq_v`, the final logit soft-cap, the embedding scale. `ModelDims::head_dim` /
/// `n_kv_heads` / `rope_theta` are the SLIDING layers' (`head_dim`, `num_key_value_heads`, `rope_parameters.sliding_attention`).
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct GemmaDims {
    /// Keys a sliding layer attends (itself included).
    pub sliding_window: usize,
    /// Global (`full_attention`) layers: head size, KV heads, RoPE base and the fraction of the head that rotates.
    pub global_head_dim: usize,
    pub global_kv_heads: usize,
    pub global_theta: f32,
    pub global_rotary_factor: f32,
    /// Global layers have no `v_proj`: V is the (unit-normed) K projection (`attention_k_eq_v`).
    pub k_eq_v: bool,
    /// `final_logit_softcapping` (`cap * tanh(logit / cap)`), 0 = none.
    pub softcap: f32,
    /// Embedding multiplier, `sqrt(hidden_size)` rounded to bf16 like HF's `embed_scale.to(weight.dtype)`.
    pub embed_scale: f32,
}

impl GemmaDims {
    /// `Some` for a `gemma4_text` config (the text config of a `Gemma4ForConditionalGeneration` wrapper or the config itself).
    pub fn from_hf_config(cfg: &serde_json::Value) -> Result<Option<GemmaDims>> {
        let tc = text_config(cfg);
        if tc.get("model_type").and_then(|v| v.as_str()) != Some("gemma4_text") {
            return Ok(None);
        }
        let u = |k: &str| tc.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| Error::model(format!("config.json missing '{k}'")));
        let rp = tc.get("rope_parameters").filter(|v| v.is_object()).ok_or_else(|| Error::model("gemma4 config: rope_parameters missing"))?;
        let ty = |k: &str, want: &str| -> Result<serde_json::Value> {
            let v = rp.get(k).filter(|v| v.is_object()).ok_or_else(|| Error::model(format!("gemma4 config: rope_parameters.{k} missing")))?;
            let t = v.get("rope_type").and_then(|t| t.as_str()).unwrap_or("default");
            if t != want {
                return Err(Error::Unsupported(format!("gemma4 config: rope_parameters.{k}.rope_type = \"{t}\" (only \"{want}\" is implemented)")));
            }
            Ok(v.clone())
        };
        let _local = ty("sliding_attention", "default")?;
        let global = ty("full_attention", "proportional")?;
        let g = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_f64());
        if g(&global, "factor").is_some_and(|f| (f - 1.0).abs() > 1e-9) {
            return Err(Error::Unsupported("gemma4 config: proportional rope with factor != 1".into()));
        }
        let hidden = u("hidden_size")?;
        // HF rounds sqrt(hidden) to the weight dtype (bf16) before the multiply
        let embed_scale = half::bf16::from_f32((hidden as f32).sqrt()).to_f32();
        let rot = g(&global, "partial_rotary_factor").unwrap_or(1.0) as f32;
        let ghd = u("global_head_dim")?;
        if !(rot > 0.0 && rot <= 1.0) || ((rot * ghd as f32) as usize) % 2 != 0 {
            return Err(Error::model(format!("gemma4 config: global partial_rotary_factor {rot} on head_dim {ghd}")));
        }
        if tc.get("num_kv_shared_layers").and_then(|v| v.as_u64()).unwrap_or(0) != 0 {
            return Err(Error::Unsupported("gemma4 config: num_kv_shared_layers != 0".into()));
        }
        if tc.get("hidden_size_per_layer_input").and_then(|v| v.as_u64()).unwrap_or(0) != 0 {
            return Err(Error::Unsupported("gemma4 config: hidden_size_per_layer_input != 0 (per-layer embeddings)".into()));
        }
        if tc.get("hidden_activation").and_then(|v| v.as_str()) != Some("gelu_pytorch_tanh") {
            return Err(Error::Unsupported(format!("gemma4 config: hidden_activation = {:?} (only gelu_pytorch_tanh)", tc.get("hidden_activation"))));
        }
        if !tc.get("enable_moe_block").and_then(|v| v.as_bool()).unwrap_or(false) {
            return Err(Error::Unsupported("gemma4 config: enable_moe_block = false (dense Gemma 4 is not implemented)".into()));
        }
        if tc.get("attention_bias").and_then(|v| v.as_bool()) == Some(true) {
            return Err(Error::Unsupported("gemma4 config: attention_bias = true".into()));
        }
        Ok(Some(GemmaDims {
            sliding_window: u("sliding_window")?,
            global_head_dim: ghd,
            global_kv_heads: u("num_global_key_value_heads")?,
            global_theta: g(&global, "rope_theta").ok_or_else(|| Error::model("gemma4 config: full_attention rope_theta missing"))? as f32,
            global_rotary_factor: rot,
            k_eq_v: tc.get("attention_k_eq_v").and_then(|v| v.as_bool()).unwrap_or(false),
            softcap: tc.get("final_logit_softcapping").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            embed_scale,
        }))
    }
}

/// Dimensions of a GatedDeltaNet (linear attention) layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct LinearDims {
    pub k_heads: usize,
    pub v_heads: usize,
    pub k_head_dim: usize,
    pub v_head_dim: usize,
    pub conv_kernel: usize,
}

impl LinearDims {
    /// Width of the fused q/k/v projection and of the conv: `2 * Hk * dk + Hv * dv`.
    pub fn conv_dim(&self) -> usize {
        2 * self.k_heads * self.k_head_dim + self.v_heads * self.v_head_dim
    }
}

/// The sub-config holding the text model: `text_config` of multimodal wrappers
/// (`Qwen3_5MoeForConditionalGeneration`, ...), otherwise the config itself.
pub fn text_config(cfg: &serde_json::Value) -> &serde_json::Value {
    cfg.get("text_config").filter(|v| v.is_object()).unwrap_or(cfg)
}

/// Reject configuration this engine would silently run wrong. Errors name the offending key.
pub fn check_supported(cfg: &serde_json::Value) -> Result<()> {
    let tc = text_config(cfg);
    // Gemma 4 (sliding-window layers, per-layer-type proportional rope, GELU) is validated by `GemmaDims::from_hf_config`
    if GemmaDims::from_hf_config(cfg)?.is_some() {
        return Ok(());
    }
    let unsupported = |m: String| Err(Error::Unsupported(m));
    if let Some(act) = tc.get("hidden_act").and_then(|v| v.as_str()) {
        if act != "silu" {
            return unsupported(format!("config hidden_act = \"{act}\" (only silu is implemented)"));
        }
    }
    // q / k / v biases are loaded when the recipe maps them (`[layer.attn] q_bias / k_bias / v_bias`, Qwen2 sets them
    // without any config key); a config `attention_bias = true` means HF Llama's layout, which also biases `o_proj`
    if tc.get("attention_bias").and_then(|v| v.as_bool()) == Some(true) {
        return unsupported("config attention_bias = true (o_proj bias is not implemented; q / k / v biases are mapped in the recipe, [layer.attn] q_bias / k_bias / v_bias)".into());
    }
    if tc.get("use_sliding_window").and_then(|v| v.as_bool()) == Some(true) {
        return unsupported("config use_sliding_window = true (sliding-window attention is not implemented)".into());
    }
    if let Some(lt) = tc.get("layer_types").and_then(|v| v.as_array()) {
        if lt.iter().any(|t| t.as_str() == Some("sliding_attention")) {
            return unsupported("config layer_types contains sliding_attention (not implemented)".into());
        }
    }
    // rope: the plain embedding, llama3 and yarn scaling; every other rope_type would silently change positions
    if let Some(v) = tc.get("rope_parameters") {
        crate::rope::check_rope_type("rope_parameters", v)?;
    }
    if let Some(v) = tc.get("rope_scaling") {
        crate::rope::check_rope_type("rope_scaling", v)?;
    }
    Ok(())
}

impl ModelDims {
    /// The dimensions one tensor-parallel rank works with: attention heads, KV heads and the dense FFN width divided
    /// by `world`; GatedDeltaNet key and value heads (the recurrent state follows them); the MoE expert and shared-expert
    /// widths (tensor parallel inside every expert, docs/design/35b-tp2.md). Hidden size, head sizes, the expert count,
    /// top-k, vocabulary and the rest are replicated.
    pub fn sharded(&self, world: usize) -> crate::error::Result<ModelDims> {
        use crate::error::Error;
        if world <= 1 {
            return Ok(self.clone());
        }
        if self.gemma.is_some() {
            return Err(Error::Unsupported("tensor parallelism of Gemma 4 (per-layer-type head counts, parallel dense + MoE FFN)".into()));
        }
        let mut checks = vec![("attention heads", self.n_heads), ("KV heads", self.n_kv_heads), ("FFN width", self.intermediate)];
        if let Some(l) = &self.linear {
            checks.extend([("GatedDeltaNet key heads", l.k_heads), ("GatedDeltaNet value heads", l.v_heads)]);
        }
        if let Some(m) = &self.moe {
            checks.extend([("MoE expert width", m.inter), ("MoE shared-expert width", m.shared_inter)]);
        }
        for (what, v) in checks {
            if v % world != 0 {
                return Err(Error::Unsupported(format!(
                    "tensor parallelism {world}: {what} ({v}) must be divisible by {world} (n_heads {}, n_kv_heads {}, intermediate {})",
                    self.n_heads, self.n_kv_heads, self.intermediate
                )));
            }
        }
        Ok(ModelDims {
            n_heads: self.n_heads / world,
            n_kv_heads: self.n_kv_heads / world,
            intermediate: self.intermediate / world,
            linear: self.linear.map(|l| LinearDims { k_heads: l.k_heads / world, v_heads: l.v_heads / world, ..l }),
            moe: self.moe.clone().map(|m| crate::quant::moe::MoeConfig { inter: m.inter / world, shared_inter: m.shared_inter / world, ..m }),
            ..self.clone()
        })
    }
}

impl ModelDims {
    pub fn from_hf_config(cfg: &serde_json::Value, keys: &ConfigKeys) -> Result<ModelDims> {
        check_supported(cfg)?;
        let tc = text_config(cfg);
        let gemma = GemmaDims::from_hf_config(cfg)?;
        let get_u = |k: &str| tc.get(k).and_then(|v| v.as_u64()).map(|v| v as usize);
        let need_u = |k: &str| get_u(k).ok_or_else(|| Error::model(format!("config.json missing '{k}'")));
        let get_f = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_f64());
        let hidden = need_u(&keys.hidden_size)?;
        let n_heads = need_u(&keys.num_heads)?;
        let n_kv_heads = get_u(&keys.num_kv_heads).unwrap_or(n_heads);
        // head_dim is explicit for Qwen3 (128 with hidden 1024 / 16 heads) - never assume hidden/heads.
        let head_dim = get_u(&keys.head_dim).unwrap_or(hidden / n_heads);

        // No silent defaults for the numbers that change every logit: eps, rope theta, context length.
        let rms_eps = get_f(tc, &keys.rms_eps)
            .ok_or_else(|| Error::model(format!("config.json missing '{}' (no default is assumed)", keys.rms_eps)))?;
        // transformers 5 keeps rope settings in `rope_parameters`; older checkpoints have a top-level key.
        let rp = tc.get("rope_parameters").filter(|v| v.is_object());
        // Gemma 4 keeps one rope table per layer type: `ModelDims` carries the sliding layers', `GemmaDims` the global ones'
        let rp = if gemma.is_some() { rp.and_then(|r| r.get("sliding_attention")).filter(|v| v.is_object()) } else { rp };
        let theta_new = rp.and_then(|r| get_f(r, &keys.rope_theta));
        let theta_old = get_f(tc, &keys.rope_theta);
        let rope_theta = match (theta_new, theta_old) {
            (Some(a), Some(b)) if (a - b).abs() > 1e-6 * a.abs().max(1.0) => {
                return Err(Error::model(format!(
                    "config.json: rope_parameters.{k} = {a} conflicts with {k} = {b}",
                    k = keys.rope_theta
                )))
            }
            (Some(a), _) | (None, Some(a)) => a,
            (None, None) => {
                return Err(Error::model(format!(
                    "config.json has neither rope_parameters.{k} nor {k} (no default is assumed)",
                    k = keys.rope_theta
                )))
            }
        };
        let partial_rotary_factor = rp
            .and_then(|r| get_f(r, "partial_rotary_factor"))
            .or_else(|| get_f(tc, "partial_rotary_factor"))
            .unwrap_or(1.0);
        if !(partial_rotary_factor > 0.0 && partial_rotary_factor <= 1.0) {
            return Err(Error::model(format!("partial_rotary_factor {partial_rotary_factor} outside (0, 1]")));
        }
        let max_positions = need_u(&keys.max_positions)
            .map_err(|_| Error::model(format!("config.json missing '{}' (no default is assumed)", keys.max_positions)))?;
        // frequency scaling: `rope_parameters` (transformers >= 5) or the older `rope_scaling` block
        let rope_scaling = {
            let from = |k: &str| -> Result<RopeScaling> {
                match tc.get(k) {
                    Some(v) => RopeScaling::from_config(v, tc, max_positions),
                    None => Ok(RopeScaling::None),
                }
            };
            match (from("rope_parameters")?, from("rope_scaling")?) {
                (RopeScaling::None, b) => b,
                (a, RopeScaling::None) => a,
                (a, b) if a == b => a,
                (a, b) => {
                    return Err(Error::model(format!(
                        "config.json: rope_parameters ({}) and rope_scaling ({}) disagree",
                        a.name(),
                        b.name()
                    )))
                }
            }
        };
        let is_moe = ["num_experts", "num_local_experts", "moe_intermediate_size"].iter().any(|k| tc.get(*k).is_some());
        let gemma_moe = || -> Result<crate::quant::moe::MoeConfig> {
            let u = |k: &str| get_u(k).ok_or_else(|| Error::model(format!("config.json missing '{k}'")));
            Ok(crate::quant::moe::MoeConfig {
                hidden,
                experts: u("num_experts")?,
                top_k: u("top_k_experts")?,
                inter: u("moe_intermediate_size")?,
                shared_inter: 0,
                norm_topk: true,
            })
        };
        let intermediate = match get_u(&keys.intermediate_size) {
            Some(v) => v,
            None if is_moe => 0,
            None => need_u(&keys.intermediate_size)?,
        };
        let tie_embeddings = tc
            .get(&keys.tie_embeddings)
            .or_else(|| cfg.get(&keys.tie_embeddings))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let n_layers = need_u(&keys.num_layers)?;
        let layer_types: Vec<String> = match tc.get(&keys.layer_types).and_then(|v| v.as_array()) {
            Some(a) => a.iter().map(|v| v.as_str().map(String::from)).collect::<Option<Vec<_>>>().ok_or_else(|| {
                Error::model(format!("config.json '{}' must be a list of strings", keys.layer_types))
            })?,
            None => Vec::new(),
        };
        if !layer_types.is_empty() && layer_types.len() != n_layers {
            return Err(Error::model(format!(
                "config.json '{}' has {} entries for {n_layers} layers",
                keys.layer_types,
                layer_types.len()
            )));
        }
        let linear_keys = [
            &keys.linear_k_heads,
            &keys.linear_v_heads,
            &keys.linear_k_head_dim,
            &keys.linear_v_head_dim,
            &keys.linear_conv_kernel,
        ];
        let linear = if linear_keys.iter().any(|k| tc.get(k.as_str()).is_some()) {
            let lu = |k: &String| need_u(k.as_str());
            let l = LinearDims {
                k_heads: lu(&keys.linear_k_heads)?,
                v_heads: lu(&keys.linear_v_heads)?,
                k_head_dim: lu(&keys.linear_k_head_dim)?,
                v_head_dim: lu(&keys.linear_v_head_dim)?,
                conv_kernel: lu(&keys.linear_conv_kernel)?,
            };
            if l.k_heads == 0 || l.v_heads % l.k_heads != 0 || l.conv_kernel < 2 {
                return Err(Error::model(format!(
                    "linear attention dims {l:?}: value heads must be a multiple of key heads and conv kernel >= 2"
                )));
            }
            Some(l)
        } else {
            None
        };
        Ok(ModelDims {
            hidden,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            intermediate,
            vocab: need_u(&keys.vocab_size)?,
            rms_eps: rms_eps as f32,
            rope_theta: rope_theta as f32,
            partial_rotary_factor: partial_rotary_factor as f32,
            rope_scaling,
            tie_embeddings,
            max_positions,
            layer_types,
            linear,
            moe: if gemma.is_some() {
                Some(gemma_moe()?)
            } else if is_moe {
                Some(crate::quant::moe::MoeConfig::from_hf_config(cfg)?)
            } else {
                None
            },
            gemma,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn base() -> serde_json::Value {
        json!({
            "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 16, "intermediate_size": 128, "vocab_size": 300, "rms_norm_eps": 1e-6,
            "max_position_embeddings": 4096, "rope_theta": 1000000.0, "hidden_act": "silu",
            "attention_bias": false, "use_sliding_window": false, "rope_scaling": null,
            "tie_word_embeddings": true
        })
    }
    fn dims(c: &serde_json::Value) -> Result<ModelDims> {
        ModelDims::from_hf_config(c, &ConfigKeys::default())
    }

    #[test]
    fn legacy_config_loads() {
        let d = dims(&base()).unwrap();
        assert_eq!((d.rope_theta, d.rms_eps, d.max_positions), (1_000_000.0, 1e-6, 4096));
        assert!(d.tie_embeddings);
        assert_eq!(d.partial_rotary_factor, 1.0);
    }

    #[test]
    fn rope_parameters_layout_and_text_config() {
        let mut inner = base();
        inner.as_object_mut().unwrap().remove("rope_theta");
        inner["rope_parameters"] = json!({"rope_theta": 10000000, "rope_type": "default", "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10], "mrope_interleaved": true});
        let wrapped = json!({"architectures": ["Qwen3_5MoeForConditionalGeneration"], "text_config": inner,
            "tie_word_embeddings": false});
        let d = dims(&wrapped).unwrap();
        assert_eq!(d.rope_theta, 1e7);
        assert_eq!(d.partial_rotary_factor, 0.25);
        assert_eq!(d.hidden, 64);
    }

    #[test]
    fn missing_numbers_are_errors_not_defaults() {
        for key in ["rope_theta", "rms_norm_eps", "max_position_embeddings"] {
            let mut c = base();
            c.as_object_mut().unwrap().remove(key);
            let e = dims(&c).unwrap_err().to_string();
            assert!(e.contains(key) || e.contains("rope_theta"), "{key}: {e}");
        }
    }

    #[test]
    fn conflicting_rope_theta_is_an_error() {
        let mut c = base();
        c["rope_parameters"] = json!({"rope_theta": 10000.0});
        assert!(dims(&c).unwrap_err().to_string().contains("conflicts"));
        c["rope_parameters"] = json!({"rope_theta": 1000000.0});
        dims(&c).unwrap();
    }

    #[test]
    fn unsupported_fields_are_rejected() {
        let cases = [
            ("hidden_act", json!("gelu")),
            ("attention_bias", json!(true)),
            ("use_sliding_window", json!(true)),
            ("rope_scaling", json!({"rope_type": "dynamic", "factor": 4.0})),
            ("rope_scaling", json!({"type": "linear", "factor": 2.0})),
            ("rope_parameters", json!({"rope_type": "longrope", "rope_theta": 5e5})),
            ("layer_types", json!(["full_attention", "sliding_attention"])),
        ];
        for (k, v) in cases {
            let mut c = base();
            c[k] = v;
            let e = dims(&c).unwrap_err();
            assert!(matches!(e, Error::Unsupported(_)), "{k}: {e}");
        }
        let mut c = base();
        c["rope_scaling"] = json!({"rope_type": "default"});
        dims(&c).unwrap();
    }

    #[test]
    fn llama3_and_yarn_scaling_are_read_from_either_block() {
        use crate::rope::RopeScaling;
        let llama3 = json!({"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192});
        let want = RopeScaling::Llama3 { factor: 8.0, low_freq_factor: 1.0, high_freq_factor: 4.0, original_max: 8192.0 };
        // legacy `rope_scaling` next to a top-level rope_theta
        let mut c = base();
        c["rope_scaling"] = llama3.clone();
        assert_eq!(dims(&c).unwrap().rope_scaling, want);
        // `rope_parameters` carrying rope_theta (transformers >= 5)
        let mut c = base();
        let mut rp = llama3.clone();
        rp["rope_theta"] = json!(1000000.0);
        c["rope_parameters"] = rp;
        assert_eq!(dims(&c).unwrap().rope_scaling, want);
        // yarn with only a factor: the pre-training length falls back to max_position_embeddings
        let mut c = base();
        c["rope_scaling"] = json!({"rope_type": "yarn", "factor": 4.0});
        match dims(&c).unwrap().rope_scaling {
            RopeScaling::Yarn { factor, original_max, .. } => assert_eq!((factor, original_max), (4.0, 4096.0)),
            other => panic!("{other:?}"),
        }
        // an unscaled model serialises without the field (plan / dims JSON of existing models is unchanged)
        let plain = serde_json::to_value(dims(&base()).unwrap()).unwrap();
        assert!(plain.get("rope_scaling").is_none(), "{plain}");
        let scaled = serde_json::to_value(dims(&c).unwrap()).unwrap();
        assert!(scaled.get("rope_scaling").is_some());
    }

    #[test]
    fn moe_checkpoint_has_no_dense_width() {
        let mut c = base();
        c.as_object_mut().unwrap().remove("intermediate_size");
        assert!(dims(&c).is_err());
        c["num_experts"] = json!(8);
        // MoE dims are read with the rest: a MoE config missing them is an error, not a dense model
        assert!(dims(&c).unwrap_err().to_string().contains("num_experts_per_tok"));
        c["num_experts_per_tok"] = json!(2);
        c["moe_intermediate_size"] = json!(32);
        let d = dims(&c).unwrap();
        assert_eq!(d.intermediate, 0);
        assert_eq!(d.moe.unwrap().top_k, 2);
    }
}
