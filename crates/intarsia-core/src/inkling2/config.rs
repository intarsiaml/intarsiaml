//! Inkling-2 text-model dimensions (`model_type = "inkling2_loop_text"`, the `text_config` of
//! `Inkling2LoopForConditionalGeneration`), docs/plans/2026-11-21-inkling2-port.md sections a and f.

use crate::error::{Error, Result};
use crate::recipe::text_config;

/// What an Inkling-2 config adds to `ModelDims` (which carries hidden, layers, heads, head dim, the REAL vocabulary
/// `unpadded_vocab_size` and `rms_norm_eps`).
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct InkDims {
    /// Rows of `embed` / `unembed` (`vocab_size`, padded); `ModelDims::vocab` is the real width.
    pub vocab_padded: usize,
    pub d_rel: usize,
    /// Distance-bias extent of the full-attention layers (`rel_extent`); the SWA layers' is the window.
    pub rel_extent: usize,
    /// Keys an SWA layer attends, itself included (`sliding_window_size`).
    pub window: usize,
    /// Per physical layer: sliding-window (`local_layer_ids`) or full attention.
    pub local: Vec<bool>,
    pub k_conv: usize,
    pub qk_clip: f32,
    pub rel_clip: f32,
    /// `log_scaling_n_floor` (0 = no log scaling), `log_scaling_alpha`.
    pub n_floor: usize,
    pub alpha: f32,
    /// Layers per attn-res block (`n_layers_per_attn_res_block`).
    pub block: usize,
    pub experts: usize,
    pub top_k: usize,
    pub inter: usize,
    pub gate_cap: f32,
    pub up_cap: f32,
    pub route_scale: f32,
    /// `final_logit_softcapping`, 0 = none.
    pub softcap: f32,
    /// The looped core `core_start..core_end`, `passes` times (no loop: `core_start == core_end`, 1 pass).
    pub core_start: usize,
    pub core_end: usize,
    pub passes: usize,
    pub gate_rank: usize,
}

impl InkDims {
    /// `Some` for an `inkling2_loop_text` config (or a wrapper whose `text_config` is one).
    pub fn from_hf_config(cfg: &serde_json::Value) -> Result<Option<InkDims>> {
        let tc = text_config(cfg);
        if tc.get("model_type").and_then(|v| v.as_str()) != Some("inkling2_loop_text") {
            return Ok(None);
        }
        let bad = |m: String| Err(Error::model(format!("inkling2 config: {m}")));
        let u = |k: &str| tc.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| Error::model(format!("inkling2 config: missing '{k}'")));
        let f = |k: &str| tc.get(k).and_then(|v| v.as_f64()).ok_or_else(|| Error::model(format!("inkling2 config: missing '{k}'")));
        let flag = |k: &str, want: bool| tc.get(k).and_then(|v| v.as_bool()).unwrap_or(want) == want;
        // what the port implements (the oracle asserts the same, tools/inkling2_ref.py `Cfg.load`)
        for (k, want) in [("use_value_norm_no_gamma", true), ("attn_res_normalize_values", true), ("use_gate_bias", true), ("q_bias", false), ("o_bias", false)] {
            if !flag(k, want) {
                return Err(Error::Unsupported(format!("inkling2 config: {k} = {} (only {want} is implemented)", !want)));
            }
        }
        if tc.get("n_shared_experts").and_then(|v| v.as_u64()).unwrap_or(0) != 0 {
            return Err(Error::Unsupported("inkling2 config: n_shared_experts != 0".into()));
        }
        let n_layers = u("num_hidden_layers")?;
        for (a, b) in [("swa_num_attention_heads", "num_attention_heads"), ("swa_num_key_value_heads", "num_key_value_heads"), ("swa_head_dim", "head_dim")] {
            if let Some(v) = tc.get(a).and_then(|v| v.as_u64()) {
                if v as usize != u(b)? {
                    return Err(Error::Unsupported(format!("inkling2 config: {a} = {v} differs from {b} (one head shape for both layer kinds is implemented)")));
                }
            }
        }
        let ids = tc.get("local_layer_ids").and_then(|v| v.as_array()).ok_or_else(|| Error::model("inkling2 config: missing 'local_layer_ids'"))?;
        let mut local = vec![false; n_layers];
        for v in ids {
            match v.as_u64().map(|x| x as usize) {
                Some(i) if i < n_layers => local[i] = true,
                _ => return bad(format!("local_layer_ids entry {v} outside 0..{n_layers}")),
            }
        }
        let (core_start, core_end, passes, gate_rank) = match tc.get("loop_config").filter(|v| v.is_object()) {
            Some(l) => {
                let lu = |k: &str| l.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| Error::model(format!("inkling2 config: loop_config.{k} missing")));
                if l.get("is_varloop").and_then(|v| v.as_bool()).unwrap_or(false) {
                    return Err(Error::Unsupported("inkling2 config: loop_config.is_varloop = true".into()));
                }
                (lu("core_start")?, lu("core_end")?, lu("num_passes")?, lu("gate_rank")?)
            }
            None => (0, 0, 1, 0),
        };
        let block = u("n_layers_per_attn_res_block")?;
        if block == 0 || core_start > core_end || core_end > n_layers || passes == 0 {
            return bad(format!("block {block}, core {core_start}..{core_end} of {n_layers}, {passes} passes"));
        }
        if core_end > core_start {
            // the messages mix exactly two bank entries (vLLM loop.py:228-236): the core must be two whole attn-res blocks
            if core_start % block != 0 || core_end - core_start != 2 * block || passes < 2 || gate_rank == 0 {
                return Err(Error::Unsupported(format!(
                    "inkling2 config: loop core {core_start}..{core_end} with block {block}, {passes} passes, gate_rank {gate_rank} (the port needs a core of exactly two attn-res blocks, block-aligned)"
                )));
            }
        }
        let vocab_padded = u("vocab_size")?;
        let vocab = tc.get("unpadded_vocab_size").and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(vocab_padded);
        if vocab == 0 || vocab > vocab_padded {
            return bad(format!("unpadded_vocab_size {vocab} > vocab_size {vocab_padded}"));
        }
        let d = InkDims {
            vocab_padded,
            d_rel: u("d_rel")?,
            rel_extent: u("rel_extent")?,
            window: u("sliding_window_size")?,
            local,
            k_conv: u("sconv_kernel_size")?,
            qk_clip: f("qk_l2_clip_threshold")? as f32,
            rel_clip: f("rel_l2_clip_threshold")? as f32,
            n_floor: tc.get("log_scaling_n_floor").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
            alpha: tc.get("log_scaling_alpha").and_then(|v| v.as_f64()).unwrap_or(0.1) as f32,
            block,
            experts: u("n_routed_experts")?,
            top_k: u("num_experts_per_tok")?,
            inter: u("intermediate_size")?,
            gate_cap: f("mlp_gate_softcap")? as f32,
            up_cap: f("mlp_up_softcap")? as f32,
            route_scale: f("route_scale")? as f32,
            softcap: tc.get("final_logit_softcapping").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            core_start,
            core_end,
            passes,
            gate_rank,
        };
        if d.k_conv < 2 || d.window == 0 || d.d_rel == 0 || d.top_k == 0 || d.top_k > d.experts {
            return bad(format!("k_conv {}, window {}, d_rel {}, top-{} of {}", d.k_conv, d.window, d.d_rel, d.top_k, d.experts));
        }
        Ok(Some(d))
    }

    /// `(layer, pass)` of every visit in execution order: prelude, each pass of the core, coda (spec a12).
    pub fn visits(&self) -> Vec<(usize, usize)> {
        let n = self.local.len();
        if self.core_end == self.core_start {
            return (0..n).map(|i| (i, 0)).collect();
        }
        let mut v: Vec<(usize, usize)> = (0..self.core_start).map(|i| (i, 0)).collect();
        for p in 0..self.passes {
            v.extend((self.core_start..self.core_end).map(|i| (i, p)));
        }
        v.extend((self.core_end..n).map(|i| (i, 0)));
        v
    }

    pub fn looped(&self) -> bool {
        self.core_end > self.core_start
    }
}
