//! Gemma 4 decoder layer (HF `Gemma4TextDecoderLayer` of transformers 5.6.2, `enable_moe_block`), one block per layer.
//!
//! ```text
//! n   = rmsnorm(h, input_layernorm)
//! q   = rope(rmsnorm_head(q_proj n, q_norm))                 sliding: default rope, theta 1e4, whole head
//!                                                            global : "proportional" rope (Op::Rope rope_prop), theta 1e6
//! k   = rope(rmsnorm_head(k_proj n, k_norm))
//! v   = rmsnorm_head(v_proj n, no scale)                     global layers (attention_k_eq_v): v_proj is k_proj's output
//! a   = attention(q, k, v), scale 1.0, sliding layers: window 1024
//! h1  = h + rmsnorm(o_proj a, post_attention_layernorm)
//! d1  = rmsnorm(down(gelu_tanh(gate n2) * up n2), post_feedforward_layernorm_1),   n2 = rmsnorm(h1, pre_feedforward_layernorm)
//! d2  = rmsnorm(experts(rmsnorm(h1, pre_feedforward_layernorm_2)), post_feedforward_layernorm_2)
//!       routing: softmax(router.proj(rmsnorm(h1) * router.scale * hidden^-0.5)) -> top-k -> renormalise -> x per_expert_scale
//! h'  = (h1 + rmsnorm(d1 + d2, post_feedforward_layernorm)) * layer_scalar
//! ```
//!
//! Folded at load time (no op): `router.scale * hidden^-0.5` into the router norm's weight, `per_expert_scale` into each expert's
//! down-projection global scale ([`crate::quant::moe::MoeWeights::load_gemma`]). The embedding scale and `layer_scalar` are the
//! `scale` attr of `Embedding` / the last `Add`. The final logit soft-cap is applied to the downloaded logits on the host
//! (it does not change the argmax).

use super::*;
use crate::recipe::GemmaDims;

pub(super) struct GemmaLayer {
    pub nh: usize,
    pub nkv: usize,
    pub hd: usize,
    /// Rotated values per head for `rope_prop` (0 = sliding layer: default rope over the whole head).
    pub rotary_dim: usize,
    pub theta: f32,
    /// Sliding window (keys incl. the current one), 0 = global attention.
    pub window: usize,
    pub slot: usize,
    pub input_norm: TensorRef,
    pub post_attn_norm: TensorRef,
    pub pre_ffn_norm: TensorRef,
    pub post_ffn_norm: TensorRef,
    pub pre_ffn2_norm: TensorRef,
    pub post_ffn1_norm: TensorRef,
    pub post_ffn2_norm: TensorRef,
    pub q_norm: TensorRef,
    pub k_norm: TensorRef,
    /// `[hd]` ones: the scale-less `v_norm`.
    pub ones: TensorRef,
    pub wq: QTensor,
    pub wk: QTensor,
    pub wv: Option<QTensor>,
    pub wo: QTensor,
    pub gate: QTensor,
    pub up: QTensor,
    pub down: QTensor,
    /// `router.scale * hidden^-0.5` as the weight of the router's input norm.
    pub router_norm: TensorRef,
    pub moe: MoeWeights,
    pub layer_scalar: f32,
}

/// Scratch of the Gemma layers (one set, reused by every layer).
pub(super) struct GemmaWs {
    pub max_tokens: usize,
    /// Six `[T, hidden]` buffers (the layer's liveness map is in [`Model::gemma_layer`]).
    pub h: [Buffer; 6],
    pub q: [Buffer; 2],
    pub k: [Buffer; 2],
    pub v: [Buffer; 2],
    pub g: Buffer,
    pub u: Buffer,
    pub a: Buffer,
    pub logits: Buffer,
    pub topk_w: Buffer,
    pub topk_id: Buffer,
}

impl GemmaWs {
    pub(super) fn alloc(wb: &mut dyn FnMut(usize) -> Result<Buffer>, mt: usize, esz: usize, dims: &ModelDims, g: &GemmaDims) -> Result<GemmaWs> {
        let hn = mt * dims.hidden * esz;
        let qmax = dims.n_heads * dims.head_dim.max(g.global_head_dim);
        let kvmax = (dims.n_kv_heads * dims.head_dim).max(g.global_kv_heads * g.global_head_dim);
        let mc = dims.moe.as_ref().ok_or_else(|| Error::model("gemma4 without MoE dimensions"))?;
        Ok(GemmaWs {
            max_tokens: mt,
            h: [wb(hn)?, wb(hn)?, wb(hn)?, wb(hn)?, wb(hn)?, wb(hn)?],
            q: [wb(mt * qmax * esz)?, wb(mt * qmax * esz)?],
            k: [wb(mt * kvmax * esz)?, wb(mt * kvmax * esz)?],
            v: [wb(mt * kvmax * esz)?, wb(mt * kvmax * esz)?],
            g: wb(mt * dims.intermediate * esz)?,
            u: wb(mt * dims.intermediate * esz)?,
            a: wb(mt * dims.intermediate * esz)?,
            logits: wb(mt * mc.experts * esz)?,
            topk_w: wb(mt * mc.top_k * 4)?,
            topk_id: wb(mt * mc.top_k * 4)?,
        })
    }
}

/// `values` in the activation dtype's little-endian bytes.
pub(super) fn encode_floats(dtype: DType, values: &[f32]) -> Result<Vec<u8>> {
    Ok(match dtype {
        DType::Bf16 => values.iter().flat_map(|&v| half::bf16::from_f32(v).to_le_bytes()).collect(),
        DType::F16 => values.iter().flat_map(|&v| half::f16::from_f32(v).to_le_bytes()).collect(),
        DType::F32 => values.iter().flat_map(|&v| v.to_le_bytes()).collect(),
        d => return Err(Error::Unsupported(format!("gemma4 activation dtype {d}"))),
    })
}

impl Model {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gemma_layer(
        &self,
        g: &GemmaLayer,
        st: &SequenceState,
        h_in: &TensorRef,
        h_out: &TensorRef,
        t: usize,
        pos0: usize,
        profile: bool,
    ) -> Result<()> {
        let ws = self.gemma.as_ref().ok_or_else(|| Error::invalid("gemma layer without its workspace"))?;
        let d = &self.dims;
        let mc = d.moe.as_ref().ok_or_else(|| Error::invalid("gemma layer without MoE dims"))?;
        let (act, hdn) = (self.act, d.hidden);
        let (nh, nkv, hd) = (g.nh, g.nkv, g.hd);
        let eps = OpAttrs { eps: d.rms_eps, w_offset: self.recipe.model.norm_offset, ..Default::default() };
        let hv = |i: usize| self.view(&ws.h[i], act, &[t, hdn]);
        // liveness: B0 n/rn/n3/s2, B1 o/d/routed/s, B2 o2/d1, B3 h1, B5 d2
        let (b0, b1, b2, b3, b5) = (hv(0), hv(1), hv(2), hv(3), hv(5));
        let norm = |x: &TensorRef, w: &TensorRef, out: &TensorRef| self.exec(Op::RmsNorm, vec![x.clone(), w.clone()], vec![out.clone()], eps.clone(), profile);

        // ---- attention ----
        let slot = match &st.slots[g.slot] {
            StateSlot::AttnKv(s) => s,
            StateSlot::Recurrent(_) => return Err(Error::invalid("gemma attention bound to a recurrent state slot")),
        };
        norm(h_in, &g.input_norm, &b0)?;
        let q3 = |i: usize| self.view(&ws.q[i], act, &[t, nh, hd]);
        let k3 = |b: &Buffer| self.view(b, act, &[t, nkv, hd]);
        let (q0, q1) = (q3(0), q3(1));
        let (k0, k1) = (k3(&ws.k[0]), k3(&ws.k[1]));
        let (v0, v1) = (k3(&ws.v[0]), k3(&ws.v[1]));
        self.proj(&b0, &g.wq, &q0.reshaped(&[t, nh * hd]), profile)?;
        self.proj(&b0, &g.wk, &k0.reshaped(&[t, nkv * hd]), profile)?;
        let vsrc = match &g.wv {
            Some(wv) => {
                self.proj(&b0, wv, &v0.reshaped(&[t, nkv * hd]), profile)?;
                &v0
            }
            None => &k0, // attention_k_eq_v: V is the K projection (before k_norm and rope)
        };
        // v_norm: RMSNorm without scale = QkNorm with a ones weight
        self.exec(Op::QkNorm, vec![vsrc.clone(), g.ones.clone()], vec![v1.clone()], eps.clone(), profile)?;
        let rope = OpAttrs {
            pos0,
            theta: g.theta,
            head_dim: hd,
            rotary_dim: g.rotary_dim,
            rope_prop: g.rotary_dim != 0,
            ..Default::default()
        };
        self.exec(Op::QkNorm, vec![q0.clone(), g.q_norm.clone()], vec![q1.clone()], eps.clone(), profile)?;
        self.exec(Op::Rope, vec![q1.clone()], vec![q0.clone()], rope.clone(), profile)?;
        self.exec(Op::QkNorm, vec![k0.clone(), g.k_norm.clone()], vec![k1.clone()], eps.clone(), profile)?;
        self.exec(Op::Rope, vec![k1.clone()], vec![k0.clone()], rope, profile)?;
        let cache_shape = [nkv, slot.cap, hd];
        let kcache = self.view(&slot.k, slot.dtype, &cache_shape);
        let vcache = self.view(&slot.v, slot.dtype, &cache_shape);
        let append = OpAttrs { pos0, kv_cap: slot.cap, n_kv_heads: nkv, head_dim: hd, ..Default::default() };
        self.exec(Op::KvAppend, vec![k0.clone()], vec![kcache.clone()], append.clone(), profile)?;
        self.exec(Op::KvAppend, vec![v1.clone()], vec![vcache.clone()], append, profile)?;
        let attn = OpAttrs {
            pos0,
            scale: 1.0, // Gemma 4: no 1/sqrt(head_dim)
            n_heads: nh,
            n_kv_heads: nkv,
            head_dim: hd,
            kv_cap: slot.cap,
            window: g.window,
            ..Default::default()
        };
        self.exec(if t == 1 { Op::AttnDecode } else { Op::AttnPrefill }, vec![q0, kcache, vcache], vec![q1.clone()], attn, profile)?;
        self.proj(&q1.reshaped(&[t, nh * hd]), &g.wo, &b1, profile)?;
        norm(&b1, &g.post_attn_norm, &b2)?;
        self.exec(Op::Add, vec![h_in.clone(), b2.clone()], vec![b3.clone()], OpAttrs::default(), profile)?; // h1 in B3

        // ---- dense GeGLU MLP ----
        let inter = |b: &Buffer| self.view(b, act, &[t, d.intermediate]);
        let (gg, uu, aa) = (inter(&ws.g), inter(&ws.u), inter(&ws.a));
        norm(&b3, &g.pre_ffn_norm, &b0)?;
        self.proj(&b0, &g.gate, &gg, profile)?;
        self.proj(&b0, &g.up, &uu, profile)?;
        self.exec(Op::GeluGlu, vec![gg, uu], vec![aa.clone()], OpAttrs::default(), profile)?;
        self.proj(&aa, &g.down, &b1, profile)?;
        norm(&b1, &g.post_ffn1_norm, &b2)?; // d1 in B2

        // ---- routed experts ----
        norm(&b3, &g.router_norm, &b0)?;
        let logits = self.view(&ws.logits, act, &[t, mc.experts]);
        self.proj(&b0, &g.moe.router, &logits, profile)?;
        let tw = self.view(&ws.topk_w, DType::F32, &[t, mc.top_k]);
        let ti = self.view(&ws.topk_id, DType::I32, &[t, mc.top_k]);
        let topk = OpAttrs { top_k: mc.top_k, norm_topk: true, ..Default::default() };
        self.exec(Op::MoeTopk, vec![logits], vec![tw.clone(), ti.clone()], topk.clone(), profile)?;
        norm(&b3, &g.pre_ffn2_norm, &b0)?;
        let e = &g.moe.experts;
        self.exec(
            Op::MoeExperts,
            vec![b0.clone(), ti, tw, e.gate_up_w.clone(), e.gate_up_s.clone(), e.down_w.clone(), e.down_s.clone(), e.gscales.clone()],
            vec![b1.clone()],
            OpAttrs { gelu: true, ..topk },
            profile,
        )?;
        norm(&b1, &g.post_ffn2_norm, &b5)?; // d2 in B5

        // ---- combine ----
        self.exec(Op::Add, vec![b2.clone(), b5.clone()], vec![b1.clone()], OpAttrs::default(), profile)?;
        norm(&b1, &g.post_ffn_norm, &b0)?;
        // layer_scalar multiplies the whole hidden state, residual included
        self.exec(Op::Add, vec![b3, b0], vec![h_out.clone()], OpAttrs { scale: g.layer_scalar, ..Default::default() }, profile)?;
        Ok(())
    }
}
