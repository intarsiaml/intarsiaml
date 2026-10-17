//! The GatedDeltaNet (linear attention) block of the generic decoder, plus the activation buffers
//! the gated-attention and GatedDeltaNet blocks need. Semantics: `crate::gdn` (HF `Qwen3_5MoeGatedDeltaNet`).
//!
//! Dataflow of one block (`n` = pre-normed input `[T, H]`, all activations in the model's act dtype):
//! ```text
//!   qkv = n @ in_qkv^T            [T, C]        C = 2*Hk*dk + Hv*dv
//!   z   = n @ in_z^T              [T, Hv*dv]
//!   b   = n @ in_b^T, a = n @ in_a^T   [T, Hv]
//!   conv = GdnConv(qkv, taps)     [T, C]        silu(causal depthwise conv), conv state in the slot
//!   core = GdnDeltaRule(conv, a, b, A_log, dt_bias)   [T, Hv, dv]   recurrent state S in the slot
//!   core = silu(z) * rmsnorm(core, w)           (RmsNorm + SiluGlu over rows of dv)
//!   out  = core @ out_proj^T      [T, H]
//! ```

use super::*;
use crate::quant::{BF16, FP8_E4M3};

/// Bytes the recurrent slot of one GatedDeltaNet layer holds: conv state `[C, K-1]` f32, then (256-byte
/// aligned) the delta-rule state `[Hv, dv, dk]` f32.
pub(super) fn state_bytes(l: &LinearDims) -> usize {
    state_s_offset(l) + l.v_heads * l.v_head_dim * l.k_head_dim * 4
}

fn state_s_offset(l: &LinearDims) -> usize {
    (l.conv_dim() * (l.conv_kernel - 1) * 4).div_ceil(256) * 256
}

pub(super) struct GdnBlock {
    pub pre_norm: TensorRef,
    pub in_qkv: QTensor,
    pub in_z: QTensor,
    pub in_b: QTensor,
    pub in_a: QTensor,
    /// Conv taps `[C, K]`.
    pub conv: TensorRef,
    pub a_log: TensorRef,
    pub dt_bias: TensorRef,
    /// Gated RMSNorm weight `[dv]`.
    pub norm: TensorRef,
    pub out: QTensor,
    pub slot: usize,
    pub lin: LinearDims,
}

struct GdnWs {
    qkv: Buffer,
    z: Buffer,
    a: Buffer,
    b: Buffer,
    conv: Buffer,
    core: Buffer,
    normed: Buffer,
}

/// Buffers that only some models need (allocated only when the recipe uses the block).
pub(super) struct ExtraWs {
    /// Gated attention: the fused q+gate projection `[T, Nh*2*hd]` and the split-off gate `[T, Nh*hd]`.
    qg: Option<Buffer>,
    gate: Option<Buffer>,
    gdn: Option<GdnWs>,
}

pub(super) fn alloc_extra(
    wb: &mut dyn FnMut(usize) -> Result<Buffer>,
    mt: usize,
    esz: usize,
    dims: &ModelDims,
    has_gdn: bool,
    has_gated_attn: bool,
) -> Result<ExtraWs> {
    let qdim = dims.n_heads * dims.head_dim;
    let (qg, gate) = if has_gated_attn {
        (Some(wb(mt * 2 * qdim * esz)?), Some(wb(mt * qdim * esz)?))
    } else {
        (None, None)
    };
    let gdn = match (has_gdn, dims.linear) {
        (true, Some(l)) => {
            let vdim = l.v_heads * l.v_head_dim;
            Some(GdnWs {
                qkv: wb(mt * l.conv_dim() * esz)?,
                z: wb(mt * vdim * esz)?,
                a: wb(mt * l.v_heads * esz)?,
                b: wb(mt * l.v_heads * esz)?,
                conv: wb(mt * l.conv_dim() * esz)?,
                core: wb(mt * vdim * esz)?,
                normed: wb(mt * vdim * esz)?,
            })
        }
        _ => None,
    };
    Ok(ExtraWs { qg, gate, gdn })
}

impl Model {
    /// `(q+gate projection [T, Nh*2*hd], gate [T, Nh, hd])` workspace views of a gated-attention block.
    pub(super) fn gated_q_buffers(&self, t: usize) -> (TensorRef, TensorRef) {
        let d = &self.dims;
        let x = &self.ws.x;
        let qg = x.qg.as_ref().expect("gated attention workspace is allocated at load");
        let gate = x.gate.as_ref().expect("gated attention workspace is allocated at load");
        (
            self.view(qg, self.act, &[t, d.n_heads * 2 * d.head_dim]),
            self.view(gate, self.act, &[t, d.n_heads, d.head_dim]),
        )
    }

    pub(super) fn gdn_block(
        &self,
        g: &GdnBlock,
        st: &SequenceState,
        n: &TensorRef,
        out: &TensorRef,
        t: usize,
        profile: bool,
        fold: Option<&NormFold>,
    ) -> Result<()> {
        let x = self.ws.x.gdn.as_ref().ok_or_else(|| Error::invalid("GatedDeltaNet workspace missing"))?;
        let l = &g.lin;
        let act = self.act;
        let (c, hv, dv, dk, ks) = (l.conv_dim(), l.v_heads, l.v_head_dim, l.k_head_dim, l.conv_kernel);
        let vdim = hv * dv;
        let rec = match &st.slots[g.slot] {
            StateSlot::Recurrent(r) => r,
            StateSlot::AttnKv(_) => return Err(Error::invalid("GatedDeltaNet block bound to an attention state slot")),
        };
        let conv_state = TensorRef::new(rec.buf.clone(), DType::F32, &[c, ks - 1]);
        let mut s_state = TensorRef::new(rec.buf.clone(), DType::F32, &[hv, dv, dk]);
        s_state.offset = state_s_offset(l);

        let qkv = self.view(&x.qkv, act, &[t, c]);
        let z = self.view(&x.z, act, &[t, vdim]);
        let a = self.view(&x.a, act, &[t, hv]);
        let b = self.view(&x.b, act, &[t, hv]);
        let conv = self.view(&x.conv, act, &[t, c]);
        let core = self.view(&x.core, act, &[t, hv, dv]);
        let normed = self.view(&x.normed, act, &[t, hv, dv]);

        // `n` is not computed yet when `fold` is set (the residual add + norm in front of the block). The b and a gates
        // (dense, tiny) take the norm into their launch when the backend has the fused kernel, so they run first (they do
        // not depend on the other projections); otherwise the norm runs here unfused, before anything reads `n`.
        let mut ba_fused = false;
        if let Some(f) = fold {
            if let (BF16, BF16) = (g.in_b.format, g.in_a.format) {
                let (inputs, outputs) = f.pair(n, &g.in_b.parts[0], &g.in_a.parts[0], &b, &a);
                ba_fused = self.try_fused(Op::RmsNormAddMatmulPair, inputs, outputs, f.attrs.clone(), profile)?;
            }
            if !ba_fused {
                let (inputs, outputs) = f.unfused(n);
                self.exec(Op::RmsNormAdd, inputs, outputs, f.attrs.clone(), profile)?;
            }
        }
        // The four projections read the same activation. qkv and z (fp8) go through one launch, and so do the b and a
        // gates (dense), when the backend has the fused pair kernels; otherwise each runs on its own.
        let qz_fused = match (g.in_qkv.format, g.in_z.format) {
            (FP8_E4M3, FP8_E4M3) => self.try_fused(
                Op::MatmulFp8Pair,
                vec![n.clone(), g.in_qkv.parts[0].clone(), g.in_qkv.parts[1].clone(), g.in_z.parts[0].clone(), g.in_z.parts[1].clone()],
                vec![qkv.clone(), z.clone()],
                OpAttrs::default(),
                profile,
            )?,
            _ => false,
        };
        if !qz_fused {
            self.proj(n, &g.in_qkv, &qkv, profile)?;
            self.proj(n, &g.in_z, &z, profile)?;
        }
        let ba_fused = ba_fused
            || match (g.in_b.format, g.in_a.format) {
                (BF16, BF16) => self.try_fused(
                    Op::MatmulPair,
                    vec![n.clone(), g.in_b.parts[0].clone(), g.in_a.parts[0].clone()],
                    vec![b.clone(), a.clone()],
                    OpAttrs::default(),
                    profile,
                )?,
                _ => false,
            };
        if !ba_fused {
            self.proj(n, &g.in_b, &b, profile)?;
            self.proj(n, &g.in_a, &a, profile)?;
        }
        self.exec(
            Op::GdnConv,
            vec![qkv, g.conv.clone()],
            vec![conv.clone(), conv_state],
            OpAttrs::default(),
            profile,
        )?;
        self.exec(
            Op::GdnDeltaRule,
            vec![conv, a, b, g.a_log.clone(), g.dt_bias.clone()],
            vec![core.clone(), s_state],
            // HF l2norm epsilon (a constant of the architecture, not rms_norm_eps)
            OpAttrs { eps: 1e-6, ..Default::default() },
            profile,
        )?;
        // gated RMSNorm: rmsnorm(core, w) * silu(z), over rows of dv (the weight is never zero-centred)
        let rows = [t * hv, dv];
        let norm_attrs = OpAttrs { eps: self.dims.rms_eps, ..Default::default() };
        // the norm folded into the fp8 out_proj when the backend has that kernel: the block's output in one launch
        if g.out.format == FP8_E4M3 {
            let folded = self.try_fused(
                Op::GatedRmsnormMatmulFp8,
                vec![core.reshaped(&rows), z.reshaped(&rows), g.norm.clone(), g.out.parts[0].clone(), g.out.parts[1].clone()],
                vec![out.clone()],
                norm_attrs.clone(),
                profile,
            )?;
            if folded {
                return Ok(());
            }
        }
        // one launch for both when the backend has it (the result then lands in `normed`, not `core`)
        let gated = self.try_fused(
            Op::GatedRmsnorm,
            vec![core.reshaped(&rows), z.reshaped(&rows), g.norm.clone()],
            vec![normed.reshaped(&rows)],
            norm_attrs.clone(),
            profile,
        )?;
        if gated {
            return self.proj(&normed.reshaped(&[t, vdim]), &g.out, out, profile);
        }
        self.exec(
            Op::RmsNorm,
            vec![core.reshaped(&rows), g.norm.clone()],
            vec![normed.reshaped(&rows)],
            norm_attrs,
            profile,
        )?;
        self.exec(
            Op::SiluGlu,
            vec![z.reshaped(&rows), normed.reshaped(&rows)],
            vec![core.reshaped(&rows)],
            OpAttrs::default(),
            profile,
        )?;
        self.proj(&core.reshaped(&[t, vdim]), &g.out, out, profile)
    }
}
