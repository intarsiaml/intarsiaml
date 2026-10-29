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
use crate::quant::{BF16, FP8_E4M3, MXFP4};

/// Bytes the recurrent slot of one GatedDeltaNet layer holds: conv state `[C, K-1]` in `conv_dt` (`Backend::gdn_conv_dtype`:
/// f32, bf16 on CUDA), then (256-byte aligned) the delta-rule state `[Hv, dv, dk]` f32.
pub(super) fn state_bytes(l: &LinearDims, conv_dt: DType) -> usize {
    state_s_offset(l, conv_dt) + l.v_heads * l.v_head_dim * l.k_head_dim * 4
}

pub(super) fn state_s_offset(l: &LinearDims, conv_dt: DType) -> usize {
    (l.conv_dim() * (l.conv_kernel - 1) * conv_dt.size_bytes()).div_ceil(256) * 256
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

pub(super) struct GdnWs {
    pub(super) qkv: Buffer,
    pub(super) z: Buffer,
    pub(super) a: Buffer,
    pub(super) b: Buffer,
    pub(super) conv: Buffer,
    pub(super) core: Buffer,
    pub(super) normed: Buffer,
}

/// Buffers that only some models need (allocated only when the recipe uses the block).
pub(super) struct ExtraWs {
    /// Gated attention: the fused q+gate projection `[T, Nh*2*hd]` and the split-off gate `[T, Nh*hd]`.
    qg: Option<Buffer>,
    gate: Option<Buffer>,
    pub(super) gdn: Option<GdnWs>,
    /// LFM2's short-conv block (`model/short_conv.rs`).
    pub(super) short: Option<super::short_conv::ShortConvWs>,
}

pub(super) fn alloc_extra(
    wb: &mut dyn FnMut(usize) -> Result<Buffer>,
    mt: usize,
    esz: usize,
    dims: &ModelDims,
    has_gdn: bool,
    has_gated_attn: bool,
    has_conv: bool,
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
    let short = if has_conv { Some(super::short_conv::alloc_ws(wb, mt, esz, dims.hidden)?) } else { None };
    Ok(ExtraWs { qg, gate, gdn, short })
}

impl Model {
    /// `(q+gate projection [T, Nh*2*hd], gate [T, Nh, hd])` workspace views of a gated-attention block.
    pub(super) fn gated_q_buffers(&self, t: usize) -> (TensorRef, TensorRef) {
        // this rank's heads (equal to the model's without tensor parallelism)
        let d = &self.ldims;
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
        self.gdn_block_segs(g, &[Seg { st, lo: 0, len: t, pos: st.len }], n, out, t, profile, fold)
    }

    /// [`Model::gdn_block`] over the rows of one or several segments (a packed prefill pass): the projections, the gated norm and
    /// the output projection run once on all `t` rows, the conv + delta rule once per segment on that segment's rows and its own
    /// recurrent state. One segment is the solo block, call for call.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gdn_block_segs(
        &self,
        g: &GdnBlock,
        segs: &[Seg<'_>],
        n: &TensorRef,
        out: &TensorRef,
        t: usize,
        profile: bool,
        fold: Option<&NormFold>,
    ) -> Result<()> {
        let st = segs[0].st;
        let x = self.ws.x.gdn.as_ref().ok_or_else(|| Error::invalid("GatedDeltaNet workspace missing"))?;
        let l = &g.lin;
        let act = self.act;
        let (c, hv, dv, dk, ks) = (l.conv_dim(), l.v_heads, l.v_head_dim, l.k_head_dim, l.conv_kernel);
        let vdim = hv * dv;
        let conv_dt = self.backend.gdn_conv_dtype();
        let rec = match &st.slots[g.slot] {
            StateSlot::Recurrent(r) => r,
            StateSlot::AttnKv(_) => return Err(Error::invalid("GatedDeltaNet block bound to an attention state slot")),
        };
        let conv_state = TensorRef::new(rec.buf.clone(), conv_dt, &[c, ks - 1]);
        let mut s_state = TensorRef::new(rec.buf.clone(), DType::F32, &[hv, dv, dk]);
        s_state.offset = state_s_offset(l, conv_dt);

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
        // The four projections read the same activation. qkv and z (fp8 / mxfp4) go through one launch, and so do the b and a
        // gates (dense / mxfp4), when the backend has the fused pair kernels; otherwise each runs on its own. Lane moe16b: z, b
        // and a run on the side stream beside qkv (same ops in the same order; forked here, joined before the conv reads them).
        let fork = self.side_fork(true, t, profile)?;
        let qz_fused = match (g.in_qkv.format, g.in_z.format) {
            (FP8_E4M3, FP8_E4M3) => self.try_fused(
                Op::MatmulFp8Pair,
                vec![n.clone(), g.in_qkv.parts[0].clone(), g.in_qkv.parts[1].clone(), g.in_z.parts[0].clone(), g.in_z.parts[1].clone()],
                vec![qkv.clone(), z.clone()],
                OpAttrs { fp8_static_act: g.in_qkv.act_quant && g.in_z.act_quant, ..OpAttrs::default() },
                profile,
            )?,
            (MXFP4, MXFP4) => self.try_mxfp4_pair(n, &g.in_qkv, &g.in_z, &qkv, &z, profile)?,
            _ => false,
        };
        if !qz_fused {
            self.proj(n, &g.in_qkv, &qkv, profile)?;
        }
        let mut rest = || -> Result<()> {
            if !qz_fused {
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
                    (MXFP4, MXFP4) => self.try_mxfp4_pair(n, &g.in_b, &g.in_a, &b, &a, profile)?,
                    _ => false,
                };
            if !ba_fused {
                self.proj(n, &g.in_b, &b, profile)?;
                self.proj(n, &g.in_a, &a, profile)?;
            }
            Ok(())
        };
        match fork {
            Some(tok) => self.side_run(tok, &mut rest)?,
            None => rest()?,
        }
        // HF l2norm epsilon (a constant of the architecture, not rms_norm_eps)
        let delta_attrs = OpAttrs { eps: 1e-6, ..Default::default() };
        if segs.len() > 1 {
            for sg in segs {
                let rec = match &sg.st.slots[g.slot] {
                    StateSlot::Recurrent(r) => r,
                    StateSlot::AttnKv(_) => return Err(Error::invalid("GatedDeltaNet block bound to an attention state slot")),
                };
                let conv_state = TensorRef::new(rec.buf.clone(), conv_dt, &[c, ks - 1]);
                let mut s_state = TensorRef::new(rec.buf.clone(), DType::F32, &[hv, dv, dk]);
                s_state.offset = state_s_offset(l, conv_dt);
                let (lo, len) = (sg.lo, sg.len);
                let (qkv, a, b, conv, core) = (qkv.rows(lo, len), a.rows(lo, len), b.rows(lo, len), conv.rows(lo, len), core.rows(lo, len));
                self.conv_delta(g, &qkv, &a, &b, &conv, &core, &s_state, &conv_state, &delta_attrs, profile)?;
            }
        } else if self.verifying() && t > 1 {
            // speculative verify: one row at a time (each call is the decode step's), and the slot's state after every row
            // but the last saved for the rollback (`SequenceState::spec_rollback`)
            let save = st.spec_rows.bufs.len() >= t - 1 && self.recorder.lock().unwrap().is_none();
            let off = st.recurrent_offset(g.slot);
            // HIP (lane vrc): the T rows in one launch that also stores the state after every row but the last into the save
            // buffers: bitwise the row-by-row calls and copies below, without their launches (INTARSIA_HIP_GDN_VERIFY_FUSE=0: off)
            let mut done = false;
            if save && self.gdn_verify_fuse_ok(t, off, rec.bytes, state_s_offset(l, conv_dt)) {
                let snaps: Vec<TensorRef> = st.spec_rows.bufs[..t - 1]
                    .iter()
                    .map(|b| {
                        let mut v = TensorRef::new(b.clone(), conv_dt, &[c, ks - 1]);
                        v.offset = off;
                        v
                    })
                    .collect();
                let mut outs = vec![core.clone(), s_state.clone(), conv_state.clone()];
                outs.extend(snaps);
                done = self.try_fused(
                    Op::GdnConvDelta,
                    vec![qkv.clone(), g.conv.clone(), a.clone(), b.clone(), g.a_log.clone(), g.dt_bias.clone()],
                    outs,
                    delta_attrs.clone(),
                    profile,
                )?;
            }
            for r in 0..if done { 0 } else { t } {
                let (qkv, a, b, conv, core) = (qkv.rows(r, 1), a.rows(r, 1), b.rows(r, 1), conv.rows(r, 1), core.rows(r, 1));
                self.conv_delta(g, &qkv, &a, &b, &conv, &core, &s_state, &conv_state, &delta_attrs, profile)?;
                if save && r + 1 < t {
                    self.backend.copy(&st.spec_rows.bufs[r], off, &rec.buf, 0, rec.bytes, 0)?;
                }
            }
        } else {
            self.conv_delta(g, &qkv, &a, &b, &conv, &core, &s_state, &conv_state, &delta_attrs, profile)?;
        }
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
            return self.proj_reduce(&normed.reshaped(&[t, vdim]), &g.out, out, t, profile);
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
        // row-parallel under tensor parallelism (out_proj holds this rank's value heads' columns): the partial sum, then
        // the all-reduce; without it, exactly the projection
        self.proj_reduce(&core.reshaped(&[t, vdim]), &g.out, out, t, profile)
    }

    /// The conv and the delta rule of a block's `qkv` rows: folded into one launch (`GdnConvDelta`: the conv output is never
    /// stored) when the backend has it, otherwise the two ops, in that order.
    #[allow(clippy::too_many_arguments)]
    fn conv_delta(
        &self,
        g: &GdnBlock,
        qkv: &TensorRef,
        a: &TensorRef,
        b: &TensorRef,
        conv: &TensorRef,
        core: &TensorRef,
        s_state: &TensorRef,
        conv_state: &TensorRef,
        delta_attrs: &OpAttrs,
        profile: bool,
    ) -> Result<()> {
        let folded = self.try_fused(
            Op::GdnConvDelta,
            vec![qkv.clone(), g.conv.clone(), a.clone(), b.clone(), g.a_log.clone(), g.dt_bias.clone()],
            vec![core.clone(), s_state.clone(), conv_state.clone()],
            delta_attrs.clone(),
            profile,
        )?;
        if !folded {
            self.exec(Op::GdnConv, vec![qkv.clone(), g.conv.clone()], vec![conv.clone(), conv_state.clone()], OpAttrs::default(), profile)?;
            self.exec(
                Op::GdnDeltaRule,
                vec![conv.clone(), a.clone(), b.clone(), g.a_log.clone(), g.dt_bias.clone()],
                vec![core.clone(), s_state.clone()],
                delta_attrs.clone(),
                profile,
            )?;
        }
        Ok(())
    }

    /// Whether a speculative-verify GatedDeltaNet block of `t` rows may take the one-launch snapshotting `GdnConvDelta` (HIP only):
    /// `INTARSIA_HIP_GDN_VERIFY_FUSE` not `0`, `t - 1` snapshots within the kernel's table, and every address the kernel stores to
    /// 16-byte aligned (the save buffers' slot offset, the slot size, the delta state's offset in the slot).
    fn gdn_verify_fuse_ok(&self, t: usize, off: usize, slot_bytes: usize, s_off: usize) -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("INTARSIA_HIP_GDN_VERIFY_FUSE").map_or(true, |v| v != "0"))
            && self.backend.name() == "hip"
            && (2..=16).contains(&t)
            && off % 16 == 0
            && slot_bytes % 16 == 0
            && s_off % 16 == 0
    }
}
