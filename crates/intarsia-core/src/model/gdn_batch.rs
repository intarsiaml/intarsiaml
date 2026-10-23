//! Batched decode of a GatedDeltaNet block (hybrid opt-in, `Model::allow_recurrent_batch`; cpu prototype,
//! docs/results/2026-10-23-rb-recurrent-batching.md sections 4 and 12).
//!
//! One decode token for each of `B` sequences, each with its own recurrent state. The structure follows the attention
//! batch step (`attn_block_batch`): the four input projections and the output projection run once on all `B` rows
//! (weights read once); the two ops that touch per-sequence state, the causal conv and the delta rule, run **once per
//! row, as the one-row (T = 1) calls of the solo decode step, on that row's own state slot**, reading and writing the
//! state in place. A batched row is therefore the solo row by construction: nothing is gathered, copied or shared.
//!
//! The block body of `gdn_block` is deliberately not shared: `gdn_block_batch` repeats its projection and gated-norm
//! sequence (same fold and fusion decisions, `B` rows instead of `t`), so the solo path stays textually untouched.
//! Once PR #110 (`conv_delta` helper) lands the per-row loop below should call that helper.

use super::gdn_block::state_s_offset;
use super::*;
use crate::quant::{BF16, FP8_E4M3, MXFP4};

impl Model {
    pub(super) fn gdn_block_batch(
        &self,
        g: &GdnBlock,
        sts: &[&SequenceState],
        n: &TensorRef,
        out: &TensorRef,
        profile: bool,
        fold: Option<&NormFold>,
    ) -> Result<()> {
        let rows = sts.len();
        let x = self.ws.x.gdn.as_ref().ok_or_else(|| Error::invalid("GatedDeltaNet workspace missing"))?;
        let l = &g.lin;
        let act = self.act;
        let (c, hv, dv, dk, ks) = (l.conv_dim(), l.v_heads, l.v_head_dim, l.k_head_dim, l.conv_kernel);
        let vdim = hv * dv;

        let qkv = self.view(&x.qkv, act, &[rows, c]);
        let z = self.view(&x.z, act, &[rows, vdim]);
        let a = self.view(&x.a, act, &[rows, hv]);
        let bg = self.view(&x.b, act, &[rows, hv]);
        let conv = self.view(&x.conv, act, &[rows, c]);
        let core = self.view(&x.core, act, &[rows, hv, dv]);
        let normed = self.view(&x.normed, act, &[rows, hv, dv]);

        // projections over all rows, with the same fold / fusion decisions as the solo step
        let mut ba_fused = false;
        if let Some(f) = fold {
            if let (BF16, BF16) = (g.in_b.format, g.in_a.format) {
                let (inputs, outputs) = f.pair(n, &g.in_b.parts[0], &g.in_a.parts[0], &bg, &a);
                ba_fused = self.try_fused(Op::RmsNormAddMatmulPair, inputs, outputs, f.attrs.clone(), profile)?;
            }
            if !ba_fused {
                let (inputs, outputs) = f.unfused(n);
                self.exec(Op::RmsNormAdd, inputs, outputs, f.attrs.clone(), profile)?;
            }
        }
        let qz_fused = match (g.in_qkv.format, g.in_z.format) {
            (FP8_E4M3, FP8_E4M3) => self.try_fused(
                Op::MatmulFp8Pair,
                vec![
                    n.clone(),
                    g.in_qkv.parts[0].clone(),
                    g.in_qkv.parts[1].clone(),
                    g.in_z.parts[0].clone(),
                    g.in_z.parts[1].clone(),
                ],
                vec![qkv.clone(), z.clone()],
                OpAttrs::default(),
                profile,
            )?,
            (MXFP4, MXFP4) => self.try_mxfp4_pair(n, &g.in_qkv, &g.in_z, &qkv, &z, profile)?,
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
                    vec![bg.clone(), a.clone()],
                    OpAttrs::default(),
                    profile,
                )?,
                (MXFP4, MXFP4) => self.try_mxfp4_pair(n, &g.in_b, &g.in_a, &bg, &a, profile)?,
                _ => false,
            };
        if !ba_fused {
            self.proj(n, &g.in_b, &bg, profile)?;
            self.proj(n, &g.in_a, &a, profile)?;
        }

        // conv + delta rule: one T = 1 call per row, on that row's own recurrent slot. These are the solo step's one-row
        // calls, so the row-exact mark is off for them (as for the per-row attention in `attn_block_batch`): a backend
        // that registers row-exact kernels for other ops must not restrict these to them.
        let delta_attrs = OpAttrs { eps: 1e-6, ..Default::default() };
        // Row tables (R3a): the same two calls for ALL rows in two launches, each row on its own recurrent state; the state
        // tensors name row 0's buffers, the kernel takes row r's from the table.
        let mut table_done = false;
        if let (Some(conv_tab), Some(delta_tab)) = (
            self.row_tab_for(rows, g.slot, 0, 0),
            self.row_tab_for(rows, g.slot, 0, state_s_offset(l)),
        ) {
            if let StateSlot::Recurrent(rec) = &sts[0].slots[g.slot] {
                let conv_state = TensorRef::new(rec.buf.clone(), DType::F32, &[c, ks - 1]);
                let mut s_state = TensorRef::new(rec.buf.clone(), DType::F32, &[hv, dv, dk]);
                s_state.offset = state_s_offset(l);
                let mut cc = OpCall::new(
                    Op::GdnConv,
                    vec![qkv.clone(), g.conv.clone()],
                    vec![conv.clone(), conv_state],
                    OpAttrs::default(),
                );
                cc.rows = Some(conv_tab);
                let mut dc = OpCall::new(
                    Op::GdnDeltaRule,
                    vec![conv.clone(), a.clone(), bg.clone(), g.a_log.clone(), g.dt_bias.clone()],
                    vec![core.clone(), s_state],
                    delta_attrs.clone(),
                );
                dc.rows = Some(delta_tab);
                let bad = [&cc, &dc].into_iter().find_map(|x| x.check().and_then(|_| self.registry.select(x).map(|_| ())).err().map(|e| format!("GatedDeltaNet {}: {e}", x.op)));
                match bad {
                    None => {
                        self.exec_call(cc, profile)?;
                        self.exec_call(dc, profile)?;
                        table_done = true;
                    }
                    Some(why) => self.row_table_fallback(&why),
                }
            }
        }
        let mark = self.exact_rows.swap(false, std::sync::atomic::Ordering::Relaxed);
        if !table_done {
            self.mark_state_bound();
        }
        let per_row = (|| -> Result<()> {
            if table_done {
                return Ok(());
            }
            for (r, st) in sts.iter().enumerate() {
                let rec = match &st.slots[g.slot] {
                    StateSlot::Recurrent(r) => r,
                    StateSlot::AttnKv(_) => {
                        return Err(Error::invalid("GatedDeltaNet block bound to an attention state slot"))
                    }
                };
                let conv_state = TensorRef::new(rec.buf.clone(), DType::F32, &[c, ks - 1]);
                let mut s_state = TensorRef::new(rec.buf.clone(), DType::F32, &[hv, dv, dk]);
                s_state.offset = state_s_offset(l);
                let (qkv_r, a_r, b_r, conv_r, core_r) =
                    (qkv.rows(r, 1), a.rows(r, 1), bg.rows(r, 1), conv.rows(r, 1), core.rows(r, 1));
                let folded = self.try_fused(
                    Op::GdnConvDelta,
                    vec![qkv_r.clone(), g.conv.clone(), a_r.clone(), b_r.clone(), g.a_log.clone(), g.dt_bias.clone()],
                    vec![core_r.clone(), s_state.clone(), conv_state.clone()],
                    delta_attrs.clone(),
                    profile,
                )?;
                if !folded {
                    self.exec(
                        Op::GdnConv,
                        vec![qkv_r, g.conv.clone()],
                        vec![conv_r.clone(), conv_state],
                        OpAttrs::default(),
                        profile,
                    )?;
                    self.exec(
                        Op::GdnDeltaRule,
                        vec![conv_r, a_r, b_r, g.a_log.clone(), g.dt_bias.clone()],
                        vec![core_r, s_state],
                        delta_attrs.clone(),
                        profile,
                    )?;
                }
            }
            Ok(())
        })();
        self.exact_rows.store(mark, std::sync::atomic::Ordering::Relaxed);
        per_row?;

        // gated RMSNorm over rows of dv and the output projection, over all rows
        let nrows = [rows * hv, dv];
        let norm_attrs = OpAttrs { eps: self.dims.rms_eps, ..Default::default() };
        if g.out.format == FP8_E4M3 {
            let folded = self.try_fused(
                Op::GatedRmsnormMatmulFp8,
                vec![
                    core.reshaped(&nrows),
                    z.reshaped(&nrows),
                    g.norm.clone(),
                    g.out.parts[0].clone(),
                    g.out.parts[1].clone(),
                ],
                vec![out.clone()],
                norm_attrs.clone(),
                profile,
            )?;
            if folded {
                return Ok(());
            }
            // The solo step folds the norm into the fp8 out-projection with a one-row kernel. Where no multi-row form exists
            // the rows run the unfused chain below (the norm over all rows, then the row-exact projection: the weights are
            // read once) if the fusion is a registered bitwise equivalent of that chain (`crate::fusion`, CUDA); only for a
            // fusion without that proof does the fold run once per row (same op, same bits, the weights read once per row:
            // the correct-but-slow stage S2 of the rb design).
            if crate::fusion::fusion(self.backend.name(), Op::GatedRmsnormMatmulFp8).is_none() {
                let row = [hv, dv];
                let one_row = |r: usize| {
                    OpCall::new(
                        Op::GatedRmsnormMatmulFp8,
                        vec![
                            core.rows(r, 1).reshaped(&row),
                            z.rows(r, 1).reshaped(&row),
                            g.norm.clone(),
                            g.out.parts[0].clone(),
                            g.out.parts[1].clone(),
                        ],
                        vec![out.rows(r, 1)],
                        norm_attrs.clone(),
                    )
                };
                let mark = self.exact_rows.swap(false, std::sync::atomic::Ordering::Relaxed);
                let usable = self.fused_ok(&one_row(0));
                let ran = (|| -> Result<()> {
                    if usable {
                        for r in 0..rows {
                            let c = one_row(r);
                            self.exec(c.op, c.inputs, c.outputs, c.attrs, profile)?;
                        }
                    }
                    Ok(())
                })();
                self.exact_rows.store(mark, std::sync::atomic::Ordering::Relaxed);
                ran?;
                if usable {
                    return Ok(());
                }
            }
        }
        let gated = self.try_fused(
            Op::GatedRmsnorm,
            vec![core.reshaped(&nrows), z.reshaped(&nrows), g.norm.clone()],
            vec![normed.reshaped(&nrows)],
            norm_attrs.clone(),
            profile,
        )?;
        if gated {
            return self.proj_reduce(&normed.reshaped(&[rows, vdim]), &g.out, out, rows, profile);
        }
        self.exec(
            Op::RmsNorm,
            vec![core.reshaped(&nrows), g.norm.clone()],
            vec![normed.reshaped(&nrows)],
            norm_attrs,
            profile,
        )?;
        self.exec(
            Op::SiluGlu,
            vec![z.reshaped(&nrows), normed.reshaped(&nrows)],
            vec![core.reshaped(&nrows)],
            OpAttrs::default(),
            profile,
        )?;
        self.proj_reduce(&core.reshaped(&[rows, vdim]), &g.out, out, rows, profile)
    }
}
