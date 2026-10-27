//! The gated short-conv block of Liquid's LFM2 / LFM2.5 (HF `Lfm2ShortConv`) in the generic decoder.
//!
//! Dataflow of one block (`n` = pre-normed input `[T, H]`, all activations in the model's act dtype):
//! ```text
//!   bcx = n @ in_proj^T           [T, 3H]       rows [B | C | x]
//!   y   = ShortConv(bcx, taps)    [T, H]        C * causalconv(B * x): no activation, conv state in the slot
//!   out = y @ out_proj^T          [T, H]
//! ```
//! The only state is the conv's last `K - 1` columns of `B * x` (f32, `[H, K-1]`), the same `SlotSpec::Recurrent` slot the
//! GatedDeltaNet block uses, so prefix snapshots, restore and the failed-forward reset need nothing new. Batched decode
//! follows `gdn_batch`: the projections run once on all rows, the conv once per row on that row's own slot.

use super::*;
use crate::state::RecurrentState;

/// Bytes the recurrent slot of one short-conv layer holds: the conv state `[H, K-1]` f32.
pub(super) fn state_bytes(hidden: usize, kernel: usize) -> usize {
    hidden * (kernel - 1) * 4
}

pub(super) struct ConvBlock {
    pub pre_norm: TensorRef,
    /// `[3H, H]`, rows `[B | C | x]`.
    pub in_proj: QTensor,
    /// Conv taps `[H, K]`.
    pub conv: TensorRef,
    pub out: QTensor,
    pub slot: usize,
    pub hidden: usize,
    pub kernel: usize,
}

/// Activation buffers of the short-conv blocks (allocated only when the recipe has the block).
pub(super) struct ShortConvWs {
    /// The `in_proj` output `[T, 3H]`.
    pub(super) bcx: Buffer,
    /// The conv output `[T, H]`.
    pub(super) y: Buffer,
}

pub(super) fn alloc_ws(wb: &mut dyn FnMut(usize) -> Result<Buffer>, mt: usize, esz: usize, hidden: usize) -> Result<ShortConvWs> {
    Ok(ShortConvWs { bcx: wb(mt * 3 * hidden * esz)?, y: wb(mt * hidden * esz)? })
}

impl Model {
    pub(super) fn conv_block(&self, c: &ConvBlock, st: &SequenceState, n: &TensorRef, out: &TensorRef, t: usize, profile: bool) -> Result<()> {
        self.conv_block_segs(c, &[Seg { st, lo: 0, len: t, pos: st.len }], n, out, t, profile)
    }

    /// [`Model::conv_block`] over the rows of one or several segments (a packed prefill pass): the projections run once on all
    /// `t` rows, the conv once per segment on that segment's rows and its own recurrent slot.
    pub(super) fn conv_block_segs(&self, c: &ConvBlock, segs: &[Seg<'_>], n: &TensorRef, out: &TensorRef, t: usize, profile: bool) -> Result<()> {
        let x = self.ws.x.short.as_ref().ok_or_else(|| Error::invalid("short-conv workspace missing"))?;
        let (h, ks, act) = (c.hidden, c.kernel, self.act);
        let bcx = self.view(&x.bcx, act, &[t, 3 * h]);
        let y = self.view(&x.y, act, &[t, h]);
        self.proj(n, &c.in_proj, &bcx, profile)?;
        let st = segs[0].st;
        if segs.len() == 1 && self.verifying() && t > 1 {
            // speculative verify: one row at a time (each call is the decode step's), and the slot's state after every row but
            // the last saved for the rollback (`SequenceState::spec_rollback`)
            let rec = Self::conv_slot(c, st)?;
            let state = TensorRef::new(rec.buf.clone(), DType::F32, &[h, ks - 1]);
            let save = st.spec_rows.bufs.len() >= t - 1 && self.recorder.lock().unwrap().is_none();
            let off = st.recurrent_offset(c.slot);
            for r in 0..t {
                self.conv_op(c, &bcx.rows(r, 1), &y.rows(r, 1), &state, profile)?;
                if save && r + 1 < t {
                    self.backend.copy(&st.spec_rows.bufs[r], off, &rec.buf, 0, rec.bytes, 0)?;
                }
            }
        } else {
            for sg in segs {
                let rec = Self::conv_slot(c, sg.st)?;
                let state = TensorRef::new(rec.buf.clone(), DType::F32, &[h, ks - 1]);
                self.conv_op(c, &bcx.rows(sg.lo, sg.len), &y.rows(sg.lo, sg.len), &state, profile)?;
            }
        }
        self.proj_reduce(&y, &c.out, out, t, profile)
    }

    /// One decode token of each of `sts.len()` sequences: the projections once on all rows, the conv once per row (the solo
    /// step's one-row call) on that row's own slot, so a batched row is the solo row by construction.
    pub(super) fn conv_block_batch(&self, c: &ConvBlock, sts: &[&SequenceState], n: &TensorRef, out: &TensorRef, profile: bool) -> Result<()> {
        let rows = sts.len();
        let x = self.ws.x.short.as_ref().ok_or_else(|| Error::invalid("short-conv workspace missing"))?;
        let (h, ks, act) = (c.hidden, c.kernel, self.act);
        let bcx = self.view(&x.bcx, act, &[rows, 3 * h]);
        let y = self.view(&x.y, act, &[rows, h]);
        self.proj(n, &c.in_proj, &bcx, profile)?;
        for (r, st) in sts.iter().enumerate() {
            let rec = Self::conv_slot(c, st)?;
            let state = TensorRef::new(rec.buf.clone(), DType::F32, &[h, ks - 1]);
            self.conv_op(c, &bcx.rows(r, 1), &y.rows(r, 1), &state, profile)?;
        }
        self.proj_reduce(&y, &c.out, out, rows, profile)
    }

    fn conv_slot<'a>(c: &ConvBlock, st: &'a SequenceState) -> Result<&'a RecurrentState> {
        match &st.slots[c.slot] {
            StateSlot::Recurrent(r) => Ok(r),
            StateSlot::AttnKv(_) => Err(Error::invalid("short-conv block bound to an attention state slot")),
        }
    }

    fn conv_op(&self, c: &ConvBlock, bcx: &TensorRef, y: &TensorRef, state: &TensorRef, profile: bool) -> Result<()> {
        self.exec(Op::ShortConv, vec![bcx.clone(), c.conv.clone()], vec![y.clone(), state.clone()], OpAttrs::default(), profile)
    }
}
