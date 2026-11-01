//! The DeepSeek-V4.1 ops (`Op::HcExpand` .. `Op::EngramMerge`, appended once by lane 1a, plus the fused `Op::HcSublayer`): their attributes, call checks
//! and shape facts. Exact semantics, shapes and the official function each one mirrors are in the `Op` doc table
//! (`crate::ops`) and in `docs/design/dsv41-ops.md`, which is the contract the cpu reference (`intarsia-backend-cpu`'s
//! `dsv41_ref`), the CUDA port and the Engram lane implement against.
//!
//! Every op is defined for an arbitrary chunk: `T` rows at absolute positions `pos0 .. pos0 + T`. The official code only
//! runs a prefill from position 0 or a one-token decode step (`model.py` `Compressor.forward`, `_window_kv`, the indexer
//! mask); the general rules here reduce to those two cases.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{Op, OpCall, ShapeInfo};

/// What a registered-but-unimplemented kernel's error says (`intarsia kernels test` reports such kernels as STUB instead
/// of failing; the cpu stubs in `intarsia-backend-cpu`'s `dsv41_ref::STUBS` use it).
pub const STUB_MARK: &str = "not implemented (lane 1b)";

/// Element format of `Op::ActQdq` (the reference's in-place `act_quant` / `fp4_act_quant`: quantise, then dequantise
/// back to the activation dtype).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum QdqFmt {
    /// e4m3, saturating at 448.
    #[default]
    Fp8E4m3,
    /// e2m1 (values 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives), saturating at 6.
    Fp4E2m1,
}

/// Scale format of `Op::ActQdq`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum QdqScale {
    /// Power of two, `2^ceil(log2(amax / fmax))` (`scale_fmt = "ue8m0"`, stored as E8M0 by the reference).
    #[default]
    Ue8m0,
    /// `e4m3(amax / fmax)` (the compressed KV's fp4 groups of 16).
    E4m3,
    /// Plain f32 `amax / fmax`.
    F32,
}

/// Router score function of `Op::MoeRoute` (`ModelArgs.score_func`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScoreFunc {
    Softmax,
    Sigmoid,
    /// `sqrt(softplus(x))`, DeepSeek-V4.1.
    #[default]
    SqrtSoftplus,
}

/// Attributes of the DeepSeek-V4.1 ops (`OpAttrs::dsv`); every other op ignores them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DsvAttrs {
    /// `SiluGluClamp`: `swiglu_limit` (up clamped to `[-limit, limit]`, gate to `<= limit`); 0 = no clamp.
    pub limit: f32,
    /// `CompressPool`, `IndexerBlocks`, `IndexerTopk`: tokens per compressed row (`compress_ratios[layer]`, >= 1).
    pub ratio: usize,
    /// `HcMixes`, `HcSublayer`: Sinkhorn iterations (`hc_sinkhorn_iters`).
    pub iters: usize,
    /// `HcMixes`: the epsilon added inside the split and every Sinkhorn normalisation (`hc_eps`).
    pub hc_eps: f32,
    /// `ActQdq`: element format, block length along the last dimension, scale format.
    pub qdq_fmt: QdqFmt,
    pub qdq_block: usize,
    pub qdq_scale: QdqScale,
    /// `RopeTail`: rotate by the negated angle (the attention output's de-rotation).
    pub inverse: bool,
    /// `RopeTail`: row `t` sits at position `pos0 + t * pos_stride` (a compressed latent of ratio `r` stands for the first
    /// token of its group, so its rows are `r` positions apart).
    pub pos_stride: usize,
    /// `IndexerBlocks`: candidate block length in compressed rows (`candidate_block_size`). `MatmulFp8Block`: the weight
    /// and activation block (32).
    pub block: usize,
    /// `MoeRoute`: score function.
    pub score_func: ScoreFunc,
    /// Host calls only (admission cases, `exec::run_host`): upload this value as the call's device position
    /// (`OpCall::pos`, `i32 [1]`), so a sweep case exercises the device-position path. Kernels never read it; an
    /// `OpCall` built by a graph carries its position in `OpCall::pos` and leaves this `None`.
    pub host_dev_pos: Option<i32>,
}

impl Default for DsvAttrs {
    fn default() -> Self {
        DsvAttrs {
            limit: 0.0,
            ratio: 1,
            iters: 20,
            hc_eps: 1e-6,
            qdq_fmt: QdqFmt::Fp8E4m3,
            qdq_block: 32,
            qdq_scale: QdqScale::Ue8m0,
            inverse: false,
            pos_stride: 1,
            block: 8,
            score_func: ScoreFunc::SqrtSoftplus,
            host_dev_pos: None,
        }
    }
}

/// The DeepSeek-V4.1 ops, in discriminant order (44 ..).
pub const OPS: [Op; 19] = [
    Op::HcExpand,
    Op::HcMixes,
    Op::HcPre,
    Op::HcPost,
    Op::MoeRoute,
    Op::SiluGluClamp,
    Op::ActQdq,
    Op::RopeTail,
    Op::WindowKv,
    Op::AttnSparse,
    Op::CompressPool,
    Op::IndexerBlocks,
    Op::IndexerTopk,
    Op::GroupedMatmul,
    Op::Transpose01,
    Op::MatmulFp8Block,
    Op::EngramGather,
    Op::EngramMerge,
    Op::HcSublayer,
];

/// Ops whose position comes from `attrs.pos0` or a device-side `OpCall::pos` (`i32 [1]`, rewritten before a captured
/// decode step replays). With `pos` the call ignores `attrs.pos0` (semantics and checks) and takes capacity-sized shapes;
/// the rules are in docs/design/dsv41-ops.md ("Device position"). `ActQdq` is positioned only in that form (x `[G,N]`
/// written to rows `floor(P / ratio) ..` of y `[Ry,N]`); without `pos` it has no position.
pub const POSITIONED: [Op; 7] =
    [Op::RopeTail, Op::WindowKv, Op::AttnSparse, Op::CompressPool, Op::IndexerBlocks, Op::IndexerTopk, Op::ActQdq];

/// `(cmp_rows, groups)` of a chunk: compressed rows visible to the last query (`(pos0 + T) / ratio`) and the groups that
/// complete inside it (`(pos0 + T) / ratio - pos0 / ratio`).
pub fn compressed_rows(pos0: usize, t: usize, ratio: usize) -> (usize, usize) {
    let r = ratio.max(1);
    ((pos0 + t) / r, (pos0 + t) / r - pos0 / r)
}

/// Compressed rows query `t` of a chunk may attend (`compress_lens` of `Indexer.forward`): a group becomes visible once
/// the query has passed its last token.
pub fn visible_rows(pos0: usize, t: usize, ratio: usize) -> usize {
    (pos0 + t + 1) / ratio.max(1)
}

/// Payload width of an Engram record of `r` bytes (`Dh` e4m3 bytes plus `Dh / 32` E8M0 scales, so `r = 33 * Dh / 32`);
/// 0 when `r` is not such a width.
pub fn engram_head_dim(r: usize) -> usize {
    if r > 0 && r % 33 == 0 { r / 33 * 32 } else { 0 }
}

/// `(rows, positions)` of `Op::IndexerBlocks`' output for `n` compressed rows: `ceil(n / block)` blocks.
pub fn n_blocks(n: usize, block: usize) -> usize {
    n.div_ceil(block.max(1))
}

fn bad(op: Op, m: String) -> Result<()> {
    Err(Error::invalid(format!("{op}: {m}")))
}

/// Call checks of the DeepSeek-V4.1 ops (`OpCall::check` has validated arity and bounds).
pub fn check(call: &OpCall) -> Result<()> {
    let (i, o, a) = (&call.inputs, &call.outputs, &call.attrs);
    let op = call.op;
    // device position: attrs.pos0 is ignored, shapes are the capacity's
    let dev = call.pos.is_some();
    if dev && a.dsv.ratio == 0 {
        return bad(op, "ratio must be >= 1".into());
    }
    let rank = |t: &crate::backend::TensorRef, r: usize, what: &str| -> Result<()> {
        if t.shape.len() != r {
            return bad(op, format!("{what} must be rank {r}, got {:?}", t.shape));
        }
        Ok(())
    };
    let float = |t: &crate::backend::TensorRef, what: &str| -> Result<()> {
        if !t.dtype.is_float() || t.dtype == DType::Fp8E4m3 || t.dtype == DType::Fp8E5m2 {
            return bad(op, format!("{what} must be bf16 / f16 / f32, got {}", t.dtype));
        }
        Ok(())
    };
    let f32_ = |t: &crate::backend::TensorRef, what: &str| -> Result<()> {
        if t.dtype != DType::F32 {
            return bad(op, format!("{what} must be f32, got {}", t.dtype));
        }
        Ok(())
    };
    match op {
        Op::HcExpand => {
            rank(&i[0], 2, "x")?;
            rank(&o[0], 3, "y")?;
            let (t, d) = (i[0].shape[0], i[0].shape[1]);
            if o[0].shape[0] != t || o[0].shape[2] != d || o[0].shape[1] == 0 {
                return bad(op, format!("x {:?} y {:?}: need y [T,S,D]", i[0].shape, o[0].shape));
            }
            float(&i[0], "x")?;
            float(&o[0], "y")?;
        }
        Op::HcMixes => {
            rank(&i[0], 3, "x")?;
            let (t, s, d) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
            let m = (2 + s) * s;
            if i[1].shape != [m, s * d] || i[2].shape != [3] || i[3].shape != [m] {
                return bad(op, format!("fn {:?} scale {:?} base {:?}: need [{m},{}], [3], [{m}]", i[1].shape, i[2].shape, i[3].shape, s * d));
            }
            if o[0].shape != [t, s] || o[1].shape != [t, s] || o[2].shape != [t, s, s] {
                return bad(op, format!("pre {:?} post {:?} comb {:?}: need [T,S], [T,S], [T,S,S]", o[0].shape, o[1].shape, o[2].shape));
            }
            float(&i[0], "x")?;
            for (t, w) in [(&i[1], "fn"), (&i[2], "scale"), (&i[3], "base"), (&o[0], "pre"), (&o[1], "post"), (&o[2], "comb")] {
                f32_(t, w)?;
            }
            if a.dsv.iters == 0 {
                return bad(op, "iters must be >= 1".into());
            }
        }
        Op::HcPre => {
            rank(&i[0], 3, "x")?;
            let (t, s, d) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
            if i[1].shape != [t, s] || o[0].shape != [t, d] {
                return bad(op, format!("x {:?} mix {:?} y {:?}: need mix [T,S], y [T,D]", i[0].shape, i[1].shape, o[0].shape));
            }
            float(&i[0], "x")?;
            f32_(&i[1], "mix")?;
            float(&o[0], "y")?;
        }
        Op::HcSublayer => {
            rank(&i[0], 3, "x")?;
            let (t, s, d) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
            let m = (2 + s) * s;
            if i[1].shape != [m, s * d] || i[2].shape != [3] || i[3].shape != [m] {
                return bad(op, format!("fn {:?} scale {:?} base {:?}: need [{m},{}], [3], [{m}]", i[1].shape, i[2].shape, i[3].shape, s * d));
            }
            if i[4].shape != [t, s] || i[5].shape != [d] {
                return bad(op, format!("pre_in {:?} norm_w {:?}: need [T,S], [D]", i[4].shape, i[5].shape));
            }
            if o[0].shape != [t, d] || o[1].shape != [t, s] || o[2].shape != [t, s] || o[3].shape != [t, s, s] {
                return bad(op, format!("n {:?} pre {:?} post {:?} comb {:?}: need [T,D], [T,S], [T,S], [T,S,S]", o[0].shape, o[1].shape, o[2].shape, o[3].shape));
            }
            // the HcPost fold: all of post_x, post_mix, post_comb, h_out present, or all four empty `[0]`
            let empty = |x: &crate::backend::TensorRef| x.shape == [0];
            let fold = !empty(&i[6]);
            if fold {
                if i[6].shape != [t, d] || i[7].shape != [t, s] || i[8].shape != [t, s, s] || o[4].shape != [t, s, d] {
                    return bad(
                        op,
                        format!("fold: post_x {:?} post_mix {:?} post_comb {:?} h_out {:?}: need [T,D], [T,S], [T,S,S], [T,S,D]", i[6].shape, i[7].shape, i[8].shape, o[4].shape),
                    );
                }
                float(&i[6], "post_x")?;
                f32_(&i[7], "post_mix")?;
                f32_(&i[8], "post_comb")?;
                float(&o[4], "h_out")?;
            } else if !empty(&i[7]) || !empty(&i[8]) || !empty(&o[4]) {
                return bad(op, "without the fold post_mix, post_comb and h_out must be empty [0]".into());
            }
            float(&i[0], "x")?;
            float(&i[5], "norm_w")?;
            float(&o[0], "n")?;
            for (t, w) in [(&i[1], "fn"), (&i[2], "scale"), (&i[3], "base"), (&i[4], "pre_in"), (&o[1], "pre"), (&o[2], "post"), (&o[3], "comb")] {
                f32_(t, w)?;
            }
            if a.dsv.iters == 0 {
                return bad(op, "iters must be >= 1".into());
            }
        }
        Op::HcPost => {
            rank(&i[0], 2, "x")?;
            rank(&i[1], 3, "res")?;
            let (t, s, d) = (i[1].shape[0], i[1].shape[1], i[1].shape[2]);
            if i[0].shape != [t, d] || i[2].shape != [t, s] || i[3].shape != [t, s, s] || o[0].shape != i[1].shape {
                return bad(
                    op,
                    format!("x {:?} res {:?} post {:?} comb {:?} y {:?}", i[0].shape, i[1].shape, i[2].shape, i[3].shape, o[0].shape),
                );
            }
            float(&i[0], "x")?;
            float(&i[1], "res")?;
            f32_(&i[2], "post")?;
            f32_(&i[3], "comb")?;
            float(&o[0], "y")?;
        }
        Op::MoeRoute => {
            rank(&i[0], 2, "logits")?;
            let (t, e) = (i[0].shape[0], i[0].shape[1]);
            let k = a.top_k;
            if k == 0 || k > e || i[1].shape != [e] || o[0].shape != [t, k] || o[1].shape != [t, k] {
                return bad(op, format!("logits {:?} bias {:?} weights {:?} ids {:?} top_k {k}", i[0].shape, i[1].shape, o[0].shape, o[1].shape));
            }
            f32_(&i[0], "logits")?;
            f32_(&i[1], "bias")?;
            f32_(&o[0], "weights")?;
            if o[1].dtype != DType::I32 {
                return bad(op, "ids must be i32".into());
            }
        }
        Op::SiluGluClamp => {
            rank(&i[0], 2, "gate")?;
            if i[1].shape != i[0].shape || o[0].shape != i[0].shape {
                return bad(op, "shape mismatch".into());
            }
            if !(a.dsv.limit >= 0.0) {
                return bad(op, format!("limit {} must be >= 0", a.dsv.limit));
            }
            float(&i[0], "gate")?;
            float(&i[1], "up")?;
            float(&o[0], "out")?;
        }
        Op::ActQdq if dev => {
            rank(&i[0], 2, "x")?;
            rank(&o[0], 2, "y")?;
            if o[0].shape[1] != i[0].shape[1] || o[0].dtype != i[0].dtype {
                return bad(op, format!("x {} {:?} y {} {:?}: device position needs y [Ry,N] of x's dtype", i[0].dtype, i[0].shape, o[0].dtype, o[0].shape));
            }
            let (n, b) = (i[0].shape[1], a.dsv.qdq_block);
            if b == 0 || n % b != 0 {
                return bad(op, format!("last dim {n} is not a multiple of the block {b}"));
            }
            float(&i[0], "x")?;
        }
        Op::ActQdq => {
            if i[0].shape.is_empty() || o[0].shape != i[0].shape || o[0].dtype != i[0].dtype {
                return bad(op, format!("x {} {:?} y {} {:?}: same shape and dtype", i[0].dtype, i[0].shape, o[0].dtype, o[0].shape));
            }
            let n = *i[0].shape.last().unwrap();
            let b = a.dsv.qdq_block;
            if b == 0 || n % b != 0 {
                return bad(op, format!("last dim {n} is not a multiple of the block {b}"));
            }
            float(&i[0], "x")?;
        }
        Op::RopeTail => {
            rank(&i[0], 3, "x")?;
            rank(&i[1], 1, "inv_freq")?;
            let d = i[0].shape[2];
            let rd = 2 * i[1].shape[0];
            if rd == 0 || rd > d || o[0].shape != i[0].shape {
                return bad(op, format!("x {:?} inv_freq {:?}: rotated tail 2*{} must fit the head", i[0].shape, i[1].shape, i[1].shape[0]));
            }
            if a.dsv.pos_stride == 0 {
                return bad(op, "pos_stride must be >= 1".into());
            }
            float(&i[0], "x")?;
            f32_(&i[1], "inv_freq")?;
        }
        Op::WindowKv => {
            rank(&i[0], 2, "kv")?;
            rank(&o[0], 2, "win")?;
            rank(&o[1], 2, "ring")?;
            let (t, d) = (i[0].shape[0], i[0].shape[1]);
            let (w, wc) = (a.window, o[1].shape[0]);
            if w == 0 || wc < w || o[1].shape[1] != d || o[0].shape != [w + t, d] {
                return bad(op, format!("kv {:?} win {:?} ring {:?} window {w}: need win [W+T,D], ring [Wc>=W,D]", i[0].shape, o[0].shape, o[1].shape));
            }
            if o[0].dtype != i[0].dtype || o[1].dtype != i[0].dtype {
                return bad(op, "kv, win and ring share one dtype".into());
            }
        }
        Op::AttnSparse => {
            rank(&i[0], 3, "q")?;
            rank(&i[1], 2, "win")?;
            rank(&i[2], 2, "cmp")?;
            rank(&i[3], 2, "idx")?;
            rank(&i[4], 1, "sink")?;
            let (t, h, d) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
            let w = a.window;
            if w == 0 || i[1].shape != [w + t, d] || i[2].shape[1] != d || i[3].shape[0] != t || i[4].shape != [h] || o[0].shape != i[0].shape {
                return bad(
                    op,
                    format!("q {:?} win {:?} cmp {:?} idx {:?} sink {:?} window {w}", i[0].shape, i[1].shape, i[2].shape, i[3].shape, i[4].shape),
                );
            }
            if i[3].dtype != DType::I32 {
                return bad(op, "idx must be i32".into());
            }
            f32_(&i[4], "sink")?;
            for (x, w) in [(&i[0], "q"), (&i[1], "win"), (&i[2], "cmp"), (&o[0], "o")] {
                float(x, w)?;
            }
        }
        Op::CompressPool => {
            rank(&i[0], 2, "kv")?;
            let (t, d) = (i[0].shape[0], i[0].shape[1]);
            let r = a.dsv.ratio;
            let rc = o[1].shape.first().copied().unwrap_or(0);
            // device position: out [Gmax,D] for any Gmax (the groups completing at P that fit are written)
            let g = if dev { o[0].shape.first().copied().unwrap_or(0) } else { compressed_rows(a.pos0, t, r).1 };
            if r < 2 || o[0].shape.len() != 2 || i[1].shape != [t, d] || o[0].shape != [g, d] || o[1].shape != [rc, d] || o[2].shape != [rc, d] || rc < r || rc % r != 0 {
                return bad(
                    op,
                    format!(
                        "kv {:?} score {:?} out {:?} ring {:?}/{:?} ratio {r} pos0 {}: need out [{g},D], rings [Rc,D] with Rc a multiple of ratio >= 2",
                        i[0].shape, i[1].shape, o[0].shape, o[1].shape, o[2].shape, a.pos0
                    ),
                );
            }
            for (x, w) in [(&i[0], "kv"), (&i[1], "score"), (&o[1], "kv ring"), (&o[2], "score ring")] {
                f32_(x, w)?;
            }
            float(&o[0], "out")?;
        }
        Op::IndexerBlocks | Op::IndexerTopk => {
            rank(&i[0], 3, "q")?;
            rank(&i[1], 2, "k")?;
            rank(&i[2], 2, "w")?;
            let (t, hi, di) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
            let nk = i[1].shape[0];
            // device position: k is a capacity view [Ncap,Di] (any Ncap), every width below is the capacity's
            let want_nk = if dev { nk } else { compressed_rows(a.pos0, t, a.dsv.ratio).0 };
            if i[1].shape[1] != di || i[2].shape != [t, hi] || nk != want_nk {
                return bad(
                    op,
                    format!("q {:?} k {:?} w {:?} pos0 {} ratio {}: need k [(pos0+T)/ratio = {want_nk},Di], w [T,Hi]", i[0].shape, i[1].shape, i[2].shape, a.pos0, a.dsv.ratio),
                );
            }
            float(&i[0], "q")?;
            float(&i[1], "k")?;
            float(&i[2], "w")?;
            if op == Op::IndexerBlocks {
                let nb = n_blocks(nk, a.dsv.block);
                if a.top_k == 0 || o[0].shape != [t, nb] || o[0].dtype != DType::U8 {
                    return bad(op, format!("mask {} {:?}: need u8 [T,{nb}], top_k (blocks kept) >= 1", o[0].dtype, o[0].shape));
                }
            } else {
                let k = a.top_k.min(nk);
                if a.top_k == 0 || o[0].shape != [t, k] || o[0].dtype != DType::I32 {
                    return bad(op, format!("idx {} {:?}: need i32 [T,min(top_k,Nk) = {k}]", o[0].dtype, o[0].shape));
                }
                // optional candidate mask: `[T, ceil(Nk / block)]` u8, or an empty `[0]` (no candidates)
                let m = &i[3];
                let none = m.numel() == 0;
                if !none && (m.dtype != DType::U8 || m.shape != [t, n_blocks(nk, a.dsv.block)]) {
                    return bad(op, format!("candidates {} {:?}: need u8 [T,ceil(Nk/block)] or empty", m.dtype, m.shape));
                }
            }
        }
        Op::GroupedMatmul => {
            rank(&i[0], 2, "x")?;
            rank(&i[1], 3, "w")?;
            let (t, gk) = (i[0].shape[0], i[0].shape[1]);
            let (g, n, k) = (i[1].shape[0], i[1].shape[1], i[1].shape[2]);
            if g == 0 || gk != g * k || o[0].shape != [t, g * n] {
                return bad(op, format!("x {:?} w {:?} y {:?}: need x [T,G*K], w [G,N,K], y [T,G*N]", i[0].shape, i[1].shape, o[0].shape));
            }
            float(&i[0], "x")?;
            float(&i[1], "w")?;
            float(&o[0], "y")?;
        }
        Op::Transpose01 => {
            rank(&i[0], 3, "x")?;
            let s = &i[0].shape;
            if o[0].shape != [s[1], s[0], s[2]] || o[0].dtype != i[0].dtype {
                return bad(op, format!("x {:?} y {:?}: need y [B,A,C] of x's dtype", i[0].shape, o[0].shape));
            }
        }
        Op::MatmulFp8Block => {
            rank(&i[0], 2, "x")?;
            rank(&i[1], 2, "w")?;
            rank(&i[2], 2, "w_scale")?;
            let (m, k) = (i[0].shape[0], i[0].shape[1]);
            let n = i[1].shape[0];
            let b = a.dsv.block;
            if b == 0 || i[1].shape[1] != k || k % b != 0 || i[2].shape != [n.div_ceil(b), k / b] || o[0].shape != [m, n] {
                return bad(op, format!("x {:?} w {:?} w_scale {:?} y {:?} block {b}", i[0].shape, i[1].shape, i[2].shape, o[0].shape));
            }
            if i[1].dtype != DType::Fp8E4m3 || i[2].dtype != DType::U8 {
                return bad(op, format!("w must be fp8_e4m3 and w_scale u8 (e8m0), got {} / {}", i[1].dtype, i[2].dtype));
            }
            float(&i[0], "x")?;
        }
        Op::EngramGather => {
            rank(&i[0], 3, "records")?;
            let (t, c, r) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
            let dh = engram_head_dim(r);
            if dh == 0 || o[0].shape != [t, c * dh] {
                return bad(op, format!("records {:?} y {:?}: need records [T,C,Dh+Dh/32] (Dh a multiple of 32), y [T,C*Dh]", i[0].shape, o[0].shape));
            }
            if i[0].dtype != DType::U8 {
                return bad(op, format!("records must be u8 (e4m3 payload + E8M0 scales), got {}", i[0].dtype));
            }
            float(&o[0], "y")?;
        }
        Op::EngramMerge => {
            rank(&i[0], 3, "h")?;
            let (t, s, d) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
            if i[1].shape != [t, (s + 1) * d] || i[2].shape != [s, d] || i[3].shape != [s, d] || o[0].shape != i[0].shape {
                return bad(op, format!("h {:?} kv {:?} q_w {:?} k_w {:?}: need kv [T,(S+1)*D], q_w / k_w [S,D]", i[0].shape, i[1].shape, i[2].shape, i[3].shape));
            }
            for (x, w) in [(&i[0], "h"), (&i[1], "kv"), (&i[2], "q_w"), (&i[3], "k_w"), (&o[0], "out")] {
                float(x, w)?;
            }
        }
        _ => unreachable!("dsv41::ops::check called for {op}"),
    }
    Ok(())
}

/// Shape facts for kernel predicates (`ShapeInfo::from_call`).
pub fn shape(call: &OpCall, s: &mut ShapeInfo) {
    let (i, o) = (&call.inputs, &call.outputs);
    let first = |t: &crate::backend::TensorRef, d: usize| t.shape.get(d).copied().unwrap_or(0);
    match call.op {
        Op::HcExpand => {
            s.m = first(&i[0], 0);
            s.k = first(&i[0], 1);
            s.n = first(&o[0], 1);
        }
        Op::HcMixes | Op::HcPre | Op::EngramMerge | Op::HcSublayer => {
            s.m = first(&i[0], 0);
            s.n = first(&i[0], 1);
            s.k = first(&i[0], 2);
        }
        Op::HcPost => {
            s.m = first(&i[1], 0);
            s.n = first(&i[1], 1);
            s.k = first(&i[1], 2);
        }
        Op::MoeRoute => {
            s.m = first(&i[0], 0);
            s.experts = first(&i[0], 1);
            s.top_k = call.attrs.top_k;
        }
        Op::SiluGluClamp | Op::ActQdq => {
            s.m = first(&i[0], 0);
            s.k = i[0].numel() / s.m.max(1);
        }
        Op::RopeTail | Op::AttnSparse => {
            s.m = first(&i[0], 0);
            s.n_heads = first(&i[0], 1);
            s.head_dim = first(&i[0], 2);
            s.k = s.head_dim;
            if call.op == Op::AttnSparse {
                s.n_kv_heads = 1;
                s.kv_len = first(&i[2], 0);
                s.n = first(&i[3], 1);
            }
        }
        Op::WindowKv => {
            s.m = first(&i[0], 0);
            s.k = first(&i[0], 1);
            s.n = first(&o[1], 0);
        }
        Op::CompressPool => {
            s.m = first(&i[0], 0);
            s.k = first(&i[0], 1);
            s.n = first(&o[0], 0);
        }
        Op::IndexerBlocks | Op::IndexerTopk => {
            s.m = first(&i[0], 0);
            s.n_heads = first(&i[0], 1);
            s.head_dim = first(&i[0], 2);
            s.k = s.head_dim;
            s.kv_len = first(&i[1], 0);
            s.top_k = call.attrs.top_k;
        }
        Op::GroupedMatmul => {
            s.m = first(&i[0], 0);
            s.n = first(&i[1], 0) * first(&i[1], 1);
            s.k = first(&i[1], 2);
            s.weight = Some(i[1].dtype);
        }
        Op::Transpose01 => {
            s.m = first(&i[0], 0);
            s.n = first(&i[0], 1);
            s.k = first(&i[0], 2);
        }
        Op::MatmulFp8Block => {
            s.m = first(&i[0], 0);
            s.n = first(&i[1], 0);
            s.k = first(&i[1], 1);
            s.weight = Some(i[1].dtype);
        }
        Op::EngramGather => {
            s.m = first(&i[0], 0);
            s.n = first(&i[0], 1);
            s.k = engram_head_dim(first(&i[0], 2));
            s.weight = Some(DType::Fp8E4m3);
        }
        _ => {}
    }
}
