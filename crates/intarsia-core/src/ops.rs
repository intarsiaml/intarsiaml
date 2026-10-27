//! Op ids, call descriptors and conventions (ARCHITECTURE 3.2).
//!
//! # Conventions (all tensors dense row-major, see `kernels/abi/ik_abi.h`)
//!
//! | Op | inputs | outputs | attrs |
//! |---|---|---|---|
//! | `Embedding` | ids `[T]` i32, table `[V,H]` | `[T,H]` | |
//! | `RmsNorm` | x `[R,H]`, w `[H]` | `[R,H]` | eps, w_offset (scale = `w + w_offset`; 1.0 for zero-centred norms) |
//! | `RmsNormAdd` | x `[R,H]`, res `[R,H]`, w `[H]` | normed `[R,H]`, sum `[R,H]` (= x + res) | eps, w_offset |
//! | `Add` | a `[R,H]`, b `[R,H]` | `[R,H]` | |
//! | `QkNorm` | x `[T,N,D]`, w `[D]` | `[T,N,D]` | eps, w_offset |
//! | `Rope` | x `[T,N,D]` | `[T,N,D]` | pos0 (or `OpCall::pos`), theta, rotary_dim (neox / rotate-half over the first `rotary_dim` values, 0 = all `D`; the rest is copied) |
//! | `RopeScaled` | x `[T,N,D]`, inv_freq `[rd/2]` f32 | `[T,N,D]` | as `Rope`, but pair `i` rotates by `pos * inv_freq[i]` (instead of `theta^(-2i/rd)`) and cos / sin are multiplied by `rope_mscale`; llama3 / yarn scaling, table from `rope::RopeScaling::freqs` |
//! | `AddBias` | x `[R,N]`, bias `[N]` (x's dtype) | `[R,N]` (= x + bias, one rounding; may alias x) | |
//! | `Matmul` | x `[M,K]`, w `[N,K]` | `[M,N]` (= x @ w^T) | |
//! | `Logits` | x `[M,K]`, w `[V,K]` | `[M,V]` f32 | same math as `Matmul`, f32 output |
//! | `KvAppend` | new `[T,Nkv,D]` | cache `[Nkv,cap,D]` (written in place) | pos0 |
//! | `AttnPrefill` | q `[T,N,D]`, kc, vc `[Nkv,cap,D]` | `[T,N,D]` | pos0, scale, n_heads, n_kv_heads, head_dim, kv_cap |
//! | `AttnDecode` | as prefill with T == 1 | `[1,N,D]` | same |
//! | `SiluGlu` | gate `[T,I]`, up `[T,I]` | `[T,I]` | |
//! | `DynConv` | x `[T,H]`, k `[T,K*G]` (row t: K taps x G groups, tap-major), base `[K,H]` | `[T,H]` (= sum over taps o < K with t >= o of `(base[o,c] + k[t, o*G + c / (H/G)]) * x[t-o, c]`, f32 inside, one rounding; rows before row 0 are zero) | | DFlash2 draft's grouped dynamic causal conv along the block rows (lane dfh) |
//! | `GeluGlu` | gate `[T,I]`, up `[T,I]` | `[T,I]` (= `gelu_tanh(gate) * up`, `gelu_pytorch_tanh`: `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`, f32 inside, one rounding) | |
//! | `Argmax` | logits `[R,V]` | `[R]` i32 | |
//! | `Sample` | logits `[R,V]`, u `[R]` f32 in [0,1) | `[R]` i32 | temperature, top_k, top_p |
//! | `MatmulFp8` | x `[M,K]`, w `[N,K]` fp8_e4m3, scales `[2]` f32 = (weight_scale, input_scale) | `[M,N]` (bf16/f16/f32) | see `quant::fp8` |
//! | `MatmulExl3` | x `[M,K]`, trellis `[K/16,N/16,8*k2]` i16, suh `[K]` f16, svh `[N]` f16, codebook `[1]` i32 (0 3inst, 1 mcg, 2 mul1) | `[M,N]` (bf16/f16/f32) | see `quant::exl3` |
//! | `MatmulNvfp4` | x `[M,K]`, w `[N,K/2]` u8 (e2m1 pairs), block scales `[N,K/16]` fp8_e4m3, global `[1]` f32 | `[M,N]` (bf16/f16/f32) | see `quant::nvfp4` |
//! | `MatmulMxfp4` | x `[M,K]`, w `[N,K/2]` u8 (e2m1 pairs), scales `[N,K/32]` u8 (e8m0), act `[1]` f32 (0 = 16-bit x, 1 = dynamic MXFP4 QDQ of x, 2 = dynamic per-token fp8 e4m3 x) | `[M,N]` (bf16/f16/f32) | see `quant::mxfp4` |
//! | `MatmulMxfp4Pair` | x `[M,K]`, (w0 `[N0,K/2]` u8, scales0 `[N0,K/32]` u8), (w1 `[N1,K/2]` u8, scales1 `[N1,K/32]` u8), act `[1]` f32 | `[M,N0]`, `[M,N1]` (= two `MatmulMxfp4`s of the same x and act) | fused, see `quant::mxfp4` |
//! | `MoeTopk` | router logits `[T,E]` | weights `[T,K]` f32, expert ids `[T,K]` i32 | top_k, norm_topk |
//! | `MoeExperts` | x `[T,H]`, ids `[T,K]`, weights `[T,K]`, gate_up w `[E,2I,H/2]` u8, gate_up scales `[E,2I,H/16]` fp8, down w `[E,H,I/2]` u8, down scales `[E,H,I/16]` fp8, global `[E,3]` f32 | `[T,H]` | see `quant::moe` |
//! | `MoeCombine` | routed `[T,H]`, shared `[T,H]`, shared-gate logit `[T,1]` | `[T,H]` | |
//! | `GateSplit` | x `[T,N,2D]` (per head: `D` values then `D` gate values) | q `[T,N,D]`, gate `[T,N,D]` | |
//! | `SigmoidMul` | a `[R,X]`, gate `[R,X]` | `[R,X]` (= a * sigmoid(gate)) | |
//! | `GdnConv` | x `[T,C]`, taps `[C,K]` | y `[T,C]` (= silu(causal conv)), conv state `[C,K-1]` f32 (**in place**) | |
//! | `ShortConv` | bcx `[T,3C]` (the `in_proj` output, rows `[B \| C \| x]`), taps `[C,K]` | y `[T,C]` (= `C * causalconv(B * x)`: depthwise causal conv, no activation, no bias, tap `K-1` on the current token; f32 inside, one rounding), conv state `[C,K-1]` f32 (the last `K-1` values of `B * x`, **in place**) | LFM2's short-conv block (lane nab2) |
//! | `GdnDeltaRule` | qkv `[T,C]` (`C = 2*Hk*dk + Hv*dv`, rows `[q|k|v]`), a `[T,Hv]`, b `[T,Hv]`, A_log `[Hv]`, dt_bias `[Hv]` | o `[T,Hv,dv]`, state `[Hv,dv,dk]` f32 (**in place**) | eps (l2norm) |
//!
//! | `MatmulPair` | x `[M,K]`, w0 `[N0,K]`, w1 `[N1,K]` (dense) | `[M,N0]`, `[M,N1]` (= two `Matmul`s of one x) | fused decode GEMV pair, see `quant::fused` |
//! | `MatmulFp8Pair` | x `[M,K]`, (w0 `[N0,K]`, scales0 `[2]`), (w1 `[N1,K]`, scales1 `[2]`) fp8 | `[M,N0]`, `[M,N1]` (= two `MatmulFp8`s) | fused, see `quant::fused` |
//! | `MatmulNvfp4Glu` | x `[M,K]`, gate (w, block scales, global), up (w, block scales, global), nvfp4 `[N,K]` each | `[M,N]` (= `SiluGlu(MatmulNvfp4(gate), MatmulNvfp4(up))`) | fused, see `quant::fused` |
//! | `GatedRmsnorm` | x `[R,D]`, z `[R,D]`, w `[D]` | `[R,D]` (= `SiluGlu(z, RmsNorm(x, w))`) | eps, w_offset; fused, see `quant::fused` |
//! | `RmsNormAddMatmulPair` | x `[1,K]`, res `[1,K]`, norm w `[K]`, w0 `[N0,K]`, w1 `[N1,K]` (dense) | normed `[1,K]`, sum `[1,K]`, `[1,N0]`, `[1,N1]` (= `RmsNormAdd(x, res, w)`, then `MatmulPair` of the normed row) | eps, w_offset; fused decode norm + GEMV pair, see `quant::fused` |
//! | `GatedRmsnormMatmulFp8` | x `[R,128]`, z `[R,128]`, norm w `[128]`, w `[N,R*128]` fp8, scales `[2]` | `[1,N]` (= `MatmulFp8(GatedRmsnorm(x, z, w) as [1,R*128], w)`) | eps, w_offset; fused decode norm + GEMV, see `quant::fused` |
//! | `QkNormRopeKv` | q `[T,N,D]`, k `[T,Nkv,D]`, v `[T,Nkv,D]`, q_w `[D]`, k_w `[D]` (all one float dtype) | q_out `[T,N,D]`, k cache, v cache `[Nkv,cap,D]` (**in place**) | eps, w_offset, theta, rotary_dim, pos0 (or `OpCall::pos`); fused attention prologue, see below |
//! | `RmsNormAddMatmulFp8Pair` | x `[M,K]`, res `[M,K]`, norm w `[K]`, (w0 `[N0,K]`, scales0 `[2]`), (w1 `[N1,K]`, scales1 `[2]`) fp8 | normed `[M,K]`, sum `[M,K]`, `[M,N0]`, `[M,N1]` (= `RmsNormAdd(x, res, w)`, then `MatmulFp8Pair` of the normed rows) | eps, w_offset; the attention input norm folded into the k / v projections (P5) |
//! | `SigmoidMulMatmulFp8` | a `[M,K]`, gate `[M,K]`, w `[N,K]` fp8, scales `[2]` | `[M,N]` (= `MatmulFp8(SigmoidMul(a, gate), w)`) | the attention output gate folded into the o_proj input (P4b) |
//! | `SharedDownCombine` | x `[M,K]`, w `[N,K/2]` u8, block scales `[N,K/16]` fp8, global `[1]` f32 (nvfp4), routed `[M,N]`, shared-gate logit `[M,1]` | `[M,N]` (= `MoeCombine(routed, MatmulNvfp4(x, w), logit)`) | the MoE shared-expert down projection with the combine as its epilogue (P2a) |
//! | `GateSplitQkNormRopeKv` | q+gate `[T,N,2D]` (per head: `D` query values then `D` gate values), k `[T,Nkv,D]`, v `[T,Nkv,D]`, q_w `[D]`, k_w `[D]` | q_out `[T,N,D]`, gate `[T,N,D]`, k cache, v cache `[Nkv,cap,D]` (**in place**) (= `GateSplit`, then `QkNormRopeKv` of the split-off query) | as `QkNormRopeKv`; the gated attention's split folded into the prologue (P4a) |
//! | `GdnConvDelta` | x `[T,C]` (the projection, before the conv), taps `[C,K]`, a `[T,Hv]`, b `[T,Hv]`, A_log `[Hv]`, dt_bias `[Hv]` | o `[T,Hv,dv]`, state `[Hv,dv,dk]` f32, conv state `[C,K-1]` f32 (both **in place**) (= `GdnConv(x, taps)`, then `GdnDeltaRule` of its output) | eps (l2norm); the conv folded into the delta rule (P3) |
//!
//! The quantised-weight ops are specified in `quant/` (the reference is `intarsia-backend-cpu`'s `ref_*`). The fused ops
//! above are optional: a backend registers kernels for them only when it has a fast path, and the model
//! falls back to the unfused ops it replaces when `Registry::select` finds none.
//!
//! Attention is causal GQA: query `t` sits at absolute position `pos0 + t` and
//! attends cache rows `0..=pos0+t`; the caller must have appended K/V for
//! those rows first (`KvAppend`). Output tensors never alias inputs except
//! the caches of `KvAppend` and `QkNormRopeKv` (in place by definition).
//!
//! `QkNormRopeKv` is the per-layer sequence `q_out = Rope(QkNorm(q, q_w))`, `KvAppend(Rope(QkNorm(k, k_w)), k cache)`,
//! `KvAppend(v, v cache)` in one op (six launches become one). It is defined as exactly that composition, including
//! the rounding of every intermediate to the input dtype (`intarsia-backend-cpu`'s `fused_ref`), and it needs both
//! norms, so models without q/k norms keep the unfused ops. Like the other fused ops it is optional.

use crate::backend::TensorRef;
use crate::dtype::DType;
use crate::error::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Op {
    Embedding,
    RmsNorm,
    RmsNormAdd,
    Add,
    QkNorm,
    Rope,
    Matmul,
    Logits,
    KvAppend,
    AttnPrefill,
    AttnDecode,
    SiluGlu,
    Argmax,
    Sample,
    MatmulFp8,
    MatmulNvfp4,
    MoeTopk,
    MoeExperts,
    MoeCombine,
    GateSplit,
    SigmoidMul,
    GdnConv,
    GdnDeltaRule,
    MatmulPair,
    MatmulFp8Pair,
    MatmulNvfp4Glu,
    GatedRmsnorm,
    QkNormRopeKv,
    RmsNormAddMatmulPair,
    GatedRmsnormMatmulFp8,
    // appended (2026-10-19, core-op fusion phase 1): new ops go at the end, because admission sweeps seed their case data
    // from the op's discriminant (`quant::cases::sweep`), so inserting one would change every later op's cases
    RmsNormAddMatmulFp8Pair,
    SigmoidMulMatmulFp8,
    SharedDownCombine,
    // appended (2026-10-19, core-op fusion phase 2), for the same reason
    GateSplitQkNormRopeKv,
    GdnConvDelta,
    // appended (2026-10-19, core-op fusion P2'): the whole MoE block after the router logits, for the same reason
    MoeBlock,
    // appended (2026-10-21, M1 MXFP4), for the same reason
    MatmulMxfp4,
    // appended (2026-10-21, M2 EXL3), for the same reason
    MatmulExl3,
    // appended (2026-10-21, P1 MXFP4 decode, after M2 EXL3): `MatmulFp8Pair` for OCP MXFP4 weights
    MatmulMxfp4Pair,
    // appended (2026-10-22, rope scaling + attention bias), for the same reason
    RopeScaled,
    AddBias,
    // appended (2026-10-22, g4 Gemma-4): `SiluGlu` with the tanh-GELU activation (`gelu_pytorch_tanh`), for the same reason
    GeluGlu,
    DynConv,
    // appended (2026-10-27, nab2 LFM2.5): the gated short conv of Liquid's LFM2 (no activation, state in a `Recurrent` slot), for the same reason
    ShortConv,
}

impl Op {
    pub const ALL: [Op; 44] = [
        Op::Embedding,
        Op::RmsNorm,
        Op::RmsNormAdd,
        Op::Add,
        Op::QkNorm,
        Op::Rope,
        Op::Matmul,
        Op::Logits,
        Op::KvAppend,
        Op::AttnPrefill,
        Op::AttnDecode,
        Op::SiluGlu,
        Op::Argmax,
        Op::Sample,
        Op::MatmulFp8,
        Op::MatmulNvfp4,
        Op::MoeTopk,
        Op::MoeExperts,
        Op::MoeCombine,
        Op::GateSplit,
        Op::SigmoidMul,
        Op::GdnConv,
        Op::GdnDeltaRule,
        Op::MatmulPair,
        Op::MatmulFp8Pair,
        Op::MatmulNvfp4Glu,
        Op::GatedRmsnorm,
        Op::QkNormRopeKv,
        Op::RmsNormAddMatmulPair,
        Op::GatedRmsnormMatmulFp8,
        Op::RmsNormAddMatmulFp8Pair,
        Op::SigmoidMulMatmulFp8,
        Op::SharedDownCombine,
        Op::GateSplitQkNormRopeKv,
        Op::GdnConvDelta,
        Op::MoeBlock,
        Op::MatmulMxfp4,
        Op::MatmulExl3,
        Op::MatmulMxfp4Pair,
        Op::RopeScaled,
        Op::AddBias,
        Op::GeluGlu,
        Op::DynConv,
        Op::ShortConv,
    ];

    /// Stable snake_case id (also the `<op>` part of `ik_<op>_<variant>`).
    pub fn as_str(self) -> &'static str {
        match self {
            Op::Embedding => "embedding",
            Op::RmsNorm => "rmsnorm",
            Op::RmsNormAdd => "rmsnorm_add",
            Op::Add => "add",
            Op::QkNorm => "qk_norm",
            Op::Rope => "rope",
            Op::Matmul => "matmul",
            Op::Logits => "logits",
            Op::KvAppend => "kv_append",
            Op::AttnPrefill => "attn_prefill",
            Op::AttnDecode => "attn_decode",
            Op::SiluGlu => "silu_glu",
            Op::Argmax => "argmax",
            Op::Sample => "sample",
            Op::MatmulFp8 => "matmul_fp8",
            Op::MatmulNvfp4 => "matmul_nvfp4",
            Op::MoeTopk => "moe_topk",
            Op::MoeExperts => "moe_experts",
            Op::MoeCombine => "moe_combine",
            Op::GateSplit => "gate_split",
            Op::SigmoidMul => "sigmoid_mul",
            Op::GdnConv => "gdn_conv",
            Op::GdnDeltaRule => "gdn_delta_rule",
            Op::MatmulPair => "matmul_pair",
            Op::MatmulFp8Pair => "matmul_fp8_pair",
            Op::MatmulNvfp4Glu => "matmul_nvfp4_glu",
            Op::GatedRmsnorm => "gated_rmsnorm",
            Op::QkNormRopeKv => "qk_norm_rope_kv",
            Op::RmsNormAddMatmulPair => "rmsnorm_add_matmul_pair",
            Op::GatedRmsnormMatmulFp8 => "gated_rmsnorm_matmul_fp8",
            Op::RmsNormAddMatmulFp8Pair => "rmsnorm_add_matmul_fp8_pair",
            Op::SigmoidMulMatmulFp8 => "sigmoid_mul_matmul_fp8",
            Op::SharedDownCombine => "shared_down_combine",
            Op::GateSplitQkNormRopeKv => "gate_split_qk_norm_rope_kv",
            Op::GdnConvDelta => "gdn_conv_delta",
            Op::MoeBlock => "moe_block",
            Op::MatmulMxfp4 => "matmul_mxfp4",
            Op::MatmulExl3 => "matmul_exl3",
            Op::MatmulMxfp4Pair => "matmul_mxfp4_pair",
            Op::RopeScaled => "rope_scaled",
            Op::AddBias => "add_bias",
            Op::GeluGlu => "gelu_glu",
            Op::DynConv => "dyn_conv",
            Op::ShortConv => "short_conv",
        }
    }

    pub fn parse(s: &str) -> Option<Op> {
        Op::ALL.iter().copied().find(|o| o.as_str() == s)
    }

    /// (number of inputs, number of outputs)
    pub fn arity(self) -> (usize, usize) {
        match self {
            Op::Embedding => (2, 1),
            Op::RmsNorm => (2, 1),
            Op::RmsNormAdd => (3, 2),
            Op::Add => (2, 1),
            Op::QkNorm => (2, 1),
            Op::Rope => (1, 1),
            Op::Matmul | Op::Logits => (2, 1),
            Op::KvAppend => (1, 1),
            Op::AttnPrefill | Op::AttnDecode => (3, 1),
            Op::SiluGlu => (2, 1),
            Op::Argmax => (1, 1),
            Op::Sample => (2, 1),
            Op::MatmulFp8 => (3, 1),
            Op::MatmulNvfp4 => (4, 1),
            Op::MoeTopk => (1, 2),
            Op::MoeExperts => (8, 1),
            Op::MoeCombine => (3, 1),
            Op::GateSplit => (1, 2),
            Op::SigmoidMul => (2, 1),
            Op::GdnConv => (2, 2),
            Op::GdnDeltaRule => (5, 2),
            Op::MatmulPair => (3, 2),
            Op::MatmulFp8Pair => (5, 2),
            Op::MatmulNvfp4Glu => (7, 1),
            Op::GatedRmsnorm => (3, 1),
            Op::QkNormRopeKv => (5, 3),
            Op::RmsNormAddMatmulPair => (5, 4),
            Op::GatedRmsnormMatmulFp8 => (5, 1),
            Op::RmsNormAddMatmulFp8Pair => (7, 4),
            Op::SigmoidMulMatmulFp8 => (4, 1),
            Op::SharedDownCombine => (6, 1),
            Op::GateSplitQkNormRopeKv => (5, 4),
            Op::GdnConvDelta => (6, 3),
            Op::MoeBlock => (17, 1),
            Op::MatmulMxfp4 => (4, 1),
            Op::MatmulExl3 => (5, 1),
            Op::MatmulMxfp4Pair => (6, 2),
            Op::RopeScaled => (2, 1),
            Op::AddBias => (2, 1),
            Op::GeluGlu => (2, 1),
            Op::DynConv => (3, 1),
            Op::ShortConv => (2, 2),
        }
    }
}

impl std::fmt::Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Scalar attributes; each op reads the ones listed in the table above.
#[derive(Clone, Debug)]
pub struct OpAttrs {
    pub eps: f32,
    pub theta: f32,
    pub scale: f32,
    pub pos0: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub kv_cap: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    /// `MoeTopk`: renormalise the selected experts' probabilities to sum to 1 (HF `norm_topk_prob`).
    pub norm_topk: bool,
    /// `Rope`: number of leading values of each head that rotate (HF `partial_rotary_factor * head_dim`); 0 = the whole head.
    pub rotary_dim: usize,
    /// `RopeScaled`: multiplier on cos and sin (YaRN's attention factor; 1.0 otherwise).
    pub rope_mscale: f32,
    /// Norm ops: the scale applied is `w + w_offset` (1.0 for zero-centred norms such as Qwen3.5's), added in f32.
    pub w_offset: f32,
    /// `AttnPrefill` / `AttnDecode`: sliding window. Query at absolute position `p` attends keys `max(0, p + 1 - window)..=p`
    /// (the last `window` keys including itself, HF `sliding_window` with `sliding_window_mask_function`); 0 = all keys.
    /// A call with `window > 0` selects only kernels registered `.ext()`.
    pub window: usize,
    /// `Rope`: "proportional" rotation (Gemma 4 global layers, HF `rope_type = "proportional"`). Only the first
    /// `rotary_dim / 2` angles are non-zero: pair `(j, j + D/2)` for `j < rotary_dim / 2` is rotated by `pos * theta^(-2j/D)`
    /// (the exponent's denominator is the full head dim `D`, not `rotary_dim`), every other value is copied. `rotary_dim` is
    /// the number of rotated values (`partial_rotary_factor * D`). Selects only `.ext()` kernels.
    pub rope_prop: bool,
    /// `Add` / `Embedding` use the existing `scale` attr: `scale != 1.0` multiplies the result (`out = round(f32(a + b) * scale)`,
    /// `round(f32(table row) * scale)`, one rounding; Gemma's `layer_scalar` and embedding scale) and selects only `.ext()`
    /// kernels. Attention keeps reading `scale` as its softmax scale; that is not an extended use.
    ///
    /// `MoeExperts`: expert activation is tanh-GELU instead of SiLU (`gelu_tanh(gate) * up`). Selects only `.ext()` kernels.
    pub gelu: bool,
    /// `MatmulFp8` / `MatmulFp8Pair`: every fp8 weight of the call carries a static activation scale (W8A8,
    /// `QTensor::act_quant`). Read by kernel selection only (`ShapeInfo::fp8_static_act`: CUDA's default cuBLASLt e4m3 GEMM
    /// serves W8A8 calls only), never by a kernel; false for W8A16 weights and for every other op.
    pub fp8_static_act: bool,
    /// `MatmulMxfp4` / `MatmulMxfp4Pair`: the weights' activations are not quantised (W4A16, `QTensor::act_quant` false, the
    /// call's `act` input is 0). Read by kernel launch routing only (HIP: the `_rows` launchers keep the bf16 kernel for W4A16),
    /// never changes a result; false for W4A4 and for every other op.
    pub mxfp4_w4a16: bool,
    /// `MatmulMxfp4` / `MatmulMxfp4Pair`: the call's `act` input is 2, W4A8 (dynamic per-token fp8 e4m3 activations,
    /// `QTensor::act_fp8`). Read by kernel selection (`ShapeInfo::mxfp4_a8`): only kernels registered for W4A8 serve it, every
    /// other MXFP4 kernel declines it, because they read any nonzero `act` as W4A4; false for every other op.
    pub mxfp4_a8: bool,
}

impl Default for OpAttrs {
    fn default() -> Self {
        OpAttrs {
            eps: 1e-6,
            theta: 10000.0,
            scale: 1.0,
            pos0: 0,
            n_heads: 0,
            n_kv_heads: 0,
            head_dim: 0,
            kv_cap: 0,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            norm_topk: true,
            rotary_dim: 0,
            rope_mscale: 1.0,
            w_offset: 0.0,
            window: 0,
            rope_prop: false,
            gelu: false,
            fp8_static_act: false,
            mxfp4_w4a16: false,
            mxfp4_a8: false,
        }
    }
}

/// One op invocation: what to compute and on which buffers.
#[derive(Clone, Debug)]
pub struct OpCall {
    pub op: Op,
    pub inputs: Vec<TensorRef>,
    pub outputs: Vec<TensorRef>,
    pub attrs: OpAttrs,
    /// Pre-allocated scratch workspace, set by the caller when the selected kernel declares
    /// a scratch requirement (`KernelImpl::scratch`). `u8` view of at least the declared size,
    /// 256-byte aligned, uninitialised, stream-ordered (see kernels/README.md). `None` otherwise.
    pub scratch: Option<TensorRef>,
    /// Device-side position (`Rope`, `KvAppend`, `AttnDecode`): an `i32 [1]` tensor holding `pos0`. A kernel
    /// that runs with it reads the position when it executes, not when it was launched, so a captured CUDA graph
    /// replays at every position. `attrs.pos0` then only bounds the call (validation, shape predicates); results
    /// are bit-identical to a launch whose host `pos0` equals the tensor's value. `None` = use `attrs.pos0`.
    pub pos: Option<TensorRef>,
    /// Speculative-decoding verify call: row `t` of a multi-row call must equal, bit for bit, the single-row
    /// decode launch at position `pos0 + t`. `Registry::select` then picks among the kernels registered
    /// `rows_exact` for this op (and fails if none fits) instead of the ordinary priority order. `false` (every
    /// ordinary call) leaves selection as it was, except that `rows_exact` kernels are never picked.
    pub exact_rows: bool,
    /// Row-table call (batched decode, CUDA): the call covers `rows.rows` sequences at once, each with its own state buffers.
    /// See [`RowTab`]. `None` (every ordinary call): the state tensors of the call are the state.
    pub rows: Option<RowTab>,
}

/// The per-sequence buffers of a batched decode call, as a table of device addresses that the host rewrites before every step
/// (so a captured graph replays with any set of sequences). Defined for `GdnConv`, `GdnDeltaRule`, `KvAppend`, `AttnDecode`, `QkNormRopeKv`.
///
/// Such a call is the one-row (`T = 1`) call of the solo decode step on each of `rows` rows at once: row `r` of the activations
/// (`[rows, ...]`) is the single input row of sequence `r`, and the tensors that name a sequence's state (the conv and delta
/// state, the K / V cache) describe **row 0's buffers** and serve as shape / dtype carriers only: the kernel reads, and for
/// `GdnConv` / `GdnDeltaRule` / `KvAppend` writes, the buffer whose address is table entry `[r, index]` (plus `byte_off`) for
/// row `r`. `AttnDecode` and `QkNormRopeKv` use the K cache at entry `index` and the V cache at `index + 1`. `KvAppend`, `AttnDecode` and
/// `Rope` also take the per-row positions in `OpCall::pos` (i32 `[rows]`). Row `r` is bitwise the solo one-row call on
/// sequence `r` (G-B for the row-table kernels, `tests/row_tables.rs`).
#[derive(Clone, Debug)]
pub struct RowTab {
    /// u8 `[rows, stride, 8]`: little-endian u64 device addresses, row-major.
    pub ptrs: TensorRef,
    /// Rows of the call (the leading dimension of the activations).
    pub rows: usize,
    /// Entries per row of the table.
    pub stride: usize,
    /// This call's entry within a row.
    pub index: usize,
    /// Bytes added to the table entry (the delta state sits after the conv state in a layer's recurrent buffer).
    pub byte_off: usize,
}

impl OpCall {
    pub fn new(op: Op, inputs: Vec<TensorRef>, outputs: Vec<TensorRef>, attrs: OpAttrs) -> OpCall {
        OpCall { op, inputs, outputs, attrs, scratch: None, pos: None, exact_rows: false, rows: None }
    }

    /// The solo-step calls a call stands for: `rows` for a row-table call and for the per-row ops (`QkNorm`, `Rope`,
    /// `RopeScaled`) run once over a `[B, ...]` batch, one otherwise. `plan::check_batch_ops` counts a batched step's per-row
    /// ops with this weight.
    pub fn row_weight(&self) -> usize {
        match (&self.rows, self.op) {
            (Some(r), _) => r.rows,
            (None, Op::QkNorm | Op::Rope | Op::RopeScaled) => self.inputs.first().map_or(1, |t| t.shape.first().copied().unwrap_or(1)),
            _ => 1,
        }
    }

    /// The same call on `rows` rows, for the row-local fused decode ops whose activation tensors all lead with the row
    /// count (`RmsNormAddMatmulFp8Pair`, `SigmoidMulMatmulFp8`, `SharedDownCombine`, `MoeBlock`); `None` for any other op. Only the
    /// shapes change (the views keep their buffers), so the result is a `Registry::select` probe, never a launch: the
    /// model asks whether a batched step of `rows` rows would select a kernel for the op too, and does not fuse the solo
    /// step when it would not (the batch must run the solo step's op multiset, `plan::check_batch_ops`).
    pub fn with_rows(&self, rows: usize) -> Option<OpCall> {
        let (ins, outs): (&[usize], &[usize]) = match self.op {
            Op::RmsNormAddMatmulFp8Pair => (&[0, 1], &[0, 1, 2, 3]),
            Op::SigmoidMulMatmulFp8 => (&[0, 1], &[0]),
            Op::SharedDownCombine => (&[0, 4, 5], &[0]),
            Op::MoeBlock => (&[0, 1, 2], &[0]),
            _ => return None,
        };
        let mut c = self.clone();
        for &i in ins {
            if let Some(t) = c.inputs.get_mut(i).filter(|t| !t.shape.is_empty()) {
                t.shape[0] = rows;
            }
        }
        for &o in outs {
            if let Some(t) = c.outputs.get_mut(o).filter(|t| !t.shape.is_empty()) {
                t.shape[0] = rows;
            }
        }
        Some(c)
    }

    /// `GateSplitQkNormRopeKv` as the `QkNormRopeKv` call it ends in: the query input is the first half of each head of
    /// the gated row (`[T,N,D]` over `[T,N,2D]`; the view keeps the buffer, only the shape is the split-off query's, so
    /// its row stride is not that view's and the call is a shape / dtype probe, not a launch), the gate output dropped.
    /// Attributes and the device position carry over. Panics for any other op.
    pub fn gate_split_inner(&self) -> OpCall {
        assert_eq!(self.op, Op::GateSplitQkNormRopeKv, "gate_split_inner of {}", self.op);
        let mut q = self.inputs[0].clone();
        if let [_, _, d2] = q.shape[..] {
            q.shape[2] = d2 / 2;
        }
        let mut inputs = self.inputs.clone();
        inputs[0] = q;
        let mut outputs = self.outputs.clone();
        outputs.remove(1);
        let mut c = OpCall::new(Op::QkNormRopeKv, inputs, outputs, self.attrs.clone());
        c.pos = self.pos.clone();
        c.exact_rows = self.exact_rows;
        c
    }

    /// Validate arity, ranks, shapes, dtypes and bounds. Backends and the
    /// harness call this so every implementation sees well-formed calls.
    pub fn check(&self) -> Result<()> {
        let (ni, no) = self.op.arity();
        let bad = |m: String| Err(Error::invalid(format!("{}: {m}", self.op)));
        if self.inputs.len() != ni || self.outputs.len() != no {
            return bad(format!(
                "expected {ni} inputs / {no} outputs, got {} / {}",
                self.inputs.len(),
                self.outputs.len()
            ));
        }
        for t in self.inputs.iter().chain(self.outputs.iter()) {
            t.check_bounds()?;
        }
        if let Some(p) = &self.pos {
            p.check_bounds()?;
            if !matches!(self.op, Op::Rope | Op::RopeScaled | Op::KvAppend | Op::AttnDecode | Op::QkNormRopeKv | Op::GateSplitQkNormRopeKv) {
                return Err(Error::invalid(format!("{}: takes no device-side position", self.op)));
            }
            // one position (token t sits at pos + t), or one per token (`[T]`: batched decode rows, each at its own position)
            let t = self.inputs.first().and_then(|x| x.shape.first().copied()).unwrap_or(1);
            let per_token = p.numel() == t && t > 1 && matches!(self.op, Op::Rope | Op::RopeScaled | Op::KvAppend | Op::AttnDecode | Op::QkNormRopeKv);
            if p.dtype != DType::I32 || (p.numel() != 1 && !per_token) {
                return Err(Error::invalid(format!("{}: device position must be i32 [1] (or [T] for a batch of decode rows), got {} {:?}", self.op, p.dtype, p.shape)));
            }
            if per_token && matches!(self.op, Op::KvAppend | Op::AttnDecode | Op::QkNormRopeKv) && self.rows.is_none() {
                return Err(Error::invalid(format!("{}: per-token positions need a row table", self.op)));
            }
        }
        if let Some(r) = &self.rows {
            r.ptrs.check_bounds()?;
            if !matches!(self.op, Op::GdnConv | Op::GdnDeltaRule | Op::KvAppend | Op::AttnDecode | Op::QkNormRopeKv) {
                return Err(Error::invalid(format!("{}: takes no row table", self.op)));
            }
            let t = self.inputs.first().and_then(|x| x.shape.first().copied()).unwrap_or(0);
            if r.ptrs.dtype != DType::U8
                || r.ptrs.shape != [r.rows, r.stride, 8]
                || r.rows != t
                || r.rows == 0
                || r.index >= r.stride
                || (matches!(self.op, Op::AttnDecode | Op::QkNormRopeKv) && r.index + 1 >= r.stride)
            {
                return Err(Error::invalid(format!(
                    "{}: row table {:?} {:?} rows {} stride {} index {} does not fit a call over {t} rows",
                    self.op, r.ptrs.dtype, r.ptrs.shape, r.rows, r.stride, r.index
                )));
            }
            if matches!(self.op, Op::KvAppend | Op::AttnDecode | Op::QkNormRopeKv) && self.pos.is_none() {
                return Err(Error::invalid(format!("{}: a row-table call needs the per-row positions (`pos`)", self.op)));
            }
        }
        let i = &self.inputs;
        let o = &self.outputs;
        let a = &self.attrs;
        let rank = |t: &TensorRef, r: usize, what: &str| -> Result<()> {
            if t.shape.len() != r {
                return Err(Error::invalid(format!("{}: {what} must be rank {r}, got {:?}", self.op, t.shape)));
            }
            Ok(())
        };
        let float = |t: &TensorRef, what: &str| -> Result<()> {
            if !t.dtype.is_float() {
                return Err(Error::invalid(format!("{}: {what} must be a float dtype", self.op)));
            }
            Ok(())
        };
        match self.op {
            Op::Embedding => {
                rank(&i[0], 1, "ids")?;
                rank(&i[1], 2, "table")?;
                rank(&o[0], 2, "out")?;
                if i[0].dtype != DType::I32 {
                    return bad("ids must be i32".into());
                }
                if o[0].shape != [i[0].shape[0], i[1].shape[1]] {
                    return bad(format!("out shape {:?} != [T,H]", o[0].shape));
                }
            }
            Op::RmsNorm => {
                rank(&i[0], 2, "x")?;
                rank(&i[1], 1, "w")?;
                if i[1].shape[0] != i[0].shape[1] || o[0].shape != i[0].shape {
                    return bad("shape mismatch".into());
                }
            }
            Op::RmsNormAdd => {
                rank(&i[0], 2, "x")?;
                if i[1].shape != i[0].shape
                    || i[2].shape != [i[0].shape[1]]
                    || o[0].shape != i[0].shape
                    || o[1].shape != i[0].shape
                {
                    return bad("shape mismatch".into());
                }
            }
            Op::Add => {
                if i[0].shape != i[1].shape || o[0].shape != i[0].shape {
                    return bad("shape mismatch".into());
                }
            }
            Op::QkNorm => {
                rank(&i[0], 3, "x")?;
                if i[1].shape != [i[0].shape[2]] || o[0].shape != i[0].shape {
                    return bad("shape mismatch".into());
                }
            }
            Op::Rope => {
                rank(&i[0], 3, "x")?;
                if o[0].shape != i[0].shape || i[0].shape[2] % 2 != 0 {
                    return bad("shape mismatch / odd head_dim".into());
                }
                if a.rotary_dim > i[0].shape[2] || a.rotary_dim % 2 != 0 {
                    return bad(format!("rotary_dim {} must be even and <= head_dim {}", a.rotary_dim, i[0].shape[2]));
                }
            }
            Op::RopeScaled => {
                rank(&i[0], 3, "x")?;
                rank(&i[1], 1, "inv_freq")?;
                if o[0].shape != i[0].shape || i[0].shape[2] % 2 != 0 {
                    return bad("shape mismatch / odd head_dim".into());
                }
                let rd = if a.rotary_dim == 0 { i[0].shape[2] } else { a.rotary_dim };
                if rd > i[0].shape[2] || rd % 2 != 0 {
                    return bad(format!("rotary_dim {} must be even and <= head_dim {}", a.rotary_dim, i[0].shape[2]));
                }
                if i[1].dtype != DType::F32 || i[1].shape[0] != rd / 2 {
                    return bad(format!("inv_freq must be f32 [{}], got {} {:?}", rd / 2, i[1].dtype, i[1].shape));
                }
                if !(a.rope_mscale > 0.0) || !a.rope_mscale.is_finite() {
                    return bad(format!("rope_mscale {} must be positive and finite", a.rope_mscale));
                }
            }
            Op::DynConv => {
                rank(&i[0], 2, "x")?;
                rank(&i[1], 2, "k")?;
                rank(&i[2], 2, "base")?;
                let (t, h) = (i[0].shape[0], i[0].shape[1]);
                let taps = i[2].shape[0];
                let ok = i[2].shape[1] == h
                    && taps > 0
                    && i[1].shape[0] == t
                    && i[1].shape[1] % taps == 0
                    && i[1].shape[1] / taps > 0
                    && h % (i[1].shape[1] / taps) == 0
                    && o[0].shape == i[0].shape
                    && i[1].dtype == i[0].dtype
                    && i[2].dtype == i[0].dtype
                    && o[0].dtype == i[0].dtype;
                if !ok {
                    return bad(format!("x {:?} k {:?} base {:?} out {:?}", i[0].shape, i[1].shape, i[2].shape, o[0].shape));
                }
                float(&i[0], "x")?;
            }
            Op::AddBias => {
                rank(&i[0], 2, "x")?;
                rank(&i[1], 1, "bias")?;
                if i[1].shape[0] != i[0].shape[1] || o[0].shape != i[0].shape || i[1].dtype != i[0].dtype {
                    return bad(format!("x {:?} bias {:?} out {:?}", i[0].shape, i[1].shape, o[0].shape));
                }
                float(&i[0], "x")?;
            }
            Op::Matmul | Op::Logits => {
                rank(&i[0], 2, "x")?;
                rank(&i[1], 2, "w")?;
                rank(&o[0], 2, "out")?;
                if i[0].shape[1] != i[1].shape[1] || o[0].shape != [i[0].shape[0], i[1].shape[0]] {
                    return bad(format!("x {:?} w {:?} out {:?}", i[0].shape, i[1].shape, o[0].shape));
                }
                float(&i[0], "x")?;
                float(&i[1], "w")?;
                if self.op == Op::Logits && o[0].dtype != DType::F32 {
                    return bad("logits output must be f32".into());
                }
            }
            Op::KvAppend => {
                rank(&i[0], 3, "new")?;
                rank(&o[0], 3, "cache")?;
                let (t, nkv, d) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
                if o[0].shape[0] != nkv || o[0].shape[2] != d || a.pos0 + t > o[0].shape[1] {
                    return bad(format!("append {t} rows at {} into cache {:?}", a.pos0, o[0].shape));
                }
            }
            Op::AttnPrefill | Op::AttnDecode => {
                rank(&i[0], 3, "q")?;
                rank(&i[1], 3, "k cache")?;
                rank(&i[2], 3, "v cache")?;
                let (t, n, d) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
                let (nkv, cap) = (i[1].shape[0], i[1].shape[1]);
                if i[1].shape != i[2].shape || i[1].shape[2] != d || n % nkv != 0 || o[0].shape != i[0].shape {
                    return bad(format!("q {:?} cache {:?}", i[0].shape, i[1].shape));
                }
                if a.pos0 + t > cap {
                    return bad(format!("pos0 {} + T {t} exceeds cache capacity {cap}", a.pos0));
                }
                if self.op == Op::AttnDecode && t != 1 && self.rows.is_none() {
                    return bad("decode requires T == 1".into());
                }
            }
            Op::SiluGlu | Op::GeluGlu => {
                if i[0].shape != i[1].shape || o[0].shape != i[0].shape {
                    return bad("shape mismatch".into());
                }
            }
            Op::QkNormRopeKv => {
                for (t, what) in [(&i[0], "q"), (&i[1], "k"), (&i[2], "v"), (&o[0], "q_out"), (&o[1], "k cache"), (&o[2], "v cache")] {
                    rank(t, 3, what)?;
                }
                let (t, n, d) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
                let (nkv, cap) = (o[1].shape[0], o[1].shape[1]);
                if i[1].shape != [t, nkv, d] || i[2].shape != i[1].shape || o[0].shape != i[0].shape {
                    return bad(format!("q {:?} k {:?} v {:?} q_out {:?}", i[0].shape, i[1].shape, i[2].shape, o[0].shape));
                }
                if o[2].shape != o[1].shape || o[1].shape[2] != d || o[2].dtype != o[1].dtype {
                    return bad(format!("caches {:?} {:?} must match each other and [Nkv,cap,{d}]", o[1].shape, o[2].shape));
                }
                if i[3].shape != [d] || i[4].shape != [d] {
                    return bad(format!("norm weights {:?} {:?} != [{d}]", i[3].shape, i[4].shape));
                }
                if nkv == 0 || n % nkv != 0 || d % 2 != 0 {
                    return bad(format!("N={n} Nkv={nkv} D={d}: N must be a multiple of Nkv and D even"));
                }
                if a.rotary_dim > d || a.rotary_dim % 2 != 0 {
                    return bad(format!("rotary_dim {} must be even and <= head_dim {d}", a.rotary_dim));
                }
                if a.pos0 + t > cap {
                    return bad(format!("append {t} rows at {} into cache {:?}", a.pos0, o[1].shape));
                }
                for (tn, what) in [(&i[0], "q"), (&i[1], "k"), (&i[2], "v"), (&i[3], "q_w"), (&i[4], "k_w"), (&o[0], "q_out")] {
                    float(tn, what)?;
                    if tn.dtype != i[0].dtype {
                        return bad(format!("{what} is {}, q is {}: one activation dtype", tn.dtype, i[0].dtype));
                    }
                }
            }
            Op::GateSplitQkNormRopeKv => {
                // GateSplit's checks on the gated row, then QkNormRopeKv's on the split-off query (same checks, same message)
                rank(&i[0], 3, "q+gate")?;
                rank(&o[1], 3, "gate")?;
                let (t, n, d2) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
                if d2 % 2 != 0 {
                    return bad(format!("last dim {d2} must be even (query + gate)"));
                }
                if o[1].shape != [t, n, d2 / 2] || o[1].dtype != i[0].dtype {
                    return bad(format!("gate {} {:?} must be {} [{t},{n},{}]", o[1].dtype, o[1].shape, i[0].dtype, d2 / 2));
                }
                self.gate_split_inner().check()?;
            }
            Op::Argmax => {
                rank(&i[0], 2, "logits")?;
                if o[0].dtype != DType::I32 || o[0].shape != [i[0].shape[0]] {
                    return bad("out must be i32 [R]".into());
                }
            }
            Op::Sample => {
                rank(&i[0], 2, "logits")?;
                if i[1].dtype != DType::F32 || i[1].shape != [i[0].shape[0]] {
                    return bad("u must be f32 [R]".into());
                }
                if o[0].dtype != DType::I32 || o[0].shape != [i[0].shape[0]] {
                    return bad("out must be i32 [R]".into());
                }
            }
            Op::MatmulFp8
            | Op::MatmulNvfp4
            | Op::MoeTopk
            | Op::MoeExperts
            | Op::MoeCombine
            | Op::MatmulPair
            | Op::MatmulFp8Pair
            | Op::MatmulNvfp4Glu
            | Op::GatedRmsnorm
            | Op::RmsNormAddMatmulPair
            | Op::GatedRmsnormMatmulFp8
            | Op::RmsNormAddMatmulFp8Pair
            | Op::SigmoidMulMatmulFp8
            | Op::SharedDownCombine
            | Op::MoeBlock
            | Op::MatmulMxfp4
            | Op::MatmulMxfp4Pair => crate::quant::ops::check(self)?,
            Op::MatmulExl3 => crate::quant::exl3::check(self)?,
            Op::GateSplit | Op::SigmoidMul | Op::GdnConv | Op::GdnDeltaRule | Op::GdnConvDelta | Op::ShortConv => crate::gdn::check(self)?,
        }
        Ok(())
    }
}

/// Shape facts handed to kernel predicates (`KernelImpl::when`).
#[derive(Clone, Debug, Default)]
pub struct ShapeInfo {
    /// Rows / tokens (M for matmul, T for attention, R for norms).
    pub m: usize,
    /// Output features (matmul N, vocab for argmax).
    pub n: usize,
    /// Reduction dim (matmul K, hidden for norms, head_dim for attention).
    pub k: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Number of keys visible to the last query (attention).
    pub kv_len: usize,
    pub act: Option<DType>,
    pub weight: Option<DType>,
    /// Dtype of the first output (`None` when the call has no outputs). Lets a registration pick its
    /// per-output-dtype tolerance (`KernelImpl::tol_for`).
    pub out: Option<DType>,
    /// MoE ops: number of routed experts / experts selected per token.
    pub experts: usize,
    pub top_k: usize,
    /// The call uses an extended attribute (`OpAttrs::window`, `rope_prop`, `gelu`): only kernels registered `.ext()` accept it,
    /// so a kernel that predates the attribute can never run it silently wrong.
    pub ext: bool,
    /// `OpAttrs::fp8_static_act` of the call (fp8 matmuls: W8A8 with a static input scale).
    pub fp8_static_act: bool,
    /// `OpAttrs::mxfp4_a8` of the call (MXFP4 matmuls: W4A8, dynamic per-token fp8 activations).
    pub mxfp4_a8: bool,
    /// `OpAttrs::mxfp4_w4a16` of the call (MXFP4 matmuls: the activations are not quantised, the call's `act` input is 0). Read by kernel
    /// selection: a kernel that reads only bf16 activations (CUDA's tensor-core `matmul_mxfp4_mma`) is registered for these calls alone,
    /// because the device-side `act` value is invisible at plan time.
    pub mxfp4_w4a16: bool,
    /// An input of the call is in a non-row-major layout (`backend::Layout`: HIP fragment-order fp8 weights). Only kernels registered
    /// `.frag_ok()` accept the call, so a kernel that reads row-major bytes can never be selected for it.
    pub frag: bool,
}

impl ShapeInfo {
    pub fn from_call(call: &OpCall) -> ShapeInfo {
        let mut s = Self::from_call_untagged(call);
        // an input in a non-row-major layout (backend::Layout, HIP fragment-order fp8 weights): only kernels registered `.frag_ok()` accept it
        s.frag = call.inputs.iter().any(|t| t.layout() != crate::backend::Layout::RowMajor);
        s
    }

    fn from_call_untagged(call: &OpCall) -> ShapeInfo {
        let i = &call.inputs;
        let o = &call.outputs;
        let mut s = ShapeInfo {
            act: i.first().map(|t| t.dtype),
            out: o.first().map(|t| t.dtype),
            ext: call.attrs.window > 0
                || call.attrs.rope_prop
                || call.attrs.gelu
                || (matches!(call.op, Op::Add | Op::Embedding) && call.attrs.scale != 1.0),
            fp8_static_act: call.attrs.fp8_static_act,
            mxfp4_a8: call.attrs.mxfp4_a8,
            mxfp4_w4a16: call.attrs.mxfp4_w4a16,
            ..Default::default()
        };
        match call.op {
            Op::Embedding => {
                s.m = i[0].shape[0];
                s.n = i[1].shape[0];
                s.k = i[1].shape[1];
                s.act = Some(i[1].dtype);
            }
            Op::RmsNorm | Op::RmsNormAdd | Op::Add | Op::SiluGlu | Op::GeluGlu | Op::AddBias => {
                s.m = i[0].shape[0];
                s.k = i[0].shape[1];
                // the norm weight `w` (x, w | x, res, w): a backend whose norm kernel reads one weight dtype declares it with
                // `.weights(..)`, and a checkpoint tensor of another dtype then fails at plan time, naming the tensor
                s.weight = match call.op {
                    Op::RmsNorm => i.get(1),
                    Op::RmsNormAdd => i.get(2),
                    _ => None,
                }
                .map(|w| w.dtype);
            }
            Op::DynConv => {
                s.m = i[0].shape[0];
                s.k = i[0].shape[1];
                s.n = i[1].shape[1];
            }
            Op::QkNorm | Op::Rope | Op::RopeScaled | Op::KvAppend => {
                s.m = i[0].shape[0];
                s.n_heads = i[0].shape[1];
                s.head_dim = i[0].shape[2];
                s.k = s.head_dim;                if call.op == Op::QkNorm {
                    s.weight = i.get(1).map(|w| w.dtype); // the norm weight, as for RmsNorm
                }
            }
            Op::Matmul | Op::Logits => {
                s.m = i[0].shape[0];
                s.k = i[0].shape[1];
                s.n = i[1].shape[0];
                s.weight = Some(i[1].dtype);
            }
            Op::QkNormRopeKv => {
                s.m = i[0].shape[0];
                s.n_heads = i[0].shape[1];
                s.head_dim = i[0].shape[2];
                s.k = s.head_dim;
                s.n_kv_heads = i[1].shape[1];
                s.weight = Some(o[1].dtype); // kv cache dtype
            }
            // QkNormRopeKv's facts for the split-off query (head_dim is half the gated row)
            Op::GateSplitQkNormRopeKv => {
                s.m = i[0].shape[0];
                s.n_heads = i[0].shape[1];
                s.head_dim = i[0].shape[2] / 2;
                s.k = s.head_dim;
                s.n_kv_heads = i[1].shape[1];
                s.weight = Some(o[2].dtype); // kv cache dtype
            }
            Op::AttnPrefill | Op::AttnDecode => {
                s.m = i[0].shape[0];
                s.n_heads = i[0].shape[1];
                s.head_dim = i[0].shape[2];
                s.k = s.head_dim;
                s.n_kv_heads = i[1].shape[0];
                s.kv_len = call.attrs.pos0 + s.m;
                s.weight = Some(i[1].dtype); // kv dtype
            }
            Op::Argmax | Op::Sample => {
                s.m = i[0].shape[0];
                s.n = i[0].shape[1];
            }
            Op::MatmulFp8
            | Op::MatmulNvfp4
            | Op::MoeTopk
            | Op::MoeExperts
            | Op::MoeCombine
            | Op::MatmulPair
            | Op::MatmulFp8Pair
            | Op::MatmulNvfp4Glu
            | Op::GatedRmsnorm
            | Op::RmsNormAddMatmulPair
            | Op::GatedRmsnormMatmulFp8
            | Op::RmsNormAddMatmulFp8Pair
            | Op::SigmoidMulMatmulFp8
            | Op::SharedDownCombine
            | Op::MoeBlock
            | Op::MatmulMxfp4
            | Op::MatmulMxfp4Pair => crate::quant::ops::shape(call, &mut s),
            Op::MatmulExl3 => crate::quant::exl3::shape(call, &mut s),
            Op::GateSplit | Op::SigmoidMul | Op::GdnConv | Op::GdnDeltaRule | Op::GdnConvDelta | Op::ShortConv => crate::gdn::shape(call, &mut s),
        }
        let _ = o;
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Buffer, BufferImpl};
    use std::any::Any;

    struct Dummy(usize);
    impl BufferImpl for Dummy {
        fn bytes(&self) -> usize {
            self.0
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }
    fn t(dtype: DType, shape: &[usize]) -> TensorRef {
        let n = shape.iter().product::<usize>() * dtype.size_bytes();
        TensorRef::new(Buffer::new(Dummy(n)), dtype, shape)
    }

    #[test]
    fn norm_calls_report_the_norm_weight_dtype_so_a_kernel_can_declare_it() {
        let call = |op: Op, inputs: Vec<TensorRef>, out: TensorRef| OpCall::new(op, inputs, vec![out], OpAttrs::default());
        let x = t(DType::Bf16, &[4, 128]);
        let rms = call(Op::RmsNorm, vec![x.clone(), t(DType::F32, &[128])], x.clone());
        assert_eq!(ShapeInfo::from_call(&rms).weight, Some(DType::F32));
        let add = call(Op::RmsNormAdd, vec![x.clone(), x.clone(), t(DType::Bf16, &[128])], x.clone());
        assert_eq!(ShapeInfo::from_call(&add).weight, Some(DType::Bf16));
        let qk = call(Op::QkNorm, vec![t(DType::Bf16, &[4, 8, 64]), t(DType::F32, &[64])], t(DType::Bf16, &[4, 8, 64]));
        assert_eq!(ShapeInfo::from_call(&qk).weight, Some(DType::F32));
        // ops without a norm weight stay as they were
        let plain = call(Op::Add, vec![x.clone(), x.clone()], x.clone());
        assert_eq!(ShapeInfo::from_call(&plain).weight, None);
        // a kernel that declares bf16 weights takes the bf16 norm and refuses the f32 one (a backend that declares nothing takes both)
        let k = crate::registry::KernelImpl::new(Op::RmsNorm, "rmsnorm_bf16", "metal").acts([DType::Bf16]).weights([DType::Bf16]);
        let open = crate::registry::KernelImpl::new(Op::RmsNorm, "rmsnorm_any", "cuda").acts([DType::Bf16]);
        let bf = call(Op::RmsNorm, vec![x.clone(), t(DType::Bf16, &[128])], x.clone());
        assert!(k.accepts(&ShapeInfo::from_call(&bf)) && !k.accepts(&ShapeInfo::from_call(&rms)));
        assert!(open.accepts(&ShapeInfo::from_call(&rms)));
    }

    #[test]
    fn op_names_roundtrip() {
        for op in Op::ALL {
            assert_eq!(Op::parse(op.as_str()), Some(op));
        }
    }

    #[test]
    fn check_matmul() {
        let ok = OpCall::new(
            Op::Matmul,
            vec![t(DType::Bf16, &[3, 8]), t(DType::Bf16, &[5, 8])],
            vec![t(DType::Bf16, &[3, 5])],
            OpAttrs::default(),
        );
        ok.check().unwrap();
        let bad = OpCall::new(
            Op::Matmul,
            vec![t(DType::Bf16, &[3, 8]), t(DType::Bf16, &[5, 7])],
            vec![t(DType::Bf16, &[3, 5])],
            OpAttrs::default(),
        );
        assert!(bad.check().is_err());
        let s = ShapeInfo::from_call(&ok);
        assert_eq!((s.m, s.n, s.k), (3, 5, 8));
        // the first output's dtype is recorded for per-output-dtype tolerances
        assert_eq!(s.out, Some(DType::Bf16));
        assert_eq!(ShapeInfo::default().out, None);
        let f32_out = OpCall::new(
            Op::Logits,
            vec![t(DType::Bf16, &[3, 8]), t(DType::Bf16, &[5, 8])],
            vec![t(DType::F32, &[3, 5])],
            OpAttrs::default(),
        );
        assert_eq!(ShapeInfo::from_call(&f32_out).out, Some(DType::F32));
    }

    /// The phase-1 folds: shapes and dtypes are checked like the ops they replace, the shape facts follow the first GEMV,
    /// and `with_rows` re-shapes exactly the row-leading tensors (weights, scales and the norm weight keep theirs).
    #[test]
    fn check_phase1_folds() {
        let b = |s: &[usize]| t(DType::Bf16, s);
        let f8 = |s: &[usize]| t(DType::Fp8E4m3, s);
        let sc = || t(DType::F32, &[2]);
        let norm_kv = OpCall::new(
            Op::RmsNormAddMatmulFp8Pair,
            vec![b(&[1, 64]), b(&[1, 64]), b(&[64]), f8(&[16, 64]), sc(), f8(&[8, 64]), sc()],
            vec![b(&[1, 64]), b(&[1, 64]), b(&[1, 16]), b(&[1, 8])],
            OpAttrs::default(),
        );
        norm_kv.check().unwrap();
        let s = ShapeInfo::from_call(&norm_kv);
        assert_eq!((s.m, s.n, s.k, s.weight), (1, 16, 64, Some(DType::Fp8E4m3)));
        let r3 = norm_kv.with_rows(3).unwrap();
        let shapes: Vec<_> = r3.inputs.iter().chain(&r3.outputs).map(|t| t.shape.clone()).collect();
        assert_eq!(
            shapes,
            [vec![3, 64], vec![3, 64], vec![64], vec![16, 64], vec![2], vec![8, 64], vec![2], vec![3, 64], vec![3, 64], vec![3, 16], vec![3, 8]]
        );
        let mut bad = norm_kv.clone();
        bad.inputs[5] = b(&[8, 64]); // a bf16 weight
        assert!(bad.check().is_err());

        let gate_o = OpCall::new(Op::SigmoidMulMatmulFp8, vec![b(&[1, 64]), b(&[1, 64]), f8(&[32, 64]), sc()], vec![b(&[1, 32])], OpAttrs::default());
        gate_o.check().unwrap();
        assert_eq!(ShapeInfo::from_call(&gate_o).n, 32);
        let mut bad = gate_o.clone();
        bad.inputs[1] = b(&[1, 32]); // gate not the shape of a
        assert!(bad.check().is_err());
        assert_eq!(gate_o.with_rows(2).unwrap().outputs[0].shape, [2, 32]);

        let down = OpCall::new(
            Op::SharedDownCombine,
            vec![b(&[2, 32]), t(DType::U8, &[64, 16]), t(DType::Fp8E4m3, &[64, 2]), t(DType::F32, &[1]), b(&[2, 64]), b(&[2, 1])],
            vec![b(&[2, 64])],
            OpAttrs::default(),
        );
        down.check().unwrap();
        let s = ShapeInfo::from_call(&down);
        assert_eq!((s.m, s.n, s.k, s.weight), (2, 64, 32, Some(DType::U8)));
        let r1 = down.with_rows(1).unwrap();
        assert_eq!((r1.inputs[0].shape.clone(), r1.inputs[1].shape.clone(), r1.inputs[5].shape.clone()), (vec![1, 32], vec![64, 16], vec![1, 1]));
        let mut bad = down.clone();
        bad.inputs[5] = b(&[2, 2]); // the logit is one value per row
        assert!(bad.check().is_err());
        // ops outside the phase-1 folds are never re-shaped
        assert!(OpCall::new(Op::Matmul, vec![b(&[1, 8]), b(&[4, 8])], vec![b(&[1, 4])], OpAttrs::default()).with_rows(2).is_none());
    }

    /// The phase-2 folds: each is checked as the chain it replaces (with the same messages), takes a device position only
    /// where that chain does (the prologue), and reports the shape facts its kernels select on.
    #[test]
    fn check_phase2_folds() {
        let b = |s: &[usize]| t(DType::Bf16, s);
        let f = |s: &[usize]| t(DType::F32, s);
        let attrs = OpAttrs { pos0: 3, kv_cap: 16, rotary_dim: 16, ..OpAttrs::default() };
        let (n, nkv, d) = (4, 2, 64);
        let p4a = OpCall::new(
            Op::GateSplitQkNormRopeKv,
            vec![b(&[2, n, 2 * d]), b(&[2, nkv, d]), b(&[2, nkv, d]), b(&[d]), b(&[d])],
            vec![b(&[2, n, d]), b(&[2, n, d]), b(&[nkv, 16, d]), b(&[nkv, 16, d])],
            attrs.clone(),
        );
        p4a.check().unwrap();
        let inner = p4a.gate_split_inner();
        assert_eq!((inner.op, inner.inputs[0].shape.clone(), inner.outputs.len()), (Op::QkNormRopeKv, vec![2, n, d], 3));
        let s = ShapeInfo::from_call(&p4a);
        assert_eq!((s.m, s.n_heads, s.n_kv_heads, s.head_dim, s.weight), (2, n, nkv, d, Some(DType::Bf16)));
        // a device position (graph replay) is accepted, as for QkNormRopeKv; without it the decode graph would never fuse
        let mut with_pos = p4a.clone();
        with_pos.pos = Some(t(DType::I32, &[1]));
        with_pos.check().unwrap();
        assert!(with_pos.gate_split_inner().pos.is_some());
        let mut bad = p4a.clone();
        bad.outputs[1] = b(&[2, n, d + 2]); // the gate is not the split-off half
        assert!(bad.check().is_err());
        let mut bad = p4a.clone();
        bad.inputs[0] = b(&[2, n, 2 * d + 1]); // odd gated row
        assert!(bad.check().is_err());
        let mut bad = p4a.clone();
        bad.attrs.pos0 = 15; // the 2 rows run past the cache (QkNormRopeKv's check)
        assert!(bad.check().is_err());

        // GdnConvDelta: Hk = 1, Hv = 2, dk = 32, dv = 16 -> C = 96; K = 4
        let (tt, c, hv, dv, dk, k) = (3, 96, 2, 16, 32, 4);
        let p3 = OpCall::new(
            Op::GdnConvDelta,
            vec![b(&[tt, c]), b(&[c, k]), b(&[tt, hv]), b(&[tt, hv]), b(&[hv]), b(&[hv])],
            vec![b(&[tt, hv, dv]), f(&[hv, dv, dk]), f(&[c, k - 1])],
            OpAttrs { eps: 1e-6, ..OpAttrs::default() },
        );
        p3.check().unwrap();
        let s = ShapeInfo::from_call(&p3);
        assert_eq!((s.m, s.n, s.k, s.head_dim, s.n_heads, s.n_kv_heads), (tt, dv, k, dk, hv, 1));
        let mut bad = p3.clone();
        bad.outputs[2] = f(&[c, k]); // conv state [C, K-1]
        assert!(bad.check().is_err());
        let mut bad = p3.clone();
        bad.outputs[2] = b(&[c, k - 1]); // conv state must be f32
        assert!(bad.check().is_err());
        let mut bad = p3.clone();
        bad.outputs[1] = f(&[hv, dv, dk + 1]); // delta state does not split the qkv row
        assert!(bad.check().is_err());
        let mut with_pos = p3.clone();
        with_pos.pos = Some(t(DType::I32, &[1]));
        assert!(with_pos.check().is_err(), "the GDN ops take no device position");
        // neither is a row-local fold of phase 1
        assert!(p4a.with_rows(2).is_none() && p3.with_rows(2).is_none());
    }

    /// P2' `MoeBlock`: arity, the composed checks, the shape facts (IS rides in `head_dim`) and the row-local probe.
    #[test]
    fn check_moeblock() {
        let b = |s: &[usize]| t(DType::Bf16, s);
        let (u8_, f8, f32_) = (|s: &[usize]| t(DType::U8, s), |s: &[usize]| t(DType::Fp8E4m3, s), |s: &[usize]| t(DType::F32, s));
        // T = 2, E = 8, K = 2, H = 64, I = 32, IS = 48
        let (tt, e, k, h, i, is) = (2usize, 8usize, 2usize, 64usize, 32usize, 48usize);
        let attrs = OpAttrs { top_k: k, norm_topk: true, ..OpAttrs::default() };
        let call = OpCall::new(
            Op::MoeBlock,
            vec![
                b(&[tt, h]),
                b(&[tt, e]),
                b(&[tt, 1]),
                u8_(&[e, 2 * i, h / 2]),
                f8(&[e, 2 * i, h / 16]),
                u8_(&[e, h, i / 2]),
                f8(&[e, h, i / 16]),
                f32_(&[e, 3]),
                u8_(&[is, h / 2]),
                f8(&[is, h / 16]),
                f32_(&[1]),
                u8_(&[is, h / 2]),
                f8(&[is, h / 16]),
                f32_(&[1]),
                u8_(&[h, is / 2]),
                f8(&[h, is / 16]),
                f32_(&[1]),
            ],
            vec![b(&[tt, h])],
            attrs,
        );
        assert_eq!(Op::MoeBlock.arity(), (17, 1));
        assert_eq!(Op::MoeBlock.as_str(), "moe_block");
        assert_eq!(Op::ALL[Op::ALL.len() - 9], Op::MoeBlock, "appended at the end");
        assert_eq!(Op::ALL[Op::ALL.len() - 8], Op::MatmulMxfp4, "appended after MoeBlock");
        assert_eq!(Op::ALL[Op::ALL.len() - 7], Op::MatmulExl3, "appended after MatmulMxfp4");
        assert_eq!(Op::ALL[Op::ALL.len() - 6], Op::MatmulMxfp4Pair, "appended after MatmulExl3");
        assert_eq!(Op::ALL[Op::ALL.len() - 5], Op::RopeScaled, "appended after MatmulMxfp4Pair");
        assert_eq!(Op::ALL[Op::ALL.len() - 4], Op::AddBias, "appended after RopeScaled");
        assert_eq!(Op::ALL[Op::ALL.len() - 3], Op::GeluGlu, "appended after AddBias");
        assert_eq!(Op::ALL[Op::ALL.len() - 2], Op::DynConv, "appended after GeluGlu");
        assert_eq!(Op::ALL.last(), Some(&Op::ShortConv), "appended after DynConv");
        call.check().unwrap();
        let s = ShapeInfo::from_call(&call);
        assert_eq!((s.m, s.k, s.n, s.experts, s.top_k, s.head_dim), (tt, h, i, e, k, is));
        assert_eq!((s.act, s.weight, s.out), (Some(DType::Bf16), Some(DType::U8), Some(DType::Bf16)));
        assert_eq!(crate::quant::ops::moe_block_scratch(&s), crate::quant::ops::moe_block_scratch_bytes(tt, k, i, is));
        // h region: 2*2*32*2 = 256 bytes; sa region: 2*48*2 = 192 -> 256
        assert_eq!(crate::quant::ops::moe_block_scratch(&s), 512);
        let r1 = call.with_rows(1).unwrap();
        assert_eq!((r1.inputs[0].shape.clone(), r1.inputs[1].shape.clone(), r1.inputs[2].shape.clone()), (vec![1, h], vec![1, e], vec![1, 1]));
        assert_eq!((r1.inputs[3].shape.clone(), r1.outputs[0].shape.clone()), (vec![e, 2 * i, h / 2], vec![1, h]));
        r1.check().unwrap();
        // every tensor is checked
        let breaks: Vec<(usize, TensorRef)> = vec![
            (1, b(&[tt, e + 1])),
            (2, b(&[tt, 2])),
            (3, u8_(&[e, 2 * i, h / 2 + 1])),
            (4, f8(&[e, 2 * i, h / 16 + 1])),
            (5, u8_(&[e, h, i / 2 + 1])),
            (6, t(DType::F32, &[e, h, i / 16])),
            (7, f32_(&[e, 2])),
            (8, u8_(&[is, h / 2 + 1])),
            (9, f8(&[is + 16, h / 16])),
            (10, f32_(&[2])),
            (11, u8_(&[is + 16, h / 2])),
            (12, f8(&[is, h / 16 + 1])),
            (13, b(&[1])),
            (14, u8_(&[h, is / 2 + 1])),
            (15, f8(&[h + 1, is / 16])),
            (16, f32_(&[1, 1])),
        ];
        for (idx, bad_t) in breaks {
            let mut bad = call.clone();
            bad.inputs[idx] = bad_t;
            assert!(bad.check().is_err(), "input {idx} not checked");
        }
        let mut bad = call.clone();
        bad.outputs[0] = b(&[tt, h + 16]);
        assert!(bad.check().is_err());
        for top_k in [0, e + 1] {
            let mut bad = call.clone();
            bad.attrs.top_k = top_k;
            assert!(bad.check().is_err(), "top_k {top_k}");
        }
        // the shared intermediate must be a multiple of 16
        let mut bad = call.clone();
        bad.inputs[8] = u8_(&[40, h / 2]);
        bad.inputs[9] = f8(&[40, h / 16]);
        bad.inputs[11] = u8_(&[40, h / 2]);
        bad.inputs[12] = f8(&[40, h / 16]);
        bad.inputs[14] = u8_(&[h, 20]);
        bad.inputs[15] = f8(&[h, 2]);
        assert!(bad.check().is_err());
    }

    #[test]
    fn check_qk_norm_rope_kv() {
        let f = |shape: &[usize]| t(DType::Bf16, shape);
        let attrs = OpAttrs { pos0: 3, rotary_dim: 8, kv_cap: 16, n_kv_heads: 2, head_dim: 16, ..OpAttrs::default() };
        let mk = |a: OpAttrs, kcache: TensorRef| {
            OpCall::new(
                Op::QkNormRopeKv,
                vec![f(&[2, 4, 16]), f(&[2, 2, 16]), f(&[2, 2, 16]), f(&[16]), f(&[16])],
                vec![f(&[2, 4, 16]), kcache, f(&[2, 16, 16])],
                a,
            )
        };
        let ok = mk(attrs.clone(), f(&[2, 16, 16]));
        ok.check().unwrap();
        let s = ShapeInfo::from_call(&ok);
        assert_eq!((s.m, s.n_heads, s.n_kv_heads, s.head_dim, s.weight), (2, 4, 2, 16, Some(DType::Bf16)));
        // a device-side position is accepted, only this op among the norm / matmul ops
        let mut dev = mk(attrs.clone(), f(&[2, 16, 16]));
        dev.pos = Some(t(DType::I32, &[1]));
        dev.check().unwrap();
        // 3 + 2 rows fit a 16-row cache; 15 + 2 do not
        assert!(mk(OpAttrs { pos0: 15, ..attrs.clone() }, f(&[2, 16, 16])).check().is_err());
        // caches must agree; the cache width must be the head dim; odd rotary dim; a mixed activation dtype
        assert!(mk(attrs.clone(), f(&[2, 8, 16])).check().is_err());
        assert!(mk(OpAttrs { rotary_dim: 7, ..attrs.clone() }, f(&[2, 16, 16])).check().is_err());
        let mut mixed = mk(attrs, f(&[2, 16, 16]));
        mixed.inputs[3] = t(DType::F32, &[16]);
        assert!(mixed.check().is_err());
    }
}
