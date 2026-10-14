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
//! | `Matmul` | x `[M,K]`, w `[N,K]` | `[M,N]` (= x @ w^T) | |
//! | `Logits` | x `[M,K]`, w `[V,K]` | `[M,V]` f32 | same math as `Matmul`, f32 output |
//! | `KvAppend` | new `[T,Nkv,D]` | cache `[Nkv,cap,D]` (written in place) | pos0 |
//! | `AttnPrefill` | q `[T,N,D]`, kc, vc `[Nkv,cap,D]` | `[T,N,D]` | pos0, scale, n_heads, n_kv_heads, head_dim, kv_cap |
//! | `AttnDecode` | as prefill with T == 1 | `[1,N,D]` | same |
//! | `SiluGlu` | gate `[T,I]`, up `[T,I]` | `[T,I]` | |
//! | `Argmax` | logits `[R,V]` | `[R]` i32 | |
//! | `Sample` | logits `[R,V]`, u `[R]` f32 in [0,1) | `[R]` i32 | temperature, top_k, top_p |
//! | `MatmulFp8` | x `[M,K]`, w `[N,K]` fp8_e4m3, scales `[2]` f32 = (weight_scale, input_scale) | `[M,N]` (bf16/f16/f32) | see `quant::fp8` |
//! | `MatmulNvfp4` | x `[M,K]`, w `[N,K/2]` u8 (e2m1 pairs), block scales `[N,K/16]` fp8_e4m3, global `[1]` f32 | `[M,N]` (bf16/f16/f32) | see `quant::nvfp4` |
//! | `MoeTopk` | router logits `[T,E]` | weights `[T,K]` f32, expert ids `[T,K]` i32 | top_k, norm_topk |
//! | `MoeExperts` | x `[T,H]`, ids `[T,K]`, weights `[T,K]`, gate_up w `[E,2I,H/2]` u8, gate_up scales `[E,2I,H/16]` fp8, down w `[E,H,I/2]` u8, down scales `[E,H,I/16]` fp8, global `[E,3]` f32 | `[T,H]` | see `quant::moe` |
//! | `MoeCombine` | routed `[T,H]`, shared `[T,H]`, shared-gate logit `[T,1]` | `[T,H]` | |
//! | `GateSplit` | x `[T,N,2D]` (per head: `D` values then `D` gate values) | q `[T,N,D]`, gate `[T,N,D]` | |
//! | `SigmoidMul` | a `[R,X]`, gate `[R,X]` | `[R,X]` (= a * sigmoid(gate)) | |
//! | `GdnConv` | x `[T,C]`, taps `[C,K]` | y `[T,C]` (= silu(causal conv)), conv state `[C,K-1]` f32 (**in place**) | |
//! | `GdnDeltaRule` | qkv `[T,C]` (`C = 2*Hk*dk + Hv*dv`, rows `[q|k|v]`), a `[T,Hv]`, b `[T,Hv]`, A_log `[Hv]`, dt_bias `[Hv]` | o `[T,Hv,dv]`, state `[Hv,dv,dk]` f32 (**in place**) | eps (l2norm) |
//!
//! | `MatmulPair` | x `[M,K]`, w0 `[N0,K]`, w1 `[N1,K]` (dense) | `[M,N0]`, `[M,N1]` (= two `Matmul`s of one x) | fused decode GEMV pair, see `quant::fused` |
//! | `MatmulFp8Pair` | x `[M,K]`, (w0 `[N0,K]`, scales0 `[2]`), (w1 `[N1,K]`, scales1 `[2]`) fp8 | `[M,N0]`, `[M,N1]` (= two `MatmulFp8`s) | fused, see `quant::fused` |
//! | `MatmulNvfp4Glu` | x `[M,K]`, gate (w, block scales, global), up (w, block scales, global), nvfp4 `[N,K]` each | `[M,N]` (= `SiluGlu(MatmulNvfp4(gate), MatmulNvfp4(up))`) | fused, see `quant::fused` |
//! | `GatedRmsnorm` | x `[R,D]`, z `[R,D]`, w `[D]` | `[R,D]` (= `SiluGlu(z, RmsNorm(x, w))`) | eps, w_offset; fused, see `quant::fused` |
//! | `QkNormRopeKv` | q `[T,N,D]`, k `[T,Nkv,D]`, v `[T,Nkv,D]`, q_w `[D]`, k_w `[D]` (all one float dtype) | q_out `[T,N,D]`, k cache, v cache `[Nkv,cap,D]` (**in place**) | eps, w_offset, theta, rotary_dim, pos0 (or `OpCall::pos`); fused attention prologue, see below |
//!
//! The quantised-weight ops are specified in `quant/` (the reference is `intarsia-backend-cpu`'s `ref_*`). The five
//! fused ops above are optional: a backend registers kernels for them only when it has a fast path, and the model
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
}

impl Op {
    pub const ALL: [Op; 28] = [
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
    /// Norm ops: the scale applied is `w + w_offset` (1.0 for zero-centred norms such as Qwen3.5's), added in f32.
    pub w_offset: f32,
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
            w_offset: 0.0,
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
}

impl OpCall {
    pub fn new(op: Op, inputs: Vec<TensorRef>, outputs: Vec<TensorRef>, attrs: OpAttrs) -> OpCall {
        OpCall { op, inputs, outputs, attrs, scratch: None, pos: None }
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
            if !matches!(self.op, Op::Rope | Op::KvAppend | Op::AttnDecode | Op::QkNormRopeKv) {
                return Err(Error::invalid(format!("{}: takes no device-side position", self.op)));
            }
            if p.dtype != DType::I32 || p.numel() != 1 {
                return Err(Error::invalid(format!("{}: device position must be i32 [1], got {} {:?}", self.op, p.dtype, p.shape)));
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
                if self.op == Op::AttnDecode && t != 1 {
                    return bad("decode requires T == 1".into());
                }
            }
            Op::SiluGlu => {
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
            | Op::GatedRmsnorm => crate::quant::ops::check(self)?,
            Op::GateSplit | Op::SigmoidMul | Op::GdnConv | Op::GdnDeltaRule => crate::gdn::check(self)?,
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
}

impl ShapeInfo {
    pub fn from_call(call: &OpCall) -> ShapeInfo {
        let i = &call.inputs;
        let o = &call.outputs;
        let mut s = ShapeInfo {
            act: i.first().map(|t| t.dtype),
            out: o.first().map(|t| t.dtype),
            ..Default::default()
        };
        match call.op {
            Op::Embedding => {
                s.m = i[0].shape[0];
                s.n = i[1].shape[0];
                s.k = i[1].shape[1];
                s.act = Some(i[1].dtype);
            }
            Op::RmsNorm | Op::RmsNormAdd | Op::Add | Op::SiluGlu => {
                s.m = i[0].shape[0];
                s.k = i[0].shape[1];
            }
            Op::QkNorm | Op::Rope | Op::KvAppend => {
                s.m = i[0].shape[0];
                s.n_heads = i[0].shape[1];
                s.head_dim = i[0].shape[2];
                s.k = s.head_dim;
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
            | Op::GatedRmsnorm => crate::quant::ops::shape(call, &mut s),
            Op::GateSplit | Op::SigmoidMul | Op::GdnConv | Op::GdnDeltaRule => crate::gdn::shape(call, &mut s),
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
