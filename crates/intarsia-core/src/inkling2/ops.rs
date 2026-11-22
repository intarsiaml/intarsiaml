//! The Inkling-2-Mini ops (`Op::AttnResMix` .. `Op::ConvMix`, discriminants 67..=74, appended once): attributes, call checks
//! and shape facts. This module doc is the contract a kernel is written against (with docs/plans/2026-11-21-inkling2-port.md
//! section a for the model-level formulas); the f64 reference is `intarsia-backend-cpu`'s `ink_ref`.
//!
//! # Conventions
//!
//! * `A` is the call's activation dtype (bf16 or f32; the dtype of the first activation input, every activation tensor of
//!   the call has it). `rA(x)` rounds to `A` (identity for f32). Everything inside an op is computed in f32 (the reference:
//!   f64) and rounded only where `rA` is written. Weights are any float dtype (the checkpoint's bf16); f32 inputs named so.
//! * An optional tensor is passed as `[0]` (any dtype) when absent.
//! * A chunk is `T` rows at absolute positions `p = pos0 + t` (`attrs.pos0`, or the device position `OpCall::pos` `i32 [1]`,
//!   which then replaces `pos0`). Every op is row-local except through the state it reads (rings, caches), so row `t` of a
//!   `T`-row call equals, bit for bit, the 1-row call at position `pos0 + t` made after rows `0..t` were committed: this is
//!   the rows-exact rule (G0) and every kernel must keep it (fixed reduction order per row, independent of `T`).
//! * `SRC_EPS` = 1e-6 is the fixed epsilon of the source / pair RMS inside the depth mix and the controller, and of the L2
//!   clip denominators (vLLM `ops/attn_res.py:252`, `ops/norm.py`); `attrs.eps` is the model's `rms_norm_eps` (the norms).
//! * `inv(x) = 1 / sqrt(mean(x^2) + eps)` over the last dimension; `clip(x, th) = x * min(th / (|x|_2 + SRC_EPS), 1)` per head row.
//!
//! # Rings (state that rolls back by length)
//!
//! A ring `[Rr, C]` holds in row `p % Rr` the value of position `p` (SWA K/V: `[Hk, Rr, Dh]`, row index on the middle axis).
//! A call **reads the ring as it was before the call** (positions `< pos0`; positions `< 0` read as zero, whatever the row
//! holds), takes positions `>= pos0` from its own inputs, and **then** writes the rows of positions
//! `max(pos0, pos0 + T - Rr) .. pos0 + T` (only the last `Rr` positions of a long chunk). A kernel that reads and writes the
//! same rows from different threads must order that itself (stage the history, or a second pass); for `T <= Rr - (history
//! length)` the rows written and the rows read are disjoint. No linearizing copy of the ring is part of any op. The model
//! sizes rings as `history + margin` rows (conv: `K-1 + 16`, SWA KV: `window + 16`), so a cut of up to 16 tokens
//! (speculative rollback) finds every row it needs. Values written to a ring are the call's inputs as given (already in `A`,
//! the ring's dtype must be `A`), so a later 1-row call reads exactly what an in-call row read (rows-exact, review F9).
//!
//! # The ops
//!
//! | Op | inputs | outputs | attrs |
//! |---|---|---|---|
//! | `AttnResMix` (67) | src `[N,R,D]` (N >= 0), last `[R,D]`, addend `[R,D]` or `[0]`, queries `[Q,D]` f32, norm `[Q,D]` or `[0]` | out `[Q,R,D]`: `A` with norm, f32 without | eps |
//! | `LoopBlend` (68) | dest `[Sd,R,D]` f32 (Sd = 1 broadcasts, else S), msg `[S,R,D]` f32, down `[S,G,2D]`, gain `[S,D,G]`, forget `[S,D,G]`, norm `[S,D]` | out `[S,R,D]` (its dtype is `A`) | eps |
//! | `SconvRes` (69) | x `[T,C]`, taps `[K,C]`, res `[T,C]` or `[0]`, addend `[T,C]` or `[0]` | y `[T,C]`, ring `[Rr,C]` (**in place**), sum `[T,C]` or `[0]` | pos0 |
//! | `InkKvPrologue` (70) | pq `[T,Nq]`, pk `[T,Ckv(+Hk*K)]`, pv (as pk), k taps `[K,Ckv]`, v taps `[K,Ckv]`, q scale `[H,Dh]`, r scale `[H,dr]`, rel proj `[dr,E]` | q `[T,H,Dh]`, rel `[T,H,E]`, k `[T,Hk,Dh]`, v `[T,Hk,Dh]`, k ring `[Rr,Ckv]`, v ring `[Rr,Ckv]` (rings **in place**) | pos0, n_heads, n_kv_heads, head_dim, eps, ink.{qk_clip, rel_clip, tau_floor, tau_alpha, dyn_conv, d_rel} |
//! | `RelSinkAttn` (71) | q `[T,H,Dh]`, k `[T,Hk,Dh]`, v `[T,Hk,Dh]`, k cache `[Hk,Cr,Dh]`, v cache `[Hk,Cr,Dh]`, rel `[T,H,E]`, pq `[T,Nq]` | o `[T,H*Dh]`, k cache, v cache (**in place**) | pos0, scale, n_heads, n_kv_heads, head_dim, window, ink.{kv_ring, d_rel} |
//! | `MoeExpertsMxfp8` (72) | x `[T,H]`, ids `[T,K]` i32, weights `[T,K]` f32, w13 `[E,2I,H]` e4m3, w13 scales `[E,2I,H/32]` u8 (E8M0), w2 `[E,H,I]` e4m3, w2 scales `[E,H,I/32]` u8 | `[T,H]` | gelu, ink.{situ, gate_cap, up_cap, act_quant} |
//! | `InkInProj` (73) | x `[S,T,D]` (S = 1 or 3), w `[Nq+Nk+Nv,D]` | pq `[T,Nq]`, pk `[T,Nk]`, pv `[T,Nv]` | |
//! | `InkAttnDecode` (75) | pq, pk, pv, k taps, v taps, q scale, r scale, rel proj (as `InkKvPrologue`), k cache `[Hk,Cr,Dh]`, v cache | o `[T,H*Dh]`, k cache, v cache, k ring, v ring (all **in place**) | the union of the two ops' attrs |
//! | `ConvMix` (74) | x, taps, res, addend (as `SconvRes`), src `[N,T,D]`, queries `[Q,D]` f32 (Q may be 0), norm `[S,D]` or `[0]`, msg `[S,T,D]` f32 or `[0]`, down, gain, forget (as `LoopBlend`) or three `[0]` | y, ring, sum (as `SconvRes`), mix `[S,T,D]` or `[0]` | pos0, eps |
//!
//! ## AttnResMix: the depth mix (spec a2, also the loop messages a12 and the output mix a16)
//!
//! Row `r`: sources `x_n = src[n, r]` for `n < N`, and `x_N = last[r] + addend[r]` (f32 sum, not rounded; just `last[r]`
//! without addend). `w_n = 1 / sqrt(mean(x_n^2) + SRC_EPS)`, `l[q,n] = (queries[q] . x_n) * w_n`, `p[q,.] = softmax_n(l[q,.])`,
//! `mix_q = sum_n p[q,n] * w_n * x_n`. Output: with norm, `rA(mix_q * inv(mix_q) * norm[q])` (eps = `attrs.eps`); without,
//! the f32 `mix_q`. The queries are the checkpoint's times `D^-0.5` (scaled at load).
//!
//! ## LoopBlend: the cross-pass controller (spec a12)
//!
//! Stream `s`, row `r`: `d = dest[Sd == 1 ? 0 : s, r]`, `m = msg[s, r]`; `pair = rA([d * w(d), m * w(m)])` (`w` as in
//! `AttnResMix`, SRC_EPS; dest half first, `2D` values); `h[g] = rA(sum_j down[s,g,j] * pair[j])` (G = 64 rows);
//! `gn[c] = sum_g gain[s,c,g] * h[g]`, `fg[c] = sum_g forget[s,c,g] * h[g]` (f32); `y = (1 - tanh(fg)) * d + tanh(gn) * m`;
//! `out[s,r] = rA(y * inv(y) * norm[s])` (eps = `attrs.eps`).
//!
//! ## SconvRes: static short conv with the residual epilogue (spec a6, a11, a15)
//!
//! `y[t,c] = rA((sum_{l < K} taps[l,c] * x(p - l, c) + x(p, c)) * (K + 1)^-0.5)`, lag-major taps (`l = 0` is the current
//! token; the loader turns the checkpoint's `[C,1,K]`, tap `K-1` current, into this), `x(q, c)` = `x[q - pos0, c]` for
//! `q >= pos0`, the ring row for `0 <= q < pos0`, zero for `q < 0`. The ring then gets `x` (rules above). With res:
//! `sum = rA((res + addend) + y)` (f32 sums in that order, `y` as rounded; `rA(res + y)` without addend).
//!
//! ## InkKvPrologue: K / V conv and the q / k / r post-processing (spec a6-a9)
//!
//! `pq` row layout `[q (H*Dh) | r (H*dr) | gate (H*Dh) | sink (H)]` (this op reads q and r; `RelSinkAttn` reads gate and sink).
//! `pk` row: `[k (Ckv = Hk*Dh) | taps (Hk*K, [head][lag])]` with `ink.dyn_conv`, else just `k`; `pv` likewise.
//! Conv of k (v the same with its own tensors): `kc[c] = rA((sum_l (ktaps[l,c] + dyn[c / Dh, l]) * kx(p - l, c) + kx(p, c)) * s)`,
//! `dyn` = the row's taps (zero without `dyn_conv`), `s = (2K+1)^-0.5` with `dyn_conv` else `(K+1)^-0.5`, `kx` the k part of
//! `pk` read as in `SconvRes` (the k ring holds `kx`). Then per kv head `k = rA(clip(kc, qk_clip))`,
//! `v = rA(vc * inv(vc))` (eps = `attrs.eps`, no weight). Per q head: `q = rA(clip(q_raw * qscale[h], qk_clip))`,
//! `r = rA(clip(r_raw * rscale[h], rel_clip))`; with `tau_floor > 0` both are then `rA(x * tau)`,
//! `tau = 1 + tau_alpha * ln(max((p + 1) / tau_floor, 1))` (f32, full-attention layers only). `rel[t,h,e] = rA(sum_j r[h,j] *
//! relproj[j,e])`. The K / V caches are not touched (`RelSinkAttn` writes them), so a decode kernel may fuse this op with
//! `RelSinkAttn` (one threadgroup per kv head and its `H/Hk` query heads) as long as the result equals the composition.
//!
//! ## RelSinkAttn: GQA with a distance bias, a per-query sink and the output gate (spec a9-a11)
//!
//! Query head `h` (kv head `h / (H/Hk)`) of row `t` at position `p` sees key positions `j` with `0 <= j <= p` and, with
//! `window > 0`, `p - j < window`. Key / value of `j`: the call's `k[j - pos0]` / `v[..]` for `j >= pos0`, else cache row
//! `j % Cr` (`ink.kv_ring`) or `j` (linear). `s_j = scale * (q . k_j) + (d < E ? rel[t,h,d] : 0)`, `d = p - j` (the bias is
//! not scaled; zero beyond the extent). `sink = pq[t, 2*H*Dh + H*dr + h]`, `gate = pq[t, H*Dh + H*dr + h*Dh + i]`;
//! `o = sum_j e^{s_j} v_j / (sum_j e^{s_j} + e^{sink})` (f32), `out[t, h*Dh + i] = rA(o_i * sigmoid(gate))`. After the
//! attention the caches get the call's k / v rows: linear rows `pos0 .. pos0+T` (needs `pos0 + T <= Cr`), ring rows by the
//! ring rule (`window - 1 + T` positions are needed while reading, so for a ring `window <= Cr`).
//!
//! ## InkAttnDecode: the fused decode attention (lane ink-m-attn, discriminant 75)
//!
//! Exactly `RelSinkAttn(InkKvPrologue(..))`: the prologue's q / rel / k / v are rounded to `A` as that op writes them and
//! are not outputs; `RelSinkAttn` reads `pq` (gate, sink) and the caches as above. T is 1..=16 (decode and verify rows; the
//! check refuses more, so a graph falls back to the pair for a chunk). A kernel is rows-exact like the pair.
//!
//! ## MoeExpertsMxfp8 and the `situ` activation of `MoeExperts`
//!
//! Weights: `W[n,k] = e4m3(w[n,k]) * 2^(scale[n, k/32] - 127)` (1x32 blocks, E8M0). Gate rows `0..I`, up rows `I..2I` (the
//! loader de-interleaves the checkpoint's `gate = rows 0::2, up = 1::2`). Per (token, expert slot): `g = rA(x . Wg)`,
//! `u = rA(x . Wu)`. Default SiLU / `gelu` (as `MoeExperts`): `a = rA(rA(act(g)) * u)`, `y = rA(a . W2)`,
//! `out = rA(sum_k rA(y * w_k))`. With `ink.situ` (SiTU, spec a14; also on `MoeExperts`, NVFP4):
//! `a = rA(gate_cap * tanh(g / gate_cap) * sigmoid(g) * up_cap * tanh(u / up_cap) * w_k)` (the routing weight before w2),
//! `y = rA(a . W2)`, `out = rA(sum_k y)`. A `situ` call selects only kernels registered `.situ()` (`ShapeInfo::situ`).
//! `ink.act_quant` is reserved for a W4A4 / W8A8 activation mode and must be 0.
//!
//! ## InkInProj: the attention in-projection of one visit
//!
//! `w` stacks the q-pack rows (`Nq`), the k-pack rows (`Nk`) and the v-pack rows (`Nv`); `pq = rA(x[0] . Wq^T)`,
//! `pk = rA(x[min(1, S-1)] . Wk^T)`, `pv = rA(x[min(2, S-1)] . Wv^T)`. One launch whether the three inputs differ or not.
//!
//! ## ConvMix: a sublayer boundary (vLLM `ops/conv_mix.py`, `loop.py:257-316`)
//!
//! Exactly `SconvRes(x, taps, res, addend) -> y, ring, sum`, then the depth mix of the next sublayer over `src` and the
//! source `last` = `sum` when sum is given, else `res + y` (f32, unrounded) when res is given, else `y`, with `queries`;
//! then with msg the blend `LoopBlend(mix, msg, down, gain, forget, norm)` (the mix unnormed, f32), else the norm as in
//! `AttnResMix`. `mix` is `[S,T,D]` in `A` (S = msg rows with a blend, Q otherwise), or f32 `[Q,T,D]` without norm and blend.
//! `Q = 0` (queries `[0,D]`, mix `[0]`) is the conv alone.

use crate::backend::TensorRef;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{Op, OpCall, ShapeInfo};

/// Fixed epsilon of the depth-mix source RMS, the controller pair RMS and the L2 clips (module doc).
pub const SRC_EPS: f64 = 1e-6;

/// Attributes of the Inkling-2 ops (`OpAttrs::ink`); every other op ignores them (`MoeExperts` reads the `situ` fields).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InkAttrs {
    /// `MoeExperts` / `MoeExpertsMxfp8`: SiTU activation (`gate_cap`, `up_cap`, routing weight before w2).
    pub situ: bool,
    pub gate_cap: f32,
    pub up_cap: f32,
    /// Expert ops: activation quantisation mode, reserved (0 = 16-bit activations, the only one implemented).
    pub act_quant: u8,
    /// `InkKvPrologue`: L2 clip thresholds of q / k and of r.
    pub qk_clip: f32,
    pub rel_clip: f32,
    /// `InkKvPrologue`: log-position scaling of q and r, `tau = 1 + tau_alpha * ln(max((p+1) / tau_floor, 1))`; 0 = off.
    pub tau_floor: usize,
    pub tau_alpha: f32,
    /// `InkKvPrologue`: dynamic K / V conv (full-attention layers: per-token taps in `pk` / `pv`).
    pub dyn_conv: bool,
    /// `InkKvPrologue` / `RelSinkAttn`: rel-bias rank per head (the `r` width).
    pub d_rel: usize,
    /// `RelSinkAttn`: the K / V caches are rings (row `p % Cr`), else linear (row `p`).
    pub kv_ring: bool,
}

impl Default for InkAttrs {
    fn default() -> Self {
        InkAttrs {
            situ: false,
            gate_cap: 0.0,
            up_cap: 0.0,
            act_quant: 0,
            qk_clip: 0.0,
            rel_clip: 0.0,
            tau_floor: 0,
            tau_alpha: 0.0,
            dyn_conv: false,
            d_rel: 0,
            kv_ring: false,
        }
    }
}

/// The Inkling-2 ops, in discriminant order (67 ..).
pub const OPS: [Op; 9] = [
    Op::AttnResMix,
    Op::LoopBlend,
    Op::SconvRes,
    Op::InkKvPrologue,
    Op::RelSinkAttn,
    Op::MoeExpertsMxfp8,
    Op::InkInProj,
    Op::ConvMix,
    Op::InkAttnDecode,
];

/// The ops that take a device position (`OpCall::pos`, `i32 [1]`).
pub const POSITIONED: [Op; 5] = [Op::SconvRes, Op::InkKvPrologue, Op::RelSinkAttn, Op::ConvMix, Op::InkAttnDecode];

/// Most rows of an `InkAttnDecode` call.
pub const ATTN_DECODE_MAX_ROWS: usize = 16;

/// The two calls `InkAttnDecode` is defined as (prologue, attention), with `q`, `rel`, `k`, `v` the intermediates (any
/// buffers of the right shapes; the checks only look at shapes and dtypes).
pub fn attn_decode_parts(call: &OpCall, q: TensorRef, rel: TensorRef, k: TensorRef, v: TensorRef) -> (OpCall, OpCall) {
    let (i, o) = (&call.inputs, &call.outputs);
    let mut pro = OpCall::new(
        Op::InkKvPrologue,
        i[..8].to_vec(),
        vec![q.clone(), rel.clone(), k.clone(), v.clone(), o[3].clone(), o[4].clone()],
        call.attrs.clone(),
    );
    pro.pos = call.pos.clone();
    let mut att =
        OpCall::new(Op::RelSinkAttn, vec![q, k, v, i[8].clone(), i[9].clone(), rel, i[0].clone()], o[..3].to_vec(), call.attrs.clone());
    att.pos = call.pos.clone();
    (pro, att)
}

/// Shapes of the intermediates of `InkAttnDecode`: q, rel, k, v.
pub fn attn_decode_inter(call: &OpCall) -> [Vec<usize>; 4] {
    let (i, a) = (&call.inputs, &call.attrs);
    let t = i[0].shape.first().copied().unwrap_or(0);
    let e = i[7].shape.get(1).copied().unwrap_or(0);
    [vec![t, a.n_heads, a.head_dim], vec![t, a.n_heads, e], vec![t, a.n_kv_heads, a.head_dim], vec![t, a.n_kv_heads, a.head_dim]]
}

/// `[q | r | gate | sink]` width of a `pq` row.
pub fn nq(h: usize, dh: usize, dr: usize) -> usize {
    2 * h * dh + h * dr + h
}

fn bad<T>(op: Op, m: String) -> Result<T> {
    Err(Error::invalid(format!("{op}: {m}")))
}

fn absent(t: &TensorRef) -> bool {
    t.numel() == 0
}

fn is_act(d: DType) -> bool {
    matches!(d, DType::Bf16 | DType::F32 | DType::F16)
}

fn float(d: DType) -> bool {
    matches!(d, DType::Bf16 | DType::F32 | DType::F16)
}

/// Shape and dtype of one tensor.
fn want(op: Op, t: &TensorRef, dt: Option<DType>, shape: &[usize], what: &str) -> Result<()> {
    if t.shape != shape || dt.is_some_and(|d| d != t.dtype) || (dt.is_none() && !float(t.dtype)) {
        return bad(op, format!("{what} must be {} {shape:?}, got {} {:?}", dt.map_or("float".to_string(), |d| d.to_string()), t.dtype, t.shape));
    }
    Ok(())
}

/// The conv-and-residual part shared by `SconvRes` and `ConvMix` (inputs 0..4, outputs 0..3). Returns (T, C, act).
fn check_sconv(op: Op, i: &[TensorRef], o: &[TensorRef]) -> Result<(usize, usize, DType)> {
    if i[0].shape.len() != 2 || !is_act(i[0].dtype) {
        return bad(op, format!("x must be a float [T,C], got {} {:?}", i[0].dtype, i[0].shape));
    }
    let (t, c, a) = (i[0].shape[0], i[0].shape[1], i[0].dtype);
    if i[1].shape.len() != 2 || i[1].shape[1] != c || i[1].shape[0] < 1 || !float(i[1].dtype) {
        return bad(op, format!("taps must be [K,{c}], got {:?}", i[1].shape));
    }
    let k = i[1].shape[0];
    want(op, &o[0], Some(a), &[t, c], "y")?;
    if o[1].shape.len() != 2 || o[1].shape[1] != c || o[1].dtype != a || o[1].shape[0] < k.max(1) {
        return bad(op, format!("ring must be {a} [Rr >= {k},{c}], got {} {:?}", o[1].dtype, o[1].shape));
    }
    let (res, add, sum) = (!absent(&i[2]), !absent(&i[3]), !absent(&o[2]));
    if !res && (add || sum) {
        return bad(op, "addend / sum need res".into());
    }
    if res {
        want(op, &i[2], Some(a), &[t, c], "res")?;
    }
    if add {
        want(op, &i[3], Some(a), &[t, c], "addend")?;
        if !sum {
            return bad(op, "an addend needs the sum output".into());
        }
    }
    if sum {
        want(op, &o[2], Some(a), &[t, c], "sum")?;
    }
    if op == Op::SconvRes && res && !sum {
        return bad(op, "res needs the sum output".into());
    }
    Ok((t, c, a))
}

/// The depth mix (+ blend) part of `ConvMix` / the whole `AttnResMix` / `LoopBlend` operand checks.
fn check_mix(op: Op, src: &TensorRef, q: &TensorRef, t: usize, d: usize, a: DType) -> Result<usize> {
    if src.shape.len() != 3 || src.shape[1] != t || src.shape[2] != d || (src.shape[0] > 0 && src.dtype != a) {
        return bad(op, format!("src must be {a} [N,{t},{d}], got {} {:?}", src.dtype, src.shape));
    }
    if q.shape.len() != 2 || q.shape[1] != d || q.dtype != DType::F32 {
        return bad(op, format!("queries must be f32 [Q,{d}], got {} {:?}", q.dtype, q.shape));
    }
    Ok(q.shape[0])
}

fn check_blend(op: Op, msg: &TensorRef, down: &TensorRef, gain: &TensorRef, forget: &TensorRef, norm: &TensorRef, t: usize, d: usize) -> Result<usize> {
    if msg.shape.len() != 3 || msg.shape[1..] != [t, d] || msg.dtype != DType::F32 {
        return bad(op, format!("msg must be f32 [S,{t},{d}], got {} {:?}", msg.dtype, msg.shape));
    }
    let s = msg.shape[0];
    if down.shape.len() != 3 || down.shape[0] != s || down.shape[2] != 2 * d || !float(down.dtype) {
        return bad(op, format!("down must be [{s},G,{}], got {:?}", 2 * d, down.shape));
    }
    let g = down.shape[1];
    want(op, gain, None, &[s, d, g], "gain")?;
    want(op, forget, None, &[s, d, g], "forget")?;
    want(op, norm, None, &[s, d], "norm")?;
    if s == 0 || g == 0 {
        return bad(op, "empty blend".into());
    }
    Ok(s)
}

/// Call checks of the Inkling-2 ops (`OpCall::check` has validated arity and bounds).
pub fn check(call: &OpCall) -> Result<()> {
    let (i, o, at) = (&call.inputs, &call.outputs, &call.attrs);
    let op = call.op;
    let ink = &at.ink;
    match op {
        Op::AttnResMix => {
            if i[1].shape.len() != 2 || !is_act(i[1].dtype) {
                return bad(op, format!("last must be a float [R,D], got {:?}", i[1].shape));
            }
            let (r, d, a) = (i[1].shape[0], i[1].shape[1], i[1].dtype);
            let nq = check_mix(op, &i[0], &i[3], r, d, a)?;
            if !absent(&i[2]) {
                want(op, &i[2], Some(a), &[r, d], "addend")?;
            }
            if nq == 0 {
                return bad(op, "no queries".into());
            }
            if absent(&i[4]) {
                want(op, &o[0], Some(DType::F32), &[nq, r, d], "out (no norm)")?;
            } else {
                want(op, &i[4], None, &[nq, d], "norm")?;
                want(op, &o[0], Some(a), &[nq, r, d], "out")?;
            }
        }
        Op::LoopBlend => {
            if o[0].shape.len() != 3 || !is_act(o[0].dtype) {
                return bad(op, format!("out must be a float [S,R,D], got {:?}", o[0].shape));
            }
            let (s, r, d) = (o[0].shape[0], o[0].shape[1], o[0].shape[2]);
            let sb = check_blend(op, &i[1], &i[2], &i[3], &i[4], &i[5], r, d)?;
            if sb != s {
                return bad(op, format!("msg has {sb} streams, out {s}"));
            }
            if i[0].shape.len() != 3 || !(i[0].shape[0] == 1 || i[0].shape[0] == s) || i[0].shape[1..] != [r, d] || i[0].dtype != DType::F32 {
                return bad(op, format!("dest must be f32 [1 or {s},{r},{d}], got {} {:?}", i[0].dtype, i[0].shape));
            }
        }
        Op::SconvRes => {
            check_sconv(op, i, o)?;
        }
        Op::ConvMix => {
            let (t, d, a) = check_sconv(op, i, o)?;
            let nq = check_mix(op, &i[4], &i[5], t, d, a)?;
            let blend = !absent(&i[7]);
            if nq == 0 {
                if !absent(&o[3]) || blend || !absent(&i[6]) {
                    return bad(op, "no queries: mix, norm and msg must be [0]".into());
                }
                return Ok(());
            }
            if blend {
                let s = check_blend(op, &i[7], &i[8], &i[9], &i[10], &i[6], t, d)?;
                if !(nq == 1 || nq == s) {
                    return bad(op, format!("{nq} queries for {s} blended streams"));
                }
                want(op, &o[3], Some(a), &[s, t, d], "mix")?;
            } else {
                if [&i[8], &i[9], &i[10]].iter().any(|x| !absent(x)) {
                    return bad(op, "controller weights without msg".into());
                }
                if absent(&i[6]) {
                    want(op, &o[3], Some(DType::F32), &[nq, t, d], "mix (no norm)")?;
                } else {
                    want(op, &i[6], None, &[nq, d], "norm")?;
                    want(op, &o[3], Some(a), &[nq, t, d], "mix")?;
                }
            }
        }
        Op::InkKvPrologue => {
            let (h, hk, dh, dr) = (at.n_heads, at.n_kv_heads, at.head_dim, ink.d_rel);
            if h == 0 || hk == 0 || h % hk != 0 || dh == 0 || dr == 0 {
                return bad(op, format!("n_heads {h}, n_kv_heads {hk}, head_dim {dh}, d_rel {dr}"));
            }
            if i[0].shape.len() != 2 || !is_act(i[0].dtype) {
                return bad(op, format!("pq must be a float [T,Nq], got {:?}", i[0].shape));
            }
            let (t, a) = (i[0].shape[0], i[0].dtype);
            want(op, &i[0], Some(a), &[t, nq(h, dh, dr)], "pq")?;
            let ckv = hk * dh;
            if i[3].shape.len() != 2 || i[3].shape[1] != ckv {
                return bad(op, format!("k taps must be [K,{ckv}], got {:?}", i[3].shape));
            }
            let k = i[3].shape[0];
            let pw = if ink.dyn_conv { ckv + hk * k } else { ckv };
            want(op, &i[1], Some(a), &[t, pw], "pk")?;
            want(op, &i[2], Some(a), &[t, pw], "pv")?;
            want(op, &i[3], None, &[k, ckv], "k taps")?;
            want(op, &i[4], None, &[k, ckv], "v taps")?;
            want(op, &i[5], None, &[h, dh], "q scale")?;
            want(op, &i[6], None, &[h, dr], "r scale")?;
            if i[7].shape.len() != 2 || i[7].shape[0] != dr || i[7].shape[1] == 0 {
                return bad(op, format!("rel proj must be [{dr},E], got {:?}", i[7].shape));
            }
            let e = i[7].shape[1];
            want(op, &i[7], None, &[dr, e], "rel proj")?;
            want(op, &o[0], Some(a), &[t, h, dh], "q")?;
            want(op, &o[1], Some(a), &[t, h, e], "rel")?;
            want(op, &o[2], Some(a), &[t, hk, dh], "k")?;
            want(op, &o[3], Some(a), &[t, hk, dh], "v")?;
            for (x, w) in [(&o[4], "k ring"), (&o[5], "v ring")] {
                if x.shape.len() != 2 || x.shape[1] != ckv || x.shape[0] < k.max(1) || x.dtype != a {
                    return bad(op, format!("{w} must be {a} [Rr >= {k},{ckv}], got {} {:?}", x.dtype, x.shape));
                }
            }
            if o[4].shape != o[5].shape {
                return bad(op, "k and v rings differ".into());
            }
            if !(ink.qk_clip > 0.0) || !(ink.rel_clip > 0.0) || !(at.eps >= 0.0) || !(ink.tau_alpha.is_finite()) {
                return bad(op, format!("clip thresholds {} / {} must be > 0", ink.qk_clip, ink.rel_clip));
            }
        }
        Op::RelSinkAttn => {
            let (h, hk, dh, dr) = (at.n_heads, at.n_kv_heads, at.head_dim, ink.d_rel);
            if h == 0 || hk == 0 || h % hk != 0 || dh == 0 || dr == 0 {
                return bad(op, format!("n_heads {h}, n_kv_heads {hk}, head_dim {dh}, d_rel {dr}"));
            }
            if i[0].shape.len() != 3 || !is_act(i[0].dtype) {
                return bad(op, format!("q must be a float [T,H,Dh], got {:?}", i[0].shape));
            }
            let (t, a) = (i[0].shape[0], i[0].dtype);
            want(op, &i[0], Some(a), &[t, h, dh], "q")?;
            want(op, &i[1], Some(a), &[t, hk, dh], "k")?;
            want(op, &i[2], Some(a), &[t, hk, dh], "v")?;
            if i[3].shape.len() != 3 || i[3].shape[0] != hk || i[3].shape[2] != dh {
                return bad(op, format!("k cache must be [{hk},Cr,{dh}], got {:?}", i[3].shape));
            }
            let cr = i[3].shape[1];
            want(op, &i[3], Some(a), &[hk, cr, dh], "k cache (input)")?;
            want(op, &i[4], Some(a), &[hk, cr, dh], "v cache (input)")?;
            if i[5].shape.len() != 3 || i[5].shape[..2] != [t, h] || i[5].shape[2] == 0 {
                return bad(op, format!("rel must be [{t},{h},E], got {:?}", i[5].shape));
            }
            want(op, &i[5], Some(a), &i[5].shape.clone(), "rel")?;
            want(op, &i[6], Some(a), &[t, nq(h, dh, dr)], "pq")?;
            want(op, &o[0], Some(a), &[t, h * dh], "o")?;
            want(op, &o[1], Some(a), &[hk, cr, dh], "k cache")?;
            want(op, &o[2], Some(a), &[hk, cr, dh], "v cache")?;
            if o[1].buf.ptr_eq(&o[2].buf) && o[1].offset == o[2].offset {
                return bad(op, "k and v caches alias".into());
            }
            let dev = call.pos.is_some();
            if ink.kv_ring {
                if at.window == 0 || at.window > cr {
                    return bad(op, format!("ring cache of {cr} rows with window {} (needs 1 <= window <= rows)", at.window));
                }
            } else if !dev && at.pos0 + t > cr {
                return bad(op, format!("linear cache of {cr} rows cannot hold positions {}..{}", at.pos0, at.pos0 + t));
            }
            if !(at.scale.is_finite()) {
                return bad(op, "scale".into());
            }
        }
        Op::MoeExpertsMxfp8 => {
            if i[0].shape.len() != 2 || !is_act(i[0].dtype) || i[1].shape.len() != 2 || i[3].shape.len() != 3 {
                return bad(op, format!("x [T,H], ids [T,K], w13 [E,2I,H]: got {:?} {:?} {:?}", i[0].shape, i[1].shape, i[3].shape));
            }
            let (t, hd) = (i[0].shape[0], i[0].shape[1]);
            let k = i[1].shape[1];
            let (e, two_i) = (i[3].shape[0], i[3].shape[1]);
            let inter = two_i / 2;
            if two_i % 2 != 0 || hd % 32 != 0 || inter % 32 != 0 || e == 0 || k == 0 || k > e {
                return bad(op, format!("H {hd} and I {inter} must be multiples of 32 (E {e}, K {k})"));
            }
            want(op, &i[1], Some(DType::I32), &[t, k], "ids")?;
            want(op, &i[2], Some(DType::F32), &[t, k], "weights")?;
            want(op, &i[3], Some(DType::Fp8E4m3), &[e, two_i, hd], "w13")?;
            want(op, &i[4], Some(DType::U8), &[e, two_i, hd / 32], "w13 scales")?;
            want(op, &i[5], Some(DType::Fp8E4m3), &[e, hd, inter], "w2")?;
            want(op, &i[6], Some(DType::U8), &[e, hd, inter / 32], "w2 scales")?;
            if !is_act(o[0].dtype) || o[0].shape != [t, hd] {
                return bad(op, format!("out must be a float [{t},{hd}], got {:?}", o[0].shape));
            }
            check_act_epilogue(call)?;
        }
        Op::InkInProj => {
            if i[0].shape.len() != 3 || !(i[0].shape[0] == 1 || i[0].shape[0] == 3) || !is_act(i[0].dtype) {
                return bad(op, format!("x must be a float [1 or 3,T,D], got {:?}", i[0].shape));
            }
            let (t, d, a) = (i[0].shape[1], i[0].shape[2], i[0].dtype);
            let ns: Vec<usize> = o.iter().map(|x| x.shape.get(1).copied().unwrap_or(0)).collect();
            for (x, w) in o.iter().zip(["pq", "pk", "pv"]) {
                if x.shape.len() != 2 || x.shape[0] != t || x.dtype != a || x.shape[1] == 0 {
                    return bad(op, format!("{w} must be {a} [{t},N], got {} {:?}", x.dtype, x.shape));
                }
            }
            want(op, &i[1], None, &[ns.iter().sum(), d], "w")?;
        }
        Op::InkAttnDecode => {
            let t = i[0].shape.first().copied().unwrap_or(0);
            if !(1..=ATTN_DECODE_MAX_ROWS).contains(&t) || i[7].shape.len() != 2 {
                return bad(op, format!("1..={ATTN_DECODE_MAX_ROWS} rows (got {t}) and a [dr,E] rel proj"));
            }
            let a = i[0].dtype;
            let [qs, rs, ks, vs] = attn_decode_inter(call);
            let ph = |shape: Vec<usize>| TensorRef { buf: i[0].buf.clone(), offset: 0, dtype: a, shape };
            let (pro, att) = attn_decode_parts(call, ph(qs), ph(rs), ph(ks), ph(vs));
            check(&pro)?;
            check(&att)?;
        }
        _ => return bad(op, "not an Inkling-2 op".into()),
    }
    Ok(())
}

/// The expert-activation attributes (`MoeExperts`, `MoeExpertsMxfp8`).
pub fn check_act_epilogue(call: &OpCall) -> Result<()> {
    let (a, ink) = (&call.attrs, &call.attrs.ink);
    if ink.act_quant != 0 {
        return bad(call.op, format!("ink.act_quant {} is reserved (only 16-bit activations are implemented)", ink.act_quant));
    }
    if ink.situ && (a.gelu || !(ink.gate_cap > 0.0) || !(ink.up_cap > 0.0)) {
        return bad(call.op, format!("situ needs gate_cap, up_cap > 0 and no gelu (caps {} / {}, gelu {})", ink.gate_cap, ink.up_cap, a.gelu));
    }
    Ok(())
}

/// Shape facts for kernel predicates (`ShapeInfo::from_call`).
pub fn shape(call: &OpCall, s: &mut ShapeInfo) {
    let (i, o, a) = (&call.inputs, &call.outputs, &call.attrs);
    let dim = |t: &TensorRef, d: usize| t.shape.get(d).copied().unwrap_or(0);
    s.situ = a.ink.situ;
    match call.op {
        Op::AttnResMix => {
            s.m = dim(&i[1], 0);
            s.k = dim(&i[1], 1);
            s.n = dim(&i[3], 0);
            s.kv_len = dim(&i[0], 0) + 1;
            s.act = Some(i[1].dtype);
        }
        Op::LoopBlend => {
            s.m = dim(&o[0], 1);
            s.k = dim(&o[0], 2);
            s.n = dim(&o[0], 0);
            s.act = Some(o[0].dtype);
            s.weight = Some(i[2].dtype);
        }
        Op::SconvRes | Op::ConvMix => {
            s.m = dim(&i[0], 0);
            s.k = dim(&i[0], 1);
            s.n = if call.op == Op::ConvMix { dim(&i[5], 0) } else { 0 };
            s.kv_len = dim(&i[1], 0);
            s.weight = Some(i[1].dtype);
        }
        Op::InkKvPrologue | Op::RelSinkAttn | Op::InkAttnDecode => {
            s.m = dim(&i[0], 0);
            s.n_heads = a.n_heads;
            s.n_kv_heads = a.n_kv_heads;
            s.head_dim = a.head_dim;
            s.k = a.head_dim;
            s.kv_len = a.pos0 + dim(&i[0], 0);
            if call.op == Op::RelSinkAttn {
                s.weight = Some(i[3].dtype);
            }
            if call.op == Op::InkAttnDecode {
                s.weight = Some(i[8].dtype);
            }
        }
        Op::MoeExpertsMxfp8 => {
            s.m = dim(&i[0], 0);
            s.k = dim(&i[0], 1);
            s.n = dim(&i[3], 1) / 2;
            s.experts = dim(&i[3], 0);
            s.top_k = dim(&i[1], 1);
            s.weight = Some(i[3].dtype);
        }
        Op::InkInProj => {
            s.m = dim(&i[0], 1);
            s.k = dim(&i[0], 2);
            s.n = dim(&i[1], 0);
            s.weight = Some(i[1].dtype);
        }
        _ => {}
    }
}
