//! EXL3 routed experts as an `Op::MoeExperts` weight format (DeepSeek-V4.1 Flash on the Mia EXL3 checkpoint, where every
//! expert matrix is an EXL3 trellis): no new op, the same eight inputs, told apart from the NVFP4 encoding by the dtype of
//! input 3 (i16 here, u8 there).
//!
//! ```text
//! 0 x        [T, H]   bf16 / f16 / f32
//! 1 ids      [T, K]   i32   expert of each (token, slot); an id outside 0..E contributes nothing
//! 2 weights  [T, K]   f32
//! 3 trellis  [W]      i16   arena: every expert's gate [H/16, I/16, 8*k2], up (same) and down [I/16, H/16, 8*k2] tiles
//! 4 scales   [E, 3H + 3I] f16  per expert: suh_g [H] | suh_u [H] | svh_g [I] | svh_u [I] | suh_d [I] | svh_d [H]
//! 5 meta     [E, 8]   i32   per expert: gate, up, down offsets into the arena in 16-byte units, gate, up, down k2, 0, 0
//! 6 codebook [1]      i32   0 3inst, 1 mcg, 2 mul1 (one per call, as for `MatmulExl3`)
//! 7 limit    [1]      f32   SwiGLU clamp (DeepSeek `swiglu_limit`): silu(min(g, L)) * clamp(u, -L, L); +inf = none
//! out        [T, H]   bf16 / f16 / f32: out[t] = sum_j weights[t, j] * down_e(act(gate_e(x[t]), up_e(x[t]))), e = ids[t, j]
//! ```
//!
//! Each matrix carries its own width (`k2`, twice the bits per weight) as device data, so one call mixes widths: the Mia
//! checkpoint's routed experts are 2 or 3 bits and its shared expert 4 or 5 bits, and the shared expert can ride in the same
//! launch as an extra expert (id E-1 of an arena of `n_routed + 1`) picked by an extra slot of weight 1 in every row, as
//! TensorFold's served DeepSeek-V4.1 decode does (`shared_slot = slots - 1`). E, H and I come from the shapes (E: meta rows,
//! H: x, I: the scales row), the widths and offsets only from device data, which plan-time checks cannot see; kernels and
//! the cpu reference refuse (NaN / error) an offset outside the arena or a width that is no EXL3 width.
//!
//! Tensor parallelism ([`load`]): rank `r` of `world` keeps every expert's intermediate columns `[r I/world, (r+1) I/world)`:
//! gate / up split their N (`ShardAxis::Rows`), down its K (`ShardAxis::Cols`), so a rank's output is its partial sum of the
//! block output and the split must sit on the 128-point Hadamard grid (`exl3::shard_ranges`; DeepSeek-V4.1 at TP=2:
//! I = 2304 = 2 x 9 x 128).

use crate::admission::{ShapeCaps, SweepConfig};
use crate::backend::{upload_new, Backend, MemKind, TensorRef};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::exec::{HostCall, OutSpec};
use crate::ops::{Op, OpAttrs, OpCall, ShapeInfo};
use crate::quant::exl3::{self, Codebook, HAD, TILE};
use crate::quant::WeightMeta;
use crate::rng::SplitMix64;
use crate::tensor::HostTensor;
use crate::weights::WeightSource;

/// Columns of a meta row.
pub const META_COLS: usize = 8;
/// Most slots per token (the kernels' member lists pack a slot into 5 bits).
pub const MAX_SLOTS: usize = 32;

/// The three matrices of an expert, in meta order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mat {
    Gate = 0,
    Up = 1,
    Down = 2,
}

/// Offsets (in f16 elements) of the six scale vectors in a scales row of width `3h + 3i`.
pub fn scale_offsets(h: usize, i: usize) -> [usize; 6] {
    [0, h, 2 * h, 2 * h + i, 2 * h + 2 * i, 2 * h + 3 * i]
}

/// Is this `MoeExperts` call the EXL3 encoding (input 3 an i16 arena)?
pub fn is_exl3(call: &OpCall) -> bool {
    call.op == Op::MoeExperts && call.inputs.get(3).is_some_and(|t| t.dtype == DType::I16)
}

/// (T, H, I, E, K) of an EXL3 `MoeExperts` call that passed [`check`].
pub fn dims(call: &OpCall) -> (usize, usize, usize, usize, usize) {
    let i = &call.inputs;
    let (t, h) = (i[0].shape[0], i[0].shape[1]);
    let e = i[5].shape[0];
    let inter = (i[4].shape[1].saturating_sub(3 * h)) / 3;
    (t, h, inter, e, i[1].shape[1])
}

fn bad<T>(m: String) -> Result<T> {
    Err(Error::invalid(format!("{} (exl3 experts): {m}", Op::MoeExperts)))
}

/// Validate the EXL3 encoding (shapes and dtypes; offsets and widths are device data).
pub fn check(call: &OpCall) -> Result<()> {
    let (i, o) = (&call.inputs, &call.outputs);
    let act = |t: &TensorRef| matches!(t.dtype, DType::Bf16 | DType::F16 | DType::F32);
    if i[0].shape.len() != 2 || !act(&i[0]) || o[0].shape.len() != 2 || !act(&o[0]) {
        return bad(format!("x and out must be 2-D bf16/f16/f32, got {} {:?} and {} {:?}", i[0].dtype, i[0].shape, o[0].dtype, o[0].shape));
    }
    let (t, h) = (i[0].shape[0], i[0].shape[1]);
    if i[1].dtype != DType::I32 || i[1].shape.len() != 2 || i[1].shape[0] != t {
        return bad(format!("ids must be i32 [T={t}, K], got {} {:?}", i[1].dtype, i[1].shape));
    }
    let k = i[1].shape[1];
    if i[2].dtype != DType::F32 || i[2].shape != [t, k] {
        return bad(format!("weights must be f32 [{t}, {k}], got {} {:?}", i[2].dtype, i[2].shape));
    }
    if i[3].shape.len() != 1 || i[3].shape[0] == 0 {
        return bad(format!("trellis arena must be i16 [W], got {:?}", i[3].shape));
    }
    if i[5].dtype != DType::I32 || i[5].shape.len() != 2 || i[5].shape[1] != META_COLS || i[5].shape[0] == 0 {
        return bad(format!("meta must be i32 [E, {META_COLS}], got {} {:?}", i[5].dtype, i[5].shape));
    }
    let e = i[5].shape[0];
    if i[4].dtype != DType::F16 || i[4].shape.len() != 2 || i[4].shape[0] != e || i[4].shape[1] <= 3 * h || (i[4].shape[1] - 3 * h) % 3 != 0 {
        return bad(format!("scales must be f16 [E={e}, 3H + 3I] with H={h}, got {} {:?}", i[4].dtype, i[4].shape));
    }
    let inter = (i[4].shape[1] - 3 * h) / 3;
    if h % HAD != 0 || inter % HAD != 0 {
        return bad(format!("H={h} and I={inter} must be multiples of {HAD} (the Hadamard blocks)"));
    }
    if i[6].dtype != DType::I32 || i[6].shape != [1] {
        return bad(format!("codebook must be i32 [1], got {} {:?}", i[6].dtype, i[6].shape));
    }
    if i[7].dtype != DType::F32 || i[7].shape != [1] {
        return bad(format!("limit must be f32 [1], got {} {:?}", i[7].dtype, i[7].shape));
    }
    if k == 0 || k > MAX_SLOTS {
        return bad(format!("{k} slots per token (1..={MAX_SLOTS})"));
    }
    if o[0].shape != [t, h] {
        return bad(format!("out {:?} != [{t},{h}]", o[0].shape));
    }
    Ok(())
}

/// Shape facts: m = T, k = H, n = I, experts = E, top_k = K, weight = i16 (no other field: the NVFP4 encoding's facts are
/// unchanged).
pub fn shape(call: &OpCall, s: &mut ShapeInfo) {
    let i = &call.inputs;
    s.m = i[0].shape[0];
    s.k = i[0].shape[1];
    s.n = i[4].shape.get(1).map_or(0, |c| c.saturating_sub(3 * s.k) / 3);
    s.experts = i[5].shape[0];
    s.top_k = i[1].shape.get(1).copied().unwrap_or(0);
    s.weight = Some(DType::I16);
}

// ---- host side: random experts, the f64 reference -------------------------------------------------------------------

/// A layer's experts on the host, in the op's encoding (inputs 3..8).
#[derive(Clone, Debug)]
pub struct HostExperts {
    pub e: usize,
    pub h: usize,
    pub i: usize,
    pub trellis: HostTensor,
    pub scales: HostTensor,
    pub meta: HostTensor,
    pub codebook: HostTensor,
    pub limit: HostTensor,
}

impl HostExperts {
    pub fn parts(&self) -> Vec<HostTensor> {
        vec![self.trellis.clone(), self.scales.clone(), self.meta.clone(), self.codebook.clone(), self.limit.clone()]
    }
}

/// Random experts `e` x (gate, up, down) at hidden `h`, intermediate `i`; expert `x`'s matrix `m` has width `k2(x, m)`
/// (valid with `cb`); arena offsets in expert order with `gap` 16-byte units of padding before each matrix (the kernels must
/// not assume a dense arena).
pub fn random_experts(
    rng: &mut SplitMix64,
    e: usize,
    h: usize,
    i: usize,
    k2: &dyn Fn(usize, Mat) -> usize,
    cb: Codebook,
    limit: f32,
    gap: usize,
) -> HostExperts {
    let mut arena: Vec<u8> = Vec::new();
    let mut meta = vec![0i32; e * META_COLS];
    let cols = 3 * h + 3 * i;
    let mut scales = vec![0f32; e * cols];
    let off = scale_offsets(h, i);
    for x in 0..e {
        for (m, (kk, nn)) in [(Mat::Gate, (h, i)), (Mat::Up, (h, i)), (Mat::Down, (i, h))] {
            let w = k2(x, m);
            let q = exl3::random(rng, nn, kk, w, cb);
            arena.extend(std::iter::repeat_n(0xA5u8, 16 * gap));
            meta[x * META_COLS + m as usize] = (arena.len() / 16) as i32;
            meta[x * META_COLS + 3 + m as usize] = w as i32;
            arena.extend_from_slice(&q.parts[0].data);
            let (suh, svh) = (q.parts[1].to_f32(), q.parts[2].to_f32());
            let (so, vo) = match m {
                Mat::Gate => (off[0], off[2]),
                Mat::Up => (off[1], off[3]),
                Mat::Down => (off[4], off[5]),
            };
            scales[x * cols + so..x * cols + so + kk].copy_from_slice(&suh);
            scales[x * cols + vo..x * cols + vo + nn].copy_from_slice(&svh);
        }
    }
    let words = arena.len() / 2;
    HostExperts {
        e,
        h,
        i,
        trellis: HostTensor { dtype: DType::I16, shape: vec![words], data: arena },
        scales: HostTensor::from_f32_as(DType::F16, &[e, cols], &scales),
        meta: HostTensor::from_i32(&[e, META_COLS], &meta),
        codebook: HostTensor::from_i32(&[1], &[cb as i32]),
        limit: HostTensor::from_f32(&[1], &[limit]),
    }
}

/// One expert matrix read out of the encoding, validated: (trellis bytes, k, n, k2, suh, svh).
pub struct MatView<'a> {
    pub trellis: &'a [u8],
    pub k: usize,
    pub n: usize,
    pub k2: usize,
    pub suh: Vec<f64>,
    pub svh: Vec<f64>,
}

/// The encoding's byte views (`trellis`, `scales`, `meta` as i32, `cb`), for the reference and tests.
pub struct View<'a> {
    pub e: usize,
    pub h: usize,
    pub i: usize,
    pub trellis: &'a [u8],
    pub scales: &'a [u8],
    pub meta: Vec<i32>,
    pub cb: Codebook,
    pub limit: f32,
}

impl<'a> View<'a> {
    pub fn new(e: usize, h: usize, i: usize, trellis: &'a [u8], scales: &'a [u8], meta: &[u8], cb: &[u8], limit: &[u8]) -> Result<View<'a>> {
        let cbv = i32::from_le_bytes(cb[..4].try_into().unwrap());
        let cb = Codebook::from_i32(cbv).ok_or_else(|| Error::invalid(format!("exl3 experts: codebook {cbv} is not 0, 1 or 2")))?;
        let meta: Vec<i32> = meta.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let limit = f32::from_le_bytes(limit[..4].try_into().unwrap());
        Ok(View { e, h, i, trellis, scales, meta, cb, limit })
    }

    /// Expert `x`'s matrix `m`, or an error for an offset outside the arena or a width that is no EXL3 width (with `cb`).
    pub fn mat(&self, x: usize, m: Mat) -> Result<MatView<'a>> {
        let (k, n) = if m == Mat::Down { (self.i, self.h) } else { (self.h, self.i) };
        let off = self.meta[x * META_COLS + m as usize];
        let k2 = self.meta[x * META_COLS + 3 + m as usize];
        if k2 < 0 || !exl3::valid_k2(k2 as usize, self.cb) {
            return Err(Error::invalid(format!("exl3 experts: expert {x} {m:?} width k2={k2} is not valid with {}", self.cb.name())));
        }
        let k2 = k2 as usize;
        let len = (k / TILE) * (n / TILE) * exl3::tile_bytes(k2);
        if off < 0 || off as usize * 16 + len > self.trellis.len() {
            return Err(Error::invalid(format!(
                "exl3 experts: expert {x} {m:?} at 16-byte offset {off} ({len} bytes) is outside the {}-byte arena",
                self.trellis.len()
            )));
        }
        let cols = 3 * self.h + 3 * self.i;
        let so = scale_offsets(self.h, self.i);
        let (a, b) = match m {
            Mat::Gate => (so[0], so[2]),
            Mat::Up => (so[1], so[3]),
            Mat::Down => (so[4], so[5]),
        };
        let row = &self.scales[2 * x * cols..2 * (x + 1) * cols];
        Ok(MatView {
            trellis: &self.trellis[off as usize * 16..off as usize * 16 + len],
            k,
            n,
            k2,
            suh: exl3::f16_bytes(&row[2 * a..2 * (a + k)]),
            svh: exl3::f16_bytes(&row[2 * b..2 * (b + n)]),
        })
    }

    /// One expert on one row, f64: `down(act(gate(x), up(x)))` with the SwiGLU clamp.
    pub fn expert(&self, x: usize, row: &[f32]) -> Result<Vec<f64>> {
        let run = |m: Mat, v: &[f64]| -> Result<Vec<f64>> {
            let q = self.mat(x, m)?;
            Ok(exl3::forward64(v, 1, q.k, q.n, q.trellis, &q.suh, &q.svh, q.k2, self.cb))
        };
        let r64: Vec<f64> = row.iter().map(|v| *v as f64).collect();
        let (g, u) = (run(Mat::Gate, &r64)?, run(Mat::Up, &r64)?);
        let l = self.limit as f64;
        let a: Vec<f64> = g
            .iter()
            .zip(&u)
            .map(|(g, u)| {
                let g = g.min(l);
                let u = u.max(-l).min(l);
                g / (1.0 + (-g).exp()) * u
            })
            .collect();
        run(Mat::Down, &a)
    }
}

/// The f64 reference of the EXL3 encoding: `out[t] = sum_j w[t,j] * expert_{ids[t,j]}(x[t])` (`[T, H]`), ids outside
/// 0..E skipped. Pairs run in parallel through `par` (the caller's thread pool; `|n, f| (0..n).map(f).collect()` serially).
pub fn forward(
    v: &View,
    x: &[f32],
    ids: &[i32],
    wts: &[f32],
    t: usize,
    k: usize,
    par: &dyn Fn(usize, &(dyn Fn(usize) -> Result<Vec<f64>> + Sync)) -> Vec<Result<Vec<f64>>>,
) -> Result<Vec<f64>> {
    let h = v.h;
    let pairs = par(t * k, &|p| {
        let (ti, j) = (p / k, p % k);
        let e = ids[ti * k + j];
        if e < 0 || e as usize >= v.e {
            return Ok(Vec::new());
        }
        v.expert(e as usize, &x[ti * h..(ti + 1) * h])
    });
    let mut out = vec![0f64; t * h];
    for (p, y) in pairs.into_iter().enumerate() {
        let y = y?;
        if y.is_empty() {
            continue;
        }
        let w = wts[p] as f64;
        for (o, v) in out[(p / k) * h..(p / k + 1) * h].iter_mut().zip(&y) {
            *o += w * v;
        }
    }
    Ok(out)
}

// ---- admission cases ------------------------------------------------------------------------------------------------

/// A random EXL3 `MoeExperts` call: `t` tokens, `e` experts, `k` slots; widths per expert from `k2`; routing distinct
/// experts per row (the last slot picks expert `e - 1`, the shared expert's place, when `shared`).
#[allow(clippy::too_many_arguments)]
pub fn case(
    rng: &mut SplitMix64,
    t: usize,
    e: usize,
    k: usize,
    h: usize,
    i: usize,
    k2: &dyn Fn(usize, Mat) -> usize,
    cb: Codebook,
    shared: bool,
    xdt: DType,
    odt: DType,
    limit: f32,
) -> HostCall {
    let ex = random_experts(rng, e, h, i, k2, cb, limit, 1);
    let (ids, ws) = if shared {
        let (ids, ws) = crate::quant::cases::random_routing(rng, t, e - 1, k - 1);
        let (ids, ws) = (ids.to_i32().unwrap(), ws.to_f32());
        let mut ids2 = Vec::with_capacity(t * k);
        let mut ws2 = Vec::with_capacity(t * k);
        for r in 0..t {
            ids2.extend_from_slice(&ids[r * (k - 1)..(r + 1) * (k - 1)]);
            ids2.push((e - 1) as i32);
            ws2.extend(ws[r * (k - 1)..(r + 1) * (k - 1)].iter().map(|w| w * 1.5));
            ws2.push(1.0);
        }
        (HostTensor::from_i32(&[t, k], &ids2), HostTensor::from_f32(&[t, k], &ws2))
    } else {
        crate::quant::cases::random_routing(rng, t, e, k)
    };
    // x of unit size; the scales of `exl3::random` keep each projection's outputs about unit size, so the clamp at 10 bites
    // on the rare large gate / up value only (the cases also run with L = 1 to exercise it everywhere)
    let mut inputs = vec![HostTensor::from_f32_as(xdt, &[t, h], &rng.normal_vec(t * h, 1.0)), ids, ws];
    inputs.extend(ex.parts());
    HostCall { op: Op::MoeExperts, inputs, outputs: vec![OutSpec::new(odt, &[t, h])], attrs: OpAttrs { top_k: k, ..OpAttrs::default() } }
}

fn widths_label(k2s: &[usize]) -> String {
    k2s.iter().map(|k| format!("{}", *k as f64 / 2.0)).collect::<Vec<_>>().join("/")
}

/// The EXL3 cases of the `MoeExperts` sweep (appended to the NVFP4 ones; only kernels taking i16 weights accept them).
/// Every codebook; uniform and mixed widths (MUL1 half-integer ones included); the shared-expert slot; decode rows (1, 4, 8),
/// a batch past the decode path (40) and prompt-chunk rows (70, 130; non-quick 300); bf16 / f16 / f32 activations and
/// outputs; the clamp at L = 1 and none (inf). Non-quick: DeepSeek-V4.1's per-rank shape at TP=2 (H = 5120, I = 1152: K
/// chains of 9 Hadamard blocks), 3-bit routed + 5-bit shared.
pub fn sweep_cases(cfg: &SweepConfig, rng: &mut SplitMix64, push: &mut dyn FnMut(String, HostCall)) {
    // (t, e, k, h, i, widths cycled per expert, cb, shared, x, out, limit)
    type Row = (usize, usize, usize, usize, usize, Vec<usize>, Codebook, bool, DType, DType, f32);
    let (bf, f16, f32) = (DType::Bf16, DType::F16, DType::F32);
    let mut rows: Vec<Row> = vec![
        (1, 8, 2, 256, 128, vec![6], Codebook::Mul1, false, bf, bf, 10.0),
        (1, 8, 4, 384, 256, vec![4, 6], Codebook::Inst3, true, bf, f32, 10.0),
        (4, 12, 7, 256, 256, vec![5, 6, 7, 10], Codebook::Mul1, true, bf, bf, 1.0),
        (8, 16, 6, 256, 128, vec![8, 2, 16], Codebook::Mcg, false, f16, f16, f32::INFINITY),
        (40, 16, 4, 256, 128, vec![6, 10], Codebook::Mul1, true, bf, bf, 10.0),
        (70, 8, 3, 256, 256, vec![4, 6], Codebook::Inst3, false, bf, bf, 10.0),
        (130, 16, 6, 384, 128, vec![6, 3, 10], Codebook::Mul1, true, f32, f32, 1.0),
    ];
    if !cfg.quick {
        rows.extend([
            (1, 33, 7, 5120, 1152, vec![6], Codebook::Mul1, true, bf, bf, 10.0),
            (1, 33, 7, 5120, 1152, vec![4], Codebook::Mul1, true, bf, f32, 10.0),
            (4, 17, 7, 5120, 1152, vec![6], Codebook::Mul1, true, bf, bf, 10.0),
            (1, 16, 8, 1024, 1152, vec![12, 14], Codebook::Mcg, false, bf, bf, f32::INFINITY),
            (3, 16, 5, 640, 384, vec![8, 2], Codebook::Inst3, false, bf, f16, 10.0),
            (64, 24, 6, 1024, 1152, vec![6, 10], Codebook::Mul1, true, bf, bf, 10.0),
            (300, 16, 7, 256, 384, vec![6, 4, 10], Codebook::Mcg, true, bf, bf, 10.0),
            (33, 8, 8, 128, 128, vec![16], Codebook::Inst3, false, f16, f32, 10.0),
        ]);
    }
    for (t, e, k, h, i, ws, cb, shared, xdt, odt, limit) in rows {
        let last = *ws.last().unwrap();
        let wsc = ws.clone();
        // the shared expert (e - 1) takes the last width; routed experts cycle through the others
        let k2 = move |x: usize, _m: Mat| if shared && x == e - 1 { last } else { wsc[x % wsc.len().saturating_sub(shared as usize).max(1)] };
        let label = format!(
            "exl3 T={t} E={e} K={k} H={h} I={i} bits={} cb={} shared={shared} x={xdt} out={odt} limit={limit}",
            widths_label(&ws),
            cb.name()
        );
        push(label, case(rng, t, e, k, h, i, &k2, cb, shared, xdt, odt, limit));
    }
}

/// Plan-time label of an EXL3 call (rows clamped by `caps`, experts to 32 (at least K); H, I, K and dtypes exact). Widths
/// and the codebook are device data: the plan case grades 3-bit routed experts with a 5-bit last expert (the Mia
/// checkpoint's mix) in codebook `T % 3`.
pub fn label(call: &OpCall, caps: &ShapeCaps) -> String {
    let (t, h, i, e, k) = dims(call);
    format!(
        "exl3 T={} E={} K={k} H={h} I={i} x={} out={}",
        t.min(caps.rows),
        e.min(32).max(k),
        call.inputs[0].dtype,
        call.outputs[0].dtype
    )
}

pub fn case_from_call(call: &OpCall, caps: &ShapeCaps, rng: &mut SplitMix64) -> (HostCall, String) {
    let (t, h, i, e, k) = dims(call);
    let (t, e) = (t.min(caps.rows), e.min(32).max(k));
    let cb = Codebook::ALL[t % 3];
    let k2 = move |x: usize, _m: Mat| if x == e - 1 { 10 } else { 6 };
    let hc = case(rng, t, e, k, h, i, &k2, cb, k >= 2, call.inputs[0].dtype, call.outputs[0].dtype, 10.0);
    (hc, label(call, caps))
}

// ---- loader ---------------------------------------------------------------------------------------------------------

/// Load a layer's EXL3 experts into the op's encoding, as rank `rank` of `world` (intermediate-width split, see the module
/// doc). `experts[x] = [gate, up, down]` checkpoint prefixes (DeepSeek-V4.1: `layers.{l}.ffn.experts.{x}.w1` / `.w3` /
/// `.w2`; append `layers.{l}.ffn.shared_experts.*` as the last expert to run the shared expert in the same launch).
/// `hidden` is H, `inter` the checkpoint's (whole) expert width; every matrix may have its own width, all must share one
/// codebook. `limit` is the SwiGLU clamp (`f32::INFINITY` for none).
///
/// Returns the five weight inputs (3..8) as [`crate::quant::moe::Experts`]: `gate_up_w` = arena, `gate_up_s` = scales,
/// `down_w` = meta, `down_s` = codebook, `gscales` = limit. Only this rank's slice of each trellis is uploaded.
#[allow(clippy::too_many_arguments)]
pub fn load(
    backend: &dyn Backend,
    src: &dyn WeightSource,
    experts: &[[String; 3]],
    hidden: usize,
    inter: usize,
    rank: usize,
    world: usize,
    limit: f32,
) -> Result<crate::quant::moe::Experts> {
    use crate::weights::{Shard, ShardAxis};
    let e = experts.len();
    if e == 0 {
        return Err(Error::model("exl3 experts: no experts".to_string()));
    }
    let sh = |axis| Shard { axis, rank, world };
    // pass 1: validate every matrix, its rank slice and its place in the arena
    struct Plan {
        k2: usize,
        k: usize,
        n: usize,
        kr: (usize, usize),
        nr: (usize, usize),
        off16: usize,
    }
    let mut cb: Option<Codebook> = None;
    let mut plans: Vec<[Plan; 3]> = Vec::with_capacity(e);
    let mut arena16 = 0usize; // arena size in 16-byte units
    let mut il = 0usize; // the rank's width
    for names in experts {
        let mut one = |j: usize| -> Result<Plan> {
            let p = &names[j];
            let lm = exl3::layer_meta(&WeightMeta::new(src, p))?;
            let (wk, wn) = if j == 2 { (inter, hidden) } else { (hidden, inter) };
            if (lm.k, lm.n) != (wk, wn) {
                return Err(Error::model(format!("exl3 expert '{p}' is [K={}, N={}], expected [{wk}, {wn}]", lm.k, lm.n)));
            }
            match cb {
                None => cb = Some(lm.cb),
                Some(c) if c != lm.cb => {
                    return Err(Error::Unsupported(format!(
                        "exl3 experts: '{p}' uses the {} codebook, expert 0 {}: one codebook per layer",
                        lm.cb.name(),
                        c.name()
                    )))
                }
                _ => {}
            }
            let (kr, nr) = if world > 1 {
                exl3::shard_ranges(lm.k, lm.n, sh(if j == 2 { ShardAxis::Cols } else { ShardAxis::Rows }), p)?
            } else {
                ((0, lm.k), (0, lm.n))
            };
            let bytes = (kr.1 / TILE) * (nr.1 / TILE) * exl3::tile_bytes(lm.k2);
            let off16 = arena16;
            arena16 += bytes / 16;
            Ok(Plan { k2: lm.k2, k: lm.k, n: lm.n, kr, nr, off16 })
        };
        let g = one(0)?;
        let u = one(1)?;
        let d = one(2)?;
        il = g.nr.1;
        if off16_max(arena16) {
            return Err(Error::Unsupported(format!("exl3 experts: arena of {} bytes exceeds the 32 GiB the i32 offsets address", arena16 * 16)));
        }
        plans.push([g, u, d]);
    }
    let cb = cb.unwrap();
    let cols = 3 * hidden + 3 * il;
    let so = scale_offsets(hidden, il);
    let mut scales = vec![0u8; 2 * e * cols];
    let mut meta = vec![0i32; e * META_COLS];
    let arena = backend.alloc((arena16 * 16).max(16), MemKind::Device)?;
    for (x, (names, pl)) in experts.iter().zip(&plans).enumerate() {
        for (j, p) in pl.iter().enumerate() {
            let wm = WeightMeta::new(src, &names[j]);
            let tname = format!("{}.trellis", names[j]);
            let tb = src.bytes(&tname)?;
            if world == 1 {
                backend.upload(&arena, p.off16 * 16, tb, 0)?;
            } else {
                backend.upload(&arena, p.off16 * 16, &exl3::shard_trellis(tb, p.k, p.n, p.k2, p.kr, p.nr)?, 0)?;
            }
            meta[x * META_COLS + j] = p.off16 as i32;
            meta[x * META_COLS + 3 + j] = p.k2 as i32;
            let suh = exl3::scale_bytes(&wm, "suh", "su", p.k, p.kr.0, p.kr.1)?;
            let svh = exl3::scale_bytes(&wm, "svh", "sv", p.n, p.nr.0, p.nr.1)?;
            let (a, b) = [(so[0], so[2]), (so[1], so[3]), (so[4], so[5])][j];
            let row = &mut scales[2 * x * cols..2 * (x + 1) * cols];
            row[2 * a..2 * a + suh.len()].copy_from_slice(&suh);
            row[2 * b..2 * b + svh.len()].copy_from_slice(&svh);
        }
    }
    let meta_b: Vec<u8> = meta.iter().flat_map(|v| v.to_le_bytes()).collect();
    Ok(crate::quant::moe::Experts {
        gate_up_w: TensorRef::new(arena, DType::I16, &[arena16 * 8]),
        gate_up_s: TensorRef::new(upload_new(backend, &scales)?, DType::F16, &[e, cols]),
        down_w: TensorRef::new(upload_new(backend, &meta_b)?, DType::I32, &[e, META_COLS]),
        down_s: TensorRef::new(upload_new(backend, &(cb as i32).to_le_bytes())?, DType::I32, &[1]),
        gscales: TensorRef::new(upload_new(backend, &limit.to_le_bytes())?, DType::F32, &[1]),
    })
}

fn off16_max(units: usize) -> bool {
    units > i32::MAX as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    struct P(usize);
    impl crate::backend::BufferImpl for P {
        fn bytes(&self) -> usize {
            self.0
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    fn calls(quick: bool) -> Vec<(String, OpCall)> {
        let mut v = Vec::new();
        let mut rng = SplitMix64::new(3);
        sweep_cases(&SweepConfig { quick, ..SweepConfig::default() }, &mut rng, &mut |label, hc| {
            let mk = |t: &HostTensor| TensorRef::new(crate::backend::Buffer::new(P(t.nbytes())), t.dtype, &t.shape);
            let outs = hc.outputs.iter().map(|o| mk(&HostTensor::zeros(o.dtype, &o.shape))).collect();
            v.push((label, OpCall::new(hc.op, hc.inputs.iter().map(mk).collect(), outs, hc.attrs.clone())));
        });
        v
    }

    #[test]
    fn every_case_validates_and_reports_its_shape() {
        for (label, call) in calls(true) {
            call.check().unwrap_or_else(|e| panic!("{label}: {e}"));
            assert!(is_exl3(&call));
            let s = ShapeInfo::from_call(&call);
            let (t, h, i, e, k) = dims(&call);
            assert_eq!((s.m, s.k, s.n, s.experts, s.top_k, s.weight), (t, h, i, e, k, Some(DType::I16)), "{label}");
        }
    }

    #[test]
    fn a_wrong_scales_width_is_refused() {
        let (label, mut call) = calls(true).remove(0);
        let s = &call.inputs[4];
        call.inputs[4] = TensorRef::new(s.buf.clone(), DType::F16, &[s.shape[0], s.shape[1] - 1]);
        assert!(call.check().is_err(), "{label}");
    }

    /// The f64 reference against a direct dequantised computation (weights from `exl3::dequant_t`), clamp included.
    #[test]
    fn reference_matches_dequantised_math() {
        let mut rng = SplitMix64::new(9);
        let (t, e, k, h, i) = (2usize, 4usize, 3usize, 256usize, 128usize);
        for limit in [f32::INFINITY, 0.5] {
            let hc = case(&mut rng, t, e, k, h, i, &|x, _| [4, 5, 6, 10][x], Codebook::Mul1, true, DType::F32, DType::F32, limit);
            let inp = &hc.inputs;
            let v = View::new(e, h, i, &inp[3].data, &inp[4].data, &inp[5].data, &inp[6].data, &inp[7].data).unwrap();
            let x = inp[0].to_f32();
            let (ids, ws) = (inp[1].to_i32().unwrap(), inp[2].to_f32());
            let got = forward(&v, &x, &ids, &ws, t, k, &|n, f| (0..n).map(f).collect()).unwrap();
            let l = limit as f64;
            for r in 0..t {
                let mut want = vec![0f64; h];
                for j in 0..k {
                    let ex = ids[r * k + j] as usize;
                    let w = |m: Mat| {
                        let q = v.mat(ex, m).unwrap();
                        exl3::dequant_t(q.trellis, &q.suh, &q.svh, q.k, q.n, q.k2, Codebook::Mul1)
                    };
                    let (wg, wu, wd) = (w(Mat::Gate), w(Mat::Up), w(Mat::Down));
                    let xr = &x[r * h..(r + 1) * h];
                    let a: Vec<f64> = (0..i)
                        .map(|c| {
                            let g: f64 = (0..h).map(|q| xr[q] as f64 * wg[c * h + q]).sum::<f64>().min(l);
                            let u: f64 = (0..h).map(|q| xr[q] as f64 * wu[c * h + q]).sum::<f64>().max(-l).min(l);
                            g / (1.0 + (-g).exp()) * u
                        })
                        .collect();
                    for (o, wo) in want.iter_mut().enumerate() {
                        *wo += ws[r * k + j] as f64 * (0..i).map(|c| a[c] * wd[o * i + c]).sum::<f64>();
                    }
                }
                for (g, w) in got[r * h..(r + 1) * h].iter().zip(&want) {
                    assert!((g - w).abs() < 1e-9 * (1.0 + w.abs()), "limit {limit}: {g} vs {w}");
                }
            }
        }
    }

    #[test]
    fn a_bad_offset_or_width_is_an_error() {
        let mut rng = SplitMix64::new(10);
        let hc = case(&mut rng, 1, 2, 2, 128, 128, &|_, _| 6, Codebook::Inst3, false, DType::F32, DType::F32, 10.0);
        let inp = &hc.inputs;
        let mut meta = inp[5].to_i32().unwrap();
        meta[3] = 3; // 1.5 bits needs mul1
        let mb: Vec<u8> = meta.iter().flat_map(|v| v.to_le_bytes()).collect();
        let v = View::new(2, 128, 128, &inp[3].data, &inp[4].data, &mb, &inp[6].data, &inp[7].data).unwrap();
        assert!(v.mat(0, Mat::Gate).is_err());
        let mut meta = inp[5].to_i32().unwrap();
        meta[META_COLS + 2] = 1 << 28;
        let mb: Vec<u8> = meta.iter().flat_map(|v| v.to_le_bytes()).collect();
        let v = View::new(2, 128, 128, &inp[3].data, &inp[4].data, &mb, &inp[6].data, &inp[7].data).unwrap();
        assert!(v.mat(1, Mat::Down).is_err());
    }
}
