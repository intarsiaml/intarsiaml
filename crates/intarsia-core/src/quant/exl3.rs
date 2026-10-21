//! EXL3 weights (ExLlamaV3's trellis quantisation, QTIP-style; MIT, Copyright (c) 2025 Turboderp). Ported from the
//! MIT numpy reference decoder of TensorFold (`tensorfold/cuda/exl3/format.py`) and checked against exllamav3's own
//! reconstruction on real tensors (`crates/intarsia-backend-cpu/tests/quant_exl3.rs`).
//!
//! ```text
//! {p}.trellis  i16 [K/16, N/16, 16*bits]  one 16x16 tile of W_q per (k tile, n tile), 16*bits words of a bit stream
//! {p}.suh      f16 [K]                    input-side scales (sign and magnitude); old checkpoints: {p}.su i16 [K/16]
//! {p}.svh      f16 [N]                    output-side scales;                       old checkpoints: {p}.sv i16 [N/16]
//! {p}.mcg      i32 scalar 0xCBAC1FED      codebook marker: MCG; {p}.mul1 (0x83DCD12D): MUL1; neither: 3INST
//! ```
//!
//! The checkpoint stores the weight as `[K, N]` (in, out; `y = x @ W`):
//! `W = diag(suh) . H_K . W_q . H_N . diag(svh)`, `H` the 128-point Sylvester Hadamard / sqrt(128) applied blockwise,
//! so `K` and `N` are multiples of 128. Value `p` (0..256) of a tile is a 16-bit window of the tile's bit stream
//! ending at bit `E(p)` (`stream_end`), read as a ring of 32-bit little-endian words with the most significant bit
//! first; the 16-bit state maps to an fp16 value through a multiply-based codebook (`decode`); `tile_pos` places it
//! in the tile. Bits per weight are 1..8, or 1.5 / 2.5 / 3.5 with the MUL1 codebook; we carry them as `k2 = 2 * bits`.
//!
//! In the engine the layer is a [`QTensor`] `[n, k]` (out, in; `y = x @ W^T`) with parts
//! `[trellis i16 [K/16, N/16, 8*k2], suh f16 [K], svh f16 [N], codebook i32 [1]]` (packed `su`/`sv` are unpacked to
//! f16 at load). The codebook is data, not a launch parameter: a kernel reads it on the device and branches once
//! (uniformly), so launches never synchronise; `k2` comes from the trellis shape.

use crate::admission::{Case, SweepConfig};
use crate::backend::{upload_new, Backend, TensorRef};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::exec::{HostCall, OutSpec};
use crate::ops::{Op, OpAttrs, OpCall, ShapeInfo};
use crate::quant::{HostQTensor, QTensor, QuantCfg, QuantFormat, QuantId, QuantLoadOpts, WeightMeta, EXL3};
use crate::rng::SplitMix64;
use crate::tensor::HostTensor;
use half::f16;
use std::sync::OnceLock;

/// Hadamard block (both sides).
pub const HAD: usize = 128;
/// Tile edge.
pub const TILE: usize = 16;

pub const INST3_MUL: u32 = 89_226_354;
pub const INST3_ADD: u32 = 64_248_484;
pub const MCG_MUL: u32 = 0xCBAC_1FED;
pub const MUL1_MUL: u32 = 0x83DC_D12D;
const MASK: u32 = 0x8FFF_8FFF;
const FLIP: u32 = 0x3B60_3B60;
/// fp16 bit patterns of MUL1's scale (1/147.7) and bias (-10.39).
const MUL1_SCALE: u16 = 0x1EEE;
const MUL1_BIAS: u16 = 0xC931;

/// The three procedural codebooks; the discriminant is the value of the codebook part (`i32 [1]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Codebook {
    Inst3 = 0,
    Mcg = 1,
    Mul1 = 2,
}

impl Codebook {
    pub const ALL: [Codebook; 3] = [Codebook::Inst3, Codebook::Mcg, Codebook::Mul1];
    pub fn from_i32(v: i32) -> Option<Codebook> {
        Codebook::ALL.into_iter().find(|c| *c as i32 == v)
    }
    pub fn name(self) -> &'static str {
        match self {
            Codebook::Inst3 => "3inst",
            Codebook::Mcg => "mcg",
            Codebook::Mul1 => "mul1",
        }
    }
    pub fn parse(s: &str) -> Option<Codebook> {
        Codebook::ALL.into_iter().find(|c| c.name().eq_ignore_ascii_case(s))
    }
    /// The checkpoint's marker tensor suffix and the value it holds (`None` for 3INST, which has no marker).
    pub fn marker(self) -> Option<(&'static str, u32)> {
        match self {
            Codebook::Inst3 => None,
            Codebook::Mcg => Some(("mcg", MCG_MUL)),
            Codebook::Mul1 => Some(("mul1", MUL1_MUL)),
        }
    }
}

/// Is `k2` (twice the bits per weight) a width ExLlamaV3 writes with codebook `cb`? 2..=16 even (1..8 bits); 3, 5, 7
/// (1.5, 2.5, 3.5 bits) with MUL1 only.
pub fn valid_k2(k2: usize, cb: Codebook) -> bool {
    (k2 % 2 == 0 && (2..=16).contains(&k2)) || (matches!(k2, 3 | 5 | 7) && cb == Codebook::Mul1)
}

/// Any width at all (codebook not known yet).
pub fn valid_k2_any(k2: usize) -> bool {
    valid_k2(k2, Codebook::Mul1)
}

fn h(bits: u16) -> f64 {
    f16::from_bits(bits).to_f64()
}

/// The fp16 value (bits) of 16-bit state `s` in codebook `cb`, computed (one fp16 rounding, as `__hadd` / `__hfma`).
pub fn decode_state(cb: Codebook, s: u16) -> u16 {
    let s = s as u32;
    match cb {
        Codebook::Mul1 => {
            let x = s.wrapping_mul(MUL1_MUL);
            let sum = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24);
            // fp16 1024 + sum (exact: sum <= 1020); h * scale + bias is exact in f64, so one rounding like __hfma
            let hv = 1024.0 + sum as f64;
            f16::from_f64(hv * h(MUL1_SCALE) + h(MUL1_BIAS)).to_bits()
        }
        _ => {
            let x = if cb == Codebook::Mcg { s.wrapping_mul(MCG_MUL) } else { s.wrapping_mul(INST3_MUL).wrapping_add(INST3_ADD) };
            let x = (x & MASK) ^ FLIP;
            // the sum of two fp16 values is exact in f64: one rounding like __hadd
            f16::from_f64(h((x & 0xFFFF) as u16) + h((x >> 16) as u16)).to_bits()
        }
    }
}

/// `decode_state` for every state, per codebook (built once, 128 KiB each).
pub fn codebook_table(cb: Codebook) -> &'static [u16] {
    static T: OnceLock<[Vec<u16>; 3]> = OnceLock::new();
    let t = T.get_or_init(|| Codebook::ALL.map(|c| (0..=u16::MAX).map(|s| decode_state(c, s)).collect()));
    &t[cb as usize]
}

/// `E(p)`: one past the last stream bit of value `p`'s 16-bit window (format.py `stream_ends`).
#[inline]
pub fn stream_end(k2: usize, p: usize) -> usize {
    let p1 = p + 1;
    if k2 % 2 == 1 {
        (p1 * k2 - (p1 & 1)) / 2
    } else {
        p1 * (k2 / 2)
    }
}

/// 32-bit words of one tile (256 values of `k2 / 2` bits).
pub const fn tile_words(k2: usize) -> usize {
    4 * k2
}

/// Bytes of one tile.
pub const fn tile_bytes(k2: usize) -> usize {
    16 * k2
}

/// (row = k, column = n) in its 16x16 tile of a tile's value `p` (stream order); lane `p / 8` of the mma fragment.
#[inline]
pub fn tile_pos(p: usize) -> (usize, usize) {
    let (lane, j) = (p / 8, p % 8);
    (2 * (lane % 4) + (j & 1) + 8 * ((j >> 1) & 1), lane / 4 + 8 * (j >> 2))
}

/// The 256 16-bit states of one tile (`tile`: its `16 * k2` bytes), in stream order.
pub fn tile_states(tile: &[u8], k2: usize) -> [u16; 256] {
    let nw = tile_words(k2);
    debug_assert_eq!(tile.len(), 4 * nw);
    let word = |i: usize| u32::from_le_bytes([tile[4 * i], tile[4 * i + 1], tile[4 * i + 2], tile[4 * i + 3]]) as u64;
    let ring = 32 * nw;
    let mut s = [0u16; 256];
    for (p, out) in s.iter_mut().enumerate() {
        let first = stream_end(k2, p) + ring - 16; // the window's first bit, made non-negative
        let (i0, off) = ((first / 32) % nw, first % 32);
        let pair = (word(i0) << 32) | word((i0 + 1) % nw);
        *out = ((pair >> (48 - off)) & 0xFFFF) as u16;
    }
    s
}

/// One tile of `W_q` as fp16 bits, row-major 16x16 (`[r * 16 + c]`, r along K, c along N).
pub fn tile_values(tile: &[u8], k2: usize, cb: Codebook) -> [u16; 256] {
    let table = codebook_table(cb);
    let st = tile_states(tile, k2);
    let mut v = [0u16; 256];
    for (p, s) in st.iter().enumerate() {
        let (r, c) = tile_pos(p);
        v[r * TILE + c] = table[*s as usize];
    }
    v
}

/// `W_q` `[K, N]` in the rotated basis as fp16 bits (exllamav3 `reconstruct`) from the trellis bytes.
pub fn unpack(trellis: &[u8], k: usize, n: usize, k2: usize, cb: Codebook) -> Vec<u16> {
    let (kt, nt, tb) = (k / TILE, n / TILE, tile_bytes(k2));
    assert_eq!(trellis.len(), kt * nt * tb, "trellis bytes for [{k}, {n}] at k2={k2}");
    let mut w = vec![0u16; k * n];
    for a in 0..kt {
        for b in 0..nt {
            let off = (a * nt + b) * tb;
            let v = tile_values(&trellis[off..off + tb], k2, cb);
            for r in 0..TILE {
                w[(a * TILE + r) * n + b * TILE..(a * TILE + r) * n + b * TILE + TILE].copy_from_slice(&v[r * TILE..r * TILE + TILE]);
            }
        }
    }
    w
}

/// In-place 128-point Walsh-Hadamard transform (Sylvester order), scaled by 1/sqrt(128).
pub fn fwht128(v: &mut [f64]) {
    debug_assert_eq!(v.len(), HAD);
    let mut hlen = 1;
    while hlen < HAD {
        for i in (0..HAD).step_by(2 * hlen) {
            for j in i..i + hlen {
                let (a, b) = (v[j], v[j + hlen]);
                v[j] = a + b;
                v[j + hlen] = a - b;
            }
        }
        hlen *= 2;
    }
    let s = 1.0 / (HAD as f64).sqrt();
    for x in v.iter_mut() {
        *x *= s;
    }
}

/// f16 bytes to f64.
pub fn f16_bytes(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(2).map(|c| h(u16::from_le_bytes([c[0], c[1]]))).collect()
}

/// Packed sign vector (`su` / `sv`, i16 `[n/16]`) to f16 bytes of +-1: bit b of word w set means element 16w + b is -1.
pub fn unpack_signs(packed: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(packed.len() * 16);
    for c in packed.chunks_exact(2) {
        let w = u16::from_le_bytes([c[0], c[1]]);
        for b in 0..16 {
            let v: u16 = if (w >> b) & 1 == 1 { 0xBC00 } else { 0x3C00 };
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// The layer's weight as the engine holds it: `W^T` `[N, K]` row-major, f64
/// (`W^T[n, k] = svh[n] * (H_K W_q H_N)[k, n] * suh[k]`).
pub fn dequant_t(trellis: &[u8], suh: &[f64], svh: &[f64], k: usize, n: usize, k2: usize, cb: Codebook) -> Vec<f64> {
    let wq = unpack(trellis, k, n, k2, cb);
    // transposed: row n holds column n of W_q over k
    let mut wt = vec![0f64; n * k];
    for kk in 0..k {
        for nn in 0..n {
            wt[nn * k + kk] = h(wq[kk * n + nn]);
        }
    }
    // H_K on the left: along k (rows of W_q), i.e. contiguous blocks of each row of W^T
    for row in wt.chunks_exact_mut(k) {
        for blk in row.chunks_exact_mut(HAD) {
            fwht128(blk);
        }
        for (x, s) in row.iter_mut().zip(suh) {
            *x *= s;
        }
    }
    // H_N on the right: along n, a strided column of W^T per k
    let mut col = vec![0f64; HAD];
    for nb in 0..n / HAD {
        for kk in 0..k {
            for j in 0..HAD {
                col[j] = wt[(nb * HAD + j) * k + kk];
            }
            fwht128(&mut col);
            for j in 0..HAD {
                wt[(nb * HAD + j) * k + kk] = col[j] * svh[nb * HAD + j];
            }
        }
    }
    wt
}

/// `x * suh` rotated blockwise: the kernels' rotated input `[m, k]`, f64 (no fp16 rounding).
pub fn rotate_input(x: &[f32], m: usize, k: usize, suh: &[f64]) -> Vec<f64> {
    let mut xh: Vec<f64> = x.iter().zip(suh.iter().cycle()).map(|(a, s)| *a as f64 * s).collect();
    debug_assert_eq!(xh.len(), m * k);
    for blk in xh.chunks_exact_mut(HAD) {
        fwht128(blk);
    }
    xh
}

/// Output columns `[nb*128, nb*128 + 128)` of `y = x @ W` (f64, `[m, 128]`) from the rotated input `xh` `[m, k]`:
/// `(xh @ W_q[:, block]) H * svh[block]`. `trellis` is the whole tensor (`[k/16, n/16, 16*k2]` bytes).
pub fn forward_block(xh: &[f64], m: usize, k: usize, n: usize, trellis: &[u8], svh: &[f64], nb: usize, k2: usize, cb: Codebook) -> Vec<f64> {
    let (nt, tb) = (n / TILE, tile_bytes(k2));
    let table = codebook_table(cb);
    let mut acc = vec![0f64; m * HAD];
    let mut wv = [0f64; 256];
    for a in 0..k / TILE {
        for t in 0..HAD / TILE {
            let b = nb * (HAD / TILE) + t;
            let off = (a * nt + b) * tb;
            let st = tile_states(&trellis[off..off + tb], k2);
            for (p, s) in st.iter().enumerate() {
                let (r, c) = tile_pos(p);
                wv[r * TILE + c] = h(table[*s as usize]);
            }
            for mi in 0..m {
                let xr = &xh[mi * k + a * TILE..mi * k + a * TILE + TILE];
                let out = &mut acc[mi * HAD + t * TILE..mi * HAD + t * TILE + TILE];
                for (r, xv) in xr.iter().enumerate() {
                    for c in 0..TILE {
                        out[c] += xv * wv[r * TILE + c];
                    }
                }
            }
        }
    }
    for (mi, row) in acc.chunks_exact_mut(HAD).enumerate() {
        let _ = mi;
        fwht128(row);
        for (j, v) in row.iter_mut().enumerate() {
            *v *= svh[nb * HAD + j];
        }
    }
    acc
}

/// `y = x @ W` `[m, n]` in f64 (the reference of `Op::MatmulExl3`), single-threaded.
pub fn forward(x: &[f32], m: usize, k: usize, n: usize, trellis: &[u8], suh: &[f64], svh: &[f64], k2: usize, cb: Codebook) -> Vec<f64> {
    let xh = rotate_input(x, m, k, suh);
    let mut y = vec![0f64; m * n];
    for nb in 0..n / HAD {
        let blk = forward_block(&xh, m, k, n, trellis, svh, nb, k2, cb);
        for mi in 0..m {
            y[mi * n + nb * HAD..mi * n + nb * HAD + HAD].copy_from_slice(&blk[mi * HAD..(mi + 1) * HAD]);
        }
    }
    y
}

/// `k2` from a trellis shape `[K/16, N/16, 8*k2]` (its last dim is `16 * bits` i16 words).
pub fn k2_of(shape: &[usize]) -> Option<usize> {
    let last = *shape.last()?;
    (shape.len() == 3 && last % 8 == 0 && valid_k2_any(last / 8)).then_some(last / 8)
}

/// (k, n, k2, codebook) of host parts `[trellis, suh, svh, codebook]`.
fn host_meta(parts: &[HostTensor]) -> (usize, usize, usize, Codebook) {
    let s = &parts[0].shape;
    let cbv = i32::from_le_bytes(parts[3].data[..4].try_into().unwrap());
    (16 * s[0], 16 * s[1], k2_of(s).expect("valid trellis"), Codebook::from_i32(cbv).expect("valid codebook"))
}

/// `[N, K]` f32 weights of host parts `[trellis, suh, svh, codebook]` (for `HostQTensor::dequant`).
pub fn dequant_parts(parts: &[HostTensor]) -> Vec<f32> {
    let (k, n, k2, cb) = host_meta(parts);
    dequant_t(&parts[0].data, &f16_bytes(&parts[1].data), &f16_bytes(&parts[2].data), k, n, k2, cb).into_iter().map(|v| v as f32).collect()
}

/// The `exl3` quantisation format (ExLlamaV3 checkpoints, `quant_method = "exl3"`).
pub struct Exl3;

impl QuantFormat for Exl3 {
    fn id(&self) -> QuantId {
        EXL3
    }

    fn detect(&self, meta: &WeightMeta) -> bool {
        meta.info("trellis").is_some_and(|t| t.dtype == DType::I16 && t.shape.len() == 3)
    }

    fn load(&self, meta: &WeightMeta, backend: &dyn Backend, opts: &QuantLoadOpts) -> Result<QTensor> {
        opts.refuse_shard("EXL3", meta.prefix)?;
        let p = meta.prefix;
        let fail = |why: String| Error::model(format!("exl3 '{p}': {why}"));
        let ti = meta.info("trellis").ok_or_else(|| fail("no .trellis".into()))?;
        if ti.dtype != DType::I16 || ti.shape.len() != 3 {
            return Err(fail(format!("trellis must be i16 [K/16, N/16, 16*bits], got {} {:?}", ti.dtype, ti.shape)));
        }
        let k2 = k2_of(&ti.shape).ok_or_else(|| fail(format!("trellis last dim {} is no EXL3 width", ti.shape[2])))?;
        let (k, n) = (16 * ti.shape[0], 16 * ti.shape[1]);
        if k % HAD != 0 || n % HAD != 0 {
            return Err(fail(format!("K={k} and N={n} must be multiples of {HAD} (the Hadamard blocks)")));
        }
        if meta.info("bias").is_some() {
            return Err(Error::Unsupported(format!("exl3 '{p}': a bias next to the trellis is not implemented")));
        }
        // the codebook: at most one marker, and it must hold the multiplier exllamav3 writes
        let mut cb = Codebook::Inst3;
        for c in [Codebook::Mcg, Codebook::Mul1] {
            let (suffix, want) = c.marker().unwrap();
            if let Some(mi) = meta.info(suffix) {
                if cb != Codebook::Inst3 {
                    return Err(fail("both mcg and mul1 markers".into()));
                }
                let b = meta.src.bytes(&format!("{p}.{suffix}"))?;
                if mi.dtype != DType::I32 || b.len() != 4 || u32::from_le_bytes([b[0], b[1], b[2], b[3]]) != want {
                    return Err(fail(format!(".{suffix} marker must be the i32 scalar {want:#x}")));
                }
                cb = c;
            }
        }
        if !valid_k2(k2, cb) {
            return Err(fail(format!("{}-bit tiles need the mul1 codebook, found {}", k2 as f64 / 2.0, cb.name())));
        }
        let scales = |full: &str, packed: &str, len: usize| -> Result<TensorRef> {
            if let Some(i) = meta.info(full) {
                if i.dtype != DType::F16 || i.shape != [len] {
                    return Err(fail(format!(".{full} must be f16 [{len}], got {} {:?}", i.dtype, i.shape)));
                }
                return crate::weights::load_tensor(backend, meta.src, &format!("{p}.{full}"), Some(&[len]));
            }
            let i = meta.info(packed).ok_or_else(|| fail(format!("no .{full} or .{packed}")))?;
            if i.dtype != DType::I16 || i.shape != [len / 16] {
                return Err(fail(format!(".{packed} must be i16 [{}], got {} {:?}", len / 16, i.dtype, i.shape)));
            }
            let f = unpack_signs(meta.src.bytes(&format!("{p}.{packed}"))?);
            Ok(TensorRef::new(upload_new(backend, &f)?, DType::F16, &[len]))
        };
        let suh = scales("suh", "su", k)?;
        let svh = scales("svh", "sv", n)?;
        let tr = crate::weights::load_tensor(backend, meta.src, &format!("{p}.trellis"), None)?;
        let cbt = TensorRef::new(upload_new(backend, &(cb as i32).to_le_bytes())?, DType::I32, &[1]);
        Ok(QTensor { format: EXL3, n, k, parts: vec![tr, suh, svh, cbt], act_quant: false })
    }

    fn dequant_ref(&self, q: &QTensor, backend: &dyn Backend) -> Result<HostTensor> {
        Ok(HostTensor::from_f32(&[q.n, q.k], &q.download(backend)?.dequant()))
    }

    /// No EXL3 quantiser here (the format needs ExLlamaV3's trellis search); `random` makes test tensors.
    fn quantize(&self, _t: &HostTensor, _cfg: &QuantCfg) -> Option<HostQTensor> {
        None
    }
}

/// A random EXL3 layer `[n, k]` (any trellis bits are a valid stream): scales of either sign with magnitudes in
/// [0.5, 1.5), the input side divided by sqrt(k), so `x @ W` of a unit-normal `x` has outputs of about unit size.
pub fn random(rng: &mut SplitMix64, n: usize, k: usize, k2: usize, cb: Codebook) -> HostQTensor {
    assert!(k % HAD == 0 && n % HAD == 0 && valid_k2(k2, cb));
    let tr: Vec<u8> = (0..(k / TILE) * (n / TILE) * tile_bytes(k2)).map(|_| rng.next_u64() as u8).collect();
    let mut sc = |len: usize, mul: f32| -> HostTensor {
        let v: Vec<f32> = (0..len)
            .map(|_| {
                let u = rng.next_u64();
                let mag = 0.5 + (u >> 11) as f32 / (1u64 << 53) as f32;
                if u & 1 == 1 { -mag * mul } else { mag * mul }
            })
            .collect();
        HostTensor::from_f32_as(DType::F16, &[len], &v)
    };
    let suh = sc(k, 1.0 / (k as f32).sqrt());
    let svh = sc(n, 1.0);
    HostQTensor {
        format: EXL3,
        n,
        k,
        parts: vec![
            HostTensor { dtype: DType::I16, shape: vec![k / TILE, n / TILE, 8 * k2], data: tr },
            suh,
            svh,
            HostTensor { dtype: DType::I32, shape: vec![1], data: (cb as i32).to_le_bytes().to_vec() },
        ],
        act_quant: false,
    }
}

// ---- Op::MatmulExl3: validation, shape facts, admission cases --------------------------------------------------

fn bad<T>(m: String) -> Result<T> {
    Err(Error::invalid(format!("{}: {m}", Op::MatmulExl3)))
}

/// Validate `Op::MatmulExl3`: x `[M,K]`, trellis i16 `[K/16,N/16,8*k2]`, suh f16 `[K]`, svh f16 `[N]`, codebook i32
/// `[1]` -> out `[M,N]` (bf16/f16/f32 activations and output; K, N multiples of 128).
pub fn check(call: &OpCall) -> Result<()> {
    let (i, o) = (&call.inputs, &call.outputs);
    let act = |t: &TensorRef| matches!(t.dtype, DType::Bf16 | DType::F16 | DType::F32);
    if i[0].shape.len() != 2 || o[0].shape.len() != 2 {
        return bad(format!("x and out must be 2-D, got {:?} and {:?}", i[0].shape, o[0].shape));
    }
    if !act(&i[0]) || !act(&o[0]) {
        return bad("x and out must be bf16/f16/f32".into());
    }
    let (m, k) = (i[0].shape[0], i[0].shape[1]);
    let t = &i[1];
    let Some(k2) = k2_of(&t.shape).filter(|_| t.dtype == DType::I16) else {
        return bad(format!("trellis must be i16 [K/16,N/16,8*k2] with k2 in 2..=16 or 3/5/7, got {} {:?}", t.dtype, t.shape));
    };
    let n = 16 * t.shape[1];
    if 16 * t.shape[0] != k {
        return bad(format!("trellis covers K={} but x has K={k}", 16 * t.shape[0]));
    }
    if k % HAD != 0 || n % HAD != 0 {
        return bad(format!("K={k} and N={n} must be multiples of {HAD}"));
    }
    if i[2].dtype != DType::F16 || i[2].shape != [k] {
        return bad(format!("suh must be f16 [{k}], got {} {:?}", i[2].dtype, i[2].shape));
    }
    if i[3].dtype != DType::F16 || i[3].shape != [n] {
        return bad(format!("svh must be f16 [{n}], got {} {:?}", i[3].dtype, i[3].shape));
    }
    if i[4].dtype != DType::I32 || i[4].shape != [1] {
        return bad(format!("codebook must be i32 [1], got {} {:?}", i[4].dtype, i[4].shape));
    }
    if o[0].shape != [m, n] {
        return bad(format!("out {:?} != [{m},{n}]", o[0].shape));
    }
    let _ = k2;
    Ok(())
}

/// `ShapeInfo` of `Op::MatmulExl3`: m, k, n, weight i16, and `head_dim` = k2 (twice the bits per weight) so kernel
/// predicates and labels see the width.
pub fn shape(call: &OpCall, s: &mut ShapeInfo) {
    let i = &call.inputs;
    s.m = i[0].shape[0];
    s.k = i[0].shape[1];
    s.n = 16 * i[1].shape[1];
    s.weight = Some(i[1].dtype);
    s.head_dim = k2_of(&i[1].shape).unwrap_or(0);
}

pub fn case(rng: &mut SplitMix64, m: usize, n: usize, k: usize, k2: usize, cb: Codebook, xdt: DType, odt: DType) -> HostCall {
    let q = random(rng, n, k, k2, cb);
    let x = HostTensor::from_f32_as(xdt, &[m, k], &rng.normal_vec(m * k, 1.0));
    let mut inputs = vec![x];
    inputs.extend(q.parts);
    HostCall { op: Op::MatmulExl3, inputs, outputs: vec![OutSpec::new(odt, &[m, n])], attrs: OpAttrs::default() }
}

fn bits_label(k2: usize) -> String {
    format!("{}", k2 as f64 / 2.0)
}

/// Every width / codebook pair exllamav3 writes: 1..8 bits with each codebook, 1.5 / 2.5 / 3.5 with MUL1.
pub fn widths() -> Vec<(usize, Codebook)> {
    let mut v = Vec::new();
    for k2 in [2usize, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16] {
        for cb in Codebook::ALL {
            if valid_k2(k2, cb) {
                v.push((k2, cb));
            }
        }
    }
    v
}

/// The `intarsia kernels test` sweep of `Op::MatmulExl3`: every width and codebook at a small shape, the decode and
/// prefill row counts, both activation dtypes, and (non-quick) the Qwen3-8B projection shapes at the common widths.
pub fn sweep(cfg: &SweepConfig) -> Vec<Case> {
    let mut rng = SplitMix64::new(cfg.seed ^ (Op::MatmulExl3 as u64).wrapping_mul(0x9E37) ^ 0x51ED);
    let mut cases = Vec::new();
    let mut push = |rng: &mut SplitMix64, m: usize, n: usize, k: usize, k2: usize, cb: Codebook, xdt: DType, odt: DType| {
        let label = format!("M={m} N={n} K={k} bits={} cb={} x={xdt} out={odt}", bits_label(k2), cb.name());
        cases.push(Case { label, call: case(rng, m, n, k, k2, cb, xdt, odt) });
    };
    let ms: &[usize] = if cfg.quick { &[1, 5, 40] } else { &[1, 3, 8, 17, 40, 256, 300] };
    for (k2, cb) in widths() {
        for &m in ms {
            push(&mut rng, m, 256, 384, k2, cb, DType::Bf16, DType::Bf16);
        }
        push(&mut rng, 2, 128, 256, k2, cb, DType::F32, DType::F32);
    }
    if !cfg.quick {
        // Qwen3-8B (turboderp 4.0bpw: 4-bit 3INST projections, 6-bit head): q/o, k/v, gate/up, down; and the head
        for &(n, k) in &[(4096usize, 4096usize), (1024, 4096), (12288, 4096), (4096, 12288)] {
            for (k2, cb) in [(8usize, Codebook::Inst3), (6, Codebook::Mcg), (5, Codebook::Mul1)] {
                for &m in &[1usize, 4] {
                    push(&mut rng, m, n, k, k2, cb, DType::Bf16, DType::Bf16);
                }
            }
        }
        for &m in &[1usize, 4] {
            push(&mut rng, m, 151_936, 4096, 12, Codebook::Inst3, DType::Bf16, DType::F32);
        }
    }
    cases
}

/// Plan-time label (rows and N clamped by `caps`; K and the width exact). The codebook is device data the plan
/// cannot see; the case grades one codebook chosen by the row count (`case_from_call`), the sweep grades all three.
pub fn label(call: &OpCall, caps: &crate::admission::ShapeCaps) -> String {
    let (i, o) = (&call.inputs, &call.outputs);
    let k2 = k2_of(&i[1].shape).unwrap_or(0);
    format!(
        "{} M={} N={} K={} bits={} x={} out={}",
        Op::MatmulExl3,
        i[0].shape[0].min(caps.rows),
        (16 * i[1].shape[1]).min(caps.n.max(HAD)) / HAD * HAD,
        i[0].shape[1],
        bits_label(k2),
        i[0].dtype,
        o[0].dtype
    )
}

/// Plan-time admission case of `call` (see [`label`]): rows `m` grade codebook `m % 3` (MUL1 for half-integer widths).
pub fn case_from_call(call: &OpCall, caps: &crate::admission::ShapeCaps, rng: &mut SplitMix64) -> (HostCall, String) {
    let (i, o) = (&call.inputs, &call.outputs);
    let k2 = k2_of(&i[1].shape).unwrap_or(8);
    let m = i[0].shape[0].min(caps.rows);
    let n = (16 * i[1].shape[1]).min(caps.n.max(HAD)) / HAD * HAD;
    let cb = if k2 % 2 == 1 { Codebook::Mul1 } else { Codebook::ALL[m % 3] };
    (case(rng, m, n, i[0].shape[1], k2, cb, i[0].dtype, o[0].dtype), label(call, caps))
}

// ---- the declared scheme (config.json) --------------------------------------------------------------------------

/// The EXL3 block a checkpoint declares: `quantization_config` or `quantization` of `config.json` (top level or its
/// `text_config`) with `quant_method = "exl3"`, or the same object as a sidecar `quantization_config.json`.
pub fn config_fields(cfg: &serde_json::Value) -> Option<serde_json::Value> {
    let is = |b: &serde_json::Value| b.get("quant_method").and_then(|m| m.as_str()).is_some_and(|m| m.eq_ignore_ascii_case("exl3"));
    if is(cfg) {
        return Some(cfg.clone());
    }
    let text = cfg.get("text_config");
    [cfg, text.unwrap_or(&serde_json::Value::Null)]
        .iter()
        .flat_map(|c| ["quantization_config", "quantization"].map(|k| c.get(k)))
        .flatten()
        .find(|b| is(b))
        .cloned()
}

/// Refuse, from the config alone, a checkpoint whose EXL3 weights this engine does not read: a codebook other than
/// 3inst / mcg / mul1, a head or MTP width that is neither an integer 1..8 nor 16 (16: a plain fp16 head), an average
/// width outside 1..8. Per-tensor checks (shapes, markers) happen at load.
pub fn check_config(f: &serde_json::Value) -> Result<()> {
    let refuse = |why: String| {
        Err(Error::Unsupported(format!(
            "EXL3 checkpoint ({}): {why}; this engine reads the 3inst, mcg and mul1 codebooks at 1..8 bits per weight \
             (1.5, 2.5, 3.5 with mul1)",
            describe_config(f)
        )))
    };
    if let Some(cb) = f.get("codebook").and_then(|v| v.as_str()) {
        if Codebook::parse(cb).is_none() {
            return refuse(format!("codebook '{cb}'"));
        }
    }
    for key in ["head_bits", "mtp_bits"] {
        if let Some(v) = f.get(key).filter(|v| !v.is_null()) {
            match v.as_f64() {
                Some(b) if b == 16.0 || (b.fract() == 0.0 && (1.0..=8.0).contains(&b)) => {}
                _ => return refuse(format!("{key} {v}")),
            }
        }
    }
    if let Some(v) = f.get("bits").filter(|v| !v.is_null()) {
        match v.as_f64() {
            Some(b) if (1.0..=8.0).contains(&b) => {}
            _ => return refuse(format!("average bits {v}")),
        }
    }
    Ok(())
}

/// `"exl3 0.0.1, 3inst, average 4 bits, head 6"` (the fields that are present).
pub fn describe_config(f: &serde_json::Value) -> String {
    let s = |k: &str| f.get(k).filter(|v| !v.is_null()).map(|v| v.as_str().map(String::from).unwrap_or_else(|| v.to_string()));
    let mut parts = vec!["exl3".to_string()];
    parts.extend(s("version"));
    parts.extend(s("codebook"));
    parts.extend(s("bits").map(|b| format!("average {b} bits")));
    parts.extend(s("head_bits").map(|b| format!("head {b}")));
    parts.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_fields_and_refusals() {
        let q8b: serde_json::Value = serde_json::from_str(
            r#"{"architectures":["Qwen3ForCausalLM"],"quantization_config":{"quant_method":"exl3","version":"0.0.1","bits":4.0,"head_bits":6,"calibration":{"rows":100,"cols":2048},"out_scales":"auto"}}"#,
        )
        .unwrap();
        let f = config_fields(&q8b).unwrap();
        check_config(&f).unwrap();
        assert_eq!(describe_config(&f), "exl3, 0.0.1, average 4.0 bits, head 6");
        let glm: serde_json::Value =
            serde_json::from_str(r#"{"text_config":{"quantization_config":{"quant_method":"exl3","bits":4,"codebook":"mcg","head_bits":16}}}"#).unwrap();
        check_config(&config_fields(&glm).unwrap()).unwrap();
        assert!(config_fields(&serde_json::json!({"quantization_config":{"quant_method":"fp8"}})).is_none());
        for bad in [r#"{"quant_method":"exl3","codebook":"2mad"}"#, r#"{"quant_method":"exl3","head_bits":6.5}"#, r#"{"quant_method":"exl3","bits":9}"#] {
            let v: serde_json::Value = serde_json::from_str(bad).unwrap();
            assert!(check_config(&config_fields(&v).unwrap()).is_err(), "{bad}");
        }
    }

    #[test]
    fn codebook_values_spot_checks() {
        // MUL1 of state 0: x = 0, byte sum 0, 1024 * 0x1EEE + 0xC931 rounded once
        let v = h(decode_state(Codebook::Mul1, 0));
        assert!((v - (1024.0 * h(MUL1_SCALE) + h(MUL1_BIAS))).abs() < 1e-2, "{v}");
        // every value is finite and the codebooks are roughly zero-mean, unit-ish spread
        for cb in Codebook::ALL {
            let t = codebook_table(cb);
            let (mut s, mut s2) = (0f64, 0f64);
            for b in t {
                let x = h(*b);
                assert!(x.is_finite());
                s += x;
                s2 += x * x;
            }
            let (mean, var) = (s / 65536.0, s2 / 65536.0 - (s / 65536.0).powi(2));
            assert!(mean.abs() < 0.1 && (0.3..3.0).contains(&var), "{cb:?} mean {mean} var {var}");
        }
    }

    #[test]
    fn stream_ends_and_positions() {
        assert_eq!((0..4).map(|p| stream_end(4, p)).collect::<Vec<_>>(), vec![2, 4, 6, 8]);
        // 2.5 bits: alternating 2 and 3 bit steps
        assert_eq!((0..4).map(|p| stream_end(5, p)).collect::<Vec<_>>(), vec![2, 5, 7, 10]);
        assert_eq!(stream_end(5, 255), 640); // 256 * 2.5: the tile's 40 words
        let mut seen = [false; 256];
        for p in 0..256 {
            let (r, c) = tile_pos(p);
            assert!(!seen[r * 16 + c]);
            seen[r * 16 + c] = true;
        }
    }

    #[test]
    fn hadamard_is_orthonormal() {
        let mut e = vec![0f64; HAD];
        e[5] = 1.0;
        fwht128(&mut e);
        fwht128(&mut e);
        assert!((e[5] - 1.0).abs() < 1e-12 && e.iter().enumerate().all(|(i, v)| i == 5 || v.abs() < 1e-12));
    }

    #[test]
    fn forward_equals_dequant_matmul() {
        let mut rng = SplitMix64::new(7);
        for (k2, cb) in [(4usize, Codebook::Inst3), (5, Codebook::Mul1), (6, Codebook::Mcg)] {
            let (n, k, m) = (256, 128, 3);
            let q = random(&mut rng, n, k, k2, cb);
            let x = rng.normal_vec(m * k, 1.0);
            let (suh, svh) = (f16_bytes(&q.parts[1].data), f16_bytes(&q.parts[2].data));
            let y = forward(&x, m, k, n, &q.parts[0].data, &suh, &svh, k2, cb);
            let wt = dequant_t(&q.parts[0].data, &suh, &svh, k, n, k2, cb);
            for mi in 0..m {
                for nn in 0..n {
                    let r: f64 = (0..k).map(|kk| x[mi * k + kk] as f64 * wt[nn * k + kk]).sum();
                    assert!((r - y[mi * n + nn]).abs() < 1e-9 * (1.0 + r.abs()), "{k2} {cb:?}: {r} vs {}", y[mi * n + nn]);
                }
            }
            assert_eq!(q.dequant().len(), n * k);
        }
    }

    #[test]
    fn signs_unpack_low_bit_first() {
        let f = unpack_signs(&[0b0000_0101, 0x80]);
        let v: Vec<f64> = f16_bytes(&f);
        assert_eq!(&v[..4], &[-1.0, 1.0, -1.0, 1.0]);
        assert_eq!(v[15], -1.0);
    }

    #[test]
    fn every_case_validates() {
        use crate::admission::SweepConfig;
        for c in sweep(&SweepConfig { quick: true, ..SweepConfig::default() }) {
            let mk = |t: &HostTensor| TensorRef::new(crate::backend::Buffer::new(P(t.nbytes())), t.dtype, &t.shape);
            let call = OpCall::new(
                Op::MatmulExl3,
                c.call.inputs.iter().map(mk).collect(),
                c.call.outputs.iter().map(|o| mk(&HostTensor::zeros(o.dtype, &o.shape))).collect(),
                c.call.attrs.clone(),
            );
            call.check().unwrap_or_else(|e| panic!("{}: {e}", c.label));
            let s = ShapeInfo::from_call(&call);
            assert_eq!(s.n, c.call.outputs[0].shape[1]);
        }
    }

    struct P(usize);
    impl crate::backend::BufferImpl for P {
        fn bytes(&self) -> usize {
            self.0
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }
}
