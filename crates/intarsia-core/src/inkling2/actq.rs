//! The W4A4 activation quantisers of the Inkling-2 experts (`ink.act_quant = 1`; formulas in the `ops` module doc). Every
//! value is an f32 operation in the order written there, so a GPU kernel that uses the same IEEE operations (no fused
//! multiply-adds, round-to-nearest division) reproduces `xq` and `outer` bit for bit. vLLM: `ops/nvfp4.py` (`_quantize_row`,
//! "fouroversix_mae") and `ops/mxfp8.py` (`_mxfp8_block`).

use crate::dtype::f32_to_fp8_e4m3;
use crate::quant::nvfp4::{f32_to_fp4, fp4_to_f32};
use crate::quant::reference::e4m3_table;

/// NVFP4 block (16 values along the row).
pub const NV_BLOCK: usize = 16;
/// MXFP8 block (32 values along the row).
pub const MX_BLOCK: usize = 32;

fn e4m3_round(v: f32) -> f32 {
    e4m3_table()[f32_to_fp8_e4m3(v) as usize]
}

/// Sum of 16 errors as the pairwise tree (adjacent pairs, then pairs of those).
fn tree16(e: &[f32; 16]) -> f32 {
    let mut v = *e;
    let mut n = 16;
    while n > 1 {
        for j in 0..n / 2 {
            v[j] = v[2 * j] + v[2 * j + 1];
        }
        n /= 2;
    }
    v[0]
}

/// One NVFP4 row: `(xq, outer)` with `xq[c] = e2m1 code * e4m3 block scale` (exact in f32 and bf16) and the row scale `outer`
/// (`x ~ xq * outer`). `codes` / `scales` receive the packed codes (`[C/2]`, low nibble first) and the e4m3 block-scale bytes
/// (`[C/16]`) when given (the FP4 tensor-core operands).
pub fn nvfp4_row_parts(x: &[f32], mut codes: Option<&mut [u8]>, mut scales: Option<&mut [u8]>) -> (Vec<f32>, f32) {
    assert!(x.len() % NV_BLOCK == 0, "NVFP4 activation row of {} values", x.len());
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let enc = (1536f32 / amax).min(f32::MAX);
    let dec = 1f32 / enc;
    let outer = amax * (1f32 / 1536f32);
    let mut xq = vec![0f32; x.len()];
    for (b, blk) in x.chunks_exact(NV_BLOCK).enumerate() {
        let bmax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let cand = |s: f32| {
            let inv = (1f32 / (s * dec)).min(f32::MAX);
            let mut c = [0u8; 16];
            let mut err = [0f32; 16];
            for (j, &v) in blk.iter().enumerate() {
                c[j] = f32_to_fp4(v * inv);
                err[j] = ((fp4_to_f32(c[j]) * s) * outer - v).abs();
            }
            (c, tree16(&err))
        };
        let s6 = e4m3_round((bmax * (enc * (1f32 / 6f32))).min(448.0));
        let s4 = e4m3_round((bmax * (enc * 0.25f32)).min(448.0));
        let (c6, e6) = cand(s6);
        let (c4, e4) = cand(s4);
        let (c, s) = if e4 < e6 { (c4, s4) } else { (c6, s6) };
        for j in 0..NV_BLOCK {
            xq[b * NV_BLOCK + j] = fp4_to_f32(c[j]) * s;
        }
        if let Some(cs) = codes.as_deref_mut() {
            for j in 0..NV_BLOCK / 2 {
                cs[b * 8 + j] = (c[2 * j] & 15) | ((c[2 * j + 1] & 15) << 4);
            }
        }
        if let Some(ss) = scales.as_deref_mut() {
            ss[b] = f32_to_fp8_e4m3(s);
        }
    }
    (xq, outer)
}

/// [`nvfp4_row_parts`] without the packed operands.
pub fn nvfp4_row(x: &[f32]) -> (Vec<f32>, f32) {
    nvfp4_row_parts(x, None, None)
}

/// One MXFP8 row: `xq[c] = e4m3 value * 2^(e-127)` per block of 32 (f32; the caller rounds to the activation dtype).
pub fn mxfp8_row(x: &[f32]) -> Vec<f32> {
    assert!(x.len() % MX_BLOCK == 0, "MXFP8 activation row of {} values", x.len());
    let min_scale = f32::from_bits(0x0040_0000); // 2^-127
    let mut out = vec![0f32; x.len()];
    for (b, blk) in x.chunks_exact(MX_BLOCK).enumerate() {
        let bmax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let raw = bmax * (1f32 / 448f32);
        let mut e = ((raw.to_bits().wrapping_add(0x007F_FFFF)) & 0x7F80_0000) >> 23;
        if raw <= min_scale {
            e = 0;
        }
        e = e.min(254);
        let scale = f32::from_bits(e << 23).max(min_scale);
        let inv = if e == 0 { f32::from_bits(254 << 23) } else { 1f32 / scale };
        for (j, &v) in blk.iter().enumerate() {
            let q = (v * inv).clamp(-448.0, 448.0);
            out[b * MX_BLOCK + j] = e4m3_round(q) * scale;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvfp4_zero_row_and_exact_values() {
        let (q, o) = nvfp4_row(&[0f32; 32]);
        assert!(q.iter().all(|v| *v == 0.0) && o == 0.0);
        // a row whose values are e2m1 codes times one scale comes back as itself
        let x: Vec<f32> = (0..32).map(|i| [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][i % 8] * if i % 3 == 0 { -1.0 } else { 1.0 }).collect();
        let (q, o) = nvfp4_row(&x);
        for (a, b) in q.iter().zip(&x) {
            assert!((a * o - b).abs() <= 1e-6 * b.abs().max(1.0), "{a} * {o} vs {b}");
        }
    }

    #[test]
    fn nvfp4_values_are_bf16_exact_and_close() {
        let mut rng = crate::rng::SplitMix64::new(7);
        let x: Vec<f32> = (0..2304).map(|_| (rng.next_f32() - 0.5) * 3.0).collect();
        let (q, o) = nvfp4_row(&x);
        let mut num = 0f64;
        let mut den = 0f64;
        for (a, b) in q.iter().zip(&x) {
            assert_eq!(a.to_bits() & 0xffff, 0, "{a} not bf16");
            num += ((a * o - b) as f64).powi(2);
            den += (*b as f64).powi(2);
        }
        assert!((num / den).sqrt() < 0.12, "rel err {}", (num / den).sqrt());
    }

    #[test]
    fn mxfp8_powers_of_two() {
        let x: Vec<f32> = (0..32).map(|i| i as f32 - 16.0).collect();
        let q = mxfp8_row(&x);
        for (a, b) in q.iter().zip(&x) {
            assert!((a - b).abs() <= b.abs() / 16.0, "{a} vs {b}");
        }
        assert!(mxfp8_row(&[0f32; 32]).iter().all(|v| *v == 0.0));
    }
}
