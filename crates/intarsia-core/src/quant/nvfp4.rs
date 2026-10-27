//! NVFP4 (ModelOpt) weights: 4-bit e2m1 values in groups of 16 along K, one fp8 e4m3 scale per
//! group and one fp32 tensor scale (`weight_scale_2`).
//!
//! ```text
//! weight        u8  [N, K/2]    two e2m1 codes per byte: element 2i in the LOW nibble, 2i+1 in the HIGH nibble
//! weight_scale  e4m3 [N, K/16]  one scale per 16 consecutive K elements of a row (plain row-major, no swizzle)
//! weight_scale_2 f32 scalar     W[n,k] = e2m1(code) * float(weight_scale[n,k/16]) * weight_scale_2
//! input_scale   f32 scalar      only used by W4A4 checkpoints; the Qwen3.6 checkpoint is W4A16 (`quant_algo`
//!                               = "W4A16_NVFP4"), activations stay 16-bit, so it is ignored.
//! ```
//!
//! e2m1 code (sign bit 3, magnitude bits 2..0) values: `0 0.5 1 1.5 2 3 4 6`, sign-flipped for codes 8..15.
//! Nibble order and the multiplier role of `weight_scale_2` were checked against TensorFold's independent
//! numpy dequantizer (`tensorfold/cuda/nvfp4/format.py`) and against real tensors (dequantised std ~0.01,
//! max |raw e2m1*e4m3| <= 6*448 = 2688).

use crate::backend::{upload_new, Backend, TensorRef};
use crate::dtype::{f32_to_fp8_e4m3, fp8_e4m3_to_f32, DType};
use crate::error::{Error, Result};
use crate::quant::{HostQTensor, QTensor, QuantCfg, QuantFormat, QuantId, QuantLoadOpts, WeightMeta, NVFP4};
use crate::tensor::HostTensor;
use crate::weights::load_tensor;

/// Elements per block scale.
pub const GROUP: usize = 16;

/// e2m1 magnitudes indexed by the low three bits of a code.
pub const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Value of a 4-bit e2m1 code.
#[inline]
pub fn fp4_to_f32(code: u8) -> f32 {
    let m = E2M1[(code & 7) as usize];
    if code & 8 != 0 {
        -m
    } else {
        m
    }
}

/// Round-to-nearest-even e2m1 encode (saturating at +-6). Ties go to the even code like ModelOpt's `_cast_fp4`.
pub fn f32_to_fp4(x: f32) -> u8 {
    let sign = if x.is_sign_negative() && x != 0.0 { 8u8 } else { 0 };
    let a = x.abs();
    // decision boundaries between 0, .5, 1, 1.5, 2, 3, 4, 6 and whether the tie rounds up (even code above)
    const B: [(f32, bool); 7] =
        [(0.25, false), (0.75, true), (1.25, false), (1.75, true), (2.5, false), (3.5, true), (5.0, false)];
    let mut code = 0u8;
    for (i, (b, tie_up)) in B.iter().enumerate() {
        if a > *b || (a == *b && *tie_up) {
            code = i as u8 + 1;
        }
    }
    sign | code
}

/// Dequantise `n` rows of `k` elements (`w` = `n*k/2` bytes, `bs` = `n*k/16` e4m3 scales) to f32.
pub fn dequant_rows(w: &[u8], bs: &[u8], gs: f32, n: usize, k: usize) -> Vec<f32> {
    assert!(k % GROUP == 0 && w.len() >= n * k / 2 && bs.len() >= n * k / GROUP);
    let mut out = vec![0f32; n * k];
    for r in 0..n {
        for j in 0..k {
            let byte = w[r * (k / 2) + j / 2];
            let code = if j % 2 == 0 { byte & 0xF } else { byte >> 4 };
            let s = fp8_e4m3_to_f32(bs[r * (k / GROUP) + j / GROUP]);
            out[r * k + j] = fp4_to_f32(code) * s * gs;
        }
    }
    out
}

/// ModelOpt-style RTN quantisation of an `[n, k]` row-major matrix: `weight_scale_2 = amax / (6*448)`,
/// per-16 scale = e4m3(block_amax / 6 / weight_scale_2). Returns (packed `[n,k/2]`, scales `[n,k/16]`, gs).
pub fn quantize_rows(vals: &[f32], n: usize, k: usize) -> (Vec<u8>, Vec<u8>, f32) {
    assert!(k % GROUP == 0 && vals.len() == n * k);
    let amax = vals.iter().fold(0f32, |m, v| m.max(v.abs()));
    let gs = if amax > 0.0 { amax / (6.0 * 448.0) } else { 1.0 };
    let mut packed = vec![0u8; n * k / 2];
    let mut scales = vec![0u8; n * k / GROUP];
    for r in 0..n {
        for g in 0..k / GROUP {
            let blk = &vals[r * k + g * GROUP..r * k + (g + 1) * GROUP];
            let bmax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
            let sc = f32_to_fp8_e4m3(bmax / 6.0 / gs);
            scales[r * (k / GROUP) + g] = sc;
            let sd = fp8_e4m3_to_f32(sc) * gs;
            for (j, v) in blk.iter().enumerate() {
                let code = if sd > 0.0 { f32_to_fp4(*v / sd) } else { 0 };
                let idx = r * (k / 2) + (g * GROUP + j) / 2;
                if j % 2 == 0 {
                    packed[idx] |= code;
                } else {
                    packed[idx] |= code << 4;
                }
            }
        }
    }
    (packed, scales, gs)
}

/// The `nvfp4` quantisation format (ModelOpt checkpoints).
pub struct Nvfp4;

impl QuantFormat for Nvfp4 {
    fn id(&self) -> QuantId {
        NVFP4
    }

    fn detect(&self, meta: &WeightMeta) -> bool {
        let w = meta.info("weight");
        let s = meta.info("weight_scale");
        let s2 = meta.info("weight_scale_2");
        matches!((w, s, s2), (Some(w), Some(s), Some(_)) if w.dtype == DType::U8 && s.dtype == DType::Fp8E4m3)
    }

    fn load(&self, meta: &WeightMeta, backend: &dyn Backend, opts: &QuantLoadOpts) -> Result<QTensor> {
        let name = |s: &str| format!("{}.{s}", meta.prefix);
        let wi = meta.info("weight").ok_or_else(|| Error::model(format!("'{}.weight' missing", meta.prefix)))?;
        if wi.shape.len() != 2 {
            return Err(Error::model(format!("nvfp4 '{}': weight must be 2-D, got {:?}", meta.prefix, wi.shape)));
        }
        let (n, kb) = (wi.shape[0], wi.shape[1]);
        let k = kb * 2;
        if k % GROUP != 0 {
            return Err(Error::model(format!("nvfp4 '{}': K={k} is not a multiple of {GROUP}", meta.prefix)));
        }
        let si = meta.info("weight_scale").ok_or_else(|| Error::model(format!("'{}.weight_scale' missing", meta.prefix)))?;
        if si.shape != [n, k / GROUP] {
            return Err(Error::model(format!("nvfp4 '{}': weight_scale is {:?}, expected [{n}, {}]", meta.prefix, si.shape, k / GROUP)));
        }
        // tensor parallelism: the packed nibbles `[n, k/2]` and the block scales `[n, k/16]` are cut along the same axis; a
        // column slice must hold whole 16-element blocks. The global scale stays.
        let (w, bs, n, k) = match opts.shard {
            Some(sh) => {
                if sh.axis == crate::weights::ShardAxis::Cols && (sh.world == 0 || k % (sh.world * GROUP) != 0) {
                    return Err(Error::model(format!(
                        "nvfp4 '{}': K={k} does not split into {} column slices of whole {GROUP}-element blocks",
                        meta.prefix, sh.world
                    )));
                }
                let (ln, lk) = sh.local_shape(n, k);
                let w = crate::weights::load_tensor_shard(backend, meta.src, &name("weight"), sh)?;
                let bs = crate::weights::load_tensor_shard(backend, meta.src, &name("weight_scale"), sh)?;
                (w, bs, ln, lk)
            }
            None => {
                let w = load_tensor(backend, meta.src, &name("weight"), None)?;
                let bs = load_tensor(backend, meta.src, &name("weight_scale"), Some(&[n, k / GROUP]))?;
                (w, bs, n, k)
            }
        };
        if bs.dtype != DType::Fp8E4m3 {
            return Err(Error::model(format!("nvfp4 '{}': weight_scale must be fp8_e4m3, got {}", meta.prefix, bs.dtype)));
        }
        let gs = meta.scalar_f32("weight_scale_2")?;
        let gst = upload_new(backend, &gs.to_le_bytes())?;
        Ok(QTensor {
            format: NVFP4,
            n,
            k,
            parts: vec![w, bs, TensorRef::new(gst, DType::F32, &[1])],
            act_fp8: false, act_quant: false,
        })
    }

    fn dequant_ref(&self, q: &QTensor, backend: &dyn Backend) -> Result<HostTensor> {
        Ok(HostTensor::from_f32(&[q.n, q.k], &q.download(backend)?.dequant()))
    }

    fn quantize(&self, t: &HostTensor, _cfg: &QuantCfg) -> Option<HostQTensor> {
        if t.shape.len() != 2 || t.shape[1] % GROUP != 0 {
            return None;
        }
        let (n, k) = (t.shape[0], t.shape[1]);
        let (packed, scales, gs) = quantize_rows(&t.to_f32(), n, k);
        Some(HostQTensor {
            format: NVFP4,
            n,
            k,
            parts: vec![
                HostTensor { dtype: DType::U8, shape: vec![n, k / 2], data: packed },
                HostTensor { dtype: DType::Fp8E4m3, shape: vec![n, k / GROUP], data: scales },
                HostTensor::from_f32(&[1], &[gs]),
            ],
            act_quant: false,
        })
    }
}

/// Deterministic random NVFP4 matrix for tests and admission: every nibble pattern is valid, scales are
/// finite positive e4m3 codes around 1 and `gs` makes the dequantised values ~unit-variance / sqrt(k).
pub fn random_rows(rng: &mut crate::rng::SplitMix64, n: usize, k: usize) -> HostQTensor {
    let packed: Vec<u8> = (0..n * k / 2).map(|_| rng.next_u64() as u8).collect();
    // e4m3 0x30..=0x41 is 0.5 .. 2.25 (positive, finite)
    let scales: Vec<u8> = (0..n * k / GROUP).map(|_| 0x30 + (rng.next_u64() % 0x12) as u8).collect();
    let gs = 1.0 / (2.9 * (k as f32).sqrt());
    HostQTensor {
        format: NVFP4,
        n,
        k,
        parts: vec![
            HostTensor { dtype: DType::U8, shape: vec![n, k / 2], data: packed },
            HostTensor { dtype: DType::Fp8E4m3, shape: vec![n, k / GROUP], data: scales },
            HostTensor::from_f32(&[1], &[gs]),
        ],
        act_quant: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_table_and_rounding() {
        let want = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];
        for (c, w) in want.iter().enumerate() {
            assert_eq!(fp4_to_f32(c as u8), *w);
            assert_eq!(fp4_to_f32(f32_to_fp4(*w)), *w);
        }
        // ties go to the even code: 0.25 -> 0 (code 0), 0.75 -> 1.0 (code 2), 1.25 -> 1.0, 1.75 -> 2.0, 2.5 -> 2.0, 3.5 -> 4.0, 5 -> 4.0
        for (x, c) in [(0.25, 0u8), (0.75, 2), (1.25, 2), (1.75, 4), (2.5, 4), (3.5, 6), (5.0, 6), (7.0, 7), (0.2501, 1)] {
            assert_eq!(f32_to_fp4(x), c, "{x}");
            assert_eq!(f32_to_fp4(-x), c | 8, "-{x}");
        }
    }

    #[test]
    fn nibble_order_is_low_first() {
        // byte 0x21 = codes (1, 2) = (0.5, 1.0); byte 0x87 = codes (7, 8) = (6.0, -0.0); scale 1.0 (e4m3 0x38), gs 2
        let mut w = vec![0u8; 8];
        w[0] = 0x21;
        w[1] = 0x87;
        let d = dequant_rows(&w, &[0x38], 2.0, 1, 16);
        assert_eq!(&d[..4], &[1.0, 2.0, 12.0, 0.0]);
        assert!(d[3].is_sign_negative());
    }

    #[test]
    fn quantize_roundtrip_error_is_small() {
        let mut rng = crate::rng::SplitMix64::new(3);
        let v = rng.normal_vec(8 * 64, 0.02);
        let t = HostTensor::from_f32(&[8, 64], &v);
        let q = Nvfp4.quantize(&t, &QuantCfg::default()).unwrap();
        let d = q.dequant();
        let (mut num, mut den) = (0f64, 0f64);
        for (a, b) in d.iter().zip(&v) {
            num += ((a - b) as f64).powi(2);
            den += (*b as f64).powi(2);
        }
        assert!((num / den).sqrt() < 0.15, "rel err {}", (num / den).sqrt());
    }
}
