//! FP8 e4m3 weights (ModelOpt / compressed-tensors per-tensor scheme).
//!
//! ```text
//! weight        e4m3 [N, K]   W[n,k] = float(weight[n,k]) * weight_scale
//! weight_scale  f32 scalar
//! input_scale   f32 scalar    static activation scale (`input_activations.dynamic = false`)
//! ```
//!
//! Two activation modes, chosen per recipe through [`Fp8Act`] and encoded in the op's `scales` tensor
//! `[weight_scale, input_scale]` (`input_scale <= 0` = no activation quantisation):
//!
//! * `Static` (W8A8, the checkpoint's declared scheme): `xq = e4m3(x / input_scale)` (RNE, saturating at 448),
//!   `y = input_scale * weight_scale * sum_k xq[k] * wq[k]`. Products of two fp8 values are exact in
//!   f32, so a GPU kernel only differs from the reference in accumulation order.
//! * `None` (W8A16): activations stay bf16, `y = weight_scale * sum_k x[k] * wq[k]`.
//!
//! Which one tracks the vLLM reference better is decided on the accuracy gate once the model is assembled
//! (see `docs/quant-moe.md`); `Static` matches the scheme that the checkpoint was calibrated for.

use crate::backend::{upload_new, Backend, TensorRef};
use crate::dtype::{f32_to_fp8_e4m3, fp8_e4m3_to_f32, DType};
use crate::error::{Error, Result};
use crate::quant::{
    Fp8Act, HostQTensor, QTensor, QuantCfg, QuantFormat, QuantId, QuantLoadOpts, WeightMeta, FP8_E4M3,
};
use crate::tensor::HostTensor;
use crate::weights::load_tensor;

/// Largest finite e4m3 value.
pub const MAX: f32 = 448.0;

/// `e4m3(x / input_scale) * input_scale`: what a W8A8 kernel feeds the dot product (the activation snapped to the fp8 grid).
pub fn quantize_activation(x: f32, input_scale: f32) -> f32 {
    fp8_e4m3_to_f32(f32_to_fp8_e4m3(x / input_scale)) * input_scale
}

/// Encode one scale/activation pair of the op's `scales` tensor.
pub fn scales_tensor(weight_scale: f32, input_scale: f32) -> HostTensor {
    HostTensor::from_f32(&[2], &[weight_scale, input_scale])
}

pub struct Fp8E4m3;

impl QuantFormat for Fp8E4m3 {
    fn id(&self) -> QuantId {
        FP8_E4M3
    }

    fn detect(&self, meta: &WeightMeta) -> bool {
        let w = meta.info("weight");
        let s = meta.info("weight_scale");
        matches!((w, s), (Some(w), Some(s)) if w.dtype == DType::Fp8E4m3 && s.dtype == DType::F32 && s.shape.iter().product::<usize>() == 1)
    }

    fn load(&self, meta: &WeightMeta, backend: &dyn Backend, opts: &QuantLoadOpts) -> Result<QTensor> {
        let wi = meta.info("weight").ok_or_else(|| Error::model(format!("'{}.weight' missing", meta.prefix)))?;
        if wi.shape.len() != 2 {
            return Err(Error::model(format!("fp8 '{}': weight must be 2-D, got {:?}", meta.prefix, wi.shape)));
        }
        // tensor parallelism: the scale is per tensor, so a slice keeps it (rows, columns or row groups of the e4m3 bytes)
        let wname = format!("{}.weight", meta.prefix);
        let (w, (n, k)) = match opts.shard {
            Some(sh) => (crate::weights::load_tensor_shard(backend, meta.src, &wname, sh)?, sh.local_shape(wi.shape[0], wi.shape[1])),
            None => (load_tensor(backend, meta.src, &wname, None)?, (wi.shape[0], wi.shape[1])),
        };
        let ws = meta.scalar_f32("weight_scale")?;
        let (is, act_quant) = match opts.fp8_act {
            Fp8Act::None => (0.0, false),
            Fp8Act::Static => {
                let s = meta.scalar_f32("input_scale").map_err(|_| {
                    Error::model(format!("fp8 '{}': static activation quantisation needs '.input_scale'", meta.prefix))
                })?;
                if !(s > 0.0 && s.is_finite()) {
                    return Err(Error::model(format!("fp8 '{}': input_scale {s} is not a positive number", meta.prefix)));
                }
                (s, true)
            }
        };
        let st = upload_new(backend, &scales_tensor(ws, is).data)?;
        Ok(QTensor { format: FP8_E4M3, n, k, parts: vec![w, TensorRef::new(st, DType::F32, &[2])], act_quant })
    }

    fn dequant_ref(&self, q: &QTensor, backend: &dyn Backend) -> Result<HostTensor> {
        Ok(HostTensor::from_f32(&[q.n, q.k], &q.download(backend)?.dequant()))
    }

    fn quantize(&self, t: &HostTensor, cfg: &QuantCfg) -> Option<HostQTensor> {
        if t.shape.len() != 2 {
            return None;
        }
        let v = t.to_f32();
        let amax = v.iter().fold(0f32, |m, x| m.max(x.abs()));
        let ws = if amax > 0.0 { amax / MAX } else { 1.0 };
        let data: Vec<u8> = v.iter().map(|x| f32_to_fp8_e4m3(x / ws)).collect();
        Some(HostQTensor {
            format: FP8_E4M3,
            n: t.shape[0],
            k: t.shape[1],
            parts: vec![
                HostTensor { dtype: DType::Fp8E4m3, shape: t.shape.clone(), data },
                scales_tensor(ws, cfg.input_scale.unwrap_or(0.0)),
            ],
            act_quant: cfg.input_scale.is_some(),
        })
    }
}

/// Deterministic random FP8 matrix for tests and admission: finite e4m3 codes (no NaN encodings),
/// `weight_scale` such that the dequantised values are ~unit / sqrt(k), static `input_scale` = 1/8 when `act_quant`.
pub fn random_rows(rng: &mut crate::rng::SplitMix64, n: usize, k: usize, act_quant: bool) -> HostQTensor {
    let data: Vec<u8> = (0..n * k)
        .map(|_| {
            let mut b = rng.next_u64() as u8;
            if b & 0x7F == 0x7F {
                b &= 0xF0; // never the NaN encoding
            }
            b
        })
        .collect();
    // e4m3 codes are uniform over exponents, so the rms is dominated by the top binade (~100): scale it down
    let ws = 1.0 / (60.0 * (k as f32).sqrt());
    HostQTensor {
        format: FP8_E4M3,
        n,
        k,
        parts: vec![
            HostTensor { dtype: DType::Fp8E4m3, shape: vec![n, k], data },
            scales_tensor(ws, if act_quant { 0.125 } else { 0.0 }),
        ],
        act_quant,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_snaps_to_the_fp8_grid() {
        assert_eq!(quantize_activation(1.0, 1.0), 1.0);
        assert_eq!(quantize_activation(1.01, 1.0), 1.0);
        assert_eq!(quantize_activation(1e9, 0.5), 448.0 * 0.5); // saturates
        assert_eq!(quantize_activation(-0.3, 0.1), -0.1 * 3.0); // -3.0 is on the grid (2 and 3 are 1.5*2)
    }

    #[test]
    fn quantize_then_dequant() {
        let mut rng = crate::rng::SplitMix64::new(5);
        let v = rng.normal_vec(4 * 32, 0.05);
        let q = Fp8E4m3.quantize(&HostTensor::from_f32(&[4, 32], &v), &QuantCfg::default()).unwrap();
        let d = q.dequant();
        for (a, b) in d.iter().zip(&v) {
            assert!((a - b).abs() <= b.abs() / 14.0 + 1e-4, "{a} vs {b}");
        }
    }
}
