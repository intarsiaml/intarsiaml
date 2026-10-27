//! MXFP4 (OCP Microscaling, AMD Quark export) weights: 4-bit e2m1 values in groups of 32 along K, one e8m0
//! (power-of-two) scale per group, no tensor scale.
//!
//! ```text
//! weight        u8 [N, K/2]   two e2m1 codes per byte: element 2i in the LOW nibble, 2i+1 in the HIGH nibble
//! weight_scale  u8 [N, K/32]  e8m0: biased exponent, value 2^(b-127); 255 is NaN. Plain row-major, no swizzle.
//!                             W[n,k] = e2m1(code) * 2^(weight_scale[n,k/32] - 127)
//! ```
//!
//! Activations: Quark's MXFP4 export declares `input_tensors` as dynamic MXFP4 too (W4A4): at run time every
//! group of 32 consecutive activations along K is quantised to e2m1 with its own e8m0 scale and dequantised again
//! (QDQ) before the dot product. [`Mxfp4Act::Dynamic`] runs that declared scheme; [`Mxfp4Act::None`] keeps the
//! activations 16-bit (W4A16, a deviation from the checkpoint: the choice of `--mxfp4-act none` or a recipe's `[defaults] mxfp4_act`). The op carries the mode as its fourth input,
//! `act [1] f32` (0 = 16-bit activations, 1 = dynamic MXFP4 QDQ), like FP8's `input_scale`.
//!
//! Activation scale, Quark `scale_calculation_mode = "even"`: round the group's amax at the element's mantissa width
//! (e2m1 has 1 mantissa bit: add 2^21 to the f32 bit pattern and clear the mantissa), take its exponent, subtract
//! e2m1's largest exponent (2), clamp to e8m0's range. The element is then `x / scale` rounded to nearest-even e2m1,
//! saturating at +-6. With `amax = m * 2^e`, m in [1, 1.75) gives scale 2^(e-2) (elements up to 7 -> saturate at 6),
//! m in [1.75, 2) gives 2^(e-1).
//!
//! Nibble order, e8m0 decode and the "even" rule were checked against independent code: radiance's exact-reference
//! `_exact_ref` (`/opt/code/vllm-mxfp4/radiance_mxfp4.py`: `codes = [w & 0xF, w >> 4]`, `2^(scale-127)` repeated over
//! 32 columns) and the real Qwen3.8-27B Quark checkpoint (`crates/intarsia-backend-cpu/tests/quant_mxfp4.rs`).

use crate::backend::{upload_new, Backend, TensorRef};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::quant::nvfp4::{f32_to_fp4, fp4_to_f32};
use crate::quant::{HostQTensor, QTensor, QuantCfg, QuantFormat, QuantId, QuantLoadOpts, WeightMeta, MXFP4};
use crate::tensor::HostTensor;
use crate::weights::load_tensor;

/// Elements per e8m0 scale.
pub const GROUP: usize = 32;

/// How the MXFP4 format treats activations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Mxfp4Act {
    /// W4A4 (Quark's declared scheme): activations are QDQ'd to MXFP4 per group of 32 at run time.
    #[default]
    Dynamic,
    /// W4A16: activations stay 16-bit (a deviation from a W4A4 checkpoint: `--mxfp4-act none`, or a recipe's default as in the Qwen3.8-27B's).
    None,
}

impl Mxfp4Act {
    pub fn parse(s: &str) -> Result<Mxfp4Act> {
        match s {
            "dynamic" | "w4a4" => Ok(Mxfp4Act::Dynamic),
            "none" | "w4a16" => Ok(Mxfp4Act::None),
            other => Err(Error::model(format!("unknown mxfp4 activation mode '{other}' (dynamic | none)"))),
        }
    }
    /// Value of the op's `act [1] f32` input.
    pub fn flag(self) -> f32 {
        match self {
            Mxfp4Act::Dynamic => 1.0,
            Mxfp4Act::None => 0.0,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Mxfp4Act::Dynamic => "dynamic",
            Mxfp4Act::None => "none",
        }
    }
}

/// The activation mode the MXFP4 projections run: `--mxfp4-act` (`cli`), else the recipe's `[defaults] mxfp4_act`, else what
/// the checkpoint declares, else W4A16.
pub fn effective_act(cli: Option<Mxfp4Act>, recipe: Option<Mxfp4Act>, declared: Option<Mxfp4Act>) -> Mxfp4Act {
    cli.or(recipe).or(declared).unwrap_or(Mxfp4Act::None)
}

/// Which input of [`effective_act`] decided (for the load log): `--mxfp4-act`, the recipe, or the checkpoint.
pub fn act_source(cli: Option<Mxfp4Act>, recipe: Option<Mxfp4Act>) -> ActSource {
    if cli.is_some() {
        ActSource::Cli
    } else if recipe.is_some() {
        ActSource::Recipe
    } else {
        ActSource::Checkpoint
    }
}

/// Where the MXFP4 activation mode came from ([`act_source`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActSource {
    Cli,
    Recipe,
    Checkpoint,
}

/// Value of an e8m0 scale byte: `2^(b-127)` (exact, including the subnormal `2^-127`); 255 is NaN.
#[inline]
pub fn e8m0_to_f32(b: u8) -> f32 {
    match b {
        255 => f32::NAN,
        0 => f32::from_bits(0x0040_0000), // 2^-127, an f32 subnormal
        _ => f32::from_bits((b as u32) << 23),
    }
}

/// e8m0 byte of the shared scale for a group whose largest magnitude is `amax` (Quark "even" mode, e2m1 elements).
#[inline]
pub fn e8m0_scale_even(amax: f32) -> u8 {
    if amax.is_nan() {
        return 255;
    }
    // round amax at e2m1's 1 mantissa bit (add half of the last kept mantissa ulp, 2^21), keep only the exponent
    let bits = amax.abs().to_bits();
    let rounded = (bits + (1 << 21)) & 0x7F80_0000;
    let biased = (rounded >> 23) as i32; // 0 for zero / tiny amax
    if biased == 0 {
        return 0; // smallest scale 2^-127: every element of the group quantises to 0 or a tiny value
    }
    if biased == 255 {
        return 254; // amax rounded up to inf (inf, or >= 1.75 * 2^127): Quark's log2 gives inf, clamped to 2^127
    }
    // unbiased exponent of the rounded amax, minus e2m1's emax (2); clamp to e8m0's [-127, 127]
    let e = (biased - 127 - 2).clamp(-127, 127);
    (e + 127) as u8
}

/// QDQ one group (any length, normally 32) of activations in place: Quark dynamic MXFP4, "even" scale.
pub fn qdq_group(x: &mut [f32]) {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let sb = e8m0_scale_even(amax);
    let s = e8m0_to_f32(sb);
    for v in x.iter_mut() {
        *v = fp4_to_f32(f32_to_fp4(*v / s)) * s;
    }
}

/// QDQ an `[m, k]` activation matrix along K in groups of 32 (`k % 32 == 0`).
pub fn qdq_rows(x: &[f32], m: usize, k: usize) -> Vec<f32> {
    assert!(k % GROUP == 0 && x.len() >= m * k);
    let mut out = x[..m * k].to_vec();
    for g in out.chunks_exact_mut(GROUP) {
        qdq_group(g);
    }
    out
}

/// Dequantise `n` rows of `k` elements (`w` = `n*k/2` bytes, `s` = `n*k/32` e8m0 bytes) to f32.
pub fn dequant_rows(w: &[u8], s: &[u8], n: usize, k: usize) -> Vec<f32> {
    assert!(k % GROUP == 0 && w.len() >= n * k / 2 && s.len() >= n * k / GROUP);
    let mut out = vec![0f32; n * k];
    for r in 0..n {
        for j in 0..k {
            let byte = w[r * (k / 2) + j / 2];
            let code = if j % 2 == 0 { byte & 0xF } else { byte >> 4 };
            out[r * k + j] = fp4_to_f32(code) * e8m0_to_f32(s[r * (k / GROUP) + j / GROUP]);
        }
    }
    out
}

/// Round-to-nearest MXFP4 quantisation of an `[n, k]` row-major matrix with the "even" scale rule.
/// Returns (packed `[n,k/2]`, e8m0 scales `[n,k/32]`).
pub fn quantize_rows(vals: &[f32], n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    assert!(k % GROUP == 0 && vals.len() == n * k);
    let mut packed = vec![0u8; n * k / 2];
    let mut scales = vec![0u8; n * k / GROUP];
    for r in 0..n {
        for g in 0..k / GROUP {
            let blk = &vals[r * k + g * GROUP..r * k + (g + 1) * GROUP];
            let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
            let sb = e8m0_scale_even(amax);
            scales[r * (k / GROUP) + g] = sb;
            let s = e8m0_to_f32(sb);
            for (j, v) in blk.iter().enumerate() {
                let code = f32_to_fp4(*v / s);
                let idx = r * (k / 2) + (g * GROUP + j) / 2;
                packed[idx] |= if j % 2 == 0 { code } else { code << 4 };
            }
        }
    }
    (packed, scales)
}

/// The `act [1] f32` op input for a mode.
pub fn act_tensor(act: Mxfp4Act) -> HostTensor {
    HostTensor::from_f32(&[1], &[act.flag()])
}

/// The `mxfp4` quantisation format (AMD Quark OCP MX checkpoints).
pub struct Mxfp4;

impl QuantFormat for Mxfp4 {
    fn id(&self) -> QuantId {
        MXFP4
    }

    /// `weight` u8 `[N, K/2]` next to a u8 `weight_scale` `[N, K/32]`, and no `weight_scale_2` (that is NVFP4).
    fn detect(&self, meta: &WeightMeta) -> bool {
        let (Some(w), Some(s)) = (meta.info("weight"), meta.info("weight_scale")) else { return false };
        meta.info("weight_scale_2").is_none()
            && w.dtype == DType::U8
            && s.dtype == DType::U8
            && w.shape.len() == 2
            && s.shape.len() == 2
            && s.shape[0] == w.shape[0]
            && s.shape[1] * GROUP == w.shape[1] * 2
    }

    fn load(&self, meta: &WeightMeta, backend: &dyn Backend, opts: &QuantLoadOpts) -> Result<QTensor> {
        let name = |s: &str| format!("{}.{s}", meta.prefix);
        let wi = meta.info("weight").ok_or_else(|| Error::model(format!("'{}.weight' missing", meta.prefix)))?;
        if wi.shape.len() != 2 {
            return Err(Error::model(format!("mxfp4 '{}': weight must be 2-D, got {:?}", meta.prefix, wi.shape)));
        }
        let (n, k) = (wi.shape[0], wi.shape[1] * 2);
        if k % GROUP != 0 {
            return Err(Error::model(format!("mxfp4 '{}': K={k} is not a multiple of {GROUP}", meta.prefix)));
        }
        // tensor parallelism: the packed nibbles `[n, k/2]` and the e8m0 scales `[n, k/32]` are cut along the same axis
        // (rows, the GatedDeltaNet q|k|v row groups, or columns). A column slice must hold whole 32-element groups, so
        // each rank's groups, their scales and (W4A4) the activation QDQ groups are exactly those of the full matrix.
        let (w, s, n, k) = match opts.shard {
            Some(sh) => {
                if sh.axis == crate::weights::ShardAxis::Cols && (sh.world == 0 || k % (sh.world * GROUP) != 0) {
                    return Err(Error::model(format!(
                        "mxfp4 '{}': K={k} does not split into {} column slices of whole {GROUP}-element groups",
                        meta.prefix, sh.world
                    )));
                }
                let si = meta.info("weight_scale").ok_or_else(|| Error::model(format!("'{}.weight_scale' missing", meta.prefix)))?;
                if si.shape != [n, k / GROUP] {
                    return Err(Error::model(format!(
                        "mxfp4 '{}': weight_scale is {:?}, expected [{n}, {}]",
                        meta.prefix,
                        si.shape,
                        k / GROUP
                    )));
                }
                let (ln, lk) = sh.local_shape(n, k);
                let w = crate::weights::load_tensor_shard(backend, meta.src, &name("weight"), sh)?;
                let s = crate::weights::load_tensor_shard(backend, meta.src, &name("weight_scale"), sh)?;
                (w, s, ln, lk)
            }
            None => {
                let w = load_tensor(backend, meta.src, &name("weight"), None)?;
                let s = load_tensor(backend, meta.src, &name("weight_scale"), Some(&[n, k / GROUP]))?;
                (w, s, n, k)
            }
        };
        if s.dtype != DType::U8 {
            return Err(Error::model(format!("mxfp4 '{}': weight_scale must be u8 (e8m0), got {}", meta.prefix, s.dtype)));
        }
        let act = upload_new(backend, &act_tensor(opts.mxfp4_act).data)?;
        Ok(QTensor {
            format: MXFP4,
            n,
            k,
            parts: vec![w, s, TensorRef::new(act, DType::F32, &[1])],
            act_quant: opts.mxfp4_act == Mxfp4Act::Dynamic,
        })
    }

    fn dequant_ref(&self, q: &QTensor, backend: &dyn Backend) -> Result<HostTensor> {
        Ok(HostTensor::from_f32(&[q.n, q.k], &q.download(backend)?.dequant()))
    }

    fn quantize(&self, t: &HostTensor, cfg: &QuantCfg) -> Option<HostQTensor> {
        if t.shape.len() != 2 || t.shape[1] % GROUP != 0 {
            return None;
        }
        let (n, k) = (t.shape[0], t.shape[1]);
        let (packed, scales) = quantize_rows(&t.to_f32(), n, k);
        let act = cfg.mxfp4_act.unwrap_or_default();
        Some(HostQTensor {
            format: MXFP4,
            n,
            k,
            parts: vec![
                HostTensor { dtype: DType::U8, shape: vec![n, k / 2], data: packed },
                HostTensor { dtype: DType::U8, shape: vec![n, k / GROUP], data: scales },
                act_tensor(act),
            ],
            act_quant: act == Mxfp4Act::Dynamic,
        })
    }
}

/// Deterministic random MXFP4 matrix for tests and admission: every nibble pattern is valid, scales are e8m0 codes
/// around `2^-log2(sqrt k)` so the dequantised values are ~unit-variance / sqrt(k) (never the NaN code).
pub fn random_rows(rng: &mut crate::rng::SplitMix64, n: usize, k: usize, act: Mxfp4Act) -> HostQTensor {
    let packed: Vec<u8> = (0..n * k / 2).map(|_| rng.next_u64() as u8).collect();
    // e2m1 codes have rms ~3.2; centre the scale exponent on 1 / (3.2 sqrt k), spread over 4 binades
    let centre = 127.0 - ((3.2 * (k as f32).sqrt()).log2()).round();
    let scales: Vec<u8> = (0..n * k / GROUP).map(|_| (centre as i32 - 2 + (rng.next_u64() % 4) as i32).clamp(1, 254) as u8).collect();
    HostQTensor {
        format: MXFP4,
        n,
        k,
        parts: vec![
            HostTensor { dtype: DType::U8, shape: vec![n, k / 2], data: packed },
            HostTensor { dtype: DType::U8, shape: vec![n, k / GROUP], data: scales },
            act_tensor(act),
        ],
        act_quant: act == Mxfp4Act::Dynamic,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn act_precedence_is_cli_then_recipe_then_checkpoint() {
        use Mxfp4Act::{Dynamic as D, None as N};
        // (cli, recipe, declared) -> (mode, source)
        let cases = [
            ((Some(D), Some(N), Some(D)), (D, ActSource::Cli)),
            ((Some(N), Some(D), Some(D)), (N, ActSource::Cli)),
            ((None, Some(N), Some(D)), (N, ActSource::Recipe)),
            ((None, Some(D), Some(N)), (D, ActSource::Recipe)),
            ((None, None, Some(D)), (D, ActSource::Checkpoint)),
            ((None, None, Some(N)), (N, ActSource::Checkpoint)),
            ((None, None, None), (N, ActSource::Checkpoint)),
        ];
        for ((cli, recipe, declared), (mode, src)) in cases {
            assert_eq!(effective_act(cli, recipe, declared), mode, "{cli:?} {recipe:?} {declared:?}");
            assert_eq!(act_source(cli, recipe), src, "{cli:?} {recipe:?} {declared:?}");
        }
    }

    #[test]
    fn e8m0_decode() {
        assert_eq!(e8m0_to_f32(127), 1.0);
        assert_eq!(e8m0_to_f32(128), 2.0);
        assert_eq!(e8m0_to_f32(120), 2f32.powi(-7));
        assert_eq!(e8m0_to_f32(254), 2f32.powi(127));
        assert_eq!(e8m0_to_f32(1), 2f32.powi(-126));
        assert_eq!(e8m0_to_f32(0) as f64, 2f64.powi(-127));
        assert!(e8m0_to_f32(255).is_nan());
    }

    #[test]
    fn even_scale_rule() {
        // amax = m * 2^e: m in [1, 1.75) -> 2^(e-2); m in [1.75, 2) -> 2^(e-1)
        for (amax, want) in [(1.0f32, -2), (1.5, -2), (1.74, -2), (1.75, -1), (1.99, -1), (6.0, 0), (7.0, 1), (8.0, 1), (0.5, -3)] {
            let b = e8m0_scale_even(amax);
            assert_eq!(b as i32 - 127, want, "amax {amax}");
        }
        assert_eq!(e8m0_scale_even(0.0), 0);
        assert_eq!(e8m0_scale_even(f32::NAN), 255);
        assert_eq!(e8m0_scale_even(f32::INFINITY), 254, "Quark: log2(inf) clamps to 127");
        assert_eq!(e8m0_scale_even(f32::MAX), 254);
        assert_eq!(e8m0_scale_even(1e-45), 0);
    }

    #[test]
    fn qdq_keeps_representable_groups_and_saturates() {
        // a group whose values are exactly e2m1 * 2^-3 survives QDQ unchanged (amax 6 * 2^-3 = 0.75 -> m 1.5)
        let mut g: Vec<f32> = (0..32).map(|i| fp4_to_f32((i % 16) as u8) * 0.125).collect();
        let want = g.clone();
        qdq_group(&mut g);
        assert_eq!(g, want);
        // amax 7 (m 1.75) -> scale 2: 7 -> 8 (code 4.0 * 2), 1 -> 1 (0.5 * 2), 0.4 -> 0
        let mut h = vec![0f32; 32];
        h[0] = 7.0;
        h[1] = 1.0;
        h[2] = 0.4;
        qdq_group(&mut h);
        assert_eq!(&h[..3], &[8.0, 1.0, 0.0]);
        // amax 6.9 (m 1.725) -> scale 1: 6.9 saturates at 6
        let mut s = vec![0f32; 32];
        s[0] = -6.9;
        qdq_group(&mut s);
        assert_eq!(s[0], -6.0);
    }

    #[test]
    fn nibble_order_is_low_first() {
        // byte 0x21 = codes (1, 2) = (0.5, 1.0); byte 0x87 = codes (7, 8) = (6.0, -0.0); scale 2 (e8m0 128)
        let mut w = vec![0u8; 16];
        w[0] = 0x21;
        w[1] = 0x87;
        let d = dequant_rows(&w, &[128], 1, 32);
        assert_eq!(&d[..4], &[1.0, 2.0, 12.0, 0.0]);
        assert!(d[3].is_sign_negative());
    }

    #[test]
    fn quantize_roundtrip_error_is_small() {
        let mut rng = crate::rng::SplitMix64::new(5);
        let v = rng.normal_vec(8 * 128, 0.02);
        let t = HostTensor::from_f32(&[8, 128], &v);
        let q = Mxfp4.quantize(&t, &QuantCfg::default()).unwrap();
        let d = q.dequant();
        let (mut num, mut den) = (0f64, 0f64);
        for (a, b) in d.iter().zip(&v) {
            num += ((a - b) as f64).powi(2);
            den += (*b as f64).powi(2);
        }
        assert!((num / den).sqrt() < 0.2, "rel err {}", (num / den).sqrt());
        // re-quantising the dequantised matrix is exact (codes and scales are a fixed point)
        let q2 = Mxfp4.quantize(&HostTensor::from_f32(&[8, 128], &d), &QuantCfg::default()).unwrap();
        assert_eq!(q2.dequant(), d);
    }

    #[test]
    fn random_rows_are_finite_and_scaled() {
        let mut rng = crate::rng::SplitMix64::new(9);
        let q = random_rows(&mut rng, 16, 1024, Mxfp4Act::None);
        let d = q.dequant();
        assert!(d.iter().all(|v| v.is_finite()));
        let rms = (d.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / d.len() as f64).sqrt();
        let want = 1.0 / (1024f64).sqrt();
        assert!(rms > want / 4.0 && rms < want * 4.0, "rms {rms} vs ~{want}");
    }
}
