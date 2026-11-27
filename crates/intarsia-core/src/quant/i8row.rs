//! Int8 dense weights with one f32 scale per output row (lane ink-hip-i8, docs/results/2026-11-26-inkling2-q8-study.md arm C and
//! docs/results/2026-11-26-inkling2-hip-i8.md). Opt-in and lossy: `INTARSIA_INK_ATTN_QUANT=int8_row` / `INTARSIA_LMHEAD_QUANT=int8_row`
//! on a backend with the kernels (`Op::MatmulI8`, the i8 form of `Op::InkInProj`), W8A16.
//!
//! A weight `[n, k]` is `q[n, k]` (i8, levels -127..=127) and `scale[n]` (f32): `w[r, c] = q[r, c] * scale[r]`. The scale of a
//! row is `clip / 127`, `clip` the MSE-optimal symmetric clip of the row (`head::int8_best_clip`; round to nearest even).
//! `QTensor::parts = [q, scales]` with scales `[n, 1]` f32 (rank 2 so a row prefix / gather cuts it with the weight).
//!
//! The ops: `MatmulI8`: `out[m, n] = rA(scale[n] * sum_k(x[m, k] * q[n, k]))` (the f32 / f64 sum of the products of the activation
//! and the code, then the row scale, then the rounding of the output: bf16 / f16 for `Matmul` shape, f32 for `Logits`), and
//! `InkInProj` with an i8 `w [Nq + Nk + Nv, D]` and a third input `scales [Nq + Nk + Nv, 1]` f32: pack j of the three outputs is
//! the same sum over its rows. A float `w` takes exactly two inputs, e4m3 exactly three with scales `[3]`, P12 codes three with
//! aux, i8 exactly three with scales `[N, 1]` (the validator is one match over the w dtype).
//!
//! The kernels read only the 1..=16-row calls; larger calls (prefill chunks) and calls a kernel does not select take the bf16
//! weight the model keeps next to the codes: `bf16(q * scale)` (the weight the codes mean, rounded to bf16), so prefill and
//! decode see the same int8 weights (decode's kernel multiplies the exact `q * scale` in f32, prefill's GEMM the bf16 rounding of
//! it: a 2^-9 relative difference per weight, far under the int8 step).

use crate::backend::TensorRef;
use crate::dtype::DType;
use crate::tensor::HostTensor;

/// The `QuantFormat` id of an int8 per-row weight (`QTensor::parts = [q, scales]`, the op `MatmulI8`).
pub const I8ROW: super::QuantId = "i8row";

/// Most activation rows an i8 kernel takes (decode and verify; prefill chunks take the bf16 weight).
pub const ROWS_MAX: usize = 16;

/// Most rows for which Inkling-2 takes the i8 copy of in_proj and wo (larger calls take the dequantised bf16 weight). Set by the
/// kernel microbench (docs/results/2026-11-26-inkling2-hip-i8.md); the unembed takes i8 up to [`ROWS_MAX`].
pub const DENSE_ROWS_MAX: usize = 16;

/// Validate an i8 weight and its scales for `n` rows of `k` columns (a message on failure).
pub fn check_tensors(w: &TensorRef, scales: &TensorRef, n: usize, k: usize, what: &str) -> std::result::Result<(), String> {
    if k == 0 || k % 16 != 0 || k > 65536 {
        return Err(format!("{what}: K = {k} must be a positive multiple of 16, at most 65536"));
    }
    if w.dtype != DType::I8 || w.shape != [n, k] {
        return Err(format!("{what} must be i8 [{n},{k}], got {} {:?}", w.dtype, w.shape));
    }
    if scales.dtype != DType::F32 || scales.shape != [n, 1] {
        return Err(format!("{what} scales must be f32 [{n},1], got {} {:?}", scales.dtype, scales.shape));
    }
    Ok(())
}

/// One row quantised: the codes in `q`, the scale returned. `clip_search`: the MSE-optimal clip (else `amax`). An all-zero row has
/// zero codes and scale 1. Exactly the arithmetic of `head::int8_row_dequant` (which is `q * scale` of these codes).
pub fn quantize_row(x: &[f32], q: &mut [i8], clip_search: bool) -> f32 {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    if !(amax > 0.0) {
        q.iter_mut().for_each(|o| *o = 0);
        return 1.0;
    }
    let s = if clip_search { super::head::int8_best_clip(x, amax) } else { amax } / 127.0;
    let inv = 1.0 / s;
    for (o, v) in q.iter_mut().zip(x) {
        *o = (v * inv).round_ties_even().clamp(-127.0, 127.0) as i8;
    }
    s
}

/// `n` rows of an `[n, k]` f32 matrix as a host i8 weight: `parts = [q i8 [n,k], scales f32 [n,1]]`.
pub fn quantize_host(x: &[f32], n: usize, k: usize, clip_search: bool) -> (HostTensor, HostTensor) {
    let mut q = vec![0i8; n * k];
    let mut sc = Vec::with_capacity(n * 4);
    for (xr, qr) in x.chunks(k).zip(q.chunks_mut(k)) {
        sc.extend_from_slice(&quantize_row(xr, qr, clip_search).to_le_bytes());
    }
    let qb: Vec<u8> = q.iter().map(|v| *v as u8).collect();
    (HostTensor { dtype: DType::I8, shape: vec![n, k], data: qb }, HostTensor { dtype: DType::F32, shape: vec![n, 1], data: sc })
}

/// The dequantised `[n, k]` f32 values of an i8 weight (`q * scale`).
pub fn dequant(q: &[u8], scales: &[f32], k: usize) -> Vec<f32> {
    q.chunks(k).zip(scales).flat_map(|(r, s)| r.iter().map(move |c| (*c as i8) as f32 * s)).collect()
}

/// The bf16 weight the codes mean, as bytes (`bf16(q * scale)`, little endian): what prefill and the calls without an i8 kernel read.
pub fn dequant_bf16_bytes(q: &[u8], scales: &[f32], k: usize) -> Vec<u8> {
    dequant(q, scales, k).iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_times_scale_are_the_emulated_dequant_bit_for_bit() {
        let mut rng = crate::rng::SplitMix64::new(5);
        let (n, k) = (9, 96);
        let x: Vec<f32> = (0..n * k).map(|i| ((rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * if i % 61 == 0 { 9.0 } else { 0.3 }).collect();
        for clip in [true, false] {
            let (q, s) = quantize_host(&x, n, k, clip);
            let scales = s.to_f32();
            let d = dequant(&q.data, &scales, k);
            let mut e = vec![0f32; n * k];
            for (xr, er) in x.chunks(k).zip(e.chunks_mut(k)) {
                super::super::head::int8_row_dequant(xr, er, clip);
            }
            // equal values (the emulation's `-0.0` for a small negative weight is the codes' `+0.0`: `==` treats them as equal)
            assert!(d.iter().zip(&e).all(|(a, b)| a == b), "clip {clip}");
            assert!(q.data.iter().all(|b| (*b as i8) != -128), "levels are -127..=127");
        }
    }

    #[test]
    fn a_zero_row_has_zero_codes_and_scale_one() {
        let (q, s) = quantize_host(&vec![0f32; 32], 1, 32, true);
        assert!(q.data.iter().all(|b| *b == 0));
        assert_eq!(s.to_f32(), vec![1.0]);
    }
}
