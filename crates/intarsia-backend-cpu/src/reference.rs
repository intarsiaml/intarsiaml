//! The oracle: deliberately naive implementations with f64 accumulation.
//! Correctness over speed; every other implementation (including `fast.rs`
//! and all GPU kernels) is graded against these.

use crate::mem::{bytes_mut, read_f32};
use intarsia_core::dtype::encode_from_f32;
use intarsia_core::sampler::{argmax, sample_row, SamplingParams};
use intarsia_core::{DType, Error, Op, OpCall, Result, TensorRef};
use rayon::prelude::*;

fn rd(t: &TensorRef) -> Result<Vec<f64>> {
    Ok(read_f32(t)?.into_iter().map(|v| v as f64).collect())
}

fn wr(t: &TensorRef, v: &[f64]) -> Result<()> {
    let f: Vec<f32> = v.iter().map(|x| *x as f32).collect();
    // SAFETY: exclusive output.
    let dst = unsafe { bytes_mut(t)? };
    encode_from_f32(t.dtype, &f, dst);
    Ok(())
}

/// Round f64 values to what a tensor of `dtype` would store.
fn quantise(dtype: DType, v: &[f64]) -> Vec<f64> {
    let f: Vec<f32> = v.iter().map(|x| *x as f32).collect();
    let mut bytes = vec![0u8; f.len() * dtype.size_bytes()];
    encode_from_f32(dtype, &f, &mut bytes);
    let mut back = vec![0f32; f.len()];
    intarsia_core::dtype::decode_to_f32(dtype, &bytes, &mut back);
    back.into_iter().map(|x| x as f64).collect()
}

fn rmsnorm_rows(x: &[f64], w: &[f64], eps: f64, w_offset: f64) -> Result<Vec<f64>> {
    let d = w.len();
    if d == 0 {
        return Err(Error::invalid("rmsnorm: zero-width weight"));
    }
    let mut out = vec![0f64; x.len()];
    for (xr, or) in x.chunks(d).zip(out.chunks_mut(d)) {
        let mut ss = 0.0;
        for v in xr {
            ss += v * v;
        }
        let r = 1.0 / (ss / d as f64 + eps).sqrt();
        for i in 0..d {
            or[i] = xr[i] * r * (w[i] + w_offset);
        }
    }
    Ok(out)
}

pub fn run(call: &OpCall) -> Result<()> {
    let i = &call.inputs;
    let o = &call.outputs;
    let a = &call.attrs;
    match call.op {
        Op::GateSplit | Op::SigmoidMul | Op::GdnConv | Op::GdnDeltaRule => crate::gdn_ref::run(call),
        Op::MatmulFp8 | Op::MatmulNvfp4 | Op::MoeTopk | Op::MoeExperts | Op::MoeCombine => crate::quant_ref::run(call),
        Op::Embedding => {
            let ids = {
                let b = crate::mem::bytes(&i[0])?;
                b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<_>>()
            };
            let (v, h) = (i[1].shape[0], i[1].shape[1]);
            let table = rd(&i[1])?;
            let mut out = Vec::with_capacity(ids.len() * h);
            for id in ids {
                if id < 0 || id as usize >= v {
                    return Err(Error::invalid("embedding id out of range"));
                }
                out.extend_from_slice(&table[id as usize * h..(id as usize + 1) * h]);
            }
            wr(&o[0], &out)
        }
        Op::RmsNorm | Op::QkNorm => {
            let x = rd(&i[0])?;
            let w = rd(&i[1])?;
            wr(&o[0], &rmsnorm_rows(&x, &w, a.eps as f64, a.w_offset as f64)?)
        }
        Op::RmsNormAdd => {
            let x = rd(&i[0])?;
            let r = rd(&i[1])?;
            let w = rd(&i[2])?;
            let sum: Vec<f64> = x.iter().zip(&r).map(|(p, q)| p + q).collect();
            let sum = quantise(o[1].dtype, &sum);
            wr(&o[0], &rmsnorm_rows(&sum, &w, a.eps as f64, a.w_offset as f64)?)?;
            wr(&o[1], &sum)
        }
        Op::Add => {
            let x = rd(&i[0])?;
            let y = rd(&i[1])?;
            wr(&o[0], &x.iter().zip(&y).map(|(p, q)| p + q).collect::<Vec<_>>())
        }
        Op::Rope => {
            let (t, n, d) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
            let x = rd(&i[0])?;
            // partial rotary: only the first `rd_` values of each head rotate, the rest pass through
            let rd_ = if a.rotary_dim == 0 { d } else { a.rotary_dim };
            let half = rd_ / 2;
            let mut out = x.clone();
            for ti in 0..t {
                let pos = (a.pos0 + ti) as f64;
                for h in 0..n {
                    let base = (ti * n + h) * d;
                    for k in 0..half {
                        let ang = pos * (a.theta as f64).powf(-2.0 * k as f64 / rd_ as f64);
                        let (s, c) = ang.sin_cos();
                        // narrowing cos/sin to f32 like every real kernel does with its tables
                        let (s, c) = (s as f32 as f64, c as f32 as f64);
                        out[base + k] = x[base + k] * c - x[base + k + half] * s;
                        out[base + k + half] = x[base + k + half] * c + x[base + k] * s;
                    }
                }
            }
            wr(&o[0], &out)
        }
        Op::Matmul | Op::Logits => {
            let (m, k) = (i[0].shape[0], i[0].shape[1]);
            let n = i[1].shape[0];
            let x = rd(&i[0])?;
            let w = rd(&i[1])?;
            if n == 0 {
                return Err(Error::invalid(format!("{}: zero output dim (n=0, m={m}, k={k})", call.op.as_str())));
            }
            let mut out = vec![0f64; m * n];
            out.par_chunks_mut(n).enumerate().for_each(|(mi, row)| {
                for (ni, o) in row.iter_mut().enumerate() {
                    let mut s = 0.0;
                    for ki in 0..k {
                        s += x[mi * k + ki] * w[ni * k + ki];
                    }
                    *o = s;
                }
            });
            wr(&o[0], &out)
        }
        Op::KvAppend => {
            let (t, nkv, d) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
            let cap = o[0].shape[1];
            let src = rd(&i[0])?;
            // untouched rows must keep their exact stored bits: write only the new rows
            let f: Vec<f32> = src.iter().map(|x| *x as f32).collect();
            let csz = o[0].dtype.size_bytes();
            // SAFETY: in-place output.
            let cb = unsafe { bytes_mut(&o[0])? };
            for ti in 0..t {
                for h in 0..nkv {
                    let off = (h * cap + a.pos0 + ti) * d * csz;
                    encode_from_f32(
                        o[0].dtype,
                        &f[(ti * nkv + h) * d..(ti * nkv + h + 1) * d],
                        &mut cb[off..off + d * csz],
                    );
                }
            }
            Ok(())
        }
        Op::AttnPrefill | Op::AttnDecode => {
            let (t, n, d) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
            let (nkv, cap) = (i[1].shape[0], i[1].shape[1]);
            let group = n / nkv;
            let q = rd(&i[0])?;
            let kc = rd(&i[1])?;
            let vc = rd(&i[2])?;
            let scale = a.scale as f64;
            let mut out = vec![0f64; t * n * d];
            for ti in 0..t {
                let visible = a.pos0 + ti + 1;
                for h in 0..n {
                    let kvh = h / group;
                    let qrow = &q[(ti * n + h) * d..(ti * n + h + 1) * d];
                    let mut sc = Vec::with_capacity(visible);
                    for j in 0..visible {
                        let kr = &kc[(kvh * cap + j) * d..(kvh * cap + j + 1) * d];
                        let mut s = 0.0;
                        for e in 0..d {
                            s += qrow[e] * kr[e];
                        }
                        sc.push(s * scale);
                    }
                    let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let ex: Vec<f64> = sc.iter().map(|s| (s - mx).exp()).collect();
                    let den: f64 = ex.iter().sum();
                    for j in 0..visible {
                        let vr = &vc[(kvh * cap + j) * d..(kvh * cap + j + 1) * d];
                        for e in 0..d {
                            out[(ti * n + h) * d + e] += ex[j] / den * vr[e];
                        }
                    }
                }
            }
            wr(&o[0], &out)
        }
        Op::SiluGlu => {
            let g = rd(&i[0])?;
            let u = rd(&i[1])?;
            wr(&o[0], &g.iter().zip(&u).map(|(g, u)| g / (1.0 + (-g).exp()) * u).collect::<Vec<_>>())
        }
        Op::Argmax => {
            let (r, v) = (i[0].shape[0], i[0].shape[1]);
            let l = read_f32(&i[0])?;
            // SAFETY: exclusive output.
            let ob = unsafe { bytes_mut(&o[0])? };
            for ri in 0..r {
                let id = argmax(&l[ri * v..(ri + 1) * v]) as i32;
                ob[ri * 4..ri * 4 + 4].copy_from_slice(&id.to_le_bytes());
            }
            Ok(())
        }
        Op::Sample => {
            let (r, v) = (i[0].shape[0], i[0].shape[1]);
            let l = read_f32(&i[0])?;
            let u = read_f32(&i[1])?;
            let p = SamplingParams { temperature: a.temperature, top_k: a.top_k, top_p: a.top_p };
            // SAFETY: exclusive output.
            let ob = unsafe { bytes_mut(&o[0])? };
            for ri in 0..r {
                let id = sample_row(&l[ri * v..(ri + 1) * v], &p, u[ri]) as i32;
                ob[ri * 4..ri * 4 + 4].copy_from_slice(&id.to_le_bytes());
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod zero_dim_tests {
    use super::*;
    use intarsia_core::exec::{run_host, HostCall, OutSpec};
    use intarsia_core::HostTensor;

    fn run_ref(op: Op, inputs: Vec<HostTensor>, out_shape: &[usize]) -> Result<()> {
        let b = crate::CpuBackend::arc();
        let hc = HostCall { op, inputs, outputs: vec![OutSpec::new(DType::F32, out_shape)], attrs: Default::default() };
        run_host(&*b, &format!("ref_{}", op.as_str()), &hc).map(|_| ())
    }

    #[test]
    fn matmul_with_zero_output_dim_is_an_error_not_a_panic() {
        let x = HostTensor::from_f32(&[2, 4], &[0.5; 8]);
        let w = HostTensor::from_f32(&[0, 4], &[]);
        let e = run_ref(Op::Matmul, vec![x, w], &[2, 0]).unwrap_err();
        assert!(e.to_string().contains("zero output dim"), "{e}");
    }

    #[test]
    fn rmsnorm_with_zero_width_is_an_error_not_a_panic() {
        let x = HostTensor::from_f32(&[2, 0], &[]);
        let w = HostTensor::from_f32(&[0], &[]);
        assert!(run_ref(Op::RmsNorm, vec![x, w], &[2, 0]).is_err());
    }
}
