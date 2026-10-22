//! Optimised cpu implementations: f32 accumulate, rayon over independent
//! outputs (so results are run-to-run deterministic). Graded against
//! `reference.rs` by the admission harness.

use crate::mem::{bytes, bytes_mut, read_f32, write_f32};
use intarsia_core::dtype::{decode_to_f32, encode_from_f32};
use intarsia_core::sampler::{argmax, sample_row, SamplingParams};
use intarsia_core::{Error, Op, OpCall, Result};
use rayon::prelude::*;

pub fn run(call: &OpCall) -> Result<()> {
    match call.op {
        // the gated-delta-net ops reuse the f64 reference math (no separate fast path yet)
        Op::GateSplit | Op::SigmoidMul | Op::GdnConv | Op::GdnDeltaRule => crate::gdn_ref::run(call),
        // quantised matmuls and MoE reuse the f64 reference math (parallel over rows / tokens)
        Op::MatmulFp8 | Op::MatmulNvfp4 | Op::MatmulMxfp4 | Op::MoeTopk | Op::MoeExperts | Op::MoeCombine => crate::quant_ref::run(call),
        Op::MatmulExl3 => crate::exl3_ref::run(call),
        Op::MatmulPair
        | Op::MatmulFp8Pair
        | Op::MatmulMxfp4Pair
        | Op::MatmulNvfp4Glu
        | Op::GatedRmsnorm
        | Op::QkNormRopeKv
        | Op::RmsNormAddMatmulPair
        | Op::GatedRmsnormMatmulFp8
        | Op::RmsNormAddMatmulFp8Pair
        | Op::SigmoidMulMatmulFp8
        | Op::SharedDownCombine
        | Op::GateSplitQkNormRopeKv
        | Op::GdnConvDelta
        | Op::MoeBlock => crate::fused_ref::run(call, run),
        Op::Embedding => embedding(call),
        Op::RmsNorm | Op::QkNorm => rmsnorm(call),
        Op::RmsNormAdd => rmsnorm_add(call),
        Op::Add => add(call),
        Op::Rope | Op::RopeScaled => rope(call),
        Op::AddBias => add_bias(call),
        Op::Matmul | Op::Logits => matmul(call),
        Op::KvAppend => kv_append(call),
        Op::AttnPrefill | Op::AttnDecode => attention(call),
        Op::SiluGlu => silu_glu(call),
        Op::Argmax => argmax_op(call),
        Op::Sample => sample_op(call),
    }
}

/// 8-way unrolled dot product (vectorises on NEON/AVX).
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s = (acc[0] + acc[4]) + (acc[1] + acc[5]) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

fn embedding(call: &OpCall) -> Result<()> {
    let ids = call.inputs[0].clone();
    let table = &call.inputs[1];
    let out = &call.outputs[0];
    let (v, h) = (table.shape[0], table.shape[1]);
    let idb = bytes(&ids)?;
    let tb = bytes(table)?;
    let esz = table.dtype.size_bytes();
    // SAFETY: exclusive output.
    let ob = unsafe { bytes_mut(out)? };
    let same = table.dtype == out.dtype;
    let osz = out.dtype.size_bytes();
    ob.par_chunks_mut(h * osz).enumerate().try_for_each(|(t, row)| -> Result<()> {
        let id = i32::from_le_bytes(idb[t * 4..t * 4 + 4].try_into().unwrap());
        if id < 0 || id as usize >= v {
            return Err(Error::invalid(format!("embedding id {id} out of range {v}")));
        }
        let src = &tb[id as usize * h * esz..(id as usize + 1) * h * esz];
        if same {
            row.copy_from_slice(src);
        } else {
            let mut tmp = vec![0f32; h];
            decode_to_f32(table.dtype, src, &mut tmp);
            encode_from_f32(out.dtype, &tmp, row);
        }
        Ok(())
    })
}

fn norm_row(x: &[f32], w: &[f32], eps: f32, w_offset: f32, out: &mut [f32]) {
    let ss = dot(x, x);
    let r = 1.0 / (ss / x.len() as f32 + eps).sqrt();
    for ((o, xv), wv) in out.iter_mut().zip(x).zip(w) {
        *o = xv * r * (wv + w_offset);
    }
}

/// RmsNorm (rows = all leading dims, normalised over the last).
fn rmsnorm(call: &OpCall) -> Result<()> {
    let x = read_f32(&call.inputs[0])?;
    let w = read_f32(&call.inputs[1])?;
    let d = w.len();
    let mut o = vec![0f32; x.len()];
    o.par_chunks_mut(d).zip(x.par_chunks(d)).for_each(|(orow, xrow)| norm_row(xrow, &w, call.attrs.eps, call.attrs.w_offset, orow));
    write_f32(&call.outputs[0], &o)
}

fn round_to(dtype: intarsia_core::DType, v: &mut [f32]) {
    // emulate storing in `dtype` and reading back
    let mut tmp = vec![0u8; v.len() * dtype.size_bytes()];
    encode_from_f32(dtype, v, &mut tmp);
    decode_to_f32(dtype, &tmp, v);
}

fn rmsnorm_add(call: &OpCall) -> Result<()> {
    let x = read_f32(&call.inputs[0])?;
    let res = read_f32(&call.inputs[1])?;
    let w = read_f32(&call.inputs[2])?;
    let d = w.len();
    let mut sum: Vec<f32> = x.iter().zip(&res).map(|(a, b)| a + b).collect();
    // the residual stream is stored in the output dtype; normalise what is stored
    round_to(call.outputs[1].dtype, &mut sum);
    let mut o = vec![0f32; x.len()];
    o.par_chunks_mut(d).zip(sum.par_chunks(d)).for_each(|(orow, srow)| norm_row(srow, &w, call.attrs.eps, call.attrs.w_offset, orow));
    write_f32(&call.outputs[0], &o)?;
    write_f32(&call.outputs[1], &sum)
}

fn add(call: &OpCall) -> Result<()> {
    let a = read_f32(&call.inputs[0])?;
    let b = read_f32(&call.inputs[1])?;
    let o: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    write_f32(&call.outputs[0], &o)
}

fn add_bias(call: &OpCall) -> Result<()> {
    let cols = call.inputs[0].shape[1];
    let x = read_f32(&call.inputs[0])?;
    let b = read_f32(&call.inputs[1])?;
    let o: Vec<f32> = x.iter().enumerate().map(|(i, v)| v + b[i % cols]).collect();
    write_f32(&call.outputs[0], &o)
}

fn rope(call: &OpCall) -> Result<()> {
    let xt = &call.inputs[0];
    let (n, d) = (xt.shape[1], xt.shape[2]);
    let x = read_f32(xt)?;
    // partial rotary: only the first `rd` values of each head rotate, the rest pass through
    let rd = if call.attrs.rotary_dim == 0 { d } else { call.attrs.rotary_dim };
    let half = rd / 2;
    // `RopeScaled`: the caller's frequency table and cos / sin multiplier; `Rope`: `theta^(-2i/rd)`, multiplier 1
    let (inv, mscale): (Vec<f64>, f64) = if call.op == Op::RopeScaled {
        (read_f32(&call.inputs[1])?.iter().map(|&f| f as f64).collect(), call.attrs.rope_mscale as f64)
    } else {
        let theta = call.attrs.theta as f64;
        ((0..half).map(|i| theta.powf(-2.0 * i as f64 / rd as f64)).collect(), 1.0)
    };
    let mut o = x.clone();
    o.par_chunks_mut(n * d).enumerate().for_each(|(ti, trow)| {
        let pos = (call.attrs.pos0 + ti) as f64;
        let cs: Vec<(f32, f32)> = inv
            .iter()
            .map(|f| {
                let a = pos * f;
                if mscale == 1.0 {
                    (a.cos() as f32, a.sin() as f32)
                } else {
                    ((a.cos() * mscale) as f32, (a.sin() * mscale) as f32)
                }
            })
            .collect();
        for h in 0..n {
            let xr = &x[(ti * n + h) * d..(ti * n + h + 1) * d];
            let or = &mut trow[h * d..(h + 1) * d];
            for i in 0..half {
                let (c, s) = cs[i];
                or[i] = xr[i] * c - xr[i + half] * s;
                or[i + half] = xr[i + half] * c + xr[i] * s;
            }
        }
    });
    write_f32(&call.outputs[0], &o)
}

fn matmul(call: &OpCall) -> Result<()> {
    let xt = &call.inputs[0];
    let wt = &call.inputs[1];
    let (m, k) = (xt.shape[0], xt.shape[1]);
    let n = wt.shape[0];
    let xf = read_f32(xt)?;
    let wb = bytes(wt)?;
    let wsz = wt.dtype.size_bytes();
    const NB: usize = 8;
    let mut ot = vec![0f32; n * m];
    ot.par_chunks_mut(m * NB).enumerate().for_each_init(
        || vec![0f32; k],
        |wbuf, (ci, chunk)| {
            for r in 0..chunk.len() / m {
                let ni = ci * NB + r;
                decode_to_f32(wt.dtype, &wb[ni * k * wsz..(ni + 1) * k * wsz], wbuf);
                for mi in 0..m {
                    chunk[r * m + mi] = dot(&xf[mi * k..(mi + 1) * k], wbuf);
                }
            }
        },
    );
    let mut of = vec![0f32; m * n];
    of.par_chunks_mut(n).enumerate().for_each(|(mi, row)| {
        for (ni, o) in row.iter_mut().enumerate() {
            *o = ot[ni * m + mi];
        }
    });
    write_f32(&call.outputs[0], &of)
}

fn kv_append(call: &OpCall) -> Result<()> {
    let newt = &call.inputs[0];
    let cache = &call.outputs[0];
    let (t, nkv, d) = (newt.shape[0], newt.shape[1], newt.shape[2]);
    let cap = cache.shape[1];
    let src = read_f32(newt)?;
    let csz = cache.dtype.size_bytes();
    // SAFETY: the cache is the op's in-place output.
    let cb = unsafe { bytes_mut(cache)? };
    for ti in 0..t {
        for h in 0..nkv {
            let off = (h * cap + call.attrs.pos0 + ti) * d * csz;
            encode_from_f32(cache.dtype, &src[(ti * nkv + h) * d..(ti * nkv + h + 1) * d], &mut cb[off..off + d * csz]);
        }
    }
    Ok(())
}

fn attention(call: &OpCall) -> Result<()> {
    let qt = &call.inputs[0];
    let (t, n, d) = (qt.shape[0], qt.shape[1], qt.shape[2]);
    let kct = &call.inputs[1];
    let vct = &call.inputs[2];
    let (nkv, cap) = (kct.shape[0], kct.shape[1]);
    let group = n / nkv;
    let pos0 = call.attrs.pos0;
    let l = pos0 + t;
    let scale = call.attrs.scale;
    let q = read_f32(qt)?;
    let kb = bytes(kct)?;
    let vb = bytes(vct)?;
    let csz = kct.dtype.size_bytes();
    // decode the used part of each kv head once
    let dec = |b: &[u8], h: usize| -> Vec<f32> {
        let mut v = vec![0f32; l * d];
        decode_to_f32(kct.dtype, &b[h * cap * d * csz..(h * cap + l) * d * csz], &mut v);
        v
    };
    let ks: Vec<Vec<f32>> = (0..nkv).into_par_iter().map(|h| dec(kb, h)).collect();
    let vs: Vec<Vec<f32>> = (0..nkv).into_par_iter().map(|h| dec(vb, h)).collect();
    let mut o = vec![0f32; t * n * d];
    o.par_chunks_mut(d).enumerate().for_each(|(i, orow)| {
        let (ti, head) = (i / n, i % n);
        let kvh = head / group;
        let visible = pos0 + ti + 1;
        let qrow = &q[(ti * n + head) * d..(ti * n + head + 1) * d];
        let (k, v) = (&ks[kvh], &vs[kvh]);
        let mut s: Vec<f32> = (0..visible).map(|j| scale * dot(qrow, &k[j * d..(j + 1) * d])).collect();
        let mx = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut den = 0f32;
        for x in s.iter_mut() {
            *x = (*x - mx).exp();
            den += *x;
        }
        for (j, p) in s.iter().enumerate() {
            let w = p / den;
            let vr = &v[j * d..(j + 1) * d];
            for (a, b) in orow.iter_mut().zip(vr) {
                *a += w * b;
            }
        }
    });
    write_f32(&call.outputs[0], &o)
}

fn silu_glu(call: &OpCall) -> Result<()> {
    let g = read_f32(&call.inputs[0])?;
    let u = read_f32(&call.inputs[1])?;
    let o: Vec<f32> = g.par_iter().zip(u.par_iter()).map(|(g, u)| (g / (1.0 + (-g).exp())) * u).collect();
    write_f32(&call.outputs[0], &o)
}

fn argmax_op(call: &OpCall) -> Result<()> {
    let lt = &call.inputs[0];
    let (r, v) = (lt.shape[0], lt.shape[1]);
    let l = read_f32(lt)?;
    let ids: Vec<i32> = (0..r).map(|i| argmax(&l[i * v..(i + 1) * v]) as i32).collect();
    // SAFETY: exclusive output.
    let ob = unsafe { bytes_mut(&call.outputs[0])? };
    for (i, id) in ids.iter().enumerate() {
        ob[i * 4..i * 4 + 4].copy_from_slice(&id.to_le_bytes());
    }
    Ok(())
}

fn sample_op(call: &OpCall) -> Result<()> {
    let lt = &call.inputs[0];
    let (r, v) = (lt.shape[0], lt.shape[1]);
    let l = read_f32(lt)?;
    let u = read_f32(&call.inputs[1])?;
    let p = SamplingParams { temperature: call.attrs.temperature, top_k: call.attrs.top_k, top_p: call.attrs.top_p };
    // SAFETY: exclusive output.
    let ob = unsafe { bytes_mut(&call.outputs[0])? };
    for i in 0..r {
        let id = sample_row(&l[i * v..(i + 1) * v], &p, u[i]) as i32;
        ob[i * 4..i * 4 + 4].copy_from_slice(&id.to_le_bytes());
    }
    Ok(())
}
