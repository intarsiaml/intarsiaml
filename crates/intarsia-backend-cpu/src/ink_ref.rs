//! The Inkling-2 ops (`Op::AttnResMix` .. `Op::ConvMix`; contract in `intarsia_core::inkling2::ops`): the f64 reference, one
//! implementation for both cpu kernel sets. Every reduction is f64; values are rounded to the call's activation dtype exactly
//! where the contract writes `rA(..)`. Rows run in parallel (rayon), each in a fixed order, so results do not depend on the
//! number of rows in a call (rows-exact). `InkInProj`'s fast kernel is three `Matmul` launches of the fast path.

use crate::mem::{read_f32, write_f32};
use intarsia_core::inkling2::ops::{nq, SRC_EPS};
use intarsia_core::quant::reference::{e4m3_table, round};
use intarsia_core::{DType, Error, Op, OpAttrs, OpCall, Result, TensorRef};
use rayon::prelude::*;

fn rd(t: &TensorRef) -> Result<Vec<f64>> {
    Ok(read_f32(t)?.into_iter().map(|v| v as f64).collect())
}

fn wr(t: &TensorRef, v: &[f64]) -> Result<()> {
    let f: Vec<f32> = v.iter().map(|&x| x as f32).collect();
    write_f32(t, &f)
}

fn absent(t: &TensorRef) -> bool {
    t.numel() == 0
}

#[inline]
fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// `1 / sqrt(mean(x^2) + eps)`.
fn inv_rms(x: &[f64], eps: f64) -> f64 {
    let ss: f64 = x.iter().map(|v| v * v).sum();
    1.0 / (ss / x.len() as f64 + eps).sqrt()
}

/// `x * min(th / (|x| + SRC_EPS), 1)`.
fn clip(x: &mut [f64], th: f64) {
    let n: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    let f = (th / (n + SRC_EPS)).min(1.0);
    x.iter_mut().for_each(|v| *v *= f);
}

/// The depth mix of one row: `srcs` the N+1 sources (the last one already summed with its addend), queries `[Q,D]`.
/// Returns the Q unnormed f64 mixes.
fn mix_row(srcs: &[&[f64]], last: &[f64], queries: &[f64], nqry: usize, d: usize) -> Vec<Vec<f64>> {
    let all: Vec<&[f64]> = srcs.iter().copied().chain(std::iter::once(last)).collect();
    let w: Vec<f64> = all.iter().map(|x| inv_rms(x, SRC_EPS)).collect();
    (0..nqry)
        .map(|q| {
            let qv = &queries[q * d..(q + 1) * d];
            let l: Vec<f64> = all.iter().zip(&w).map(|(x, wn)| qv.iter().zip(x.iter()).map(|(a, b)| a * b).sum::<f64>() * wn).collect();
            let mx = l.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = l.iter().map(|v| (v - mx).exp()).collect();
            let s: f64 = e.iter().sum();
            let mut out = vec![0f64; d];
            for ((x, wn), en) in all.iter().zip(&w).zip(&e) {
                let c = en / s * wn;
                for (o, v) in out.iter_mut().zip(x.iter()) {
                    *o += c * v;
                }
            }
            out
        })
        .collect()
}

/// `rA(x * inv(x) * w)`.
fn norm_row(x: &[f64], w: &[f64], eps: f64, a: DType) -> Vec<f64> {
    let r = inv_rms(x, eps);
    x.iter().zip(w).map(|(v, g)| round(a, v * r * g)).collect()
}

/// The controller of one stream and row (`LoopBlend`): down `[G,2D]`, gain / forget `[D,G]`, norm `[D]`.
#[allow(clippy::too_many_arguments)]
fn blend_row(dest: &[f64], msg: &[f64], down: &[f64], gain: &[f64], forget: &[f64], norm: &[f64], g: usize, eps: f64, a: DType) -> Vec<f64> {
    let d = dest.len();
    let (wd, wm) = (inv_rms(dest, SRC_EPS), inv_rms(msg, SRC_EPS));
    let pair: Vec<f64> = dest.iter().map(|v| round(a, v * wd)).chain(msg.iter().map(|v| round(a, v * wm))).collect();
    let h: Vec<f64> = (0..g).map(|gi| round(a, down[gi * 2 * d..(gi + 1) * 2 * d].iter().zip(&pair).map(|(x, y)| x * y).sum())).collect();
    let y: Vec<f64> = (0..d)
        .map(|c| {
            let gn: f64 = gain[c * g..(c + 1) * g].iter().zip(&h).map(|(x, y)| x * y).sum();
            let fg: f64 = forget[c * g..(c + 1) * g].iter().zip(&h).map(|(x, y)| x * y).sum();
            (1.0 - fg.tanh()) * dest[c] + gn.tanh() * msg[c]
        })
        .collect();
    norm_row(&y, norm, eps, a)
}

/// The causal conv over a chunk: `x` `[T,C]` (the call's rows), lag-major `taps` `[K,C]`, optional per-row dynamic taps
/// `dyn_[t][c / dh][l]` (row stride `ds`, at column offset `doff` of each row of `dynsrc`), the ring `[Rr,C]` (read as before
/// the call, then updated). Returns the unrounded `(sum_l w * x(p - l) + x(p)) * scale`.
#[allow(clippy::too_many_arguments)]
fn conv(x: &[f64], t: usize, c: usize, taps: &[f64], k: usize, dynsrc: Option<(&[f64], usize, usize, usize)>, ring: &mut [f64], pos0: usize, scale: f64) -> Vec<f64> {
    let rr = ring.len() / c;
    let at = |q: isize, ch: usize| -> f64 {
        if q < 0 {
            0.0
        } else if q as usize >= pos0 {
            x[(q as usize - pos0) * c + ch]
        } else {
            ring[(q as usize % rr) * c + ch]
        }
    };
    let mut y = vec![0f64; t * c];
    for ti in 0..t {
        let p = (pos0 + ti) as isize;
        for ch in 0..c {
            let mut acc = 0f64;
            for l in 0..k {
                let mut w = taps[l * c + ch];
                if let Some((src, stride, off, dh)) = dynsrc {
                    w += src[ti * stride + off + (ch / dh) * k + l];
                }
                acc += w * at(p - l as isize, ch);
            }
            y[ti * c + ch] = (acc + x[ti * c + ch]) * scale;
        }
    }
    // the ring gets the last `rr` positions of the chunk
    for q in (pos0 + t).saturating_sub(rr).max(pos0)..pos0 + t {
        ring[(q % rr) * c..(q % rr + 1) * c].copy_from_slice(&x[(q - pos0) * c..(q - pos0 + 1) * c]);
    }
    y
}

/// `SconvRes`' part of `SconvRes` / `ConvMix`: writes y, the ring and the sum; returns (y, sum or empty, act).
fn sconv_part(call: &OpCall) -> Result<(Vec<f64>, Vec<f64>, DType)> {
    let (i, o) = (&call.inputs, &call.outputs);
    let (t, c, a) = (i[0].shape[0], i[0].shape[1], i[0].dtype);
    let k = i[1].shape[0];
    let x = rd(&i[0])?;
    let taps = rd(&i[1])?;
    let mut ring = rd(&o[1])?;
    let y: Vec<f64> = conv(&x, t, c, &taps, k, None, &mut ring, call.attrs.pos0, ((k + 1) as f64).powf(-0.5)).into_iter().map(|v| round(a, v)).collect();
    wr(&o[0], &y)?;
    wr(&o[1], &ring)?;
    let sum = if absent(&o[2]) {
        Vec::new()
    } else {
        let res = rd(&i[2])?;
        let add = if absent(&i[3]) { vec![0f64; t * c] } else { rd(&i[3])? };
        // f32 sums in the contract's order: (res + addend) + y
        let s: Vec<f64> = (0..t * c).map(|j| round(a, ((res[j] as f32 + add[j] as f32) + y[j] as f32) as f64)).collect();
        wr(&o[2], &s)?;
        s
    };
    Ok((y, sum, a))
}

/// The mix (+ blend / norm) of `ConvMix` and `AttnResMix`. `src` `[N,R,D]`, `last` `[R,D]` (addend summed), returns `[S|Q,R,D]`.
#[allow(clippy::too_many_arguments)]
fn mix_all(
    src: &[f64],
    n: usize,
    last: &[f64],
    r: usize,
    d: usize,
    queries: &[f64],
    nqry: usize,
    norm: Option<&[f64]>,
    blend: Option<(&[f64], &[f64], &[f64], &[f64], usize, usize)>,
    eps: f64,
    a: DType,
) -> Vec<f64> {
    let rows: Vec<Vec<Vec<f64>>> = (0..r)
        .into_par_iter()
        .map(|ri| {
            let srcs: Vec<&[f64]> = (0..n).map(|ni| &src[(ni * r + ri) * d..(ni * r + ri + 1) * d]).collect();
            let mixes = mix_row(&srcs, &last[ri * d..(ri + 1) * d], queries, nqry, d);
            match (blend, norm) {
                (Some((msg, down, gain, forget, s, g)), Some(nw)) => (0..s)
                    .map(|si| {
                        let dest = &mixes[if nqry == 1 { 0 } else { si }];
                        let m = &msg[(si * r + ri) * d..(si * r + ri + 1) * d];
                        blend_row(dest, m, &down[si * g * 2 * d..(si + 1) * g * 2 * d], &gain[si * d * g..(si + 1) * d * g], &forget[si * d * g..(si + 1) * d * g], &nw[si * d..(si + 1) * d], g, eps, a)
                    })
                    .collect(),
                (None, Some(nw)) => mixes.iter().enumerate().map(|(q, m)| norm_row(m, &nw[q * d..(q + 1) * d], eps, a)).collect(),
                _ => mixes.into_iter().map(|m| m.into_iter().map(|v| v as f32 as f64).collect()).collect(),
            }
        })
        .collect();
    let s = rows.first().map_or(0, |x| x.len());
    let mut out = vec![0f64; s * r * d];
    for (ri, row) in rows.iter().enumerate() {
        for (si, v) in row.iter().enumerate() {
            out[(si * r + ri) * d..(si * r + ri + 1) * d].copy_from_slice(v);
        }
    }
    out
}

fn opt(t: &TensorRef) -> Result<Option<Vec<f64>>> {
    if absent(t) {
        Ok(None)
    } else {
        rd(t).map(Some)
    }
}

pub fn run(call: &OpCall, fast: bool) -> Result<()> {
    let (i, o, at) = (&call.inputs, &call.outputs, &call.attrs);
    let eps = at.eps as f64;
    match call.op {
        Op::AttnResMix => {
            let (r, d, a) = (i[1].shape[0], i[1].shape[1], i[1].dtype);
            let n = i[0].shape[0];
            let mut last = rd(&i[1])?;
            if let Some(add) = opt(&i[2])? {
                // f32 sum, not rounded
                last.iter_mut().zip(&add).for_each(|(l, x)| *l = (*l as f32 + *x as f32) as f64);
            }
            let q = rd(&i[3])?;
            let norm = opt(&i[4])?;
            let out = mix_all(&rd(&i[0])?, n, &last, r, d, &q, i[3].shape[0], norm.as_deref(), None, eps, a);
            wr(&o[0], &out)
        }
        Op::LoopBlend => {
            let (s, r, d, a) = (o[0].shape[0], o[0].shape[1], o[0].shape[2], o[0].dtype);
            let sd = i[0].shape[0];
            let g = i[2].shape[1];
            let (dest, msg, down, gain, forget, norm) = (rd(&i[0])?, rd(&i[1])?, rd(&i[2])?, rd(&i[3])?, rd(&i[4])?, rd(&i[5])?);
            let rows: Vec<Vec<Vec<f64>>> = (0..r)
                .into_par_iter()
                .map(|ri| {
                    (0..s)
                        .map(|si| {
                            let ds = if sd == 1 { 0 } else { si };
                            blend_row(
                                &dest[(ds * r + ri) * d..(ds * r + ri + 1) * d],
                                &msg[(si * r + ri) * d..(si * r + ri + 1) * d],
                                &down[si * g * 2 * d..(si + 1) * g * 2 * d],
                                &gain[si * d * g..(si + 1) * d * g],
                                &forget[si * d * g..(si + 1) * d * g],
                                &norm[si * d..(si + 1) * d],
                                g,
                                eps,
                                a,
                            )
                        })
                        .collect()
                })
                .collect();
            let mut out = vec![0f64; s * r * d];
            for (ri, row) in rows.iter().enumerate() {
                for (si, v) in row.iter().enumerate() {
                    out[(si * r + ri) * d..(si * r + ri + 1) * d].copy_from_slice(v);
                }
            }
            wr(&o[0], &out)
        }
        Op::SconvRes => sconv_part(call).map(|_| ()),
        Op::ConvMix => {
            let (y, sum, a) = sconv_part(call)?;
            let nqry = i[5].shape[0];
            if nqry == 0 {
                return Ok(());
            }
            let (t, d) = (i[0].shape[0], i[0].shape[1]);
            let last: Vec<f64> = if !sum.is_empty() {
                sum
            } else if !absent(&i[2]) {
                let res = rd(&i[2])?;
                res.iter().zip(&y).map(|(r, v)| (*r as f32 + *v as f32) as f64).collect()
            } else {
                y
            };
            let q = rd(&i[5])?;
            let norm = opt(&i[6])?;
            let blend = if absent(&i[7]) { None } else { Some((rd(&i[7])?, rd(&i[8])?, rd(&i[9])?, rd(&i[10])?, i[7].shape[0], i[8].shape[1])) };
            let out = mix_all(
                &rd(&i[4])?,
                i[4].shape[0],
                &last,
                t,
                d,
                &q,
                nqry,
                norm.as_deref(),
                blend.as_ref().map(|(m, dn, gn, fg, s, g)| (m.as_slice(), dn.as_slice(), gn.as_slice(), fg.as_slice(), *s, *g)),
                eps,
                a,
            );
            wr(&o[3], &out)
        }
        Op::InkKvPrologue => prologue(call),
        Op::RelSinkAttn => attention(call),
        Op::InkAttnDecode => attn_decode(call),
        Op::MoeExpertsMxfp8 => experts_mxfp8(call),
        // e4m3 weights (lane ink-fp8): the reference is also the fast path (f64, rows in parallel)
        Op::InkInProj if fast && i.len() == 2 => in_proj_fast(call),
        Op::InkInProj => {
            let (s, t, d, a) = (i[0].shape[0], i[0].shape[1], i[0].shape[2], i[0].dtype);
            let (x, w) = (rd(&i[0])?, rd(&i[1])?);
            // `scale * sum(x * e4m3(w))` per pack, `MatmulFp8`'s W8A16 order; 1 for float weights
            let sc: Vec<f64> = if i.len() == 3 { rd(&i[2])? } else { vec![1.0; 3] };
            let mut row0 = 0;
            for (j, out) in o.iter().enumerate() {
                let n = out.shape[1];
                let xs = &x[j.min(s - 1) * t * d..(j.min(s - 1) + 1) * t * d];
                let ws = &w[row0 * d..(row0 + n) * d];
                let y: Vec<f64> = (0..t * n)
                    .into_par_iter()
                    .map(|idx| {
                        let (ti, ni) = (idx / n, idx % n);
                        let dot: f64 = xs[ti * d..(ti + 1) * d].iter().zip(&ws[ni * d..(ni + 1) * d]).map(|(p, q)| p * q).sum();
                        round(a, if i.len() == 3 { dot * sc[j] } else { dot })
                    })
                    .collect();
                wr(out, &y)?;
                row0 += n;
            }
            Ok(())
        }
        other => Err(Error::invalid(format!("ink_ref cannot run {other}"))),
    }
}

/// `InkInProj` as three `Matmul`s of the fast path (views of the input row block and of the weight rows).
fn in_proj_fast(call: &OpCall) -> Result<()> {
    let (i, o) = (&call.inputs, &call.outputs);
    let (s, t, d) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
    let mut row0 = 0;
    for (j, out) in o.iter().enumerate() {
        let n = out.shape[1];
        let x = i[0].rows(j.min(s - 1), 1).reshaped(&[t, d]);
        let w = i[1].rows(row0, n);
        crate::fast::run(&OpCall::new(Op::Matmul, vec![x, w], vec![out.clone()], OpAttrs::default()))?;
        row0 += n;
    }
    Ok(())
}

fn prologue(call: &OpCall) -> Result<()> {
    let (i, o, at) = (&call.inputs, &call.outputs, &call.attrs);
    let ink = &at.ink;
    let (h, hk, dh, dr) = (at.n_heads, at.n_kv_heads, at.head_dim, ink.d_rel);
    let (t, a) = (i[0].shape[0], i[0].dtype);
    let ckv = hk * dh;
    let k = i[3].shape[0];
    let e = i[7].shape[1];
    let pw = i[1].shape[1];
    let (pq, pk, pv) = (rd(&i[0])?, rd(&i[1])?, rd(&i[2])?);
    let (wk, wv, qs, rs, rp) = (rd(&i[3])?, rd(&i[4])?, rd(&i[5])?, rd(&i[6])?, rd(&i[7])?);
    let (mut kring, mut vring) = (rd(&o[4])?, rd(&o[5])?);
    let scale = if ink.dyn_conv { ((2 * k + 1) as f64).powf(-0.5) } else { ((k + 1) as f64).powf(-0.5) };
    let part = |p: &[f64]| -> Vec<f64> { (0..t).flat_map(|ti| p[ti * pw..ti * pw + ckv].to_vec()).collect() };
    let (kx, vx) = (part(&pk), part(&pv));
    let dynk = ink.dyn_conv.then_some((pk.as_slice(), pw, ckv, dh));
    let dynv = ink.dyn_conv.then_some((pv.as_slice(), pw, ckv, dh));
    let kc: Vec<f64> = conv(&kx, t, ckv, &wk, k, dynk, &mut kring, at.pos0, scale).into_iter().map(|v| round(a, v)).collect();
    let vc: Vec<f64> = conv(&vx, t, ckv, &wv, k, dynv, &mut vring, at.pos0, scale).into_iter().map(|v| round(a, v)).collect();
    let (qth, rth) = (ink.qk_clip as f64, ink.rel_clip as f64);
    let mut kk = vec![0f64; t * ckv];
    let mut vv = vec![0f64; t * ckv];
    for row in 0..t * hk {
        let sl = row * dh..(row + 1) * dh;
        let mut x = kc[sl.clone()].to_vec();
        clip(&mut x, qth);
        kk[sl.clone()].iter_mut().zip(&x).for_each(|(o, v)| *o = round(a, *v));
        let r = inv_rms(&vc[sl.clone()], eps_of(at));
        vv[sl.clone()].iter_mut().zip(&vc[sl]).for_each(|(o, v)| *o = round(a, v * r));
    }
    let nqw = nq(h, dh, dr);
    let tau = |p: usize| -> f64 {
        if ink.tau_floor == 0 {
            1.0
        } else {
            1.0 + ink.tau_alpha as f64 * (((p + 1) as f64 / ink.tau_floor as f64).max(1.0)).ln()
        }
    };
    let mut q = vec![0f64; t * h * dh];
    let mut rel = vec![0f64; t * h * e];
    for ti in 0..t {
        let tv = tau(at.pos0 + ti);
        for hi in 0..h {
            let mut qv: Vec<f64> = (0..dh).map(|j| pq[ti * nqw + hi * dh + j] * qs[hi * dh + j]).collect();
            clip(&mut qv, qth);
            let mut qv: Vec<f64> = qv.into_iter().map(|v| round(a, v)).collect();
            let roff = h * dh + hi * dr;
            let mut rv: Vec<f64> = (0..dr).map(|j| pq[ti * nqw + roff + j] * rs[hi * dr + j]).collect();
            clip(&mut rv, rth);
            let mut rv: Vec<f64> = rv.into_iter().map(|v| round(a, v)).collect();
            if ink.tau_floor > 0 {
                qv.iter_mut().for_each(|v| *v = round(a, *v * tv));
                rv.iter_mut().for_each(|v| *v = round(a, *v * tv));
            }
            q[(ti * h + hi) * dh..(ti * h + hi + 1) * dh].copy_from_slice(&qv);
            for ei in 0..e {
                rel[(ti * h + hi) * e + ei] = round(a, (0..dr).map(|j| rv[j] * rp[j * e + ei]).sum());
            }
        }
    }
    wr(&o[0], &q)?;
    wr(&o[1], &rel)?;
    wr(&o[2], &kk)?;
    wr(&o[3], &vv)?;
    wr(&o[4], &kring)?;
    wr(&o[5], &vring)
}

/// `InkAttnDecode`: the prologue into scratch intermediates (rounded to the activation dtype as that op writes them), then
/// the attention; bitwise the pair by construction.
fn attn_decode(call: &OpCall) -> Result<()> {
    let a = call.inputs[0].dtype;
    let tmp = |shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        TensorRef::new(intarsia_core::backend::Buffer::new(crate::mem::CpuBuffer::zeroed((n * a.size_bytes()).max(1))), a, &shape)
    };
    let [qs, rs, ks, vs] = intarsia_core::inkling2::ops::attn_decode_inter(call);
    let (pro, att) = intarsia_core::inkling2::ops::attn_decode_parts(call, tmp(qs), tmp(rs), tmp(ks), tmp(vs));
    prologue(&pro)?;
    attention(&att)
}

fn eps_of(at: &OpAttrs) -> f64 {
    at.eps as f64
}

fn attention(call: &OpCall) -> Result<()> {
    let (i, o, at) = (&call.inputs, &call.outputs, &call.attrs);
    let (h, hk, dh, dr) = (at.n_heads, at.n_kv_heads, at.head_dim, at.ink.d_rel);
    let (t, a) = (i[0].shape[0], i[0].dtype);
    let cr = i[3].shape[1];
    let e = i[5].shape[2];
    let grp = h / hk;
    let (q, kn, vn, kc, vc, rel, pq) = (rd(&i[0])?, rd(&i[1])?, rd(&i[2])?, rd(&o[1])?, rd(&o[2])?, rd(&i[5])?, rd(&i[6])?);
    let nqw = nq(h, dh, dr);
    let (gate_off, sink_off) = (h * dh + h * dr, 2 * h * dh + h * dr);
    let ring = at.ink.kv_ring;
    let pos0 = at.pos0;
    let scale = at.scale as f64;
    let row = |j: usize| if ring { j % cr } else { j };
    let out: Vec<Vec<f64>> = (0..t * h)
        .into_par_iter()
        .map(|idx| {
            let (ti, hi) = (idx / h, idx % h);
            let g = hi / grp;
            let p = pos0 + ti;
            let lo = if at.window > 0 { (p + 1).saturating_sub(at.window) } else { 0 };
            let qv = &q[(ti * h + hi) * dh..(ti * h + hi + 1) * dh];
            let key = |j: usize| -> (&[f64], &[f64]) {
                if j >= pos0 {
                    let b = ((j - pos0) * hk + g) * dh;
                    (&kn[b..b + dh], &vn[b..b + dh])
                } else {
                    let b = (g * cr + row(j)) * dh;
                    (&kc[b..b + dh], &vc[b..b + dh])
                }
            };
            let s: Vec<f64> = (lo..=p)
                .map(|j| {
                    let d = p - j;
                    let bias = if d < e { rel[(ti * h + hi) * e + d] } else { 0.0 };
                    scale * qv.iter().zip(key(j).0).map(|(x, y)| x * y).sum::<f64>() + bias
                })
                .collect();
            let sink = pq[ti * nqw + sink_off + hi];
            let m = s.iter().cloned().fold(sink, f64::max);
            let ex: Vec<f64> = s.iter().map(|v| (v - m).exp()).collect();
            let den: f64 = ex.iter().sum::<f64>() + (sink - m).exp();
            let mut o = vec![0f64; dh];
            for (j, w) in (lo..=p).zip(&ex) {
                for (oi, v) in o.iter_mut().zip(key(j).1) {
                    *oi += w * v;
                }
            }
            o.iter()
                .enumerate()
                .map(|(c, v)| round(a, v / den * sigmoid(pq[ti * nqw + gate_off + hi * dh + c])))
                .collect()
        })
        .collect();
    wr(&o[0], &out.concat())?;
    // the caches get the call's rows after the attention read them
    let (mut kc, mut vc) = (kc, vc);
    let from = if ring { (pos0 + t).saturating_sub(cr).max(pos0) } else { pos0 };
    for j in from..pos0 + t {
        for g in 0..hk {
            let (src, dst) = (((j - pos0) * hk + g) * dh, (g * cr + row(j)) * dh);
            kc[dst..dst + dh].copy_from_slice(&kn[src..src + dh]);
            vc[dst..dst + dh].copy_from_slice(&vn[src..src + dh]);
        }
    }
    wr(&o[1], &kc)?;
    wr(&o[2], &vc)
}

/// One expert's activation epilogue on (gate, up) pre-activations (already rounded), weight `w` of the slot.
pub(crate) fn expert_act(g: f64, u: f64, w: f64, at: &OpAttrs, dt: DType) -> f64 {
    let ink = &at.ink;
    if ink.situ {
        let (gc, uc) = (ink.gate_cap as f64, ink.up_cap as f64);
        round(dt, gc * (g / gc).tanh() * sigmoid(g) * uc * (u / uc).tanh() * w)
    } else {
        let act = if at.gelu { intarsia_core::quant::reference::gelu_tanh(g) } else { g * sigmoid(g) };
        round(dt, round(dt, act) * u)
    }
}

fn experts_mxfp8(call: &OpCall) -> Result<()> {
    let (i, o, at) = (&call.inputs, &call.outputs, &call.attrs);
    let (t, hd, dt) = (i[0].shape[0], i[0].shape[1], i[0].dtype);
    let k = i[1].shape[1];
    let (e, two_i) = (i[3].shape[0], i[3].shape[1]);
    let inter = two_i / 2;
    let od = o[0].dtype;
    let x = rd(&i[0])?;
    let ids: Vec<i32> = crate::mem::bytes(&i[1])?.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let wts = read_f32(&i[2])?;
    let (w13, s13, w2, s2) = (crate::mem::bytes(&i[3])?, crate::mem::bytes(&i[4])?, crate::mem::bytes(&i[5])?, crate::mem::bytes(&i[6])?);
    let lut = e4m3_table();
    let p2: Vec<f64> = (0..256).map(|v| (2f64).powi(v as i32 - 127)).collect();
    // `x . W[n]` for row `n` of expert `ex` of an `[E,N,Kd]` e4m3 tensor with `[E,N,Kd/32]` E8M0 scales (f64, block by block)
    let dot = |w: &[u8], s: &[u8], ex: usize, n_rows: usize, kd: usize, n: usize, x: &[f64]| -> f64 {
        let base = (ex * n_rows + n) * kd;
        let sb = (ex * n_rows + n) * (kd / 32);
        let mut acc = 0f64;
        for blk in 0..kd / 32 {
            let sc = p2[s[sb + blk] as usize];
            for c in blk * 32..(blk + 1) * 32 {
                acc += lut[w[base + c] as usize] as f64 * sc * x[c];
            }
        }
        acc
    };
    let token = |ti: usize| -> Vec<f64> {
        let xr = &x[ti * hd..(ti + 1) * hd];
        let mut acc = vec![0f64; hd];
        for slot in 0..k {
            let id = ids[ti * k + slot];
            if id < 0 || id as usize >= e {
                continue;
            }
            let ex = id as usize;
            let w = wts[ti * k + slot] as f64;
            let act: Vec<f64> = (0..inter)
                .into_par_iter()
                .map(|r| {
                    let g = round(dt, dot(w13, s13, ex, two_i, hd, r, xr));
                    let u = round(dt, dot(w13, s13, ex, two_i, hd, inter + r, xr));
                    expert_act(g, u, w, at, dt)
                })
                .collect();
            let ys: Vec<f64> = (0..hd).into_par_iter().map(|c| round(dt, dot(w2, s2, ex, hd, inter, c, &act))).collect();
            for (av, y) in acc.iter_mut().zip(ys) {
                *av += if at.ink.situ { y } else { round(dt, y * w) };
            }
        }
        acc.into_iter().map(|v| round(od, v)).collect()
    };
    let rows: Vec<Vec<f64>> = (0..t).into_par_iter().map(token).collect();
    wr(&o[0], &rows.concat())
}
