//! cpu implementations of the DeepSeek-V4.1 ops (`intarsia_core::dsv41::ops`, contract in docs/design/dsv41-ops.md).
//! Plain f64 math transcribed from the official `inference/` code (MIT, DeepSeek); both kernel sets (`ref_*` and
//! `cpu_f32_*`) run this code. Ops in [`STUBS`] have no reference yet: their kernels are registered (so `plan show`
//! lists the full op sequence) and fail with a "not implemented (lane 1b)" error, which admission reports.

use crate::mem::{read_f32, write_f32};
use intarsia_core::dsv41::ops::{compressed_rows, n_blocks, visible_rows, QdqFmt, QdqScale, ScoreFunc};
use intarsia_core::dtype::{f32_to_fp8_e4m3, fp8_e4m3_to_f32};
use intarsia_core::quant::nvfp4::{f32_to_fp4, fp4_to_f32};
use intarsia_core::{Error, Op, OpCall, Result, TensorRef};

/// DeepSeek-V4.1 ops without a cpu reference yet (lane 1b removes entries as it lands them). `MatmulFp8Block` serves only
/// the official FP8 base checkpoint (the Mia EXL3 checkpoint has no fp8 linear); `MoeExpertsExl3` is the grouped EXL3
/// expert GEMM (lane 1b composes it from the EXL3 decoder in `exl3_ref`, lane 2 ports TensorFold's `experts_grouped`).
pub const STUBS: [Op; 2] = [Op::MatmulFp8Block, Op::MoeExpertsExl3];

/// The error a stub op's kernels return.
pub fn stub_error(op: Op) -> Error {
    Error::Unsupported(format!("{op}: not implemented (lane 1b), see docs/design/dsv41-ops.md"))
}

fn rd(t: &TensorRef) -> Result<Vec<f64>> {
    Ok(read_f32(t)?.into_iter().map(|v| v as f64).collect())
}

fn wr(t: &TensorRef, v: &[f64]) -> Result<()> {
    let f: Vec<f32> = v.iter().map(|x| *x as f32).collect();
    write_f32(t, &f)
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// `2^ceil(log2(x))` from the f32 bits (the reference's `fast_round_scale`).
fn pow2_ceil(x: f32) -> f32 {
    let b = x.to_bits();
    let e = ((b >> 23) & 0xFF) as i32 - 127 + i32::from(b & 0x7F_FFFF != 0);
    f32::from_bits(((e + 127).clamp(1, 254) as u32) << 23)
}

/// One block of `ActQdq`: quantise to `fmt` with a scale of `scale` format, dequantise (f32, as the reference kernels).
pub fn qdq_block(x: &mut [f32], fmt: QdqFmt, scale: QdqScale) {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let (fmax, floor) = match (fmt, scale) {
        (QdqFmt::Fp8E4m3, _) => (448f32, 1e-4f32),
        (QdqFmt::Fp4E2m1, QdqScale::E4m3) => (6.0, 6.0 * 2f32.powi(-9)),
        (QdqFmt::Fp4E2m1, _) => (6.0, 6.0 * 2f32.powi(-126)),
    };
    let amax = amax.max(floor);
    let s = match scale {
        QdqScale::Ue8m0 => pow2_ceil(amax * (1.0 / fmax)),
        QdqScale::E4m3 => fp8_e4m3_to_f32(f32_to_fp8_e4m3(amax / fmax)),
        QdqScale::F32 => amax * (1.0 / fmax),
    };
    for v in x.iter_mut() {
        let c = (*v / s).clamp(-fmax, fmax);
        let q = match fmt {
            QdqFmt::Fp8E4m3 => fp8_e4m3_to_f32(f32_to_fp8_e4m3(c)),
            QdqFmt::Fp4E2m1 => fp4_to_f32(f32_to_fp4(c)),
        };
        *v = q * s;
    }
}

/// Indices of the `k` largest values, largest first, ties to the lower index (`torch.topk` leaves ties unspecified).
fn topk_desc(v: &[f64], k: usize) -> Vec<usize> {
    let mut ix: Vec<usize> = (0..v.len()).collect();
    ix.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    ix.truncate(k);
    ix
}

/// Index score of query row `t` against every compressed row (`Indexer.forward`): `scale * sum_h w[t,h] * relu(q[t,h] . k[c])`
/// for the rows the query can see, `-inf` past them.
fn index_scores(call: &OpCall, q: &[f64], k: &[f64], w: &[f64], t: usize) -> Vec<f64> {
    let (hi, di) = (call.inputs[0].shape[1], call.inputs[0].shape[2]);
    let nk = call.inputs[1].shape[0];
    let a = &call.attrs;
    let n_t = visible_rows(a.pos0, t, a.dsv.ratio);
    (0..nk)
        .map(|c| {
            if c >= n_t {
                return f64::NEG_INFINITY;
            }
            let kr = &k[c * di..(c + 1) * di];
            let mut s = 0.0;
            for h in 0..hi {
                let qr = &q[(t * hi + h) * di..(t * hi + h + 1) * di];
                let dot: f64 = qr.iter().zip(kr).map(|(x, y)| x * y).sum();
                s += w[t * hi + h] * dot.max(0.0);
            }
            s * a.scale as f64
        })
        .collect()
}

pub fn run(call: &OpCall) -> Result<()> {
    let (i, o, a) = (&call.inputs, &call.outputs, &call.attrs);
    if STUBS.contains(&call.op) {
        return Err(stub_error(call.op));
    }
    match call.op {
        Op::HcExpand => {
            let x = rd(&i[0])?;
            let (t, s, d) = (o[0].shape[0], o[0].shape[1], o[0].shape[2]);
            let mut y = vec![0f64; t * s * d];
            for r in 0..t {
                for c in 0..s {
                    y[(r * s + c) * d..(r * s + c + 1) * d].copy_from_slice(&x[r * d..(r + 1) * d]);
                }
            }
            wr(&o[0], &y)
        }
        Op::HcMixes => {
            // Block.hc_mixes + kernel.py hc_split_sinkhorn
            let (t, s, d) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
            let (x, f, sc, base) = (rd(&i[0])?, rd(&i[1])?, rd(&i[2])?, rd(&i[3])?);
            let (m, sd) = ((2 + s) * s, s * d);
            let eps = a.dsv.hc_eps as f64;
            let (mut pre, mut post, mut comb) = (vec![0f64; t * s], vec![0f64; t * s], vec![0f64; t * s * s]);
            for r in 0..t {
                let xr = &x[r * sd..(r + 1) * sd];
                let rs = 1.0 / (xr.iter().map(|v| v * v).sum::<f64>() / sd as f64 + a.eps as f64).sqrt();
                let mix: Vec<f64> = (0..m).map(|j| f[j * sd..(j + 1) * sd].iter().zip(xr).map(|(p, q)| p * q).sum::<f64>() * rs).collect();
                for j in 0..s {
                    pre[r * s + j] = sigmoid(mix[j] * sc[0] + base[j]) + eps;
                    post[r * s + j] = 2.0 * sigmoid(mix[s + j] * sc[1] + base[s + j]);
                }
                let mut c: Vec<f64> = (0..s * s).map(|j| mix[2 * s + j] * sc[2] + base[2 * s + j]).collect();
                for row in c.chunks_mut(s) {
                    let mx = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    row.iter_mut().for_each(|v| *v = (*v - mx).exp());
                    let sum: f64 = row.iter().sum();
                    row.iter_mut().for_each(|v| *v = *v / sum + eps);
                }
                let col_norm = |c: &mut Vec<f64>| {
                    for k in 0..s {
                        let sum: f64 = (0..s).map(|j| c[j * s + k]).sum();
                        (0..s).for_each(|j| c[j * s + k] /= sum + eps);
                    }
                };
                col_norm(&mut c);
                for _ in 1..a.dsv.iters {
                    for row in c.chunks_mut(s) {
                        let sum: f64 = row.iter().sum();
                        row.iter_mut().for_each(|v| *v /= sum + eps);
                    }
                    col_norm(&mut c);
                }
                comb[r * s * s..(r + 1) * s * s].copy_from_slice(&c);
            }
            wr(&o[0], &pre)?;
            wr(&o[1], &post)?;
            wr(&o[2], &comb)
        }
        Op::HcPre => {
            let (t, s, d) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
            let (x, mix) = (rd(&i[0])?, rd(&i[1])?);
            let mut y = vec![0f64; t * d];
            for r in 0..t {
                for c in 0..s {
                    let m = mix[r * s + c];
                    for e in 0..d {
                        y[r * d + e] += m * x[(r * s + c) * d + e];
                    }
                }
            }
            wr(&o[0], &y)
        }
        Op::HcPost => {
            let (t, s, d) = (i[1].shape[0], i[1].shape[1], i[1].shape[2]);
            let (x, res, post, comb) = (rd(&i[0])?, rd(&i[1])?, rd(&i[2])?, rd(&i[3])?);
            let mut y = vec![0f64; t * s * d];
            for r in 0..t {
                for k in 0..s {
                    for e in 0..d {
                        let mut v = post[r * s + k] * x[r * d + e];
                        for j in 0..s {
                            v += comb[(r * s + j) * s + k] * res[(r * s + j) * d + e];
                        }
                        y[(r * s + k) * d + e] = v;
                    }
                }
            }
            wr(&o[0], &y)
        }
        Op::MoeRoute => {
            // Gate.forward: the bias picks the experts, the weights come from the unbiased scores
            let (t, e) = (i[0].shape[0], i[0].shape[1]);
            let (lg, bias) = (rd(&i[0])?, rd(&i[1])?);
            let k = a.top_k;
            let (mut w, mut ids) = (vec![0f64; t * k], vec![0f64; t * k]);
            for r in 0..t {
                let l: Vec<f64> = lg[r * e..(r + 1) * e].iter().map(|v| v / a.temperature as f64).collect();
                let sc: Vec<f64> = match a.dsv.score_func {
                    ScoreFunc::SqrtSoftplus => l.iter().map(|v| (v.max(0.0) + (-v.abs()).exp().ln_1p()).sqrt()).collect(),
                    ScoreFunc::Sigmoid => l.iter().map(|v| sigmoid(*v)).collect(),
                    ScoreFunc::Softmax => {
                        let mx = l.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let ex: Vec<f64> = l.iter().map(|v| (v - mx).exp()).collect();
                        let s: f64 = ex.iter().sum();
                        ex.iter().map(|v| v / s).collect()
                    }
                };
                let biased: Vec<f64> = sc.iter().zip(&bias).map(|(s, b)| s + b).collect();
                let top = topk_desc(&biased, k);
                let mut wr_: Vec<f64> = top.iter().map(|&j| sc[j]).collect();
                if a.norm_topk && k > 1 {
                    let s: f64 = wr_.iter().sum::<f64>() + 1e-20;
                    wr_.iter_mut().for_each(|v| *v /= s);
                }
                for (j, (&id, v)) in top.iter().zip(&wr_).enumerate() {
                    w[r * k + j] = v * a.scale as f64;
                    ids[r * k + j] = id as f64;
                }
            }
            wr(&o[0], &w)?;
            wr(&o[1], &ids)
        }
        Op::SiluGluClamp => {
            let (g, u) = (rd(&i[0])?, rd(&i[1])?);
            let l = a.dsv.limit as f64;
            let y: Vec<f64> = g
                .iter()
                .zip(&u)
                .map(|(&g, &u)| {
                    let (g, u) = if l > 0.0 { (g.min(l), u.clamp(-l, l)) } else { (g, u) };
                    g / (1.0 + (-g).exp()) * u
                })
                .collect();
            wr(&o[0], &y)
        }
        Op::ActQdq => {
            let mut x = read_f32(&i[0])?;
            for blk in x.chunks_mut(a.dsv.qdq_block) {
                qdq_block(blk, a.dsv.qdq_fmt, a.dsv.qdq_scale);
            }
            write_f32(&o[0], &x)
        }
        Op::RopeTail => {
            // apply_rotary_emb on x[..., -rd:]: adjacent pairs as complex numbers, angle pos * inv_freq[i] (f32 as the reference)
            let (t, n, d) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
            let (mut x, f) = (rd(&i[0])?, read_f32(&i[1])?);
            let base = d - 2 * f.len();
            for r in 0..t {
                let pos = (a.pos0 + r * a.dsv.pos_stride) as f32;
                for (j, fr) in f.iter().enumerate() {
                    let ang = (pos * fr) as f64;
                    let (sn, cs) = ang.sin_cos();
                    let sn = if a.dsv.inverse { -sn } else { sn };
                    for h in 0..n {
                        let p = (r * n + h) * d + base + 2 * j;
                        let (re, im) = (x[p], x[p + 1]);
                        x[p] = re * cs - im * sn;
                        x[p + 1] = re * sn + im * cs;
                    }
                }
            }
            wr(&o[0], &x)
        }
        Op::WindowKv => {
            let (t, d) = (i[0].shape[0], i[0].shape[1]);
            let (w, wc) = (a.window, o[1].shape[0]);
            let (kv, mut ring) = (rd(&i[0])?, rd(&o[1])?);
            let mut win = vec![0f64; (w + t) * d];
            for r in 0..w {
                if a.pos0 + r >= w {
                    let p = a.pos0 + r - w;
                    win[r * d..(r + 1) * d].copy_from_slice(&ring[(p % wc) * d..(p % wc + 1) * d]);
                }
            }
            win[w * d..].copy_from_slice(&kv);
            for j in t.saturating_sub(wc)..t {
                let slot = (a.pos0 + j) % wc;
                ring[slot * d..(slot + 1) * d].copy_from_slice(&kv[j * d..(j + 1) * d]);
            }
            wr(&o[0], &win)?;
            wr(&o[1], &ring)
        }
        Op::AttnSparse => {
            // kernel.py sparse_attn: one KV vector is key and value; the sink joins the softmax denominator only
            let (t, h, d) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
            let w = a.window;
            let (q, win, cmp, idx, sink) = (rd(&i[0])?, rd(&i[1])?, rd(&i[2])?, read_f32(&i[3])?, rd(&i[4])?);
            let (nc, kc) = (i[2].shape[0], i[3].shape[1]);
            let mut out = vec![0f64; t * h * d];
            for r in 0..t {
                let mut keys: Vec<&[f64]> = Vec::new();
                for row in r + 1..=r + w {
                    if a.pos0 + row >= w {
                        keys.push(&win[row * d..(row + 1) * d]);
                    }
                }
                for j in 0..kc {
                    let c = idx[r * kc + j] as i64;
                    if c >= 0 {
                        let c = c as usize;
                        if c >= nc {
                            return Err(Error::invalid(format!("attn_sparse: index {c} >= {nc} compressed rows")));
                        }
                        keys.push(&cmp[c * d..(c + 1) * d]);
                    }
                }
                if keys.is_empty() {
                    continue;
                }
                for hh in 0..h {
                    let qr = &q[(r * h + hh) * d..(r * h + hh + 1) * d];
                    let s: Vec<f64> = keys.iter().map(|k| qr.iter().zip(k.iter()).map(|(x, y)| x * y).sum::<f64>() * a.scale as f64).collect();
                    let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let e: Vec<f64> = s.iter().map(|v| (v - mx).exp()).collect();
                    let den = e.iter().sum::<f64>() + (sink[hh] - mx).exp();
                    let o_ = &mut out[(r * h + hh) * d..(r * h + hh + 1) * d];
                    for (ej, k) in e.iter().zip(&keys) {
                        for (ov, kv) in o_.iter_mut().zip(k.iter()) {
                            *ov += ej / den * kv;
                        }
                    }
                }
            }
            wr(&o[0], &out)
        }
        Op::CompressPool => {
            // Compressor.forward (ratio > 1): softmax-gated pooling per channel over each complete group of `ratio` tokens
            let (t, d) = (i[0].shape[0], i[0].shape[1]);
            let r = a.dsv.ratio;
            let rc = o[1].shape[0];
            let (kv, sc, mut rkv, mut rsc) = (rd(&i[0])?, rd(&i[1])?, rd(&o[1])?, rd(&o[2])?);
            let p0 = a.pos0;
            let (_, g) = compressed_rows(p0, t, r);
            let mut out = vec![0f64; g * d];
            for (gi, grp) in (p0 / r..p0 / r + g).enumerate() {
                let rows: Vec<(&[f64], &[f64])> = (grp * r..(grp + 1) * r)
                    .map(|p| {
                        if p >= p0 {
                            let j = p - p0;
                            (&kv[j * d..(j + 1) * d], &sc[j * d..(j + 1) * d])
                        } else {
                            let s = p % rc;
                            (&rkv[s * d..(s + 1) * d], &rsc[s * d..(s + 1) * d])
                        }
                    })
                    .collect();
                for c in 0..d {
                    let mx = rows.iter().map(|(_, s)| s[c]).fold(f64::NEG_INFINITY, f64::max);
                    let den: f64 = rows.iter().map(|(_, s)| (s[c] - mx).exp()).sum();
                    out[gi * d + c] = rows.iter().map(|(k, s)| k[c] * (s[c] - mx).exp() / den).sum();
                }
            }
            for j in t.saturating_sub(rc)..t {
                let s = (p0 + j) % rc;
                rkv[s * d..(s + 1) * d].copy_from_slice(&kv[j * d..(j + 1) * d]);
                rsc[s * d..(s + 1) * d].copy_from_slice(&sc[j * d..(j + 1) * d]);
            }
            wr(&o[0], &out)?;
            wr(&o[1], &rkv)?;
            wr(&o[2], &rsc)
        }
        Op::IndexerBlocks => {
            // select_candidate_blocks: block score = best position; the query's newest block is pinned in
            let t = i[0].shape[0];
            let nk = i[1].shape[0];
            let b = a.dsv.block;
            let nb = n_blocks(nk, b);
            let (q, k, w) = (rd(&i[0])?, rd(&i[1])?, rd(&i[2])?);
            let mut mask = vec![0f64; t * nb];
            for r in 0..t {
                let s = index_scores(call, &q, &k, &w, r);
                let mut bs: Vec<f64> = (0..nb).map(|j| s[j * b..((j + 1) * b).min(nk)].iter().cloned().fold(f64::NEG_INFINITY, f64::max)).collect();
                let n_t = visible_rows(a.pos0, r, a.dsv.ratio);
                if n_t > 0 {
                    bs[(n_t - 1) / b] = f64::INFINITY;
                }
                for j in topk_desc(&bs, a.top_k.min(nb)) {
                    if bs[j] > f64::NEG_INFINITY {
                        mask[r * nb + j] = 1.0;
                    }
                }
            }
            wr(&o[0], &mask)
        }
        Op::IndexerTopk => {
            let t = i[0].shape[0];
            let nk = i[1].shape[0];
            let b = a.dsv.block;
            let nb = n_blocks(nk, b);
            let (q, k, w) = (rd(&i[0])?, rd(&i[1])?, rd(&i[2])?);
            let cand = if i[3].numel() == 0 { None } else { Some(read_f32(&i[3])?) };
            let kk = o[0].shape[1];
            let mut idx = vec![0f64; t * kk];
            for r in 0..t {
                let mut s = index_scores(call, &q, &k, &w, r);
                if let Some(m) = &cand {
                    for (c, v) in s.iter_mut().enumerate() {
                        if m[r * nb + c / b] == 0.0 {
                            *v = f64::NEG_INFINITY;
                        }
                    }
                }
                let mut top = topk_desc(&s, kk);
                top.sort_unstable();
                let n_t = visible_rows(a.pos0, r, a.dsv.ratio);
                for (j, c) in top.into_iter().enumerate() {
                    idx[r * kk + j] = if c < n_t { c as f64 } else { -1.0 };
                }
            }
            wr(&o[0], &idx)
        }
        Op::GroupedMatmul => {
            let t = i[0].shape[0];
            let (g, n, k) = (i[1].shape[0], i[1].shape[1], i[1].shape[2]);
            let (x, w) = (rd(&i[0])?, rd(&i[1])?);
            let mut y = vec![0f64; t * g * n];
            for r in 0..t {
                for gg in 0..g {
                    let xr = &x[r * g * k + gg * k..r * g * k + (gg + 1) * k];
                    for nn in 0..n {
                        let wr_ = &w[(gg * n + nn) * k..(gg * n + nn + 1) * k];
                        y[r * g * n + gg * n + nn] = xr.iter().zip(wr_).map(|(p, q)| p * q).sum();
                    }
                }
            }
            wr(&o[0], &y)
        }
        Op::Transpose01 => {
            let (aa, bb, c) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
            let x = read_f32(&i[0])?;
            let mut y = vec![0f32; aa * bb * c];
            for p in 0..aa {
                for q in 0..bb {
                    y[(q * aa + p) * c..(q * aa + p + 1) * c].copy_from_slice(&x[(p * bb + q) * c..(p * bb + q + 1) * c]);
                }
            }
            write_f32(&o[0], &y)
        }
        Op::EngramGather => {
            // ParallelEngramEmbedding.forward: fp8 rows times their E8M0 block scales
            let (rows, sc) = (read_f32(&i[0])?, read_f32(&i[1])?);
            let y: Vec<f32> = rows.iter().enumerate().map(|(j, v)| v * 2f32.powi(sc[j / 32] as i32 - 127)).collect();
            write_f32(&o[0], &y)
        }
        Op::EngramMerge => {
            // Engram.forward: per stream, a normalised dot of the stream with its key gates the shared value
            let (t, s, d) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
            let (h, kv, qw, kw) = (rd(&i[0])?, rd(&i[1])?, rd(&i[2])?, rd(&i[3])?);
            let eps = a.eps as f64;
            let mut y = h.clone();
            for r in 0..t {
                let row = &kv[r * (s + 1) * d..(r + 1) * (s + 1) * d];
                let value = &row[s * d..];
                for c in 0..s {
                    let hs = &h[(r * s + c) * d..(r * s + c + 1) * d];
                    let key = &row[c * d..(c + 1) * d];
                    let ms = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>() / d as f64;
                    let rstd = 1.0 / (ms(hs) + eps).sqrt() / (ms(key) + eps).sqrt();
                    let dot: f64 = (0..d).map(|e| hs[e] * qw[c * d + e] * kw[c * d + e] * key[e]).sum::<f64>() * rstd / (d as f64).sqrt();
                    let gate = sigmoid(dot.abs().max(1e-6).sqrt().copysign(dot));
                    for e in 0..d {
                        y[(r * s + c) * d + e] = hs[e] + gate * value[e];
                    }
                }
            }
            wr(&o[0], &y)
        }
        op => Err(Error::invalid(format!("dsv41_ref::run called for {op}"))),
    }
}
