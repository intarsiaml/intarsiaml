//! cpu implementations of the quantised-weight and MoE ops. The math lives in
//! `intarsia_core::quant::reference` (f64, checked against HF/numpy numbers in the unit tests); this file reads
//! the tensors, parallelises over independent rows/tokens and rounds to the output dtype.
//! Both kernel sets (`ref_*` and `cpu_f32_*`) run this code.

use crate::mem::{bytes, read_f32, write_f32};
use intarsia_core::quant::reference as r;
use intarsia_core::quant::Mxfp4Act;
use intarsia_core::{Error, Op, OpCall, Result, TensorRef};
use rayon::prelude::*;

fn wr(t: &TensorRef, v: &[f64]) -> Result<()> {
    let f: Vec<f32> = v.iter().map(|x| *x as f32).collect();
    write_f32(t, &f)
}

fn scalar(t: &TensorRef) -> Result<f32> {
    Ok(read_f32(t)?[0])
}

pub fn run(call: &OpCall) -> Result<()> {
    let (i, o, a) = (&call.inputs, &call.outputs, &call.attrs);
    match call.op {
        Op::MatmulFp8 => {
            let (m, k, n) = (i[0].shape[0], i[0].shape[1], i[1].shape[0]);
            let x = read_f32(&i[0])?;
            let w = bytes(&i[1])?;
            let sc = read_f32(&i[2])?;
            let chunk = 64.max(n / (rayon::current_num_threads() * 4)).min(n.max(1));
            let cols: Vec<(usize, Vec<f64>)> = (0..n)
                .step_by(chunk)
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|n0| {
                    let nc = chunk.min(n - n0);
                    (n0, r::matmul_fp8(&x, m, k, &w[n0 * k..(n0 + nc) * k], nc, [sc[0], sc[1]]))
                })
                .collect();
            let mut out = vec![0f64; m * n];
            for (n0, blk) in cols {
                let nc = blk.len() / m.max(1);
                for mi in 0..m {
                    out[mi * n + n0..mi * n + n0 + nc].copy_from_slice(&blk[mi * nc..(mi + 1) * nc]);
                }
            }
            wr(&o[0], &out)
        }
        Op::MatmulNvfp4 => {
            let (m, k, n) = (i[0].shape[0], i[0].shape[1], i[1].shape[0]);
            let x = read_f32(&i[0])?;
            let (w, bs) = (bytes(&i[1])?, bytes(&i[2])?);
            let gs = scalar(&i[3])?;
            let chunk = 64.max(n / (rayon::current_num_threads() * 4)).min(n.max(1));
            let cols: Vec<(usize, Vec<f64>)> = (0..n)
                .step_by(chunk)
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|n0| {
                    let nc = chunk.min(n - n0);
                    let ws = &w[n0 * (k / 2)..(n0 + nc) * (k / 2)];
                    let bss = &bs[n0 * (k / 16)..(n0 + nc) * (k / 16)];
                    (n0, r::matmul_nvfp4(&x, m, k, ws, bss, gs, nc))
                })
                .collect();
            let mut out = vec![0f64; m * n];
            for (n0, blk) in cols {
                let nc = blk.len() / m.max(1);
                for mi in 0..m {
                    out[mi * n + n0..mi * n + n0 + nc].copy_from_slice(&blk[mi * nc..(mi + 1) * nc]);
                }
            }
            wr(&o[0], &out)
        }
        Op::MatmulMxfp4 => {
            let (m, k, n) = (i[0].shape[0], i[0].shape[1], i[1].shape[0]);
            let g = intarsia_core::quant::mxfp4::GROUP;
            let mut x = read_f32(&i[0])?;
            let mode = Mxfp4Act::from_flag(scalar(&i[3])?)
                .ok_or_else(|| Error::model(format!("matmul_mxfp4: act input {} is not 0, 1 or 2", scalar(&i[3]).unwrap_or(f32::NAN))))?;
            if mode == Mxfp4Act::Dynamic {
                // QDQ once for all rows (the per-row oracle would redo it per column block)
                x = intarsia_core::quant::mxfp4::qdq_rows(&x, m, k);
            }
            // W4A8: quantise once for all rows too; the e4m3 values go through the plain dot, the token scale after it
            let a8_scales = if mode == Mxfp4Act::Fp8 {
                let (q, s) = intarsia_core::quant::mxfp4::a8_quant_rows(&x, m, k);
                x = q.iter().map(|b| intarsia_core::dtype::fp8_e4m3_to_f32(*b)).collect();
                Some(s)
            } else {
                None
            };
            let (w, sc) = (bytes(&i[1])?, bytes(&i[2])?);
            let chunk = 64.max(n / (rayon::current_num_threads() * 4)).min(n.max(1));
            let cols: Vec<(usize, Vec<f64>)> = (0..n)
                .step_by(chunk)
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|n0| {
                    let nc = chunk.min(n - n0);
                    let ws = &w[n0 * (k / 2)..(n0 + nc) * (k / 2)];
                    let ss = &sc[n0 * (k / g)..(n0 + nc) * (k / g)];
                    (n0, r::matmul_mxfp4(&x, m, k, ws, ss, false, nc))
                })
                .collect();
            let mut out = vec![0f64; m * n];
            for (n0, blk) in cols {
                let nc = blk.len() / m.max(1);
                for mi in 0..m {
                    out[mi * n + n0..mi * n + n0 + nc].copy_from_slice(&blk[mi * nc..(mi + 1) * nc]);
                }
            }
            if let Some(s) = &a8_scales {
                for mi in 0..m {
                    for v in &mut out[mi * n..(mi + 1) * n] {
                        *v *= s[mi] as f64;
                    }
                }
            }
            wr(&o[0], &out)
        }
        Op::MoeTopk => {
            let (t, e, k) = (i[0].shape[0], i[0].shape[1], a.top_k);
            let (w, ids) = r::moe_topk(&read_f32(&i[0])?, t, e, k, a.norm_topk, i[0].dtype);
            write_f32(&o[0], &w)?;
            // SAFETY: exclusive output tensor.
            let ob = unsafe { crate::mem::bytes_mut(&o[1])? };
            for (c, v) in ob.chunks_exact_mut(4).zip(&ids) {
                c.copy_from_slice(&v.to_le_bytes());
            }
            Ok(())
        }
        Op::MoeExperts if intarsia_core::quant::exl3_moe::is_exl3(call) => crate::exl3_ref::run_moe(call),
        Op::MoeExperts => {
            let (t, h) = (i[0].shape[0], i[0].shape[1]);
            let k = i[1].shape[1];
            let (e, inter) = (i[3].shape[0], i[3].shape[1] / 2);
            let x = read_f32(&i[0])?;
            let ids: Vec<i32> =
                bytes(&i[1])?.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let wts = read_f32(&i[2])?;
            let gs = read_f32(&i[7])?;
            let ex = r::ExpertsRef {
                e,
                h,
                i: inter,
                gate_up_w: bytes(&i[3])?,
                gate_up_s: bytes(&i[4])?,
                down_w: bytes(&i[5])?,
                down_s: bytes(&i[6])?,
                gscales: &gs,
            };
            let dt = i[0].dtype;
            let od = o[0].dtype;
            if a.ink.situ && t > 1 {
                // a prefill chunk: each expert's rows decoded once for all its tokens (same sums, same bits as per token)
                let out = situ_grouped(&ex, &x, &ids, &wts, t, k, dt, od, a.ink.gate_cap as f64, a.ink.up_cap as f64);
                return wr(&o[0], &out);
            }
            let rows: Vec<Vec<f64>> = (0..t)
                .into_par_iter()
                .map(|ti| {
                    let (xr, ir, wr_) = (&x[ti * h..(ti + 1) * h], &ids[ti * k..(ti + 1) * k], &wts[ti * k..(ti + 1) * k]);
                    if a.ink.situ {
                        situ_token(&ex, xr, ir, wr_, dt, od, a.ink.gate_cap as f64, a.ink.up_cap as f64)
                    } else {
                        r::moe_experts_token_act(&ex, xr, ir, wr_, dt, od, a.gelu)
                    }
                })
                .collect();
            wr(&o[0], &rows.concat())
        }
        Op::MoeCombine => {
            let (t, h) = (i[0].shape[0], i[0].shape[1]);
            let out = r::moe_combine(&read_f32(&i[0])?, &read_f32(&i[1])?, &read_f32(&i[2])?, t, h, i[0].dtype);
            wr(&o[0], &out)
        }
        other => Err(Error::invalid(format!("quant_ref cannot run {other}"))),
    }
}

/// `r::moe_experts_token_situ` with the rows of each expert in parallel (decode has one token): the same f64 sums in the same
/// order per output value, so the bits are those of the serial reference.
#[allow(clippy::too_many_arguments)]
fn situ_token(ex: &r::ExpertsRef, x: &[f32], ids: &[i32], weights: &[f32], dt: intarsia_core::DType, od: intarsia_core::DType, gc: f64, uc: f64) -> Vec<f64> {
    use intarsia_core::quant::nvfp4::GROUP;
    let (h, i) = (ex.h, ex.i);
    let sig = |z: f64| 1.0 / (1.0 + (-z).exp());
    let ys: Vec<Vec<f64>> = ids
        .par_iter()
        .zip(weights.par_iter())
        .map(|(&id, &w)| {
            if id < 0 || id as usize >= ex.e {
                return Vec::new();
            }
            let e = id as usize;
            let gw = &ex.gate_up_w[e * 2 * i * (h / 2)..(e + 1) * 2 * i * (h / 2)];
            let gsb = &ex.gate_up_s[e * 2 * i * (h / GROUP)..(e + 1) * 2 * i * (h / GROUP)];
            let dw = &ex.down_w[e * h * (i / 2)..(e + 1) * h * (i / 2)];
            let dsb = &ex.down_s[e * h * (i / GROUP)..(e + 1) * h * (i / GROUP)];
            let g3 = &ex.gscales[e * 3..e * 3 + 3];
            let act: Vec<f64> = (0..i)
                .into_par_iter()
                .map_init(
                    || vec![0f64; h],
                    |row, rr| {
                        r::nvfp4_row(gw, gsb, g3[0], rr, h, row);
                        let g = r::round(dt, (0..h).map(|c| x[c] as f64 * row[c]).sum());
                        r::nvfp4_row(gw, gsb, g3[1], i + rr, h, row);
                        let u = r::round(dt, (0..h).map(|c| x[c] as f64 * row[c]).sum());
                        r::round(dt, gc * (g / gc).tanh() * sig(g) * uc * (u / uc).tanh() * w as f64)
                    },
                )
                .collect();
            (0..h)
                .into_par_iter()
                .map_init(
                    || vec![0f64; i],
                    |row, rr| {
                        r::nvfp4_row(dw, dsb, g3[2], rr, i, row);
                        r::round(dt, (0..i).map(|c| act[c] * row[c]).sum())
                    },
                )
                .collect()
        })
        .collect();
    let mut acc = vec![0f64; h];
    for y in ys.iter().filter(|y| !y.is_empty()) {
        acc.iter_mut().zip(y).for_each(|(a, v)| *a += v);
    }
    acc.iter().map(|v| r::round(od, *v)).collect()
}

/// [`situ_token`] for `t` tokens at once, grouped by expert: every (token, slot) value is the same f64 sum in the same order, and
/// the slots of a token are summed in slot order, so the result is bit for bit the per-token one.
#[allow(clippy::too_many_arguments)]
fn situ_grouped(ex: &r::ExpertsRef, x: &[f32], ids: &[i32], weights: &[f32], t: usize, k: usize, dt: intarsia_core::DType, od: intarsia_core::DType, gc: f64, uc: f64) -> Vec<f64> {
    use intarsia_core::quant::nvfp4::GROUP;
    let (h, i) = (ex.h, ex.i);
    let sig = |z: f64| 1.0 / (1.0 + (-z).exp());
    // (expert, [(token * k + slot)])
    let mut by: Vec<Vec<usize>> = vec![Vec::new(); ex.e];
    for (j, &id) in ids.iter().enumerate().take(t * k) {
        if id >= 0 && (id as usize) < ex.e {
            by[id as usize].push(j);
        }
    }
    let ys: Vec<(usize, Vec<f64>)> = by
        .par_iter()
        .enumerate()
        .filter(|(_, js)| !js.is_empty())
        .flat_map_iter(|(e, js)| {
            let gw = &ex.gate_up_w[e * 2 * i * (h / 2)..(e + 1) * 2 * i * (h / 2)];
            let gsb = &ex.gate_up_s[e * 2 * i * (h / GROUP)..(e + 1) * 2 * i * (h / GROUP)];
            let dw = &ex.down_w[e * h * (i / 2)..(e + 1) * h * (i / 2)];
            let dsb = &ex.down_s[e * h * (i / GROUP)..(e + 1) * h * (i / GROUP)];
            let g3 = &ex.gscales[e * 3..e * 3 + 3];
            let n = js.len();
            // act[r][p] for pair p
            let act: Vec<Vec<f64>> = (0..i)
                .into_par_iter()
                .map_init(
                    || (vec![0f64; h], vec![0f64; h]),
                    |(rg, ru), rr| {
                        r::nvfp4_row(gw, gsb, g3[0], rr, h, rg);
                        r::nvfp4_row(gw, gsb, g3[1], i + rr, h, ru);
                        js.iter()
                            .map(|&j| {
                                let xr = &x[(j / k) * h..(j / k + 1) * h];
                                let g = r::round(dt, (0..h).map(|c| xr[c] as f64 * rg[c]).sum());
                                let u = r::round(dt, (0..h).map(|c| xr[c] as f64 * ru[c]).sum());
                                r::round(dt, gc * (g / gc).tanh() * sig(g) * uc * (u / uc).tanh() * weights[j] as f64)
                            })
                            .collect()
                    },
                )
                .collect();
            // y[c][p]
            let y: Vec<Vec<f64>> = (0..h)
                .into_par_iter()
                .map_init(
                    || vec![0f64; i],
                    |row, c| {
                        r::nvfp4_row(dw, dsb, g3[2], c, i, row);
                        (0..n).map(|p| r::round(dt, (0..i).map(|q| act[q][p] * row[q]).sum())).collect()
                    },
                )
                .collect();
            js.iter().enumerate().map(move |(p, &j)| (j, (0..h).map(|c| y[c][p]).collect::<Vec<f64>>())).collect::<Vec<_>>()
        })
        .collect();
    let mut per: Vec<Option<Vec<f64>>> = vec![None; t * k];
    for (j, v) in ys {
        per[j] = Some(v);
    }
    let mut out = vec![0f64; t * h];
    for ti in 0..t {
        let acc = &mut out[ti * h..(ti + 1) * h];
        for slot in 0..k {
            if let Some(v) = &per[ti * k + slot] {
                acc.iter_mut().zip(v).for_each(|(a, b)| *a += b);
            }
        }
        acc.iter_mut().for_each(|a| *a = r::round(od, *a));
    }
    out
}
