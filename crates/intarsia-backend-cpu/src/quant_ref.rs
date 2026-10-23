//! cpu implementations of the quantised-weight and MoE ops. The math lives in
//! `intarsia_core::quant::reference` (f64, checked against HF/numpy numbers in the unit tests); this file reads
//! the tensors, parallelises over independent rows/tokens and rounds to the output dtype.
//! Both kernel sets (`ref_*` and `cpu_f32_*`) run this code.

use crate::mem::{bytes, read_f32, write_f32};
use intarsia_core::quant::reference as r;
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
            let act_qdq = scalar(&i[3])? != 0.0;
            if act_qdq {
                // QDQ once for all rows (the per-row oracle would redo it per column block)
                x = intarsia_core::quant::mxfp4::qdq_rows(&x, m, k);
            }
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
            let rows: Vec<Vec<f64>> = (0..t)
                .into_par_iter()
                .map(|ti| {
                    r::moe_experts_token_act(&ex, &x[ti * h..(ti + 1) * h], &ids[ti * k..(ti + 1) * k], &wts[ti * k..(ti + 1) * k], dt, od, a.gelu)
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
