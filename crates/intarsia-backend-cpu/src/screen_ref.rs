//! The screened greedy head on the cpu (`intarsia_core::screen`, lane ink-fp8). `fast`: `LogitsGather` uses the fast
//! backend's `Logits` dot (`fast::dot`, bitwise its `Logits`); otherwise the reference's f64 sum rounded to f32 (bitwise the
//! reference `Logits`). HeadScreen and ScreenArgmax are the core's canonical f32 functions in both.

use crate::mem::{read_f32, write_f32};
use intarsia_core::screen::{argmax_row, candidate_path, screen_row};
use intarsia_core::{Error, Op, OpCall, Result, TensorRef};
use rayon::prelude::*;

fn read_i32(t: &TensorRef) -> Result<Vec<i32>> {
    let b = crate::mem::bytes(t)?;
    Ok(b[..t.numel() * 4].chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn write_i32(t: &TensorRef, v: &[i32]) -> Result<()> {
    // SAFETY: the op's exclusive output.
    let b = unsafe { crate::mem::bytes_mut(t)? };
    for (j, x) in v.iter().enumerate() {
        b[j * 4..j * 4 + 4].copy_from_slice(&x.to_le_bytes());
    }
    Ok(())
}

pub fn run(call: &OpCall, fast: bool) -> Result<()> {
    let (i, o) = (&call.inputs, &call.outputs);
    match call.op {
        Op::HeadScreen => {
            let (t, v, c) = (i[0].shape[0], i[0].shape[1], o[0].shape[1]);
            let d = i[1].shape[1];
            let (a, h, beta) = (read_f32(&i[0])?, read_f32(&i[1])?, read_f32(&i[2])?);
            let mut cand = vec![-1i32; t * c];
            let cnt: Vec<i32> = cand
                .par_chunks_mut(c)
                .enumerate()
                .map(|(r, cr)| screen_row(&a[r * v..(r + 1) * v], &h[r * d..(r + 1) * d], &beta, cr))
                .collect();
            // every slot is defined: the first min(cnt, C) candidates, -1 elsewhere (screen_row)
            write_i32(&o[0], &cand)?;
            write_i32(&o[1], &cnt)
        }
        Op::LogitsGather => {
            let (_t, d, v, c) = (i[0].shape[0], i[0].shape[1], i[1].shape[0], i[2].shape[1]);
            let (h, w, cand, cnt) = (read_f32(&i[0])?, read_f32(&i[1])?, read_i32(&i[2])?, read_i32(&i[3])?);
            let mut y = read_f32(&o[0])?;
            let logit = |r: usize, n: usize| -> f32 {
                let (hr, wn) = (&h[r * d..(r + 1) * d], &w[n * d..(n + 1) * d]);
                if fast {
                    crate::fast::dot(hr, wn)
                } else {
                    hr.iter().zip(wn).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() as f32
                }
            };
            y.par_chunks_mut(v).enumerate().for_each(|(r, yr)| {
                if candidate_path(cnt[r], c) {
                    for j in 0..cnt[r] as usize {
                        yr[j] = logit(r, cand[r * c + j] as usize);
                    }
                } else {
                    for (n, yn) in yr.iter_mut().enumerate() {
                        *yn = logit(r, n);
                    }
                }
            });
            write_f32(&o[0], &y)
        }
        Op::ScreenArgmax => {
            let (t, c, v) = (i[0].shape[0], i[0].shape[1], i[2].shape[1]);
            let (cand, cnt, y) = (read_i32(&i[0])?, read_i32(&i[1])?, read_f32(&i[2])?);
            let tok: Vec<i32> = (0..t).map(|r| argmax_row(&cand[r * c..(r + 1) * c], cnt[r], &y[r * v..(r + 1) * v])).collect();
            write_i32(&o[0], &tok)
        }
        op => Err(Error::invalid(format!("{op}: not a screen op"))),
    }
}
