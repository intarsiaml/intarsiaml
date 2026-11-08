//! The DSpine code bridge's row-wise ops (`Op::BridgeNorm`, `BridgeCode`, `BridgeGate`, `BridgeWrite`; definitions in
//! `intarsia_core::ops`). One implementation serves both CPU kernel sets (the oracle and the engine's): reductions accumulate in
//! f64, every other step is f32 with the reference's rounding points (`bf()` = round to nearest even bf16, applied where the op's
//! definition stores a bf16 value; a tensor of another dtype is not pre-rounded).

use crate::mem::{read_f32, write_f32};
use intarsia_core::{DType, Op, OpCall, Result};

const FLT_MIN: f32 = 1.175_494_35e-38;

/// Round to the nearest-even bf16 (as an f32).
#[inline]
fn bf(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let b = x.to_bits();
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

/// Round to what a tensor of `dtype` stores, where the definition rounds (bf16); otherwise leave the f32 as it is.
#[inline]
fn rnd(dtype: DType, x: f32) -> f32 {
    if dtype == DType::Bf16 {
        bf(x)
    } else {
        x
    }
}

#[inline]
fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

fn sum_sq(row: &[f32]) -> f32 {
    row.iter().map(|&v| v as f64 * v as f64).sum::<f64>() as f32
}

pub fn run(call: &OpCall) -> Result<()> {
    let (i, o, a) = (&call.inputs, &call.outputs, &call.attrs);
    let dt = o[0].dtype;
    match call.op {
        Op::BridgeNorm => {
            let (t, d) = (i[0].shape[0], i[0].shape[1]);
            let x = read_f32(&i[0])?;
            let (mut y, mut rms) = (vec![0f32; t * d], vec![0f32; t]);
            for r in 0..t {
                let row = &x[r * d..(r + 1) * d];
                let ms = (sum_sq(row) as f64 / d as f64) as f32;
                let rr = ms.max(FLT_MIN).sqrt();
                rms[r] = rr;
                for c in 0..d {
                    y[r * d + c] = rnd(dt, row[c] / (rr + a.eps));
                }
            }
            write_f32(&o[0], &y)?;
            write_f32(&o[1], &rms)
        }
        Op::BridgeCode => {
            let (t, d) = (i[0].shape[0], i[0].shape[1]);
            let x = read_f32(&i[0])?;
            let mut y = vec![0f32; t * d];
            for r in 0..t {
                let row = &x[r * d..(r + 1) * d];
                let n = sum_sq(row).sqrt();
                for c in 0..d {
                    y[r * d + c] = rnd(dt, (a.scale * row[c]) / (n + a.eps));
                }
            }
            write_f32(&o[0], &y)
        }
        Op::BridgeGate => {
            let (t, r3) = (i[0].shape[0], i[0].shape[1]);
            let r = r3 / 3;
            let (p, bias, s0) = (read_f32(&i[0])?, read_f32(&i[1])?, read_f32(&i[2])?);
            let mut msg = vec![0f32; t * r];
            for row in 1..t {
                for c in 0..r {
                    let q = p[row * r3 + c];
                    let (s, v) = if row == 1 { (s0[c], s0[r + c]) } else { (p[(row - 1) * r3 + r + c], p[(row - 1) * r3 + 2 * r + c]) };
                    msg[row * r + c] = rnd(dt, sigmoid((q + s) + bias[c]) * v);
                }
            }
            write_f32(&o[0], &msg)
        }
        Op::BridgeWrite => {
            let (t, h) = (i[0].shape[0], i[0].shape[1]);
            let (raw, rms, delta, w) = (read_f32(&i[0])?, read_f32(&i[1])?, read_f32(&i[2])?, read_f32(&i[3])?);
            let (mut out, mut normed) = (vec![0f32; t * h], vec![0f32; t * h]);
            for r in 0..t {
                let na = if r == 0 { 0.0f32 } else { 1.0 };
                for c in 0..h {
                    out[r * h + c] = rnd(dt, raw[r * h + c] + (rms[r] * delta[r * h + c]) * na);
                }
                let row = &out[r * h..(r + 1) * h];
                let ms = (sum_sq(row) as f64 / h as f64) as f32;
                let inv = 1.0 / (ms + a.eps).sqrt();
                for c in 0..h {
                    let y = rnd(dt, row[c] * inv);
                    normed[r * h + c] = rnd(dt, y * w[c]);
                }
            }
            write_f32(&o[0], &out)?;
            write_f32(&o[1], &normed)
        }
        other => Err(intarsia_core::Error::invalid(format!("bridge_ref: {other} is not a bridge op"))),
    }
}
