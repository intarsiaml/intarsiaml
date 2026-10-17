//! Plain-Rust reference of the quantised arithmetic of `moe_experts_nvfp4_mma_bf16` (kernels/cuda/moe_nvfp4_mma_sm120a.cu),
//! shared by tests/moe_nvfp4.rs (GPU) and tests/moe_nvfp4_cpu.rs (no GPU needed).
#![allow(dead_code)]
use intarsia_core::dtype::f32_to_fp8_e4m3;
use intarsia_core::exec::HostCall;
use intarsia_core::quant::nvfp4::fp4_to_f32;
use intarsia_core::quant::reference::e4m3_table;
use intarsia_core::{DType, HostTensor};
use std::collections::HashMap;

// ---- the quantised arithmetic, in plain Rust -------------------------------------------------------------------------

pub fn bf16r(v: f32) -> f32 {
    let b = v.to_bits();
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

pub const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// e2m1 code of `f`: round to nearest, ties to the even code, saturating at 6 (cvt.rn.satfinite.e2m1x2.f32).
pub fn e2m1_rne(f: f32) -> u8 {
    let a = f.abs();
    let code = if a <= 0.25 {
        0
    } else if a < 0.75 {
        1
    } else if a <= 1.25 {
        2
    } else if a < 1.75 {
        3
    } else if a <= 2.5 {
        4
    } else if a < 3.5 {
        5
    } else if a <= 5.0 {
        6
    } else {
        7
    };
    code | if f.is_sign_negative() { 8 } else { 0 }
}

pub fn e2m1_value(code: u8) -> f64 {
    let v = E2M1[(code & 7) as usize] as f64;
    if code & 8 != 0 {
        -v
    } else {
        v
    }
}

/// The kernel's activation quantiser (`act_quant_kernel`) for one row: returns the dequantised values WITHOUT the
/// row scale (e2m1 * block scale, exact) and the row scale `ginv` (= amax / 2688) the epilogue multiplies back in.
pub fn quant_row(x: &[f32]) -> (Vec<f64>, f32) {
    let t = e4m3_table();
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let gx = if amax > 0.0 { 2688f32 / amax } else { 0.0 };
    let ginv = if amax > 0.0 { amax / 2688f32 } else { 0.0 };
    let mut out = Vec::with_capacity(x.len());
    for blk in x.chunks(16) {
        let bmax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let sb = f32_to_fp8_e4m3((bmax * gx) * (1f32 / 6f32));
        let sd = t[sb as usize];
        let inv = if sd > 0.0 { gx / sd } else { 0.0 };
        for &v in blk {
            out.push(e2m1_value(e2m1_rne(v * inv)) * sd as f64);
        }
    }
    (out, ginv)
}

/// Dequantised weight row WITHOUT the expert's global scale.
pub fn weight_row(w: &[u8], s: &[u8], row: usize, k: usize) -> Vec<f32> {
    let t = e4m3_table();
    (0..k)
        .map(|c| {
            let byte = w[row * (k / 2) + c / 2];
            let code = if c % 2 == 0 { byte & 15 } else { byte >> 4 };
            fp4_to_f32(code) * t[s[row * (k / 16) + c / 16] as usize]
        })
        .collect()
}

pub struct Quantised<'a> {
    hc: &'a HostCall,
    h: usize,
    i: usize,
    cache: HashMap<(usize, usize), Vec<f32>>, // (expert, row in 0..2I for gate_up | 2I + row for down) -> dequantised row
}

impl<'a> Quantised<'a> {
    pub fn new(hc: &'a HostCall) -> Self {
        let h = hc.inputs[0].shape[1];
        let i = hc.inputs[3].shape[1] / 2;
        Quantised { hc, h, i, cache: HashMap::new() }
    }
    fn gu_row(&mut self, e: usize, r: usize) -> &Vec<f32> {
        let (h, i) = (self.h, self.i);
        let hc = self.hc;
        self.cache.entry((e, r)).or_insert_with(|| {
            weight_row(&hc.inputs[3].data[e * 2 * i * (h / 2)..], &hc.inputs[4].data[e * 2 * i * (h / 16)..], r, h)
        })
    }
    fn dn_row(&mut self, e: usize, r: usize) -> &Vec<f32> {
        let (h, i) = (self.h, self.i);
        let hc = self.hc;
        self.cache.entry((e, 2 * i + r)).or_insert_with(|| {
            weight_row(&hc.inputs[5].data[e * h * (i / 2)..], &hc.inputs[6].data[e * h * (i / 16)..], r, i)
        })
    }
    /// The kernel's result for the whole call, computed with the quantised arithmetic.
    pub fn run(&mut self) -> Vec<f32> {
        let hc = self.hc;
        let (t, h, i) = (hc.inputs[0].shape[0], self.h, self.i);
        let k = hc.inputs[1].shape[1];
        let x = hc.inputs[0].to_f32();
        let ids = hc.inputs[1].to_i32().unwrap();
        let wts = hc.inputs[2].to_f32();
        let gs = hc.inputs[7].to_f32();
        let e_n = hc.inputs[3].shape[0] as i32;
        let mut out = vec![0f32; t * h];
        for ti in 0..t {
            let (xq, xginv) = quant_row(&x[ti * h..(ti + 1) * h]);
            let mut acc = vec![0f32; h];
            for j in 0..k {
                let e = ids[ti * k + j];
                if e < 0 || e >= e_n {
                    continue;
                }
                let e = e as usize;
                let (gg, gu, gd) = (gs[e * 3], gs[e * 3 + 1], gs[e * 3 + 2]);
                let mut act = vec![0f32; i];
                for r in 0..i {
                    let g: f64 = self.gu_row(e, r).iter().zip(&xq).map(|(w, x)| *w as f64 * x).sum();
                    let u: f64 = self.gu_row(e, i + r).iter().zip(&xq).map(|(w, x)| *w as f64 * x).sum();
                    let g = bf16r(g as f32 * (xginv * gg));
                    let u = bf16r(u as f32 * (xginv * gu));
                    let a = bf16r(g / (1.0 + (-g).exp()));
                    act[r] = bf16r(a * u);
                }
                let (hq, hginv) = quant_row(&act);
                for n in 0..h {
                    let y: f64 = self.dn_row(e, n).iter().zip(&hq).map(|(w, x)| *w as f64 * x).sum();
                    let y = bf16r(y as f32 * (hginv * gd));
                    acc[n] += bf16r(y * wts[ti * k + j]);
                }
            }
            for n in 0..h {
                out[ti * h + n] = bf16r(acc[n]);
            }
        }
        out
    }
}

pub fn with_rows_scaled(hc: &HostCall, scales: &[(usize, f32)]) -> HostCall {
    let mut hc = hc.clone();
    let (t, h) = (hc.inputs[0].shape[0], hc.inputs[0].shape[1]);
    let mut x = hc.inputs[0].to_f32();
    for &(row, s) in scales {
        assert!(row < t);
        for v in &mut x[row * h..(row + 1) * h] {
            *v *= s;
        }
    }
    hc.inputs[0] = HostTensor::from_f32_as(DType::Bf16, &[t, h], &x);
    hc
}

