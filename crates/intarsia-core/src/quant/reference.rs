//! The oracle math of the quantised-weight and MoE ops: slices in, f64 out, no backend. The cpu backend's
//! `ref_*` kernels wrap these (parallelising over independent rows / tokens); the Mac unit tests check them
//! against numbers computed by independent Python (HF transformers + a numpy dequantiser) from real checkpoint slices.
//!
//! Rounding mirrors the HF bf16 module chain where it is cheap (`round`): linear outputs, `act(gate)`, `act(gate)*up`,
//! the expert output, `expert_out * routing_weight`, `sigmoid(gate_logit)` and `routed + gated shared` are rounded to the
//! activation dtype. For `f32` activations `round` is the identity, so the f32 path equals the HF float32 module.

use crate::dtype::{decode_to_f32, encode_from_f32, f32_to_fp8_e4m3, fp8_e4m3_to_f32, DType};
use crate::quant::nvfp4::{fp4_to_f32, GROUP};

/// Round `v` to what a tensor of `dtype` stores (identity for f32).
#[inline]
pub fn round(dtype: DType, v: f64) -> f64 {
    if dtype == DType::F32 {
        return v as f32 as f64;
    }
    let f = [v as f32];
    let mut b = [0u8; 2];
    encode_from_f32(dtype, &f, &mut b);
    let mut o = [0f32];
    decode_to_f32(dtype, &b, &mut o);
    o[0] as f64
}

/// 256-entry decode table of e4m3 bytes.
pub fn e4m3_table() -> &'static [f32; 256] {
    use std::sync::OnceLock;
    static T: OnceLock<[f32; 256]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0f32; 256];
        for (i, v) in t.iter_mut().enumerate() {
            *v = fp8_e4m3_to_f32(i as u8);
        }
        t
    })
}

/// One FP8 `MatmulFp8` row-block: `x` `[m,k]`, `w` the weight rows `[n,k]` (e4m3 bytes), `scales` = (weight_scale,
/// input_scale). Returns `[m,n]` f64 (not yet rounded to the output dtype).
pub fn matmul_fp8(x: &[f32], m: usize, k: usize, w: &[u8], n: usize, scales: [f32; 2]) -> Vec<f64> {
    let t = e4m3_table();
    let (ws, is) = (scales[0], scales[1]);
    let act = is > 0.0;
    let xq: Vec<f64> = x
        .iter()
        .map(|v| if act { t[f32_to_fp8_e4m3(*v / is) as usize] as f64 } else { *v as f64 })
        .collect();
    let scale = ws as f64 * if act { is as f64 } else { 1.0 };
    let mut out = vec![0f64; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut s = 0f64;
            for ki in 0..k {
                s += xq[mi * k + ki] * t[w[ni * k + ki] as usize] as f64;
            }
            out[mi * n + ni] = s * scale;
        }
    }
    out
}

/// Dequantised NVFP4 row `r` of a `[n,k]` matrix as f64 (`fp4 * block_scale * global`).
pub fn nvfp4_row(w: &[u8], bs: &[u8], gs: f32, r: usize, k: usize, out: &mut [f64]) {
    let t = e4m3_table();
    for j in 0..k {
        let byte = w[r * (k / 2) + j / 2];
        let code = if j % 2 == 0 { byte & 0xF } else { byte >> 4 };
        out[j] = fp4_to_f32(code) as f64 * t[bs[r * (k / GROUP) + j / GROUP] as usize] as f64 * gs as f64;
    }
}

/// `MatmulNvfp4` for `n` rows: `x` `[m,k]` -> `[m,n]` f64.
pub fn matmul_nvfp4(x: &[f32], m: usize, k: usize, w: &[u8], bs: &[u8], gs: f32, n: usize) -> Vec<f64> {
    let mut out = vec![0f64; m * n];
    let mut row = vec![0f64; k];
    for ni in 0..n {
        nvfp4_row(w, bs, gs, ni, k, &mut row);
        for mi in 0..m {
            let mut s = 0f64;
            for ki in 0..k {
                s += x[mi * k + ki] as f64 * row[ki];
            }
            out[mi * n + ni] = s;
        }
    }
    out
}

/// `MoeTopk`: per token the `k` largest logits (ties: lowest index first), softmax weights.
/// Selecting on the raw logits is exactly top-k of the softmax probabilities (monotone), and with `norm` the
/// renormalised weights equal the softmax over the selected logits. Weights are rounded to `dtype` (HF casts the
/// routing weights back to the logits dtype). Returns (weights `[t,k]`, ids `[t,k]`).
pub fn moe_topk(logits: &[f32], t: usize, e: usize, k: usize, norm: bool, dtype: DType) -> (Vec<f32>, Vec<i32>) {
    let mut w = vec![0f32; t * k];
    let mut ids = vec![0i32; t * k];
    for ti in 0..t {
        let row = &logits[ti * e..(ti + 1) * e];
        let mut order: Vec<usize> = (0..e).collect();
        // stable sort by value descending keeps the lowest index first among equals
        order.sort_by(|a, b| row[*b].partial_cmp(&row[*a]).unwrap_or(std::cmp::Ordering::Equal));
        let sel = &order[..k];
        let mx = row[sel[0]] as f64;
        let denom: f64 = if norm {
            sel.iter().map(|&j| (row[j] as f64 - mx).exp()).sum()
        } else {
            row.iter().map(|&v| (v as f64 - mx).exp()).sum()
        };
        for (j, &idx) in sel.iter().enumerate() {
            w[ti * k + j] = round(dtype, (row[idx] as f64 - mx).exp() / denom) as f32;
            ids[ti * k + j] = idx as i32;
        }
    }
    (w, ids)
}

/// Stacked NVFP4 expert weights (`MoeExperts` inputs 3..8).
pub struct ExpertsRef<'a> {
    pub e: usize,
    pub h: usize,
    pub i: usize,
    pub gate_up_w: &'a [u8],
    pub gate_up_s: &'a [u8],
    pub down_w: &'a [u8],
    pub down_s: &'a [u8],
    /// `[e, 3]` = (gate, up, down) `weight_scale_2`.
    pub gscales: &'a [f32],
}

/// One (token, expert) pair: `down(silu(gate(x)) * up(x))` with HF's bf16 rounding points (`dt`).
pub fn expert_forward(ex: &ExpertsRef, expert: usize, x: &[f32], dt: DType) -> Vec<f64> {
    let (h, i) = (ex.h, ex.i);
    let gw = &ex.gate_up_w[expert * 2 * i * (h / 2)..(expert + 1) * 2 * i * (h / 2)];
    let gsb = &ex.gate_up_s[expert * 2 * i * (h / GROUP)..(expert + 1) * 2 * i * (h / GROUP)];
    let dw = &ex.down_w[expert * h * (i / 2)..(expert + 1) * h * (i / 2)];
    let dsb = &ex.down_s[expert * h * (i / GROUP)..(expert + 1) * h * (i / GROUP)];
    let g3 = &ex.gscales[expert * 3..expert * 3 + 3];
    let mut row = vec![0f64; h.max(i)];
    let mut act = vec![0f64; i];
    for r in 0..i {
        nvfp4_row(gw, gsb, g3[0], r, h, &mut row);
        let g = round(dt, (0..h).map(|c| x[c] as f64 * row[c]).sum());
        nvfp4_row(gw, gsb, g3[1], i + r, h, &mut row);
        let u = round(dt, (0..h).map(|c| x[c] as f64 * row[c]).sum());
        let a = round(dt, g / (1.0 + (-g).exp()));
        act[r] = round(dt, a * u);
    }
    let mut y = vec![0f64; h];
    for (r, yr) in y.iter_mut().enumerate() {
        nvfp4_row(dw, dsb, g3[2], r, i, &mut row);
        *yr = round(dt, (0..i).map(|c| act[c] * row[c]).sum());
    }
    y
}

/// `MoeExperts` for one token: `sum_k round(y_k * w_k)` (ids outside `0..e` are skipped), rounded to `out_dt`.
pub fn moe_experts_token(
    ex: &ExpertsRef,
    x: &[f32],
    ids: &[i32],
    weights: &[f32],
    dt: DType,
    out_dt: DType,
) -> Vec<f64> {
    let mut acc = vec![0f64; ex.h];
    for (&id, &w) in ids.iter().zip(weights) {
        if id < 0 || id as usize >= ex.e {
            continue;
        }
        let y = expert_forward(ex, id as usize, x, dt);
        for (a, v) in acc.iter_mut().zip(&y) {
            *a += round(dt, v * w as f64);
        }
    }
    acc.iter().map(|v| round(out_dt, *v)).collect()
}

/// `MoeCombine`: `round(routed + round(round(sigmoid(gate)) * shared))`.
pub fn moe_combine(routed: &[f32], shared: &[f32], gate: &[f32], t: usize, h: usize, dt: DType) -> Vec<f64> {
    let mut out = vec![0f64; t * h];
    for ti in 0..t {
        let g = gate[ti] as f64;
        let sg = round(dt, 1.0 / (1.0 + (-g).exp()));
        for c in 0..h {
            let s = round(dt, sg * shared[ti * h + c] as f64);
            out[ti * h + c] = round(dt, routed[ti * h + c] as f64 + s);
        }
    }
    out
}
