//! Gated DeltaNet (linear attention) and gated-attention helper ops: validation, shape facts and
//! the f64 reference math (the oracle `intarsia-backend-cpu` wraps and every GPU kernel is graded
//! against). Semantics are those of HF `transformers` `Qwen3_5MoeGatedDeltaNet`
//! (`models/qwen3_5_moe/modeling_qwen3_5_moe.py`), which is the spec.
//!
//! # Ops (all dense row-major; `T` tokens)
//!
//! * `GateSplit`: `x [T,N,2D]` -> `q [T,N,D]`, `gate [T,N,D]`. HF fuses q and the attention output
//!   gate in one projection, interleaved per head: head `h` owns `2D` consecutive values, the first
//!   `D` are the query and the next `D` the gate.
//! * `SigmoidMul`: `a`, `g` of equal shape -> `a * sigmoid(g)` (the attention output gate).
//! * `GdnConv`: depthwise causal conv over the token axis, per channel `c` with taps `w [C,K]`
//!   (tap `K-1` multiplies the current token, tap `0` the token `K-1` steps back), no bias, then
//!   SiLU. `state [C,K-1]` f32 holds the last `K-1` inputs of the channel (index 0 = oldest) and is
//!   updated in place; a zeroed state is HF's left zero padding. Sequences are processed in any
//!   chunking: carrying the state across calls gives the same result as one call.
//! * `GdnDeltaRule`: the gated delta rule recurrence. Input `qkv [T,C]` is the conv output with rows
//!   `[q (Hk*dk) | k (Hk*dk) | v (Hv*dv)]`; value head `h` reads key head `h / (Hv/Hk)` (HF
//!   `repeat_interleave`). Per token and value head:
//!   ```text
//!   q = l2norm(q) / sqrt(dk)        l2norm(x) = x * rsqrt(sum(x^2) + eps)
//!   k = l2norm(k)
//!   beta = sigmoid(b)               g = -exp(A_log) * softplus(a + dt_bias)    (f32)
//!   S = S * exp(g)                  S: [dk, dv]
//!   delta = (v - S^T k) * beta
//!   S = S + k (x) delta
//!   o = S^T q
//!   ```
//!   The state is stored **transposed**, `[Hv, dv, dk]` (dk contiguous), f32, updated in place.
//!   `o` is `[T, Hv, dv]` in the activation dtype. softplus uses PyTorch's threshold 20.

use crate::error::{Error, Result};
use crate::ops::{Op, OpCall, ShapeInfo};
use crate::dtype::DType;

/// Dimensions of a `GdnDeltaRule` call, derived from its tensor shapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeltaDims {
    pub t: usize,
    /// Key heads.
    pub hk: usize,
    /// Value heads.
    pub hv: usize,
    pub dk: usize,
    pub dv: usize,
}

impl DeltaDims {
    /// Row width of the fused `[q|k|v]` input.
    pub fn conv_dim(&self) -> usize {
        2 * self.hk * self.dk + self.hv * self.dv
    }

    pub fn from_call(call: &OpCall) -> DeltaDims {
        let (qkv, o, st) = (&call.inputs[0], &call.outputs[0], &call.outputs[1]);
        let (t, hv, dv) = (o.shape[0], o.shape[1], o.shape[2]);
        let dk = st.shape[2];
        let c = qkv.shape[1];
        let hk = if dk == 0 { 0 } else { (c.saturating_sub(hv * dv)) / (2 * dk) };
        DeltaDims { t, hk, hv, dk, dv }
    }
}

/// Validation of the four ops (called from `OpCall::check`).
pub fn check(call: &OpCall) -> Result<()> {
    let (i, o) = (&call.inputs, &call.outputs);
    let bad = |m: String| Err(Error::invalid(format!("{}: {m}", call.op)));
    let rank = |t: &crate::backend::TensorRef, r: usize, what: &str| -> Result<()> {
        if t.shape.len() != r {
            return Err(Error::invalid(format!("{}: {what} must be rank {r}, got {:?}", call.op, t.shape)));
        }
        Ok(())
    };
    match call.op {
        Op::GateSplit => {
            rank(&i[0], 3, "x")?;
            let (t, n, d2) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
            if d2 % 2 != 0 {
                return bad(format!("last dim {d2} must be even (query + gate)"));
            }
            let want = [t, n, d2 / 2];
            if o[0].shape != want || o[1].shape != want {
                return bad(format!("outputs {:?} {:?}, expected {want:?}", o[0].shape, o[1].shape));
            }
        }
        Op::SigmoidMul => {
            if i[0].shape != i[1].shape || o[0].shape != i[0].shape {
                return bad("shape mismatch".into());
            }
        }
        Op::GdnConv => {
            rank(&i[0], 2, "x")?;
            rank(&i[1], 2, "taps")?;
            rank(&o[0], 2, "y")?;
            rank(&o[1], 2, "state")?;
            let (t, c) = (i[0].shape[0], i[0].shape[1]);
            let k = i[1].shape[1];
            if k < 2 || i[1].shape[0] != c || o[0].shape != [t, c] || o[1].shape != [c, k - 1] {
                return bad(format!(
                    "x {:?} taps {:?} y {:?} state {:?}: need taps [C,K>=2], y [T,C], state [C,K-1]",
                    i[0].shape, i[1].shape, o[0].shape, o[1].shape
                ));
            }
            if o[1].dtype != DType::F32 {
                return bad("conv state must be f32".into());
            }
        }
        Op::GdnDeltaRule => {
            rank(&i[0], 2, "qkv")?;
            rank(&o[0], 3, "o")?;
            rank(&o[1], 3, "state")?;
            let (t, c) = (i[0].shape[0], i[0].shape[1]);
            let (hv, dv, dk) = (o[0].shape[1], o[0].shape[2], o[1].shape[2]);
            if o[0].shape[0] != t || o[1].shape[0] != hv || o[1].shape[1] != dv {
                return bad(format!("o {:?} / state {:?} inconsistent", o[0].shape, o[1].shape));
            }
            if dk == 0 || dv == 0 || hv == 0 || c <= hv * dv || (c - hv * dv) % (2 * dk) != 0 {
                return bad(format!("qkv width {c} does not split into 2*Hk*{dk} + {hv}*{dv}"));
            }
            let hk = (c - hv * dv) / (2 * dk);
            if hv % hk != 0 {
                return bad(format!("{hv} value heads are not a multiple of {hk} key heads"));
            }
            for (idx, what) in [(1usize, "a"), (2, "b")] {
                if i[idx].shape != [t, hv] {
                    return bad(format!("{what} {:?} must be [T={t},Hv={hv}]", i[idx].shape));
                }
            }
            for (idx, what) in [(3usize, "A_log"), (4, "dt_bias")] {
                if i[idx].shape != [hv] {
                    return bad(format!("{what} {:?} must be [Hv={hv}]", i[idx].shape));
                }
            }
            if o[1].dtype != DType::F32 {
                return bad("recurrent state must be f32".into());
            }
        }
        _ => unreachable!("gdn::check called for {}", call.op),
    }
    Ok(())
}

/// Shape facts for kernel predicates (`ShapeInfo::from_call`).
pub fn shape(call: &OpCall, s: &mut ShapeInfo) {
    let (i, o) = (&call.inputs, &call.outputs);
    match call.op {
        Op::GateSplit => {
            s.m = i[0].shape[0];
            s.n_heads = i[0].shape[1];
            s.head_dim = i[0].shape[2] / 2;
            s.k = s.head_dim;
        }
        Op::SigmoidMul => {
            s.m = i[0].shape[0];
            s.k = i[0].numel() / i[0].shape[0].max(1);
        }
        Op::GdnConv => {
            s.m = i[0].shape[0];
            s.n = i[0].shape[1];
            s.k = i[1].shape[1];
            s.weight = Some(i[1].dtype);
        }
        Op::GdnDeltaRule => {
            let d = DeltaDims::from_call(call);
            s.m = d.t;
            s.n_heads = d.hv;
            s.n_kv_heads = d.hk;
            s.head_dim = d.dk;
            s.k = d.dk;
            s.n = d.dv;
            s.weight = Some(i[3].dtype);
        }
        _ => unreachable!(),
    }
    let _ = o;
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// PyTorch `softplus` (beta 1, threshold 20).
fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// Reference `GateSplit`.
pub fn gate_split(x: &[f64], t: usize, n: usize, d: usize) -> (Vec<f64>, Vec<f64>) {
    let mut q = Vec::with_capacity(t * n * d);
    let mut g = Vec::with_capacity(t * n * d);
    for row in x.chunks(2 * d).take(t * n) {
        q.extend_from_slice(&row[..d]);
        g.extend_from_slice(&row[d..]);
    }
    (q, g)
}

/// Reference `SigmoidMul`.
pub fn sigmoid_mul(a: &[f64], g: &[f64]) -> Vec<f64> {
    a.iter().zip(g).map(|(a, g)| a * sigmoid(*g)).collect()
}

/// Reference `GdnConv`: `x [T,C]`, `w [C,K]`, `state [C,K-1]` (updated) -> `y [T,C]`.
pub fn conv_silu(x: &[f64], w: &[f64], state: &mut [f64], t: usize, c: usize, k: usize) -> Vec<f64> {
    let keep = k - 1;
    let mut y = vec![0f64; t * c];
    let mut ext = vec![0f64; keep + t];
    for ch in 0..c {
        ext[..keep].copy_from_slice(&state[ch * keep..(ch + 1) * keep]);
        for ti in 0..t {
            ext[keep + ti] = x[ti * c + ch];
        }
        for ti in 0..t {
            let mut acc = 0.0;
            for j in 0..k {
                acc += w[ch * k + j] * ext[ti + j];
            }
            y[ti * c + ch] = silu(acc);
        }
        state[ch * keep..(ch + 1) * keep].copy_from_slice(&ext[t..t + keep]);
    }
    y
}

/// Reference `GdnDeltaRule`. `qkv [T,C]`, `a`,`b [T,Hv]`, `a_log`,`dt_bias [Hv]`, `state [Hv,dv,dk]`
/// (updated) -> `o [T,Hv,dv]`.
#[allow(clippy::too_many_arguments)]
pub fn delta_rule(
    d: &DeltaDims,
    eps: f64,
    qkv: &[f64],
    a: &[f64],
    b: &[f64],
    a_log: &[f64],
    dt_bias: &[f64],
    state: &mut [f64],
) -> Vec<f64> {
    let DeltaDims { t, hk, hv, dk, dv } = *d;
    let c = d.conv_dim();
    let rep = hv / hk;
    let scale = 1.0 / (dk as f64).sqrt();
    let mut o = vec![0f64; t * hv * dv];
    let mut qn = vec![0f64; dk];
    let mut kn = vec![0f64; dk];
    for ti in 0..t {
        let row = &qkv[ti * c..(ti + 1) * c];
        for h in 0..hv {
            let kh = h / rep;
            let q = &row[kh * dk..(kh + 1) * dk];
            let k = &row[hk * dk + kh * dk..hk * dk + (kh + 1) * dk];
            let v = &row[2 * hk * dk + h * dv..2 * hk * dk + (h + 1) * dv];
            let qr = 1.0 / (q.iter().map(|x| x * x).sum::<f64>() + eps).sqrt();
            let kr = 1.0 / (k.iter().map(|x| x * x).sum::<f64>() + eps).sqrt();
            for i in 0..dk {
                qn[i] = q[i] * qr * scale;
                kn[i] = k[i] * kr;
            }
            let beta = sigmoid(b[ti * hv + h]);
            let g = -a_log[h].exp() * softplus(a[ti * hv + h] + dt_bias[h]);
            let decay = g.exp();
            for j in 0..dv {
                let s = &mut state[(h * dv + j) * dk..(h * dv + j + 1) * dk];
                let mut kv = 0.0;
                for i in 0..dk {
                    s[i] *= decay;
                    kv += s[i] * kn[i];
                }
                let delta = (v[j] - kv) * beta;
                let mut out = 0.0;
                for i in 0..dk {
                    s[i] += kn[i] * delta;
                    out += s[i] * qn[i];
                }
                o[(ti * hv + h) * dv + j] = out;
            }
        }
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    fn rnd(rng: &mut SplitMix64, n: usize, s: f32) -> Vec<f64> {
        rng.normal_vec(n, s).into_iter().map(|x| x as f64).collect()
    }

    #[test]
    fn conv_state_carries_across_chunks() {
        let (c, k, t) = (5usize, 4usize, 9usize);
        let mut rng = SplitMix64::new(3);
        let x = rnd(&mut rng, t * c, 1.0);
        let w = rnd(&mut rng, c * k, 0.5);
        let mut s1 = vec![0f64; c * (k - 1)];
        let whole = conv_silu(&x, &w, &mut s1, t, c, k);
        let mut s2 = vec![0f64; c * (k - 1)];
        let mut parts = Vec::new();
        for (lo, hi) in [(0usize, 2usize), (2, 3), (3, 9)] {
            parts.extend(conv_silu(&x[lo * c..hi * c], &w, &mut s2, hi - lo, c, k));
        }
        assert_eq!(whole, parts);
        assert_eq!(s1, s2);
        // newest input is the last state entry
        assert_eq!(s1[2], x[(t - 1) * c]);
    }

    #[test]
    fn delta_rule_state_carries_across_chunks() {
        let d = DeltaDims { t: 7, hk: 2, hv: 4, dk: 8, dv: 6 };
        let mut rng = SplitMix64::new(5);
        let qkv = rnd(&mut rng, d.t * d.conv_dim(), 1.0);
        let a = rnd(&mut rng, d.t * d.hv, 1.0);
        let b = rnd(&mut rng, d.t * d.hv, 1.0);
        let a_log = rnd(&mut rng, d.hv, 0.5);
        let dtb = rnd(&mut rng, d.hv, 0.5);
        let mut s1 = vec![0f64; d.hv * d.dv * d.dk];
        let whole = delta_rule(&d, 1e-6, &qkv, &a, &b, &a_log, &dtb, &mut s1);
        let mut s2 = vec![0f64; s1.len()];
        let mut parts = Vec::new();
        let c = d.conv_dim();
        let mut lo = 0;
        for n in [3usize, 1, 3] {
            let dd = DeltaDims { t: n, ..d };
            parts.extend(delta_rule(
                &dd,
                1e-6,
                &qkv[lo * c..(lo + n) * c],
                &a[lo * d.hv..(lo + n) * d.hv],
                &b[lo * d.hv..(lo + n) * d.hv],
                &a_log,
                &dtb,
                &mut s2,
            ));
            lo += n;
        }
        assert_eq!(whole, parts);
        assert_eq!(s1, s2);
        assert!(whole.iter().all(|x| x.is_finite()) && whole.iter().any(|x| x.abs() > 1e-6));
    }

    #[test]
    fn delta_rule_first_token_is_beta_scaled_v() {
        // zero state: kv_mem = 0, delta = beta*v, S = k (x) delta, o = S^T q = (k . q) * beta * v
        let d = DeltaDims { t: 1, hk: 1, hv: 1, dk: 4, dv: 3 };
        let qkv: Vec<f64> = vec![1., 2., 3., 4., /*k*/ 4., 3., 2., 1., /*v*/ 0.5, -1.0, 2.0];
        let mut s = vec![0f64; 12];
        let o = delta_rule(&d, 0.0, &qkv, &[0.3], &[0.7], &[0.1], &[0.2], &mut s);
        let nq = (30f64).sqrt();
        let qn: Vec<f64> = [1., 2., 3., 4.].iter().map(|x| x / nq / 2.0).collect();
        let kn: Vec<f64> = [4., 3., 2., 1.].iter().map(|x| x / nq).collect();
        let kq: f64 = qn.iter().zip(&kn).map(|(a, b)| a * b).sum();
        let beta = sigmoid(0.7);
        for (j, v) in [0.5, -1.0, 2.0].iter().enumerate() {
            assert!((o[j] - kq * beta * v).abs() < 1e-12, "{j}: {} vs {}", o[j], kq * beta * v);
        }
    }

    #[test]
    fn split_and_gate() {
        let x: Vec<f64> = (0..12).map(|v| v as f64).collect(); // T=1, N=2, D=3
        let (q, g) = gate_split(&x, 1, 2, 3);
        assert_eq!(q, vec![0., 1., 2., 6., 7., 8.]);
        assert_eq!(g, vec![3., 4., 5., 9., 10., 11.]);
        assert_eq!(sigmoid_mul(&[2.0], &[0.0]), vec![1.0]);
    }
}
