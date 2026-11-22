//! Exact screened greedy head (lane ink-fp8, opt-in `INTARSIA_INK_HEAD_SCREEN=1`; ops 88-90, 91 stays reserved).
//!
//! Greedy decode only needs the argmax of the bf16 head, `x_n = Logits(h, W_n)` (f32, the device `Logits` kernel). An fp8 copy
//! `q` of the head (one scale `s`) gives screen values `a_n = MatmulFp8(h, q_n)`; a rigorous per-row bound picks every row that
//! can be the argmax; only those rows are rescored with the bf16 weights, by a kernel whose per-row arithmetic is the `Logits`
//! kernel's, so the token is bitwise today's argmax (ties included: the lowest index of the maximum).
//!
//! | Op | Inputs | Outputs |
//! |---|---|---|
//! | `HeadScreen` (88) | a `[T,V]` f32, h `[T,D]` (act), beta `[V]` f32 | cand `[T,C]` i32, cnt `[T]` i32 |
//! | `LogitsGather` (89) | h `[T,D]` bf16, W `[V,D]` bf16, cand `[T,C]` i32, cnt `[T]` i32 | y `[T,V]` f32 |
//! | `ScreenArgmax` (90) | cand `[T,C]` i32, cnt `[T]` i32, y `[T,V]` f32 | tok `[T]` i32 |
//!
//! **HeadScreen.** Per row t, in f32 with these exact operations (the cpu reference [`screen_row`] and every kernel agree
//! bitwise): `ss = sum_k fma(h_k, h_k, .)` over 32 lane partials (lane l takes k = l, l + 32, ... in order) then the xor tree
//! (`p_l += p_{l^16}`, `^8`, `^4`, `^2`, `^1`, lane 0); `hn = sqrt(ss) * (1 + 2^-11)`; for every row n
//! `m = fma(hn, beta_n, |a_n|)`, `lo = fma(-C, m, fma(-hn, beta_n, a_n))`, `hi = fma(C, m, fma(hn, beta_n, a_n))` with
//! `C = 2^-20`; `L = max_n lo`; the candidates are `{n : hi_n >= L}` written in ascending n to `cand[t, ..]` (at most C_cap of
//! them; C_cap is cand's width), `cnt[t]` their number (which may exceed C_cap). A non-finite `hn`, `a_n` or `L` sets
//! `cnt[t] = i32::MAX`.
//!
//! **LogitsGather.** `cnt[t]` in `1..=C_cap`: `y[t, j] = Logits(h_t, W[cand[t, j]])` for `j < cnt[t]`; otherwise (the
//! fallback) `y[t, n] = Logits(h_t, W_n)` for every n, i.e. the full head row. Each value is bitwise the backend's `Logits`
//! for that (row, weight row); the rest of `y[t]` is left as it was.
//!
//! **ScreenArgmax.** `cnt[t]` in `1..=C_cap`: `tok[t] = cand[t, j*]`, j* the lowest j of the maximum of `y[t, 0..cnt)`;
//! otherwise `tok[t] = Argmax(y[t, 0..V))`. Comparisons as `Argmax`: strict `>`, `+0 == -0`, NaN never wins.
//!
//! **Why the token is exact.** Let `x_n` be the device logit and `a_n` the screen value. Products of a bf16 activation and a
//! bf16 weight, or an e4m3 weight (decoded exactly), are exact in f32, so both are f32 sums of exact products: whatever the
//! summation order (fma included), `|x_n - h.W_n| <= g ||h|| ||W_n||` and `|a_n - s h.q_n| <= g ||h|| s ||q_n||` with
//! `g = gamma_{D+2} = (D+2) u / (1 - (D+2) u)`, `u = 2^-24` (the `+2` covers the scale multiply). With
//! `e_n = ||W_n - s q_n||_2`, Cauchy-Schwarz gives `|x_n - a_n| <= ||h|| (e_n + g (||W_n|| + s ||q_n||)) <= hn * beta_n`, where
//! [`row_bounds`] computes `beta_n` in f64, scales it by `1 + 2^-20` and rounds it up to f32, and `hn >= ||h||` (the f32 sum
//! of squares is within `gamma_D` of the exact one, i.e. 1.4e-4 relative at D = 2304, far inside the `2^-11` margin; the
//! module tests check `hn >= ||h||` in f64). `C = 2^-20` (16 u) absorbs every rounding of the `lo` / `hi` formulas, so
//! `lo_n <= x_n <= hi_n` for every row. The row n0 with `lo_{n0} = L` has `x_{n0} >= L`, so the maximum `x* >= L`, and every
//! row with `x_n = x*` has `hi_n >= x_n = x* >= L`: every maximiser is a candidate. The rescoring reproduces each `x_n`
//! bitwise and ScreenArgmax takes the lowest index of their maximum, which is `Argmax` over the full row.
//!
//! Memory: the fp8 copy of the head (V x D bytes, 0.46 GB for Inkling-2-Mini) plus beta (4 V bytes) on top of the bf16 head.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{Op, OpCall, ShapeInfo};

/// The three screen ops, discriminants 88..=90.
pub const OPS: [Op; 3] = [Op::HeadScreen, Op::LogitsGather, Op::ScreenArgmax];

/// `cnt` of a row whose screen saw a non-finite value (the fallback).
pub const NONFINITE: i32 = i32::MAX;

/// The widening factor of the `lo` / `hi` formulas (`2^-20`).
pub const WIDEN: f32 = 1.0 / (1u32 << 20) as f32;

/// The `hn` margin (`1 + 2^-11`).
pub const HN_MARGIN: f32 = 1.0 + 1.0 / 2048.0;

/// `INTARSIA_INK_HEAD_SCREEN`: `1` = the screened greedy head (needs `INTARSIA_LMHEAD_QUANT` unset: the head stays bf16, the
/// fp8 copy is the screen's); unset / anything else = off.
pub const ENV: &str = "INTARSIA_INK_HEAD_SCREEN";

/// Candidate cap per row (`INTARSIA_INK_HEAD_SCREEN_CAP`, default 64): rows with more candidates rescore the full head.
pub const CAP_ENV: &str = "INTARSIA_INK_HEAD_SCREEN_CAP";
pub const CAP_DEFAULT: usize = 64;

fn bad(op: Op, msg: String) -> Result<()> {
    Err(Error::invalid(format!("{op}: {msg}")))
}

fn want(op: Op, t: &crate::backend::TensorRef, dt: Option<DType>, shape: &[usize], what: &str) -> Result<()> {
    let float = matches!(t.dtype, DType::Bf16 | DType::F16 | DType::F32);
    if t.shape != shape || dt.is_some_and(|d| d != t.dtype) || (dt.is_none() && !float) {
        return bad(op, format!("{what} must be {} {shape:?}, got {} {:?}", dt.map_or("float".to_string(), |d| d.to_string()), t.dtype, t.shape));
    }
    Ok(())
}

/// Shapes and dtypes of a screen call (arity is checked by `OpCall::check`).
pub fn check(call: &OpCall) -> Result<()> {
    let (op, i, o) = (call.op, &call.inputs, &call.outputs);
    match op {
        Op::HeadScreen => {
            if i[0].shape.len() != 2 || o[0].shape.len() != 2 {
                return bad(op, "a and cand must be rank 2".into());
            }
            let (t, v, c) = (i[0].shape[0], i[0].shape[1], o[0].shape[1]);
            want(op, &i[0], Some(DType::F32), &[t, v], "a")?;
            if i[1].shape.len() != 2 || i[1].shape[0] != t {
                return bad(op, format!("h must be [{t}, D], got {:?}", i[1].shape));
            }
            want(op, &i[1], None, &i[1].shape.clone(), "h")?;
            want(op, &i[2], Some(DType::F32), &[v], "beta")?;
            want(op, &o[0], Some(DType::I32), &[t, c], "cand")?;
            want(op, &o[1], Some(DType::I32), &[t], "cnt")?;
            if c == 0 {
                return bad(op, "cand needs at least one column".into());
            }
        }
        Op::LogitsGather => {
            if i[0].shape.len() != 2 || i[1].shape.len() != 2 || i[2].shape.len() != 2 {
                return bad(op, "h, W and cand must be rank 2".into());
            }
            let (t, d, v, c) = (i[0].shape[0], i[0].shape[1], i[1].shape[0], i[2].shape[1]);
            want(op, &i[0], Some(DType::Bf16), &[t, d], "h")?;
            want(op, &i[1], Some(DType::Bf16), &[v, d], "W")?;
            want(op, &i[2], Some(DType::I32), &[t, c], "cand")?;
            want(op, &i[3], Some(DType::I32), &[t], "cnt")?;
            want(op, &o[0], Some(DType::F32), &[t, v], "y")?;
        }
        Op::ScreenArgmax => {
            if i[0].shape.len() != 2 || i[2].shape.len() != 2 {
                return bad(op, "cand and y must be rank 2".into());
            }
            let (t, c, v) = (i[0].shape[0], i[0].shape[1], i[2].shape[1]);
            want(op, &i[0], Some(DType::I32), &[t, c], "cand")?;
            want(op, &i[1], Some(DType::I32), &[t], "cnt")?;
            want(op, &i[2], Some(DType::F32), &[t, v], "y")?;
            want(op, &o[0], Some(DType::I32), &[t], "tok")?;
        }
        _ => unreachable!("not a screen op"),
    }
    Ok(())
}

/// [`ShapeInfo`] of a screen call: `m` = T, `n` = V, `k` = D (HeadScreen, LogitsGather) or the cap (ScreenArgmax).
pub fn shape(call: &OpCall, s: &mut ShapeInfo) {
    let i = &call.inputs;
    match call.op {
        Op::HeadScreen => {
            s.m = i[0].shape[0];
            s.n = i[0].shape[1];
            s.k = i[1].shape[1];
            s.act = Some(i[1].dtype);
        }
        Op::LogitsGather => {
            s.m = i[0].shape[0];
            s.n = i[1].shape[0];
            s.k = i[0].shape[1];
            s.act = Some(i[0].dtype);
            s.weight = Some(i[1].dtype);
        }
        Op::ScreenArgmax => {
            s.m = i[0].shape[0];
            s.n = i[2].shape[1];
            s.k = i[0].shape[1];
            s.act = Some(DType::F32);
        }
        _ => {}
    }
}

/// `hn` of one row as the contract defines it (32 lane partials of fma squares, xor tree, sqrt, margin).
pub fn row_norm(h: &[f32]) -> f32 {
    let mut p = [0f32; 32];
    for (k, &x) in h.iter().enumerate() {
        p[k % 32] = x.mul_add(x, p[k % 32]);
    }
    for off in [16usize, 8, 4, 2, 1] {
        let q = p;
        for l in 0..32 {
            p[l] = q[l] + q[l ^ off];
        }
    }
    p[0].sqrt() * HN_MARGIN
}

/// `(lo, hi)` of one screen value.
#[inline]
pub fn lo_hi(a: f32, hn: f32, beta: f32) -> (f32, f32) {
    let m = hn.mul_add(beta, a.abs());
    let lo = (-WIDEN).mul_add(m, (-hn).mul_add(beta, a));
    let hi = WIDEN.mul_add(m, hn.mul_add(beta, a));
    (lo, hi)
}

/// HeadScreen of one row: writes the first `cand.len()` candidates in ascending order, returns `cnt`.
pub fn screen_row(a: &[f32], h: &[f32], beta: &[f32], cand: &mut [i32]) -> i32 {
    let hn = row_norm(h);
    if !hn.is_finite() {
        return NONFINITE;
    }
    let mut l = f32::NEG_INFINITY;
    for (n, &an) in a.iter().enumerate() {
        if !an.is_finite() {
            return NONFINITE;
        }
        l = l.max(lo_hi(an, hn, beta[n]).0);
    }
    if !l.is_finite() {
        return NONFINITE;
    }
    let mut c = 0usize;
    for (n, &an) in a.iter().enumerate() {
        if lo_hi(an, hn, beta[n]).1 >= l {
            if c < cand.len() {
                cand[c] = n as i32;
            }
            c += 1;
        }
    }
    c.min(i32::MAX as usize - 1) as i32
}

/// Whether a row's `cnt` takes the candidate path (else the full-row fallback).
#[inline]
pub fn candidate_path(cnt: i32, cap: usize) -> bool {
    cnt >= 1 && (cnt as usize) <= cap
}

/// ScreenArgmax of one row (`y` the full `[V]` row).
pub fn argmax_row(cand: &[i32], cnt: i32, y: &[f32]) -> i32 {
    if candidate_path(cnt, cand.len()) {
        let (mut bv, mut bi) = (f32::NEG_INFINITY, cand[0]);
        for j in 0..cnt as usize {
            if y[j] > bv {
                bv = y[j];
                bi = cand[j];
            }
        }
        bi
    } else {
        crate::sampler::argmax(y) as i32
    }
}

/// Per-row bounds `beta_n` of an fp8 copy of a head (see the module doc): `w` the head rows in `dt` (bf16 / f16 / f32
/// bytes, `[n, k]`), `q` the e4m3 codes `[n, k]`, `s` the copy's scale. f64, scaled by `1 + 2^-20`, rounded up to f32.
pub fn row_bounds(dt: DType, w: &[u8], q: &[u8], s: f32, n: usize, k: usize) -> Result<Vec<f32>> {
    let es = dt.size_bytes();
    if w.len() != n * k * es || q.len() != n * k {
        return Err(Error::invalid(format!("row_bounds: {} weight / {} code bytes for [{n}, {k}]", w.len(), q.len())));
    }
    let t = crate::quant::reference::e4m3_table();
    let uk = (k + 2) as f64 * f64::powi(2.0, -24);
    let g = uk / (1.0 - uk);
    let s = s as f64;
    let mut row = vec![0f32; k];
    let mut out = Vec::with_capacity(n);
    for r in 0..n {
        crate::dtype::decode_to_f32(dt, &w[r * k * es..(r + 1) * k * es], &mut row);
        let (mut e2, mut w2, mut q2) = (0f64, 0f64, 0f64);
        for c in 0..k {
            let wv = row[c] as f64;
            let qv = s * t[q[r * k + c] as usize] as f64;
            e2 += (wv - qv) * (wv - qv);
            w2 += wv * wv;
            q2 += qv * qv;
        }
        let b = (e2.sqrt() + g * (w2.sqrt() + q2.sqrt())) * (1.0 + f64::powi(2.0, -20));
        let mut f = b as f32;
        if (f as f64) < b {
            f = f32::from_bits(f.to_bits() + 1);
        }
        out.push(f);
    }
    Ok(out)
}

/// Admission sweep cases (`admission::sweep`): T 1 / 4 / 16, caps 1 / 16 / 64, small widths and the real Inkling-2-Mini head
/// (D 2304, V 200058) outside `--quick`; candidate, full-row (cnt 0, above the cap, non-finite) and tied rows.
pub fn sweep(op: Op, cfg: &crate::admission::SweepConfig) -> Vec<crate::admission::Case> {
    use crate::exec::{HostCall, OutSpec};
    use crate::tensor::HostTensor;
    let mut rng = crate::rng::SplitMix64::new(cfg.seed ^ (op as u64).wrapping_mul(0x9E37));
    let mut cases = Vec::new();
    let mut shapes = vec![(64usize, 1000usize)];
    if !cfg.quick {
        shapes.push((2304, 200058));
    }
    let bf = |v: &[f32], shape: &[usize]| HostTensor::from_f32_as(DType::Bf16, shape, v);
    for &(d, v) in &shapes {
        for &t in &[1usize, 4, 16] {
            for &c in &[1usize, 16, 64] {
                let label = format!("T={t} D={d} V={v} C={c}");
                match op {
                    Op::HeadScreen => {
                        let h = rng.normal_vec(t * d, 1.0);
                        // screen values: a spread plus a few near the top of each row, so the counts vary around the caps
                        let mut a = rng.normal_vec(t * v, 3.0);
                        for r in 0..t {
                            let top = 12.0 + r as f32;
                            for j in 0..(r * 5 + 1).min(v) {
                                let n = (j * 7919 + r * 104729) % v;
                                a[r * v + n] = top - 0.01 * j as f32 * (r % 3) as f32;
                            }
                        }
                        if t == 4 {
                            a[2 * v + v / 2] = f32::NAN; // row 2: the non-finite fallback
                        }
                        let beta: Vec<f32> = (0..v).map(|_| 0.002 + 0.003 * rng.next_f32()).collect();
                        let inputs = vec![HostTensor::from_f32(&[t, v], &a), bf(&h, &[t, d]), HostTensor::from_f32(&[v], &beta)];
                        let outputs = vec![OutSpec::new(DType::I32, &[t, c]), OutSpec::new(DType::I32, &[t])];
                        cases.push(crate::admission::Case { label, call: HostCall { op, inputs, outputs, attrs: Default::default() } });
                    }
                    Op::LogitsGather | Op::ScreenArgmax => {
                        // per row: a candidate count (1, c, then 0 / above the cap / non-finite as the full-row cases)
                        let cnts: Vec<i32> = (0..t)
                            .map(|r| match r % 5 {
                                0 => 1,
                                1 => c as i32,
                                2 => (c as i32 / 2).max(1),
                                3 if t > 4 => [0, c as i32 + 7, NONFINITE][(r / 5) % 3],
                                _ => (c as i32 - 1).max(1),
                            })
                            .collect();
                        let mut cand = vec![0i32; t * c];
                        for r in 0..t {
                            let mut n = (rng.next_u64() % 97) as usize;
                            for j in 0..c {
                                cand[r * c + j] = n.min(v - 1) as i32;
                                n += 1 + (rng.next_u64() % (v as u64 / c as u64 / 2).max(1)) as usize;
                            }
                        }
                        let cin = HostTensor::from_i32(&[t, c], &cand);
                        let nin = HostTensor::from_i32(&[t], &cnts);
                        if op == Op::LogitsGather {
                            let h = rng.normal_vec(t * d, 1.0);
                            let w = rng.normal_vec(v * d, (d as f32).powf(-0.5));
                            let inputs = vec![bf(&h, &[t, d]), bf(&w, &[v, d]), cin, nin];
                            let outputs = vec![OutSpec::new(DType::F32, &[t, v])];
                            cases.push(crate::admission::Case { label, call: HostCall { op, inputs, outputs, attrs: Default::default() } });
                        } else {
                            // y with exact ties at the maximum (the lowest index must win) and +-0
                            let mut y = rng.normal_vec(t * v, 1.0);
                            for r in 0..t {
                                let k = cnts[r].clamp(1, c as i32) as usize;
                                for j in (0..k).step_by(3) {
                                    y[r * v + j] = 9.5;
                                }
                                for n in (1..v).step_by(4999) {
                                    y[r * v + n] = 9.5; // ties in the full-row case too
                                }
                                y[r * v] = if r % 2 == 0 { 0.0 } else { -0.0 };
                            }
                            let inputs = vec![cin, nin, HostTensor::from_f32(&[t, v], &y)];
                            let outputs = vec![OutSpec::new(DType::I32, &[t])];
                            cases.push(crate::admission::Case { label, call: HostCall { op, inputs, outputs, attrs: Default::default() } });
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    cases
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
    }

    fn bf_bits(v: f32) -> u16 {
        let mut o = [0u8; 2];
        crate::dtype::encode_from_f32(DType::Bf16, &[v], &mut o);
        u16::from_le_bytes(o)
    }

    fn bf(v: f32) -> f32 {
        f32::from_bits((bf_bits(v) as u32) << 16)
    }

    #[test]
    fn hn_bounds_the_norm() {
        let mut rng = Rng(7);
        for k in [1usize, 31, 64, 2304, 5000] {
            for scale in [1e-20f32, 1e-3, 1.0, 1e3, 1e15] {
                let h: Vec<f32> = (0..k).map(|_| bf(rng.unit() * scale)).collect();
                let exact = h.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
                let hn = row_norm(&h) as f64;
                assert!(hn >= exact, "k {k} scale {scale}: hn {hn} < ||h|| {exact}");
            }
        }
    }

    /// The property behind the op: the screen's candidates always contain every maximiser of the exact f32 logits, and
    /// argmax_row over them is `Argmax` of the full row, for random, tied and adversarial rows (lane ink-fp8, brief 4.3).
    #[test]
    fn screen_finds_the_argmax() {
        let mut rng = Rng(0x5c4ee7);
        let (k, n) = (256usize, 512usize);
        for trial in 0..400 {
            let kind = trial % 8;
            // weights: bf16 values; a quarter of the trials put duplicate rows (exact ties) or huge / tiny magnitudes
            let mag = match kind {
                5 => 1e18,
                6 => 1e-18,
                _ => 0.05,
            };
            let mut w: Vec<f32> = (0..n * k).map(|_| bf(rng.unit() * mag)).collect();
            if kind == 1 || kind == 2 {
                for r in (0..n).step_by(7) {
                    let src = (r * 31 + 3) % n;
                    let row: Vec<f32> = w[src * k..(src + 1) * k].to_vec();
                    w[r * k..(r + 1) * k].copy_from_slice(&row);
                }
            }
            if kind == 3 {
                // rows that differ from row 0 in one low bit (near ties)
                for r in 1..n {
                    if r % 3 == 0 {
                        let mut row: Vec<f32> = w[..k].to_vec();
                        let c = r % k;
                        row[c] = f32::from_bits(row[c].to_bits() ^ 0x1_0000);
                        w[r * k..(r + 1) * k].copy_from_slice(&row);
                    }
                }
            }
            let mut h: Vec<f32> = (0..k).map(|_| bf(rng.unit())).collect();
            if kind == 4 {
                for x in h.iter_mut().step_by(2) {
                    *x = if rng.next() & 1 == 0 { 0.0 } else { -0.0 };
                }
            }
            if kind == 7 {
                h.iter_mut().for_each(|x| *x = 0.0); // every logit +-0: all tie, the answer is row 0
            }
            // the fp8 copy (per-tensor amax / 448, as quant::head)
            let amax = w.iter().fold(0f32, |m, x| m.max(x.abs()));
            let s = if amax > 0.0 { amax / crate::quant::fp8::MAX } else { 1.0 };
            let q: Vec<u8> = w.iter().map(|x| crate::dtype::f32_to_fp8_e4m3(x / s)).collect();
            let wb: Vec<u8> = w.iter().flat_map(|x| bf_bits(*x).to_le_bytes()).collect();
            let beta = row_bounds(DType::Bf16, &wb, &q, s, n, k).unwrap();
            let t = crate::quant::reference::e4m3_table();
            // device-like values: the exact logits in one summation order, the screen in another
            let x: Vec<f32> = (0..n).map(|r| h.iter().zip(&w[r * k..(r + 1) * k]).fold(0f32, |acc, (a, b)| a.mul_add(*b, acc))).collect();
            let a: Vec<f32> = (0..n)
                .map(|r| {
                    let mut p = [0f32; 4];
                    for c in 0..k {
                        p[c % 4] += h[c] * t[q[r * k + c] as usize];
                    }
                    ((p[0] + p[1]) + (p[2] + p[3])) * s
                })
                .collect();
            let want = crate::sampler::argmax(&x) as i32;
            for cap in [1usize, 4, 64, n] {
                let mut cand = vec![-1i32; cap];
                let cnt = screen_row(&a, &h, &beta, &mut cand);
                // y as LogitsGather leaves it
                let mut y = vec![f32::NAN; n];
                if candidate_path(cnt, cap) {
                    for j in 0..cnt as usize {
                        y[j] = x[cand[j] as usize];
                    }
                    let xmax = x.iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v));
                    for (r, &xr) in x.iter().enumerate() {
                        if xr == xmax {
                            assert!(cand[..cnt as usize].contains(&(r as i32)), "trial {trial} kind {kind}: maximiser {r} not a candidate");
                        }
                    }
                    assert!(cand[..cnt as usize].windows(2).all(|p| p[0] < p[1]), "ascending");
                } else {
                    y.copy_from_slice(&x);
                }
                assert_eq!(argmax_row(&cand, cnt, &y), want, "trial {trial} kind {kind} cap {cap} cnt {cnt}");
            }
        }
    }

    #[test]
    fn nonfinite_falls_back() {
        let h = vec![1.0f32; 8];
        let beta = vec![0.1f32; 4];
        let mut cand = vec![0i32; 4];
        assert_eq!(screen_row(&[0.0, f32::NAN, 1.0, 2.0], &h, &beta, &mut cand), NONFINITE);
        assert_eq!(screen_row(&[0.0, f32::INFINITY, 1.0, 2.0], &h, &beta, &mut cand), NONFINITE);
        let hinf = vec![f32::MAX; 8];
        assert_eq!(screen_row(&[0.0, 0.5, 1.0, 2.0], &hinf, &beta, &mut cand), NONFINITE);
        assert!(!candidate_path(NONFINITE, 4) && !candidate_path(0, 4) && !candidate_path(5, 4) && candidate_path(4, 4));
    }
}
