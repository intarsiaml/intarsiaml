//! Kernel admission harness (ARCHITECTURE 3.2 / 6.1).
//!
//! For every registered implementation of a candidate backend, run it against
//! the reference backend's reference implementation over a shape sweep and
//! report `max_abs`, `rel_l2` and pass/fail. `intarsia kernels test` is a thin
//! wrapper around [`run_admission`].
//!
//! Pass criterion per case: `rel_l2 <= tol.rel_l2` and
//! `max_abs <= tol.max_abs * max(1, max|reference|)` (max_abs is relative to
//! the output scale so one tolerance works for bf16 outputs of any magnitude).
//! Integer outputs (argmax, sample) must match exactly.

use crate::backend::{Backend, DeviceInfo};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::exec::{run_host, HostCall, OutSpec};
use crate::ops::{Op, OpAttrs, ShapeInfo};
use crate::registry::{Admission, Registry};
use crate::rng::SplitMix64;
use crate::tensor::HostTensor;

/// Real model dimensions the sweep should include (from a recipe + config).
#[derive(Clone, Debug)]
pub struct SweepConfig {
    pub hidden: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub rope_theta: f32,
    /// Fewer/smaller cases.
    pub quick: bool,
    pub seed: u64,
}

impl Default for SweepConfig {
    /// Qwen3-0.6B-like shapes (the reranker-tuned Mac test model).
    fn default() -> Self {
        SweepConfig {
            hidden: 1024,
            n_heads: 16,
            n_kv_heads: 8,
            head_dim: 128,
            intermediate: 3072,
            vocab: 151_669,
            rope_theta: 1_000_000.0,
            quick: false,
            seed: 1234,
        }
    }
}

pub struct Case {
    pub label: String,
    pub call: HostCall,
}

fn act_dtypes() -> [DType; 2] {
    [DType::Bf16, DType::F32]
}

fn rand_t(rng: &mut SplitMix64, dt: DType, shape: &[usize], scale: f32) -> HostTensor {
    let n: usize = shape.iter().product();
    HostTensor::from_f32_as(dt, shape, &rng.normal_vec(n, scale))
}

fn norm_w(rng: &mut SplitMix64, dt: DType, n: usize) -> HostTensor {
    let v: Vec<f32> = (0..n).map(|_| 1.0 + 0.25 * rng.normal()).collect();
    HostTensor::from_f32_as(dt, &[n], &v)
}

fn attrs() -> OpAttrs {
    OpAttrs::default()
}

/// Generate the sweep cases for one op.
pub fn sweep(op: Op, cfg: &SweepConfig) -> Vec<Case> {
    if let Some(cases) = crate::quant::cases::sweep(op, cfg) {
        return cases; // quantised-weight and MoE ops build their own (format-aware) data
    }
    let mut rng = SplitMix64::new(cfg.seed ^ (op as u64).wrapping_mul(0x9E37));
    let mut cases = Vec::new();
    let quick = cfg.quick;
    let rows: &[usize] = if quick { &[1, 5] } else { &[1, 5, 32] };
    let hs = [cfg.hidden.min(256).max(8) & !1, cfg.hidden];
    let hs: Vec<usize> = if quick { vec![hs[0]] } else { hs.to_vec() };
    let mut push = |label: String, call: HostCall| cases.push(Case { label, call });
    match op {
        Op::Embedding => {
            let v = cfg.vocab.min(2048);
            for &t in rows {
                for &h in &hs {
                    for dt in act_dtypes() {
                        let ids: Vec<i32> = (0..t).map(|_| (rng.next_u64() % v as u64) as i32).collect();
                        let table = rand_t(&mut rng, dt, &[v, h], 1.0);
                        push(
                            format!("T={t} V={v} H={h} {dt}"),
                            HostCall {
                                op,
                                inputs: vec![HostTensor::from_i32(&[t], &ids), table],
                                outputs: vec![OutSpec::new(dt, &[t, h])],
                                attrs: attrs(),
                            },
                        );
                    }
                }
            }
        }
        Op::RmsNorm | Op::RmsNormAdd | Op::Add | Op::SiluGlu => {
            let mut hs2 = hs.clone();
            // `intermediate` is 0 for MoE-only checkpoints (no dense FFN, e.g. Qwen3.6-35B-A3B): no dense width to sweep.
            if op == Op::SiluGlu && !quick && cfg.intermediate > 0 {
                hs2.push(cfg.intermediate);
            }
            for &r in rows {
                for &h in &hs2 {
                    for (dt, woff) in act_dtypes().into_iter().flat_map(|d| [(d, 0.0f32), (d, 1.0)]) {
                        if woff != 0.0 && !matches!(op, Op::RmsNorm | Op::RmsNormAdd) {
                            continue;
                        }
                        let x = rand_t(&mut rng, dt, &[r, h], 1.0);
                        let (inputs, outputs) = match op {
                            Op::RmsNorm => (vec![x, norm_w(&mut rng, dt, h)], vec![OutSpec::new(dt, &[r, h])]),
                            Op::RmsNormAdd => (
                                vec![x, rand_t(&mut rng, dt, &[r, h], 2.0), norm_w(&mut rng, dt, h)],
                                vec![OutSpec::new(dt, &[r, h]), OutSpec::new(dt, &[r, h])],
                            ),
                            _ => (vec![x, rand_t(&mut rng, dt, &[r, h], 1.0)], vec![OutSpec::new(dt, &[r, h])]),
                        };
                        push(
                            format!("R={r} H={h} {dt}{}", if woff != 0.0 { " w_offset=1" } else { "" }),
                            HostCall { op, inputs, outputs, attrs: OpAttrs { eps: 1e-6, w_offset: woff, ..attrs() } },
                        );
                    }
                }
            }
        }
        Op::QkNorm | Op::Rope => {
            let dims: Vec<(usize, usize)> = if quick {
                vec![(cfg.n_heads.min(4), cfg.head_dim.min(64))]
            } else {
                vec![(4, 64), (cfg.n_heads, cfg.head_dim), (cfg.n_kv_heads, cfg.head_dim)]
            };
            let pos0s: &[usize] = if op == Op::Rope { &[0, 37] } else { &[0] };
            for &t in rows {
                for &(n, d) in &dims {
                    for &p in pos0s {
                        // rope: full head and partial rotary (a quarter of the head, Qwen3.5); qk_norm: plain and zero-centred
                        let variants: [(usize, f32); 2] = if op == Op::Rope { [(0, 0.0), ((d / 4) & !1, 0.0)] } else { [(0, 0.0), (0, 1.0)] };
                        for (dt, (rdim, woff)) in act_dtypes().into_iter().flat_map(|d| variants.map(|v| (d, v))) {
                            let x = rand_t(&mut rng, dt, &[t, n, d], 1.0);
                            let inputs = if op == Op::QkNorm { vec![x, norm_w(&mut rng, dt, d)] } else { vec![x] };
                            push(
                                format!("T={t} N={n} D={d} pos0={p} {dt} rot={rdim} off={woff}"),
                                HostCall {
                                    op,
                                    inputs,
                                    outputs: vec![OutSpec::new(dt, &[t, n, d])],
                                    attrs: OpAttrs {
                                        pos0: p,
                                        theta: cfg.rope_theta,
                                        head_dim: d,
                                        rotary_dim: rdim,
                                        w_offset: woff,
                                        ..attrs()
                                    },
                                },
                            );
                        }
                    }
                }
            }
        }
        Op::Matmul | Op::Logits => {
            let mut nk: Vec<(usize, usize)> = vec![(64, 64), (33, 70)];
            if !quick {
                let q = cfg.n_heads * cfg.head_dim;
                let kv = cfg.n_kv_heads * cfg.head_dim;
                nk.extend([
                    (q, cfg.hidden),
                    (kv, cfg.hidden),
                    (cfg.hidden, q),
                    (cfg.intermediate, cfg.hidden),
                    (cfg.hidden, cfg.intermediate),
                ]);
                // A zero dim is skipped, not swept: `intermediate` is 0 for MoE-only checkpoints (no dense FFN, e.g.
                // Qwen3.6-35B-A3B), and an N=0 or K=0 matmul has nothing to compare.
                nk.retain(|&(n, k)| n > 0 && k > 0);
            }
            if op == Op::Logits {
                nk = vec![(
                    cfg.vocab.min(if quick { 2048 } else { 8192 }),
                    cfg.hidden.min(if quick { 256 } else { 4096 }),
                )];
                if !quick {
                    // full-vocab N with a small K: catches grid-dimension limits on the N axis
                    nk.push((cfg.vocab, 128));
                }
            }
            let ms: &[usize] = match (op, quick) {
                (Op::Logits, _) => &[1, 4],
                (_, true) => &[1, 3],
                (_, false) => &[1, 3, 32],
            };
            for &m in ms {
                for &(n, k) in &nk {
                    for (dt, wt) in [(DType::Bf16, DType::Bf16), (DType::F32, DType::F32)] {
                        let x = rand_t(&mut rng, dt, &[m, k], 1.0);
                        let w = rand_t(&mut rng, wt, &[n, k], 1.0 / (k as f32).sqrt());
                        let od = if op == Op::Logits { DType::F32 } else { dt };
                        push(
                            format!("M={m} N={n} K={k} x={dt} w={wt}"),
                            HostCall {
                                op,
                                inputs: vec![x, w],
                                outputs: vec![OutSpec::new(od, &[m, n])],
                                attrs: attrs(),
                            },
                        );
                    }
                }
            }
        }
        Op::KvAppend => {
            let (nkv, d) = if quick { (2, 32) } else { (cfg.n_kv_heads, cfg.head_dim) };
            for &t in &[1usize, 6] {
                for &p in &[0usize, 9] {
                    for (dt, cdt) in
                        [(DType::Bf16, DType::Bf16), (DType::Bf16, DType::Fp8E4m3), (DType::F32, DType::F32)]
                    {
                        let cap = p + t + 5;
                        let newt = rand_t(&mut rng, dt, &[t, nkv, d], 1.0);
                        let cache = rand_t(&mut rng, cdt, &[nkv, cap, d], 1.0);
                        push(
                            format!("T={t} pos0={p} Nkv={nkv} D={d} cap={cap} new={dt} cache={cdt}"),
                            HostCall {
                                op,
                                inputs: vec![newt],
                                outputs: vec![OutSpec { dtype: cdt, shape: vec![nkv, cap, d], init: Some(cache) }],
                                attrs: OpAttrs { pos0: p, kv_cap: cap, n_kv_heads: nkv, head_dim: d, ..attrs() },
                            },
                        );
                    }
                }
            }
        }
        Op::QkNormRopeKv => {
            // (heads, kv heads, head_dim): a small GQA group, the model's, and the widest head (256) the GPU kernels take
            let dims: Vec<(usize, usize, usize)> = if quick {
                vec![(4, 2, 64)]
            } else {
                vec![(4, 2, 64), (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim), (8, 2, 256)]
            };
            for &t in rows {
                for &(n, nkv, d) in &dims {
                    for &p in &[0usize, 37] {
                        // full and partial rotary (a quarter of the head, Qwen3.5), plain and zero-centred norms
                        for (rdim, woff) in [(0usize, 0.0f32), ((d / 4) & !1, 1.0)] {
                            // the kv cache is a bf16 / f32 one here: an fp8 cache re-rounds a one-ulp difference of the
                            // f32 and f64 stages into a whole fp8 step (KvAppend's own cases cover fp8)
                            for dt in act_dtypes() {
                                let cap = p + t + 5;
                                let cache = |rng: &mut SplitMix64| OutSpec {
                                    dtype: dt,
                                    shape: vec![nkv, cap, d],
                                    init: Some(rand_t(rng, dt, &[nkv, cap, d], 1.0)),
                                };
                                let inputs = vec![
                                    rand_t(&mut rng, dt, &[t, n, d], 1.0),
                                    rand_t(&mut rng, dt, &[t, nkv, d], 1.0),
                                    rand_t(&mut rng, dt, &[t, nkv, d], 1.0),
                                    norm_w(&mut rng, dt, d),
                                    norm_w(&mut rng, dt, d),
                                ];
                                let outputs = vec![OutSpec::new(dt, &[t, n, d]), cache(&mut rng), cache(&mut rng)];
                                push(
                                    format!("T={t} pos0={p} N={n} Nkv={nkv} D={d} {dt} rot={rdim} off={woff}"),
                                    HostCall {
                                        op,
                                        inputs,
                                        outputs,
                                        attrs: OpAttrs {
                                            pos0: p,
                                            theta: cfg.rope_theta,
                                            eps: 1e-6,
                                            w_offset: woff,
                                            rotary_dim: rdim,
                                            n_heads: n,
                                            n_kv_heads: nkv,
                                            head_dim: d,
                                            kv_cap: cap,
                                            ..attrs()
                                        },
                                    },
                                );
                            }
                        }
                    }
                }
            }
        }
        Op::GateSplitQkNormRopeKv => {
            // QkNormRopeKv's cases with the query read from the gated row (the gate half is copied out)
            let dims: Vec<(usize, usize, usize)> = if quick {
                vec![(4, 2, 64)]
            } else {
                vec![(4, 2, 64), (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim), (8, 2, 256)]
            };
            for &t in rows {
                for &(n, nkv, d) in &dims {
                    for &p in &[0usize, 37] {
                        for (rdim, woff) in [(0usize, 0.0f32), ((d / 4) & !1, 1.0)] {
                            for dt in act_dtypes() {
                                let cap = p + t + 5;
                                let cache = |rng: &mut SplitMix64| OutSpec {
                                    dtype: dt,
                                    shape: vec![nkv, cap, d],
                                    init: Some(rand_t(rng, dt, &[nkv, cap, d], 1.0)),
                                };
                                let inputs = vec![
                                    rand_t(&mut rng, dt, &[t, n, 2 * d], 1.0),
                                    rand_t(&mut rng, dt, &[t, nkv, d], 1.0),
                                    rand_t(&mut rng, dt, &[t, nkv, d], 1.0),
                                    norm_w(&mut rng, dt, d),
                                    norm_w(&mut rng, dt, d),
                                ];
                                let outputs =
                                    vec![OutSpec::new(dt, &[t, n, d]), OutSpec::new(dt, &[t, n, d]), cache(&mut rng), cache(&mut rng)];
                                push(
                                    format!("T={t} pos0={p} N={n} Nkv={nkv} D={d} {dt} rot={rdim} off={woff}"),
                                    HostCall {
                                        op,
                                        inputs,
                                        outputs,
                                        attrs: OpAttrs {
                                            pos0: p,
                                            theta: cfg.rope_theta,
                                            eps: 1e-6,
                                            w_offset: woff,
                                            rotary_dim: rdim,
                                            n_heads: n,
                                            n_kv_heads: nkv,
                                            head_dim: d,
                                            kv_cap: cap,
                                            ..attrs()
                                        },
                                    },
                                );
                            }
                        }
                    }
                }
            }
        }
        Op::GdnConvDelta => {
            for c in gdn_conv_delta_sweep(cfg, &mut rng) {
                push(c.label, c.call);
            }
        }
        Op::AttnPrefill | Op::AttnDecode => {
            let heads: Vec<(usize, usize, usize)> =
                if quick { vec![(4, 2, 32)] } else { vec![(4, 2, 32), (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim), (16, 2, 256)] };
            let mut tp: Vec<(usize, usize)> =
                if op == Op::AttnDecode { vec![(1, 0), (1, 7), (1, 40)] } else { vec![(5, 0), (16, 0), (7, 13)] };
            // long context / multi-tile cases (kernels often break past a 64/128-row tile or a few k keys)
            if !quick {
                if op == Op::AttnDecode {
                    tp.extend([(1, 2000), (1, 5000)]);
                } else {
                    tp.extend([(130, 0), (70, 131), (3, 1500), (200, 1100)]);
                }
            }
            for &(n, nkv, d) in &heads {
                for &(t, p) in &tp {
                    for (dt, kdt) in
                        [(DType::Bf16, DType::Bf16), (DType::F32, DType::F32), (DType::Bf16, DType::Fp8E4m3)]
                    {
                        let cap = p + t + 3;
                        let q = rand_t(&mut rng, dt, &[t, n, d], 1.0);
                        let kc = rand_t(&mut rng, kdt, &[nkv, cap, d], 1.0);
                        let vc = rand_t(&mut rng, kdt, &[nkv, cap, d], 1.0);
                        push(
                            format!("T={t} pos0={p} N={n} Nkv={nkv} D={d} q={dt} kv={kdt}"),
                            HostCall {
                                op,
                                inputs: vec![q, kc, vc],
                                outputs: vec![OutSpec::new(dt, &[t, n, d])],
                                attrs: OpAttrs {
                                    pos0: p,
                                    scale: 1.0 / (d as f32).sqrt(),
                                    n_heads: n,
                                    n_kv_heads: nkv,
                                    head_dim: d,
                                    kv_cap: cap,
                                    ..attrs()
                                },
                            },
                        );
                    }
                }
            }
        }
        Op::GateSplit | Op::SigmoidMul | Op::GdnConv | Op::GdnDeltaRule => {
            for c in gdn_sweep(op, cfg, &mut rng) {
                push(c.label, c.call);
            }
        }
        Op::Argmax | Op::Sample => {
            let vs: &[usize] = if quick { &[100] } else { &[100, 4096] };
            for &r in &[1usize, 5] {
                for &v in vs {
                    for dt in [DType::F32, DType::Bf16] {
                        let logits = rand_t(&mut rng, dt, &[r, v], 2.0);
                        if op == Op::Argmax {
                            push(
                                format!("R={r} V={v} {dt}"),
                                HostCall {
                                    op,
                                    inputs: vec![logits],
                                    outputs: vec![OutSpec::new(DType::I32, &[r])],
                                    attrs: attrs(),
                                },
                            );
                        } else {
                            for (temp, tk, tp) in [(1.0f32, 0usize, 1.0f32), (0.7, 20, 0.9), (0.0, 0, 1.0)] {
                                let u: Vec<f32> = (0..r).map(|_| rng.next_f32()).collect();
                                push(
                                    format!("R={r} V={v} {dt} T={temp} k={tk} p={tp}"),
                                    HostCall {
                                        op,
                                        inputs: vec![logits.clone(), HostTensor::from_f32(&[r], &u)],
                                        outputs: vec![OutSpec::new(DType::I32, &[r])],
                                        attrs: OpAttrs { temperature: temp, top_k: tk, top_p: tp, ..attrs() },
                                    },
                                );
                            }
                        }
                    }
                }
            }
        }
        _ => {} // quant::cases::sweep handled the quantised-weight / MoE ops above
    }
    cases
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct AdmissionRow {
    pub op: Op,
    pub kernel: String,
    pub backend: String,
    pub case: String,
    pub exactness: crate::registry::Exactness,
    pub max_abs: f64,
    pub rel_l2: f64,
    pub tol_max_abs: f32,
    pub tol_rel_l2: f32,
    pub pass: bool,
    pub error: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct KernelSummary {
    pub op: Op,
    pub kernel: String,
    pub backend: String,
    pub cases: usize,
    pub failed: usize,
    pub worst_max_abs: f64,
    pub worst_rel_l2: f64,
    pub admitted: bool,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct AdmissionReport {
    pub candidate: DeviceInfo,
    pub reference_backend: String,
    pub rows: Vec<AdmissionRow>,
    pub kernels: Vec<KernelSummary>,
}

impl AdmissionReport {
    pub fn all_pass(&self) -> bool {
        !self.kernels.is_empty() && self.kernels.iter().all(|k| k.admitted)
    }

    /// Mark admitted/rejected kernels in a registry (rejected ones become unselectable).
    pub fn apply(&self, reg: &mut Registry) {
        for k in &self.kernels {
            let st = if k.cases == 0 {
                Admission::Untested
            } else if k.admitted {
                Admission::Admitted
            } else {
                Admission::Rejected
            };
            reg.set_admission(&k.kernel, st);
        }
    }

    /// Plain-text table for the CLI.
    pub fn render(&self, verbose: bool) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        let _ = writeln!(
            s,
            "candidate: {} ({}/{})   reference: {}",
            self.candidate.name, self.candidate.backend, self.candidate.arch, self.reference_backend
        );
        if verbose {
            for r in &self.rows {
                let _ = writeln!(
                    s,
                    "  {:<14} {:<22} {:<5} max_abs={:.3e} rel_l2={:.3e} {}{}",
                    r.op.as_str(),
                    r.kernel,
                    if r.pass { "ok" } else { "FAIL" },
                    r.max_abs,
                    r.rel_l2,
                    r.case,
                    r.error.as_deref().map(|e| format!("  ERROR: {e}")).unwrap_or_default()
                );
            }
        } else {
            for r in self.rows.iter().filter(|r| !r.pass) {
                let _ = writeln!(
                    s,
                    "  FAIL {:<14} {:<22} max_abs={:.3e} rel_l2={:.3e} {}{}",
                    r.op.as_str(),
                    r.kernel,
                    r.max_abs,
                    r.rel_l2,
                    r.case,
                    r.error.as_deref().map(|e| format!("  ERROR: {e}")).unwrap_or_default()
                );
            }
        }
        let _ = writeln!(
            s,
            "\n{:<14} {:<26} {:>6} {:>6} {:>11} {:>11}  result",
            "op", "kernel", "cases", "fail", "max_abs", "rel_l2"
        );
        for k in &self.kernels {
            let _ = writeln!(
                s,
                "{:<14} {:<26} {:>6} {:>6} {:>11.3e} {:>11.3e}  {}",
                k.op.as_str(),
                k.kernel,
                k.cases,
                k.failed,
                k.worst_max_abs,
                k.worst_rel_l2,
                if k.cases == 0 {
                    "NO CASES (untested)"
                } else if k.admitted {
                    "ADMITTED"
                } else {
                    "REJECTED"
                }
            );
        }
        s
    }
}

#[derive(Clone, Debug, Default)]
pub struct AdmissionOpts {
    pub sweep: Option<SweepConfig>,
    pub only_ops: Option<Vec<Op>>,
    pub only_kernel: Option<String>,
}

/// Shapes the gated-delta-net sweep covers: (key heads, value heads, dk, dv, conv taps).
fn gdn_shapes(quick: bool) -> Vec<(usize, usize, usize, usize, usize)> {
    // small shapes catch indexing bugs; the Qwen3.5/3.6-35B-A3B shape is the one that must be right
    let mut v = vec![(2, 4, 32, 32, 4)];
    if !quick {
        v.extend([(1, 3, 64, 48, 4), (2, 2, 128, 64, 3), (1, 2, 256, 40, 4), (16, 32, 128, 128, 4)]);
    }
    v
}

/// Cases for `GateSplit`, `SigmoidMul`, `GdnConv` and `GdnDeltaRule` (non-zero initial state, several
/// chunk lengths: the state must carry).
fn gdn_sweep(op: Op, cfg: &SweepConfig, rng: &mut SplitMix64) -> Vec<Case> {
    let mut cases = Vec::new();
    let ts: &[usize] = if cfg.quick { &[1, 5] } else { &[1, 7, 64, 130] };
    for (hk, hv, dk, dv, ks) in gdn_shapes(cfg.quick) {
        let c = 2 * hk * dk + hv * dv;
        for &t in ts {
            for dt in act_dtypes() {
                let mut push = |label: String, call: HostCall| cases.push(Case { label, call });
                match op {
                    Op::GateSplit => {
                        let (n, d) = (hv, dk);
                        push(
                            format!("T={t} N={n} D={d} {dt}"),
                            HostCall {
                                op,
                                inputs: vec![rand_t(rng, dt, &[t, n, 2 * d], 1.0)],
                                outputs: vec![OutSpec::new(dt, &[t, n, d]), OutSpec::new(dt, &[t, n, d])],
                                attrs: attrs(),
                            },
                        );
                    }
                    Op::SigmoidMul => {
                        let x = hv * dv;
                        push(
                            format!("R={t} X={x} {dt}"),
                            HostCall {
                                op,
                                inputs: vec![rand_t(rng, dt, &[t, x], 1.0), rand_t(rng, dt, &[t, x], 2.0)],
                                outputs: vec![OutSpec::new(dt, &[t, x])],
                                attrs: attrs(),
                            },
                        );
                    }
                    Op::GdnConv => {
                        push(
                            format!("T={t} C={c} K={ks} {dt}"),
                            HostCall {
                                op,
                                inputs: vec![rand_t(rng, dt, &[t, c], 1.0), rand_t(rng, dt, &[c, ks], 0.5)],
                                outputs: vec![
                                    OutSpec::new(dt, &[t, c]),
                                    OutSpec {
                                        dtype: DType::F32,
                                        shape: vec![c, ks - 1],
                                        init: Some(rand_t(rng, DType::F32, &[c, ks - 1], 1.0)),
                                    },
                                ],
                                attrs: attrs(),
                            },
                        );
                    }
                    Op::GdnDeltaRule => {
                        push(
                            format!("T={t} Hk={hk} Hv={hv} dk={dk} dv={dv} {dt}"),
                            HostCall {
                                op,
                                inputs: vec![
                                    rand_t(rng, dt, &[t, c], 1.0),
                                    rand_t(rng, dt, &[t, hv], 1.0),
                                    rand_t(rng, dt, &[t, hv], 1.0),
                                    rand_t(rng, dt, &[hv], 0.5),
                                    rand_t(rng, dt, &[hv], 0.5),
                                ],
                                outputs: vec![
                                    OutSpec::new(dt, &[t, hv, dv]),
                                    OutSpec {
                                        dtype: DType::F32,
                                        shape: vec![hv, dv, dk],
                                        init: Some(rand_t(rng, DType::F32, &[hv, dv, dk], 0.1)),
                                    },
                                ],
                                attrs: attrs(),
                            },
                        );
                    }
                    _ => unreachable!(),
                }
            }
        }
        if matches!(op, Op::GateSplit | Op::SigmoidMul) {
            break; // shapes do not depend on the conv/delta configuration beyond the first entry
        }
    }
    cases
}

/// One `GdnConvDelta` case: random projection rows `x`, taps, gates, and non-zero initial states (conv state and delta
/// state both carry: a kernel that ignores either fails).
fn gdn_conv_delta_case(rng: &mut SplitMix64, t: usize, hk: usize, hv: usize, dk: usize, dv: usize, ks: usize, dt: DType) -> HostCall {
    let c = 2 * hk * dk + hv * dv;
    HostCall {
        op: Op::GdnConvDelta,
        inputs: vec![
            rand_t(rng, dt, &[t, c], 1.0),
            rand_t(rng, dt, &[c, ks], 0.5),
            rand_t(rng, dt, &[t, hv], 1.0),
            rand_t(rng, dt, &[t, hv], 1.0),
            rand_t(rng, dt, &[hv], 0.5),
            rand_t(rng, dt, &[hv], 0.5),
        ],
        outputs: vec![
            OutSpec::new(dt, &[t, hv, dv]),
            OutSpec { dtype: DType::F32, shape: vec![hv, dv, dk], init: Some(rand_t(rng, DType::F32, &[hv, dv, dk], 0.1)) },
            OutSpec { dtype: DType::F32, shape: vec![c, ks - 1], init: Some(rand_t(rng, DType::F32, &[c, ks - 1], 1.0)) },
        ],
        attrs: OpAttrs { eps: 1e-6, ..attrs() },
    }
}

/// Cases for `GdnConvDelta` (the fused conv + delta rule): decode-sized chunks up to 32 tokens, small shapes, a dk = 128 /
/// K = 4 one with two value heads per key head (the shape the GPU kernels take, in the quick sweep too), the 35B's.
fn gdn_conv_delta_sweep(cfg: &SweepConfig, rng: &mut SplitMix64) -> Vec<Case> {
    let mut cases = Vec::new();
    let ts: &[usize] = if cfg.quick { &[1, 5] } else { &[1, 2, 5, 17, 32] };
    let mut shapes = vec![(2usize, 4usize, 32usize, 32usize, 4usize), (1, 2, 128, 32, 4)];
    if !cfg.quick {
        shapes.extend([(2, 2, 128, 64, 3), (16, 32, 128, 128, 4)]);
    }
    for (hk, hv, dk, dv, ks) in shapes {
        for &t in ts {
            for dt in act_dtypes() {
                cases.push(Case {
                    label: format!("T={t} Hk={hk} Hv={hv} dk={dk} dv={dv} K={ks} {dt}"),
                    call: gdn_conv_delta_case(rng, t, hk, hv, dk, dv, ks, dt),
                });
            }
        }
    }
    cases
}

/// Compare two output tensors (f32-decoded).
fn compare(cand: &HostTensor, refr: &HostTensor) -> (f64, f64, bool) {
    if cand.dtype == DType::I32 {
        let a = cand.to_i32().unwrap_or_default();
        let b = refr.to_i32().unwrap_or_default();
        let mism = a.iter().zip(&b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len());
        return (mism as f64, mism as f64, mism == 0);
    }
    let a = cand.to_f32();
    let b = refr.to_f32();
    let mut max_abs = 0f64;
    let mut num = 0f64;
    let mut den = 0f64;
    let mut ref_max = 0f64;
    let mut nonfinite = false;
    for (x, y) in a.iter().zip(&b) {
        if !x.is_finite() || !y.is_finite() {
            if x.to_bits() != y.to_bits() && !(x.is_nan() && y.is_nan()) {
                nonfinite = true;
            }
            continue;
        }
        let d = (*x as f64 - *y as f64).abs();
        max_abs = max_abs.max(d);
        num += d * d;
        den += (*y as f64) * (*y as f64);
        ref_max = ref_max.max((*y as f64).abs());
    }
    let rel = if den > 0.0 { (num / den).sqrt() } else { num.sqrt() };
    // encode the scale into max_abs so callers can apply the relative criterion
    let scaled = max_abs / ref_max.max(1.0);
    (scaled, rel, !nonfinite && a.len() == b.len())
}

/// Run the admission tests of `candidate`'s kernels against `reference`'s reference impls.
pub fn run_admission(
    reference: &dyn Backend,
    candidate: &dyn Backend,
    opts: &AdmissionOpts,
) -> Result<AdmissionReport> {
    let cfg = opts.sweep.clone().unwrap_or_default();
    let cinfo = candidate.info();
    let ref_kernels = reference.kernels();
    let cand_kernels = candidate.kernels();
    let mut rows = Vec::new();
    let mut summaries = Vec::new();

    for kern in cand_kernels.iter() {
        if kern.is_reference && reference.name() == candidate.name() {
            continue; // never grade the reference against itself
        }
        if !kern.arch_matches(&cinfo.arch) {
            continue;
        }
        if let Some(ops) = &opts.only_ops {
            if !ops.contains(&kern.op) {
                continue;
            }
        }
        if let Some(name) = &opts.only_kernel {
            if &kern.name != name {
                continue;
            }
        }
        let refimpl = ref_kernels
            .iter()
            .find(|k| k.op == kern.op && k.is_reference)
            .ok_or_else(|| Error::NoKernel(format!("reference implementation of {}", kern.op)))?;

        let mut n_cases = 0usize;
        let mut failed = 0usize;
        let mut worst_abs = 0f64;
        let mut worst_rel = 0f64;
        for case in sweep(kern.op, &cfg) {
            // shape predicate from the call's own descriptor
            let s = shape_of(&case.call)?;
            if !kern.accepts(&s) {
                continue;
            }
            n_cases += 1;
            // Tolerance applied per output tensor by dtype (`tol_for`); the row reports the one of the
            // first output, which is the governing output of every op with a single result.
            let row_tol = kern.tol_for(case.call.outputs.first().map_or(DType::Bf16, |o| o.dtype));
            let mut row = AdmissionRow {
                op: kern.op,
                kernel: kern.name.clone(),
                backend: kern.backend.clone(),
                case: case.label.clone(),
                exactness: kern.exact,
                max_abs: 0.0,
                rel_l2: 0.0,
                tol_max_abs: row_tol.max_abs,
                tol_rel_l2: row_tol.rel_l2,
                pass: false,
                error: None,
            };
            let outcome = (|| -> Result<(Vec<HostTensor>, Vec<HostTensor>)> {
                let want = run_host(reference, &refimpl.name, &case.call)?;
                let got = run_host(candidate, &kern.name, &case.call)?;
                Ok((got, want))
            })();
            match outcome {
                Err(e) => {
                    row.error = Some(e.to_string());
                    failed += 1;
                }
                Ok((got, want)) => {
                    let mut ok = true;
                    let (mut ma, mut rl) = (0f64, 0f64);
                    for (g, w) in got.iter().zip(&want) {
                        let (a, r, finite) = compare(g, w);
                        ma = ma.max(a);
                        rl = rl.max(r);
                        let int_out = g.dtype == DType::I32;
                        let tol = kern.tol_for(g.dtype);
                        let within = if int_out {
                            finite
                        } else if kern.exact == crate::registry::Exactness::Exact {
                            a == 0.0 && r == 0.0
                        } else {
                            a <= tol.max_abs as f64 && r <= tol.rel_l2 as f64
                        };
                        ok &= finite && within;
                    }
                    row.max_abs = ma;
                    row.rel_l2 = rl;
                    row.pass = ok;
                    if !ok {
                        failed += 1;
                    }
                }
            }
            worst_abs = worst_abs.max(row.max_abs);
            worst_rel = worst_rel.max(row.rel_l2);
            rows.push(row);
        }
        summaries.push(KernelSummary {
            op: kern.op,
            kernel: kern.name.clone(),
            backend: kern.backend.clone(),
            cases: n_cases,
            failed,
            worst_max_abs: worst_abs,
            worst_rel_l2: worst_rel,
            admitted: n_cases > 0 && failed == 0,
        });
    }
    Ok(AdmissionReport { candidate: cinfo, reference_backend: reference.name().to_string(), rows, kernels: summaries })
}

/// Shape facts of a host call (builds throwaway device-free descriptors).
fn shape_of(hc: &HostCall) -> Result<ShapeInfo> {
    use crate::backend::{Buffer, BufferImpl, TensorRef};
    struct Phantom(usize);
    impl BufferImpl for Phantom {
        fn bytes(&self) -> usize {
            self.0
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }
    let mk = |dt: DType, shape: &[usize]| {
        TensorRef::new(Buffer::new(Phantom(shape.iter().product::<usize>() * dt.size_bytes())), dt, shape)
    };
    let call = crate::ops::OpCall::new(
        hc.op,
        hc.inputs.iter().map(|t| mk(t.dtype, &t.shape)).collect(),
        hc.outputs.iter().map(|o| mk(o.dtype, &o.shape)).collect(),
        hc.attrs.clone(),
    );
    call.check()?;
    Ok(ShapeInfo::from_call(&call))
}

// ---------------------------------------------------------------------------
// admission on a model's exact shapes (plan::admit)
// ---------------------------------------------------------------------------

/// Cost bounds for building a host case from a planned call: the f32 reference is serial and slow,
/// so rows (`M`/`T`), matmul `N`, attention positions and embedding vocabulary are clamped; `K`,
/// head counts, head dims, dtypes and attributes stay exact.
#[derive(Clone, Copy, Debug)]
pub struct ShapeCaps {
    pub rows: usize,
    pub n: usize,
    pub pos0: usize,
    pub emb_vocab: usize,
}

/// Clamped dimensions of a planned call, or `None` for ops without an exact-shape case builder.
struct Clamped {
    label: String,
}

fn clamp_call(call: &crate::ops::OpCall, caps: &ShapeCaps) -> Option<Clamped> {
    if let Some(label) = crate::quant::cases::label(call, caps) {
        return Some(Clamped { label });
    }
    let i = &call.inputs;
    let o = &call.outputs;
    let a = &call.attrs;
    // the signatures this builder was written for; an op whose arity changed falls back to the
    // kernel's own sweep instead of being graded on a case that does not match what the kernel gets
    let known = match call.op {
        Op::Embedding | Op::RmsNorm | Op::Add | Op::QkNorm | Op::Matmul | Op::Logits | Op::SiluGlu => (2, 1),
        Op::RmsNormAdd => (3, 2),
        Op::Rope | Op::KvAppend | Op::Argmax => (1, 1),
        Op::AttnPrefill | Op::AttnDecode => (3, 1),
        Op::QkNormRopeKv => (5, 3),
        Op::GateSplit => (1, 2),
        Op::SigmoidMul => (2, 1),
        Op::GdnConv => (2, 2),
        Op::GdnDeltaRule => (5, 2),
        Op::GateSplitQkNormRopeKv => (5, 4),
        Op::GdnConvDelta => (6, 3),
        _ => return None,
    };
    if known != (i.len(), o.len()) {
        return None;
    }
    let label = match call.op {
        Op::Embedding => format!(
            "T={} V={} H={} {}",
            i[0].shape[0].min(caps.rows),
            i[1].shape[0].min(caps.emb_vocab),
            i[1].shape[1],
            i[1].dtype
        ),
        Op::RmsNorm | Op::RmsNormAdd | Op::Add | Op::SiluGlu => {
            format!("R={} H={} {} off={}", i[0].shape[0].min(caps.rows), i[0].shape[1], i[0].dtype, a.w_offset)
        }
        Op::QkNorm => format!(
            "T={} N={} D={} {} off={}",
            i[0].shape[0].min(caps.rows),
            i[0].shape[1],
            i[0].shape[2],
            i[0].dtype,
            a.w_offset
        ),
        Op::Rope => format!(
            "T={} N={} D={} pos0={} theta={} rot={} {}",
            i[0].shape[0].min(caps.rows),
            i[0].shape[1],
            i[0].shape[2],
            a.pos0,
            a.theta,
            a.rotary_dim,
            i[0].dtype
        ),
        Op::GateSplit => format!(
            "T={} N={} D={} {}",
            i[0].shape[0].min(caps.rows),
            i[0].shape[1],
            i[0].shape[2] / 2,
            i[0].dtype
        ),
        Op::SigmoidMul => format!("R={} X={} {}", i[0].shape[0].min(caps.rows), i[0].shape[1], i[0].dtype),
        Op::GdnConv => format!(
            "T={} C={} K={} {} w={}",
            i[0].shape[0].min(caps.rows),
            i[0].shape[1],
            i[1].shape[1],
            i[0].dtype,
            i[1].dtype
        ),
        Op::GdnDeltaRule => {
            let d = crate::gdn::DeltaDims::from_call(call);
            format!("T={} Hk={} Hv={} dk={} dv={} {} eps={}", d.t.min(caps.rows), d.hk, d.hv, d.dk, d.dv, i[0].dtype, a.eps)
        }
        Op::GdnConvDelta => {
            let d = crate::gdn::DeltaDims::from_call(call);
            format!(
                "T={} Hk={} Hv={} dk={} dv={} K={} {} w={} eps={}",
                d.t.min(caps.rows),
                d.hk,
                d.hv,
                d.dk,
                d.dv,
                i[1].shape[1],
                i[0].dtype,
                i[1].dtype,
                a.eps
            )
        }
        Op::GateSplitQkNormRopeKv => {
            let p = a.pos0.min(caps.pos0);
            format!(
                "T={} pos0={p} N={} Nkv={} D={} {} cache={} rot={} off={}",
                i[0].shape[0].min(caps.rows),
                i[0].shape[1],
                i[1].shape[1],
                i[0].shape[2] / 2,
                i[0].dtype,
                o[2].dtype,
                a.rotary_dim,
                a.w_offset
            )
        }
        Op::Matmul | Op::Logits => {
            if !i[1].dtype.is_float() {
                return None; // packed / quantised weights need their own generator
            }
            format!(
                "M={} N={} K={} x={} w={} out={}",
                i[0].shape[0].min(caps.rows),
                i[1].shape[0].min(caps.n),
                i[1].shape[1],
                i[0].dtype,
                i[1].dtype,
                o[0].dtype
            )
        }
        Op::KvAppend => {
            let p = a.pos0.min(caps.pos0);
            format!(
                "T={} pos0={p} Nkv={} D={} new={} cache={}",
                i[0].shape[0].min(caps.rows),
                i[0].shape[1],
                i[0].shape[2],
                i[0].dtype,
                o[0].dtype
            )
        }
        Op::QkNormRopeKv => {
            let p = a.pos0.min(caps.pos0);
            format!(
                "T={} pos0={p} N={} Nkv={} D={} {} cache={} rot={} off={}",
                i[0].shape[0].min(caps.rows),
                i[0].shape[1],
                i[1].shape[1],
                i[0].shape[2],
                i[0].dtype,
                o[1].dtype,
                a.rotary_dim,
                a.w_offset
            )
        }
        Op::AttnPrefill | Op::AttnDecode => {
            let p = a.pos0.min(caps.pos0);
            format!(
                "T={} pos0={p} N={} Nkv={} D={} q={} kv={}",
                i[0].shape[0].min(caps.rows),
                i[0].shape[1],
                i[1].shape[0],
                i[0].shape[2],
                i[0].dtype,
                i[1].dtype
            )
        }
        Op::Argmax => format!("R={} V={} {}", i[0].shape[0].min(caps.rows), i[0].shape[1], i[0].dtype),
        _ => return None,
    };
    Some(Clamped { label: format!("{} {}", call.op, label) })
}

/// Label of the clamped host case for a planned call (the admission cache key part).
pub fn case_label(call: &crate::ops::OpCall, caps: &ShapeCaps) -> Option<String> {
    clamp_call(call, caps).map(|c| c.label)
}

/// Build a host-side case that exercises the same kernel path as `call` (same dtypes, `K`, heads,
/// head dims, attributes; rows / `N` / positions clamped per `caps`) with random data.
pub fn case_from_call(call: &crate::ops::OpCall, caps: &ShapeCaps, rng: &mut SplitMix64) -> Option<(HostCall, String)> {
    if let Some(r) = crate::quant::cases::case_from_call(call, caps, rng) {
        return Some(r);
    }
    let label = clamp_call(call, caps)?.label;
    let i = &call.inputs;
    let o = &call.outputs;
    let a = &call.attrs;
    let op = call.op;
    let hc = match op {
        Op::Embedding => {
            let (t, v, h) = (i[0].shape[0].min(caps.rows), i[1].shape[0].min(caps.emb_vocab), i[1].shape[1]);
            let ids: Vec<i32> = (0..t).map(|_| (rng.next_u64() % v as u64) as i32).collect();
            HostCall {
                op,
                inputs: vec![HostTensor::from_i32(&[t], &ids), rand_t(rng, i[1].dtype, &[v, h], 1.0)],
                outputs: vec![OutSpec::new(o[0].dtype, &[t, h])],
                attrs: a.clone(),
            }
        }
        Op::RmsNorm | Op::RmsNormAdd | Op::Add | Op::SiluGlu => {
            let (r, h, dt) = (i[0].shape[0].min(caps.rows), i[0].shape[1], i[0].dtype);
            let x = rand_t(rng, dt, &[r, h], 1.0);
            let (inputs, outputs) = match op {
                Op::RmsNorm => (vec![x, norm_w(rng, i[1].dtype, h)], vec![OutSpec::new(o[0].dtype, &[r, h])]),
                Op::RmsNormAdd => (
                    vec![x, rand_t(rng, dt, &[r, h], 2.0), norm_w(rng, i[2].dtype, h)],
                    vec![OutSpec::new(o[0].dtype, &[r, h]), OutSpec::new(o[1].dtype, &[r, h])],
                ),
                _ => (vec![x, rand_t(rng, i[1].dtype, &[r, h], 1.0)], vec![OutSpec::new(o[0].dtype, &[r, h])]),
            };
            HostCall { op, inputs, outputs, attrs: a.clone() }
        }
        Op::QkNorm | Op::Rope => {
            let (t, n, d, dt) = (i[0].shape[0].min(caps.rows), i[0].shape[1], i[0].shape[2], i[0].dtype);
            let x = rand_t(rng, dt, &[t, n, d], 1.0);
            let inputs = if op == Op::QkNorm { vec![x, norm_w(rng, i[1].dtype, d)] } else { vec![x] };
            HostCall { op, inputs, outputs: vec![OutSpec::new(o[0].dtype, &[t, n, d])], attrs: a.clone() }
        }
        Op::Matmul | Op::Logits => {
            let (m, n, k) = (i[0].shape[0].min(caps.rows), i[1].shape[0].min(caps.n), i[1].shape[1]);
            HostCall {
                op,
                inputs: vec![
                    rand_t(rng, i[0].dtype, &[m, k], 1.0),
                    rand_t(rng, i[1].dtype, &[n, k], 1.0 / (k as f32).sqrt()),
                ],
                outputs: vec![OutSpec::new(o[0].dtype, &[m, n])],
                attrs: a.clone(),
            }
        }
        Op::KvAppend => {
            let (t, nkv, d) = (i[0].shape[0].min(caps.rows), i[0].shape[1], i[0].shape[2]);
            let p = a.pos0.min(caps.pos0);
            let cap = p + t + 5;
            HostCall {
                op,
                inputs: vec![rand_t(rng, i[0].dtype, &[t, nkv, d], 1.0)],
                outputs: vec![OutSpec {
                    dtype: o[0].dtype,
                    shape: vec![nkv, cap, d],
                    init: Some(rand_t(rng, o[0].dtype, &[nkv, cap, d], 1.0)),
                }],
                attrs: OpAttrs { pos0: p, kv_cap: cap, n_kv_heads: nkv, head_dim: d, ..a.clone() },
            }
        }
        Op::QkNormRopeKv => {
            let (t, n, nkv, d) = (i[0].shape[0].min(caps.rows), i[0].shape[1], i[1].shape[1], i[0].shape[2]);
            let (p, dt) = (a.pos0.min(caps.pos0), i[0].dtype);
            let cap = p + t + 5;
            let cache = |rng: &mut SplitMix64| OutSpec {
                dtype: o[1].dtype,
                shape: vec![nkv, cap, d],
                init: Some(rand_t(rng, o[1].dtype, &[nkv, cap, d], 1.0)),
            };
            HostCall {
                op,
                inputs: vec![
                    rand_t(rng, dt, &[t, n, d], 1.0),
                    rand_t(rng, dt, &[t, nkv, d], 1.0),
                    rand_t(rng, dt, &[t, nkv, d], 1.0),
                    norm_w(rng, i[3].dtype, d),
                    norm_w(rng, i[4].dtype, d),
                ],
                outputs: vec![OutSpec::new(o[0].dtype, &[t, n, d]), cache(rng), cache(rng)],
                attrs: OpAttrs { pos0: p, kv_cap: cap, ..a.clone() },
            }
        }
        Op::AttnPrefill | Op::AttnDecode => {
            let (t, n, d) = (i[0].shape[0].min(caps.rows), i[0].shape[1], i[0].shape[2]);
            let nkv = i[1].shape[0];
            let p = a.pos0.min(caps.pos0);
            let cap = p + t + 3;
            HostCall {
                op,
                inputs: vec![
                    rand_t(rng, i[0].dtype, &[t, n, d], 1.0),
                    rand_t(rng, i[1].dtype, &[nkv, cap, d], 1.0),
                    rand_t(rng, i[2].dtype, &[nkv, cap, d], 1.0),
                ],
                outputs: vec![OutSpec::new(o[0].dtype, &[t, n, d])],
                attrs: OpAttrs { pos0: p, kv_cap: cap, ..a.clone() },
            }
        }
        Op::GateSplit => {
            let (t, n, d2) = (i[0].shape[0].min(caps.rows), i[0].shape[1], i[0].shape[2]);
            HostCall {
                op,
                inputs: vec![rand_t(rng, i[0].dtype, &[t, n, d2], 1.0)],
                outputs: vec![OutSpec::new(o[0].dtype, &[t, n, d2 / 2]), OutSpec::new(o[1].dtype, &[t, n, d2 / 2])],
                attrs: a.clone(),
            }
        }
        Op::SigmoidMul => {
            let (r, x) = (i[0].shape[0].min(caps.rows), i[0].shape[1]);
            HostCall {
                op,
                inputs: vec![rand_t(rng, i[0].dtype, &[r, x], 1.0), rand_t(rng, i[1].dtype, &[r, x], 2.0)],
                outputs: vec![OutSpec::new(o[0].dtype, &[r, x])],
                attrs: a.clone(),
            }
        }
        Op::GdnConv => {
            let (t, c, k) = (i[0].shape[0].min(caps.rows), i[0].shape[1], i[1].shape[1]);
            HostCall {
                op,
                inputs: vec![rand_t(rng, i[0].dtype, &[t, c], 1.0), rand_t(rng, i[1].dtype, &[c, k], 0.5)],
                outputs: vec![
                    OutSpec::new(o[0].dtype, &[t, c]),
                    OutSpec { dtype: DType::F32, shape: vec![c, k - 1], init: Some(rand_t(rng, DType::F32, &[c, k - 1], 1.0)) },
                ],
                attrs: a.clone(),
            }
        }
        Op::GdnDeltaRule => {
            let d = crate::gdn::DeltaDims::from_call(call);
            let t = d.t.min(caps.rows);
            let dt = i[0].dtype;
            HostCall {
                op,
                inputs: vec![
                    rand_t(rng, dt, &[t, d.conv_dim()], 1.0),
                    rand_t(rng, i[1].dtype, &[t, d.hv], 1.0),
                    rand_t(rng, i[2].dtype, &[t, d.hv], 1.0),
                    rand_t(rng, i[3].dtype, &[d.hv], 0.5),
                    rand_t(rng, i[4].dtype, &[d.hv], 0.5),
                ],
                outputs: vec![
                    OutSpec::new(o[0].dtype, &[t, d.hv, d.dv]),
                    OutSpec {
                        dtype: DType::F32,
                        shape: vec![d.hv, d.dv, d.dk],
                        init: Some(rand_t(rng, DType::F32, &[d.hv, d.dv, d.dk], 0.1)),
                    },
                ],
                attrs: a.clone(),
            }
        }
        Op::GateSplitQkNormRopeKv => {
            let (t, n, nkv, d) = (i[0].shape[0].min(caps.rows), i[0].shape[1], i[1].shape[1], i[0].shape[2] / 2);
            let (p, dt) = (a.pos0.min(caps.pos0), i[0].dtype);
            let cap = p + t + 5;
            let cache = |rng: &mut SplitMix64| OutSpec {
                dtype: o[2].dtype,
                shape: vec![nkv, cap, d],
                init: Some(rand_t(rng, o[2].dtype, &[nkv, cap, d], 1.0)),
            };
            HostCall {
                op,
                inputs: vec![
                    rand_t(rng, dt, &[t, n, 2 * d], 1.0),
                    rand_t(rng, dt, &[t, nkv, d], 1.0),
                    rand_t(rng, dt, &[t, nkv, d], 1.0),
                    norm_w(rng, i[3].dtype, d),
                    norm_w(rng, i[4].dtype, d),
                ],
                outputs: vec![OutSpec::new(o[0].dtype, &[t, n, d]), OutSpec::new(o[1].dtype, &[t, n, d]), cache(rng), cache(rng)],
                attrs: OpAttrs { pos0: p, kv_cap: cap, ..a.clone() },
            }
        }
        Op::GdnConvDelta => {
            let d = crate::gdn::DeltaDims::from_call(call);
            let mut hc = gdn_conv_delta_case(rng, d.t.min(caps.rows), d.hk, d.hv, d.dk, d.dv, i[1].shape[1], i[0].dtype);
            hc.attrs = a.clone();
            hc
        }
        Op::Argmax => {
            let (r, v) = (i[0].shape[0].min(caps.rows), i[0].shape[1]);
            HostCall {
                op,
                inputs: vec![rand_t(rng, i[0].dtype, &[r, v], 2.0)],
                outputs: vec![OutSpec::new(DType::I32, &[r])],
                attrs: a.clone(),
            }
        }
        _ => return None,
    };
    Some((hc, label))
}

/// Judge candidate outputs against the reference outputs under `kern`'s tolerance:
/// `(pass, max_abs (scaled), rel_l2)`.
pub fn judge(kern: &crate::registry::KernelImpl, got: &[HostTensor], want: &[HostTensor]) -> (bool, f64, f64) {
    let mut ok = got.len() == want.len();
    let (mut ma, mut rl) = (0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        let (a, r, finite) = compare(g, w);
        ma = ma.max(a);
        rl = rl.max(r);
        let tol = kern.tol_for(g.dtype);
        let within = if g.dtype == DType::I32 {
            finite
        } else if kern.exact == crate::registry::Exactness::Exact {
            a == 0.0 && r == 0.0
        } else {
            a <= tol.max_abs as f64 && r <= tol.rel_l2 as f64
        };
        ok &= finite && within;
    }
    (ok, ma, rl)
}

#[cfg(test)]
mod sweep_dim_tests {
    use super::*;

    /// Qwen3.6-35B-A3B text_config: MoE-only, so `ModelDims::intermediate` is 0.
    fn moe_35b() -> SweepConfig {
        SweepConfig {
            hidden: 2048,
            n_heads: 16,
            n_kv_heads: 2,
            head_dim: 256,
            intermediate: 0,
            vocab: 248_320,
            rope_theta: 1e7,
            quick: false,
            seed: 1,
        }
    }

    #[test]
    fn moe_only_config_sweeps_no_zero_dim_case() {
        let cfg = moe_35b();
        for op in [Op::Matmul, Op::Logits, Op::SiluGlu] {
            let cases = sweep(op, &cfg);
            assert!(!cases.is_empty(), "{op:?}");
            for c in &cases {
                for t in &c.call.inputs {
                    assert!(t.shape.iter().all(|&d| d > 0), "{op:?} {}: zero dim in {:?}", c.label, t.shape);
                }
                for o in &c.call.outputs {
                    assert!(o.shape.iter().all(|&d| d > 0), "{op:?} {}: zero dim out {:?}", c.label, o.shape);
                }
            }
        }
    }
}

#[cfg(test)]
mod plan_case_tests {
    use super::*;
    use crate::backend::{Buffer, BufferImpl, TensorRef};
    use crate::ops::OpCall;

    struct P(usize);
    impl BufferImpl for P {
        fn bytes(&self) -> usize {
            self.0
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }
    fn t(dt: DType, shape: &[usize]) -> TensorRef {
        TensorRef::new(Buffer::new(P(shape.iter().product::<usize>() * dt.size_bytes() + 16)), dt, shape)
    }

    #[test]
    fn matmul_case_keeps_k_and_clamps_rows_and_n() {
        let call = OpCall::new(
            Op::Matmul,
            vec![t(DType::Bf16, &[256, 4096]), t(DType::Bf16, &[151936, 4096])],
            vec![t(DType::Bf16, &[256, 151936])],
            OpAttrs::default(),
        );
        let caps = ShapeCaps { rows: 32, n: 1024, pos0: 100, emb_vocab: 64 };
        let (hc, label) = case_from_call(&call, &caps, &mut SplitMix64::new(1)).unwrap();
        assert_eq!(hc.inputs[0].shape, vec![32, 4096]);
        assert_eq!(hc.inputs[1].shape, vec![1024, 4096]);
        assert!(label.contains("K=4096"), "{label}");
        assert_eq!(case_label(&call, &caps).unwrap(), label);
    }

    #[test]
    fn unknown_ops_have_no_case() {
        let call = OpCall::new(Op::Sample, vec![t(DType::F32, &[1, 8]), t(DType::F32, &[1])], vec![t(DType::I32, &[1])], OpAttrs::default());
        assert!(case_from_call(&call, &ShapeCaps { rows: 4, n: 4, pos0: 4, emb_vocab: 4 }, &mut SplitMix64::new(1)).is_none());
    }
}
