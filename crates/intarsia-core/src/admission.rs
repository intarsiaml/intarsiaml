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
            if op == Op::SiluGlu && !quick {
                hs2.push(cfg.intermediate);
            }
            for &r in rows {
                for &h in &hs2 {
                    for dt in act_dtypes() {
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
                            format!("R={r} H={h} {dt}"),
                            HostCall { op, inputs, outputs, attrs: OpAttrs { eps: 1e-6, ..attrs() } },
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
                        for dt in act_dtypes() {
                            let x = rand_t(&mut rng, dt, &[t, n, d], 1.0);
                            let inputs = if op == Op::QkNorm { vec![x, norm_w(&mut rng, dt, d)] } else { vec![x] };
                            push(
                                format!("T={t} N={n} D={d} pos0={p} {dt}"),
                                HostCall {
                                    op,
                                    inputs,
                                    outputs: vec![OutSpec::new(dt, &[t, n, d])],
                                    attrs: OpAttrs { pos0: p, theta: cfg.rope_theta, head_dim: d, ..attrs() },
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
        Op::AttnPrefill | Op::AttnDecode => {
            let heads: Vec<(usize, usize, usize)> =
                if quick { vec![(4, 2, 32)] } else { vec![(4, 2, 32), (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim)] };
            let mut tp: Vec<(usize, usize)> =
                if op == Op::AttnDecode { vec![(1, 0), (1, 7), (1, 40)] } else { vec![(5, 0), (16, 0), (7, 13)] };
            // long context / multi-tile cases (kernels often break past a 64/128-row tile or a few k keys)
            if !quick {
                if op == Op::AttnDecode {
                    tp.extend([(1, 2000)]);
                } else {
                    tp.extend([(130, 0), (70, 131)]);
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
            let mut row = AdmissionRow {
                op: kern.op,
                kernel: kern.name.clone(),
                backend: kern.backend.clone(),
                case: case.label.clone(),
                exactness: kern.exact,
                max_abs: 0.0,
                rel_l2: 0.0,
                tol_max_abs: kern.tol.max_abs,
                tol_rel_l2: kern.tol.rel_l2,
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
                        let within = if int_out {
                            finite
                        } else if kern.exact == crate::registry::Exactness::Exact {
                            a == 0.0 && r == 0.0
                        } else {
                            a <= kern.tol.max_abs as f64 && r <= kern.tol.rel_l2 as f64
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
