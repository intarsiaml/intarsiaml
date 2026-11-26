//! Gates of the Inkling-2-Mini HIP mix / projection kernels (lane ink-hip-mix; port of the CUDA `ink_rows.rs`):
//!  * admission of ops 67 / 68 / 69 / 73 / 74 against the cpu reference (`ink_ref`) at the CUDA tolerances,
//!  * rows-exact (BITWISE): row t of a T-row call is row 0 of the one-row call on row t's inputs, made after rows 0..t (the conv ring
//!    carried from call to call): InkInProj (S 1 and 3, the real stacked shape and an odd small one, T 1..=16), ConvMix / SconvRes
//!    (every epilogue, mix and blend variant at D 2304 with the real ring, and D 64 with a ring the call's own rows wrap),
//!    AttnResMix, LoopBlend,
//!  * a bench of the in-projection GEMV against the 627 GB/s stream rate (`ink_in_proj_bench`, ignored by default).
//!
//! Needs a GPU: without one the tests print SKIPPED (and fail under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_hip)]

mod common;

use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, Op, OpAttrs, OpCall, ShapeInfo};
use std::sync::Arc;

struct Rig {
    be: Arc<dyn Backend>,
    /// device copies of large inputs (the expert weights), keyed by host address and length: uploaded once
    cache: std::cell::RefCell<std::collections::HashMap<(usize, usize, u64), TensorRef>>,
    /// on only in the experts tests, whose large inputs are the weights (alive for the whole test)
    cache_on: std::cell::Cell<bool>,
}

/// the process-wide mix mode switch (`inkmix::set_mix_mode`) is held while a test flips it
static MIX_MODE: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// an output: dtype, shape, initial contents (in-place state) or None (0xA5 fill: NaN in bf16 / f32)
type Out = (DType, Vec<usize>, Option<HostTensor>);

impl Rig {
    fn new(test: &str) -> Option<Rig> {
        let ord = std::env::var("INTARSIA_HIP_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        match intarsia_backend_hip::create(ord) {
            Ok(be) => Some(Rig { be, cache: Default::default(), cache_on: Default::default() }),
            Err(e) => {
                common::skip(test, &format!("no usable HIP device ({e})"));
                None
            }
        }
    }

    fn put(&self, t: &HostTensor) -> TensorRef {
        // (keyed by address, length and a sample of the bytes: a dropped tensor's address can be reused by a new one)
        if t.nbytes() >= 1 << 20 {
            let n = t.nbytes();
            let fp = (0..64).fold(0xcbf29ce484222325u64, |h, j| (h ^ t.data[j * (n - 1) / 63] as u64).wrapping_mul(0x100000001b3));
            let key = (t.data.as_ptr() as usize, n, fp);
            if let Some(r) = self.cache.borrow().get(&key) {
                return r.clone();
            }
            let r = self.put_new(t);
            if self.cache_on.get() {
                self.cache.borrow_mut().insert(key, r.clone());
            }
            return r;
        }
        self.put_new(t)
    }

    fn put_new(&self, t: &HostTensor) -> TensorRef {
        let buf = self.be.alloc(t.nbytes().max(16), MemKind::Device).unwrap();
        self.be.zero(&buf, 0).unwrap();
        if t.nbytes() > 0 {
            self.be.upload(&buf, 0, &t.data, 0).unwrap();
        }
        TensorRef::new(buf, t.dtype, &t.shape)
    }

    fn run(&self, kernel: &str, op: Op, inputs: &[HostTensor], outs: &[Out], attrs: &OpAttrs) -> Vec<HostTensor> {
        self.run_r(kernel, op, &inputs.iter().collect::<Vec<_>>(), outs, attrs)
    }

    /// The call in the default forms, and for the mix kernels also with every fast form on (mask 15) and in the first forms (mask 0): all three must
    /// agree to the bit, so every rows-exact check below also gates "fast == first form".
    fn run_r(&self, kernel: &str, op: Op, inputs: &[&HostTensor], outs: &[Out], attrs: &OpAttrs) -> Vec<HostTensor> {
        let _g = MIX_MODE.lock().unwrap_or_else(|e| e.into_inner());
        let fast = self.run_once(kernel, op, inputs, outs, attrs);
        if matches!(kernel, "sconv_res_bf16" | "attn_res_mix_bf16" | "conv_mix_bf16" | "loop_blend_bf16") {
            // every fast form (mask 15, the fast row kernel at every row count too) and the default (3) against the first forms (mask 0)
            intarsia_backend_hip::inkmix::set_mix_mode(15);
            let all = self.run_once(kernel, op, inputs, outs, attrs);
            intarsia_backend_hip::inkmix::set_mix_mode(0);
            let first = self.run_once(kernel, op, inputs, outs, attrs);
            intarsia_backend_hip::inkmix::set_mix_mode(3);
            for (j, ((a, b), c)) in fast.iter().zip(&first).zip(&all).enumerate() {
                assert_eq!(a.data, b.data, "{kernel}: output {j} of the default forms differs from the first forms");
                assert_eq!(c.data, b.data, "{kernel}: output {j} of the fast forms (mask 15) differs from the first forms");
            }
        }
        fast
    }

    fn run_once(&self, kernel: &str, op: Op, inputs: &[&HostTensor], outs: &[Out], attrs: &OpAttrs) -> Vec<HostTensor> {
        let ins: Vec<TensorRef> = inputs.iter().map(|t| self.put(t)).collect();
        let outputs: Vec<TensorRef> = outs
            .iter()
            .map(|(dt, shape, init)| match init {
                Some(t) => self.put(t),
                None => {
                    let mut t = HostTensor::zeros(*dt, shape);
                    t.data.fill(0xA5);
                    self.put(&t)
                }
            })
            .collect();
        let mut call = OpCall::new(op, ins, outputs.clone(), attrs.clone());
        call.check().unwrap_or_else(|e| panic!("{kernel}: {e}"));
        let k = self.be.kernels().into_iter().find(|k| k.name == kernel && k.op == op).unwrap_or_else(|| panic!("no {kernel}"));
        assert!(k.accepts(&ShapeInfo::from_call(&call)), "{kernel} refuses the call");
        let need = k.scratch_bytes(&ShapeInfo::from_call(&call));
        if need > 0 {
            call.scratch = Some(TensorRef::new(self.be.alloc(need.max(16), MemKind::Device).unwrap(), DType::U8, &[need]));
        }
        self.be.launch(kernel, &call, 0).unwrap_or_else(|e| panic!("{kernel}: {e}"));
        self.be.synchronize(0).unwrap();
        outputs
            .iter()
            .zip(outs)
            .map(|(o, (dt, shape, _))| {
                let mut data = vec![0u8; o.nbytes()];
                self.be.download(&o.buf, o.offset, &mut data, 0).unwrap();
                HostTensor::new(*dt, shape.clone(), data).unwrap()
            })
            .collect()
    }
}

fn rnd(rng: &mut SplitMix64, dt: DType, shape: &[usize], s: f32) -> HostTensor {
    let n: usize = shape.iter().product();
    HostTensor::from_f32_as(dt, shape, &rng.normal_vec(n, s))
}

fn pos(rng: &mut SplitMix64, shape: &[usize]) -> HostTensor {
    let n: usize = shape.iter().product();
    HostTensor::from_f32_as(DType::Bf16, shape, &(0..n).map(|_| 0.5 + rng.next_f32()).collect::<Vec<_>>())
}

fn none() -> HostTensor {
    HostTensor::zeros(DType::Bf16, &[0])
}

/// rows `r` of a tensor whose row axis is `axis` (shape [.., R, ..]): the bytes of the slice as a new tensor with R = 1
fn row_of(t: &HostTensor, axis: usize, r: usize) -> HostTensor {
    if t.shape.iter().product::<usize>() == 0 {
        // an absent operand ([0]) stays absent; an empty source stack [0, R, D] becomes [0, 1, D]
        let mut c = t.clone();
        if c.shape.len() > 1 && axis < c.shape.len() {
            c.shape[axis] = 1;
        }
        return c;
    }
    let es = t.dtype.size_bytes();
    let outer: usize = t.shape[..axis].iter().product();
    let inner: usize = t.shape[axis + 1..].iter().product::<usize>() * es;
    let rows = t.shape[axis];
    let mut data = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        let b = (o * rows + r) * inner;
        data.extend_from_slice(&t.data[b..b + inner]);
    }
    let mut shape = t.shape.clone();
    shape[axis] = 1;
    HostTensor::new(t.dtype, shape, data).unwrap()
}

fn assert_rows_eq(what: &str, full: &HostTensor, axis: usize, r: usize, one: &HostTensor) {
    let a = row_of(full, axis, r);
    assert_eq!(a.data, one.data, "{what}: row {r} differs from the one-row call");
}

fn shape_rows(shape: &[usize], axis: usize, r: usize) -> Vec<usize> {
    let mut s = shape.to_vec();
    if !s.is_empty() && s.iter().product::<usize>() > 0 {
        s[axis] = r;
    }
    s
}

#[test]
fn ink_in_proj_rows_exact() {
    let Some(rig) = Rig::new("ink_in_proj_rows_exact") else { return };
    let mut rng = SplitMix64::new(0x1b);
    for (d, ns) in [(2304usize, [6552usize, 1056, 1056]), (64, [246, 40, 40])] {
        let w = rnd(&mut rng, DType::Bf16, &[ns.iter().sum(), d], (d as f32).powf(-0.5));
        for s in [1usize, 3] {
            for t in [1usize, 2, 3, 5, 8, 9, 16] {
                let x = rnd(&mut rng, DType::Bf16, &[s, t, d], 1.0);
                let outs: Vec<Out> = ns.iter().map(|&n| (DType::Bf16, vec![t, n], None)).collect();
                for kname in ["ink_in_proj_bf16_gemv", "ink_in_proj_bf16_gemv_rows"] {
                    let full = rig.run(kname, Op::InkInProj, &[x.clone(), w.clone()], &outs, &OpAttrs::default());
                    for r in 0..t {
                        let o1: Vec<Out> = ns.iter().map(|&n| (DType::Bf16, vec![1, n], None)).collect();
                        let one = rig.run("ink_in_proj_bf16_gemv", Op::InkInProj, &[row_of(&x, 1, r), w.clone()], &o1, &OpAttrs::default());
                        for j in 0..3 {
                            assert_rows_eq(&format!("{kname} D={d} S={s} T={t} seg {j}"), &full[j], 0, r, &one[j]);
                        }
                    }
                }
            }
        }
    }
}

/// ConvMix / SconvRes: a T-row call against T one-row calls at pos0 + r, the ring carried
#[test]
fn ink_conv_mix_rows_exact() {
    let Some(rig) = Rig::new("ink_conv_mix_rows_exact") else { return };
    let mut rng = SplitMix64::new(0x3d);
    let bf = DType::Bf16;
    let kt = 4usize;
    // (D, Rr): the real ring, and a small ring the call's own rows wrap (deferred ring write above 4 rows)
    for (d, rr, g) in [(2304usize, kt - 1 + 16, 64usize), (64, kt - 1 + 4, 8)] {
        for t in [2usize, 3, 7, 16, 20] {
            for pos0 in [0usize, 29] {
                // (op, res, add, sum, N, Q, norm, S)
                let variants: &[(Op, bool, bool, bool, usize, usize, bool, usize)] = &[
                    (Op::SconvRes, true, true, true, 0, 0, false, 0),
                    (Op::SconvRes, false, false, false, 0, 0, false, 0),
                    (Op::ConvMix, true, true, true, 4, 1, true, 0),
                    (Op::ConvMix, true, false, false, 3, 3, true, 0),
                    (Op::ConvMix, false, false, false, 2, 1, false, 0),
                    (Op::ConvMix, true, true, true, 0, 0, false, 0),
                    (Op::ConvMix, true, true, true, 4, 1, false, 3),
                    (Op::ConvMix, true, false, true, 3, 3, false, 3),
                    (Op::ConvMix, true, true, true, 2, 1, false, 1),
                ];
                for &(op, res, add, sum, n, q, norm, s) in variants {
                    let x = rnd(&mut rng, bf, &[t, d], 1.0);
                    let taps = rnd(&mut rng, bf, &[kt, d], 0.5);
                    let resv = if res { rnd(&mut rng, bf, &[t, d], 1.0) } else { none() };
                    let addv = if add { rnd(&mut rng, bf, &[t, d], 1.0) } else { none() };
                    let ring0 = rnd(&mut rng, bf, &[rr, d], 1.0);
                    let src = rnd(&mut rng, bf, &[n, t, d], 1.5);
                    let qs = rnd(&mut rng, DType::F32, &[q, d], 0.2);
                    let nrows = if s > 0 { s } else { q };
                    let nw = if norm || s > 0 { pos(&mut rng, &[nrows, d]) } else { none() };
                    let (msg, down, gain, forget) = if s > 0 {
                        (
                            rnd(&mut rng, DType::F32, &[s, t, d], 1.0),
                            rnd(&mut rng, bf, &[s, g, 2 * d], (2.0 * d as f32).powf(-0.5)),
                            rnd(&mut rng, bf, &[s, d, g], 0.7),
                            rnd(&mut rng, bf, &[s, d, g], 0.7),
                        )
                    } else {
                        (none(), none(), none(), none())
                    };
                    let mix_dt = if s > 0 || norm { bf } else { DType::F32 };
                    let mix_rows = if s > 0 { s } else { q };
                    let kname = if op == Op::SconvRes { "sconv_res_bf16" } else { "conv_mix_bf16" };
                    let build = |rows: usize, r0: usize, ring: &HostTensor, p0: usize| -> (Vec<HostTensor>, Vec<Out>, OpAttrs) {
                        let sl = |v: &HostTensor, axis: usize| -> HostTensor {
                            if rows == t {
                                v.clone()
                            } else {
                                row_of(v, axis, r0)
                            }
                        };
                        let mut ins = vec![sl(&x, 0), taps.clone(), sl(&resv, 0), sl(&addv, 0)];
                        let mut outs: Vec<Out> = vec![
                            (bf, vec![rows, d], None),
                            (bf, vec![rr, d], Some(ring.clone())),
                            if sum { (bf, vec![rows, d], None) } else { (bf, vec![0], None) },
                        ];
                        if op == Op::ConvMix {
                            ins.extend([sl(&src, 1), qs.clone(), nw.clone(), sl(&msg, 1), down.clone(), gain.clone(), forget.clone()]);
                            outs.push(if q == 0 { (bf, vec![0], None) } else { (mix_dt, vec![mix_rows, rows, d], None) });
                        }
                        (ins, outs, OpAttrs { eps: 1e-6, pos0: p0, ..OpAttrs::default() })
                    };
                    let (ins, outs, a) = build(t, 0, &ring0, pos0);
                    let full = rig.run(kname, op, &ins, &outs, &a);
                    let mut ring = ring0.clone();
                    let what = format!("{kname} D={d} Rr={rr} T={t} pos0={pos0} res={res} add={add} sum={sum} N={n} Q={q} norm={norm} S={s}");
                    for r in 0..t {
                        let (ins, outs, a) = build(1, r, &ring, pos0 + r);
                        let one = rig.run(kname, op, &ins, &outs, &a);
                        assert_rows_eq(&format!("{what} y"), &full[0], 0, r, &one[0]);
                        if sum {
                            assert_rows_eq(&format!("{what} sum"), &full[2], 0, r, &one[2]);
                        }
                        if op == Op::ConvMix && q > 0 {
                            assert_rows_eq(&format!("{what} mix"), &full[3], 1, r, &one[3]);
                        }
                        ring = one[1].clone();
                    }
                    assert_eq!(full[1].data, ring.data, "{what}: final ring differs");
                }
            }
        }
    }
}

#[test]
fn ink_attn_res_mix_and_blend_rows_exact() {
    let Some(rig) = Rig::new("ink_attn_res_mix_and_blend_rows_exact") else { return };
    let mut rng = SplitMix64::new(0x4e);
    let bf = DType::Bf16;
    for d in [2304usize, 64] {
        for r in [2usize, 5, 16, 300] {
            for &(n, q, addend, norm) in &[(4usize, 1usize, true, true), (7, 3, false, true), (0, 2, false, false), (2, 4, true, false)] {
                let src = rnd(&mut rng, bf, &[n, r, d], 1.5);
                let last = rnd(&mut rng, bf, &[r, d], 1.0);
                let add = if addend { rnd(&mut rng, bf, &[r, d], 1.0) } else { none() };
                let qs = rnd(&mut rng, DType::F32, &[q, d], 0.2);
                let nw = if norm { pos(&mut rng, &[q, d]) } else { none() };
                let odt = if norm { bf } else { DType::F32 };
                let a = OpAttrs { eps: 1e-6, ..OpAttrs::default() };
                let full = rig.run("attn_res_mix_bf16", Op::AttnResMix, &[src.clone(), last.clone(), add.clone(), qs.clone(), nw.clone()], &[(odt, vec![q, r, d], None)], &a);
                for t in [0, r / 2, r - 1] {
                    let one = rig.run(
                        "attn_res_mix_bf16",
                        Op::AttnResMix,
                        &[row_of(&src, 1, t), row_of(&last, 0, t), row_of(&add, 0, t), qs.clone(), nw.clone()],
                        &[(odt, vec![q, 1, d], None)],
                        &a,
                    );
                    assert_rows_eq(&format!("attn_res_mix D={d} R={r} N={n} Q={q}"), &full[0], 1, t, &one[0]);
                }
            }
            let g = if d > 64 { 64 } else { 8 };
            for &(s, sd) in &[(3usize, 1usize), (3, 3), (1, 1)] {
                let ins = vec![
                    rnd(&mut rng, DType::F32, &[sd, r, d], 1.0),
                    rnd(&mut rng, DType::F32, &[s, r, d], 1.0),
                    rnd(&mut rng, bf, &[s, g, 2 * d], (2.0 * d as f32).powf(-0.5)),
                    rnd(&mut rng, bf, &[s, d, g], 0.5),
                    rnd(&mut rng, bf, &[s, d, g], 0.5),
                    pos(&mut rng, &[s, d]),
                ];
                let a = OpAttrs { eps: 1e-6, ..OpAttrs::default() };
                let full = rig.run("loop_blend_bf16", Op::LoopBlend, &ins, &[(bf, vec![s, r, d], None)], &a);
                for t in [0, r / 2, r - 1] {
                    let mut one_in = ins.clone();
                    one_in[0] = row_of(&ins[0], 1, t);
                    one_in[1] = row_of(&ins[1], 1, t);
                    let one = rig.run("loop_blend_bf16", Op::LoopBlend, &one_in, &[(bf, shape_rows(&[s, r, d], 1, 1), None)], &a);
                    assert_rows_eq(&format!("loop_blend D={d} R={r} S={s} Sd={sd}"), &full[0], 1, t, &one[0]);
                }
            }
        }
    }
}


/// Admission of the five ops against the cpu reference (the CUDA tolerances, `inkmix.rs`): every kernel of the table runs a case and
/// is admitted (`intarsia kernels test --backend hip --op <op>`).
#[test]
fn ink_mix_ops_match_the_reference() {
    use intarsia_core::admission::{run_admission, AdmissionOpts};
    let ord = std::env::var("INTARSIA_HIP_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let hip = match intarsia_backend_hip::create(ord) {
        Ok(b) => b,
        Err(e) => return common::skip("ink_mix_ops_match_the_reference", &format!("no usable HIP device ({e})")),
    };
    let cpu = intarsia_backend_cpu::CpuBackend::arc();
    let report = run_admission(
        &*cpu,
        &*hip,
        &AdmissionOpts {
            sweep: None,
            only_ops: Some(vec![Op::AttnResMix, Op::LoopBlend, Op::SconvRes, Op::InkInProj, Op::ConvMix, Op::InkRoute]),
            only_kernel: None,
        },
    )
    .unwrap();
    println!("{}", report.render(true));
    for (name, _) in intarsia_backend_hip::inkmix::TABLE {
        let k = report.kernels.iter().find(|k| k.kernel == *name).unwrap_or_else(|| panic!("{name}: no summary"));
        assert!(k.cases > 0, "{name}: no admission case ran it");
        assert!(k.admitted, "{name}: not admitted");
    }
}

/// Launches `kernel` `iters` times back to back (inputs uploaded once, outputs filled once) and returns microseconds per launch.
fn bench(rig: &Rig, kernel: &str, op: Op, inputs: &[HostTensor], outs: &[Out], attrs: &OpAttrs, iters: usize) -> f64 {
    let ins: Vec<TensorRef> = inputs.iter().map(|t| rig.put_new(t)).collect();
    let outputs: Vec<TensorRef> = outs
        .iter()
        .map(|(dt, shape, init)| match init {
            Some(t) => rig.put_new(t),
            None => rig.put_new(&HostTensor::zeros(*dt, shape)),
        })
        .collect();
    let mut call = OpCall::new(op, ins, outputs, attrs.clone());
    call.check().unwrap();
    let k = rig.be.kernels().into_iter().find(|k| k.name == kernel && k.op == op).unwrap();
    assert!(k.accepts(&ShapeInfo::from_call(&call)), "{kernel} refuses the call");
    let need = k.scratch_bytes(&ShapeInfo::from_call(&call));
    if need > 0 {
        call.scratch = Some(TensorRef::new(rig.be.alloc(need.max(16), MemKind::Device).unwrap(), DType::U8, &[need]));
    }
    for _ in 0..20 {
        rig.be.launch(kernel, &call, 0).unwrap();
    }
    rig.be.synchronize(0).unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        rig.be.launch(kernel, &call, 0).unwrap();
    }
    rig.be.synchronize(0).unwrap();
    t0.elapsed().as_secs_f64() * 1e6 / iters as f64
}

/// Bench of the five ops at the real decode / prefill shapes (D 2304, Nq 6552, Nk = Nv 1056, controller rank 64):
/// `cargo test --release -p intarsia-backend-hip --test ink_mix ink_mix_bench -- --ignored --nocapture`.
/// The in-projection streams 39.9 MB of weight per call: its roofline at 627 GB/s is 63.6 us.
#[test]
#[ignore]
fn ink_mix_bench() {
    let Some(rig) = Rig::new("ink_mix_bench") else { return };
    let mut rng = SplitMix64::new(0xbe);
    let bf = DType::Bf16;
    let (d, ns, g, kt) = (2304usize, [6552usize, 1056, 1056], 64usize, 4usize);
    let nsum: usize = ns.iter().sum();
    let w = rnd(&mut rng, bf, &[nsum, d], (d as f32).powf(-0.5));
    let wbytes = (nsum * d * 2) as f64;
    for (s, t) in [(1usize, 1usize), (1, 4), (1, 16), (3, 1), (3, 16), (1, 64), (1, 512)] {
        let x = rnd(&mut rng, bf, &[s, t, d], 1.0);
        let outs: Vec<Out> = ns.iter().map(|&n| (bf, vec![t, n], None)).collect();
        let kname = if t <= 16 { "ink_in_proj_bf16_gemv" } else { "ink_in_proj_bf16_matmul" };
        let us = bench(&rig, kname, Op::InkInProj, &[x, w.clone()], &outs, &OpAttrs::default(), 200);
        println!(
            "BENCH in_proj {kname} S={s} T={t}: {us:.1} us, weights {:.0} GB/s ({:.0}% of 627), {:.1} TFLOP/s",
            wbytes / us / 1e3,
            wbytes / us / 1e3 / 627.0 * 100.0,
            2.0 * (t * nsum * d) as f64 / us / 1e6
        );
    }
    // the mix ops in both forms (mode 0: the first forms, 1: the fast forms): same-window before / after
    for t in [1usize, 2, 4, 16, 256, 1024] {
        let a = OpAttrs { eps: 1e-6, ..OpAttrs::default() };
        let rr = kt - 1 + 16;
        let s = 3usize;
        let sconv_in = vec![rnd(&mut rng, bf, &[t, d], 1.0), rnd(&mut rng, bf, &[kt, d], 0.5), rnd(&mut rng, bf, &[t, d], 1.0), rnd(&mut rng, bf, &[t, d], 1.0)];
        let souts: Vec<Out> = vec![(bf, vec![t, d], None), (bf, vec![rr, d], Some(rnd(&mut rng, bf, &[rr, d], 1.0))), (bf, vec![t, d], None)];
        let mix_in = vec![rnd(&mut rng, bf, &[6, t, d], 1.5), rnd(&mut rng, bf, &[t, d], 1.0), none(), rnd(&mut rng, DType::F32, &[1, d], 0.2), pos(&mut rng, &[1, d])];
        let blend_in = vec![
            rnd(&mut rng, DType::F32, &[1, t, d], 1.0),
            rnd(&mut rng, DType::F32, &[s, t, d], 1.0),
            rnd(&mut rng, bf, &[s, g, 2 * d], (2.0 * d as f32).powf(-0.5)),
            rnd(&mut rng, bf, &[s, d, g], 0.5),
            rnd(&mut rng, bf, &[s, d, g], 0.5),
            pos(&mut rng, &[s, d]),
        ];
        // ConvMix: conv + 6 sources + 1 query + norm (no blend), and with the blend (3 streams)
        let cm_common = |msg: bool| -> (Vec<HostTensor>, Vec<Out>) {
            let mut rng = SplitMix64::new(0xc0 + t as u64);
            let mut ins = vec![
                rnd(&mut rng, bf, &[t, d], 1.0),
                rnd(&mut rng, bf, &[kt, d], 0.5),
                rnd(&mut rng, bf, &[t, d], 1.0),
                rnd(&mut rng, bf, &[t, d], 1.0),
                rnd(&mut rng, bf, &[6, t, d], 1.5),
                rnd(&mut rng, DType::F32, &[1, d], 0.2),
                pos(&mut rng, &[if msg { s } else { 1 }, d]),
            ];
            let mut outs: Vec<Out> = vec![(bf, vec![t, d], None), (bf, vec![rr, d], Some(rnd(&mut rng, bf, &[rr, d], 1.0))), (bf, vec![t, d], None)];
            if msg {
                ins.extend([
                    rnd(&mut rng, DType::F32, &[s, t, d], 1.0),
                    rnd(&mut rng, bf, &[s, g, 2 * d], (2.0 * d as f32).powf(-0.5)),
                    rnd(&mut rng, bf, &[s, d, g], 0.5),
                    rnd(&mut rng, bf, &[s, d, g], 0.5),
                ]);
                outs.push((bf, vec![s, t, d], None));
            } else {
                ins.extend([none(), none(), none(), none()]);
                outs.push((bf, vec![1, t, d], None));
            }
            (ins, outs)
        };
        let (cm_in, cm_out) = cm_common(false);
        let (cb_in, cb_out) = cm_common(true);
        let cases: [(&str, &str, Op, &[HostTensor], &[Out]); 5] = [
            ("sconv_res", "sconv_res_bf16", Op::SconvRes, &sconv_in, &souts),
            ("attn_res_mix N6 Q1", "attn_res_mix_bf16", Op::AttnResMix, &mix_in, &[(bf, vec![1, t, d], None)]),
            ("loop_blend S3 G64", "loop_blend_bf16", Op::LoopBlend, &blend_in, &[(bf, vec![s, t, d], None)]),
            ("conv_mix N6 Q1 norm", "conv_mix_bf16", Op::ConvMix, &cm_in, &cm_out),
            ("conv_mix N6 Q1 blend S3", "conv_mix_bf16", Op::ConvMix, &cb_in, &cb_out),
        ];
        let iters = if t >= 256 { 100 } else { 300 };
        for (label, kname, op, ins, outs) in cases {
            let masks = [0i32, 1, 2, 3, 8];
            let mut us = [0f64; 5];
            for (k, &mode) in masks.iter().enumerate() {
                intarsia_backend_hip::inkmix::set_mix_mode(mode);
                us[k] = bench(&rig, kname, op, ins, outs, &a, iters);
            }
            intarsia_backend_hip::inkmix::set_mix_mode(3);
            println!(
                "BENCH {label} T={t}: first {:.1} us | row loads (<=16 rows) {:.1} | fused blend {:.1} | default (3) {:.1} | row loads any T {:.1}",
                us[0], us[1], us[2], us[3], us[4]
            );
        }
    }
    // the router: Logits + MoeRoute (two launches) against the fused InkRoute, the real shape (D 2304, 128 experts, top-8)
    {
        let (e, k) = (128usize, 8usize);
        let w = rnd(&mut rng, bf, &[e, d], (d as f32).powf(-0.5));
        let bias = rnd(&mut rng, DType::F32, &[e], 0.1);
        let mut a = OpAttrs { top_k: k, scale: 8.0, norm_topk: true, temperature: 1.0, ..OpAttrs::default() };
        a.dsv.score_func = intarsia_core::dsv41::ops::ScoreFunc::Sigmoid;
        for t in [1usize, 4] {
            let x = rnd(&mut rng, bf, &[t, d], 1.0);
            let lk = if t <= 4 { "logits_bf16_gemv" } else { "logits_bf16_gemv_rows" };
            let us_l = bench(&rig, lk, Op::Logits, &[x.clone(), w.clone()], &[(DType::F32, vec![t, e], None)], &OpAttrs::default(), 300);
            let lg = rnd(&mut rng, DType::F32, &[t, e], 1.0);
            let us_r = bench(&rig, "moe_route", Op::MoeRoute, &[lg, bias.clone()], &[(DType::F32, vec![t, k], None), (DType::I32, vec![t, k], None)], &a, 300);
            let outs: Vec<Out> = vec![(DType::F32, vec![t, k], None), (DType::I32, vec![t, k], None), (DType::F32, vec![t, e], None)];
            let us_f = bench(&rig, "ink_route_bf16", Op::InkRoute, &[x, w.clone(), bias.clone()], &outs, &a, 300);
            println!("BENCH router T={t}: {lk} {us_l:.1} + moe_route {us_r:.1} = {:.1} us, fused ink_route_bf16 {us_f:.1} us", us_l + us_r);
        }
    }
}

/// InkRoute (76) is BITWISE its pair on the same call: `logits_bf16_gemv` +
/// `moe_route`, for the real router (D 2304, 128 experts, top-8, route_scale 8, sigmoid) and the other score functions, T 1..=4 (the registered range: above it the
/// fused kernel is slower than the pair), with and
/// without the logits output; a row is bitwise the one-row call.
#[test]
fn ink_route_bitwise_pair() {
    let Some(rig) = Rig::new("ink_route_bitwise_pair") else { return };
    let mut rng = SplitMix64::new(0x7a);
    let bf = DType::Bf16;
    for &(d, e, k, scale) in &[(2304usize, 128usize, 8usize, 8.0f32), (64, 40, 6, 1.0), (72, 17, 3, 1.5)] {
        let w = rnd(&mut rng, bf, &[e, d], (d as f32).powf(-0.5));
        let bias = rnd(&mut rng, DType::F32, &[e], 0.1);
        for t in [1usize, 2, 3, 4] {
            let x = rnd(&mut rng, bf, &[t, d], 1.0);
            for (sf, norm, temp) in [(1u8, true, 1.0f32), (0, true, 1.0), (2, true, 1.0), (1, false, 0.7)] {
                let mut a = OpAttrs { top_k: k, scale, norm_topk: norm, temperature: temp, ..OpAttrs::default() };
                a.dsv.score_func = match sf {
                    0 => intarsia_core::dsv41::ops::ScoreFunc::Softmax,
                    1 => intarsia_core::dsv41::ops::ScoreFunc::Sigmoid,
                    _ => intarsia_core::dsv41::ops::ScoreFunc::SqrtSoftplus,
                };
                // the pair
                let lk = if t <= 4 { "logits_bf16_gemv" } else { "logits_bf16_gemv_rows" };
                let lg = rig.run(lk, Op::Logits, &[x.clone(), w.clone()], &[(DType::F32, vec![t, e], None)], &OpAttrs::default());
                let rt = rig.run("moe_route", Op::MoeRoute, &[lg[0].clone(), bias.clone()], &[(DType::F32, vec![t, k], None), (DType::I32, vec![t, k], None)], &a);
                for with_logits in [true, false] {
                    let outs: Vec<Out> = vec![
                        (DType::F32, vec![t, k], None),
                        (DType::I32, vec![t, k], None),
                        if with_logits { (DType::F32, vec![t, e], None) } else { (DType::F32, vec![0], None) },
                    ];
                    let f = rig.run("ink_route_bf16", Op::InkRoute, &[x.clone(), w.clone(), bias.clone()], &outs, &a);
                    let what = format!("D={d} E={e} K={k} T={t} sf={sf} norm={norm} logits={with_logits}");
                    assert_eq!(f[0].data, rt[0].data, "{what}: weights differ from Logits + MoeRoute");
                    assert_eq!(f[1].data, rt[1].data, "{what}: ids differ from Logits + MoeRoute");
                    if with_logits {
                        assert_eq!(f[2].data, lg[0].data, "{what}: logits differ from {lk}");
                    }
                    for r in 0..t {
                        let o1: Vec<Out> = vec![(DType::F32, vec![1, k], None), (DType::I32, vec![1, k], None), (DType::F32, vec![0], None)];
                        let one = rig.run("ink_route_bf16", Op::InkRoute, &[row_of(&x, 0, r), w.clone(), bias.clone()], &o1, &a);
                        assert_rows_eq(&format!("{what} weights"), &f[0], 0, r, &one[0]);
                        assert_rows_eq(&format!("{what} ids"), &f[1], 0, r, &one[1]);
                    }
                }
            }
        }
    }
}
