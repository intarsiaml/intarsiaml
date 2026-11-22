//! Rows-exact gate of the Inkling-2-Mini CUDA kernels (lane ink-cuda-mixmoe): row t of a T-row call is BITWISE row 0 of the
//! one-row call on row t's inputs, made after rows 0..t (the conv ring carried from call to call). No tolerance: outputs
//! (and the final ring) are compared as bytes. InkInProj (S 1 and 3, the real stacked shape and an odd small one, T 1..=16),
//! the experts (NVFP4 situ, MXFP8 silu / gelu / situ; random, identical and partly invalid routing; T 2..=16), ConvMix /
//! SconvRes (every epilogue, mix and blend variant at D 2304 with the real ring, and D 64 with a ring the call's own rows
//! wrap), AttnResMix, LoopBlend. A negative control checks that the comparison sees a one-element change.
//!
//! Needs a GPU: without one the tests print SKIPPED (and fail under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_cuda)]

mod common;

use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::quant::cases::{random_experts, random_routing};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, Op, OpAttrs, OpCall, ShapeInfo};
use std::sync::Arc;

struct Rig {
    be: Arc<dyn Backend>,
}

/// an output: dtype, shape, initial contents (in-place state) or None (0xA5 fill: NaN in bf16 / f32)
type Out = (DType, Vec<usize>, Option<HostTensor>);

impl Rig {
    fn new(test: &str) -> Option<Rig> {
        match intarsia_backend_cuda::create(0) {
            Ok(be) => Some(Rig { be }),
            Err(e) => {
                common::skip(test, &format!("no usable CUDA device ({e})"));
                None
            }
        }
    }

    fn put(&self, t: &HostTensor) -> TensorRef {
        let buf = self.be.alloc(t.nbytes().max(16), MemKind::Device).unwrap();
        self.be.zero(&buf, 0).unwrap();
        if t.nbytes() > 0 {
            self.be.upload(&buf, 0, &t.data, 0).unwrap();
        }
        TensorRef::new(buf, t.dtype, &t.shape)
    }

    fn run(&self, kernel: &str, op: Op, inputs: &[HostTensor], outs: &[Out], attrs: &OpAttrs) -> Vec<HostTensor> {
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
                for kname in ["ink_in_proj_bf16_mma", "ink_in_proj_bf16_mma_rows"] {
                    let full = rig.run(kname, Op::InkInProj, &[x.clone(), w.clone()], &outs, &OpAttrs::default());
                    for r in 0..t {
                        let o1: Vec<Out> = ns.iter().map(|&n| (DType::Bf16, vec![1, n], None)).collect();
                        let one = rig.run("ink_in_proj_bf16_mma", Op::InkInProj, &[row_of(&x, 1, r), w.clone()], &o1, &OpAttrs::default());
                        for j in 0..3 {
                            assert_rows_eq(&format!("{kname} D={d} S={s} T={t} seg {j}"), &full[j], 0, r, &one[j]);
                        }
                    }
                }
            }
        }
    }
}

fn mx_bytes(rng: &mut SplitMix64, n: usize) -> Vec<u8> {
    (0..n)
        .map(|_| {
            let r = rng.next_u64();
            ((r & 0x80) as u8) | ((4 + (r >> 8) % 7) as u8) << 3 | ((r >> 16) & 7) as u8
        })
        .collect()
}

#[test]
fn ink_experts_rows_exact() {
    let Some(rig) = Rig::new("ink_experts_rows_exact") else { return };
    let mut rng = SplitMix64::new(0x2c);
    let (e, k, h, i) = (32usize, 8usize, 2304usize, 1152usize);
    let ex = random_experts(&mut rng, e, h, i);
    let nv_w = vec![ex.gate_up_w.clone(), ex.gate_up_s.clone(), ex.down_w.clone(), ex.down_s.clone(), ex.gscales.clone()];
    let mx_w = vec![
        HostTensor::new(DType::Fp8E4m3, vec![e, 2 * i, h], mx_bytes(&mut rng, e * 2 * i * h)).unwrap(),
        HostTensor::new(DType::U8, vec![e, 2 * i, h / 32], (0..e * 2 * i * h / 32).map(|_| 116 + (rng.next_u64() % 4) as u8).collect()).unwrap(),
        HostTensor::new(DType::Fp8E4m3, vec![e, h, i], mx_bytes(&mut rng, e * h * i)).unwrap(),
        HostTensor::new(DType::U8, vec![e, h, i / 32], (0..e * h * i / 32).map(|_| 116 + (rng.next_u64() % 4) as u8).collect()).unwrap(),
    ];
    // 64 x 8 = 512 pairs: the last call grouped in shared memory; 72 and 128: the pair table in scratch and the gather
    for t in [2usize, 3, 4, 7, 16, 64, 72, 128] {
        for routing in ["random", "same", "invalid"] {
            let (ids, ws) = random_routing(&mut rng, t, e, k);
            let mut idv = ids.to_i32().unwrap();
            match routing {
                "same" => {
                    let r0: Vec<i32> = idv[..k].to_vec();
                    for tt in 0..t {
                        idv[tt * k..(tt + 1) * k].copy_from_slice(&r0);
                    }
                }
                "invalid" => {
                    idv[1] = -1;
                    idv[k + 2] = e as i32;
                }
                _ => {}
            }
            let ids = HostTensor::new(DType::I32, vec![t, k], idv.iter().flat_map(|v| v.to_le_bytes()).collect()).unwrap();
            let ws = HostTensor::from_f32(&[t, k], &ws.to_f32().iter().map(|v| v * 8.0).collect::<Vec<_>>());
            let x = rnd(&mut rng, DType::Bf16, &[t, h], 1.0);
            for (kname, op, act) in [
                ("moe_experts_situ_bf16", Op::MoeExperts, "situ"),
                ("moe_experts_mxfp8_bf16", Op::MoeExpertsMxfp8, "situ"),
                ("moe_experts_mxfp8_bf16", Op::MoeExpertsMxfp8, "silu"),
                ("moe_experts_mxfp8_bf16", Op::MoeExpertsMxfp8, "gelu"),
            ] {
                let mut a = OpAttrs { top_k: k, gelu: act == "gelu", ..OpAttrs::default() };
                if act == "situ" {
                    (a.ink.situ, a.ink.gate_cap, a.ink.up_cap) = (true, 4.0, 25.0);
                }
                let wts = if op == Op::MoeExperts { &nv_w } else { &mx_w };
                let call_in = |x: HostTensor, ids: HostTensor, ws: HostTensor| -> Vec<HostTensor> {
                    let mut v = vec![x, ids, ws];
                    v.extend(wts.iter().cloned());
                    v
                };
                let full = rig.run(kname, op, &call_in(x.clone(), ids.clone(), ws.clone()), &[(DType::Bf16, vec![t, h], None)], &a);
                if t <= 16 {
                    let rows_k = format!("{kname}_rows");
                    let full2 = rig.run(&rows_k, op, &call_in(x.clone(), ids.clone(), ws.clone()), &[(DType::Bf16, vec![t, h], None)], &a);
                    assert_eq!(full[0].data, full2[0].data, "{rows_k} differs from {kname}");
                }
                for r in 0..t {
                    let one = rig.run(kname, op, &call_in(row_of(&x, 0, r), row_of(&ids, 0, r), row_of(&ws, 0, r)), &[(DType::Bf16, vec![1, h], None)], &a);
                    assert_rows_eq(&format!("{kname} {act} T={t} {routing}"), &full[0], 0, r, &one[0]);
                }
            }
        }
    }
    // negative control: one activation of the last token changes only that token's output bytes
    let t = 4;
    let (ids, ws) = random_routing(&mut rng, t, e, k);
    let ws = HostTensor::from_f32(&[t, k], &ws.to_f32().iter().map(|v| v * 8.0).collect::<Vec<_>>());
    let x = rnd(&mut rng, DType::Bf16, &[t, h], 1.0);
    let mut x2 = x.clone();
    x2.data[(t - 1) * h * 2 + 10] ^= 0x40;
    let mut a = OpAttrs { top_k: k, ..OpAttrs::default() };
    (a.ink.situ, a.ink.gate_cap, a.ink.up_cap) = (true, 4.0, 25.0);
    let mk = |x: &HostTensor| {
        let mut v = vec![x.clone(), ids.clone(), ws.clone()];
        v.extend(nv_w.iter().cloned());
        v
    };
    let o1 = rig.run("moe_experts_situ_bf16", Op::MoeExperts, &mk(&x), &[(DType::Bf16, vec![t, h], None)], &a);
    let o2 = rig.run("moe_experts_situ_bf16", Op::MoeExperts, &mk(&x2), &[(DType::Bf16, vec![t, h], None)], &a);
    for r in 0..t - 1 {
        assert_eq!(row_of(&o1[0], 0, r).data, row_of(&o2[0], 0, r).data, "negative control: row {r} changed");
    }
    assert_ne!(row_of(&o1[0], 0, t - 1).data, row_of(&o2[0], 0, t - 1).data, "negative control: the changed row did not change");
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
        for t in [2usize, 3, 7, 16] {
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
