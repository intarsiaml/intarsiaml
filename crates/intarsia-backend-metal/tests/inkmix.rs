#![cfg(target_os = "macos")]
//! Rows-exact (G0) for the Inkling-2 mix / conv / in-projection kernels (inkmix.rs): an R-row call is bitwise R one-row calls
//! made in order (the conv ring evolving between them), outputs and the final ring alike, at the real widths (D 2304, the
//! stacked in-projection 6552 + 1056 + 1056, up to 7 stored sources, 3 blend streams) and a small one, for rows 1..16 and a
//! 40-row chunk longer than the ring (the deferred ring write). Numerics against the f64 reference are `intarsia kernels
//! test --backend metal` (the admission sweep).

mod common;
use common::*;
use intarsia_backend_metal::bench::{f32_to_bf16, Rng};
use intarsia_backend_metal::inkmix::*;
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};

/// A host tensor: dtype, shape, raw little-endian bytes.
#[derive(Clone)]
struct Ht {
    dt: DType,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl Ht {
    fn rand(dt: DType, shape: &[usize], scale: f32, rng: &mut Rng) -> Ht {
        let n: usize = shape.iter().product();
        let data = match dt {
            DType::Bf16 => (0..n).flat_map(|_| f32_to_bf16(rng.unit() * scale).to_le_bytes()).collect(),
            _ => (0..n).flat_map(|_| (rng.unit() * scale).to_le_bytes()).collect(),
        };
        Ht { dt, shape: shape.to_vec(), data }
    }
    fn pos(shape: &[usize], rng: &mut Rng) -> Ht {
        let n: usize = shape.iter().product();
        Ht { dt: DType::Bf16, shape: shape.to_vec(), data: (0..n).flat_map(|_| f32_to_bf16(1.0 + 0.5 * rng.unit()).to_le_bytes()).collect() }
    }
    fn zeros(dt: DType, shape: &[usize]) -> Ht {
        Ht { dt, shape: shape.to_vec(), data: vec![0; shape.iter().product::<usize>() * dt.size_bytes()] }
    }
    fn none() -> Ht {
        Ht::zeros(DType::Bf16, &[0])
    }
    /// Row `t` of the middle axis of an `[A, T, D]` tensor (or of the first axis of `[T, D]`), keeping the rank.
    fn row(&self, t: usize) -> Ht {
        if self.shape.iter().product::<usize>() == 0 {
            let mut z = self.clone();
            if z.shape.len() == 3 {
                z.shape[1] = 1;
            }
            return z;
        }
        let es = self.dt.size_bytes();
        if self.shape.len() == 2 {
            let w = self.shape[1] * es;
            return Ht { dt: self.dt, shape: vec![1, self.shape[1]], data: self.data[t * w..(t + 1) * w].to_vec() };
        }
        let (a, tt, d) = (self.shape[0], self.shape[1], self.shape[2]);
        let w = d * es;
        let mut data = Vec::new();
        for i in 0..a {
            data.extend_from_slice(&self.data[(i * tt + t) * w..(i * tt + t + 1) * w]);
        }
        Ht { dt: self.dt, shape: vec![a, 1, d], data }
    }
    fn up(&self, be: &MetalBackend) -> TensorRef {
        let b = be.alloc(self.data.len().max(16), MemKind::Device).unwrap();
        if !self.data.is_empty() {
            be.upload(&b, 0, &self.data, 0).unwrap();
        }
        TensorRef { buf: b, offset: 0, dtype: self.dt, shape: self.shape.clone() }
    }
}

fn down(be: &MetalBackend, t: &TensorRef) -> Vec<u8> {
    let mut v = vec![0u8; t.numel() * t.dtype.size_bytes()];
    if !v.is_empty() {
        be.download(&t.buf, t.offset, &mut v, 0).unwrap();
    }
    v
}

fn scratch(be: &MetalBackend, call: &OpCall) -> Option<TensorRef> {
    let s = ShapeInfo::from_call(call);
    let n = blend_scratch(&s);
    let b: Buffer = be.alloc(n, MemKind::Device).unwrap();
    Some(TensorRef { buf: b, offset: 0, dtype: DType::U8, shape: vec![n] })
}

/// Run `kernel` on `op` with host inputs / outputs (outputs as initial contents); returns the outputs' bytes.
fn run(be: &MetalBackend, kernel: &str, op: Op, ins: &[Ht], outs: &[Ht], attrs: &OpAttrs, exact: bool, dev_pos: Option<i32>) -> Vec<Vec<u8>> {
    let it: Vec<TensorRef> = ins.iter().map(|h| h.up(be)).collect();
    let ot: Vec<TensorRef> = outs.iter().map(|h| h.up(be)).collect();
    let mut call = OpCall::new(op, it, ot.clone(), attrs.clone());
    call.exact_rows = exact;
    if let Some(p) = dev_pos {
        let pb = be.alloc(16, MemKind::Device).unwrap();
        be.upload(&pb, 0, &p.to_le_bytes(), 0).unwrap();
        call.pos = Some(TensorRef { buf: pb, offset: 0, dtype: DType::I32, shape: vec![1] });
    }
    call.check().unwrap();
    call.scratch = scratch(be, &call);
    be.launch(kernel, &call, 0).unwrap();
    be.synchronize(0).unwrap();
    ot.iter().map(|t| down(be, t)).collect()
}

/// Byte rows of `[A, T, D]` (or `[T, D]`) output `bytes`, row t of every A block, for comparison with a 1-row call.
fn out_row(bytes: &[u8], shape: &[usize], es: usize, t: usize) -> Vec<u8> {
    if shape.iter().product::<usize>() == 0 {
        return Vec::new();
    }
    let h = Ht { dt: if es == 2 { DType::Bf16 } else { DType::F32 }, shape: shape.to_vec(), data: bytes.to_vec() };
    h.row(t).data
}

#[test]
fn in_proj_rows_exact() {
    let be = backend();
    let mut rng = Rng(0x1a2b3c);
    let (d, ns) = (2304usize, [6552usize, 1056, 1056]);
    let n: usize = ns.iter().sum();
    let w = Ht::rand(DType::Bf16, &[n, d], 0.03, &mut rng);
    for s in [1usize, 3] {
        for r in [1usize, 2, 3, 4, 5, 7, 8, 9, 12, 16] {
            let x = Ht::rand(DType::Bf16, &[s, r, d], 1.0, &mut rng);
            let outs: Vec<Ht> = ns.iter().map(|&k| Ht::zeros(DType::Bf16, &[r, k])).collect();
            let multi = run(&be, INK_IN_PROJ_BF16_GEMV_ROWS, Op::InkInProj, &[x.clone(), w.clone()], &outs, &OpAttrs::default(), r > 1, None);
            for t in 0..r {
                let outs1: Vec<Ht> = ns.iter().map(|&k| Ht::zeros(DType::Bf16, &[1, k])).collect();
                let one = run(&be, INK_IN_PROJ_BF16_GEMV, Op::InkInProj, &[x.row(t), w.clone()], &outs1, &OpAttrs::default(), false, None);
                for j in 0..3 {
                    assert_eq!(out_row(&multi[j], &[r, ns[j]], 2, t), one[j], "S={s} R={r} row {t} segment {j}");
                }
            }
        }
    }
}

/// The e4m3 in-projection (lane ink-fp8): rows 1..16 bitwise the 1-row launch, one scale per segment, every compiled family.
#[test]
fn in_proj_f8_rows_exact() {
    let be = backend();
    let mut rng = Rng(0xf8f8);
    let (d, ns) = (2304usize, [6552usize, 1056, 1056]);
    let n: usize = ns.iter().sum();
    // random e4m3 codes without the NaN codes 0x7F / 0xFF
    let data: Vec<u8> = (0..n * d)
        .map(|_| {
            let b = ((rng.unit() * 0.5 + 0.5) * 255.0) as u8;
            if b & 0x7F == 0x7F { b ^ 1 } else { b }
        })
        .collect();
    let w = Ht { dt: DType::Fp8E4m3, shape: vec![n, d], data };
    let sc = Ht { dt: DType::F32, shape: vec![3], data: [1.3e-4f32, 4.1e-5, 2.2e-4].iter().flat_map(|v| v.to_le_bytes()).collect() };
    for s in [1usize, 3] {
        for r in [1usize, 2, 3, 4, 5, 7, 8, 9, 12, 16] {
            let x = Ht::rand(DType::Bf16, &[s, r, d], 1.0, &mut rng);
            let outs: Vec<Ht> = ns.iter().map(|&k| Ht::zeros(DType::Bf16, &[r, k])).collect();
            let multi = run(&be, INK_IN_PROJ_F8_GEMV_ROWS, Op::InkInProj, &[x.clone(), w.clone(), sc.clone()], &outs, &OpAttrs::default(), r > 1, None);
            for t in 0..r {
                let outs1: Vec<Ht> = ns.iter().map(|&k| Ht::zeros(DType::Bf16, &[1, k])).collect();
                let one = run(&be, INK_IN_PROJ_F8_GEMV, Op::InkInProj, &[x.row(t), w.clone(), sc.clone()], &outs1, &OpAttrs::default(), false, None);
                for j in 0..3 {
                    assert_eq!(out_row(&multi[j], &[r, ns[j]], 2, t), one[j], "S={s} R={r} row {t} segment {j}");
                }
            }
        }
    }
}

/// ConvMix / SconvRes case: (T, C, Rr, pos0, res, add, sum, N, Q, norm, blend streams).
#[allow(clippy::too_many_arguments)]
fn conv_case(be: &MetalBackend, rng: &mut Rng, op: Op, t: usize, c: usize, rr: usize, pos0: usize, epi: (bool, bool, bool), mix: (usize, usize, bool, usize), dev: bool) {
    const K: usize = 4;
    const G: usize = 64;
    let (res, add, sum) = epi;
    let (n, q, norm, s) = mix;
    let bf = DType::Bf16;
    let x = Ht::rand(bf, &[t, c], 1.0, rng);
    let taps = Ht::rand(bf, &[K, c], 0.5, rng);
    let resv = if res { Ht::rand(bf, &[t, c], 1.0, rng) } else { Ht::none() };
    let addv = if add { Ht::rand(bf, &[t, c], 1.0, rng) } else { Ht::none() };
    let ring0 = Ht::rand(bf, &[rr, c], 1.0, rng);
    let mut ins = vec![x, taps, resv, addv];
    let mut outs = vec![Ht::zeros(bf, &[t, c]), ring0.clone(), if sum { Ht::zeros(bf, &[t, c]) } else { Ht::none() }];
    let kernel = if op == Op::SconvRes { SCONV_RES_BF16 } else { CONV_MIX_BF16 };
    if op == Op::ConvMix {
        ins.push(Ht::rand(bf, &[n, t, c], 1.5, rng));
        ins.push(Ht::rand(DType::F32, &[q, c], 0.2, rng));
        let nrows = if s > 0 { s } else { q };
        ins.push(if norm || s > 0 { Ht::pos(&[nrows, c], rng) } else { Ht::none() });
        if s > 0 {
            ins.push(Ht::rand(DType::F32, &[s, t, c], 1.0, rng));
            ins.push(Ht::rand(bf, &[s, G, 2 * c], (2.0 * c as f32).powf(-0.5), rng));
            ins.push(Ht::rand(bf, &[s, c, G], 0.25, rng));
            ins.push(Ht::rand(bf, &[s, c, G], 0.25, rng));
        } else {
            ins.extend([Ht::none(), Ht::none(), Ht::none(), Ht::none()]);
        }
        outs.push(if q == 0 {
            Ht::none()
        } else if s > 0 {
            Ht::zeros(bf, &[s, t, c])
        } else if norm {
            Ht::zeros(bf, &[q, t, c])
        } else {
            Ht::zeros(DType::F32, &[q, t, c])
        });
    }
    let attrs = OpAttrs { pos0: if dev { 0 } else { pos0 }, eps: 1e-6, ..OpAttrs::default() };
    let dp = dev.then_some(pos0 as i32);
    let multi = run(be, kernel, op, &ins, &outs, &attrs, t > 1, dp);
    // the same rows one at a time, the ring carried over
    let mut ring = ring0;
    let mut rows: Vec<Vec<Vec<u8>>> = Vec::new();
    for ti in 0..t {
        let ins1: Vec<Ht> = ins
            .iter()
            .enumerate()
            .map(|(k, h)| if [1usize, 5, 6, 8, 9, 10].contains(&k) { h.clone() } else { h.row(ti) })
            .collect();
        let mut outs1: Vec<Ht> = outs.iter().enumerate().map(|(k, h)| if k == 1 { ring.clone() } else { h.row(ti) }).collect();
        outs1[1] = ring.clone();
        let a1 = OpAttrs { pos0: if dev { 0 } else { pos0 + ti }, ..attrs.clone() };
        let r1 = run(be, kernel, op, &ins1, &outs1, &a1, false, dev.then_some((pos0 + ti) as i32));
        ring.data = r1[1].clone();
        rows.push(r1);
    }
    let label = format!("{op} T={t} C={c} Rr={rr} pos0={pos0} epi={epi:?} mix={mix:?} dev={dev}");
    assert_eq!(multi[1], ring.data, "{label}: final ring");
    for (oi, o) in outs.iter().enumerate() {
        if oi == 1 {
            continue;
        }
        for (ti, r1) in rows.iter().enumerate() {
            assert_eq!(out_row(&multi[oi], &o.shape, o.dt.size_bytes(), ti), r1[oi], "{label}: output {oi} row {ti}");
        }
    }
}

#[test]
fn conv_mix_rows_exact() {
    let be = backend();
    let mut rng = Rng(0x5eed);
    // the model's ring (K-1 + 16) and a short one (K-1 + 4: rows 7.. of a call read rows it also writes -> deferred write)
    for &(c, rr) in &[(2304usize, 19usize), (64, 7)] {
        for &t in &[1usize, 2, 3, 7, 16, 40] {
            if c > 64 && t > 16 {
                continue;
            }
            for &pos0 in &[0usize, 2, 29] {
                conv_case(&be, &mut rng, Op::SconvRes, t, c, rr, pos0, (true, true, true), (0, 0, false, 0), false);
                for &(epi, mix) in &[
                    ((true, true, true), (4usize, 1usize, true, 0usize)),
                    ((true, false, false), (7, 3, true, 0)),
                    ((false, false, false), (2, 1, false, 0)),
                    ((true, false, true), (0, 0, false, 0)),
                    ((true, true, true), (3, 1, false, 3)),
                    ((true, false, false), (5, 3, false, 3)),
                    ((true, true, true), (2, 1, false, 1)),
                ] {
                    conv_case(&be, &mut rng, Op::ConvMix, t, c, rr, pos0, epi, mix, false);
                }
            }
        }
    }
    // a chunk longer than the blend's 256-row scratch chunk (two blend chunks, the h GEMV's full 16-row chunks and its tail)
    conv_case(&be, &mut rng, Op::ConvMix, 300, 64, 7, 2, (true, true, true), (3, 1, false, 3), false);
    conv_case(&be, &mut rng, Op::ConvMix, 300, 64, 7, 2, (true, false, false), (2, 3, false, 3), false);
    // the device position
    conv_case(&be, &mut rng, Op::ConvMix, 7, 2304, 19, 29, (true, true, true), (4, 1, true, 0), true);
    conv_case(&be, &mut rng, Op::ConvMix, 16, 64, 7, 3, (true, true, true), (3, 1, false, 3), true);
}

#[test]
fn attn_res_mix_and_loop_blend_rows_exact() {
    let be = backend();
    let mut rng = Rng(0xab5);
    let bf = DType::Bf16;
    for &d in &[2304usize, 64] {
        for &r in &[2usize, 5, 16, 40, 300] {
            if d > 64 && r > 16 {
                continue;
            }
            for &(n, q, add, norm) in &[(4usize, 1usize, true, true), (7, 3, false, true), (0, 4, true, false), (3, 6, false, false)] {
                let ins = vec![
                    Ht::rand(bf, &[n, r, d], 1.5, &mut rng),
                    Ht::rand(bf, &[r, d], 1.0, &mut rng),
                    if add { Ht::rand(bf, &[r, d], 1.0, &mut rng) } else { Ht::none() },
                    Ht::rand(DType::F32, &[q, d], 0.2, &mut rng),
                    if norm { Ht::pos(&[q, d], &mut rng) } else { Ht::none() },
                ];
                let out = if norm { Ht::zeros(bf, &[q, r, d]) } else { Ht::zeros(DType::F32, &[q, r, d]) };
                let a = OpAttrs { eps: 1e-6, ..OpAttrs::default() };
                let multi = run(&be, ATTN_RES_MIX_BF16, Op::AttnResMix, &ins, &[out.clone()], &a, true, None);
                for t in 0..r {
                    let ins1: Vec<Ht> = ins.iter().enumerate().map(|(k, h)| if k >= 3 { h.clone() } else { h.row(t) }).collect();
                    let one = run(&be, ATTN_RES_MIX_BF16, Op::AttnResMix, &ins1, &[out.row(t)], &a, false, None);
                    assert_eq!(out_row(&multi[0], &out.shape, out.dt.size_bytes(), t), one[0], "AttnResMix D={d} R={r} N={n} Q={q} row {t}");
                }
            }
            for &(s, sd) in &[(3usize, 1usize), (3, 3), (1, 1)] {
                let g = 64;
                let ins = vec![
                    Ht::rand(DType::F32, &[sd, r, d], 1.0, &mut rng),
                    Ht::rand(DType::F32, &[s, r, d], 1.0, &mut rng),
                    Ht::rand(bf, &[s, g, 2 * d], (2.0 * d as f32).powf(-0.5), &mut rng),
                    Ht::rand(bf, &[s, d, g], 0.25, &mut rng),
                    Ht::rand(bf, &[s, d, g], 0.25, &mut rng),
                    Ht::pos(&[s, d], &mut rng),
                ];
                let out = Ht::zeros(bf, &[s, r, d]);
                let a = OpAttrs { eps: 1e-6, ..OpAttrs::default() };
                let multi = run(&be, LOOP_BLEND_BF16, Op::LoopBlend, &ins, &[out.clone()], &a, true, None);
                for t in 0..r {
                    let ins1: Vec<Ht> = ins.iter().enumerate().map(|(k, h)| if k >= 2 { h.clone() } else { h.row(t) }).collect();
                    let one = run(&be, LOOP_BLEND_BF16, Op::LoopBlend, &ins1, &[out.row(t)], &a, false, None);
                    assert_eq!(out_row(&multi[0], &out.shape, 2, t), one[0], "LoopBlend D={d} R={r} S={s} Sd={sd} row {t}");
                }
            }
        }
    }
}
