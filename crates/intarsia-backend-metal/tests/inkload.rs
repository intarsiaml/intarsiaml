#![cfg(target_os = "macos")]
//! Lane ink-dfuse: the load-order variants of the Inkling-2 row kernels are bitwise the kernels they replace.
//! `INTARSIA_INK_MIX_HOIST` (`ink_mix_row_h_*`: every operand load before its first use) on vs off for SconvRes, ConvMix (with
//! and without the blend) and AttnResMix over the widths of every compiled NE, taps K 1..5 (K > 4 keeps the loop), rows 1..16,
//! the model's ring and a short one, host and device positions, one and two query batches; `INTARSIA_INK_BLEND_NORM_C`
//! (`ink_blend_norm_c`, one channel per thread) on vs off for LoopBlend and the ConvMix blend. Rows-exact with the switches
//! on is tests/inkmix.rs run with the variables set.

mod common;
use common::*;
use intarsia_backend_metal::bench::{f32_to_bf16, Rng};
use intarsia_backend_metal::inkctrl::{set_blend_tail, BLEND_NORM_C, MIX_HOIST};
use intarsia_backend_metal::inkmix::*;
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};

/// The switches are process-wide: one case at a time.
static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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

/// Run `kernel` on fresh copies of `ins` / `outs` with the switches set (hoist, norm_c); every output's bytes.
#[allow(clippy::too_many_arguments)]
fn run(be: &MetalBackend, kernel: &str, op: Op, ins: &[Ht], outs: &[Ht], attrs: &OpAttrs, dev_pos: Option<i32>, sw: (bool, bool)) -> Vec<Vec<u8>> {
    MIX_HOIST.set(Some(sw.0));
    BLEND_NORM_C.set(Some(sw.1));
    set_blend_tail(Some(false));
    let it: Vec<TensorRef> = ins.iter().map(|h| h.up(be)).collect();
    let ot: Vec<TensorRef> = outs.iter().map(|h| h.up(be)).collect();
    let mut call = OpCall::new(op, it, ot.clone(), attrs.clone());
    call.exact_rows = ins[0].shape.len() == 2 && ins[0].shape[0] > 1;
    if let Some(p) = dev_pos {
        let pb = be.alloc(16, MemKind::Device).unwrap();
        be.upload(&pb, 0, &p.to_le_bytes(), 0).unwrap();
        call.pos = Some(TensorRef { buf: pb, offset: 0, dtype: DType::I32, shape: vec![1] });
    }
    call.check().unwrap();
    let n = blend_scratch(&ShapeInfo::from_call(&call));
    let b = be.alloc(n.max(16), MemKind::Device).unwrap();
    call.scratch = Some(TensorRef { buf: b, offset: 0, dtype: DType::U8, shape: vec![n.max(16)] });
    be.launch(kernel, &call, 0).unwrap();
    be.synchronize(0).unwrap();
    MIX_HOIST.set(None);
    BLEND_NORM_C.set(None);
    set_blend_tail(None);
    ot.iter().map(|t| down(be, t)).collect()
}

/// A ConvMix / SconvRes case: (inputs, outputs) for T rows of width C, K taps, ring Rr, (res, add, sum), (N, Q, norm, S).
fn conv_io(rng: &mut Rng, op: Op, t: usize, c: usize, k: usize, rr: usize, epi: (bool, bool, bool), mix: (usize, usize, bool, usize)) -> (Vec<Ht>, Vec<Ht>) {
    const G: usize = 64;
    let bf = DType::Bf16;
    let (res, add, sum) = epi;
    let (n, q, norm, s) = mix;
    let mut ins = vec![
        Ht::rand(bf, &[t, c], 1.0, rng),
        Ht::rand(bf, &[k, c], 0.5, rng),
        if res { Ht::rand(bf, &[t, c], 1.0, rng) } else { Ht::none() },
        if add { Ht::rand(bf, &[t, c], 1.0, rng) } else { Ht::none() },
    ];
    let mut outs = vec![Ht::zeros(bf, &[t, c]), Ht::rand(bf, &[rr, c], 1.0, rng), if sum { Ht::zeros(bf, &[t, c]) } else { Ht::none() }];
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
    (ins, outs)
}

/// The HO instances admit the thread counts the row kernels launch (D 2304: NE 4 x 576 threads since ink/metal's mix_geom
/// change; NE 2 below D 2048: up to 1024), else the switch would fall back to the plain instance and the bitwise test below
/// would compare a kernel with itself.
#[test]
fn mix_hoist_instances_admit_their_rows() {
    use objc2_metal::MTLComputePipelineState;
    let be = backend();
    for ne in [1usize, 2, 3, 4] {
        for qb in [1usize, 4] {
            for ns in 1..=8usize {
                let (h, p) = (format!("ink_mix_row_h_e{ne}n{ns}q{qb}"), format!("ink_mix_row_e{ne}n{ns}q{qb}"));
                let mh = be.pso(&h).unwrap_or_else(|| panic!("no pipeline {h}")).maxTotalThreadsPerThreadgroup();
                let mp = be.pso(&p).unwrap().maxTotalThreadsPerThreadgroup();
                eprintln!("{h}: max threads {mh} (plain {mp}){}", if mh < mp { " (falls back above this)" } else { "" });
                // the model's rows (D 2304: NE 4 x 576 threads) must take it
                assert!(ne != 4 || mh >= 576, "{h}: max threads {mh}, the model's rows need 576");
            }
        }
    }
}

#[test]
fn mix_hoist_is_bitwise() {
    let be = backend();
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut rng = Rng(0x4015);
    let mut n = 0;
    // widths of NE 4 (the model, D 2304), 1, 4 (full) and 2; K = 4 (the model), shorter, and 5 (the loop)
    for &c in &[2304usize, 1000, 4096, 1500] {
        for &k in &[4usize, 2, 1, 5] {
            for &t in &[1usize, 2, 3, 4, 7, 16] {
                for &(rr, pos0, dev) in &[(k - 1 + 16, 0usize, false), (k - 1 + 16, 29, true), (k - 1 + 4, 2, false), (k - 1 + 4, 29, false)] {
                    if c != 2304 && !(t == 1 || t == 16) {
                        continue;
                    }
                    let a = OpAttrs { pos0: if dev { 0 } else { pos0 }, eps: 1e-6, ..OpAttrs::default() };
                    let dp = dev.then_some(pos0 as i32);
                    let (ins, outs) = conv_io(&mut rng, Op::SconvRes, t, c, k, rr, (true, true, true), (0, 0, false, 0));
                    let off = run(&be, SCONV_RES_BF16, Op::SconvRes, &ins, &outs, &a, dp, (false, false));
                    let on = run(&be, SCONV_RES_BF16, Op::SconvRes, &ins, &outs, &a, dp, (true, false));
                    assert_eq!(on, off, "SconvRes C={c} K={k} T={t} Rr={rr} pos0={pos0} dev={dev}");
                    n += 1;
                    for &(epi, mix) in &[
                        ((true, true, true), (4usize, 1usize, true, 0usize)),
                        ((true, false, false), (7, 3, true, 0)),
                        ((false, false, false), (2, 1, false, 0)),
                        ((true, false, false), (3, 6, true, 0)),
                        ((true, false, false), (1, 6, false, 0)),
                        ((true, false, true), (0, 0, false, 0)),
                        ((true, true, true), (3, 1, false, 3)),
                        ((true, false, false), (5, 3, false, 3)),
                        ((true, true, true), (2, 1, false, 1)),
                    ] {
                        let (ins, outs) = conv_io(&mut rng, Op::ConvMix, t, c, k, rr, epi, mix);
                        let off = run(&be, CONV_MIX_BF16, Op::ConvMix, &ins, &outs, &a, dp, (false, false));
                        let on = run(&be, CONV_MIX_BF16, Op::ConvMix, &ins, &outs, &a, dp, (true, false));
                        assert_eq!(on, off, "ConvMix C={c} K={k} T={t} Rr={rr} pos0={pos0} dev={dev} epi={epi:?} mix={mix:?}");
                        n += 1;
                    }
                }
            }
        }
    }
    // AttnResMix: one batch, a partial second batch, the per-pass message call (Q 48)
    let bf = DType::Bf16;
    for &d in &[2304usize, 1000, 4096] {
        for &r in &[1usize, 2, 4, 16] {
            for &(ns, q, add, norm) in &[(4usize, 1usize, true, true), (7, 3, false, true), (0, 4, true, false), (3, 6, false, false), (1, 48, false, false)] {
                let ins = vec![
                    Ht::rand(bf, &[ns, r, d], 1.5, &mut rng),
                    Ht::rand(bf, &[r, d], 1.0, &mut rng),
                    if add { Ht::rand(bf, &[r, d], 1.0, &mut rng) } else { Ht::none() },
                    Ht::rand(DType::F32, &[q, d], 0.2, &mut rng),
                    if norm { Ht::pos(&[q, d], &mut rng) } else { Ht::none() },
                ];
                let out = if norm { Ht::zeros(bf, &[q, r, d]) } else { Ht::zeros(DType::F32, &[q, r, d]) };
                let a = OpAttrs { eps: 1e-6, ..OpAttrs::default() };
                let off = run(&be, ATTN_RES_MIX_BF16, Op::AttnResMix, &ins, &[out.clone()], &a, None, (false, false));
                let on = run(&be, ATTN_RES_MIX_BF16, Op::AttnResMix, &ins, &[out.clone()], &a, None, (true, false));
                assert_eq!(on, off, "AttnResMix D={d} R={r} N={ns} Q={q} add={add} norm={norm}");
                n += 1;
            }
        }
    }
    eprintln!("{n} cases: ink_mix_row_h is bitwise ink_mix_row");
}

#[test]
fn blend_norm_c_is_bitwise() {
    let be = backend();
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut rng = Rng(0xb1e0);
    let (bf, f) = (DType::Bf16, DType::F32);
    let mut n = 0;
    for &(d, g) in &[(2304usize, 64usize), (200, 64), (256, 8), (320, 128), (1000, 16)] {
        for &r in &[1usize, 2, 3, 4, 7, 16, 40] {
            for &(s, sd) in &[(3usize, 1usize), (3, 3), (1, 1)] {
                let ins = vec![
                    Ht::rand(f, &[sd, r, d], 1.0, &mut rng),
                    Ht::rand(f, &[s, r, d], 1.0, &mut rng),
                    Ht::rand(bf, &[s, g, 2 * d], (2.0 * d as f32).powf(-0.5), &mut rng),
                    Ht::rand(bf, &[s, d, g], 0.25, &mut rng),
                    Ht::rand(bf, &[s, d, g], 0.25, &mut rng),
                    Ht::pos(&[s, d], &mut rng),
                ];
                let outs = vec![Ht::zeros(bf, &[s, r, d])];
                let a = OpAttrs { eps: 1e-6, ..OpAttrs::default() };
                let off = run(&be, LOOP_BLEND_BF16, Op::LoopBlend, &ins, &outs, &a, None, (false, false));
                let on = run(&be, LOOP_BLEND_BF16, Op::LoopBlend, &ins, &outs, &a, None, (false, true));
                assert_eq!(on, off, "LoopBlend D={d} G={g} R={r} S={s} Sd={sd}");
                n += 1;
            }
            if g != 64 {
                continue;
            }
            for &mix in &[(3usize, 1usize, false, 3usize), (5, 3, false, 3), (2, 1, false, 1)] {
                let (ins, outs) = conv_io(&mut rng, Op::ConvMix, r, d, 4, 19, (true, true, true), mix);
                let a = OpAttrs { pos0: 5, eps: 1e-6, ..OpAttrs::default() };
                let off = run(&be, CONV_MIX_BF16, Op::ConvMix, &ins, &outs, &a, None, (false, false));
                let on = run(&be, CONV_MIX_BF16, Op::ConvMix, &ins, &outs, &a, None, (false, true));
                let both = run(&be, CONV_MIX_BF16, Op::ConvMix, &ins, &outs, &a, None, (true, true));
                assert_eq!(on, off, "ConvMix blend D={d} R={r} mix={mix:?}: norm_c");
                assert_eq!(both, off, "ConvMix blend D={d} R={r} mix={mix:?}: hoist + norm_c");
                n += 1;
            }
        }
    }
    eprintln!("{n} cases: ink_blend_norm_c is bitwise ink_blend_norm");
}
