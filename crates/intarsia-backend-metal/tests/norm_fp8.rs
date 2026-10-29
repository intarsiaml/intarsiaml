//! Lane mnq (docs/results/2026-10-29-mnq-metal-norm-fp8.md): the producers of an fp8 GEMV's input (`rmsnorm`, `rmsnorm_add`,
//! `silu_glu`, `sigmoid_mul`) write q(x) of their output (`prequant`), and the one-row fp8 GEMV reads it (`_m1_xd`). Every
//! launch sequence here is run twice on the same inputs, once with the handoff (producer, then the GEMVs back to back) and
//! once with a synchronize between producer and GEMV (the held producer is encoded plain, the GEMV stages x itself, which is
//! main's path), and the bytes of the producer outputs and of every GEMV output, guards included, must be equal.
//!
//! The norms hand off only at `INTARSIA_METAL_NORM_FP8=2` (run this file at 1 and at 2).
//!
//! The synthetic tests use the GPU: on endor take a fleetq slot first.

#![cfg(target_os = "macos")]

mod common;
use common::*;
use intarsia_backend_metal::bench::{f32_to_bf16, Rng};
use intarsia_backend_metal::{norm_fp8_enabled, norm_fp8_level, MetalBackend};
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::{DType, Op, OpAttrs, OpCall};

fn bytes_u16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn put(be: &MetalBackend, off: usize, data: &[u8]) -> Buffer {
    let b = be.alloc(off + data.len() + 16, MemKind::Device).unwrap();
    be.upload(&b, 0, &vec![0xEEu8; off + data.len() + 16], 0).unwrap();
    be.upload(&b, off, data, 0).unwrap();
    b
}

fn guarded(be: &MetalBackend, bytes: usize) -> Buffer {
    let b = be.alloc(bytes + 32, MemKind::Device).unwrap();
    be.upload(&b, 0, &vec![0xEEu8; bytes + 32], 0).unwrap();
    b
}

fn dl(be: &MetalBackend, b: &Buffer) -> Vec<u8> {
    be.synchronize(0).unwrap();
    let mut raw = vec![0u8; b.bytes()];
    be.download(b, 0, &mut raw, 0).unwrap();
    raw
}

fn tr(b: &Buffer, off: usize, dt: DType, shape: &[usize]) -> TensorRef {
    TensorRef { buf: b.clone(), offset: off, dtype: dt, shape: shape.to_vec() }
}

fn e4m3(n: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed | 1);
    (0..n)
        .map(|i| {
            let b = (r.next_u64() >> 32) as u8;
            if b & 0x7F == 0x7F {
                b & 0xFE
            } else if i % 4999 == 4998 {
                0x7F
            } else {
                b
            }
        })
        .collect()
}

/// x values of one row, by `kind`: random in [-1, 1), wide range, large (saturates after a small scale), small noise, one NaN.
fn row(k: usize, kind: usize, seed: u64) -> Vec<u16> {
    let mut r = Rng(seed | 1);
    (0..k)
        .map(|j| {
            let v = match kind % 5 {
                0 => r.unit(),
                1 => r.unit() * 2f32.powi((r.next_u64() % 25) as i32 - 12),
                2 => r.unit() * 400.0,
                3 => r.unit() * 1e-3,
                _ => {
                    if j == 17 {
                        f32::NAN
                    } else {
                        r.unit()
                    }
                }
            };
            f32_to_bf16(v)
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
enum Prod {
    Norm,
    NormAdd,
    Silu,
    Sigmoid,
}

fn sequence(be: &MetalBackend, prod: Prod, k: usize, kind: usize, x_off: usize, gemvs: &[(usize, bool, f32)]) {
    let base = be.norm_fp8_hits();
    // handoff run: producer then GEMVs with nothing between; the producer outputs are read after (a download synchronizes)
    let a = run(be, prod, k, kind, x_off, gemvs, true);
    let used = be.norm_fp8_hits();
    let b = run(be, prod, k, kind, x_off, gemvs, false);
    assert_eq!(be.norm_fp8_hits(), used, "{prod:?}: the reference run must not hand off");
    let claims = match prod {
        Prod::Silu | Prod::Sigmoid => norm_fp8_level() >= 1,
        Prod::Norm | Prod::NormAdd => norm_fp8_level() >= 2,
    };
    if claims && gemvs[0].2 > 0.0 {
        assert!(used.0 > base.0, "{prod:?} k={k}: no producer was claimed");
    }
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        if let Some(j) = x.iter().zip(y).position(|(p, q)| p != q) {
            panic!("{prod:?} k={k} kind={kind} x_off={x_off} gemvs={gemvs:?}: output {i} differs at byte {j}: handoff {:#04x} reference {:#04x}", x[j], y[j]);
        }
    }
}

/// One sequence: `prod`, then GEMVs (n, f32_out, input scale) on its output; `handoff` = nothing between them. Returns every byte
/// the sequence wrote: the producer's outputs and each GEMV's output (downloaded after all launches).
fn run(be: &MetalBackend, prod: Prod, k: usize, kind: usize, x_off: usize, gemvs: &[(usize, bool, f32)], handoff: bool) -> Vec<Vec<u8>> {
    let a = put(be, x_off, &bytes_u16(&row(k, kind, 11)));
    let b = put(be, 0, &bytes_u16(&row(k, kind + 1, 12)));
    let w = put(
        be,
        0,
        &bytes_u16(&row(k, 0, 13).iter().map(|v| f32_to_bf16(1.0 + 0.1 * intarsia_backend_metal::bench::bf16_to_f32(*v))).collect::<Vec<_>>()),
    );
    let attrs = OpAttrs { eps: 1e-6, w_offset: 0.0, ..Default::default() };
    let out = guarded(be, k * 2);
    let sum = guarded(be, k * 2);
    // the model's shapes: the gated norm's silu_glu works on [heads, dv] rows and the attention output is [1, heads, d]; the
    // GEMV reads either as [1, K]
    let ps: Vec<usize> = match (prod, kind % 2) {
        (Prod::Silu, 1) => vec![32, k / 32],
        (Prod::Sigmoid, 1) => vec![1, 16, k / 16],
        _ => vec![1, k],
    };
    let at = tr(&a, x_off, DType::Bf16, &ps);
    let bt = tr(&b, 0, DType::Bf16, &ps);
    let wt = tr(&w, 0, DType::Bf16, &[k]);
    let pt = tr(&out, 0, DType::Bf16, &ps);
    let ot = tr(&out, 0, DType::Bf16, &[1, k]);
    let st = tr(&sum, 0, DType::Bf16, &[1, k]);
    let (kernel, call) = match prod {
        Prod::Norm => ("rmsnorm_bf16", OpCall::new(Op::RmsNorm, vec![at, wt], vec![ot.clone()], attrs)),
        Prod::NormAdd => ("rmsnorm_add_bf16", OpCall::new(Op::RmsNormAdd, vec![at, bt, wt], vec![ot.clone(), st], attrs)),
        Prod::Silu => ("silu_glu_bf16", OpCall::new(Op::SiluGlu, vec![at, bt], vec![pt], attrs)),
        Prod::Sigmoid => ("sigmoid_mul_bf16", OpCall::new(Op::SigmoidMul, vec![at, bt], vec![pt], attrs)),
    };
    // inputs of the GEMVs first (uploads synchronize), then the producer and the GEMVs back to back
    let mut stuff = Vec::new();
    for (i, &(n, f32_out, is)) in gemvs.iter().enumerate() {
        let sc = put(be, 4, &[0.0123f32, is].iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<_>>());
        let wq = put(be, 8, &e4m3(n * k, 77 + i as u64));
        let (odt, esz) = if f32_out { (DType::F32, 4) } else { (DType::Bf16, 2) };
        let y = guarded(be, n * esz);
        let g = OpCall::new(
            Op::MatmulFp8,
            vec![ot.clone(), tr(&wq, 8, DType::Fp8E4m3, &[n, k]), tr(&sc, 4, DType::F32, &[2])],
            vec![tr(&y, 0, odt, &[1, n])],
            OpAttrs::default(),
        );
        stuff.push((g, y, sc, wq));
    }
    be.launch(kernel, &call, 0).unwrap();
    if !handoff {
        be.synchronize(0).unwrap();
    }
    for (g, ..) in &stuff {
        be.launch("matmul_fp8_gemv", g, 0).unwrap();
    }
    let mut outs = vec![dl(be, &out), dl(be, &sum)];
    for (_, y, ..) in &stuff {
        outs.push(dl(be, y));
    }
    outs
}

const SHAPES: &[(usize, usize)] = &[(2048, 8192), (2048, 130), (4096, 2048), (4096, 33), (512, 65), (96, 16), (8192, 512)];

fn all(prods: &[Prod]) {
    let be = backend();
    for &prod in prods {
        for &(k, n) in SHAPES {
            for kind in 0..5 {
                for (si, is) in [0.37f32, 0.0021, 8.0, 300.0].into_iter().enumerate() {
                    let x_off = if (kind + si) % 2 == 0 { 0 } else { 2 };
                    // two consumers with the scale the slot was made with, one with another scale (staged), one more equal
                    // (3 and 4: bf16 / f32 output)
                    let gemvs = [(n, kind % 2 == 0, is), (n.min(512), kind % 2 == 1, is), (n, false, is * 0.5), (130, true, is)];
                    sequence(&be, prod, k, kind, x_off, &gemvs);
                }
            }
        }
    }
}

#[test]
fn rmsnorm_handoff_is_bitwise_the_staged_gemv() {
    all(&[Prod::Norm]);
}

#[test]
fn rmsnorm_add_handoff_is_bitwise_the_staged_gemv() {
    all(&[Prod::NormAdd]);
}

#[test]
fn silu_glu_and_sigmoid_mul_handoff_are_bitwise_the_staged_gemv() {
    all(&[Prod::Silu, Prod::Sigmoid]);
}

/// A write to x between the producer and the GEMV (here a second producer's output on the same tensor) drops the slot.
#[test]
fn a_write_to_x_drops_the_slot() {
    let be = backend();
    let k = 2048;
    let a = put(&be, 0, &bytes_u16(&row(k, 0, 1)));
    let b = put(&be, 0, &bytes_u16(&row(k, 3, 2)));
    let out = guarded(&be, k * 2);
    let ot = tr(&out, 0, DType::Bf16, &[1, k]);
    let sc = put(&be, 4, &[0.0123f32, 0.37].iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<_>>());
    let wq = put(&be, 8, &e4m3(256 * k, 5));
    let ys = [guarded(&be, 512), guarded(&be, 512)];
    let g = |y: &Buffer| {
        OpCall::new(
            Op::MatmulFp8,
            vec![ot.clone(), tr(&wq, 8, DType::Fp8E4m3, &[256, k]), tr(&sc, 4, DType::F32, &[2])],
            vec![tr(y, 0, DType::Bf16, &[1, 256])],
            OpAttrs::default(),
        )
    };
    let silu = |p: &Buffer, q: &Buffer| OpCall::new(Op::SiluGlu, vec![tr(p, 0, DType::Bf16, &[1, k]), tr(q, 0, DType::Bf16, &[1, k])], vec![ot.clone()], OpAttrs::default());
    be.launch("silu_glu_bf16", &silu(&a, &b), 0).unwrap();
    be.launch("matmul_fp8_gemv", &g(&ys[0]), 0).unwrap();
    // x is rewritten (plain add into the same tensor): the next GEMV must see the new x
    let add = OpCall::new(Op::Add, vec![tr(&a, 0, DType::Bf16, &[1, k]), tr(&a, 0, DType::Bf16, &[1, k])], vec![ot.clone()], OpAttrs::default());
    be.launch("add_bf16", &add, 0).unwrap();
    be.launch("matmul_fp8_gemv", &g(&ys[1]), 0).unwrap();
    let got = dl(&be, &ys[1]);
    // reference: the same GEMV on the same x after a synchronize
    let yr = guarded(&be, 512);
    be.synchronize(0).unwrap();
    be.launch("matmul_fp8_gemv", &g(&yr), 0).unwrap();
    assert_eq!(got, dl(&be, &yr));
}

/// Timing (ignored; run once with `INTARSIA_METAL_NORM_FP8=0` and once without, in a `--quiet gpu` fleetq slot): µs per
/// `producer + GEMVs` sequence over weights rotated through 24 copies so they do not sit in the system cache.
#[test]
#[ignore]
fn timing_sequences() {
    let be = backend();
    let reps = 400usize;
    let copies = 24usize;
    for (label, prod, k, ns) in [
        ("norm_add alone", Prod::NormAdd, 2048usize, vec![]),
        ("norm_add+qkv", Prod::NormAdd, 2048, vec![8192]),
        ("norm_add+z", Prod::NormAdd, 2048, vec![4096]),
        ("norm_add+qkv+z (GDN in)", Prod::NormAdd, 2048, vec![8192, 4096]),
        ("norm_add+q+k+v (attn in)", Prod::NormAdd, 2048, vec![8192, 512, 512]),
        ("norm_add+tiny gemv (producer cost)", Prod::NormAdd, 2048, vec![16]),
        ("silu+out (GDN out)", Prod::Silu, 4096, vec![2048]),
        ("sigmoid+o_proj (attn out)", Prod::Sigmoid, 4096, vec![2048]),
    ] {
        let a = put(&be, 0, &bytes_u16(&row(k, 0, 11)));
        let b = put(&be, 0, &bytes_u16(&row(k, 1, 12)));
        let w = put(&be, 0, &bytes_u16(&vec![f32_to_bf16(1.0); k]));
        let attrs = OpAttrs { eps: 1e-6, w_offset: 0.0, ..Default::default() };
        let (out, sum) = (guarded(&be, k * 2), guarded(&be, k * 2));
        let (at, bt, wt) = (tr(&a, 0, DType::Bf16, &[1, k]), tr(&b, 0, DType::Bf16, &[1, k]), tr(&w, 0, DType::Bf16, &[k]));
        let (ot, st) = (tr(&out, 0, DType::Bf16, &[1, k]), tr(&sum, 0, DType::Bf16, &[1, k]));
        let (kernel, call) = match prod {
            Prod::NormAdd => ("rmsnorm_add_bf16", OpCall::new(Op::RmsNormAdd, vec![at, bt, wt], vec![ot.clone(), st], attrs)),
            Prod::Silu => ("silu_glu_bf16", OpCall::new(Op::SiluGlu, vec![at, bt], vec![ot.clone()], attrs)),
            _ => ("sigmoid_mul_bf16", OpCall::new(Op::SigmoidMul, vec![at, bt], vec![ot.clone()], attrs)),
        };
        let sc = put(&be, 4, &[0.0123f32, 0.37].iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<_>>());
        let mut gs: Vec<Vec<(OpCall, Buffer, Buffer)>> = Vec::new();
        for c in 0..copies {
            let mut v = Vec::new();
            for (i, &n) in ns.iter().enumerate() {
                let wq = put(&be, 8, &e4m3(n * k, 100 + (c * 7 + i) as u64));
                let y = guarded(&be, n * 2);
                let g = OpCall::new(
                    Op::MatmulFp8,
                    vec![ot.clone(), tr(&wq, 8, DType::Fp8E4m3, &[n, k]), tr(&sc, 4, DType::F32, &[2])],
                    vec![tr(&y, 0, DType::Bf16, &[1, n])],
                    OpAttrs::default(),
                );
                v.push((g, y, wq));
            }
            gs.push(v);
        }
        let mut best = f64::MAX;
        for round in 0..5 {
            let t0 = std::time::Instant::now();
            for r in 0..reps {
                be.launch(kernel, &call, 0).unwrap();
                for (g, ..) in &gs[r % copies] {
                    be.launch("matmul_fp8_gemv", g, 0).unwrap();
                }
            }
            be.synchronize(0).unwrap();
            let us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
            if round > 0 {
                best = best.min(us);
            }
        }
        eprintln!("TIMING {label}: {best:.2} us/seq (handoff on: {})", norm_fp8_enabled());
    }
}

/// Timing (ignored): the GEMV alone, x produced once (so every later launch reuses the slot with the handoff on, and stages
/// without it), weights rotated through 24 copies.
#[test]
#[ignore]
fn timing_gemv_only() {
    let be = backend();
    let reps = 600usize;
    for (n, k) in [(8192usize, 2048usize), (4096, 2048), (512, 2048), (2048, 4096)] {
        let a = put(&be, 0, &bytes_u16(&row(k, 0, 11)));
        let b = put(&be, 0, &bytes_u16(&row(k, 1, 12)));
        let out = guarded(&be, k * 2);
        let ot = tr(&out, 0, DType::Bf16, &[1, k]);
        let silu = OpCall::new(Op::SiluGlu, vec![tr(&a, 0, DType::Bf16, &[1, k]), tr(&b, 0, DType::Bf16, &[1, k])], vec![ot.clone()], OpAttrs::default());
        let sc = put(&be, 4, &[0.0123f32, 0.37].iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<_>>());
        let mut gs = Vec::new();
        for c in 0..24 {
            let wq = put(&be, 8, &e4m3(n * k, 100 + c as u64));
            let y = guarded(&be, n * 2);
            let g = OpCall::new(
                Op::MatmulFp8,
                vec![ot.clone(), tr(&wq, 8, DType::Fp8E4m3, &[n, k]), tr(&sc, 4, DType::F32, &[2])],
                vec![tr(&y, 0, DType::Bf16, &[1, n])],
                OpAttrs::default(),
            );
            gs.push((g, y, wq));
        }
        let mut best = f64::MAX;
        for round in 0..5 {
            be.launch("silu_glu_bf16", &silu, 0).unwrap();
            let t0 = std::time::Instant::now();
            for r in 0..reps {
                be.launch("matmul_fp8_gemv", &gs[r % 24].0, 0).unwrap();
            }
            be.synchronize(0).unwrap();
            let us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
            if round > 0 {
                best = best.min(us);
            }
        }
        eprintln!("TIMING gemv {n}x{k}: {best:.2} us/gemv (handoff on: {})", norm_fp8_enabled());
    }
}
