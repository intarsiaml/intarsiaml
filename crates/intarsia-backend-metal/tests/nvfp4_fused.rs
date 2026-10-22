//! `MatmulNvfp4Glu` (`matmul_nvfp4_glu_gemv`) and `SharedDownCombine` (`shared_down_combine_gemv`) against the unfused
//! launches they replace, BYTE for byte: `matmul_nvfp4_gemv` x 2 + `silu_glu_bf16`, and `matmul_nvfp4_gemv` +
//! `moe_combine_bf16`. The whole output buffer (guard bytes included) must be identical, at the real 35B shared-expert shapes,
//! at odd N and K (multiples of 16), with weights at 8 mod 16, block scales / x / outputs at odd byte offsets, random global
//! scales and random routed values / gate logits (large and small ones to exercise the sigmoid and the silu tails).

#![cfg(target_os = "macos")]

mod common;

use common::*;
use intarsia_backend_metal::bench::{f32_to_bf16, Rng};
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};

const GLU: &str = "matmul_nvfp4_glu_gemv";
const DOWN: &str = "shared_down_combine_gemv";

fn upload_bytes(be: &MetalBackend, off: usize, data: &[u8]) -> Buffer {
    let b = be.alloc(off + data.len() + 16, MemKind::Device).unwrap();
    be.upload(&b, 0, &vec![0xEEu8; off + data.len() + 16], 0).unwrap();
    be.upload(&b, off, data, 0).unwrap();
    b
}

fn t(buf: &Buffer, off: usize, dt: DType, shape: &[usize]) -> TensorRef {
    TensorRef { buf: buf.clone(), offset: off, dtype: dt, shape: shape.to_vec() }
}

fn bf16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn rand_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed | 1);
    (0..n).map(|_| (r.next_u64() >> 32) as u8).collect()
}

/// Block scales of moderate magnitude (e4m3 exponents 4..=11, any sign and mantissa; never the NaN code).
fn rand_block_scales(n: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed | 1);
    (0..n)
        .map(|_| {
            let v = (r.next_u64() >> 32) as u8;
            (v & 0x87) | (((v >> 3) % 8 + 4) << 3)
        })
        .collect()
}

fn alloc_guarded(be: &MetalBackend, size: usize) -> Buffer {
    let b = be.alloc(size, MemKind::Device).unwrap();
    be.upload(&b, 0, &vec![0xEEu8; size], 0).unwrap();
    b
}

fn download(be: &MetalBackend, b: &Buffer, size: usize) -> Vec<u8> {
    let mut v = vec![0u8; size];
    be.download(b, 0, &mut v, 0).unwrap();
    v
}

fn accepts(be: &MetalBackend, kernel: &str, cl: &OpCall) -> bool {
    be.kernels().iter().find(|k| k.name == kernel).unwrap().accepts(&ShapeInfo::from_call(cl))
}

/// Byte offsets of the bound tensors (all inside larger guarded buffers).
#[derive(Clone, Copy)]
struct Offs {
    x: usize,
    w: usize,
    bs: usize,
    gs: usize,
    out: usize,
    routed: usize,
    logit: usize,
}

const ALIGNED: Offs = Offs { x: 0, w: 8, bs: 0, gs: 4, out: 0, routed: 0, logit: 0 };
const SKEWED: Offs = Offs { x: 2, w: 16, bs: 1, gs: 8, out: 2, routed: 6, logit: 2 };
const ODD: Offs = Offs { x: 6, w: 24, bs: 3, gs: 12, out: 10, routed: 2, logit: 6 };

/// A random global scale in [lo, hi).
fn rand_gs(seed: u64, lo: f32, hi: f32) -> f32 {
    let mut r = Rng(seed | 1);
    lo + (r.unit().abs() * (hi - lo))
}

struct Weights {
    w: Vec<u8>,
    bs: Vec<u8>,
    gs: f32,
}

fn weights(n: usize, k: usize, seed: u64, gs: f32) -> Weights {
    Weights { w: rand_bytes(n * k / 2, seed), bs: rand_block_scales(n * k / 16, seed + 1), gs }
}

/// Gate logits that exercise the sigmoid: tiny, moderate, saturated both ways, zero, negative zero.
const LOGITS: &[f32] =
    &[0.0, -0.0, 1e-3, -1e-3, 0.5, -0.5, 3.0, -3.0, 8.0, -8.0, 20.0, -20.0, 88.0, -88.0, 200.0, -200.0];

fn gemv_call(x: TensorRef, wb: (&Buffer, &Buffer, &Buffer), o: Offs, n: usize, k: usize, out: TensorRef) -> OpCall {
    OpCall::new(
        Op::MatmulNvfp4,
        vec![
            x,
            t(wb.0, o.w, DType::U8, &[n, k / 2]),
            t(wb.1, o.bs, DType::Fp8E4m3, &[n, k / 16]),
            t(wb.2, o.gs, DType::F32, &[1]),
        ],
        vec![out],
        OpAttrs::default(),
    )
}

fn upload_weights(be: &MetalBackend, w: &Weights, o: Offs) -> (Buffer, Buffer, Buffer) {
    (upload_bytes(be, o.w, &w.w), upload_bytes(be, o.bs, &w.bs), upload_bytes(be, o.gs, &w.gs.to_le_bytes()))
}

fn glu_case(be: &MetalBackend, m: usize, n: usize, k: usize, o: Offs, gs: (f32, f32), seed: u64) {
    let what = format!("glu m={m} n={n} k={k} gs={gs:?} seed={seed}");
    let x = rand_bf16(m * k, seed);
    let xb = upload_bytes(be, o.x, &bf16_bytes(&x));
    let (wg, wu) = (weights(n, k, seed + 10, gs.0), weights(n, k, seed + 20, gs.1));
    let (gb, ub) = (upload_weights(be, &wg, o), upload_weights(be, &wu, o));
    let osz = o.out + m * n * 2 + 32;

    // fused
    let fo = alloc_guarded(be, osz);
    let cl = OpCall::new(
        Op::MatmulNvfp4Glu,
        vec![
            t(&xb, o.x, DType::Bf16, &[m, k]),
            t(&gb.0, o.w, DType::U8, &[n, k / 2]),
            t(&gb.1, o.bs, DType::Fp8E4m3, &[n, k / 16]),
            t(&gb.2, o.gs, DType::F32, &[1]),
            t(&ub.0, o.w, DType::U8, &[n, k / 2]),
            t(&ub.1, o.bs, DType::Fp8E4m3, &[n, k / 16]),
            t(&ub.2, o.gs, DType::F32, &[1]),
        ],
        vec![t(&fo, o.out, DType::Bf16, &[m, n])],
        OpAttrs::default(),
    );
    cl.check().unwrap();
    assert!(accepts(be, GLU, &cl), "{what}: not accepted");
    be.launch(GLU, &cl, 0).unwrap();

    // unfused: gemv(gate) -> g, gemv(up) -> u (both bf16, offset 2), silu_glu_bf16
    let (gt, ut) = (alloc_guarded(be, m * n * 2 + 32), alloc_guarded(be, m * n * 2 + 32));
    let uo = alloc_guarded(be, osz);
    let xr = || t(&xb, o.x, DType::Bf16, &[m, k]);
    be.launch(
        "matmul_nvfp4_gemv",
        &gemv_call(xr(), (&gb.0, &gb.1, &gb.2), o, n, k, t(&gt, 2, DType::Bf16, &[m, n])),
        0,
    )
    .unwrap();
    be.launch(
        "matmul_nvfp4_gemv",
        &gemv_call(xr(), (&ub.0, &ub.1, &ub.2), o, n, k, t(&ut, 2, DType::Bf16, &[m, n])),
        0,
    )
    .unwrap();
    let silu = OpCall::new(
        Op::SiluGlu,
        vec![t(&gt, 2, DType::Bf16, &[m, n]), t(&ut, 2, DType::Bf16, &[m, n])],
        vec![t(&uo, o.out, DType::Bf16, &[m, n])],
        OpAttrs::default(),
    );
    be.launch("silu_glu_bf16", &silu, 0).unwrap();
    be.synchronize(0).unwrap();

    let (a, b) = (download(be, &fo, osz), download(be, &uo, osz));
    if a != b {
        let i = a.iter().zip(&b).position(|(p, q)| p != q).unwrap();
        panic!(
            "{what}: fused and unfused outputs differ at byte {i} (fused {:#04x}, unfused {:#04x}), {} bytes differ",
            a[i],
            b[i],
            a.iter().zip(&b).filter(|(p, q)| p != q).count()
        );
    }
    // the guards around the output are untouched
    assert!(
        a[..o.out].iter().all(|v| *v == 0xEE) && a[o.out + m * n * 2..].iter().all(|v| *v == 0xEE),
        "{what}: stray stores"
    );
    eprintln!("{what}: identical ({} bytes)", a.len());
}

fn down_case(be: &MetalBackend, m: usize, n: usize, k: usize, o: Offs, gs: f32, logit: f32, seed: u64) {
    let what = format!("down m={m} n={n} k={k} gs={gs} logit={logit} seed={seed}");
    let a = rand_bf16(m * k, seed);
    let routed = rand_bf16(m * n, seed + 5);
    let ab = upload_bytes(be, o.x, &bf16_bytes(&a));
    let rb = upload_bytes(be, o.routed, &bf16_bytes(&routed));
    // row r gets the logit rotated through LOGITS from `logit`
    let li = LOGITS.iter().position(|l| l.to_bits() == logit.to_bits()).unwrap();
    let logits: Vec<u16> = (0..m).map(|r| f32_to_bf16(LOGITS[(li + r) % LOGITS.len()])).collect();
    let lb = upload_bytes(be, o.logit, &bf16_bytes(&logits));
    let w = weights(n, k, seed + 10, gs);
    let wb = upload_weights(be, &w, o);
    let osz = o.out + m * n * 2 + 32;

    let fo = alloc_guarded(be, osz);
    let cl = OpCall::new(
        Op::SharedDownCombine,
        vec![
            t(&ab, o.x, DType::Bf16, &[m, k]),
            t(&wb.0, o.w, DType::U8, &[n, k / 2]),
            t(&wb.1, o.bs, DType::Fp8E4m3, &[n, k / 16]),
            t(&wb.2, o.gs, DType::F32, &[1]),
            t(&rb, o.routed, DType::Bf16, &[m, n]),
            t(&lb, o.logit, DType::Bf16, &[m, 1]),
        ],
        vec![t(&fo, o.out, DType::Bf16, &[m, n])],
        OpAttrs::default(),
    );
    cl.check().unwrap();
    assert!(accepts(be, DOWN, &cl), "{what}: not accepted");
    be.launch(DOWN, &cl, 0).unwrap();

    let so = alloc_guarded(be, m * n * 2 + 32);
    let uo = alloc_guarded(be, osz);
    be.launch(
        "matmul_nvfp4_gemv",
        &gemv_call(t(&ab, o.x, DType::Bf16, &[m, k]), (&wb.0, &wb.1, &wb.2), o, n, k, t(&so, 2, DType::Bf16, &[m, n])),
        0,
    )
    .unwrap();
    let comb = OpCall::new(
        Op::MoeCombine,
        vec![
            t(&rb, o.routed, DType::Bf16, &[m, n]),
            t(&so, 2, DType::Bf16, &[m, n]),
            t(&lb, o.logit, DType::Bf16, &[m, 1]),
        ],
        vec![t(&uo, o.out, DType::Bf16, &[m, n])],
        OpAttrs::default(),
    );
    be.launch("moe_combine_bf16", &comb, 0).unwrap();
    be.synchronize(0).unwrap();

    let (fa, ub) = (download(be, &fo, osz), download(be, &uo, osz));
    if fa != ub {
        let i = fa.iter().zip(&ub).position(|(p, q)| p != q).unwrap();
        panic!(
            "{what}: fused and unfused outputs differ at byte {i} (fused {:#04x}, unfused {:#04x}), {} bytes differ",
            fa[i],
            ub[i],
            fa.iter().zip(&ub).filter(|(p, q)| p != q).count()
        );
    }
    assert!(
        fa[..o.out].iter().all(|v| *v == 0xEE) && fa[o.out + m * n * 2..].iter().all(|v| *v == 0xEE),
        "{what}: stray stores"
    );
}

/// (n, k): the 35B shared expert's gate / up (512 x 2048) and down (2048 x 512) shapes, odd N (not a multiple of the 32 rows
/// of a threadgroup, nor of the 4 of a simdgroup) and odd K (multiples of 16, not of 512: lanes own unequal block counts).
const GLU_SHAPES: &[(usize, usize)] =
    &[(512, 2048), (2048, 512), (40, 48), (33, 16), (17, 80), (130, 112), (1, 16), (3, 2064), (1001, 528)];

/// Down projection shapes (n, k): the real 2048 x 512, and odd ones as above (K a multiple of 16).
const DOWN_SHAPES: &[(usize, usize)] =
    &[(2048, 512), (512, 2048), (48, 48), (16, 32), (80, 16), (112, 128), (1001, 528), (3, 64), (2064, 1008)];

#[test]
fn glu_is_byte_identical_to_the_unfused_chain() {
    let be = backend();
    for (ci, &(n, k)) in GLU_SHAPES.iter().enumerate() {
        for (oi, o) in [ALIGNED, SKEWED, ODD].into_iter().enumerate() {
            let seed = 100 + ci as u64 * 31 + oi as u64 * 7;
            let g = (rand_gs(seed, 0.0003, 0.002), rand_gs(seed + 1, 0.0003, 0.002));
            for m in [1usize, 2, 3, 8] {
                glu_case(&be, m, n, k, o, g, seed + m as u64);
            }
        }
        // larger global scales: saturated silu (gate far in the tails), different scale per projection
        for (gi, gs) in [(0.05f32, 0.9f32), (3.0, 0.0001), (40.0, 40.0), (0.0, 1.0)].into_iter().enumerate() {
            for m in [1usize, 5] {
                glu_case(&be, m, n, k, SKEWED, gs, 900 + ci as u64 * 13 + gi as u64 + m as u64);
            }
        }
    }
}

#[test]
fn down_combine_is_byte_identical_to_the_unfused_chain() {
    let be = backend();
    for (ci, &(n, k)) in DOWN_SHAPES.iter().enumerate() {
        for (oi, o) in [ALIGNED, SKEWED, ODD].into_iter().enumerate() {
            for (li, &lg) in LOGITS.iter().enumerate() {
                let seed = 2000 + ci as u64 * 97 + oi as u64 * 17 + li as u64;
                for m in [1usize, 2, 8] {
                    down_case(&be, m, n, k, o, rand_gs(seed, 0.0003, 0.05), lg, seed + m as u64);
                }
            }
        }
    }
    // the real shape (down 2048 x 512), large global scales (saturated shared output)
    for (gi, gs) in [0.00071f32, 0.3, 5.0, 100.0].into_iter().enumerate() {
        for (li, &lg) in LOGITS.iter().enumerate() {
            for m in [1usize, 3] {
                down_case(&be, m, 2048, 512, SKEWED, gs, lg, 7000 + gi as u64 * 50 + li as u64 + m as u64);
            }
        }
    }
}

/// M = 1..=8 is registered; weights the plain GEMV refuses are refused here too.
#[test]
fn registration_and_refusals() {
    let be = backend();
    let ks = be.kernels();
    for (name, op) in [(GLU, Op::MatmulNvfp4Glu), (DOWN, Op::SharedDownCombine)] {
        let k = ks.iter().find(|k| k.name == name).unwrap_or_else(|| panic!("{name} not registered"));
        assert_eq!(k.op, op);
        let sh = |m: usize, kk: usize| ShapeInfo {
            m,
            n: 512,
            k: kk,
            act: Some(DType::Bf16),
            out: Some(DType::Bf16),
            weight: Some(DType::U8),
            ..Default::default()
        };
        assert!(k.accepts(&sh(1, 2048)) && k.accepts(&sh(2, 2048)) && k.accepts(&sh(8, 2048)));
        assert!(!k.accepts(&sh(9, 2048)), "{name}: M = 9");
        assert!(!k.accepts(&sh(1, 2040)), "{name}: K % 16 != 0");
    }
    assert!(!ks.iter().any(|k| k.op == Op::MoeBlock));

    // weights at a 4-byte offset
    let (n, k) = (16usize, 32usize);
    let x = upload_bytes(&be, 0, &bf16_bytes(&rand_bf16(k, 1)));
    let bs = upload_bytes(&be, 0, &rand_block_scales(n * k / 16, 3));
    let gs = upload_bytes(&be, 0, &1.0f32.to_le_bytes());
    let routed = upload_bytes(&be, 0, &bf16_bytes(&rand_bf16(n, 2)));
    let logit = upload_bytes(&be, 0, &bf16_bytes(&[f32_to_bf16(0.5)]));
    let y = be.alloc(n * 2 + 16, MemKind::Device).unwrap();
    for off in [4usize, 2, 1] {
        let w = upload_bytes(&be, off, &rand_bytes(n * k / 2, 4));
        let wt = t(&w, off, DType::U8, &[n, k / 2]);
        let (bt, gt) = (t(&bs, 0, DType::Fp8E4m3, &[n, k / 16]), t(&gs, 0, DType::F32, &[1]));
        let glu = OpCall::new(
            Op::MatmulNvfp4Glu,
            vec![
                t(&x, 0, DType::Bf16, &[1, k]),
                wt.clone(),
                bt.clone(),
                gt.clone(),
                wt.clone(),
                bt.clone(),
                gt.clone(),
            ],
            vec![t(&y, 0, DType::Bf16, &[1, n])],
            OpAttrs::default(),
        );
        assert!(be.launch(GLU, &glu, 0).is_err(), "glu weights at byte {off}");
        let down = OpCall::new(
            Op::SharedDownCombine,
            vec![
                t(&x, 0, DType::Bf16, &[1, k]),
                wt,
                bt,
                gt,
                t(&routed, 0, DType::Bf16, &[1, n]),
                t(&logit, 0, DType::Bf16, &[1, 1]),
            ],
            vec![t(&y, 0, DType::Bf16, &[1, n])],
            OpAttrs::default(),
        );
        assert!(be.launch(DOWN, &down, 0).is_err(), "down weights at byte {off}");
    }
}
