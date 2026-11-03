//! `matmul_mxfp4_gemv` (kernels/metal/mxfp4.metal): the decode GEMV (M = 1) and its multi-row form (M > 1), W4A16 and W4A4.
//! * the output stays close to an f64 reference over the decoded weights (bf16 / f32 outputs, K tails of one group);
//! * the output **bits** are pinned: `PINNED` holds the FNV-1a hash of each case's output as the GEMV of lane m27 (the kernel before
//!   the half-trick decode and the activation fusion, lane `u` summing groups `u, u + 32, ...` with an f32 fma chain, then `simd_sum`) produced it, so a decode or
//!   load change that is meant to keep the numerics must keep these (the current hashes are printed on stderr, `--nocapture`).
//! * a row's bits do not depend on the call's M.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use intarsia_backend_metal::bench::{bf16_to_f32, Rng};
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};

const GEMV: &str = "matmul_mxfp4_gemv";
const FP4: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

fn up(be: &MetalBackend, data: &[u8]) -> Buffer {
    let b = be.alloc(data.len() + 16, MemKind::Device).unwrap();
    be.upload(&b, 0, data, 0).unwrap();
    b
}

fn t(buf: &Buffer, dt: DType, shape: &[usize]) -> TensorRef {
    TensorRef { buf: buf.clone(), offset: 0, dtype: dt, shape: shape.to_vec() }
}

/// Packed codes `[n, k/2]`, scale bytes `[n, k/32]` (2^-8 .. 2^3) and the decoded f32 weights `[n, k]`.
fn weights(n: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<u8>, Vec<f32>) {
    let mut r = Rng(seed | 1);
    let w: Vec<u8> = (0..n * k / 2).map(|_| (r.next_u64() >> 40) as u8).collect();
    let sc: Vec<u8> = (0..n * k / 32).map(|_| 119 + ((r.next_u64() >> 40) % 12) as u8).collect();
    let mut dec = vec![0f32; n * k];
    for row in 0..n {
        for e in 0..k {
            let byte = w[row * (k / 2) + e / 2];
            let code = if e % 2 == 0 { byte & 0xF } else { byte >> 4 };
            dec[row * k + e] = FP4[code as usize] * f32::from_bits((sc[row * (k / 32) + e / 32] as u32) << 23);
        }
    }
    (w, sc, dec)
}

fn fnv(raw: &[u8]) -> u64 {
    raw.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3))
}

/// Output bytes of the GEMV on rows `r0..r0+m` of `x`.
fn run(be: &MetalBackend, x: &Buffer, w: &Buffer, sc: &Buffer, r0: usize, m: usize, n: usize, k: usize, f32_out: bool) -> Vec<u8> {
    run_at(be, x, r0 * k * 2, w, sc, m, n, k, f32_out, 0.0)
}

/// As [`run`], with x at byte offset `x_off` and the `act` flag (1: W4A4, the activations QDQ'd to MXFP4 per group of 32).
fn run_at(be: &MetalBackend, x: &Buffer, x_off: usize, w: &Buffer, sc: &Buffer, m: usize, n: usize, k: usize, f32_out: bool, act: f32) -> Vec<u8> {
    let esz = if f32_out { 4 } else { 2 };
    let yb = be.alloc(m * n * esz + 16, MemKind::Device).unwrap();
    let ab = up(be, &act.to_le_bytes());
    let mut cl = OpCall::new(
        Op::MatmulMxfp4,
        vec![
            TensorRef { buf: x.clone(), offset: x_off, dtype: DType::Bf16, shape: vec![m, k] },
            t(w, DType::U8, &[n, k / 2]),
            t(sc, DType::U8, &[n, k / 32]),
            t(&ab, DType::F32, &[1]),
        ],
        vec![t(&yb, if f32_out { DType::F32 } else { DType::Bf16 }, &[m, n])],
        // the model sets the W4A16 marker when the activations are not quantised: such calls take the fused GEMV (x read directly)
        OpAttrs { mxfp4_w4a16: act == 0.0, ..OpAttrs::default() },
    );
    let need = be.kernels().iter().find(|q| q.name == GEMV).unwrap().scratch_bytes(&ShapeInfo::from_call(&cl));
    let sb = be.alloc(need + 8, MemKind::Device).unwrap();
    cl.scratch = Some(t(&sb, DType::U8, &[need]));
    be.launch(GEMV, &cl, 0).unwrap();
    be.synchronize(0).unwrap();
    let mut raw = vec![0u8; m * n * esz];
    be.download(&yb, 0, &mut raw, 0).unwrap();
    raw
}

/// (M, N, K, f32 out, act): decode, MTP-verify and wider-M calls, K = 32 (one group), 544 (17 groups, a tail), 5120 and 17408 (the 27B's).
const CASES: &[(usize, usize, usize, bool, f32)] = &[
    (1, 100, 32, false, 0.0),
    (1, 130, 544, false, 0.0),
    (1, 1024, 5120, false, 0.0),
    (1, 515, 5120, true, 0.0),
    (1, 5120, 17408, false, 0.0),
    (3, 130, 1024, false, 0.0),
    (3, 130, 1024, true, 0.0),
    (8, 70, 544, false, 0.0),
    (13, 64, 5120, false, 0.0),
    // W4A4: the activations QDQ'd per group of 32 by the pre-pass
    (1, 130, 544, false, 1.0),
    (1, 515, 5120, true, 1.0),
    (3, 130, 1024, false, 1.0),
    (8, 70, 544, true, 1.0),
    (13, 64, 5120, false, 1.0),
];

/// FNV-1a of the output bytes of each case with the seeds of `case_data`, from the GEMV of lane m27 (arithmetic e2m1 decode, pre-pass dispatch; commit 9bad344e7).
const PINNED: &[u64] = &[
    // 14 cases; the first nine are the file's first version
    0x65880f8cedb93641,
    0x5d3578fa00011ac2,
    0x1c893ceae6174fba,
    0x43cfc1c517cc6e47,
    0xd9585f21e41268e4,
    0xf26e4815c35cf82a,
    0x47d6b5fdca459a02,
    0xfefd654b46c19d7a,
    0xac2caa0658ca3e60,
    0xc69df258a56ba1e8,
    0xff6bcd72a7f71c6f,
    0x759fe9a88d78b18b,
    0x49896804784d8402,
    0x30b77bebfa2059e4,
];

fn case_data(be: &MetalBackend, i: usize) -> (Buffer, Buffer, Buffer, Vec<u16>, Vec<f32>) {
    let (m, n, k, ..) = CASES[i];
    let x = rand_bf16(m * k, 900 + i as u64);
    let (w, sc, dec) = weights(n, k, 950 + i as u64);
    (upload(be, &x), up(be, &w), up(be, &sc), x, dec)
}

#[test]
fn gemv_matches_the_f64_reference() {
    let be = backend();
    for (i, &(m, n, k, f32_out, act)) in CASES.iter().enumerate() {
        if act != 0.0 {
            continue; // the W4A4 cases are pinned below and swept against the cpu backend by `intarsia kernels test`
        }
        let (xb, wb, sb, x, dec) = case_data(&be, i);
        let raw = run(&be, &xb, &wb, &sb, 0, m, n, k, f32_out);
        let mut worst = 0f64;
        for r in 0..m {
            for c in 0..n {
                let (mut s, mut mag) = (0f64, 0f64);
                for e in 0..k {
                    let p = bf16_to_f32(x[r * k + e]) as f64 * dec[c * k + e] as f64;
                    s += p;
                    mag += p.abs();
                }
                let o = r * n + c;
                let got = if f32_out {
                    f32::from_le_bytes(raw[o * 4..o * 4 + 4].try_into().unwrap()) as f64
                } else {
                    bf16_to_f32(u16::from_le_bytes([raw[o * 2], raw[o * 2 + 1]])) as f64
                };
                // bf16 output rounding (2^-8 relative) plus the f32 accumulation error, scaled by the sum of magnitudes
                let tol = if f32_out { 1e-5 } else { 5e-3 } * s.abs().max(mag * 1e-2).max(1e-6);
                worst = worst.max((got - s).abs() / tol);
                assert!((got - s).abs() <= tol, "case {i} (M={m} N={n} K={k}) row {r} col {c}: got {got}, reference {s}");
            }
        }
        eprintln!("case {i}: M={m} N={n} K={k} f32_out={f32_out}: worst error / tolerance {worst:.3}");
    }
}

#[test]
fn gemv_bits_are_pinned() {
    let be = backend();
    let mut got = vec![];
    for (i, &(m, n, k, f32_out, act)) in CASES.iter().enumerate() {
        let (xb, wb, sb, ..) = case_data(&be, i);
        got.push(fnv(&run_at(&be, &xb, 0, &wb, &sb, m, n, k, f32_out, act)));
    }
    eprintln!("current hashes: {got:#018x?}");
    assert_eq!(got, PINNED);
}

#[test]
fn a_rows_bits_do_not_depend_on_the_calls_m() {
    let be = backend();
    let (n, k) = (130, 5120);
    let x = rand_bf16(13 * k, 31);
    let (w, sc, _) = weights(n, k, 32);
    let (xb, wb, sb) = (upload(&be, &x), up(&be, &w), up(&be, &sc));
    let all = run(&be, &xb, &wb, &sb, 0, 13, n, k, false);
    for r in [0usize, 2, 7, 8, 12] {
        let one = run(&be, &xb, &wb, &sb, r, 1, n, k, false);
        assert_eq!(&all[r * n * 2..(r + 1) * n * 2], &one[..], "row {r}: M = 13 call vs M = 1 call");
    }
    let three = run(&be, &xb, &wb, &sb, 5, 3, n, k, false);
    assert_eq!(&all[5 * n * 2..8 * n * 2], &three[..], "rows 5..8: M = 13 call vs M = 3 call");
}

#[test]
fn an_unaligned_x_takes_the_prepass_and_gives_the_same_bits() {
    // x at a byte offset that is not a multiple of 16 cannot be read by the fused GEMV: the pre-pass copies it into 16-byte aligned scratch
    let be = backend();
    let (m, n, k) = (3, 130, 1024);
    let x = rand_bf16(m * k + 8, 77);
    let (w, sc, _) = weights(n, k, 78);
    let (xb, wb, sb) = (upload(&be, &x), up(&be, &w), up(&be, &sc));
    for act in [0.0f32, 1.0] {
        for off in [16usize, 4, 8] {
            let got = run_at(&be, &xb, off, &wb, &sb, m, n, k, false, act);
            // the same values at an aligned offset
            let shifted: Vec<u16> = x[off / 2..off / 2 + m * k].to_vec();
            let xa = upload(&be, &shifted);
            let want = run_at(&be, &xa, 0, &wb, &sb, m, n, k, false, act);
            assert_eq!(got, want, "x at byte offset {off}, act {act}");
        }
    }
}
