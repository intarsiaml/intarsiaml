//! `MatmulFp8` and `MatmulNvfp4` (kernels/metal/{fp8,nvfp4}.metal) against the oracle math of
//! `intarsia_core::quant::reference`: W8A8 and W8A16, bf16 and f32 outputs, M on both sides of the GEMV / GEMM split,
//! weights at 8 mod 16 (the real checkpoints' alignment), block scales and f32 scales at odd offsets, a weight alignment the
//! kernels refuse, and the W8A8 activation quantiser bit for bit over every bf16 pattern. The sweep over the registered
//! shapes against the cpu backend is `intarsia kernels test --backend metal --op matmul_fp8` (and `matmul_nvfp4`).

#![cfg(target_os = "macos")]

mod common;

use common::*;
use intarsia_backend_metal::bench::{bf16_to_f32, f32_to_bf16, Rng};
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::dtype::{f32_to_fp8_e4m3, fp8_e4m3_to_f32};
use intarsia_core::quant::reference::{matmul_fp8, matmul_nvfp4};
use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};

/// Upload `data` at byte offset `off` of a fresh buffer (0xEE guards around it).
fn upload_bytes(be: &MetalBackend, off: usize, data: &[u8]) -> Buffer {
    let b = be.alloc(off + data.len() + 16, MemKind::Device).unwrap();
    be.upload(&b, 0, &vec![0xEEu8; off + data.len() + 16], 0).unwrap();
    be.upload(&b, off, data, 0).unwrap();
    b
}

fn t(buf: &Buffer, off: usize, dt: DType, shape: &[usize]) -> TensorRef {
    TensorRef { buf: buf.clone(), offset: off, dtype: dt, shape: shape.to_vec() }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bf16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The scratch workspace the registry hands a kernel that declares one (0xEE-filled).
fn attach_scratch(be: &MetalBackend, kernel: &str, cl: &mut OpCall) {
    let need = be.kernels().iter().find(|k| k.name == kernel).map_or(0, |k| k.scratch_bytes(&ShapeInfo::from_call(cl)));
    if need > 0 {
        let b = be.alloc(need, MemKind::Device).unwrap();
        be.upload(&b, 0, &vec![0xEEu8; need], 0).unwrap();
        cl.scratch = Some(TensorRef { buf: b, offset: 0, dtype: DType::U8, shape: vec![need] });
    }
}

fn accepts(be: &MetalBackend, kernel: &str, cl: &OpCall) -> bool {
    be.kernels().iter().find(|k| k.name == kernel).unwrap().accepts(&ShapeInfo::from_call(cl))
}

/// e4m3 bytes that are never the NaN code.
fn rand_e4m3(n: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed | 1);
    (0..n)
        .map(|_| {
            let b = (r.next_u64() >> 32) as u8;
            if b & 0x7F == 0x7F {
                b & 0xFE
            } else {
                b
            }
        })
        .collect()
}

fn rand_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed | 1);
    (0..n).map(|_| (r.next_u64() >> 32) as u8).collect()
}

/// Block scales of moderate magnitude (e4m3 exponents 4..=11, any sign and mantissa).
fn rand_block_scales(n: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed | 1);
    (0..n)
        .map(|_| {
            let v = (r.next_u64() >> 32) as u8;
            (v & 0x87) | (((v >> 3) % 8 + 4) << 3)
        })
        .collect()
}

fn read_out(be: &MetalBackend, b: &Buffer, off: usize, n: usize, f32_out: bool) -> Vec<f64> {
    if f32_out {
        let mut raw = vec![0u8; n * 4];
        be.download(b, off, &mut raw, 0).unwrap();
        raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect()
    } else {
        let mut raw = vec![0u8; n * 2];
        be.download(b, off, &mut raw, 0).unwrap();
        raw.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])) as f64).collect()
    }
}

/// Compare with the oracle: half a bf16 ulp of the result (2% slack) or the f32 accumulation noise, plus an f32 reduction
/// bound relative to the L1 mass `sum |x w|`.
fn check(got: &[f64], want: &[f64], l1: &[f64], f32_out: bool, what: &str) {
    let mut worst = 0.0f64;
    for (i, ((g, r), m)) in got.iter().zip(want).zip(l1).enumerate() {
        let tol = if f32_out { r.abs() * 2e-6 + 2e-6 * m } else { r.abs() * 2f64.powi(-8) * 1.02 + 2e-6 * m } + 1e-30;
        assert!((g - r).abs() <= tol, "{what}: element {i}: gpu {g} ref {r} (|err| {} > tol {tol})", (g - r).abs());
        worst = worst.max((g - r).abs() / tol);
    }
    eprintln!("{what}: {} elements, worst error {worst:.3} of tolerance", got.len());
}

struct Fp8Case {
    m: usize,
    n: usize,
    k: usize,
    is: f32,
    f32_out: bool,
}

const FP8_CASES: &[(usize, usize, usize)] =
    &[(1, 40, 96), (1, 512, 2048), (1, 2048, 4096), (3, 33, 48), (8, 130, 512), (8, 64, 2048), (9, 33, 48), (70, 130, 512), (200, 100, 64)];

fn run_fp8(be: &MetalBackend, kernel: &str, c: &Fp8Case, w_off: usize, seed: u64) {
    let (m, n, k) = (c.m, c.n, c.k);
    let x = rand_bf16(m * k, seed);
    let w = rand_e4m3(n * k, seed + 1);
    let ws = 0.0123f32;
    let scales = [ws, c.is];
    // oracle (f64) and the L1 mass from |x|, |w|
    let xf: Vec<f32> = x.iter().map(|v| bf16_to_f32(*v)).collect();
    let want = matmul_fp8(&xf, m, k, &w, n, scales);
    let xa: Vec<f32> = xf.iter().map(|v| v.abs()).collect();
    let wa: Vec<u8> = w.iter().map(|b| b & 0x7F).collect();
    let l1 = matmul_fp8(&xa, m, k, &wa, n, [ws.abs(), c.is]);
    let (xo, so, yo) = (2usize, 4usize, 2usize);
    let (xb, wb, sb) = (upload_bytes(be, xo, &bf16_bytes(&x)), upload_bytes(be, w_off, &w), upload_bytes(be, so, &f32_bytes(&scales)));
    let ysz = yo + m * n * 4 + 16;
    let yb = be.alloc(ysz, MemKind::Device).unwrap();
    be.upload(&yb, 0, &vec![0xEEu8; ysz], 0).unwrap();
    let odt = if c.f32_out { DType::F32 } else { DType::Bf16 };
    let yo = if c.f32_out { 4 } else { yo };
    let mut cl = OpCall::new(
        Op::MatmulFp8,
        vec![t(&xb, xo, DType::Bf16, &[m, k]), t(&wb, w_off, DType::Fp8E4m3, &[n, k]), t(&sb, so, DType::F32, &[2])],
        vec![t(&yb, yo, odt, &[m, n])],
        OpAttrs::default(),
    );
    attach_scratch(be, kernel, &mut cl);
    be.launch(kernel, &cl, 0).unwrap();
    be.synchronize(0).unwrap();
    let got = read_out(be, &yb, yo, m * n, c.f32_out);
    check(&got, &want, &l1, c.f32_out, &format!("{kernel} m={m} n={n} k={k} is={} f32_out={}", c.is, c.f32_out));
}

#[test]
fn fp8_matches_the_oracle() {
    let be = backend();
    for (ci, &(m, n, k)) in FP8_CASES.iter().enumerate() {
        for is in [0.0f32, 0.37, 2.0] {
            for f32_out in [false, true] {
                let c = Fp8Case { m, n, k, is, f32_out };
                let seed = 1000 + ci as u64 * 7;
                // weights at 8 mod 16 (and 0 mod 16) of a larger buffer
                for (kernel, w_off) in [("matmul_fp8_gemv", 8usize), ("matmul_fp8_gemv", 16), ("matmul_fp8_mma", 8), ("matmul_fp8_mma", 16)] {
                    let probe = OpCall::new(
                        Op::MatmulFp8,
                        vec![
                            t(&be.alloc(16, MemKind::Device).unwrap(), 0, DType::Bf16, &[m, k]),
                            t(&be.alloc(16, MemKind::Device).unwrap(), 0, DType::Fp8E4m3, &[n, k]),
                            t(&be.alloc(16, MemKind::Device).unwrap(), 0, DType::F32, &[2]),
                        ],
                        vec![t(&be.alloc(16, MemKind::Device).unwrap(), 0, DType::Bf16, &[m, n])],
                        OpAttrs::default(),
                    );
                    if accepts(&be, kernel, &probe) {
                        run_fp8(&be, kernel, &c, w_off, seed);
                    }
                }
            }
        }
    }
}

fn run_nvfp4(be: &MetalBackend, kernel: &str, m: usize, n: usize, k: usize, f32_out: bool, w_off: usize, seed: u64) {
    let x = rand_bf16(m * k, seed);
    let w = rand_bytes(n * k / 2, seed + 1);
    let bs = rand_block_scales(n * k / 16, seed + 2);
    let gs = 0.00071f32;
    let xf: Vec<f32> = x.iter().map(|v| bf16_to_f32(*v)).collect();
    let want = matmul_nvfp4(&xf, m, k, &w, &bs, gs, n);
    let xa: Vec<f32> = xf.iter().map(|v| v.abs()).collect();
    let wa: Vec<u8> = w.iter().map(|b| b & 0x77).collect();
    let ba: Vec<u8> = bs.iter().map(|b| b & 0x7F).collect();
    let l1 = matmul_nvfp4(&xa, m, k, &wa, &ba, gs, n);
    let (xo, bo, go) = (2usize, 3usize, 4usize);
    let xb = upload_bytes(be, xo, &bf16_bytes(&x));
    let (wb, bb, gb) = (upload_bytes(be, w_off, &w), upload_bytes(be, bo, &bs), upload_bytes(be, go, &gs.to_le_bytes()));
    let ysz = m * n * 4 + 32;
    let yb = be.alloc(ysz, MemKind::Device).unwrap();
    be.upload(&yb, 0, &vec![0xEEu8; ysz], 0).unwrap();
    let (odt, yo) = if f32_out { (DType::F32, 4) } else { (DType::Bf16, 2) };
    let cl = OpCall::new(
        Op::MatmulNvfp4,
        vec![
            t(&xb, xo, DType::Bf16, &[m, k]),
            t(&wb, w_off, DType::U8, &[n, k / 2]),
            t(&bb, bo, DType::Fp8E4m3, &[n, k / 16]),
            t(&gb, go, DType::F32, &[1]),
        ],
        vec![t(&yb, yo, odt, &[m, n])],
        OpAttrs::default(),
    );
    if !accepts(be, kernel, &cl) {
        return;
    }
    be.launch(kernel, &cl, 0).unwrap();
    be.synchronize(0).unwrap();
    let got = read_out(be, &yb, yo, m * n, f32_out);
    check(&got, &want, &l1, f32_out, &format!("{kernel} m={m} n={n} k={k} f32_out={f32_out} w_off={w_off}"));
}

#[test]
fn nvfp4_matches_the_oracle() {
    let be = backend();
    for (ci, &(m, n, k)) in
        [(1usize, 40usize, 48usize), (1, 512, 2048), (1, 2048, 512), (4, 33, 80), (8, 130, 512), (9, 33, 48), (70, 130, 512), (200, 100, 64), (1, 3000, 2048)]
            .iter()
            .enumerate()
    {
        for f32_out in [false, true] {
            for kernel in ["matmul_nvfp4_gemv", "matmul_nvfp4_mma"] {
                for w_off in [8usize, 16] {
                    run_nvfp4(&be, kernel, m, n, k, f32_out, w_off, 5000 + ci as u64 * 11);
                }
            }
        }
    }
}

/// An unaligned weight is refused with an error, never read wrongly.
#[test]
fn weights_not_8_byte_aligned_are_refused() {
    let be = backend();
    let (m, n, k) = (1usize, 16usize, 32usize);
    let x = upload_bytes(&be, 0, &bf16_bytes(&rand_bf16(m * k, 1)));
    let sc = upload_bytes(&be, 0, &f32_bytes(&[1.0, 0.5]));
    let y = be.alloc(m * n * 2 + 16, MemKind::Device).unwrap();
    for off in [4usize, 2, 1] {
        let w = upload_bytes(&be, off, &rand_e4m3(n * k, 2));
        let cl = OpCall::new(
            Op::MatmulFp8,
            vec![t(&x, 0, DType::Bf16, &[m, k]), t(&w, off, DType::Fp8E4m3, &[n, k]), t(&sc, 0, DType::F32, &[2])],
            vec![t(&y, 0, DType::Bf16, &[m, n])],
            OpAttrs::default(),
        );
        assert!(be.launch("matmul_fp8_gemv", &cl, 0).is_err(), "fp8 weights at byte {off}");
        let bs = upload_bytes(&be, 0, &rand_block_scales(n * k / 16, 3));
        let gs = upload_bytes(&be, 0, &1.0f32.to_le_bytes());
        let w4 = upload_bytes(&be, off, &rand_bytes(n * k / 2, 4));
        let cl = OpCall::new(
            Op::MatmulNvfp4,
            vec![
                t(&x, 0, DType::Bf16, &[m, k]),
                t(&w4, off, DType::U8, &[n, k / 2]),
                t(&bs, 0, DType::Fp8E4m3, &[n, k / 16]),
                t(&gs, 0, DType::F32, &[1]),
            ],
            vec![t(&y, 0, DType::Bf16, &[m, n])],
            OpAttrs::default(),
        );
        assert!(be.launch("matmul_nvfp4_gemv", &cl, 0).is_err(), "nvfp4 weights at byte {off}");
    }
}

/// The W8A8 activation quantiser is `f32_to_fp8_e4m3(x / is)` bit for bit, for every bf16 pattern (NaN, inf, subnormals, every
/// tie of the e4m3 grid at is = 1). x is `[4096, 16]`, W the 16 x 16 identity (e4m3 1.0), so `out[m][j]` = decode(q(x[m][j])) * is
/// unless the row holds a NaN code, which poisons the whole row (NaN * 0). Run through the GEMM, the 8-row GEMV and the 1-row split-K GEMV.
#[test]
fn w8a8_quantiser_is_bit_exact_over_every_bf16_pattern() {
    let be = backend();
    let (rows, k) = (4096usize, 16usize);
    let x: Vec<u16> = (0..=65535u32).map(|v| v as u16).collect();
    let mut w = vec![0u8; k * k];
    for j in 0..k {
        w[j * k + j] = 0x38;
    }
    for is in [1.0f32, 0.37, 0.003, 300.0] {
        let want: Vec<f32> = (0..rows * k)
            .map(|i| {
                let row = &x[i / k * k..][..k];
                let nan = row.iter().any(|v| fp8_e4m3_to_f32(f32_to_fp8_e4m3(bf16_to_f32(*v) / is)).is_nan());
                if nan {
                    f32::NAN
                } else {
                    fp8_e4m3_to_f32(f32_to_fp8_e4m3(bf16_to_f32(x[i]) / is)) * is
                }
            })
            .collect();
        let (xb, wb, sb) = (upload_bytes(&be, 0, &bf16_bytes(&x)), upload_bytes(&be, 8, &w), upload_bytes(&be, 0, &f32_bytes(&[1.0, is])));
        // the GEMM, the 8-row GEMV (staged q(x)) and the 1-row split-K GEMV (each thread quantises its own activations)
        for (kernel, step) in [("matmul_fp8_mma", rows), ("matmul_fp8_gemv", 8), ("matmul_fp8_gemv", 1)] {
            let yb = be.alloc(rows * k * 4 + 16, MemKind::Device).unwrap();
            be.upload(&yb, 0, &vec![0xEEu8; rows * k * 4 + 16], 0).unwrap();
            for r0 in (0..rows).step_by(step) {
                let mut cl = OpCall::new(
                    Op::MatmulFp8,
                    vec![t(&xb, r0 * k * 2, DType::Bf16, &[step, k]), t(&wb, 8, DType::Fp8E4m3, &[k, k]), t(&sb, 0, DType::F32, &[2])],
                    vec![t(&yb, r0 * k * 4, DType::F32, &[step, k])],
                    OpAttrs::default(),
                );
                attach_scratch(&be, kernel, &mut cl);
                be.launch(kernel, &cl, 0).unwrap();
            }
            be.synchronize(0).unwrap();
            let got = read_out(&be, &yb, 0, rows * k, true);
            let bad = got
                .iter()
                .zip(&want)
                .enumerate()
                .filter(|(_, (g, w))| !(((**g as f32) == **w) || ((**g as f32).is_nan() && w.is_nan())))
                .map(|(i, (g, w))| (i, x[i], *g, *w))
                .take(5)
                .collect::<Vec<_>>();
            assert!(bad.is_empty(), "{kernel} M={step} is={is}: quantiser differs (element, bf16 bits, gpu, want): {bad:?}");
            eprintln!("{kernel} M={step} is={is}: {} elements bit-exact", rows * k);
        }
    }
    let _ = f32_to_bf16;
}
