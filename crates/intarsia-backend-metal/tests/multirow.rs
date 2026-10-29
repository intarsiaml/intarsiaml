#![cfg(target_os = "macos")]
//! G0: the multi-row GEMVs (`matmul_bf16_gemv_rows`, `matmul_pair_bf16_gemv_rows`, `logits_bf16_gemv_rows`) must give,
//! for every activation row, BITWISE the output of the serving 1-row decode launch (`matmul_bf16_gemv`,
//! `matmul_pair_bf16_gemv`, `logits_bf16_gemv`) for that row: random data at the Qwen3-8B shapes, K tails, odd N, every
//! compiled row count, 16-byte and 8-byte operand alignment (and the scalar fallback), and the real Qwen3-8B weights
//! mmap'd the way the model loader maps them (opt-in: `INTARSIA_TEST_REAL_MODEL=1`, model dir `INTARSIA_METAL_G0_MODEL`, else
//! `~/models/Qwen3-8B`; the synthetic tests need no model but still use the GPU, so take an exclusive fleetq slot on endor).

mod common;
use common::*;
use intarsia_backend_metal::bench::{f32_to_bf16, Rng};
use intarsia_backend_metal::kernels::{
    LOGITS_BF16, LOGITS_BF16_GEMV, MATMUL_BF16, MATMUL_BF16_GEMV, MATMUL_PAIR_BF16_GEMV,
};
use intarsia_backend_metal::multirow::{
    self, shader, shader_split, MR_SPLITS, Family, LOGITS_BF16_GEMV_ROWS, MATMUL_BF16_GEMV_ROWS, MATMUL_PAIR_BF16_GEMV_ROWS, MR_PER_TG, MR_ROWS,
};
use intarsia_backend_metal::{GemvParams, MetalBackend, MetalBuffer};
use intarsia_core::backend::{Backend, Buffer, HostMemory, MemKind, TensorRef};
use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};
use memmap2::Mmap;
use std::path::PathBuf;
use std::sync::Arc;

/// Distinct rows with a spread of magnitudes (rows differ in scale so a row mix-up cannot cancel).
fn rand_rows(m: usize, k: usize, seed: u64) -> Vec<u16> {
    let mut r = Rng(seed | 1);
    (0..m * k)
        .map(|i| {
            let scale = 0.25 * (1 + (i / k) % 7) as f32;
            f32_to_bf16(r.unit() * scale)
        })
        .collect()
}

fn padded(be: &MetalBackend, lead_bytes: usize, data: &[u16], trail: usize) -> Buffer {
    let mut v = vec![0xEEEEu16; lead_bytes / 2];
    v.extend_from_slice(data);
    v.extend(std::iter::repeat_n(0xEEEEu16, trail / 2));
    upload(be, &v)
}

fn out_buffer(be: &MetalBackend, off: usize, bytes: usize) -> Buffer {
    let total = off + bytes + 16;
    let b = be.alloc(total, MemKind::Device).unwrap();
    be.upload(&b, 0, &vec![0xEEu8; total], 0).unwrap();
    b
}

fn dl(be: &MetalBackend, b: &Buffer, n: usize) -> Vec<u8> {
    let mut raw = vec![0u8; n];
    be.download(b, 0, &mut raw, 0).unwrap();
    raw
}

fn tref_dt(buf: &Buffer, off: usize, dt: DType, shape: &[usize]) -> TensorRef {
    TensorRef { buf: buf.clone(), offset: off, dtype: dt, shape: shape.to_vec() }
}

/// First differing byte, as a message.
fn same(a: &[u8], b: &[u8], what: &str) {
    if let Some(i) = a.iter().zip(b).position(|(x, y)| x != y) {
        panic!("{what}: first differing byte {i} of {} (serial {:#04x}, multi-row {:#04x})", a.len(), a[i], b[i]);
    }
    assert_eq!(a.len(), b.len(), "{what}: length");
}

fn registered_but_not_chosen(be: &MetalBackend, op: Op, name: &str, call: &OpCall) {
    // the kernel is in the table, and the registry never prefers it over what serving runs today
    let ks = be.kernels();
    let me = ks.iter().find(|k| k.name == name && k.op == op).expect("registered");
    assert!(me.priority < 0, "{name} must rank below every serving kernel");
    let s = ShapeInfo::from_call(call);
    let best = ks.iter().filter(|k| k.op == op && k.accepts(&s)).map(|k| k.priority).max();
    if let Some(best) = best {
        assert!(best >= 0 || !me.accepts(&s), "{name} would be the best kernel for {op} M={}", s.m);
    }
}

/// `Matmul`: serial 1-row launches vs one multi-row launch, `w` already placed (any alignment).
fn check_matmul(be: &MetalBackend, w: &TensorRef, m: usize, x_off: usize, y_off: usize, seed: u64, what: &str) {
    let (n, k) = (w.shape[0], w.shape[1]);
    let x = rand_rows(m, k, seed);
    let xb = padded(be, x_off, &x, 16);
    let xt = tref(&xb, x_off, &[m, k]);
    let ybytes = m * n * 2;
    let (ys, ym) = (out_buffer(be, y_off, ybytes), out_buffer(be, y_off, ybytes));
    let (yst, ymt) = (tref(&ys, y_off, &[m, n]), tref(&ym, y_off, &[m, n]));
    for r in 0..m {
        be.launch(MATMUL_BF16_GEMV, &matmul_call(xt.rows(r, 1), w.clone(), yst.rows(r, 1)), 0).unwrap();
    }
    let call = matmul_call(xt.clone(), w.clone(), ymt);
    be.launch(MATMUL_BF16_GEMV_ROWS, &call, 0).unwrap();
    be.synchronize(0).unwrap();
    let tot = y_off + ybytes + 16;
    same(&dl(be, &ys, tot), &dl(be, &ym, tot), &format!("matmul {what} M={m} K={k} N={n} x_off={x_off} y_off={y_off}"));
    if m >= 2 && m <= 8 {
        registered_but_not_chosen(be, Op::Matmul, MATMUL_BF16_GEMV_ROWS, &call);
    }
}

fn pair_call(x: TensorRef, w0: TensorRef, w1: TensorRef, y0: TensorRef, y1: TensorRef) -> OpCall {
    OpCall::new(Op::MatmulPair, vec![x, w0, w1], vec![y0, y1], OpAttrs::default())
}

/// `MatmulPair`: serial 1-row pair launches vs one multi-row launch.
#[allow(clippy::too_many_arguments)]
fn check_pair(
    be: &MetalBackend,
    w0: &TensorRef,
    w1: &TensorRef,
    m: usize,
    x_off: usize,
    y_off: usize,
    seed: u64,
    what: &str,
) {
    let (n0, n1, k) = (w0.shape[0], w1.shape[0], w0.shape[1]);
    let x = rand_rows(m, k, seed);
    let xb = padded(be, x_off, &x, 16);
    let xt = tref(&xb, x_off, &[m, k]);
    let (ys0, ys1) = (out_buffer(be, y_off, m * n0 * 2), out_buffer(be, y_off, m * n1 * 2));
    let (ym0, ym1) = (out_buffer(be, y_off, m * n0 * 2), out_buffer(be, y_off, m * n1 * 2));
    let (s0, s1) = (tref(&ys0, y_off, &[m, n0]), tref(&ys1, y_off, &[m, n1]));
    let (m0, m1) = (tref(&ym0, y_off, &[m, n0]), tref(&ym1, y_off, &[m, n1]));
    for r in 0..m {
        let call = pair_call(xt.rows(r, 1), w0.clone(), w1.clone(), s0.rows(r, 1), s1.rows(r, 1));
        be.launch(MATMUL_PAIR_BF16_GEMV, &call, 0).unwrap();
    }
    let call = pair_call(xt.clone(), w0.clone(), w1.clone(), m0, m1);
    be.launch(MATMUL_PAIR_BF16_GEMV_ROWS, &call, 0).unwrap();
    be.synchronize(0).unwrap();
    for (a, b, n, which) in [(&ys0, &ym0, n0, "y0"), (&ys1, &ym1, n1, "y1")] {
        let tot = y_off + m * n * 2 + 16;
        same(
            &dl(be, a, tot),
            &dl(be, b, tot),
            &format!("pair {which} {what} M={m} K={k} N=({n0},{n1}) x_off={x_off} y_off={y_off}"),
        );
    }
}

/// `Logits` (f32 out).
fn check_logits(be: &MetalBackend, w: &TensorRef, m: usize, x_off: usize, seed: u64, what: &str) {
    let (n, k) = (w.shape[0], w.shape[1]);
    let x = rand_rows(m, k, seed);
    let xb = padded(be, x_off, &x, 16);
    let xt = tref(&xb, x_off, &[m, k]);
    let ybytes = m * n * 4;
    let (ys, ym) = (out_buffer(be, 0, ybytes), out_buffer(be, 0, ybytes));
    let (yst, ymt) = (tref_dt(&ys, 0, DType::F32, &[m, n]), tref_dt(&ym, 0, DType::F32, &[m, n]));
    let lcall = |x, y| OpCall::new(Op::Logits, vec![x, w.clone()], vec![y], OpAttrs::default());
    for r in 0..m {
        be.launch(LOGITS_BF16_GEMV, &lcall(xt.rows(r, 1), yst.rows(r, 1)), 0).unwrap();
    }
    be.launch(LOGITS_BF16_GEMV_ROWS, &lcall(xt.clone(), ymt), 0).unwrap();
    be.synchronize(0).unwrap();
    let tot = ybytes + 16;
    same(&dl(be, &ys, tot), &dl(be, &ym, tot), &format!("logits {what} M={m} K={k} N={n} x_off={x_off}"));
}

fn weights(be: &MetalBackend, n: usize, k: usize, w_off: usize, seed: u64) -> TensorRef {
    let mut r = Rng(seed | 1);
    let s = 1.0 / (k as f32).sqrt();
    let w: Vec<u16> = (0..n * k).map(|_| f32_to_bf16(r.unit() * s)).collect();
    let b = padded(be, w_off, &w, 16);
    tref(&b, w_off, &[n, k])
}

#[test]
fn matmul_qwen3_8b_shapes_all_rows() {
    let be = backend();
    for (k, n) in [(4096, 4096), (12288, 4096), (4096, 1024)] {
        let w = weights(&be, n, k, 0, 11);
        for &m in &MR_ROWS {
            check_matmul(&be, &w, m, 0, 0, 100 + m as u64, "aligned");
        }
    }
}

#[test]
fn matmul_a8_alignment_and_scalar_fallback() {
    let be = backend();
    // weights at 8 mod 16 (a safetensors shard with header length 8 mod 16): the 8-byte kernels
    for (k, n) in [(4096, 4096), (12288, 4096), (4100, 33)] {
        let w = weights(&be, n, k, 8, 21);
        for &m in &[1usize, 2, 3, 4, 5, 8, 16] {
            check_matmul(&be, &w, m, 0, 0, 200 + m as u64, "w@8");
            check_matmul(&be, &w, m, 8, 4, 300 + m as u64, "w@8 x@8 y@4");
        }
    }
    // 2-byte alignment / K % 4 != 0: the scalar 1-row kernel, multi-row launch falls back to per-row launches
    for (k, n, w_off, x_off) in [(4097, 65, 0, 0), (4096, 129, 2, 0), (4096, 129, 0, 6), (13, 7, 2, 2)] {
        let w = weights(&be, n, k, w_off, 31);
        for &m in &[2usize, 5, 8] {
            check_matmul(&be, &w, m, x_off, 2, 400 + m as u64, "scalar");
        }
    }
}

#[test]
fn matmul_k_tails_and_odd_n() {
    let be = backend();
    // K = 4104 is 8 mod 512 chunks (tail loop), N odd and not a multiple of anything
    for (k, n) in [(4104, 1031), (8200, 257), (520, 5), (8, 3), (1032, 100), (24, 2)] {
        let w = weights(&be, n, k, 0, 41);
        for &m in &[2usize, 3, 4, 6, 7, 8, 16] {
            check_matmul(&be, &w, m, 0, 0, 500 + m as u64, "tail");
        }
        let w8 = weights(&be, n, k - k % 4, 8, 42);
        for &m in &[2usize, 4, 8] {
            check_matmul(&be, &w8, m, 0, 0, 600 + m as u64, "tail a8");
        }
    }
}

#[test]
fn pair_qwen3_8b_shapes_all_rows() {
    let be = backend();
    for (k, n0, n1) in [(4096, 1024, 1024), (4096, 12288, 12288), (4104, 33, 70), (4096, 5, 4096)] {
        for w_off in [0usize, 8] {
            let (w0, w1) = (weights(&be, n0, k, w_off, 51), weights(&be, n1, k, w_off, 52));
            for &m in &MR_ROWS {
                check_pair(&be, &w0, &w1, m, 0, 0, 700 + m as u64, if w_off == 0 { "aligned" } else { "w@8" });
            }
        }
    }
    // mixed alignment and 2-byte alignment: per-row fallback
    let (w0, w1) = (weights(&be, 300, 4096, 0, 61), weights(&be, 200, 4096, 8, 62));
    check_pair(&be, &w0, &w1, 4, 0, 0, 800, "mixed");
    let (w0, w1) = (weights(&be, 300, 4096, 2, 63), weights(&be, 200, 4096, 2, 64));
    check_pair(&be, &w0, &w1, 4, 0, 2, 801, "scalar");
}

/// Mixed header parity: w0 and w1 resolve to DIFFERENT 1-row GEMV families (a safetensors pair straddling two shards
/// whose header lengths differ mod 16). The serving 1-row pair plans each half on its own (16-byte kernel for the
/// 16-aligned half, 8-byte kernel for the other), so the multi-row launch must too: each half bitwise equal to the
/// 1-row serving pair launch for R = 2..=8, with 16-byte and 8-byte x as well.
#[test]
fn pair_mixed_family_halves_all_rows() {
    let be = backend();
    for (k, n0, n1) in [(4096, 1024, 1024), (4096, 12288, 4096), (4104, 33, 70), (4096, 5, 300)] {
        for (o0, o1) in [(0usize, 8usize), (8, 0)] {
            let (w0, w1) = (weights(&be, n0, k, o0, 151), weights(&be, n1, k, o1, 152));
            for m in 2..=8usize {
                for x_off in [0usize, 8] {
                    check_pair(&be, &w0, &w1, m, x_off, 0, 1700 + m as u64, &format!("mixed w0@{o0} w1@{o1}"));
                }
            }
        }
    }
}

#[test]
fn logits_qwen3_8b_shapes_all_rows() {
    let be = backend();
    for (k, n) in [(4096, 151936), (4104, 1031), (256, 2048), (8, 3)] {
        let w = weights(&be, n, k, 0, 71);
        for &m in &MR_ROWS {
            check_logits(&be, &w, m, 0, 900 + m as u64, "aligned");
        }
    }
    // 8-byte aligned and odd alignments: the 1-row path is the scalar f32 kernel, the multi-row falls back per row
    let w = weights(&be, 1031, 4096, 8, 72);
    for &m in &[2usize, 4, 8] {
        check_logits(&be, &w, m, 0, 1000 + m as u64, "w@8");
        check_logits(&be, &w, m, 6, 1100 + m as u64, "x@6");
    }
}

/// Every (family, R, P) instance exists as a pipeline.
#[test]
fn every_instance_exists() {
    let be = backend();
    for f in Family::ALL {
        for &r in &MR_ROWS {
            for &p in &MR_PER_TG {
                let name = shader(f, r, p).unwrap();
                assert!(be.pso(name).is_some(), "no pipeline {name}");
            }
        }
        for &(r, p, rb) in &MR_SPLITS {
            let name = shader_split(f, r, p, rb).unwrap();
            assert!(be.pso(name).is_some(), "no pipeline {name}");
        }
    }
    assert_eq!(shader(Family::Logits, 9, 1), None);
    assert_eq!(shader(Family::Logits, 4, 3), None);
}

fn mtl_of(t: &TensorRef) -> (&objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>, usize) {
    let mb = t.buf.downcast::<MetalBuffer>().unwrap();
    (mb.mtl(), mb.base() + t.offset)
}

const MAXR: usize = 16;

/// Direct encode of every (R, P) instance of the matmul families against R serial serving 1-row launches (computed once
/// for 16 rows; output rows do not depend on R).
fn check_matmul_variants(be: &MetalBackend, k: usize, n: usize, w_off: usize) {
    let w = weights(be, n, k, w_off, 1234 + k as u64);
    let a8 = w_off % 16 != 0;
    let x = rand_rows(MAXR, k, 99);
    let xb = padded(be, 0, &x, 16);
    let xt = tref(&xb, 0, &[MAXR, k]);
    let ys = out_buffer(be, 0, MAXR * n * 2);
    let yst = tref(&ys, 0, &[MAXR, n]);
    for r in 0..MAXR {
        be.launch(MATMUL_BF16_GEMV, &matmul_call(xt.rows(r, 1), w.clone(), yst.rows(r, 1)), 0).unwrap();
    }
    be.synchronize(0).unwrap();
    let reference = dl(be, &ys, MAXR * n * 2);
    let fam = if a8 { Family::MatmulA8 } else { Family::Matmul16 };
    let (wm, wo) = mtl_of(&w);
    let (xm, xo) = mtl_of(&xt);
    let mut variants: Vec<(usize, usize, &str)> = vec![];
    for &r in &MR_ROWS {
        for &p in &MR_PER_TG {
            variants.push((r, p, shader(fam, r, p).unwrap()));
        }
    }
    for &(r, p, rb) in &MR_SPLITS {
        variants.push((r, p, shader_split(fam, r, p, rb).unwrap()));
    }
    for (r, p, name) in variants {
        {
            let ym = out_buffer(be, 0, MAXR * n * 2);
            let ymt = tref(&ym, 0, &[MAXR, n]);
            let (ymm, yo) = mtl_of(&ymt);
            let pso = be.pso(name).unwrap();
            let params = GemvParams { k: k as u32, n: n as u32, ..Default::default() };
            let cb = be
                .encode_and_commit(|enc| {
                    multirow::encode_matmul_rows(enc, pso, (wm, wo), (xm, xo), (ymm, yo), &params, p)
                })
                .unwrap();
            MetalBackend::wait(&cb).unwrap();
            let got = dl(be, &ym, MAXR * n * 2);
            same(&reference[..r * n * 2], &got[..r * n * 2], &format!("{fam:?} R={r} P={p} K={k} N={n} w_off={w_off}"));
            assert!(got[r * n * 2..].iter().all(|b| *b == 0xEE), "{fam:?} R={r} P={p}: stray store past row R");
        }
    }
}

#[test]
fn all_variants_matmul() {
    let be = backend();
    for (k, n) in [(4096, 4096), (12288, 4096), (4104, 1031), (520, 5), (8, 3), (1032, 100)] {
        check_matmul_variants(&be, k, n, 0);
    }
    for (k, n) in [(4096, 4096), (12288, 4096), (4100, 33), (28, 7)] {
        check_matmul_variants(&be, k, n, 8);
    }
}

fn check_pair_variants(be: &MetalBackend, k: usize, n0: usize, n1: usize, w_off: usize) {
    let (w0, w1) = (weights(be, n0, k, w_off, 7 + k as u64), weights(be, n1, k, w_off, 8 + k as u64));
    let a8 = w_off % 16 != 0;
    let x = rand_rows(MAXR, k, 98);
    let xb = padded(be, 0, &x, 16);
    let xt = tref(&xb, 0, &[MAXR, k]);
    let (ys0, ys1) = (out_buffer(be, 0, MAXR * n0 * 2), out_buffer(be, 0, MAXR * n1 * 2));
    let (s0, s1) = (tref(&ys0, 0, &[MAXR, n0]), tref(&ys1, 0, &[MAXR, n1]));
    for r in 0..MAXR {
        let call = pair_call(xt.rows(r, 1), w0.clone(), w1.clone(), s0.rows(r, 1), s1.rows(r, 1));
        be.launch(MATMUL_PAIR_BF16_GEMV, &call, 0).unwrap();
    }
    be.synchronize(0).unwrap();
    let (ref0, ref1) = (dl(be, &ys0, MAXR * n0 * 2), dl(be, &ys1, MAXR * n1 * 2));
    let fam = if a8 { Family::PairA8 } else { Family::Pair16 };
    let ((w0m, w0o), (w1m, w1o), (xm, xo)) = (mtl_of(&w0), mtl_of(&w1), mtl_of(&xt));
    let mut variants: Vec<(usize, usize, &str)> = vec![];
    for &r in &MR_ROWS {
        for &p in &MR_PER_TG {
            variants.push((r, p, shader(fam, r, p).unwrap()));
        }
    }
    for &(r, p, rb) in &MR_SPLITS {
        variants.push((r, p, shader_split(fam, r, p, rb).unwrap()));
    }
    for (r, p, name) in variants {
        {
            let (y0, y1) = (out_buffer(be, 0, MAXR * n0 * 2), out_buffer(be, 0, MAXR * n1 * 2));
            let (y0t, y1t) = (tref(&y0, 0, &[MAXR, n0]), tref(&y1, 0, &[MAXR, n1]));
            let ((y0m, y0o), (y1m, y1o)) = (mtl_of(&y0t), mtl_of(&y1t));
            let pso = be.pso(name).unwrap();
            let pp = multirow::PairParams { k: k as u32, n0: n0 as u32, n1: n1 as u32, ..Default::default() };
            let cb = be
                .encode_and_commit(|enc| {
                    multirow::encode_pair_rows(
                        enc,
                        pso,
                        (w0m, w0o),
                        (w1m, w1o),
                        (xm, xo),
                        (y0m, y0o),
                        (y1m, y1o),
                        &pp,
                        p,
                    )
                })
                .unwrap();
            MetalBackend::wait(&cb).unwrap();
            let (g0, g1) = (dl(be, &y0, MAXR * n0 * 2), dl(be, &y1, MAXR * n1 * 2));
            let what = format!("{fam:?} R={r} P={p} K={k} N=({n0},{n1}) w_off={w_off}");
            same(&ref0[..r * n0 * 2], &g0[..r * n0 * 2], &format!("{what} y0"));
            same(&ref1[..r * n1 * 2], &g1[..r * n1 * 2], &format!("{what} y1"));
            assert!(g0[r * n0 * 2..].iter().all(|b| *b == 0xEE) && g1[r * n1 * 2..].iter().all(|b| *b == 0xEE), "{what}: stray store");
        }
    }
}

#[test]
fn all_variants_pair() {
    let be = backend();
    for (k, n0, n1) in [(4096, 1024, 1024), (4096, 12288, 12288), (4104, 33, 70), (4096, 5, 4096), (8, 1, 1)] {
        check_pair_variants(&be, k, n0, n1, 0);
    }
    for (k, n0, n1) in [(4096, 1024, 1024), (4096, 12288, 12288), (4100, 33, 70)] {
        check_pair_variants(&be, k, n0, n1, 8);
    }
}

fn check_logits_variants(be: &MetalBackend, k: usize, n: usize) {
    let w = weights(be, n, k, 0, 55 + k as u64);
    let x = rand_rows(MAXR, k, 97);
    let xb = padded(be, 0, &x, 16);
    let xt = tref(&xb, 0, &[MAXR, k]);
    let ys = out_buffer(be, 0, MAXR * n * 4);
    let yst = tref_dt(&ys, 0, DType::F32, &[MAXR, n]);
    for r in 0..MAXR {
        let call = OpCall::new(Op::Logits, vec![xt.rows(r, 1), w.clone()], vec![yst.rows(r, 1)], OpAttrs::default());
        be.launch(LOGITS_BF16_GEMV, &call, 0).unwrap();
    }
    be.synchronize(0).unwrap();
    let reference = dl(be, &ys, MAXR * n * 4);
    let ((wm, wo), (xm, xo)) = (mtl_of(&w), mtl_of(&xt));
    let mut variants: Vec<(usize, usize, &str)> = vec![];
    for &r in &MR_ROWS {
        for &p in &MR_PER_TG {
            variants.push((r, p, shader(Family::Logits, r, p).unwrap()));
        }
    }
    for &(r, p, rb) in &MR_SPLITS {
        variants.push((r, p, shader_split(Family::Logits, r, p, rb).unwrap()));
    }
    for (r, p, name) in variants {
        {
            let ym = out_buffer(be, 0, MAXR * n * 4);
            let ymt = tref_dt(&ym, 0, DType::F32, &[MAXR, n]);
            let (ymm, yo) = mtl_of(&ymt);
            let pso = be.pso(name).unwrap();
            let params = GemvParams { k: k as u32, n: n as u32, ..Default::default() };
            let cb = be
                .encode_and_commit(|enc| multirow::encode_logits_rows(enc, pso, (wm, wo), (xm, xo), (ymm, yo), &params, p))
                .unwrap();
            MetalBackend::wait(&cb).unwrap();
            let got = dl(be, &ym, MAXR * n * 4);
            same(&reference[..r * n * 4], &got[..r * n * 4], &format!("logits R={r} P={p} K={k} N={n}"));
            assert!(got[r * n * 4..].iter().all(|b| *b == 0xEE), "logits R={r} P={p}: stray store");
        }
    }
}

#[test]
fn all_variants_logits() {
    let be = backend();
    for (k, n) in [(4096, 151936), (4104, 1031), (256, 2049), (8, 3)] {
        check_logits_variants(&be, k, n);
    }
}

/// Ordinary calls (`exact_rows` false) get what serving picked before the rows kernels were listed, through
/// `Registry::select` itself: NAX GEMM / NAX pair for M = 2..15 (`NAX_MIN_M` = 2 since the tail-chunk fix; tiled GEMM
/// and no pair kernel without the metal4 library) and the tiled logits GEMM. Verify / batched calls (`exact_rows`) of
/// M = 2..16 get the multi-row GEMVs and nothing else (9..16 rows run as two launches, lane mdec), and M > 16 fail instead of
/// landing on another kernel.
#[test]
fn serving_selection_unchanged() {
    let be = backend();
    let reg = intarsia_core::Registry::new(be.name(), &be.info(), be.kernels());
    let pick = |c: &OpCall| reg.select(c).map(|k| k.name.clone());
    let nax = be.kernels().iter().any(|k| k.name == intarsia_backend_metal::kernels::MATMUL_BF16_NAX);
    let (mm, pair) = if nax {
        (intarsia_backend_metal::kernels::MATMUL_BF16_NAX, Some(intarsia_backend_metal::kernels::MATMUL_PAIR_BF16_NAX))
    } else {
        (MATMUL_BF16, None)
    };
    for m in [2usize, 3, 8, 15, 16, 17] {
        let w = weights(&be, 64, 64, 0, 1);
        let xb = padded(&be, 0, &rand_rows(m, 64, 2), 16);
        let yb = out_buffer(&be, 0, m * 64 * 2);
        let mut call = matmul_call(tref(&xb, 0, &[m, 64]), w.clone(), tref(&yb, 0, &[m, 64]));
        assert_eq!(pick(&call).unwrap(), mm, "M={m}");
        let yf = out_buffer(&be, 0, m * 64 * 4);
        let mut lcall = OpCall::new(
            Op::Logits,
            vec![tref(&xb, 0, &[m, 64]), w],
            vec![tref_dt(&yf, 0, DType::F32, &[m, 64])],
            OpAttrs::default(),
        );
        assert_eq!(pick(&lcall).unwrap(), LOGITS_BF16, "M={m}");
        let mut pcall = pair_call(
            tref(&xb, 0, &[m, 64]),
            weights(&be, 64, 64, 0, 3),
            weights(&be, 64, 64, 0, 4),
            tref(&yb, 0, &[m, 64]),
            tref(&yb, 0, &[m, 64]),
        );
        assert_eq!(pick(&pcall).ok().as_deref(), pair, "M={m}");
        call.exact_rows = true;
        lcall.exact_rows = true;
        pcall.exact_rows = true;
        if m <= 16 {
            assert_eq!(pick(&call).unwrap(), MATMUL_BF16_GEMV_ROWS, "M={m}");
            assert_eq!(pick(&lcall).unwrap(), LOGITS_BF16_GEMV_ROWS, "M={m}");
            assert_eq!(pick(&pcall).unwrap(), MATMUL_PAIR_BF16_GEMV_ROWS, "M={m}");
        } else {
            assert!(pick(&call).is_err() && pick(&lcall).is_err() && pick(&pcall).is_err(), "M={m}");
        }
    }
}

// ---- real Qwen3-8B weights, mmap'd and wrapped zero-copy as the model loader does ----------------------------------

struct MmapHost(Mmap);
impl HostMemory for MmapHost {
    fn as_slice(&self) -> &[u8] {
        &self.0
    }
}

fn model_dir() -> PathBuf {
    std::env::var_os("INTARSIA_METAL_G0_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("models/Qwen3-8B"))
}

/// (mmap, tensor name -> (begin, end, shape)) of one shard.
fn open_shard(name: &str) -> Option<(Arc<MmapHost>, serde_json::Value, usize)> {
    let f = std::fs::File::open(model_dir().join(name)).ok()?;
    // SAFETY: read-only mapping of a file nothing modifies (same as the loader).
    let mm = unsafe { Mmap::map(&f) }.ok()?;
    let hl = u64::from_le_bytes(mm[..8].try_into().unwrap()) as usize;
    let hdr: serde_json::Value = serde_json::from_slice(&mm[8..8 + hl]).unwrap();
    Some((Arc::new(MmapHost(mm)), hdr, 8 + hl))
}

fn tensor(be: &MetalBackend, shard: &(Arc<MmapHost>, serde_json::Value, usize), name: &str) -> TensorRef {
    let v = &shard.1[name];
    let o = v["data_offsets"].as_array().unwrap();
    let (b, e) = (o[0].as_u64().unwrap() as usize, o[1].as_u64().unwrap() as usize);
    let shape: Vec<usize> = v["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as usize).collect();
    let buf = be.wrap_host(shard.0.clone(), shard.2 + b, e - b).expect("wrap_host");
    TensorRef { buf, offset: 0, dtype: DType::Bf16, shape }
}

#[test]
fn real_qwen3_8b_weights_bitwise() {
    if !real_model_enabled() {
        return;
    }
    let be = backend();
    let Some(s1) = open_shard("model-00001-of-00005.safetensors") else {
        eprintln!("SKIP: Qwen3-8B not found at {}", model_dir().display());
        return;
    };
    let s5 = open_shard("model-00005-of-00005.safetensors").expect("shard 5");
    let s3 = open_shard("model-00003-of-00005.safetensors").expect("shard 3");
    let rows = [2usize, 3, 4, 5, 6, 7, 8, 16];
    let l = |i: usize, p: &str| format!("model.layers.{i}.{p}.weight");
    for (shard, layer) in [(&s1, 0usize), (&s3, 20)] {
        for p in ["self_attn.q_proj", "self_attn.o_proj", "mlp.down_proj"] {
            let w = tensor(&be, shard, &l(layer, p));
            for &m in &rows {
                check_matmul(&be, &w, m, 0, 0, 1200 + m as u64, &format!("real L{layer} {p}"));
            }
        }
        for (a, b) in [("self_attn.k_proj", "self_attn.v_proj"), ("mlp.gate_proj", "mlp.up_proj")] {
            let (w0, w1) = (tensor(&be, shard, &l(layer, a)), tensor(&be, shard, &l(layer, b)));
            for &m in &rows {
                check_pair(&be, &w0, &w1, m, 0, 0, 1300 + m as u64, &format!("real L{layer} {a}+{b}"));
            }
        }
    }
    let lm = tensor(&be, &s5, "lm_head.weight");
    for &m in &[2usize, 4, 8] {
        check_logits(&be, &lm, m, 0, 1400 + m as u64, "real lm_head");
    }
}
