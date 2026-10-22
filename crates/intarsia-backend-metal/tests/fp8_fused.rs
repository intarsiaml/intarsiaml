//! `MatmulFp8Pair` (`matmul_fp8_pair_gemv`) and `RmsNormAddMatmulFp8Pair` (`rmsnorm_add_matmul_fp8_pair_gemv`) against the
//! unfused launches they replace (`matmul_fp8_gemv` per matrix, `rmsnorm_add_bf16` in front): the outputs must be BYTE
//! identical, not within a tolerance. Two input scales (W8A8 with different values, W8A16), bf16 and f32 outputs, M on both
//! sides of the m1 / m8 GEMV variants, N that is not a multiple of a threadgroup's rows, weights at 8 mod 16, outputs at
//! odd element offsets.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use intarsia_backend_metal::bench::Rng;
use intarsia_backend_metal::kernels::RMSNORM_ADD_BF16;
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};

const MATMUL_FP8_GEMV: &str = "matmul_fp8_gemv";
const MATMUL_FP8_PAIR_GEMV: &str = "matmul_fp8_pair_gemv";
const RMSNORM_ADD_MATMUL_FP8_PAIR_GEMV: &str = "rmsnorm_add_matmul_fp8_pair_gemv";

fn up(be: &MetalBackend, off: usize, data: &[u8]) -> Buffer {
    let b = be.alloc(off + data.len() + 16, MemKind::Device).unwrap();
    be.upload(&b, 0, &vec![0xEEu8; off + data.len() + 16], 0).unwrap();
    be.upload(&b, off, data, 0).unwrap();
    b
}

fn poison(be: &MetalBackend, bytes: usize) -> Buffer {
    let b = be.alloc(bytes, MemKind::Device).unwrap();
    be.upload(&b, 0, &vec![0xEEu8; bytes], 0).unwrap();
    b
}

fn t(buf: &Buffer, off: usize, dt: DType, shape: &[usize]) -> TensorRef {
    TensorRef { buf: buf.clone(), offset: off, dtype: dt, shape: shape.to_vec() }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

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

fn download(be: &MetalBackend, b: &Buffer, off: usize, bytes: usize) -> Vec<u8> {
    let mut v = vec![0u8; bytes];
    be.download(b, off, &mut v, 0).unwrap();
    v
}

struct Case {
    m: usize,
    n0: usize,
    n1: usize,
    k: usize,
    is: [f32; 2],
    f32_out: bool,
    w_off: usize,
    y_off: usize, // elements
}

fn dt(c: &Case) -> DType {
    if c.f32_out {
        DType::F32
    } else {
        DType::Bf16
    }
}

fn kernel_accepts(be: &MetalBackend, name: &str, call: &OpCall) -> bool {
    be.kernels().iter().find(|k| k.name == name).unwrap().accepts(&ShapeInfo::from_call(call))
}

fn run_pair(be: &MetalBackend, c: &Case, seed: u64) {
    let (m, n0, n1, k) = (c.m, c.n0, c.n1, c.k);
    let x = rand_bf16(m * k, seed);
    let (w0, w1) = (rand_e4m3(n0 * k, seed + 1), rand_e4m3(n1 * k, seed + 2));
    let (s0, s1) = ([0.0123f32, c.is[0]], [0.0071f32, c.is[1]]);
    let xb = upload(be, &x);
    let (w0b, w1b) = (up(be, c.w_off, &w0), up(be, c.w_off, &w1));
    let (s0b, s1b) = (up(be, 4, &f32_bytes(&s0)), up(be, 4, &f32_bytes(&s1)));
    let es = dt(c).size_bytes();
    let ysz = c.y_off * es + m * n0.max(n1) * es + 16;
    let mk = |n: usize, wb: &Buffer, sb: &Buffer| (t(wb, c.w_off, DType::Fp8E4m3, &[n, k]), t(sb, 4, DType::F32, &[2]));
    let (tw0, ts0) = mk(n0, &w0b, &s0b);
    let (tw1, ts1) = mk(n1, &w1b, &s1b);
    let xt = tref(&xb, 0, &[m, k]);
    let (py0, py1) = (poison(be, ysz), poison(be, ysz));
    let (sy0, sy1) = (poison(be, ysz), poison(be, ysz));
    let yo = c.y_off * es;
    let pair = OpCall::new(
        Op::MatmulFp8Pair,
        vec![xt.clone(), tw0.clone(), ts0.clone(), tw1.clone(), ts1.clone()],
        vec![t(&py0, yo, dt(c), &[m, n0]), t(&py1, yo, dt(c), &[m, n1])],
        OpAttrs::default(),
    );
    pair.check().unwrap();
    assert!(kernel_accepts(be, MATMUL_FP8_PAIR_GEMV, &pair), "the pair kernel does not accept m={m} n0={n0} k={k}");
    be.launch(MATMUL_FP8_PAIR_GEMV, &pair, 0).unwrap();
    for (tw, ts, sy, n) in [(&tw0, &ts0, &sy0, n0), (&tw1, &ts1, &sy1, n1)] {
        let single = OpCall::new(
            Op::MatmulFp8,
            vec![xt.clone(), tw.clone(), ts.clone()],
            vec![t(sy, yo, dt(c), &[m, n])],
            OpAttrs::default(),
        );
        be.launch(MATMUL_FP8_GEMV, &single, 0).unwrap();
    }
    be.synchronize(0).unwrap();
    let what =
        format!("m={m} n0={n0} n1={n1} k={k} is={:?} f32_out={} w_off={} y_off={}", c.is, c.f32_out, c.w_off, c.y_off);
    for (p, s) in [(&py0, &sy0), (&py1, &sy1)] {
        assert!(download(be, p, 0, ysz) == download(be, s, 0, ysz), "pair differs from the single GEMVs ({what})");
    }
    let nonzero = download(be, &py0, yo, m * n0 * es).iter().any(|b| *b != 0xEE);
    assert!(nonzero, "the pair wrote nothing ({what})");
}

#[test]
fn fp8_pair_is_bitwise_two_gemvs() {
    let be = backend();
    let mut seed = 100;
    // (m, n0, n1, k): the 35B shapes (qkv 8192 + z 4096 at K = 2048; k / v 512 + 512) and odd splits around tile edges
    let shapes = [
        (1, 8192, 4096, 2048),
        (1, 512, 512, 2048),
        (1, 33, 5, 96),
        (1, 4, 1, 16),
        (1, 31, 33, 160),
        (1, 32, 32, 64),
        (2, 130, 70, 256),
        (3, 300, 17, 512),
        (8, 512, 512, 2048),
        (8, 65, 9, 48),
    ];
    for &(m, n0, n1, k) in &shapes {
        for is in [[0.37f32, 2.0], [0.0, 0.5], [1.5, 0.0], [0.0, 0.0]] {
            for f32_out in [false, true] {
                seed += 7;
                run_pair(&be, &Case { m, n0, n1, k, is, f32_out, w_off: 8, y_off: 0 }, seed);
            }
        }
    }
    // weights at 0 mod 16 and outputs at odd element offsets
    run_pair(&be, &Case { m: 1, n0: 130, n1: 70, k: 256, is: [0.37, 2.0], f32_out: false, w_off: 16, y_off: 1 }, 900);
    run_pair(&be, &Case { m: 8, n0: 130, n1: 70, k: 256, is: [0.37, 2.0], f32_out: false, w_off: 16, y_off: 3 }, 901);
    run_pair(&be, &Case { m: 1, n0: 130, n1: 70, k: 256, is: [0.37, 2.0], f32_out: true, w_off: 8, y_off: 1 }, 902);
}

struct NormCase {
    n0: usize,
    n1: usize,
    k: usize,
    is: [f32; 2],
    eps: f32,
    w_offset: f32,
    skip: usize, // leading bf16 elements of x / res / outputs
}

fn run_norm_pair(be: &MetalBackend, c: &NormCase, seed: u64) {
    let (n0, n1, k) = (c.n0, c.n1, c.k);
    let x = rand_bf16(k + c.skip, seed);
    let res = rand_bf16(k + c.skip, seed + 1);
    let nw = rand_bf16(k + c.skip, seed + 2);
    let (w0, w1) = (rand_e4m3(n0 * k, seed + 3), rand_e4m3(n1 * k, seed + 4));
    let (s0, s1) = ([0.0123f32, c.is[0]], [0.0071f32, c.is[1]]);
    let (xb, rb, nb) = (upload(be, &x), upload(be, &res), upload(be, &nw));
    let (w0b, w1b) = (up(be, 8, &w0), up(be, 8, &w1));
    let (s0b, s1b) = (up(be, 4, &f32_bytes(&s0)), up(be, 4, &f32_bytes(&s1)));
    let sz = (k + c.skip + n0.max(n1)) * 2 + 16;
    let o = c.skip * 2;
    let bf = DType::Bf16;
    let tx = t(&xb, o, bf, &[1, k]);
    let tr = t(&rb, o, bf, &[1, k]);
    let tn = t(&nb, o, bf, &[k]);
    let tw0 = t(&w0b, 8, DType::Fp8E4m3, &[n0, k]);
    let tw1 = t(&w1b, 8, DType::Fp8E4m3, &[n1, k]);
    let (ts0, ts1) = (t(&s0b, 4, DType::F32, &[2]), t(&s1b, 4, DType::F32, &[2]));
    let attrs = OpAttrs { eps: c.eps, w_offset: c.w_offset, ..Default::default() };
    // fused
    let (fn_, fs, fy0, fy1) = (poison(be, sz), poison(be, sz), poison(be, sz), poison(be, sz));
    let fused = OpCall::new(
        Op::RmsNormAddMatmulFp8Pair,
        vec![tx.clone(), tr.clone(), tn.clone(), tw0.clone(), ts0.clone(), tw1.clone(), ts1.clone()],
        vec![t(&fn_, o, bf, &[1, k]), t(&fs, o, bf, &[1, k]), t(&fy0, o, bf, &[1, n0]), t(&fy1, o, bf, &[1, n1])],
        attrs.clone(),
    );
    fused.check().unwrap();
    assert!(kernel_accepts(be, RMSNORM_ADD_MATMUL_FP8_PAIR_GEMV, &fused), "the norm pair kernel does not accept k={k}");
    be.launch(RMSNORM_ADD_MATMUL_FP8_PAIR_GEMV, &fused, 0).unwrap();
    // unfused: rmsnorm_add, then the two GEMVs on its output
    let (un, us, uy0, uy1) = (poison(be, sz), poison(be, sz), poison(be, sz), poison(be, sz));
    let norm = OpCall::new(
        Op::RmsNormAdd,
        vec![tx.clone(), tr.clone(), tn.clone()],
        vec![t(&un, o, bf, &[1, k]), t(&us, o, bf, &[1, k])],
        attrs,
    );
    be.launch(RMSNORM_ADD_BF16, &norm, 0).unwrap();
    for (tw, ts, y, n) in [(&tw0, &ts0, &uy0, n0), (&tw1, &ts1, &uy1, n1)] {
        let single = OpCall::new(
            Op::MatmulFp8,
            vec![t(&un, o, bf, &[1, k]), tw.clone(), ts.clone()],
            vec![t(y, o, bf, &[1, n])],
            OpAttrs::default(),
        );
        be.launch(MATMUL_FP8_GEMV, &single, 0).unwrap();
    }
    be.synchronize(0).unwrap();
    let what = format!("n0={n0} n1={n1} k={k} is={:?} eps={} w_offset={} skip={}", c.is, c.eps, c.w_offset, c.skip);
    for (name, f, u) in [("normed", &fn_, &un), ("sum", &fs, &us), ("y0", &fy0, &uy0), ("y1", &fy1, &uy1)] {
        assert!(download(be, f, 0, sz) == download(be, u, 0, sz), "{name} differs from the unfused launches ({what})");
    }
}

#[test]
fn fp8_norm_pair_is_bitwise_norm_then_two_gemvs() {
    let be = backend();
    let mut seed = 500;
    // (n0, n1, k): the 35B attention input norm (k / v 512 + 512, hidden 2048) and widths that exercise every norm layout:
    // a row narrower than a threadgroup, not a multiple of 256, wider than 4096
    for &(n0, n1, k) in
        &[(512, 512, 2048), (130, 70, 256), (33, 5, 96), (64, 64, 16), (40, 24, 800), (96, 32, 4112), (8, 8, 8192)]
    {
        for is in [[0.37f32, 2.0], [0.0, 0.5], [0.0, 0.0]] {
            for (eps, w_offset) in [(1e-6f32, 0.0f32), (1e-6, 1.0)] {
                seed += 9;
                run_norm_pair(&be, &NormCase { n0, n1, k, is, eps, w_offset, skip: 0 }, seed);
            }
        }
    }
    run_norm_pair(&be, &NormCase { n0: 130, n1: 70, k: 2048, is: [0.37, 2.0], eps: 1e-6, w_offset: 0.0, skip: 1 }, 800);
    run_norm_pair(&be, &NormCase { n0: 130, n1: 70, k: 2048, is: [0.37, 2.0], eps: 1e-6, w_offset: 0.0, skip: 3 }, 801);
}

// ---- SigmoidMulMatmulFp8 -------------------------------------------------------------------------------------------------

const SIGMOID_MUL_MATMUL_FP8_GEMV: &str = "sigmoid_mul_matmul_fp8_gemv";
const GATED_RMSNORM_MATMUL_FP8_GEMV: &str = "gated_rmsnorm_matmul_fp8_gemv";

/// bf16 values of magnitude up to `amp` (so the gate saturates for large `amp`).
fn rand_bf16_amp(n: usize, seed: u64, amp: f32) -> Vec<u16> {
    rand_bf16(n, seed)
        .iter()
        .map(|v| intarsia_backend_metal::bench::f32_to_bf16(intarsia_backend_metal::bench::bf16_to_f32(*v) * amp))
        .collect()
}

fn run_sigmoid(be: &MetalBackend, m: usize, n: usize, k: usize, is: f32, amp: f32, skip: usize, seed: u64) {
    let a = rand_bf16_amp(m * k + skip, seed, 3.0);
    let g = rand_bf16_amp(m * k + skip, seed + 1, amp);
    let w = rand_e4m3(n * k, seed + 2);
    let sc = [0.0123f32, is];
    let (ab, gb) = (upload(be, &a), upload(be, &g));
    let wb = up(be, 8, &w);
    let sb = up(be, 4, &f32_bytes(&sc));
    let bf = DType::Bf16;
    let o = skip * 2;
    let (ta, tg) = (t(&ab, o, bf, &[m, k]), t(&gb, o, bf, &[m, k]));
    let tw = t(&wb, 8, DType::Fp8E4m3, &[n, k]);
    let ts = t(&sb, 4, DType::F32, &[2]);
    let sz = (m * n.max(k) + skip) * 2 + 16;
    let (fo, uo, gated) = (poison(be, sz), poison(be, sz), poison(be, sz));
    let fused = OpCall::new(
        Op::SigmoidMulMatmulFp8,
        vec![ta.clone(), tg.clone(), tw.clone(), ts.clone()],
        vec![t(&fo, o, bf, &[m, n])],
        OpAttrs::default(),
    );
    fused.check().unwrap();
    assert!(kernel_accepts(be, SIGMOID_MUL_MATMUL_FP8_GEMV, &fused), "sigmoid fold does not accept m={m} k={k}");
    be.launch(SIGMOID_MUL_MATMUL_FP8_GEMV, &fused, 0).unwrap();
    let mul = OpCall::new(Op::SigmoidMul, vec![ta, tg], vec![t(&gated, o, bf, &[m, k])], OpAttrs::default());
    be.launch("sigmoid_mul_bf16", &mul, 0).unwrap();
    let single = OpCall::new(
        Op::MatmulFp8,
        vec![t(&gated, o, bf, &[m, k]), tw, ts],
        vec![t(&uo, o, bf, &[m, n])],
        OpAttrs::default(),
    );
    be.launch(MATMUL_FP8_GEMV, &single, 0).unwrap();
    be.synchronize(0).unwrap();
    assert!(
        download(be, &fo, 0, sz) == download(be, &uo, 0, sz),
        "sigmoid fold differs from sigmoid_mul + GEMV (m={m} n={n} k={k} is={is} amp={amp} skip={skip})"
    );
}

#[test]
fn fp8_sigmoid_fold_is_bitwise_sigmoid_mul_then_gemv() {
    let be = backend();
    let mut seed = 2000;
    // (m, n, k): the 35B o_proj (2048 x 4096), odd sizes, the 8-row variant
    for &(m, n, k) in &[(1, 2048, 4096), (1, 33, 96), (1, 5, 16), (2, 130, 256), (8, 64, 2048), (3, 17, 48)] {
        for is in [0.37f32, 0.0] {
            for amp in [1.0f32, 40.0] {
                seed += 5;
                run_sigmoid(&be, m, n, k, is, amp, 0, seed);
            }
        }
    }
    run_sigmoid(&be, 1, 130, 256, 0.37, 1.0, 1, 3000);
    run_sigmoid(&be, 4, 130, 256, 0.37, 1.0, 3, 3001);
}

// ---- GatedRmsnormMatmulFp8 -----------------------------------------------------------------------------------------------

fn run_gated(
    be: &MetalBackend,
    rows: usize,
    d: usize,
    n: usize,
    is: f32,
    eps: f32,
    w_offset: f32,
    skip: usize,
    seed: u64,
) {
    let k = rows * d;
    let x = rand_bf16_amp(k + skip, seed, 2.0);
    let z = rand_bf16_amp(k + skip, seed + 1, 6.0);
    let nw = rand_bf16(d + skip, seed + 2);
    let w = rand_e4m3(n * k, seed + 3);
    let sc = [0.0123f32, is];
    let (xb, zb, nb) = (upload(be, &x), upload(be, &z), upload(be, &nw));
    let wb = up(be, 8, &w);
    let sb = up(be, 4, &f32_bytes(&sc));
    let bf = DType::Bf16;
    let o = skip * 2;
    let (tx, tz, tn) = (t(&xb, o, bf, &[rows, d]), t(&zb, o, bf, &[rows, d]), t(&nb, o, bf, &[d]));
    let tw = t(&wb, 8, DType::Fp8E4m3, &[n, k]);
    let ts = t(&sb, 4, DType::F32, &[2]);
    let sz = (k.max(n) + skip) * 2 + 16;
    let (fo, uo, normed, core) = (poison(be, sz), poison(be, sz), poison(be, sz), poison(be, sz));
    let attrs = OpAttrs { eps, w_offset, ..Default::default() };
    let fused = OpCall::new(
        Op::GatedRmsnormMatmulFp8,
        vec![tx.clone(), tz.clone(), tn.clone(), tw.clone(), ts.clone()],
        vec![t(&fo, o, bf, &[1, n])],
        attrs.clone(),
    );
    fused.check().unwrap();
    assert!(kernel_accepts(be, GATED_RMSNORM_MATMUL_FP8_GEMV, &fused), "gated fold does not accept rows={rows} d={d}");
    be.launch(GATED_RMSNORM_MATMUL_FP8_GEMV, &fused, 0).unwrap();
    let norm = OpCall::new(Op::RmsNorm, vec![tx, tn], vec![t(&normed, o, bf, &[rows, d])], attrs);
    be.launch("rmsnorm_bf16", &norm, 0).unwrap();
    let glu = OpCall::new(
        Op::SiluGlu,
        vec![tz, t(&normed, o, bf, &[rows, d])],
        vec![t(&core, o, bf, &[rows, d])],
        OpAttrs::default(),
    );
    be.launch("silu_glu_bf16", &glu, 0).unwrap();
    let single = OpCall::new(
        Op::MatmulFp8,
        vec![t(&core, o, bf, &[1, k]), tw, ts],
        vec![t(&uo, o, bf, &[1, n])],
        OpAttrs::default(),
    );
    be.launch(MATMUL_FP8_GEMV, &single, 0).unwrap();
    be.synchronize(0).unwrap();
    assert!(
        download(be, &fo, 0, sz) == download(be, &uo, 0, sz),
        "gated fold differs from rmsnorm + silu_glu + GEMV (rows={rows} d={d} n={n} is={is} eps={eps} w_offset={w_offset} skip={skip})"
    );
}

#[test]
fn fp8_gated_norm_fold_is_bitwise_norm_glu_then_gemv() {
    let be = backend();
    let mut seed = 4000;
    // (rows, d, n): the 35B GatedDeltaNet out_proj (32 value heads of 128, N = 2048) and other row widths / counts
    for &(rows, d, n) in
        &[(32, 128, 2048), (32, 128, 33), (1, 128, 64), (5, 64, 40), (12, 32, 24), (8, 256, 130), (3, 160, 17)]
    {
        for is in [0.37f32, 0.0] {
            for (eps, w_offset) in [(1e-6f32, 0.0f32), (1e-6, 1.0)] {
                seed += 11;
                run_gated(&be, rows, d, n, is, eps, w_offset, 0, seed);
            }
        }
    }
    run_gated(&be, 32, 128, 130, 0.37, 1e-6, 0.0, 1, 5000);
    run_gated(&be, 32, 128, 130, 0.37, 1e-6, 0.0, 3, 5001);
}
