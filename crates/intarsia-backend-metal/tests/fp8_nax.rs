//! `matmul_fp8_nax` (kernels/metal/gemm_nax_fp8.metal + `fp8_w_to_bf16`, metal4.0 library): the fp8 projection GEMM on the
//! neural accelerators. Properties the engine relies on:
//! * the output is the bf16 NAX GEMM of the decoded weights, scaled by `ws * is` in float and rounded once (so the only
//!   change against `matmul_fp8_mma` is the accumulation order), bitwise, bf16 and f32 outputs;
//! * a row's bits depend on neither the call's M, its position in the call, the tile, nor the output offset;
//! * run-to-run deterministic; a NaN weight code poisons its output column as in `matmul_fp8_mma`;
//! * registered for M > 8 only (the decode GEMV class and M <= 8 `mma` calls are untouched) and ahead of `mma`;
//! * within the oracle's tolerance (tests/quant.rs runs the shared cases through this kernel too).
//! The sweep against the cpu backend is `intarsia kernels test --backend metal --op matmul_fp8`.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use intarsia_backend_metal::bench::{bf16_to_f32, f32_to_bf16, Rng};
use intarsia_backend_metal::gemm_nax::NAX_VARIANTS;
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::dtype::fp8_e4m3_to_f32;
use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};

const NAX: &str = "matmul_fp8_nax";
const MMA: &str = "matmul_fp8_mma";

fn up(be: &MetalBackend, off: usize, data: &[u8]) -> Buffer {
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

fn e4m3(n: usize, seed: u64, nan_at: Option<usize>) -> Vec<u8> {
    let mut r = Rng(seed | 1);
    let mut w: Vec<u8> = (0..n)
        .map(|_| {
            let b = (r.next_u64() >> 32) as u8;
            if b & 0x7F == 0x7F { b & 0xFE } else { b }
        })
        .collect();
    if let Some(i) = nan_at {
        w[i] = 0x7F;
    }
    w
}

/// One `OpCall` of the kernel: x at `x_off` (rows `r0..r0+m` of `xb`), W at `w_off`, scales, output at `y_off`.
struct Run<'a> {
    be: &'a MetalBackend,
    k: usize,
    n: usize,
    xb: Buffer,
    wb: Buffer,
    sb: Buffer,
    w_off: usize,
}

impl<'a> Run<'a> {
    fn new(be: &'a MetalBackend, m_total: usize, n: usize, k: usize, scales: [f32; 2], seed: u64, nan_at: Option<usize>) -> Self {
        let x = rand_bf16(m_total * k, seed);
        let w = e4m3(n * k, seed + 1, nan_at);
        Run { be, k, n, xb: up(be, 0, as_bytes(&x)), wb: up(be, 8, &w), sb: up(be, 4, &f32_bytes(&scales)), w_off: 8 }
    }

    /// Output bytes of `kernel` for rows `r0..r0+m`, written at byte `y_off` of a fresh buffer (bf16 or f32).
    fn go(&self, kernel: &str, r0: usize, m: usize, y_off: usize, f32_out: bool, tile: Option<usize>) -> Vec<u8> {
        let (n, k) = (self.n, self.k);
        let esz = if f32_out { 4 } else { 2 };
        let ysz = y_off + m * n * esz + 16;
        let yb = self.be.alloc(ysz, MemKind::Device).unwrap();
        self.be.upload(&yb, 0, &vec![0xEEu8; ysz], 0).unwrap();
        let odt = if f32_out { DType::F32 } else { DType::Bf16 };
        let mut cl = OpCall::new(
            Op::MatmulFp8,
            vec![t(&self.xb, r0 * k * 2, DType::Bf16, &[m, k]), t(&self.wb, self.w_off, DType::Fp8E4m3, &[n, k]), t(&self.sb, 4, DType::F32, &[2])],
            vec![t(&yb, y_off, odt, &[m, n])],
            OpAttrs::default(),
        );
        let need = self.be.kernels().iter().find(|q| q.name == kernel).unwrap().scratch_bytes(&ShapeInfo::from_call(&cl));
        if need > 0 {
            let b = self.be.alloc(need, MemKind::Device).unwrap();
            self.be.upload(&b, 0, &vec![0xEEu8; need], 0).unwrap();
            cl.scratch = Some(TensorRef { buf: b, offset: 0, dtype: DType::U8, shape: vec![need] });
        }
        let launch = || self.be.launch(kernel, &cl, 0).unwrap();
        match tile {
            Some(i) => intarsia_backend_metal::fp8::with_nax_tile(i, launch),
            None => launch(),
        }
        self.be.synchronize(0).unwrap();
        let mut raw = vec![0u8; m * n * esz];
        self.be.download(&yb, y_off, &mut raw, 0).unwrap();
        raw
    }
}

fn bf(raw: &[u8]) -> Vec<u16> {
    raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

fn f32s(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Shapes of the 35B projections at chunk-sized and ragged M, plus odd tails of M, N and K.
const SHAPES: &[(usize, usize, usize)] =
    &[(9, 64, 64), (37, 130, 48), (130, 100, 80), (300, 512, 2048), (333, 777, 2048), (2048, 4096, 2048), (700, 2048, 4096)];

/// f32 out = the unscaled f32 product times the scale (one float multiply), bf16 out = that rounded to nearest even once.
#[test]
fn nax_scales_the_float_product_and_rounds_once() {
    let be = backend();
    for (si, &(m, n, k)) in SHAPES.iter().enumerate() {
        for is in [0.0f32, 0.37] {
            let ws = 0.0123f32;
            let scale = ws * if is > 0.0 { is } else { 1.0 };
            let unit = Run::new(&be, m, n, k, [1.0, 0.0], 100 + si as u64, None);
            let u = f32s(&unit.go(NAX, 0, m, 0, true, None));
            // same data, scales (ws, is): W8A8 re-quantises x, so only is == 0 can be derived from the unit run
            let scaled = Run { sb: up(&be, 4, &f32_bytes(&[ws, is])), xb: unit.xb.clone(), wb: unit.wb.clone(), ..unit };
            let f = f32s(&scaled.go(NAX, 0, m, 0, true, None));
            let b = bf(&scaled.go(NAX, 0, m, 0, false, None));
            if is == 0.0 {
                for i in 0..m * n {
                    assert_eq!(f[i].to_bits(), (u[i] * scale).to_bits(), "f32 m={m} n={n} k={k} elem {i}");
                    assert_eq!(b[i], f32_to_bf16(u[i] * scale), "bf16 m={m} n={n} k={k} elem {i}");
                }
            } else {
                // W8A8: the f32 and bf16 launches agree with each other (one rounding of the same float)
                for i in 0..m * n {
                    assert_eq!(b[i], f32_to_bf16(f[i]), "w8a8 bf16 vs f32 m={m} n={n} k={k} elem {i}");
                }
            }
        }
    }
}

/// W8A16 (is = 0), scale 1: bitwise the existing bf16 NAX GEMM on the decoded weights.
#[test]
fn nax_equals_the_bf16_nax_gemm_on_decoded_weights() {
    let be = backend();
    for (si, &(m, n, k)) in SHAPES.iter().enumerate() {
        let r = Run::new(&be, m, n, k, [1.0, 0.0], 300 + si as u64, None);
        let got = bf(&r.go(NAX, 0, m, 0, false, None));
        let w = e4m3(n * k, 300 + si as u64 + 1, None);
        let wbf: Vec<u16> = w.iter().map(|b| f32_to_bf16(fp8_e4m3_to_f32(*b))).collect();
        let x = rand_bf16(m * k, 300 + si as u64);
        let (xb, wb) = (upload(&be, &x), upload(&be, &wbf));
        let yb = be.alloc(m * n * 2, MemKind::Device).unwrap();
        let cl = matmul_call(tref(&xb, 0, &[m, k]), tref(&wb, 0, &[n, k]), tref(&yb, 0, &[m, n]));
        be.launch("matmul_bf16_nax", &cl, 0).unwrap();
        be.synchronize(0).unwrap();
        let mut raw = vec![0u8; m * n * 2];
        be.download(&yb, 0, &mut raw, 0).unwrap();
        let want = bf(&raw);
        let bad = (0..m * n).filter(|&i| got[i] != want[i]).count();
        assert_eq!(bad, 0, "m={m} n={n} k={k}: {bad} elements differ from matmul_bf16_nax on the decoded weights");
    }
}

/// The bits of a row do not depend on the call's M, the row's position in it, the tile, or the output's byte offset.
#[test]
fn nax_rows_do_not_depend_on_the_call() {
    let be = backend();
    let (m_big, n, k) = (1100usize, 1024usize, 2048usize);
    let r = Run::new(&be, m_big, n, k, [0.0123, 0.0], 7, None);
    let want = bf(&r.go(NAX, 0, m_big, 0, false, None));
    for tile in 0..NAX_VARIANTS.len() {
        let got = bf(&r.go(NAX, 0, m_big, 0, false, Some(tile)));
        assert!(got == want, "{} differs from the picked tile on a {m_big}-row call", NAX_VARIANTS[tile].shader);
    }
    for (r0, rows) in [(0usize, 9usize), (7, 10), (40, 11), (1000, 15), (17, 17), (100, 64), (819, 205), (1091, 9), (512, 512), (3, 1097)] {
        for tile in [None, Some(0), Some(2), Some(3)] {
            let y_off = 2 * (r0 % 3);
            let got = bf(&r.go(NAX, r0, rows, y_off, false, tile));
            assert!(got == want[r0 * n..(r0 + rows) * n], "rows {r0}..{} of a {rows}-row call (tile {tile:?}) differ", r0 + rows);
            let gf = f32s(&r.go(NAX, r0, rows, 4 * (r0 % 3), true, tile));
            let wf = f32s(&r.go(NAX, 0, m_big, 0, true, None));
            assert!(
                gf.iter().zip(&wf[r0 * n..(r0 + rows) * n]).all(|(a, b)| a.to_bits() == b.to_bits()),
                "f32 rows {r0}..{} of a {rows}-row call (tile {tile:?}) differ",
                r0 + rows
            );
        }
    }
}

#[test]
fn nax_is_run_to_run_deterministic() {
    let be = backend();
    let (m, n, k) = (333usize, 777usize, 2048usize);
    let r = Run::new(&be, m, n, k, [0.0123, 0.37], 11, None);
    let first = r.go(NAX, 0, m, 0, false, None);
    for _ in 0..4 {
        assert!(r.go(NAX, 0, m, 0, false, None) == first, "run-to-run difference");
    }
}

/// A NaN weight code (0x7F) poisons its output column in every row, like `matmul_fp8_mma` (and no other column).
#[test]
fn nax_nan_weight_poisons_its_column_only() {
    let be = backend();
    let (m, n, k) = (70usize, 130usize, 80usize);
    let (bad_n, bad_k) = (37usize, 53usize);
    let r = Run::new(&be, m, n, k, [0.5, 0.0], 21, Some(bad_n * k + bad_k));
    let a = bf(&r.go(NAX, 0, m, 0, false, None));
    let b = bf(&r.go(MMA, 0, m, 0, false, None));
    for i in 0..m * n {
        let (na, nb) = (bf16_to_f32(a[i]).is_nan(), bf16_to_f32(b[i]).is_nan());
        assert_eq!(na, nb, "elem {i}: nax NaN {na}, mma NaN {nb}");
        assert_eq!(na, i % n == bad_n, "elem {i}: NaN only in column {bad_n}");
    }
}

#[test]
fn nax_is_registered_for_m_above_the_gemv_rows_only() {
    let be = backend();
    let ks = be.kernels();
    let nax = ks.iter().find(|k| k.name == NAX).unwrap_or_else(|| panic!("{NAX} not registered"));
    let mma = ks.iter().find(|k| k.name == MMA).unwrap();
    let gemv = ks.iter().find(|k| k.name == "matmul_fp8_gemv").unwrap();
    assert_eq!(nax.op, Op::MatmulFp8);
    let shape = |m: usize, n: usize, k: usize| ShapeInfo {
        m,
        n,
        k,
        act: Some(DType::Bf16),
        weight: Some(DType::Fp8E4m3),
        out: Some(DType::Bf16),
        ..Default::default()
    };
    for m in 1..=8 {
        assert!(!nax.accepts(&shape(m, 8192, 2048)) && !nax.accepts(&shape(m, 2048, 4096)), "M = {m} stays on the decode kernels");
    }
    for m in [9usize, 15, 16, 300, 2048] {
        assert!(nax.accepts(&shape(m, 8192, 2048)) && nax.accepts(&shape(m, 2048, 4096)), "M = {m}");
    }
    assert!(!nax.accepts(&shape(300, 8192, 40)), "K % 16 != 0 stays on mma only if mma takes it");
    assert!(nax.priority > mma.priority && nax.priority < gemv.priority, "above mma, below the GEMV class");
    assert!(nax.scratch_bytes(&shape(2048, 8192, 2048)) >= (2048 + 8192) * 2048 * 2);
}
