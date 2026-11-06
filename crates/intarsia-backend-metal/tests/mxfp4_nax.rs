//! `matmul_mxfp4_nax` (`mxfp4_w_to_bf16` + the fp8 NAX GEMM, metal4.0 library): the MXFP4 prefill GEMM. Properties:
//! * with `act` = 0 (W4A16) the output is bitwise the bf16 NAX GEMM of the decoded weights (e2m1 * 2^(scale - 127), exact in
//!   bf16), bf16 and f32 outputs, with M, N and K tails;
//! * a row's bits depend on neither the call's M, its position in the call, nor the tile;
//! * with `act` = 1 (W4A4) the activations are the same QDQ'd bf16 the GEMV uses, and the result stays within the
//!   accumulation-order difference of `matmul_mxfp4_gemv`;
//! * registered for M > 8 only, ahead of the GEMV.
//! The sweep against the cpu backend is `intarsia kernels test --backend metal --op matmul_mxfp4`.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use intarsia_backend_metal::bench::{f32_to_bf16, Rng};
use intarsia_backend_metal::gemm_nax::NAX_VARIANTS;
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};

const NAX: &str = "matmul_mxfp4_nax";
const GEMV: &str = "matmul_mxfp4_gemv";
const FP4: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

fn up(be: &MetalBackend, data: &[u8]) -> Buffer {
    let b = be.alloc(data.len() + 16, MemKind::Device).unwrap();
    be.upload(&b, 0, &vec![0xEEu8; data.len() + 16], 0).unwrap();
    be.upload(&b, 0, data, 0).unwrap();
    b
}

fn t(buf: &Buffer, off: usize, dt: DType, shape: &[usize]) -> TensorRef {
    TensorRef { buf: buf.clone(), offset: off, dtype: dt, shape: shape.to_vec() }
}

/// Random packed codes `[n, k/2]` and scale bytes `[n, k/32]` (e8m0, 2^-8 .. 2^3), plus the decoded bf16 weights `[n, k]`.
fn weights(n: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<u8>, Vec<u16>) {
    let mut r = Rng(seed | 1);
    let w: Vec<u8> = (0..n * k / 2).map(|_| (r.next_u64() >> 40) as u8).collect();
    let sc: Vec<u8> = (0..n * k / 32).map(|_| 119 + ((r.next_u64() >> 40) % 12) as u8).collect();
    let mut dec = vec![0u16; n * k];
    for row in 0..n {
        for e in 0..k {
            let byte = w[row * (k / 2) + e / 2];
            let code = if e % 2 == 0 { byte & 0xF } else { byte >> 4 };
            let s = f32::from_bits((sc[row * (k / 32) + e / 32] as u32) << 23);
            dec[row * k + e] = f32_to_bf16(FP4[code as usize] * s);
        }
    }
    (w, sc, dec)
}

struct Run<'a> {
    be: &'a MetalBackend,
    n: usize,
    k: usize,
    xb: Buffer,
    wb: Buffer,
    sb: Buffer,
}

impl<'a> Run<'a> {
    fn new(be: &'a MetalBackend, m_total: usize, n: usize, k: usize, seed: u64) -> (Self, Vec<u16>, Vec<u16>) {
        let x = rand_bf16(m_total * k, seed);
        let (w, sc, dec) = weights(n, k, seed + 1);
        (Run { be, n, k, xb: up(be, as_bytes(&x)), wb: up(be, &w), sb: up(be, &sc) }, x, dec)
    }

    /// Output bytes of `kernel` for rows `r0..r0+m` of x, written at byte `y_off` of a fresh buffer.
    fn go(
        &self,
        kernel: &str,
        act: f32,
        r0: usize,
        m: usize,
        y_off: usize,
        f32_out: bool,
        tile: Option<usize>,
    ) -> Vec<u8> {
        let (n, k) = (self.n, self.k);
        let esz = if f32_out { 4 } else { 2 };
        let ysz = y_off + m * n * esz + 16;
        let yb = self.be.alloc(ysz, MemKind::Device).unwrap();
        self.be.upload(&yb, 0, &vec![0xEEu8; ysz], 0).unwrap();
        let ab = up(self.be, &act.to_le_bytes());
        let odt = if f32_out { DType::F32 } else { DType::Bf16 };
        let mut cl = OpCall::new(
            Op::MatmulMxfp4,
            vec![
                t(&self.xb, r0 * k * 2, DType::Bf16, &[m, k]),
                t(&self.wb, 0, DType::U8, &[n, k / 2]),
                t(&self.sb, 0, DType::U8, &[n, k / 32]),
                t(&ab, 0, DType::F32, &[1]),
            ],
            vec![t(&yb, y_off, odt, &[m, n])],
            OpAttrs::default(),
        );
        let need =
            self.be.kernels().iter().find(|q| q.name == kernel).unwrap().scratch_bytes(&ShapeInfo::from_call(&cl));
        let b = self.be.alloc(need + 8, MemKind::Device).unwrap();
        self.be.upload(&b, 0, &vec![0xEEu8; need + 8], 0).unwrap();
        // an unaligned scratch start (4-byte aligned only) must work: the launcher pads to 16 itself
        cl.scratch = Some(TensorRef { buf: b, offset: 4, dtype: DType::U8, shape: vec![need] });
        let launch = || self.be.launch(kernel, &cl, 0).unwrap();
        match tile {
            Some(i) => intarsia_backend_metal::mxfp4::with_nax_tile(i, launch),
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

/// M, N, K tails and the 27B's projection widths (hidden 5120).
const SHAPES: &[(usize, usize, usize)] =
    &[(9, 64, 64), (37, 130, 96), (130, 100, 160), (300, 1000, 512), (257, 5120, 5120), (64, 6144, 5120)];

#[test]
fn nax_w4a16_equals_the_bf16_nax_gemm_on_decoded_weights() {
    let be = backend();
    for (si, &(m, n, k)) in SHAPES.iter().enumerate() {
        let (r, x, dec) = Run::new(&be, m, n, k, 500 + si as u64);
        let got = bf(&r.go(NAX, 0.0, 0, m, 0, false, None));
        let (xb, wb) = (upload(&be, &x), upload(&be, &dec));
        let yb = be.alloc(m * n * 2, MemKind::Device).unwrap();
        let cl = matmul_call(tref(&xb, 0, &[m, k]), tref(&wb, 0, &[n, k]), tref(&yb, 0, &[m, n]));
        be.launch("matmul_bf16_nax", &cl, 0).unwrap();
        be.synchronize(0).unwrap();
        let mut raw = vec![0u8; m * n * 2];
        be.download(&yb, 0, &mut raw, 0).unwrap();
        let want = bf(&raw);
        let bad = (0..m * n).filter(|&i| got[i] != want[i]).count();
        assert_eq!(bad, 0, "m={m} n={n} k={k}: {bad} elements differ from matmul_bf16_nax on the decoded weights");
        // f32 out is the same float product, rounded once to bf16 by the bf16 form
        let f = f32s(&r.go(NAX, 0.0, 0, m, 0, true, None));
        let bad = (0..m * n).filter(|&i| f32_to_bf16(f[i]) != got[i]).count();
        assert_eq!(bad, 0, "m={m} n={n} k={k}: {bad} f32 elements round to a different bf16");
    }
}

#[test]
fn nax_rows_do_not_depend_on_the_call() {
    let be = backend();
    let (m_big, n, k) = (700usize, 1024usize, 1024usize);
    for act in [0.0f32, 1.0] {
        let (r, _, _) = Run::new(&be, m_big, n, k, 9);
        let want = bf(&r.go(NAX, act, 0, m_big, 0, false, None));
        for tile in 0..NAX_VARIANTS.len() {
            let got = bf(&r.go(NAX, act, 0, m_big, 0, false, Some(tile)));
            assert!(got == want, "{} differs from the picked tile (act {act})", NAX_VARIANTS[tile].shader);
        }
        for (r0, rows) in [(0usize, 9usize), (7, 10), (40, 11), (600, 15), (100, 64), (219, 205), (3, 697)] {
            for tile in [None, Some(0), Some(2)] {
                let got = bf(&r.go(NAX, act, r0, rows, 2 * (r0 % 3), false, tile));
                assert!(
                    got == want[r0 * n..(r0 + rows) * n],
                    "rows {r0}..{} (act {act}, tile {tile:?}) differ",
                    r0 + rows
                );
            }
        }
    }
}

/// Against the GEMV (same QDQ'd bf16 activations, fp32 FMA chain): only the K accumulation order differs.
#[test]
fn nax_matches_the_gemv_within_accumulation_order() {
    let be = backend();
    for act in [0.0f32, 1.0] {
        for &(m, n, k) in &[(40usize, 512usize, 1024usize), (300, 1000, 512), (129, 2048, 5120)] {
            let (r, _, _) = Run::new(&be, m, n, k, 77);
            let a = f32s(&r.go(NAX, act, 0, m, 0, true, None));
            let b = f32s(&r.go(GEMV, act, 0, m, 0, true, None));
            let (mut num, mut den) = (0f64, 0f64);
            for i in 0..m * n {
                num += ((a[i] - b[i]) as f64).powi(2);
                den += (b[i] as f64).powi(2);
            }
            let rel = (num / den).sqrt();
            assert!(rel < 1e-5, "act {act} m={m} n={n} k={k}: relative RMS difference {rel:e}");
        }
    }
}

#[test]
fn nax_is_registered_for_m_above_the_gemv_rows_only() {
    let be = backend();
    let ks = be.kernels();
    let nax = ks.iter().find(|k| k.name == NAX).unwrap_or_else(|| panic!("{NAX} not registered"));
    let gemv = ks.iter().find(|k| k.name == GEMV).unwrap();
    assert_eq!(nax.op, Op::MatmulMxfp4);
    let shape = |m: usize, n: usize, k: usize| ShapeInfo {
        m,
        n,
        k,
        act: Some(DType::Bf16),
        weight: Some(DType::U8),
        out: Some(DType::Bf16),
        ..Default::default()
    };
    for m in 1..=8 {
        assert!(!nax.accepts(&shape(m, 17408, 5120)), "M = {m} stays on the GEMV");
    }
    for m in [9usize, 15, 256, 2048] {
        assert!(nax.accepts(&shape(m, 17408, 5120)) && nax.accepts(&shape(m, 5120, 6144)), "M = {m}");
    }
    assert!(!nax.accepts(&ShapeInfo { mxfp4_a8: true, ..shape(300, 5120, 5120) }), "W4A8 is not served");
    assert!(nax.priority > gemv.priority, "above the GEMV for M > 8");
    assert!(nax.scratch_bytes(&shape(2048, 17408, 5120)) >= (2048 + 17408) * 5120 * 2);
}
