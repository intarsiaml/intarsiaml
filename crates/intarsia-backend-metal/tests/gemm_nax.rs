//! The NAX bf16 GEMM (kernels/metal/gemm_nax_bf16.metal, metal4.0 library): against an f64 reference at the Qwen3-8B
//! shapes including M tails and odd operand offsets, row independence (a row's bits do not depend on the call's M, the
//! row's position or the tile configuration), strict against relaxed precision, and the registration. The cpu-oracle
//! admission is `intarsia kernels test --backend metal`; the metal4.0 fallback is tests/metal4_fallback.rs.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use intarsia_backend_metal::gemm_nax::{encode_nax, NaxParams, NaxVariant, NAX_VARIANTS};
use intarsia_backend_metal::kernels::{MATMUL_BF16, MATMUL_BF16_NAX, MATMUL_BF16_NAX_RELAXED};
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, Buffer, MemKind};
use intarsia_core::DType;

fn upload_at(be: &MetalBackend, off: usize, data: &[u16]) -> Buffer {
    let b = be.alloc(off + data.len() * 2 + 16, MemKind::Device).unwrap();
    be.upload(&b, 0, &vec![0xEEu8; off + data.len() * 2 + 16], 0).unwrap();
    be.upload(&b, off, as_bytes(data), 0).unwrap();
    b
}

fn read_u16(be: &MetalBackend, b: &Buffer, off: usize, n: usize) -> Vec<u16> {
    let mut raw = vec![0u8; n * 2];
    be.download(b, off, &mut raw, 0).unwrap();
    raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

/// Run variant `v` on x `[m,k]` at byte offset `xo`, w `[n,k]` at `wo`, y `[m,n]` at `yo` of their buffers (the NAX
/// kernels bind at the 4-byte aligned offset and take the leftover element as a skip, exactly like `launch_nax`).
fn run_variant(
    be: &MetalBackend,
    v: &NaxVariant,
    (xb, xo): (&Buffer, usize),
    (wb, wo): (&Buffer, usize),
    (yb, yo): (&Buffer, usize),
    (m, n, k): (usize, usize, usize),
) {
    let pso = be.pso(v.shader).unwrap_or_else(|| panic!("no pipeline {}", v.shader));
    let p = NaxParams {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        swz_log: 0,
        s0: ((xo & 3) / 2) as u32,
        s1: ((wo & 3) / 2) as u32,
        s2: ((yo & 3) / 2) as u32,
        pad: 0,
    };
    fn mtl(b: &Buffer) -> &intarsia_backend_metal::MetalBuffer {
        b.downcast::<intarsia_backend_metal::MetalBuffer>().unwrap()
    }
    let (xm, wm, ym) = (mtl(xb), mtl(wb), mtl(yb));
    be.with_encoder(|enc| encode_nax(enc, pso, v, (xm.mtl(), xo & !3), (wm.mtl(), wo & !3), (ym.mtl(), yo & !3), &p)).unwrap();
    be.synchronize(0).unwrap();
}

fn sample_rows(m: usize) -> Vec<usize> {
    let mut r = vec![0, 1, 15.min(m - 1), 16.min(m - 1), m / 3, m / 2, m.saturating_sub(17), m - 2.min(m), m - 1];
    r.sort();
    r.dedup();
    r
}

/// (m, n, k, x offset, w offset, y offset): Qwen3-8B shapes at prefill chunk tails, N / M / K tile edges, odd and 8-byte
/// aligned offsets (a safetensors shard with header length 8 mod 16 puts every weight at 8 mod 16).
const CASES: &[(usize, usize, usize, usize, usize, usize)] = &[
    (16, 64, 64, 0, 0, 0),
    (17, 4096, 4096, 0, 0, 0),
    (205, 1024, 4096, 0, 8, 0),
    (222, 4096, 4096, 0, 0, 0),
    (256, 4096, 12288, 0, 0, 0),
    (973, 1024, 4096, 0, 6, 2),
    (1024, 12288, 4096, 0, 8, 0),
    (130, 100, 4096, 2, 2, 2),
    (33, 65, 128, 0, 2, 0),
    (200, 130, 1024, 2, 0, 2),
    (64, 33, 64, 0, 0, 0),
];

#[test]
fn nax_matches_f64_reference() {
    let be = backend();
    assert!(be.has_metal4(), "metal4.0 library not loaded on this machine");
    for (si, &(m, n, k, xo, wo, yo)) in CASES.iter().enumerate() {
        let x = rand_bf16(m * k, 10 + si as u64);
        let w: Vec<u16> = rand_bf16(n * k, 99 + si as u64);
        // w ~ U[-1,1) / sqrt(k) like the admission sweep: keeps outputs O(1) so the f32 accumulation noise is small
        let scale = 1.0 / (k as f32).sqrt();
        let w: Vec<u16> = w
            .iter()
            .map(|&b| intarsia_backend_metal::bench::f32_to_bf16(intarsia_backend_metal::bench::bf16_to_f32(b) * scale))
            .collect();
        let (xb, wb) = (upload_at(&be, xo, &x), upload_at(&be, wo, &w));
        for v in NAX_VARIANTS {
            let yb = be.alloc(yo + m * n * 2 + 16, MemKind::Device).unwrap();
            be.upload(&yb, 0, &vec![0xEEu8; yo + m * n * 2 + 16], 0).unwrap();
            run_variant(&be, v, (&xb, xo), (&wb, wo), (&yb, yo), (m, n, k));
            let y = read_u16(&be, &yb, yo, m * n);
            for &mi in &sample_rows(m) {
                let refs = reference(&w, &x[mi * k..(mi + 1) * k], k, n);
                check(&y[mi * n..(mi + 1) * n], &refs, 0..n, &format!("{} m={m} n={n} k={k} off={xo},{wo},{yo} row {mi}", v.shader));
            }
            // guard bytes around y untouched (stores past the M / N edge would land here)
            let raw = read_u16(&be, &yb, 0, (yo + m * n * 2 + 16) / 2);
            assert!(
                raw[..yo / 2].iter().all(|v| *v == 0xEEEE) && raw[(yo / 2 + m * n)..].iter().all(|v| *v == 0xEEEE),
                "{}: guard bytes of y overwritten (m={m} n={n} k={k})",
                v.shader
            );
        }
    }
}

/// K that is not a multiple of the 16-wide K step: the primitive's dynamic K must zero-fill the tail. Decides whether
/// the registration predicate may drop `k % 16 == 0`.
#[test]
fn nax_k_tails() {
    let be = backend();
    for &(m, n, k) in &[(40usize, 48usize, 70usize), (40, 48, 33), (20, 130, 100), (64, 64, 18), (17, 31, 17)] {
        let x = rand_bf16(m * k, 41 + k as u64);
        let w = rand_bf16(n * k, 42 + k as u64);
        let (xb, wb) = (upload_at(&be, 0, &x), upload_at(&be, 0, &w));
        let yb = be.alloc(m * n * 2, MemKind::Device).unwrap();
        run_variant(&be, &NAX_VARIANTS[1], (&xb, 0), (&wb, 0), (&yb, 0), (m, n, k));
        let y = read_u16(&be, &yb, 0, m * n);
        for mi in 0..m {
            let refs = reference(&w, &x[mi * k..(mi + 1) * k], k, n);
            check(&y[mi * n..(mi + 1) * n], &refs, 0..n, &format!("k tail m={m} n={n} k={k} row {mi}"));
        }
    }
}

/// The bits of a row depend on neither the call's M, the row's position in the call, the tile configuration nor the
/// precision flag: rows of a 1024-row call equal the same rows computed in calls of other sizes at other offsets.
#[test]
fn nax_rows_do_not_depend_on_the_call() {
    let be = backend();
    let (n, k) = (1024usize, 4096usize);
    let m_big = 1024usize;
    let x = rand_bf16(m_big * k, 5);
    let w = rand_bf16(n * k, 6);
    let (xb, wb) = (upload_at(&be, 0, &x), upload_at(&be, 0, &w));
    let base = &NAX_VARIANTS[0];
    let yb = be.alloc(m_big * n * 2, MemKind::Device).unwrap();
    run_variant(&be, base, (&xb, 0), (&wb, 0), (&yb, 0), (m_big, n, k));
    let want = read_u16(&be, &yb, 0, m_big * n);
    // every tile configuration and precision flag gives the same bits for the whole call
    for v in NAX_VARIANTS {
        let y2 = be.alloc(m_big * n * 2, MemKind::Device).unwrap();
        run_variant(&be, v, (&xb, 0), (&wb, 0), (&y2, 0), (m_big, n, k));
        assert!(read_u16(&be, &y2, 0, m_big * n) == want, "{} differs from {}", v.shader, base.shader);
    }
    // calls of other sizes over row ranges of x, outputs written at odd offsets
    for (r0, rows) in [(0usize, 5usize), (17, 17), (100, 64), (819, 205), (1019, 5), (512, 512), (3, 1021)] {
        for v in [&NAX_VARIANTS[0], &NAX_VARIANTS[1], &NAX_VARIANTS[7]] {
            let yo = 2 * (r0 % 3);
            let y2 = be.alloc(yo + rows * n * 2 + 16, MemKind::Device).unwrap();
            run_variant(&be, v, (&xb, r0 * k * 2), (&wb, 0), (&y2, yo), (rows, n, k));
            let got = read_u16(&be, &y2, yo, rows * n);
            assert!(
                got == want[r0 * n..(r0 + rows) * n],
                "{}: rows {r0}..{} of a {rows}-row call differ from the 1024-row call",
                v.shader,
                r0 + rows
            );
        }
    }
}

/// Repeated runs give the same bits (no atomics, no split-K).
#[test]
fn nax_is_run_to_run_deterministic() {
    let be = backend();
    let (m, n, k) = (333usize, 777usize, 2048usize);
    let (x, w) = (rand_bf16(m * k, 1), rand_bf16(n * k, 2));
    let (xb, wb) = (upload_at(&be, 0, &x), upload_at(&be, 0, &w));
    let mut first: Option<Vec<u16>> = None;
    for _ in 0..4 {
        let yb = be.alloc(m * n * 2, MemKind::Device).unwrap();
        run_variant(&be, &NAX_VARIANTS[1], (&xb, 0), (&wb, 0), (&yb, 0), (m, n, k));
        let y = read_u16(&be, &yb, 0, m * n);
        match &first {
            None => first = Some(y),
            Some(f) => assert!(*f == y, "run-to-run difference"),
        }
    }
}

#[test]
fn nax_kernels_are_registered_for_prefill_shapes_only() {
    let be = backend();
    let ks = be.kernels();
    for name in [MATMUL_BF16_NAX, MATMUL_BF16_NAX_RELAXED] {
        let k = ks.iter().find(|k| k.name == name).unwrap_or_else(|| panic!("{name} not registered"));
        let shape = |m: usize, k_: usize| intarsia_core::ShapeInfo {
            m,
            n: 1024,
            k: k_,
            act: Some(DType::Bf16),
            weight: Some(DType::Bf16),
            out: Some(DType::Bf16),
            ..Default::default()
        };
        assert!(k.accepts(&shape(16, 4096)) && k.accepts(&shape(1024, 12288)), "{name}: prefill shapes");
        assert!(!k.accepts(&shape(1, 4096)) && !k.accepts(&shape(15, 4096)), "{name}: small M is left to the GEMV / tiled GEMM");
        assert!(k.priority < ks.iter().find(|k| k.name == MATMUL_BF16).unwrap().priority + 1, "{name}: not selected in the spike");
    }
}

/// Launch through `Backend::launch` (the registry path): both registered kernels, odd offsets, bitwise equal to each other.
#[test]
fn nax_launch_through_the_backend() {
    let be = backend();
    let (m, n, k) = (205usize, 4096usize, 4096usize);
    let (xo, wo, yo) = (2usize, 8usize, 2usize);
    let x = rand_bf16(m * k, 31);
    let scale = 1.0 / (k as f32).sqrt();
    let w: Vec<u16> = rand_bf16(n * k, 32)
        .iter()
        .map(|&b| intarsia_backend_metal::bench::f32_to_bf16(intarsia_backend_metal::bench::bf16_to_f32(b) * scale))
        .collect();
    let (xb, wb) = (upload_at(&be, xo, &x), upload_at(&be, wo, &w));
    let mut outs = Vec::new();
    for name in [MATMUL_BF16_NAX, MATMUL_BF16_NAX_RELAXED, MATMUL_BF16] {
        let yb = be.alloc(yo + m * n * 2 + 16, MemKind::Device).unwrap();
        let call = matmul_call(tref(&xb, xo, &[m, k]), tref(&wb, wo, &[n, k]), tref(&yb, yo, &[m, n]));
        be.launch(name, &call, 0).unwrap();
        be.synchronize(0).unwrap();
        let y = read_u16(&be, &yb, yo, m * n);
        for &mi in &[0usize, 77, 204] {
            let refs = reference(&w, &x[mi * k..(mi + 1) * k], k, n);
            check(&y[mi * n..(mi + 1) * n], &refs, 0..n, &format!("{name} row {mi}"));
        }
        outs.push(y);
    }
    assert!(outs[0] == outs[1], "strict and relaxed differ");
}
