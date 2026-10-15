//! The NAX bf16 GEMM (kernels/metal/gemm_nax_bf16.metal, metal4.0 library): against an f64 reference at the Qwen3-8B
//! shapes including M tails and odd operand offsets, row independence (a row's bits do not depend on the call's M, the
//! row's position or the tile configuration), strict against relaxed precision, and the registration. The cpu-oracle
//! admission is `intarsia kernels test --backend metal`; the metal4.0 fallback is tests/metal4_fallback.rs.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use intarsia_backend_metal::gemm_nax::{
    encode_nax, pick_variant, NaxParams, NaxVariant, NAX_MIN_M, NAX_VARIANTS,
};
use intarsia_backend_metal::kernels::{MATMUL_BF16, MATMUL_BF16_NAX, MATMUL_PAIR_BF16_NAX};
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

/// The bits of a row depend on neither the call's M, the row's position in the call, nor the tile configuration:
/// rows of a 1024-row call equal the same rows computed in calls of other sizes at other offsets.
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
    // every tile configuration gives the same bits for the whole call
    for v in NAX_VARIANTS {
        let y2 = be.alloc(m_big * n * 2, MemKind::Device).unwrap();
        run_variant(&be, v, (&xb, 0), (&wb, 0), (&y2, 0), (m_big, n, k));
        assert!(read_u16(&be, &y2, 0, m_big * n) == want, "{} differs from {}", v.shader, base.shader);
    }
    // calls of other sizes over row ranges of x, outputs written at odd offsets
    for (r0, rows) in [(0usize, 5usize), (17, 17), (100, 64), (819, 205), (1019, 5), (512, 512), (3, 1021)] {
        for v in NAX_VARIANTS {
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
    let gemm = ks.iter().find(|k| k.name == MATMUL_BF16).unwrap();
    let gemv = ks.iter().find(|k| k.name == "matmul_bf16_gemv").unwrap();
    for (name, pair) in [(MATMUL_BF16_NAX, false), (MATMUL_PAIR_BF16_NAX, true)] {
        let k = ks.iter().find(|k| k.name == name).unwrap_or_else(|| panic!("{name} not registered"));
        assert_eq!(k.op, if pair { intarsia_core::Op::MatmulPair } else { intarsia_core::Op::Matmul });
        let shape = |m: usize, k_: usize| intarsia_core::ShapeInfo {
            m,
            n: 1024,
            k: k_,
            act: Some(DType::Bf16),
            weight: Some(DType::Bf16),
            out: Some(DType::Bf16),
            ..Default::default()
        };
        assert!(k.accepts(&shape(NAX_MIN_M, 4096)) && k.accepts(&shape(1024, 12288)), "{name}: prefill shapes");
        assert!(!k.accepts(&shape(1, 4096)) && !k.accepts(&shape(NAX_MIN_M - 1, 4096)), "{name}: small M is left to the GEMV / tiled GEMM");
        assert!(k.priority > gemm.priority && k.priority < gemv.priority, "{name}: above the tiled GEMM, below the GEMV class");
    }
}

#[test]
fn nax_tile_pick_is_by_shape() {
    let name = |m: usize, n: usize| pick_variant(m, n).shader;
    // P1 rule outside 128..=512 rows: 64x64 / four simdgroups, or 128x64 when it leaves enough threadgroups
    assert_eq!(name(16, 12288), "gemm_nax_t64x64w4");
    assert_eq!(name(64, 12288), "gemm_nax_t64x64w4");
    assert_eq!(name(127, 24576), "gemm_nax_t64x64w4");
    assert_eq!(name(1024, 24576), "gemm_nax_t128x64w4");
    assert_eq!(name(2048, 4096), "gemm_nax_t128x64w4");
    assert_eq!(name(973, 4096), "gemm_nax_t128x64w4");
    assert_eq!(name(1024, 1024), "gemm_nax_t64x64w4");
    // chunk-sized rows: k + v pair on 32x32 / one simdgroup
    assert_eq!(name(256, 2048), "gemm_nax_t32x32w1");
    assert_eq!(name(222, 2048), "gemm_nax_t32x32w1");
    assert_eq!(name(128, 4096), "gemm_nax_t32x32w1");
    // q / o, down, gate + up at a multiple of 64 rows: 64x64 / two simdgroups
    assert_eq!(name(256, 4096), "gemm_nax_t64x64w2");
    assert_eq!(name(256, 24576), "gemm_nax_t64x64w2");
    assert_eq!(name(512, 4096), "gemm_nax_t64x64w2");
    // M tails that leave a 64-row tile mostly empty: 32x128 / two simdgroups, except for the wide gate + up pair
    assert_eq!(name(205, 4096), "gemm_nax_t32x128w2");
    assert_eq!(name(222, 4096), "gemm_nax_t32x128w2");
    assert_eq!(name(222, 24576), "gemm_nax_t64x64w2");
    assert_eq!(name(250, 4096), "gemm_nax_t64x64w2"); // 250 % 64 = 58: nearly full tiles
}

/// Through `Backend::launch` (the registry path, tile picked by shape): a call's output columns do not depend on how
/// many other columns the call has, nor on the rows around them, so the pick cannot change a result.
#[test]
fn nax_launch_picks_tiles_without_changing_bits() {
    let be = backend();
    let k = 4096usize;
    let x = rand_bf16(1024 * k, 61);
    let w = rand_bf16(12288 * k, 62);
    let (xb, wb) = (upload_at(&be, 0, &x), upload_at(&be, 0, &w));
    let run = |m: usize, n: usize, r0: usize| -> Vec<u16> {
        let yb = be.alloc(m * n * 2, MemKind::Device).unwrap();
        let call = matmul_call(tref(&xb, r0 * k * 2, &[m, k]), tref(&wb, 0, &[n, k]), tref(&yb, 0, &[m, n]));
        be.launch(MATMUL_BF16_NAX, &call, 0).unwrap();
        be.synchronize(0).unwrap();
        read_u16(&be, &yb, 0, m * n)
    };
    let big = run(1024, 12288, 0); // 128x64 tile
    for (m, n, r0) in [(256usize, 1024usize, 300usize), (205, 256, 0), (17, 12288, 1007), (1024, 1024, 0), (222, 4096, 300), (256, 4096, 5)] {
        let got = run(m, n, r0);
        for r in 0..m {
            assert!(
                got[r * n..(r + 1) * n] == big[(r0 + r) * 12288..(r0 + r) * 12288 + n],
                "m={m} n={n} r0={r0}: row {r} differs from the 1024x12288 call"
            );
        }
    }
}

/// `matmul_pair_bf16_nax`: bitwise two single launches, at M tails, unequal N0 / N1, odd offsets.
#[test]
fn nax_pair_is_bitwise_two_singles() {
    let be = backend();
    for &(m, n0, n1, k, xo, w0o, w1o, y0o, y1o) in &[
        (205usize, 1024usize, 1024usize, 4096usize, 0usize, 0usize, 8usize, 0usize, 2usize),
        (1024, 12288, 12288, 4096, 0, 0, 0, 0, 0),
        (17, 70, 33, 64, 2, 2, 6, 2, 0),
        (222, 1024, 1024, 4096, 0, 8, 8, 0, 0),
        (16, 130, 64, 100, 0, 0, 0, 0, 0),
    ] {
        let x = rand_bf16(m * k, 71);
        let w0 = rand_bf16(n0 * k, 72);
        let w1 = rand_bf16(n1 * k, 73);
        let (xb, w0b, w1b) = (upload_at(&be, xo, &x), upload_at(&be, w0o, &w0), upload_at(&be, w1o, &w1));
        let (a0, a1) = (be.alloc(y0o + m * n0 * 2 + 16, MemKind::Device).unwrap(), be.alloc(y1o + m * n1 * 2 + 16, MemKind::Device).unwrap());
        for (b, nb) in [(&a0, y0o + m * n0 * 2 + 16), (&a1, y1o + m * n1 * 2 + 16)] {
            be.upload(b, 0, &vec![0xEEu8; nb], 0).unwrap();
        }
        let call = intarsia_core::OpCall::new(
            intarsia_core::Op::MatmulPair,
            vec![tref(&xb, xo, &[m, k]), tref(&w0b, w0o, &[n0, k]), tref(&w1b, w1o, &[n1, k])],
            vec![tref(&a0, y0o, &[m, n0]), tref(&a1, y1o, &[m, n1])],
            Default::default(),
        );
        be.launch(MATMUL_PAIR_BF16_NAX, &call, 0).unwrap();
        be.synchronize(0).unwrap();
        for (wb, wo, n, ab, yo) in [(&w0b, w0o, n0, &a0, y0o), (&w1b, w1o, n1, &a1, y1o)] {
            let yb = be.alloc(yo + m * n * 2 + 16, MemKind::Device).unwrap();
            be.upload(&yb, 0, &vec![0xEEu8; yo + m * n * 2 + 16], 0).unwrap();
            let call = matmul_call(tref(&xb, xo, &[m, k]), tref(wb, wo, &[n, k]), tref(&yb, yo, &[m, n]));
            be.launch(MATMUL_BF16_NAX, &call, 0).unwrap();
            be.synchronize(0).unwrap();
            // whole buffers incl. guard bytes equal: the pair wrote exactly what the single wrote
            let len = (yo + m * n * 2 + 16) / 2;
            assert!(read_u16(&be, ab, 0, len) == read_u16(&be, &yb, 0, len), "pair != singles at m={m} n={n} k={k}");
        }
    }
}

/// Launch through `Backend::launch` (the registry path): odd offsets, correct against the f64 reference.
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
    let yb = be.alloc(yo + m * n * 2 + 16, MemKind::Device).unwrap();
    let call = matmul_call(tref(&xb, xo, &[m, k]), tref(&wb, wo, &[n, k]), tref(&yb, yo, &[m, n]));
    be.launch(MATMUL_BF16_NAX, &call, 0).unwrap();
    be.synchronize(0).unwrap();
    let y = read_u16(&be, &yb, yo, m * n);
    for &mi in &[0usize, 77, 204] {
        let refs = reference(&w, &x[mi * k..(mi + 1) * k], k, n);
        check(&y[mi * n..(mi + 1) * n], &refs, 0..n, &format!("{MATMUL_BF16_NAX} row {mi}"));
    }
}
