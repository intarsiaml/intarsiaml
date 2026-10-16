#![cfg(target_os = "macos")]
//! bf16 GEMV against an f64 CPU reference at Qwen3-8B shapes, odd sizes, unaligned operands, and every shader variant.

mod common;
use common::*;
use intarsia_backend_metal::kernels::{GEMV_A8_VARIANTS, GEMV_VARIANTS, MATMUL_BF16_GEMV};
use intarsia_backend_metal::{encode_gemv, GemvParams};
use intarsia_core::backend::Backend;
use intarsia_core::{Op, ShapeInfo};

fn run_shape(k: usize, n: usize, w_off: usize, x_off: usize, y_off: usize, seed: u64) {
    let be = backend();
    // operands live inside bigger buffers at the requested byte offsets (offsets are multiples of 2: bf16 elements)
    let w = rand_bf16(n * k, seed);
    let x = rand_bf16(k, seed + 1);
    let wb = {
        let mut v = vec![0u16; w_off / 2];
        v.extend_from_slice(&w);
        upload(&be, &v)
    };
    let xb = {
        let mut v = vec![0u16; x_off / 2];
        v.extend_from_slice(&x);
        upload(&be, &v)
    };
    let yb = be.alloc(y_off + n * 2 + 16, intarsia_core::backend::MemKind::Device).unwrap();
    // poison y so a missed or stray store shows up
    be.upload(&yb, 0, &vec![0xEEu8; y_off + n * 2 + 16], 0).unwrap();
    let call = matmul_call(tref(&xb, x_off, &[1, k]), tref(&wb, w_off, &[n, k]), tref(&yb, y_off, &[1, n]));

    // the registry predicate accepts exactly this call
    let shape = ShapeInfo::from_call(&call);
    let ks = be.kernels();
    assert!(ks.iter().any(|kn| kn.op == Op::Matmul && kn.name == MATMUL_BF16_GEMV && kn.accepts(&shape)));

    be.launch(MATMUL_BF16_GEMV, &call, 0).unwrap();
    be.synchronize(0).unwrap();
    let mut raw = vec![0u8; y_off + n * 2 + 16];
    be.download(&yb, 0, &mut raw, 0).unwrap();
    assert!(raw[..y_off].iter().all(|b| *b == 0xEE), "stray store before y");
    assert!(raw[y_off + n * 2..].iter().all(|b| *b == 0xEE), "stray store after y");
    let gpu: Vec<u16> = raw[y_off..y_off + n * 2].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let refs = reference(&w, &x, k, n);
    let checked = check(&gpu, &refs, 0..n, &format!("K={k} N={n} offs=({w_off},{x_off},{y_off})"));
    assert_eq!(checked, n);
}

#[test]
fn qwen3_8b_shapes() {
    for (i, (k, n)) in [(4096, 4096), (4096, 12288), (12288, 4096), (4096, 151936)].into_iter().enumerate() {
        run_shape(k, n, 0, 0, 0, 100 + i as u64);
    }
}

#[test]
fn k_not_multiple_of_8_and_odd_n() {
    // scalar fallback: K % 8 != 0, N not a multiple of any rows-per-threadgroup
    for (k, n) in [(1, 1), (7, 3), (13, 17), (4100, 33), (4097, 1025), (1000, 5)] {
        run_shape(k, n, 0, 0, 0, 7);
    }
    // vector path with N not a multiple of rows/tg and a K tail shorter than one lane sweep
    for (k, n) in [(8, 1), (16, 5), (264, 7), (4104, 129), (8200, 2049)] {
        run_shape(k, n, 0, 0, 0, 9);
    }
}

#[test]
fn unaligned_operand_offsets() {
    // W / x at 2, 4, 6, 16, 18 byte offsets: 16-aligned ones keep the vector kernel, the rest take the scalar one
    // 8-byte aligned W/x (offset 8, 24) takes the 8-byte-load kernel when K % 4 == 0
    for (wo, xo, yo) in [
        (2, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (4, 6, 2),
        (16, 32, 16),
        (18, 16, 0),
        (6, 10, 14),
        (8, 0, 0),
        (8, 8, 8),
        (24, 16, 0),
        (8, 8, 6),
    ] {
        run_shape(4096, 130, wo, xo, yo, 21);
        run_shape(4100, 33, wo, xo, yo, 23);
        run_shape(77, 9, wo, xo, yo, 22);
    }
}

#[test]
fn every_variant_matches_reference() {
    let be = backend();
    let (k, n) = (4104, 1031);
    let w = rand_bf16(n * k, 5);
    let x = rand_bf16(k, 6);
    let refs = reference(&w, &x, k, n);
    let (wb, xb) = (upload(&be, &w), upload(&be, &x));
    for v in GEMV_VARIANTS.iter().chain(GEMV_A8_VARIANTS) {
        let yb = be.new_mtl_buffer(n * 2, false).unwrap();
        let wm = wb.downcast::<intarsia_backend_metal::MetalBuffer>().unwrap();
        let xm = xb.downcast::<intarsia_backend_metal::MetalBuffer>().unwrap();
        let params = GemvParams { k: k as u32, n: n as u32, ..Default::default() };
        let cb = be
            .encode_and_commit(|enc| {
                encode_gemv(
                    enc,
                    be.pso(v.name).unwrap(),
                    (wm.mtl(), 0),
                    (xm.mtl(), 0),
                    (&yb, 0),
                    &params,
                    (n as u32).div_ceil(v.rows_per_tg()),
                    v.threads_per_tg(),
                )
            })
            .unwrap();
        intarsia_backend_metal::MetalBackend::wait(&cb).unwrap();
        // SAFETY: n u16 results written by the kernel.
        let gpu: Vec<u16> =
            unsafe { std::slice::from_raw_parts(objc2_metal::MTLBuffer::contents(&*yb).as_ptr() as *const u16, n) }
                .to_vec();
        check(&gpu, &refs, 0..n, v.name);
    }
}

#[test]
fn backend_info_and_alloc_roundtrip() {
    let be = backend();
    let info = be.info();
    eprintln!("{info:?}");
    assert!(info.unified && info.total_mem > 0 && info.free_mem <= info.total_mem);
    let bw = info.bandwidth_gbs.expect("copy bandwidth measured");
    assert!(bw > 50.0 && bw < 2000.0, "implausible copy bandwidth {bw}");
    let b = be.alloc(1000, intarsia_core::backend::MemKind::Device).unwrap();
    let data: Vec<u8> = (0..1000u32).map(|i| (i * 7) as u8).collect();
    be.upload(&b, 0, &data, 0).unwrap();
    let c = be.alloc(1000, intarsia_core::backend::MemKind::Device).unwrap();
    be.copy(&c, 10, &b, 100, 500, 0).unwrap();
    let mut out = vec![0u8; 500];
    be.download(&c, 10, &mut out, 0).unwrap();
    assert_eq!(out, data[100..600]);
    assert!(be.download(&c, 900, &mut [0u8; 200], 0).is_err());
}

#[test]
fn shader_choice_follows_alignment_and_k() {
    let be = backend();
    let w = upload(&be, &rand_bf16(64 * 4104 + 64, 1));
    let x = upload(&be, &rand_bf16(4104 + 64, 2));
    let y = be.alloc(64 * 2, intarsia_core::backend::MemKind::Device).unwrap();
    let pick = |k: usize, wo: usize, xo: usize| {
        let call = matmul_call(tref(&x, xo, &[1, k]), tref(&w, wo, &[64, k]), tref(&y, 0, &[1, 64]));
        be.gemv_shader_for(&call).unwrap()
    };
    assert_eq!(pick(4096, 0, 0), intarsia_backend_metal::DEFAULT_GEMV.name);
    assert_eq!(pick(4096, 16, 32), intarsia_backend_metal::DEFAULT_GEMV.name);
    assert_eq!(pick(4096, 8, 0), intarsia_backend_metal::kernels::DEFAULT_GEMV_A8.name);
    assert_eq!(pick(4100, 0, 0), intarsia_backend_metal::kernels::DEFAULT_GEMV_A8.name);
    assert_eq!(pick(4096, 4, 0), intarsia_backend_metal::kernels::GEMV_SCALAR.name);
    assert_eq!(pick(4097, 0, 0), intarsia_backend_metal::kernels::GEMV_SCALAR.name);
    assert_eq!(pick(4096, 0, 2), intarsia_backend_metal::kernels::GEMV_SCALAR.name);
}
