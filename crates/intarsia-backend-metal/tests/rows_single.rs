//! G-B of the one-launch 9..=16-row GEMVs (`INTARSIA_METAL_ROWS_SINGLE`, `<base>_mr8t<T>` in kernels/metal/gemv_multirow.metal;
//! lane ink-batch): every row of a `matmul_bf16_gemv_rows` / `logits_bf16_gemv_rows` call of 9..=16 rows run as ONE launch is
//! bitwise the one-row launch on that row and the default two-chunk launch, on the 16-byte and 8-byte load families and the
//! f32-out logits kernel, with N not a multiple of the threadgroup's rows.

#![cfg(target_os = "macos")]

use intarsia_backend_metal::{with_rows_single, MetalBackend};
use intarsia_core::backend::{Backend, Buffer, MemKind};
use intarsia_core::{DType, Op, OpAttrs, OpCall, TensorRef};

fn backend() -> MetalBackend {
    MetalBackend::new().expect("metal device")
}

fn bf16_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut v = Vec::with_capacity(n * 2);
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let f = ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.25;
        v.extend_from_slice(&((f.to_bits() >> 16) as u16).to_le_bytes());
    }
    v
}

fn put(be: &MetalBackend, off: usize, bytes: &[u8]) -> Buffer {
    let b = be.alloc(off + bytes.len() + 64, MemKind::Device).unwrap();
    be.upload(&b, off, bytes, 0).unwrap();
    b
}

fn t(b: &Buffer, off: usize, dt: DType, shape: &[usize]) -> TensorRef {
    let mut x = TensorRef::new(b.clone(), dt, shape);
    x.offset = off;
    x
}

fn dl(be: &MetalBackend, b: &Buffer, n: usize) -> Vec<u8> {
    be.synchronize(0).unwrap();
    let mut v = vec![0u8; n];
    be.download(b, 0, &mut v, 0).unwrap();
    v
}

/// One case: `op` with `m` rows of K, N outputs, weights / activations at byte offset `off` (16: the 16-byte family, 8: the
/// 8-byte family of `Matmul`).
fn case(be: &MetalBackend, kernel: &str, op: Op, m: usize, k: usize, n: usize, off: usize) {
    let w = put(be, off, &bf16_bytes(n * k, 7 + n as u64));
    let x = put(be, off, &bf16_bytes(m * k, 11 + m as u64));
    let (odt, esz) = if op == Op::Logits { (DType::F32, 4) } else { (DType::Bf16, 2) };
    let ybytes = m * n * esz;
    let call = |rows: usize, r0: usize, y: &Buffer| {
        OpCall::new(
            op,
            vec![t(&x, off + r0 * k * 2, DType::Bf16, &[rows, k]), t(&w, off, DType::Bf16, &[n, k])],
            vec![t(y, r0 * n * esz, odt, &[rows, n])],
            OpAttrs::default(),
        )
    };
    let mk = || {
        let b = be.alloc(ybytes + 64, MemKind::Device).unwrap();
        be.upload(&b, 0, &vec![0xa5u8; ybytes + 64], 0).unwrap();
        b
    };
    let (one, chunks, single) = (mk(), mk(), mk());
    for r in 0..m {
        be.launch(kernel, &call(1, r, &one), 0).unwrap();
    }
    with_rows_single(false, || be.launch(kernel, &call(m, 0, &chunks), 0).unwrap());
    with_rows_single(true, || be.launch(kernel, &call(m, 0, &single), 0).unwrap());
    let (a, b, c) = (dl(be, &one, ybytes + 64), dl(be, &chunks, ybytes + 64), dl(be, &single, ybytes + 64));
    assert!(a == b, "{kernel} M={m} K={k} N={n} off={off}: two chunk launches differ from the one-row launches");
    assert!(a == c, "{kernel} M={m} K={k} N={n} off={off}: the one-launch form differs from the one-row launches");
}

#[test]
fn one_launch_9_to_16_rows_equal_one_row_launches() {
    let be = backend();
    for m in 9..=16usize {
        // Inkling-2's wo (K 4096 -> 2304) on both load families, a K that is not a multiple of the 16-byte block
        for (k, n, off) in [(4096usize, 2304usize, 0usize), (4096, 2304, 8), (2312, 515, 0)] {
            case(&be, "matmul_bf16_gemv_rows", Op::Matmul, m, k, n, off);
        }
        // logits: the unembed shape class (K 2304, N not a multiple of 4 rows per threadgroup), and a router-sized N
        for (k, n) in [(2304usize, 8195usize), (2304, 128)] {
            case(&be, "logits_bf16_gemv_rows", Op::Logits, m, k, n, 0);
        }
    }
}

#[test]
fn rows_single_is_off_by_default_and_only_for_9_to_16_rows() {
    // the env var is unset in the test run
    assert!(!intarsia_backend_metal::rows_single("matmul_bf16_gemv_rows", 16));
    with_rows_single(true, || {
        assert!(intarsia_backend_metal::rows_single("matmul_bf16_gemv_rows", 9));
        assert!(intarsia_backend_metal::rows_single("logits_bf16_gemv_rows", 16));
        assert!(!intarsia_backend_metal::rows_single("matmul_bf16_gemv_rows", 8));
        assert!(!intarsia_backend_metal::rows_single("matmul_bf16_gemv_rows", 17));
        assert!(!intarsia_backend_metal::rows_single("matmul_fp8_gemv_rows", 16));
    });
}
