#![cfg(target_os = "macos")]
//! `MatmulPair` (`matmul_pair_bf16_gemv`): bitwise the two `matmul_bf16_gemv` launches it replaces, one dispatch on the
//! vector path, correct (two dispatches) on every other operand alignment, no stray stores.

mod common;
use common::*;
use intarsia_backend_metal::kernels::{MATMUL_BF16_GEMV, MATMUL_PAIR_BF16_GEMV};
use intarsia_core::backend::{Backend, MemKind};
use intarsia_core::{Op, OpAttrs, OpCall, ShapeInfo};

/// Byte offsets of w0, w1, x, y0, y1 inside their buffers.
#[derive(Clone, Copy)]
struct Offs {
    w0: usize,
    w1: usize,
    x: usize,
    y0: usize,
    y1: usize,
}
const ZERO: Offs = Offs { w0: 0, w1: 0, x: 0, y0: 0, y1: 0 };

fn pair_vs_singles(n0: usize, n1: usize, k: usize, o: Offs, seed: u64, expect_dispatches: usize) {
    let be = backend();
    let (w0, w1, x) = (rand_bf16(n0 * k, seed), rand_bf16(n1 * k, seed + 1), rand_bf16(k, seed + 2));
    let with_off = |v: &[u16], off: usize| {
        let mut p = vec![0u16; off / 2];
        p.extend_from_slice(v);
        upload(&be, &p)
    };
    let (w0b, w1b, xb) = (with_off(&w0, o.w0), with_off(&w1, o.w1), with_off(&x, o.x));
    let poison = |n: usize, off: usize| {
        let b = be.alloc(off + n * 2 + 16, MemKind::Device).unwrap();
        be.upload(&b, 0, &vec![0xEEu8; off + n * 2 + 16], 0).unwrap();
        b
    };
    let (y0b, y1b) = (poison(n0, o.y0), poison(n1, o.y1));
    let (s0b, s1b) = (poison(n0, o.y0), poison(n1, o.y1));
    let xt = tref(&xb, o.x, &[1, k]);
    let pair = OpCall::new(
        Op::MatmulPair,
        vec![xt.clone(), tref(&w0b, o.w0, &[n0, k]), tref(&w1b, o.w1, &[n1, k])],
        vec![tref(&y0b, o.y0, &[1, n0]), tref(&y1b, o.y1, &[1, n1])],
        OpAttrs::default(),
    );
    pair.check().unwrap();
    let ks = be.kernels();
    let shape = ShapeInfo::from_call(&pair);
    let registered = ks.iter().any(|kn| kn.op == Op::MatmulPair && kn.name == MATMUL_PAIR_BF16_GEMV);
    assert!(registered, "matmul_pair_bf16_gemv is registered");
    if k % 8 == 0 {
        assert!(ks.iter().any(|kn| kn.name == MATMUL_PAIR_BF16_GEMV && kn.accepts(&shape)));
    }
    be.launch(MATMUL_PAIR_BF16_GEMV, &pair, 0).unwrap();
    assert_eq!(be.pending_dispatches(), expect_dispatches, "dispatches of the pair launch");
    be.launch(MATMUL_BF16_GEMV, &matmul_call(xt.clone(), tref(&w0b, o.w0, &[n0, k]), tref(&s0b, o.y0, &[1, n0])), 0)
        .unwrap();
    be.launch(MATMUL_BF16_GEMV, &matmul_call(xt, tref(&w1b, o.w1, &[n1, k]), tref(&s1b, o.y1, &[1, n1])), 0).unwrap();
    be.synchronize(0).unwrap();
    for (pairb, singleb, n, off, w, name) in [(&y0b, &s0b, n0, o.y0, &w0, "y0"), (&y1b, &s1b, n1, o.y1, &w1, "y1")] {
        let (mut a, mut b) = (vec![0u8; off + n * 2 + 16], vec![0u8; off + n * 2 + 16]);
        be.download(pairb, 0, &mut a, 0).unwrap();
        be.download(singleb, 0, &mut b, 0).unwrap();
        assert!(a[..off].iter().all(|v| *v == 0xEE), "{name}: stray store before");
        assert!(a[off + n * 2..].iter().all(|v| *v == 0xEE), "{name}: stray store after");
        assert_eq!(a, b, "{name}: pair differs from the single GEMV (K={k} N0={n0} N1={n1})");
        let gpu: Vec<u16> = a[off..off + n * 2].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        check(&gpu, &reference(w, &x, k, n), 0..n, &format!("{name} K={k} N0={n0} N1={n1}"));
    }
}

#[test]
fn qwen3_8b_shapes_are_bitwise_two_gemvs_in_one_dispatch() {
    // gate + up, k + v
    pair_vs_singles(12288, 12288, 4096, ZERO, 10, 1);
    pair_vs_singles(1024, 1024, 4096, ZERO, 20, 1);
}

#[test]
fn odd_sizes_and_unequal_halves() {
    // the admission shapes (K % 8 == 0), N not a multiple of the 4 rows per threadgroup, N1 == 1, K < one block
    for (n0, n1, k) in [(64, 40, 64), (33, 1, 96), (9, 17, 40), (300, 5, 520), (4097, 3, 4104), (1, 1, 8)] {
        pair_vs_singles(n0, n1, k, ZERO, 30, 1);
    }
}

#[test]
fn output_offsets_with_a_leftover_element() {
    // y at 2 mod 4 bytes (the y_skip trick), weights and x still 16-byte aligned: still one dispatch
    pair_vs_singles(130, 70, 256, Offs { y0: 2, y1: 6, ..ZERO }, 40, 1);
}

#[test]
fn other_alignments_fall_back_to_two_dispatches() {
    // W only 8-byte aligned (8-byte kernel), or 2-byte aligned (scalar), or K % 8 != 0: two launches, same results
    pair_vs_singles(130, 70, 256, Offs { w0: 8, ..ZERO }, 50, 2);
    pair_vs_singles(130, 70, 256, Offs { w1: 2, x: 2, ..ZERO }, 51, 2);
    pair_vs_singles(130, 70, 100, ZERO, 52, 2);
}
