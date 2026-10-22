#![cfg(target_os = "macos")]
//! `RmsNormAddMatmulPair` (`rmsnorm_add_matmul_pair_bf16_gemv`) against the launches it replaces, `rmsnorm_add_bf16` then
//! `matmul_pair_bf16_gemv`: all four outputs must be BYTE identical. Weights at 0 / 8 mod 16 (the 16-byte and 8-byte pair
//! kernels), unequal halves, K that is not a multiple of a threadgroup's step, a 1-row second matrix (the shared-expert
//! gate), outputs at odd element offsets, `w_offset` 0 and 1.

mod common;

use common::*;
use intarsia_backend_metal::kernels::{MATMUL_PAIR_BF16_GEMV, RMSNORM_ADD_BF16, RMSNORM_ADD_MATMUL_PAIR_BF16_GEMV};
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::{Op, OpAttrs, OpCall, ShapeInfo};

fn up_at(be: &MetalBackend, byte_off: usize, v: &[u16]) -> Buffer {
    let mut p = vec![0u16; byte_off / 2];
    p.extend_from_slice(v);
    upload(be, &p)
}

fn poison(be: &MetalBackend, bytes: usize) -> Buffer {
    let b = be.alloc(bytes, MemKind::Device).unwrap();
    be.upload(&b, 0, &vec![0xEEu8; bytes], 0).unwrap();
    b
}

fn download(be: &MetalBackend, b: &Buffer, bytes: usize) -> Vec<u8> {
    let mut v = vec![0u8; bytes];
    be.download(b, 0, &mut v, 0).unwrap();
    v
}

#[allow(clippy::too_many_arguments)]
fn case(be: &MetalBackend, n0: usize, n1: usize, k: usize, w_off: usize, skip: usize, w_offset: f32, seed: u64) {
    let x = rand_bf16(k + skip, seed);
    let res = rand_bf16(k + skip, seed + 1);
    let nw = rand_bf16(k + skip, seed + 2);
    let (w0, w1) = (rand_bf16(n0 * k, seed + 3), rand_bf16(n1 * k, seed + 4));
    let (xb, rb, nb) = (upload(be, &x), upload(be, &res), upload(be, &nw));
    let (w0b, w1b) = (up_at(be, w_off, &w0), up_at(be, w_off, &w1));
    let o = skip * 2;
    let sz = (k + skip + n0.max(n1)) * 2 + 32;
    let (tx, tr) = (tref(&xb, o, &[1, k]), tref(&rb, o, &[1, k]));
    let tn = tref(&nb, o, &[k]);
    let (tw0, tw1) = (tref(&w0b, w_off, &[n0, k]), tref(&w1b, w_off, &[n1, k]));
    let attrs = OpAttrs { eps: 1e-6, w_offset, ..Default::default() };
    let mut outs = Vec::new();
    for fused in [true, false] {
        let (normed, sum, y0, y1) = (poison(be, sz), poison(be, sz), poison(be, sz), poison(be, sz));
        let t = |b: &Buffer, shape: &[usize]| -> TensorRef { tref(b, o, shape) };
        if fused {
            let call = OpCall::new(
                Op::RmsNormAddMatmulPair,
                vec![tx.clone(), tr.clone(), tn.clone(), tw0.clone(), tw1.clone()],
                vec![t(&normed, &[1, k]), t(&sum, &[1, k]), t(&y0, &[1, n0]), t(&y1, &[1, n1])],
                attrs.clone(),
            );
            call.check().unwrap();
            let ks = be.kernels();
            let kn = ks.iter().find(|kn| kn.name == RMSNORM_ADD_MATMUL_PAIR_BF16_GEMV).unwrap();
            assert!(kn.accepts(&ShapeInfo::from_call(&call)), "the fold does not accept n0={n0} n1={n1} k={k}");
            be.launch(RMSNORM_ADD_MATMUL_PAIR_BF16_GEMV, &call, 0).unwrap();
        } else {
            let norm = OpCall::new(
                Op::RmsNormAdd,
                vec![tx.clone(), tr.clone(), tn.clone()],
                vec![t(&normed, &[1, k]), t(&sum, &[1, k])],
                attrs.clone(),
            );
            be.launch(RMSNORM_ADD_BF16, &norm, 0).unwrap();
            let pair = OpCall::new(
                Op::MatmulPair,
                vec![t(&normed, &[1, k]), tw0.clone(), tw1.clone()],
                vec![t(&y0, &[1, n0]), t(&y1, &[1, n1])],
                OpAttrs::default(),
            );
            be.launch(MATMUL_PAIR_BF16_GEMV, &pair, 0).unwrap();
        }
        be.synchronize(0).unwrap();
        outs.push([normed, sum, y0, y1].map(|b| download(be, &b, sz)));
    }
    for (name, (f, u)) in ["normed", "sum", "y0", "y1"].iter().zip(outs[0].iter().zip(outs[1].iter())) {
        assert!(f == u, "{name} differs (n0={n0} n1={n1} k={k} w_off={w_off} skip={skip} w_offset={w_offset})");
    }
}

#[test]
fn norm_pair_is_bitwise_norm_then_pair() {
    let be = backend();
    let mut seed = 10;
    // (n0, n1, k): router + shared gate logit (256 + 1, K = 2048), GatedDeltaNet b / a (32 + 32), odd splits and Ks
    for &(n0, n1, k) in &[
        (256, 1, 2048),
        (32, 32, 2048),
        (33, 5, 104),
        (7, 9, 4104),
        (1, 1, 8),
        (130, 70, 520),
        (64, 64, 8192),
        (16, 3, 4096),
    ] {
        // 0 / 16 mod 16: the 16-byte rt pair; 8 mod 16: the 8-byte one
        for w_off in [0usize, 16, 8, 24] {
            for w_offset in [0.0f32, 1.0] {
                seed += 7;
                case(&be, n0, n1, k, w_off, 0, w_offset, seed);
            }
        }
    }
    // outputs / activations at odd element offsets (the pair then runs unfused inside the launcher or takes the skip)
    case(&be, 256, 1, 2048, 8, 1, 0.0, 900);
    case(&be, 64, 8, 520, 16, 3, 0.0, 901);
}
