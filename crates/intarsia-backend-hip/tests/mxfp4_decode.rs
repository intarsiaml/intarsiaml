//! The MXFP4 decode kernels of lane P1 against the launches they replace, bitwise (the `fused_decode.rs` pattern).
//!
//! 1. `matmul_mxfp4_gemv` at M = 1 runs `mxfp4_gemv1_lds_kernel` (x staged in LDS, the W4A4 QDQ fused, grid-stride rows)
//!    when K <= 8192; at M >= 2 the same entry runs the pre-pass (`mxfp4_xprep_kernel`) and `mxfp4_gemv_kernel<M>`, whose
//!    per-row arithmetic is that of `<1>`. So the M = 1 output must equal, bit for bit, every row of an M = 2 call on x
//!    stacked twice: the LDS path against the pre-pass path in one process, without the `INTARSIA_HIP_MXFP4_LDS` knob
//!    (read once per process).
//! 2. `matmul_mxfp4_pair_gemv` (one launch over the rows of both matrices) must equal two single `matmul_mxfp4_gemv`
//!    calls, bitwise, at M = 1 (LDS pair) and M = 2 (the two-singles fallback).
//!
//! Shapes: Qwen3.8-27B decode (K = 5120: gate / up, qkv, z, b / a, q, k / v; K = 6144: out / o; K = 17408: down, which
//! keeps the pre-pass path) and edges (rows not a multiple of 8 warps, one group per row, K past one 8-group batch).
//! Both activation modes (W4A4 dynamic QDQ, W4A16 none) and both output dtypes.
//!
//! Needs a GPU: skips (passes) without one.
#![cfg(ik_hip)]

use intarsia_core::backend::Backend;
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::quant::cases::{matmul_mxfp4_case, matmul_mxfp4_pair_case};
use intarsia_core::quant::Mxfp4Act;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, Op, OpAttrs};
use std::sync::Arc;

fn device() -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_HIP_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_hip::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping: no usable HIP device ({e})");
            None
        }
    }
}

fn bits_eq(label: &str, what: &str, got: &[u8], want: &[u8], dt: DType) {
    if got != want {
        let w = dt.size_bytes();
        let bad = got.chunks(w).zip(want.chunks(w)).filter(|(a, b)| a != b).count();
        let first = got.chunks(w).zip(want.chunks(w)).position(|(a, b)| a != b).unwrap();
        panic!("{label}: {what} differs in {bad} of {} elements (first at {first})", got.len() / w);
    }
}

const SHAPES: [(usize, usize); 12] = [
    (17408, 5120), // ffn gate / up
    (10240, 5120), // gdn in_qkv
    (6144, 5120),  // gdn in_z
    (12288, 5120), // attn q
    (1024, 5120),  // attn k / v
    (48, 5120),    // gdn in_b / in_a
    (5120, 6144),  // gdn out / attn o
    (5120, 17408), // ffn down (pre-pass path at M = 1 too)
    (17, 64),
    (130, 1056),
    (1, 32),
    (9, 8192), // largest K on the LDS path, two 8-group batches per lane
];

#[test]
fn mxfp4_gemv_lds_is_bitwise_prepass_gemv() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(71);
    for (n, k) in SHAPES {
        for act in [Mxfp4Act::Dynamic, Mxfp4Act::None] {
            for odt in [DType::Bf16, DType::F32] {
                let label = format!("mxfp4 gemv n={n} k={k} act={act:?} out={odt:?}");
                let one = matmul_mxfp4_case(&mut rng, 1, n, k, act, DType::Bf16, odt);
                let got = run_host(&*hip, "matmul_mxfp4_gemv", &one).unwrap();
                let x = &one.inputs[0];
                let mut two = one.clone();
                two.inputs[0] = HostTensor::new(x.dtype, vec![2, k], [x.data.clone(), x.data.clone()].concat()).unwrap();
                two.outputs = vec![OutSpec::new(odt, &[2, n])];
                let want = run_host(&*hip, "matmul_mxfp4_gemv", &two).unwrap();
                let row = n * odt.size_bytes();
                bits_eq(&label, "M=1 vs row 0 of M=2", &got[0].data, &want[0].data[..row], odt);
                bits_eq(&label, "M=1 vs row 1 of M=2", &got[0].data, &want[0].data[row..], odt);
            }
        }
    }
}

#[test]
fn matmul_mxfp4_pair_is_bitwise_two_gemvs() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(72);
    // (n0, n1, k): GDN qkv + z, b + a, FFN gate + up, attention k + v; unequal and odd segments; the K > 8192 fallback
    for (n0, n1, k) in [
        (10240usize, 6144usize, 5120usize),
        (48, 48, 5120),
        (17408, 17408, 5120),
        (1024, 1024, 5120),
        (17, 33, 64),
        (130, 7, 1056),
        (1, 1, 32),
        (24, 40, 17408),
    ] {
        for m in [1usize, 2] {
            for act in [Mxfp4Act::Dynamic, Mxfp4Act::None] {
                let label = format!("mxfp4 pair m={m} n0={n0} n1={n1} k={k} act={act:?}");
                let hc = matmul_mxfp4_pair_case(&mut rng, m, n0, n1, k, act, DType::Bf16, DType::Bf16);
                let got = run_host(&*hip, "matmul_mxfp4_pair_gemv", &hc).unwrap();
                for (j, w, sc) in [(0usize, 1usize, 2usize), (1, 3, 4)] {
                    let one = HostCall {
                        op: Op::MatmulMxfp4,
                        inputs: vec![hc.inputs[0].clone(), hc.inputs[w].clone(), hc.inputs[sc].clone(), hc.inputs[5].clone()],
                        outputs: vec![hc.outputs[j].clone()],
                        attrs: OpAttrs::default(),
                    };
                    let want = run_host(&*hip, "matmul_mxfp4_gemv", &one).unwrap();
                    bits_eq(&label, &format!("output {j}"), &got[j].data, &want[0].data, DType::Bf16);
                }
            }
        }
    }
}
