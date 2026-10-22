//! `matmul_nvfp4_gemm_reg` and `matmul_nvfp4_glu_gemm` (kernels/cuda/nvfp4_gemm_reg.cu: dense NVFP4 prefill GEMMs with the
//! weights dequantised in registers) must give the bytes of `matmul_nvfp4_gemm` and, for the GLU, of two `matmul_nvfp4_gemm`
//! calls followed by `silu_glu_bf16`: M tails, the 35B shared-expert shapes, bf16 (vectorised) and f32 outputs. Needs a GPU:
//! without one the tests print SKIPPED (and fail under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_cuda)]

mod common;

use intarsia_core::backend::Backend;
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::quant::cases::{matmul_nvfp4_case, matmul_nvfp4_glu_case};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, Op, OpAttrs};
use std::sync::Arc;

fn device() -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_cuda::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            common::skip("nvfp4_gemm_reg", &format!("no usable CUDA device ({e})"));
            None
        }
    }
}

fn same(label: &str, want: &HostTensor, got: &HostTensor) {
    assert_eq!(want.data.len(), got.data.len(), "{label}: output size");
    if want.data != got.data {
        let (w, g) = (want.to_f32(), got.to_f32());
        let diff = w.iter().zip(&g).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let first = w.iter().zip(&g).position(|(a, b)| a.to_bits() != b.to_bits()).unwrap_or(0);
        panic!("{label}: {diff} of {} outputs differ (first at {first}: {} vs {})", w.len(), w[first], g[first]);
    }
}

/// M below, at and above the 64-row tile and not a multiple of it (256, 300, 777, 1000, 2048), N of one and of many
/// 128-row weight tiles, K of one to sixteen 128-wide units, bf16 and f32 outputs, the 35B shared expert's gate / up
/// (512 x 2048) and down (2048 x 512) projections.
#[test]
fn nvfp4_gemm_reg_is_bitwise_the_gemm() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(95);
    for (m, n, k, out) in [
        (2048usize, 512usize, 2048usize, DType::Bf16),
        (2048, 2048, 512, DType::Bf16),
        (256, 512, 2048, DType::Bf16),
        (300, 640, 384, DType::Bf16),
        (777, 1024, 256, DType::F32),
        (1000, 128, 128, DType::Bf16),
        (257, 384, 1024, DType::F32),
    ] {
        let hc = matmul_nvfp4_case(&mut rng, m, n, k, DType::Bf16, out);
        let label = format!("M={m} N={n} K={k} {out:?}");
        let old = run_host(&*cuda, "matmul_nvfp4_gemm", &hc).unwrap();
        let new = run_host(&*cuda, "matmul_nvfp4_gemm_reg", &hc).unwrap();
        same(&label, &old[0], &new[0]);
        let again = run_host(&*cuda, "matmul_nvfp4_gemm_reg", &hc).unwrap();
        assert!(new[0].data == again[0].data, "{label}: not repeatable");
    }
}

/// The fused shared-expert gate + up + silu_glu against the chain it replaces (the model's unfused fallback).
#[test]
fn nvfp4_glu_gemm_is_bitwise_the_chain() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(96);
    for (m, n, k) in
        [(2048usize, 512usize, 2048usize), (256, 512, 2048), (300, 192, 384), (1500, 64, 256), (999, 128, 1024)]
    {
        let hc = matmul_nvfp4_glu_case(&mut rng, m, n, k, DType::Bf16, DType::Bf16);
        let label = format!("GLU M={m} N={n} K={k}");
        let proj = |w: usize| HostCall {
            op: Op::MatmulNvfp4,
            inputs: vec![
                hc.inputs[0].clone(),
                hc.inputs[w].clone(),
                hc.inputs[w + 1].clone(),
                hc.inputs[w + 2].clone(),
            ],
            outputs: vec![OutSpec::new(DType::Bf16, &[m, n])],
            attrs: OpAttrs::default(),
        };
        let g = run_host(&*cuda, "matmul_nvfp4_gemm", &proj(1)).unwrap().remove(0);
        let u = run_host(&*cuda, "matmul_nvfp4_gemm", &proj(4)).unwrap().remove(0);
        let glu = HostCall {
            op: Op::SiluGlu,
            inputs: vec![g, u],
            outputs: vec![OutSpec::new(DType::Bf16, &[m, n])],
            attrs: OpAttrs::default(),
        };
        let want = run_host(&*cuda, "silu_glu_bf16", &glu).unwrap();
        let got = run_host(&*cuda, "matmul_nvfp4_glu_gemm", &hc).unwrap();
        same(&label, &want[0], &got[0]);
        let again = run_host(&*cuda, "matmul_nvfp4_glu_gemm", &hc).unwrap();
        assert!(got[0].data == again[0].data, "{label}: not repeatable");
    }
}
