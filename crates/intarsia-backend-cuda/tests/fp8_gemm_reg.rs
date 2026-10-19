//! `matmul_fp8_gemm_reg` (kernels/cuda/fp8_gemm.cu: weights dequantised in registers, 128 x 128 tiles) must give the bytes
//! `matmul_fp8_gemm` gives, for every finite e4m3 weight, both activation modes (W8A8 with a static input scale, weight
//! only), bf16 (vectorised and scalar epilogue) and f32 outputs, M and N tails, and the 35B projection shapes; and stay
//! within the admission budget of the cpu reference. Needs a GPU: without one the tests print SKIPPED (and fail under
//! CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_cuda)]

mod common;

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::backend::Backend;
use intarsia_core::exec::{run_host, HostCall};
use intarsia_core::quant::cases::matmul_fp8_case;
use intarsia_core::rng::SplitMix64;
use intarsia_core::DType;
use std::sync::Arc;

fn device() -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_cuda::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            common::skip("fp8_gemm_reg", &format!("no usable CUDA device ({e})"));
            None
        }
    }
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

/// Overwrite the weights with every finite e4m3 code (all but 0x7F / 0xFF), placed so that each code meets every
/// (row, k) position class: byte (r, c) = code number (r * 37 + c * 11 + seed) mod 254 of the 254 finite codes.
fn all_codes(hc: &mut HostCall, seed: usize) {
    let (n, k) = (hc.inputs[1].shape[0], hc.inputs[1].shape[1]);
    let codes: Vec<u8> = (0u16..256).map(|v| v as u8).filter(|&b| b & 0x7F != 0x7F).collect();
    assert_eq!(codes.len(), 254);
    for r in 0..n {
        for c in 0..k {
            hc.inputs[1].data[r * k + c] = codes[(r * 37 + c * 11 + seed) % codes.len()];
        }
    }
}

fn check(cuda: &dyn Backend, hc: &HostCall, label: &str, vs_ref: bool) {
    let old = run_host(cuda, "matmul_fp8_gemm", hc).unwrap();
    let new = run_host(cuda, "matmul_fp8_gemm_reg", hc).unwrap();
    assert_eq!(old[0].data.len(), new[0].data.len(), "{label}: output size");
    if old[0].data != new[0].data {
        let (o, n) = (old[0].to_f32(), new[0].to_f32());
        let diff = o.iter().zip(&n).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let first = o.iter().zip(&n).position(|(a, b)| a.to_bits() != b.to_bits()).unwrap_or(0);
        panic!(
            "{label}: {diff} of {} outputs differ from matmul_fp8_gemm (first at {first}: {} vs {})",
            o.len(),
            o[first],
            n[first]
        );
    }
    let again = run_host(cuda, "matmul_fp8_gemm_reg", hc).unwrap();
    assert_eq!(new[0].data, again[0].data, "{label}: not repeatable");
    if vs_ref {
        let want = run_host(&CpuBackend::new(), "ref_matmul_fp8", hc).unwrap()[0].to_f32();
        let r = rel_l2(&new[0].to_f32(), &want);
        assert!(r < 1e-3, "{label}: rel_l2 {r:.3e} against the cpu reference");
    }
}

/// Tails and modes on random checkpoint-like weights: M below, at and above the 128-row tile (33, 100, 128, 1025), N not a
/// multiple of the 128-row weight tile (192, 200 vectorised; 196 takes the scalar epilogue: N % 8 != 0), K of one and
/// several 64-wide steps (128, 256, 1024), bf16 and f32 outputs, W8A8 and weight-only activations.
#[test]
fn fp8_gemm_reg_is_bitwise_the_gemm() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(91);
    for act_quant in [true, false] {
        for (m, n, k, out) in [
            (33usize, 192usize, 256usize, DType::Bf16),
            (100, 200, 1024, DType::Bf16),
            (128, 128, 128, DType::Bf16),
            (130, 196, 256, DType::Bf16),
            (256, 384, 1024, DType::F32),
            (1025, 256, 512, DType::Bf16),
            (77, 136, 384, DType::F32),
        ] {
            let hc = matmul_fp8_case(&mut rng, m, n, k, act_quant, DType::Bf16, out);
            check(&*cuda, &hc, &format!("act_quant={act_quant} M={m} N={n} K={k} {out:?}"), m <= 256);
        }
    }
}

/// Every finite e4m3 code in every position class of the weight tile (signs, zero, -0, the 7 subnormals, the largest
/// normal 448), in both activation modes and both output types.
#[test]
fn fp8_gemm_reg_every_e4m3_code() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(92);
    for (seed, act_quant, out) in
        [(0usize, true, DType::Bf16), (5, false, DType::Bf16), (77, true, DType::F32), (131, false, DType::F32)]
    {
        let mut hc = matmul_fp8_case(&mut rng, 160, 256, 512, act_quant, DType::Bf16, out);
        all_codes(&mut hc, seed);
        check(&*cuda, &hc, &format!("all codes seed={seed} act_quant={act_quant} {out:?}"), false);
    }
}

/// The 35B W8A8 projection shapes at a 2048-token prefill chunk (and a 300-row tail chunk): GatedDeltaNet in_proj_qkv
/// (8192 x 2048), in_proj_z (4096 x 2048), out_proj (2048 x 4096), attention k/v (512 x 2048).
#[test]
fn fp8_gemm_reg_35b_shapes() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(93);
    for (m, n, k) in [
        (2048usize, 8192usize, 2048usize),
        (2048, 4096, 2048),
        (2048, 2048, 4096),
        (2048, 512, 2048),
        (300, 2048, 4096),
    ] {
        let hc = matmul_fp8_case(&mut rng, m, n, k, true, DType::Bf16, DType::Bf16);
        check(&*cuda, &hc, &format!("35B M={m} N={n} K={k}"), false);
    }
}
