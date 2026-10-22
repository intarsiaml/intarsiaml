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

/// Tails and modes on random checkpoint-like weights. The launcher runs the 128 x 128 kernel for grids of at least 24 tiles
/// (else the 64-row wq_gemm kernel, also checked: the last two shapes), so the tail shapes are wide: M below, at and above
/// the 128-row tile (33, 100, 130, 256, 300, 1025), N not a multiple of the 128-row weight tile (3080, 1608 vectorised;
/// 3196 and 1100 take the scalar epilogue: N % 8 != 0), K of two to sixteen 64-wide steps, bf16 and f32 outputs, W8A8 and
/// weight-only activations.
#[test]
fn fp8_gemm_reg_is_bitwise_the_gemm() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(91);
    for act_quant in [true, false] {
        for (m, n, k, out) in [
            (33usize, 3080usize, 256usize, DType::Bf16),
            (100, 3196, 1024, DType::Bf16),
            (130, 1608, 256, DType::Bf16),
            (256, 1600, 1024, DType::F32),
            (300, 1100, 384, DType::F32),
            (1025, 384, 512, DType::Bf16),
            (128, 128, 128, DType::Bf16),
            (77, 136, 384, DType::F32),
        ] {
            let hc = matmul_fp8_case(&mut rng, m, n, k, act_quant, DType::Bf16, out);
            check(&*cuda, &hc, &format!("act_quant={act_quant} M={m} N={n} K={k} {out:?}"), m <= 300);
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
        let mut hc = matmul_fp8_case(&mut rng, 160, 1536, 512, act_quant, DType::Bf16, out);
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

/// `matmul_fp8_pair_gemm_reg` (Op::MatmulFp8Pair at prefill: one activation quantiser when the input scales are equal, one
/// grid for both GEMMs) must give the bytes of two `matmul_fp8_gemm_reg` calls, for equal input scales (the shared-xq path,
/// the 35B's case: every in_proj_qkv / in_proj_z and k / v pair has one input scale), different ones (the second
/// quantiser), weight-only and mixed halves, bf16 and f32 outputs, the 35B pair shapes, N tails, and grids below 24 tiles
/// (the launcher's single-call path).
#[test]
fn fp8_pair_gemm_reg_is_bitwise_two_calls() {
    use intarsia_core::quant::cases::matmul_fp8_pair_case;
    use intarsia_core::{HostTensor, Op};
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(94);
    let single = |hc: &HostCall, w: usize, out: usize| HostCall {
        op: Op::MatmulFp8,
        inputs: vec![hc.inputs[0].clone(), hc.inputs[w].clone(), hc.inputs[w + 1].clone()],
        outputs: vec![hc.outputs[out].clone()],
        attrs: hc.attrs.clone(),
    };
    for (m, n0, n1, k, out) in [
        (2048usize, 8192usize, 4096usize, 2048usize, DType::Bf16),
        (2048, 512, 512, 2048, DType::Bf16),
        (1000, 512, 512, 2048, DType::Bf16),
        (300, 512, 512, 2048, DType::Bf16),  // 12 tiles per half: the single-call path
        (300, 3080, 136, 384, DType::Bf16),  // one half above, one below 24 tiles
        (777, 3196, 1608, 256, DType::Bf16), // N % 8 != 0: scalar epilogue
        (256, 1600, 1100, 512, DType::F32),
    ] {
        for (mode, aq0, aq1) in
            [("equal", true, true), ("differ", true, true), ("weight-only", false, false), ("mixed", true, false)]
        {
            let mut hc = matmul_fp8_pair_case(&mut rng, m, n0, n1, k, aq0, aq1, DType::Bf16, out);
            let (s0, mut s1) = (hc.inputs[2].to_f32(), hc.inputs[4].to_f32());
            if mode == "equal" {
                s1[1] = s0[1];
            } else if mode == "differ" {
                assert!(s1[1] > 0.0 && s0[1] > 0.0 && s1[1] != s0[1], "case must differ: {s0:?} {s1:?}");
            }
            hc.inputs[4] = HostTensor::from_f32_as(hc.inputs[4].dtype, &hc.inputs[4].shape, &s1);
            let label = format!("{mode} M={m} N0={n0} N1={n1} K={k} {out:?}");
            let pair = run_host(&*cuda, "matmul_fp8_pair_gemm_reg", &hc).unwrap();
            let a = run_host(&*cuda, "matmul_fp8_gemm_reg", &single(&hc, 1, 0)).unwrap();
            let b = run_host(&*cuda, "matmul_fp8_gemm_reg", &single(&hc, 3, 1)).unwrap();
            assert!(pair[0].data == a[0].data, "{label}: first output differs from matmul_fp8_gemm_reg");
            assert!(pair[1].data == b[0].data, "{label}: second output differs from matmul_fp8_gemm_reg");
            let again = run_host(&*cuda, "matmul_fp8_pair_gemm_reg", &hc).unwrap();
            assert!(pair[0].data == again[0].data && pair[1].data == again[1].data, "{label}: not repeatable");
        }
    }
}
