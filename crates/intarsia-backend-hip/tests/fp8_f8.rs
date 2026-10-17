//! `matmul_fp8_mma_f8` (kernels/hip/ik_f8_gemm.h): the W8A8 prefill GEMM on the gfx12 fp8 x fp8 WMMA.
//!
//! Each activation is quantised to e4m3 once, every product e4m3 x e4m3 is exact in f32, so the kernel differs from the
//! cpu reference only by the order of the f32 additions. That makes a much tighter check than the admission budget
//! (rel_l2 1e-3 / max_abs 1.5e-2 on bf16 outputs, 2e-4 on f32 ones) possible, and these tests use it:
//!
//! * `f8_matches_cpu_reference_tightly`: f32 outputs within 2e-6 relative (L2) of the cpu reference, bf16 outputs within 3e-4
//!   (the admission sweep measured 1.5e-4 for the bf16 kernel this replaces), W8A8 with a non-trivial weight and input scale;
//!   shapes are the model's projections and edges (K not a multiple of the 64-byte stage, N not a multiple of the tile, M from 5).
//! * `f8_rows_do_not_depend_on_m`: a row computed in a call of M = 5, 37, 129 or 300 rows has the bits it has in a call of 300
//!   (every tile shape and every position in a tile sums k in the same order): this is what keeps a tail chunk equal to a full one.
//! * `f8_is_not_the_bf16_path`: the f32 output is within 1e-6 of `matmul_fp8_mma`'s yet differs from it in some bits (a different
//!   order of additions): the launch really ran the e4m3 kernel, and the weight-only half of the launcher did not overwrite it.
//! * `f8_weight_only_is_the_bf16_path`: with input_scale 0 the output is bitwise `matmul_fp8_mma`'s (the e4m3 kernel and the
//!   quantiser exit, the bf16 kernel runs).
//!
//! scripts/hip-decode/mutate.py (`f8-*` mutations) breaks the kernel in ways each of these must catch. Needs a GPU: skips (passes)
//! without one.
#![cfg(ik_hip)]

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::backend::Backend;
use intarsia_core::exec::run_host;
use intarsia_core::quant::cases::matmul_fp8_case;
use intarsia_core::quant::fp8::scales_tensor;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, Op};
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

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

fn nbits_differ(a: &HostTensor, b: &HostTensor) -> usize {
    a.to_f32().iter().zip(b.to_f32().iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

/// W8A8 case with the weight scale scaled by `ws` and the input scale `is` (0 = weight-only activations).
fn case(rng: &mut SplitMix64, m: usize, n: usize, k: usize, odt: DType, ws: f32, is: f32) -> intarsia_core::exec::HostCall {
    let mut hc = matmul_fp8_case(rng, m, n, k, is > 0.0, DType::Bf16, odt);
    let s = hc.inputs[2].to_f32();
    hc.inputs[2] = scales_tensor(s[0] * ws, is);
    hc
}

const SHAPES: &[(usize, usize, usize)] = &[
    (5, 512, 2048),
    (17, 130, 48),
    (33, 259, 1040),
    (64, 4096, 2048),
    (100, 8192, 2048),
    (129, 2048, 4096),
    (257, 4096, 2048),
    (300, 17, 96),
    (130, 131, 4112),
    (1024, 512, 2048),
];

#[test]
fn f8_matches_cpu_reference_tightly() {
    let Some(hip) = device() else { return };
    let cpu = CpuBackend::new();
    let reference = cpu.kernels().into_iter().find(|k| k.op == Op::MatmulFp8 && k.is_reference).unwrap().name;
    let mut rng = SplitMix64::new(81);
    let mut worst = [0f64; 2];
    for &(m, n, k) in SHAPES {
        for (ws, is) in [(1.0f32, 0.31f32), (2.5, 0.07), (0.4, 4.0)] {
            let hc = case(&mut rng, m, n, k, DType::F32, ws, is);
            let want = run_host(&cpu, &reference, &hc).unwrap().remove(0);
            let want_f32 = want.to_f32();
            let want_bf16 = HostTensor::from_f32_as(DType::Bf16, &[m, n], &want_f32);
            for (o, (odt, bound, w)) in [(DType::F32, 2e-6, &want), (DType::Bf16, 3e-4, &want_bf16)].into_iter().enumerate() {
                let mut hc = hc.clone();
                hc.outputs[0].dtype = odt;
                let got = run_host(&*hip, "matmul_fp8_mma_f8", &hc).unwrap().remove(0);
                let r = rel_l2(&got.to_f32(), &w.to_f32());
                worst[o] = worst[o].max(r);
                assert!(r < bound, "m={m} n={n} k={k} ws={ws} is={is} {odt:?}: rel_l2 {r} vs {reference} (bound {bound})");
            }
        }
    }
    eprintln!("f8 vs cpu reference: worst rel_l2 f32 out {:.3e}, bf16 out {:.3e}", worst[0], worst[1]);
}

#[test]
fn f8_rows_do_not_depend_on_m() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(82);
    for (n, k) in [(1024usize, 2048usize), (259, 1040), (4096, 4096)] {
        let big = case(&mut rng, 300, n, k, DType::F32, 1.3, 0.2);
        let whole = run_host(&*hip, "matmul_fp8_mma_f8", &big).unwrap().remove(0);
        let rowb = k * 2;
        for (r0, mm) in [(0usize, 5usize), (100, 37), (7, 129), (171, 129), (295, 5)] {
            let mm = mm.min(300 - r0);
            if mm < 5 {
                continue;
            }
            let mut part = big.clone();
            part.inputs[0] = HostTensor::new(DType::Bf16, vec![mm, k], big.inputs[0].data[r0 * rowb..(r0 + mm) * rowb].to_vec()).unwrap();
            part.outputs[0].shape = vec![mm, n];
            let got = run_host(&*hip, "matmul_fp8_mma_f8", &part).unwrap().remove(0);
            let want = &whole.data[r0 * n * 4..(r0 + mm) * n * 4];
            assert!(got.data == want, "n={n} k={k}: rows {r0}..{} computed in a call of {mm} rows differ from the same rows of a call of 300", r0 + mm);
        }
    }
}

#[test]
fn f8_is_not_the_bf16_path() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(83);
    let mut differing = 0usize;
    for (m, n, k) in [(129usize, 2048usize, 2048usize), (300, 1024, 4096), (64, 4096, 2048)] {
        let hc = case(&mut rng, m, n, k, DType::F32, 1.0, 0.25);
        let f8 = run_host(&*hip, "matmul_fp8_mma_f8", &hc).unwrap().remove(0);
        let bf = run_host(&*hip, "matmul_fp8_mma", &hc).unwrap().remove(0);
        let r = rel_l2(&f8.to_f32(), &bf.to_f32());
        assert!(r < 1e-6, "m={m} n={n} k={k}: f8 differs from the bf16 path by rel_l2 {r}");
        differing += nbits_differ(&f8, &bf);
    }
    assert!(differing > 0, "the e4m3 kernel produced the bf16 kernel's bits exactly: it did not run (or the bf16 kernel overwrote it)");
    eprintln!("f8 vs matmul_fp8_mma: {differing} elements differ in the last bits (f32 out)");
}

#[test]
fn f8_weight_only_is_the_bf16_path() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(84);
    for (m, n, k) in [(5usize, 512usize, 2048usize), (129, 1000, 1040), (300, 2048, 4096), (33, 17, 48)] {
        for odt in [DType::F32, DType::Bf16] {
            let hc = case(&mut rng, m, n, k, odt, 1.7, 0.0);
            let f8 = run_host(&*hip, "matmul_fp8_mma_f8", &hc).unwrap().remove(0);
            let bf = run_host(&*hip, "matmul_fp8_mma", &hc).unwrap().remove(0);
            assert!(f8.data == bf.data, "m={m} n={n} k={k} {odt:?}: weight-only launch differs from matmul_fp8_mma");
        }
    }
}
