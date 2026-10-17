//! No GPU needed: the plain-Rust reference of the quantised arithmetic of `moe_experts_nvfp4_mma_bf16`
//! (tests/moe_nvfp4_ref) against the e2m1 grid and against the bf16-activation cpu reference. The second one is where
//! `NVFP4_ACT_TOL` (the tolerance the GPU kernel is registered with) comes from: the measured activation-quantisation
//! error over the shapes the admission sweep gives this kernel, plus a margin.
mod moe_nvfp4_ref;

use intarsia_backend_cpu::CpuBackend;
use intarsia_backend_cuda::quantmoe::NVFP4_ACT_TOL;
use intarsia_core::exec::run_host;
use intarsia_core::quant::cases::experts_case;
use intarsia_core::rng::SplitMix64;
use intarsia_core::DType;
use moe_nvfp4_ref::{bf16r, e2m1_rne, e2m1_value, quant_row, Quantised, E2M1};

#[test]
fn e2m1_rounding_is_nearest_even_and_saturates() {
    // the grid itself, and the ties (from vLLM's cast_to_fp4 reference thresholds)
    for (c, v) in E2M1.iter().enumerate() {
        assert_eq!(e2m1_rne(*v), c as u8);
        assert_eq!(e2m1_rne(-*v), c as u8 | 8);
    }
    for (x, want) in [(0.25f32, 0.0f64), (0.75, 1.0), (1.25, 1.0), (1.75, 2.0), (2.5, 2.0), (3.5, 4.0), (5.0, 4.0)] {
        assert_eq!(e2m1_value(e2m1_rne(x)), want, "tie at {x}");
        assert_eq!(e2m1_value(e2m1_rne(-x)), -want, "tie at -{x}");
    }
    for (x, want) in [(0.26f32, 0.5f64), (0.74, 0.5), (1.26, 1.5), (2.51, 3.0), (3.49, 3.0), (5.01, 6.0), (100.0, 6.0)] {
        assert_eq!(e2m1_value(e2m1_rne(x)), want, "{x}");
    }
}

#[test]
fn quant_row_reproduces_the_row_within_the_grid_step_and_handles_zeros() {
    let mut rng = SplitMix64::new(3);
    let x: Vec<f32> = rng.normal_vec(256, 1.0).into_iter().map(bf16r).collect();
    let (q, ginv) = quant_row(&x);
    // dequantised value * ginv should reproduce x: the e2m1 grid is at worst 1/3 of the value apart (steps 1 -> 1.5 -> 2
    // are 50% / 33%), so no element is further than a quarter of its block maximum off
    for (b, (xb, qb)) in x.chunks(16).zip(q.chunks(16)).enumerate() {
        let bmax = xb.iter().fold(0f32, |m, v| m.max(v.abs())) as f64;
        for (a, d) in xb.iter().zip(qb) {
            let err = (*a as f64 - *d * ginv as f64).abs();
            assert!(err <= 0.2 * bmax + 1e-6, "block {b}: {a} vs {}", d * ginv as f64);
        }
    }
    let (qz, gz) = quant_row(&[0f32; 32]);
    assert!(qz.iter().all(|v| *v == 0.0) && gz == 0.0);
}

/// Activation-quantisation error of the W4A4 arithmetic against the bf16-activation reference (`ref_moe_experts`) on the
/// shapes the admission sweep gives the kernel (T >= 32, H % 128, I % 128; bf16 activations). The 256-expert case is left
/// out (its dequantised weights do not fit comfortably in memory here); the largest one (60 tokens, 16 experts) is the
/// real layer size.
#[test]
fn activation_quantisation_error_is_inside_the_registered_tolerance() {
    let cpu = CpuBackend::new();
    let mut rng = SplitMix64::new(0x71);
    let mut worst = (0f64, 0f64);
    for (t, e, k, h, i) in [
        (36usize, 16usize, 4usize, 128usize, 128usize),
        (130, 16, 8, 256, 128),
        (45, 16, 4, 640, 384),
        (60, 16, 8, 2048, 512),
    ] {
        let hc = experts_case(&mut rng, t, e, k, h, i, DType::Bf16);
        let want = run_host(&cpu, "ref_moe_experts", &hc).unwrap()[0].to_f32();
        let got = Quantised::new(&hc).run();
        let num: f64 = got.iter().zip(&want).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
        let den: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum();
        let rel = (num / den).sqrt();
        let scale = want.iter().fold(1f32, |m, v| m.max(v.abs())) as f64;
        let max_abs = got.iter().zip(&want).fold(0f64, |m, (a, b)| m.max((a - b).abs() as f64)) / scale;
        println!("T={t} E={e} K={k} H={h} I={i}: rel_l2 {rel:.4}, max_abs / max(1, max|ref|) {max_abs:.4}");
        worst = (worst.0.max(max_abs), worst.1.max(rel));
        assert!(rel < NVFP4_ACT_TOL.rel_l2 as f64, "rel_l2 {rel}");
        assert!(max_abs < NVFP4_ACT_TOL.max_abs as f64, "max_abs {max_abs}");
    }
    println!("worst: max_abs {:.4}, rel_l2 {:.4}; registered tolerance max_abs {}, rel_l2 {}", worst.0, worst.1, NVFP4_ACT_TOL.max_abs, NVFP4_ACT_TOL.rel_l2);
}
