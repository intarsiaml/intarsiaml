//! The fused decode kernels (`matmul_pair_bf16_gemv`, `matmul_fp8_pair_gemv`, `matmul_nvfp4_glu_gemv`,
//! `gated_rmsnorm_bf16`) against the launches they replace, and against the cpu reference.
//!
//! A fused kernel runs the arithmetic of the kernels it merges in the same order, so its outputs must be *bitwise* the
//! outputs of that chain on the same device (this is what keeps the 35B accuracy numbers and the graph/eager proof
//! unmoved: the model computes the same bits with fewer launches). The cpu-reference check (admission budget) is
//! independent of that. The shapes include the model's (Qwen3.6-35B-A3B: GDN qkv + z, attention k + v, GDN b + a, router
//! + shared-expert gate, shared gate + up, the dv = 128 gated norm) and the edges: row counts that are not a block
//! multiple, a segment shorter than one block, K past the unrolled window, odd widths (the scalar path of the norm), W8A8
//! and W8A16 segments in one call, segments with different weight and input scales. Segments with *different* parameters
//! are the point: a kernel that read the first segment's scale for the second, or shifted the second segment's block
//! index, must fail here (scripts/cuda-decode/mutate.py applies such mutations to a copy of the tree and requires this
//! file to fail on each).
//!
//! Needs a GPU: without one the tests print SKIPPED (and fail under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_cuda)]

mod common;

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::backend::Backend;
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::quant::cases::{gated_rmsnorm_case, matmul_fp8_pair_case, matmul_nvfp4_glu_case, matmul_pair_case};
use intarsia_core::quant::fp8::scales_tensor;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, Op, OpAttrs};
use std::sync::Arc;

fn device(test: &str) -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_cuda::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            common::skip(test, &format!("no usable CUDA device ({e})"));
            None
        }
    }
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

fn call(op: Op, inputs: Vec<HostTensor>, outputs: Vec<OutSpec>, attrs: OpAttrs) -> HostCall {
    HostCall { op, inputs, outputs, attrs }
}

fn bits_eq(label: &str, what: &str, got: &HostTensor, want: &HostTensor) {
    assert_eq!(got.shape, want.shape, "{label}: {what} shape");
    if got.data != want.data {
        let (g, w) = (got.to_f32(), want.to_f32());
        let bad = g.iter().zip(&w).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let first = g.iter().zip(&w).position(|(a, b)| a.to_bits() != b.to_bits()).unwrap();
        panic!(
            "{label}: {what} differs from the unfused launches in {bad} of {} elements (first at {first}: {} vs {})",
            g.len(),
            g[first],
            w[first]
        );
    }
}

/// The fused kernel's outputs are within `rel` of the cpu reference kernel (admission-style check).
fn near_reference(op: Op, hc: &HostCall, got: &[HostTensor], rel: f64, label: &str) {
    let cpu = CpuBackend::new();
    let reference = cpu.kernels().into_iter().find(|k| k.op == op && k.is_reference).unwrap().name;
    let want = run_host(&cpu, &reference, hc).unwrap();
    for (o, (g, w)) in got.iter().zip(&want).enumerate() {
        let r = rel_l2(&g.to_f32(), &w.to_f32());
        assert!(r < rel, "{label}: output {o} vs {reference}: rel_l2 {r}");
    }
}

fn out_bf16(shape: &[usize]) -> OutSpec {
    OutSpec::new(DType::Bf16, shape)
}

#[test]
fn matmul_pair_is_bitwise_two_gemvs() {
    let Some(cuda) = device("matmul_pair_is_bitwise_two_gemvs") else { return };
    let mut rng = SplitMix64::new(61);
    // (n0, n1, k): GDN b + a, MoE router + shared-expert gate, a segment shorter than a block, an odd split, K = 8
    for (n0, n1, k) in [
        (32usize, 32usize, 2048usize),
        (256, 1, 2048),
        (33, 1, 96),
        (300, 5, 520),
        (9, 17, 40),
        (1, 1, 8),
        (7, 4097, 128),
        (2048, 2048, 2048),
    ] {
        let hc = matmul_pair_case(&mut rng, 1, n0, n1, k, DType::Bf16);
        let label = format!("pair n0={n0} n1={n1} k={k}");
        let got = run_host(&*cuda, "matmul_pair_bf16_gemv", &hc).unwrap();
        for (j, w) in [(0usize, 1usize), (1, 2)] {
            let one = call(
                Op::Matmul,
                vec![hc.inputs[0].clone(), hc.inputs[w].clone()],
                vec![hc.outputs[j].clone()],
                OpAttrs::default(),
            );
            let want = run_host(&*cuda, "matmul_bf16_gemv", &one).unwrap();
            bits_eq(&label, &format!("output {j}"), &got[j], &want[0]);
        }
        near_reference(Op::MatmulPair, &hc, &got, 1e-3, &label);
    }
}

/// Replace the second segment's scales: a different weight scale (and input scale) from the first one's.
fn with_second_scales(mut hc: HostCall, ws_factor: f32, input_scale: f32) -> HostCall {
    let s = hc.inputs[4].to_f32();
    hc.inputs[4] = scales_tensor(s[0] * ws_factor, input_scale);
    hc
}

#[test]
fn matmul_fp8_pair_is_bitwise_two_gemvs() {
    let Some(cuda) = device("matmul_fp8_pair_is_bitwise_two_gemvs") else { return };
    let mut rng = SplitMix64::new(62);
    // (n0, n1, k): GDN qkv + z, attention k + v, small / odd shapes, a long K
    for (n0, n1, k) in [
        (8192usize, 4096usize, 2048usize),
        (512, 512, 2048),
        (64, 48, 64),
        (17, 33, 48),
        (130, 7, 1040),
        (1, 1, 16),
        (4096, 2048, 4096),
        (16, 16, 16),
    ] {
        // (segment 0 W8A8, segment 1 W8A8 / W8A16, input scale of segment 1)
        for (aq0, aq1, is1) in
            [(true, true, 0.125f32), (true, true, 0.3), (false, false, 0.0), (true, false, 0.0), (false, true, 0.25)]
        {
            let hc = matmul_fp8_pair_case(&mut rng, 1, n0, n1, k, aq0, aq1, DType::Bf16, DType::Bf16);
            let hc = with_second_scales(hc, 1.7, if aq1 { is1 } else { 0.0 });
            let label = format!("fp8 pair n0={n0} n1={n1} k={k} aq={aq0}/{aq1} is1={is1}");
            let got = run_host(&*cuda, "matmul_fp8_pair_gemv", &hc).unwrap();
            for (j, w, sc) in [(0usize, 1usize, 2usize), (1, 3, 4)] {
                let one = call(
                    Op::MatmulFp8,
                    vec![hc.inputs[0].clone(), hc.inputs[w].clone(), hc.inputs[sc].clone()],
                    vec![hc.outputs[j].clone()],
                    OpAttrs::default(),
                );
                let want = run_host(&*cuda, "matmul_fp8_gemv", &one).unwrap();
                bits_eq(&label, &format!("output {j}"), &got[j], &want[0]);
            }
            near_reference(Op::MatmulFp8Pair, &hc, &got, 1e-3, &label);
        }
    }
}

#[test]
fn matmul_nvfp4_glu_is_bitwise_gate_up_silu() {
    let Some(cuda) = device("matmul_nvfp4_glu_is_bitwise_gate_up_silu") else { return };
    let mut rng = SplitMix64::new(63);
    // (n, k): the shared expert (512 x 2048), its transpose, K past the 4-iteration unrolled window (4112), N not a
    // multiple of the 8 warps of a block, one K block
    for (n, k) in [(512usize, 2048usize), (2048, 512), (300, 4112), (9, 16), (64, 64), (40, 96), (1, 16), (1030, 160)] {
        let mut hc = matmul_nvfp4_glu_case(&mut rng, 1, n, k, DType::Bf16, DType::Bf16);
        // different global scales for gate and up: using one for both must be visible
        let gs = hc.inputs[3].to_f32()[0];
        hc.inputs[3] = HostTensor::from_f32(&[1], &[gs * 0.8]);
        hc.inputs[6] = HostTensor::from_f32(&[1], &[gs * 1.3]);
        let label = format!("nvfp4 glu n={n} k={k}");
        let got = run_host(&*cuda, "matmul_nvfp4_glu_gemv", &hc).unwrap();
        let proj = |w: usize| {
            let one = call(
                Op::MatmulNvfp4,
                vec![hc.inputs[0].clone(), hc.inputs[w].clone(), hc.inputs[w + 1].clone(), hc.inputs[w + 2].clone()],
                vec![out_bf16(&[1, n])],
                OpAttrs::default(),
            );
            run_host(&*cuda, "matmul_nvfp4_gemv", &one).unwrap().remove(0)
        };
        let (g, u) = (proj(1), proj(4));
        let glu = call(Op::SiluGlu, vec![g, u], vec![out_bf16(&[1, n])], OpAttrs::default());
        let want = run_host(&*cuda, "silu_glu_bf16", &glu).unwrap();
        bits_eq(&label, "output", &got[0], &want[0]);
        near_reference(Op::MatmulNvfp4Glu, &hc, &got, 6e-3, &label);
    }
}

#[test]
fn gated_rmsnorm_is_bitwise_rmsnorm_then_silu_glu() {
    let Some(cuda) = device("gated_rmsnorm_is_bitwise_rmsnorm_then_silu_glu") else { return };
    let mut rng = SplitMix64::new(64);
    // (rows, d): the GDN output norm at decode (32 value heads of 128) and prefill sizes, the scalar path (d % 8 != 0), a
    // block of more than 32 threads (d = 2048), tiny widths
    for (r, d) in [(32usize, 128usize), (1, 128), (5, 64), (33, 130), (960, 128), (7, 256), (3, 8), (2, 2048), (4, 1)] {
        for w_offset in [0.0f32, 1.0] {
            let mut hc = gated_rmsnorm_case(&mut rng, r, d, DType::Bf16);
            hc.attrs.w_offset = w_offset;
            let label = format!("gated rmsnorm r={r} d={d} w_offset={w_offset}");
            let got = run_host(&*cuda, "gated_rmsnorm_bf16", &hc).unwrap();
            let norm = call(
                Op::RmsNorm,
                vec![hc.inputs[0].clone(), hc.inputs[2].clone()],
                vec![out_bf16(&[r, d])],
                hc.attrs.clone(),
            );
            let normed = run_host(&*cuda, "rmsnorm_bf16", &norm).unwrap().remove(0);
            let glu = call(Op::SiluGlu, vec![hc.inputs[1].clone(), normed], vec![out_bf16(&[r, d])], OpAttrs::default());
            let want = run_host(&*cuda, "silu_glu_bf16", &glu).unwrap();
            bits_eq(&label, "output", &got[0], &want[0]);
            near_reference(Op::GatedRmsnorm, &hc, &got, 6e-3, &label);
        }
    }
}
