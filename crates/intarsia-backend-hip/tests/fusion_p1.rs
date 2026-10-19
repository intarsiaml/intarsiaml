//! The core-op fusion phase-1 kernels (docs/results/2026-10-19-core-op-fusion-p1.md) against the launches they replace on
//! the same device, bitwise, and against the cpu reference (admission budget):
//!
//! * `rmsnorm_add_matmul_fp8_pair_gemv` (P5) = `rmsnorm_add_bf16` -> `matmul_fp8_pair_gemv` on the normed row; all four
//!   outputs (normed, sum, k, v) compared.
//! * `sigmoid_mul_matmul_fp8_gemv` (P4b) = `sigmoid_mul_bf16` -> `matmul_fp8_gemv_v2`.
//! * `shared_down_combine_nvfp4_gemv` (P2a) = `matmul_nvfp4_gemv_v2` -> `moe_combine_bf16`, M = 1..8 with a distinct logit
//!   per row and non-zero routed values.
//!
//! The shapes include the 35B's (attention k + v at hidden 2048, o_proj 2048 x 4096, shared down 2048 x 512) and the edges:
//! K below one 256-thread block of chunks, K past the GEMV prefetch window and with a chunk tail (4112), row counts that are
//! not a block multiple, segments with different scales and activation modes. scripts/hip-decode/mutate.py has the mutants
//! (`p1-*`) this file must fail on.
//!
//! Needs a GPU: skips (passes) without one.
#![cfg(ik_hip)]

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::backend::Backend;
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::quant::cases::{rmsnorm_add_matmul_fp8_pair_case, shared_down_combine_case, sigmoid_mul_matmul_fp8_case};
use intarsia_core::quant::fp8::scales_tensor;
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

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

fn bits_eq(label: &str, what: &str, got: &HostTensor, want: &HostTensor) {
    assert_eq!(got.shape, want.shape, "{label}: {what} shape");
    if got.data != want.data {
        let (g, w) = (got.to_f32(), want.to_f32());
        let bad = g.iter().zip(&w).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let first = g.iter().zip(&w).position(|(a, b)| a.to_bits() != b.to_bits()).unwrap();
        panic!("{label}: {what} differs from the unfused launches in {bad} of {} elements (first at {first}: {} vs {})", g.len(), g[first], w[first]);
    }
}

fn near_reference(op: Op, hc: &HostCall, got: &[HostTensor], rel: f64, label: &str) {
    let cpu = CpuBackend::new();
    let reference = cpu.kernels().into_iter().find(|k| k.op == op && k.is_reference).unwrap().name;
    let want = run_host(&cpu, &reference, hc).unwrap();
    for (o, (g, w)) in got.iter().zip(&want).enumerate() {
        let r = rel_l2(&g.to_f32(), &w.to_f32());
        assert!(r < rel, "{label}: output {o} vs {reference}: rel_l2 {r}");
    }
}

fn call(op: Op, inputs: Vec<HostTensor>, outputs: Vec<OutSpec>, attrs: OpAttrs) -> HostCall {
    HostCall { op, inputs, outputs, attrs }
}

#[test]
fn rmsnorm_add_matmul_fp8_pair_is_bitwise_norm_then_pair() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(71);
    // (n0, n1, k): the 35B's attention k + v (2 kv heads x 256) at hidden 2048, the 8B-like 4096, a K past the prefetch
    // window with a chunk tail, K below one block of 256 chunks (the norm's extra threads idle), segments shorter than a block
    let shapes = [(512usize, 512usize, 2048usize), (1024, 1024, 4096), (24, 40, 4112), (64, 48, 64), (17, 33, 48), (9, 300, 1024)];
    let mut n = 0;
    for &(n0, n1, k) in &shapes {
        for (aq0, aq1, w_offset, second) in
            [(true, true, 1.0f32, None), (false, false, 0.0, None), (true, false, 1.0, None), (true, true, 1.0, Some((3.0f32, 0.02f32)))]
        {
            let mut hc = rmsnorm_add_matmul_fp8_pair_case(&mut rng, 1, n0, n1, k, aq0, aq1, DType::Bf16, w_offset);
            if let Some((f, is)) = second {
                // the second segment's own weight and input scale (a kernel that read segment 0's scales fails here)
                let s = hc.inputs[6].to_f32();
                hc.inputs[6] = scales_tensor(s[0] * f, is);
            }
            let label = format!("P5 n0={n0} n1={n1} k={k} aq={aq0}/{aq1} w_offset={w_offset} second={second:?}");
            let got = run_host(&*hip, "rmsnorm_add_matmul_fp8_pair_gemv", &hc).unwrap_or_else(|e| panic!("{label}: {e}"));
            let norm = call(Op::RmsNormAdd, hc.inputs[..3].to_vec(), hc.outputs[..2].to_vec(), hc.attrs.clone());
            let nw = run_host(&*hip, "rmsnorm_add_bf16", &norm).unwrap();
            let pair = call(
                Op::MatmulFp8Pair,
                vec![nw[0].clone(), hc.inputs[3].clone(), hc.inputs[4].clone(), hc.inputs[5].clone(), hc.inputs[6].clone()],
                hc.outputs[2..].to_vec(),
                OpAttrs::default(),
            );
            let pw = run_host(&*hip, "matmul_fp8_pair_gemv", &pair).unwrap();
            for (j, (what, want)) in [("normed", &nw[0]), ("sum", &nw[1]), ("k", &pw[0]), ("v", &pw[1])].into_iter().enumerate() {
                bits_eq(&label, what, &got[j], want);
            }
            near_reference(Op::RmsNormAddMatmulFp8Pair, &hc, &got, 6e-3, &label);
            n += 1;
        }
    }
    eprintln!("P5: {n} cases bitwise");
}

#[test]
fn sigmoid_mul_matmul_fp8_is_bitwise_gate_then_gemv() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(72);
    // (n, k): the 35B's o_proj (2048 x 4096), small / odd shapes, a K past the prefetch window with a tail, a wide N (4 rows
    // per warp in gemv_rows) and a narrow one (1 row per warp)
    let shapes = [(2048usize, 4096usize), (64, 64), (40, 96), (17, 48), (520, 2064), (8192, 512), (3, 4112)];
    let mut n = 0;
    for &(nn, k) in &shapes {
        for aq in [true, false] {
            let hc = sigmoid_mul_matmul_fp8_case(&mut rng, 1, nn, k, aq, DType::Bf16, DType::Bf16);
            let label = format!("P4b n={nn} k={k} aq={aq}");
            let got = run_host(&*hip, "sigmoid_mul_matmul_fp8_gemv", &hc).unwrap_or_else(|e| panic!("{label}: {e}"));
            let gate =
                call(Op::SigmoidMul, hc.inputs[..2].to_vec(), vec![OutSpec::new(DType::Bf16, &[1, k])], OpAttrs::default());
            let g = run_host(&*hip, "sigmoid_mul_bf16", &gate).unwrap();
            let mm = call(
                Op::MatmulFp8,
                vec![g[0].clone(), hc.inputs[2].clone(), hc.inputs[3].clone()],
                hc.outputs.clone(),
                OpAttrs::default(),
            );
            let want = run_host(&*hip, "matmul_fp8_gemv_v2", &mm).unwrap();
            bits_eq(&label, "out", &got[0], &want[0]);
            near_reference(Op::SigmoidMulMatmulFp8, &hc, &got, 6e-3, &label);
            n += 1;
        }
    }
    eprintln!("P4b: {n} cases bitwise");
}

#[test]
fn shared_down_combine_is_bitwise_down_then_combine() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(73);
    // (n, k): the 35B's shared-expert down (2048 x 512), small / odd shapes, a K past the prefetch window (4112: 257
    // lane-iterations), an N that is not a multiple of the 8 warps
    let shapes = [(2048usize, 512usize), (64, 64), (40, 96), (9, 16), (300, 4112), (2049, 512)];
    let mut n = 0;
    for &(nn, k) in &shapes {
        for m in 1..=8usize {
            let hc = shared_down_combine_case(&mut rng, m, nn, k, DType::Bf16, DType::Bf16);
            let label = format!("P2a m={m} n={nn} k={k}");
            let got = run_host(&*hip, "shared_down_combine_nvfp4_gemv", &hc).unwrap_or_else(|e| panic!("{label}: {e}"));
            let down = call(Op::MatmulNvfp4, hc.inputs[..4].to_vec(), hc.outputs.clone(), OpAttrs::default());
            let so = run_host(&*hip, "matmul_nvfp4_gemv_v2", &down).unwrap();
            let comb = call(
                Op::MoeCombine,
                vec![hc.inputs[4].clone(), so[0].clone(), hc.inputs[5].clone()],
                hc.outputs.clone(),
                OpAttrs::default(),
            );
            let want = run_host(&*hip, "moe_combine_bf16", &comb).unwrap();
            bits_eq(&label, "out", &got[0], &want[0]);
            near_reference(Op::SharedDownCombine, &hc, &got, 6e-3, &label);
            n += 1;
        }
    }
    eprintln!("P2a: {n} cases bitwise");
}

/// Shapes the fused kernels do not take return an error (the plan then never selects them: `when()` refuses them first),
/// never a wrong result: two rows for the fp8 folds, nine for the nvfp4 one.
#[test]
fn out_of_range_shapes_are_refused() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(74);
    let hc = rmsnorm_add_matmul_fp8_pair_case(&mut rng, 2, 64, 48, 64, true, true, DType::Bf16, 1.0);
    assert!(run_host(&*hip, "rmsnorm_add_matmul_fp8_pair_gemv", &hc).is_err());
    let hc = sigmoid_mul_matmul_fp8_case(&mut rng, 2, 64, 64, true, DType::Bf16, DType::Bf16);
    assert!(run_host(&*hip, "sigmoid_mul_matmul_fp8_gemv", &hc).is_err());
    let hc = shared_down_combine_case(&mut rng, 9, 64, 64, DType::Bf16, DType::Bf16);
    assert!(run_host(&*hip, "shared_down_combine_nvfp4_gemv", &hc).is_err());
}
