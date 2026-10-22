//! The decode-path "v2" kernels (latency-restructured twins of `matmul_nvfp4_gemv`, `moe_topk_bf16` and the decode path
//! of `moe_experts_bf16`) against their v1 kernels and the cpu reference.
//!
//! They are registered kernels (so the admission sweep grades them like any other), and these tests pin more than the
//! sweep's tolerance does: a v2 kernel computes the same arithmetic in the same order as v1, so its outputs must be
//! *bitwise* the v1 outputs, on shapes that exercise every special case (rows and K past the prefetch window, row
//! tails, ties in the top-k, skewed and invalid routing). Any mutation of a v2 kernel that changes one accumulated value
//! (a dropped iteration, a wrong scale, a swapped tie-break) breaks bit-equality, and the cpu-reference check (rel_l2
//! 1e-3, the admission budget) catches the same mutation independently of v1.
//!
//! Needs a GPU: skips (passes) without one.
#![cfg(ik_hip)]

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::backend::Backend;
use intarsia_core::exec::{run_host, HostCall};
use intarsia_core::quant::cases::{experts_case, matmul_fp8_case, matmul_nvfp4_case, topk_case};
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

/// v2 == v1 bit for bit on every output, and both within the admission budget of the cpu reference kernel.
fn same_as_v1(hip: &dyn Backend, v1: &str, v2: &str, op: Op, hc: &HostCall, label: &str) {
    let cpu = CpuBackend::new();
    let reference = cpu.kernels().into_iter().find(|k| k.op == op && k.is_reference).unwrap().name;
    let a = run_host(hip, v1, hc).unwrap();
    let b = run_host(hip, v2, hc).unwrap();
    let want = run_host(&cpu, &reference, hc).unwrap();
    for (o, (x, y)) in a.iter().zip(&b).enumerate() {
        assert!(x.data == y.data, "{label}: output {o} of {v2} differs from {v1}");
    }
    // the first output carries the values (top-k's second output is the ids, compared exactly above)
    if a[0].dtype != DType::I32 {
        let r = rel_l2(&b[0].to_f32(), &want[0].to_f32());
        assert!(r < 1e-3, "{label}: {v2} vs {reference}: rel_l2 {r}");
    }
}

#[test]
fn nvfp4_gemv_v2_is_bitwise_v1() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(31);
    // (m, n, k): the model's shared-expert and router-adjacent shapes; K past the 4-iteration prefetch window (4112 =
    // 257 blocks per row, 9 lane-iterations for lane 0); N not a multiple of the 8 warps of a block; K = 16 (one block);
    // M up to the kernel's limit of 8
    for (m, n, k) in [
        (1usize, 512usize, 2048usize),
        (1, 2048, 512),
        (1, 300, 48),
        (1, 17, 4112),
        (2, 513, 2064),
        (4, 96, 1024),
        (8, 64, 16),
        (1, 4096, 160),
    ] {
        for odt in [DType::Bf16, DType::F32] {
            let hc = matmul_nvfp4_case(&mut rng, m, n, k, DType::Bf16, odt);
            same_as_v1(&*hip, "matmul_nvfp4_gemv", "matmul_nvfp4_gemv_v2", Op::MatmulNvfp4, &hc, &format!("nvfp4 gemv m={m} n={n} k={k} {odt:?}"));
        }
    }
}

#[test]
fn moe_topk_v2_is_bitwise_v1() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(32);
    // ties on (coarse grid values), both normalisations, partial last block of tokens (T = 5, 64), E = 8 .. 512, K = 1 .. 8
    for (t, e, k) in [(1usize, 256usize, 8usize), (5, 256, 8), (64, 512, 8), (3, 64, 1), (2, 8, 8), (7, 128, 3), (1, 512, 32)] {
        for ties in [false, true] {
            for norm in [true, false] {
                let hc = topk_case(&mut rng, t, e, k, ties, DType::Bf16, norm);
                same_as_v1(&*hip, "moe_topk_bf16", "moe_topk_bf16_v2", Op::MoeTopk, &hc, &format!("topk t={t} e={e} k={k} ties={ties} norm={norm}"));
            }
        }
    }
}

#[test]
fn moe_experts_v2_is_bitwise_v1() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(33);
    // decode (P = T*K <= 16, the p1 path) at the real shape and at small / odd ones (I / 16 below the 32 lanes, H / 16 above
    // the prefetch window), plus the grouped paths v2 hands to the v1 kernels (T = 4 -> P = 32, T = 40)
    for (t, e, k, h, i) in [
        (1usize, 256usize, 8usize, 2048usize, 512usize),
        (1, 256, 8, 2048, 256), // the 35B tp=2 shard (lane H1)
        (2, 64, 8, 2048, 512),
        (1, 32, 4, 256, 48),
        (1, 16, 2, 4112, 160),
        (2, 16, 8, 64, 16),
        (4, 32, 8, 512, 128),
        (40, 16, 4, 256, 64),
    ] {
        let hc = experts_case(&mut rng, t, e, k, h, i, DType::Bf16);
        same_as_v1(&*hip, "moe_experts_bf16", "moe_experts_bf16_v2", Op::MoeExperts, &hc, &format!("experts t={t} e={e} k={k} h={h} i={i}"));
    }
    // routing the sweep does not draw: every token on the same experts, a duplicated expert, out-of-range ids
    let base = experts_case(&mut rng, 2, 16, 4, 256, 64, DType::Bf16);
    let mut odd = base.clone();
    let mut ids = odd.inputs[1].to_i32().unwrap();
    ids[0] = 3;
    ids[1] = 3;
    ids[2] = -1;
    ids[3] = 16 + 5;
    odd.inputs[1] = HostTensor::from_i32(&[2, 4], &ids);
    same_as_v1(&*hip, "moe_experts_bf16", "moe_experts_bf16_v2", Op::MoeExperts, &odd, "duplicate and invalid ids");
}

/// The W8A8 / W8A16 fp8 GEMV: v2 stages the activations eight per thread (and, with INTARSIA_HIP_FP8_Q=2/3, converts to
/// e4m3 with the hardware instruction); the staged values must be exactly the software conversion's, so the outputs are
/// bitwise v1's. x is also drawn scaled up (most of it saturating past +-448 * input_scale) and down (e4m3 subnormals and
/// underflow to zero), where the rounding rules differ most.
#[test]
fn fp8_gemv_v2_is_bitwise_v1() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(34);
    for (m, n, k) in [(1usize, 4096usize, 2048usize), (1, 8192, 2048), (1, 2048, 4096), (1, 512, 2048), (1, 37, 48), (2, 130, 1040), (4, 64, 16)] {
        for act_quant in [true, false] {
            for odt in [DType::Bf16, DType::F32] {
                let base = matmul_fp8_case(&mut rng, m, n, k, act_quant, DType::Bf16, odt);
                for gain in [1.0f32, 400.0, 3e-4, 1e-8] {
                    let mut hc = base.clone();
                    let xs: Vec<f32> = hc.inputs[0].to_f32().iter().map(|v| v * gain).collect();
                    hc.inputs[0] = HostTensor::from_f32_as(DType::Bf16, &[m, k], &xs);
                    same_as_v1(
                        &*hip,
                        "matmul_fp8_gemv",
                        "matmul_fp8_gemv_v2",
                        Op::MatmulFp8,
                        &hc,
                        &format!("fp8 gemv m={m} n={n} k={k} actq={act_quant} {odt:?} gain={gain}"),
                    );
                }
            }
        }
    }
}

/// Fusions inside existing launchers: the single-tile `gdn_conv_bf16` writes its own new state (no `gdn_conv_state` launch),
/// and the partial-rotary `rope_bf16` copies the un-rotated tail itself (no `rope_pass` launch). Both against the cpu
/// reference: the conv state is a copy of earlier inputs, so it must match *exactly* (a dropped, shifted or stale state write
/// fails), the rotated values to the bf16 admission budget, the copied tail exactly.
#[test]
fn fused_conv_state_and_rope_tail_match_the_reference() {
    use intarsia_core::exec::OutSpec;
    use intarsia_core::{OpAttrs};
    let Some(hip) = device() else { return };
    let cpu = CpuBackend::new();
    let mut rng = SplitMix64::new(35);
    let rt = |rng: &mut SplitMix64, shape: &[usize], scale: f32| {
        let n: usize = shape.iter().product();
        HostTensor::from_f32_as(DType::Bf16, shape, &rng.normal_vec(n, scale))
    };
    let refname = |op: Op| cpu.kernels().into_iter().find(|k| k.op == op && k.is_reference).unwrap().name;
    // conv: T = 1 (decode), 2, 32 (the last single-tile length), 33 and 70 (two kernels, unchanged path); K = 4 as in the model
    for (t, c, k) in [(1usize, 8192usize, 4usize), (2, 300, 4), (32, 257, 4), (33, 257, 4), (70, 129, 3), (1, 128, 2)] {
        let init = HostTensor::from_f32_as(DType::F32, &[c, k - 1], &rng.normal_vec(c * (k - 1), 1.0));
        let hc = HostCall {
            op: Op::GdnConv,
            inputs: vec![rt(&mut rng, &[t, c], 1.0), rt(&mut rng, &[c, k], 0.5)],
            outputs: vec![OutSpec::new(DType::Bf16, &[t, c]), OutSpec { dtype: DType::F32, shape: vec![c, k - 1], init: Some(init) }],
            attrs: OpAttrs::default(),
        };
        let got = run_host(&*hip, "gdn_conv_bf16", &hc).unwrap();
        let want = run_host(&cpu, &refname(Op::GdnConv), &hc).unwrap();
        let r = rel_l2(&got[0].to_f32(), &want[0].to_f32());
        assert!(r < 1e-3, "conv T={t} C={c} K={k}: y rel_l2 {r}");
        assert!(got[1].data == want[1].data, "conv T={t} C={c} K={k}: the new state differs from the reference");
    }
    // rope: partial rotary (64 of 256, the 35B's attention heads), one decode token and a chunk, x != out (run_host always
    // uses separate buffers), plus a full rotary control
    for (t, n, d, rd) in [(1usize, 16usize, 256usize, 64usize), (1, 2, 256, 64), (5, 16, 256, 64), (1, 16, 128, 128), (3, 5, 96, 32)] {
        let hc = HostCall {
            op: Op::Rope,
            inputs: vec![rt(&mut rng, &[t, n, d], 1.0)],
            outputs: vec![OutSpec::new(DType::Bf16, &[t, n, d])],
            attrs: OpAttrs { rotary_dim: rd, theta: 1e7, pos0: 37, ..OpAttrs::default() },
        };
        let got = run_host(&*hip, "rope_bf16", &hc).unwrap();
        let want = run_host(&cpu, &refname(Op::Rope), &hc).unwrap();
        let (g, w) = (got[0].to_f32(), want[0].to_f32());
        assert!(rel_l2(&g, &w) < 1e-3, "rope t={t} n={n} d={d} rd={rd}");
        for r in 0..t * n {
            for j in rd..d {
                assert!(g[r * d + j] == w[r * d + j], "rope tail value ({r},{j}) not copied: t={t} n={n} d={d} rd={rd}");
            }
        }
    }
}
