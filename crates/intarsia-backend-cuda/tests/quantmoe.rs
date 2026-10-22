//! cuda quantised-weight / MoE kernels: edge cases beyond the admission sweep (skewed and invalid routing,
//! multi-chunk experts) and an ignored microbenchmark on the real Qwen3.6-35B-A3B shapes
//! (`cargo test --release -p intarsia-backend-cuda --test quantmoe -- --ignored --nocapture`).
//! Needs a GPU: without one the tests print SKIPPED (and fail under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_cuda)]

mod common;

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::backend::{upload_new, Backend, MemKind, TensorRef};
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::quant::cases::{experts_case, matmul_fp8_case, matmul_nvfp4_case};
use intarsia_core::quant::ops::moe_experts_scratch;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, Op, OpCall, ShapeInfo};
use std::sync::Arc;
use std::time::Instant;

fn device() -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_cuda::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            common::skip("quantmoe", &format!("no usable CUDA device ({e})"));
            None
        }
    }
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

fn check_experts(cuda: &dyn Backend, hc: &HostCall, label: &str) {
    check_experts_with(cuda, "moe_experts_bf16", hc, label);
}

fn check_experts_with(cuda: &dyn Backend, kernel: &str, hc: &HostCall, label: &str) {
    let cpu = CpuBackend::new();
    let want = run_host(&cpu, "ref_moe_experts", hc).unwrap();
    let got = run_host(cuda, kernel, hc).unwrap();
    let r = rel_l2(&got[0].to_f32(), &want[0].to_f32());
    assert!(r < 1e-3, "{kernel} {label}: rel_l2 {r}");
    let again = run_host(cuda, kernel, hc).unwrap();
    assert_eq!(got[0].data, again[0].data, "{kernel} {label}: not repeatable");
}

/// The two-launch decode kernel does the same arithmetic as the warp-per-row kernel, so for T <= 4 the outputs are
/// bit-identical (including duplicate and invalid ids), and both match the reference.
#[test]
fn moe_experts_decode_matches_old_kernel() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(33);
    // (T, E, K, H, I): the real layer shape, more tokens, K below / above the 8 warps of a block (idle warps; a warp owning
    // two slots), I of one and of several lane iterations (I / 16 = 3 active lanes; 64 lane iterations of 32), K = E
    for (t, e, k, h, i) in [
        (1usize, 64usize, 8usize, 2048usize, 512usize),
        (2, 16, 8, 2048, 512),
        (4, 32, 8, 2048, 512),
        (3, 16, 4, 256, 64),
        (1, 8, 2, 64, 32),
        (1, 32, 10, 256, 1024),
        (2, 32, 16, 128, 32),
        (1, 16, 1, 64, 48),
        (2, 4, 4, 48, 16),
    ] {
        let base = experts_case(&mut rng, t, e, k, h, i, DType::Bf16);
        let mut odd = base.clone();
        let mut ids = odd.inputs[1].to_i32().unwrap();
        ids[0] = 3.min(e as i32 - 1);
        if ids.len() > 1 {
            ids[1] = ids[0]; // duplicate within a token
        }
        if ids.len() > 3 {
            ids[2] = -1;
            ids[3] = e as i32 + 5;
        }
        odd.inputs[1] = HostTensor::from_i32(&[t, k], &ids);
        for (label, hc) in [("random", &base), ("duplicate and invalid ids", &odd)] {
            let label = format!("{label} T={t} E={e} K={k} H={h} I={i}");
            let old = run_host(&*cuda, "moe_experts_bf16", hc).unwrap();
            // the v2 kernel (down projection restructured) must give the same bits as the original decode kernel
            for kernel in ["moe_experts_decode_bf16", "moe_experts_decode_v2_bf16"] {
                check_experts_with(&*cuda, kernel, hc, &label);
                let new = run_host(&*cuda, kernel, hc).unwrap();
                assert_eq!(old[0].data, new[0].data, "{label}: {kernel} differs from moe_experts_bf16");
            }
        }
    }
}

/// The grouped tensor-core kernel on skewed, duplicated and invalid routing at tile heights 16 / 32 / 64
/// (the kernel picks the height from pairs per expert: T*K/E <= 10 -> 16, <= 21 -> 32, else 64), including an
/// expert that owns many tiles and experts that own none.
#[test]
fn moe_experts_mma_matches_reference() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(21);
    let k = 4usize;
    for (t, e) in [(40usize, 16usize), (80, 16), (200, 16), (40, 64)] {
        let base = experts_case(&mut rng, t, e, k, 256, 128, DType::Bf16);
        check_experts_with(&*cuda, "moe_experts_mma_bf16", &base, &format!("random T={t} E={e}"));
        // every token picks experts 0..k: T pairs per expert, many tiles each, the other experts get none
        let mut hot = base.clone();
        let ids: Vec<i32> = (0..t).flat_map(|_| (0..k as i32)).collect();
        hot.inputs[1] = HostTensor::from_i32(&[t, k], &ids);
        check_experts_with(&*cuda, "moe_experts_mma_bf16", &hot, &format!("hot experts T={t} E={e}"));
        // duplicate ids within a token, out-of-range ids
        let mut odd = base.clone();
        let mut ids = odd.inputs[1].to_i32().unwrap();
        for j in 0..k {
            ids[j] = 3;
        }
        ids[k] = -1;
        ids[k + 1] = e as i32 + 5;
        odd.inputs[1] = HostTensor::from_i32(&[t, k], &ids);
        check_experts_with(&*cuda, "moe_experts_mma_bf16", &odd, &format!("duplicate and invalid ids T={t} E={e}"));
    }
    // real layer shape (H=2048, I=512), a handful of experts, tile counts that are not multiples of the tile height
    for (t, e) in [(33usize, 16usize), (100, 16)] {
        let hc = experts_case(&mut rng, t, e, 8, 2048, 512, DType::Bf16);
        check_experts_with(&*cuda, "moe_experts_mma_bf16", &hc, &format!("real shape T={t} E={e}"));
    }
}

/// The grouped tensor-core kernel against the warp-per-row kernel it replaces for prefill batches, on the same
/// inputs (both are also graded against the cpu reference above). The two sum in a different order, so the bf16
/// outputs may differ by rounding, never by more than rel_l2 1e-3 (the admission budget). Same shapes as
/// `moe_experts_mma_matches_reference` plus a larger real-layer one (H=2048, I=512). Runs only in the spark1
/// acceptance run (needs a GPU).
#[test]
fn moe_experts_mma_matches_old_kernel() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(21);
    let k = 4usize;
    let mut cases: Vec<(String, HostCall)> = Vec::new();
    for (t, e) in [(40usize, 16usize), (80, 16), (200, 16), (40, 64)] {
        let base = experts_case(&mut rng, t, e, k, 256, 128, DType::Bf16);
        let mut hot = base.clone();
        let ids: Vec<i32> = (0..t).flat_map(|_| (0..k as i32)).collect();
        hot.inputs[1] = HostTensor::from_i32(&[t, k], &ids);
        let mut odd = base.clone();
        let mut ids = odd.inputs[1].to_i32().unwrap();
        for j in 0..k {
            ids[j] = 3;
        }
        ids[k] = -1;
        ids[k + 1] = e as i32 + 5;
        odd.inputs[1] = HostTensor::from_i32(&[t, k], &ids);
        cases.push((format!("random T={t} E={e}"), base));
        cases.push((format!("hot experts T={t} E={e}"), hot));
        cases.push((format!("duplicate and invalid ids T={t} E={e}"), odd));
    }
    for (t, e) in [(33usize, 16usize), (100, 16), (256, 64)] {
        cases.push((format!("real shape T={t} E={e}"), experts_case(&mut rng, t, e, 8, 2048, 512, DType::Bf16)));
    }
    for (label, hc) in &cases {
        let old = run_host(&*cuda, "moe_experts_bf16", hc).unwrap();
        let new = run_host(&*cuda, "moe_experts_mma_bf16", hc).unwrap();
        let r = rel_l2(&new[0].to_f32(), &old[0].to_f32());
        assert!(r <= 1e-3, "{label}: moe_experts_mma_bf16 vs moe_experts_bf16 rel_l2 {r}");
    }
}

/// `matmul_*_gemm` (dequantise-once, cp.async pipeline) and `matmul_*_mma` run the same arithmetic in the same order,
/// so their bf16 outputs are claimed bit-identical (docs/quant-moe.md, "rel_l2 0"). Asserts equal output bytes on one
/// host call per shape, M in {33, 40, 100, 256} (the gemm kernels need M >= 33), K % 128 == 0. Runs only in the
/// spark1 acceptance run (needs a GPU); nothing has run it yet.
fn gemm_vs_mma_bitwise(cuda: &dyn Backend, gemm: &str, mma: &str, mk: &mut dyn FnMut(usize, usize, usize) -> HostCall) {
    for (m, n, k) in [(33usize, 192usize, 256usize), (40, 256, 128), (100, 192, 384), (256, 512, 2048)] {
        let hc = mk(m, n, k);
        let a = run_host(cuda, mma, &hc).unwrap();
        let b = run_host(cuda, gemm, &hc).unwrap();
        assert_eq!(a[0].data, b[0].data, "{gemm} vs {mma} differ at M={m} N={n} K={k}");
    }
}

/// See `gemm_vs_mma_bitwise`. Runs only in the spark1 acceptance run.
#[test]
fn matmul_nvfp4_gemm_is_bitwise_the_mma_kernel() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(71);
    gemm_vs_mma_bitwise(&*cuda, "matmul_nvfp4_gemm", "matmul_nvfp4_mma", &mut |m, n, k| {
        matmul_nvfp4_case(&mut rng, m, n, k, DType::Bf16, DType::Bf16)
    });
}

/// See `gemm_vs_mma_bitwise`; both W8A8 (activations quantised once) and weight-only. Runs only in the spark1
/// acceptance run.
#[test]
fn matmul_fp8_gemm_is_bitwise_the_mma_kernel() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(72);
    for act_quant in [true, false] {
        gemm_vs_mma_bitwise(&*cuda, "matmul_fp8_gemm", "matmul_fp8_mma", &mut |m, n, k| {
            matmul_fp8_case(&mut rng, m, n, k, act_quant, DType::Bf16, DType::Bf16)
        });
    }
}

#[test]
fn moe_experts_skewed_and_invalid_routing() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(11);
    let (t, e, k, h, i) = (40usize, 16usize, 4usize, 256usize, 64usize);
    let base = experts_case(&mut rng, t, e, k, h, i, DType::Bf16);
    // every token picks experts 0..k: 40 pairs per expert, five 8-pair chunks each
    let mut hot = base.clone();
    let ids: Vec<i32> = (0..t).flat_map(|_| (0..k as i32)).collect();
    hot.inputs[1] = HostTensor::from_i32(&[t, k], &ids);
    check_experts(&*cuda, &hot, "hot experts");
    // one expert receives every pair of token 0 (duplicate ids are legal input) and some ids are out of range
    let mut odd = base.clone();
    let mut ids = odd.inputs[1].to_i32().unwrap();
    for j in 0..k {
        ids[j] = 3;
    }
    ids[k] = -1;
    ids[k + 1] = e as i32 + 5;
    odd.inputs[1] = HostTensor::from_i32(&[t, k], &ids);
    check_experts(&*cuda, &odd, "duplicate and invalid ids");
    // a single token
    let one = experts_case(&mut rng, 1, 64, 8, 2048, 512, DType::Bf16);
    check_experts(&*cuda, &one, "T=1 real shape");
}

fn dev_tensor(b: &dyn Backend, t: &HostTensor) -> TensorRef {
    TensorRef::new(upload_new(b, &t.data).unwrap(), t.dtype, &t.shape)
}

fn bench(b: &dyn Backend, kernel: &str, hc: &HostCall, reps: usize) -> f64 {
    let inputs: Vec<TensorRef> = hc.inputs.iter().map(|t| dev_tensor(b, t)).collect();
    let outputs: Vec<TensorRef> =
        hc.outputs.iter().map(|o| dev_tensor(b, &HostTensor::zeros(o.dtype, &o.shape))).collect();
    let mut call = OpCall::new(hc.op, inputs, outputs, hc.attrs.clone());
    let need = b.kernels().iter().find(|k| k.name == kernel).map_or(0, |k| k.scratch_bytes(&ShapeInfo::from_call(&call)));
    if need > 0 {
        call.scratch = Some(TensorRef::new(b.alloc(need, MemKind::Device).unwrap(), DType::U8, &[need]));
    }
    for _ in 0..10 {
        b.launch(kernel, &call, 0).unwrap();
    }
    b.synchronize(0).unwrap();
    let t0 = Instant::now();
    for _ in 0..reps {
        b.launch(kernel, &call, 0).unwrap();
    }
    b.synchronize(0).unwrap();
    t0.elapsed().as_secs_f64() / reps as f64
}

/// Cold-weight timing of the T = 1 routed experts at the real layer shape (256 experts, top-8, H 2048, I 512). 96 different
/// routings cycle through one weight set, so consecutive launches read different experts (as the 40 layers of a decode step
/// do) instead of the L2-warm same eight, and all 96 launches are captured in one CUDA graph so the host launch cost (a few
/// us per call here) does not count. Prints microseconds per call and the effective bandwidth over the bytes touched.
/// Knobs: INTARSIA_MOE_DOWN_R, INTARSIA_MOE_GU_UNROLL (moe_decode.cu).
#[test]
#[ignore]
fn bench_moe_decode_cold() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(9);
    const N: usize = 96;
    let big = experts_case(&mut rng, N, 256, 8, 2048, 512, DType::Bf16);
    let weights: Vec<TensorRef> = big.inputs[3..8].iter().map(|t| dev_tensor(&*cuda, t)).collect();
    let row = |t: &HostTensor, i: usize, width: usize, bytes: usize| HostTensor {
        shape: vec![1, width],
        data: t.data[i * width * bytes..(i + 1) * width * bytes].to_vec(),
        ..t.clone()
    };
    let mut calls = Vec::new();
    for i in 0..N {
        let mut inputs = vec![
            dev_tensor(&*cuda, &row(&big.inputs[0], i, 2048, 2)),
            dev_tensor(&*cuda, &row(&big.inputs[1], i, 8, 4)),
            dev_tensor(&*cuda, &row(&big.inputs[2], i, 8, 4)),
        ];
        inputs.extend(weights.iter().cloned());
        let out = dev_tensor(&*cuda, &HostTensor::zeros(DType::Bf16, &[1, 2048]));
        calls.push(OpCall::new(Op::MoeExperts, inputs, vec![out], big.attrs.clone()));
    }
    let nv = |n: usize, k: usize| (n * k) as f64 * (0.5 + 1.0 / 16.0) / 1e6;
    let touched = 8.0 * (nv(1024, 2048) + nv(2048, 512));
    let need = cuda.kernels().iter().find(|k| k.name == "moe_experts_decode_bf16").unwrap().scratch_bytes(&ShapeInfo::from_call(&calls[0]));
    let scratch = TensorRef::new(cuda.alloc(need, MemKind::Device).unwrap(), DType::U8, &[need]);
    for c in &mut calls {
        c.scratch = Some(scratch.clone());
    }
    println!("{:<34} {:>10} {:>10}", "variant (T=1, 8 of 256 experts)", "us/call", "GB/s");
    for (label, kernel, down_r, gu) in [
        ("decode (original)", "moe_experts_decode_bf16", "", ""),
        ("decode v2 R=1", "moe_experts_decode_v2_bf16", "1", "2"),
        ("decode v2 R=2", "moe_experts_decode_v2_bf16", "2", "2"),
        ("decode v2 R=4", "moe_experts_decode_v2_bf16", "4", "2"),
        ("decode v2 R=8", "moe_experts_decode_v2_bf16", "8", "2"),
        ("decode v2 R=4 gate_up unroll 1", "moe_experts_decode_v2_bf16", "4", "1"),
        ("decode v2 R=4 gate_up unroll 4", "moe_experts_decode_v2_bf16", "4", "4"),
    ] {
        std::env::set_var("INTARSIA_MOE_DOWN_R", down_r);
        std::env::set_var("INTARSIA_MOE_GU_UNROLL", gu);
        let g = cuda
            .graph_capture(0, &mut || {
                for c in &calls {
                    cuda.launch(kernel, c, 0)?;
                }
                Ok(())
            })
            .unwrap();
        for _ in 0..3 {
            cuda.graph_replay(&g, 0).unwrap();
        }
        cuda.synchronize(0).unwrap();
        let reps = 20;
        let t0 = Instant::now();
        for _ in 0..reps {
            cuda.graph_replay(&g, 0).unwrap();
        }
        cuda.synchronize(0).unwrap();
        let s = t0.elapsed().as_secs_f64() / (reps * N) as f64;
        println!("{:<34} {:>10.1} {:>10.0}", label, s * 1e6, touched / 1e3 / s);
    }
    std::env::remove_var("INTARSIA_MOE_DOWN_R");
    std::env::remove_var("INTARSIA_MOE_GU_UNROLL");
}

#[test]
#[ignore]
fn bench_real_shapes() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(5);
    println!("{:<34} {:>10} {:>12} {:>10}", "case", "us", "weight MB", "GB/s");
    // NVFP4 bytes: 0.5 B/weight + 1/16 B scale
    let nv = |n: usize, k: usize| (n * k) as f64 * (0.5 + 1.0 / 16.0) / 1e6;
    for (name, m, n, k, out) in [
        ("lm_head M=1 (248320x2048)", 1usize, 248_320usize, 2048usize, DType::F32),
        ("shared gate/up M=1 (512x2048)", 1, 512, 2048, DType::Bf16),
        ("shared down M=1 (2048x512)", 1, 2048, 512, DType::Bf16),
        ("shared gate M=4", 4, 512, 2048, DType::Bf16),
        ("lm_head prefill M=256 mma", 256, 248_320, 2048, DType::F32),
    ] {
        let hc = matmul_nvfp4_case(&mut rng, m, n, k, DType::Bf16, out);
        let kern = if m <= 8 { "matmul_nvfp4_gemv" } else { "matmul_nvfp4_mma" };
        let s = bench(&*cuda, kern, &hc, if m > 8 { 5 } else { 50 });
        println!("{:<34} {:>10.1} {:>12.2} {:>10.0}", name, s * 1e6, nv(n, k), nv(n, k) / 1e3 / s);
    }
    for (name, m, n, k) in [
        ("fp8 qkv M=1 (8192x2048)", 1usize, 8192usize, 2048usize),
        ("fp8 o_proj M=1 (2048x4096)", 1, 2048, 4096),
        ("fp8 qkv M=256 mma", 256, 8192, 2048),
    ] {
        let hc = intarsia_core::quant::cases::matmul_fp8_case(&mut rng, m, n, k, true, DType::Bf16, DType::Bf16);
        let kern = if m <= 4 { "matmul_fp8_gemv" } else { "matmul_fp8_mma" };
        let s = bench(&*cuda, kern, &hc, if m > 4 { 5 } else { 50 });
        let mb = (n * k) as f64 / 1e6;
        println!("{:<34} {:>10.1} {:>12.2} {:>10.0}", name, s * 1e6, mb, mb / 1e3 / s);
    }
    // prefill-shaped W8A8 projections: bf16 dequantise-once gemm vs the e4m3 tensor-core gemm (TFLOPS = 2*M*N*K)
    for (name, m, n, k) in [
        ("fp8 qkv M=256", 256usize, 8192usize, 2048usize),
        ("fp8 qkv M=2048", 2048, 8192, 2048),
        ("fp8 o_proj M=2048", 2048, 2048, 4096),
        ("fp8 qkv M=4096", 4096, 8192, 2048),
    ] {
        let hc = intarsia_core::quant::cases::matmul_fp8_case(&mut rng, m, n, k, true, DType::Bf16, DType::Bf16);
        for kern in ["matmul_fp8_gemm", "matmul_fp8_gemm_f8"] {
            let s = bench(&*cuda, kern, &hc, 10);
            println!("{:<22} {:<20} {:>10.1} us {:>7.1} TFLOPS", name, kern, s * 1e6, 2.0 * (m * n * k) as f64 / s / 1e12);
        }
    }
    // routed experts at the real layer shape: one 256-expert weight set, the first `t` rows of one routing
    let big = experts_case(&mut rng, 4096, 256, 8, 2048, 512, DType::Bf16);
    for (name, t) in [
        ("moe_experts T=1 (8 of 256)", 1usize),
        ("moe_experts T=4", 4),
        ("moe_experts T=16", 16),
        ("moe_experts T=64", 64),
        ("moe_experts T=128", 128),
        ("moe_experts T=256", 256),
        ("moe_experts T=1024", 1024),
        ("moe_experts T=2048", 2048),
        ("moe_experts T=4096", 4096),
    ] {
        let mut hc = big.clone();
        hc.inputs[0] = HostTensor { shape: vec![t, 2048], data: big.inputs[0].data[..t * 2048 * 2].to_vec(), ..big.inputs[0].clone() };
        hc.inputs[1] = HostTensor { shape: vec![t, 8], data: big.inputs[1].data[..t * 8 * 4].to_vec(), ..big.inputs[1].clone() };
        hc.inputs[2] = HostTensor { shape: vec![t, 8], data: big.inputs[2].data[..t * 8 * 4].to_vec(), ..big.inputs[2].clone() };
        hc.outputs = vec![OutSpec::new(DType::Bf16, &[t, 2048])];
        let touched = (t * 8).min(256) as f64 * (nv(1024, 2048) + nv(2048, 512));
        // flops: 2 * pairs * (H*2I + I*H)
        let tflop = 2.0 * (t * 8) as f64 * (2048.0 * 1024.0 + 512.0 * 2048.0) / 1e12;
        for kern in ["moe_experts_bf16", "moe_experts_mma_bf16"] {
            if kern == "moe_experts_mma_bf16" && t < 16 {
                continue;
            }
            let s = bench(&*cuda, kern, &hc, if t >= 1024 { 5 } else { 20 });
            println!(
                "{:<28} {:<22} {:>10.1} us {:>8.2} MB {:>6.0} GB/s {:>6.1} TFLOPS",
                name,
                kern,
                s * 1e6,
                touched,
                touched / 1e3 / s,
                tflop / s
            );
        }
    }
    let _ = (moe_experts_scratch, Op::MoeTopk, OutSpec::new(DType::F32, &[1]));
}

/// Prefill-shaped dense projections of the 35B (docs/results/2026-10-19-cuda-prefill-3.md): the fp8 W8A8 projections
/// (GatedDeltaNet in_proj_qkv / in_proj_z / out_proj, attention q (with gate) / k / v / o) and the NVFP4 shared expert
/// (gate, up, down), one call each at M tokens (INTARSIA_BENCH_M, comma separated, default 2048), per kernel in
/// INTARSIA_BENCH_WQ (comma separated; default `matmul_fp8_gemm,matmul_fp8_gemm_f8`; nvfp4 rows use `matmul_nvfp4_gemm`).
/// Prints us and TFLOPS per call and the per-chunk total (30 GDN layers x 3 + 10 attention layers x 4 fp8 calls, 40 x 3
/// nvfp4 calls). Median of INTARSIA_BENCH_REPS (default 5) timings of 10 launches each.
#[test]
#[ignore]
fn bench_wq_prefill_shapes() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(11);
    let ms: Vec<usize> = std::env::var("INTARSIA_BENCH_M")
        .unwrap_or_else(|_| "2048".into())
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect();
    let reps: usize = std::env::var("INTARSIA_BENCH_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
    let kernels: Vec<String> = std::env::var("INTARSIA_BENCH_WQ")
        .unwrap_or_else(|_| "matmul_fp8_gemm_reg,matmul_fp8_gemm_f8,matmul_fp8_gemm_lt".into())
        .split(',')
        .map(|s| s.to_string())
        .collect();
    let time = |kern: &str, hc: &HostCall| {
        let mut v: Vec<f64> = (0..reps).map(|_| bench(&*cuda, kern, hc, 10)).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    // (name, N, K, calls per 2048-token chunk)
    let fp8 = [
        ("gdn in_proj_qkv", 8192usize, 2048usize, 30usize),
        ("gdn in_proj_z", 4096, 2048, 30),
        ("gdn out_proj", 2048, 4096, 30),
        ("attn q_proj+gate", 8192, 2048, 10),
        ("attn k/v_proj", 512, 2048, 20),
        ("attn o_proj", 2048, 4096, 10),
    ];
    let nv = [("shared gate/up", 512usize, 2048usize, 80usize), ("shared down", 2048, 512, 40)];
    for &m in &ms {
        for kern in kernels.iter().filter(|k| k.starts_with("matmul_fp8") && !k.contains("pair")) {
            let mut total = 0.0;
            for (name, n, k, calls) in fp8 {
                let hc = matmul_fp8_case(&mut rng, m, n, k, true, DType::Bf16, DType::Bf16);
                let s = time(kern, &hc);
                total += s * calls as f64;
                let tf = 2.0 * (m * n * k) as f64 / s / 1e12;
                println!("M={m} {name:<18} {kern:<22} {:>9.1} us {:>6.1} TFLOPS", s * 1e6, tf);
            }
            println!("M={m} fp8 per-chunk total {kern:<22} {:>9.1} us", total * 1e6);
        }
        for kern in kernels.iter().filter(|k| k.starts_with("matmul_nvfp4") && !k.contains("glu")) {
            let mut total = 0.0;
            for (name, n, k, calls) in nv {
                let hc = matmul_nvfp4_case(&mut rng, m, n, k, DType::Bf16, DType::Bf16);
                let s = time(kern, &hc);
                total += s * calls as f64;
                let tf = 2.0 * (m * n * k) as f64 / s / 1e12;
                println!("M={m} {name:<18} {kern:<22} {:>9.1} us {:>6.1} TFLOPS", s * 1e6, tf);
            }
            println!("M={m} nvfp4 per-chunk total {kern:<22} {:>9.1} us", total * 1e6);
        }
        // lane S1: cuBLAS bf16 GEMM (nvjet) at the same shapes, as the bf16 tensor-core reference for gemm_reg's ceiling
        if kernels.iter().any(|k| k == "matmul_bf16") {
            let mut total = 0.0;
            for (name, n, k, calls) in fp8 {
                let mut r = SplitMix64::new(5);
                let mut v = |len: usize| (0..len).map(|_| (r.next_f32() - 0.5) * 0.25).collect::<Vec<f32>>();
                let hc = HostCall {
                    op: Op::Matmul,
                    inputs: vec![
                        HostTensor::from_f32_as(DType::Bf16, &[m, k], &v(m * k)),
                        HostTensor::from_f32_as(DType::Bf16, &[n, k], &v(n * k)),
                    ],
                    outputs: vec![OutSpec::new(DType::Bf16, &[m, n])],
                    attrs: Default::default(),
                };
                let s = time("matmul_bf16", &hc);
                total += s * calls as f64;
                println!("M={m} {name:<18} {:<22} {:>9.1} us {:>6.1} TFLOPS", "matmul_bf16 (cuBLAS)", s * 1e6, 2.0 * (m * n * k) as f64 / s / 1e12);
            }
            println!("M={m} bf16 cuBLAS per-chunk total {:>9.1} us", total * 1e6);
        }
        // lane S1: the prefill pair (GDN qkv + z, attention k + v; equal input scales as in the 35B) and the shared-expert
        // GLU against the calls they replace
        if kernels.iter().any(|k| k == "matmul_fp8_pair_gemm_reg") {
            for (name, n0, n1, k, calls) in [("gdn qkv+z pair", 8192usize, 4096usize, 2048usize, 30usize), ("attn k+v pair", 512, 512, 2048, 10)] {
                let mut hc = intarsia_core::quant::cases::matmul_fp8_pair_case(&mut rng, m, n0, n1, k, true, true, DType::Bf16, DType::Bf16);
                let mut s1 = hc.inputs[4].to_f32();
                s1[1] = hc.inputs[2].to_f32()[1];
                hc.inputs[4] = HostTensor::from_f32_as(hc.inputs[4].dtype, &hc.inputs[4].shape, &s1);
                let single = |w: usize, out: usize| HostCall {
                    op: Op::MatmulFp8,
                    inputs: vec![hc.inputs[0].clone(), hc.inputs[w].clone(), hc.inputs[w + 1].clone()],
                    outputs: vec![hc.outputs[out].clone()],
                    attrs: hc.attrs.clone(),
                };
                let two = time("matmul_fp8_gemm_reg", &single(1, 0)) + time("matmul_fp8_gemm_reg", &single(3, 1));
                let pair = time("matmul_fp8_pair_gemm_reg", &hc);
                println!(
                    "M={m} {name:<18} two calls {:>9.1} us  pair {:>9.1} us  saved per chunk {:>8.1} us",
                    two * 1e6,
                    pair * 1e6,
                    (two - pair) * calls as f64 * 1e6
                );
            }
        }
        if kernels.iter().any(|k| k == "matmul_nvfp4_glu_gemm") {
            let hc = intarsia_core::quant::cases::matmul_nvfp4_glu_case(&mut rng, m, 512, 2048, DType::Bf16, DType::Bf16);
            let proj = |w: usize| HostCall {
                op: Op::MatmulNvfp4,
                inputs: vec![hc.inputs[0].clone(), hc.inputs[w].clone(), hc.inputs[w + 1].clone(), hc.inputs[w + 2].clone()],
                outputs: vec![OutSpec::new(DType::Bf16, &[m, 512])],
                attrs: Default::default(),
            };
            let glu = HostCall {
                op: Op::SiluGlu,
                inputs: vec![HostTensor::zeros(DType::Bf16, &[m, 512]), HostTensor::zeros(DType::Bf16, &[m, 512])],
                outputs: vec![OutSpec::new(DType::Bf16, &[m, 512])],
                attrs: Default::default(),
            };
            let chain = time("matmul_nvfp4_gemm", &proj(1)) + time("matmul_nvfp4_gemm", &proj(4)) + time("silu_glu_bf16", &glu);
            let fused = time("matmul_nvfp4_glu_gemm", &hc);
            println!(
                "M={m} shared gate+up+glu   chain {:>9.1} us  fused {:>9.1} us  saved per chunk {:>8.1} us",
                chain * 1e6,
                fused * 1e6,
                (chain - fused) * 40.0 * 1e6
            );
        }
    }
}

/// The grouped kernel must be about as close to the exact result as the warp-per-row kernel it replaces. The two
/// round differently inside (the old one sums fp32 products in order, this one runs bf16 mma), but the old kernel's
/// outputs differ from the exact (cpu reference) bf16 outputs in roughly 0.1 % of the elements. Accumulating the
/// whole K = 2048 dot product inside the tensor core's accumulator put 0.6 to 1 % of the elements one bf16 step off
/// (5 to 10x as many; the mechanism, low bits lost at every accumulate step, is inferred from the effect of the fix).
/// The kernel now adds each k16 mma result into the fp32 accumulator itself. This is a kernel-precision test only: on
/// the 35B the end-to-end per-prompt metrics against the vLLM golden did not follow it (docs/results/
/// 2026-10-13-phase1b.md, section 8). Real layer shape (H=2048, I=512); the bound is relative to the old kernel's own
/// error on the same inputs, so it does not depend on the random data's scale.
/// Runs only in the spark1 acceptance run (needs a GPU).
#[test]
fn moe_experts_mma_is_as_accurate_as_old_kernel() {
    let Some(cuda) = device() else { return };
    let cpu = CpuBackend::new();
    let mut rng = SplitMix64::new(5);
    for (t, e) in [(128usize, 64usize), (512, 64)] {
        let hc = experts_case(&mut rng, t, e, 8, 2048, 512, DType::Bf16);
        let want = run_host(&cpu, "ref_moe_experts", &hc).unwrap()[0].to_f32();
        let old = run_host(&*cuda, "moe_experts_bf16", &hc).unwrap()[0].to_f32();
        let new = run_host(&*cuda, "moe_experts_mma_bf16", &hc).unwrap()[0].to_f32();
        let frac = |a: &[f32]| a.iter().zip(&want).filter(|(x, y)| x != y).count() as f64 / want.len() as f64;
        let (fo, fn_) = (frac(&old), frac(&new));
        let (ro, rn) = (rel_l2(&old, &want), rel_l2(&new, &want));
        eprintln!("T={t} E={e}: elements off the reference: old {fo:.4} new {fn_:.4}; rel_l2 old {ro:.3e} new {rn:.3e}");
        assert!(fn_ <= 3.0 * fo.max(1e-3), "T={t} E={e}: {:.2} % of the mma outputs differ from the reference, the old kernel {:.2} %", fn_ * 100.0, fo * 100.0);
        assert!(rn <= 2.5 * ro, "T={t} E={e}: mma rel_l2 {rn:.3e} against the reference, old kernel {ro:.3e}");
    }
}

/// Diagnostic for docs/results/2026-10-14-35b-accuracy.md: the fraction of bf16 outputs off the cpu reference for the
/// grouped (mma) kernel and the warp-per-row kernel over small token counts (the kernel serves T >= 32; a 58-token
/// prompt's decode-mode prefill is T = 42), the largest error in bf16 steps, and whether a case that breaks the bounds of
/// `moe_experts_mma_is_as_accurate_as_old_kernel` repeats bit for bit. Prints only; run with `--include-ignored
/// --nocapture`. `INTARSIA_SWEEP_SEEDS=a..b` (default 1..17) and `INTARSIA_SWEEP_T=t1,t2,..` choose the cases.
#[test]
#[ignore]
fn moe_experts_mma_small_t_sweep() {
    let Some(cuda) = device() else { return };
    let cpu = CpuBackend::new();
    let seeds: Vec<u64> = std::env::var("INTARSIA_SWEEP_SEEDS")
        .ok()
        .and_then(|v| v.split_once("..").map(|(a, b)| (a.parse().unwrap()..b.parse().unwrap()).collect()))
        .unwrap_or_else(|| (1..17).collect());
    let ts: Vec<usize> = std::env::var("INTARSIA_SWEEP_T")
        .map(|v| v.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or_else(|_| vec![32, 33, 34, 35, 40, 41, 42, 48, 49, 57, 63, 64]);
    let ulps = |a: &[f32], b: &[f32]| {
        let key = |x: f32| {
            let u = (x.to_bits() >> 16) as i32;
            if u & 0x8000 != 0 { -(u & 0x7fff) } else { u }
        };
        a.iter().zip(b).map(|(x, y)| (key(*x) - key(*y)).unsigned_abs()).max().unwrap_or(0)
    };
    let (mut n, mut breaches) = (0, 0);
    for &seed in &seeds {
        for &t in &ts {
            let mut rng = SplitMix64::new(seed * 1000 + t as u64);
            let hc = experts_case(&mut rng, t, 64, 8, 2048, 512, DType::Bf16);
            let want = run_host(&cpu, "ref_moe_experts", &hc).unwrap()[0].to_f32();
            let old = run_host(&*cuda, "moe_experts_bf16", &hc).unwrap()[0].to_f32();
            let new = run_host(&*cuda, "moe_experts_mma_bf16", &hc).unwrap()[0].to_f32();
            let frac = |a: &[f32]| a.iter().zip(&want).filter(|(x, y)| x != y).count() as f64 / want.len() as f64;
            let (fo, fn_) = (frac(&old), frac(&new));
            let (ro, rn) = (rel_l2(&old, &want), rel_l2(&new, &want));
            let breach = fn_ > 3.0 * fo.max(1e-3) || rn > 2.5 * ro;
            n += 1;
            let mut extra = String::new();
            if breach {
                breaches += 1;
                let again = run_host(&*cuda, "moe_experts_mma_bf16", &hc).unwrap()[0].to_f32();
                let ids: Vec<i32> = hc.inputs[1].to_f32().iter().map(|&x| x as i32).collect();
                let mut per = vec![0usize; 64];
                for &e in &ids {
                    if (0..64).contains(&e) {
                        per[e as usize] += 1;
                    }
                }
                extra = format!(
                    " BREACH repeat-bitwise={} max-pairs/expert={} experts-used={}",
                    again.iter().zip(&new).all(|(x, y)| x.to_bits() == y.to_bits()),
                    per.iter().max().unwrap(),
                    per.iter().filter(|&&c| c > 0).count()
                );
            }
            eprintln!(
                "SWEEP seed={seed:3} T={t:4}: off-reference old {:.4} % mma {:.4} %; rel_l2 old {ro:.3e} mma {rn:.3e}; max ulp old {} mma {}{extra}",
                fo * 100.0,
                fn_ * 100.0,
                ulps(&old, &want),
                ulps(&new, &want)
            );
        }
    }
    eprintln!("SWEEP done: {n} cases, {breaches} break the bounds of moe_experts_mma_is_as_accurate_as_old_kernel");
}

/// The tile height (16, 32 or 64 pairs, normally chosen from the mean pairs per expert; `INTARSIA_MOE_BM` forces it)
/// only changes how rows are batched into mma tiles, never a row's own arithmetic, so the outputs are bit-identical for
/// every height. Includes experts that own fewer pairs than a tile and ones that own several tiles. Runs only in the
/// spark1 acceptance run (needs a GPU).
#[test]
fn moe_experts_mma_is_independent_of_tile_height() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(9);
    let hc = experts_case(&mut rng, 200, 16, 8, 2048, 512, DType::Bf16);
    let mut outs: Vec<(&str, Vec<u8>)> = Vec::new();
    for bm in ["16", "32", "64"] {
        std::env::set_var("INTARSIA_MOE_BM", bm);
        outs.push((bm, run_host(&*cuda, "moe_experts_mma_bf16", &hc).unwrap()[0].data.clone()));
    }
    std::env::remove_var("INTARSIA_MOE_BM");
    for (bm, o) in &outs[1..] {
        assert!(o == &outs[0].1, "BM={bm} differs from BM=16");
    }
}

/// `matmul_fp8_gemm_f8` runs the products on e4m3 tensor cores instead of dequantising the weights to bf16, so it must be
/// at least as close to the exact (cpu f64 reference) result as the bf16 `matmul_fp8_gemm` is, in both activation modes:
/// W8A8 (static input scale: the e4m3 kernel) and weight-only (the launcher's bf16 fallback). The shapes cover an M tail
/// tile (33, 100), an N that is not a multiple of the 128-row weight tile (192, 384), a tile-height-128 launch
/// (M >= 1024) and f32 outputs. Runs only in the spark1 run (needs a GPU).
#[test]
fn matmul_fp8_gemm_f8_is_as_accurate_as_the_bf16_gemm() {
    e4m3_gemm_is_as_accurate_as_the_bf16_gemm("matmul_fp8_gemm_f8", 1e-7);
}

/// The cuBLASLt e4m3 GEMM (`matmul_fp8_gemm_lt`, opt-in, docs/results/2026-10-21-cuda-round7.md), same bounds except
/// one, measured on spark1: where the bf16 GEMM is exact (rel_l2 0, the K = 256 cases) the f8 kernel's floor of 1e-7
/// is replaced by `LT_REL_FLOOR`. There cuBLAS puts one output in a thousand one bf16 step off (M = 5, N = 192:
/// rel_l2 8.1e-5, 0.1 % off), which is what accumulating the whole K inside the tensor core's f32 accumulator does (it
/// drops low bits at every step; `moe_experts_mma_is_as_accurate_as_old_kernel` documents it for the bf16 mma). The f8
/// kernel and the bf16 GEMM instead add every k-step's result with a normal fp32 add. vLLM's CUTLASS sm120 kernel
/// accumulates like cuBLAS. On every case with a non-zero bf16 error the lt kernel is at or below it (e.g. K = 4096:
/// 1.10e-4 against 1.45e-4), so only the floor changes. `INTARSIA_LT_FAST_ACCUM=1` gives the same errors on sm_121.
#[test]
fn matmul_fp8_gemm_lt_is_as_accurate_as_the_bf16_gemm() {
    e4m3_gemm_is_as_accurate_as_the_bf16_gemm("matmul_fp8_gemm_lt", LT_REL_FLOOR);
}

/// rel_l2 floor of the ratio bound for `matmul_fp8_gemm_lt` (a tenth of the 1e-3 admission budget; one bf16 step in a
/// thousand outputs is about 8e-5).
const LT_REL_FLOOR: f64 = 1e-4;

/// rel_l2 at most 2x the bf16 GEMM's (or `rel_floor`), at most 3x its share of outputs off the reference (or 0.1 %).
fn e4m3_gemm_is_as_accurate_as_the_bf16_gemm(kern: &str, rel_floor: f64) {
    let (rel_ratio, frac_ratio) = (2.0, 3.0);
    let Some(cuda) = device() else { return };
    let cpu = CpuBackend::new();
    let mut rng = SplitMix64::new(73);
    let mut fails = Vec::new();
    for act_quant in [true, false] {
        for (m, n, k, out) in [
            (5usize, 192usize, 256usize, DType::Bf16),
            (33, 192, 256, DType::Bf16),
            (100, 512, 2048, DType::Bf16),
            (256, 384, 1024, DType::F32),
            (1100, 256, 512, DType::Bf16),
            (256, 2048, 4096, DType::Bf16),
        ] {
            let hc = matmul_fp8_case(&mut rng, m, n, k, act_quant, DType::Bf16, out);
            let want = run_host(&cpu, "ref_matmul_fp8", &hc).unwrap()[0].to_f32();
            let old = run_host(&*cuda, "matmul_fp8_gemm", &hc).unwrap()[0].to_f32();
            let got = run_host(&*cuda, kern, &hc).unwrap();
            let new = got[0].to_f32();
            let frac = |a: &[f32]| a.iter().zip(&want).filter(|(x, y)| x != y).count() as f64 / want.len() as f64;
            let (ro, rn) = (rel_l2(&old, &want), rel_l2(&new, &want));
            let label = format!("{kern} act_quant={act_quant} M={m} N={n} K={k} {out:?}");
            eprintln!("{label}: rel_l2 bf16 gemm {ro:.3e} e4m3 gemm {rn:.3e}; elements off: {:.4} / {:.4}", frac(&old), frac(&new));
            if rn >= 1e-3 {
                fails.push(format!("{label}: rel_l2 {rn:.3e}"));
            }
            if rn > rel_ratio * ro.max(rel_floor) {
                fails.push(format!("{label}: e4m3 rel_l2 {rn:.3e}, bf16 gemm {ro:.3e} (bound x{rel_ratio}, floor {rel_floor:.0e})"));
            }
            if frac(&new) > frac_ratio * frac(&old).max(1e-3) {
                fails.push(format!("{label}: {:.4} of the outputs off the reference, bf16 gemm {:.4} (bound x{frac_ratio})", frac(&new), frac(&old)));
            }
            let again = run_host(&*cuda, kern, &hc).unwrap();
            assert_eq!(got[0].data, again[0].data, "{label}: not repeatable");
        }
    }
    assert!(fails.is_empty(), "{kern}:\n{}", fails.join("\n"));
}

/// A row's result must not depend on how many rows share the call or where the row sits in its tile: `matmul_fp8_gemm_f8`
/// serves every M >= 5, so a prompt chunked differently (tail chunks, prefix reuse) must give each token the same bits.
/// Compares row slices of one 1100-row call (128-row tiles) with calls of 5, 17, 64 and 100 rows at other offsets (64-row
/// tiles, tail rows, shifted tile origins), bitwise.
#[test]
fn matmul_fp8_gemm_f8_rows_do_not_depend_on_the_call() {
    e4m3_gemm_rows_do_not_depend_on_the_call("matmul_fp8_gemm_f8", 1100, 384, 1024, &[(0, 5), (3, 17), (130, 64), (1000, 100)]);
}

/// The same for the cuBLASLt e4m3 GEMM (one algorithm pinned per (N, K) for every M), also on a 35B projection shape
/// at the 2048-token chunk against the tail-chunk sizes a prompt produces.
#[test]
fn matmul_fp8_gemm_lt_rows_do_not_depend_on_the_call() {
    let rows: &[(usize, usize)] = &[(0, 5), (3, 17), (130, 64), (1000, 100), (7, 255), (512, 517)];
    e4m3_gemm_rows_do_not_depend_on_the_call("matmul_fp8_gemm_lt", 1100, 384, 1024, rows);
    let rows: &[(usize, usize)] = &[(0, 5), (100, 93), (1, 1024), (1500, 548), (2000, 48)];
    e4m3_gemm_rows_do_not_depend_on_the_call("matmul_fp8_gemm_lt", 2048, 2048, 4096, rows);
}

fn e4m3_gemm_rows_do_not_depend_on_the_call(kern: &str, m: usize, n: usize, k: usize, slices: &[(usize, usize)]) {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(74);
    let whole = matmul_fp8_case(&mut rng, m, n, k, true, DType::Bf16, DType::Bf16);
    let all = run_host(&*cuda, kern, &whole).unwrap();
    let (xk, on) = (k * 2, n * 2);
    for &(r0, rows) in slices {
        assert!(rows >= 5 && r0 + rows <= m, "the registry serves M < 5 with the gemv");
        let mut hc = whole.clone();
        hc.inputs[0] = HostTensor { shape: vec![rows, k], data: whole.inputs[0].data[r0 * xk..(r0 + rows) * xk].to_vec(), ..whole.inputs[0].clone() };
        hc.outputs = vec![OutSpec::new(DType::Bf16, &[rows, n])];
        let got = run_host(&*cuda, kern, &hc).unwrap();
        assert!(got[0].data == all[0].data[r0 * on..(r0 + rows) * on], "{kern}: rows {r0}..{} differ when run as a {rows}-row call", r0 + rows);
    }
}

/// Lane S1 probe: does cuBLAS's bf16 GEMM (`matmul_bf16`, cublasGemmEx) give the bytes of `matmul_fp8_gemm_reg` on
/// e4m3-valued operands (weight-only mode, weight scale 1, so both epilogues are one bf16 rounding of the fp32 sum)?
/// Prints the number of differing outputs per 35B prefill shape (INTARSIA_BENCH_M, default 2048). Equal bytes would mean
/// cuBLAS runs the same m16n8k16 accumulation chain there (no split-K, k order); any difference rules it out as a bitwise
/// replacement at that shape.
#[test]
#[ignore]
fn probe_cublas_bf16_vs_gemm_reg_bits() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(77);
    let ms: Vec<usize> = std::env::var("INTARSIA_BENCH_M")
        .unwrap_or_else(|_| "2048".into())
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect();
    for &m in &ms {
        for (n, k) in [(8192usize, 2048usize), (4096, 2048), (2048, 4096), (512, 2048)] {
            let mut hc = matmul_fp8_case(&mut rng, m, n, k, false, DType::Bf16, DType::Bf16);
            // weight scale 1, no input scale: gemm_reg's output is bf16(sum of x * e4m3 weight)
            hc.inputs[2] = HostTensor::from_f32_as(hc.inputs[2].dtype, &hc.inputs[2].shape, &[1.0, 0.0]);
            // x on the e4m3 grid (as W8A8 activations are), so both kernels see the same small-mantissa operands
            let xs: Vec<f32> = hc.inputs[0]
                .to_f32()
                .iter()
                .map(|&v| intarsia_core::dtype::fp8_e4m3_to_f32(intarsia_core::dtype::f32_to_fp8_e4m3(v * 8.0)))
                .collect();
            hc.inputs[0] = HostTensor::from_f32_as(DType::Bf16, &[m, k], &xs);
            let wf: Vec<f32> = hc.inputs[1].data.iter().map(|&b| intarsia_core::dtype::fp8_e4m3_to_f32(b)).collect();
            let bf = HostCall {
                op: Op::Matmul,
                inputs: vec![hc.inputs[0].clone(), HostTensor::from_f32_as(DType::Bf16, &[n, k], &wf)],
                outputs: vec![OutSpec::new(DType::Bf16, &[m, n])],
                attrs: Default::default(),
            };
            let a = run_host(&*cuda, "matmul_fp8_gemm_reg", &hc).unwrap().remove(0);
            let b = run_host(&*cuda, "matmul_bf16", &bf).unwrap().remove(0);
            let (fa, fb) = (a.to_f32(), b.to_f32());
            let diff = fa.iter().zip(&fb).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            let maxrel = fa.iter().zip(&fb).map(|(x, y)| ((x - y).abs() / x.abs().max(1e-3))).fold(0f32, f32::max);
            println!("M={m} N={n} K={k}: {diff} of {} bf16 outputs differ (max rel {maxrel:.2e})", fa.len());
        }
    }
}
