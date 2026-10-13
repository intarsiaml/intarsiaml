//! cuda quantised-weight / MoE kernels: edge cases beyond the admission sweep (skewed and invalid routing,
//! multi-chunk experts) and an ignored microbenchmark on the real Qwen3.6-35B-A3B shapes
//! (`cargo test --release -p intarsia-backend-cuda --test quantmoe -- --ignored --nocapture`).
//! Needs a GPU: without one the tests print SKIPPED (and fail under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_cuda)]

mod common;

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::backend::{upload_new, Backend, MemKind, TensorRef};
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::quant::cases::{experts_case, matmul_nvfp4_case};
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
    assert!(r < 1e-2, "{kernel} {label}: rel_l2 {r}");
    let again = run_host(cuda, kernel, hc).unwrap();
    assert_eq!(got[0].data, again[0].data, "{kernel} {label}: not repeatable");
}

/// The two-launch decode kernel does the same arithmetic as the warp-per-row kernel, so for T <= 4 the outputs are
/// bit-identical (including duplicate and invalid ids), and both match the reference.
#[test]
fn moe_experts_decode_matches_old_kernel() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(33);
    for (t, e, k, h, i) in [(1usize, 64usize, 8usize, 2048usize, 512usize), (2, 16, 8, 2048, 512), (4, 32, 8, 2048, 512), (3, 16, 4, 256, 64), (1, 8, 2, 64, 32)] {
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
            check_experts_with(&*cuda, "moe_experts_decode_bf16", hc, &label);
            let old = run_host(&*cuda, "moe_experts_bf16", hc).unwrap();
            let new = run_host(&*cuda, "moe_experts_decode_bf16", hc).unwrap();
            assert_eq!(old[0].data, new[0].data, "{label}: decode kernel differs from moe_experts_bf16");
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
