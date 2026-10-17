//! `moe_experts_nvfp4_mma_bf16` (kernels/cuda/moe_nvfp4_mma_sm120a.cu): grouped MoE experts on the block-scaled FP4 tensor
//! core, W4A4. Three kinds of checks:
//!   * against a cpu reference of the QUANTISED arithmetic (activations quantised exactly as the kernel does, products and
//!     sums in f64): equal up to fp32 summation order and the bf16 rounding points that order can flip;
//!   * against the bf16-activation reference `ref_moe_experts`, within the stated activation-quantisation tolerance;
//!   * structural: routing edge cases, independence of the tile height, repeatability.
//! Plus an ignored microbenchmark against `moe_experts_mma_bf16` on the real layer shape
//! (`cargo test --release -p intarsia-backend-cuda --test moe_nvfp4 -- --ignored --nocapture`).
//! Needs a GPU built for sm_120a / sm_121a: without one the tests print SKIPPED (and fail under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_cuda)]
mod common;
mod moe_nvfp4_ref;

use moe_nvfp4_ref::{with_rows_scaled, Quantised};

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::backend::{upload_new, Backend, MemKind, TensorRef};
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::quant::cases::experts_case;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, OpCall, ShapeInfo};
use std::sync::Arc;
use std::time::Instant;

const KERNEL: &str = "moe_experts_nvfp4_mma_bf16";

fn device() -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_cuda::create(ord) {
        Ok(b) => {
            if !b.kernels().iter().any(|k| k.name == KERNEL) {
                common::skip("moe_nvfp4", "kernels not built for sm_120a / sm_121a");
                return None;
            }
            Some(b)
        }
        Err(e) => {
            common::skip("moe_nvfp4", &format!("no usable CUDA device ({e})"));
            None
        }
    }
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

/// The generator gives every expert the same global scales; give gate, up and down (and every expert) their own so a
/// kernel that mixes them up is caught.
fn with_distinct_gscales(hc: &HostCall) -> HostCall {
    let mut hc = hc.clone();
    let mut gs = hc.inputs[7].to_f32();
    for (j, g) in gs.iter_mut().enumerate() {
        *g *= [1.0f32, 0.6, 1.7][j % 3] * (1.0 + 0.05 * (j / 3 % 7) as f32);
    }
    hc.inputs[7] = HostTensor::from_f32(&hc.inputs[7].shape.clone(), &gs);
    hc
}

/// Quantised-arithmetic check: the kernel against `Quantised::run`. `tol` is relative L2.
fn check_quantised(cuda: &dyn Backend, hc: &HostCall, label: &str, tol: f64) -> f64 {
    let want = Quantised::new(hc).run();
    let got = run_host(cuda, KERNEL, hc).unwrap();
    let g = got[0].to_f32();
    let r = rel_l2(&g, &want);
    let differ = g.iter().zip(&want).filter(|(a, b)| a != b).count();
    println!("{label}: vs quantised reference rel_l2 {r:.3e}, {differ}/{} elements differ", g.len());
    assert!(r < tol, "{label}: rel_l2 {r} vs the quantised reference");
    let again = run_host(cuda, KERNEL, hc).unwrap();
    assert_eq!(got[0].data, again[0].data, "{label}: not repeatable");
    r
}

#[test]
fn moe_nvfp4_matches_the_quantised_reference() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(71);
    // (T, E, K, H, I): tile heights 16 / 32 / 64 are picked from T*K/E (<= 10, <= 21, else); an I of one and of several
    // 64-column blocks; K above and below E/2
    for (t, e, k, h, i) in [
        (40usize, 64usize, 8usize, 256usize, 128usize), // ~5 pairs per expert -> BM 16
        (64, 32, 8, 256, 128),                          // 16 -> BM 32
        (128, 16, 8, 256, 256),                         // 64 -> BM 64
        (33, 8, 4, 128, 128),
        (96, 16, 2, 384, 128),
        (50, 4, 4, 2048, 512), // real hidden / intermediate sizes, every expert hot
    ] {
        let hc = with_distinct_gscales(&experts_case(&mut rng, t, e, k, h, i, DType::Bf16));
        check_quantised(&*cuda, &hc, &format!("random T={t} E={e} K={k} H={h} I={i}"), 3e-3);
    }
}

/// Rows with very different magnitude (the per-row scale), an all-zero row, duplicate / invalid / skewed routing.
#[test]
fn moe_nvfp4_handles_range_zero_rows_and_odd_routing() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(72);
    let (t, e, k, h, i) = (48usize, 16usize, 4usize, 256usize, 128usize);
    let base = with_distinct_gscales(&experts_case(&mut rng, t, e, k, h, i, DType::Bf16));
    let scaled = with_rows_scaled(&base, &[(0, 1e3), (1, 1e-3), (2, 0.0), (3, 5e4), (4, 3e-6), (5, 1e-37), (6, 1e-33)]);
    check_quantised(&*cuda, &scaled, "row magnitudes 1e3 / 1e-3 / 0 / 5e4 / 3e-6 / 1e-37 / 1e-33", 3e-3);
    let mut odd = scaled.clone();
    let mut ids = odd.inputs[1].to_i32().unwrap();
    for j in 0..k {
        ids[j] = 3; // one expert receives every pair of token 0
    }
    ids[k] = -1;
    ids[k + 1] = e as i32 + 5;
    for tok in 8..t {
        ids[tok * k] = 0; // a hot expert owning several tiles
    }
    odd.inputs[1] = HostTensor::from_i32(&[t, k], &ids);
    check_quantised(&*cuda, &odd, "duplicate, invalid and hot-expert routing", 3e-3);
    let one = experts_case(&mut rng, 1, 64, 8, 2048, 512, DType::Bf16);
    // T = 1 is below the registered kernel's 32-token threshold but the launcher itself handles it
    check_quantised(&*cuda, &one, "T=1 real shape", 3e-3);
}

/// The only difference to the bf16-activation reference is the activation quantisation; it must stay under the tolerance
/// the kernel is registered with (`NVFP4_ACT_TOL`), and the measured value is printed for the results page.
#[test]
fn moe_nvfp4_is_within_activation_quantisation_error_of_the_bf16_reference() {
    let Some(cuda) = device() else { return };
    let cpu = CpuBackend::new();
    let mut rng = SplitMix64::new(73);
    let tol = intarsia_backend_cuda::quantmoe::NVFP4_ACT_TOL;
    for (t, e, k, h, i) in [(64usize, 16usize, 8usize, 256usize, 128usize), (100, 8, 4, 2048, 512)] {
        let hc = experts_case(&mut rng, t, e, k, h, i, DType::Bf16);
        let want = run_host(&cpu, "ref_moe_experts", &hc).unwrap()[0].to_f32();
        let got = run_host(&*cuda, KERNEL, &hc).unwrap()[0].to_f32();
        let old = run_host(&*cuda, "moe_experts_mma_bf16", &hc).unwrap()[0].to_f32();
        let r = rel_l2(&got, &want);
        let maxabs = got.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        println!(
            "T={t} E={e} K={k} H={h} I={i}: fp4 vs bf16 reference rel_l2 {r:.4} max_abs {maxabs:.4} (bf16 kernel: {:.1e})",
            rel_l2(&old, &want)
        );
        let scale = want.iter().fold(1f32, |m, v| m.max(v.abs()));
        assert!(
            r < tol.rel_l2 as f64 && (maxabs / scale) < tol.max_abs,
            "outside NVFP4_ACT_TOL: rel_l2 {r}, max_abs {} (relative to max(1, max|ref|))",
            maxabs / scale
        );
    }
}

/// The real prefill shape (256 experts, top-8, H 2048, I 512, 1024 tokens: about 32 pairs per expert, several tiles for
/// the hot ones): too big for the cpu reference, so the check is against the bf16 GPU kernel (the activation quantisation
/// error again, ~0.16), plus repeatability and bit-equality across tile heights.
#[test]
fn moe_nvfp4_at_the_real_prefill_shape() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(75);
    let hc = with_distinct_gscales(&experts_case(&mut rng, 1024, 256, 8, 2048, 512, DType::Bf16));
    let old = run_host(&*cuda, "moe_experts_mma_bf16", &hc).unwrap()[0].to_f32();
    let got = run_host(&*cuda, KERNEL, &hc).unwrap();
    let r = rel_l2(&got[0].to_f32(), &old);
    println!("T=1024 E=256 K=8 H=2048 I=512: fp4 vs bf16 kernel rel_l2 {r:.4}");
    assert!(r > 0.05 && r < 0.25, "rel_l2 {r}");
    assert_eq!(got[0].data, run_host(&*cuda, KERNEL, &hc).unwrap()[0].data, "not repeatable");
}

/// The tile height changes only how rows are batched into mma tiles, never a row's own arithmetic: bit-identical outputs.
#[test]
fn moe_nvfp4_is_independent_of_tile_height() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(74);
    let hc = experts_case(&mut rng, 200, 16, 8, 2048, 512, DType::Bf16);
    let mut outs: Vec<(&str, Vec<u8>)> = Vec::new();
    for bm in ["16", "32", "64"] {
        std::env::set_var("INTARSIA_MOE_BM", bm);
        outs.push((bm, run_host(&*cuda, KERNEL, &hc).unwrap()[0].data.clone()));
    }
    std::env::remove_var("INTARSIA_MOE_BM");
    for (bm, o) in &outs[1..] {
        assert!(o == &outs[0].1, "BM={bm} differs from BM=16");
    }
}

// ---- microbenchmark ----------------------------------------------------------------------------------------------------

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

/// Real layer shape (256 experts, top-8, H 2048, I 512): bf16 grouped kernel vs the FP4 one at prefill chunk sizes.
/// Run with vLLM up the numbers are shared-GPU timings (the GPU is idle but memory is shared): relative, not absolute.
#[test]
#[ignore]
fn bench_moe_nvfp4_real_shapes() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(5);
    let big = experts_case(&mut rng, 4096, 256, 8, 2048, 512, DType::Bf16);
    println!("{:<12} {:<28} {:>10} {:>12}", "T", "kernel", "us", "TFLOPS");
    // INTARSIA_BENCH_T=2048,4096 picks the batch sizes (for profiling a single one)
    let ts: Vec<usize> = std::env::var("INTARSIA_BENCH_T")
        .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|_| vec![64, 256, 1024, 2048, 4096]);
    let kernels: Vec<&str> = match std::env::var("INTARSIA_BENCH_KERNEL") {
        Ok(v) if v == "fp4" => vec![KERNEL],
        _ => vec!["moe_experts_mma_bf16", KERNEL],
    };
    for t in ts {
        let mut hc = big.clone();
        hc.inputs[0] = HostTensor { shape: vec![t, 2048], data: big.inputs[0].data[..t * 2048 * 2].to_vec(), ..big.inputs[0].clone() };
        hc.inputs[1] = HostTensor { shape: vec![t, 8], data: big.inputs[1].data[..t * 8 * 4].to_vec(), ..big.inputs[1].clone() };
        hc.inputs[2] = HostTensor { shape: vec![t, 8], data: big.inputs[2].data[..t * 8 * 4].to_vec(), ..big.inputs[2].clone() };
        hc.outputs = vec![OutSpec::new(DType::Bf16, &[t, 2048])];
        let tflop = 2.0 * (t * 8) as f64 * (2048.0 * 1024.0 + 512.0 * 2048.0) / 1e12;
        let reps = if t >= 1024 { 10 } else { 30 };
        for &kern in &kernels {
            let s = bench(&*cuda, kern, &hc, reps);
            println!("{:<12} {:<28} {:>10.1} {:>12.1}", t, kern, s * 1e6, tflop / s / 1.0);
        }
    }
}
