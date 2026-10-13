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
    let Some(cuda) = device() else { return };
    let cpu = CpuBackend::new();
    let mut rng = SplitMix64::new(73);
    for act_quant in [true, false] {
        for (m, n, k, out) in [
            (5usize, 192usize, 256usize, DType::Bf16),
            (33, 192, 256, DType::Bf16),
            (100, 512, 2048, DType::Bf16),
            (256, 384, 1024, DType::F32),
            (1100, 256, 512, DType::Bf16),
        ] {
            let hc = matmul_fp8_case(&mut rng, m, n, k, act_quant, DType::Bf16, out);
            let want = run_host(&cpu, "ref_matmul_fp8", &hc).unwrap()[0].to_f32();
            let old = run_host(&*cuda, "matmul_fp8_gemm", &hc).unwrap()[0].to_f32();
            let got = run_host(&*cuda, "matmul_fp8_gemm_f8", &hc).unwrap();
            let new = got[0].to_f32();
            let frac = |a: &[f32]| a.iter().zip(&want).filter(|(x, y)| x != y).count() as f64 / want.len() as f64;
            let (ro, rn) = (rel_l2(&old, &want), rel_l2(&new, &want));
            let label = format!("act_quant={act_quant} M={m} N={n} K={k} {out:?}");
            eprintln!("{label}: rel_l2 bf16 gemm {ro:.3e} f8 gemm {rn:.3e}; elements off: {:.4} / {:.4}", frac(&old), frac(&new));
            assert!(rn < 1e-3, "{label}: rel_l2 {rn:.3e}");
            assert!(rn <= 2.0 * ro.max(1e-7), "{label}: f8 rel_l2 {rn:.3e}, bf16 gemm {ro:.3e}");
            assert!(
                frac(&new) <= 3.0 * frac(&old).max(1e-3),
                "{label}: {:.4} of the outputs off the reference, bf16 gemm {:.4}",
                frac(&new),
                frac(&old)
            );
            let again = run_host(&*cuda, "matmul_fp8_gemm_f8", &hc).unwrap();
            assert_eq!(got[0].data, again[0].data, "{label}: not repeatable");
        }
    }
}

/// A row's result must not depend on how many rows share the call or where the row sits in its tile: `matmul_fp8_gemm_f8`
/// serves every M >= 5, so a prompt chunked differently (tail chunks, prefix reuse) must give each token the same bits.
/// Compares row slices of one 1100-row call (128-row tiles) with calls of 5, 17, 64 and 100 rows at other offsets (64-row
/// tiles, tail rows, shifted tile origins), bitwise.
#[test]
fn matmul_fp8_gemm_f8_rows_do_not_depend_on_the_call() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(74);
    let (m, n, k) = (1100usize, 384usize, 1024usize);
    let whole = matmul_fp8_case(&mut rng, m, n, k, true, DType::Bf16, DType::Bf16);
    let all = run_host(&*cuda, "matmul_fp8_gemm_f8", &whole).unwrap();
    let (xk, on) = (k * 2, n * 2);
    for (r0, rows) in [(0usize, 5usize), (3, 17), (130, 64), (1000, 100), (1099, 1)] {
        if rows < 5 {
            continue; // the registry serves M < 5 with the gemv
        }
        let mut hc = whole.clone();
        hc.inputs[0] = HostTensor { shape: vec![rows, k], data: whole.inputs[0].data[r0 * xk..(r0 + rows) * xk].to_vec(), ..whole.inputs[0].clone() };
        hc.outputs = vec![OutSpec::new(DType::Bf16, &[rows, n])];
        let got = run_host(&*cuda, "matmul_fp8_gemm_f8", &hc).unwrap();
        assert!(got[0].data == all[0].data[r0 * on..(r0 + rows) * on], "rows {r0}..{} differ when run as a {rows}-row call", r0 + rows);
    }
}
