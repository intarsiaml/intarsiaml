//! `moe_experts_w4a16_bf16` (kernels/cuda/moe_w4a16.cu): the grouped MoE experts with the fp4 weights dequantised in
//! registers. It is a different schedule of the same arithmetic, so the contract is stronger than a tolerance: the output
//! bytes equal those of `moe_experts_mma_bf16` for every input, every tile height, and every routing. These tests pin that,
//! plus the accuracy bound of `moe_experts_mma_is_as_accurate_as_old_kernel` (tests/quantmoe.rs) for the new kernel.
//! Needs a GPU: without one the tests print SKIPPED (and fail under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_cuda)]
mod common;

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::backend::Backend;
use intarsia_core::exec::{run_host, HostCall};
use intarsia_core::quant::cases::experts_case;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor};
use std::sync::Arc;

const V1: &str = "moe_experts_mma_bf16";
const V2: &str = "moe_experts_w4a16_bf16";
const PF_ENV: &str = "INTARSIA_CUDA_MOE_W4A16_PF";

fn device() -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_cuda::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            common::skip("moe_w4a16", &format!("no usable CUDA device ({e})"));
            None
        }
    }
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

/// Every expert (and gate / up / down) gets its own global scale, so a kernel that mixes them up is caught.
fn with_distinct_gscales(hc: &HostCall) -> HostCall {
    let mut hc = hc.clone();
    let mut gs = hc.inputs[7].to_f32();
    for (j, g) in gs.iter_mut().enumerate() {
        *g *= [1.0f32, 0.6, 1.7][j % 3] * (1.0 + 0.05 * (j / 3 % 7) as f32);
    }
    hc.inputs[7] = HostTensor::from_f32(&hc.inputs[7].shape.clone(), &gs);
    hc
}

fn routed(hc: &HostCall, ids: &[i32], t: usize, k: usize) -> HostCall {
    let mut hc = hc.clone();
    hc.inputs[1] = HostTensor::from_i32(&[t, k], ids);
    hc
}

/// The same call through v1 and v2 under both tile heights v2 offers (`INTARSIA_MOE_BM`; "" = the automatic choice), with the
/// L2-prefetch schedule off and on (`INTARSIA_CUDA_MOE_W4A16_PF`; also in its automatic tile height). v1 is independent of its
/// own tile height (tests/quantmoe.rs).
fn assert_bitwise(cuda: &dyn Backend, hc: &HostCall, label: &str) {
    std::env::remove_var("INTARSIA_MOE_BM");
    std::env::remove_var(PF_ENV);
    let want = run_host(cuda, V1, hc).unwrap()[0].data.clone();
    for pf in ["0", "1"] {
        std::env::set_var(PF_ENV, pf);
        for bm in ["", "32", "64"] {
            if bm.is_empty() {
                std::env::remove_var("INTARSIA_MOE_BM");
            } else {
                std::env::set_var("INTARSIA_MOE_BM", bm);
            }
            let got = run_host(cuda, V2, hc).unwrap()[0].data.clone();
            let differ = got.chunks(2).zip(want.chunks(2)).filter(|(a, b)| a != b).count();
            assert!(
                differ == 0,
                "{label}: PF={pf} BM={}: {differ} of {} bf16 outputs differ from {V1}",
                if bm.is_empty() { "auto" } else { bm },
                got.len() / 2
            );
            let again = run_host(cuda, V2, hc).unwrap()[0].data.clone();
            assert!(again == got, "{label}: PF={pf} BM={bm}: not repeatable");
        }
    }
    std::env::remove_var("INTARSIA_MOE_BM");
    std::env::remove_var(PF_ENV);
}

#[test]
fn moe_w4a16_is_bitwise_the_grouped_kernel() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(81);
    // (T, E, K, H, I): small experts (fewer pairs than a tile), several tiles per expert, one and several 64-column
    // blocks, K above and below E/2, the real hidden / intermediate sizes
    for (t, e, k, h, i) in [
        (40usize, 64usize, 8usize, 256usize, 128usize),
        (64, 32, 8, 256, 128),
        (128, 16, 8, 256, 256),
        (33, 8, 4, 128, 128),
        (96, 16, 2, 384, 128),
        (50, 4, 4, 2048, 512),
        (300, 64, 8, 2048, 512),
    ] {
        let hc = with_distinct_gscales(&experts_case(&mut rng, t, e, k, h, i, DType::Bf16));
        assert_bitwise(&*cuda, &hc, &format!("random T={t} E={e} K={k} H={h} I={i}"));
    }
}

/// Hot expert owning many tiles, experts owning none, duplicate and out-of-range ids.
#[test]
fn moe_w4a16_is_bitwise_on_odd_routing() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(82);
    let (t, e, k, h, i) = (200usize, 16usize, 4usize, 256usize, 128usize);
    let base = with_distinct_gscales(&experts_case(&mut rng, t, e, k, h, i, DType::Bf16));
    let hot: Vec<i32> = (0..t).flat_map(|_| 0..k as i32).collect();
    assert_bitwise(&*cuda, &routed(&base, &hot, t, k), "every token picks experts 0..k");
    let mut odd = base.inputs[1].to_i32().unwrap();
    for j in 0..k {
        odd[j] = 3;
    }
    odd[k] = -1;
    odd[k + 1] = e as i32 + 5;
    for tok in 8..t {
        odd[tok * k] = 0;
    }
    assert_bitwise(&*cuda, &routed(&base, &odd, t, k), "duplicate, invalid and hot-expert routing");
    // a real-layer expert whose pair count is not a multiple of any tile height (1 .. 130 rows)
    let (t, e, k) = (130usize, 8usize, 4usize);
    let real = with_distinct_gscales(&experts_case(&mut rng, t, e, k, 2048, 512, DType::Bf16));
    let skew: Vec<i32> = (0..t * k).map(|p| if p % 5 == 0 { 7 } else { (p % 4) as i32 }).collect();
    assert_bitwise(&*cuda, &routed(&real, &skew, t, k), "skewed real-layer routing");
    // T = 1 is below the registered threshold but the launcher itself handles it
    let one = experts_case(&mut rng, 1, 64, 8, 2048, 512, DType::Bf16);
    assert_bitwise(&*cuda, &one, "T=1 real shape");
}

/// The real layer shape at the sizes a 2048-token prefill chunk produces.
#[test]
fn moe_w4a16_is_bitwise_at_the_real_prefill_shape() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(83);
    for t in [1024usize, 2048] {
        let hc = with_distinct_gscales(&experts_case(&mut rng, t, 256, 8, 2048, 512, DType::Bf16));
        assert_bitwise(&*cuda, &hc, &format!("real prefill shape T={t}"));
    }
}

/// `moe_experts_mma_is_as_accurate_as_old_kernel` (tests/quantmoe.rs), run with v2: the bound is relative to the old
/// kernel's own error on the same inputs. Implied by the bitwise tests; kept as the bar the task states.
#[test]
fn moe_experts_w4a16_is_as_accurate_as_old_kernel() {
    let Some(cuda) = device() else { return };
    let cpu = CpuBackend::new();
    let mut rng = SplitMix64::new(5);
    for (t, e) in [(128usize, 64usize), (512, 64)] {
        let hc = experts_case(&mut rng, t, e, 8, 2048, 512, DType::Bf16);
        let want = run_host(&cpu, "ref_moe_experts", &hc).unwrap()[0].to_f32();
        let old = run_host(&*cuda, "moe_experts_bf16", &hc).unwrap()[0].to_f32();
        let new = run_host(&*cuda, V2, &hc).unwrap()[0].to_f32();
        std::env::set_var(PF_ENV, "1");
        let new_pf = run_host(&*cuda, V2, &hc).unwrap()[0].to_f32();
        std::env::remove_var(PF_ENV);
        assert!(new_pf == new, "T={t} E={e}: the prefetch schedule changed the output");
        let frac = |a: &[f32]| a.iter().zip(&want).filter(|(x, y)| x != y).count() as f64 / want.len() as f64;
        let (fo, fn_) = (frac(&old), frac(&new));
        let (ro, rn) = (rel_l2(&old, &want), rel_l2(&new, &want));
        eprintln!("T={t} E={e}: elements off the reference: old {fo:.4} w4a16 {fn_:.4}; rel_l2 old {ro:.3e} w4a16 {rn:.3e}");
        assert!(fn_ <= 3.0 * fo.max(1e-3), "T={t} E={e}: {:.2} % of the outputs differ from the reference, the old kernel {:.2} %", fn_ * 100.0, fo * 100.0);
        assert!(rn <= 2.5 * ro, "T={t} E={e}: w4a16 rel_l2 {rn:.3e} against the reference, old kernel {ro:.3e}");
    }
}
