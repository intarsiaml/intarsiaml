//! Lane l5 (docs/results/2026-10-25-l5-moe-fp8-wmma.md): the opt-in fp8-WMMA routed experts (INTARSIA_HIP_MOE_FP8_WMMA=1,
//! kernels/hip/moe_wmma_fp8.h) through the registered launcher `moe_experts_bf16_v2`, with the threshold lowered so that small cases take it.
//!  * accuracy: against the cpu reference (bf16 activations) the relative L2 error is the cost of the e4m3 activations (about 0.05), never
//!    near the 1e-3 of the bf16 path; outputs finite;
//!  * grouping independence: a token's output is the same bits whatever else is in the call (rows reversed, other tokens around it,
//!    routing that leaves tiles partly filled, an expert with more pair tiles than the grid);
//!  * repeatability: the same call twice gives the same bits.
//! Its own test binary with a single test (the knobs are read once per process; they are set before the first call). Needs a GPU, skips
//! (passes) without one.
#![cfg(ik_hip)]

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::backend::Backend;
use intarsia_core::exec::{run_host, HostCall};
use intarsia_core::quant::cases::experts_case;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor};
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

/// rows of a [rows, cols] bf16 / i32 / f32 tensor in the order `perm` (new row r = old row perm[r])
fn permute_rows(t: &HostTensor, perm: &[usize]) -> HostTensor {
    let rows = t.shape[0];
    let rb = t.data.len() / rows;
    let mut data = Vec::with_capacity(t.data.len());
    for &r in perm {
        data.extend_from_slice(&t.data[r * rb..(r + 1) * rb]);
    }
    HostTensor { data, ..t.clone() }
}

#[test]
fn fp8_experts_are_accurate_grouping_independent_and_repeatable() {
    std::env::set_var("INTARSIA_HIP_MOE_FP8_WMMA", "1");
    std::env::set_var("INTARSIA_HIP_MOE_FP8_MIN", "16");
    let Some(hip) = device() else { return };
    let cpu = CpuBackend::new();
    let mut rng = SplitMix64::new(2025);
    let mut cases: Vec<(String, HostCall)> = Vec::new();
    // h, i multiples of 256 (the quantiser's K); partly filled tiles; the 35B's shape; more tiles than the grid
    for &(t, e, k, h, i) in &[(37usize, 16usize, 4usize, 256usize, 256usize), (300, 16, 4, 512, 256), (1100, 64, 8, 2048, 512)] {
        cases.push((format!("t={t} e={e} k={k} h={h} i={i}"), experts_case(&mut rng, t, e, k, h, i, DType::Bf16)));
    }
    let mut hot = experts_case(&mut rng, 1000, 16, 4, 256, 256, DType::Bf16);
    let ids: Vec<i32> = (0..1000).flat_map(|_| 0..4i32).collect();
    hot.inputs[1] = HostTensor::from_i32(&[1000, 4], &ids);
    cases.push(("hot experts".into(), hot));
    for (what, hc) in &cases {
        let got = run_host(&*hip, "moe_experts_bf16_v2", hc).unwrap().remove(0);
        let again = run_host(&*hip, "moe_experts_bf16_v2", hc).unwrap().remove(0);
        assert!(got.data == again.data, "{what}: not repeatable");
        let want = run_host(&cpu, "ref_moe_experts", hc).unwrap().remove(0);
        let g = got.to_f32();
        assert!(g.iter().all(|v| v.is_finite()), "{what}: non-finite output");
        let r = rel_l2(&g, &want.to_f32());
        assert!(r > 5e-3, "{what}: rel_l2 {r} is as small as the bf16 path's: the fp8 path did not run");
        assert!(r < 0.12, "{what}: rel_l2 {r} against the cpu reference");
        // grouping independence: the call with the tokens in reverse order gives the reversed rows, bit for bit
        let t = hc.inputs[0].shape[0];
        let perm: Vec<usize> = (0..t).rev().collect();
        let mut rev = hc.clone();
        rev.inputs[0] = permute_rows(&hc.inputs[0], &perm); // x
        rev.inputs[1] = permute_rows(&hc.inputs[1], &perm); // ids
        rev.inputs[2] = permute_rows(&hc.inputs[2], &perm); // weights
        let out_rev = run_host(&*hip, "moe_experts_bf16_v2", &rev).unwrap().remove(0);
        let back = permute_rows(&out_rev, &perm);
        assert!(back.data == got.data, "{what}: a token's output depends on the other tokens of the call");
    }
}
