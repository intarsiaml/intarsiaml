//! The pair tile of the grouped MoE projections (kernels/hip/moe_wmma.h, `PAIRS`; INTARSIA_HIP_MOE_PAIRS = 32 | 64 | auto) must
//! not change a bit of the output: every pair accumulates over k in the same order whatever else is in its tile. This runs the
//! same calls with 32 and with 64 pairs per block (the launcher reads the variable at every launch) and compares the outputs
//! bitwise, on routings that leave tiles partly filled, an expert with more tiles than the grid, the 35B's real shape, and
//! token counts around the 16-token threshold of the tensor-core path. Accuracy against the cpu reference (both tile sizes,
//! since auto takes 64 for busy experts) is tests/quantmoe.rs.
//!
//! Its own test binary with a single test: it sets an environment variable, which no other test may race with.
#![cfg(ik_hip)]

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

fn run(hip: &dyn Backend, hc: &HostCall, pairs: &str) -> HostTensor {
    run_mw(hip, hc, pairs, "0")
}

/// with the projection variant bits INTARSIA_HIP_MOE_MW (moe_wmma.h: skip empty fragments, byte table, occupancy)
fn run_mw(hip: &dyn Backend, hc: &HostCall, pairs: &str, mw: &str) -> HostTensor {
    std::env::set_var("INTARSIA_HIP_MOE_PAIRS", pairs);
    std::env::set_var("INTARSIA_HIP_MOE_MW", mw);
    run_host(hip, "moe_experts_bf16_v2", hc).unwrap().remove(0)
}

#[test]
fn moe_pair_tile_does_not_change_a_bit() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(77);
    let mut cases: Vec<(String, HostCall)> = Vec::new();
    for &(t, e, k, h, i) in &[
        (16usize, 16usize, 4usize, 256usize, 64usize),
        (37, 16, 4, 256, 64),
        (300, 16, 4, 256, 128),
        (1024, 256, 8, 2048, 512),
        (2048, 256, 8, 2048, 512),
    ] {
        cases.push((format!("t={t} e={e} k={k} h={h} i={i}"), experts_case(&mut rng, t, e, k, h, i, DType::Bf16)));
    }
    // every token on experts 0..k: 1000 pairs each, more pair tiles than the grid of either tile size holds
    let mut hot = experts_case(&mut rng, 1000, 16, 4, 256, 64, DType::Bf16);
    let ids: Vec<i32> = (0..1000).flat_map(|_| 0..4i32).collect();
    hot.inputs[1] = HostTensor::from_i32(&[1000, 4], &ids);
    cases.push(("hot experts".into(), hot));
    let mut differing_from_zero = 0usize;
    for (what, hc) in &cases {
        let a = run(&*hip, hc, "32");
        let b = run(&*hip, hc, "64");
        let c = run(&*hip, hc, "0");
        assert!(a.data == b.data, "{what}: the 64-pair tile changed the output");
        assert!(a.data == c.data, "{what}: the auto tile changed the output");
        // every projection variant at both tile sizes (variant 0 at 32 pairs = a, the reference)
        for mw in [1, 4, 5] {
            for pairs in ["32", "64"] {
                let v = run_mw(&*hip, hc, pairs, &mw.to_string());
                assert!(a.data == v.data, "{what}: INTARSIA_HIP_MOE_MW={mw} at {pairs} pairs changed the output");
            }
        }
        differing_from_zero += a.to_f32().iter().filter(|v| **v != 0.0).count();
    }
    std::env::remove_var("INTARSIA_HIP_MOE_PAIRS");
    std::env::remove_var("INTARSIA_HIP_MOE_MW");
    assert!(differing_from_zero > 0, "all outputs zero: nothing was computed");
}
