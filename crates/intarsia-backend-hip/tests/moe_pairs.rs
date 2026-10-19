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

/// with the projection variant bits INTARSIA_HIP_MOE_MW (moe_wmma.h: skip empty fragments, occupancy; moe_wmma_r7.h: fp8-conversion
/// decode, two A fragments per wave)
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
        // h = 320: the two-fragment down's 256-feature tile runs past the last row; i = 192: three k blocks in down, fewer
        // than the deep ring's six
        (300, 16, 4, 320, 192),
    ] {
        cases.push((format!("t={t} e={e} k={k} h={h} i={i}"), experts_case(&mut rng, t, e, k, h, i, DType::Bf16)));
    }
    // every token on experts 0..k: 1000 pairs each, more pair tiles than the grid of either tile size holds
    let mut hot = experts_case(&mut rng, 1000, 16, 4, 256, 64, DType::Bf16);
    let ids: Vec<i32> = (0..1000).flat_map(|_| 0..4i32).collect();
    hot.inputs[1] = HostTensor::from_i32(&[1000, 4], &ids);
    cases.push(("hot experts".into(), hot));
    // every finite e4m3 block-scale byte (subnormals, zero, negative; random_rows only draws 0x30..0x41), at both projections:
    // the round-7 fp4 decode (moe_wmma_r7.h kMwCvt) folds 2^6 into the scale, which must stay exact for all of them
    let mut scl = experts_case(&mut rng, 300, 16, 4, 256, 128, DType::Bf16);
    // 0..=253 -> every byte but the NaNs 0x7f and 0xff
    let finite = |j: usize| -> u8 { (j % 254) as u8 + u8::from(j % 254 >= 0x7f) };
    for idx in [4usize, 6] {
        for (j, b) in scl.inputs[idx].data.iter_mut().enumerate() {
            *b = finite(j.wrapping_mul(97) + j / 254);
        }
    }
    cases.push(("all finite scale bytes".into(), scl));
    // a few NaN scale bytes (0x7f / 0xff) among ordinary ones: NaN outputs, bit for bit too
    let mut nan = experts_case(&mut rng, 64, 16, 4, 256, 128, DType::Bf16);
    for idx in [4usize, 6] {
        let n = nan.inputs[idx].data.len();
        for j in (0..n).step_by(997) {
            nan.inputs[idx].data[j] = if j % 2 == 0 { 0x7f } else { 0xff };
        }
    }
    cases.push(("NaN scale bytes".into(), nan));
    let mut differing_from_zero = 0usize;
    // the reference and every comparison above the round-8 block: one launch per projection
    std::env::set_var("INTARSIA_HIP_SPLIT_US", "0");
    for (what, hc) in &cases {
        let a = run(&*hip, hc, "32");
        // round 8: the weighted expert sum with 8 features per thread (moe_reduce_v8_kernel) against the scalar kernel
        std::env::set_var("INTARSIA_HIP_MOE_REDUCE", "0");
        let r = run(&*hip, hc, "32");
        std::env::remove_var("INTARSIA_HIP_MOE_REDUCE");
        assert!(a.data == r.data, "{what}: moe_reduce_v8_kernel differs from moe_reduce_kernel");
        let b = run(&*hip, hc, "64");
        let c = run(&*hip, hc, "0");
        assert!(a.data == b.data, "{what}: the 64-pair tile changed the output");
        assert!(a.data == c.data, "{what}: the auto tile changed the output");
        // every projection variant at both tile sizes (variant 0 at 32 pairs = a, the reference): round 6's 1, 4, 5
        // (moe_wmma.h) and round 7's 13, 25, 29, 41, 45 (moe_wmma_r7.h: 8 fp8-conversion fp4 decode, 16 two A fragments in
        // down, 32 six weight blocks in flight)
        for mw in [1, 4, 5, 13, 25, 29, 41, 45] {
            for pairs in ["32", "64"] {
                let v = run_mw(&*hip, hc, pairs, &mw.to_string());
                assert!(a.data == v.data, "{what}: INTARSIA_HIP_MOE_MW={mw} at {pairs} pairs changed the output");
            }
        }
        // the round-7 defaults (no knob) and gate_up / down set apart, at both tiles
        for (pairs, mw) in [("32", None), ("64", None), ("32", Some("0:25")), ("64", Some("45:0")), ("64", Some("5:41"))] {
            std::env::set_var("INTARSIA_HIP_MOE_PAIRS", pairs);
            match mw {
                Some(m) => std::env::set_var("INTARSIA_HIP_MOE_MW", m),
                None => std::env::remove_var("INTARSIA_HIP_MOE_MW"),
            }
            let v = run_host(&*hip, "moe_experts_bf16_v2", hc).unwrap().remove(0);
            assert!(a.data == v.data, "{what}: INTARSIA_HIP_MOE_MW={mw:?} at {pairs} pairs changed the output");
        }
        // round 8: each projection launched as pieces over its feature tiles (INTARSIA_HIP_SPLIT_US, kernels/hip/ik_hip.h;
        // 1 = one feature tile per launch, 150 = a few uneven pieces at the larger cases), every kernel family (0 = the
        // round-5 kernel, 5 = moe_wmma_kernel_v, the r7 defaults and two r7 splits) at both tiles, against the single launch
        for split in ["1", "150"] {
            std::env::set_var("INTARSIA_HIP_SPLIT_US", split);
            std::env::set_var("INTARSIA_HIP_SPLIT_MIN_US", "0"); // round 9 trigger off: pieces at every size
            for (pairs, mw) in [("32", "0"), ("64", "0"), ("32", "5"), ("64", "5"), ("32", "0:41"), ("64", "45:25"), ("64", "13:29")] {
                let v = run_mw(&*hip, hc, pairs, mw);
                assert!(
                    a.data == v.data,
                    "{what}: INTARSIA_HIP_SPLIT_US={split} with INTARSIA_HIP_MOE_MW={mw} at {pairs} pairs changed the output"
                );
            }
            std::env::set_var("INTARSIA_HIP_SPLIT_US", "0");
        }
        differing_from_zero += a.to_f32().iter().filter(|v| **v != 0.0).count();
    }
    std::env::remove_var("INTARSIA_HIP_MOE_PAIRS");
    std::env::remove_var("INTARSIA_HIP_MOE_MW");
    std::env::remove_var("INTARSIA_HIP_SPLIT_US");
    std::env::remove_var("INTARSIA_HIP_SPLIT_MIN_US");
    assert!(differing_from_zero > 0, "all outputs zero: nothing was computed");
}
