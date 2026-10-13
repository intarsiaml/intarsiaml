//! `gdn_delta_rule_bf16_lanes` (several lanes per state row, prefill) against the cpu f64 reference and against the
//! one-thread-per-row kernel it replaces for T >= 16, at the Qwen3.6-35B-A3B shape (16 key / 32 value heads of 128) and
//! a dk = 64 shape. Needs a GPU: without one the tests print SKIPPED (and fail under CI=1 / INTARSIA_REQUIRE_GPU=1).
//! Runs only in the spark1 run (see docs/results/2026-10-13-prefill-cuda.md).
#![cfg(ik_cuda)]

mod common;

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::backend::Backend;
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, Op, OpAttrs};
use std::sync::Arc;

fn device() -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_cuda::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            common::skip("gdn_lanes", &format!("no usable CUDA device ({e})"));
            None
        }
    }
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

fn rand_t(rng: &mut SplitMix64, dt: DType, shape: &[usize], scale: f32) -> HostTensor {
    let n: usize = shape.iter().product();
    HostTensor::from_f32_as(dt, shape, &rng.normal_vec(n, scale))
}

/// A delta-rule call with a non-zero initial state (so the carry matters).
fn delta_case(rng: &mut SplitMix64, t: usize, hk: usize, hv: usize, dk: usize, dv: usize) -> HostCall {
    let c = 2 * hk * dk + hv * dv;
    let dt = DType::Bf16;
    HostCall {
        op: Op::GdnDeltaRule,
        inputs: vec![
            rand_t(rng, dt, &[t, c], 1.0),
            rand_t(rng, dt, &[t, hv], 1.0),
            rand_t(rng, dt, &[t, hv], 1.0),
            rand_t(rng, dt, &[hv], 0.5),
            rand_t(rng, dt, &[hv], 0.5),
        ],
        outputs: vec![
            OutSpec::new(dt, &[t, hv, dv]),
            OutSpec { dtype: DType::F32, shape: vec![hv, dv, dk], init: Some(rand_t(rng, DType::F32, &[hv, dv, dk], 0.1)) },
        ],
        attrs: OpAttrs { eps: 1e-6, ..Default::default() },
    }
}

const SHAPES: [(usize, usize, usize, usize); 3] = [(16, 32, 128, 128), (2, 4, 64, 40), (2, 2, 128, 72)];

/// The multi-lane kernel is as close to the f64 reference as the kernel it replaces (output and carried state), at
/// token counts around the lane/warp boundaries and a row count (dv = 40, 72) that is not a multiple of the rows per warp.
#[test]
fn gdn_lanes_is_as_accurate_as_the_rows_kernel() {
    let Some(cuda) = device() else { return };
    let cpu = CpuBackend::new();
    let mut rng = SplitMix64::new(61);
    for (hk, hv, dk, dv) in SHAPES {
        for t in [16usize, 17, 130, 513] {
            let hc = delta_case(&mut rng, t, hk, hv, dk, dv);
            let want = run_host(&cpu, "ref_gdn_delta_rule", &hc).unwrap();
            let old = run_host(&*cuda, "gdn_delta_rule_bf16", &hc).unwrap();
            let new = run_host(&*cuda, "gdn_delta_rule_bf16_lanes", &hc).unwrap();
            for (i, what) in ["o", "state"].iter().enumerate() {
                let (w, o, n) = (want[i].to_f32(), old[i].to_f32(), new[i].to_f32());
                let (ro, rn) = (rel_l2(&o, &w), rel_l2(&n, &w));
                eprintln!("dk={dk} dv={dv} T={t} {what}: rel_l2 vs f64 reference: rows kernel {ro:.3e}, lanes kernel {rn:.3e}");
                assert!(rn <= 1e-3 && rn <= 2.0 * ro.max(1e-6), "dk={dk} dv={dv} T={t} {what}: lanes rel_l2 {rn:.3e}, rows kernel {ro:.3e}");
            }
            // repeatable
            let again = run_host(&*cuda, "gdn_delta_rule_bf16_lanes", &hc).unwrap();
            assert_eq!(new[0].data, again[0].data, "dk={dk} T={t}: not repeatable");
            assert_eq!(new[1].data, again[1].data, "dk={dk} T={t}: state not repeatable");
        }
    }
}

/// The recurrence is token-sequential, so splitting a sequence into two calls that carry the state must give the same
/// bits as one call (outputs and final state). A wrong state load/store, a dropped token or a decay applied at the
/// wrong place breaks this.
#[test]
fn gdn_lanes_state_carries_across_calls() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(62);
    let (hk, hv, dk, dv) = SHAPES[0];
    let (t1, t2) = (70usize, 59usize);
    let whole = delta_case(&mut rng, t1 + t2, hk, hv, dk, dv);
    let out_whole = run_host(&*cuda, "gdn_delta_rule_bf16_lanes", &whole).unwrap();
    let c = 2 * hk * dk + hv * dv;
    let rows = |t: &HostTensor, width: usize, r0: usize, r1: usize| {
        let w = t.dtype.size_bytes() * width;
        HostTensor::new(t.dtype, vec![r1 - r0, width], t.data[r0 * w..r1 * w].to_vec()).unwrap()
    };
    let part = |r0: usize, r1: usize, state: &HostTensor| HostCall {
        op: Op::GdnDeltaRule,
        inputs: vec![
            rows(&whole.inputs[0], c, r0, r1),
            rows(&whole.inputs[1], hv, r0, r1),
            rows(&whole.inputs[2], hv, r0, r1),
            whole.inputs[3].clone(),
            whole.inputs[4].clone(),
        ],
        outputs: vec![
            OutSpec::new(DType::Bf16, &[r1 - r0, hv, dv]),
            OutSpec { dtype: DType::F32, shape: vec![hv, dv, dk], init: Some(state.clone()) },
        ],
        attrs: whole.attrs.clone(),
    };
    let init = whole.outputs[1].init.clone().unwrap();
    let a = run_host(&*cuda, "gdn_delta_rule_bf16_lanes", &part(0, t1, &init)).unwrap();
    let b = run_host(&*cuda, "gdn_delta_rule_bf16_lanes", &part(t1, t1 + t2, &a[1])).unwrap();
    let mut joined = a[0].data.clone();
    joined.extend_from_slice(&b[0].data);
    assert!(joined == out_whole[0].data, "outputs of the split run differ from the single call");
    assert!(b[1].data == out_whole[1].data, "final state of the split run differs from the single call");
}
