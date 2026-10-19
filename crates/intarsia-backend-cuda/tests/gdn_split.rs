//! `gdn_delta_rule_bf16_split` (the one-thread-per-row GatedDeltaNet recurrence's arithmetic split over several lanes per
//! row) against `gdn_delta_rule_bf16` (the one-thread-per-row kernel it replaces for dk 64 / 128): the outputs and the
//! carried state must be the SAME BITS, at the Qwen3.6-35B-A3B shape (16 key / 32 value heads of 128), dk = 64 shapes and
//! row counts that are not a multiple of the rows per warp. The bits must not depend on the lanes per row: the dev script
//! runs this file under INTARSIA_GDN_SPLIT_LPR=2 (default), 4 and 8. Needs a GPU: without one the tests print SKIPPED (and
//! fail under CI=1 / INTARSIA_REQUIRE_GPU=1). docs/results/2026-10-19-cuda-prefill-4.md.
#![cfg(ik_cuda)]

mod common;

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
            common::skip("gdn_split", &format!("no usable CUDA device ({e})"));
            None
        }
    }
}

fn rand_t(rng: &mut SplitMix64, dt: DType, shape: &[usize], scale: f32) -> HostTensor {
    let n: usize = shape.iter().product();
    HostTensor::from_f32_as(dt, shape, &rng.normal_vec(n, scale))
}

/// A delta-rule call with a non-zero initial state. `gate` scales the raw a / b inputs: 1 is the usual range, 12 pushes
/// softplus past its x > 20 branch and the decay / beta to 0 and 1.
fn delta_case(rng: &mut SplitMix64, t: usize, (hk, hv, dk, dv): (usize, usize, usize, usize), gate: f32) -> HostCall {
    let c = 2 * hk * dk + hv * dv;
    let dt = DType::Bf16;
    HostCall {
        op: Op::GdnDeltaRule,
        inputs: vec![
            rand_t(rng, dt, &[t, c], 1.0),
            rand_t(rng, dt, &[t, hv], gate),
            rand_t(rng, dt, &[t, hv], gate),
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

/// (hk, hv, dk, dv): the 35B, dk 64 (dv 40 and 20), and dv 36 / 20 (not a multiple of 4 or 8 rows per warp).
const SHAPES: [(usize, usize, usize, usize); 5] =
    [(16, 32, 128, 128), (2, 4, 64, 40), (2, 2, 128, 36), (2, 4, 64, 20), (1, 2, 128, 128)];

fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
}

/// Bit-identical outputs and final state, for T around the lane / warp boundaries and a 2048-token prefill chunk.
#[test]
fn gdn_split_is_bitwise_the_rows_kernel() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(71);
    for shape in SHAPES {
        let ts: &[usize] = if shape.0 == 16 { &[1, 2, 3, 7, 16, 17, 130, 513, 2048] } else { &[1, 2, 17, 130, 513] };
        for &t in ts {
            for gate in [1.0f32, 12.0] {
                let hc = delta_case(&mut rng, t, shape, gate);
                let old = run_host(&*cuda, "gdn_delta_rule_bf16", &hc).unwrap();
                let new = run_host(&*cuda, "gdn_delta_rule_bf16_split", &hc).unwrap();
                for (i, what) in ["o", "state"].iter().enumerate() {
                    assert!(
                        old[i].data == new[i].data,
                        "{shape:?} T={t} gate={gate} {what}: differs from gdn_delta_rule_bf16 at byte {:?} of {}",
                        first_diff(&old[i].data, &new[i].data),
                        old[i].data.len()
                    );
                }
            }
        }
    }
}

/// Splitting a sequence into calls that carry the state gives the bits of one call (outputs and final state), tails of
/// 1 token (decode) included. Implied by the bitwise test for a chunk-invariant rows kernel; kept so that a wrong state
/// load / store of this kernel alone is named directly.
#[test]
fn gdn_split_state_carries_across_calls() {
    let Some(cuda) = device() else { return };
    for (t1, t2) in [(70usize, 59usize), (120, 10), (3, 1), (16, 1), (1, 1)] {
        let mut rng = SplitMix64::new(72 + (t1 * 31 + t2) as u64);
        let (hk, hv, dk, dv) = SHAPES[0];
        let whole = delta_case(&mut rng, t1 + t2, SHAPES[0], 1.0);
        let out_whole = run_host(&*cuda, "gdn_delta_rule_bf16_split", &whole).unwrap();
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
        let a = run_host(&*cuda, "gdn_delta_rule_bf16_split", &part(0, t1, &init)).unwrap();
        let b = run_host(&*cuda, "gdn_delta_rule_bf16_split", &part(t1, t1 + t2, &a[1])).unwrap();
        let mut joined = a[0].data.clone();
        joined.extend_from_slice(&b[0].data);
        assert!(joined == out_whole[0].data, "split {t1}+{t2}: outputs differ from the single call");
        assert!(b[1].data == out_whole[1].data, "split {t1}+{t2}: final state differs from the single call");
    }
}
