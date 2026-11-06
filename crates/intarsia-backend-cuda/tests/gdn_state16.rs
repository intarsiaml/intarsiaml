//! Lane spark35: the GatedDeltaNet delta-rule state stored as bf16 (`Backend::gdn_state_supported`, `INTARSIA_GDN_STATE=bf16`) is the f32
//! state rounded to bf16 (to nearest even) after every kernel launch, and nothing else. `gdn_delta_rule_bf16_split` (the T > 1 prefill
//! kernel and, for dk = 128, the T = 1 decode kernel) is run on a bf16 state and on an f32 state that holds the same (bf16-representable)
//! values; the outputs must be the same bytes and the bf16 state after the call must be the f32 state after the call, rounded. Each
//! step feeds the next one with the bf16 state and with the rounded f32 state, so a rounding that differs, a load that mis-reads the
//! 8-byte groups or a store that lands in the wrong row shows within the first steps.
//!
//! Needs a GPU: without one the tests print SKIPPED (and fail under CI=1 / INTARSIA_REQUIRE_GPU=1).
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
            common::skip("gdn_state16", &format!("no usable CUDA device ({e})"));
            None
        }
    }
}

fn rand_t(rng: &mut SplitMix64, dt: DType, shape: &[usize], scale: f32) -> HostTensor {
    let n: usize = shape.iter().product();
    HostTensor::from_f32_as(dt, shape, &rng.normal_vec(n, scale))
}

/// The inputs of one delta-rule call of `t` tokens, and the call with `state` (f32 or bf16) as its in-place state.
fn call(inputs: &[HostTensor], t: usize, (hv, dk, dv): (usize, usize, usize), state: HostTensor) -> HostCall {
    HostCall {
        op: Op::GdnDeltaRule,
        inputs: inputs.to_vec(),
        outputs: vec![OutSpec::new(DType::Bf16, &[t, hv, dv]), OutSpec { dtype: state.dtype, shape: vec![hv, dv, dk], init: Some(state) }],
        attrs: OpAttrs { eps: 1e-6, ..Default::default() },
    }
}

fn round_bf16(shape: &[usize], v: &[f32]) -> Vec<f32> {
    HostTensor::from_f32_as(DType::Bf16, shape, v).to_f32()
}

/// (hk, hv, dk, dv): the 35B, dk 64, and dv 36 (rows that do not fill a warp's row group).
const SHAPES: [(usize, usize, usize, usize); 4] = [(16, 32, 128, 128), (2, 4, 64, 40), (2, 2, 128, 36), (1, 2, 128, 128)];

#[test]
fn bf16_state_is_the_f32_state_rounded_after_every_launch() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(0x535);
    let mut launches = 0;
    for (hk, hv, dk, dv) in SHAPES {
        let shape = [hv, dv, dk];
        let c = 2 * hk * dk + hv * dv;
        // a non-zero start that is exactly representable in bf16 (so both layouts start from the same values)
        let mut s32 = HostTensor::from_f32_as(DType::F32, &shape, &round_bf16(&shape, &rng.normal_vec(hv * dv * dk, 0.1)));
        let mut s16 = HostTensor::from_f32_as(DType::Bf16, &shape, &s32.to_f32());
        // T = 1 is the decode kernel (dk = 128), the rest the split kernel: tails, a lane group, a chunk
        for t in [1usize, 1, 2, 17, 1, 130, 1, 3, 1, 513, 1, 1, 64, 1] {
            let gate = 1.0;
            let inputs = [
                rand_t(&mut rng, DType::Bf16, &[t, c], 1.0),
                rand_t(&mut rng, DType::Bf16, &[t, hv], gate),
                rand_t(&mut rng, DType::Bf16, &[t, hv], gate),
                rand_t(&mut rng, DType::Bf16, &[hv], 0.5),
                rand_t(&mut rng, DType::Bf16, &[hv], 0.5),
            ];
            let r32 = run_host(&*cuda, "gdn_delta_rule_bf16_split", &call(&inputs, t, (hv, dk, dv), s32.clone())).unwrap();
            let r16 = run_host(&*cuda, "gdn_delta_rule_bf16_split", &call(&inputs, t, (hv, dk, dv), s16.clone())).unwrap();
            assert!(r16[0].data == r32[0].data, "{:?} T={t}: outputs differ between the bf16 and the f32 state", (hk, hv, dk, dv));
            assert_eq!(r16[1].dtype, DType::Bf16);
            let want = round_bf16(&shape, &r32[1].to_f32());
            let got = r16[1].to_f32();
            if let Some(i) = got.iter().zip(&want).position(|(a, b)| a.to_bits() != b.to_bits()) {
                panic!("{:?} T={t}: bf16 state value {i} is {} but the rounded f32 state is {}", (hk, hv, dk, dv), got[i], want[i]);
            }
            // carry on from the same values: the bf16 state as stored, the f32 state rounded
            s16 = r16[1].clone();
            s32 = HostTensor::from_f32_as(DType::F32, &shape, &want);
            launches += 1;
        }
    }
    println!("gdn_state16: bf16 state == f32 state rounded after every launch, {launches} launches over {} shapes", SHAPES.len());
}

/// How far the bf16 state wanders from the f32 one over 1000 decode steps of random inputs (informational: the relative L2 distance
/// between the two layouts' states is printed every 100 steps; the quality gate is the model-level KL, not this number).
#[test]
fn bf16_state_stays_near_the_f32_state_over_many_decode_steps() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(0x536);
    let (hk, hv, dk, dv) = (2usize, 4usize, 128usize, 128usize);
    let shape = [hv, dv, dk];
    let c = 2 * hk * dk + hv * dv;
    let mut s32 = HostTensor::zeros(DType::F32, &shape);
    let mut s16 = HostTensor::zeros(DType::Bf16, &shape);
    // one fixed (a, b, A_log, dt_bias) so the decay is stationary, fresh q / k / v every step
    let abg = [rand_t(&mut rng, DType::Bf16, &[1, hv], 1.0), rand_t(&mut rng, DType::Bf16, &[1, hv], 1.0)];
    let heads = [rand_t(&mut rng, DType::Bf16, &[hv], 0.5), rand_t(&mut rng, DType::Bf16, &[hv], 0.5)];
    let mut worst = 0.0f64;
    for step in 0..1000 {
        let inputs = [rand_t(&mut rng, DType::Bf16, &[1, c], 1.0), abg[0].clone(), abg[1].clone(), heads[0].clone(), heads[1].clone()];
        s32 = run_host(&*cuda, "gdn_delta_rule_bf16_split", &call(&inputs, 1, (hv, dk, dv), s32)).unwrap().remove(1);
        s16 = run_host(&*cuda, "gdn_delta_rule_bf16_split", &call(&inputs, 1, (hv, dk, dv), s16)).unwrap().remove(1);
        if step % 100 == 99 {
            let (a, b) = (s32.to_f32(), s16.to_f32());
            let num: f64 = a.iter().zip(&b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
            let den: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum();
            let rel = (num / den.max(1e-30)).sqrt();
            println!("gdn_state16: step {}: |S16 - S32| / |S32| = {rel:.3e}", step + 1);
            worst = worst.max(rel);
        }
    }
    assert!(worst.is_finite(), "non-finite state distance");
}
