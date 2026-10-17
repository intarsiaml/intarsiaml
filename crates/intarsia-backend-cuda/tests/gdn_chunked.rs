//! `gdn_delta_rule_bf16_chunked` (WY chunk form, kernels/cuda/gdn_chunked.cu) against the cpu f64 reference and against the
//! sequential one-thread-per-row kernel, at the Qwen3.6-35B-A3B shape (16 key / 32 value heads of 128) and smaller shapes
//! (dk = 64, several heads per key head). Needs a GPU: without one the tests print SKIPPED (and fail under CI=1 /
//! INTARSIA_REQUIRE_GPU=1). Runs only in the spark1 run (docs/results/2026-10-17-cuda-prefill-2.md).
//!
//! The chunked form is not bitwise independent of where a sequence is split into calls (the chunk boundaries move), so the
//! carry tests bound the error against the f64 reference of the whole sequence instead of comparing bits.
#![cfg(ik_cuda)]

mod common;

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::backend::Backend;
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, Op, OpAttrs};
use std::sync::Arc;

const CHUNKED: &str = "gdn_delta_rule_bf16_chunked";
const ROWS: &str = "gdn_delta_rule_bf16";

fn device() -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_cuda::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            common::skip("gdn_chunked", &format!("no usable CUDA device ({e})"));
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

/// A delta-rule call with a non-zero initial state (so the carry matters). `dt_bias_mean` shifts the decay: 0 gives a
/// decay of about exp(-1) per token, -4 a decay close to 1 (a long memory, so the inter-chunk decay and carry matter).
fn delta_case(rng: &mut SplitMix64, t: usize, hk: usize, hv: usize, dk: usize, dv: usize, dt_bias_mean: f32) -> HostCall {
    let c = 2 * hk * dk + hv * dv;
    let dt = DType::Bf16;
    let dt_bias: Vec<f32> = rng.normal_vec(hv, 0.5).into_iter().map(|x| x + dt_bias_mean).collect();
    HostCall {
        op: Op::GdnDeltaRule,
        inputs: vec![
            rand_t(rng, dt, &[t, c], 1.0),
            rand_t(rng, dt, &[t, hv], 1.0),
            rand_t(rng, dt, &[t, hv], 1.0),
            rand_t(rng, dt, &[hv], 0.5),
            HostTensor::from_f32_as(dt, &[hv], &dt_bias),
        ],
        outputs: vec![
            OutSpec::new(dt, &[t, hv, dv]),
            OutSpec { dtype: DType::F32, shape: vec![hv, dv, dk], init: Some(rand_t(rng, DType::F32, &[hv, dv, dk], 0.1)) },
        ],
        attrs: OpAttrs { eps: 1e-6, ..Default::default() },
    }
}

/// (hk, hv, dk, dv); dv must be a multiple of 32.
const SHAPES: [(usize, usize, usize, usize); 4] = [(16, 32, 128, 128), (2, 4, 64, 64), (2, 2, 128, 96), (1, 4, 128, 32)];

/// The bound every comparison uses: as close to the f64 reference as the sequential kernel (the reference rounds its output
/// to bf16, so a kernel that is exact in fp32 still differs by an occasional one-ulp flip: rel_l2 ~ 5e-5).
fn within(rn: f64, ro: f64) -> bool {
    rn <= 1e-3 && rn <= 2.0 * ro.max(5e-5)
}

/// Token counts around the chunk boundaries (a tail of 1, 2, 36, 63 tokens past a multiple of 64).
const TS: [usize; 9] = [64, 65, 66, 100, 127, 128, 129, 300, 513];

#[test]
fn gdn_chunked_is_as_accurate_as_the_rows_kernel() {
    let Some(cuda) = device() else { return };
    let cpu = CpuBackend::new();
    let mut rng = SplitMix64::new(71);
    for (hk, hv, dk, dv) in SHAPES {
        for t in TS {
            // the large shape against a long reference only at a few lengths: the f64 reference is slow
            if hv == 32 && !matches!(t, 64 | 65 | 129 | 513) {
                continue;
            }
            for dt_mean in [0.0f32, -4.0] {
                let hc = delta_case(&mut rng, t, hk, hv, dk, dv, dt_mean);
                let want = run_host(&cpu, "ref_gdn_delta_rule", &hc).unwrap();
                let old = run_host(&*cuda, ROWS, &hc).unwrap();
                let new = run_host(&*cuda, CHUNKED, &hc).unwrap();
                for (i, what) in ["o", "state"].iter().enumerate() {
                    let (w, o, n) = (want[i].to_f32(), old[i].to_f32(), new[i].to_f32());
                    let (ro, rn) = (rel_l2(&o, &w), rel_l2(&n, &w));
                    eprintln!("dk={dk} dv={dv} hv={hv} T={t} dtb={dt_mean} {what}: rel_l2 vs f64 reference: rows kernel {ro:.3e}, chunked kernel {rn:.3e}");
                    assert!(within(rn, ro), "dk={dk} dv={dv} T={t} dtb={dt_mean} {what}: chunked rel_l2 {rn:.3e}, rows kernel {ro:.3e}");
                }
                let again = run_host(&*cuda, CHUNKED, &hc).unwrap();
                assert_eq!(new[0].data, again[0].data, "dk={dk} T={t}: not repeatable");
                assert_eq!(new[1].data, again[1].data, "dk={dk} T={t}: state not repeatable");
            }
        }
    }
}

/// Strongly correlated keys (every k is the same direction plus a little noise) with beta close to 1 and almost no decay:
/// the worst conditioning of I + L, where a missing or wrong triangular-solve correction is largest.
#[test]
fn gdn_chunked_handles_correlated_keys() {
    let Some(cuda) = device() else { return };
    let cpu = CpuBackend::new();
    let mut rng = SplitMix64::new(72);
    let (hk, hv, dk, dv) = (2usize, 4usize, 128usize, 64usize);
    for t in [64usize, 130, 257] {
        let mut hc = delta_case(&mut rng, t, hk, hv, dk, dv, -6.0);
        let c = 2 * hk * dk + hv * dv;
        let mut qkv = hc.inputs[0].to_f32();
        let base: Vec<f32> = rng.normal_vec(hk * dk, 1.0);
        let noise = rng.normal_vec(t * hk * dk, 0.05);
        for ti in 0..t {
            for i in 0..hk * dk {
                qkv[ti * c + hk * dk + i] = base[i] + noise[ti * hk * dk + i];
            }
        }
        hc.inputs[0] = HostTensor::from_f32_as(DType::Bf16, &[t, c], &qkv);
        hc.inputs[2] = HostTensor::from_f32_as(DType::Bf16, &[t, hv], &vec![6.0f32; t * hv]);
        let want = run_host(&cpu, "ref_gdn_delta_rule", &hc).unwrap();
        let old = run_host(&*cuda, ROWS, &hc).unwrap();
        let new = run_host(&*cuda, CHUNKED, &hc).unwrap();
        for (i, what) in ["o", "state"].iter().enumerate() {
            let (w, o, n) = (want[i].to_f32(), old[i].to_f32(), new[i].to_f32());
            let (ro, rn) = (rel_l2(&o, &w), rel_l2(&n, &w));
            eprintln!("correlated T={t} {what}: rel_l2 vs f64 reference: rows kernel {ro:.3e}, chunked kernel {rn:.3e}");
            assert!(within(rn, ro.max(1e-4)), "correlated T={t} {what}: chunked rel_l2 {rn:.3e}, rows kernel {ro:.3e}");
        }
    }
}

/// Splitting a sequence into two calls that carry the state gives the f64 reference's answer to the same accuracy as one
/// call (outputs and final state), for splits that cut a chunk and splits that do not. Not bitwise: the second call's
/// chunks start at the split. A wrong state load/store, a dropped chunk, a decay applied at the wrong place breaks it.
#[test]
fn gdn_chunked_state_carries_across_calls() {
    for (t1, t2, dt_mean) in [(70usize, 59usize, 0.0f32), (128, 128, -4.0), (200, 57, -4.0), (64, 64, 0.0), (130, 70, -4.0)] {
        gdn_chunked_split_case(t1, t2, dt_mean);
    }
}

fn gdn_chunked_split_case(t1: usize, t2: usize, dt_mean: f32) {
    let Some(cuda) = device() else { return };
    let cpu = CpuBackend::new();
    let mut rng = SplitMix64::new(73 + (t1 * 31 + t2) as u64);
    let (hk, hv, dk, dv) = (2usize, 4usize, 128usize, 96usize);
    let whole = delta_case(&mut rng, t1 + t2, hk, hv, dk, dv, dt_mean);
    let want = run_host(&cpu, "ref_gdn_delta_rule", &whole).unwrap();
    let out_whole = run_host(&*cuda, CHUNKED, &whole).unwrap();
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
    let a = run_host(&*cuda, CHUNKED, &part(0, t1, &init)).unwrap();
    let b = run_host(&*cuda, CHUNKED, &part(t1, t1 + t2, &a[1])).unwrap();
    let mut joined = a[0].data.clone();
    joined.extend_from_slice(&b[0].data);
    let joined = HostTensor::new(DType::Bf16, vec![t1 + t2, hv, dv], joined).unwrap();
    let (w_o, w_s) = (want[0].to_f32(), want[1].to_f32());
    let (r_whole_o, r_split_o) = (rel_l2(&out_whole[0].to_f32(), &w_o), rel_l2(&joined.to_f32(), &w_o));
    let (r_whole_s, r_split_s) = (rel_l2(&out_whole[1].to_f32(), &w_s), rel_l2(&b[1].to_f32(), &w_s));
    eprintln!("split {t1}+{t2} dtb={dt_mean}: o rel_l2 whole {r_whole_o:.3e} split {r_split_o:.3e}; state whole {r_whole_s:.3e} split {r_split_s:.3e}");
    assert!(within(r_split_o, r_whole_o), "split {t1}+{t2}: outputs rel_l2 {r_split_o:.3e} (one call {r_whole_o:.3e})");
    assert!(within(r_split_s, r_whole_s), "split {t1}+{t2}: final state rel_l2 {r_split_s:.3e} (one call {r_whole_s:.3e})");
}
