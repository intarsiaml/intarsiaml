//! `moe_experts_w4a16_bf16` with `INTARSIA_CUDA_MOE_W4A16_IP=1` (in-place mma accumulation, kernels/cuda/moe_w4a16.cu): not
//! bitwise the grouped kernel (that is the point of the knob), so the bar is the accuracy bound of
//! `moe_experts_w4a16_is_as_accurate_as_old_kernel` (tests/moe_w4a16.rs): against the cpu reference, rel_l2 within 2.5x of the
//! old kernel's own and at most 3x its fraction of off-reference elements, on all tile heights, plus repeatability and that
//! a call's rows do not depend on the other rows of the batch. Own test binary: the launcher reads the knob once per process.
//! Needs a GPU: without one the tests print SKIPPED (and fail under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_cuda)]
mod common;

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::backend::Backend;
use intarsia_core::exec::run_host;
use intarsia_core::quant::cases::experts_case;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor};
use std::sync::{Arc, Once};

const V2: &str = "moe_experts_w4a16_bf16";

fn device() -> Option<Arc<dyn Backend>> {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| std::env::set_var("INTARSIA_CUDA_MOE_W4A16_IP", "1"));
    let ord = std::env::var("INTARSIA_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_cuda::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            common::skip("moe_w4a16_ip", &format!("no usable CUDA device ({e})"));
            None
        }
    }
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

#[test]
fn moe_w4a16_in_place_is_as_accurate_as_old_kernel() {
    let Some(cuda) = device() else { return };
    let cpu = CpuBackend::new();
    let mut rng = SplitMix64::new(6);
    for (t, e) in [(128usize, 64usize), (512, 64), (1100, 64)] {
        let hc = experts_case(&mut rng, t, e, 8, 2048, 512, DType::Bf16);
        let want = run_host(&cpu, "ref_moe_experts", &hc).unwrap()[0].to_f32();
        let old = run_host(&*cuda, "moe_experts_bf16", &hc).unwrap()[0].to_f32();
        for bm in ["32", "64"] {
            std::env::set_var("INTARSIA_MOE_BM", bm);
            let new = run_host(&*cuda, V2, &hc).unwrap()[0].to_f32();
            let again = run_host(&*cuda, V2, &hc).unwrap()[0].to_f32();
            assert!(new == again, "T={t} BM={bm}: not repeatable");
            let frac = |a: &[f32]| a.iter().zip(&want).filter(|(x, y)| x != y).count() as f64 / want.len() as f64;
            let (fo, fn_) = (frac(&old), frac(&new));
            let (ro, rn) = (rel_l2(&old, &want), rel_l2(&new, &want));
            eprintln!("T={t} E={e} BM={bm}: off-reference old {fo:.4} in-place {fn_:.4}; rel_l2 old {ro:.3e} in-place {rn:.3e}");
            assert!(fn_ <= 3.0 * fo.max(1e-3), "T={t} BM={bm}: {:.2} % off the reference, old kernel {:.2} %", fn_ * 100.0, fo * 100.0);
            assert!(rn <= 2.5 * ro, "T={t} BM={bm}: in-place rel_l2 {rn:.3e}, old kernel {ro:.3e}");
        }
        std::env::remove_var("INTARSIA_MOE_BM");
    }
}

/// A token's output must not depend on which other tokens share the call (the prefill chunk length changes, a tail chunk
/// runs the same tokens in a smaller batch): row 0..n of a big call equal the same rows run as a small call, when the
/// routing of those rows is the same. Expert grouping changes the tile an expert's pairs land in, not the k-order of a
/// pair's sum, so the bits must agree.
#[test]
fn moe_w4a16_in_place_rows_do_not_depend_on_the_batch() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(7);
    let (t_big, t_small, e, k, h, i) = (1100usize, 70usize, 64usize, 8usize, 2048usize, 512usize);
    let big = experts_case(&mut rng, t_big, e, k, h, i, DType::Bf16);
    let mut small = big.clone();
    small.inputs[0] = HostTensor { shape: vec![t_small, h], data: big.inputs[0].data[..t_small * h * 2].to_vec(), ..big.inputs[0].clone() };
    small.inputs[1] = HostTensor { shape: vec![t_small, k], data: big.inputs[1].data[..t_small * k * 4].to_vec(), ..big.inputs[1].clone() };
    small.inputs[2] = HostTensor { shape: vec![t_small, k], data: big.inputs[2].data[..t_small * k * 4].to_vec(), ..big.inputs[2].clone() };
    small.outputs = vec![intarsia_core::exec::OutSpec::new(DType::Bf16, &[t_small, h])];
    let yb = run_host(&*cuda, V2, &big).unwrap()[0].data.clone();
    let ys = run_host(&*cuda, V2, &small).unwrap()[0].data.clone();
    assert!(yb[..t_small * h * 2] == ys[..], "in-place accumulation made a row depend on the batch it ran in");
}
