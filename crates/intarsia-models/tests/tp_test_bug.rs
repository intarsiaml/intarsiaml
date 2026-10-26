//! `INTARSIA_TP_TEST_BUG=scale:F`: the deliberate TP bug the accuracy gate's sensitivity is measured with
//! (docs/accuracy.md, "TP self-golden budget"). Its own test binary because it sets a process environment variable
//! before the models load. The tests here serialise on a mutex so none reads the variable while another sets it.

mod common;
use common::*;
use intarsia_backend_cpu::CpuBackend;
use intarsia_core::comm::HostCommGroup;
use intarsia_core::model::{HeadMode, LoadOptions, Model};
use intarsia_core::{Backend, DType, Telemetry};
use std::sync::{Arc, Mutex};

static ENV: Mutex<()> = Mutex::new(());

fn tp2_logits(t: &Tiny, opts: &LoadOptions, ids: &[u32]) -> Vec<f32> {
    let backends: Vec<Arc<dyn Backend>> = (0..2).map(|_| CpuBackend::arc()).collect();
    let comms = HostCommGroup::new(backends.clone());
    let m = Model::load_tp(backends, comms, &recipe(), &dims(), &t.weights, opts, Arc::new(Telemetry::new())).unwrap();
    let mut st = m.new_state().unwrap();
    let mut last = vec![];
    let mut cb = |_p: usize, r: &[f32]| {
        last = r.to_vec();
        Ok(())
    };
    m.forward(&mut st, ids, HeadMode::AllLogits, Some(&mut cb)).unwrap();
    last
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den).sqrt()
}

/// The guard that does catch a 1% partial-scale bug (the GPU accuracy gates do not, docs/accuracy.md "TP self-golden
/// budget"): f32 weights and KV, host comm, tp=2 against the unsharded model. Summation order moves the logits by ~1e-7
/// (the bound in `tp.rs` is 1e-5); the bug by orders more.
#[test]
fn f32_tp2_vs_tp1_separates_a_one_percent_partial_scale_bug_from_summation_order() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let t = tiny_weights(DType::F32, 5);
    let opts = LoadOptions { kv_dtype: Some(DType::F32), ..Default::default() };
    let ids: Vec<u32> = (0..9).map(|i| ((i * 37 + 11) % V) as u32).collect();
    unsafe { std::env::remove_var("INTARSIA_TP_TEST_BUG") };
    let one = load(&t, &opts);
    let mut st = one.new_state().unwrap();
    let mut reference = vec![];
    let mut cb = |_p: usize, r: &[f32]| {
        reference = r.to_vec();
        Ok(())
    };
    one.forward(&mut st, &ids, HeadMode::AllLogits, Some(&mut cb)).unwrap();
    let clean = rel_l2(&tp2_logits(&t, &opts, &ids), &reference);
    unsafe { std::env::set_var("INTARSIA_TP_TEST_BUG", "scale:1.01") };
    let bug = rel_l2(&tp2_logits(&t, &opts, &ids), &reference);
    unsafe { std::env::remove_var("INTARSIA_TP_TEST_BUG") };
    eprintln!("tp2 vs tp1, f32: summation order rel_l2 {clean:.3e}; scale:1.01 on rank 1 rel_l2 {bug:.3e}");
    assert!(clean < 1e-5, "clean tp=2 rel_l2 {clean}");
    assert!(bug > 100.0 * 1e-5, "a 1% partial-scale bug must sit far above the 1e-5 bound (rel_l2 {bug})");
}

#[test]
fn test_bug_moves_the_logits_and_scale_one_does_not() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let t = tiny_weights(DType::F32, 5);
    let opts = LoadOptions { kv_dtype: Some(DType::F32), ..Default::default() };
    let ids: Vec<u32> = (0..9).map(|i| ((i * 37 + 11) % V) as u32).collect();
    // SAFETY: the tests of this binary hold ENV, nothing else reads the environment concurrently.
    unsafe { std::env::remove_var("INTARSIA_TP_TEST_BUG") };
    let clean = tp2_logits(&t, &opts, &ids);
    unsafe { std::env::set_var("INTARSIA_TP_TEST_BUG", "scale:1.0") };
    let one = tp2_logits(&t, &opts, &ids);
    assert_eq!(clean, one, "scale:1.0 is the identity");
    unsafe { std::env::set_var("INTARSIA_TP_TEST_BUG", "scale:1.01") };
    let bug = tp2_logits(&t, &opts, &ids);
    let d = rel_l2(&bug, &clean);
    assert!(d > 1e-4, "a 1% scale of one rank's partial must move the logits (rel_l2 {d})");
    unsafe { std::env::set_var("INTARSIA_TP_TEST_BUG", "scale:1.01@0") };
    let bug0 = tp2_logits(&t, &opts, &ids);
    assert!(rel_l2(&bug0, &clean) > 1e-4);
    unsafe { std::env::set_var("INTARSIA_TP_TEST_BUG", "bogus") };
    let backends: Vec<Arc<dyn Backend>> = (0..2).map(|_| CpuBackend::arc()).collect();
    let comms = HostCommGroup::new(backends.clone());
    let r = Model::load_tp(backends, comms, &recipe(), &dims(), &t.weights, &opts, Arc::new(Telemetry::new()));
    assert!(r.is_err(), "a malformed INTARSIA_TP_TEST_BUG is an error, not a silent no-op");
    unsafe { std::env::remove_var("INTARSIA_TP_TEST_BUG") };
}

fn tp1_logits(t: &Tiny, opts: &LoadOptions, ids: &[u32]) -> Vec<f32> {
    let one = load(t, opts);
    let mut st = one.new_state().unwrap();
    let mut reference = vec![];
    let mut cb = |_p: usize, r: &[f32]| {
        reference = r.to_vec();
        Ok(())
    };
    one.forward(&mut st, ids, HeadMode::AllLogits, Some(&mut cb)).unwrap();
    reference
}

/// Direct check of the tensor-parallel partial sums, independent of KL: tp=2 against tp=1 on the f32 tiny model, with
/// a tolerance DERIVED from the fp32 reduction-order bound instead of tuned, and the 0.3% and 0.1% partial-scale
/// mutants (`INTARSIA_TP_TEST_BUG=scale:1.003` / `scale:1.001`, rank 1) required to land above it.
///
/// Derivation. Only the row-parallel projections change their summation order under tp=2 (o_proj, K = NH*HD = 64, and
/// down_proj, K = I = 48; q/k/v/gate/up/head are column-parallel, each output element is summed by one rank over the
/// same K terms in the same order as tp=1; attention is per head). For one K-term f32 dot product every summation order
/// (recursive, pairwise, blocked, "two halves then one add") satisfies |fl(s) - s| <= gamma_{K-1} * sum_i |a_i b_i|,
/// gamma_n = n*u / (1 - n*u), u = 2^-24, so two orders differ by at most 2 * gamma_{K-1} * sum|a_i b_i| (Higham,
/// Accuracy and Stability of Numerical Algorithms, 2nd ed., sec. 3.1 and 4.2). tp=2 sums the halves K/2 each and adds
/// the two f32 partials in the all-reduce: that is one of the orders covered. Per site this is a relative perturbation
/// of the projection output of at most 2 * gamma_{K-1} * kappa, kappa = sum|a_i b_i| / |sum a_i b_i|. There are
/// 2 sites per layer * LAYERS = 6 of them. Taking kappa * (the Lipschitz amplification of the rest of the network) = 1
/// (no cancellation, non-expansive residual/RMSNorm path) and adding the sites, the logits satisfy
/// rel_l2(tp2, tp1) <= 6 * 2 * gamma_63 = 4.5e-5. Over 8 seeds the measured clean distance is >=78x below that, the 0.3%
/// mutant >=24x above it and the 0.1% mutant >=8x above it; every margin is printed.
///
/// What the bound does not cover: (1) kappa and the amplification through attention softmax / SwiGLU / RoPE are not
/// bounded here, they are assumed 1 (the ~80x clean margin is the evidence the assumption has room on this model, not a
/// proof); (2) it is an f32 model on the host communicator: bf16 weights/KV, GPU kernels and the bf16 round of the
/// all-reduce output have other (larger) noise and are checked in
/// crates/intarsia-backend-hip/tests/tp_partial_sum.rs; (3) a mutant is only detected when its effect exceeds the bound: a
/// partial scale of 1+e moves the logits by ~(0.4..1.8)*e here, so e below ~1.3e-4 would sit under the bound (the
/// 0.1% mutant, e = 1e-3, has 8x room on the weakest seed).
#[test]
fn f32_tp2_vs_tp1_direct_bound_kills_the_point_three_percent_partial_scale_mutant() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let u = 2f64.powi(-24);
    let gamma = |n: f64| n * u / (1.0 - n * u);
    let k_max = (NH * HD).max(I) as f64; // 64, the longest row-parallel dot product
    let sites = 2 * LAYERS; // o_proj + down_proj per layer
    let bound = sites as f64 * 2.0 * gamma(k_max - 1.0);
    eprintln!("derived bound: {sites} sites * 2 * gamma_{} = {bound:.3e}", k_max - 1.0);
    let opts = LoadOptions { kv_dtype: Some(DType::F32), ..Default::default() };
    let (mut worst_clean, mut best_3, mut best_1, mut best_3_r0) = (0f64, f64::INFINITY, f64::INFINITY, f64::INFINITY);
    for seed in [5u64, 6, 7, 8, 9, 10, 11, 12] {
        let t = tiny_weights(DType::F32, seed);
        let ids: Vec<u32> = (0..9).map(|i| ((i * 37 + 11 + seed as usize) % V) as u32).collect();
        unsafe { std::env::remove_var("INTARSIA_TP_TEST_BUG") };
        let one = tp1_logits(&t, &opts, &ids);
        assert!(one.iter().any(|v| v.abs() > 1e-3), "seed {seed}: tp1 logits are all ~0 (vacuous)");
        let clean = rel_l2(&tp2_logits(&t, &opts, &ids), &one);
        let run = |spec: &str| {
            unsafe { std::env::set_var("INTARSIA_TP_TEST_BUG", spec) };
            let d = rel_l2(&tp2_logits(&t, &opts, &ids), &one);
            unsafe { std::env::remove_var("INTARSIA_TP_TEST_BUG") };
            d
        };
        let (b3, b1, b3r0) = (run("scale:1.003"), run("scale:1.001"), run("scale:1.003@0"));
        eprintln!(
            "seed {seed}: clean {clean:.3e} ({:.0}x under the bound); scale:1.003 {b3:.3e} ({:.1}x over, {:.0}x clean); scale:1.001 {b1:.3e} ({:.1}x over, {:.0}x clean); scale:1.003@0 {b3r0:.3e}",
            bound / clean,
            b3 / bound,
            b3 / clean,
            b1 / bound,
            b1 / clean
        );
        worst_clean = worst_clean.max(clean);
        best_3 = best_3.min(b3);
        best_1 = best_1.min(b1);
        best_3_r0 = best_3_r0.min(b3r0);
    }
    eprintln!(
        "worst clean {worst_clean:.3e} (bound/clean {:.0}x); weakest scale:1.003 {best_3:.3e} ({:.1}x bound); weakest scale:1.001 {best_1:.3e} ({:.1}x bound)",
        bound / worst_clean,
        best_3 / bound,
        best_1 / bound
    );
    assert!(worst_clean <= bound, "clean tp=2 vs tp=1 rel_l2 {worst_clean:.3e} exceeds the derived bound {bound:.3e}");
    assert!(worst_clean * 10.0 <= bound, "the derived bound must have >=10x room over the clean noise ({worst_clean:.3e} vs {bound:.3e}): the kappa*amplification = 1 assumption is suspect");
    assert!(best_3 > bound, "scale:1.003 must exceed the derived bound ({best_3:.3e} <= {bound:.3e})");
    assert!(best_1 > bound, "scale:1.001 must exceed the derived bound ({best_1:.3e} <= {bound:.3e})");
    assert!(best_3_r0 > bound, "scale:1.003@0 must exceed the derived bound ({best_3_r0:.3e} <= {bound:.3e})");
}
