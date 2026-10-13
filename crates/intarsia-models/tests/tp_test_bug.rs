//! `INTARSIA_TP_TEST_BUG=scale:F`: the deliberate TP bug the accuracy gate's sensitivity is measured with
//! (docs/accuracy.md, "TP self-golden budget"). Its own test binary because it sets a process environment variable
//! before the models load; this file holds a single test so nothing else reads the variable concurrently.

mod common;
use common::*;
use intarsia_backend_cpu::CpuBackend;
use intarsia_core::comm::HostCommGroup;
use intarsia_core::model::{HeadMode, LoadOptions, Model};
use intarsia_core::{Backend, DType, Telemetry};
use std::sync::Arc;

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

#[test]
fn test_bug_moves_the_logits_and_scale_one_does_not() {
    let t = tiny_weights(DType::F32, 5);
    let opts = LoadOptions { kv_dtype: Some(DType::F32), ..Default::default() };
    let ids: Vec<u32> = (0..9).map(|i| ((i * 37 + 11) % V) as u32).collect();
    // SAFETY: the only test of this binary, nothing reads the environment concurrently.
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
