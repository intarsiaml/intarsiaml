//! Opt-in quantised lm_head (`INTARSIA_LMHEAD_QUANT`, lane lmhead-fp8) on the cpu backend, the reference kernels of the fp8 and
//! mxfp4 formats: it loads (tied and untied heads), is a small perturbation of the bf16 head, is bitwise the same split over two
//! ranks as replicated (vocab-parallel head and the draft head's row prefix come from the one quantised tensor), and does
//! nothing while the variable is unset. The variable is process-wide: every test takes `ENV_LOCK`.

mod common;
use common::*;
use intarsia_backend_cpu::CpuBackend;
use intarsia_core::comm::HostCommGroup;
use intarsia_core::model::{HeadMode, LoadOptions, Model, StepOut};
use intarsia_core::recipe::ModelDims;
use intarsia_core::tensor::HostTensor;
use intarsia_core::{Backend, DType, Telemetry};
use std::sync::{Arc, Mutex};

static ENV_LOCK: Mutex<()> = Mutex::new(());
const ENV: &str = "INTARSIA_LMHEAD_QUANT";

fn ids(n: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * 37 + 11) % V) as u32).collect()
}

fn setup(untied: bool) -> (Tiny, ModelDims) {
    let mut t = tiny_weights(DType::Bf16, 31);
    let mut d = dims();
    if untied {
        let (shape, vals) = t.raw["model.embed_tokens.weight"].clone();
        let vals: Vec<f32> = vals.iter().rev().map(|x| x * 0.7).collect();
        t.weights.insert("lm_head.weight", HostTensor::from_f32_as(DType::Bf16, &shape, &vals));
        d.tie_embeddings = false;
    }
    (t, d)
}

fn load_tp1(t: &Tiny, d: &ModelDims, q: Option<&str>) -> Model {
    match q {
        Some(v) => std::env::set_var(ENV, v),
        None => std::env::remove_var(ENV),
    }
    let m = Model::load(CpuBackend::arc(), &recipe(), d, &t.weights, &LoadOptions::default(), Arc::new(Telemetry::new())).unwrap();
    std::env::remove_var(ENV);
    m
}

fn load_tp2(t: &Tiny, d: &ModelDims, q: Option<&str>, tp_vocab: bool) -> Model {
    match q {
        Some(v) => std::env::set_var(ENV, v),
        None => std::env::remove_var(ENV),
    }
    let backends: Vec<Arc<dyn Backend>> = (0..2).map(|_| CpuBackend::arc()).collect();
    let comms = HostCommGroup::new(backends.clone());
    let m = Model::load_tp(backends, comms, &recipe(), d, &t.weights, &LoadOptions { tp_vocab, ..Default::default() }, Arc::new(Telemetry::new())).unwrap();
    std::env::remove_var(ENV);
    m
}

/// All rows' logits of a prompt, then a greedy decode of 6 tokens.
fn outputs(m: &Model) -> (Vec<Vec<f32>>, Vec<u32>) {
    let mut rows = Vec::new();
    let mut st = m.new_state().unwrap();
    let mut cb = |_p: usize, r: &[f32]| {
        rows.push(r.to_vec());
        Ok(())
    };
    m.forward(&mut st, &ids(19), HeadMode::AllLogits, Some(&mut cb)).unwrap();
    let mut st = m.new_state().unwrap();
    let mut tok = match m.forward(&mut st, &ids(9), HeadMode::LastArgmax, None).unwrap() {
        StepOut::Token(t) => t,
        _ => unreachable!(),
    };
    let mut toks = vec![tok];
    for _ in 0..6 {
        tok = match m.forward(&mut st, &[tok], HeadMode::LastArgmax, None).unwrap() {
            StepOut::Token(t) => t,
            _ => unreachable!(),
        };
        toks.push(tok);
    }
    (rows, toks)
}

fn bits(r: &[Vec<f32>]) -> Vec<Vec<u32>> {
    r.iter().map(|v| v.iter().map(|x| x.to_bits()).collect()).collect()
}

/// rms of (a - b) over rms of a, all rows
fn rel(a: &[Vec<f32>], b: &[Vec<f32>]) -> f64 {
    let (mut n, mut d) = (0f64, 0f64);
    for (x, y) in a.iter().flatten().zip(b.iter().flatten()) {
        n += ((x - y) as f64).powi(2);
        d += (*x as f64).powi(2);
    }
    (n / d).sqrt()
}

#[test]
fn quantised_heads_load_and_stay_close_to_bf16() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    for untied in [false, true] {
        let (t, d) = setup(untied);
        let base = load_tp1(&t, &d, None);
        let (r0, _) = outputs(&base);
        for (q, tol, bytes) in [("fp8", 0.06, V * H + 8), ("mxfp4", 0.35, V * H / 2 + V * H / 32 + 4)] {
            let m = load_tp1(&t, &d, Some(q));
            let (r, toks) = outputs(&m);
            assert_eq!(toks.len(), 7);
            let e = rel(&r0, &r);
            assert!(e > 0.0 && e < tol, "{q} untied={untied}: relative rms logit error {e}");
            // untied: the quantised copy replaces the bf16 head (smaller); tied: the embeddings stay and the copy is added
            let head_bf16 = V * H * 2;
            let delta = m.weights_bytes as i64 - base.weights_bytes as i64;
            let expect = if untied { bytes as i64 - head_bf16 as i64 } else { bytes as i64 };
            assert_eq!(delta, expect, "{q} untied={untied}: resident weight bytes");
        }
    }
}

#[test]
fn unset_and_off_are_bitwise_the_bf16_head() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (t, d) = setup(true);
    let a = outputs(&load_tp1(&t, &d, None));
    for off in ["off", "0", ""] {
        let b = outputs(&load_tp1(&t, &d, Some(off)));
        assert_eq!(bits(&a.0), bits(&b.0), "{off:?}");
        assert_eq!(a.1, b.1);
    }
}

#[test]
fn split_quantised_head_is_bitwise_the_replicated_quantised_head() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    for untied in [false, true] {
        let (t, d) = setup(untied);
        for q in ["fp8", "mxfp4"] {
            let rep = load_tp2(&t, &d, Some(q), false);
            let split = load_tp2(&t, &d, Some(q), true);
            assert!(!rep.vocab_parallel() && split.vocab_parallel());
            let (a, b) = (outputs(&split), outputs(&rep));
            assert_eq!(bits(&a.0), bits(&b.0), "{q} untied={untied}: split vs replicated quantised head");
            assert_eq!(a.1, b.1);
            // and it is a real quantised head, not the bf16 one
            let plain = outputs(&load_tp2(&t, &d, None, true));
            assert_ne!(bits(&a.0), bits(&plain.0), "{q} untied={untied}: the variable changed nothing");
        }
    }
}

#[test]
fn a_bad_value_is_a_load_error() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (t, d) = setup(true);
    std::env::set_var(ENV, "fp4");
    let r = Model::load(CpuBackend::arc(), &recipe(), &d, &t.weights, &LoadOptions::default(), Arc::new(Telemetry::new()));
    std::env::remove_var(ENV);
    assert!(r.err().expect("must not load").to_string().contains("fp8 | mxfp4"));
}
