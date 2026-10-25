//! Vocab-parallel lm_head (`LoadOptions::tp_vocab`) on the cpu backend with the host communicator: each rank computes half of
//! the vocabulary's logits, one all-gather puts them in rank 0's logits, and everything that reads them (AllLogits rows, the
//! last row's logits, greedy ids, prefill chunks, decode steps, prefix rollback) is bit for bit what the replicated head
//! (rank 0 alone) gives. The comparison is tp=2 against tp=2 with only the head changed, so no summation order differs.
//! A shard offset off by one row must fail these tests (checked by hand with the mutation in the lane's report).

mod common;
use common::*;
use intarsia_backend_cpu::CpuBackend;
use intarsia_core::comm::HostCommGroup;
use intarsia_core::model::{HeadMode, LoadOptions, Model, StepOut};
use intarsia_core::recipe::ModelDims;
use intarsia_core::tensor::HostTensor;
use intarsia_core::{Backend, DType, Telemetry};
use std::sync::Arc;

fn ids(n: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * 37 + 11) % V) as u32).collect()
}

/// `untied`: the checkpoint carries its own `lm_head.weight` (loaded with the row-sharded loader); otherwise the head is the
/// embedding table (each rank's head is a row range of it).
fn setup(dt: DType, seed: u64, untied: bool) -> (Tiny, ModelDims) {
    let mut t = tiny_weights(dt, seed);
    let mut d = dims();
    if untied {
        let (shape, vals) = t.raw["model.embed_tokens.weight"].clone();
        // a different matrix from the embeddings, so a head read from the wrong tensor would show
        let vals: Vec<f32> = vals.iter().rev().map(|x| x * 0.7).collect();
        t.weights.insert("lm_head.weight", HostTensor::from_f32_as(dt, &shape, &vals));
        d.tie_embeddings = false;
    }
    (t, d)
}

fn load_tp(t: &Tiny, d: &ModelDims, opts: &LoadOptions) -> Model {
    let backends: Vec<Arc<dyn Backend>> = (0..2).map(|_| CpuBackend::arc()).collect();
    let comms = HostCommGroup::new(backends.clone());
    Model::load_tp(backends, comms, &recipe(), d, &t.weights, opts, Arc::new(Telemetry::new())).unwrap()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Every output the head feeds, as bits: all rows of several prompts (chunking, 16-row head groups, 1-row tails), the last
/// row's logits, and a greedy decode with a rollback in the middle.
fn outputs(m: &Model) -> Vec<Vec<u32>> {
    let mut out = Vec::new();
    for n in [1usize, 2, 3, 15, 16, 17, 18, 33, 40] {
        let mut st = m.new_state().unwrap();
        let mut cb = |_p: usize, r: &[f32]| {
            out.push(bits(r));
            Ok(())
        };
        m.forward(&mut st, &ids(n), HeadMode::AllLogits, Some(&mut cb)).unwrap();
    }
    let mut st = m.new_state().unwrap();
    match m.forward(&mut st, &ids(11), HeadMode::LastLogits, None).unwrap() {
        StepOut::Logits(l) => out.push(bits(&l)),
        _ => unreachable!(),
    }
    let mut st = m.new_state().unwrap();
    let mut tok = match m.forward(&mut st, &ids(9), HeadMode::LastArgmax, None).unwrap() {
        StepOut::Token(t) => t,
        _ => unreachable!(),
    };
    let mut toks = vec![tok];
    for step in 0..8 {
        tok = match m.forward(&mut st, &[tok], HeadMode::LastArgmax, None).unwrap() {
            StepOut::Token(t) => t,
            _ => unreachable!(),
        };
        toks.push(tok);
        if step == 3 {
            // prefix reuse: roll back and decode the same position again, this time for its logits
            st.truncate(11, &*m.backend).unwrap();
            if let StepOut::Logits(l) = m.forward(&mut st, &[toks[0]], HeadMode::LastLogits, None).unwrap() {
                out.push(bits(&l));
            }
        }
    }
    out.push(toks);
    out
}

fn check(dt: DType, seed: u64, untied: bool, chunk: usize) {
    let (t, d) = setup(dt, seed, untied);
    let base = LoadOptions { prefill_chunk: chunk, kv_dtype: Some(dt), ..Default::default() };
    let replicated = load_tp(&t, &d, &base);
    let split = load_tp(&t, &d, &LoadOptions { tp_vocab: true, ..base.clone() });
    assert!(!replicated.vocab_parallel());
    assert!(split.vocab_parallel(), "the head must be split for this test to mean anything");
    assert!(split.rank_model(1).unwrap().vocab_parallel());
    let (a, b) = (outputs(&split), outputs(&replicated));
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert!(x == y, "output {i} differs between the vocab-parallel and the replicated head ({dt:?}, untied={untied}, chunk={chunk})");
    }
}

#[test]
fn split_head_is_bitwise_the_replicated_head_tied_f32() {
    check(DType::F32, 21, false, 256);
}

#[test]
fn split_head_is_bitwise_the_replicated_head_tied_bf16_in_small_chunks() {
    check(DType::Bf16, 22, false, 8);
}

#[test]
fn split_head_is_bitwise_the_replicated_head_untied_f32() {
    check(DType::F32, 23, true, 256);
}

#[test]
fn split_head_is_bitwise_the_replicated_head_untied_bf16_in_small_chunks() {
    check(DType::Bf16, 24, true, 5);
}

/// Memory and plan: an untied head's bytes move from rank 0 to half on each rank; a tied head is a view of the embedding
/// table (nothing new); the plan counts the gather and says where the head lives.
#[test]
fn split_head_halves_the_head_weights_per_rank_and_the_plan_counts_the_gather() {
    let esz = 2u64;
    let head = (V * H) as u64 * esz;
    for untied in [false, true] {
        let (t, d) = setup(DType::Bf16, 25, untied);
        let base = LoadOptions::default();
        let rep = load_tp(&t, &d, &base);
        let split = load_tp(&t, &d, &LoadOptions { tp_vocab: true, ..base });
        let w = |m: &Model, r: usize| m.rank_model(r).unwrap().weights_bytes;
        if untied {
            assert_eq!(w(&rep, 0), w(&split, 0) + head / 2, "rank 0 gives up half of the head");
            assert_eq!(w(&split, 1), w(&rep, 1) + head / 2, "rank 1 takes the other half");
        } else {
            assert_eq!((w(&rep, 0), w(&rep, 1)), (w(&split, 0), w(&split, 1)), "a tied head is a view of the embeddings");
        }
        let (pr, ps) = (intarsia_core::plan::plan_json(&rep, "t"), intarsia_core::plan::plan_json(&split, "t"));
        let calls = |p: &serde_json::Value| p["tp"]["all_reduce_calls_per_decode_step"].as_u64();
        assert_eq!(calls(&ps), calls(&pr).map(|c| c + 1), "one gather more per decode step\n{ps:#}");
        assert!(ps.to_string().contains("vocab-parallel"));
        assert!(pr.to_string().contains("replicated on rank 0 only"));
    }
}
