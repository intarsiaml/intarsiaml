//! Tensor parallelism on the cpu backend: two shards joined by the host-staged communicator must compute what one
//! unsharded model computes (summation order aside), hold only their slice of the weights, and plan per-rank shapes.
//! No GPU needed: this is the Mac gate for the sharding logic; `intarsia-backend-hip` tests the device collective.

mod common;
use common::*;
use intarsia_backend_cpu::CpuBackend;
use intarsia_core::comm::{Comm, HostCommGroup};
use intarsia_core::model::{HeadMode, LoadOptions, Model, StepOut};
use intarsia_core::{Backend, DType, Telemetry};
use std::sync::Arc;

fn ids(n: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * 37 + 11) % V) as u32).collect()
}

fn load_tp(t: &Tiny, world: usize, opts: &LoadOptions) -> Model {
    let backends: Vec<Arc<dyn Backend>> = (0..world).map(|_| CpuBackend::arc()).collect();
    let comms = HostCommGroup::new(backends.clone());
    Model::load_tp(backends, comms, &recipe(), &dims(), &t.weights, opts, Arc::new(Telemetry::new())).unwrap()
}

fn all_logits(m: &Model, ids: &[u32]) -> Vec<Vec<f32>> {
    let mut st = m.new_state().unwrap();
    let mut rows = vec![];
    let mut cb = |_p: usize, r: &[f32]| {
        rows.push(r.to_vec());
        Ok(())
    };
    m.forward(&mut st, ids, HeadMode::AllLogits, Some(&mut cb)).unwrap();
    rows
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den).sqrt()
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
}

#[test]
fn tp2_matches_tp1_prefill_and_decode_f32() {
    let t = tiny_weights(DType::F32, 5);
    let opts = LoadOptions { kv_dtype: Some(DType::F32), ..Default::default() };
    let one = load(&t, &opts);
    let two = load_tp(&t, 2, &opts);
    assert_eq!(two.tp_world(), 2);
    let input = ids(11);
    let (a, b) = (all_logits(&two, &input), all_logits(&one, &input));
    assert_eq!(a.len(), b.len());
    for (p, (x, y)) in a.iter().zip(&b).enumerate() {
        assert!(rel_l2(x, y) < 1e-5, "pos {p}: rel_l2 {}", rel_l2(x, y));
        assert_eq!(argmax(x), argmax(y), "pos {p}");
    }
}

#[test]
fn tp2_greedy_decode_follows_tp1_in_bf16_with_chunks_and_truncate() {
    let t = tiny_weights(DType::Bf16, 6);
    let opts = LoadOptions { prefill_chunk: 4, ..Default::default() };
    let one = load(&t, &opts);
    let two = load_tp(&t, 2, &opts);
    let run = |m: &Model| -> Vec<u32> {
        let mut st = m.new_state().unwrap();
        let mut tok = match m.forward(&mut st, &ids(9), HeadMode::LastArgmax, None).unwrap() {
            StepOut::Token(t) => t,
            _ => unreachable!(),
        };
        let mut out = vec![tok];
        for _ in 0..6 {
            tok = match m.forward(&mut st, &[tok], HeadMode::LastArgmax, None).unwrap() {
                StepOut::Token(t) => t,
                _ => unreachable!(),
            };
            out.push(tok);
        }
        // prefix reuse: roll back to 9 tokens and decode the same continuation again
        st.truncate(9, &*m.backend).unwrap();
        assert_eq!(st.len, 9);
        for p in &st.peers {
            assert_eq!(p.st.len, 9, "peers follow truncate");
        }
        let again = match m.forward(&mut st, &[out[0]], HeadMode::LastArgmax, None).unwrap() {
            StepOut::Token(t) => t,
            _ => unreachable!(),
        };
        assert_eq!(again, out[1]);
        out
    };
    // bf16 partial sums round differently from one big matmul, so near-ties may flip a token; the tiny model's
    // logits are well separated, so the whole sequence should agree
    assert_eq!(run(&two), run(&one));
}

#[test]
fn shards_hold_only_their_slice_and_the_plan_shows_per_rank_shapes() {
    let t = tiny_weights(DType::Bf16, 7);
    let opts = LoadOptions::default();
    let one = load(&t, &opts);
    let two = load_tp(&t, 2, &opts);
    // per-rank dims
    assert_eq!((two.ldims.n_heads, two.ldims.n_kv_heads, two.ldims.intermediate), (NH / 2, NKV / 2, I / 2));
    assert_eq!((two.dims.n_heads, two.dims.n_kv_heads, two.dims.intermediate), (NH, NKV, I));
    // each rank holds strictly less than the whole model
    let (r0, r1) = (two.rank_model(0).unwrap().weights_bytes, two.rank_model(1).unwrap().weights_bytes);
    let embed = (V * H * 2) as u64;
    assert!(r0 < one.weights_bytes && r1 < one.weights_bytes, "{r0} {r1} vs {}", one.weights_bytes);
    // everything except the replicated embedding / norms is split exactly in two: the ranks together hold one copy of
    // the sharded weights plus two copies of the replicated ones
    let replicated = embed + (H * 2) as u64 + norms_per_rank();
    assert_eq!(r0 + r1, one.weights_bytes + replicated);
    // the plan: all-reduce count and the per-rank matmul shapes
    assert_eq!(two.resolved.comm_calls_per_decode_step, 2 * LAYERS);
    assert_eq!(two.resolved.comm_bytes_per_decode_step, 2 * LAYERS * H * 2);
    let plan = intarsia_core::plan::plan_json(&two, "tiny");
    assert_eq!(plan["tp"]["world"], 2);
    assert_eq!(plan["tp"]["per_rank"]["n_heads"], NH / 2);
    assert_eq!(plan["tp"]["all_reduce_calls_per_decode_step"], 2 * LAYERS);
    let probe = two.dry_run(intarsia_core::plan::Phase::Decode, 1, 0).unwrap();
    let mm: Vec<(usize, usize)> = probe
        .calls
        .iter()
        .filter(|c| c.call.op == intarsia_core::Op::Matmul)
        .map(|c| (c.call.inputs[1].shape[0], c.call.inputs[1].shape[1]))
        .collect();
    // q [nh/2*hd, H], k/v [nkv/2*hd, H], o [H, nh/2*hd], gate/up [I/2, H], down [H, I/2]
    for want in [((NH / 2) * HD, H), ((NKV / 2) * HD, H), (H, (NH / 2) * HD), (I / 2, H), (H, I / 2)] {
        assert!(mm.contains(&want), "no matmul with weight {want:?} in {mm:?}");
    }
    // KV is split by heads too
    let st = two.new_state().unwrap();
    assert_eq!(st.peers.len(), 1);
    assert_eq!(st.total_bytes() * 2, one.new_state().unwrap().total_bytes());
}

fn norms_per_rank() -> u64 {
    // per layer: pre-attn norm, q_norm, k_norm, pre-ffn norm (bf16), replicated on both ranks
    (LAYERS * (2 * H + 2 * HD) * 2) as u64
}

#[test]
fn indivisible_degree_and_unshardable_blocks_are_refused() {
    let t = tiny_weights(DType::Bf16, 8);
    let backends: Vec<Arc<dyn Backend>> = (0..3).map(|_| CpuBackend::arc()).collect();
    let comms = HostCommGroup::new(backends.clone());
    let e = Model::load_tp(backends, comms, &recipe(), &dims(), &t.weights, &LoadOptions::default(), Arc::new(Telemetry::new()))
        .err()
        .expect("3-way must fail: 4 heads, 2 kv heads");
    assert!(e.to_string().contains("divisible"), "{e}");
}

#[test]
fn all_gather_concatenates_in_rank_order_and_a_failed_rank_releases_the_others() {
    let backends: Vec<Arc<dyn Backend>> = (0..2).map(|_| CpuBackend::arc()).collect();
    let comms: Vec<Arc<dyn Comm>> = HostCommGroup::new(backends.clone());
    let mk = |b: &Arc<dyn Backend>, vals: &[f32]| {
        let mut bytes = vec![0u8; vals.len() * 4];
        intarsia_core::dtype::encode_from_f32(DType::F32, vals, &mut bytes);
        let buf = intarsia_core::backend::upload_new(&**b, &bytes).unwrap();
        intarsia_core::TensorRef::new(buf, DType::F32, &[vals.len()])
    };
    let read = |b: &Arc<dyn Backend>, t: &intarsia_core::TensorRef| {
        let mut bytes = vec![0u8; t.nbytes()];
        b.download(&t.buf, t.offset, &mut bytes, 0).unwrap();
        let mut out = vec![0f32; t.numel()];
        intarsia_core::dtype::decode_to_f32(DType::F32, &bytes, &mut out);
        out
    };
    let sends: Vec<_> = (0..2).map(|r| mk(&backends[r], &[r as f32 * 10.0 + 1.0, r as f32 * 10.0 + 2.0])).collect();
    let recvs: Vec<_> = (0..2).map(|r| mk(&backends[r], &[0.0; 4])).collect();
    std::thread::scope(|sc| {
        for r in 0..2 {
            let (c, s, rv) = (&comms[r], &sends[r], &recvs[r]);
            sc.spawn(move || c.all_gather(s, rv, 0).unwrap());
        }
    });
    for r in 0..2 {
        assert_eq!(read(&backends[r], &recvs[r]), vec![1.0, 2.0, 11.0, 12.0]);
    }
    // all_reduce in place
    std::thread::scope(|sc| {
        for r in 0..2 {
            let (c, s) = (&comms[r], &sends[r]);
            sc.spawn(move || c.all_reduce_sum(s, 0).unwrap());
        }
    });
    for r in 0..2 {
        assert_eq!(read(&backends[r], &sends[r]), vec![12.0, 14.0]);
    }
    assert_eq!(comms[0].stats().calls, 2);
    // one rank aborts: the other gets an error instead of waiting
    let c1 = comms[1].clone();
    let h = std::thread::spawn({
        let c0 = comms[0].clone();
        let s = sends[0].clone();
        move || c0.all_reduce_sum(&s, 0)
    });
    std::thread::sleep(std::time::Duration::from_millis(50));
    c1.abort();
    assert!(h.join().unwrap().is_err());
}
