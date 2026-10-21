//! Device-side token feedback (lane T1, `model/feedback.rs`): the pipelined greedy loop must be indistinguishable from
//! the plain one except for one discarded step after a stop. Every request runs on two engines of the same weights, one
//! with feedback, one without; tokens, text, the `Done` event, the KV length left behind and what the next request reuses
//! must agree. The cpu backend runs launches synchronously, so this checks the engine's bookkeeping (the stop one step
//! late, the rollback, the prefix-reuse key); the Metal gates check the overlap itself.

mod common;
use common::*;
use intarsia_core::model::LoadOptions;
use intarsia_core::{DType, Engine, EngineMeta, FinishReason, GenEvent, GenRequest, LocalEngine, Telemetry};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

const EOS: u32 = 299;

fn engine(seed: u64, feedback: bool, max_seq: usize) -> LocalEngine {
    let t = tiny_weights(DType::Bf16, seed);
    let tel = Arc::new(Telemetry::new());
    let m = intarsia_core::model::Model::load(
        intarsia_backend_cpu::CpuBackend::arc(),
        &recipe(),
        &dims(),
        &t.weights,
        &LoadOptions { max_seq, prefill_chunk: 8, ..Default::default() },
        tel.clone(),
    )
    .unwrap();
    assert!(!m.token_feedback(), "off by default on the cpu backend");
    assert_eq!(m.set_token_feedback(feedback), feedback);
    LocalEngine::new(m, Arc::new(ByteCodec), EngineMeta { id: "tiny".into(), eos_ids: vec![EOS], ..Default::default() }, tel)
        .unwrap()
}

#[derive(Debug, PartialEq)]
struct Run {
    ids: Vec<u32>,
    text: String,
    reason: FinishReason,
    completion: usize,
    cached: usize,
    stop_sequence: Option<String>,
    /// KV rows the engine holds after the request (what the next one can reuse).
    kv_after: u64,
}

fn run(e: &LocalEngine, req: GenRequest) -> (Run, u64) {
    let steps0 = e.telemetry().steps_total;
    let (mut ids, mut text, mut done) = (vec![], String::new(), None);
    e.generate(req, &mut |ev| match ev {
        GenEvent::Token { id, text: t } => {
            ids.push(id);
            text.push_str(&t);
        }
        GenEvent::Done { finish_reason, completion_tokens, cached_tokens, stop_sequence, .. } => {
            done = Some((finish_reason, completion_tokens, cached_tokens, stop_sequence))
        }
        GenEvent::Error { message } => panic!("engine error: {message}"),
    })
    .unwrap();
    let (reason, completion, cached, stop_sequence) = done.unwrap();
    // the held-back tail (partial UTF-8) is released as one more event under the last id
    ids.truncate(completion);
    let t = e.telemetry();
    (Run { ids, text, reason, completion, cached, stop_sequence, kv_after: t.kv_used_tokens }, t.steps_total - steps0)
}

/// A plain run of `r` on both engines (they must see the same request sequence: prefix reuse depends on it).
fn probe(base: &LocalEngine, fb: &LocalEngine, r: GenRequest) -> Run {
    let (a, _) = run(base, r.clone());
    let (b, _) = run(fb, r);
    assert_eq!(a, b, "probe");
    a
}

fn bytes(s: &str) -> Vec<u32> {
    s.bytes().map(|b| b as u32).collect()
}

fn req(prompt: &[u32], n: usize) -> GenRequest {
    GenRequest { prompt: prompt.to_vec(), max_tokens: n, ignore_eos: true, ..Default::default() }
}

/// Run `r` on both engines, then two follow-ups that reuse its KV (the reply with and without the final token, then
/// more text); everything must match. Returns the plain run and how many more steps the feedback engine ran (0 or 1).
fn same(base: &LocalEngine, fb: &LocalEngine, r: GenRequest, what: &str) -> (Run, u64) {
    let (a, sa) = run(base, r.clone());
    let (b, sb) = run(fb, r.clone());
    assert_eq!(a, b, "{what}");
    assert_eq!(a.kv_after as usize, r.prompt.len() + a.completion - 1, "{what}: every emitted token but the last is fed");
    assert!(sb == sa || sb == sa + 1, "{what}: feedback runs at most one discarded step ({sa} vs {sb})");
    for with_last in [true, false] {
        let mut p = r.prompt.clone();
        p.extend(&a.ids);
        if with_last && a.completion > a.ids.len() && r.stop_ids.len() == 1 {
            // the stop id was counted but not streamed; a chat follow-up carries it
            p.push(r.stop_ids[0]);
        }
        p.extend(bytes(" and more"));
        let follow = req(&p, 5);
        let (fa, _) = run(base, follow.clone());
        let (fb2, _) = run(fb, follow);
        assert_eq!(fa, fb2, "{what}: follow-up (with_last {with_last})");
    }
    (a, sb - sa)
}

#[test]
fn forced_length_matches_and_never_speculates_past_max_tokens() {
    let (base, fb) = (engine(11, false, 64), engine(11, true, 64));
    let p = bytes("hello world");
    for n in [1, 2, 3, 4, 17] {
        let (a, extra) = same(&base, &fb, req(&p, n), &format!("max_tokens {n}"));
        assert_eq!(a.reason, FinishReason::Length);
        assert_eq!(a.ids.len(), n);
        assert_eq!(extra, 0, "max_tokens {n}: no step past the last token");
    }
    // the context window caps the request: 60 prompt tokens in 64 leave 4
    let long: Vec<u32> = (0..60).map(|i| (i * 7 % 250) as u32).collect();
    let ((a, sa), (b, sb)) = (run(&base, req(&long, 100)), run(&fb, req(&long, 100)));
    assert_eq!(a, b, "context cap");
    assert_eq!((a.reason, a.ids.len(), a.kv_after, sb - sa), (FinishReason::Length, 4, 63, 0));
    // what the window-filling request left behind is reused faithfully
    let next: Vec<u32> = long[..50].to_vec();
    assert_eq!(run(&base, req(&next, 6)).0, run(&fb, req(&next, 6)).0, "after the context cap");
}

#[test]
fn eos_at_every_position_rolls_the_extra_step_back() {
    let (base, fb) = (engine(12, false, 96), engine(12, true, 96));
    let p = bytes("xyz");
    let free = probe(&base, &fb, req(&p, 12));
    let mut saw_extra = false;
    for k in 0..6 {
        let mut r = req(&p, 12);
        r.ignore_eos = false;
        r.stop_ids = vec![free.ids[k]];
        let (a, extra) = same(&base, &fb, r, &format!("stop id at token {k}"));
        assert_eq!(a.reason, FinishReason::Stop);
        let first = free.ids.iter().position(|&t| t == free.ids[k]).unwrap();
        assert_eq!(a.ids.len(), first);
        // a stop on the first token (from prefill) never enters the feedback loop; any later one ran one extra step
        assert_eq!(extra, u64::from(first >= 1), "stop id at token {k}");
        saw_extra |= extra == 1;
    }
    assert!(saw_extra, "the feedback path was exercised");
}

#[test]
fn min_tokens_window_then_feedback() {
    let (base, fb) = (engine(13, false, 96), engine(13, true, 96));
    let p = bytes("abc");
    let free = probe(&base, &fb, req(&p, 16));
    for (min_tokens, k) in [(3, 1), (3, 5), (6, 2), (6, 8)] {
        let mut r = req(&p, 16);
        r.ignore_eos = false;
        r.stop_ids = vec![free.ids[k]];
        r.min_tokens = min_tokens;
        same(&base, &fb, r, &format!("min_tokens {min_tokens}, stop id from token {k}"));
    }
}

#[test]
fn stop_strings_match() {
    let (base, fb) = (engine(14, false, 96), engine(14, true, 96));
    let p = bytes("hello");
    let free = probe(&base, &fb, req(&p, 30));
    assert!(free.text.len() >= 10, "need some text: {:?}", free.text);
    let mut hits = 0;
    for start in [0usize, 1, 3, 6] {
        let probe: String = free.text.chars().skip(start).take(2).collect();
        let mut r = req(&p, 30);
        r.stop = vec![probe.clone()];
        let (a, extra) = same(&base, &fb, r, &format!("stop string {probe:?}"));
        if a.reason == FinishReason::Stop {
            hits += 1;
            assert_eq!(a.stop_sequence.as_deref(), Some(probe.as_str()));
            assert!(extra <= 1);
        }
    }
    assert!(hits > 0);
}

#[test]
fn eos_mid_run_in_a_queue_of_requests() {
    // several requests back to back on one engine (the single-sequence engine's queue): each one's leftovers must not
    // leak into the next
    let (base, fb) = (engine(15, false, 96), engine(15, true, 96));
    let prompts = [bytes("one"), bytes("two two"), bytes("three"), bytes("one")];
    for (i, p) in prompts.iter().enumerate() {
        let free = probe(&base, &fb, req(p, 10));
        let mut r = req(p, 10);
        r.ignore_eos = false;
        r.stop_ids = vec![free.ids[2 + i % 3]];
        same(&base, &fb, r, &format!("request {i}"));
    }
}

#[test]
fn cancel_leaves_a_consistent_state() {
    // where a cancel lands is timing-dependent, so check the invariants instead of equality with the plain engine
    let fb = engine(16, true, 128);
    let base = engine(16, false, 128);
    let p = bytes("cancel me");
    for after in [1usize, 3, 7] {
        let cancel = Arc::new(AtomicBool::new(false));
        let mut r = req(&p, 60);
        r.cancel = Some(cancel.clone());
        let (mut n, mut done) = (0, None);
        fb.generate(r, &mut |ev| match ev {
            GenEvent::Token { .. } => {
                n += 1;
                if n == after {
                    cancel.store(true, Ordering::Relaxed);
                }
            }
            GenEvent::Done { finish_reason, completion_tokens, .. } => done = Some((finish_reason, completion_tokens)),
            GenEvent::Error { message } => panic!("{message}"),
        })
        .unwrap();
        let (reason, completion) = done.unwrap();
        assert_eq!(reason, FinishReason::Cancelled);
        assert_eq!(fb.telemetry().kv_used_tokens as usize, p.len() + completion - 1, "cancel after {after}");
        // the next request on the cancelled engine equals a plain cold run
        let next = req(&bytes("cancel me too"), 6);
        let (x, _) = run(&fb, next.clone());
        let (y, _) = run(&base, next);
        assert_eq!(x.ids, y.ids, "cancel after {after}");
    }
}

/// `BatchedEngine` keeps its own decode loop: with token feedback switched on in the model (the Metal default), a batch
/// in which one request stops mid-run on a stop id while the others continue must give exactly what it gives with
/// feedback off.
#[test]
fn batched_engine_is_unchanged_with_feedback_on() {
    use intarsia_core::{BatchedEngine, BatchedOptions};
    type Done = (Vec<u32>, (FinishReason, usize, usize));
    fn batched(feedback: bool) -> BatchedEngine {
        let t = tiny_weights(DType::Bf16, 17);
        let tel = Arc::new(Telemetry::new());
        let mut m = intarsia_core::model::Model::load(
            intarsia_backend_cpu::CpuBackend::arc(),
            &recipe(),
            &dims(),
            &t.weights,
            &LoadOptions { max_seq: 128, prefill_chunk: 8, ..Default::default() },
            tel.clone(),
        )
        .unwrap();
        m.enable_batch(3).unwrap();
        m.configure_plan(&Default::default(), None).unwrap();
        assert_eq!(m.set_token_feedback(feedback), feedback);
        let o = BatchedOptions::new(3, 16);
        let meta = EngineMeta { id: "tiny".into(), eos_ids: vec![EOS], ..Default::default() };
        BatchedEngine::new(m, Arc::new(ByteCodec), meta, tel, o).unwrap()
    }
    fn collect(rx: std::sync::mpsc::Receiver<GenEvent>) -> Done {
        let mut ids = vec![];
        for ev in rx.iter() {
            match ev {
                GenEvent::Token { id, .. } => ids.push(id),
                GenEvent::Done { finish_reason, completion_tokens, cached_tokens, .. } => {
                    ids.truncate(completion_tokens);
                    return (ids, (finish_reason, completion_tokens, cached_tokens));
                }
                GenEvent::Error { message } => panic!("{message}"),
            }
        }
        panic!("no Done event")
    }
    let prompts = [bytes("alpha beta"), bytes("gamma"), bytes("delta epsilon zeta")];
    let mut outs = vec![];
    for feedback in [false, true] {
        let e = batched(feedback);
        // the stop id: what the first request generates as its 4th token
        let free = collect(e.submit(req(&prompts[0], 12)).unwrap());
        let mut stop = req(&prompts[0], 12);
        stop.ignore_eos = false;
        stop.stop_ids = vec![free.0[3]];
        let rxs = vec![e.submit(stop).unwrap(), e.submit(req(&prompts[1], 20)).unwrap(), e.submit(req(&prompts[2], 16)).unwrap()];
        let batch: Vec<Done> = rxs.into_iter().map(collect).collect();
        assert_eq!(batch[0].1 .0, FinishReason::Stop, "feedback {feedback}");
        // one more request after the batch drained
        let after = collect(e.submit(req(&bytes("alpha beta again"), 6)).unwrap());
        outs.push((free, batch, after, e.telemetry().kv_used_tokens));
    }
    assert_eq!(outs[0], outs[1]);
}
