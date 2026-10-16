//! `BatchedEngine` (the scheduler wired to the real `Model` adapters) against `LocalEngine` on the tiny Qwen3-shaped
//! fixture, cpu backend: requests submitted together to a batched engine must come out exactly as the same requests
//! run one at a time. The cpu kernels compute every row of a multi-row call with the one-row loop, so this checks the
//! adapter plumbing (slot states, split borrows, chunking, first-token handling, speculation gate, telemetry), as the
//! Qwen3-8B Metal run (`tools/cbidentity.py`) checks it with the real row-exact kernels. Prefix reuse is off in the
//! batched engine (cold-prefill solo reference); the chunk size is the same on both sides.
mod common;
use common::*;
use intarsia_core::model::{LoadOptions, Model};
use intarsia_core::spec::{SpecConfig, SpecMode};
use intarsia_core::{
    BatchedEngine, BatchedOptions, DType, Engine, EngineMeta, FinishReason, GenEvent, GenRequest, LocalEngine, Telemetry,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::Duration;

const CHUNK: usize = 16;

fn model(seed: u64, rows: usize, spec_k: Option<usize>, tel: &Arc<Telemetry>) -> (Model, SpecConfig) {
    let t = tiny_weights(DType::Bf16, seed);
    let mut m = Model::load(
        intarsia_backend_cpu::CpuBackend::arc(),
        &recipe(),
        &dims(),
        &t.weights,
        &LoadOptions { max_seq: 160, prefill_chunk: CHUNK, ..Default::default() },
        tel.clone(),
    )
    .unwrap();
    let spec = match spec_k {
        Some(k) => {
            m.enable_verify(k + 1).unwrap();
            SpecConfig { mode: SpecMode::Ngram, k }
        }
        None => SpecConfig::default(),
    };
    if rows > 1 {
        m.enable_batch(rows).unwrap();
    }
    m.configure_plan(&Default::default(), None).unwrap();
    (m, spec)
}

fn meta(spec: SpecConfig) -> EngineMeta {
    EngineMeta { id: "tiny".into(), eos_ids: vec![299], spec, ..Default::default() }
}

fn solo(seed: u64, spec_k: Option<usize>) -> LocalEngine {
    let tel = Arc::new(Telemetry::new());
    let (m, spec) = model(seed, 1, spec_k, &tel);
    LocalEngine::new(m, Arc::new(ByteCodec), meta(spec), tel).unwrap()
}

fn batched(seed: u64, rows: usize, spec_k: Option<usize>, max_queue: usize) -> (BatchedEngine, Arc<Telemetry>) {
    batched_with(seed, rows, spec_k, max_queue, |_| {})
}

fn batched_with(
    seed: u64,
    rows: usize,
    spec_k: Option<usize>,
    max_queue: usize,
    tweak: impl FnOnce(&mut BatchedOptions),
) -> (BatchedEngine, Arc<Telemetry>) {
    let tel = Arc::new(Telemetry::new());
    let (m, spec) = model(seed, rows, spec_k, &tel);
    let mut o = BatchedOptions::new(rows, max_queue);
    o.prefix_reuse = false;
    tweak(&mut o);
    (BatchedEngine::new(m, Arc::new(ByteCodec), meta(spec), tel.clone(), o).unwrap(), tel)
}

#[derive(Debug, PartialEq)]
struct Out {
    ids: Vec<u32>,
    text: String,
    reason: Option<FinishReason>,
    prompt_tokens: usize,
    completion_tokens: usize,
    cached_tokens: usize,
    stop_sequence: Option<String>,
}

fn collect(rx: &Receiver<GenEvent>) -> Result<Out, String> {
    let (mut ids, mut text) = (vec![], String::new());
    loop {
        match rx.recv_timeout(Duration::from_secs(120)).expect("engine stalled") {
            GenEvent::Token { id, text: t } => {
                ids.push(id);
                text.push_str(&t);
            }
            GenEvent::Done { finish_reason, prompt_tokens, completion_tokens, cached_tokens, stop_sequence, .. } => {
                return Ok(Out {
                    ids,
                    text,
                    reason: Some(finish_reason),
                    prompt_tokens,
                    completion_tokens,
                    cached_tokens,
                    stop_sequence,
                })
            }
            GenEvent::Error { message } => return Err(message),
        }
    }
}

fn run_solo(e: &LocalEngine, req: GenRequest) -> Out {
    // a throw-away request first so no KV prefix of the previous case is reused
    let flush = GenRequest { prompt: vec![250], max_tokens: 1, ignore_eos: true, ..Default::default() };
    collect(&e.submit(flush).unwrap()).unwrap();
    collect(&e.submit(req).unwrap()).unwrap()
}

/// Deterministic cases: prompts of 3..110 tokens (first token unique per case), greedy and seeded sampling,
/// `min_tokens`, stop strings, EOS, `max_tokens` from 1 to 40; some periodic so the n-gram drafter hits.
fn cases(n: usize) -> Vec<GenRequest> {
    let mut rng = intarsia_core::rng::SplitMix64::new(0xC0FFEE);
    (0..n)
        .map(|i| {
            let len = [3, 9, 15, 16, 17, 33, 48, 64, 90, 110, 7, 31][i % 12];
            let period = 3 + (i % 5);
            let mut prompt: Vec<u32> = vec![1 + i as u32];
            let base: Vec<u32> = (0..period).map(|_| 1 + (rng.next_u64() % 200) as u32).collect();
            while prompt.len() < len {
                if i % 3 == 0 {
                    prompt.push(base[prompt.len() % period]);
                } else {
                    prompt.push(1 + (rng.next_u64() % 250) as u32);
                }
            }
            let mut r = GenRequest { prompt, max_tokens: 1 + (i * 7) % 40, ignore_eos: i % 4 == 0, ..Default::default() };
            match i % 6 {
                1 => {
                    r.temperature = 0.8;
                    r.top_k = 40;
                    r.top_p = 0.95;
                    r.seed = Some(100 + i as u64);
                }
                2 => r.min_tokens = 6,
                3 => r.stop = vec![String::from_utf8_lossy(&[(1 + (i * 13) % 100) as u8]).into_owned()],
                _ => {}
            }
            r
        })
        .collect()
}

fn check(seed: u64, rows: usize, spec_k: Option<usize>, n_cases: usize) {
    let reference = solo(seed, None);
    let want: Vec<Out> = cases(n_cases).into_iter().map(|r| run_solo(&reference, r)).collect();
    let (eng, tel) = batched(seed, rows, spec_k, 64);
    // all at once: more requests than slots, so they queue, overlap, and finish at different times
    let rxs: Vec<_> = cases(n_cases).into_iter().map(|r| eng.submit(r).unwrap()).collect();
    for (i, (rx, w)) in rxs.iter().zip(&want).enumerate() {
        let mut got = collect(rx).unwrap();
        assert_eq!(got.cached_tokens, 0, "case {i}: prefix reuse is off");
        got.cached_tokens = w.cached_tokens;
        assert_eq!(&got, w, "case {i} (rows {rows}, spec {spec_k:?}, seed {seed})");
    }
    let m = eng.metrics();
    assert!(rows == 1 || m.batch_hist[2..].iter().sum::<u64>() > 0, "no batched decode pass ran: {:?}", m.batch_hist);
    // every request balanced its telemetry counters
    let snap = tel.snapshot();
    assert_eq!((snap.running, snap.waiting), (0, 0), "{snap:?}");
    assert_eq!(snap.requests_total as usize, n_cases);
    if spec_k.is_some() && rows < 4 {
        assert!(snap.spec_steps > 0 && snap.spec_accepted > 0, "no accepted drafts: {snap:?}");
    }
}

#[test]
fn batched_equals_solo_plain() {
    for (seed, rows) in [(7u64, 2usize), (7, 4), (23, 8)] {
        check(seed, rows, None, 30);
    }
}

#[test]
fn batched_equals_solo_with_speculation() {
    // B < 4: speculation offered; B >= 4: gated off. Either way the stream is the plain one (G1).
    for rows in [2usize, 4] {
        check(11, rows, Some(3), 24);
    }
}

#[test]
fn cancel_and_errors_leave_others_alone() {
    let reference = solo(7, None);
    let all = cases(8);
    let want: Vec<Out> = all.iter().cloned().map(|r| run_solo(&reference, r)).collect();
    let (eng, tel) = batched(7, 3, None, 64);
    let flag = Arc::new(AtomicBool::new(false));
    let mut rxs = Vec::new();
    for (i, r) in all.iter().cloned().enumerate() {
        let mut r = r;
        if i == 2 {
            r.cancel = Some(flag.clone());
            r.max_tokens = 100_000;
            r.ignore_eos = true;
        }
        rxs.push(eng.submit(r).unwrap());
    }
    // an out-of-vocab token is rejected at admission, a too-long prompt as well
    let bad = eng.submit(GenRequest { prompt: vec![1, 2, 99_999], ..Default::default() }).unwrap();
    let long = eng.submit(GenRequest { prompt: vec![5; 4000], ..Default::default() }).unwrap();
    flag.store(true, Ordering::Relaxed);
    for (i, rx) in rxs.iter().enumerate() {
        let got = collect(rx).unwrap();
        if i == 2 {
            assert_eq!(got.reason, Some(FinishReason::Cancelled));
        } else {
            assert_eq!(got, Out { cached_tokens: got.cached_tokens, ..clone_out(&want[i]) }, "case {i}");
        }
    }
    assert!(collect(&bad).unwrap_err().contains("vocab"));
    assert!(collect(&long).unwrap_err().contains("does not fit"));
    let snap = tel.snapshot();
    assert_eq!((snap.running, snap.waiting), (0, 0), "{snap:?}");
}

fn clone_out(o: &Out) -> Out {
    Out {
        ids: o.ids.clone(),
        text: o.text.clone(),
        reason: o.reason,
        prompt_tokens: o.prompt_tokens,
        completion_tokens: o.completion_tokens,
        cached_tokens: o.cached_tokens,
        stop_sequence: o.stop_sequence.clone(),
    }
}

#[test]
fn queue_full_is_refused_and_unwinds_telemetry() {
    let (eng, tel) = batched(7, 2, None, 2);
    let stop = Arc::new(AtomicBool::new(false));
    let long = |i: u32| GenRequest {
        prompt: vec![1 + i, 2, 3],
        max_tokens: 30_000,
        ignore_eos: true,
        cancel: Some(stop.clone()),
        ..Default::default()
    };
    // submit never blocks; the scheduler thread admits two, two wait, further ones are refused
    let mut rxs = Vec::new();
    let mut refused = 0;
    for i in 0..60 {
        match eng.submit(long(i)) {
            Ok(rx) => rxs.push(rx),
            Err(e) => {
                assert!(matches!(e, intarsia_core::Error::QueueFull(2)), "{e}");
                refused += 1;
            }
        }
    }
    assert!(refused > 0, "a bounded queue refused nothing");
    stop.store(true, Ordering::Relaxed);
    for rx in &rxs {
        let _ = collect(rx);
    }
    std::thread::sleep(Duration::from_millis(200));
    let snap = tel.snapshot();
    assert_eq!((snap.running, snap.waiting), (0, 0), "{snap:?}");
    assert_eq!(snap.requests_total as usize, rxs.len(), "{snap:?}");
}

#[test]
fn solo_only_endpoints_are_refused() {
    let (eng, _) = batched(7, 2, None, 4);
    assert!(eng.forward_logits(&[1, 2, 3]).unwrap_err().to_string().contains("--max-batch 1"));
}

#[test]
fn bad_decode_row_fails_only_its_request() {
    let reference = solo(7, None);
    let mk = |i: u32| GenRequest { prompt: vec![10 + i, 20, 30], max_tokens: 24, ignore_eos: true, ..Default::default() };
    let want: Vec<Out> = (0..4).map(|i| run_solo(&reference, mk(i))).collect();
    // three slots: requests 0..3 take slots 0..3 in order; slot 1 is fed an out-of-vocab id once it holds 3 + 4 tokens
    let (eng, tel) = batched_with(7, 3, None, 16, |o| o.row_fault = Some((1, 7, 99_999)));
    let rxs: Vec<_> = (0..3).map(|i| eng.submit(mk(i)).unwrap()).collect();
    let err = collect(&rxs[1]).unwrap_err();
    assert!(err.contains("vocab"), "{err}");
    for i in [0usize, 2] {
        let got = collect(&rxs[i]).unwrap();
        assert_eq!(got, Out { cached_tokens: got.cached_tokens, ..clone_out(&want[i]) }, "request {i}");
    }
    // the engine did not fail, and the poisoned slot serves the next request exactly
    assert!(eng.failed().is_none(), "{:?}", eng.failed());
    let again = collect(&eng.submit(mk(3)).unwrap()).unwrap();
    assert_eq!(again, Out { cached_tokens: again.cached_tokens, ..clone_out(&want[3]) });
    let m = eng.metrics();
    assert_eq!((m.failed, m.finished), (1, 3), "{m:?}");
    std::thread::sleep(Duration::from_millis(100));
    let snap = tel.snapshot();
    assert_eq!((snap.running, snap.waiting), (0, 0), "{snap:?}");
}

/// The `Done` event's prefill chunk: reported by the batched engine only (so CUDA / HIP, which never batch, answer as
/// before).
fn done_chunk(rx: &Receiver<GenEvent>) -> Option<usize> {
    loop {
        match rx.recv_timeout(Duration::from_secs(120)).expect("engine stalled") {
            GenEvent::Done { prefill_chunk, .. } => return prefill_chunk,
            GenEvent::Error { message } => panic!("{message}"),
            GenEvent::Token { .. } => {}
        }
    }
}

#[test]
fn only_the_batched_engine_reports_the_prefill_chunk() {
    let req = || GenRequest { prompt: vec![5, 6, 7], max_tokens: 2, ignore_eos: true, ..Default::default() };
    assert_eq!(done_chunk(&solo(7, None).submit(req()).unwrap()), None);
    let (eng, _) = batched(7, 2, None, 4);
    assert_eq!(done_chunk(&eng.submit(req()).unwrap()), Some(CHUNK));
}
