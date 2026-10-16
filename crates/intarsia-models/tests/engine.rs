mod common;
use common::*;
use intarsia_core::model::LoadOptions;
use intarsia_core::{DType, Engine, EngineMeta, FinishReason, GenEvent, GenRequest, LocalEngine, Telemetry};
use std::sync::Arc;

fn engine(seed: u64) -> LocalEngine {
    let t = tiny_weights(DType::Bf16, seed);
    let tel = Arc::new(Telemetry::new());
    let m = intarsia_core::model::Model::load(
        intarsia_backend_cpu::CpuBackend::arc(),
        &recipe(),
        &dims(),
        &t.weights,
        &LoadOptions { max_seq: 64, ..Default::default() },
        tel.clone(),
    )
    .unwrap();
    LocalEngine::new(
        m,
        Arc::new(ByteCodec),
        EngineMeta { id: "tiny".into(), eos_ids: vec![299], ..Default::default() },
        tel,
    )
    .unwrap()
}

fn run(e: &LocalEngine, req: GenRequest) -> (Vec<u32>, String, GenEvent) {
    let mut ids = vec![];
    let mut text = String::new();
    let mut last = None;
    e.generate(req, &mut |ev| match ev {
        GenEvent::Token { id, text: t } => {
            ids.push(id);
            text.push_str(&t);
        }
        d @ GenEvent::Done { .. } => last = Some(d),
        GenEvent::Error { message } => panic!("engine error: {message}"),
    })
    .unwrap();
    (ids, text, last.unwrap())
}

fn req(prompt: &str, n: usize) -> GenRequest {
    GenRequest {
        prompt: prompt.bytes().map(|b| b as u32).collect(),
        max_tokens: n,
        ignore_eos: true,
        ..Default::default()
    }
}

#[test]
fn greedy_is_deterministic_and_reports_usage() {
    let e = engine(1);
    let (a, _, d) = run(&e, req("hello world", 10));
    assert_eq!(a.len(), 10);
    match d {
        GenEvent::Done { finish_reason, prompt_tokens, completion_tokens, cached_tokens, .. } => {
            assert_eq!(finish_reason, FinishReason::Length);
            assert_eq!((prompt_tokens, completion_tokens, cached_tokens), (11, 10, 0));
        }
        _ => unreachable!(),
    }
    // second identical request reuses the KV prefix and yields identical tokens
    let (b, _, d2) = run(&e, req("hello world", 10));
    assert_eq!(a, b);
    if let GenEvent::Done { cached_tokens, .. } = d2 {
        assert_eq!(cached_tokens, 10, "prompt minus the one token that must be recomputed");
    }
    // a fresh engine agrees (cache reuse does not change results)
    let (c, _, _) = run(&engine(1), req("hello world", 10));
    assert_eq!(a, c);
}

#[test]
fn seeded_sampling_reproducible_and_telemetry_counts() {
    let e = engine(2);
    let mut r = req("abc", 12);
    r.temperature = 0.9;
    r.top_k = 20;
    r.top_p = 0.95;
    r.seed = Some(7);
    let (a, _, _) = run(&e, r.clone());
    let (b, _, _) = run(&e, r.clone());
    assert_eq!(a, b);
    let t = e.telemetry();
    assert_eq!(t.requests_total, 2);
    assert_eq!((t.running, t.waiting), (0, 0));
    assert_eq!(t.generation_tokens_total, 24);
    assert!(t.steps_total >= 2 * 12, "steps {}", t.steps_total);
    assert!(t.ops["matmul"].calls > 0 && t.ops["matmul"].timed_calls > 0);
    assert!(t.kv_total_bytes > 0 && t.kv_capacity_tokens == 64);
    assert!(t.last_decode_tok_s > 0.0);
}

#[test]
fn eos_min_tokens_and_ignore_eos() {
    let e = engine(3);
    // find what the model generates greedily, then declare the 4th token an EOS
    let (base, _, _) = run(&e, req("xyz", 8));
    let eos = base[3];
    let mut r = req("xyz", 8);
    r.ignore_eos = false;
    r.stop_ids = vec![eos];
    let (ids, _, d) = run(&e, r.clone());
    // stops at the first occurrence of `eos`
    let first = base.iter().position(|t| *t == eos).unwrap();
    assert_eq!(ids.len(), first);
    if let GenEvent::Done { finish_reason, completion_tokens, .. } = d {
        assert_eq!(finish_reason, FinishReason::Stop);
        assert_eq!(completion_tokens, first + 1);
    }
    // min_tokens suppresses the stop token until enough tokens exist
    r.min_tokens = 6;
    let (ids2, _, _) = run(&e, r);
    assert!(ids2.len() >= 6 || ids2.len() == 8);
    assert!(!ids2[..6.min(ids2.len())].contains(&eos));
}

#[test]
fn stop_strings_trim_and_holdback() {
    let e = engine(4);
    let (_, full, _) = run(&e, req("hello", 24));
    assert!(full.len() >= 8, "need some text: {full:?}");
    let probe: String = full.chars().skip(3).take(2).collect();
    let mut r = req("hello", 24);
    r.stop = vec![probe.clone()];
    let (_, text, d) = run(&e, r);
    let idx = full.find(&probe).unwrap();
    assert_eq!(text, &full[..idx]);
    if let GenEvent::Done { finish_reason, .. } = d {
        assert_eq!(finish_reason, FinishReason::Stop);
    }
}

#[test]
fn dropped_receiver_cancels() {
    let e = engine(5);
    let rx = e.submit(req("cancel me", 60)).unwrap();
    // read one event then drop
    let _ = rx.recv().unwrap();
    drop(rx);
    // engine must stay healthy and finish the cancelled request quickly
    let (ids, _, _) = run(&e, req("next", 4));
    assert_eq!(ids.len(), 4);
    let t = e.telemetry();
    assert_eq!(t.finished.get("cancelled").copied().unwrap_or(0), 1);
    assert_eq!(t.running + t.waiting, 0);
}

#[test]
fn bad_requests_become_error_events() {
    let e = engine(6);
    let mut got_err = false;
    e.generate(GenRequest { prompt: vec![], max_tokens: 4, ..Default::default() }, &mut |ev| {
        if let GenEvent::Error { .. } = ev {
            got_err = true;
        }
    })
    .unwrap();
    assert!(got_err);
    got_err = false;
    e.generate(GenRequest { prompt: vec![65; 200], max_tokens: 4, ..Default::default() }, &mut |ev| {
        if let GenEvent::Error { message } = ev {
            assert!(message.contains("context"), "{message}");
            got_err = true;
        }
    })
    .unwrap();
    assert!(got_err);
    assert_eq!(e.telemetry().finished["error"], 2);
}

#[test]
fn forward_logits_all_positions() {
    let e = engine(7);
    let ids: Vec<u32> = (0..9).map(|i| (i * 13 % 250) as u32).collect();
    let rows = e.forward_logits(&ids).unwrap();
    assert_eq!(rows.len(), 9);
    assert_eq!(rows[0].len(), V);
    // consistent with a generation request's first token (greedy)
    let (g, _, _) = run(&e, GenRequest { prompt: ids.clone(), max_tokens: 1, ignore_eos: true, ..Default::default() });
    let last = rows.last().unwrap();
    let best = last.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    assert_eq!(g[0] as usize, best);
}

// ---- prefix reuse is faithful: the logits of a warm run equal those of a cold run -------------------------------

fn engine_chunk(seed: u64, chunk: usize) -> LocalEngine {
    let t = tiny_weights(DType::Bf16, seed);
    let tel = Arc::new(Telemetry::new());
    let m = intarsia_core::model::Model::load(
        intarsia_backend_cpu::CpuBackend::arc(),
        &recipe(),
        &dims(),
        &t.weights,
        &LoadOptions { max_seq: 96, prefill_chunk: chunk, ..Default::default() },
        tel.clone(),
    )
    .unwrap();
    LocalEngine::new(m, Arc::new(ByteCodec), EngineMeta { id: "tiny".into(), eos_ids: vec![299], ..Default::default() }, tel).unwrap()
}

type Trace = std::sync::Arc<std::sync::Mutex<Vec<Vec<f32>>>>;

/// Greedy run of `n` tokens recording the logits row of every step: (tokens, cached prompt tokens, rows).
fn traced(e: &LocalEngine, prompt: &[u32], n: usize) -> (Vec<u32>, usize, Vec<Vec<f32>>) {
    let trace: Trace = Default::default();
    let (mut ids, mut cached) = (vec![], 0);
    e.generate(
        GenRequest { prompt: prompt.to_vec(), max_tokens: n, ignore_eos: true, trace_logits: Some(trace.clone()), ..Default::default() },
        &mut |ev| match ev {
            GenEvent::Token { id, .. } => ids.push(id),
            GenEvent::Done { cached_tokens, completion_tokens, .. } => {
                cached = cached_tokens;
                ids.truncate(completion_tokens);
            }
            GenEvent::Error { message } => panic!("engine error: {message}"),
        },
    )
    .unwrap();
    let rows = trace.lock().unwrap().clone();
    (ids, cached, rows)
}

/// Forget what the engine holds: serve an unrelated prompt whose first token differs from `first`.
fn flush(e: &LocalEngine, first: u32) {
    traced(e, &[(first + 1) % 250, 3, 4, 5, 6, 7, 8, 9], 1);
}

fn prompt_ids(n: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * 37 + 11) % 250) as u32).collect()
}

fn max_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs())).fold(0.0, f32::max)
}

#[test]
fn prefix_reuse_restore_on_a_chunk_boundary_is_bit_identical_to_cold() {
    let chunk = 8;
    let e = engine_chunk(3, chunk);
    let p = prompt_ids(45);
    for split in [16usize, 24, 40] {
        // turn 1: the same first `split` tokens, then a different tail
        let mut t1 = p[..split].to_vec();
        t1.extend(p[split..].iter().take(11).map(|&t| (t + 1) % 250));
        flush(&e, p[0]);
        traced(&e, &t1, 4);
        let (warm, cached, wrows) = traced(&e, &p, 8);
        assert_eq!(cached, split, "split {split}: the shared prefix is reused, not more, not less");
        flush(&e, p[0]);
        let (cold, cold_cached, crows) = traced(&e, &p, 8);
        assert_eq!(cold_cached, 0);
        assert_eq!(warm, cold, "split {split}");
        assert_eq!(wrows.len(), 8);
        assert_eq!(max_diff(&wrows, &crows), 0.0, "split {split}: warm and cold prefill the same chunks, so the logits are the same bits");
    }
}

#[test]
fn prefix_reuse_continue_and_unaligned_restore_match_cold() {
    let chunk = 8;
    let e = engine_chunk(5, chunk);
    let p = prompt_ids(47);
    // continue: turn 2 = turn 1 + its reply + more; the reply rows were written one token at a time
    let split = 21;
    flush(&e, p[0]);
    let (reply, _, _) = traced(&e, &p[..split], 7);
    let mut t2 = p[..split].to_vec();
    t2.extend(&reply);
    t2.extend(&p[split..]);
    let (warm, cached, wrows) = traced(&e, &t2, 8);
    assert_eq!(cached, split + reply.len() - 1, "everything the engine holds: the last reply token was never fed");
    flush(&e, p[0]);
    let (cold, _, crows) = traced(&e, &t2, 8);
    assert_eq!(warm, cold);
    // other chunk shapes and decode-written rows: rounding only (the cpu backend is in fact bit-exact here)
    assert!(max_diff(&wrows, &crows) < 1e-3, "continue: warm vs cold {}", max_diff(&wrows, &crows));
    // restore at a split inside a chunk
    let split = 21;
    let mut t1 = p[..split].to_vec();
    t1.extend(p[split..].iter().take(9).map(|&t| (t + 1) % 250));
    flush(&e, p[0]);
    traced(&e, &t1, 4);
    let (warm, cached, wrows) = traced(&e, &p, 8);
    assert_eq!(cached, split);
    flush(&e, p[0]);
    let (cold, _, crows) = traced(&e, &p, 8);
    assert_eq!(warm, cold);
    assert!(max_diff(&wrows, &crows) < 1e-3, "restore: warm vs cold {}", max_diff(&wrows, &crows));
}

#[test]
fn scripted_replay_reproduces_a_warm_generate_bit_for_bit() {
    use intarsia_core::ScriptStep;
    let chunk = 8;
    let e = engine_chunk(9, chunk);
    let p = prompt_ids(44);
    let split = 19;
    flush(&e, p[0]);
    let (reply, _, _) = traced(&e, &p[..split], 6);
    let mut t2 = p[..split].to_vec();
    t2.extend(&reply);
    t2.extend(&p[split..]);
    let (warm, cached, wrows) = traced(&e, &t2, 6);
    // the same operations as a script: turn 1's prefill chunks, its decode steps, turn 2's prefill from the reuse point
    let step = |ids: &[u32], logits, truncate| ScriptStep { ids: ids.to_vec(), logits, truncate };
    let mut steps = vec![];
    let mut pos = 0;
    while split - pos > chunk {
        steps.push(step(&p[pos..pos + chunk], false, None));
        pos += chunk;
    }
    steps.push(step(&p[pos..split], true, None));
    for t in &reply[..reply.len() - 1] {
        steps.push(step(&[*t], true, None));
    }
    let mut pos = cached;
    let mut first = true;
    while t2.len() - pos > chunk {
        steps.push(step(&t2[pos..pos + chunk], false, first.then_some(cached)));
        first = false;
        pos += chunk;
    }
    steps.push(step(&t2[pos..], true, first.then_some(cached)));
    for t in &warm[..warm.len() - 1] {
        steps.push(step(&[*t], true, None));
    }
    let rows: Vec<Vec<f32>> = e.forward_script(&steps).unwrap().into_iter().flatten().collect();
    let tail = &rows[rows.len() - wrows.len()..];
    assert_eq!(max_diff(tail, &wrows), 0.0, "the engine's warm run is exactly its scripted replay");
}
