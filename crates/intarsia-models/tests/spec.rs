//! Speculative decoding (milestone G1) through the engine on the cpu backend: spec on must emit exactly the token
//! stream, text, finish reason and counts of spec off, including when `max_tokens`, a stop string, an EOS or the
//! `min_tokens` boundary lands inside an accepted run, and must leave the KV state where sequential decode leaves it
//! (a follow-up request reuses the same prefix). The cpu kernels compute every row of a multi-row call with the
//! single-row loop, so the verify rows equal the decode rows here as the row-exact kernels make them equal on Metal.
//! (`load_engine` only enables `--spec` on Metal; these tests build the engine directly.)
mod common;
use common::*;
use intarsia_core::model::LoadOptions;
use intarsia_core::spec::{SpecConfig, SpecMode};
use intarsia_core::{DType, Engine, EngineMeta, FinishReason, GenEvent, GenRequest, LocalEngine, Telemetry};
use std::sync::Arc;

fn engine(seed: u64, eos: Vec<u32>, spec_k: Option<usize>) -> (LocalEngine, Arc<Telemetry>) {
    let t = tiny_weights(DType::Bf16, seed);
    let tel = Arc::new(Telemetry::new());
    let mut m = intarsia_core::model::Model::load(
        intarsia_backend_cpu::CpuBackend::arc(),
        &recipe(),
        &dims(),
        &t.weights,
        &LoadOptions { max_seq: 160, ..Default::default() },
        tel.clone(),
    )
    .unwrap();
    let spec = match spec_k {
        Some(k) => {
            m.enable_verify(k + 1).unwrap();
            m.configure_plan(&Default::default(), None).unwrap();
            SpecConfig { mode: SpecMode::Ngram, k }
        }
        None => SpecConfig::default(),
    };
    let e = LocalEngine::new(m, Arc::new(ByteCodec), EngineMeta { id: "tiny".into(), eos_ids: eos, spec, ..Default::default() }, tel.clone())
        .unwrap();
    (e, tel)
}

#[derive(Debug, PartialEq)]
struct Out {
    ids: Vec<u32>,
    text: String,
    reason: FinishReason,
    completion_tokens: usize,
    cached_tokens: usize,
    stop_sequence: Option<String>,
}

fn run(e: &LocalEngine, req: GenRequest) -> Out {
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
    let (reason, completion_tokens, cached_tokens, stop_sequence) = done.unwrap();
    Out { ids, text, reason, completion_tokens, cached_tokens, stop_sequence }
}

fn ids(s: &str) -> Vec<u32> {
    s.bytes().map(|b| b as u32).collect()
}

fn req(prompt: &str, n: usize) -> GenRequest {
    GenRequest { prompt: ids(prompt), max_tokens: n, ignore_eos: true, ..Default::default() }
}

const PROMPTS: [&str; 3] = ["abcabcabcabcabc", "the cat sat on the mat; the cat sat on the", "xyz"];

/// Greedy spec on == spec off over seeds, prompts and draft lengths; verify passes do run and accept drafts.
#[test]
fn spec_identity_greedy() {
    for seed in [3u64, 11, 29] {
        let (off, _) = engine(seed, vec![299], None);
        for k in [1usize, 3, 6] {
            let (on, tel) = engine(seed, vec![299], Some(k));
            for p in PROMPTS {
                let a = run(&off, req(p, 60));
                let b = run(&on, req(p, 60));
                assert_eq!(a, b, "seed {seed} k {k} prompt {p:?}");
            }
            let t = tel.snapshot();
            assert!(t.spec_steps > 0 && t.spec_accepted > 0, "seed {seed} k {k}: no accepted drafts ({t:?})");
            assert!(t.spec_accepted <= t.spec_proposed);
        }
    }
}

/// `max_tokens` at every length, and a stop string ending at every position of the reference output: whatever the
/// run boundaries, the output and the follow-up request (prefix reuse of the KV the first one left) are identical.
#[test]
fn spec_stops_inside_accepted_runs() {
    let seed = 11;
    let (off, _) = engine(seed, vec![299], None);
    let (on, tel) = engine(seed, vec![299], Some(4));
    let p = PROMPTS[0];
    let full = run(&off, req(p, 48));
    assert_eq!(full, run(&on, req(p, 48)));
    for n in 1..=48 {
        let a = run(&off, req(p, n));
        let b = run(&on, req(p, n));
        assert_eq!(a, b, "max_tokens {n}");
        assert_eq!(a.reason, FinishReason::Length);
        // follow-up: prompt + what was generated + one more byte, so the KV prefix of the first request is reused
        let next = format!("{p}{}q", a.text);
        let (a2, b2) = (run(&off, req(&next, 6)), run(&on, req(&next, 6)));
        assert_eq!(a2, b2, "follow-up after max_tokens {n}");
    }
    let bytes: Vec<u32> = full.ids.iter().copied().filter(|&i| i < 128 && i >= 32).collect();
    let text = full.text.clone();
    let mut checked = 0;
    for end in 2..text.len() {
        if !text.is_char_boundary(end) || !text.is_char_boundary(end - 2) {
            continue;
        }
        let stop = text[end - 2..end].to_string();
        let r = |e: &LocalEngine| run(e, GenRequest { stop: vec![stop.clone()], ..req(p, 48) });
        let (a, b) = (r(&off), r(&on));
        assert_eq!(a, b, "stop {stop:?}");
        checked += 1;
    }
    assert!(checked > 10, "{checked} stop strings ({} printable ids)", bytes.len());
    assert!(tel.snapshot().spec_accepted > 0);
}

/// EOS = a token of the reference output, `min_tokens` at every length: the EOS mask (host argmax of masked logits on
/// the plain path, device argmax plus a cut before the masked row on the verify path) gives the same stream, also
/// when the boundary or the EOS falls inside a run. Seed 19 / "xyz" is one of the few tiny-model streams that changes
/// token a few times (71 71 211 211 211 286 286 178 178 178 291 185 ...), so the EOS candidates are first seen at
/// positions 3..12, where verify rows predict them.
#[test]
fn spec_min_tokens_and_eos_inside_runs() {
    let seed = 19;
    let p = "xyz";
    let (probe, _) = engine(seed, vec![299], None);
    let base = run(&probe, req(p, 40));
    let mut eos_set: Vec<u32> = Vec::new();
    for &t in &base.ids[2..] {
        if !base.ids[..2].contains(&t) && !eos_set.contains(&t) {
            eos_set.push(t);
        }
    }
    assert!(eos_set.len() >= 3, "{:?}", base.ids);
    let mut steps = 0;
    for eos in eos_set {
        let (off, _) = engine(seed, vec![eos], None);
        let (on, tel) = engine(seed, vec![eos], Some(4));
        for min in 0..=24 {
            let mk = || GenRequest { prompt: ids(p), max_tokens: 40, min_tokens: min, ..Default::default() };
            let (a, b) = (run(&off, mk()), run(&on, mk()));
            assert_eq!(a, b, "eos {eos} min_tokens {min}");
            let mk2 = || GenRequest { stop: vec!["<1".into()], ..mk() };
            let (a, b) = (run(&off, mk2()), run(&on, mk2()));
            assert_eq!(a, b, "eos {eos} min_tokens {min} + stop string");
        }
        steps += tel.snapshot().spec_steps;
    }
    assert!(steps > 0);
}

/// Sampled requests (temperature > 0) take plain decode steps with spec on and give the seeded stream of spec off.
#[test]
fn spec_sampled_requests_fall_back() {
    let (off, _) = engine(3, vec![299], None);
    let (on, tel) = engine(3, vec![299], Some(4));
    let mk = || GenRequest { temperature: 0.8, top_p: 0.9, seed: Some(7), ..req(PROMPTS[0], 40) };
    assert_eq!(run(&off, mk()), run(&on, mk()));
    assert_eq!(tel.snapshot().spec_steps, 0);
}

