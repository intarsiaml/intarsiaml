//! Cancellation and stop-sequence reporting of the local engine.
mod common;
use common::*;
use intarsia_core::model::LoadOptions;
use intarsia_core::{DType, Engine, EngineMeta, FinishReason, GenEvent, GenRequest, LocalEngine, Telemetry};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn engine(seed: u64, chunk: usize) -> LocalEngine {
    let t = tiny_weights(DType::Bf16, seed);
    let tel = Arc::new(Telemetry::new());
    let m = intarsia_core::model::Model::load(
        intarsia_backend_cpu::CpuBackend::arc(),
        &recipe(),
        &dims(),
        &t.weights,
        &LoadOptions { max_seq: 128, prefill_chunk: chunk, ..Default::default() },
        tel.clone(),
    )
    .unwrap();
    LocalEngine::new(m, Arc::new(ByteCodec), EngineMeta { id: "tiny".into(), eos_ids: vec![299], ..Default::default() }, tel)
        .unwrap()
}

fn req(prompt: &str, n: usize) -> GenRequest {
    GenRequest { prompt: prompt.bytes().map(|b| b as u32).collect(), max_tokens: n, ignore_eos: true, ..Default::default() }
}

fn run(e: &LocalEngine, r: GenRequest) -> (String, GenEvent) {
    let mut text = String::new();
    let mut last = None;
    e.generate(r, &mut |ev| match ev {
        GenEvent::Token { text: t, .. } => text.push_str(&t),
        d @ GenEvent::Done { .. } => last = Some(d),
        GenEvent::Error { message } => panic!("engine error: {message}"),
    })
    .unwrap();
    (text, last.unwrap())
}

#[test]
fn done_carries_the_matched_stop_string() {
    let e = engine(4, 256);
    let (full, _) = run(&e, req("hello", 24));
    let a: String = full.chars().skip(3).take(2).collect();
    let mut r = req("hello", 24);
    r.stop = vec!["\u{1}never\u{1}".into(), a.clone()];
    let (_, d) = run(&e, r);
    match d {
        GenEvent::Done { finish_reason, stop_sequence, .. } => {
            assert_eq!(finish_reason, FinishReason::Stop);
            assert_eq!(stop_sequence.as_deref(), Some(a.as_str()));
        }
        _ => unreachable!(),
    }
    // a plain length stop has none
    if let (_, GenEvent::Done { stop_sequence, .. }) = run(&e, req("hello", 4)) {
        assert_eq!(stop_sequence, None);
    }
}

#[test]
fn cancel_before_prefill_runs_no_step_and_frees_the_worker() {
    let e = engine(6, 4);
    let flag = Arc::new(AtomicBool::new(true));
    let mut r = req(&"p".repeat(40), 8);
    r.cancel = Some(flag);
    let (text, d) = run(&e, r);
    assert!(text.is_empty());
    match d {
        GenEvent::Done { finish_reason, completion_tokens, .. } => {
            assert_eq!(finish_reason, FinishReason::Cancelled);
            assert_eq!(completion_tokens, 0);
        }
        _ => unreachable!(),
    }
    assert_eq!(e.telemetry().steps_total, 0, "no prefill chunk may run for a cancelled request");
    // the worker is free again, and a normal request still produces tokens
    let (text, d) = run(&e, req(&"p".repeat(40), 5));
    assert!(matches!(d, GenEvent::Done { finish_reason: FinishReason::Length, .. }), "{text}");
    let t = e.telemetry();
    assert_eq!((t.running, t.waiting), (0, 0));
}

#[test]
fn cancel_flag_stops_decoding() {
    let e = engine(7, 256);
    let flag = Arc::new(AtomicBool::new(false));
    let mut r = req("abc", 60);
    r.cancel = Some(flag.clone());
    let mut n = 0;
    let mut fin = None;
    e.generate(r, &mut |ev| match ev {
        GenEvent::Token { .. } => {
            n += 1;
            if n == 3 {
                flag.store(true, Ordering::SeqCst);
            }
        }
        d @ GenEvent::Done { .. } => fin = Some(d),
        GenEvent::Error { message } => panic!("{message}"),
    })
    .unwrap();
    match fin.unwrap() {
        GenEvent::Done { finish_reason, completion_tokens, .. } => {
            assert_eq!(finish_reason, FinishReason::Cancelled);
            assert!(completion_tokens < 10, "{completion_tokens}");
        }
        _ => unreachable!(),
    }
}
