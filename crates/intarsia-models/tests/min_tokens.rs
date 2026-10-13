//! `min_tokens` end to end (review finding 9): EOS masking with `min_tokens` alone (no `ignore_eos`),
//! the forced-length path keeping the on-device argmax, and stop strings waiting for `min_tokens`.
mod common;
use common::*;
use intarsia_core::model::LoadOptions;
use intarsia_core::{DType, Engine, EngineMeta, FinishReason, GenEvent, GenRequest, LocalEngine, Telemetry};
use std::sync::Arc;

fn engine(seed: u64, eos: Vec<u32>) -> LocalEngine {
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
    LocalEngine::new(m, Arc::new(ByteCodec), EngineMeta { id: "tiny".into(), eos_ids: eos, ..Default::default() }, tel)
        .unwrap()
}

struct Out {
    ids: Vec<u32>,
    text: String,
    reason: FinishReason,
    completion_tokens: usize,
}

fn run(e: &LocalEngine, req: GenRequest) -> Out {
    let (mut ids, mut text, mut done) = (vec![], String::new(), None);
    e.generate(req, &mut |ev| match ev {
        GenEvent::Token { id, text: t } => {
            ids.push(id);
            text.push_str(&t);
        }
        GenEvent::Done { finish_reason, completion_tokens, .. } => done = Some((finish_reason, completion_tokens)),
        GenEvent::Error { message } => panic!("engine error: {message}"),
    })
    .unwrap();
    let (reason, completion_tokens) = done.unwrap();
    Out { ids, text, reason, completion_tokens }
}

fn req(n: usize) -> GenRequest {
    GenRequest { prompt: "xyz".bytes().map(|b| b as u32).collect(), max_tokens: n, ..Default::default() }
}

#[test]
fn min_tokens_alone_masks_eos_without_ignore_eos() {
    // what the model says greedily; its 3rd token becomes the EOS of a second, identical engine
    let base = run(&engine(31, vec![299]), GenRequest { ignore_eos: true, ..req(12) });
    let eos = base.ids[2];
    assert!(!base.ids[..2].contains(&eos), "pick a seed where the third token is new");
    let e = engine(31, vec![eos]);

    // without min_tokens it stops at the EOS
    let plain = run(&e, req(12));
    assert_eq!(plain.ids, base.ids[..2]);
    assert_eq!((plain.reason, plain.completion_tokens), (FinishReason::Stop, 3));

    // min_tokens alone (ignore_eos false): the EOS cannot appear in the first 6 tokens, so the
    // token at the EOS position is the best *other* token, and generation reaches 6 tokens
    let masked = run(&e, GenRequest { min_tokens: 6, ..req(12) });
    assert!(masked.completion_tokens >= 6, "{}", masked.completion_tokens);
    assert!(!masked.ids[..6.min(masked.ids.len())].contains(&eos), "{:?}", masked.ids);
    assert_eq!(masked.ids[..2], base.ids[..2]);
    assert_ne!(masked.ids[2], eos);
    // and it is not just "stopped late": past min_tokens the EOS ends the run again or the cap does
    assert!(matches!(masked.reason, FinishReason::Stop | FinishReason::Length));
}

#[test]
fn forced_length_greedy_keeps_the_on_device_argmax() {
    // the speed-benchmark shape: ignore_eos + min_tokens == max_tokens, greedy
    let e = engine(32, vec![299]);
    let n = 10;
    let forced = run(&e, GenRequest { ignore_eos: true, min_tokens: n, ..req(n) });
    assert_eq!(forced.completion_tokens, n);
    assert_eq!(forced.reason, FinishReason::Length);
    let t = e.telemetry();
    eprintln!(
        "DBG ids {} argmax calls {} logits calls {}",
        forced.ids.len(),
        t.ops["argmax"].calls,
        t.ops["logits"].calls
    );
    // every one of the n tokens came from the device argmax: no host-side sampling path
    assert_eq!(t.ops["argmax"].calls, n as u64, "forced-length decode must not pull logits to the host");
    // same tokens as plain greedy
    let plain = run(&engine(32, vec![299]), GenRequest { ignore_eos: true, ..req(n) });
    assert_eq!(forced.ids, plain.ids);
    assert_eq!(forced.text, plain.text);

    // with a real stop id and min_tokens the logits are needed (mask), so the argmax op is not used
    let e2 = engine(32, vec![forced.ids[1]]);
    let masked = run(&e2, GenRequest { min_tokens: 4, ..req(n) });
    assert!(masked.completion_tokens >= 4);
    assert!(e2.telemetry().ops["argmax"].calls < masked.completion_tokens as u64);
}

#[test]
fn stop_strings_wait_for_min_tokens() {
    let e = engine(33, vec![299]);
    let full = run(&e, GenRequest { ignore_eos: true, ..req(24) });
    assert!(full.text.chars().count() >= 12, "{:?}", full.text);
    let chars: Vec<char> = full.text.chars().collect();
    let probe: String = chars[2..4].iter().collect();
    let first = full.text.find(&probe).unwrap();

    // stop string alone: stops at its first match
    let s1 = run(&e, GenRequest { ignore_eos: true, stop: vec![probe.clone()], ..req(24) });
    assert_eq!(s1.text, &full.text[..first]);
    assert_eq!(s1.reason, FinishReason::Stop);

    // with min_tokens 10 the early match (at token 3) is ignored; generation goes on to >= 10 tokens
    let s2 = run(&e, GenRequest { ignore_eos: true, min_tokens: 10, stop: vec![probe.clone()], ..req(24) });
    assert!(s2.completion_tokens >= 10, "{} tokens, text {:?}", s2.completion_tokens, s2.text);
    assert!(s2.text.starts_with(&full.text[..first + probe.len()]), "{:?} vs {:?}", s2.text, full.text);
}
