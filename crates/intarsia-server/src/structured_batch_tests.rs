//! Lane ink-hostgap: json_schema constraints (the llguidance [`LlgSpec`](super) path) on batched sampled decode. With the host
//! pool on (`INTARSIA_HOST_THREADS` > 1), the rows of a batched decode step are accepted one per pool task
//! (`Scheduler::accept_rows`), so a constrained row's mask, apply, sample and commit run on a pool worker. Every request's
//! tokens must be bit for bit what the same request gives alone (`LocalEngine`), with the pool on and off: run this test
//! with `INTARSIA_HOST_THREADS=8` and `=1`. The tiny Qwen3-shaped fixture of `intarsia-models`, cpu backend; its byte
//! vocabulary is the grammar env's (token i = byte i, EOS 256, padded to 300).

use super::{GrammarEnv, Plan, CLOSE, OPEN, THINK_CLOSE};
use intarsia_core::engine::GenEvent;
use intarsia_core::model::{LoadOptions, Model};
use intarsia_core::spec::SpecConfig;
use intarsia_core::{BatchedEngine, BatchedOptions, DType, Engine, EngineMeta, FinishReason, GenRequest, LocalEngine, Telemetry};
use serde_json::json;
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::Duration;

#[path = "../../intarsia-models/tests/common/mod.rs"]
#[allow(clippy::all)]
mod tiny;

const EOS: u32 = 256;

fn env() -> GrammarEnv {
    let mut b: Vec<Vec<u8>> = (0..=255u8).map(|x| vec![x]).collect();
    b.push(b"\xFF<eos>".to_vec());
    b.push(OPEN.as_bytes().to_vec());
    b.push(CLOSE.as_bytes().to_vec());
    b.push(THINK_CLOSE.as_bytes().to_vec());
    GrammarEnv::from_parts(b, tiny::V, &[EOS], Box::new(|s| Some(s.bytes().map(u32::from).collect()))).unwrap()
}

fn model(rows: usize) -> Model {
    let t = tiny::tiny_weights(DType::Bf16, 19);
    let mut m = Model::load(
        intarsia_backend_cpu::CpuBackend::arc(),
        &tiny::recipe(),
        &tiny::dims(),
        &t.weights,
        &LoadOptions { max_seq: 160, prefill_chunk: 16, ..Default::default() },
        Arc::new(Telemetry::new()),
    )
    .unwrap();
    if rows > 1 {
        m.enable_batch(rows).unwrap();
    }
    m.configure_plan(&Default::default(), None).unwrap();
    m
}

fn meta() -> EngineMeta {
    EngineMeta { id: "tiny".into(), eos_ids: vec![EOS], spec: SpecConfig::default(), ..Default::default() }
}

#[derive(Debug, PartialEq)]
struct Out {
    ids: Vec<u32>,
    reason: FinishReason,
}

fn collect(rx: &Receiver<GenEvent>) -> Out {
    let mut ids = vec![];
    loop {
        match rx.recv_timeout(Duration::from_secs(120)).expect("engine stalled") {
            GenEvent::Token { id, .. } => ids.push(id),
            GenEvent::Done { finish_reason, .. } => return Out { ids, reason: finish_reason },
            GenEvent::Error { message } => panic!("request failed: {message}"),
        }
    }
}

/// Every request sampled; even ones constrained to a JSON object schema, odd ones free (next to constrained rows in one step).
fn requests(env: &GrammarEnv, n: usize) -> Vec<GenRequest> {
    let schema = json!({
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "string", "maxLength": 6}, "c": {"type": "boolean"}},
        "required": ["a", "b"],
        "additionalProperties": false
    });
    (0..n)
        .map(|i| {
            let prompt: Vec<u32> = format!("req {i}: give json {}", "xyz".repeat(i % 4)).bytes().map(u32::from).collect();
            let mut r = GenRequest {
                prompt,
                max_tokens: 24 + (i * 5) % 20,
                temperature: 0.7 + 0.15 * (i % 5) as f32,
                top_k: if i % 3 == 0 { 0 } else { 40 },
                top_p: if i % 4 == 1 { 0.9 } else { 1.0 },
                seed: Some(500 + 11 * i as u64),
                ..Default::default()
            };
            if i % 2 == 0 {
                r.constraint = Some(env.compile(&Plan::Json(schema.clone())).unwrap());
            }
            r
        })
        .collect()
}

#[test]
fn json_schema_batched_sampled_equals_solo() {
    let env = env();
    let solo = LocalEngine::new(model(1), Arc::new(tiny::ByteCodec), meta(), Arc::new(Telemetry::new())).unwrap();
    let n = 16;
    let want: Vec<Out> = requests(&env, n).into_iter().map(|r| collect(&solo.submit(r).unwrap())).collect();
    // the constraint did constrain: every constrained stream is JSON bytes only
    for (i, w) in want.iter().enumerate().filter(|(i, _)| i % 2 == 0) {
        assert!(w.ids.iter().all(|&t| t < 128 || t == EOS), "request {i}: {:?}", w.ids);
        assert!(w.ids.contains(&u32::from(b'{')), "request {i} opens a JSON object: {:?}", w.ids);
    }
    for rows in [4usize, 8] {
        let mut o = BatchedOptions::new(rows, 64);
        o.prefix_reuse = false;
        let eng = BatchedEngine::new(model(rows), Arc::new(tiny::ByteCodec), meta(), Arc::new(Telemetry::new()), o).unwrap();
        let rxs: Vec<_> = requests(&env, n).into_iter().map(|r| eng.submit(r).unwrap()).collect();
        for (i, (rx, w)) in rxs.iter().zip(&want).enumerate() {
            assert_eq!(&collect(rx), w, "request {i} (rows {rows}, host threads {})", intarsia_core::par::threads());
        }
        let m = eng.metrics();
        assert!(m.batch_hist[2..].iter().sum::<u64>() > 0, "no batched decode pass ran: {:?}", m.batch_hist);
    }
}
