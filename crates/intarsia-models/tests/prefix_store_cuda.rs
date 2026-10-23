//! The prefix store's byte-identity gate on a real model on the CUDA backend (the 35B hybrid by default, or any model
//! through `INTARSIA_CUDA_TEST_MODEL`); the same scenario as `prefix_store_metal.rs`. Through the engine and the model's own chat template:
//!
//! 1. engine A (store on): turn 1 of a chat with thinking, then an unrelated chat (it replaces the held sequence), then
//!    turn 2, which restores turn 1's history from the store;
//! 2. engine B (store on, fresh): turn 2 cold;
//! 3. engine C (store off): turn 2 cold.
//!
//! A's turn 2 must equal B's bit for bit (logits of every step and the greedy ids). C is reported, not asserted: CUDA
//! prefill is chunk-sensitive (recipes/qwen3_5_moe.toml [tuning.cuda]), so the boundary-cut schedule changes bits.
//!
//! Opt-in: `INTARSIA_TEST_REAL_MODEL=1` (skips otherwise, and without a CUDA device). Take a fleetq slot on the Spark.
#![cfg(not(target_os = "macos"))]
use intarsia_core::engine::ChatOpts;
use intarsia_core::model::LoadOptions;
use intarsia_core::prefix_store::PrefixStoreConfig;
use intarsia_core::{Engine, GenEvent, GenRequest};
use intarsia_models::{load_engine, EngineOptions};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn enabled() -> bool {
    if std::env::var("INTARSIA_TEST_REAL_MODEL").is_ok_and(|v| v == "1") {
        return true;
    }
    eprintln!("SKIP: real-model test (set INTARSIA_TEST_REAL_MODEL=1; run inside a fleetq slot)");
    false
}

fn model_dir() -> PathBuf {
    std::env::var_os("INTARSIA_CUDA_TEST_MODEL").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join(".cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5")
    })
}

fn engine(store: bool) -> Arc<dyn Engine> {
    let backend = intarsia_backend_cuda::create(0).unwrap();
    let opts = EngineOptions {
        load: LoadOptions { max_seq: 8192, profile_every: 0, ..Default::default() },
        prefix_store: store.then(|| PrefixStoreConfig { host_bytes: 16 << 30, ..Default::default() }),
        max_batch: Some(1),
        ..Default::default()
    };
    load_engine(&model_dir(), backend, &opts).unwrap().engine
}

struct Out {
    ids: Vec<u32>,
    logits: Vec<Vec<f32>>,
    cached: usize,
    prompt: usize,
    ms: f64,
}

fn run(e: &dyn Engine, msgs: &[Value], n: usize) -> Out {
    let opts = ChatOpts { add_generation_prompt: true, enable_thinking: Some(true), ..Default::default() };
    let text = e.apply_chat_template(msgs, &opts).unwrap();
    let prompt = e.tokenize(&text, false).unwrap();
    let trace = Arc::new(Mutex::new(Vec::new()));
    let req = GenRequest { prompt: prompt.clone(), max_tokens: n, trace_logits: Some(trace.clone()), ..Default::default() };
    let (mut ids, mut cached) = (vec![], 0);
    let t0 = Instant::now();
    e.generate(req, &mut |ev| match ev {
        GenEvent::Token { id, .. } => ids.push(id),
        GenEvent::Done { cached_tokens, .. } => cached = cached_tokens,
        GenEvent::Error { message } => panic!("{message}"),
    })
    .unwrap();
    let logits = trace.lock().unwrap().clone();
    Out { ids, logits, cached, prompt: prompt.len(), ms: t0.elapsed().as_secs_f64() * 1e3 }
}

fn bits_equal(a: &Out, b: &Out) -> (bool, Option<usize>) {
    let first = a.logits.iter().zip(&b.logits).position(|(x, y)| x.iter().zip(y).any(|(p, q)| p.to_bits() != q.to_bits()));
    (a.ids == b.ids && a.logits.len() == b.logits.len() && first.is_none(), first)
}

fn corpus(skip: usize, chars: usize) -> String {
    let t = include_str!("../../../tools/prompts/corpus_docs.txt");
    t.chars().skip(skip).take(chars).collect()
}

#[test]
fn prefix_store_restore_is_bitwise_cold_cuda() {
    if !enabled() {
        return;
    }
    if !intarsia_backend_cuda::AVAILABLE {
        eprintln!("SKIP: built without CUDA");
        return;
    }
    if !model_dir().join("config.json").exists() {
        eprintln!("SKIP: no model at {}", model_dir().display());
        return;
    }
    let gen = 48;
    let sys = json!({"role": "system", "content": "You are a careful reading companion. Keep answers short."});
    let u1 = json!({"role": "user", "content": format!("{}\n\nWhat is this passage about?", corpus(1000, 6000))});
    let other = vec![json!({"role": "user", "content": format!("{}\n\nSummarise.", corpus(40_000, 3000))})];

    let a = engine(true);
    let t1 = run(&*a, &[sys.clone(), u1.clone()], gen);
    let answer = a.detokenize(&t1.ids).unwrap();
    let o = run(&*a, &other, 8);
    let msgs2 = vec![
        sys,
        u1,
        json!({"role": "assistant", "content": answer}),
        json!({"role": "user", "content": format!("{}\n\nAnd this one?", corpus(9000, 2000))}),
    ];
    let warm = run(&*a, &msgs2, gen);
    let snap = a.telemetry().prefix_store.unwrap();
    drop(a);
    let cold_on = run(&*engine(true), &msgs2, gen);
    let cold_off = run(&*engine(false), &msgs2, gen);
    let (eq_on, d_on) = bits_equal(&warm, &cold_on);
    let (eq_off, d_off) = bits_equal(&warm, &cold_off);
    println!(
        "PREFIX-STORE-CUDA model {} turn1 {} tok ({:.0} ms), other {} tok, turn2 {} tok: restored {} cached in {:.0} ms vs cold {:.0} ms (store on) / {:.0} ms (store off)",
        model_dir().display(),
        t1.prompt,
        t1.ms,
        o.prompt,
        warm.prompt,
        warm.cached,
        warm.ms,
        cold_on.ms,
        cold_off.ms
    );
    println!("PREFIX-STORE-CUDA restored vs cold(store on): bitwise {eq_on} (first differing step {d_on:?})");
    println!("PREFIX-STORE-CUDA restored vs cold(store off): bitwise {eq_off} (first differing step {d_off:?}); ids equal {}", warm.ids == cold_off.ids);
    println!("PREFIX-STORE-CUDA store: {}", serde_json::to_string(&snap).unwrap());
    assert!(warm.cached > t1.prompt / 2, "turn 2 restored only {} tokens", warm.cached);
    assert_eq!(cold_on.cached, 0);
    assert!(eq_on, "restored turn 2 differs from cold (store on) at step {d_on:?}");
}
