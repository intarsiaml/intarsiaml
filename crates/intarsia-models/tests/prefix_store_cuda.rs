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
//! `prefix_store_batched_is_single_engine_cuda` (slice 1 of the store in the batched scheduler): two chats, two turns each, every
//! round's two requests in flight together on a `--max-batch 2` engine with the store; each reply must equal the single-sequence
//! engine's (store on, the same requests one at a time) bit for bit, and the second turns must restore from the store.
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
    engine_b(store, 1)
}

fn engine_b(store: bool, max_batch: usize) -> Arc<dyn Engine> {
    let backend = intarsia_backend_cuda::create(0).unwrap();
    let opts = EngineOptions {
        // INTARSIA_TEST_KV_DTYPE=fp8_e4m3 (lane fp8kv): the opt-in fp8 cache (default: the recipe's bf16)
        load: LoadOptions {
            max_seq: 8192,
            profile_every: 0,
            kv_dtype: std::env::var("INTARSIA_TEST_KV_DTYPE").ok().map(|v| intarsia_core::DType::parse(&v).unwrap()),
            ..Default::default()
        },
        prefix_store: store.then(|| PrefixStoreConfig { host_bytes: 16 << 30, ..Default::default() }),
        max_batch: Some(max_batch),
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

fn chat_prompt(e: &dyn Engine, msgs: &[Value]) -> Vec<u32> {
    let opts = ChatOpts { add_generation_prompt: true, enable_thinking: Some(true), ..Default::default() };
    let text = e.apply_chat_template(msgs, &opts).unwrap();
    e.tokenize(&text, false).unwrap()
}

/// Submit every chat's request at once (all in flight on the batched engine), then collect each reply.
fn run_together(e: &dyn Engine, chats: &[Vec<Value>], n: usize) -> Vec<Out> {
    let t0 = Instant::now();
    let pending: Vec<_> = chats
        .iter()
        .map(|msgs| {
            let prompt = chat_prompt(e, msgs);
            let trace = Arc::new(Mutex::new(Vec::new()));
            let req = GenRequest { prompt: prompt.clone(), max_tokens: n, trace_logits: Some(trace.clone()), ..Default::default() };
            (e.submit(req).unwrap(), trace, prompt.len())
        })
        .collect();
    pending
        .into_iter()
        .map(|(rx, trace, plen)| {
            let (mut ids, mut cached) = (vec![], usize::MAX);
            for ev in rx.iter() {
                match ev {
                    GenEvent::Token { id, .. } => ids.push(id),
                    GenEvent::Done { cached_tokens, .. } => {
                        cached = cached_tokens;
                        break;
                    }
                    GenEvent::Error { message } => panic!("{message}"),
                }
            }
            assert_ne!(cached, usize::MAX, "no Done");
            let logits = trace.lock().unwrap().clone();
            Out { ids, logits, cached, prompt: plen, ms: t0.elapsed().as_secs_f64() * 1e3 }
        })
        .collect()
}

fn run(e: &dyn Engine, msgs: &[Value], n: usize) -> Out {
    let prompt = chat_prompt(e, msgs);
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

/// How fast a device-to-host copy is on this machine (the prefix store's settle cost): 64 MiB into a fresh `Vec`, into
/// a touched `Vec`, and as 512 segments of 128 KiB. Opt-in like the gate.
#[test]
fn cuda_d2h_bandwidth_report() {
    if !enabled() || !intarsia_backend_cuda::AVAILABLE {
        return;
    }
    use intarsia_core::backend::MemKind;
    let b = intarsia_backend_cuda::create(0).unwrap();
    let n = 64 << 20;
    let buf = b.alloc(n, MemKind::Device).unwrap();
    b.upload(&buf, 0, &vec![7u8; n], 0).unwrap();
    b.synchronize(0).unwrap();
    let t = Instant::now();
    let mut v = vec![0u8; n];
    b.download(&buf, 0, &mut v, 0).unwrap();
    let fresh = t.elapsed().as_secs_f64();
    let t = Instant::now();
    b.download(&buf, 0, &mut v, 0).unwrap();
    let touched = t.elapsed().as_secs_f64();
    let t = Instant::now();
    for i in 0..512 {
        b.download(&buf, i << 17, &mut v[i << 17..(i + 1) << 17], 0).unwrap();
    }
    let seg = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let mut w = vec![0u8; n];
    w.copy_from_slice(&v);
    let memcpy = t.elapsed().as_secs_f64();
    let gb = n as f64 / 1e9;
    println!(
        "D2H 64 MiB: fresh Vec {:.2} GB/s, touched Vec {:.2} GB/s, 512 x 128 KiB {:.2} GB/s; host memcpy {:.2} GB/s",
        gb / fresh,
        gb / touched,
        gb / seg,
        gb / memcpy
    );
}

#[test]
fn prefix_store_batched_is_single_engine_cuda() {
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
    let mut chats: Vec<Vec<Value>> = vec![
        vec![sys.clone(), json!({"role": "user", "content": format!("{}\n\nWhat is this passage about?", corpus(1000, 6000))})],
        vec![sys, json!({"role": "user", "content": format!("{}\n\nSummarise.", corpus(40_000, 4000))})],
    ];
    let follow = [corpus(9000, 2000), corpus(50_000, 2500)];
    let b = engine_b(true, 2);
    let mut rounds: Vec<(Vec<Vec<Value>>, Vec<Out>)> = Vec::new();
    for round in 0..2 {
        let outs = run_together(&*b, &chats, gen);
        for (i, o) in outs.iter().enumerate() {
            println!("PREFIX-STORE-CUDA-BATCHED round {round} chat {i}: {} tok, cached {}, {:.0} ms", o.prompt, o.cached, o.ms);
        }
        let next: Vec<Vec<Value>> = chats
            .iter()
            .zip(&outs)
            .zip(&follow)
            .map(|((c, o), f)| {
                let mut c = c.clone();
                c.push(json!({"role": "assistant", "content": b.detokenize(&o.ids).unwrap()}));
                c.push(json!({"role": "user", "content": format!("{f}\n\nAnd this one?")}));
                c
            })
            .collect();
        rounds.push((std::mem::replace(&mut chats, next), outs));
    }
    let snap = b.telemetry().prefix_store.unwrap();
    println!("PREFIX-STORE-CUDA-BATCHED store: {}", serde_json::to_string(&snap).unwrap());
    drop(b);
    let solo = engine(true);
    let mut all_eq = true;
    for (round, (chats, outs)) in rounds.iter().enumerate() {
        for (i, (msgs, got)) in chats.iter().zip(outs).enumerate() {
            let want = run(&*solo, msgs, gen);
            let (eq, d) = bits_equal(got, &want);
            println!(
                "PREFIX-STORE-CUDA-BATCHED round {round} chat {i}: batched vs single-sequence (store on) bitwise {eq} (first differing step {d:?}); ids equal {}; cached {} vs {}",
                got.ids == want.ids,
                got.cached,
                want.cached
            );
            all_eq &= eq;
        }
    }
    let r2 = &rounds[1].1;
    assert!(r2.iter().all(|o| o.cached > o.prompt / 2), "second turns restored too little: {:?}", r2.iter().map(|o| (o.cached, o.prompt)).collect::<Vec<_>>());
    assert!(snap.hits >= 2 && snap.inserts > 0, "{snap:?}");
    assert!(all_eq, "a batched reply differs from the single-sequence engine's (store on)");
}

/// Distinct prompts through a small host tier: every settle evicts, so demotes either take fresh host buffers (a page
/// fault per 4 KiB) or recycle the evicted ones. Prints the demote and capture throughput from the stats
/// (`INTARSIA_PREFIX_STORE_DEBUG=1` adds the per-object lines). Opt-in like the gate.
#[test]
fn cuda_host_pressure_report() {
    if !enabled() || !intarsia_backend_cuda::AVAILABLE || !model_dir().join("config.json").exists() {
        return;
    }
    let host_mb: u64 = std::env::var("INTARSIA_TEST_STORE_HOST_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
    let backend = intarsia_backend_cuda::create(0).unwrap();
    let opts = EngineOptions {
        load: LoadOptions { max_seq: 8192, profile_every: 0, ..Default::default() },
        prefix_store: Some(PrefixStoreConfig { host_bytes: host_mb << 20, ..Default::default() }),
        max_batch: Some(1),
        ..Default::default()
    };
    let e = load_engine(&model_dir(), backend, &opts).unwrap().engine;
    let t0 = Instant::now();
    let mut prompt_tokens = 0;
    for i in 0..12 {
        let msgs = vec![json!({"role": "user", "content": format!("{}\n\nSummarise.", corpus(2000 + i * 5000, 6000))})];
        prompt_tokens += run(&*e, &msgs, 4).prompt;
    }
    let wall = t0.elapsed().as_secs_f64();
    let snap = e.telemetry().prefix_store.unwrap();
    let gb = 1e9;
    println!(
        "PREFIX-STORE-CUDA pressure: {} prompts {} tok in {:.2} s; inserts {} ({:.0} MiB) insert {:.3} s of which settle {:.3} s; host {} objs {:.0} MiB / {} MiB, {} evictions",
        12,
        prompt_tokens,
        wall,
        snap.inserts,
        snap.insert_bytes as f64 / 1048576.0,
        snap.insert_seconds_total,
        snap.settle_seconds_total,
        snap.host.objects,
        snap.host.bytes as f64 / 1048576.0,
        host_mb,
        snap.host.evictions
    );
    println!(
        "PREFIX-STORE-CUDA pressure: settle {:.2} GB/s of captured bytes",
        snap.insert_bytes as f64 / gb / snap.settle_seconds_total.max(1e-9)
    );
    assert!(snap.host.bytes <= host_mb << 20, "host tier over budget: {} > {} MiB", snap.host.bytes, host_mb);
}
