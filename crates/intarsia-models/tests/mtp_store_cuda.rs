//! Lane mtp-store on a real model on the CUDA backend (the Qwen3.6-35B-A3B NVFP4 by default, `INTARSIA_CUDA_TEST_MODEL` as in
//! `prefix_store_cuda.rs`): `--spec mtp` on prefix-store requests. The store restores the main model's state only; the MTP
//! head's context restarts where a request resumes (`MtpState::base`), so a restored request drafts.
//!
//! Two chats, three turns each, greedy, through the model's chat template, the chats alternating so every later turn restores
//! its history from the store:
//!
//! 1. reference: the single-sequence engine, store on, `--spec off`;
//! 2. the single-sequence engine, store on, the recipe's MTP default (which the store no longer skips on CUDA): every reply's
//!    ids and cached tokens must equal the reference's, and the later turns must draft after their restore;
//! 3. the single-sequence engine, store OFF, MTP on, over the same prompts: reported only, the acceptance with the head's whole
//!    context (the store-off chunk schedule differs on CUDA, so ids are not compared);
//! 4. `--max-batch 2`, store on, the recipe's MTP default (batched MTP N = 1): each round's two requests in flight together,
//!    then every request once more alone (each restores, then drafts): every reply's ids must equal the reference's.
//!
//! Prints the acceptance rate per request (`MTP-STORE-CUDA` lines). Opt-in: `INTARSIA_TEST_REAL_MODEL=1`; take a fleetq slot.
#![cfg(not(target_os = "macos"))]
use intarsia_core::engine::ChatOpts;
use intarsia_core::model::LoadOptions;
use intarsia_core::prefix_store::PrefixStoreConfig;
use intarsia_core::{Engine, GenEvent, GenRequest};
use intarsia_models::{load_engine, EngineOptions};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
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

/// `mtp`: the recipe's MTP default applies (no `--spec` given); else `--spec off`.
fn engine(store: bool, mtp: bool, max_batch: usize) -> Arc<dyn Engine> {
    let backend = intarsia_backend_cuda::create(0).unwrap();
    let opts = EngineOptions {
        load: LoadOptions { max_seq: 8192, profile_every: 0, ..Default::default() },
        prefix_store: store.then(|| PrefixStoreConfig { host_bytes: 16 << 30, ..Default::default() }),
        max_batch: Some(max_batch),
        spec_auto: mtp,
        ..Default::default()
    };
    load_engine(&model_dir(), backend, &opts).unwrap().engine
}

#[derive(Clone)]
struct Out {
    ids: Vec<u32>,
    cached: usize,
    prompt: usize,
    prefill_ms: f64,
    decode_ms: f64,
}

fn chat_prompt(e: &dyn Engine, msgs: &[Value]) -> Vec<u32> {
    let opts = ChatOpts { add_generation_prompt: true, enable_thinking: Some(true), ..Default::default() };
    let text = e.apply_chat_template(msgs, &opts).unwrap();
    e.tokenize(&text, false).unwrap()
}

fn greedy(prompt: &[u32], n: usize) -> GenRequest {
    GenRequest { prompt: prompt.to_vec(), max_tokens: n, temperature: 0.0, ..Default::default() }
}

fn collect(rx: std::sync::mpsc::Receiver<GenEvent>, plen: usize) -> Out {
    let mut ids = vec![];
    for ev in rx.iter() {
        match ev {
            GenEvent::Token { id, .. } => ids.push(id),
            GenEvent::Done { cached_tokens, prefill_ms, decode_ms, .. } => {
                return Out { ids, cached: cached_tokens, prompt: plen, prefill_ms, decode_ms };
            }
            GenEvent::Error { message } => panic!("{message}"),
        }
    }
    panic!("no Done");
}

/// (verify passes, drafted, accepted) so far.
fn spec(e: &dyn Engine) -> (u64, u64, u64) {
    let t = e.telemetry();
    (t.spec_steps, t.spec_proposed, t.spec_accepted)
}

/// One request alone; returns the reply and the request's (passes, drafted, accepted).
fn run(e: &dyn Engine, prompt: &[u32], n: usize) -> (Out, (u64, u64, u64)) {
    let s0 = spec(e);
    let out = collect(e.submit(greedy(prompt, n)).unwrap(), prompt.len());
    let s1 = spec(e);
    (out, (s1.0 - s0.0, s1.1 - s0.1, s1.2 - s0.2))
}

fn rate(s: (u64, u64, u64)) -> f64 {
    if s.1 == 0 {
        0.0
    } else {
        s.2 as f64 / s.1 as f64
    }
}

fn tps(o: &Out) -> f64 {
    if o.decode_ms > 0.0 {
        (o.ids.len().saturating_sub(1)) as f64 / (o.decode_ms / 1e3)
    } else {
        0.0
    }
}

fn corpus(skip: usize, chars: usize) -> String {
    let t = include_str!("../../../tools/prompts/corpus_docs.txt");
    t.chars().skip(skip).take(chars).collect()
}

fn line(tag: &str, i: usize, o: &Out, s: (u64, u64, u64)) {
    println!(
        "MTP-STORE-CUDA {tag} request {i}: prompt {} cached {} | {} tokens, ttft {:.0} ms, decode {:.1} tok/s | {} passes, drafted {}, accepted {} ({:.1} %)",
        o.prompt,
        o.cached,
        o.ids.len(),
        o.prefill_ms,
        tps(o),
        s.0,
        s.1,
        s.2,
        100.0 * rate(s)
    );
}

#[test]
fn mtp_on_store_requests_equals_mtp_off_cuda() {
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
    let gen = 160;
    let rounds = 3;
    let sys = json!({"role": "system", "content": "You are a careful reading companion. Keep answers short."});
    let first = [
        format!("{}\n\nWhat is this passage about?", corpus(1000, 6000)),
        format!("{}\n\nSummarise.", corpus(40_000, 4000)),
    ];
    let follow = [
        [format!("{}\n\nAnd this one?", corpus(9000, 2000)), format!("{}\n\nCompare the two passages.", corpus(15_000, 1500))],
        [format!("{}\n\nAnd this one?", corpus(50_000, 2500)), "Which of the two was clearer, and why?".to_string()],
    ];

    // 1. the reference (store on, spec off); its replies build the later turns' prompts for every arm
    let t0 = Instant::now();
    let off = engine(true, false, 1);
    let mut chats: Vec<Vec<Value>> = first.iter().map(|u| vec![sys.clone(), json!({"role": "user", "content": u})]).collect();
    let mut prompts: Vec<Vec<u32>> = Vec::new();
    let mut want: Vec<Out> = Vec::new();
    for round in 0..rounds {
        for c in 0..chats.len() {
            let p = chat_prompt(&*off, &chats[c]);
            let (o, s) = run(&*off, &p, gen);
            assert_eq!(s, (0, 0, 0), "the reference does not draft");
            line("store-on mtp-off", prompts.len(), &o, s);
            if round + 1 < rounds {
                chats[c].push(json!({"role": "assistant", "content": off.detokenize(&o.ids).unwrap()}));
                chats[c].push(json!({"role": "user", "content": &follow[c][round]}));
            }
            prompts.push(p);
            want.push(o);
        }
    }
    let n_first = chats.len();
    assert!(want[n_first..].iter().all(|o| o.cached > o.prompt / 2), "the later turns restore: {:?}", want.iter().map(|o| (o.cached, o.prompt)).collect::<Vec<_>>());
    drop(off);
    println!("MTP-STORE-CUDA reference done at {:.0} s", t0.elapsed().as_secs_f64());

    // 2. single-sequence engine, store on, the recipe's MTP default
    let on = engine(true, true, 1);
    let mut solo_ok = true;
    let (mut cold, mut restored) = ((0, 0, 0), (0, 0, 0));
    let mut on_outs = Vec::new();
    for (i, (p, w)) in prompts.iter().zip(&want).enumerate() {
        let (o, s) = run(&*on, p, gen);
        line("store-on mtp-on", i, &o, s);
        let eq = o.ids == w.ids && o.cached == w.cached;
        if !eq {
            let at = o.ids.iter().zip(&w.ids).position(|(a, b)| a != b);
            println!("MTP-STORE-CUDA MISMATCH single request {i}: first differing token {at:?}, cached {} vs {}", o.cached, w.cached);
        }
        solo_ok &= eq;
        let acc = if i < n_first { &mut cold } else { &mut restored };
        *acc = (acc.0 + s.0, acc.1 + s.1, acc.2 + s.2);
        on_outs.push(o);
    }
    let snap = on.telemetry().prefix_store.unwrap();
    println!("MTP-STORE-CUDA single store: {}", serde_json::to_string(&snap).unwrap());
    drop(on);
    println!(
        "MTP-STORE-CUDA single engine: ids and cached equal mtp-off {solo_ok}; acceptance cold turns {:.1} % ({} / {}), restored turns {:.1} % ({} / {})",
        100.0 * rate(cold),
        cold.2,
        cold.1,
        100.0 * rate(restored),
        restored.2,
        restored.1
    );

    // 3. store off, MTP on: the head follows every prompt from position 0 (reported, not asserted)
    let full = engine(false, true, 1);
    let (mut f_cold, mut f_later) = ((0, 0, 0), (0, 0, 0));
    for (i, (p, w)) in prompts.iter().zip(&want).enumerate() {
        let (o, s) = run(&*full, p, gen);
        line("store-off mtp-on", i, &o, s);
        println!("MTP-STORE-CUDA store-off request {i}: ids equal the store-on reference {}", o.ids == w.ids);
        let acc = if i < n_first { &mut f_cold } else { &mut f_later };
        *acc = (acc.0 + s.0, acc.1 + s.1, acc.2 + s.2);
    }
    drop(full);
    println!(
        "MTP-STORE-CUDA whole head context (store off): acceptance first turns {:.1} % ({} / {}), later turns {:.1} % ({} / {})",
        100.0 * rate(f_cold),
        f_cold.2,
        f_cold.1,
        100.0 * rate(f_later),
        f_later.2,
        f_later.1
    );

    // 4. --max-batch 2, store on, the recipe's MTP default
    let b = engine(true, true, 2);
    let mut batched_ok = true;
    let s0 = spec(&*b);
    for round in 0..rounds {
        let idx: Vec<usize> = (0..n_first).map(|c| round * n_first + c).collect();
        let rxs: Vec<_> = idx.iter().map(|&i| (i, b.submit(greedy(&prompts[i], gen)).unwrap())).collect();
        for (i, rx) in rxs {
            let o = collect(rx, prompts[i].len());
            let eq = o.ids == want[i].ids;
            println!(
                "MTP-STORE-CUDA batched together request {i}: prompt {} cached {} | ttft {:.0} ms, decode {:.1} tok/s | ids equal mtp-off {eq}",
                o.prompt,
                o.cached,
                o.prefill_ms,
                tps(&o)
            );
            batched_ok &= eq;
        }
    }
    let s1 = spec(&*b);
    let together = (s1.0 - s0.0, s1.1 - s0.1, s1.2 - s0.2);
    let mut alone = (0, 0, 0);
    let mut alone_cached = true;
    for (i, (p, w)) in prompts.iter().zip(&want).enumerate() {
        let (o, s) = run(&*b, p, gen);
        line("batched alone store-on mtp-on", i, &o, s);
        let eq = o.ids == w.ids;
        if !eq {
            let at = o.ids.iter().zip(&w.ids).position(|(a, b)| a != b);
            println!("MTP-STORE-CUDA MISMATCH batched request {i}: first differing token {at:?}");
        }
        batched_ok &= eq;
        alone_cached &= o.cached > o.prompt / 2;
        alone = (alone.0 + s.0, alone.1 + s.1, alone.2 + s.2);
    }
    let bsnap = b.telemetry().prefix_store.unwrap();
    println!("MTP-STORE-CUDA batched store: {}", serde_json::to_string(&bsnap).unwrap());
    drop(b);
    println!(
        "MTP-STORE-CUDA batched engine: ids equal mtp-off {batched_ok}; together: {} passes, acceptance {:.1} % ({} / {}); alone after a restore: {} passes, acceptance {:.1} % ({} / {})",
        together.0,
        100.0 * rate(together),
        together.2,
        together.1,
        alone.0,
        100.0 * rate(alone),
        alone.2,
        alone.1
    );
    println!("MTP-STORE-CUDA total {:.0} s", t0.elapsed().as_secs_f64());

    assert!(solo_ok, "single-sequence engine: a store-on MTP-on reply differs from the store-on MTP-off reply");
    assert!(snap.hits as usize >= prompts.len() - n_first, "single-sequence engine: the drafting requests restored: {snap:?}");
    assert!(restored.1 > 0 && restored.2 > 0, "single-sequence engine: the restored turns drafted and accepted: {restored:?}");
    assert!(batched_ok, "batched engine: a store-on MTP-on reply differs from the store-on MTP-off reply");
    assert!(alone_cached && bsnap.hits > 0, "batched engine: the requests restored: {bsnap:?}");
    assert!(alone.1 > 0 && alone.2 > 0, "batched engine: the restored requests drafted and accepted: {alone:?}");
    let _ = on_outs;
}
