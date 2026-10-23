//! CUDA gate G-C and the engine-level identity on the real models (docs/results/2026-10-23-rb2-cuda-batching.md):
//!
//! * `real_35b_batched_decode_is_bitwise_the_solo_step`: Qwen3.6-35B-A3B NVFP4. `Model::forward_batch` of N = 2, 3, 4, 8, 16
//!   sequences at mixed context lengths, greedy and seeded-sampled rows, 130 consecutive steps each, must return for every
//!   row the bits of that sequence's solo decode step (logits bit for bit, the on-device argmax token for token) and leave
//!   identical state (`common/batch_identity.rs`); decode rows beside prefill chunks too.
//! * `real_35b_batched_engine_equals_the_local_engine`: the whole `BatchedEngine` (scheduler, slots, snapshots, prefill
//!   chunks, sampling) over greedy and seeded-sampled requests: at one stream it equals `LocalEngine` token for token, and
//!   with 4 and 16 requests in flight each request's tokens still equal its solo run. (Prefill of a busy engine runs in
//!   smaller chunks than a solo one: for the greedy prompts below the chunk schedule is the same, see the test.)
//! * `real_qwen3_8b_batched_decode_is_bitwise_the_solo_step`: the dense bf16 model on the same CUDA kernels.
//!
//! Opt-in (loads 20 GB onto the GPU: take a fleetq slot on the Spark first): `INTARSIA_QWEN36_DIR` (35B checkpoint) and
//! `INTARSIA_QWEN3_DIR` (8B); a test without its directory prints SKIPPED (and panics under `INTARSIA_REQUIRE_GPU=1`).
use intarsia_core::model::{LoadOptions, Model};
use intarsia_core::{Engine, FinishReason, GenEvent, GenRequest, Telemetry};
use intarsia_models::load::SchemedSource;
use intarsia_models::{load_engine, read_model_spec, EngineOptions};
use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::Duration;

#[path = "common/batch_identity.rs"]
mod batch_identity;

fn skip(why: &str) {
    eprintln!("SKIPPED: {why}");
    if std::env::var("INTARSIA_REQUIRE_GPU").is_ok_and(|v| v == "1") {
        panic!("SKIPPED ({why}) but INTARSIA_REQUIRE_GPU=1");
    }
}

fn model_dir(var: &str) -> Option<PathBuf> {
    match std::env::var_os(var).map(PathBuf::from) {
        Some(d) if d.join("config.json").exists() => Some(d),
        Some(d) => {
            skip(&format!("{var}={} has no config.json", d.display()));
            None
        }
        None => {
            skip(&format!("{var} not set"));
            None
        }
    }
}

fn cuda() -> Option<Arc<dyn intarsia_core::backend::Backend>> {
    match intarsia_backend_cuda::create(0) {
        Ok(b) => Some(b),
        Err(e) => {
            skip(&format!("no usable CUDA device ({e})"));
            None
        }
    }
}

/// The model of `dir` on the GPU with batching enabled for `rows` sequences of `max_seq` tokens.
fn load(dir: &PathBuf, backend: Arc<dyn intarsia_core::backend::Backend>, rows: usize, max_seq: usize) -> Model {
    load_with(dir, backend, rows, max_seq, 0).0
}

fn load_with(
    dir: &PathBuf,
    backend: Arc<dyn intarsia_core::backend::Backend>,
    rows: usize,
    max_seq: usize,
    profile_every: u32,
) -> (Model, Arc<Telemetry>) {
    let (_, recipe, dims, _) = read_model_spec(dir, &EngineOptions::default()).unwrap();
    let weights = SchemedSource::open(dir).unwrap();
    let chunk = recipe.resolve_prefill_chunk(None, "cuda").value;
    let opts = LoadOptions {
        max_seq,
        prefill_chunk: chunk,
        profile_every,
        // the Spark's `free` after the weights is the page-cache-reduced figure (about 2 GB on a shared box): take most of it
        kv_fraction: 0.95,
        ..Default::default()
    };
    let tel = Arc::new(Telemetry::new());
    let mut m = Model::load(backend, &recipe, &dims, &weights, &opts, tel.clone()).unwrap();
    assert!(m.batch_capable(), "the cuda backend registers the row-exact kernels");
    eprintln!("plan before batching: {:?}", m.plan);
    eprintln!("recurrent bytes per state {}", m.recurrent_bytes_per_state());
    m.allow_recurrent_batch().unwrap();
    m.enable_batch(rows).unwrap();
    // plan resolution dry-runs every batch size at both ends of the context: a batched op that would pick a kernel that is
    // neither the solo step's nor a row-exact one fails here (`plan::check_batch_ops`)
    m.configure_plan(&Default::default(), None).unwrap();
    assert!(m.plan.kv_cap_tokens >= max_seq, "{} tokens per sequence", m.plan.kv_cap_tokens);
    (m, tel)
}

fn gate(m: &Model, lens: Vec<usize>, batches: Vec<usize>, steps: usize, vocab_range: (u32, u32)) {
    let expect: usize = batches.iter().sum::<usize>() * steps;
    let t0 = std::time::Instant::now();
    let n = batch_identity::run(m, &batch_identity::Cfg { lens, batches, steps, vocab_range });
    eprintln!("{n} (row, step) pairs bitwise identical in {:.1} s", t0.elapsed().as_secs_f64());
    assert_eq!(n, expect);
}

#[test]
fn real_35b_batched_decode_is_bitwise_the_solo_step() {
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), cuda()) else { return };
    let m = load(&dir, be, 16, 2048);
    eprintln!("batch plan notes: {:?}", m.plan.notes);
    // N = 2, 3, 4 (the gate), then 8 and 16; 130 consecutive batched steps per size on the same, growing states
    let lens = vec![5, 37, 130, 300, 17, 64, 9, 222, 400, 3, 77, 150, 28, 11, 90, 45];
    gate(&m, lens, vec![2, 3, 4, 8, 16], 130, (1000, 30000));
    // decode rows beside prefill chunks (the engine's busy chunk is 256): the decode rows stay bitwise, the prefilled state
    // equals the prompt prefilled alone with the same chunks
    let n = batch_identity::run_with_prefill(&m, &[37, 130, 300, 17], 700, 256, (1000, 30000));
    assert_eq!(n, 4 * 2 * 700usize.div_ceil(256));
}

#[test]
fn real_qwen3_8b_batched_decode_is_bitwise_the_solo_step() {
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN3_DIR"), cuda()) else { return };
    let m = load(&dir, be, 16, 2048);
    let lens = vec![5, 37, 130, 300, 17, 64, 9, 222, 400, 3, 77, 150, 28, 11, 90, 45];
    gate(&m, lens, vec![2, 3, 4, 8, 16], 130, (1000, 30000));
}

// ---- engine level ---------------------------------------------------------------------------------------------------

#[derive(Debug, PartialEq)]
struct Out {
    ids: Vec<u32>,
    text: String,
    reason: Option<FinishReason>,
}

fn collect(rx: &Receiver<GenEvent>) -> Out {
    let (mut ids, mut text) = (vec![], String::new());
    loop {
        match rx.recv_timeout(Duration::from_secs(600)).expect("engine stalled") {
            GenEvent::Token { id, text: t } => {
                ids.push(id);
                text.push_str(&t);
            }
            GenEvent::Done { finish_reason, .. } => return Out { ids, text, reason: Some(finish_reason) },
            GenEvent::Error { message } => panic!("request failed: {message}"),
        }
    }
}

const PROMPTS: [&str; 6] = [
    "The history of the printing press begins",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Photosynthesis is the process by which",
    "In 1969, the Apollo 11 mission",
    "Explain, step by step, why the sky is blue. First,",
    "The three laws of thermodynamics state that",
];

fn requests(engine: &dyn Engine, n: usize, max_tokens: usize) -> Vec<GenRequest> {
    (0..n)
        .map(|i| {
            let mut ids = engine.tokenize(PROMPTS[i % PROMPTS.len()], false).unwrap();
            if i >= PROMPTS.len() {
                // distinct prompts beyond the six texts: a different tail
                ids.extend(engine.tokenize(&format!(" ({i})"), false).unwrap());
            }
            let mut r = GenRequest { prompt: ids, max_tokens, ignore_eos: true, ..Default::default() };
            if i % 4 == 3 {
                r.temperature = 0.8;
                r.top_k = 40;
                r.top_p = 0.95;
                r.seed = Some(11 + i as u64);
            }
            r
        })
        .collect()
}

#[test]
fn real_35b_batched_engine_equals_the_local_engine() {
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), cuda()) else { return };
    let mk = |max_batch: Option<usize>| EngineOptions {
        load: LoadOptions { max_seq: 2048, kv_fraction: 0.95, profile_every: 0, ..Default::default() },
        max_batch,
        ..Default::default()
    };
    // the reference: every request alone on the single-sequence engine
    let reference: Vec<Out> = {
        let l = load_engine(&dir, be.clone(), &mk(None)).unwrap();
        assert_eq!(l.max_batch, 1);
        let reqs = requests(&*l.engine, 20, 130);
        reqs.into_iter().map(|r| collect(&l.engine.submit(r).unwrap())).collect()
    };
    assert!(reference.iter().all(|o| o.ids.len() == 130));
    for slots in [4usize, 16] {
        let l = load_engine(&dir, be.clone(), &mk(Some(slots))).unwrap();
        assert_eq!(l.max_batch, slots);
        // one stream at a time: the batched engine at N = 1 is the local engine
        for (i, r) in requests(&*l.engine, 3, 130).into_iter().enumerate() {
            assert_eq!(collect(&l.engine.submit(r).unwrap()), reference[i], "one stream, request {i}, {slots} slots");
        }
        // all at once: 20 requests on `slots` slots (queueing, overlap, slot reuse)
        let rxs: Vec<_> = requests(&*l.engine, 20, 130).into_iter().map(|r| l.engine.submit(r).unwrap()).collect();
        for (i, rx) in rxs.iter().enumerate() {
            let got = collect(rx);
            let want = &reference[i];
            // prefill of a request that starts while others decode runs in busy chunks; a prompt this short is one chunk
            // either way, so the whole output is compared
            assert_eq!(&got, want, "concurrent request {i} on {slots} slots");
        }
    }
}

// ---- timing (not a gate): `cargo test ... -- --ignored --nocapture real_35b_batch_step_timing` --------------------------

/// Wall time of `forward_batch` (all rows `LastArgmax`, the serving mode) at B = 1, 2, 4, 8, 16 over 24 steps, the solo
/// `forward` step beside it, and the per-op profile of one profiled B = 4 and B = 16 step.
#[test]
#[ignore]
fn real_35b_batch_step_timing() {
    use intarsia_core::model::{HeadMode, StepOut};
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), cuda()) else { return };
    let m = load(&dir, be, 16, 2048);
    let tok = |i: usize| 1000 + 37 * i as u32;
    let mut states: Vec<_> = (0..16)
        .map(|i| {
            let mut st = m.new_state().unwrap();
            let p: Vec<u32> = (0..900 + 5 * i).map(|j| 1000 + ((j * 7 + i * 13) % 20000) as u32).collect();
            m.forward(&mut st, &p, HeadMode::None, None).unwrap();
            st
        })
        .collect();
    let steps = 24;
    let t0 = std::time::Instant::now();
    for _ in 0..steps {
        let StepOut::Token(_) = m.forward(&mut states[0], &[tok(0)], HeadMode::LastArgmax, None).unwrap() else { panic!() };
    }
    let solo = t0.elapsed().as_secs_f64() * 1e3 / steps as f64;
    eprintln!("TIMING solo step {solo:.2} ms ({:.1} tok/s)", 1e3 / solo);
    for b in [2usize, 4, 8, 16] {
        let ids: Vec<u32> = (0..b).map(tok).collect();
        let modes = vec![HeadMode::LastArgmax; b];
        let t0 = std::time::Instant::now();
        for _ in 0..steps {
            let mut refs: Vec<&mut _> = states.iter_mut().take(b).collect();
            m.forward_batch(&mut refs, &ids, &modes).unwrap();
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / steps as f64;
        eprintln!("TIMING batch B={b:>2}: {ms:.2} ms/step, {:.1} tok/s aggregate ({:.2}x solo step)", b as f64 * 1e3 / ms, ms / solo);
    }
}

/// Per-op calls and time of 8 profiled batched steps at B = 4 and at B = 16 (profiling syncs after each op, so the sum
/// is above the unprofiled step; the shares are what to read).
#[test]
#[ignore]
fn real_35b_batch_step_profile() {
    use intarsia_core::model::HeadMode;
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), cuda()) else { return };
    let (m, tel) = load_with(&dir, be, 16, 2048, 1);
    let mut states: Vec<_> = (0..16)
        .map(|i| {
            let mut st = m.new_state().unwrap();
            let p: Vec<u32> = (0..900 + 5 * i).map(|j| 1000 + ((j * 7 + i * 13) % 20000) as u32).collect();
            m.forward(&mut st, &p, HeadMode::None, None).unwrap();
            st
        })
        .collect();
    for b in [1usize, 4, 16] {
        let before = tel.snapshot().ops;
        let ids: Vec<u32> = (0..b).map(|i| 1000 + 37 * i as u32).collect();
        let modes = vec![HeadMode::LastArgmax; b];
        let t0 = std::time::Instant::now();
        for _ in 0..8 {
            let mut refs: Vec<&mut _> = states.iter_mut().take(b).collect();
            m.forward_batch(&mut refs, &ids, &modes).unwrap();
        }
        let wall = t0.elapsed().as_secs_f64() * 1e3 / 8.0;
        let after = tel.snapshot().ops;
        let mut rows: Vec<(String, u64, f64)> = after
            .iter()
            .map(|(k, a)| {
                let b0 = before.get(k);
                (k.clone(), a.calls - b0.map_or(0, |x| x.calls), a.time_ms - b0.map_or(0.0, |x| x.time_ms))
            })
            .filter(|r| r.1 > 0)
            .collect();
        rows.sort_by(|x, y| y.2.partial_cmp(&x.2).unwrap());
        let tot: f64 = rows.iter().map(|r| r.2).sum::<f64>() / 8.0;
        eprintln!("PROFILE B={b}: wall {wall:.2} ms/step (profiled), op time sum {tot:.2} ms/step");
        for (k, calls, ms) in rows.iter().take(14) {
            eprintln!("PROFILE B={b}   {k:<28} {:>6.1} calls/step {:>7.3} ms/step", *calls as f64 / 8.0, ms / 8.0);
        }
    }
}
