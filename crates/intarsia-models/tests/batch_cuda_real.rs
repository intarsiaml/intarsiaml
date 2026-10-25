//! CUDA (and, with `INTARSIA_BATCH_TEST_BACKEND=hip`, HIP) gate G-C and the engine-level identity on the real models (docs/results/2026-10-23-rb2-cuda-batching.md):
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

/// The GPU backend: CUDA, or HIP with `INTARSIA_BATCH_TEST_BACKEND=hip` (lane hb, docs/results/2026-10-23-hb-hip-hybrid-batching.md).
fn cuda() -> Option<Arc<dyn intarsia_core::backend::Backend>> {
    let hip = std::env::var("INTARSIA_BATCH_TEST_BACKEND").is_ok_and(|v| v == "hip");
    let made = if hip { intarsia_backend_hip::create(0) } else { intarsia_backend_cuda::create(0) };
    match made {
        Ok(b) => Some(b),
        Err(e) => {
            skip(&format!("no usable {} device ({e})", if hip { "HIP" } else { "CUDA" }));
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
    let chunk = recipe.resolve_prefill_chunk(None, backend.name()).value;
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
    assert!(m.batch_capable(), "the GPU backend registers the row-exact kernels");
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
    // on HIP (lane hb) every N from 2 to the cap, not only the sizes the CUDA gate samples
    let hip = std::env::var("INTARSIA_BATCH_TEST_BACKEND").is_ok_and(|v| v == "hip");
    let sizes: Vec<usize> = if hip { (2..=16).collect() } else { vec![2, 3, 4, 8, 16] };
    gate(&m, lens, sizes, 130, (1000, 30000));
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
    // on HIP (lane hb) every N from 2 to the cap, not only the sizes the CUDA gate samples
    let hip = std::env::var("INTARSIA_BATCH_TEST_BACKEND").is_ok_and(|v| v == "hip");
    let sizes: Vec<usize> = if hip { (2..=16).collect() } else { vec![2, 3, 4, 8, 16] };
    gate(&m, lens, sizes, 130, (1000, 30000));
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
        let StepOut::Token(_) = m.forward(&mut states[0], &[tok(0)], HeadMode::LastArgmax, None).unwrap() else {
            panic!()
        };
    }
    let solo = t0.elapsed().as_secs_f64() * 1e3 / steps as f64;
    eprintln!("TIMING solo step {solo:.2} ms ({:.1} tok/s)", 1e3 / solo);
    // the row counts timed: 2, 4, 8, 16, or INTARSIA_TIMING_ROWS="2,3,4,..." (test-only)
    let rows: Vec<usize> = std::env::var("INTARSIA_TIMING_ROWS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![2, 4, 8, 16]);
    for b in rows {
        let ids: Vec<u32> = (0..b).map(tok).collect();
        let modes = vec![HeadMode::LastArgmax; b];
        let t0 = std::time::Instant::now();
        for _ in 0..steps {
            let mut refs: Vec<&mut _> = states.iter_mut().take(b).collect();
            m.forward_batch(&mut refs, &ids, &modes).unwrap();
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / steps as f64;
        eprintln!(
            "TIMING batch B={b:>2}: {ms:.2} ms/step, {:.1} tok/s aggregate ({:.2}x solo step)",
            b as f64 * 1e3 / ms,
            ms / solo
        );
    }
}

/// B2 (Fable review): the cost of a batched step whose rows are SAMPLED, on the host path (`LastLogits`: one [B, V] download and a
/// host top-k/top-p walk per row) against the device path (`LastTopK(40)`: the k candidates only; opt-in: `INTARSIA_DEVICE_SAMPLING=1` turns it on).
/// Same seeds, so the two draw the same tokens when the states are the same; the token parity is gated by the engine test
/// (`real_35b_batched_engine_equals_the_local_engine`, seeded-sampled requests, run with `INTARSIA_DEVICE_SAMPLING=1`).
#[test]
#[ignore]
fn real_35b_batch_step_sampled_timing() {
    use intarsia_core::model::{HeadMode, StepOut};
    use intarsia_core::sampler::{Sampler, SamplingParams};
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), cuda()) else { return };
        let m = load(&dir, be, 16, 2048);
    let params = SamplingParams { temperature: 0.8, top_k: 40, top_p: 0.9 };
    let mut states: Vec<_> = (0..16)
        .map(|i| {
            let mut st = m.new_state().unwrap();
            let p: Vec<u32> = (0..900 + 5 * i).map(|j| 1000 + ((j * 7 + i * 13) % 20000) as u32).collect();
            m.forward(&mut st, &p, HeadMode::None, None).unwrap();
            st
        })
        .collect();
    let steps = 24;
    for b in [4usize, 16] {
        for device in [false, true] {
            let mut samplers: Vec<Sampler> = (0..b).map(|i| Sampler::new(params.clone(), Some(7 + i as u64))).collect();
            let mut toks: Vec<u32> = (0..b).map(|i| 1000 + 37 * i as u32).collect();
            let modes = vec![if device { HeadMode::LastTopK(40) } else { HeadMode::LastLogits }; b];
            let t0 = std::time::Instant::now();
            for _ in 0..steps {
                let mut refs: Vec<&mut _> = states.iter_mut().take(b).collect();
                let out = m.forward_batch(&mut refs, &toks, &modes).unwrap();
                for (r, o) in out.iter().enumerate() {
                    toks[r] = match o {
                        StepOut::Logits(l) => samplers[r].sample(l) as u32,
                        StepOut::TopK { ids, vals } => samplers[r].sample_topk(ids, vals) as u32,
                        o => panic!("unexpected {o:?}"),
                    };
                }
            }
            let ms = t0.elapsed().as_secs_f64() * 1e3 / steps as f64;
            eprintln!("TIMING sampled B={b:>2} {}: {ms:.2} ms/step incl. host sampling, {:.1} tok/s aggregate", if device { "device top-k" } else { "host      " }, b as f64 * 1e3 / ms);
        }
    }
}

/// Per-op calls and time of 8 profiled batched steps at B = 1, 4 and 16, or at `INTARSIA_TIMING_ROWS="1,8,16"` (profiling
/// syncs after each op, so the sum is above the unprofiled step; the shares are what to read). Two unprofiled-in-the-table
/// warm-up steps per B come first.
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
    let rows_list: Vec<usize> = std::env::var("INTARSIA_TIMING_ROWS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 4, 16]);
    for b in rows_list {
        let ids: Vec<u32> = (0..b).map(|i| 1000 + 37 * i as u32).collect();
        let modes = vec![HeadMode::LastArgmax; b];
        // B = 1 is the solo step (`forward`, eager, the same per-op syncs): the batched path replays a captured graph at one row,
        // whose ops are not timed one by one
        let step = |states: &mut Vec<_>| {
            if b == 1 {
                m.forward(&mut states[0], &ids, HeadMode::LastArgmax, None).unwrap();
            } else {
                let mut refs: Vec<&mut _> = states.iter_mut().take(b).collect();
                m.forward_batch(&mut refs, &ids, &modes).unwrap();
            }
        };
        for _ in 0..2 {
            step(&mut states);
        }
        let before = tel.snapshot().ops;
        let t0 = std::time::Instant::now();
        for _ in 0..8 {
            step(&mut states);
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
        for (k, calls, ms) in rows.iter().take(40) {
            eprintln!("PROFILE B={b}   {k:<40} {:>6.1} calls/step {:>7.3} ms/step", *calls as f64 / 8.0, ms / 8.0);
        }
    }
}

/// The repository's `docs/*.md` as token ids of the model's tokenizer (about 106k tokens): realistic English / markdown text for
/// prompts of any length up to 100k.
fn corpus_tokens(dir: &std::path::Path) -> Vec<u32> {
    use intarsia_core::engine::TextCodec;
    let codec = intarsia_models::HfCodec::load(dir, &intarsia_models::ChatTemplateSource::Auto).unwrap();
    let docs = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs");
    let mut names: Vec<_> = std::fs::read_dir(&docs)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "md"))
        .collect();
    names.sort();
    let text: String = names.iter().map(|p| std::fs::read_to_string(p).unwrap()).collect::<Vec<_>>().join("\n\n");
    codec.encode(&text, false).unwrap()
}

// ---- busy-chunk prefill KL gate (not bitwise by design) -------------------------------------------------------------------

/// Gate of the one non-bitwise step of the batching engine: a request that starts prefilling while others decode runs in busy
/// chunks (1024 tokens for the 35B on CUDA) instead of the solo chunk (2048), and the GEMM tiling depends on the chunk. For
/// prompts longer than the busy chunk (1.5k, 2.5k, 4k, 8k, 16k tokens of repository docs text) this prefills the same prompt
/// both ways, then decodes 64 greedy tokens from the solo state and teacher-forces the same tokens through the busy state.
/// Compared per row over the full vocabulary: KL(solo || busy) and top-1 agreement. Floor: the 35B short golden's ref-noise
/// (`golden/ref_noise/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors.json`, kl21_mean 3.161e-3: vLLM's own run-to-run spread on
/// the same prompts). Pass: the mean KL over the 64 decode rows of every prompt is at most that floor (factor 1, the
/// default of the ref-noise rule). Controls: the solo chunk driven by hand equals `forward` of the whole prefix (harness
/// check, KL exactly 0), and a chunk of 256 (the old default) is reported beside 1024.
#[test]
#[ignore]
fn real_35b_busy_chunk_prefill_kl_gate() {
    use intarsia_core::engine::TextCodec;
    use intarsia_core::model::{HeadMode, StepOut};
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), cuda()) else { return };
    const FLOOR: f64 = 3.161e-3; // ref-noise kl21_mean of the 35B short golden
    const SOLO: usize = 2048;
    const DECODE_ROWS: usize = 64;
    let lens = [1536usize, 2560, 4096, 8192, 16384];
    let (m, _) = load_with(&dir, be, 2, 16384 + DECODE_ROWS + 8, 0);
    assert_eq!(m.prefill_chunk(), SOLO, "the solo prefill chunk of the 35B recipe");

    // text: the repository docs, tokenized by the model's tokenizer
    let codec = intarsia_models::HfCodec::load(&dir, &intarsia_models::ChatTemplateSource::Auto).unwrap();
    let docs = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs");
    let mut names: Vec<_> = std::fs::read_dir(&docs)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "md"))
        .collect();
    names.sort();
    let text: String = names.iter().map(|p| std::fs::read_to_string(p).unwrap()).collect::<Vec<_>>().join("\n\n");
    let all = codec.encode(&text, false).unwrap();
    eprintln!("KLGATE corpus: {} docs files, {} tokens", names.len(), all.len());
    let stride = (all.len() - 16384 - 1) / (lens.len() - 1);

    // prefill `prompt` in slices of `chunk` tokens (0: one `forward` over the whole prefix, the solo engine's schedule), the
    // last prompt token alone (the recurrent schedule of the engine), then `rows` decode steps; returns (rows of logits, tokens)
    let run = |prompt: &[u32], chunk: usize, forced: Option<&[u32]>| -> (Vec<Vec<f32>>, Vec<u32>) {
        let mut st = m.new_state().unwrap();
        let (head, last) = prompt.split_at(prompt.len() - 1);
        if chunk == 0 {
            m.forward(&mut st, head, HeadMode::None, None).unwrap();
        } else {
            for c in head.chunks(chunk) {
                m.forward(&mut st, c, HeadMode::None, None).unwrap();
            }
        }
        let logits = |o: StepOut| match o {
            StepOut::Logits(l) => l,
            o => panic!("expected logits, got {o:?}"),
        };
        let mut rows = vec![logits(m.forward(&mut st, last, HeadMode::LastLogits, None).unwrap())];
        let mut toks = Vec::new();
        for i in 0..DECODE_ROWS {
            let next = match forced {
                Some(f) => f[i],
                None => {
                    rows.last()
                        .unwrap()
                        .iter()
                        .enumerate()
                        .fold((0usize, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b })
                        .0 as u32
                }
            };
            toks.push(next);
            rows.push(logits(m.forward(&mut st, &[next], HeadMode::LastLogits, None).unwrap()));
        }
        (rows, toks)
    };
    // KL(p || q) over the full vocabulary in f64, and top-1 agreement, for rows[1..] (the decode rows) and row 0
    let kl = |p: &[f32], q: &[f32]| -> f64 {
        let lse = |x: &[f32]| {
            let mx = x.iter().cloned().fold(f32::MIN, f32::max) as f64;
            mx + x.iter().map(|&v| (v as f64 - mx).exp()).sum::<f64>().ln()
        };
        let (lp, lq) = (lse(p), lse(q));
        p.iter()
            .zip(q)
            .map(|(&a, &b)| {
                let lpa = a as f64 - lp;
                lpa.exp() * (lpa - (b as f64 - lq))
            })
            .sum()
    };
    let top1 =
        |x: &[f32]| x.iter().enumerate().fold((0usize, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0;
    // the estimator of the ref-noise floor: 21-way KL(p || q), the 20 largest probabilities of p plus the tail bucket
    let kl21 = |p: &[f32], q: &[f32]| -> f64 {
        let norm = |x: &[f32]| {
            let mx = x.iter().cloned().fold(f32::MIN, f32::max) as f64;
            let z: f64 = x.iter().map(|&v| (v as f64 - mx).exp()).sum();
            move |v: f32| (v as f64 - mx).exp() / z
        };
        let (np, nq) = (norm(p), norm(q));
        let mut idx: Vec<usize> = (0..p.len()).collect();
        idx.select_nth_unstable_by(20, |&a, &b| p[b].partial_cmp(&p[a]).unwrap());
        let (mut sp, mut sq, mut kl) = (0f64, 0f64, 0f64);
        for &i in &idx[..20] {
            let (a, b) = (np(p[i]), nq(q[i]));
            sp += a;
            sq += b;
            kl += a * (a / b).ln();
        }
        let (tp, tq) = ((1.0 - sp).max(1e-12), (1.0 - sq).max(1e-12));
        kl + tp * (tp / tq).ln()
    };
    let mut worst_mean = 0f64;
    let mut sum_all = (0f64, 0usize, 0usize, 0usize);
    let mut ctl_max = 0f64;
    let mut failed: Vec<String> = Vec::new();
    let mut failed_full: Vec<String> = Vec::new();
    let mut sum21 = 0f64;
    for (k, &len) in lens.iter().enumerate() {
        let prompt = &all[k * stride..k * stride + len];
        let (solo, toks) = run(prompt, 0, None);
        // control: the solo chunk by hand is the same computation
        let (hand, _) = run(prompt, SOLO, Some(&toks));
        let ctl = (0..solo.len()).map(|i| kl(&solo[i], &hand[i])).fold(0f64, f64::max);
        ctl_max = ctl_max.max(ctl);
        for chunk in [1024usize, 256] {
            let (busy, _) = run(prompt, chunk, Some(&toks));
            let kls: Vec<f64> = (1..solo.len()).map(|i| kl(&solo[i], &busy[i])).collect();
            let k21: Vec<f64> = (1..solo.len()).map(|i| kl21(&solo[i], &busy[i])).collect();
            let mean21 = k21.iter().sum::<f64>() / k21.len() as f64;
            let flips = (1..solo.len()).filter(|&i| top1(&solo[i]) != top1(&busy[i])).count();
            let mean = kls.iter().sum::<f64>() / kls.len() as f64;
            let max = kls.iter().cloned().fold(0f64, f64::max);
            let kl0 = kl(&solo[0], &busy[0]);
            eprintln!(
                "KLGATE len {len:>5} chunk {chunk:>4} vs solo {SOLO}: decode rows {} mean KL {mean:.3e} (21-way {mean21:.3e}) max {max:.3e} top-1 flips {flips}/{} | prefill-last row KL {kl0:.3e} flip {} | control (hand 2048 vs forward) max KL {ctl:.1e}",
                kls.len(), kls.len(), top1(&solo[0]) != top1(&busy[0])
            );
            let head: Vec<String> = kls.iter().take(8).map(|v| format!("{v:.1e}")).collect();
            let tail = kls[8..].iter().sum::<f64>() / (kls.len() - 8) as f64;
            eprintln!("KLGATE   rows 1-8 KL [{}] rows 9-64 mean {tail:.3e}", head.join(" "));
            if chunk == 1024 {
                worst_mean = worst_mean.max(mean);
                sum_all.0 += kls.iter().sum::<f64>();
                sum_all.1 += kls.len();
                sum_all.2 += flips;
                sum_all.3 += 1;
                sum21 += k21.iter().sum::<f64>();
                if mean21 > FLOOR {
                    failed.push(format!("len {len}: mean 21-way KL {mean21:.3e}"));
                }
                if mean > FLOOR {
                    failed_full.push(format!("len {len}: mean full-vocab KL {mean:.3e}"));
                }
            }
        }
    }
    eprintln!(
        "KLGATE SUMMARY chunk 1024 vs 2048: {} prompts, {} decode rows, mean KL {:.3e} (21-way {:.3e}), worst prompt mean {:.3e}, top-1 flips {}/{} | floor {FLOOR:.3e} | control max KL {ctl_max:.1e}",
        sum_all.3, sum_all.1, sum_all.0 / sum_all.1 as f64, sum21 / sum_all.1 as f64, worst_mean, sum_all.2, sum_all.1
    );
    assert_eq!(ctl_max, 0.0, "the harness: the solo chunk by hand must equal forward() of the prefix");
    eprintln!("KLGATE full-vocab prompts above the floor: {failed_full:?}");
    assert!(failed.is_empty(), "above the ref-noise floor {FLOOR:.3e}: {failed:?}");
}

// ---- engine identity on prompts longer than the solo chunk ----------------------------------------------------------------

/// The busy chunk of the batching engine defaults to the solo chunk (2048) for the 35B on CUDA, so a request that prefills while
/// others decode runs the solo chunk schedule: its output must be token for token the `LocalEngine` run, for prompts of 2.6k, 4k
/// and 16k tokens (more than one chunk), with 2 and 4 slots. The decoding neighbour (R0, a short prompt, 900 tokens) is still
/// generating while the long prompts prefill (checked from the scheduler's mixed-iteration counter). Control, printed and not
/// asserted: the same scenario with `--busy-chunk 1024` (the opt-in knob), which is KL-gated and not bitwise.
#[test]
#[ignore]
fn real_35b_batched_engine_long_prompts_equal_the_local_engine() {
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), cuda()) else { return };
    let all = corpus_tokens(&dir);
    let window = |start: usize, len: usize| all[start..start + len].to_vec();
    let mk = |max_batch: Option<usize>, busy_chunk: Option<usize>| EngineOptions {
        load: LoadOptions { max_seq: 16384 + 256, kv_fraction: 0.95, profile_every: 0, ..Default::default() },
        max_batch,
        busy_chunk,
        ..Default::default()
    };
    // (name, request): R0 the decoding neighbour, then the long prompts
    let req = |prompt: Vec<u32>, gen: usize, sampled: Option<u64>| {
        let mut r = GenRequest { prompt, max_tokens: gen, ignore_eos: true, ..Default::default() };
        if let Some(seed) = sampled {
            r.temperature = 0.8;
            r.top_k = 40;
            r.top_p = 0.95;
            r.seed = Some(seed);
        }
        r
    };
    let reqs = || {
        vec![
            ("R0 200 tok, 900 gen", req(window(500, 200), 900, None)),
            ("R1 4096 tok", req(window(3000, 4096), 130, None)),
            ("R2 16384 tok (sampled)", req(window(20000, 16384), 130, Some(21))),
            ("R3 2600 tok", req(window(60000, 2600), 130, None)),
        ]
    };
    let reference: Vec<Out> = {
        let l = load_engine(&dir, be.clone(), &mk(None, None)).unwrap();
        assert_eq!(l.max_batch, 1);
        reqs().into_iter().map(|(_, r)| collect(&l.engine.submit(r).unwrap())).collect()
    };
    assert!(reference.iter().zip(reqs()).all(|(o, (_, r))| o.ids.len() == r.max_tokens));
    // `slots` slots; request indices to run (R0 first, the rest together once R0 is decoding)
    let scenario = |slots: usize,
                    which: &[usize],
                    busy_chunk: Option<usize>|
     -> (Vec<Out>, intarsia_core::scheduler::BatchMetrics) {
        let l = load_engine(&dir, be.clone(), &mk(Some(slots), busy_chunk)).unwrap();
        assert_eq!(l.max_batch, slots);
        let all_reqs = reqs();
        let rx0 = l.engine.submit(all_reqs[which[0]].1.clone()).unwrap();
        // R0 decoding: its first token is out
        let first = rx0.recv_timeout(Duration::from_secs(600)).expect("R0 first token");
        let rest: Vec<_> = which[1..].iter().map(|&i| l.engine.submit(all_reqs[i].1.clone()).unwrap()).collect();
        let mut outs = Vec::new();
        std::thread::scope(|sc| {
            let h0 = sc.spawn(move || {
                // the event already taken, then the rest of the stream
                let (mut ids, mut text, mut reason) = (vec![], String::new(), None);
                let mut ev = Some(first);
                loop {
                    let e = ev
                        .take()
                        .unwrap_or_else(|| rx0.recv_timeout(Duration::from_secs(600)).expect("engine stalled"));
                    match e {
                        GenEvent::Token { id, text: t } => {
                            ids.push(id);
                            text.push_str(&t);
                        }
                        GenEvent::Done { finish_reason, .. } => {
                            reason = Some(finish_reason);
                            break;
                        }
                        GenEvent::Error { message } => panic!("request failed: {message}"),
                    }
                }
                Out { ids, text, reason }
            });
            let hs: Vec<_> = rest.into_iter().map(|rx| sc.spawn(move || collect(&rx))).collect();
            outs.push(h0.join().unwrap());
            outs.extend(hs.into_iter().map(|h| h.join().unwrap()));
        });
        (outs, l.engine.batch_metrics().expect("a batching engine has metrics"))
    };
    let mut failures = Vec::new();
    for (slots, which) in [(2usize, vec![0usize, 1]), (4, vec![0, 1, 2, 3]), (4, vec![0, 2, 3, 1])] {
        let (outs, m) = scenario(slots, &which, None);
        let hist: Vec<String> =
            m.batch_hist.iter().enumerate().filter(|(_, &n)| n > 0).map(|(b, n)| format!("B{b}:{n}")).collect();
        eprintln!(
            "LONGENG {slots} slots, requests {which:?}, busy chunk default: mixed iterations {}, prefill chunks {}, decode batches {}",
            m.mixed_iterations, m.prefill_chunks, hist.join(" ")
        );
        assert!(
            m.mixed_iterations > 0,
            "no iteration prefilled beside decoding: the scenario did not exercise the busy path"
        );
        for (k, &i) in which.iter().enumerate() {
            let same = outs[k] == reference[i];
            eprintln!(
                "LONGENG   {} : {}",
                reqs()[i].0,
                if same { "IDENTICAL to the LocalEngine run" } else { "DIFFERS" }
            );
            if !same {
                let first = outs[k].ids.iter().zip(&reference[i].ids).position(|(a, b)| a != b);
                failures.push(format!("{slots} slots, {}: first differing token {first:?}", reqs()[i].0));
            }
        }
    }
    // control: the opt-in 1024 chunk
    let (outs, _) = scenario(4, &[0, 1, 2, 3], Some(1024));
    for (k, i) in [0usize, 1, 2, 3].iter().enumerate() {
        let first = outs[k].ids.iter().zip(&reference[*i].ids).position(|(a, b)| a != b);
        eprintln!(
            "LONGENG control busy chunk 1024: {} : {}",
            reqs()[*i].0,
            match first {
                None => "identical".to_string(),
                Some(t) => format!("first differing token {t}"),
            }
        );
    }
    assert!(failures.is_empty(), "{failures:?}");
}

/// Diagnostic (ignored): the ops and kernels of one solo decode step and of a 4-row and a 16-row batched step of the 35B, with
/// their call counts. What the R3 work replaces (per-row launches) is the difference between the two lists.
#[test]
#[ignore]
fn real_35b_batch_op_listing() {
    use intarsia_core::plan::Phase;
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), cuda()) else { return };
    let m = load(&dir, be, 16, 2048);
    let count = |rec: &intarsia_core::plan::Recorder| {
        let mut v: std::collections::BTreeMap<(String, String), usize> = Default::default();
        for c in rec.calls.iter().filter(|c| !c.extra) {
            *v.entry((c.call.op.to_string(), c.kernel.clone())).or_default() += 1;
        }
        v
    };
    for (name, phase, rows) in
        [("solo", Phase::Decode, 1usize), ("batch4", Phase::Batch, 4), ("batch16", Phase::Batch, 16)]
    {
        let rec = m.dry_run(phase, rows, 100).unwrap();
        let v = count(&rec);
        let total: usize = v.values().sum();
        eprintln!("OPLIST {name}: {total} launches");
        for ((op, k), n) in &v {
            eprintln!("OPLIST   {name:<8} {n:>5} x {op:<26} {k}");
        }
    }
}

// ---- packed prefill (INTARSIA_PACK_PREFILL=1) ----------------------------------------------------------------------------

/// Prompts that prefill together run their chunks in one pass (`Model::forward_packed`); each request must come out exactly as
/// on the single-sequence engine. 24 requests of 40..1100 tokens of repository docs text (some sampled), submitted at once on
/// 8 and 16 slots, so several are prefilling together; at least one pass must have packed two or more chunks. `--ignored`: it
/// loads the model four times.
#[test]
#[ignore]
fn real_35b_packed_prefill_equals_the_local_engine() {
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), cuda()) else { return };
    let all = corpus_tokens(&dir);
    let lens = [300usize, 1100, 40, 700, 520, 90, 1024, 260, 33, 880, 410, 150];
    let reqs = || -> Vec<GenRequest> {
        (0..24)
            .map(|i| {
                let len = lens[i % lens.len()] + (i / lens.len()) * 17;
                let start = 1000 + i * 1300;
                let mut r = GenRequest { prompt: all[start..start + len].to_vec(), max_tokens: 100, ignore_eos: true, ..Default::default() };
                if i % 4 == 3 {
                    r.temperature = 0.8;
                    r.top_k = 40;
                    r.top_p = 0.95;
                    r.seed = Some(31 + i as u64);
                }
                r
            })
            .collect()
    };
    let mk = |max_batch: Option<usize>| EngineOptions {
        load: LoadOptions { max_seq: 2048, kv_fraction: 0.95, profile_every: 0, ..Default::default() },
        max_batch,
        ..Default::default()
    };
    std::env::remove_var("INTARSIA_PACK_PREFILL");
    let reference: Vec<Out> = {
        let l = load_engine(&dir, be.clone(), &mk(None)).unwrap();
        reqs().into_iter().map(|r| collect(&l.engine.submit(r).unwrap())).collect()
    };
    for slots in [8usize, 16] {
        std::env::set_var("INTARSIA_PACK_PREFILL", "1");
        let l = load_engine(&dir, be.clone(), &mk(Some(slots))).unwrap();
        std::env::remove_var("INTARSIA_PACK_PREFILL");
        let rxs: Vec<_> = reqs().into_iter().map(|r| l.engine.submit(r).unwrap()).collect();
        for (i, rx) in rxs.iter().enumerate() {
            assert_eq!(&collect(rx), &reference[i], "request {i} ({} prompt tokens) on {slots} slots, packed prefill", reqs()[i].prompt.len());
        }
        let m = l.engine.batch_metrics().expect("batched engine metrics");
        eprintln!("{slots} slots: {} packed passes carried {} chunks ({} chunks in all)", m.pack_passes, m.packed_chunks, m.prefill_chunks);
        assert!(m.pack_passes > 0 && m.packed_chunks >= 2 * m.pack_passes, "nothing was packed: {m:?}");
    }
}
