//! Metal gate G-C and the engine-level identity on the hybrid Qwen3.6-35B-A3B (lane mb,
//! docs/results/2026-10-23-mb-metal-hybrid-batching.md; the Metal twin of `batch_cuda_real.rs`):
//!
//! * `real_35b_batched_decode_is_bitwise_the_solo_step`: `Model::forward_batch` of N = 2..=8 sequences at mixed context lengths,
//!   greedy and seeded-sampled rows, 130 consecutive steps each, must return for every row the bits of that sequence's solo
//!   decode step (logits bit for bit, the on-device argmax token for token) and leave identical state
//!   (`common/batch_identity.rs`); decode rows beside prefill chunks too.
//! * `real_35b_batched_engine_equals_the_local_engine`: the whole `BatchedEngine` (scheduler, slots, sampling) over greedy and
//!   seeded-sampled requests: at one stream it equals `LocalEngine` token for token, and with 4 and 8 requests in flight each
//!   request's tokens still equal its solo run.
//! * `real_35b_batched_engine_long_prompts_equal_the_local_engine` (ignored: minutes): prompts of 2.6k, 4k and 16k tokens
//!   prefilled beside a decoding neighbour.
//!
//! Opt-in (loads 22 GB onto the GPU: take a fleetq slot on endor first): `INTARSIA_QWEN36_DIR` (35B checkpoint, default
//! `~/models/Qwen3.6-35B-A3B-NVFP4` when `INTARSIA_TEST_REAL_MODEL=1`); without it a test prints SKIPPED (and panics under
//! `INTARSIA_REQUIRE_GPU=1`).
#![cfg(target_os = "macos")]
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

/// `INTARSIA_QWEN36_DIR`, else `~/models/Qwen3.6-35B-A3B-NVFP4` under `INTARSIA_TEST_REAL_MODEL=1`.
fn model_dir(var: &str) -> Option<PathBuf> {
    let d = std::env::var_os(var).map(PathBuf::from).or_else(|| {
        std::env::var("INTARSIA_TEST_REAL_MODEL")
            .is_ok_and(|v| v == "1")
            .then(|| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("models/Qwen3.6-35B-A3B-NVFP4"))
    });
    match d {
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

fn metal() -> Option<Arc<dyn intarsia_core::backend::Backend>> {
    match intarsia_backend_metal::MetalBackend::arc() {
        Ok(b) => Some(b),
        Err(e) => {
            skip(&format!("no usable Metal device ({e})"));
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
    let chunk = recipe.resolve_prefill_chunk(None, "metal").value;
    let opts = LoadOptions { max_seq, prefill_chunk: chunk, profile_every, ..Default::default() };
    let tel = Arc::new(Telemetry::new());
    let mut m = Model::load(backend, &recipe, &dims, &weights, &opts, tel.clone()).unwrap();
    assert!(m.batch_capable(), "the metal backend registers the row-exact kernels");
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
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), metal()) else { return };
    let m = load(&dir, be, 16, 2048);
    eprintln!("batch plan notes: {:?}", m.plan.notes);
    // N = 2, 3, 4 (the gate), then up to the cap 16 (9..=16 rows: two chunks of the 8-row GEMVs, one MoE launch); 130
    // consecutive batched steps per size on the same, growing states
    let lens = vec![5, 37, 130, 300, 17, 64, 9, 222, 41, 450, 77, 23, 190, 29, 601, 12];
    gate(&m, lens, vec![2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16], 130, (1000, 30000));
    // decode rows beside prefill chunks (the engine's busy chunk): the decode rows stay bitwise, the prefilled state equals
    // the prompt prefilled alone with the same chunks
    let n = batch_identity::run_with_prefill(&m, &[37, 130, 300, 17], 700, 256, (1000, 30000));
    assert_eq!(n, 4 * 2 * 700usize.div_ceil(256));
}

/// LFM2.5-1.2B (lane nab2): the gated short-conv block's conv state is a `Recurrent` slot, and a batched step runs the conv once per
/// row on that row's own slot (`Model::conv_block_batch`). `INTARSIA_LFM2_DIR` (the checkpoint directory; 2.4 GB, no default).
#[test]
fn real_lfm2_batched_decode_is_bitwise_the_solo_step() {
    // `model_dir` would fall back to the 35B checkpoint under INTARSIA_TEST_REAL_MODEL=1: LFM2 has no default
    if std::env::var_os("INTARSIA_LFM2_DIR").is_none() {
        return skip("INTARSIA_LFM2_DIR not set");
    }
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_LFM2_DIR"), metal()) else { return };
    let m = load(&dir, be, 16, 2048);
    eprintln!("batch plan notes: {:?}", m.plan.notes);
    let lens = vec![5, 37, 130, 300, 17, 64, 9, 222, 41, 450, 77, 23, 190, 29, 601, 12];
    gate(&m, lens, vec![2, 3, 4, 5, 6, 7, 8, 9, 12, 16], 130, (1000, 30000));
    let n = batch_identity::run_with_prefill(&m, &[37, 130, 300, 17], 700, 256, (1000, 30000));
    assert_eq!(n, 4 * 2 * 700usize.div_ceil(256));
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
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), metal()) else { return };
    let mk = |max_batch: Option<usize>| EngineOptions {
        load: LoadOptions { max_seq: 2048, profile_every: 0, ..Default::default() },
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
    for slots in [4usize, 8, 16] {
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

/// Wall time of `forward_batch` (all rows `LastArgmax`, the serving mode) at B = 1, 2, 4, 8 over 24 steps, the solo
/// `forward` step beside it, and the per-op profile of one profiled B = 4 and B = 16 step.
#[test]
#[ignore]
fn real_35b_batch_step_timing() {
    use intarsia_core::model::{HeadMode, StepOut};
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), metal()) else { return };
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
    for b in [2usize, 4, 8, 12, 16] {
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

/// Per-op calls and time of 8 profiled batched steps at B = 4 and at B = 8 (profiling syncs after each op, so the sum
/// is above the unprofiled step; the shares are what to read).
#[test]
#[ignore]
fn real_35b_batch_step_profile() {
    use intarsia_core::model::HeadMode;
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), metal()) else { return };
    let (m, tel) = load_with(&dir, be, 16, 2048, 1);
    let mut states: Vec<_> = (0..16)
        .map(|i| {
            let mut st = m.new_state().unwrap();
            let p: Vec<u32> = (0..900 + 5 * i).map(|j| 1000 + ((j * 7 + i * 13) % 20000) as u32).collect();
            m.forward(&mut st, &p, HeadMode::None, None).unwrap();
            st
        })
        .collect();
    for b in [1usize, 4, 8, 16] {
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

/// Step time of `forward_batch` on realistic traffic (lane mm, docs/results/2026-10-25-mm-metal-moe-grouped.md): B sequences
/// prefilled with ~1k-token slices of the repository docs at different offsets, then greedy decode (each row's argmax fed
/// back), `MM_STEPS` (default 24) steps after 4 warm-up steps, for each B in `MM_BATCHES` (default `1,2,4,8`). With
/// `INTARSIA_METAL_PROFILE=kernel:FILE` it is the per-kernel trace of those steps instead (the phases are 1.5 s apart, so a
/// trace splits at the gaps; prefill comes first); the timing is then meaningless.
#[test]
#[ignore]
fn real_35b_batch_step_real_traffic() {
    use intarsia_core::model::{HeadMode, StepOut};
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), metal()) else { return };
    let batches: Vec<usize> = std::env::var("MM_BATCHES")
        .unwrap_or_else(|_| "1,2,4,8".into())
        .split(',')
        .map(|v| v.trim().parse().unwrap())
        .collect();
    let steps: usize = std::env::var("MM_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(24);
    let maxb = *batches.iter().max().unwrap();
    let m = load(&dir, be, maxb.max(2), 4096);
    let all = corpus_tokens(&dir);
    let mut states: Vec<_> = (0..maxb)
        .map(|i| {
            let mut st = m.new_state().unwrap();
            let off = (i * 6007) % (all.len() - 2000);
            let p = &all[off..off + 1000 + 7 * i];
            m.forward(&mut st, p, HeadMode::None, None).unwrap();
            st
        })
        .collect();
    let mut next: Vec<u32> = (0..maxb).map(|i| all[(i * 6007) % (all.len() - 2000) + 1000 + 7 * i]).collect();
    for &b in &batches {
        std::thread::sleep(Duration::from_millis(1500));
        let modes = vec![HeadMode::LastArgmax; b];
        let mut ms = 0.0;
        for s in 0..steps + 4 {
            let t0 = std::time::Instant::now();
            let mut refs: Vec<&mut _> = states.iter_mut().take(b).collect();
            let out = m.forward_batch(&mut refs, &next[..b], &modes).unwrap();
            if s >= 4 {
                ms += t0.elapsed().as_secs_f64() * 1e3;
            }
            for (r, o) in out.into_iter().enumerate() {
                let StepOut::Token(t) = o else { panic!("row {r}: no token") };
                next[r] = t;
            }
        }
        let ms = ms / steps as f64;
        eprintln!("REALTRAFFIC B={b:>2}: {ms:.2} ms/step, {:.1} tok/s aggregate", b as f64 * 1e3 / ms);
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
// ---- engine identity on prompts longer than the solo chunk ----------------------------------------------------------------

/// The busy chunk of the batching engine is the solo chunk for the hybrid 35B on CUDA (2048), but the recipe's own 2048 on Metal (solo chunk 4096), so a request that
/// prefills while others decode runs 2048-row steps against the `LocalEngine`'s 4096: its output must still be token for token the `LocalEngine` run, for prompts of 2.6k, 4k
/// and 16k tokens (at the 4096-row Metal chunk only the 16k prompt spans several chunks; the 2.6k and 4k ones prefill whole), with 2 and 4 slots. The decoding neighbour (R0, a short prompt, 900 tokens) is still
/// generating while the long prompts prefill (checked from the scheduler's mixed-iteration counter). Control, printed and not
/// asserted: the same scenario with `--busy-chunk 64` (smaller than the solo chunk, 4096), to show the test can see a schedule mismatch.
#[test]
#[ignore]
fn real_35b_batched_engine_long_prompts_equal_the_local_engine() {
    long_engine(false);
}

/// The control alone, in its own process (after three engines in one process the free-memory plan of a fourth reads 0 MiB).
#[test]
#[ignore]
fn real_35b_batched_engine_long_prompts_control_busy_chunk_64() {
    long_engine(true);
}

/// Drop `l` (the engine, its scheduler thread, the model) and check that the device gave its memory back. The free-memory plan
/// of the next engine reads `recommendedMaxWorkingSetSize - currentAllocatedSize`; the Metal backend used to keep every loaded
/// model's weight mapping in its host cache for its whole life, so each engine of this test planned from 22 GB less than the
/// one before and the fourth got a 16377-token context window (the 16384-token prompt was refused). `MetalBackend::info`
/// now prunes the released mappings, so `free_mem` is back at `baseline` (the free memory before the first engine) here.
fn drop_and_settle(be: &Arc<dyn intarsia_core::backend::Backend>, l: intarsia_models::load::Loaded, baseline: u64, what: &str) {
    drop(l);
    let t0 = std::time::Instant::now();
    let tol = 64 << 20;
    loop {
        let free = be.info().free_mem;
        if free + tol >= baseline || t0.elapsed() > Duration::from_secs(10) {
            eprintln!(
                "LONGENG {what}: dropped, free memory {} MiB of the {} MiB at the start, after {:.1} s",
                free >> 20,
                baseline >> 20,
                t0.elapsed().as_secs_f64()
            );
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn long_engine(control_only: bool) {
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), metal()) else { return };
    let all = corpus_tokens(&dir);
    let window = |start: usize, len: usize| all[start..start + len].to_vec();
    let mk = |max_batch: Option<usize>, busy_chunk: Option<usize>| EngineOptions {
        load: LoadOptions { max_seq: 16384 + 256, profile_every: 0, ..Default::default() },
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
    let baseline = be.info().free_mem;
    let reference: Vec<Out> = {
        let l = load_engine(&dir, be.clone(), &mk(None, None)).unwrap();
        assert_eq!(l.max_batch, 1);
        let outs = reqs().into_iter().map(|(_, r)| collect(&l.engine.submit(r).unwrap())).collect();
        drop_and_settle(&be, l, baseline, "reference engine");
        outs
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
        let metrics = l.engine.batch_metrics().expect("a batching engine has metrics");
        drop_and_settle(&be, l, baseline, &format!("{slots}-slot engine"));
        (outs, metrics)
    };
    let mut failures = Vec::new();
    let identity_cases: Vec<(usize, Vec<usize>)> =
        if control_only { vec![] } else { vec![(2, vec![0, 1]), (4, vec![0, 1, 2, 3]), (4, vec![0, 2, 3, 1])] };
    for (slots, which) in identity_cases {
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
    assert!(failures.is_empty(), "{failures:?}");
    if !control_only {
        return;
    }
    // control: busy chunk 64
    let (outs, _) = scenario(4, &[0, 1, 2, 3], Some(64));
    for (k, i) in [0usize, 1, 2, 3].iter().enumerate() {
        let first = outs[k].ids.iter().zip(&reference[*i].ids).position(|(a, b)| a != b);
        eprintln!(
            "LONGENG control busy chunk 64: {} : {}",
            reqs()[*i].0,
            match first {
                None => "identical".to_string(),
                Some(t) => format!("first differing token {t}"),
            }
        );
    }
}
