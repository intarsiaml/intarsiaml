//! Inkling-2's own DSpark draft (`--spec dflash --spec-draft <model dir>`, `crates/intarsia-core/src/model/dspark.rs`) on the tiny
//! fixture: the target is `golden/inkling2-tiny`, the draft a random 3-layer DSpark (`golden/inkling2-tiny-dspark`, written by
//! `tools/inkling2_dspark_ref.py tiny`; window 8, taps at visits 2, 6, 10 and the final norm).
//!
//! The correctness gate of lane ink-mtp: greedy output with the draft on is token-identical to `--spec off`, at `--spec-k` 1 / 3 / 7,
//! odd `max_tokens`, with the random draft's (nearly always rejected) proposals and with scripted proposals that are partly right
//! (accepted rows stay, rejected ones roll back by length). Backend: cpu, or Metal / CUDA / HIP with `INTARSIA_INKLING2_BACKEND=metal|cuda|hip`.

mod common;
use common::ByteCodec;
use intarsia_core::model::{DraftScript, LoadOptions, Model};
use intarsia_core::recipe::ModelDims;
use intarsia_core::spec::{SpecConfig, SpecMode};
use intarsia_core::{Engine, EngineMeta, FinishReason, GenEvent, GenRequest, LocalEngine, Telemetry};
use intarsia_models::safetensors::SafeTensors;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

fn dirs() -> Option<(PathBuf, PathBuf)> {
    let g = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden");
    let (t, d) = (g.join("inkling2-tiny"), g.join("inkling2-tiny-dspark"));
    (t.join("config.json").exists() && d.join("mtp.safetensors").exists()).then_some((t, d))
}

fn backend() -> Arc<dyn intarsia_core::Backend> {
    match std::env::var("INTARSIA_INKLING2_BACKEND").as_deref() {
        #[cfg(target_os = "macos")]
        Ok("metal") => intarsia_backend_metal::MetalBackend::arc().unwrap(),
        // lane ink-cuda-mtp
        Ok("cuda") => intarsia_backend_cuda::create(0).unwrap(),
        // lane ink-hip-spec: one R9700 (HIP_VISIBLE_DEVICES picks the card)
        Ok("hip") => intarsia_backend_hip::create(0).unwrap(),
        Ok("cpu") | Err(_) => intarsia_backend_cpu::CpuBackend::arc(),
        Ok(other) => panic!("INTARSIA_INKLING2_BACKEND={other}: cpu, metal, cuda or hip"),
    }
}

fn model(draft: bool, k: usize) -> (Model, Arc<Telemetry>) {
    model_p(draft, k, &Default::default())
}

fn model_p(draft: bool, k: usize, plan: &intarsia_core::plan::PlanOptions) -> (Model, Arc<Telemetry>) {
    let (tdir, ddir) = dirs().unwrap();
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/inkling2.toml")).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(tdir.join("config.json")).unwrap()).unwrap();
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    let src = SafeTensors::open_dir(&tdir).unwrap();
    let dflash = draft.then(|| intarsia_models::load::open_dflash(Some(&ddir), false).unwrap());
    let tel = Arc::new(Telemetry::new());
    let opts = LoadOptions { max_seq: 128, prefill_chunk: 16, graphs: false, dflash, ..Default::default() };
    let mut m = Model::load(backend(), &recipe, &dims, &src, &opts, tel.clone()).unwrap();
    if draft {
        m.enable_verify(k + 1).unwrap();
        let oracle = intarsia_backend_cpu::CpuBackend::arc();
        m.configure_plan(plan, Some(&*oracle)).unwrap();
    }
    (m, tel)
}

fn engine(k: Option<usize>) -> (LocalEngine, Arc<Telemetry>) {
    engine_s(k, None)
}

fn engine_s(k: Option<usize>, script: Option<DraftScript>) -> (LocalEngine, Arc<Telemetry>) {
    let (m, tel) = model(k.is_some(), k.unwrap_or(1));
    if script.is_some() {
        m.dflash_script(script).unwrap();
    }
    let spec = match k {
        Some(k) => SpecConfig { mode: SpecMode::Dflash, k, k_min: None },
        None => SpecConfig::default(),
    };
    let e = LocalEngine::new(m, Arc::new(ByteCodec), EngineMeta { id: "tiny-dspark".into(), eos_ids: vec![], spec, ..Default::default() }, tel.clone())
        .unwrap();
    (e, tel)
}

#[derive(Debug, PartialEq)]
struct Out {
    ids: Vec<u32>,
    reason: FinishReason,
}

fn run(e: &LocalEngine, prompt: &[u32], n: usize) -> Out {
    let (mut ids, mut reason) = (vec![], None);
    let req = GenRequest { prompt: prompt.to_vec(), max_tokens: n, temperature: 0.0, ignore_eos: true, ..Default::default() };
    e.generate(req, &mut |ev| match ev {
        GenEvent::Token { id, .. } => ids.push(id),
        GenEvent::Done { finish_reason, .. } => reason = Some(finish_reason),
        GenEvent::Error { message } => panic!("engine error: {message}"),
    })
    .unwrap();
    Out { ids, reason: reason.unwrap() }
}

const PROMPTS: [&[u32]; 3] = [
    &[3, 9, 27, 81, 50, 7, 7, 1, 60, 2, 33, 33, 12],
    &[90, 1, 2, 3, 4, 5],
    &[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 11, 21, 31, 41],
];

/// The env knob is read at load: every test sets the same value (gate off, so the random draft proposes every step).
fn gate_off() {
    static ONCE: Mutex<()> = Mutex::new(());
    let _g = ONCE.lock().unwrap();
    std::env::set_var("INTARSIA_DSPARK_TAU", "0");
}

#[test]
fn dspark_greedy_ids_equal_spec_off() {
    if dirs().is_none() {
        eprintln!("skipped: no tiny fixtures");
        return;
    }
    gate_off();
    let (off, _) = engine(None);
    for k in [1usize, 3, 7] {
        let (on, tel) = engine(Some(k));
        for p in PROMPTS {
            for n in [1usize, 2, 5, 40] {
                assert_eq!(run(&off, p, n), run(&on, p, n), "k {k} max_tokens {n} prompt {p:?}");
            }
        }
        let t = tel.snapshot();
        assert!(t.spec_proposed > 0, "k {k}: the draft never proposed anything (vacuous test)");
        eprintln!("k {k}: proposed {} accepted {}", t.spec_proposed, t.spec_accepted);
    }
}

/// `serve`'s plan settings (admission on plan shapes, admitted kernels only): the plan dry run records the draft's calls, so its
/// kernels are admitted and a request runs. Before, `serve` failed on the first request with `no kernel for qk_norm ...
/// [admitted kernels only]` (the dry run skipped the draft). Prompts past the window and longer than a prefill chunk.
#[test]
fn dspark_runs_with_admitted_kernels_only() {
    if dirs().is_none() {
        eprintln!("skipped: no tiny fixtures");
        return;
    }
    gate_off();
    let plan = intarsia_core::plan::PlanOptions { admit: true, require_admitted: true, no_cache: true, ..Default::default() };
    let (off, _) = engine(None);
    let (m, tel) = model_p(true, 3, &plan);
    assert!(m.registry.require_admitted);
    let spec = SpecConfig { mode: SpecMode::Dflash, k: 3, k_min: None };
    let on = LocalEngine::new(m, Arc::new(ByteCodec), EngineMeta { id: "tiny-dspark".into(), eos_ids: vec![], spec, ..Default::default() }, tel.clone())
        .unwrap();
    let long: Vec<u32> = (0..40u32).map(|i| (i * 37 + 5) % 250).collect();
    for p in [PROMPTS[0], PROMPTS[2], &long[..]] {
        assert_eq!(run(&off, p, 40), run(&on, p, 40), "prompt {p:?}");
    }
    assert!(tel.snapshot().spec_proposed > 0, "the draft never proposed anything (vacuous test)");
}

/// Scripted proposals that follow the reference continuation for a varying number of tokens, then go wrong: every accepted length
/// 0..=k happens, the accepted rows stay in the draft's context and the output is still `--spec off`'s.
#[test]
fn dspark_partial_acceptance_keeps_ids_identical() {
    if dirs().is_none() {
        eprintln!("skipped: no tiny fixtures");
        return;
    }
    gate_off();
    let (off, _) = engine(None);
    for k in [2usize, 3, 7] {
        let mut acc = 0;
        for p in PROMPTS {
            let reference = run(&off, p, 48);
            let mut full = p.to_vec();
            full.extend(&reference.ids);
            let step = Mutex::new(0usize);
            let script: DraftScript = Box::new(move |seq: &[u32], kk: usize| {
                let mut s = step.lock().unwrap();
                *s += 1;
                let right = *s % (kk + 1); // 0..=kk correct proposals, then a wrong one
                let at = seq.len();
                let mut d: Vec<u32> = (0..kk).filter_map(|j| full.get(at + j).copied()).collect();
                if right < d.len() {
                    d[right] = (d[right] + 1) % 300;
                }
                d
            });
            let (on, tel) = engine_s(Some(k), Some(script));
            assert_eq!(reference, run(&on, p, 48), "k {k} prompt {p:?}");
            let t = tel.snapshot();
            acc += t.spec_accepted;
            eprintln!("k {k}: proposed {} accepted {}", t.spec_proposed, t.spec_accepted);
        }
        assert!(acc > 0, "k {k}: nothing accepted (vacuous test)");
    }
}

/// Lane ink-spec-land, Brief L2a: after a request that did not draft (a sampled turn), the draft no longer covers the sequence. A
/// greedy turn that shares more than `DFLASH_RESYNC_REUSE_MAX` rows with it keeps the KV prefix and runs without drafting (before,
/// it re-prefilled the whole prompt to draft); its ids equal `--spec off`'s on the same request sequence. A new conversation (a
/// prompt sharing nothing) drafts again.
#[test]
fn dspark_unsynced_draft_keeps_the_prefix_and_skips_drafting() {
    if dirs().is_none() {
        eprintln!("skipped: no tiny fixtures");
        return;
    }
    gate_off();
    let (tdir, ddir) = dirs().unwrap();
    let mk = |draft: bool| {
        let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/inkling2.toml")).unwrap();
        let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(tdir.join("config.json")).unwrap()).unwrap();
        let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
        let src = SafeTensors::open_dir(&tdir).unwrap();
        let dflash = draft.then(|| intarsia_models::load::open_dflash(Some(&ddir), false).unwrap());
        let tel = Arc::new(Telemetry::new());
        let opts = LoadOptions { max_seq: 512, prefill_chunk: 16, graphs: false, dflash, ..Default::default() };
        let mut m = Model::load(backend(), &recipe, &dims, &src, &opts, tel.clone()).unwrap();
        let spec = if draft {
            m.enable_verify(4).unwrap();
            m.configure_plan(&Default::default(), Some(&*intarsia_backend_cpu::CpuBackend::arc())).unwrap();
            SpecConfig { mode: SpecMode::Dflash, k: 3, k_min: None }
        } else {
            SpecConfig::default()
        };
        let e = LocalEngine::new(m, Arc::new(ByteCodec), EngineMeta { id: "tiny-dspark".into(), eos_ids: vec![], spec, ..Default::default() }, tel.clone())
            .unwrap();
        (e, tel)
    };
    // one request: (ids, cached prompt tokens)
    let go = |e: &LocalEngine, prompt: &[u32], n: usize, t: f32| {
        let (mut ids, mut cached) = (vec![], usize::MAX);
        let req = GenRequest { prompt: prompt.to_vec(), max_tokens: n, temperature: t, seed: Some(7), ignore_eos: true, ..Default::default() };
        e.generate(req, &mut |ev| match ev {
            GenEvent::Token { id, .. } => ids.push(id),
            GenEvent::Done { cached_tokens, .. } => cached = cached_tokens,
            GenEvent::Error { message } => panic!("engine error: {message}"),
        })
        .unwrap();
        (ids, cached)
    };
    let turn1: Vec<u32> = (0..200u32).map(|i| (i * 37 + 11) % 290 + 5).collect();
    let (off, _) = mk(false);
    let (on, tel) = mk(true);
    let mut log = vec![];
    for (name, e) in [("off", &off), ("on", &on)] {
        let (g1, _) = go(e, &turn1, 6, 0.0); // greedy: drafts (on)
        let p1 = tel.snapshot().spec_proposed;
        let mut turn2 = turn1.clone();
        turn2.extend_from_slice(&g1);
        turn2.extend_from_slice(&[17, 18, 19]);
        let (s2, _) = go(e, &turn2, 4, 1.0); // sampled: no draft, the draft falls out of sync
        let mut turn3 = turn2.clone();
        turn3.extend_from_slice(&s2);
        turn3.extend_from_slice(&[21, 22, 23, 24]);
        let p2 = tel.snapshot().spec_proposed;
        let (g3, c3) = go(e, &turn3, 12, 0.0); // greedy, ~210 rows to reuse
        let p3 = tel.snapshot().spec_proposed;
        let fresh: Vec<u32> = (0..40u32).map(|i| (i * 53 + 201) % 280 + 2).collect();
        let (g4, _) = go(e, &fresh, 8, 0.0); // a new conversation
        let p4 = tel.snapshot().spec_proposed;
        if name == "on" {
            assert!(p1 > 0, "the first greedy turn never drafted (vacuous)");
            assert_eq!(p3, p2, "the greedy turn after the sampled one drafted (it would have re-prefilled)");
            assert!(p4 > p3, "a new conversation did not draft again");
        }
        assert!(c3 > intarsia_core::engine::DFLASH_RESYNC_REUSE_MAX, "{name}: the greedy turn after the sampled one reused only {c3} rows");
        log.push((g1, s2, g3, c3, g4));
    }
    assert_eq!(log[0], log[1], "spec off vs on: ids or reuse differ over the four turns");
}
