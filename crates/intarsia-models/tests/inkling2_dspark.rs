//! Inkling-2's own DSpark draft (`--spec dflash --spec-draft <model dir>`, `crates/intarsia-core/src/model/dspark.rs`) on the tiny
//! fixture: the target is `golden/inkling2-tiny`, the draft a random 3-layer DSpark (`golden/inkling2-tiny-dspark`, written by
//! `tools/inkling2_dspark_ref.py tiny`; window 8, taps at visits 2, 6, 10 and the final norm).
//!
//! The correctness gate of lane ink-mtp: greedy output with the draft on is token-identical to `--spec off`, at `--spec-k` 1 / 3 / 7,
//! odd `max_tokens`, with the random draft's (nearly always rejected) proposals and with scripted proposals that are partly right
//! (accepted rows stay, rejected ones roll back by length). Backend: cpu, or Metal with `INTARSIA_INKLING2_BACKEND=metal`.

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
        Ok("metal") => intarsia_backend_metal::MetalBackend::arc().unwrap(),
        Ok("cpu") | Err(_) => intarsia_backend_cpu::CpuBackend::arc(),
        Ok(other) => panic!("INTARSIA_INKLING2_BACKEND={other}: cpu or metal"),
    }
}

fn model(draft: bool, k: usize) -> (Model, Arc<Telemetry>) {
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
        m.configure_plan(&Default::default(), None).unwrap();
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
