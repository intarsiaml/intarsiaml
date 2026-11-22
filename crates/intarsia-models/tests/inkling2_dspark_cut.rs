//! The DSpark draft's cumulative top-1 cut (`INTARSIA_DSPARK_CUT`, lane ink-mtp) on the tiny fixture of tests/inkling2_dspark.rs.
//! A file of its own: the knob is read from the environment at load, and this binary's single test is the only one that sets it.
#![allow(dead_code)]

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
        Ok("cpu") | Err(_) => intarsia_backend_cpu::CpuBackend::arc(),
        Ok(other) => panic!("INTARSIA_INKLING2_BACKEND={other}: cpu or metal"),
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

/// With the gate off (tau 0) and k 7: ids equal spec off at every cut; drafts per verify pass do not grow as the cut rises; at
/// cut 1.0 (stop unless the top-1 probability is exactly 1) there are strictly fewer drafts than without the cut.
#[test]
fn dspark_cumulative_cut_keeps_ids_and_shortens_drafts() {
    if dirs().is_none() {
        eprintln!("skipped: no tiny fixtures");
        return;
    }
    std::env::set_var("INTARSIA_DSPARK_TAU", "0");
    let (off, _) = engine(None);
    let want: Vec<Out> = PROMPTS.iter().map(|p| run(&off, p, 40)).collect();
    let mut last = f64::INFINITY;
    let mut per_cut = Vec::new();
    for cut in [None, Some("0.3"), Some("0.9"), Some("1.0")] {
        match cut {
            Some(c) => std::env::set_var("INTARSIA_DSPARK_CUT", c),
            None => std::env::remove_var("INTARSIA_DSPARK_CUT"),
        }
        let (on, tel) = engine(Some(7));
        for (p, w) in PROMPTS.iter().zip(&want) {
            assert_eq!(&run(&on, p, 40), w, "cut {cut:?} prompt {p:?}");
        }
        let t = tel.snapshot();
        // drafts per cycle: a cycle with no draft takes a plain step, so count decode steps through the tokens emitted
        let per = t.spec_proposed as f64 / (PROMPTS.len() * 40) as f64;
        eprintln!("cut {cut:?}: proposed {} accepted {} over {} passes ({per:.3} drafts per token)", t.spec_proposed, t.spec_accepted, t.spec_steps);
        assert!(per <= last + 1e-9, "cut {cut:?}: more drafts ({per:.3} per token) than at a lower cut ({last:.3})");
        last = per;
        per_cut.push(per);
    }
    assert!(per_cut[0] > 0.0, "no drafts without the cut (vacuous test)");
    assert!(per_cut[3] < per_cut[0], "cut 1.0 did not shorten drafts: {:.3} vs {:.3} per token", per_cut[3], per_cut[0]);
}
