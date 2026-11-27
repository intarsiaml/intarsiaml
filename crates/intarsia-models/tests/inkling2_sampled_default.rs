//! Lane ink-hip-specflip, flip A (docs/results/2026-11-26-inkling2-hip-specflip.md): where the recipe sets `[tuning.<backend>]
//! sampled_dflash = true` (Inkling-2 on HIP), a sampled request drafts with the checkpoint's own DSpark draft by default, with the
//! per-request draft length policy (lane ink-sampled-adapt), and a seeded request still emits `--spec off`'s ids. Tiny fixtures of
//! `tests/inkling2_dspark.rs`; the tiny engine runs on cpu (or `INTARSIA_INKLING2_BACKEND`), so the test gives the recipe's key to
//! the backend it runs on, the way `recipes/inkling2.toml` gives it to `hip`. Run without `INTARSIA_SPEC_SAMPLED_DFLASH` set.

mod common;
mod ink_sampled;
use ink_sampled::*;

use intarsia_core::model::{LoadOptions, Model};
use intarsia_core::recipe::ModelDims;
use intarsia_core::spec::{SpecConfig, SpecMode};
use intarsia_core::{EngineMeta, LocalEngine, Telemetry};
use intarsia_models::safetensors::SafeTensors;
use std::sync::Arc;

/// The tiny engine of `ink_sampled::engine` (DSpark draft at `k`, or `--spec off`) on the shipped Inkling-2 recipe, with
/// `[tuning.<this backend>] sampled_dflash = key` where `key` is given; nothing set through `Model::set_spec_sampled_dflash`.
fn engine_with(k: Option<usize>, key: Option<bool>) -> (LocalEngine, Arc<Telemetry>, bool) {
    let (tdir, ddir) = dirs().unwrap();
    let mut recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/inkling2.toml")).unwrap();
    let b = backend();
    if let Some(v) = key {
        recipe.tuning.backends.entry(b.name().to_string()).or_default().sampled_dflash = Some(v);
    }
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(tdir.join("config.json")).unwrap()).unwrap();
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    let src = SafeTensors::open_dir(&tdir).unwrap();
    let dflash = k.map(|_| intarsia_models::load::open_dflash(Some(&ddir), false).unwrap());
    let tel = Arc::new(Telemetry::new());
    let opts = LoadOptions { max_seq: 160, prefill_chunk: 16, graphs: false, dflash, ..Default::default() };
    let mut m = Model::load(b, &recipe, &dims, &src, &opts, tel.clone()).unwrap();
    if let Some(k) = k {
        m.enable_verify(k + 1).unwrap();
        m.configure_plan(&Default::default(), Some(&*intarsia_backend_cpu::CpuBackend::arc())).unwrap();
    }
    let on = m.spec_sampled_dflash();
    let spec = match k {
        Some(k) => SpecConfig { mode: SpecMode::Dflash, k, k_min: None },
        None => SpecConfig::default(),
    };
    let e = LocalEngine::new(
        m,
        Arc::new(common::ByteCodec),
        EngineMeta { id: "tiny-dspark".into(), eos_ids: vec![], spec, ..Default::default() },
        tel.clone(),
    )
    .unwrap();
    (e, tel, on)
}

fn env_unset() -> bool {
    let unset = std::env::var("INTARSIA_SPEC_SAMPLED_DFLASH").ok().filter(|v| !v.is_empty()).is_none()
        && std::env::var("INTARSIA_SPEC_SAMPLED_ADAPT").ok().filter(|v| !v.is_empty()).is_none();
    if !unset {
        eprintln!("skipped: run without INTARSIA_SPEC_SAMPLED_DFLASH / INTARSIA_SPEC_SAMPLED_ADAPT set");
    }
    unset
}

/// The recipe key turns sampled drafting on by default (adaptive draft length on), and every seeded request emits the ids of
/// `--spec off`, at the recipe's temperatures 1.0 / 0.6 and with top-k / top-p; the sampled requests really drafted.
#[test]
fn recipe_default_drafts_sampled_requests_with_spec_off_ids() {
    if dirs().is_none() || !env_unset() {
        eprintln!("skipped: no tiny fixtures");
        return;
    }
    gate_off();
    let (off, _, _) = engine_with(None, Some(true));
    for k in [3usize, 5] {
        let (on, tel, resolved) = engine_with(Some(k), Some(true));
        assert!(resolved, "k {k}: [tuning.<backend>] sampled_dflash = true resolves on");
        for cfg in CFGS {
            for p in PROMPTS {
                for (n, seed) in [(5usize, 1u64), (40, 7), (40, 11)] {
                    assert_eq!(run(&on, sreq(p, n, seed, cfg)), run(&off, sreq(p, n, seed, cfg)), "k {k} cfg {cfg:?} n {n} seed {seed} prompt {p:?}");
                }
            }
        }
        let t = tel.snapshot();
        assert!(t.spec_steps > 0, "k {k}: sampled requests never drafted under the recipe default (vacuous)");
        eprintln!("k {k}: passes {} proposed {} accepted {}", t.spec_steps, t.spec_proposed, t.spec_accepted);
    }
}

/// Without the key (the shipped recipe on cpu, Metal and CUDA) a sampled request takes plain steps by default and a greedy one drafts,
/// as before the flip; `false` in the recipe is the same.
#[test]
fn without_the_key_sampled_requests_stay_plain() {
    if dirs().is_none() || !env_unset() {
        eprintln!("skipped: no tiny fixtures");
        return;
    }
    gate_off();
    let ink = intarsia_models::parse_recipe(include_str!("../../../recipes/inkling2.toml")).unwrap();
    let b = backend();
    if ink.sampled_dflash(b.name()).is_some() {
        eprintln!("skipped: the shipped recipe sets sampled_dflash on {}", b.name());
        return;
    }
    for key in [None, Some(false)] {
        let (e, tel, resolved) = engine_with(Some(3), key);
        assert!(!resolved, "{key:?}: resolves off");
        let _ = run(&e, sreq(PROMPTS[0], 24, 3, CFGS[0]));
        assert_eq!(tel.snapshot().spec_steps, 0, "{key:?}: a sampled request drafted");
        let greedy = intarsia_core::GenRequest { prompt: PROMPTS[0].to_vec(), max_tokens: 24, ignore_eos: true, ..Default::default() };
        let _ = run(&e, greedy);
        assert!(tel.snapshot().spec_steps > 0, "{key:?}: a greedy request did not draft");
    }
}
