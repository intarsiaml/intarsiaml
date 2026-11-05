//! The recipe's DFlash default (lane d8, `[tuning.<backend>] dflash_default_draft`, `intarsia_models::load::{dflash_default,
//! find_dflash_draft, dflash_draft_fits, apply_dflash_default}`): on only without `--spec`, at one local rank on a backend whose recipe
//! sets the key, one sequence, no prefix store, and not with `INTARSIA_SPEC_DEFAULT=0`; a draft that is missing from the hub cache,
//! unreadable or for another target leaves plain decode (never an error); and the shipped recipes turn it on for exactly one pair
//! (the dense Qwen3 on CUDA). The tiny hybrid fixture stands in for the target: these tests prove the decision logic and the fallbacks,
//! the real 8B runs are in docs/results/2026-11-05-dflash-8b-default.md.

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::model::{DflashConfig, LoadOptions};
use intarsia_core::recipe::ModelDims;
use intarsia_core::spec::{SpecConfig, SpecMode, DFLASH_K_DEFAULT};
use intarsia_core::{DType, Engine, GenEvent, GenRequest};
use intarsia_models::load::{apply_dflash_default, dflash_default, dflash_draft_fits, find_dflash_draft};
use std::path::{Path, PathBuf};

static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

const REPO: &str = "Test-Org/Tiny-DFlash";

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_qwen3_5")
}

fn recipe_with(table: &str) -> intarsia_core::recipe::Recipe {
    let base = include_str!("../../../recipes/qwen3_5.toml");
    intarsia_models::parse_recipe(&format!("{base}\n{table}\n")).unwrap()
}

fn dims() -> ModelDims {
    let recipe = recipe_with("");
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fixture_dir().join("config.json")).unwrap()).unwrap();
    ModelDims::from_hf_config(&cfg, &recipe.config).unwrap()
}

fn auto() -> intarsia_models::EngineOptions {
    intarsia_models::EngineOptions { spec_auto: true, ..Default::default() }
}

/// A draft checkpoint directory (a DFlash v1 `config.json` with `hidden` and `taps`, the fixture's safetensors as weights, which the
/// default's check never reads past the header).
fn write_draft(dir: &Path, hidden: usize, taps: &[usize], v2: bool) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::copy(fixture_dir().join("model.safetensors"), dir.join("model.safetensors")).unwrap();
    let d = dims();
    let cfg = serde_json::json!({
        "architectures": [if v2 { "DFlash2DraftModel" } else { "Qwen3ForCausalLM" }],
        "hidden_size": hidden, "num_hidden_layers": 2, "num_attention_heads": d.n_heads, "num_key_value_heads": d.n_kv_heads,
        "head_dim": d.head_dim, "intermediate_size": 48, "vocab_size": d.vocab, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
        "dflash_config": { "block_size": 16, "mask_token_id": 95, "target_layer_ids": taps },
    });
    std::fs::write(dir.join("config.json"), serde_json::to_string(&cfg).unwrap()).unwrap();
}

fn hub(name: &str) -> PathBuf {
    let hub = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&hub);
    std::fs::create_dir_all(&hub).unwrap();
    hub
}

fn snapshot(hub: &Path, sha: &str) -> PathBuf {
    hub.join("models--Test-Org--Tiny-DFlash/snapshots").join(sha)
}

/// Runs `f` with the three hub-cache variables pointing at `hub` only and `INTARSIA_DFLASH_DRAFT` / `INTARSIA_SPEC_DEFAULT` unset.
fn with_hub<T>(hub: &Path, f: impl FnOnce() -> T) -> T {
    let saved: Vec<(&str, Option<std::ffi::OsString>)> =
        ["HF_HUB_CACHE", "HF_HOME", "HOME", "INTARSIA_DFLASH_DRAFT", "INTARSIA_SPEC_DEFAULT"].iter().map(|k| (*k, std::env::var_os(k))).collect();
    std::env::set_var("HF_HUB_CACHE", hub);
    std::env::remove_var("INTARSIA_DFLASH_DRAFT");
    std::env::remove_var("INTARSIA_SPEC_DEFAULT");
    let out = f();
    for (k, v) in saved {
        match v {
            Some(v) => std::env::set_var(k, v),
            None => std::env::remove_var(k),
        }
    }
    out
}

#[test]
fn the_default_needs_no_spec_one_local_rank_the_backend_one_sequence_no_store_and_no_opt_out() {
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let with = recipe_with("[tuning.cuda]\ndflash_default_draft = \"Test-Org/Tiny-DFlash\"");
    let without = recipe_with("");
    let blank = recipe_with("[tuning.cuda]\ndflash_default_draft = \"\"");
    let a = auto();
    let want = Some((REPO.to_string(), DFLASH_K_DEFAULT));
    with_hub(&hub("d8_gate"), || {
        assert_eq!(dflash_default(&a, &with, "cuda", 1, true), want, "on: no --spec, cuda, one rank");
        assert_eq!(dflash_default(&a, &without, "cuda", 1, true), None, "a recipe without the key");
        assert_eq!(dflash_default(&a, &blank, "cuda", 1, true), None, "an empty repo id is unset");
        assert_eq!(dflash_default(&a, &with, "metal", 1, true), None, "another backend (Metal has no DFlash)");
        assert_eq!(dflash_default(&a, &with, "hip", 1, true), None, "another backend (HIP runs DFlash2 only)");
        assert_eq!(dflash_default(&a, &with, "cuda", 2, true), None, "tensor parallelism");
        assert_eq!(dflash_default(&a, &with, "cuda", 1, false), None, "a rank in another process");
        let explicit = intarsia_models::EngineOptions { spec_auto: false, ..Default::default() };
        assert_eq!(dflash_default(&explicit, &with, "cuda", 1, true), None, "--spec given (--spec off too)");
        let batch = intarsia_models::EngineOptions { max_batch: Some(4), ..auto() };
        assert_eq!(dflash_default(&batch, &with, "cuda", 1, true), None, "--max-batch 4 falls back to plain decode, no error");
        let one = intarsia_models::EngineOptions { max_batch: Some(1), ..auto() };
        assert_eq!(dflash_default(&one, &with, "cuda", 1, true), want, "--max-batch 1 keeps it");
        let mtp = intarsia_models::EngineOptions { load: LoadOptions { mtp: true, ..Default::default() }, ..auto() };
        assert_eq!(dflash_default(&mtp, &with, "cuda", 1, true), None, "--mtp");
        let draft = intarsia_models::EngineOptions { spec_draft: Some("/x".into()), ..auto() };
        assert_eq!(dflash_default(&draft, &with, "cuda", 1, true), None, "an explicit --spec-draft is not overridden");
        let dco = intarsia_models::EngineOptions { dcoco: true, ..auto() };
        assert_eq!(dflash_default(&dco, &with, "cuda", 1, true), None, "--dcoco");
        // a lone --spec-k N sets the draft length only
        let lone = intarsia_models::EngineOptions { spec_k_given: true, spec: SpecConfig { mode: SpecMode::Off, k: 7 }, ..auto() };
        assert_eq!(dflash_default(&lone, &with, "cuda", 1, true), Some((REPO.to_string(), 7)));
        let lone0 = intarsia_models::EngineOptions { spec_k_given: true, spec: SpecConfig { mode: SpecMode::Off, k: 0 }, ..auto() };
        assert_eq!(dflash_default(&lone0, &with, "cuda", 1, true), None, "--spec-k 0 is no draft length");
        let lone99 = intarsia_models::EngineOptions { spec_k_given: true, spec: SpecConfig { mode: SpecMode::Off, k: 99 }, ..auto() };
        assert_eq!(dflash_default(&lone99, &with, "cuda", 1, true), None, "above the DFlash block");
        let stored = intarsia_models::EngineOptions { prefix_store: intarsia_core::prefix_store::PrefixStoreConfig::parse("on").unwrap(), ..auto() };
        assert_eq!(dflash_default(&stored, &with, "cuda", 1, true), None, "--prefix-store on (a drafting request bypasses the store)");
        std::env::set_var("INTARSIA_PREFIX_STORE", "on");
        let env_store = dflash_default(&a, &with, "cuda", 1, true);
        std::env::remove_var("INTARSIA_PREFIX_STORE");
        assert_eq!(env_store, None, "INTARSIA_PREFIX_STORE=on");
        assert_eq!(dflash_default(&a, &with, "cuda", 1, true), want, "and back on once it is unset");
        std::env::set_var("INTARSIA_SPEC_DEFAULT", "0");
        let off = dflash_default(&a, &with, "cuda", 1, true);
        std::env::remove_var("INTARSIA_SPEC_DEFAULT");
        assert_eq!(off, None, "INTARSIA_SPEC_DEFAULT=0");
    });
}

#[test]
fn the_draft_is_found_in_the_hub_cache_or_by_env_and_never_downloaded() {
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let h = hub("d8_find");
    with_hub(&h, || {
        assert_eq!(find_dflash_draft(REPO), None, "an empty cache");
        // a snapshot without the weights is not a draft
        std::fs::create_dir_all(snapshot(&h, "partial")).unwrap();
        std::fs::write(snapshot(&h, "partial").join("config.json"), "{}").unwrap();
        assert_eq!(find_dflash_draft(REPO), None, "config.json alone");
        write_draft(&snapshot(&h, "complete"), 64, &[1, 3], false);
        assert_eq!(find_dflash_draft(REPO), Some(snapshot(&h, "complete")), "the complete snapshot");
        assert_eq!(find_dflash_draft("Other-Org/Other"), None, "another repo id");
        // INTARSIA_DFLASH_DRAFT wins, and must itself be complete
        let own = h.join("own");
        write_draft(&own, 64, &[1, 3], false);
        std::env::set_var("INTARSIA_DFLASH_DRAFT", &own);
        assert_eq!(find_dflash_draft(REPO), Some(own.clone()));
        std::env::set_var("INTARSIA_DFLASH_DRAFT", h.join("nothing"));
        assert_eq!(find_dflash_draft(REPO), None, "an env dir without a draft: off, not the cache");
        std::env::remove_var("INTARSIA_DFLASH_DRAFT");
    });
}

#[test]
fn a_draft_for_another_target_is_not_applied_and_a_fitting_one_is() {
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let with = recipe_with("[tuning.cpu]\ndflash_default_draft = \"Test-Org/Tiny-DFlash\"");
    let d = dims();
    let h = hub("d8_apply");
    let snap = snapshot(&h, "s1");
    with_hub(&h, || {
        // missing: plain decode
        assert!(apply_dflash_default(&auto(), &with, &d, "cpu", 1, true).is_none(), "the draft is not in the cache");
        // for another target (hidden size, v2, taps beyond the model): plain decode
        write_draft(&snap, 4096, &[1, 3], false);
        assert!(apply_dflash_default(&auto(), &with, &d, "cpu", 1, true).is_none(), "hidden size 4096 against 64");
        write_draft(&snap, 64, &[1, 3], true);
        assert!(apply_dflash_default(&auto(), &with, &d, "cpu", 1, true).is_none(), "a DFlash2 draft");
        write_draft(&snap, 64, &[1, 99], false);
        assert!(apply_dflash_default(&auto(), &with, &d, "cpu", 1, true).is_none(), "tap layer 99 of 6");
        // unreadable config
        std::fs::write(snap.join("config.json"), "not json").unwrap();
        assert!(apply_dflash_default(&auto(), &with, &d, "cpu", 1, true).is_none(), "a broken config.json");
        // fits: --spec dflash, the draft dir, k 15 (or the lone --spec-k)
        write_draft(&snap, 64, &[1, 3], false);
        let o = apply_dflash_default(&auto(), &with, &d, "cpu", 1, true).expect("applies");
        assert_eq!((o.spec.mode, o.spec.k), (SpecMode::Dflash, DFLASH_K_DEFAULT));
        assert_eq!(o.spec_draft, Some(snap.clone()));
        let lone = intarsia_models::EngineOptions { spec_k_given: true, spec: SpecConfig { mode: SpecMode::Off, k: 5 }, ..auto() };
        assert_eq!(apply_dflash_default(&lone, &with, &d, "cpu", 1, true).unwrap().spec.k, 5);
        // other cases never touch the cache
        let batch = intarsia_models::EngineOptions { max_batch: Some(2), ..auto() };
        assert!(apply_dflash_default(&batch, &with, &d, "cpu", 1, true).is_none());
        assert!(apply_dflash_default(&auto(), &with, &d, "cuda", 1, true).is_none(), "no key for cuda in this recipe");
    });
}

#[test]
fn the_fit_check_names_what_differs() {
    let d = dims();
    let c = |hidden: usize, taps: Vec<usize>, v2: bool| DflashConfig {
        layer_ids: taps,
        mask_id: 95,
        block: 16,
        hidden,
        n_layers: 2,
        intermediate: 48,
        n_heads: d.n_heads,
        n_kv_heads: d.n_kv_heads,
        head_dim: d.head_dim,
        vocab: d.vocab,
        topk: 0,
        emb_dim: 0,
        state_dim: 0,
        mlp_dim: 0,
        v2,
        rms_eps: 1e-6,
        rope_theta: 10000.0,
        sliding_window: 0,
        conv_kernel: 0,
        conv_group: 0,
        selector_rank: 0,
        selector_top_k: 0,
    };
    assert!(dflash_draft_fits(&c(d.hidden, vec![0, 5], false), &d).is_ok());
    assert!(dflash_draft_fits(&c(d.hidden + 1, vec![0, 5], false), &d).unwrap_err().contains("another target"));
    assert!(dflash_draft_fits(&c(d.hidden, vec![0, 6], false), &d).unwrap_err().contains("taps layers"));
    assert!(dflash_draft_fits(&c(d.hidden, vec![], false), &d).is_err(), "no taps");
    assert!(dflash_draft_fits(&c(d.hidden, vec![0, 5], true), &d).unwrap_err().contains("DFlash2"));
    let mut vocab = c(d.hidden, vec![0, 5], false);
    vocab.vocab += 8;
    assert!(dflash_draft_fits(&vocab, &d).is_err(), "another vocabulary");
}

/// The shipped recipes: exactly the dense Qwen3 on CUDA names a DFlash draft, and it is the checkpoint the 8B was measured with.
#[test]
fn only_the_measured_recipe_backend_pair_defaults_to_dflash() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../recipes");
    let mut got = Vec::new();
    for e in std::fs::read_dir(&dir).unwrap() {
        let path = e.unwrap().path();
        if path.extension().is_none_or(|x| x != "toml") {
            continue;
        }
        let recipe = intarsia_models::parse_recipe(&std::fs::read_to_string(&path).unwrap()).unwrap();
        for backend in ["cpu", "cuda", "hip", "metal"] {
            if let Some(r) = recipe.resolve_dflash_default(backend) {
                got.push(format!("{} {backend} {r}", recipe.model.family));
            }
        }
    }
    got.sort();
    assert_eq!(got, ["qwen3 cuda TD-Labs/Qwen3-8B-DCOCO"], "the recipes that turn DFlash on by default");
}

/// Through the engine loader: a recipe asking for the default, a draft that is not on this machine: the engine loads, decodes plainly and
/// its ids equal `--spec off`'s (and `--spec off` itself stays off with a draft present).
#[test]
fn the_loader_falls_back_to_plain_decode_when_the_draft_is_missing_or_wrong() {
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("d8_model");
    std::fs::create_dir_all(&dir).unwrap();
    for f in ["config.json", "model.safetensors"] {
        std::fs::copy(fixture_dir().join(f), dir.join(f)).unwrap();
    }
    let vocab: serde_json::Map<String, serde_json::Value> = (0..96).map(|i| (format!("t{i}"), i.into())).collect();
    let tok = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [], "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "t0"}
    });
    std::fs::write(dir.join("tokenizer.json"), serde_json::to_string(&tok).unwrap()).unwrap();
    let recipe = dir.join("recipe.toml");
    std::fs::write(&recipe, format!("{}\n[tuning.cpu]\ndflash_default_draft = \"{REPO}\"\n", include_str!("../../../recipes/qwen3_5.toml"))).unwrap();
    let load = |auto: bool| {
        let o = intarsia_models::EngineOptions {
            load: LoadOptions { max_seq: 256, act_dtype: Some(DType::F32), kv_dtype: Some(DType::F32), ..Default::default() },
            recipe: Some(recipe.clone()),
            spec_auto: auto,
            ..Default::default()
        };
        intarsia_models::load_engine(&dir, CpuBackend::arc(), &o).unwrap()
    };
    let ids = |e: &dyn Engine| {
        let mut ids = vec![];
        let r = GenRequest { prompt: vec![3, 9, 27, 81, 50, 7, 7, 1], max_tokens: 24, temperature: 0.0, ignore_eos: true, ..Default::default() };
        e.generate(r, &mut |ev| match ev {
            GenEvent::Token { id, .. } => ids.push(id),
            GenEvent::Error { message } => panic!("engine error: {message}"),
            _ => {}
        })
        .unwrap();
        ids
    };
    let h = hub("d8_loader");
    with_hub(&h, || {
        let off = load(false);
        let want = ids(&*off.engine);
        // the draft is not in the cache
        let missing = load(true);
        assert_eq!(ids(&*missing.engine), want, "missing draft: plain decode, same ids");
        assert_eq!(missing.engine.telemetry().spec_steps, 0);
        // the draft is for another target (hidden 4096)
        write_draft(&snapshot(&h, "wrong"), 4096, &[1, 3], false);
        let wrong = load(true);
        assert_eq!(ids(&*wrong.engine), want, "mismatching draft: plain decode, same ids");
        assert_eq!(wrong.engine.telemetry().spec_steps, 0);
    });
}
