//! The real amd/Qwen3.8-27B-Quark-AWQ-MXFP4 config.json (tests/fixtures/qwen38_27b_quark_mxfp4/) through the loader's
//! scheme reader: Quark declares its scheme in config.json `quantization_config`, which must give W4A4 (dynamic MXFP4
//! activations) as the default activation mode and keep lm_head / mtp / visual unquantised.

use intarsia_core::quant::{mxfp4::effective_act, Mxfp4Act, BF16, MXFP4};
use intarsia_models::load::read_quant_scheme;
use std::path::Path;

#[test]
fn the_real_quark_config_declares_w4a4_mxfp4() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen38_27b_quark_mxfp4");
    let s = read_quant_scheme(&dir).unwrap().expect("config.json quantization_config (quark) is read");
    assert_eq!(s.quant_algo.as_deref(), Some("MXFP4_W4A4"));
    assert_eq!(s.mxfp4_act(), Some(Mxfp4Act::Dynamic));
    assert_eq!(effective_act(None, None, s.mxfp4_act()), Mxfp4Act::Dynamic, "without a recipe key the default runs the declared W4A4");
    assert_eq!(effective_act(Some(Mxfp4Act::None), None, s.mxfp4_act()), Mxfp4Act::None);
    s.check_linear("model.language_model.layers.0.mlp.gate_proj", MXFP4).unwrap();
    s.check_linear("model.language_model.layers.3.self_attn.o_proj", MXFP4).unwrap();
    s.check_linear("lm_head", BF16).unwrap();
    s.check_linear("mtp.fc", BF16).unwrap();
    assert!(s.check_linear("model.language_model.layers.0.mlp.gate_proj", BF16).is_err());
}

/// The recipe-level default (`[defaults] mxfp4_act`): the Qwen3.8-27B recipe runs W4A16 by default, `--mxfp4-act dynamic` restores the
/// declared W4A4, and no other built-in recipe sets the key (so no other model's activation mode moved).
#[test]
fn the_27b_recipe_defaults_to_w4a16_and_no_other_recipe_sets_the_key() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen38_27b_quark_mxfp4");
    let s = read_quant_scheme(&dir).unwrap().unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let (src, recipe) = intarsia_models::recipe::select_recipe(&cfg, &[]).unwrap();
    assert_eq!(src, "builtin:qwen3_5_vl");
    let rec = recipe.defaults.mxfp4_act;
    assert_eq!(rec, Some(Mxfp4Act::None));
    assert_eq!(effective_act(None, rec, s.mxfp4_act()), Mxfp4Act::None, "no flags: W4A16");
    assert_eq!(effective_act(Some(Mxfp4Act::None), rec, s.mxfp4_act()), Mxfp4Act::None);
    assert_eq!(effective_act(Some(Mxfp4Act::Dynamic), rec, s.mxfp4_act()), Mxfp4Act::Dynamic, "--mxfp4-act dynamic: W4A4");
    // the deviation note names the recipe as the chooser and the way back
    let note = s.mxfp4_act_note(Mxfp4Act::None, intarsia_core::quant::mxfp4::ActSource::Recipe).unwrap();
    assert!(note.contains("recipe") && note.contains("--mxfp4-act dynamic"), "{note}");
    for (name, r) in intarsia_models::recipe::all_recipes(&[]) {
        if name != "builtin:qwen3_5_vl" {
            assert_eq!(r.defaults.mxfp4_act, None, "{name} must not set mxfp4_act");
        }
    }
}

/// The per-backend activation mode (`[tuning.<backend>] mxfp4_act`) of the Qwen3.8-27B recipe: precedence `--mxfp4-act`, backend tuning,
/// recipe `[defaults]`, checkpoint. HIP resolves to W4A8 with no flag; CUDA, Metal and cpu stay W4A16 and never resolve to fp8 unless the
/// flag says so; no other recipe sets the key.
#[test]
fn the_27b_recipe_runs_w4a8_on_hip_only_and_the_flag_always_wins() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen38_27b_quark_mxfp4");
    let s = read_quant_scheme(&dir).unwrap().unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let (_, recipe) = intarsia_models::recipe::select_recipe(&cfg, &[]).unwrap();
    let declared = s.mxfp4_act();
    assert_eq!(declared, Some(Mxfp4Act::Dynamic));
    assert_eq!(recipe.defaults.mxfp4_act, Some(Mxfp4Act::None), "[defaults] stays W4A16");
    for (backend, plain) in [("hip", Mxfp4Act::Fp8), ("cuda", Mxfp4Act::None), ("metal", Mxfp4Act::None), ("cpu", Mxfp4Act::None), ("unknown", Mxfp4Act::None)] {
        let rec = recipe.mxfp4_act_for(backend);
        assert_eq!(effective_act(None, rec, declared), plain, "{backend}: no flag");
        for flag in [Mxfp4Act::None, Mxfp4Act::Dynamic, Mxfp4Act::Fp8] {
            assert_eq!(effective_act(Some(flag), rec, declared), flag, "{backend}: --mxfp4-act {} wins", flag.name());
        }
        assert_eq!(recipe.mxfp4_act_is_backend_tuned(backend), backend == "hip", "{backend}");
    }
    // the deviation note names the right chooser and the way back
    let note = s.mxfp4_act_note(Mxfp4Act::Fp8, intarsia_core::quant::mxfp4::ActSource::Recipe).unwrap();
    assert!(note.contains("0.8204") && note.contains("--mxfp4-act dynamic"), "{note}");
    // a backend table that sets only itself does not leak into the others, and "fp8" parses in both places
    let r = intarsia_models::parse_recipe(&format!(
        "{}\n[tuning.cuda]\nmxfp4_act = \"fp8\"\n",
        include_str!("../../../recipes/qwen3_5_vl.toml")
    ))
    .unwrap();
    assert_eq!(r.mxfp4_act_for("cuda"), Some(Mxfp4Act::Fp8));
    assert_eq!(r.mxfp4_act_for("metal"), Some(Mxfp4Act::None));
    for (name, r) in intarsia_models::recipe::all_recipes(&[]) {
        if name != "builtin:qwen3_5_vl" {
            for backend in ["cpu", "cuda", "hip", "metal"] {
                assert_eq!(r.mxfp4_act_for(backend), None, "{name} must not set mxfp4_act ({backend})");
            }
        }
    }
}
