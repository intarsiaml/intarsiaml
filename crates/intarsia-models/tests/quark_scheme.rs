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
