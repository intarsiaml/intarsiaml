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
    assert_eq!(effective_act(None, s.mxfp4_act()), Mxfp4Act::Dynamic, "the default runs the declared W4A4");
    assert_eq!(effective_act(Some(Mxfp4Act::None), s.mxfp4_act()), Mxfp4Act::None);
    s.check_linear("model.language_model.layers.0.mlp.gate_proj", MXFP4).unwrap();
    s.check_linear("model.language_model.layers.3.self_attn.o_proj", MXFP4).unwrap();
    s.check_linear("lm_head", BF16).unwrap();
    s.check_linear("mtp.fc", BF16).unwrap();
    assert!(s.check_linear("model.language_model.layers.0.mlp.gate_proj", BF16).is_err());
}
