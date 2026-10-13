//! Real checkpoint configs (fixtures) through the strict config loader.
use intarsia_core::recipe::{ConfigKeys, ModelDims};
use serde_json::Value;

fn fixture(name: &str) -> Value {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name);
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

#[test]
fn qwen3_checkpoints_load() {
    let d = ModelDims::from_hf_config(&fixture("config_qwen3_8b.json"), &ConfigKeys::default()).unwrap();
    assert_eq!((d.hidden, d.n_layers, d.n_heads, d.n_kv_heads, d.head_dim), (4096, 36, 32, 8, 128));
    assert_eq!((d.rope_theta, d.partial_rotary_factor), (1_000_000.0, 1.0));
    assert_eq!((d.rms_eps, d.max_positions, d.intermediate), (1e-6, 40960, 12288));
    assert!(!d.tie_embeddings);
    let r = ModelDims::from_hf_config(&fixture("config_qwen3_reranker_0.6b.json"), &ConfigKeys::default()).unwrap();
    assert!(r.tie_embeddings && r.n_layers == 28 && r.vocab == 151669);
    // both pick the built-in qwen3 recipe
    for f in ["config_qwen3_8b.json", "config_qwen3_reranker_0.6b.json"] {
        let (src, _) = intarsia_models::select_recipe(&fixture(f), &[]).unwrap();
        assert_eq!(src, "builtin:qwen3");
    }
}

#[test]
fn qwen36_multimodal_wrapper_reads_text_config() {
    let cfg = fixture("config_qwen36_35b_a3b_nvfp4.json");
    assert!(cfg.get("hidden_size").is_none(), "fixture keeps the wrapper layout");
    let d = ModelDims::from_hf_config(&cfg, &ConfigKeys::default()).unwrap();
    assert_eq!((d.hidden, d.n_layers, d.n_heads, d.n_kv_heads, d.head_dim), (2048, 40, 16, 2, 256));
    assert_eq!(d.rope_theta, 1e7, "from rope_parameters");
    assert_eq!(d.partial_rotary_factor, 0.25);
    assert_eq!((d.vocab, d.max_positions, d.intermediate), (248320, 262144, 0));
    assert!(!d.tie_embeddings);
}
