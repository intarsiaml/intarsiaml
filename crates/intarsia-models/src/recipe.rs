//! Recipe loading: TOML text -> `intarsia_core::recipe::Recipe`, built-in recipes
//! and matching against a model's `config.json` (`architectures`).

use intarsia_core::recipe::Recipe;
use intarsia_core::{Error, Result};
use std::path::{Path, PathBuf};

/// Recipes shipped in the binary (`recipes/*.toml` at the repository root).
pub const BUILTIN_RECIPES: &[(&str, &str)] = &[
    ("qwen3", include_str!("../../../recipes/qwen3.toml")),
    ("qwen3_5", include_str!("../../../recipes/qwen3_5.toml")),
    ("qwen3_5_moe", include_str!("../../../recipes/qwen3_5_moe.toml")),
    ("qwen3_5_vl", include_str!("../../../recipes/qwen3_5_vl.toml")),
    ("llama", include_str!("../../../recipes/llama.toml")),
    ("qwen2", include_str!("../../../recipes/qwen2.toml")),
];

pub fn parse_recipe(text: &str) -> Result<Recipe> {
    let r: Recipe = toml::from_str(text).map_err(|e| Error::model(format!("recipe parse error: {e}")))?;
    r.validate()?;
    Ok(r)
}

pub fn load_recipe_file(path: &Path) -> Result<Recipe> {
    let text =
        std::fs::read_to_string(path).map_err(|e| Error::model(format!("reading recipe {}: {e}", path.display())))?;
    parse_recipe(&text)
}

/// All recipes visible to the loader: `extra_dirs` first (user recipes win), then built-ins.
pub fn all_recipes(extra_dirs: &[PathBuf]) -> Vec<(String, Recipe)> {
    let mut out = Vec::new();
    for d in extra_dirs {
        if let Ok(rd) = std::fs::read_dir(d) {
            let mut files: Vec<_> = rd.flatten().map(|e| e.path()).collect();
            files.sort();
            for p in files {
                if p.extension().and_then(|e| e.to_str()) == Some("toml") {
                    if let Ok(r) = load_recipe_file(&p) {
                        out.push((p.display().to_string(), r));
                    }
                }
            }
        }
    }
    for (name, text) in BUILTIN_RECIPES {
        if let Ok(r) = parse_recipe(text) {
            out.push((format!("builtin:{name}"), r));
        }
    }
    out
}

/// Pick the recipe whose `match.architectures` contains one of the model's architectures.
pub fn select_recipe(config: &serde_json::Value, extra_dirs: &[PathBuf]) -> Result<(String, Recipe)> {
    let archs: Vec<String> = config
        .get("architectures")
        .and_then(|a| a.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    for (src, r) in all_recipes(extra_dirs) {
        if r.model.matcher.architectures.iter().any(|a| archs.contains(a)) {
            return Ok((src, r));
        }
    }
    Err(Error::model(format!(
        "no recipe matches architectures {archs:?}; write one (see docs/adding-a-model.md) and pass --recipe"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_qwen3_parses_and_matches() {
        let cfg = serde_json::json!({"architectures": ["Qwen3ForCausalLM"]});
        let (src, r) = select_recipe(&cfg, &[]).unwrap();
        assert_eq!(src, "builtin:qwen3");
        assert_eq!(r.model.family, "qwen3");
        assert_eq!(r.layer.pattern, vec!["attn", "ffn"]);
        assert_eq!(r.layer.attn.as_ref().unwrap().q_norm.as_deref(), Some("model.layers.{i}.self_attn.q_norm"));
        assert_eq!(r.top.lm_head.tied_to.as_deref(), Some("model.embed_tokens"));
        let bad = serde_json::json!({"architectures": ["NopeForCausalLM"]});
        assert!(select_recipe(&bad, &[]).is_err());
    }

    #[test]
    fn hybrid_recipes_map_blocks_by_layer_type() {
        let cfg = serde_json::json!({"architectures": ["Qwen3_5MoeForConditionalGeneration"]});
        let (src, r) = select_recipe(&cfg, &[]).unwrap();
        assert_eq!(src, "builtin:qwen3_5_moe");
        assert_eq!(r.model.norm_offset, 1.0);
        assert!(r.layer.attn.as_ref().unwrap().gated);
        // 3x linear_attention then 1x full_attention, as in the Qwen3.6-35B-A3B config
        let types: Vec<String> =
            (0..8).map(|i| if i % 4 == 3 { "full_attention" } else { "linear_attention" }.to_string()).collect();
        let kinds: Vec<&str> = (0..8).map(|i| r.layer.pattern_for(i, &types).unwrap()[0].as_str()).collect();
        assert_eq!(kinds, ["gdn", "gdn", "gdn", "attn", "gdn", "gdn", "gdn", "attn"]);
        assert!(r.layer.pattern_for(8, &types).is_err(), "a layer without a type is an error");
        let mut odd = types.clone();
        odd[2] = "sliding_attention".into();
        assert!(r.layer.pattern_for(2, &odd).unwrap_err().to_string().contains("sliding_attention"));
        let cfg = serde_json::json!({"architectures": ["Qwen3_5ForCausalLM"]});
        assert_eq!(select_recipe(&cfg, &[]).unwrap().0, "builtin:qwen3_5");
    }

    #[test]
    fn unknown_pattern_block_is_rejected() {
        let text = r#"
[model]
match.architectures = ["X"]
family = "x"
[layer]
pattern = ["gdn"]
[top]
embed = "e"
final_norm = "n"
lm_head = { tensor = "h" }
"#;
        assert!(parse_recipe(text).is_err());
    }

    const TUNED: &str = r#"
[model]
match.architectures = ["X"]
family = "x"
[layer]
pattern = ["ffn"]
[layer.ffn]
kind = "dense"
act = "silu_glu"
gate = "g"
up = "u"
down = "d"
[top]
embed = "e"
final_norm = "n"
lm_head = { tensor = "h" }
"#;

    #[test]
    fn prefill_chunk_resolution_order() {
        let none = parse_recipe(TUNED).unwrap();
        let base = parse_recipe(&format!("{TUNED}[tuning]\nprefill_chunk = 512\n")).unwrap();
        let both = parse_recipe(&format!("{TUNED}[tuning]\nprefill_chunk = 512\n[tuning.cuda]\nprefill_chunk = 2048\n")).unwrap();
        let only_backend = parse_recipe(&format!("{TUNED}[tuning.cuda]\nprefill_chunk = 2048\n")).unwrap();
        let got = |r: &Recipe, flag, be| {
            let c = r.resolve_prefill_chunk(flag, be);
            (c.value, c.source)
        };
        // flag > recipe backend > recipe base > default
        assert_eq!(got(&both, Some(64), "cuda"), (64, "--prefill-chunk".into()));
        assert_eq!(got(&both, None, "cuda"), (2048, "recipe x, cuda".into()));
        assert_eq!(got(&both, None, "hip"), (512, "recipe x, base".into()), "an unlisted backend falls back to the base value");
        assert_eq!(got(&base, None, "cuda"), (512, "recipe x, base".into()));
        assert_eq!(got(&only_backend, None, "hip"), (256, "default".into()));
        assert_eq!(got(&none, None, "cuda"), (256, "default".into()));
        assert_eq!(both.resolve_prefill_chunk(None, "cuda").describe(), "2048 (recipe x, cuda)");
        assert!(parse_recipe(&format!("{TUNED}[tuning.cuda]\nprefill_chunk = 0\n")).is_err());
    }

    #[test]
    fn built_in_recipes_name_their_chat_turns_and_the_reuse_tolerance_is_per_model() {
        for arch in ["Qwen3ForCausalLM", "Qwen3_5ForCausalLM", "Qwen3_5MoeForConditionalGeneration"] {
            let (_, r) = select_recipe(&serde_json::json!({"architectures": [arch]}), &[]).unwrap();
            assert_eq!(r.accuracy.chat_turns.as_deref(), Some("chatml"), "{arch}");
        }
        // only the dense 8B has a measured tolerance; the hybrid models leave it unset until it is measured on a GPU
        let (_, dense) = select_recipe(&serde_json::json!({"architectures": ["Qwen3ForCausalLM"]}), &[]).unwrap();
        assert_eq!((dense.accuracy.reuse_logit_tol, dense.accuracy.reuse_kl_tol), (Some(0.25), Some(1e-2)));
        let (_, moe) = select_recipe(&serde_json::json!({"architectures": ["Qwen3_5MoeForConditionalGeneration"]}), &[]).unwrap();
        assert_eq!((moe.accuracy.reuse_logit_tol, moe.accuracy.reuse_kl_tol), (None, None));
        // a recipe without [accuracy] still parses
        assert!(parse_recipe(TUNED).unwrap().accuracy.chat_turns.is_none());
        // the factor on the golden's reference noise is per model and unset in every built-in recipe (1.0)
        assert_eq!((dense.accuracy.ref_noise_factor, moe.accuracy.ref_noise_factor), (None, None));
        assert_eq!((dense.accuracy.ref_noise_prompt_factor, moe.accuracy.ref_noise_prompt_factor), (None, None));
        assert_eq!(parse_recipe(&format!("{TUNED}[accuracy]\nref_noise_factor = 1.5\n")).unwrap().accuracy.ref_noise_factor, Some(1.5));
        assert_eq!(parse_recipe(&format!("{TUNED}[accuracy]\nref_noise_prompt_factor = 4.0\n")).unwrap().accuracy.ref_noise_prompt_factor, Some(4.0));
    }

    #[test]
    fn llama_and_qwen2_recipes_select_by_architecture_and_map_what_they_need() {
        let pick = |arch: &str| select_recipe(&serde_json::json!({"architectures": [arch]}), &[]).unwrap();
        let (src, llama) = pick("LlamaForCausalLM");
        assert_eq!(src, "builtin:llama");
        let a = llama.layer.attn.as_ref().unwrap();
        assert!(a.q_norm.is_none() && a.k_norm.is_none() && a.q_bias.is_none() && !a.gated);
        assert_eq!(llama.accuracy.chat_turns.as_deref(), Some("llama3"));
        let (src, qwen2) = pick("Qwen2ForCausalLM");
        assert_eq!(src, "builtin:qwen2");
        let a = qwen2.layer.attn.as_ref().unwrap();
        assert!(a.q_norm.is_none() && a.k_norm.is_none() && !a.gated);
        assert_eq!(
            (a.q_bias.as_deref(), a.k_bias.as_deref(), a.v_bias.as_deref()),
            (
                Some("model.layers.{i}.self_attn.q_proj.bias"),
                Some("model.layers.{i}.self_attn.k_proj.bias"),
                Some("model.layers.{i}.self_attn.v_proj.bias")
            )
        );
        assert_eq!(qwen2.accuracy.chat_turns.as_deref(), Some("chatml"));
    }

    #[test]
    fn qwen3_5_moe_chunk_is_2048_on_cuda_and_4096_on_hip() {
        let cfg = serde_json::json!({"architectures": ["Qwen3_5MoeForConditionalGeneration"]});
        let (_, r) = select_recipe(&cfg, &[]).unwrap();
        assert_eq!(r.resolve_prefill_chunk(None, "cuda").value, 2048);
        let hip = r.resolve_prefill_chunk(None, "hip");
        assert_eq!((hip.value, hip.source.as_str()), (4096, "recipe qwen3_5_moe, hip"));
        assert_eq!(r.resolve_prefill_chunk(Some(256), "hip").value, 256);
        assert_eq!(r.resolve_prefill_chunk(None, "cpu").value, 256);
        assert_eq!(r.resolve_prefill_chunk(Some(256), "cuda").value, 256);
        // the other built-ins stay at 256 everywhere
        for arch in ["Qwen3ForCausalLM", "Qwen3_5ForCausalLM"] {
            let (_, r) = select_recipe(&serde_json::json!({"architectures": [arch]}), &[]).unwrap();
            assert_eq!(r.resolve_prefill_chunk(None, "cuda").value, 256, "{arch}");
        }
    }
}
