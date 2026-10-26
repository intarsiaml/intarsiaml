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
    ("minicpmv4_7", include_str!("../../../recipes/minicpmv4_7.toml")),
    ("gemma4", include_str!("../../../recipes/gemma4.toml")),
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

/// `*.toml` files in `extra_dirs` that do not parse as a recipe, with the reason. `all_recipes` skips them silently, so
/// a typo in a user recipe otherwise surfaces only as "no recipe matches".
pub fn unparsable_recipes(extra_dirs: &[PathBuf]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for d in extra_dirs {
        let Ok(rd) = std::fs::read_dir(d) else { continue };
        let mut files: Vec<_> = rd.flatten().map(|e| e.path()).collect();
        files.sort();
        for p in files.into_iter().filter(|p| p.extension().and_then(|e| e.to_str()) == Some("toml")) {
            if let Err(e) = load_recipe_file(&p) {
                out.push((p.display().to_string(), e.to_string()));
            }
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
    let recipes = all_recipes(extra_dirs);
    if let Some((src, r)) = recipes.iter().find(|(_, r)| r.model.matcher.architectures.iter().any(|a| archs.contains(a))) {
        return Ok((src.clone(), r.clone()));
    }
    let mut msg = format!("no recipe matches architectures {archs:?}");
    if archs.is_empty() {
        msg.push_str(" (config.json has no \"architectures\" list)");
    }
    let known: Vec<String> = recipes.iter().flat_map(|(_, r)| r.model.matcher.architectures.clone()).collect();
    msg.push_str(&format!("; recipes loaded cover {known:?}"));
    for (path, why) in unparsable_recipes(extra_dirs) {
        msg.push_str(&format!("\n  skipped {path}: {why}"));
    }
    msg.push_str("\nwrite one with `intarsia recipe new --model <dir>` (see docs/recipes.md) and pass --recipe FILE or --recipe-dir DIR");
    Err(Error::model(msg))
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
    fn unknown_recipe_key_is_rejected_naming_the_key_and_the_valid_ones() {
        let text = include_str!("../../../recipes/qwen3.toml").replace("kv_dtype =", "kv_dtyp =");
        let e = parse_recipe(&text).unwrap_err().to_string();
        assert!(e.contains("unknown field `kv_dtyp`") && e.contains("`kv_dtype`"), "{e}");
    }

    #[test]
    fn no_match_error_lists_what_is_loaded_and_why_a_user_recipe_was_skipped() {
        let dir = std::env::temp_dir().join(format!("intarsia-recipe-skip-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let text = include_str!("../../../recipes/qwen3.toml")
            .replace("Qwen3ForCausalLM", "MyModelForCausalLM")
            .replace("kv_dtype =", "kv_dtyp =");
        std::fs::write(dir.join("mine.toml"), text).unwrap();
        let cfg = serde_json::json!({"architectures": ["MyModelForCausalLM"]});
        let e = select_recipe(&cfg, &[dir.clone()]).unwrap_err().to_string();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(e.contains("skipped") && e.contains("mine.toml") && e.contains("kv_dtyp"), "{e}");
        assert!(e.contains("Qwen3ForCausalLM") && e.contains("recipe new"), "{e}");
        let e = select_recipe(&serde_json::json!({}), &[]).unwrap_err().to_string();
        assert!(e.contains("no \"architectures\" list"), "{e}");
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
    fn ppl_form_and_max_ppl_parse_and_default_to_todays_raw_gate() {
        use intarsia_core::recipe::PplForm;
        // absent keys: the raw gate with the built-in limit, for a recipe without [accuracy] and for every other built-in
        let plain = parse_recipe(TUNED).unwrap();
        assert_eq!((plain.accuracy.ppl_form.clone(), plain.accuracy.max_ppl), (None, None));
        assert_eq!(plain.accuracy.ppl_form().unwrap(), PplForm::Raw);
        for arch in ["Qwen3ForCausalLM", "Qwen3_5MoeForConditionalGeneration", "LlamaForCausalLM", "Qwen2ForCausalLM"] {
            let (_, r) = select_recipe(&serde_json::json!({"architectures": [arch]}), &[]).unwrap();
            assert_eq!((r.accuracy.ppl_form.as_deref(), r.accuracy.max_ppl), (None, None), "{arch}");
        }
        // the keys parse
        let chat = parse_recipe(&format!("{TUNED}[accuracy]\nppl_form = \"chat\"\nmax_ppl = 12.5\n")).unwrap();
        assert_eq!((chat.accuracy.ppl_form().unwrap(), chat.accuracy.max_ppl), (PplForm::Chat, Some(12.5)));
        assert_eq!(parse_recipe(&format!("{TUNED}[accuracy]\nppl_form = \"raw\"\n")).unwrap().accuracy.ppl_form().unwrap(), PplForm::Raw);
        // a typo or a non-positive limit is a load error, not a silent raw gate
        assert!(parse_recipe(&format!("{TUNED}[accuracy]\nppl_form = \"chatml\"\n")).is_err());
        assert!(parse_recipe(&format!("{TUNED}[accuracy]\nmax_ppl = 0.0\n")).is_err());
    }

    #[test]
    fn only_gemma4_opts_into_the_chat_gate_and_a_noise_factor_above_one() {
        use intarsia_core::recipe::PplForm;
        let pick = |a: &str| select_recipe(&serde_json::json!({"architectures": [a]}), &[]).unwrap().1;
        let g = pick("Gemma4ForConditionalGeneration");
        assert_eq!(g.accuracy.ppl_form().unwrap(), PplForm::Chat);
        assert!(g.accuracy.max_ppl.is_some());
        assert_eq!(g.accuracy.ref_noise_factor, Some(1.1));
        // every other built-in keeps the raw gate and the default factor of 1.0
        for arch in ["Qwen3ForCausalLM", "Qwen3_5ForCausalLM", "Qwen3_5MoeForConditionalGeneration", "LlamaForCausalLM", "Qwen2ForCausalLM"] {
            let r = pick(arch);
            assert_eq!(r.accuracy.ppl_form().unwrap(), PplForm::Raw, "{arch}");
            assert_eq!(r.accuracy.ref_noise_factor, None, "{arch}");
        }
        // Qwen2.5's [accuracy] decode_positions key still parses
        assert_eq!(pick("Qwen2ForCausalLM").accuracy.decode_positions, Some(1000));
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
        // the dense Qwen3 recipe has 4096 on CUDA (user-approved 2026-10-22 13:08); CPU and the hybrid dense recipe stay at 256
        let (_, q3) = select_recipe(&serde_json::json!({"architectures": ["Qwen3ForCausalLM"]}), &[]).unwrap();
        let c = q3.resolve_prefill_chunk(None, "cuda");
        assert_eq!((c.value, c.source.as_str()), (4096, "recipe qwen3, cuda"));
        assert_eq!(q3.resolve_prefill_chunk(None, "cpu").value, 256);
        let (_, q35) = select_recipe(&serde_json::json!({"architectures": ["Qwen3_5ForCausalLM"]}), &[]).unwrap();
        assert_eq!(q35.resolve_prefill_chunk(None, "cuda").value, 256);
    }

    #[test]
    fn no_builtin_recipe_sets_a_prefill_peel() {
        // the peel is opt-in (`INTARSIA_PREFILL_PEEL` with a larger `--prefill-chunk`): with chunk 4096 + peel 16 the 27B
        // is bitwise chunk 256, but on HIP serve --max-batch 16 it costs context (4930 -> 4407 tokens) and runs 4096-row
        // busy chunks, so its recipe keeps 256 (docs/results/2026-10-25-mx-hip-27b-prefill.md section 4)
        let (_, vl) =
            select_recipe(&serde_json::json!({"architectures": ["Qwen3_5ForConditionalGeneration"]}), &[]).unwrap();
        assert_eq!(vl.resolve_prefill_chunk(None, "hip").value, 256);
        let archs = [
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5ForCausalLM",
            "Qwen3_5MoeForConditionalGeneration",
            "Qwen3ForCausalLM",
        ];
        for arch in archs {
            let (_, r) = select_recipe(&serde_json::json!({"architectures": [arch]}), &[]).unwrap();
            for be in ["cuda", "hip", "metal", "cpu"] {
                assert_eq!(r.resolve_prefill_peel(be), (0, "default".to_string()), "{arch} {be}");
            }
        }
        // a peel is a tail of a 256-row pass: 256 or more is refused
        assert!(parse_recipe(&format!("{TUNED}[tuning.hip]\nprefill_peel = 256\n")).is_err());
        let r = parse_recipe(&format!("{TUNED}[tuning]\nprefill_peel = 4\n[tuning.hip]\nprefill_peel = 8\n")).unwrap();
        assert_eq!(r.resolve_prefill_peel("hip").0, 8);
        assert_eq!(r.resolve_prefill_peel("cuda"), (4, "recipe x, base".to_string()));
    }
}
