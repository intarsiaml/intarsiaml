//! Recipe loading: TOML text -> `intarsia_core::recipe::Recipe`, built-in recipes
//! and matching against a model's `config.json` (`architectures`).

use intarsia_core::recipe::Recipe;
use intarsia_core::{Error, Result};
use std::path::{Path, PathBuf};

/// Recipes shipped in the binary (`recipes/*.toml` at the repository root).
pub const BUILTIN_RECIPES: &[(&str, &str)] = &[("qwen3", include_str!("../../../recipes/qwen3.toml"))];

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
}
