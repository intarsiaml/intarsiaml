//! docs/recipes.md must stay in step with the recipe schema: every key the schema accepts is documented, and every
//! complete recipe shown there still parses and validates.

use intarsia_core::recipe::{
    AccuracySpec, AttnSpec, ConfigKeys, Defaults, FfnSpec, GdnSpec, LmHeadSpec, MatchSpec, Recipe, RecipeModel, TopSpec,
};
use intarsia_models::recipe::{parse_recipe, BUILTIN_RECIPES};
use std::collections::BTreeSet;

const DOC: &str = include_str!("../../../docs/recipes.md");

/// The keys a struct with `deny_unknown_fields` accepts, read from serde's own "expected one of" error, so the list
/// cannot drift from the schema.
fn accepted_keys<T: serde::de::DeserializeOwned>() -> Vec<String> {
    let err = match toml::from_str::<T>("zz_not_a_key = 1") {
        Ok(_) => panic!("{} accepts an unknown key: add #[serde(deny_unknown_fields)]", std::any::type_name::<T>()),
        Err(e) => e.to_string(),
    };
    let tail = err.split("expected").nth(1).unwrap_or_else(|| panic!("no key list in: {err}"));
    tail.split('`').skip(1).step_by(2).map(String::from).collect()
}

fn documented(key: &str) -> bool {
    DOC.contains(&format!("`{key}`")) || DOC.contains(&format!("`{key}.")) || DOC.contains(&format!(".{key}`"))
}

#[test]
fn every_schema_key_is_documented() {
    let mut missing = Vec::new();
    let mut check = |ty: &str, keys: Vec<String>| {
        assert!(!keys.is_empty(), "{ty}: no keys harvested");
        for k in keys {
            if !documented(&k) {
                missing.push(format!("{ty}.{k}"));
            }
        }
    };
    check("model", accepted_keys::<RecipeModel>());
    check("model.match", accepted_keys::<MatchSpec>());
    check("layer.attn", accepted_keys::<AttnSpec>());
    check("layer.gdn", accepted_keys::<GdnSpec>());
    check("layer.ffn", accepted_keys::<FfnSpec>());
    check("top", accepted_keys::<TopSpec>());
    check("top.lm_head", accepted_keys::<LmHeadSpec>());
    check("defaults", accepted_keys::<Defaults>());
    check("config", accepted_keys::<ConfigKeys>());
    check("accuracy", accepted_keys::<AccuracySpec>());
    // top-level tables, documented as `### [name]` headings
    for t in accepted_keys::<Recipe>() {
        if !DOC.contains(&format!("[{t}]")) && !DOC.contains(&format!("[{t}.")) {
            missing.push(format!("table [{t}]"));
        }
    }
    // `[layer]` and `[tuning]` use flattened maps, so serde cannot list their keys: take them from the shipped recipes
    let mut flat = BTreeSet::new();
    for (_, text) in BUILTIN_RECIPES {
        let v: toml::Value = toml::from_str(text).unwrap();
        for table in ["layer", "tuning"] {
            for (k, val) in v.get(table).and_then(|t| t.as_table()).into_iter().flatten() {
                if !val.is_table() || table == "layer" && ["by_type", "norms"].contains(&k.as_str()) {
                    flat.insert(k.clone());
                }
            }
        }
        for (k, val) in v["layer"].as_table().unwrap() {
            if val.is_table() && k != "by_type" && k != "norms" {
                flat.insert(format!("layer.{k}"));
            }
        }
    }
    for k in flat {
        let bare = k.strip_prefix("layer.").unwrap_or(&k);
        if !documented(bare) && !DOC.contains(&format!("[{k}]")) {
            missing.push(format!("layer/tuning key {k}"));
        }
    }
    assert!(missing.is_empty(), "docs/recipes.md does not mention these recipe keys: {missing:?}");
}

#[test]
fn complete_recipes_in_the_docs_parse_and_validate() {
    let mut n = 0;
    for block in DOC.split("```toml\n").skip(1) {
        let text = block.split("```").next().unwrap();
        if text.trim_start().starts_with("[model]") {
            n += 1;
            parse_recipe(text).unwrap_or_else(|e| panic!("recipe example {n} in docs/recipes.md: {e}\n{text}"));
        }
    }
    assert!(n >= 3, "expected the three worked examples, found {n}");
}
