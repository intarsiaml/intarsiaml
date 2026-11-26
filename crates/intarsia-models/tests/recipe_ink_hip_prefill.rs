//! The prefill chunk the shipped Inkling-2-Mini recipe gives HIP (user decision 2026-11-26: 2048 instead of the builtin 256, with the
//! tensor-core expert prefill and the flash prefill attention that are code defaults on HIP now; docs/results/2026-11-26-inkling2-hip-pfflip.md).
//! Only Inkling-2 on HIP changes; every other backend of the recipe keeps its value, `--prefill-chunk` wins over the recipe, and the
//! recipe sets no HIP busy chunk (batching is refused for Inkling-2 on HIP).

#[test]
fn inkling2_on_hip_prefills_in_2048_row_chunks() {
    let ink = intarsia_models::parse_recipe(include_str!("../../../recipes/inkling2.toml")).unwrap();
    let hip = ink.resolve_prefill_chunk(None, "hip");
    assert_eq!(hip.value, 2048, "recipe [tuning.hip] prefill_chunk");
    assert_eq!(ink.resolve_prefill_chunk(Some(256), "hip").value, 256, "--prefill-chunk 256 is the revert");
    assert_eq!(ink.resolve_prefill_chunk(None, "metal").value, 2048);
    assert_eq!(ink.resolve_prefill_chunk(None, "cuda").value, 2048);
    assert_eq!(ink.resolve_prefill_chunk(None, "cpu").value, 256, "an unlisted backend keeps the builtin chunk");
    assert_eq!(ink.resolve_busy_chunk("hip"), None, "no HIP busy chunk in the recipe");
}

#[test]
fn no_other_recipe_changes_its_hip_prefill_chunk() {
    for entry in std::fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../../recipes")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "toml") || path.file_stem().is_some_and(|s| s == "inkling2") {
            continue;
        }
        let r = intarsia_models::parse_recipe(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let tuned = std::fs::read_to_string(&path).unwrap().contains("[tuning.hip]");
        let hip = r.resolve_prefill_chunk(None, "hip").value;
        // a recipe without a HIP table resolves to its base value, the same as for cpu (which has no table in any shipped recipe)
        if !tuned {
            assert_eq!(hip, r.resolve_prefill_chunk(None, "cpu").value, "{}: no [tuning.hip], so the base chunk", path.display());
        }
    }
}
