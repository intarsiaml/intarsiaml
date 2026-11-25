//! The busy prefill chunk (the chunk of a prompt that prefills beside decoding streams under `serve --max-batch` > 1) the shipped
//! Inkling-2-Mini recipe gives Metal (user decision 2026-11-25: 2048 instead of the builtin 256; docs/results/2026-11-25-ink-busy2048.md).
//! Only that recipe on that backend sets it: every other recipe and backend keeps the loader's rule. `--busy-chunk` wins over the recipe
//! (`load.rs`: `opts.busy_chunk` is read before `resolve_busy_chunk`).

#[test]
fn only_inkling2_on_metal_sets_a_2048_busy_chunk() {
    let ink = intarsia_models::parse_recipe(include_str!("../../../recipes/inkling2.toml")).unwrap();
    assert_eq!(ink.resolve_busy_chunk("metal"), Some(2048));
    assert_eq!(ink.resolve_prefill_chunk(None, "metal").value, 2048, "the busy chunk is the solo chunk, so the engine's clamp keeps it");
    for b in ["cpu", "cuda", "hip"] {
        assert_eq!(ink.resolve_busy_chunk(b), None, "Inkling-2 on {b}: no recipe busy chunk");
    }
    for entry in std::fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../../recipes")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "toml") || path.file_stem().is_some_and(|s| s == "inkling2") {
            continue;
        }
        let r = intarsia_models::parse_recipe(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        for b in ["cpu", "cuda", "hip", "metal"] {
            // the recipes that already set a busy chunk keep it; no other recipe gains one
            let want = match (stem.as_str(), b) {
                ("qwen3_5_vl", "hip" | "metal") => Some(256),
                ("qwen3_5_moe", "metal") => Some(2048),
                _ => None,
            };
            assert_eq!(r.resolve_busy_chunk(b), want, "{}: {b} busy chunk", path.display());
        }
    }
}
