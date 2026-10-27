//! The prefill chunk the shipped Qwen3.8-27B recipe gives each backend (lane H27P: HIP 2048 rows instead of the builtin 256;
//! docs/results/2026-10-27-h27p-hip-27b-prefill.md). CUDA and Metal were not measured for the 27B and keep the builtin default.

#[test]
fn shipped_27b_recipe_prefill_chunk() {
    let r = intarsia_models::parse_recipe(include_str!("../../../recipes/qwen3_5_vl.toml")).unwrap();
    let hip = r.resolve_prefill_chunk(None, "hip");
    assert_eq!(hip.value, 2048, "HIP 27B prefill chunk (source {})", hip.source);
    assert_eq!(r.resolve_prefill_chunk(Some(256), "hip").value, 256, "--prefill-chunk wins over the recipe");
    for b in ["cuda", "metal", "cpu"] {
        assert_eq!(
            r.resolve_prefill_chunk(None, b).value,
            intarsia_core::recipe::DEFAULT_PREFILL_CHUNK,
            "{b} keeps the builtin chunk"
        );
    }
    assert_eq!(r.resolve_prefill_peel("hip").0, 16, "tails of 1..16 rows keep the 256 schedule (small-M row kernels)");
    for b in ["cuda", "metal", "cpu"] {
        assert_eq!(r.resolve_prefill_peel(b).0, 0, "{b} has no peel");
    }
    assert_eq!(r.resolve_busy_chunk("hip"), Some(256), "beside decoding streams the prefill chunk stays 256");
    for b in ["cuda", "metal", "cpu"] {
        assert_eq!(r.resolve_busy_chunk(b), None, "{b} has no recipe busy chunk");
    }
    // whole base chunks: a multiple of the 256-row builtin chunk
    assert_eq!(hip.value % intarsia_core::recipe::DEFAULT_PREFILL_CHUNK, 0);
}
