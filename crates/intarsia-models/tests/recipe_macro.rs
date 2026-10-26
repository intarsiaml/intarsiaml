//! The macro prefill pass sizes the shipped 35B recipe gives each backend (lane l7m: HIP 8192 rows = two 4096-row chunks; docs/results/2026-10-26-l7m-hip-macro-prefill.md).

#[test]
fn shipped_35b_recipe_macro_sizes() {
    let r = intarsia_models::parse_recipe(include_str!("../../../recipes/qwen3_5_moe.toml")).unwrap();
    assert_eq!(r.resolve_prefill_macro("cuda").0, 8192);
    assert_eq!(r.resolve_prefill_macro("hip").0, 8192);
    assert_eq!(r.resolve_prefill_macro("metal").0, 0, "Metal has no macro default");
    assert_eq!(r.resolve_prefill_macro("cpu").0, 0);
    assert_eq!(r.resolve_prefill_chunk(None, "hip").value, 4096, "the HIP chunk stays 4096: the macro size is two of them");
    assert_eq!(intarsia_core::scheduler::effective_macro_rows(16384, 4096), 16384);
    assert_eq!(intarsia_core::scheduler::effective_macro_rows(16384, 2048), 16384);
}
