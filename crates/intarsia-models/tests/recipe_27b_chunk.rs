//! The prefill chunk the shipped Qwen3.8-27B recipe gives each backend (lane H27P: HIP 2048 rows instead of the builtin 256;
//! docs/results/2026-10-27-h27p-hip-27b-prefill.md). Metal 2048 as well (lane m27, docs/results/2026-11-02-m27-metal.md). CUDA was not measured for the 27B and keeps the builtin default.

#[test]
fn shipped_27b_recipe_prefill_chunk() {
    let r = intarsia_models::parse_recipe(include_str!("../../../recipes/qwen3_5_vl.toml")).unwrap();
    let hip = r.resolve_prefill_chunk(None, "hip");
    assert_eq!(hip.value, 2048, "HIP 27B prefill chunk (source {})", hip.source);
    assert_eq!(r.resolve_prefill_chunk(Some(256), "hip").value, 256, "--prefill-chunk wins over the recipe");
    let metal = r.resolve_prefill_chunk(None, "metal");
    assert_eq!(metal.value, 2048, "Metal 27B prefill chunk (source {})", metal.source);
    for b in ["cuda", "cpu"] {
        assert_eq!(
            r.resolve_prefill_chunk(None, b).value,
            intarsia_core::recipe::DEFAULT_PREFILL_CHUNK,
            "{b} keeps the builtin chunk"
        );
    }
    assert_eq!(r.resolve_prefill_peel("hip").0, 16, "tails of 1..16 rows keep the 256 schedule (small-M row kernels)");
    assert_eq!(r.resolve_prefill_peel("metal").0, 8, "tails of 1..8 rows keep the busy schedule (GEMV at M <= 8)");
    for b in ["cuda", "cpu"] {
        assert_eq!(r.resolve_prefill_peel(b).0, 0, "{b} has no peel");
    }
    assert_eq!(r.resolve_busy_chunk("hip"), Some(256), "beside decoding streams the prefill chunk stays 256");
    assert_eq!(r.resolve_busy_chunk("metal"), Some(256), "as on HIP");
    for b in ["cuda", "cpu"] {
        assert_eq!(r.resolve_busy_chunk(b), None, "{b} has no recipe busy chunk");
    }
    // whole base chunks: a multiple of the 256-row builtin chunk
    assert_eq!(hip.value % intarsia_core::recipe::DEFAULT_PREFILL_CHUNK, 0);
    assert_eq!(metal.value % intarsia_core::recipe::DEFAULT_PREFILL_CHUNK, 0);
}

#[test]
fn shipped_27b_recipe_tp_partial() {
    // bf16 all-reduce partials on HIP only (lane TPB, docs/results/2026-10-27-tpb-hip-tp-partials.md); CUDA, Metal and CPU keep the engine default (f32)
    let text = include_str!("../../../recipes/qwen3_5_vl.toml");
    let r = intarsia_models::parse_recipe(text).unwrap();
    assert_eq!(r.resolve_tp_partial("hip"), Some("bf16"));
    for b in ["cuda", "metal", "cpu"] {
        assert_eq!(r.resolve_tp_partial(b), None, "{b} keeps the f32 partials");
    }
    // a value other than f32 / bf16 is refused when the recipe is read
    let bad = text.replace("tp_partial = \"bf16\"", "tp_partial = \"fp16\"");
    assert_ne!(bad, text);
    assert!(intarsia_models::parse_recipe(&bad).is_err(), "tp_partial = fp16 must be refused");
}

/// fp8 lm_head by default on Metal, HIP and CUDA (user decisions 2026-11-06, docs/results/2026-11-06-lmhead-fp8-default.md); the cpu backend unset;
/// every off spelling of `INTARSIA_LMHEAD_QUANT` beats the recipe key, and the other recipes (8B, 35B, Llama) set nothing.
#[test]
fn shipped_27b_recipe_lm_head_quant() {
    use intarsia_core::quant::head::HeadQuant;
    let r = intarsia_models::parse_recipe(include_str!("../../../recipes/qwen3_5_vl.toml")).unwrap();
    for b in ["metal", "hip", "cuda"] {
        let key = r.lm_head_quant(b);
        assert_eq!(key, Some("fp8"), "{b}");
        assert_eq!(HeadQuant::resolve(None, key).unwrap(), Some(HeadQuant::Fp8), "{b}: default on");
        for off in ["off", "none", "0", "", "bf16"] {
            assert_eq!(HeadQuant::resolve(Some(off), key).unwrap(), None, "{b}: INTARSIA_LMHEAD_QUANT={off:?} beats the recipe");
        }
        assert_eq!(HeadQuant::resolve(Some("mxfp4"), key).unwrap(), Some(HeadQuant::Mxfp4), "{b}: the variable also picks another format");
    }
    assert_eq!(r.lm_head_quant("cpu"), None, "cpu: no recipe key");
    for (name, text) in [
        ("qwen3", include_str!("../../../recipes/qwen3.toml")),
        ("qwen3_5_moe", include_str!("../../../recipes/qwen3_5_moe.toml")),
        ("llama", include_str!("../../../recipes/llama.toml")),
        ("qwen3_5", include_str!("../../../recipes/qwen3_5.toml")),
    ] {
        let o = intarsia_models::parse_recipe(text).unwrap();
        for b in ["metal", "hip", "cuda", "cpu"] {
            assert_eq!(o.lm_head_quant(b), None, "{name} on {b}");
        }
    }
}
