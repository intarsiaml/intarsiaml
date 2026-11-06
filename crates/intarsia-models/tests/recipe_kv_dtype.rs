//! Lane spark35: `[tuning.<backend>] kv_dtype` wins over `[defaults] kv_dtype` for that backend only
//! (`Recipe::resolve_kv_dtype`); the Qwen3.6-35B recipe sets fp8_e4m3 on CUDA and keeps bf16 elsewhere.
use intarsia_core::DType;

#[test]
fn qwen35_recipe_kv_dtype_is_per_backend() {
    let r = intarsia_models::parse_recipe(include_str!("../../../recipes/qwen3_5_moe.toml")).unwrap();
    assert_eq!(r.resolve_kv_dtype("cuda").unwrap(), DType::Fp8E4m3);
    for b in ["cpu", "metal", "hip"] {
        assert_eq!(r.resolve_kv_dtype(b).unwrap(), DType::Bf16, "{b}");
    }
    assert_eq!(r.kv_dtype().unwrap(), DType::Bf16, "[defaults] stays bf16");
}

#[test]
fn other_recipes_have_no_backend_kv_dtype() {
    let r = intarsia_models::parse_recipe(include_str!("../../../recipes/qwen3.toml")).unwrap_or_else(|_| panic!("recipe parse"));
    assert_eq!(r.resolve_kv_dtype("cuda").unwrap(), r.kv_dtype().unwrap());
}
