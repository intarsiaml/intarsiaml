//! Lane spark35 (user decision 2026-11-06: flip the C default, keep D opt-in): the shipped recipes store the GatedDeltaNet delta-rule state
//! bf16 on CUDA for the Qwen3.6-35B-A3B only (`[tuning.cuda] gdn_state`); every other backend and recipe keeps f32. `INTARSIA_GDN_STATE=f32`
//! (read in `Model::load`) is the off switch over the recipe key.
use std::path::Path;

#[test]
fn only_the_35b_on_cuda_defaults_to_a_bf16_gdn_state() {
    let moe = intarsia_models::parse_recipe(include_str!("../../../recipes/qwen3_5_moe.toml")).unwrap();
    assert_eq!(moe.resolve_gdn_state("cuda"), Some("bf16"));
    for b in ["cpu", "metal", "hip"] {
        assert_eq!(moe.resolve_gdn_state(b), None, "{b}");
    }
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../recipes");
    let mut set = Vec::new();
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.extension().is_some_and(|x| x == "toml") {
            let r = intarsia_models::parse_recipe(&std::fs::read_to_string(&p).unwrap()).unwrap();
            for b in ["cpu", "cuda", "metal", "hip"] {
                if r.resolve_gdn_state(b).is_some() {
                    set.push(format!("{}:{b}", p.file_name().unwrap().to_string_lossy()));
                }
            }
        }
    }
    assert_eq!(set, ["qwen3_5_moe.toml:cuda"], "exactly the measured pair");
}
