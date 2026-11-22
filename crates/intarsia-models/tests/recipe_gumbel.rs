//! The device sampler's recipe default (user decision 2026-11-22): only the Inkling-2 recipe turns it on, and only on Metal;
//! every other shipped recipe and backend leaves it unset (off). `INTARSIA_GUMBEL_SAMPLING` wins when set (`gumbel::resolve`).

use intarsia_core::gumbel::resolve;

#[test]
fn only_inkling2_on_metal_turns_the_device_sampler_on() {
    let ink = intarsia_models::parse_recipe(include_str!("../../../recipes/inkling2.toml")).unwrap();
    assert_eq!(ink.gumbel_sampling("metal"), Some(true));
    assert!(resolve(None, ink.gumbel_sampling("metal")), "unset: on for Inkling-2 on Metal");
    assert!(!resolve(Some("0"), ink.gumbel_sampling("metal")), "INTARSIA_GUMBEL_SAMPLING=0 reverts");
    for b in ["cpu", "cuda", "hip"] {
        assert_eq!(ink.gumbel_sampling(b), None, "Inkling-2 on {b}: no recipe key");
    }
    for entry in std::fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../../recipes")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "toml") || path.file_stem().is_some_and(|s| s == "inkling2") {
            continue;
        }
        let r = intarsia_models::parse_recipe(&std::fs::read_to_string(&path).unwrap()).unwrap();
        for b in ["cpu", "cuda", "hip", "metal"] {
            assert_eq!(r.gumbel_sampling(b), None, "{}: {b} sets gumbel_sampling", path.display());
        }
    }
}
