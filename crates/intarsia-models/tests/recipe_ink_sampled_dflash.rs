//! Lane ink-hip-specflip, flip A (user decision 2026-11-26 "Both, confirm first"; docs/results/2026-11-26-inkling2-hip-specflip.md):
//! sampled requests draft with Inkling-2's own DSpark draft by default on HIP only, through the recipe key `[tuning.hip]
//! sampled_dflash = true` (Metal failed the sampled-adapt gate and stays opt-in; CUDA and cpu unchanged). The resolution the model
//! applies ([`sampled_dflash_resolve`]): `INTARSIA_SPEC_SAMPLED_DFLASH` (or `Model::set_spec_sampled_dflash`) wins, `0` reverts; else
//! the recipe's key, which counts only for the checkpoint's own DSpark draft (an explicit `--spec dflash --spec-draft <DFlash2 dir>`
//! keeps sampled requests plain unless the variable says `1`); else off.

use intarsia_core::spec::sampled_dflash_resolve;
use std::path::PathBuf;

fn ink() -> intarsia_core::recipe::Recipe {
    intarsia_models::parse_recipe(include_str!("../../../recipes/inkling2.toml")).unwrap()
}

#[test]
fn on_for_inkling2_on_hip_only() {
    let ink = ink();
    assert_eq!(ink.sampled_dflash("hip"), Some(true), "recipes/inkling2.toml [tuning.hip] sampled_dflash");
    for b in ["metal", "cuda", "cpu"] {
        assert_eq!(ink.sampled_dflash(b), None, "Inkling-2 on {b}: no recipe key");
    }
    // the default (variable unset) with the model's own DSpark draft: on for hip, off for metal / cuda / cpu
    assert!(sampled_dflash_resolve(None, ink.sampled_dflash("hip"), true), "hip: on by default");
    for b in ["metal", "cuda", "cpu"] {
        assert!(!sampled_dflash_resolve(None, ink.sampled_dflash(b), true), "{b}: off by default");
    }
}

#[test]
fn the_variable_wins_and_zero_reverts() {
    let ink = ink();
    // the revert switch: INTARSIA_SPEC_SAMPLED_DFLASH=0 restores today's behaviour (plain steps for sampled requests)
    assert!(!sampled_dflash_resolve(Some(false), ink.sampled_dflash("hip"), true));
    // the opt-in elsewhere is unchanged
    for b in ["metal", "cuda", "cpu", "hip"] {
        assert!(sampled_dflash_resolve(Some(true), ink.sampled_dflash(b), true), "{b}: =1 turns it on");
        assert!(!sampled_dflash_resolve(Some(false), ink.sampled_dflash(b), true), "{b}: =0 turns it off");
    }
    // the recipe's default only covers the checkpoint's own DSpark draft; =1 still turns any DFlash-family draft on
    assert!(!sampled_dflash_resolve(None, ink.sampled_dflash("hip"), false), "hip with another draft: off by default");
    assert!(sampled_dflash_resolve(Some(true), ink.sampled_dflash("hip"), false));
}

#[test]
fn no_other_recipe_sets_it() {
    for entry in std::fs::read_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../recipes")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "toml") || path.file_stem().is_some_and(|s| s == "inkling2") {
            continue;
        }
        let r = intarsia_models::parse_recipe(&std::fs::read_to_string(&path).unwrap()).unwrap();
        for b in ["cpu", "cuda", "hip", "metal"] {
            assert_eq!(r.sampled_dflash(b), None, "{}: {b} sets sampled_dflash", path.display());
        }
    }
}
