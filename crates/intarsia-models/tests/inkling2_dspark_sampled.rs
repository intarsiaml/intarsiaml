//! Lane ink-sampled (docs/results/2026-11-26-inkling2-sampled-dspark.md): sampled requests (temperature > 0) draft with Inkling-2's
//! DSpark draft when `INTARSIA_SPEC_SAMPLED_DFLASH` is on (`Model::set_spec_sampled_dflash` here), by the seed-exact keyed rule
//! (`engine/keyed.rs`): Match's acceptance with every verify row drawn as the plain step at its position draws it. Tiny fixtures of
//! `tests/inkling2_dspark.rs`. Backend: cpu, or Metal / CUDA / HIP with `INTARSIA_INKLING2_BACKEND=metal|cuda|hip`. On Metal the
//! recipe's device sampler draws the plain-temperature requests (`gumbel_sampling`), so that run covers the device-drawn verify rows;
//! the cpu run of the same path is `tests/inkling2_dspark_sampled_gumbel.rs`.

mod common;
mod ink_sampled;
use ink_sampled::*;

/// Gate 2: a seeded request emits the same ids with sampled DSpark speculation on as with `--spec off`, for every configuration
/// (the recipe's temperature 1.0 / 0.6 with top-p 1, and 0.7 with top-k 20 / top-p 0.9), with the random draft and with scripted
/// proposals that are accepted for every length 0..=k.
#[test]
fn sampled_dspark_seeded_ids_equal_spec_off() {
    if dirs().is_none() {
        eprintln!("skipped: no tiny fixtures");
        return;
    }
    gate_off();
    seeded_ids_equal_spec_off("host sampler (or the recipe's device sampler on Metal)");
}

/// Off unless asked (the knob's default is off on every backend): a sampled request under `--spec dflash` takes plain steps, a greedy
/// one drafts. With the knob on, greedy requests keep the greedy acceptance: their ids equal `--spec off`'s.
#[test]
fn sampled_dspark_is_opt_in_and_leaves_greedy_alone() {
    if dirs().is_none() {
        eprintln!("skipped: no tiny fixtures");
        return;
    }
    gate_off();
    assert_eq!(
        std::env::var("INTARSIA_SPEC_SAMPLED_DFLASH").ok(),
        None,
        "run without INTARSIA_SPEC_SAMPLED_DFLASH set"
    );
    let greedy = |p: &[u32], n: usize| intarsia_core::GenRequest {
        prompt: p.to_vec(),
        max_tokens: n,
        ignore_eos: true,
        ..Default::default()
    };
    let (off, _) = engine(None, None, None);
    for sampled in [None, Some(false)] {
        let (e, tel) = engine(Some(3), sampled, None);
        let _ = run(&e, sreq(PROMPTS[0], 24, 3, CFGS[0]));
        assert_eq!(tel.snapshot().spec_steps, 0, "{sampled:?}: a sampled request drafted");
        let _ = run(&e, greedy(PROMPTS[0], 24));
        assert!(tel.snapshot().spec_steps > 0, "{sampled:?}: a greedy request did not draft");
    }
    let (on, tel) = engine(Some(3), Some(true), None);
    for p in PROMPTS {
        for n in [1usize, 5, 40] {
            assert_eq!(run(&on, greedy(p, n)), run(&off, greedy(p, n)), "greedy ids changed, prompt {p:?} n {n}");
        }
    }
    assert!(tel.snapshot().spec_steps > 0);
}

/// Gate 1: the emitted tokens' law with sampled DSpark speculation equals plain sampling's (chi-square of homogeneity per position,
/// disjoint seed sets), one test per configuration so they run in parallel. About 2 x 1500 short requests each; opt-in:
/// `cargo test --release -p intarsia-models --test inkling2_dspark_sampled -- --ignored law --nocapture`.
fn law(i: usize) {
    if dirs().is_none() {
        eprintln!("skipped: no tiny fixtures");
        return;
    }
    gate_off();
    let n = std::env::var("INTARSIA_LAW_N").ok().and_then(|v| v.parse().ok()).unwrap_or(1500);
    law_test("fixture", n, &[CFGS[i]]);
}

#[test]
#[ignore]
fn law_t10_recipe_top_p() {
    law(0);
}

#[test]
#[ignore]
fn law_t06_recipe_top_p() {
    law(1);
}

#[test]
#[ignore]
fn law_t07_top_k20_top_p09() {
    law(2);
}
