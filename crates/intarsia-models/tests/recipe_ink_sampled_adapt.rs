//! The draft-length policy of sampled DSpark (`INTARSIA_SPEC_SAMPLED_DFLASH=1`, opt-in) the shipped Inkling-2-Mini recipe gives HIP and Metal
//! (lane ink-sampled-adapt, docs/results/2026-11-26-inkling2-sampled-adapt.md): a string that does not parse is dropped with a stderr line, so the
//! recipe's strings are checked here against the parameters they were measured with. Other backends keep the built-in table.

use intarsia_core::spec::{AdaptMode, AdaptParams};

fn mode(backend: &str) -> AdaptMode {
    let ink = intarsia_models::parse_recipe(include_str!("../../../recipes/inkling2.toml")).unwrap();
    AdaptMode::resolve(backend, ink.sampled_adapt(backend), None)
}

#[test]
fn hip_and_metal_carry_their_measured_tables() {
    let AdaptMode::On(h) = mode("hip") else { panic!("hip: policy on") };
    assert_eq!(h.rows, vec![0.273, 0.498, 0.69, 0.985, 1.211]);
    assert_eq!((h.draft, h.g, h.margin, h.drop, h.probe, h.hold0, h.hold_max), (0.234, 0.95, 0.0, 0.05, 1, 16, 64));
    let AdaptMode::On(m) = mode("metal") else { panic!("metal: policy on") };
    assert_eq!(m.rows, vec![0.236, 0.418, 0.583]);
    assert_eq!((m.draft, m.g, m.margin, m.drop, m.probe, m.hold0, m.hold_max), (0.237, 0.85, 0.0, 0.0, 3, 8, 32));
}

#[test]
fn other_backends_keep_the_built_in_table() {
    for b in ["cpu", "cuda"] {
        assert_eq!(mode(b), AdaptMode::On(AdaptParams::default_for(b)), "{b}");
    }
}
