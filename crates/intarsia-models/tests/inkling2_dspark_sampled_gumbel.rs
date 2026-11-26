//! Lane ink-sampled: `tests/inkling2_dspark_sampled.rs`'s seeded gate with the device sampler on the cpu backend
//! (`INTARSIA_GUMBEL_SAMPLING=1`, the reference `GumbelArgmax`): plain-temperature requests draw every token, verify rows included
//! (`Model::forward_verify_gumbel`), with the position-keyed Gumbel draw, and the ids equal `--spec off`'s. The switch is a process
//! variable read when a model and a request are set up, so this binary holds ONE test that sets it first.

mod common;
mod ink_sampled;
use ink_sampled::*;

#[test]
fn sampled_dspark_device_sampler_seeded_ids_equal_spec_off() {
    if dirs().is_none() {
        eprintln!("skipped: no tiny fixtures");
        return;
    }
    std::env::set_var("INTARSIA_GUMBEL_SAMPLING", "1");
    gate_off();
    let (m, _) = model(Some(3), Some(true), None);
    assert!(m.gumbel_ok() && m.verify_gumbel_ok(), "the device sampler is not on (vacuous test)");
    drop(m);
    let (_, prop, acc) = seeded_ids_equal_spec_off("device sampler (cpu GumbelArgmax)");
    assert!(acc > 0 && prop > acc);
}
