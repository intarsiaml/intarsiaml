//! Lane svfflip: with `INTARSIA_SAMPLED_VERIFY_FAST` `0` is off (it wins over a recipe default). (Own file: the knob is read from the environment when `BatchedOptions::new` runs, and the
//! test binaries of one crate share no process.)

use intarsia_core::BatchedOptions;

#[test]
fn knob_zero() {
    std::env::set_var("INTARSIA_SAMPLED_VERIFY_FAST", "0");
    assert!(!BatchedOptions::new(4, 8).sampled_verify_fast);
    assert_eq!(intarsia_core::engine::sampled_verify_fast_env(), Some(false));
}
