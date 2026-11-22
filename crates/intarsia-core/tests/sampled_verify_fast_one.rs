//! Lane svfflip: with `INTARSIA_SAMPLED_VERIFY_FAST` `1` forces it on. (Own file: the knob is read from the environment when `BatchedOptions::new` runs, and the
//! test binaries of one crate share no process.)

use intarsia_core::BatchedOptions;

#[test]
fn knob_one() {
    std::env::set_var("INTARSIA_SAMPLED_VERIFY_FAST", "1");
    assert!(BatchedOptions::new(4, 8).sampled_verify_fast);
    assert_eq!(intarsia_core::engine::sampled_verify_fast_env(), Some(true));
}
