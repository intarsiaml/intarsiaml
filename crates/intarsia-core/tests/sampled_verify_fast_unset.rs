//! Lane svfflip: with `INTARSIA_SAMPLED_VERIFY_FAST` unset, `BatchedOptions::new` is off and the env answer is `None` (the loader's recipe key decides). (Own file: the knob is read from the environment when `BatchedOptions::new` runs, and the
//! test binaries of one crate share no process.)

use intarsia_core::BatchedOptions;

#[test]
fn knob_unset() {
    std::env::remove_var("INTARSIA_SAMPLED_VERIFY_FAST");
    assert!(!BatchedOptions::new(4, 8).sampled_verify_fast, "unset: BatchedOptions::new is off; the loader turns it on from the 35B-on-CUDA recipe key");
    assert_eq!(intarsia_core::engine::sampled_verify_fast_env(), None);
}
