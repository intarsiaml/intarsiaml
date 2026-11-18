//! Lane herdflip: with `INTARSIA_PREFIX_HERD` unset (the default since lane herdflip) the batched engine runs the prefix herd. (Own file: the knob is read from the environment when
//! `BatchedOptions::new` runs, and the test binaries of one crate share no process.)

use intarsia_core::BatchedOptions;

#[test]
fn knob_default() {
    std::env::remove_var("INTARSIA_PREFIX_HERD");
    assert!(BatchedOptions::new(4, 8).prefix_herd);
}
