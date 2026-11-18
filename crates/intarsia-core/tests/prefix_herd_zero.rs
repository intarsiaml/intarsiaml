//! Lane herdflip: with `INTARSIA_PREFIX_HERD` `0` the prefix herd is off (requests look the store up at admission only). (Own file: the knob is read from the environment when
//! `BatchedOptions::new` runs, and the test binaries of one crate share no process.)

use intarsia_core::BatchedOptions;

#[test]
fn knob_zero() {
    std::env::set_var("INTARSIA_PREFIX_HERD", "0");
    assert!(!BatchedOptions::new(4, 8).prefix_herd);
}
