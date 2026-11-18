//! Lane herdflip: with `INTARSIA_PREFIX_HERD` `1` (the old opt-in spelling) the prefix herd is on. (Own file: the knob is read from the environment when
//! `BatchedOptions::new` runs, and the test binaries of one crate share no process.)

use intarsia_core::BatchedOptions;

#[test]
fn knob_one() {
    std::env::set_var("INTARSIA_PREFIX_HERD", "1");
    assert!(BatchedOptions::new(4, 8).prefix_herd);
}
