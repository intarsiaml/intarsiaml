#![cfg(target_os = "macos")]
//! Lane mlen15: `INTARSIA_METAL_VERIFY16=0` turns the 9..=16-row bit-decode kernels off: today's (pre-lane mmoe16) kernels run.
//! (Own file: the knob is read once per process.)

use intarsia_backend_metal::verify16_enabled;

#[test]
fn knob_zero_turns_verify16_off() {
    std::env::set_var("INTARSIA_METAL_VERIFY16", "0");
    assert!(!verify16_enabled(), "knob 0: the old kernels must run");
}
