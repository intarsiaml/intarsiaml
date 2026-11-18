#![cfg(target_os = "macos")]
//! Lane mlen15: `INTARSIA_METAL_VERIFY16=1` (the opt-in value before the flip) still turns the 9..=16-row kernels on.
//! (Own file: the knob is read once per process.)

use intarsia_backend_metal::verify16_enabled;

#[test]
fn knob_one_stays_on() {
    std::env::set_var("INTARSIA_METAL_VERIFY16", "1");
    assert!(verify16_enabled());
}
