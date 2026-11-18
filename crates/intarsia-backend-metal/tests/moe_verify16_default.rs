#![cfg(target_os = "macos")]
//! Lane mlen15: with `INTARSIA_METAL_VERIFY16` unset (the default since lane mlen15) `verify16_enabled()` is true, so 9..=16-row
//! `moe_experts_bf16_rows` calls take the bitwise-equal bit-decode kernels (`moe_verify16.rs` proves them). `=1` still works.
//! (Own file: the knob is read once per process.)

use intarsia_backend_metal::verify16_enabled;

#[test]
fn unset_turns_verify16_on() {
    std::env::remove_var("INTARSIA_METAL_VERIFY16");
    assert!(verify16_enabled(), "knob unset: the 9..=16-row bit-decode kernels must be on");
}
