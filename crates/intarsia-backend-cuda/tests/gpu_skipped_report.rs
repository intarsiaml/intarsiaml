//! Honest accounting when this crate was built without CUDA (`ik_cuda` unset: no `nvcc`, or `INTARSIA_NO_CUDA=1`).
//!
//! `admission.rs`, `attention.rs`, `quantmoe.rs` and `gdn_timing.rs` are compiled out in that case, so a plain
//! `cargo test` reports no failures although no GPU code ran. This test says so (run with `--nocapture` to see
//! it) and fails when the run is declared to need a GPU (`CI=1` or `INTARSIA_REQUIRE_GPU=1`).
#![cfg(not(ik_cuda))]

mod common;

#[test]
fn gpu_tests_were_not_built() {
    common::skip(
        "intarsia-backend-cuda gpu tests (admission, attention, quantmoe, gdn_timing)",
        "built without CUDA (ik_cuda cfg unset): none of them ran; run `cargo test --release --features cuda` on a GPU box",
    );
}
