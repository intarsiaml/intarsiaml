//! Shared by the GPU tests: honest skips.
//!
//! A GPU test that cannot run (no HIP device, no checkpoint) must not look like a pass. `skip` prints a
//! `SKIPPED:` line on stdout and stderr (visible with `--nocapture`, and in the failure output) and, when the
//! run is declared to need a GPU (`CI=1` or `INTARSIA_REQUIRE_GPU=1`), panics so the skip fails the run.
#![allow(dead_code)]

pub mod tp_partial;

/// Is a skip a failure in this run?
pub fn require_gpu() -> bool {
    let set = |k: &str| std::env::var(k).is_ok_and(|v| !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false"));
    set("CI") || set("INTARSIA_REQUIRE_GPU")
}

/// Report a skipped test; panics when [`require_gpu`].
pub fn skip(test: &str, why: &str) {
    println!("SKIPPED: {test}: {why}");
    eprintln!("SKIPPED: {test}: {why}");
    if require_gpu() {
        panic!("{test} was SKIPPED ({why}) but this run requires a GPU (CI / INTARSIA_REQUIRE_GPU is set)");
    }
}
