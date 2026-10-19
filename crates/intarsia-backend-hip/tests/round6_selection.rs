//! Round 6 (docs/results/2026-10-19-hip-round6.md) pins the kernel variants chosen inside two ops, as the kernel library
//! resolves them with no environment knob set: flash v2 starts from 4 query heads per block (INTARSIA_HIP_FLASH_HPB), and the
//! grouped MoE projections run variant 5 (skip empty fragments + occupancy) for the 64-pair tile and the round-5 kernel (0)
//! for the 32-pair tile (INTARSIA_HIP_MOE_MW). Host-only: no device needed.
//!
//! Its own test binary: it reads and sets environment variables, which no other test may race with. The knob values are
//! checked in child processes of this binary (flash reads its knob once per process).
#![cfg(ik_hip)]

use std::process::Command;

/// Runs `print_inner_variants` in a child process with `env` set; returns (hpb, mw64, mw32).
fn child(env: &[(&str, &str)]) -> (i32, i32, i32) {
    let mut c = Command::new(std::env::current_exe().unwrap());
    c.args(["--ignored", "--exact", "print_inner_variants", "--nocapture", "--test-threads=1"]);
    c.env_remove("INTARSIA_HIP_FLASH_HPB").env_remove("INTARSIA_HIP_MOE_MW");
    for (k, v) in env {
        c.env(k, v);
    }
    let out = c.output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    // libtest prints "test print_inner_variants ... " first, so the values can share that line (and the test's own name
    // contains "inner_variants": the tag must not)
    const TAG: &str = "R6SEL ";
    let at = text.find(TAG).unwrap_or_else(|| panic!("no output: {text}"));
    let v: Vec<i32> = text[at + TAG.len()..].split_whitespace().take(3).map(|x| x.parse().unwrap()).collect();
    (v[0], v[1], v[2])
}

#[test]
#[ignore]
fn print_inner_variants() {
    let (a, b, c) = intarsia_backend_hip::inner_variants();
    println!("R6SEL {a} {b} {c}");
}

#[test]
fn round6_inner_variants_are_pinned() {
    // defaults: flash v2 HPB 4; MoE 5 (kMwSkip | kMwOcc) at 64 pairs, the round-5 kernel at 32
    assert_eq!(child(&[]), (4, 5, 0));
    // every accepted flash value is taken, anything else falls back to 4
    for (v, want) in [("1", 1), ("2", 2), ("4", 4), ("3", 4), ("x", 4)] {
        assert_eq!(child(&[("INTARSIA_HIP_FLASH_HPB", v)]).0, want, "INTARSIA_HIP_FLASH_HPB={v}");
    }
    // the MoE knob overrides both tiles (0, 1, 4, 5) and falls back to the defaults for any other value
    for (v, want) in [("0", (0, 0)), ("1", (1, 1)), ("4", (4, 4)), ("5", (5, 5)), ("3", (5, 0)), ("8", (5, 0)), ("-1", (5, 0))] {
        let (_, a, b) = child(&[("INTARSIA_HIP_MOE_MW", v)]);
        assert_eq!((a, b), want, "INTARSIA_HIP_MOE_MW={v}");
    }
}
