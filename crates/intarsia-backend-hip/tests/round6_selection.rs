//! Round 6 (docs/results/2026-10-19-hip-round6.md) pins the kernel variants chosen inside two ops, as the kernel library
//! resolves them with no environment knob set: flash v2 starts from 4 query heads per block (INTARSIA_HIP_FLASH_HPB), and the
//! grouped MoE projections run variant 5 (skip empty fragments + occupancy) for the 64-pair tile and the round-5 kernel (0)
//! for the 32-pair tile (INTARSIA_HIP_MOE_MW). Host-only: no device needed.
//!
//! Its own test binary: it reads and sets environment variables, which no other test may race with. The knob values are
//! checked in child processes of this binary (flash reads its knob once per process).
#![cfg(ik_hip)]

use std::process::Command;

/// Runs `print_inner_variants` in a child process with `env` set; returns (hpb, gate_up mw64, mw32, down mw64, mw32).
fn child(env: &[(&str, &str)]) -> (i32, i32, i32, i32, i32) {
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
    let v: Vec<i32> = text[at + TAG.len()..].split_whitespace().take(5).map(|x| x.parse().unwrap()).collect();
    (v[0], v[1], v[2], v[3], v[4])
}

/// MoE projection variants with no knob set: gate_up (64-pair tile, 32-pair tile), down (64, 32). Round 6: (5, 0) for both;
/// round 7 (docs/results/2026-10-19-hip-round7.md): gate_up 45 = cvt | skip | occupancy | deep ring (moe_wmma_r7.h) and the
/// round-5 kernel at 32 pairs; down 25 = cvt | skip | two fragments, 41 = cvt | skip | deep ring.
const DEFAULTS: (i32, i32) = (45, 0);
const DEFAULTS_DN: (i32, i32) = (25, 41);

#[test]
#[ignore]
fn print_inner_variants() {
    let (a, b, c) = intarsia_backend_hip::inner_variants();
    let (d, e) = intarsia_backend_hip::moe_down_variants();
    println!("R6SEL {a} {b} {c} {d} {e}");
}

#[test]
fn round6_inner_variants_are_pinned() {
    // defaults: flash v2 HPB 4; MoE DEFAULTS (at 64 pairs, at 32)
    assert_eq!(child(&[]), (4, DEFAULTS.0, DEFAULTS.1, DEFAULTS_DN.0, DEFAULTS_DN.1));
    // every accepted flash value is taken, anything else falls back to 4
    for (v, want) in [("1", 1), ("2", 2), ("4", 4), ("3", 4), ("x", 4)] {
        assert_eq!(child(&[("INTARSIA_HIP_FLASH_HPB", v)]).0, want, "INTARSIA_HIP_FLASH_HPB={v}");
    }
    // the MoE knob overrides both tiles and both projections (round 6: 0, 1, 4, 5; round 7, moe_wmma_r7.h: 13, 25, 29, 41, 45),
    // "g:d" sets gate_up and down apart, and anything else falls back to the defaults (variants measured and dropped in round
    // 7, 8 9 40 61 93 125, and combinations never built such as 16 / 21)
    let (d, dd) = (DEFAULTS, DEFAULTS_DN);
    let mut table: Vec<(String, (i32, i32, i32, i32))> = Vec::new();
    for v in [0, 1, 4, 5, 13, 25, 29, 41, 45] {
        table.push((v.to_string(), (v, v, v, v)));
    }
    for v in ["3", "8", "9", "16", "21", "40", "61", "93", "125", "-1"] {
        table.push((v.to_string(), (d.0, d.1, dd.0, dd.1)));
    }
    table.push(("13:29".into(), (13, 13, 29, 29)));
    table.push(("0:41".into(), (0, 0, 41, 41)));
    table.push(("5:".into(), (5, 5, dd.0, dd.1)));
    table.push((":0".into(), (d.0, d.1, 0, 0)));
    table.push(("3:5".into(), (d.0, d.1, 5, 5)));
    for (v, want) in &table {
        let (v, want) = (v.as_str(), *want);
        let (_, a, b, c, e) = child(&[("INTARSIA_HIP_MOE_MW", v)]);
        assert_eq!((a, b, c, e), want, "INTARSIA_HIP_MOE_MW={v}");
    }
}
