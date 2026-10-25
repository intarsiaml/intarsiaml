//! Lane l5 (docs/results/2026-10-25-l5-moe-fp8-wmma.md): the opt-in fp8-WMMA routed experts, host side. Pins that
//!  * with no knob set the kernel library reports it off and `moe_experts_scratch_hip` is exactly the core `moe_experts_scratch` (the default
//!    plan, and so the no-op gate, does not move);
//!  * INTARSIA_HIP_MOE_FP8_WMMA turns it on only for exactly "1", INTARSIA_HIP_MOE_FP8_MIN lowers the token threshold (never below the
//!    tensor-core path's 16);
//!  * the Rust scratch formula equals the kernel library's (`moe_f8_extra_bytes`, kernels/hip/moe.hip) for the 35B's shape and for
//!    shapes the quantiser does not take (0).
//! Host-only: no device needed. Its own test binary (the knobs are read once per process, in child processes of this binary).
#![cfg(ik_hip)]

use intarsia_backend_hip::quantmoe::{moe_experts_scratch_hip, moe_f8_extra, moe_f8_from};
use intarsia_core::quant::ops::moe_experts_scratch;
use intarsia_core::ShapeInfo;
use std::process::Command;

fn shape(m: usize, h: usize, i: usize) -> ShapeInfo {
    ShapeInfo { m, k: h, n: i, experts: 256, top_k: 8, ..Default::default() }
}

/// (on, min tokens, extra bytes of the library for T = 4096 / H = 2048 / I = 512, scratch fn of this process, core scratch)
fn child(env: &[(&str, &str)]) -> (bool, i64, i64, usize, usize) {
    let mut c = Command::new(std::env::current_exe().unwrap());
    c.args(["--ignored", "--exact", "print_f8_state", "--nocapture", "--test-threads=1"]);
    c.env_remove("INTARSIA_HIP_MOE_FP8_WMMA").env_remove("INTARSIA_HIP_MOE_FP8_MIN");
    for (k, v) in env {
        c.env(k, v);
    }
    let out = c.output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    const TAG: &str = "L5F8 ";
    let at = text.find(TAG).unwrap_or_else(|| panic!("no output: {text}"));
    let v: Vec<i64> = text[at + TAG.len()..].split_whitespace().take(5).map(|x| x.parse().unwrap()).collect();
    (v[0] != 0, v[1], v[2], v[3] as usize, v[4] as usize)
}

#[test]
#[ignore]
fn print_f8_state() {
    let (on, min, extra) = intarsia_backend_hip::moe_f8_state(4096, 8, 2048, 512);
    let s = shape(4096, 2048, 512);
    println!("L5F8 {} {min} {extra} {} {}", on as i32, moe_experts_scratch_hip(&s), moe_experts_scratch(&s));
}

#[test]
fn f8_knob_is_off_by_default_and_the_scratch_matches() {
    let s = shape(4096, 2048, 512);
    // off: the library reports 0 extra bytes and the scratch fn is the core one, byte for byte
    let (on, min, extra, scratch, core) = child(&[]);
    assert!(!on);
    assert_eq!(min, 1025, "the default threshold");
    assert_eq!(extra, 0);
    assert_eq!(scratch, core);
    // anything but "1" is off
    for v in ["0", "", "true", "11", " 1"] {
        let (on, _, extra, scratch, core) = child(&[("INTARSIA_HIP_MOE_FP8_WMMA", v)]);
        assert!(!on && extra == 0 && scratch == core, "INTARSIA_HIP_MOE_FP8_WMMA={v:?}");
    }
    // on: the library and the Rust formula agree
    let (on, _, extra, scratch, core) = child(&[("INTARSIA_HIP_MOE_FP8_WMMA", "1")]);
    assert!(on);
    assert_eq!(extra as usize, moe_f8_extra(&s));
    assert_eq!(scratch, core + extra as usize);
    // the threshold knob: lowered values are taken, never below 16, garbage falls back to 0 -> 16 (atoll), default when unset
    for (v, want) in [("16", 16), ("64", 64), ("4", 16), ("1025", 1025), ("2048", 2048)] {
        let (_, min, ..) = child(&[("INTARSIA_HIP_MOE_FP8_MIN", v)]);
        assert_eq!(min, want, "INTARSIA_HIP_MOE_FP8_MIN={v}");
    }
}

#[test]
fn f8_extra_formula() {
    assert!(moe_f8_from(Some("1")) && !moe_f8_from(Some("0")) && !moe_f8_from(None) && !moe_f8_from(Some("on")));
    // T = 4096, P = 32768, H = 2048, I = 512: 8 MiB + 16 KiB + 16 MiB + 128 KiB
    assert_eq!(moe_f8_extra(&shape(4096, 2048, 512)), 4096 * 2048 + 4096 * 4 + 32768 * 512 + 32768 * 4);
    // shapes the quantiser does not take (K not in 256 * {1, 2, 4, 8, 16}) add nothing, the bf16 path runs
    assert_eq!(moe_f8_extra(&shape(4096, 2048, 192)), 0);
    assert_eq!(moe_f8_extra(&shape(4096, 1536, 512)), 0);
}
