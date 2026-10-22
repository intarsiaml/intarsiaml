//! Lane pf (docs/results/2026-10-22-pf-prefill-ctx.md) pins the flash v2 staging / math variant (INTARSIA_HIP_FLASH_VP, a bit
//! mask) as the kernel library resolves it: 5 with no knob set (V pair staging + skipped no-op rescale), the accepted values
//! 0, 1, 3, 5 and 7 as given, anything else falls back to 5. Every value gives the same bits as the others
//! (tests/flash_v2.rs checks that on a GPU). Host-only: no device needed.
//!
//! Its own test binary: it sets environment variables for child processes of this binary (flash reads its knob once per
//! process).
#![cfg(ik_hip)]

use std::process::Command;

fn child(env: Option<&str>) -> i32 {
    let mut c = Command::new(std::env::current_exe().unwrap());
    c.args(["--ignored", "--exact", "print_flash_vp", "--nocapture", "--test-threads=1"]);
    c.env_remove("INTARSIA_HIP_FLASH_VP");
    if let Some(v) = env {
        c.env("INTARSIA_HIP_FLASH_VP", v);
    }
    let out = c.output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    const TAG: &str = "PFVP ";
    let at = text.find(TAG).unwrap_or_else(|| panic!("no output: {text}"));
    text[at + TAG.len()..].split_whitespace().next().unwrap().parse().unwrap()
}

#[test]
#[ignore]
fn print_flash_vp() {
    println!("PFVP {}", intarsia_backend_hip::flash_vp());
}

#[test]
fn flash_vp_is_pinned() {
    assert_eq!(child(None), 5, "default");
    for v in [0, 1, 3, 5, 7] {
        assert_eq!(child(Some(&v.to_string())), v, "INTARSIA_HIP_FLASH_VP={v}");
    }
    for v in ["2", "4", "6", "8", "-1", "x", ""] {
        assert_eq!(child(Some(v)), 5, "INTARSIA_HIP_FLASH_VP={v:?}");
    }
}
