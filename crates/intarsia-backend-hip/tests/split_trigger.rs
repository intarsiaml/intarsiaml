//! Round 9 (docs/results/2026-10-19-hip-round9.md): the launch-piece rule of round 8 (kernels/hip/ik_hip.h
//! `ik_split_pieces`) splits a long prefill launch only when its estimate exceeds INTARSIA_HIP_SPLIT_MIN_US (default 950),
//! into pieces of at most INTARSIA_HIP_SPLIT_US (default 600). Pins the rule on the host side (no device needed): the
//! defaults, the round-8 rule with the trigger at 0, the never-split switch, and the cap. Bits are covered elsewhere
//! (tests/moe_pairs.rs and tests/gdn_v2.rs force pieces and compare against one launch).
//!
//! Its own test binary with a single test: it sets environment variables, which no other test may race with.
#![cfg(ik_hip)]

use intarsia_backend_hip::split_pieces;

fn with(us: Option<&str>, min: Option<&str>, f: impl FnOnce()) {
    match us {
        Some(v) => std::env::set_var("INTARSIA_HIP_SPLIT_US", v),
        None => std::env::remove_var("INTARSIA_HIP_SPLIT_US"),
    }
    match min {
        Some(v) => std::env::set_var("INTARSIA_HIP_SPLIT_MIN_US", v),
        None => std::env::remove_var("INTARSIA_HIP_SPLIT_MIN_US"),
    }
    f();
    std::env::remove_var("INTARSIA_HIP_SPLIT_US");
    std::env::remove_var("INTARSIA_HIP_SPLIT_MIN_US");
}

#[test]
fn split_trigger_rule() {
    // defaults: budget 600, trigger 950
    with(None, None, || {
        assert_eq!(split_pieces(500.0, 64), 1);
        assert_eq!(split_pieces(788.0, 64), 1, "a ~1024-token GDN launch (0.77 us/token) must stay one launch");
        assert_eq!(split_pieces(949.0, 64), 1);
        assert_eq!(split_pieces(950.0, 64), 1, "the trigger is exclusive");
        assert_eq!(split_pieces(951.0, 64), 2);
        assert_eq!(split_pieces(1138.0, 64), 2, "MoE down<64> at 4096 tokens: 2 pieces, as in round 8");
        assert_eq!(split_pieces(3154.0, 4096), 6, "GDN at 4096 tokens: 6 pieces, as in round 8");
        assert_eq!(split_pieces(3154.0, 3), 3, "capped at max_pieces");
        assert_eq!(split_pieces(5000.0, 1), 1);
    });
    // round 8's rule: trigger 0, split whenever the estimate exceeds the budget
    with(None, Some("0"), || {
        assert_eq!(split_pieces(600.0, 64), 1);
        assert_eq!(split_pieces(601.0, 64), 2);
        assert_eq!(split_pieces(788.0, 64), 2);
        assert_eq!(split_pieces(3154.0, 4096), 6);
    });
    // the budget still sizes the pieces above the trigger; a small budget forces the most pieces only past the trigger
    with(Some("1"), None, || {
        assert_eq!(split_pieces(900.0, 64), 1);
        assert_eq!(split_pieces(1000.0, 64), 64);
    });
    with(Some("1"), Some("0"), || assert_eq!(split_pieces(5.0, 64), 6));
    with(Some("0"), Some("0"), || assert_eq!(split_pieces(5000.0, 64), 1, "INTARSIA_HIP_SPLIT_US=0 never splits"));
    with(Some("400"), Some("2000"), || {
        assert_eq!(split_pieces(1999.0, 64), 1);
        assert_eq!(split_pieces(2001.0, 64), 6);
    });
}
