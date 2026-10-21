#![cfg(target_os = "macos")]
//! `Backend::fence` / `read_after` (token feedback, T1): a read waits for its fence only, later work keeps its own
//! results, and a fence dropped by a `synchronize` (a profiled step, the bandwidth probe) falls back to a full download
//! instead of deadlocking on the stream lock. `hold` / `release` / `discard`: the first `after` dispatches run, the
//! held rest runs on release and never on discard.

mod common;
use common::*;
use intarsia_core::backend::{Backend, MemKind};

#[test]
fn read_after_reads_the_fenced_copy_and_survives_a_dropped_fence() {
    let be = backend();
    assert_eq!(be.token_feedback(), (true, true));
    let a = be.alloc(16, MemKind::Device).unwrap();
    let b = be.alloc(16, MemKind::Device).unwrap();
    be.upload(&a, 0, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 0).unwrap();
    be.zero(&b, 0).unwrap();
    // two fenced stream copies: a[0..4] -> b[4..8], then a[8..12] -> b[8..12]
    be.copy(&b, 4, &a, 0, 4, 0).unwrap();
    let f1 = be.fence(0).unwrap();
    be.copy(&b, 8, &a, 8, 4, 0).unwrap();
    let f2 = be.fence(0).unwrap();
    assert!(f2 > f1);
    let mut x = [0u8; 4];
    be.read_after(f1, &b, 4, &mut x, 0).unwrap();
    assert_eq!(x, [1, 2, 3, 4]);
    be.read_after(f2, &b, 8, &mut x, 0).unwrap();
    assert_eq!(x, [9, 10, 11, 12]);
    // a fence that a synchronize dropped: falls back to download (this deadlocked before the fix)
    be.copy(&b, 12, &a, 12, 4, 0).unwrap();
    let f3 = be.fence(0).unwrap();
    be.synchronize(0).unwrap();
    be.read_after(f3, &b, 12, &mut x, 0).unwrap();
    assert_eq!(x, [13, 14, 15, 16]);
    // an unknown fence too
    be.read_after(f3 + 100, &b, 4, &mut x, 0).unwrap();
    assert_eq!(x, [1, 2, 3, 4]);
}

#[test]
fn hold_submits_the_start_and_discard_drops_the_rest() {
    let be = backend();
    let a = be.alloc(16, MemKind::Device).unwrap();
    let b = be.alloc(16, MemKind::Device).unwrap();
    be.upload(&a, 0, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 0).unwrap();
    for discard in [true, false] {
        be.zero(&b, 0).unwrap();
        be.synchronize(0).unwrap();
        be.hold(0, 1).unwrap();
        be.copy(&b, 0, &a, 0, 4, 0).unwrap(); // committed: the first dispatch
        be.copy(&b, 4, &a, 4, 4, 0).unwrap(); // held
        be.copy(&b, 8, &a, 8, 4, 0).unwrap(); // held
        if discard {
            be.discard(0).unwrap();
        } else {
            be.release(0).unwrap();
        }
        // the stream works normally afterwards (no hold left behind)
        be.copy(&b, 12, &a, 12, 4, 0).unwrap();
        let mut x = [0u8; 16];
        be.download(&b, 0, &mut x, 0).unwrap();
        let held: [u8; 8] = if discard { [0; 8] } else { [5, 6, 7, 8, 9, 10, 11, 12] };
        assert_eq!(x[..4], [1, 2, 3, 4], "discard {discard}");
        assert_eq!(x[4..12], held, "discard {discard}");
        assert_eq!(x[12..], [13, 14, 15, 16], "discard {discard}");
    }
    // a synchronize while holding commits the held work; a later discard drops nothing
    be.zero(&b, 0).unwrap();
    be.hold(0, 1).unwrap();
    be.copy(&b, 0, &a, 0, 4, 0).unwrap();
    be.copy(&b, 4, &a, 4, 4, 0).unwrap();
    be.synchronize(0).unwrap();
    be.discard(0).unwrap();
    let mut x = [0u8; 8];
    be.download(&b, 0, &mut x, 0).unwrap();
    assert_eq!(x, [1, 2, 3, 4, 5, 6, 7, 8]);
}
