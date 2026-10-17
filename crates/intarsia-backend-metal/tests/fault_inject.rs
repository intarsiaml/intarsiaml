#![cfg(all(target_os = "macos", feature = "fault-inject"))]
//! The `fault-inject` hook in `Backend::synchronize`: an armed fault turns the next synchronize into the backend error a
//! failed command buffer would produce (mode Err) or a panic (mode Panic); once consumed the backend works as before.
//! Run with `cargo test -p intarsia-backend-metal --release --features fault-inject`.

mod common;
use common::*;
use intarsia_core::backend::{Backend, MemKind};
use intarsia_core::fault::{self, Mode};
use intarsia_core::Error;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Mutex, MutexGuard};

/// The armed fault is process-wide and any `synchronize` (an `upload` has one) consumes it: the tests run behind one lock.
static LOCK: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    let g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _ = fault::take();
    g
}

#[test]
fn armed_fault_fails_the_next_synchronize_once() {
    let _g = serial();
    let be = backend();
    let b = be.alloc(256, MemKind::Device).unwrap();
    be.upload(&b, 0, &[7u8; 256], 0).unwrap();
    be.synchronize(0).unwrap();

    fault::arm(Mode::Err);
    match be.synchronize(0) {
        Err(Error::Backend(m)) => assert!(m.contains("command buffer failed") && m.contains("injected"), "{m}"),
        other => panic!("expected a backend error, got {other:?}"),
    }
    // consumed: the backend is usable again (a real device error would be sticky only in the scheduler, not here)
    be.synchronize(0).unwrap();
    let mut out = [0u8; 256];
    be.download(&b, 0, &mut out, 0).unwrap();
    assert_eq!(out, [7u8; 256]);

    // a download syncs first, so an armed fault surfaces through it as well
    fault::arm(Mode::Err);
    assert!(matches!(be.download(&b, 0, &mut out, 0), Err(Error::Backend(_))));
    assert_eq!(fault::take(), None);
}

#[test]
fn armed_panic_unwinds_out_of_synchronize_and_leaves_the_backend_usable() {
    let _g = serial();
    let be = backend();
    let b = be.alloc(64, MemKind::Device).unwrap();
    be.upload(&b, 0, &[1u8; 64], 0).unwrap();
    fault::arm(Mode::Panic);
    let r = catch_unwind(AssertUnwindSafe(|| be.synchronize(0)));
    let msg = *r.expect_err("must panic").downcast::<&str>().unwrap();
    assert!(msg.contains("injected fault"), "{msg}");
    // the panic is raised after the stream lock is dropped: no poisoned mutex
    be.synchronize(0).unwrap();
}
