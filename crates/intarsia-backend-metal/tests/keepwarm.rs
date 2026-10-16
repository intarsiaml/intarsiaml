#![cfg(target_os = "macos")]
//! `set_keepwarm(ms)`: a thread keeps the wrapped weight files wired (an `MTLResidencySet` refreshed every period, or GPU
//! touches from its own queue; default: off). The wrapped bytes still
//! read back exactly while touches run, the touches stay out of the way of a busy engine, turning it on after the wrap
//! covers files already wrapped, and dropping the backend right after wrapping joins the thread before the mapping goes.
mod common;
use common::*;
use intarsia_backend_metal::{KeepMode, MetalBackend};
use intarsia_core::backend::{Backend, HostMemory, MemKind};
use memmap2::Mmap;
use std::io::Write;
use std::sync::Arc;
use std::time::{Duration, Instant};

struct MmapHost(Mmap);
impl HostMemory for MmapHost {
    fn as_slice(&self) -> &[u8] {
        &self.0
    }
}

fn mapped_file(bytes: usize, seed: u32) -> (Arc<dyn HostMemory>, Vec<u8>) {
    let data: Vec<u8> = (0..bytes as u32).map(|i| (i.wrapping_mul(2654435761).wrapping_add(seed) >> 11) as u8).collect();
    let path = std::env::temp_dir().join(format!("intarsia-keepwarm-{}-{seed}.bin", std::process::id()));
    std::fs::File::create(&path).unwrap().write_all(&data).unwrap();
    let mmap = unsafe { Mmap::map(&std::fs::File::open(&path).unwrap()).unwrap() };
    let _ = std::fs::remove_file(&path); // the mapping keeps the file alive
    (Arc::new(MmapHost(mmap)), data)
}

fn read_back(be: &MetalBackend, w: &intarsia_core::backend::Buffer, off: usize, len: usize) -> Vec<u8> {
    let out = be.alloc(len, MemKind::Device).unwrap();
    be.copy(&out, 0, w, off, len, 0).unwrap();
    let mut v = vec![0u8; len];
    be.download(&out, 0, &mut v, 0).unwrap();
    v
}

fn wait_for(what: &str, mut f: impl FnMut() -> bool) {
    let t = Instant::now();
    while !f() {
        assert!(t.elapsed() < Duration::from_secs(10), "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn keepwarm_is_off_by_default() {
    let be = backend();
    let (mem, data) = mapped_file(8 << 20, 1);
    let w = be.wrap_host(mem, 0, data.len()).expect("wrap");
    std::thread::sleep(Duration::from_millis(150));
    assert_eq!(be.keepwarm_touches(), 0);
    assert_eq!(read_back(&be, &w, 99, 4096), data[99..99 + 4096]);
}

#[test]
fn files_are_touched_when_wrapped_and_while_the_engine_is_quiet_and_still_read_back_exactly() {
    for mode in [KeepMode::Residency, KeepMode::Blit] {
        files_are_touched(mode);
    }
}

fn files_are_touched(mode: KeepMode) {
    let be = backend();
    be.set_keepwarm_mode(40, mode);
    let (mem, data) = mapped_file(48 << 20, 2);
    let w = be.wrap_host(mem, 0, data.len()).expect("wrap");
    // the new file is touched at once, then again while nothing is committed
    wait_for("the first touch", || be.keepwarm_touches() >= 1);
    let n = be.keepwarm_touches();
    wait_for("a touch of the idle engine", || be.keepwarm_touches() > n);
    for off in [0usize, 4096 + 3, data.len() - 4096] {
        assert_eq!(read_back(&be, &w, off, 4096), data[off..off + 4096], "offset {off}");
    }
    // a second file joins the registry
    let (mem2, data2) = mapped_file(8 << 20, 3);
    let w2 = be.wrap_host(mem2, 0, data2.len()).expect("wrap 2");
    std::thread::sleep(Duration::from_millis(120));
    assert_eq!(read_back(&be, &w2, 12345, 5000), data2[12345..12345 + 5000]);
    assert_eq!(read_back(&be, &w, 777, 1 << 20), data[777..777 + (1 << 20)]);
}

#[test]
fn turning_it_on_after_wrapping_covers_the_files_already_wrapped_and_off_stops_it() {
    let be = backend();
    let (mem, data) = mapped_file(16 << 20, 4);
    let w = be.wrap_host(mem, 0, data.len()).expect("wrap");
    be.set_keepwarm(30);
    wait_for("a touch of the file wrapped before", || be.keepwarm_touches() >= 1);
    be.set_keepwarm(0);
    assert_eq!(be.keepwarm_touches(), 0, "off drops the thread and its counter");
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(be.keepwarm_touches(), 0);
    assert_eq!(read_back(&be, &w, 4097, 4096), data[4097..4097 + 4096]);
}

#[test]
fn a_busy_engine_is_not_touched() {
    for mode in [KeepMode::Residency, KeepMode::Blit] {
        busy_engine(mode);
    }
}

fn busy_engine(mode: KeepMode) {
    let be = backend();
    be.set_keepwarm_mode(200, mode);
    let (mem, data) = mapped_file(8 << 20, 5);
    let w = be.wrap_host(mem, 0, data.len()).expect("wrap");
    wait_for("the first touch", || be.keepwarm_touches() >= 1);
    let n = be.keepwarm_touches();
    // commits every few ms for ~700 ms: the thread wakes every 200 ms and finds the engine quiet for < 100 ms
    let t = Instant::now();
    while t.elapsed() < Duration::from_millis(700) {
        let _ = read_back(&be, &w, 0, 64);
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(be.keepwarm_touches(), n, "no touch while the engine commits");
}

#[test]
fn dropping_the_backend_right_after_wrapping_joins_the_thread_before_the_mapping_goes() {
    for round in 0..8 {
        let be = backend();
        be.set_keepwarm_mode(5, if round % 2 == 0 { KeepMode::Blit } else { KeepMode::Residency });
        let (mem, data) = mapped_file(64 << 20, 10 + round);
        let w = be.wrap_host(mem, 0, data.len()).expect("wrap");
        drop(w);
        drop(be);
    }
}

#[test]
fn memkind_still_allocates_with_keepwarm_on() {
    let be = backend();
    be.set_keepwarm(10);
    let b = be.alloc(1 << 20, MemKind::Device).unwrap();
    be.zero(&b, 0).unwrap();
    be.synchronize(0).unwrap();
}

#[test]
fn residency_mode_touches_registered_files_and_drops_cleanly() {
    for round in 0..4 {
        let be = backend();
        be.set_keepwarm_mode(40, KeepMode::Residency);
        let (mem, data) = mapped_file(32 << 20, 20 + round);
        let w = be.wrap_host(mem, 0, data.len()).expect("wrap");
        wait_for("the first residency request", || be.keepwarm_touches() >= 1);
        let n = be.keepwarm_touches();
        wait_for("a refresh of the idle engine", || be.keepwarm_touches() > n);
        for off in [0usize, 4096 + 3, data.len() - 4096] {
            assert_eq!(read_back(&be, &w, off, 4096), data[off..off + 4096], "offset {off}");
        }
        drop(w);
        drop(be);
    }
}
