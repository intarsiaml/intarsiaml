#![cfg(target_os = "macos")]
//! With `set_prewire(true)`, `wrap_host` pre-wires each new whole-file buffer in a background command buffer (default: off): the wrapped bytes read back exactly
//! (before and after the wiring has finished), and dropping the backend right after wrapping waits for the blit that reads
//! the mapping instead of unmapping it underneath.
mod common;
use common::*;
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, HostMemory, MemKind};
use memmap2::Mmap;
use std::io::Write;
use std::sync::Arc;

struct MmapHost(Mmap);
impl HostMemory for MmapHost {
    fn as_slice(&self) -> &[u8] {
        &self.0
    }
}

fn mapped_file(bytes: usize, seed: u32) -> (Arc<dyn HostMemory>, Vec<u8>) {
    let data: Vec<u8> = (0..bytes as u32).map(|i| (i.wrapping_mul(2654435761).wrapping_add(seed) >> 11) as u8).collect();
    let path = std::env::temp_dir().join(format!("intarsia-prewire-{}-{seed}.bin", std::process::id()));
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

#[test]
fn prewired_file_reads_back_exactly_before_and_after_the_wiring_finishes() {
    let be = backend();
    be.set_prewire(true);
    let (mem, data) = mapped_file(48 << 20, 1);
    let w = be.wrap_host(mem, 0, data.len()).expect("wrap");
    // right behind the wrap: the first use waits for the wiring of the pre-wiring command buffer, if it is still running
    for off in [0usize, 4096 + 3, data.len() - 4096] {
        assert_eq!(read_back(&be, &w, off, 4096), data[off..off + 4096], "offset {off}");
    }
    // a pre-wired file wrapped again (a second window of the same mapping) and a second file
    let (mem2, data2) = mapped_file(8 << 20, 2);
    let w2 = be.wrap_host(mem2, 0, data2.len()).expect("wrap 2");
    std::thread::sleep(std::time::Duration::from_millis(300));
    assert_eq!(read_back(&be, &w2, 12345, 5000), data2[12345..12345 + 5000]);
    assert_eq!(read_back(&be, &w, 777, 1 << 20), data[777..777 + (1 << 20)]);
}

#[test]
fn dropping_the_backend_right_after_wrapping_waits_for_the_prewiring() {
    // the blit reads the mapping: a drop that unmapped it first would fault. Many rounds, each drops at once.
    for round in 0..8 {
        let be = backend();
        be.set_prewire(true);
        let (mem, data) = mapped_file(64 << 20, 10 + round);
        let w = be.wrap_host(mem, 0, data.len()).expect("wrap");
        drop(w);
        drop(be);
    }
}

#[test]
fn prewire_is_off_by_default_and_the_bytes_still_read_back() {
    let be = backend();
    let (mem, data) = mapped_file(32 << 20, 3);
    let w = be.wrap_host(mem, 0, data.len()).expect("wrap");
    assert_eq!(be.prewired_bytes(), 0, "a default backend must not pre-wire");
    assert_eq!(read_back(&be, &w, 4097, 4096), data[4097..4097 + 4096]);
    be.set_prewire(true);
    let (mem2, data2) = mapped_file(32 << 20, 4);
    let w2 = be.wrap_host(mem2, 0, data2.len()).expect("wrap 2");
    assert!(be.prewired_bytes() >= data2.len() as u64, "switched on: the next file is pre-wired");
    assert_eq!(read_back(&be, &w2, 99, 4096), data2[99..99 + 4096]);
}
