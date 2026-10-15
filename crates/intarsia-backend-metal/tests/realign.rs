#![cfg(target_os = "macos")]
//! `wrap_host` weight re-alignment (`MetalBackend::set_realign`): a tensor at offset 8 mod 16 of a mapped file is
//! copied into a 16-byte aligned buffer with equal contents, aligned tensors keep their zero-copy window, the batch-1
//! GEMV on the copy agrees with the 8-byte-load kernel on the unaligned window to within one bf16 ulp (the two kernels
//! group the f32 sums differently, `dot8` against two `dot4`s, so they are not bit-identical in general), and the
//! source mapping stays readable after its pages are released.

mod common;
use common::*;
use intarsia_backend_metal::MetalBuffer;
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

const K: usize = 4096;
const N: usize = 512;

/// A mapped file laid out like a shard with an 8 mod 16 header: weight A at 8, weight B at the next 16-byte boundary.
fn shard(tag: &str) -> (Arc<MmapHost>, Vec<u16>, Vec<u16>, usize, usize) {
    let dir = std::env::temp_dir().join(format!("ik-metal-realign-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("shard.bin");
    let a = rand_bf16(N * K, 11);
    let b = rand_bf16(N * K, 12);
    let (a_off, b_off) = (8usize, 8 + N * K * 2 + 8);
    let mut file = vec![0u8; b_off + N * K * 2 + 1000];
    file[a_off..a_off + N * K * 2].copy_from_slice(as_bytes(&a));
    file[b_off..b_off + N * K * 2].copy_from_slice(as_bytes(&b));
    std::fs::File::create(&path).unwrap().write_all(&file).unwrap();
    let f = std::fs::File::open(&path).unwrap();
    // SAFETY: private temp file nothing else touches.
    let mem = Arc::new(MmapHost(unsafe { Mmap::map(&f) }.unwrap()));
    std::fs::remove_dir_all(&dir).ok();
    (mem, a, b, a_off, b_off)
}

fn gemv(be: &impl Backend, w: &intarsia_core::backend::Buffer, x: &intarsia_core::backend::Buffer) -> Vec<u8> {
    let y = be.alloc(N * 2, MemKind::Device).unwrap();
    let call = matmul_call(tref(x, 0, &[1, K]), tref(w, 0, &[N, K]), tref(&y, 0, &[1, N]));
    be.launch(intarsia_backend_metal::kernels::MATMUL_BF16_GEMV, &call, 0).unwrap();
    be.synchronize(0).unwrap();
    let mut raw = vec![0u8; N * 2];
    be.download(&y, 0, &mut raw, 0).unwrap();
    raw
}

#[test]
fn realigned_buffer_is_16_byte_aligned_with_equal_contents() {
    let (mem, a, b, a_off, b_off) = shard("buf");
    let be = backend();
    let len = N * K * 2;
    assert_eq!((mem.as_slice().as_ptr() as usize + a_off) % 16, 8);

    // off (default in this process unless INTARSIA_METAL_REALIGN is set): zero-copy window at 8 mod 16
    be.set_realign(false);
    let w = be.try_wrap_host(mem.clone(), a_off, len).unwrap();
    let mb = w.downcast::<MetalBuffer>().unwrap();
    assert_eq!(mb.base() % 16, 8);
    assert_eq!(be.realigned(), (0, 0));

    be.set_realign(true);
    let w = be.try_wrap_host(mem.clone(), a_off, len).unwrap();
    let mb = w.downcast::<MetalBuffer>().unwrap();
    assert_eq!(mb.base(), 0);
    assert_eq!(mb.host_ptr() as usize % 16, 0, "realigned buffer is not 16-byte aligned");
    assert!(mb.is_read_only());
    assert_eq!(w.bytes(), len);
    // SAFETY: the buffer holds `len` initialised bytes.
    let copy = unsafe { std::slice::from_raw_parts(mb.host_ptr() as *const u8, len) };
    assert_eq!(copy, as_bytes(&a), "copy differs from the source weight");
    assert_eq!(copy, &mem.as_slice()[a_off..a_off + len]);
    assert_eq!(be.realigned(), (1, len));
    assert!(be.upload(&w, 0, &[0u8], 0).is_err(), "realigned buffer must stay read-only");

    // already 16-byte aligned: no copy, still the window onto the shared mapping
    let wb = be.try_wrap_host(mem.clone(), b_off, len).unwrap();
    let mbb = wb.downcast::<MetalBuffer>().unwrap();
    assert_eq!((mbb.base()) % 16, 0);
    assert_ne!(mbb.base(), 0, "aligned tensor should window the mapping, not be copied");
    assert_eq!(be.realigned(), (1, len));
    assert_eq!(&mem.as_slice()[b_off..b_off + len], as_bytes(&b));

    // the pages released after the copy refault with the right bytes
    assert_eq!(&mem.as_slice()[a_off..a_off + len], as_bytes(&a));
}

#[test]
fn batch1_gemv_on_realigned_weights_matches_the_8_byte_kernel() {
    let (mem, a, _b, a_off, _b_off) = shard("gemv");
    let be = backend();
    let len = N * K * 2;
    let x = upload(&be, &rand_bf16(K, 77));
    be.set_realign(false);
    let w8 = be.try_wrap_host(mem.clone(), a_off, len).unwrap();
    let y8 = gemv(&be, &w8, &x);
    be.set_realign(true);
    let w16 = be.try_wrap_host(mem.clone(), a_off, len).unwrap();
    let y16 = gemv(&be, &w16, &x);
    let bf = |raw: &[u8]| -> Vec<u16> { raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect() };
    let (g8, g16) = (bf(&y8), bf(&y16));
    let differ = g8.iter().zip(&g16).filter(|(p, q)| p != q).count();
    eprintln!("aligned vs 8-byte kernel: {differ} of {N} outputs differ");
    for (i, (p, q)) in g8.iter().zip(&g16).enumerate() {
        assert!(p.abs_diff(*q) <= 1, "row {i}: {p:#06x} vs {q:#06x} differ by more than one bf16 ulp");
    }
    // both agree with the host reference
    let refs = reference(&a, &rand_bf16(K, 77), K, N);
    check(&g16, &refs, 0..N, "realigned gemv");
}
