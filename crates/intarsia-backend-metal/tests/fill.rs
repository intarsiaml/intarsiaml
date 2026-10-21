#![cfg(target_os = "macos")]
//! `Backend::copy` / `Backend::zero` as stream dispatches: byte-exact for every alignment and tail, ordered with
//! surrounding launches, not waited on, memmove semantics for overlap, read-only wrapped memory refused as a target and
//! usable as a source.

mod common;
use common::*;
use intarsia_backend_metal::bench::{bf16_to_f32, f32_to_bf16};
use intarsia_backend_metal::kernels::ADD_BF16;
use intarsia_core::backend::{Backend, Buffer, HostMemory, MemKind, TensorRef};
use intarsia_core::{DType, Op, OpAttrs, OpCall};
use memmap2::Mmap;
use std::sync::Arc;

fn pattern(n: usize, seed: u32) -> Vec<u8> {
    (0..n as u32).map(|i| (i.wrapping_mul(2654435761).wrapping_add(seed) >> 13) as u8).collect()
}

fn buf_with(be: &impl Backend, data: &[u8]) -> Buffer {
    let b = be.alloc(data.len(), MemKind::Device).unwrap();
    be.upload(&b, 0, data, 0).unwrap();
    b
}

fn read(be: &impl Backend, b: &Buffer) -> Vec<u8> {
    let mut v = vec![0u8; b.bytes()];
    be.download(b, 0, &mut v, 0).unwrap();
    v
}

#[test]
fn copy_is_byte_exact_for_every_alignment_and_tail() {
    let be = backend();
    let size = 4096 + 64;
    let src_data = pattern(size, 1);
    let dst_data = pattern(size, 2);
    let src = buf_with(&be, &src_data);
    // (src offset, dst offset, bytes): aligned, aligned with a tail, one side unaligned, both unaligned, tiny, empty
    let cases = [
        (0, 0, 4096),
        (16, 32, 1024),
        (0, 0, 4093),
        (16, 48, 17),
        (3, 0, 1000),
        (0, 5, 1000),
        (7, 9, 2049),
        (1, 1, 1),
        (64, 64, 15),
        (32, 16, 4000),
        (100, 100, 0),
    ];
    for (so, d_o, n) in cases {
        let dst = buf_with(&be, &dst_data);
        be.copy(&dst, d_o, &src, so, n, 0).unwrap();
        let mut want = dst_data.clone();
        want[d_o..d_o + n].copy_from_slice(&src_data[so..so + n]);
        assert_eq!(read(&be, &dst), want, "src {so} dst {d_o} bytes {n}");
    }
    let dst = buf_with(&be, &dst_data);
    assert!(be.copy(&dst, size - 4, &src, 0, 8, 0).is_err(), "dst out of bounds");
    assert!(be.copy(&dst, 0, &src, size - 4, 8, 0).is_err(), "src out of bounds");
}

#[test]
fn copy_large_aligned_buffer() {
    let be = backend();
    let n = (96 << 20) + 48; // vector body plus a 16-byte-multiple that is not a whole thread block
    let data = pattern(n, 9);
    let (src, dst) = (buf_with(&be, &data), be.alloc(n, MemKind::Device).unwrap());
    be.copy(&dst, 0, &src, 0, n, 0).unwrap();
    assert!(read(&be, &dst) == data);
}

#[test]
fn overlapping_copy_in_one_buffer_is_a_memmove() {
    let be = backend();
    let data = pattern(512, 3);
    for (d_o, s_o, n) in [(16usize, 0usize, 400usize), (0, 16, 400), (3, 0, 300), (0, 3, 300), (100, 100, 50)] {
        let b = buf_with(&be, &data);
        be.copy(&b, d_o, &b, s_o, n, 0).unwrap();
        let mut want = data.clone();
        want.copy_within(s_o..s_o + n, d_o);
        assert_eq!(read(&be, &b), want, "dst {d_o} src {s_o} bytes {n}");
    }
    // disjoint ranges of one buffer take the GPU path and still work
    let b = buf_with(&be, &data);
    be.copy(&b, 256, &b, 0, 256, 0).unwrap();
    let mut want = data.clone();
    want.copy_within(0..256, 256);
    assert_eq!(read(&be, &b), want);
}

#[test]
fn zero_covers_exactly_the_buffer() {
    let be = backend();
    for n in [1usize, 2, 15, 16, 17, 31, 32, 1000, 4096, 100_003, (8 << 20) + 5] {
        let b = buf_with(&be, &vec![0xEE; n]);
        be.zero(&b, 0).unwrap();
        assert!(read(&be, &b).iter().all(|v| *v == 0), "n {n}");
    }
}

fn add_call(a: &Buffer, b: &Buffer, out: &Buffer, n: usize) -> OpCall {
    let t = |buf: &Buffer| TensorRef { buf: buf.clone(), offset: 0, dtype: DType::Bf16, shape: vec![1, n] };
    OpCall::new(Op::Add, vec![t(a), t(b)], vec![t(out)], OpAttrs::default())
}

/// zero and copy sit between dependent launches in the stream: a launch writes, a copy moves its result, a second launch
/// reads the copy, a zero clears the first result, and nothing is waited on until the host reads.
#[test]
fn copy_and_zero_are_ordered_with_launches_and_do_not_wait() {
    let be = backend();
    let n = 1 << 14;
    let a = rand_bf16(n, 1);
    let b = rand_bf16(n, 2);
    let (ab, bb) = (upload(&be, &a), upload(&be, &b));
    let (s1, c1, s2, junk) = (
        be.alloc(n * 2, MemKind::Device).unwrap(),
        be.alloc(n * 2, MemKind::Device).unwrap(),
        be.alloc(n * 2, MemKind::Device).unwrap(),
        upload(&be, &rand_bf16(n, 3)),
    );
    be.synchronize(0).unwrap();
    assert_eq!(be.pending_dispatches(), 0);
    be.zero(&s2, 0).unwrap();
    be.launch(ADD_BF16, &add_call(&ab, &bb, &s1, n), 0).unwrap(); // s1 = a + b
    be.copy(&c1, 0, &s1, 0, n * 2, 0).unwrap(); // c1 = s1
    be.launch(ADD_BF16, &add_call(&c1, &bb, &s2, n), 0).unwrap(); // s2 = c1 + b
    be.zero(&s1, 0).unwrap(); // s1 = 0 after c1 took its value
    be.copy(&junk, 0, &s1, 0, n * 2, 0).unwrap(); // junk = zeros
    // the first FIRST_COMMIT of them went out as their own (committed, not waited on) buffer
    let fc = intarsia_backend_metal::FIRST_COMMIT;
    let open = if fc > 0 && fc <= 6 { 6 - fc } else { 6 };
    assert_eq!(be.pending_dispatches(), open, "zero, add, copy, add, zero, copy: all appended, none waited on");
    let (mut r1, mut r2, mut rj) = (vec![0u8; n * 2], vec![0u8; n * 2], vec![0u8; n * 2]);
    be.download(&c1, 0, &mut r1, 0).unwrap();
    assert_eq!(be.pending_dispatches(), 0, "download flushed");
    be.download(&s2, 0, &mut r2, 0).unwrap();
    be.download(&junk, 0, &mut rj, 0).unwrap();
    let u16s = |v: &[u8]| v.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect::<Vec<_>>();
    let f = |x: u16| bf16_to_f32(x);
    let c1_want: Vec<u16> = a.iter().zip(&b).map(|(x, y)| f32_to_bf16(f(*x) + f(*y))).collect();
    let s2_want: Vec<u16> = c1_want.iter().zip(&b).map(|(x, y)| f32_to_bf16(f(*x) + f(*y))).collect();
    assert_eq!(u16s(&r1), c1_want);
    assert_eq!(u16s(&r2), s2_want);
    assert!(rj.iter().all(|v| *v == 0));
}

struct MmapHost(Mmap);
impl HostMemory for MmapHost {
    fn as_slice(&self) -> &[u8] {
        &self.0
    }
}

#[test]
fn wrapped_host_memory_is_a_source_but_never_a_target() {
    use std::io::Write;
    let be = backend();
    let dir = std::env::temp_dir().join(format!("ik-metal-fill-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("blob.bin");
    let len = 2 * 16384 + 777;
    let data = pattern(len, 5);
    std::fs::File::create(&path).unwrap().write_all(&data).unwrap();
    let f = std::fs::File::open(&path).unwrap();
    // SAFETY: private temp file nothing else touches.
    let mem = Arc::new(MmapHost(unsafe { Mmap::map(&f) }.unwrap()));
    for (off, l) in [(0usize, 4096usize), (8, 1000), (16384 + 3, 5000), (len - 17, 17)] {
        let w = be.try_wrap_host(mem.clone(), off, l).unwrap();
        let dst = be.alloc(l + 8, MemKind::Device).unwrap();
        be.zero(&dst, 0).unwrap();
        be.copy(&dst, 4, &w, 0, l, 0).unwrap();
        let got = read(&be, &dst);
        assert!(got[..4].iter().all(|v| *v == 0) && got[l + 4..].iter().all(|v| *v == 0), "off {off}");
        assert_eq!(got[4..l + 4], data[off..off + l], "off {off} len {l}");
        let scratch = be.alloc(l, MemKind::Device).unwrap();
        assert!(be.copy(&w, 0, &scratch, 0, l, 0).is_err(), "copy into wrapped memory");
        assert!(be.zero(&w, 0).is_err(), "zero of wrapped memory");
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// After a `synchronize` the first command buffer is committed after `FIRST_COMMIT` dispatches (it trims the GPU idle
/// gap between decode steps), later ones after `MAX_DISPATCHES`; the work and its order are unchanged.
#[test]
fn first_buffer_after_synchronize_is_committed_early() {
    if std::env::var_os("INTARSIA_METAL_FIRST_COMMIT").is_some() || std::env::var_os("INTARSIA_METAL_COMMIT_EVERY").is_some() {
        eprintln!("SKIP: commit cadence overridden by the environment");
        return;
    }
    let n = intarsia_backend_metal::FIRST_COMMIT;
    assert!(n > 0 && n < intarsia_backend_metal::MAX_DISPATCHES);
    let be = backend();
    let b = buf_with(&be, &pattern(64, 7));
    be.synchronize(0).unwrap();
    for i in 1..n {
        be.zero(&b, 0).unwrap();
        assert_eq!(be.pending_dispatches(), i);
    }
    be.zero(&b, 0).unwrap();
    assert_eq!(be.pending_dispatches(), 0, "first buffer after synchronize committed after FIRST_COMMIT dispatches");
    for i in 1..=n + 1 {
        be.zero(&b, 0).unwrap();
        assert_eq!(be.pending_dispatches(), i, "later buffers hold up to MAX_DISPATCHES");
    }
    assert_eq!(read(&be, &b), vec![0u8; 64]);
}
