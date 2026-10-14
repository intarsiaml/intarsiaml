#![cfg(target_os = "macos")]
//! `wrap_host` on a real read-only mmap'd safetensors file (the same `memmap2::Mmap` the model loader uses): unaligned
//! tensor offsets, the last tensor of each shard, GPU reads compared with the host bytes, and a GEMV straight off the
//! mapped weights.

mod common;
use common::*;
use intarsia_backend_metal::{MetalBackend, MetalBuffer};
use intarsia_core::backend::{Backend, Buffer, HostMemory, MemKind};
use memmap2::Mmap;
use objc2_metal::MTLBuffer;
use std::path::PathBuf;
use std::sync::Arc;

struct MmapHost(Mmap);
impl HostMemory for MmapHost {
    fn as_slice(&self) -> &[u8] {
        &self.0
    }
}

struct Tensor {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    /// Offsets into the file (header included).
    begin: usize,
    end: usize,
}

fn model_dir() -> PathBuf {
    std::env::var_os("INTARSIA_METAL_TEST_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/Users/josef/models/Qwen3.6-35B-A3B-NVFP4"))
}

fn shards() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = match std::fs::read_dir(model_dir()) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect(),
        Err(_) => vec![],
    };
    v.sort();
    v
}

fn open(p: &PathBuf) -> (Arc<MmapHost>, Vec<Tensor>) {
    let f = std::fs::File::open(p).unwrap();
    // SAFETY: read-only mapping of a file nothing modifies (same as intarsia-models' loader).
    let mm = unsafe { Mmap::map(&f) }.unwrap();
    let hl = u64::from_le_bytes(mm[..8].try_into().unwrap()) as usize;
    let hdr: serde_json::Value = serde_json::from_slice(&mm[8..8 + hl]).unwrap();
    let data0 = 8 + hl;
    let mut ts: Vec<Tensor> = hdr
        .as_object()
        .unwrap()
        .iter()
        .filter(|(k, _)| *k != "__metadata__")
        .map(|(k, v)| {
            let o = v["data_offsets"].as_array().unwrap();
            Tensor {
                name: k.clone(),
                dtype: v["dtype"].as_str().unwrap().to_string(),
                shape: v["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as usize).collect(),
                begin: data0 + o[0].as_u64().unwrap() as usize,
                end: data0 + o[1].as_u64().unwrap() as usize,
            }
        })
        .collect();
    ts.sort_by_key(|t| t.begin);
    (Arc::new(MmapHost(mm)), ts)
}

/// GPU-read `[off, off+len)` of `buf` with the copy kernel into a fresh buffer and return the bytes. The kernel needs
/// 16-byte aligned effective offsets, so the range is widened to 16 bytes; the widening stays inside the MTLBuffer
/// (its length is page rounded) and the requested bytes are sliced back out.
fn gpu_read(be: &MetalBackend, buf: &Buffer, off: usize, len: usize) -> Vec<u8> {
    let mb = buf.downcast::<MetalBuffer>().unwrap();
    let eff = mb.base() + off;
    let lo = eff & !15;
    let hi = (eff + len + 15) & !15;
    assert!(hi <= mb.mtl().length(), "widened range leaves the MTLBuffer");
    let bytes = hi - lo;
    let dst = be.new_mtl_buffer(bytes, false).unwrap();
    let pso = be.pso(intarsia_backend_metal::kernels::COPY_U4).unwrap();
    let n16 = (bytes / 16) as u32;
    let cb = be
        .encode_and_commit(|enc| {
            use objc2_metal::{MTLComputeCommandEncoder, MTLSize};
            enc.setComputePipelineState(pso);
            // SAFETY: lo/bytes checked against the buffer length above; dst holds `bytes`.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(mb.mtl()), lo, 0);
                enc.setBuffer_offset_atIndex(Some(&dst), 0, 1);
            }
            intarsia_backend_metal::set_bytes(enc, &n16, 2);
            enc.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: intarsia_backend_metal::bench::copy_threads(
                        n16 as usize,
                        intarsia_backend_metal::kernels::COPY_PER_THREAD,
                    ),
                    height: 1,
                    depth: 1,
                },
                MTLSize { width: 256, height: 1, depth: 1 },
            );
        })
        .unwrap();
    MetalBackend::wait(&cb).unwrap();
    // SAFETY: dst holds `bytes` initialised bytes.
    let all =
        unsafe { std::slice::from_raw_parts(objc2_metal::MTLBuffer::contents(&*dst).as_ptr() as *const u8, bytes) };
    all[eff - lo..eff - lo + len].to_vec()
}

#[test]
fn wrap_real_safetensors_unaligned_and_last_tensor() {
    let files = shards();
    if files.is_empty() {
        eprintln!("SKIP: no safetensors under {}", model_dir().display());
        return;
    }
    let be = backend();
    let page = be.page_size();
    assert_eq!(page, 16384, "Apple Silicon page size");
    let mut checked = 0usize;
    for f in &files {
        let (mem, ts) = open(f);
        let file_len = mem.as_slice().len();
        let unaligned16 = ts.iter().filter(|t| t.begin % 16 != 0).count();
        let unaligned_page = ts.iter().filter(|t| t.begin % page != 0).count();
        eprintln!(
            "{}: {} bytes ({} mod page), {} tensors, {} start off a 16 B boundary, {} off a page boundary",
            f.file_name().unwrap().to_string_lossy(),
            file_len,
            file_len % page,
            ts.len(),
            unaligned16,
            unaligned_page
        );
        assert_eq!(ts.last().unwrap().end, file_len, "last tensor ends at EOF");

        // first, last two, a spread of page-unaligned ones, and some that are not even 16-byte aligned
        let mut pick: Vec<usize> = vec![0, ts.len() - 1, ts.len() - 2];
        pick.extend((0..ts.len()).filter(|i| ts[*i].begin % page != 0).step_by((ts.len() / 40).max(1)));
        pick.extend((0..ts.len()).filter(|i| ts[*i].begin % 16 != 0).take(8));
        pick.sort();
        pick.dedup();
        for i in pick {
            let t = &ts[i];
            let len = t.end - t.begin;
            let b =
                be.try_wrap_host(mem.clone(), t.begin, len).unwrap_or_else(|e| panic!("wrap {} failed: {e}", t.name));
            assert_eq!(b.bytes(), len);
            // at most the last 64 MiB of a big tensor so the test stays quick; includes the final byte
            let take = len.min(64 << 20);
            let off = len - take;
            let got = gpu_read(&be, &b, off, take);
            assert_eq!(&got[..], &mem.as_slice()[t.begin + off..t.end], "GPU bytes differ for {}", t.name);
            let head = len.min(1 << 20);
            let got = gpu_read(&be, &b, 0, head);
            assert_eq!(&got[..], &mem.as_slice()[t.begin..t.begin + head], "GPU head bytes differ for {}", t.name);
            checked += 1;
        }
        // all tensors of the file share one MTLBuffer
        let first = &ts[0];
        let last = &ts[ts.len() - 1];
        let a = be.try_wrap_host(mem.clone(), first.begin, first.end - first.begin).unwrap();
        let z = be.try_wrap_host(mem.clone(), last.begin, last.end - last.begin).unwrap();
        let (am, zm) = (a.downcast::<MetalBuffer>().unwrap(), z.downcast::<MetalBuffer>().unwrap());
        assert!(
            std::ptr::eq(am.mtl() as *const _ as *const u8, zm.mtl() as *const _ as *const u8),
            "tensors of one file must share an MTLBuffer"
        );
        // rounding edge cases
        assert!(be.try_wrap_host(mem.clone(), 0, 0).is_err(), "empty range");
        assert!(be.try_wrap_host(mem.clone(), file_len - 1, 2).is_err(), "range past EOF");
        let one = be.try_wrap_host(mem.clone(), file_len - 1, 1).unwrap();
        assert_eq!(gpu_read(&be, &one, 0, 1)[0], mem.as_slice()[file_len - 1], "single last byte");
        // host writes into a wrapped buffer are refused
        assert!(be.upload(&one, 0, &[0u8], 0).is_err());
    }
    eprintln!("verified {checked} tensors across {} shards", files.len());
}

#[test]
fn gemv_on_wrapped_weights() {
    let files = shards();
    let Some(f) = files.last() else {
        eprintln!("SKIP: no safetensors");
        return;
    };
    let (mem, ts) = open(f);
    let be = backend();
    // the biggest 2-D bf16 matrix of the last shard
    let Some(t) = ts.iter().filter(|t| t.dtype == "BF16" && t.shape.len() == 2).max_by_key(|t| t.end - t.begin) else {
        eprintln!("SKIP: no 2-D bf16 tensor in {}", f.display());
        return;
    };
    let (n, k) = (t.shape[0], t.shape[1]);
    eprintln!(
        "GEMV on {} [{n},{k}] at file offset {} (mod 16 = {}, mod page = {})",
        t.name,
        t.begin,
        t.begin % 16,
        t.begin % be.page_size()
    );
    let wbuf = be.try_wrap_host(mem.clone(), t.begin, t.end - t.begin).unwrap();
    let x = rand_bf16(k, 77);
    let xb = upload(&be, &x);
    let yb = be.alloc(n * 2, MemKind::Device).unwrap();
    // the wrapped Buffer starts at the tensor; a non-zero TensorRef offset (second half of the rows) stacks on top
    for (row0, rows) in [(0usize, n), (n / 2, n - n / 2)] {
        let w = tref(&wbuf, row0 * k * 2, &[rows, k]);
        let call = matmul_call(tref(&xb, 0, &[1, k]), w, tref(&yb, 0, &[1, rows]));
        be.launch(intarsia_backend_metal::kernels::MATMUL_BF16_GEMV, &call, 0).unwrap();
        be.synchronize(0).unwrap();
        let mut raw = vec![0u8; rows * 2];
        be.download(&yb, 0, &mut raw, 0).unwrap();
        let gpu: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let host_w: Vec<u16> = mem.as_slice()[t.begin + row0 * k * 2..t.end]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        let refs = reference(&host_w, &x, k, rows);
        check(&gpu, &refs, 0..rows, &format!("wrapped {} rows {row0}..", t.name));
    }
}

#[test]
fn wrap_page_rounding_with_synthetic_file() {
    // A file-backed read-only mapping whose length is not a page multiple, wrapped at odd offsets, last byte included.
    use std::io::Write;
    let be = backend();
    let dir = std::env::temp_dir().join(format!("ik-metal-wrap-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("blob.bin");
    let len = 3 * 16384 + 4321;
    let data: Vec<u8> = (0..len).map(|i| (i as u32).wrapping_mul(2654435761) as u8).collect();
    std::fs::File::create(&path).unwrap().write_all(&data).unwrap();
    let f = std::fs::File::open(&path).unwrap();
    // SAFETY: private temp file nothing else touches.
    let mem = Arc::new(MmapHost(unsafe { Mmap::map(&f) }.unwrap()));
    for (off, l) in
        [(0, len), (1, 16), (16383, 2), (16384, 16384), (len - 1, 1), (len - 4321, 4321), (5, len - 5), (len - 17, 17)]
    {
        let b = be.try_wrap_host(mem.clone(), off, l).unwrap();
        assert_eq!(gpu_read(&be, &b, 0, l), data[off..off + l], "off {off} len {l}");
    }
    std::fs::remove_dir_all(&dir).ok();
}
