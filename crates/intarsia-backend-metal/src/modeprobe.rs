//! Diagnostic only (lane ink-mode): `INTARSIA_METAL_MODE_PROBE=1` logs the per-process state that could put a process into
//! the "slow mode" of docs/results/2026-11-25-inkling2-mode.md and times a fixed set of small GPU workloads in the same
//! process, so each process yields its state, its microbench times and its real prefill time together.
//!
//! State (one `modeprobe file` line per whole-file weight buffer, when it is created; see `host.rs`): host VA and GPU
//! address of the no-copy buffer, each mod 16 KiB / 2 MiB / 32 MiB / 1 GiB, its length, creation order and time.
//! [`MetalBackend::mode_probe`] adds the device's allocated size and recommended working set, `mincore` residency of every
//! wrapped file, and the times (GPU timestamps of each command buffer, median and min of the reps, ms) of:
//!
//!  * `alu`: a register-only FMA chain (clock and power state, no memory);
//!  * `seq_file`: a streaming read of every whole-file buffer (DRAM bandwidth through the no-copy mapping);
//!  * `gather_file`: one 16-byte read per random 16 KiB page over the largest file buffer (GPU page-walk / TLB cost of the
//!    mapping), `gather_file2g` the same over its first 2 GiB, and `seq_anon` / `gather_anon` over a fresh 2 GiB
//!    `newBufferWithLength` buffer (the same footprint, anonymous memory) as the control;
//!  * `copy_gbs`: the backend's own copy-bandwidth probe (`bench::measure_copy_gbs`, 1 GiB).
//!
//! Nothing here runs unless the variable is set; the engine's numerics are untouched either way.

use crate::{host, MetalBackend, MtlBuffer, Pso};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLDevice, MTLLibrary, MTLSize,
};
use std::ffi::c_void;
use std::ptr::NonNull;

const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void mp_alu(device float *out [[buffer(0)]], constant uint &iters [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    float a = float(gid) * 1e-7f, b = 1.0000001f, c = 0.9999999f, d = 0.5f, e = a + 1.0f, f = a + 2.0f, g = a + 3.0f, h = a + 4.0f;
    for (uint i = 0; i < iters; i++) {
        a = fma(a, b, c); e = fma(e, c, b); f = fma(f, b, d); g = fma(g, c, d); h = fma(h, b, c); d = fma(d, c, b);
    }
    out[gid] = a + e + f + g + h + d;
}
kernel void mp_seq(device const uint4 *src [[buffer(0)]], device uint *out [[buffer(1)]], constant ulong &n16 [[buffer(2)]],
                   uint gid [[thread_position_in_grid]], uint nth [[threads_per_grid]]) {
    uint4 acc = 0;
    for (ulong i = gid; i < n16; i += nth) acc ^= src[i];
    out[gid] = acc.x ^ acc.y ^ acc.z ^ acc.w;
}
kernel void mp_gather(device const uchar *src [[buffer(0)]], device uint *out [[buffer(1)]], constant uint2 &p [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    // p.x pages of 16 KiB, p.y reads per thread; each read is 16 bytes at a pseudo-random page and line
    uint x = gid * 2654435761u + 12345u;
    uint4 acc = 0;
    for (uint r = 0; r < p.y; r++) {
        x = x * 1664525u + 1013904223u;
        ulong pg = (ulong(x) * ulong(p.x)) >> 32;
        ulong off = pg * 16384ul + ulong((x >> 7) & 1023u) * 16ul;
        acc ^= *(device const uint4 *)(src + off);
    }
    out[gid] = acc.x ^ acc.y ^ acc.z ^ acc.w;
}
"#;

const PAGE: usize = 16384;
const ANON: usize = 2 << 30;

fn mods(v: u64) -> String {
    format!("{:#x} m16k={} m2m={:#x} m32m={:#x} m1g={:#x}", v, v % (16 << 10), v % (2 << 20), v % (32 << 20), v % (1 << 30))
}

/// Log one whole-file buffer as it is created (`host.rs`). `order` counts the files of this process.
pub(crate) fn log_file(order: usize, floor: usize, mtl: &MtlBuffer) {
    eprintln!(
        "modeprobe file order={order} t_ms={} len={} va={} gpu={}",
        host::now_ms(),
        mtl.length(),
        mods(floor as u64),
        mods(mtl.gpuAddress())
    );
}

pub(crate) fn enabled() -> bool {
    std::env::var_os("INTARSIA_METAL_MODE_PROBE").is_some_and(|v| !v.is_empty() && v != "0")
}

/// Share of the 16 KiB pages of `[floor, floor + len)` resident in memory (`mincore`), percent.
fn resident_pct(floor: usize, len: usize) -> f64 {
    let pages = len.div_ceil(PAGE);
    let mut v = vec![0u8; pages];
    // SAFETY: the range is a live mapping (the host cache holds it); `v` has one byte per page.
    let rc = unsafe { libc::mincore(floor as *mut c_void, len, v.as_mut_ptr() as *mut _) };
    if rc != 0 {
        return -1.0;
    }
    100.0 * v.iter().filter(|b| **b & 1 != 0).count() as f64 / pages as f64
}

fn pso(be: &MetalBackend, lib: &objc2::runtime::ProtocolObject<dyn MTLLibrary>, name: &str) -> Option<Pso> {
    let f = lib.newFunctionWithName(&NSString::from_str(name))?;
    be.device().newComputePipelineStateWithFunction_error(&f).ok()
}

fn add(line: &mut String, name: &str, r: (f64, f64), extra: String) {
    *line += &format!(" {name}={:.3}/{:.3}{extra}", r.0, r.1);
}

fn med_min(mut v: Vec<f64>) -> (f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[v.len() / 2], v[0])
}

impl MetalBackend {
    /// See the module docs. `tag` names the call site in the log line. Waits for the engine's queue first.
    pub fn mode_probe(&self, tag: &str) {
        if let Err(e) = self.mode_probe_inner(tag) {
            eprintln!("modeprobe {tag}: failed: {e}");
        }
    }

    fn mode_probe_inner(&self, tag: &str) -> intarsia_core::Result<()> {
        use intarsia_core::backend::Backend;
        use intarsia_core::Error;
        self.synchronize(0)?;
        let t0 = std::time::Instant::now();
        let dev = self.device();
        let lib = dev
            .newLibraryWithSource_options_error(&NSString::from_str(SRC), None)
            .map_err(|e| Error::backend(format!("modeprobe compile: {e:?}")))?;
        let (alu, seq, gat) = match (pso(self, &lib, "mp_alu"), pso(self, &lib, "mp_seq"), pso(self, &lib, "mp_gather")) {
            (Some(a), Some(b), Some(c)) => (a, b, c),
            _ => return Err(Error::backend("modeprobe: pipeline")),
        };
        let files = host::whole_buffers(self);
        let mut line = format!(
            "modeprobe {tag} t_ms={} alloc_gb={:.2} wset_gb={:.2} files={}",
            host::now_ms(),
            dev.currentAllocatedSize() as f64 / 1e9,
            dev.recommendedMaxWorkingSetSize() as f64 / 1e9,
            files.len()
        );
        for (i, (mtl, floor)) in files.iter().enumerate() {
            line += &format!(" f{i}_gb={:.2} f{i}_res={:.1}", mtl.length() as f64 / 1e9, resident_pct(*floor, mtl.length()));
        }
        let threads = 160 * 1024;
        let out = self.new_mtl_buffer(threads * 4, false)?;
        let anon = self.new_mtl_buffer(ANON, false)?;
        // first touch of the anonymous control buffer (GPU fill), not timed
        {
            let cb = self.queue().commandBuffer().ok_or_else(|| Error::backend("cb"))?;
            let enc = cb.blitCommandEncoder().ok_or_else(|| Error::backend("blit"))?;
            enc.fillBuffer_range_value(&anon, NSRange::new(0, ANON), 0x5a);
            enc.endEncoding();
            cb.commit();
            MetalBackend::wait(&cb)?;
        }
        let run = |p: &Pso, bufs: &[(&MtlBuffer, usize)], cst: &[u8], grid: usize, reps: usize| -> intarsia_core::Result<(f64, f64)> {
            let mut ts = vec![];
            for _ in 0..reps {
                let cb = self.queue().commandBuffer().ok_or_else(|| Error::backend("cb"))?;
                let enc = cb.computeCommandEncoder().ok_or_else(|| Error::backend("enc"))?;
                enc.setComputePipelineState(p);
                // SAFETY: every kernel reads inside the bound ranges (sizes passed in `cst`) and writes `out[grid]`.
                unsafe {
                    for (i, (b, o)) in bufs.iter().enumerate() {
                        enc.setBuffer_offset_atIndex(Some(b), *o, i);
                    }
                    enc.setBuffer_offset_atIndex(Some(&out), 0, bufs.len());
                    enc.setBytes_length_atIndex(NonNull::new_unchecked(cst.as_ptr() as *mut c_void), cst.len(), bufs.len() + 1);
                }
                enc.dispatchThreads_threadsPerThreadgroup(
                    MTLSize { width: grid, height: 1, depth: 1 },
                    MTLSize { width: 256, height: 1, depth: 1 },
                );
                enc.endEncoding();
                cb.commit();
                MetalBackend::wait(&cb)?;
                ts.push((cb.GPUEndTime() - cb.GPUStartTime()) * 1e3);
            }
            Ok(med_min(ts))
        };
        // alu: 6 FMA chains x 16384 iterations x 160k threads
        let r = run(&alu, &[], &16384u32.to_le_bytes(), threads, 5)?;
        add(&mut line, "alu", r, String::new());
        // streaming reads
        let mut seq_bytes = 0usize;
        let mut seq_ms = 0.0;
        for (mtl, _) in &files {
            let n16 = (mtl.length() / 16) as u64;
            let r = run(&seq, &[(mtl, 0)], &n16.to_le_bytes(), threads, 3)?;
            seq_bytes += mtl.length();
            seq_ms += r.0;
        }
        if !files.is_empty() {
            line += &format!(" seq_file_gbs={:.1}", seq_bytes as f64 / seq_ms / 1e6);
        }
        let r = run(&seq, &[(&anon, 0)], &((ANON / 16) as u64).to_le_bytes(), threads, 3)?;
        add(&mut line, "seq_anon", r, format!("({:.1}GB/s)", ANON as f64 / r.0 / 1e6));
        // page gathers: 160k threads x 16 reads = 2.6 M random pages
        let g = |pages: usize| {
            let mut b = (pages as u32).to_le_bytes().to_vec();
            b.extend_from_slice(&16u32.to_le_bytes());
            b
        };
        if let Some((big, _)) = files.iter().max_by_key(|(m, _)| m.length()) {
            let pages = big.length() / PAGE - 1;
            add(&mut line, "gather_file", run(&gat, &[(big, 0)], &g(pages), threads, 5)?, String::new());
            add(&mut line, "gather_file2g", run(&gat, &[(big, 0)], &g(ANON / PAGE - 1), threads, 5)?, String::new());
        }
        add(&mut line, "gather_anon", run(&gat, &[(&anon, 0)], &g(ANON / PAGE - 1), threads, 5)?, String::new());
        // sustained back-to-back load (lane ink-moepf2: MoE calls start ~7.2 ms and settle ~25 % slower after 0.4-2 s of GPU
        // work; a new process starts fast again). INTARSIA_METAL_MODE_PROBE_SUSTAIN_MS=<ms> alternates the alu buffer (clock) and
        // a 2 GiB anonymous streaming read (memory) that long and logs each series in 8 equal slices; then the alu on a fresh
        // command queue (does a new queue reset it), and again after 2 s of GPU idle.
        if let Some(ms) = std::env::var("INTARSIA_METAL_MODE_PROBE_SUSTAIN_MS").ok().and_then(|v| v.parse::<u64>().ok()) {
            let ts0 = std::time::Instant::now();
            let (mut ta, mut tm) = (vec![], vec![]);
            while ts0.elapsed().as_millis() < ms as u128 {
                ta.push(run(&alu, &[], &16384u32.to_le_bytes(), threads, 1)?.0);
                tm.push(run(&seq, &[(&anon, 0)], &((ANON / 16) as u64).to_le_bytes(), threads, 1)?.0);
            }
            let sl = |ts: &[f64]| -> String {
                let k = ts.len().div_ceil(8).max(1);
                ts.chunks(k).map(|c| format!("{:.2}", med_min(c.to_vec()).0)).collect::<Vec<_>>().join(",")
            };
            line += &format!(" sustain_n={} sustain={} sustain_mem={}", ta.len(), sl(&ta), sl(&tm));
            // fresh queue right after the sustained load
            let q2 = dev.newCommandQueue().ok_or_else(|| Error::backend("queue"))?;
            let mut t2 = vec![];
            for _ in 0..3 {
                let cb = q2.commandBuffer().ok_or_else(|| Error::backend("cb"))?;
                let enc = cb.computeCommandEncoder().ok_or_else(|| Error::backend("enc"))?;
                enc.setComputePipelineState(&alu);
                let it = 16384u32.to_le_bytes();
                // SAFETY: as in `run`
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&out), 0, 0);
                    enc.setBytes_length_atIndex(NonNull::new_unchecked(it.as_ptr() as *mut c_void), 4, 1);
                }
                enc.dispatchThreads_threadsPerThreadgroup(
                    MTLSize { width: threads, height: 1, depth: 1 },
                    MTLSize { width: 256, height: 1, depth: 1 },
                );
                enc.endEncoding();
                cb.commit();
                MetalBackend::wait(&cb)?;
                t2.push((cb.GPUEndTime() - cb.GPUStartTime()) * 1e3);
            }
            add(&mut line, "alu_newq", med_min(t2), String::new());
            std::thread::sleep(std::time::Duration::from_millis(2000));
            add(&mut line, "alu_idle2s", run(&alu, &[], &16384u32.to_le_bytes(), threads, 3)?, String::new());
        }
        let copy = crate::bench::measure_copy_gbs(self, 1 << 30, 3, 7).unwrap_or(-1.0);
        line += &format!(" copy_gbs={copy:.1} probe_ms={}", t0.elapsed().as_millis());
        eprintln!("{line}");
        Ok(())
    }
}
