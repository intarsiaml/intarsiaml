//! Spike measurement helpers: GPU-timestamped copy bandwidth, statistics, data fill. Used by `info()` (copy
//! bandwidth) and by the examples (`gemv_bw`, `icb_check`).

use crate::kernels::{COPY_PER_THREAD, COPY_U4, READ_PER_THREAD, READ_XOR_U4};
use crate::{set_bytes, MetalBackend, MtlBuffer};
use intarsia_core::{Error, Result};
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLComputeCommandEncoder, MTLSize};

/// GPU execution time of a completed command buffer, seconds (`GPUEndTime - GPUStartTime`).
pub fn gpu_seconds(cb: &ProtocolObject<dyn MTLCommandBuffer>) -> f64 {
    cb.GPUEndTime() - cb.GPUStartTime()
}

#[derive(Clone, Copy, Debug)]
pub struct Stats {
    pub p10: f64,
    pub median: f64,
    pub p90: f64,
}

/// p10 / median / p90 of `v` (sorted in place; linear interpolation between ranks).
pub fn stats(v: &mut [f64]) -> Stats {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| {
        let r = p * (v.len() - 1) as f64;
        let (lo, hi) = (r.floor() as usize, r.ceil() as usize);
        v[lo] + (v[hi] - v[lo]) * (r - lo as f64)
    };
    Stats { p10: q(0.1), median: q(0.5), p90: q(0.9) }
}

/// bf16 (as raw bits) of an f32, round to nearest even.
pub fn f32_to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// xorshift64* stream.
pub struct Rng(pub u64);
impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    /// Uniform in [-1, 1).
    pub fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u32 << 23) as f32) - 1.0
    }
}

/// Fill `n` bf16 values at `p` with uniform [-1, 1) noise (random bits, so no value compresses or repeats cheaply).
///
/// # Safety
/// `p` must be valid for `n` u16 writes.
pub unsafe fn fill_bf16_random(p: *mut u16, n: usize, seed: u64) {
    let mut r = Rng(seed | 1);
    for i in 0..n {
        *p.add(i) = f32_to_bf16(r.unit());
    }
}

/// Threads for `n16` 16-byte vectors at `per_thread` each: whole simdgroups (the shaders index per simdgroup).
pub fn copy_threads(n16: usize, per_thread: usize) -> usize {
    n16.div_ceil(per_thread).div_ceil(32) * 32
}

pub(crate) fn dispatch_threads(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, threads: usize) {
    enc.dispatchThreads_threadsPerThreadgroup(
        MTLSize { width: threads, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}

/// Encode `copy_u4`: `bytes` (a multiple of 16) from `src` to `dst`.
pub fn encode_copy(
    be: &MetalBackend,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    src: &ProtocolObject<dyn MTLBuffer>,
    dst: &ProtocolObject<dyn MTLBuffer>,
    bytes: usize,
) -> Result<()> {
    let n16 = (bytes / 16) as u32;
    enc.setComputePipelineState(be.pso(COPY_U4).ok_or_else(|| Error::backend("no copy pipeline"))?);
    // SAFETY: both buffers hold at least `bytes`.
    unsafe {
        enc.setBuffer_offset_atIndex(Some(src), 0, 0);
        enc.setBuffer_offset_atIndex(Some(dst), 0, 1);
    }
    set_bytes(enc, &n16, 2);
    dispatch_threads(enc, copy_threads(n16 as usize, COPY_PER_THREAD));
    Ok(())
}

/// Encode `read_xor_u4` over `bytes` of `src`; `out` needs 4 bytes per thread (`bytes / 64`, rounded up).
pub fn encode_read(
    be: &MetalBackend,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    src: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    bytes: usize,
) -> Result<()> {
    let n16 = (bytes / 16) as u32;
    enc.setComputePipelineState(be.pso(READ_XOR_U4).ok_or_else(|| Error::backend("no read pipeline"))?);
    // SAFETY: `src` holds `bytes`, `out` one u32 per thread.
    unsafe {
        enc.setBuffer_offset_atIndex(Some(src), 0, 0);
        enc.setBuffer_offset_atIndex(Some(out), 0, 1);
    }
    set_bytes(enc, &n16, 2);
    dispatch_threads(enc, copy_threads(n16 as usize, READ_PER_THREAD));
    Ok(())
}

/// Bytes of `out` that `encode_read` over `bytes` writes.
pub fn read_out_bytes(bytes: usize) -> usize {
    (bytes / 16).div_ceil(READ_PER_THREAD).div_ceil(256) * 256 * 4
}

/// Copy-kernel GB/s samples (read + write bytes over GPU time) for `bytes` of source, `warmup` untimed runs first.
pub fn copy_gbs_samples(be: &MetalBackend, bytes: usize, warmup: usize, reps: usize) -> Result<Vec<f64>> {
    let src = be.new_mtl_buffer(bytes, true)?;
    let dst = be.new_mtl_buffer(bytes, true)?;
    // SAFETY: freshly allocated, `bytes` long; touching the pages up front keeps first-touch faults out of the timing.
    unsafe {
        std::ptr::write_bytes(src.contents().as_ptr() as *mut u8, 0x5a, bytes);
        std::ptr::write_bytes(dst.contents().as_ptr() as *mut u8, 0xa5, bytes);
    }
    let mut out = Vec::with_capacity(reps);
    for i in 0..warmup + reps {
        let cb = be.encode_and_commit(|enc| {
            let _ = encode_copy(be, enc, &src, &dst, bytes);
        })?;
        MetalBackend::wait(&cb)?;
        if i >= warmup {
            out.push(2.0 * bytes as f64 / gpu_seconds(&cb) / 1e9);
        }
    }
    Ok(out)
}

/// Median copy GB/s (see [`copy_gbs_samples`]).
pub fn measure_copy_gbs(be: &MetalBackend, bytes: usize, warmup: usize, reps: usize) -> Result<f64> {
    Ok(stats(&mut copy_gbs_samples(be, bytes, warmup, reps)?).median)
}

/// Keep an `MtlBuffer` list alive helper for readability in the examples.
pub type Pool = Vec<MtlBuffer>;
