//! `Backend::zero` and `Backend::copy` as dispatches appended to the stream's open serial compute encoder: no flush, no
//! host wait, and ordered with the surrounding launches by the encoder's serial dispatch type (each dispatch starts
//! after the previous one finished and sees its writes). A blit would need the compute encoder ended and reopened around
//! it (an encoder boundary per call, and a second code path through `Stream`); a compute kernel keeps the single
//! encoder and the existing `copy_u4` is already the bandwidth-ceiling kernel.
//!
//! The 16-byte kernels take 16-byte aligned bindings and a multiple of 16 bytes; whatever is left over (unaligned
//! operands, or a tail under 16 bytes) goes through the one-thread-per-byte kernels, which bind the whole MTLBuffer and
//! take 64-bit byte offsets. Overlapping ranges inside one buffer (`copy` has memmove semantics) cannot run as a parallel
//! kernel and take the old host path (flush, wait, memmove).

use crate::bench::{copy_threads, dispatch_threads};
use crate::kernels::{COPY_BYTES, COPY_PER_THREAD, COPY_U4, ZERO_BYTES, ZERO_U4};
use crate::{set_bytes, MetalBackend, MetalBuffer};
use intarsia_core::backend::Buffer;
use intarsia_core::{Error, Result};
use objc2_metal::MTLComputeCommandEncoder;

/// Bytes per 16-byte-kernel dispatch: its element count is a u32 and its thread index must not wrap.
const VEC_CHUNK: usize = 1 << 32;
/// Bytes per byte-kernel dispatch (one thread per byte, 32-bit thread index).
const BYTE_CHUNK: usize = 1 << 30;

impl MetalBackend {
    fn check_range(mb: &MetalBuffer, offset: usize, len: usize) -> Result<()> {
        if offset.checked_add(len).is_none_or(|e| e > mb.bytes) {
            return Err(Error::backend("metal buffer access out of bounds"));
        }
        Ok(())
    }

    /// Zero all of `dst` on the GPU (appended to the stream, not waited on).
    pub(crate) fn gpu_zero(&self, dst: &Buffer) -> Result<()> {
        let mb = self.mbuf(dst)?;
        if mb.read_only {
            return Err(Error::backend("metal buffer is read-only (wrapped host memory)"));
        }
        let (base, bytes) = (mb.base, mb.bytes);
        let vec = if base % 16 == 0 { bytes & !15 } else { 0 };
        let pso = self.pso(ZERO_U4).ok_or_else(|| Error::backend("metal: no zero pipeline"))?;
        let mut done = 0;
        while done < vec {
            let n = (vec - done).min(VEC_CHUNK);
            let n16 = (n / 16) as u32;
            self.with_encoder(|enc| {
                enc.setComputePipelineState(pso);
                // SAFETY: `base + done + n <= base + bytes`, inside the MTLBuffer; 16-byte aligned.
                unsafe { enc.setBuffer_offset_atIndex(Some(&mb.mtl), base + done, 0) };
                set_bytes(enc, &n16, 1);
                dispatch_threads(enc, copy_threads(n16 as usize, COPY_PER_THREAD));
            })?;
            done += n;
        }
        let pso = self.pso(ZERO_BYTES).ok_or_else(|| Error::backend("metal: no zero pipeline"))?;
        while done < bytes {
            let n = (bytes - done).min(BYTE_CHUNK);
            let p: [u64; 2] = [(base + done) as u64, n as u64];
            self.with_encoder(|enc| {
                enc.setComputePipelineState(pso);
                // SAFETY: the kernel indexes `[base + done, base + done + n)`, inside the MTLBuffer.
                unsafe { enc.setBuffer_offset_atIndex(Some(&mb.mtl), 0, 0) };
                set_bytes(enc, &p, 1);
                dispatch_threads(enc, n);
            })?;
            done += n;
        }
        Ok(())
    }

    /// Copy `bytes` between buffers on the GPU (appended to the stream, not waited on). Returns `Ok(false)` without
    /// doing anything when the ranges overlap inside one MTLBuffer; the caller takes the host path then.
    pub(crate) fn gpu_copy(
        &self,
        dst: &Buffer,
        dst_offset: usize,
        src: &Buffer,
        src_offset: usize,
        bytes: usize,
    ) -> Result<bool> {
        let (db, sb) = (self.mbuf(dst)?, self.mbuf(src)?);
        Self::check_range(db, dst_offset, bytes)?;
        Self::check_range(sb, src_offset, bytes)?;
        if db.read_only {
            return Err(Error::backend("metal buffer is read-only (wrapped host memory)"));
        }
        // absolute byte positions inside each MTLBuffer
        let (d0, s0) = (db.base + dst_offset, sb.base + src_offset);
        let same = std::ptr::eq(&*db.mtl as *const _ as *const u8, &*sb.mtl as *const _ as *const u8);
        if same && d0 < s0 + bytes && s0 < d0 + bytes {
            return Ok(false);
        }
        let vec = if d0 % 16 == 0 && s0 % 16 == 0 { bytes & !15 } else { 0 };
        let pso = self.pso(COPY_U4).ok_or_else(|| Error::backend("metal: no copy pipeline"))?;
        let mut done = 0;
        while done < vec {
            let n = (vec - done).min(VEC_CHUNK);
            let n16 = (n / 16) as u32;
            self.with_encoder(|enc| {
                enc.setComputePipelineState(pso);
                // SAFETY: both ranges were bounds-checked above and are 16-byte aligned.
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&sb.mtl), s0 + done, 0);
                    enc.setBuffer_offset_atIndex(Some(&db.mtl), d0 + done, 1);
                }
                set_bytes(enc, &n16, 2);
                dispatch_threads(enc, copy_threads(n16 as usize, COPY_PER_THREAD));
            })?;
            done += n;
        }
        let pso = self.pso(COPY_BYTES).ok_or_else(|| Error::backend("metal: no copy pipeline"))?;
        while done < bytes {
            let n = (bytes - done).min(BYTE_CHUNK);
            let p: [u64; 3] = [(s0 + done) as u64, (d0 + done) as u64, n as u64];
            self.with_encoder(|enc| {
                enc.setComputePipelineState(pso);
                // SAFETY: the kernel reads and writes only the bounds-checked ranges.
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&sb.mtl), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&db.mtl), 0, 1);
                }
                set_bytes(enc, &p, 2);
                dispatch_threads(enc, n);
            })?;
            done += n;
        }
        Ok(true)
    }
}
