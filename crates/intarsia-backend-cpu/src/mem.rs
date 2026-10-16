//! Host buffers and tensor accessors for the cpu backend.

use intarsia_core::backend::{BufferImpl, HostMemory};
use intarsia_core::dtype::{decode_to_f32, encode_from_f32};
use intarsia_core::{Error, Result, TensorRef};
use std::any::Any;
use std::cell::UnsafeCell;
use std::sync::Arc;

enum Kind {
    Owned(UnsafeCell<Vec<u64>>),
    /// Read-only view of external memory (mmap'd weights).
    Host {
        mem: Arc<dyn HostMemory>,
        offset: usize,
    },
}

pub struct CpuBuffer {
    kind: Kind,
    bytes: usize,
}

// SAFETY: the engine serialises op launches on a stream; rayon tasks inside an op
// write disjoint ranges. Shared host memory is immutable.
unsafe impl Send for CpuBuffer {}
unsafe impl Sync for CpuBuffer {}

impl CpuBuffer {
    pub fn zeroed(bytes: usize) -> CpuBuffer {
        CpuBuffer { kind: Kind::Owned(UnsafeCell::new(vec![0u64; bytes.div_ceil(8).max(1)])), bytes }
    }

    pub fn host(mem: Arc<dyn HostMemory>, offset: usize, len: usize) -> CpuBuffer {
        CpuBuffer { kind: Kind::Host { mem, offset }, bytes: len }
    }

    pub fn writable(&self) -> bool {
        matches!(self.kind, Kind::Owned(_))
    }

    fn ptr(&self) -> *mut u8 {
        match &self.kind {
            // SAFETY: only the pointer is taken; accesses are bounds-checked by callers.
            Kind::Owned(v) => unsafe { (*v.get()).as_mut_ptr() as *mut u8 },
            Kind::Host { mem, offset } => unsafe { mem.as_slice().as_ptr().add(*offset) as *mut u8 },
        }
    }

    pub fn read(&self, offset: usize, dst: &mut [u8]) -> Result<()> {
        if offset + dst.len() > self.bytes {
            return Err(Error::backend("cpu read out of bounds"));
        }
        // SAFETY: bounds checked above.
        unsafe { std::ptr::copy_nonoverlapping(self.ptr().add(offset), dst.as_mut_ptr(), dst.len()) };
        Ok(())
    }

    pub fn write(&self, offset: usize, src: &[u8]) -> Result<()> {
        if !self.writable() {
            return Err(Error::backend("cpu buffer is read-only (wrapped host memory)"));
        }
        if offset + src.len() > self.bytes {
            return Err(Error::backend("cpu write out of bounds"));
        }
        // SAFETY: bounds checked above, buffer is owned.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr().add(offset), src.len()) };
        Ok(())
    }
}

impl BufferImpl for CpuBuffer {
    fn bytes(&self) -> usize {
        self.bytes
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub fn cpu_buf(t: &TensorRef) -> Result<&CpuBuffer> {
    t.buf.downcast::<CpuBuffer>().ok_or_else(|| Error::backend("buffer does not belong to the cpu backend"))
}

/// Raw bytes of a tensor view.
pub fn bytes(t: &TensorRef) -> Result<&[u8]> {
    let b = cpu_buf(t)?;
    let n = t.nbytes();
    if t.offset + n > b.bytes {
        return Err(Error::backend("tensor view out of bounds"));
    }
    // SAFETY: bounds checked; callers do not hold this across a write to the same range.
    Ok(unsafe { std::slice::from_raw_parts(b.ptr().add(t.offset), n) })
}

/// Mutable bytes of an output tensor view.
///
/// # Safety
/// The caller must ensure no other reference to the range is live.
#[allow(clippy::mut_from_ref)]
pub unsafe fn bytes_mut(t: &TensorRef) -> Result<&mut [u8]> {
    let b = cpu_buf(t)?;
    if !b.writable() {
        return Err(Error::backend("output tensor is read-only host memory"));
    }
    let n = t.nbytes();
    if t.offset + n > b.bytes {
        return Err(Error::backend("tensor view out of bounds"));
    }
    Ok(std::slice::from_raw_parts_mut(b.ptr().add(t.offset), n))
}

/// Decode a whole tensor to f32.
pub fn read_f32(t: &TensorRef) -> Result<Vec<f32>> {
    let mut v = vec![0f32; t.numel()];
    decode_to_f32(t.dtype, bytes(t)?, &mut v);
    Ok(v)
}

/// Encode `vals` into the output tensor (round to nearest even).
pub fn write_f32(t: &TensorRef, vals: &[f32]) -> Result<()> {
    assert_eq!(vals.len(), t.numel());
    // SAFETY: outputs are exclusively owned by the running op.
    let dst = unsafe { bytes_mut(t)? };
    encode_from_f32(t.dtype, vals, dst);
    Ok(())
}
