//! `MetalBuffer`: a shared-storage `MTLBuffer` (or a window onto one) behind the backend's [`Buffer`] handle.

use intarsia_core::backend::{BufferImpl, HostMemory};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use std::any::Any;
use std::sync::Arc;

pub type MtlBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;

pub struct MetalBuffer {
    pub(crate) mtl: MtlBuffer,
    /// Byte offset of this buffer's byte 0 inside `mtl`. Non-zero for [`crate::MetalBackend::wrap_host`] buffers,
    /// which are windows onto one MTLBuffer that covers a whole mmap'd file. `TensorRef::offset` adds on top.
    pub(crate) base: usize,
    pub(crate) bytes: usize,
    /// Wrapped host memory: the GPU only reads it, host writes are refused.
    pub(crate) read_only: bool,
    /// Keeps the mapping alive for as long as the buffer exists (no-copy buffers do not own their memory).
    pub(crate) _host: Option<Arc<dyn HostMemory>>,
}

// SAFETY: an MTLBuffer is thread-safe to use and share (it is a handle to shared-storage memory).
unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

impl MetalBuffer {
    pub fn mtl(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.mtl
    }
    /// Byte offset of byte 0 of this buffer inside [`Self::mtl`].
    pub fn base(&self) -> usize {
        self.base
    }
    pub fn is_read_only(&self) -> bool {
        self.read_only
    }
    /// Host address of byte 0 (shared storage: CPU and GPU see the same memory).
    pub fn host_ptr(&self) -> *mut u8 {
        // SAFETY: `base` lies inside the MTLBuffer by construction.
        unsafe { (self.mtl.contents().as_ptr() as *mut u8).add(self.base) }
    }
}

impl BufferImpl for MetalBuffer {
    fn bytes(&self) -> usize {
        self.bytes
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
