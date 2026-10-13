//! The Backend trait (ARCHITECTURE 3.1): device buffers, copies, and kernel launch.
//!
//! A backend is deliberately small. It owns memory and runs *named kernel
//! implementations* of ops; selection (which implementation for which shape) is
//! done by the [`crate::registry::Registry`] from the backend's `kernels()` list.
//! `Backend` is object safe: hold it as `Arc<dyn Backend>`.
//!
//! Threading: every method may be called from any thread. In particular the
//! engine worker runs generation, `Engine::forward_logits_each` runs on the
//! caller's thread, and the telemetry probe calls `info()` from whichever thread
//! serves `/metrics` or `/events`. A GPU backend with a per-thread current
//! context must bind its context inside each entry point.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::OpCall;
use crate::registry::KernelImpl;
use std::any::Any;
use std::sync::Arc;

/// Index of an execution stream. Phase 0 uses stream 0 only.
pub type StreamId = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemKind {
    /// Device-resident memory (unified/managed on GB10 is still `Device` here).
    Device,
    /// Pinned host memory for staging.
    Pinned,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct DeviceInfo {
    /// Marketing name, e.g. "NVIDIA GB10" or "cpu-ref (Apple M3 Max)".
    pub name: String,
    /// Backend id: "cpu", "cuda", "hip".
    pub backend: String,
    /// Architecture string kernels are registered against ("any", "sm_121", "gfx1201").
    pub arch: String,
    /// Total memory the engine may plan against, bytes.
    pub total_mem: u64,
    /// Free memory *measured now* (driver query / host RAM), bytes.
    pub free_mem: u64,
    /// Host and device share memory (GB10, Apple, cpu-ref).
    pub unified: bool,
    /// Measured memory bandwidth in GB/s when known.
    pub bandwidth_gbs: Option<f64>,
}

/// Backend-specific allocation. Implementors are plain handles (a device
/// pointer, a host vector, ...) and must be freed on drop.
pub trait BufferImpl: Any + Send + Sync {
    fn bytes(&self) -> usize;
    fn as_any(&self) -> &dyn Any;
}

/// A reference-counted device allocation.
#[derive(Clone)]
pub struct Buffer(Arc<dyn BufferImpl>);

impl Buffer {
    pub fn new<B: BufferImpl>(b: B) -> Buffer {
        Buffer(Arc::new(b))
    }
    pub fn bytes(&self) -> usize {
        self.0.bytes()
    }
    /// Downcast to the concrete backend buffer type.
    pub fn downcast<T: 'static>(&self) -> Option<&T> {
        self.0.as_any().downcast_ref::<T>()
    }
    pub fn ptr_eq(&self, other: &Buffer) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Buffer({} bytes)", self.bytes())
    }
}

/// A captured, replayable stream of launches (CUDA graph). Plain handle; freed on drop.
pub trait GraphImpl: Any + Send + Sync {
    fn as_any(&self) -> &dyn Any;
}

/// A reference-counted captured graph (see [`Backend::graph_capture`]).
#[derive(Clone)]
pub struct GraphHandle(Arc<dyn GraphImpl>);

impl GraphHandle {
    pub fn new<G: GraphImpl>(g: G) -> GraphHandle {
        GraphHandle(Arc::new(g))
    }
    pub fn downcast<T: 'static>(&self) -> Option<&T> {
        self.0.as_any().downcast_ref::<T>()
    }
}

impl std::fmt::Debug for GraphHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GraphHandle")
    }
}

/// Immutable host memory that a backend with host-addressable memory may use
/// without copying (mmap'd weight files). See [`Backend::wrap_host`].
pub trait HostMemory: Send + Sync {
    fn as_slice(&self) -> &[u8];
}

/// A dense row-major view into a [`Buffer`].
#[derive(Clone, Debug)]
pub struct TensorRef {
    pub buf: Buffer,
    /// Byte offset of element 0 inside `buf`.
    pub offset: usize,
    pub dtype: DType,
    pub shape: Vec<usize>,
}

impl TensorRef {
    pub fn new(buf: Buffer, dtype: DType, shape: &[usize]) -> TensorRef {
        TensorRef { buf, offset: 0, dtype, shape: shape.to_vec() }
    }
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn nbytes(&self) -> usize {
        self.numel() * self.dtype.size_bytes()
    }
    /// Reinterpret with a new shape of equal element count (no data movement).
    pub fn reshaped(&self, shape: &[usize]) -> TensorRef {
        assert_eq!(shape.iter().product::<usize>(), self.numel(), "reshape element count mismatch");
        TensorRef { shape: shape.to_vec(), ..self.clone() }
    }
    /// View rows `[start, start+n)` of the leading dimension.
    pub fn rows(&self, start: usize, n: usize) -> TensorRef {
        assert!(!self.shape.is_empty() && start + n <= self.shape[0], "row slice out of range");
        let row_elems: usize = self.shape[1..].iter().product();
        let mut shape = self.shape.clone();
        shape[0] = n;
        TensorRef {
            buf: self.buf.clone(),
            offset: self.offset + start * row_elems * self.dtype.size_bytes(),
            dtype: self.dtype,
            shape,
        }
    }
    /// Check the view lies inside its buffer.
    pub fn check_bounds(&self) -> Result<()> {
        if self.offset + self.nbytes() > self.buf.bytes() {
            return Err(Error::invalid(format!(
                "tensor view out of bounds: offset {} + {} bytes > buffer {}",
                self.offset,
                self.nbytes(),
                self.buf.bytes()
            )));
        }
        Ok(())
    }
}

pub trait Backend: Send + Sync {
    /// Backend id: "cpu", "cuda", "hip".
    fn name(&self) -> &str;

    /// Device description with memory figures measured at call time.
    fn info(&self) -> DeviceInfo;

    /// Allocate `bytes` of uninitialised (cpu-ref: zeroed) memory.
    fn alloc(&self, bytes: usize, kind: MemKind) -> Result<Buffer>;

    /// Host -> device copy. May be asynchronous on `stream`; `src` must stay
    /// valid until the call returns (implementations stage if needed).
    fn upload(&self, dst: &Buffer, dst_offset: usize, src: &[u8], stream: StreamId) -> Result<()>;

    /// Device -> host copy. Blocking: when it returns `dst` is filled.
    fn download(&self, src: &Buffer, src_offset: usize, dst: &mut [u8], stream: StreamId) -> Result<()>;

    /// Device -> device copy of `bytes`.
    fn copy(
        &self,
        dst: &Buffer,
        dst_offset: usize,
        src: &Buffer,
        src_offset: usize,
        bytes: usize,
        stream: StreamId,
    ) -> Result<()>;

    /// Fill a buffer with zero bytes.
    fn zero(&self, dst: &Buffer, stream: StreamId) -> Result<()> {
        let chunk = vec![0u8; dst.bytes().min(1 << 20)];
        let mut off = 0;
        while off < dst.bytes() {
            let n = chunk.len().min(dst.bytes() - off);
            self.upload(dst, off, &chunk[..n], stream)?;
            off += n;
        }
        Ok(())
    }

    /// Zero-copy import of immutable host memory (mmap'd weights). Backends
    /// whose device can address host memory directly return `Some`; discrete
    /// GPUs return `None` and the loader falls back to alloc + upload.
    /// The returned buffer is read-only.
    fn wrap_host(&self, _mem: Arc<dyn HostMemory>, _offset: usize, _len: usize) -> Option<Buffer> {
        None
    }

    /// The kernel implementations this backend provides (explicit list, no
    /// link-section magic). Exactly one of them per op may have `is_reference`.
    fn kernels(&self) -> Vec<KernelImpl>;

    /// Launch the implementation called `kernel` (a `KernelImpl::name` from
    /// `kernels()`) on `stream`. Asynchronous on devices; the cpu backend runs
    /// synchronously. Inputs/outputs are described by `call`.
    fn launch(&self, kernel: &str, call: &OpCall, stream: StreamId) -> Result<()>;

    /// Wait for all work on `stream`.
    fn synchronize(&self, stream: StreamId) -> Result<()>;

    /// Can this backend record launches into a replayable graph (`graph_capture`)?
    fn supports_graphs(&self) -> bool {
        false
    }

    /// Run `body` with every launch on `stream` *recorded instead of executed*, and return the recording. Nothing
    /// in `body` may synchronise, allocate, or copy from pageable host memory; it must not depend on host state
    /// that changes between replays (position and token ids go through device buffers the caller rewrites).
    /// The recording keeps the buffer addresses it saw: the caller keeps those buffers alive and in place for
    /// as long as it replays the graph. If `body` fails, the capture is discarded and the error returned.
    fn graph_capture(&self, _stream: StreamId, _body: &mut dyn FnMut() -> Result<()>) -> Result<GraphHandle> {
        Err(Error::Unsupported(format!("backend '{}' has no graph capture", self.name())))
    }

    /// Enqueue the recorded launches on `stream` (asynchronous, like any launch).
    fn graph_replay(&self, _g: &GraphHandle, _stream: StreamId) -> Result<()> {
        Err(Error::Unsupported(format!("backend '{}' has no graph replay", self.name())))
    }

    /// Identity of this host + kernel pack for admission caching: anything that can change a
    /// kernel's numerics (device, arch; backends add driver/toolkit/pack hash). Admission results
    /// are cached per fingerprint.
    fn fingerprint(&self) -> String {
        let i = self.info();
        format!("{}|{}|{}", self.name(), i.name, i.arch)
    }
}

/// Upload a whole host slice into a fresh buffer.
pub fn upload_new(backend: &dyn Backend, data: &[u8]) -> Result<Buffer> {
    let b = backend.alloc(data.len(), MemKind::Device)?;
    backend.upload(&b, 0, data, 0)?;
    Ok(b)
}
