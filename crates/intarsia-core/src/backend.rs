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
    /// How the bytes of this allocation are arranged. Always [`Layout::RowMajor`] (the checkpoint's order) unless the backend rearranged them
    /// on purpose ([`Backend::relayout_fp8_frag`]); never inferred, never set by anything but that call.
    fn layout(&self) -> Layout {
        Layout::RowMajor
    }
}

/// Byte order of a weight allocation. `RowMajor` is the checkpoint layout every kernel, the CPU oracle and the host copies assume.
/// `Fp8Frag` (HIP only, opt-in) is the dense fp8 matrix `[N, K]` (N % 16 == 0, K % 32 == 0) stored as `[N/16][K/32][lane = 16 h + col][16 B]`
/// so one 16-byte load per lane of a wave is a WMMA A fragment (`kernels/hip/ik_f8_rows.h`). The tag travels with the device bytes (it is a
/// property of the allocation), so every `TensorRef` made from the weight, however an `OpCall` is built, reports it: kernels that do not
/// declare [`crate::registry::KernelImpl::frag_ok`] are never selected for a call with a tagged input, and the HIP launchers reject the
/// tag's dtype themselves.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Layout {
    #[default]
    RowMajor,
    Fp8Frag,
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
    pub fn layout(&self) -> Layout {
        self.0.layout()
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
    /// The byte order of the allocation this view is in ([`Layout`]).
    pub fn layout(&self) -> Layout {
        self.buf.layout()
    }
    /// Reinterpret with a new shape of equal element count (no data movement).
    pub fn reshaped(&self, shape: &[usize]) -> TensorRef {
        assert_eq!(self.layout(), Layout::RowMajor, "reshape of a {:?} tensor: its bytes are not in row-major order", self.layout());
        assert_eq!(shape.iter().product::<usize>(), self.numel(), "reshape element count mismatch");
        TensorRef { shape: shape.to_vec(), ..self.clone() }
    }
    /// View rows `[start, start+n)` of the leading dimension.
    pub fn rows(&self, start: usize, n: usize) -> TensorRef {
        assert_eq!(self.layout(), Layout::RowMajor, "row slice of a {:?} tensor: its bytes are not in row-major order", self.layout());
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

    /// True when a 1-row prefill chunk is not bitwise equal to the same row inside a bigger chunk (the backend routes
    /// `m == 1` to a decode kernel), so prefill chunking should avoid ending in a 1-row tail where it can
    /// ([`crate::scheduler::next_prefill_chunk`]). Default false: the chunking is then exactly `chunk, chunk, .., rest`.
    fn avoid_single_row_prefill(&self) -> bool {
        false
    }

    /// Most sequences one batched decode step may carry on this backend (`Model::enable_batch`, `--max-batch`): the number of
    /// rows its row-exact multi-row kernels are registered and proven for (G-B: row r of an M-row call is bitwise the one-row
    /// kernel). Default 8 (Metal's, and what the cpu backend serves in tests); the CUDA backend proves 16. Never above
    /// [`crate::scheduler::MAX_BATCH_CAP`].
    fn batch_row_cap(&self) -> usize {
        crate::engine::SERVE_MAX_BATCH
    }

    /// Can this backend's fused decode ops run on one rank's shard under tensor parallelism ("row-local fusion")? True
    /// when every fused kernel takes its head counts and widths from the call (the rank's local dims), never from the
    /// unsharded model. The model still keeps the fused ops that would swallow a collective off (`Model::fused_ok`),
    /// and fusion under TP follows `LoadOptions::tp_fusion` (the CLI turns it on for CUDA, and for HIP at tp=2). Default false:
    /// fusion off at world > 1, and an explicit request is a load error.
    fn fusion_under_tp(&self) -> bool {
        false
    }

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

    /// Whether this backend would store a dense fp8 weight `[n, k]` in [`Layout::Fp8Frag`] (HIP, opt-in; false everywhere else).
    /// `act_quant`: the weight is W8A8. Asked once per tensor by the fp8 loader ([`crate::quant::fp8`]).
    fn fp8_frag_wanted(&self, _n: usize, _k: usize, _act_quant: bool) -> bool {
        false
    }

    /// Rearrange the dense fp8 weight `w` (the whole allocation, row-major, `crate::quant::fp8::frag_eligible`) into [`Layout::Fp8Frag`]
    /// in place and tag its buffer. Only a backend whose `fp8_frag_wanted` can answer true implements it.
    fn relayout_fp8_frag(&self, _w: &TensorRef) -> Result<()> {
        Err(Error::backend("this backend has no fragment-order fp8 layout"))
    }

    /// The bytes of `t` in the checkpoint's row-major order, whatever layout the allocation has (a [`Layout::Fp8Frag`] weight is
    /// un-permuted). `download` of a tagged allocation is refused, so a raw read of tagged bytes cannot happen by accident.
    fn download_row_major(&self, t: &TensorRef) -> Result<Vec<u8>> {
        if t.layout() != Layout::RowMajor {
            return Err(Error::backend(format!("{:?} tensor: this backend cannot read it back in row-major order", t.layout())));
        }
        let mut data = vec![0u8; t.nbytes()];
        self.download(&t.buf, t.offset, &mut data, 0)?;
        Ok(data)
    }

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

    /// Device-side token feedback for greedy decode (`Model::feedback_step`): `(supported, on by default)`. Supported
    /// means [`Backend::fence`] commits without waiting and [`Backend::read_after`] waits only for the work up to its
    /// fence, so the engine can encode step N+1 (fed by step N's on-device argmax) before it reads step N's token.
    /// Default `(false, false)`: the engine keeps the host round trip per token.
    fn token_feedback(&self) -> (bool, bool) {
        (false, false)
    }

    /// Submit everything enqueued on `stream` so far and return a fence for it. Default: wait for it (fence 0).
    fn fence(&self, stream: StreamId) -> Result<u64> {
        self.synchronize(stream)?;
        Ok(0)
    }

    /// Wait until the work up to `fence` has finished (later work may still run), then copy `dst.len()` bytes of
    /// `src` to the host. The caller guarantees no work after the fence writes that range. Default: [`Backend::download`].
    fn read_after(&self, fence: u64, src: &Buffer, src_offset: usize, dst: &mut [u8], stream: StreamId) -> Result<()> {
        let _ = fence;
        self.download(src, src_offset, dst, stream)
    }

    /// Token feedback: submit only the next `after` launches on `stream`, then keep encoding without submitting until
    /// [`Backend::release`] (submit the rest) or [`Backend::discard`] (drop it unrun). Lets the engine queue the start of
    /// a step it may not need, so a stop costs a few launches instead of a whole step. Default: no-op (everything is
    /// submitted as launched; `discard` then cannot undo anything, which the engine's rollback tolerates).
    fn hold(&self, _stream: StreamId, _after: usize) -> Result<()> {
        Ok(())
    }

    /// End a [`Backend::hold`]: submit what was held back.
    fn release(&self, _stream: StreamId) -> Result<()> {
        Ok(())
    }

    /// End a [`Backend::hold`]: drop what was held back without running it.
    fn discard(&self, _stream: StreamId) -> Result<()> {
        Ok(())
    }

    /// Can this backend record launches into a replayable graph (`graph_capture`)?
    fn supports_graphs(&self) -> bool {
        false
    }

    /// Must a speculative verify pass ([`Model::forward_verify`], eager or the captured [`Model::spec_step`]) run its attention
    /// as one single-row decode call per row (`Op::AttnDecode`, the row's own position), instead of one multi-row
    /// `Op::AttnPrefill` call? True where the multi-row kernel is not bitwise the decode step's per row (HIP: the split-KV
    /// partition of the keys follows the call's row count, measured on the 27B at 4k and 16k) and takes no device position, so a
    /// recording could not be replayed at another position. Costs `rows` launches per attention layer; each is the very call
    /// the decode step makes, so the rows are exact by construction. Default false: the multi-row call.
    fn verify_attn_per_row(&self) -> bool {
        false
    }

    /// Does this backend run batched decode steps with row tables (`OpCall::rows`): `GdnConv`, `GdnDeltaRule`, `KvAppend`
    /// and `AttnDecode` over all rows of a step in one launch, the sequences' state buffers named by a table of device
    /// addresses? (CUDA.) Whether a step uses them is the model's decision (`Model::row_tables`).
    fn supports_row_tables(&self) -> bool {
        false
    }

    /// Device-side sampling (R3b): the `k` largest values of each f32 row of `logits` `[R, V]` in the sampler's order (value
    /// descending, ties by ascending index; `sampler::sample_row` walks the same order), as `ids` i32 `[R, k]` and `vals` f32
    /// `[R, k]`; `nan` i32 `[R]` is 1 for a row holding a NaN (its entries mean nothing). `scratch`: `R * 32 * k * 8` bytes,
    /// `k <= 64`. Not a registered op; a backend without it returns `Unsupported` and the caller samples on the host.
    #[allow(clippy::too_many_arguments)]
    fn topk_rows(
        &self,
        _logits: &TensorRef,
        _k: usize,
        _ids: &TensorRef,
        _vals: &TensorRef,
        _nan: &TensorRef,
        _scratch: &TensorRef,
        _stream: StreamId,
    ) -> Result<()> {
        Err(Error::Unsupported(format!("{}: device-side top-k", self.name())))
    }

    /// Does `topk_rows` work on this backend?
    fn supports_topk(&self) -> bool {
        false
    }

    /// The device address of the first byte of `buf`, for a row table (`OpCall::rows`). `None`: the backend has none.
    fn device_address(&self, _buf: &Buffer) -> Option<u64> {
        None
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
