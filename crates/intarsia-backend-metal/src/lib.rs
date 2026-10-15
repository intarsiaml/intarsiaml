//! Metal backend (macOS, Apple GPU). Spike: the pieces needed to judge docs/research/macos-metal-backend-scope.md
//! phase A, not a complete backend.
//!
//!  * Memory is `MTLResourceStorageModeShared` (unified memory): `alloc`, `upload` and `download` are memcpys through
//!    `MTLBuffer::contents()` after the queue has drained.
//!  * `wrap_host` imports mmap'd weight files without copying (`newBufferWithBytesNoCopy`), see [`host`].
//!  * Kernels: the batch-1 bf16 GEMV and a tiled GEMM for `Matmul`/`Logits`, plus the dense elementwise / row ops
//!    (`Embedding`, `RmsNorm`, `RmsNormAdd`, `Add`, `QkNorm`, `Rope`, `SiluGlu`, `Argmax`, `KvAppend`), see [`kernels`]
//!    and `dense.rs`, and causal GQA attention (`AttnPrefill`, `AttnDecode`, kernels/metal/attention.metal). The shaders are AOT compiled
//!    from kernels/metal/*.metal by build.rs and embedded as one metallib.
//!  * One `MTLCommandQueue` and one open command buffer + compute encoder per backend ("the stream"). `launch` appends a
//!    dispatch to it; nothing is committed until `synchronize`, a host access to a buffer, or [`MAX_DISPATCHES`]
//!    dispatches. The encoder is `MTLDispatchTypeSerial`: every dispatch starts after the previous one finished and sees
//!    its memory writes, so dependent ops need no barriers (the spike measured barriers at about 3 percent, see
//!    docs/results/2026-10-14-metal-spike.md). Concurrent dispatch plus targeted barriers is a phase B option.
//!  * Host access (`upload`, `download`) first flushes and waits for all pending GPU work, the simplest rule that is
//!    always correct on shared storage. It is a no-op when nothing is pending. `copy` and `zero` are dispatches appended
//!    to the stream (no wait), see [`fill`]; only a `copy` whose ranges overlap inside one buffer takes the host path.
//!
//! Exports no `ik_*` symbols.

#![cfg(target_os = "macos")]

mod attn;
pub mod bench;
mod buffer;
pub mod decode;
mod dense;
mod fill;
pub mod flash;
mod fused;
mod host;
pub mod kernels;
pub mod profile;

pub use buffer::{MetalBuffer, MtlBuffer};
pub use kernels::{GemvVariant, DEFAULT_GEMV, GEMV_VARIANTS};

use dispatch2::DispatchData;
use intarsia_core::backend::{Backend, Buffer, DeviceInfo, HostMemory, MemKind, StreamId};
use intarsia_core::{DType, Error, KernelImpl, Op, OpCall, Result, TensorRef};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineDescriptor, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLDispatchType,
    MTLLibrary, MTLResourceOptions, MTLSize,
};
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, OnceLock};

pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
pub type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
pub type Pso = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type Encoder = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;

/// All kernels of this crate, built by build.rs (empty when the Metal toolchain was missing at build time).
static METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/intarsia.metallib"));

/// FNV-1a 64 of the launcher sources (`LAUNCHER_SOURCES` in build.rs), hex.
pub const LAUNCHER_HASH: &str = env!("INTARSIA_METAL_LAUNCHER_HASH");

/// FNV-1a 64 of the embedded metallib (every shader source change changes it), computed once.
pub fn metallib_hash() -> u64 {
    static HASH: OnceLock<u64> = OnceLock::new();
    *HASH.get_or_init(|| {
        METALLIB.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3))
    })
}
/// Kernel arguments of the GEMV shaders (`GemvParams` in gemv_bf16.metal; 32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GemvParams {
    pub k: u32,
    pub n: u32,
    pub w_skip: u32,
    pub x_skip: u32,
    pub y_skip: u32,
    pub pad: [u32; 3],
}

/// Dispatches appended to one command buffer before it is committed (not waited on) and a fresh one is opened. The GPU
/// starts on a decode step when its first command buffer is committed instead of after the host has encoded all of the
/// step's ~550 dispatches: the idle time between steps fell from 0.51 to 0.29 ms/token (64 measures the same as 128, 256
/// in between, see docs/results/2026-10-14-metal-phase-b.md "Phase B part 2b"). `INTARSIA_METAL_COMMIT_EVERY=<n>`
/// overrides it.
pub const MAX_DISPATCHES: usize = 128;

/// The stream: the open command buffer and encoder plus every committed buffer not yet waited on.
#[derive(Default)]
struct Stream {
    open: Option<(CommandBuffer, Encoder)>,
    dispatches: usize,
    inflight: Vec<CommandBuffer>,
    /// First GPU error seen on a buffer that was pruned from `inflight` before `synchronize` looked at it.
    failed: Option<String>,
    /// `INTARSIA_METAL_PROFILE` trace, see [`profile`].
    prof: Option<profile::Prof>,
    /// Label of the launch being encoded, taken by the next flush when profiling.
    label: Option<profile::Label>,
    /// Dispatches per command buffer before it is committed (`COMMIT_EVERY`, overridable for experiments).
    commit_every: usize,
}

pub struct MetalBackend {
    device: Device,
    queue: Queue,
    psos: HashMap<&'static str, Pso>,
    stream: Mutex<Stream>,
    hosts: Mutex<host::HostCache>,
    page_size: usize,
    /// Copy-kernel bandwidth, measured on the first `info()` call (about 100 ms, 2 GiB of scratch buffers) and
    /// cached. Construction stays cheap; the one-off cost lands on whoever asks first (the telemetry probe).
    bandwidth: OnceLock<Option<f64>>,
}

// SAFETY: Apple documents MTLDevice, MTLCommandQueue, MTLComputePipelineState and MTLBuffer as usable from any thread;
// the non-thread-safe objects we keep (the open command buffer and encoder) are behind a Mutex.
unsafe impl Send for MetalBackend {}
unsafe impl Sync for MetalBackend {}

fn err<E: std::fmt::Debug>(what: &str, e: E) -> Error {
    Error::backend(format!("metal: {what}: {e:?}"))
}

impl MetalBackend {
    pub fn new() -> Result<MetalBackend> {
        let device = MTLCreateSystemDefaultDevice().ok_or_else(|| Error::backend("metal: no system default device"))?;
        let queue = device.newCommandQueue().ok_or_else(|| Error::backend("metal: newCommandQueue failed"))?;
        if METALLIB.is_empty() {
            return Err(Error::backend(
                "metal: the metallib was not built (Xcode Metal toolchain missing at build time; see build.rs)",
            ));
        }
        let data = DispatchData::from_bytes(METALLIB);
        let lib = device.newLibraryWithData_error(&data).map_err(|e| err("newLibraryWithData", e))?;
        let mut psos = HashMap::new();
        for name in kernels::shader_names() {
            let f = lib
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| Error::backend(format!("metal: function '{name}' missing from the metallib")))?;
            let desc = MTLComputePipelineDescriptor::new();
            desc.setComputeFunction(Some(&f));
            // ICB-encodable, so the spike's indirect-command-buffer check can use the same pipelines
            desc.setSupportIndirectCommandBuffers(true);
            let pso = device
                .newComputePipelineStateWithDescriptor_options_reflection_error(
                    &desc,
                    objc2_metal::MTLPipelineOption::None,
                    None,
                )
                .map_err(|e| err(name, e))?;
            psos.insert(name, pso);
        }
        // SAFETY: sysconf has no preconditions.
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        Ok(MetalBackend {
            device,
            queue,
            psos,
            stream: Mutex::new(Stream {
                prof: profile::Prof::from_env(),
                commit_every: std::env::var("INTARSIA_METAL_COMMIT_EVERY")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .filter(|&n| n > 0)
                    .unwrap_or(MAX_DISPATCHES),
                ..Stream::default()
            }),
            hosts: Mutex::new(host::HostCache::default()),
            page_size,
            bandwidth: OnceLock::new(),
        })
    }

    pub fn arc() -> Result<Arc<dyn Backend>> {
        Ok(Arc::new(MetalBackend::new()?))
    }

    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    pub fn queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.queue
    }

    /// Host page size (16 KiB on Apple Silicon); `wrap_host` rounds to it.
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Compute pipeline for a shader entry point (see `kernels::shader_names`).
    pub fn pso(&self, name: &str) -> Option<&Pso> {
        self.psos.get(name)
    }

    /// A fresh shared-storage `MTLBuffer` of `bytes` (at least 1). `untracked` skips Metal's hazard tracking, which
    /// is only safe when the caller orders every access itself (serial encoders, a drained queue).
    pub fn new_mtl_buffer(&self, bytes: usize, untracked: bool) -> Result<MtlBuffer> {
        let mut opts = MTLResourceOptions::StorageModeShared;
        if untracked {
            opts |= MTLResourceOptions::HazardTrackingModeUntracked;
        }
        self.device
            .newBufferWithLength_options(bytes.max(1), opts)
            .ok_or_else(|| Error::Memory(format!("metal: newBuffer of {bytes} bytes failed")))
    }

    /// Commit a command buffer the caller built on [`Self::queue`] itself (the spike's benchmarks do). Flushes the open
    /// stream first so queue order matches call order, then tracks `cb` for `synchronize`.
    pub fn commit(&self, cb: &CommandBuffer) {
        let mut st = self.stream.lock().unwrap();
        Self::flush_locked(&mut st);
        cb.commit();
        Self::track(&mut st, cb.clone());
    }

    /// Wait for `cb` and turn a GPU error into an `Error`.
    pub fn wait(cb: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<()> {
        cb.waitUntilCompleted();
        if cb.status() == MTLCommandBufferStatus::Error {
            return Err(Error::backend(format!("metal: command buffer failed: {:?}", cb.error())));
        }
        Ok(())
    }

    fn track(st: &mut Stream, cb: CommandBuffer) {
        // drop finished buffers so a long run without `synchronize` does not grow the list; keep their first error
        let mut failed = st.failed.take();
        st.inflight.retain(|c| match c.status() {
            MTLCommandBufferStatus::Completed => false,
            MTLCommandBufferStatus::Error => {
                failed.get_or_insert_with(|| format!("metal: command buffer failed: {:?}", c.error()));
                false
            }
            _ => true,
        });
        st.failed = failed;
        st.inflight.push(cb);
    }

    /// End the open encoder and commit its command buffer (not waited on).
    fn flush_locked(st: &mut Stream) {
        if let Some((cb, enc)) = st.open.take() {
            enc.endEncoding();
            cb.commit();
            let label = st.label.take();
            if let Some(p) = st.prof.as_mut() {
                p.note(cb.clone(), label, st.dispatches);
            }
            Self::track(st, cb);
            st.dispatches = 0;
        }
    }

    /// Dispatches encoded into the open command buffer and not yet committed (0 after a flush / synchronize).
    pub fn pending_dispatches(&self) -> usize {
        self.stream.lock().unwrap().dispatches
    }

    /// Commit everything encoded so far without waiting for it.
    pub fn flush(&self) {
        Self::flush_locked(&mut self.stream.lock().unwrap());
    }

    /// Append work to the stream's open serial compute encoder (opening a command buffer and encoder when none is
    /// open). `f` must encode exactly one dispatch.
    pub fn with_encoder(&self, f: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>)) -> Result<()> {
        let mut st = self.stream.lock().unwrap();
        if st.open.is_none() {
            let cb = self.queue.commandBuffer().ok_or_else(|| Error::backend("metal: commandBuffer failed"))?;
            let enc = cb
                .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
                .ok_or_else(|| Error::backend("metal: computeCommandEncoder failed"))?;
            st.open = Some((cb, enc));
        }
        f(&st.open.as_ref().unwrap().1);
        st.dispatches += 1;
        let per_kernel = st.prof.as_ref().is_some_and(|p| p.per_kernel);
        if st.dispatches >= st.commit_every || per_kernel {
            Self::flush_locked(&mut st);
            if per_kernel {
                // run launches one at a time: independent ones (gate / up, k / v) would otherwise overlap across
                // command buffers and their times could not be told apart
                if let Some(cb) = st.inflight.last() {
                    cb.waitUntilCompleted();
                }
            }
        }
        Ok(())
    }

    fn mbuf<'a>(&self, b: &'a Buffer) -> Result<&'a MetalBuffer> {
        b.downcast::<MetalBuffer>().ok_or_else(|| Error::backend("foreign buffer"))
    }

    /// Shared-memory pointer of `[offset, offset+len)` of `b`, bounds-checked.
    fn host_range<'a>(&self, b: &'a Buffer, offset: usize, len: usize) -> Result<(&'a MetalBuffer, *mut u8)> {
        let mb = self.mbuf(b)?;
        if offset.checked_add(len).is_none_or(|e| e > mb.bytes) {
            return Err(Error::backend("metal buffer access out of bounds"));
        }
        // SAFETY: bounds-checked above.
        Ok((mb, unsafe { mb.host_ptr().add(offset) }))
    }

    /// Run `f` with a fresh compute encoder in its own command buffer, commit it, return it (not waited on). A
    /// benchmark / test helper: `Backend::launch` goes through the stream instead.
    pub fn encode_and_commit(
        &self,
        f: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
    ) -> Result<CommandBuffer> {
        let cb = self.queue.commandBuffer().ok_or_else(|| Error::backend("metal: commandBuffer failed"))?;
        let enc = cb.computeCommandEncoder().ok_or_else(|| Error::backend("metal: computeCommandEncoder failed"))?;
        f(&enc);
        enc.endEncoding();
        self.commit(&cb);
        Ok(cb)
    }

    /// Name of the shader `launch("matmul_bf16_gemv", call)` would dispatch (operand alignment and K pick it).
    pub fn gemv_shader_for(&self, call: &OpCall) -> Result<&'static str> {
        Ok(self.plan_gemv(call)?.name)
    }

    /// Decide kernel and bindings for a Matmul call (x `[1,K]`, w `[N,K]`, y `[1,N]`, all bf16).
    fn plan_gemv<'a>(&self, call: &'a OpCall) -> Result<GemvPlan> {
        let (x, w, y) = match (call.inputs.as_slice(), call.outputs.as_slice()) {
            ([x, w], [y]) => (x, w, y),
            _ => return Err(Error::invalid("matmul: expected 2 inputs / 1 output")),
        };
        if x.shape.len() != 2 || w.shape.len() != 2 || y.shape.len() != 2 || x.shape[0] != 1 || y.shape[0] != 1 {
            return Err(Error::invalid("matmul_bf16_gemv: needs x [1,K], w [N,K], y [1,N]"));
        }
        let (k, n) = (w.shape[1], w.shape[0]);
        if x.shape[1] != k || y.shape[1] != n {
            return Err(Error::invalid("matmul_bf16_gemv: shape mismatch"));
        }
        if [x.dtype, w.dtype, y.dtype] != [DType::Bf16; 3] {
            return Err(Error::invalid("matmul_bf16_gemv: bf16 only"));
        }
        if k > u32::MAX as usize || n > u32::MAX as usize {
            return Err(Error::invalid("matmul_bf16_gemv: dimension exceeds u32"));
        }
        let addr = |t: &'a TensorRef| -> Result<(&'a MetalBuffer, usize)> {
            let mb = self.mbuf(&t.buf)?;
            Ok((mb, mb.base + t.offset))
        };
        let (wb, wo) = addr(w)?;
        let (xb, xo) = addr(x)?;
        let (yb, yo) = addr(y)?;
        if yb.read_only {
            return Err(Error::backend("metal: output buffer is read-only (wrapped host memory)"));
        }
        // Kernel choice by what the operands allow. 16-byte vector loads need K % 8 == 0 and 16-byte aligned W and x;
        // 8-byte loads need K % 4 == 0 and 8-byte alignment (a safetensors shard with header length 8 mod 16 puts every
        // weight there); anything else takes the scalar kernel, which binds each buffer at a 4-byte aligned offset and
        // skips the leftover ushort in-kernel. y is written with 2-byte stores in all of them, so it only needs the
        // 4-byte binding trick.
        let mut params = GemvParams { k: k as u32, n: n as u32, y_skip: ((yo & 3) / 2) as u32, ..Default::default() };
        let y_bind = yo & !3;
        let (v, w_bind, x_bind) = if k % 8 == 0 && wo % 16 == 0 && xo % 16 == 0 {
            (DEFAULT_GEMV, wo, xo)
        } else if k % 4 == 0 && wo % 8 == 0 && xo % 8 == 0 {
            (kernels::DEFAULT_GEMV_A8, wo, xo)
        } else {
            params.w_skip = ((wo & 3) / 2) as u32;
            params.x_skip = ((xo & 3) / 2) as u32;
            (kernels::GEMV_SCALAR, wo & !3, xo & !3)
        };
        let (name, groups, threads) = (v.name, (n as u32).div_ceil(v.rows_per_tg()), v.threads_per_tg());
        Ok(GemvPlan {
            name,
            w: (wb.mtl.clone(), w_bind),
            x: (xb.mtl.clone(), x_bind),
            y: (yb.mtl.clone(), y_bind),
            params,
            groups,
            threads,
        })
    }
}

/// A resolved GEMV launch: pipeline, buffer bindings (MTLBuffer, byte offset) and grid.
struct GemvPlan {
    name: &'static str,
    w: (MtlBuffer, usize),
    x: (MtlBuffer, usize),
    y: (MtlBuffer, usize),
    params: GemvParams,
    groups: u32,
    threads: u32,
}

/// Pass `v` to the shader as a small inline constant buffer at `index`.
pub fn set_bytes<T: Copy>(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, v: &T, index: usize) {
    // SAFETY: `v` is a valid, initialised T for the call; Metal copies the bytes.
    unsafe {
        enc.setBytes_length_atIndex(
            NonNull::new_unchecked(v as *const T as *mut c_void),
            std::mem::size_of::<T>(),
            index,
        )
    }
}

/// Encode one GEMV dispatch (`y = W x`) with the shader's kernel arguments passed inline.
pub fn encode_gemv(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pso: &ProtocolObject<dyn MTLComputePipelineState>,
    w: (&ProtocolObject<dyn MTLBuffer>, usize),
    x: (&ProtocolObject<dyn MTLBuffer>, usize),
    y: (&ProtocolObject<dyn MTLBuffer>, usize),
    params: &GemvParams,
    groups: u32,
    threads: u32,
) {
    enc.setComputePipelineState(pso);
    // SAFETY: offsets are inside the buffers (the callers check) and aligned for the kernel's pointer types.
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w.0), w.1, 0);
        enc.setBuffer_offset_atIndex(Some(x.0), x.1, 1);
        enc.setBuffer_offset_atIndex(Some(y.0), y.1, 2);
    }
    set_bytes(enc, params, 3);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize { width: groups as usize, height: 1, depth: 1 },
        MTLSize { width: threads as usize, height: 1, depth: 1 },
    );
}

impl Drop for MetalBackend {
    fn drop(&mut self) {
        // an open encoder must be ended before it is released (Metal aborts otherwise); this also waits for the queue,
        // which writes the last trace records when INTARSIA_METAL_PROFILE is set. A no-op when nothing is pending.
        let _ = self.synchronize(0);
    }
}

impl Backend for MetalBackend {
    fn name(&self) -> &str {
        "metal"
    }

    fn info(&self) -> DeviceInfo {
        let total = self.device.recommendedMaxWorkingSetSize();
        let used = self.device.currentAllocatedSize() as u64;
        let bandwidth = *self.bandwidth.get_or_init(|| bench::measure_copy_gbs(self, 1 << 30, 3, 7).ok());
        DeviceInfo {
            name: self.device.name().to_string(),
            backend: "metal".into(),
            arch: self.device.architecture().name().to_string(),
            total_mem: total,
            free_mem: total.saturating_sub(used),
            unified: true,
            bandwidth_gbs: bandwidth,
        }
    }

    /// backend | device | GPU architecture | hash of the embedded metallib | hash of the launcher sources: anything that
    /// can change a kernel's numerics, code generation, selection or launch geometry invalidates cached admission
    /// results. The launcher hash (build.rs, `LAUNCHER_SOURCES`) covers the Rust that picks the GEMV variant and sets the
    /// grids, which the metallib bytes do not see. The default implementation is not used because it goes through
    /// `info()`, which runs the (100 ms) bandwidth probe. No env knob changes kernel selection or numerics
    /// (`IK_METAL_TRACKED` only toggles hazard tracking, `INTARSIA_METAL_DEBUG` and `INTARSIA_METAL_PROFILE` only
    /// log), so none is folded.
    fn fingerprint(&self) -> String {
        format!(
            "metal|{}|{}|metallib:{:016x}|launch:{}",
            self.device.name(),
            self.device.architecture().name(),
            metallib_hash(),
            LAUNCHER_HASH
        )
    }

    fn alloc(&self, bytes: usize, _kind: MemKind) -> Result<Buffer> {
        let mtl = self.new_mtl_buffer(bytes, false)?;
        Ok(Buffer::new(MetalBuffer { mtl, base: 0, bytes, read_only: false, _host: None }))
    }

    fn upload(&self, dst: &Buffer, dst_offset: usize, src: &[u8], _s: StreamId) -> Result<()> {
        self.synchronize(0)?;
        let (mb, p) = self.host_range(dst, dst_offset, src.len())?;
        if mb.read_only {
            return Err(Error::backend("metal buffer is read-only (wrapped host memory)"));
        }
        // SAFETY: bounds-checked, GPU idle, `src` cannot overlap a buffer the backend allocated.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), p, src.len()) };
        Ok(())
    }

    fn download(&self, src: &Buffer, src_offset: usize, dst: &mut [u8], _s: StreamId) -> Result<()> {
        self.synchronize(0)?;
        let (_, p) = self.host_range(src, src_offset, dst.len())?;
        // SAFETY: bounds-checked, GPU idle.
        unsafe { std::ptr::copy_nonoverlapping(p, dst.as_mut_ptr(), dst.len()) };
        Ok(())
    }

    fn copy(
        &self,
        dst: &Buffer,
        dst_offset: usize,
        src: &Buffer,
        src_offset: usize,
        bytes: usize,
        _s: StreamId,
    ) -> Result<()> {
        if bytes == 0 || self.gpu_copy(dst, dst_offset, src, src_offset, bytes)? {
            return Ok(());
        }
        // overlapping ranges in one buffer: memmove semantics need the host
        self.synchronize(0)?;
        let (_, dp) = self.host_range(dst, dst_offset, bytes)?;
        let (_, sp) = self.host_range(src, src_offset, bytes)?;
        // SAFETY: bounds-checked, GPU idle; `ptr::copy` handles the overlap, `gpu_copy` checked dst is writable.
        unsafe { std::ptr::copy(sp, dp, bytes) };
        Ok(())
    }

    fn wrap_host(&self, mem: Arc<dyn HostMemory>, offset: usize, len: usize) -> Option<Buffer> {
        host::wrap(self, mem, offset, len)
    }

    fn kernels(&self) -> Vec<KernelImpl> {
        kernels::all()
    }

    fn zero(&self, dst: &Buffer, _s: StreamId) -> Result<()> {
        self.gpu_zero(dst)
    }

    fn launch(&self, kernel: &str, call: &OpCall, _s: StreamId) -> Result<()> {
        call.check()?;
        {
            let mut st = self.stream.lock().unwrap();
            if st.prof.is_some() {
                st.label = Some(profile::Label::of(kernel, call));
            }
        }
        if kernel == kernels::MATMUL_BF16_GEMV && call.op == Op::Matmul {
            let plan = self.plan_gemv(call)?;
            let pso =
                self.pso(plan.name).ok_or_else(|| Error::backend(format!("metal: no pipeline '{}'", plan.name)))?;
            return self.with_encoder(|enc| {
                encode_gemv(
                    enc,
                    pso,
                    (&plan.w.0, plan.w.1),
                    (&plan.x.0, plan.x.1),
                    (&plan.y.0, plan.y.1),
                    &plan.params,
                    plan.groups,
                    plan.threads,
                )
            });
        }
        if kernel == kernels::ATTN_PREFILL_BF16_FLASH && call.op == Op::AttnPrefill {
            return self.launch_attn_flash(call);
        }
        if kernel == kernels::MATMUL_PAIR_BF16_GEMV && call.op == Op::MatmulPair {
            return self.launch_matmul_pair(call);
        }
        self.launch_dense(kernel, call)
    }

    /// Commit the open command buffer and wait for every committed one; the first GPU error is returned. The
    /// backend has one stream, so every `StreamId` means the same thing.
    fn synchronize(&self, _s: StreamId) -> Result<()> {
        // hold the lock while waiting: a concurrent launch queues behind this synchronize, not into it
        let mut st = self.stream.lock().unwrap();
        Self::flush_locked(&mut st);
        let pending = std::mem::take(&mut st.inflight);
        let mut first = st.failed.take();
        for cb in &pending {
            if let Err(e) = Self::wait(cb) {
                first.get_or_insert_with(|| e.to_string());
            }
        }
        if let Some(p) = st.prof.as_mut() {
            p.resolve();
        }
        match first {
            Some(e) => Err(Error::backend(e)),
            None => Ok(()),
        }
    }
}
