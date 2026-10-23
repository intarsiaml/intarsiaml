//! The CUDA `Backend` implementation (only compiled with a CUDA toolkit, `cfg(ik_cuda)`).
//!
//! Threading: the engine calls a backend from several threads (generation worker, the caller's
//! thread for `forward_logits_each`, the telemetry thread via `info()`). The CUDA runtime keeps a
//! *per-thread* current device, so every entry point calls `bind()` (cudaSetDevice, a cached no-op
//! when already current) before touching the runtime.

use crate::ffi::{self, IkDeviceInfo, IkTensor, IK_MAX_DIMS};
use crate::kernels;
use intarsia_core::backend::{Backend, Buffer, BufferImpl, DeviceInfo, GraphHandle, GraphImpl, MemKind, StreamId};
use intarsia_core::{DType, Error, KernelImpl, OpCall, Result, TensorRef};
use std::any::Any;
use std::ffi::{c_void, CStr};
use std::sync::Mutex;

fn cuda_err(what: &str, code: i32) -> Error {
    // SAFETY: cudaGetErrorString returns a static NUL-terminated string.
    let msg = unsafe { CStr::from_ptr(ffi::ik_rt_error_string(code)) }.to_string_lossy().into_owned();
    Error::backend(format!("{what}: {msg} (cudaError {code})"))
}

fn check(what: &str, code: i32) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(cuda_err(what, code))
    }
}

/// A `cudaMalloc` allocation.
pub struct CudaBuffer {
    ptr: *mut c_void,
    bytes: usize,
    ordinal: i32,
}

// SAFETY: a device pointer is just an address; all access goes through the backend, which binds the
// right device first. Freed exactly once in Drop.
unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

impl BufferImpl for CudaBuffer {
    fn bytes(&self) -> usize {
        self.bytes
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        // SAFETY: ptr came from ik_rt_malloc on `ordinal`; errors during teardown are ignored.
        unsafe {
            ffi::ik_rt_set_device(self.ordinal);
            ffi::ik_rt_free(self.ptr);
        }
    }
}

/// An instantiated CUDA graph (`cudaGraphExec_t`), destroyed on drop.
struct CudaGraph {
    exec: usize,
    ordinal: i32,
}

impl GraphImpl for CudaGraph {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        // SAFETY: exec came from ik_rt_graph_end and is destroyed exactly once.
        unsafe {
            ffi::ik_rt_set_device(self.ordinal);
            ffi::ik_rt_graph_destroy(self.exec as *mut c_void);
        }
    }
}

pub struct CudaBackend {
    ordinal: i32,
    name: String,
    arch: String,
    integrated: bool,
    total_at_init: u64,
    bandwidth_gbs: Option<f64>,
    /// `cudaStream_t` handles (as usize), index = StreamId; grown on demand.
    streams: Mutex<Vec<usize>>,
}

impl CudaBackend {
    pub fn new(ordinal: usize) -> Result<CudaBackend> {
        let mut n = 0;
        // SAFETY: plain FFI call with a valid out-pointer.
        check("cudaGetDeviceCount", unsafe { ffi::ik_rt_device_count(&mut n) })?;
        if ordinal >= n as usize {
            return Err(Error::backend(format!("CUDA device {ordinal} requested but only {n} device(s) found")));
        }
        let mut info: IkDeviceInfo = unsafe { std::mem::zeroed() };
        // SAFETY: info is a valid, writable IkDeviceInfo.
        check("device init", unsafe { ffi::ik_rt_init_device(ordinal as i32, &mut info) })?;
        let name = unsafe { CStr::from_ptr(info.name.as_ptr()) }.to_string_lossy().into_owned();
        let arch = format!("sm_{}{}", info.major, info.minor);
        if arch != kernels::ARCH {
            return Err(Error::Unsupported(format!(
                "device {ordinal} ({name}) is {arch} but the kernels were built for {}; rebuild with INTARSIA_CUDA_ARCH={arch}",
                kernels::ARCH
            )));
        }
        let mut bw = 0f64;
        // SAFETY: valid out-pointer.
        let bandwidth_gbs = (unsafe { ffi::ik_rt_measure_bandwidth(&mut bw) } == 0 && bw > 0.0).then_some(bw);
        let be = CudaBackend {
            ordinal: ordinal as i32,
            name,
            arch,
            integrated: info.integrated != 0,
            total_at_init: info.total_bytes,
            bandwidth_gbs,
            streams: Mutex::new(Vec::new()),
        };
        be.stream(0)?; // create the default stream now so failures surface here
        Ok(be)
    }

    pub(crate) fn bind(&self) -> Result<()> {
        // SAFETY: plain FFI call.
        check("cudaSetDevice", unsafe { ffi::ik_rt_set_device(self.ordinal) })
    }

    pub(crate) fn stream(&self, id: StreamId) -> Result<*mut c_void> {
        let mut s = self.streams.lock().unwrap();
        while s.len() <= id as usize {
            let mut h: *mut c_void = std::ptr::null_mut();
            // SAFETY: valid out-pointer; the device is bound by the caller.
            // stream 0 carries the ordinary work; every other stream is the decode lane (R3d): highest priority
            let id_new = s.len();
            check("cudaStreamCreate", unsafe {
                if id_new == 0 {
                    ffi::ik_rt_stream_create(&mut h)
                } else {
                    ffi::ik_rt_stream_create_high_priority(&mut h)
                }
            })?;
            s.push(h as usize);
        }
        Ok(s[id as usize] as *mut c_void)
    }

    pub(crate) fn ptr(buf: &Buffer, offset: usize, len: usize) -> Result<*mut u8> {
        let b =
            buf.downcast::<CudaBuffer>().ok_or_else(|| Error::backend("buffer does not belong to the cuda backend"))?;
        if offset.checked_add(len).is_none_or(|e| e > b.bytes) {
            return Err(Error::backend(format!("access of {len} bytes at {offset} outside a {}-byte buffer", b.bytes)));
        }
        Ok((b.ptr as *mut u8).wrapping_add(offset))
    }

    pub(crate) fn tensor(t: &TensorRef) -> Result<IkTensor> {
        if t.shape.len() > IK_MAX_DIMS {
            return Err(Error::invalid(format!("rank {} exceeds IK_MAX_DIMS", t.shape.len())));
        }
        let data = Self::ptr(&t.buf, t.offset, t.nbytes())? as *mut c_void;
        let mut shape = [1i64; IK_MAX_DIMS];
        let mut stride = [0i64; IK_MAX_DIMS];
        let mut acc = 1i64;
        for i in (0..t.shape.len()).rev() {
            shape[i] = t.shape[i] as i64;
            stride[i] = acc;
            acc *= t.shape[i] as i64;
        }
        Ok(IkTensor { data, dtype: t.dtype as i32, ndim: t.shape.len() as i32, shape, stride })
    }
}

impl Backend for CudaBackend {
    fn name(&self) -> &str {
        "cuda"
    }

    fn info(&self) -> DeviceInfo {
        let _ = self.bind();
        let (mut free, mut total) = (0u64, 0u64);
        // SAFETY: valid out-pointers.
        let rc = unsafe { ffi::ik_rt_mem_info(&mut free, &mut total) };
        if rc != 0 {
            total = self.total_at_init;
            free = 0;
        }
        DeviceInfo {
            name: self.name.clone(),
            backend: "cuda".into(),
            arch: self.arch.clone(),
            total_mem: total,
            free_mem: free,
            unified: self.integrated,
            bandwidth_gbs: self.bandwidth_gbs,
        }
    }

    fn alloc(&self, bytes: usize, _kind: MemKind) -> Result<Buffer> {
        self.bind()?;
        let mut p: *mut c_void = std::ptr::null_mut();
        // SAFETY: valid out-pointer. Zero-byte requests still get a (1 byte) allocation so the
        // pointer is valid and unique.
        let rc = unsafe { ffi::ik_rt_malloc(&mut p, bytes.max(1)) };
        if rc != 0 {
            return Err(Error::Memory(format!("cudaMalloc of {bytes} bytes failed: {}", cuda_err("", rc))));
        }
        Ok(Buffer::new(CudaBuffer { ptr: p, bytes, ordinal: self.ordinal }))
    }

    fn upload(&self, dst: &Buffer, dst_offset: usize, src: &[u8], stream: StreamId) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        self.bind()?;
        let d = Self::ptr(dst, dst_offset, src.len())?;
        let s = self.stream(stream)?;
        // SAFETY: bounds checked above. Copies from pageable host memory return once the source has
        // been staged, so `src` need not outlive the call.
        check("upload", unsafe { ffi::ik_rt_memcpy_h2d(d as *mut c_void, src.as_ptr() as *const c_void, src.len(), s) })
    }

    fn download(&self, src: &Buffer, src_offset: usize, dst: &mut [u8], stream: StreamId) -> Result<()> {
        if dst.is_empty() {
            return Ok(());
        }
        self.bind()?;
        let p = Self::ptr(src, src_offset, dst.len())?;
        let s = self.stream(stream)?;
        // SAFETY: bounds checked; the wrapper synchronises the stream before returning.
        check("download", unsafe {
            ffi::ik_rt_memcpy_d2h(dst.as_mut_ptr() as *mut c_void, p as *const c_void, dst.len(), s)
        })
    }

    fn copy(
        &self,
        dst: &Buffer,
        dst_offset: usize,
        src: &Buffer,
        src_offset: usize,
        bytes: usize,
        stream: StreamId,
    ) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        self.bind()?;
        let d = Self::ptr(dst, dst_offset, bytes)?;
        let s_ptr = Self::ptr(src, src_offset, bytes)?;
        let s = self.stream(stream)?;
        // SAFETY: both ranges bounds-checked.
        check("copy", unsafe { ffi::ik_rt_memcpy_d2d(d as *mut c_void, s_ptr as *const c_void, bytes, s) })
    }

    fn zero(&self, dst: &Buffer, stream: StreamId) -> Result<()> {
        if dst.bytes() == 0 {
            return Ok(());
        }
        self.bind()?;
        let d = Self::ptr(dst, 0, dst.bytes())?;
        let s = self.stream(stream)?;
        // SAFETY: whole-buffer range.
        check("zero", unsafe { ffi::ik_rt_memset(d as *mut c_void, 0, dst.bytes(), s) })
    }

    fn kernels(&self) -> Vec<KernelImpl> {
        let mut v = kernels::all();
        // A/B and debugging aid: INTARSIA_CUDA_DISABLE=name1,name2 hides kernels from the registry, so the
        // plan falls back to the next candidate (e.g. the SIMT attention kernels instead of flash / split-KV)
        if let Ok(list) = std::env::var("INTARSIA_CUDA_DISABLE") {
            let off: Vec<&str> = list.split(',').map(str::trim).filter(|x| !x.is_empty()).collect();
            v.retain(|k| !off.contains(&k.name.as_str()));
            // a row-table twin goes with the kernel it twins: it must never run where the solo step would run another kernel
            v.retain(|k| !(k.row_table && k.name.strip_suffix("_rows").is_some_and(|base| off.contains(&base))));
        }
        v
    }

    fn supports_row_tables(&self) -> bool {
        true
    }

    fn supports_topk(&self) -> bool {
        true
    }

    fn topk_rows(
        &self,
        logits: &TensorRef,
        k: usize,
        ids: &TensorRef,
        vals: &TensorRef,
        nan: &TensorRef,
        scratch: &TensorRef,
        stream: StreamId,
    ) -> Result<()> {
        self.bind()?;
        let s = self.stream(stream)?;
        let (l, i, v, n) = (Self::tensor(logits)?, Self::tensor(ids)?, Self::tensor(vals)?, Self::tensor(nan)?);
        let scr = ffi::IkScratch { data: Self::ptr(&scratch.buf, scratch.offset, scratch.nbytes())? as *mut c_void, bytes: scratch.nbytes() as i64 };
        let (mut i, mut v, mut n) = (i, v, n);
        // SAFETY: tensors are bounds-checked views of live device buffers; the launcher validates shapes and the scratch size.
        let st = unsafe { ffi::ik_topk_f32(&l, k as i64, &mut i, &mut v, &mut n, &scr, s) };
        if st != 0 {
            return Err(Error::backend(format!("topk_f32: {}", ffi::status_name(st))));
        }
        Ok(())
    }

    fn device_address(&self, buf: &Buffer) -> Option<u64> {
        Self::ptr(buf, 0, 0).ok().map(|p| p as u64)
    }

    fn launch(&self, kernel: &str, call: &OpCall, stream: StreamId) -> Result<()> {
        call.check()?;
        // quantised-weight and MoE launchers (device_quantmoe.rs)
        if let Some(r) = self.launch_quantmoe(kernel, call, stream) {
            return r;
        }
        if let Some(r) = self.launch_exl3(kernel, call, stream) {
            return r;
        }
        match kernels::TABLE.iter().find(|(n, _)| *n == kernel) {
            None => return Err(Error::NoKernel(format!("cuda kernel '{kernel}'"))),
            Some((_, op)) if *op != call.op => {
                return Err(Error::invalid(format!("kernel '{kernel}' implements {op}, not {}", call.op)))
            }
            _ => {}
        }
        self.bind()?;
        let s = self.stream(stream)?;
        let inp: Vec<IkTensor> = call.inputs.iter().map(Self::tensor).collect::<Result<_>>()?;
        let mut out: Vec<IkTensor> = call.outputs.iter().map(Self::tensor).collect::<Result<_>>()?;
        let a = &call.attrs;
        let i = |n: usize| &inp[n] as *const IkTensor;
        let o = out.as_mut_ptr();
        // pre-allocated workspace (plan-reserved; kernels that declared a scratch size read it)
        let scr: Option<ffi::IkScratch> = match &call.scratch {
            Some(t) => Some(ffi::IkScratch { data: Self::ptr(&t.buf, t.offset, t.nbytes())? as *mut c_void, bytes: t.nbytes() as i64 }),
            None => None,
        };
        let sc = scr.as_ref().map_or(std::ptr::null(), |x| x as *const ffi::IkScratch);
        // optional device-side position (rope / kv_append / attn_decode); null = the host `pos0`
        let pos_t: Option<IkTensor> = call.pos.as_ref().map(Self::tensor).transpose()?;
        let pos = pos_t.as_ref().map_or(std::ptr::null(), |t| t as *const IkTensor);
        // row table (batched decode, `OpCall::rows`): the device addresses of the rows' state buffers; null for an ordinary call,
        // which the row-table launchers refuse
        let rows_t: Option<(IkTensor, i64, i64, i64)> = call
            .rows
            .as_ref()
            .map(|r| Self::tensor(&r.ptrs).map(|t| (t, r.stride as i64, r.index as i64, r.byte_off as i64)))
            .transpose()?;
        let tab = rows_t.as_ref().map_or(std::ptr::null(), |t| &t.0 as *const IkTensor);
        let (tab_stride, tab_index, tab_off) = rows_t.as_ref().map_or((0, 0, 0), |t| (t.1, t.2, t.3));
        // SAFETY: tensors are bounds-checked views of live device buffers (held by `call`); the
        // launchers validate dtype/rank/stride and return an error status instead of faulting, and
        // never synchronise or allocate.
        let rc = unsafe {
            match kernel {
                "embedding_bf16" => ffi::ik_embedding_bf16(i(0), i(1), o, s),
                "rmsnorm_bf16" => ffi::ik_rmsnorm_bf16(i(0), i(1), a.eps, a.w_offset, o, s),
                "rmsnorm_add_bf16" => ffi::ik_rmsnorm_add_bf16(i(0), i(1), i(2), a.eps, a.w_offset, o, o.add(1), s),
                "add_bf16" => ffi::ik_add_bf16(i(0), i(1), o, s),
                "qk_norm_bf16" => ffi::ik_qk_norm_bf16(i(0), i(1), a.eps, a.w_offset, o, s),
                "rope_bf16" => ffi::ik_rope_bf16(i(0), a.pos0 as i64, a.theta, a.rotary_dim as i64, o, pos, s),
                "gate_split_bf16" => ffi::ik_gate_split_bf16(i(0), o, o.add(1), s),
                "sigmoid_mul_bf16" => ffi::ik_sigmoid_mul_bf16(i(0), i(1), o, s),
                "gdn_conv_bf16" => ffi::ik_gdn_conv_bf16(i(0), i(1), o, o.add(1), s),
                "gdn_delta_rule_bf16" => {
                    ffi::ik_gdn_delta_rule_bf16(i(0), i(1), i(2), i(3), i(4), a.eps, o, o.add(1), s)
                }
                "gdn_delta_rule_bf16_lanes" => {
                    ffi::ik_gdn_delta_rule_bf16_lanes(i(0), i(1), i(2), i(3), i(4), a.eps, o, o.add(1), s)
                }
                "gdn_delta_rule_bf16_split" => {
                    ffi::ik_gdn_delta_rule_bf16_split(i(0), i(1), i(2), i(3), i(4), a.eps, o, o.add(1), s)
                }
                "gdn_conv_bf16_rows" => ffi::ik_gdn_conv_bf16_rows(i(0), i(1), o, o.add(1), tab, tab_stride, tab_index, s),
                "gdn_delta_rule_bf16_split_rows" => ffi::ik_gdn_delta_rule_bf16_split_rows(
                    i(0), i(1), i(2), i(3), i(4), a.eps, o, o.add(1), tab, tab_stride, tab_index, tab_off, s,
                ),
                "kv_append_bf16_rows" => ffi::ik_kv_append_bf16_rows(i(0), o, tab, tab_stride, tab_index, pos, s),
                "attn_decode_bf16_splitkv_rows" => {
                    ffi::ik_attn_decode_bf16_splitkv_rows(i(0), i(1), i(2), a.scale, o, sc, tab, tab_stride, tab_index, pos, s)
                }
                "matmul_bf16" => ffi::ik_matmul_bf16(i(0), i(1), o, s),
                "matmul_bf16_gemv" => ffi::ik_matmul_bf16_gemv(i(0), i(1), o, s),
                "matmul_bf16_gemv_rows" => ffi::ik_matmul_bf16_gemv_rows(i(0), i(1), o, s),
                "logits_bf16" => ffi::ik_logits_bf16(i(0), i(1), o, s),
                "logits_bf16_gemv" => ffi::ik_logits_bf16_gemv(i(0), i(1), o, s),
                "logits_bf16_gemv_rows" => ffi::ik_logits_bf16_gemv_rows(i(0), i(1), o, s),
                "kv_append_bf16" => ffi::ik_kv_append_bf16(i(0), a.pos0 as i64, o, pos, s),
                "attn_prefill_bf16" => ffi::ik_attn_prefill_bf16(i(0), i(1), i(2), a.pos0 as i64, a.scale, o, s),
                "attn_decode_bf16" => ffi::ik_attn_decode_bf16(i(0), i(1), i(2), a.pos0 as i64, a.scale, o, pos, s),
                "attn_decode_bf16_splitkv" => {
                    ffi::ik_attn_decode_bf16_splitkv(i(0), i(1), i(2), a.pos0 as i64, a.scale, o, sc, pos, s)
                }
                "attn_prefill_bf16_splitkv" => {
                    ffi::ik_attn_prefill_bf16_splitkv(i(0), i(1), i(2), a.pos0 as i64, a.scale, o, sc, s)
                }
                "attn_prefill_bf16_flash" => ffi::ik_attn_prefill_bf16_flash(i(0), i(1), i(2), a.pos0 as i64, a.scale, o, s),
                "attn_decode_f8kv_splitkv" => {
                    ffi::ik_attn_decode_f8kv_splitkv(i(0), i(1), i(2), a.pos0 as i64, a.scale, o, sc, pos, s)
                }
                "attn_prefill_f8kv_splitkv" => {
                    ffi::ik_attn_prefill_f8kv_splitkv(i(0), i(1), i(2), a.pos0 as i64, a.scale, o, sc, s)
                }
                "attn_prefill_f8kv_flash" => ffi::ik_attn_prefill_f8kv_flash(i(0), i(1), i(2), a.pos0 as i64, a.scale, o, s),
                "silu_glu_bf16" => ffi::ik_silu_glu_bf16(i(0), i(1), o, s),
                "gelu_glu_bf16" => ffi::ik_gelu_glu_bf16(i(0), i(1), o, s),
                // registered `.ext()`, but the registry also offers it to a plain call (admission): that gets the plain rope
                "rope_prop_bf16" if a.rope_prop => {
                    ffi::ik_rope_prop_bf16(i(0), a.pos0 as i64, a.theta, a.rotary_dim as i64, o, pos, s)
                }
                "rope_prop_bf16" => ffi::ik_rope_bf16(i(0), a.pos0 as i64, a.theta, a.rotary_dim as i64, o, pos, s),
                "add_scaled_bf16" => ffi::ik_add_scaled_bf16(i(0), i(1), o, a.scale, s),
                "embedding_scaled_bf16" => ffi::ik_embedding_scaled_bf16(i(0), i(1), o, a.scale, s),
                "attn_prefill_bf16_ext" => {
                    ffi::ik_attn_prefill_bf16_ext(i(0), i(1), i(2), a.pos0 as i64, a.scale, a.window as i64, o, s)
                }
                "attn_decode_bf16_splitkv_ext" => ffi::ik_attn_decode_bf16_splitkv_ext(
                    i(0),
                    i(1),
                    i(2),
                    a.pos0 as i64,
                    a.scale,
                    a.window as i64,
                    o,
                    sc,
                    pos,
                    s,
                ),
                "attn_prefill_bf16_splitkv_ext" => {
                    ffi::ik_attn_prefill_bf16_splitkv_ext(i(0), i(1), i(2), a.pos0 as i64, a.scale, a.window as i64, o, sc, s)
                }
                "attn_decode_bf16_ext" => {
                    ffi::ik_attn_decode_bf16_ext(i(0), i(1), i(2), a.pos0 as i64, a.scale, a.window as i64, o, pos, s)
                }
                "argmax_f32" => ffi::ik_argmax_f32(i(0), o, s),
                "argmax_bf16" => ffi::ik_argmax_bf16(i(0), o, s),
                other => return Err(Error::NoKernel(format!("cuda kernel '{other}' has no dispatch entry"))),
            }
        };
        if rc != 0 {
            let dts: Vec<DType> = call.inputs.iter().map(|t| t.dtype).collect();
            return Err(Error::backend(format!(
                "ik_{kernel} returned {} ({rc}); inputs {dts:?} shapes {:?}",
                ffi::status_name(rc),
                call.inputs.iter().map(|t| t.shape.clone()).collect::<Vec<_>>()
            )));
        }
        Ok(())
    }

    fn synchronize(&self, stream: StreamId) -> Result<()> {
        self.bind()?;
        let s = self.stream(stream)?;
        // SAFETY: plain FFI call; also reports asynchronous kernel failures.
        check("stream synchronize", unsafe { ffi::ik_rt_stream_sync(s) })
    }

    fn supports_graphs(&self) -> bool {
        true
    }

    /// 16 rows: the `_rows` GEMV / MoE kernels are proven row-exact up to M = 16 (tests/rows_exact.rs, G-B).
    fn batch_row_cap(&self) -> usize {
        16
    }

    /// The fused decode kernels take head counts and widths from the call (checked on the 35B tp=2 shard shapes by
    /// admission, docs/results/2026-10-21-tp2-cuda-graphs.md).
    fn fusion_under_tp(&self) -> bool {
        true
    }

    fn graph_capture(&self, stream: StreamId, body: &mut dyn FnMut() -> Result<()>) -> Result<GraphHandle> {
        self.bind()?;
        let s = self.stream(stream)?;
        // SAFETY: plain FFI call; the capture is always ended below, so the stream never stays in capture mode.
        check("cudaStreamBeginCapture", unsafe { ffi::ik_rt_graph_begin(s) })?;
        let ran = body();
        let mut exec: *mut c_void = std::ptr::null_mut();
        // SAFETY: valid out-pointer; ends the capture even when `body` failed (the recording is then dropped).
        let ended = check("cudaStreamEndCapture/cudaGraphInstantiate", unsafe { ffi::ik_rt_graph_end(s, &mut exec) });
        ran?;
        ended?;
        Ok(GraphHandle::new(CudaGraph { exec: exec as usize, ordinal: self.ordinal }))
    }

    fn graph_replay(&self, g: &GraphHandle, stream: StreamId) -> Result<()> {
        let g = g.downcast::<CudaGraph>().ok_or_else(|| Error::backend("graph does not belong to the cuda backend"))?;
        self.bind()?;
        let s = self.stream(stream)?;
        // SAFETY: `exec` is a live cudaGraphExec_t owned by `g`.
        check("cudaGraphLaunch", unsafe { ffi::ik_rt_graph_launch(g.exec as *mut c_void, s) })
    }
}
