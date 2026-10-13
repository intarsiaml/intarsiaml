//! The HIP `Backend` implementation (only compiled with a ROCm toolchain, `cfg(ik_hip)`).
//!
//! Threading: the engine calls a backend from several threads (generation worker, the caller's
//! thread for `forward_logits_each`, the telemetry thread via `info()`). The HIP runtime keeps a
//! *per-thread* current device, so every entry point calls `bind()` (hipSetDevice, a cached no-op
//! when already current) before touching the runtime.

use crate::ffi::{self, IkDeviceInfo, IkTensor, IK_MAX_DIMS};
use crate::kernels;
use intarsia_core::backend::{Backend, Buffer, BufferImpl, DeviceInfo, MemKind, StreamId};
use intarsia_core::{DType, Error, KernelImpl, OpCall, Result, TensorRef};
use std::any::Any;
use std::ffi::{c_int, c_void, CStr};
use std::sync::Mutex;

fn hip_err(what: &str, code: i32) -> Error {
    // SAFETY: hipGetErrorString returns a static NUL-terminated string.
    let msg = unsafe { CStr::from_ptr(ffi::ik_rt_error_string(code)) }.to_string_lossy().into_owned();
    Error::backend(format!("{what}: {msg} (hipError {code})"))
}

fn check(what: &str, code: i32) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(hip_err(what, code))
    }
}

/// A `hipMalloc` allocation.
pub struct HipBuffer {
    ptr: *mut c_void,
    bytes: usize,
    ordinal: i32,
}

// SAFETY: a device pointer is just an address; all access goes through the backend, which binds the
// right device first. Freed exactly once in Drop.
unsafe impl Send for HipBuffer {}
unsafe impl Sync for HipBuffer {}

impl BufferImpl for HipBuffer {
    fn bytes(&self) -> usize {
        self.bytes
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Drop for HipBuffer {
    fn drop(&mut self) {
        // SAFETY: ptr came from ik_rt_malloc on `ordinal`; errors during teardown are ignored.
        unsafe {
            ffi::ik_rt_set_device(self.ordinal);
            ffi::ik_rt_free(self.ptr);
        }
    }
}

pub struct HipBackend {
    ordinal: i32,
    name: String,
    arch: String,
    integrated: bool,
    total_at_init: u64,
    bandwidth_gbs: Option<f64>,
    /// hipRuntimeGetVersion / hipDriverGetVersion, part of the admission-cache fingerprint
    runtime_version: i32,
    driver_version: i32,
    /// `hipStream_t` handles (as usize), index = StreamId; grown on demand.
    streams: Mutex<Vec<usize>>,
    /// per-kernel GPU-time trace (INTARSIA_HIP_TRACE=1: start/stop event pair per launch), printed to stderr when the
    /// backend is dropped
    trace: Option<Mutex<Trace>>,
}

/// Two stream events per launch (one recorded just before it, one just after), kept per stream. The elapsed time
/// between a launch's own two events is that kernel's GPU time (event overhead included, about a microsecond); the time
/// between the first start and the last stop of a batch (everything launched since the stream was last synchronised)
/// minus the summed kernel times is stream idle time, i.e. the gaps between launches (host dispatch, bubbles).
#[derive(Default)]
struct Trace {
    /// launches per stream since its last synchronisation: (kernel, start event, stop event)
    pending: std::collections::HashMap<StreamId, Vec<(String, usize, usize)>>,
    pool: Vec<usize>,
    totals: std::collections::HashMap<String, (f64, u64)>,
    batches: u64,
    wall_ms: f64,
}

impl Trace {
    fn event(&mut self) -> usize {
        if let Some(e) = self.pool.pop() {
            return e;
        }
        let mut h: *mut c_void = std::ptr::null_mut();
        // SAFETY: valid out-pointer.
        unsafe { ffi::ik_rt_event_create(&mut h) };
        h as usize
    }

    fn drain(&mut self, stream: StreamId) {
        let p = self.pending.remove(&stream).unwrap_or_default();
        let ms = |a: usize, b: usize| {
            let mut ms = 0f32;
            // SAFETY: both events were recorded on a stream that has since been synchronised.
            (unsafe { ffi::ik_rt_event_elapsed_ms(&mut ms, a as *mut c_void, b as *mut c_void) } == 0).then_some(ms as f64)
        };
        for (name, a, b) in &p {
            if let Some(t) = ms(*a, *b) {
                let e = self.totals.entry(name.clone()).or_insert((0.0, 0));
                e.0 += t;
                e.1 += 1;
            }
        }
        if let (Some(first), Some(last)) = (p.first(), p.last()) {
            if let Some(w) = ms(first.1, last.2) {
                self.batches += 1;
                self.wall_ms += w;
            }
        }
        for (_, a, b) in p {
            self.pool.push(a);
            self.pool.push(b);
        }
    }

    fn report(&mut self) {
        let mut rows: Vec<(&String, &(f64, u64))> = self.totals.iter().collect();
        rows.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
        let total: f64 = rows.iter().map(|r| r.1 .0).sum();
        let launches: u64 = rows.iter().map(|r| r.1 .1).sum();
        eprintln!(
            "hip trace: {} batches, {launches} launches, kernel GPU time {total:.1} ms, stream wall time {:.1} ms, idle between launches {:.1} ms ({:.0}% of wall)",
            self.batches,
            self.wall_ms,
            (self.wall_ms - total).max(0.0),
            100.0 * (self.wall_ms - total).max(0.0) / self.wall_ms.max(1e-9)
        );
        eprintln!("{:<30} {:>8} {:>10} {:>9} {:>6}", "kernel", "launches", "total ms", "avg us", "% kern");
        for (name, (ms, n)) in rows {
            eprintln!("{:<30} {:>8} {:>10.2} {:>9.1} {:>5.1}%", name, n, ms, ms * 1e3 / *n as f64, 100.0 * ms / total.max(1e-9));
        }
    }
}

impl Drop for HipBackend {
    fn drop(&mut self) {
        if let Some(t) = &self.trace {
            let mut t = t.lock().unwrap();
            let _ = self.bind();
            let streams: Vec<StreamId> = t.pending.keys().copied().collect();
            for st in streams {
                t.drain(st);
            }
            t.report();
        }
    }
}

impl HipBackend {
    pub fn new(ordinal: usize) -> Result<HipBackend> {
        let mut n = 0;
        // SAFETY: plain FFI call with a valid out-pointer.
        check("hipGetDeviceCount", unsafe { ffi::ik_rt_device_count(&mut n) })?;
        if ordinal >= n as usize {
            return Err(Error::backend(format!("HIP device {ordinal} requested but only {n} device(s) found")));
        }
        let mut info: IkDeviceInfo = unsafe { std::mem::zeroed() };
        // SAFETY: info is a valid, writable IkDeviceInfo.
        check("device init", unsafe { ffi::ik_rt_init_device(ordinal as i32, &mut info) })?;
        let name = unsafe { CStr::from_ptr(info.name.as_ptr()) }.to_string_lossy().into_owned();
        let arch = unsafe { CStr::from_ptr(info.arch.as_ptr()) }.to_string_lossy().into_owned();
        if arch != kernels::ARCH {
            return Err(Error::Unsupported(format!(
                "device {ordinal} ({name}) is {arch} but the kernels were built for {}; rebuild with INTARSIA_HIP_ARCH={arch} (the WMMA kernels need gfx12)",
                kernels::ARCH
            )));
        }
        if info.warp_size != 32 {
            return Err(Error::Unsupported(format!("device {ordinal} ({name}) has wavefront size {}, the kernels need wave32", info.warp_size)));
        }
        let mut bw = 0f64;
        // SAFETY: valid out-pointer.
        let bandwidth_gbs = (unsafe { ffi::ik_rt_measure_bandwidth(&mut bw) } == 0 && bw > 0.0).then_some(bw);
        let be = HipBackend {
            ordinal: ordinal as i32,
            name,
            arch,
            integrated: info.integrated != 0,
            total_at_init: info.total_bytes,
            bandwidth_gbs,
            runtime_version: info.runtime_version,
            driver_version: info.driver_version,
            streams: Mutex::new(Vec::new()),
            trace: std::env::var_os("INTARSIA_HIP_TRACE").filter(|v| v != "0").map(|_| Mutex::new(Trace::default())),
        };
        be.stream(0)?; // create the default stream now so failures surface here
        Ok(be)
    }

    /// INTARSIA_HIP_TRACE: the stream is idle-synchronised (a synchronize, or a copy whose wrapper synchronises it), so
    /// every event recorded so far has completed and the batch can be read out. Without this a decode run is one
    /// big batch whose wall time includes the host's gaps between tokens.
    fn trace_drain(&self, stream: StreamId) {
        if let Some(tr) = &self.trace {
            tr.lock().unwrap().drain(stream);
        }
    }

    pub(crate) fn bind(&self) -> Result<()> {
        // SAFETY: plain FFI call.
        check("hipSetDevice", unsafe { ffi::ik_rt_set_device(self.ordinal) })
    }

    pub(crate) fn stream(&self, id: StreamId) -> Result<*mut c_void> {
        let mut s = self.streams.lock().unwrap();
        while s.len() <= id as usize {
            let mut h: *mut c_void = std::ptr::null_mut();
            // SAFETY: valid out-pointer; the device is bound by the caller.
            check("hipStreamCreate", unsafe { ffi::ik_rt_stream_create(&mut h) })?;
            s.push(h as usize);
        }
        Ok(s[id as usize] as *mut c_void)
    }

    fn ptr(buf: &Buffer, offset: usize, len: usize) -> Result<*mut u8> {
        let b =
            buf.downcast::<HipBuffer>().ok_or_else(|| Error::backend("buffer does not belong to the hip backend"))?;
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

impl Backend for HipBackend {
    fn name(&self) -> &str {
        "hip"
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
            backend: "hip".into(),
            arch: self.arch.clone(),
            total_mem: total,
            free_mem: free,
            unified: self.integrated,
            bandwidth_gbs: self.bandwidth_gbs,
        }
    }

    /// backend | device | gfx arch | HIP runtime and driver versions | kernel-pack arch | compiler | env knobs: anything
    /// that can change a kernel's numerics or its code generation invalidates cached admission results.
    fn fingerprint(&self) -> String {
        format!(
            "hip|{}|{}|rt{}|drv{}|pack:{}|cc:{}|env:{}",
            self.name,
            self.arch,
            self.runtime_version,
            self.driver_version,
            kernels::ARCH,
            kernels::COMPILER,
            kernels::env_knobs()
        )
    }

    fn alloc(&self, bytes: usize, _kind: MemKind) -> Result<Buffer> {
        self.bind()?;
        let mut p: *mut c_void = std::ptr::null_mut();
        // SAFETY: valid out-pointer. Zero-byte requests still get a (1 byte) allocation so the
        // pointer is valid and unique.
        let rc = unsafe { ffi::ik_rt_malloc(&mut p, bytes.max(1)) };
        if rc != 0 {
            return Err(Error::Memory(format!("hipMalloc of {bytes} bytes failed: {}", hip_err("", rc))));
        }
        Ok(Buffer::new(HipBuffer { ptr: p, bytes, ordinal: self.ordinal }))
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
        let r = check("upload", unsafe { ffi::ik_rt_memcpy_h2d(d as *mut c_void, src.as_ptr() as *const c_void, src.len(), s) });
        self.trace_drain(stream); // the wrapper synchronises the stream
        r
    }

    fn download(&self, src: &Buffer, src_offset: usize, dst: &mut [u8], stream: StreamId) -> Result<()> {
        if dst.is_empty() {
            return Ok(());
        }
        self.bind()?;
        let p = Self::ptr(src, src_offset, dst.len())?;
        let s = self.stream(stream)?;
        // SAFETY: bounds checked; the wrapper synchronises the stream before returning.
        let r = check("download", unsafe {
            ffi::ik_rt_memcpy_d2h(dst.as_mut_ptr() as *mut c_void, p as *const c_void, dst.len(), s)
        });
        self.trace_drain(stream);
        r
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
        // A/B and debugging aid: INTARSIA_HIP_DISABLE=name1,name2 hides kernels from the registry, so the
        // plan falls back to the next candidate (e.g. the SIMT attention kernels instead of flash / split-KV)
        if let Ok(list) = std::env::var("INTARSIA_HIP_DISABLE") {
            let off: Vec<&str> = list.split(',').map(str::trim).filter(|x| !x.is_empty()).collect();
            v.retain(|k| !off.contains(&k.name.as_str()));
        }
        v
    }

    fn launch(&self, kernel: &str, call: &OpCall, stream: StreamId) -> Result<()> {
        call.check()?;
        match kernels::full_table().iter().find(|(n, _)| *n == kernel) {
            None => return Err(Error::NoKernel(format!("hip kernel '{kernel}'"))),
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
        // optional device-side position (rope / kv_append / attn_decode); null = the host `pos0`. The kernel reads the
        // i32 element when it runs, so the tensor view (and its device buffer) must outlive the launch: `pos_t` does.
        let pos_t: Option<IkTensor> = call.pos.as_ref().map(Self::tensor).transpose()?;
        let pos = pos_t.as_ref().map_or(std::ptr::null(), |t| t as *const IkTensor);
        // INTARSIA_HIP_TRACE: a start event on this stream right before the launch
        let t_start = self.trace.as_ref().map(|tr| {
            let mut tr = tr.lock().unwrap();
            let ev = tr.event();
            // SAFETY: ev is a live event, s the launch stream.
            unsafe { ffi::ik_rt_event_record(ev as *mut c_void, s) };
            ev
        });
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
                "matmul_bf16" => ffi::ik_matmul_bf16(i(0), i(1), o, s),
                "matmul_bf16_gemv" => ffi::ik_matmul_bf16_gemv(i(0), i(1), o, s),
                "logits_bf16" => ffi::ik_logits_bf16(i(0), i(1), o, s),
                "logits_bf16_gemv" => ffi::ik_logits_bf16_gemv(i(0), i(1), o, s),
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
                "silu_glu_bf16" => ffi::ik_silu_glu_bf16(i(0), i(1), o, s),
                "gdn_conv_bf16" => ffi::ik_gdn_conv_bf16(i(0), i(1), o, o.add(1), s),
                "gdn_delta_rule_bf16" => {
                    ffi::ik_gdn_delta_rule_bf16(i(0), i(1), i(2), i(3), i(4), a.eps, o, o.add(1), s)
                }
                "matmul_fp8_gemv" => ffi::ik_matmul_fp8_gemv(i(0), i(1), i(2), o, s),
                "matmul_fp8_mma" => ffi::ik_matmul_fp8_mma(i(0), i(1), i(2), o, s),
                "matmul_nvfp4_gemv" => ffi::ik_matmul_nvfp4_gemv(i(0), i(1), i(2), i(3), o, s),
                "matmul_nvfp4_mma" => ffi::ik_matmul_nvfp4_mma(i(0), i(1), i(2), i(3), o, s),
                "moe_topk_bf16" => ffi::ik_moe_topk_bf16(i(0), a.top_k as c_int, a.norm_topk as c_int, o, o.add(1), s),
                "moe_experts_bf16" => {
                    ffi::ik_moe_experts_bf16(i(0), i(1), i(2), i(3), i(4), i(5), i(6), i(7), o, sc, s)
                }
                "moe_combine_bf16" => ffi::ik_moe_combine_bf16(i(0), i(1), i(2), o, s),
                "argmax_f32" => ffi::ik_argmax_f32(i(0), o, s),
                "argmax_bf16" => ffi::ik_argmax_bf16(i(0), o, s),
                other => return Err(Error::NoKernel(format!("hip kernel '{other}' has no dispatch entry"))),
            }
        };
        if let (Some(tr), Some(a)) = (&self.trace, t_start) {
            let mut tr = tr.lock().unwrap();
            if rc == 0 {
                let b = tr.event();
                // SAFETY: b is a live event, s the launch stream.
                unsafe { ffi::ik_rt_event_record(b as *mut c_void, s) };
                tr.pending.entry(stream).or_default().push((kernel.to_string(), a, b));
            } else {
                tr.pool.push(a);
            }
        }
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
        let r = check("stream synchronize", unsafe { ffi::ik_rt_stream_sync(s) });
        self.trace_drain(stream);
        r
    }
}
