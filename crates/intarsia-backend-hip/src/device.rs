//! The HIP `Backend` implementation (only compiled with a ROCm toolchain, `cfg(ik_hip)`).
//!
//! Threading: the engine calls a backend from several threads (generation worker, the caller's
//! thread for `forward_logits_each`, the telemetry thread via `info()`). The HIP runtime keeps a
//! *per-thread* current device, so every entry point calls `bind()` (hipSetDevice, a cached no-op
//! when already current) before touching the runtime.

use crate::ffi::{self, IkDeviceInfo, IkTensor, IK_MAX_DIMS};
use crate::kernels;
use intarsia_core::backend::{Backend, Buffer, BufferImpl, DeviceInfo, GraphHandle, GraphImpl, MemKind, StreamId};
use intarsia_core::{DType, Error, KernelImpl, OpCall, Result, TensorRef};
use std::any::Any;
use std::ffi::{c_int, c_void, CStr};
use std::sync::atomic::{AtomicBool, Ordering};
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

/// An instantiated HIP graph (`hipGraphExec_t`), destroyed on drop.
struct HipGraph {
    exec: usize,
    ordinal: i32,
}

impl GraphImpl for HipGraph {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Drop for HipGraph {
    fn drop(&mut self) {
        // SAFETY: exec came from ik_rt_graph_end and is destroyed exactly once.
        unsafe {
            ffi::ik_rt_set_device(self.ordinal);
            ffi::ik_rt_graph_destroy(self.exec as *mut c_void);
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
    /// a graph capture is running on this backend: launches are recorded, not executed, so the trace's stream events
    /// (which would become event nodes of the graph) are skipped
    capturing: AtomicBool,
    /// round 9: page-locked host staging for small copies (created on first use; `None` after a failed allocation)
    staging: Mutex<Option<Option<Staging>>>,
}

/// Round 9 (docs/results/2026-10-19-hip-round9.md): page-locked host staging for small uploads and downloads.
///
/// A pageable `hipMemcpyAsync` host -> device has to be followed by a stream synchronisation (ik_rt_memcpy_h2d), so every
/// small upload (a prefill chunk's token ids, a decode step's id and position) drained the queue: the GPU idled from the
/// end of the queued work until the host, woken from the wait, had launched again. That is one boundary per prefill
/// chunk, the decode-graph capture of a new sequence (~3.5 ms, run while the GPU waits) and every decode step.
///
/// An upload of at most `STAGE_SLOT` bytes is copied into the next of `STAGE_SLOTS` page-locked slots and queued from
/// there without a wait; an event recorded after the copy guards the slot, and a slot is only rewritten once its event has
/// completed (in practice long before: the ring holds 8 copies). The device sees the same bytes in the same stream order,
/// so results are bitwise unchanged. A download of at most `STAGE_SLOT` bytes goes through one more page-locked slot (it
/// waits for the stream either way). Larger copies, copies during a graph capture and INTARSIA_HIP_TRACE runs take the
/// old path, as does everything with `INTARSIA_HIP_STAGE=0` (read at every copy); `INTARSIA_HIP_STAGE=up` stages only
/// uploads. 9 x 64 KiB of page-locked host memory per backend, no device memory.
struct Staging {
    host: usize,
    events: [usize; STAGE_SLOTS],
    used: [bool; STAGE_SLOTS],
    next: usize,
}

const STAGE_SLOTS: usize = 8;
const STAGE_SLOT: usize = 64 << 10;

/// INTARSIA_HIP_STAGE: (stage uploads, stage downloads)
fn stage_mode() -> (bool, bool) {
    match std::env::var("INTARSIA_HIP_STAGE").as_deref() {
        Ok("0") => (false, false),
        Ok("up") => (true, false),
        _ => (true, true),
    }
}

impl Staging {
    fn new() -> Option<Staging> {
        let mut host: *mut c_void = std::ptr::null_mut();
        // SAFETY: valid out-pointer; the device is bound by the caller.
        if unsafe { ffi::ik_rt_host_alloc(&mut host, STAGE_SLOT * (STAGE_SLOTS + 1)) } != 0 || host.is_null() {
            return None;
        }
        let mut events = [0usize; STAGE_SLOTS];
        for e in events.iter_mut() {
            let mut h: *mut c_void = std::ptr::null_mut();
            // SAFETY: valid out-pointer.
            if unsafe { ffi::ik_rt_event_create_sync(&mut h) } != 0 {
                return None; // (leaks what was made; this only happens when the runtime is out of resources)
            }
            *e = h as usize;
        }
        Some(Staging { host: host as usize, events, used: [false; STAGE_SLOTS], next: 0 })
    }

    fn slot(&self, i: usize) -> *mut u8 {
        (self.host as *mut u8).wrapping_add(i * STAGE_SLOT)
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        for (i, &e) in self.events.iter().enumerate() {
            // SAFETY: live events; a pending copy still reads its slot, so wait before the memory goes.
            unsafe {
                if self.used[i] {
                    ffi::ik_rt_event_sync(e as *mut c_void);
                }
                ffi::ik_rt_event_destroy(e as *mut c_void);
            }
        }
        // SAFETY: allocated by ik_rt_host_alloc, no copy reads it any more.
        unsafe { ffi::ik_rt_host_free(self.host as *mut c_void) };
    }
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
    /// Round 9: the gap from one launch's stop event to the next launch's start event, bucketed by how long the earlier
    /// launch ran (< 0.1 ms, 0.1 to 1 ms, >= 1 ms): [bucket] = every gap in ms. Answers whether the ~275 us stall after
    /// a >= 1 ms dispatch that rocprofv3 shows also happens without the profiler.
    gaps: [Vec<f64>; 3],
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
        for w in p.windows(2) {
            if let (Some(d), Some(g)) = (ms(w[0].1, w[0].2), ms(w[0].2, w[1].1)) {
                self.gaps[if d >= 1.0 { 2 } else if d >= 0.1 { 1 } else { 0 }].push(g);
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
        for (i, label) in ["< 0.1 ms", "0.1 to 1 ms", ">= 1 ms"].iter().enumerate() {
            let g = &mut self.gaps[i];
            if g.is_empty() {
                continue;
            }
            g.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q = |f: f64| g[((g.len() - 1) as f64 * f) as usize] * 1e3;
            eprintln!(
                "hip trace gaps after launches of {label}: n {} sum {:.2} ms, us min {:.1} p10 {:.1} median {:.1} p90 {:.1} max {:.1}",
                g.len(),
                g.iter().sum::<f64>(),
                q(0.0),
                q(0.1),
                q(0.5),
                q(0.9),
                q(1.0)
            );
        }
    }
}

impl Drop for HipBackend {
    fn drop(&mut self) {
        let _ = self.bind();
        // SAFETY: plain FFI call; a no-op unless INTARSIA_HIP_STAMP is set.
        unsafe { ffi::ik_rt_stamp_dump() };
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
            capturing: AtomicBool::new(false),
            staging: Mutex::new(None),
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

    /// Test seam (tests/staging.rs): keep `stream` busy for about `us` microseconds (a 1-thread kernel spinning on the wall
    /// clock), so copies queued behind it are still pending when the host moves on.
    pub(crate) fn spin(&self, stream: StreamId, us: u32) -> Result<()> {
        self.bind()?;
        let s = self.stream(stream)?;
        // SAFETY: plain launch on a live stream.
        check("spin", unsafe { ffi::ik_rt_spin(s, us) })
    }

    pub(crate) fn ordinal(&self) -> i32 {
        self.ordinal
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

    pub(crate) fn ptr(buf: &Buffer, offset: usize, len: usize) -> Result<*mut u8> {
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
        if src.len() <= STAGE_SLOT && self.trace.is_none() && !self.capturing.load(Ordering::Relaxed) && stage_mode().0 {
            let mut g = self.staging.lock().unwrap();
            if let Some(st) = g.get_or_insert_with(Staging::new) {
                let i = st.next;
                if st.used[i] {
                    // SAFETY: a live event, recorded after the copy that last read slot i
                    check("staging wait", unsafe { ffi::ik_rt_event_sync(st.events[i] as *mut c_void) })?;
                }
                let slot = st.slot(i);
                // SAFETY: slot i is STAGE_SLOT bytes of page-locked memory no queued copy reads any more (waited above);
                // `d` is bounds-checked; the event is recorded on the copy's stream right after it.
                unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr(), slot, src.len());
                    check("upload", ffi::ik_rt_memcpy_h2d_pinned(d as *mut c_void, slot as *const c_void, src.len(), s))?;
                    check("staging event", ffi::ik_rt_event_record(st.events[i] as *mut c_void, s))?;
                }
                st.used[i] = true;
                st.next = (i + 1) % STAGE_SLOTS;
                return Ok(());
            }
        }
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
        if dst.len() <= STAGE_SLOT && self.trace.is_none() && !self.capturing.load(Ordering::Relaxed) && stage_mode().1 {
            let mut g = self.staging.lock().unwrap();
            if let Some(st) = g.get_or_insert_with(Staging::new) {
                let slot = st.slot(STAGE_SLOTS); // the download slot, after the upload ring
                // SAFETY: the download slot is page-locked and only used here, under the lock; the wrapper waits for the
                // stream before returning, so the bytes are in place when they are read.
                unsafe {
                    check("download", ffi::ik_rt_memcpy_d2h_pinned(slot as *mut c_void, p as *const c_void, dst.len(), s))?;
                    std::ptr::copy_nonoverlapping(slot as *const u8, dst.as_mut_ptr(), dst.len());
                }
                return Ok(());
            }
        }
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
        // INTARSIA_HIP_TRACE: a start event on this stream right before the launch (not while a graph is being captured)
        let trace = self.trace.as_ref().filter(|_| !self.capturing.load(Ordering::Relaxed));
        let t_start = trace.map(|tr| {
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
                "matmul_bf16_gemv_rows" => ffi::ik_matmul_bf16_gemv_rows(i(0), i(1), o, s),
                "logits_bf16_gemv_rows" => ffi::ik_logits_bf16_gemv_rows(i(0), i(1), o, s),
                "kv_append_bf16" => ffi::ik_kv_append_bf16(i(0), a.pos0 as i64, o, pos, s),
                "attn_prefill_bf16" => ffi::ik_attn_prefill_bf16(i(0), i(1), i(2), a.pos0 as i64, a.scale, o, s),
                "attn_decode_bf16" => ffi::ik_attn_decode_bf16(i(0), i(1), i(2), a.pos0 as i64, a.scale, o, pos, s),
                "attn_decode_bf16_splitkv" => {
                    ffi::ik_attn_decode_bf16_splitkv(i(0), i(1), i(2), a.pos0 as i64, a.scale, o, sc, pos, s)
                }
                "qk_norm_rope_kv_bf16" => ffi::ik_qk_norm_rope_kv_bf16(
                    i(0),
                    i(1),
                    i(2),
                    i(3),
                    i(4),
                    a.eps,
                    a.w_offset,
                    a.pos0 as i64,
                    a.theta,
                    a.rotary_dim as i64,
                    o,
                    o.add(1),
                    o.add(2),
                    pos,
                    s,
                ),
                "gate_split_qk_norm_rope_kv_bf16" => ffi::ik_gate_split_qk_norm_rope_kv_bf16(
                    i(0),
                    i(1),
                    i(2),
                    i(3),
                    i(4),
                    a.eps,
                    a.w_offset,
                    a.pos0 as i64,
                    a.theta,
                    a.rotary_dim as i64,
                    o,
                    o.add(1),
                    o.add(2),
                    o.add(3),
                    pos,
                    s,
                ),
                "gdn_conv_delta_bf16" => {
                    ffi::ik_gdn_conv_delta_bf16(i(0), i(1), i(2), i(3), i(4), i(5), a.eps, o, o.add(1), o.add(2), s)
                }
                "attn_prefill_bf16_splitkv" => {
                    ffi::ik_attn_prefill_bf16_splitkv(i(0), i(1), i(2), a.pos0 as i64, a.scale, o, sc, s)
                }
                "attn_prefill_bf16_flash" => ffi::ik_attn_prefill_bf16_flash(i(0), i(1), i(2), a.pos0 as i64, a.scale, o, s),
                "attn_prefill_bf16_flash_v2" => {
                    ffi::ik_attn_prefill_bf16_flash_v2(i(0), i(1), i(2), a.pos0 as i64, a.scale, o, s)
                }
                "silu_glu_bf16" => ffi::ik_silu_glu_bf16(i(0), i(1), o, s),
                "gdn_conv_bf16" => ffi::ik_gdn_conv_bf16(i(0), i(1), o, o.add(1), s),
                "gdn_delta_rule_bf16" => {
                    ffi::ik_gdn_delta_rule_bf16(i(0), i(1), i(2), i(3), i(4), a.eps, o, o.add(1), s)
                }
                "gdn_delta_rule_bf16_v2" => {
                    ffi::ik_gdn_delta_rule_bf16_v2(i(0), i(1), i(2), i(3), i(4), a.eps, o, o.add(1), s)
                }
                "gdn_delta_rule_bf16_v3" => {
                    ffi::ik_gdn_delta_rule_bf16_v3(i(0), i(1), i(2), i(3), i(4), a.eps, o, o.add(1), s)
                }
                "matmul_fp8_gemv" => ffi::ik_matmul_fp8_gemv(i(0), i(1), i(2), o, s),
                "matmul_fp8_gemv_v2" => ffi::ik_matmul_fp8_gemv_v2(i(0), i(1), i(2), o, s),
                "matmul_fp8_gemv_rows" => ffi::ik_matmul_fp8_gemv_rows(i(0), i(1), i(2), o, s),
                "matmul_fp8_mma" => ffi::ik_matmul_fp8_mma(i(0), i(1), i(2), o, s),
                "matmul_fp8_mma_f8" => ffi::ik_matmul_fp8_mma_f8(i(0), i(1), i(2), o, sc, s),
                "matmul_fp8_mma_v2" => ffi::ik_matmul_fp8_mma_v2(i(0), i(1), i(2), o, sc, s),
                "matmul_nvfp4_gemv" => ffi::ik_matmul_nvfp4_gemv(i(0), i(1), i(2), i(3), o, s),
                "matmul_nvfp4_gemv_v2" => ffi::ik_matmul_nvfp4_gemv_v2(i(0), i(1), i(2), i(3), o, s),
                "matmul_nvfp4_gemv_rows" => ffi::ik_matmul_nvfp4_gemv_rows(i(0), i(1), i(2), i(3), o, s),
                "matmul_nvfp4_mma" => ffi::ik_matmul_nvfp4_mma(i(0), i(1), i(2), i(3), o, s),
                "matmul_mxfp4_gemv" => ffi::ik_matmul_mxfp4_gemv(i(0), i(1), i(2), i(3), o, sc, s),
                "matmul_mxfp4_mma" => ffi::ik_matmul_mxfp4_mma(i(0), i(1), i(2), i(3), o, sc, s),
                "matmul_mxfp4_pair_gemv" => ffi::ik_matmul_mxfp4_pair_gemv(i(0), i(1), i(2), i(3), i(4), i(5), o, o.add(1), sc, s),
                "moe_topk_bf16" => ffi::ik_moe_topk_bf16(i(0), a.top_k as c_int, a.norm_topk as c_int, o, o.add(1), s),
                "moe_topk_bf16_v2" => ffi::ik_moe_topk_bf16_v2(i(0), a.top_k as c_int, a.norm_topk as c_int, o, o.add(1), s),
                "moe_experts_bf16" => {
                    ffi::ik_moe_experts_bf16(i(0), i(1), i(2), i(3), i(4), i(5), i(6), i(7), o, sc, s)
                }
                "moe_experts_bf16_v2" => {
                    ffi::ik_moe_experts_bf16_v2(i(0), i(1), i(2), i(3), i(4), i(5), i(6), i(7), o, sc, s)
                }
                "moe_experts_bf16_rows" => {
                    ffi::ik_moe_experts_bf16_rows(i(0), i(1), i(2), i(3), i(4), i(5), i(6), i(7), o, sc, s)
                }
                "moe_experts_bf16_v3" => {
                    ffi::ik_moe_experts_bf16_v3(i(0), i(1), i(2), i(3), i(4), i(5), i(6), i(7), o, sc, s)
                }
                "moe_combine_bf16" => ffi::ik_moe_combine_bf16(i(0), i(1), i(2), o, s),
                // the 17 inputs as one array: `inp` is a contiguous Vec, i(0) its first element
                "moe_block_bf16" => ffi::ik_moe_block_bf16(i(0), a.top_k as c_int, a.norm_topk as c_int, o, sc, s),
                "matmul_pair_bf16_gemv" => ffi::ik_matmul_pair_bf16_gemv(i(0), i(1), i(2), o, o.add(1), s),
                "matmul_fp8_pair_gemv" => ffi::ik_matmul_fp8_pair_gemv(i(0), i(1), i(2), i(3), i(4), o, o.add(1), s),
                "matmul_nvfp4_glu_gemv" => ffi::ik_matmul_nvfp4_glu_gemv(i(0), i(1), i(2), i(3), i(4), i(5), i(6), o, s),
                "gated_rmsnorm_bf16" => ffi::ik_gated_rmsnorm_bf16(i(0), i(1), i(2), a.eps, a.w_offset, o, s),
                "gated_rmsnorm_matmul_fp8_gemv" => {
                    ffi::ik_gated_rmsnorm_matmul_fp8_gemv(i(0), i(1), i(2), a.eps, a.w_offset, i(3), i(4), o, s)
                }
                "rmsnorm_add_matmul_pair_bf16_gemv" => ffi::ik_rmsnorm_add_matmul_pair_bf16_gemv(
                    i(0),
                    i(1),
                    i(2),
                    a.eps,
                    a.w_offset,
                    i(3),
                    i(4),
                    o,
                    o.add(1),
                    o.add(2),
                    o.add(3),
                    s,
                ),
                "rmsnorm_add_matmul_fp8_pair_gemv" => ffi::ik_rmsnorm_add_matmul_fp8_pair_gemv(
                    i(0),
                    i(1),
                    i(2),
                    a.eps,
                    a.w_offset,
                    i(3),
                    i(4),
                    i(5),
                    i(6),
                    o,
                    o.add(1),
                    o.add(2),
                    o.add(3),
                    s,
                ),
                "sigmoid_mul_matmul_fp8_gemv" => ffi::ik_sigmoid_mul_matmul_fp8_gemv(i(0), i(1), i(2), i(3), o, s),
                "shared_down_combine_nvfp4_gemv" => {
                    ffi::ik_shared_down_combine_nvfp4_gemv(i(0), i(1), i(2), i(3), i(4), i(5), o, s)
                }
                "argmax_f32" => ffi::ik_argmax_f32(i(0), o, s),
                "argmax_bf16" => ffi::ik_argmax_bf16(i(0), o, s),
                other => return Err(Error::NoKernel(format!("hip kernel '{other}' has no dispatch entry"))),
            }
        };
        if let (Some(tr), Some(a)) = (trace, t_start) {
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

    /// The fused decode kernels take every dim from the call's tensors, and their admission and launch checks hold at the
    /// 35B tp=2 shard shapes (8 q / 1 kv heads, GDN 8 / 16 heads, MoE and shared widths 256); the halved shapes are
    /// cases in the fused-op tests (lane H1, docs/results/2026-10-21-h1-tp2-hip.md). Opt-in (`INTARSIA_TP_FUSION=1`).
    fn fusion_under_tp(&self) -> bool {
        true
    }

    fn supports_graphs(&self) -> bool {
        true
    }

    /// 16 rows: the `_rows` GEMV / MoE kernels are proven row-exact up to M = 16 (tests/rows_exact.rs, G-B).
    fn batch_row_cap(&self) -> usize {
        crate::kernels::ROWS_MAX
    }

    fn graph_capture(&self, stream: StreamId, body: &mut dyn FnMut() -> Result<()>) -> Result<GraphHandle> {
        self.bind()?;
        let s = self.stream(stream)?;
        // SAFETY: plain FFI call; the capture is always ended below, so the stream never stays in capture mode.
        check("hipStreamBeginCapture", unsafe { ffi::ik_rt_graph_begin(s) })?;
        self.capturing.store(true, Ordering::Relaxed);
        let ran = body();
        self.capturing.store(false, Ordering::Relaxed);
        let mut exec: *mut c_void = std::ptr::null_mut();
        // SAFETY: valid out-pointer; ends the capture even when `body` failed (the recording is then dropped).
        let ended = check("hipStreamEndCapture/hipGraphInstantiate", unsafe { ffi::ik_rt_graph_end(s, &mut exec) });
        // an instantiated graph is owned (and destroyed on drop) even when the body failed
        let graph = (!exec.is_null()).then(|| HipGraph { exec: exec as usize, ordinal: self.ordinal });
        ran?;
        ended?;
        let graph = graph.ok_or_else(|| Error::backend("graph capture produced no executable graph"))?;
        Ok(GraphHandle::new(graph))
    }

    fn graph_replay(&self, g: &GraphHandle, stream: StreamId) -> Result<()> {
        let g = g.downcast::<HipGraph>().ok_or_else(|| Error::backend("graph does not belong to the hip backend"))?;
        self.bind()?;
        let s = self.stream(stream)?;
        // SAFETY: `exec` is a live hipGraphExec_t owned by `g`.
        check("hipGraphLaunch", unsafe { ffi::ik_rt_graph_launch(g.exec as *mut c_void, s) })
    }
}
