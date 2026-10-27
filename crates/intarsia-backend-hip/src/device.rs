//! The HIP `Backend` implementation (only compiled with a ROCm toolchain, `cfg(ik_hip)`).
//!
//! Threading: the engine calls a backend from several threads (generation worker, the caller's
//! thread for `forward_logits_each`, the telemetry thread via `info()`). The HIP runtime keeps a
//! *per-thread* current device, so every entry point calls `bind()` (hipSetDevice, a cached no-op
//! when already current) before touching the runtime.

use crate::ffi::{self, IkDeviceInfo, IkTensor, IK_MAX_DIMS};
use crate::kernels;
use intarsia_core::backend::{Backend, Buffer, BufferImpl, DeviceInfo, GraphHandle, GraphImpl, Layout, MemKind, StreamId};
use intarsia_core::{DType, Error, KernelImpl, OpCall, Result, TensorRef};
use std::any::Any;
use std::ffi::{c_int, c_void, CStr};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
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
    /// 0 = `Layout::RowMajor`, 1 = `Layout::Fp8Frag`; stored only by `relayout_fp8_frag`, after the bytes have been rearranged.
    layout: AtomicU8,
}

/// The dtype code a `Layout::Fp8Frag` fp8 weight is passed to the launchers with (kernels/hip/ik_hip_abi.h IK_HIP_DTYPE_FP8_E4M3_FRAG).
const IK_HIP_DTYPE_FP8_E4M3_FRAG: i32 = 259;

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
    fn layout(&self) -> Layout {
        match self.layout.load(Ordering::Acquire) {
            0 => Layout::RowMajor,
            _ => Layout::Fp8Frag,
        }
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

/// The process-wide graph lock: one HIP graph API call at a time (capture with its instantiate, and each launch) in the whole process.
///
/// The two ranks of a tensor-parallel group capture their decode graphs concurrently from two threads, and a peer rank's replay is launched
/// while the other rank instantiates. On the 27B tp2 reproduction (fresh solo states and batched steps in one model, 20 reps of the size loop)
/// that interleaving made a replay intermittently run only part of its kernels (the collective launch counters 10 to 61 short for one row, then
/// the ranks one launch apart, wrong logits in every row of the next batched step, and a collective that hangs until the bounded wait traps
/// 0x1016): every control run failed within 1 to 8 reps, the same runs with this lock did not in 40 of 40, and with solo decode graphs off
/// (no concurrent capture) did not either. The captures themselves are consistent (the same node counts on every rank), so the loss is in the
/// runtime's replay. With one rank (tp 1) there is no concurrency and the lock is never contended. `INTARSIA_GRAPH_LOCK=0` turns it off.
fn graph_lock() -> Option<std::sync::MutexGuard<'static, ()>> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    if *ON.get_or_init(|| std::env::var("INTARSIA_GRAPH_LOCK").map_or(true, |v| v != "0")) {
        Some(LOCK.lock().unwrap_or_else(|e| e.into_inner()))
    } else {
        None
    }
}

/// An instantiated HIP graph (`hipGraphExec_t`), destroyed on drop.
struct HipGraph {
    exec: usize,
    ordinal: i32,
    /// Set by the first launch (INTARSIA_GRAPH_SYNC_CAPTURE syncs the stream before it).
    launched: std::sync::atomic::AtomicBool,
}

impl GraphImpl for HipGraph {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Drop for HipGraph {
    fn drop(&mut self) {
        // INTARSIA_GRAPH_SYNC_DESTROY=1 (w26): wait for the device to go idle before the exec is destroyed. A peer rank's replay is enqueue-only, so the
        // exec may still be running (or queued) when the next capture clears the cache; HIP can hand a destroyed exec's kernel-argument memory to the
        // next instantiate at once.
        static SYNC: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let sync = *SYNC.get_or_init(|| std::env::var("INTARSIA_GRAPH_SYNC_DESTROY").is_ok_and(|v| v == "1"));
        // SAFETY: exec came from ik_rt_graph_end and is destroyed exactly once.
        unsafe {
            ffi::ik_rt_set_device(self.ordinal);
            if sync {
                ffi::ik_rt_device_sync();
            }
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
    /// `hipEvent_t` handles (as usize) for [`Backend::stream_wait`], used round robin: a wait takes the state of the event at the call, so an
    /// event can be recorded again once the waits on it are queued. Created on the first use (no event exists unless the lane TPO overlap runs).
    fork_events: Mutex<(Vec<usize>, usize)>,
    /// per-kernel GPU-time trace (INTARSIA_HIP_TRACE=1: start/stop event pair per launch), printed to stderr when the
    /// backend is dropped
    trace: Option<Mutex<Trace>>,
    /// a graph capture is running on this backend: launches are recorded, not executed, so the trace's stream events
    /// (which would become event nodes of the graph) are skipped
    capturing: AtomicBool,
    /// round 9: page-locked host staging for small copies (created on first use; `None` after a failed allocation)
    staging: Mutex<Option<Option<Staging>>>,
    /// lane wg: dense fp8 weights rearranged into `Layout::Fp8Frag` by `relayout_fp8_frag` (count, bytes)
    relayout: (AtomicU64, AtomicU64),
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

const FORK_EVENTS: usize = 16;
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

/// `INTARSIA_HIP_POISON=<hex byte>`: the byte every new device allocation is filled with (unset: none).
fn poison_byte() -> Option<u8> {
    static B: std::sync::OnceLock<Option<u8>> = std::sync::OnceLock::new();
    *B.get_or_init(|| std::env::var("INTARSIA_HIP_POISON").ok().and_then(|v| u8::from_str_radix(v.trim_start_matches("0x"), 16).ok()))
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
            fork_events: Mutex::new((Vec::new(), 0)),
            trace: std::env::var_os("INTARSIA_HIP_TRACE").filter(|v| v != "0").map(|_| Mutex::new(Trace::default())),
            capturing: AtomicBool::new(false),
            staging: Mutex::new(None),
            relayout: (AtomicU64::new(0), AtomicU64::new(0)),
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
        // a fragment-order weight is never passed as plain e4m3: the launchers that read it accept the private dtype, every other one rejects it
        let dtype = match (t.layout(), t.dtype) {
            (Layout::RowMajor, d) => d as i32,
            (Layout::Fp8Frag, DType::Fp8E4m3) => IK_HIP_DTYPE_FP8_E4M3_FRAG,
            (l, d) => return Err(Error::invalid(format!("a {d} tensor in layout {l:?} has no launcher dtype"))),
        };
        Ok(IkTensor { data, dtype, ndim: t.shape.len() as i32, shape, stride })
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
        if let Some(byte) = poison_byte() {
            // debugging aid (INTARSIA_HIP_POISON=<hex byte>, lane tp2): hipMalloc memory is not zeroed, so a read before write sees whatever the
            // last tenant left; fill every new allocation with a known byte (ff = NaN as f32 / bf16 / -1 as i32, 00 = the benign case) so the
            // read becomes deterministic. Not while a graph is being captured (the fill would become part of the graph).
            if !self.capturing.load(Ordering::Relaxed) {
                let s = self.stream(0)?;
                // SAFETY: the whole fresh allocation, stream 0 of this device (bound above), synchronised before the buffer is handed out.
                check("poison", unsafe { ffi::ik_rt_memset(p, byte as c_int, bytes.max(1), s) })?;
                check("poison sync", unsafe { ffi::ik_rt_stream_sync(s) })?;
            }
        }
        Ok(Buffer::new(HipBuffer { ptr: p, bytes, ordinal: self.ordinal, layout: AtomicU8::new(0) }))
    }

    fn upload(&self, dst: &Buffer, dst_offset: usize, src: &[u8], stream: StreamId) -> Result<()> {
        if dst.layout() != Layout::RowMajor {
            return Err(Error::backend(format!("upload into a {:?} buffer (its bytes are not in the order a caller would write)", dst.layout())));
        }
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
        if src.layout() != Layout::RowMajor {
            return Err(Error::backend(format!(
                "download of a {:?} buffer: its bytes are not in row-major order (use download_row_major)",
                src.layout()
            )));
        }
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
        if dst.layout() != Layout::RowMajor || src.layout() != Layout::RowMajor {
            return Err(Error::backend("device copy of a layout-tagged buffer: the copy would not carry the tag".to_string()));
        }
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

    fn fp8_frag_wanted(&self, n: usize, k: usize, act_quant: bool) -> bool {
        use crate::quantmoe::{fp8_frag_serves, WmmaKind};
        intarsia_core::quant::fp8::frag_eligible(n, k) && fp8_frag_serves(if act_quant { WmmaKind::Fp8A8 } else { WmmaKind::Fp8A16 })
    }

    /// Rearrange a dense fp8 weight into `Layout::Fp8Frag` in place: permute into a scratch allocation of the tensor's size (the whole
    /// extra memory, freed on return), copy back, synchronise, and only then tag the allocation.
    fn relayout_fp8_frag(&self, w: &TensorRef) -> Result<()> {
        let (n, k) = match w.shape[..] {
            [n, k] if w.dtype == DType::Fp8E4m3 => (n, k),
            _ => return Err(Error::invalid(format!("fragment-order layout: want a 2-D fp8 weight, got {} {:?}", w.dtype, w.shape))),
        };
        if !intarsia_core::quant::fp8::frag_eligible(n, k) {
            return Err(Error::invalid(format!("fragment-order layout: {n} x {k} needs N % 16 == 0 and K % 32 == 0")));
        }
        if w.layout() != Layout::RowMajor {
            return Err(Error::invalid("fragment-order layout: the weight is already tagged"));
        }
        let hb = w
            .buf
            .downcast::<HipBuffer>()
            .ok_or_else(|| Error::backend("buffer does not belong to the hip backend"))?;
        if w.offset != 0 || w.buf.bytes() != w.nbytes() {
            return Err(Error::invalid("fragment-order layout: the tag covers a whole allocation, the weight must be all of it"));
        }
        self.bind()?;
        let scratch = self.alloc(w.nbytes(), MemKind::Device)?;
        let src = Self::tensor(w)?;
        let mut dst = Self::tensor(&TensorRef::new(scratch.clone(), DType::Fp8E4m3, &w.shape))?;
        let st = self.stream(0)?;
        // SAFETY: both tensors are bounds-checked views of live allocations; the launcher validates dtype, shape and alignment.
        check("relayout_fp8_frag", unsafe { ffi::ik_fp8_relayout_frag(&src, &mut dst, st) })?;
        self.copy(&w.buf, 0, &scratch, 0, w.nbytes(), 0)?;
        self.synchronize(0)?;
        hb.layout.store(1, Ordering::Release);
        self.relayout.0.fetch_add(1, Ordering::Relaxed);
        self.relayout.1.fetch_add(w.nbytes() as u64, Ordering::Relaxed);
        Ok(())
    }

    fn layout_report(&self) -> Option<String> {
        crate::quantmoe::fp8_frag_enabled().then(|| {
            format!(
                "relayout: {} tensors -> frag order ({:.1} MB)",
                self.relayout.0.load(Ordering::Relaxed),
                self.relayout.1.load(Ordering::Relaxed) as f64 / 1e6
            )
        })
    }

    fn download_row_major(&self, t: &TensorRef) -> Result<Vec<u8>> {
        let mut data = vec![0u8; t.nbytes()];
        if t.layout() == Layout::RowMajor {
            self.download(&t.buf, t.offset, &mut data, 0)?;
            return Ok(data);
        }
        // un-permute on the device into a plain scratch allocation, then an ordinary download of that
        self.bind()?;
        let plain = self.alloc(t.nbytes(), MemKind::Device)?;
        let src = Self::tensor(t)?;
        let mut dst = Self::tensor(&TensorRef::new(plain.clone(), t.dtype, &t.shape))?;
        let st = self.stream(0)?;
        // SAFETY: as in relayout_fp8_frag.
        check("unrelayout_fp8_frag", unsafe { ffi::ik_fp8_unrelayout_frag(&src, &mut dst, st) })?;
        self.download(&plain, 0, &mut data, 0)?;
        Ok(data)
    }

    fn kernels(&self) -> Vec<KernelImpl> {
        let mut v = kernels::all();
        // A/B and debugging aid: INTARSIA_HIP_DISABLE=name1,name2 hides kernels from the registry, so the
        // plan falls back to the next candidate (e.g. the SIMT attention kernels instead of flash / split-KV)
        if let Ok(list) = std::env::var("INTARSIA_HIP_DISABLE") {
            let off: Vec<&str> = list.split(',').map(str::trim).filter(|x| !x.is_empty()).collect();
            v.retain(|k| !off.contains(&k.name.as_str()));
            // a row-table twin goes with the kernel it twins: it must never run where the solo step would run another kernel
            v.retain(|k| !(k.row_table && k.name.strip_suffix("_rows").is_some_and(|base| off.contains(&base))));
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
        // row table (batched decode, `OpCall::rows`): the device addresses of the rows' state buffers; null for an ordinary call, which
        // the row-table launchers refuse
        let rows_t: Option<(IkTensor, i64, i64, i64)> = call
            .rows
            .as_ref()
            .map(|r| Self::tensor(&r.ptrs).map(|t| (t, r.stride as i64, r.index as i64, r.byte_off as i64)))
            .transpose()?;
        let tab = rows_t.as_ref().map_or(std::ptr::null(), |t| &t.0 as *const IkTensor);
        let (tab_stride, tab_index, tab_off) = rows_t.as_ref().map_or((0, 0, 0), |t| (t.1, t.2, t.3));
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
                "attn_decode_bf16_window" => {
                    ffi::ik_attn_decode_bf16_window(i(0), i(1), i(2), a.pos0 as i64, a.window as i64, a.scale, o, pos, s)
                }
                "dyn_conv_bf16" => ffi::ik_dyn_conv_bf16(i(0), i(1), i(2), o, s),
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
                "gdn_conv_bf16_rows" => ffi::ik_gdn_conv_bf16_rows(i(0), i(1), o, o.add(1), tab, tab_stride, tab_index, s),
                "gdn_delta_rule_bf16_v2_rows" => ffi::ik_gdn_delta_rule_bf16_v2_rows(
                    i(0), i(1), i(2), i(3), i(4), a.eps, o, o.add(1), tab, tab_stride, tab_index, tab_off, s,
                ),
                "kv_append_bf16_rows" => ffi::ik_kv_append_bf16_rows(i(0), o, tab, tab_stride, tab_index, pos, s),
                "attn_decode_bf16_splitkv_rows" => {
                    ffi::ik_attn_decode_bf16_splitkv_rows(i(0), i(1), i(2), a.scale, o, sc, tab, tab_stride, tab_index, pos, s)
                }
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
                "matmul_bf16_wmma" => ffi::ik_matmul_bf16_wmma(i(0), i(1), o, s),
                "matmul_bf16_wmma_rows" => ffi::ik_matmul_bf16_wmma_rows(i(0), i(1), o, s),
                "matmul_pair_bf16_wmma" => ffi::ik_matmul_pair_bf16_wmma(i(0), i(1), i(2), o, o.add(1), s),
                "rmsnorm_add_matmul_pair_bf16_wmma" => ffi::ik_rmsnorm_add_matmul_pair_bf16_wmma(
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
                "matmul_fp8_wmma" => ffi::ik_matmul_fp8_wmma(i(0), i(1), i(2), o, sc, s),
                "matmul_fp8_wmma_rows" => ffi::ik_matmul_fp8_wmma_rows(i(0), i(1), i(2), o, sc, s),
                "matmul_fp8_mma" => ffi::ik_matmul_fp8_mma(i(0), i(1), i(2), o, s),
                "matmul_fp8_mma_f8" => ffi::ik_matmul_fp8_mma_f8(i(0), i(1), i(2), o, sc, s),
                "matmul_fp8_mma_v2" => ffi::ik_matmul_fp8_mma_v2(i(0), i(1), i(2), o, sc, s),
                "matmul_nvfp4_gemv" => ffi::ik_matmul_nvfp4_gemv(i(0), i(1), i(2), i(3), o, s),
                "matmul_nvfp4_gemv_v2" => ffi::ik_matmul_nvfp4_gemv_v2(i(0), i(1), i(2), i(3), o, s),
                "matmul_nvfp4_gemv_rows" => ffi::ik_matmul_nvfp4_gemv_rows(i(0), i(1), i(2), i(3), o, s),
                "matmul_nvfp4_mma" => ffi::ik_matmul_nvfp4_mma(i(0), i(1), i(2), i(3), o, s),
                "matmul_mxfp4_gemv" => ffi::ik_matmul_mxfp4_gemv(i(0), i(1), i(2), i(3), o, sc, s),
                // W4A16 keeps the bf16 rows kernel (the int8 path would take its slower flagged fp chain), W4A4 the int8 path
                "matmul_mxfp4_gemv_rows" if a.mxfp4_w4a16 => ffi::ik_matmul_mxfp4_gemv_rows_bf16(i(0), i(1), i(2), i(3), o, sc, s),
                "matmul_mxfp4_gemv_rows" => ffi::ik_matmul_mxfp4_gemv_rows(i(0), i(1), i(2), i(3), o, sc, s),
                "matmul_mxfp4_a8_gemv" => ffi::ik_matmul_mxfp4_a8_gemv(i(0), i(1), i(2), i(3), o, sc, s),
                "matmul_mxfp4_a8_gemv_rows" => ffi::ik_matmul_mxfp4_a8_gemv_rows(i(0), i(1), i(2), i(3), o, sc, s),
                "matmul_mxfp4_mma" => ffi::ik_matmul_mxfp4_mma(i(0), i(1), i(2), i(3), o, sc, s),
                "matmul_mxfp4_mma_v1" => ffi::ik_matmul_mxfp4_mma_v1(i(0), i(1), i(2), i(3), o, sc, s),
                "matmul_mxfp4_mma_fold" => ffi::ik_matmul_mxfp4_mma_fold(i(0), i(1), i(2), i(3), o, sc, s),
                "matmul_mxfp4_a8_mma" => ffi::ik_matmul_mxfp4_a8_mma(i(0), i(1), i(2), i(3), o, sc, s),
                "matmul_mxfp4_pair_gemv" => ffi::ik_matmul_mxfp4_pair_gemv(i(0), i(1), i(2), i(3), i(4), i(5), o, o.add(1), sc, s),
                "matmul_mxfp4_pair_gemv_rows" if a.mxfp4_w4a16 => {
                    ffi::ik_matmul_mxfp4_pair_gemv_rows_bf16(i(0), i(1), i(2), i(3), i(4), i(5), o, o.add(1), sc, s)
                }
                "matmul_mxfp4_pair_gemv_rows" => ffi::ik_matmul_mxfp4_pair_gemv_rows(i(0), i(1), i(2), i(3), i(4), i(5), o, o.add(1), sc, s),
                "matmul_mxfp4_pair_a8_gemv" => ffi::ik_matmul_mxfp4_pair_a8_gemv(i(0), i(1), i(2), i(3), i(4), i(5), o, o.add(1), sc, s),
                "matmul_mxfp4_pair_a8_gemv_rows" => ffi::ik_matmul_mxfp4_pair_a8_gemv_rows(i(0), i(1), i(2), i(3), i(4), i(5), o, o.add(1), sc, s),
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
                "moe_block_bf16_rows" => ffi::ik_moe_block_bf16_rows(i(0), a.top_k as c_int, a.norm_topk as c_int, o, sc, s),
                "moe_block_bf16_grouped" => ffi::ik_moe_block_bf16_grouped(i(0), a.top_k as c_int, a.norm_topk as c_int, o, sc, s),
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
                "matmul_fp8_pair_wmma" => ffi::ik_matmul_fp8_pair_wmma(i(0), i(1), i(2), i(3), i(4), o, o.add(1), s),
                "gated_rmsnorm_matmul_fp8_wmma" => {
                    ffi::ik_gated_rmsnorm_matmul_fp8_wmma(i(0), i(1), i(2), a.eps, a.w_offset, i(3), i(4), o, s)
                }
                "rmsnorm_add_matmul_fp8_pair_wmma" => ffi::ik_rmsnorm_add_matmul_fp8_pair_wmma(
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
                "sigmoid_mul_matmul_fp8_wmma" => ffi::ik_sigmoid_mul_matmul_fp8_wmma(i(0), i(1), i(2), i(3), o, s),
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

    fn stream_overlap(&self) -> bool {
        true
    }

    fn stream_wait(&self, waiter: StreamId, signal: StreamId) -> Result<()> {
        self.bind()?;
        let (w, s) = (self.stream(waiter)?, self.stream(signal)?);
        let ev = {
            let mut pool = self.fork_events.lock().unwrap();
            if pool.0.is_empty() {
                for _ in 0..FORK_EVENTS {
                    let mut h: *mut c_void = std::ptr::null_mut();
                    // SAFETY: valid out-pointer; the device is bound.
                    check("hipEventCreate", unsafe { ffi::ik_rt_event_create_sync(&mut h) })?;
                    pool.0.push(h as usize);
                }
            }
            let i = pool.1;
            pool.1 = (i + 1) % FORK_EVENTS;
            pool.0[i] as *mut c_void
        };
        // SAFETY: a live event and two live streams of this device; the wait uses the event's state at this call.
        check("hipEventRecord", unsafe { ffi::ik_rt_event_record(ev, s) })?;
        check("hipStreamWaitEvent", unsafe { ffi::ik_rt_stream_wait_event(w, ev) })
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
    /// cases in the fused-op tests (lane H1, docs/results/2026-10-21-h1-tp2-hip.md). Default-on at tp > 1 (`INTARSIA_TP_FUSION=0` opts out).
    fn fusion_under_tp(&self) -> bool {
        true
    }

    fn verify_attn_per_row(&self) -> bool {
        true
    }

    fn supports_graphs(&self) -> bool {
        true
    }

    /// Batched decode steps launch the GatedDeltaNet conv / delta rule, the KV append and the split-KV decode attention once for
    /// all rows (`OpCall::rows`; the `_rows` row-table kernels, proven bitwise the one-row kernels in tests/row_tables.rs).
    fn supports_row_tables(&self) -> bool {
        true
    }

    /// Device-side top-k for sampling (`ik_topk_f32`, kernels/hip/topk.hip: the CUDA kernel's ABI and exact ordering). The model only
    /// uses it with `INTARSIA_DEVICE_SAMPLING=1`.
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

    /// 16 rows: the `_rows` GEMV / MoE kernels are proven row-exact up to M = 16 (tests/rows_exact.rs, G-B).
    fn batch_row_cap(&self) -> usize {
        crate::kernels::ROWS_MAX
    }

    fn graph_capture(&self, stream: StreamId, body: &mut dyn FnMut() -> Result<()>) -> Result<GraphHandle> {
        // see `graph_lock`
        let _lock = graph_lock();
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
        let graph = (!exec.is_null()).then(|| HipGraph { exec: exec as usize, ordinal: self.ordinal, launched: std::sync::atomic::AtomicBool::new(false) });
        ran?;
        ended?;
        let graph = graph.ok_or_else(|| Error::backend("graph capture produced no executable graph"))?;
        Ok(GraphHandle::new(graph))
    }

    fn graph_replay(&self, g: &GraphHandle, stream: StreamId) -> Result<()> {
        let g = g.downcast::<HipGraph>().ok_or_else(|| Error::backend("graph does not belong to the hip backend"))?;
        self.bind()?;
        let s = self.stream(stream)?;
        let _lock = graph_lock();
        // INTARSIA_GRAPH_SYNC_CAPTURE=1: the stream is idle before the first launch of a fresh exec
        // SAFETY: plain FFI calls on this rank's bound device and stream.
        if unsafe { ffi::ik_rt_sync_capture_on() } == 1 && !g.launched.swap(true, Ordering::Relaxed) {
            check("stream synchronize", unsafe { ffi::ik_rt_stream_sync(s) })?;
        }
        // SAFETY: `exec` is a live hipGraphExec_t owned by `g`.
        check("hipGraphLaunch", unsafe { ffi::ik_rt_graph_launch(g.exec as *mut c_void, s) })
    }
}
