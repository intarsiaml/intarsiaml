//! Structural check that the screened greedy head (`INTARSIA_INK_HEAD_SCREEN`, lane ink-fp8) leaves a batched decode step
//! untouched (brief 16, the exact replacement of gate item 3b): one B8 and one B16 `forward_batch` step are recorded at the
//! Metal API, and the record of a screen-off process must equal the record of a screen-on process.
//!
//!   INTARSIA_INK_HEAD_SCREEN=0|1 INTARSIA_METAL_TEST_MODEL=<Inkling-2-Mini> \
//!     cargo run --release -p intarsia-models --example screen_batch_trace -- --out <file> [--ctx 256] [--batches 8,16]
//!
//! Recorded per traced step, in order: every compute-encoder call (pipeline by kernel name, buffer bindings as buffer length +
//! offset + index, set-bytes length + content, threadgroup memory, dispatch grid and threadgroup sizes, resource-residency
//! calls, ICB executions, barriers, `endEncoding`), blit copies, and command-buffer commits. The hooks replace the methods
//! of the encoder / command-buffer classes for this process only (a diagnostic, not part of the engine). The model is
//! warmed first (two batched steps at each B), so lazy first-use work does not land in the traced step.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("screen_batch_trace needs macOS / Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    imp::main()
}

#[cfg(target_os = "macos")]
mod imp {
    use intarsia_backend_metal::MetalBackend;
    use intarsia_core::model::{HeadMode, LoadOptions, Model, StepOut};
    use intarsia_core::state::SequenceState;
    use intarsia_core::Telemetry;
    use intarsia_models::load::SchemedSource;
    use intarsia_models::{read_model_spec, EngineOptions};
    use objc2::rc::Retained;
    use objc2::runtime::{AnyClass, AnyObject, Imp, Sel};
    use objc2::{msg_send, sel};
    use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLDispatchType, MTLSize};
    use std::collections::HashMap;
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex, OnceLock};

    static ON: AtomicBool = AtomicBool::new(false);
    static LOG: Mutex<Vec<String>> = Mutex::new(Vec::new());
    static NAMES: OnceLock<HashMap<usize, &'static str>> = OnceLock::new();
    static ORIG: OnceLock<Mutex<HashMap<&'static str, usize>>> = OnceLock::new();

    fn rec(s: String) {
        if ON.load(Ordering::Relaxed) {
            LOG.lock().unwrap().push(s);
        }
    }
    fn orig(name: &'static str) -> usize {
        *ORIG.get().unwrap().lock().unwrap().get(name).unwrap()
    }
    unsafe fn blen(b: *mut AnyObject) -> usize {
        if b.is_null() {
            return 0;
        }
        msg_send![&*b, length]
    }
    fn sz(s: MTLSize) -> String {
        format!("{}x{}x{}", s.width, s.height, s.depth)
    }

    unsafe extern "C-unwind" fn h_pso(this: *mut AnyObject, cmd: Sel, p: *mut AnyObject) {
        let n = NAMES.get().and_then(|m| m.get(&(p as usize)).copied()).unwrap_or("?unknown-pso");
        rec(format!("pso {n}"));
        let f: unsafe extern "C-unwind" fn(*mut AnyObject, Sel, *mut AnyObject) = std::mem::transmute(orig("pso"));
        f(this, cmd, p)
    }
    unsafe extern "C-unwind" fn h_buf(this: *mut AnyObject, cmd: Sel, b: *mut AnyObject, off: usize, idx: usize) {
        rec(format!("buf[{idx}] len {} off {off}", blen(b)));
        let f: unsafe extern "C-unwind" fn(*mut AnyObject, Sel, *mut AnyObject, usize, usize) = std::mem::transmute(orig("buf"));
        f(this, cmd, b, off, idx)
    }
    unsafe extern "C-unwind" fn h_bufoff(this: *mut AnyObject, cmd: Sel, off: usize, idx: usize) {
        rec(format!("bufoff[{idx}] off {off}"));
        let f: unsafe extern "C-unwind" fn(*mut AnyObject, Sel, usize, usize) = std::mem::transmute(orig("bufoff"));
        f(this, cmd, off, idx)
    }
    unsafe extern "C-unwind" fn h_bytes(this: *mut AnyObject, cmd: Sel, p: *const c_void, len: usize, idx: usize) {
        let b = std::slice::from_raw_parts(p as *const u8, len);
        rec(format!("bytes[{idx}] len {len} {}", b.iter().map(|x| format!("{x:02x}")).collect::<String>()));
        let f: unsafe extern "C-unwind" fn(*mut AnyObject, Sel, *const c_void, usize, usize) = std::mem::transmute(orig("bytes"));
        f(this, cmd, p, len, idx)
    }
    unsafe extern "C-unwind" fn h_tgmem(this: *mut AnyObject, cmd: Sel, len: usize, idx: usize) {
        rec(format!("tgmem[{idx}] {len}"));
        let f: unsafe extern "C-unwind" fn(*mut AnyObject, Sel, usize, usize) = std::mem::transmute(orig("tgmem"));
        f(this, cmd, len, idx)
    }
    unsafe extern "C-unwind" fn h_dtg(this: *mut AnyObject, cmd: Sel, g: MTLSize, t: MTLSize) {
        rec(format!("dispatch groups {} threads {}", sz(g), sz(t)));
        let f: unsafe extern "C-unwind" fn(*mut AnyObject, Sel, MTLSize, MTLSize) = std::mem::transmute(orig("dtg"));
        f(this, cmd, g, t)
    }
    unsafe extern "C-unwind" fn h_dth(this: *mut AnyObject, cmd: Sel, g: MTLSize, t: MTLSize) {
        rec(format!("dispatch threads {} tg {}", sz(g), sz(t)));
        let f: unsafe extern "C-unwind" fn(*mut AnyObject, Sel, MTLSize, MTLSize) = std::mem::transmute(orig("dth"));
        f(this, cmd, g, t)
    }
    unsafe extern "C-unwind" fn h_use(this: *mut AnyObject, cmd: Sel, r: *const *mut AnyObject, n: usize, usage: usize) {
        let lens: Vec<usize> = (0..n).map(|i| blen(*r.add(i))).collect();
        rec(format!("useResources n {n} usage {usage} lens {lens:?}"));
        let f: unsafe extern "C-unwind" fn(*mut AnyObject, Sel, *const *mut AnyObject, usize, usize) = std::mem::transmute(orig("use"));
        f(this, cmd, r, n, usage)
    }
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct NSRange {
        loc: usize,
        len: usize,
    }
    unsafe impl objc2::Encode for NSRange {
        const ENCODING: objc2::Encoding = objc2::Encoding::Struct("_NSRange", &[usize::ENCODING, usize::ENCODING]);
    }
    unsafe extern "C-unwind" fn h_icb(this: *mut AnyObject, cmd: Sel, icb: *mut AnyObject, r: NSRange) {
        rec(format!("executeICB range {}+{}", r.loc, r.len));
        let f: unsafe extern "C-unwind" fn(*mut AnyObject, Sel, *mut AnyObject, NSRange) = std::mem::transmute(orig("icb"));
        f(this, cmd, icb, r)
    }
    unsafe extern "C-unwind" fn h_end(this: *mut AnyObject, cmd: Sel) {
        rec("endEncoding".into());
        let f: unsafe extern "C-unwind" fn(*mut AnyObject, Sel) = std::mem::transmute(orig("end"));
        f(this, cmd)
    }
    unsafe extern "C-unwind" fn h_bend(this: *mut AnyObject, cmd: Sel) {
        rec("blit endEncoding".into());
        let f: unsafe extern "C-unwind" fn(*mut AnyObject, Sel) = std::mem::transmute(orig("bend"));
        f(this, cmd)
    }
    unsafe extern "C-unwind" fn h_copy(this: *mut AnyObject, cmd: Sel, s: *mut AnyObject, so: usize, d: *mut AnyObject, dof: usize, n: usize) {
        rec(format!("blit copy src len {} off {so} dst len {} off {dof} size {n}", blen(s), blen(d)));
        let f: unsafe extern "C-unwind" fn(*mut AnyObject, Sel, *mut AnyObject, usize, *mut AnyObject, usize, usize) = std::mem::transmute(orig("copy"));
        f(this, cmd, s, so, d, dof, n)
    }
    unsafe extern "C-unwind" fn h_commit(this: *mut AnyObject, cmd: Sel) {
        rec("commit".into());
        let f: unsafe extern "C-unwind" fn(*mut AnyObject, Sel) = std::mem::transmute(orig("commit"));
        f(this, cmd)
    }

    /// Replace `sel` on `cls` (or the superclass that implements it) with `imp`, keeping the original under `key`.
    unsafe fn hook(cls: &AnyClass, s: Sel, key: &'static str, imp: Imp) {
        let m = cls.instance_method(s).unwrap_or_else(|| panic!("{} has no {s:?}", cls.name().to_string_lossy()));
        let old = m.set_implementation(imp);
        ORIG.get_or_init(Default::default).lock().unwrap().insert(key, old as usize);
    }

    fn install(be: &MetalBackend) {
        // pipeline object -> kernel name, over every pipeline the backend made
        let mut names = HashMap::new();
        for n in intarsia_backend_metal::kernels::shader_names().into_iter().chain(intarsia_backend_metal::kernels::metal4_shaders()) {
            if let Some(p) = be.pso(n) {
                names.insert(Retained::as_ptr(p) as *const c_void as usize, n);
            }
        }
        eprintln!("trace: {} pipelines named", names.len());
        NAMES.set(names).unwrap();
        let cb = be.queue().commandBuffer().unwrap();
        let enc = cb.computeCommandEncoderWithDispatchType(MTLDispatchType::Serial).unwrap();
        let ecls = unsafe { &*(Retained::as_ptr(&enc) as *const AnyObject) }.class();
        enc.endEncoding();
        let blit = cb.blitCommandEncoder().unwrap();
        let bcls = unsafe { &*(Retained::as_ptr(&blit) as *const AnyObject) }.class();
        blit.endEncoding();
        let ccls = unsafe { &*(Retained::as_ptr(&cb) as *const AnyObject) }.class();
        eprintln!("trace: classes {} / {} / {}", ecls.name().to_string_lossy(), bcls.name().to_string_lossy(), ccls.name().to_string_lossy());
        unsafe {
            hook(ecls, sel!(setComputePipelineState:), "pso", std::mem::transmute(h_pso as unsafe extern "C-unwind" fn(_, _, _)));
            hook(ecls, sel!(setBuffer:offset:atIndex:), "buf", std::mem::transmute(h_buf as unsafe extern "C-unwind" fn(_, _, _, _, _)));
            hook(ecls, sel!(setBufferOffset:atIndex:), "bufoff", std::mem::transmute(h_bufoff as unsafe extern "C-unwind" fn(_, _, _, _)));
            hook(ecls, sel!(setBytes:length:atIndex:), "bytes", std::mem::transmute(h_bytes as unsafe extern "C-unwind" fn(_, _, _, _, _)));
            hook(ecls, sel!(setThreadgroupMemoryLength:atIndex:), "tgmem", std::mem::transmute(h_tgmem as unsafe extern "C-unwind" fn(_, _, _, _)));
            hook(ecls, sel!(dispatchThreadgroups:threadsPerThreadgroup:), "dtg", std::mem::transmute(h_dtg as unsafe extern "C-unwind" fn(_, _, _, _)));
            hook(ecls, sel!(dispatchThreads:threadsPerThreadgroup:), "dth", std::mem::transmute(h_dth as unsafe extern "C-unwind" fn(_, _, _, _)));
            hook(ecls, sel!(useResources:count:usage:), "use", std::mem::transmute(h_use as unsafe extern "C-unwind" fn(_, _, _, _, _)));
            hook(ecls, sel!(executeCommandsInBuffer:withRange:), "icb", std::mem::transmute(h_icb as unsafe extern "C-unwind" fn(_, _, _, _)));
            hook(ecls, sel!(endEncoding), "end", std::mem::transmute(h_end as unsafe extern "C-unwind" fn(_, _)));
            hook(bcls, sel!(endEncoding), "bend", std::mem::transmute(h_bend as unsafe extern "C-unwind" fn(_, _)));
            hook(bcls, sel!(copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:), "copy", std::mem::transmute(h_copy as unsafe extern "C-unwind" fn(_, _, _, _, _, _, _)));
            hook(ccls, sel!(commit), "commit", std::mem::transmute(h_commit as unsafe extern "C-unwind" fn(_, _)));
        }
    }

    fn arg(args: &[String], name: &str, default: &str) -> String {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| default.into())
    }

    pub fn main() {
        let args: Vec<String> = std::env::args().collect();
        let ctx: usize = arg(&args, "--ctx", "256").parse().unwrap();
        let out = arg(&args, "--out", "screen_batch_trace.txt");
        let batches: Vec<usize> = arg(&args, "--batches", "8,16").split(',').map(|x| x.parse().unwrap()).collect();
        let dir = std::path::PathBuf::from(std::env::var_os("INTARSIA_METAL_TEST_MODEL").expect("INTARSIA_METAL_TEST_MODEL"));
        let be = Arc::new(MetalBackend::new().unwrap());
        let backend: Arc<dyn intarsia_core::Backend> = be.clone();
        let (_, recipe, dims, _) = read_model_spec(&dir, &EngineOptions::default()).unwrap();
        let weights = SchemedSource::open(&dir).unwrap();
        let chunk = recipe.resolve_prefill_chunk(None, "metal").value;
        let top = *batches.iter().max().unwrap();
        let opts = LoadOptions { max_seq: ctx + 64, prefill_chunk: chunk, profile_every: 0, ..Default::default() };
        let mut m = Model::load(backend, &recipe, &dims, &weights, &opts, Arc::new(Telemetry::new())).unwrap();
        m.enable_batch(top).unwrap();
        m.configure_plan(&Default::default(), None).unwrap();
        let screen = m.head_screen_stats().is_some();
        let mut seqs: Vec<SequenceState> = Vec::new();
        let mut cur: Vec<u32> = Vec::new();
        for i in 0..top {
            let mut rng = intarsia_core::rng::SplitMix64::new(0xbe7c_0000 + i as u64);
            let p: Vec<u32> = (0..ctx).map(|_| 1000 + (rng.next_u64() % 29000) as u32).collect();
            let mut s = m.new_state().unwrap();
            m.forward(&mut s, &p[..ctx - 1], HeadMode::None, None).unwrap();
            seqs.push(s);
            cur.push(p[ctx - 1]);
        }
        // one solo greedy step on sequence 0 (the screen's first use, so its lazy check is done before tracing)
        {
            let mut s = m.new_state().unwrap();
            m.forward(&mut s, &cur[..1], HeadMode::LastArgmax, None).unwrap();
        }
        let step = |m: &Model, seqs: &mut [SequenceState], cur: &mut [u32], b: usize| {
            let ids = cur[..b].to_vec();
            let modes = vec![HeadMode::LastArgmax; b];
            let mut refs: Vec<&mut SequenceState> = seqs[..b].iter_mut().collect();
            let outs = m.forward_batch(&mut refs, &ids, &modes).unwrap();
            for (i, o) in outs.iter().enumerate() {
                cur[i] = match o {
                    StepOut::Token(t) => *t,
                    _ => panic!("no token"),
                };
            }
        };
        for &b in &batches {
            step(&m, &mut seqs, &mut cur, b);
            step(&m, &mut seqs, &mut cur, b);
        }
        install(&be);
        let mut text = format!("# screen loaded: {screen}; ctx {ctx}\n");
        for &b in &batches {
            LOG.lock().unwrap().clear();
            ON.store(true, Ordering::SeqCst);
            step(&m, &mut seqs, &mut cur, b);
            ON.store(false, Ordering::SeqCst);
            let log = std::mem::take(&mut *LOG.lock().unwrap());
            let disp = log.iter().filter(|l| l.starts_with("dispatch")).count();
            let commits = log.iter().filter(|l| *l == "commit").count();
            let unknown = log.iter().filter(|l| l.contains("?unknown-pso")).count();
            eprintln!("B{b}: {} records, {disp} dispatches, {commits} command buffers, {unknown} unnamed pipelines, tokens {:?}", log.len(), &cur[..b]);
            text += &format!("## B{b}: {disp} dispatches, {commits} command buffers\n");
            for l in log {
                text += &l;
                text.push('\n');
            }
        }
        std::fs::write(&out, text).unwrap();
        eprintln!("trace: wrote {out} (screen loaded: {screen})");
    }
}
