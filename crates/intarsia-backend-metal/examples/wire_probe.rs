//! First-use latency of mmap'd weight shards on the GPU. A cold process pays about 140 ms before a Qwen3-8B prefill gets
//! going (command buffer 0 starts ~100 ms after its commit, buffers 1 and 2 ~25 ms later each): this wraps the shards of a
//! model directory the way the loader does and times one 16-byte GPU copy out of each, twice, after an optional
//! pre-wiring step.
//!
//!   wire_probe <model dir> [none|residency|touch|mlock|madvise|copy] [sleep ms] [gap ms]
//!
//!   none       time the first and second use of every shard
//!   residency  an MTLResidencySet holding every shard (requestResidency, queue.addResidencySet) before the timed uses
//!   touch      one GPU copy per shard (not timed) before the timed uses
//!   mlock      mlock every mapping (CPU-side page wiring) before the timed uses
//!   madvise    madvise(MADV_WILLNEED) and a read pass over every mapping before the timed uses
//!   keep       touch pass, then during the gap a 16-byte GPU copy every KEEP_MS ms (default 200) out of KEEP_WHAT =
//!              all (every shard, one command buffer) | first (shard 1 only) | none (no weight buffer): a keep-warm
//!   blitref    raw blit command buffers on the queue (no stream): touch pass, gap, timed use of the wrapped shards
//!   copyshared like blitref, but on shared MTLBuffers the backend allocated and filled (memcpy), touched once before the gap
//!   copyprivate like blitref, but on StorageModePrivate buffers filled by a GPU blit out of the wrapped shards
//!   residencycpu like blitref with an MTLResidencySet on the queue, and a CPU-only requestResidency() every KEEP_MS (500)
//!   copy       copy every shard into a fresh shared MTLBuffer (alloc + memcpy), the timed uses read those instead
//!
//! `sleep ms` waits that long after wrapping (a loader doing other work) before the mode's setup; `gap ms` waits that long
//! after the setup (an idle server) before the timed uses. The probe pre-wires wrapped files
//! (`MetalBackend::set_prewire`) only with `INTARSIA_METAL_PREWIRE=1` in its environment.
//! Run inside a fleetq slot.
#[cfg(not(target_os = "macos"))]
fn main() {}

#[cfg(target_os = "macos")]
fn main() {
    use intarsia_backend_metal::{MetalBackend, MetalBuffer};
    use intarsia_core::backend::{Backend, HostMemory, MemKind};
    use memmap2::Mmap;
    use std::sync::Arc;
    use std::time::Instant;

    struct MmapHost(Mmap);
    impl HostMemory for MmapHost {
        fn as_slice(&self) -> &[u8] {
            &self.0
        }
    }

    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("model dir");
    let mode = args.next().unwrap_or_else(|| "none".into());
    let sleep_ms: u64 = args.next().and_then(|v| v.parse().ok()).unwrap_or(0);
    let gap_ms: u64 = args.next().and_then(|v| v.parse().ok()).unwrap_or(0);
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    let be = MetalBackend::new().unwrap();
    be.set_prewire(std::env::var_os("INTARSIA_METAL_PREWIRE").is_some_and(|v| v == "1"));
    let small = be.alloc(4096, MemKind::Device).unwrap();
    let t0 = Instant::now();
    let mut mems: Vec<Arc<dyn HostMemory>> = Vec::new();
    let mut shards: Vec<_> = files
        .iter()
        .map(|p| {
            let mem: Arc<dyn HostMemory> = Arc::new(MmapHost(unsafe { Mmap::map(&std::fs::File::open(p).unwrap()).unwrap() }));
            let len = mem.as_slice().len();
            let b = be.wrap_host(mem.clone(), 0, len).expect("wrap");
            mems.push(mem);
            (p.file_name().unwrap().to_string_lossy().to_string(), len, b)
        })
        .collect();
    println!("mapped and wrapped {} shards in {:.1} ms (mode {mode}, sleep {sleep_ms} ms, gap {gap_ms} ms)", shards.len(), t0.elapsed().as_secs_f64() * 1e3);
    std::thread::sleep(std::time::Duration::from_millis(sleep_ms));
    if matches!(mode.as_str(), "blitref" | "copyshared" | "copyprivate" | "residencycpu") {
        raw_modes(&be, &mode, &mems, &shards, sleep_ms, gap_ms);
        return;
    }
    let use_once = |b: &intarsia_core::backend::Buffer| {
        let t = Instant::now();
        be.copy(&small, 0, b, 0, 16, 0).unwrap();
        be.synchronize(0).unwrap();
        t.elapsed().as_secs_f64() * 1e3
    };
    match mode.as_str() {
        "residency" => {
            use objc2_metal::{MTLAllocation, MTLCommandQueue, MTLDevice, MTLResidencySet, MTLResidencySetDescriptor};
            let t = Instant::now();
            let desc = MTLResidencySetDescriptor::new();
            let _ = (&desc, &t);
            let set = be.device().newResidencySetWithDescriptor_error(&desc).expect("residency set");
            for (_, _, b) in &shards {
                let mb = b.downcast::<MetalBuffer>().unwrap();
                set.addAllocation(objc2::runtime::ProtocolObject::<dyn MTLAllocation>::from_ref(mb.mtl()));
            }
            set.commit();
            let t1 = t.elapsed().as_secs_f64() * 1e3;
            set.requestResidency();
            let t2 = t.elapsed().as_secs_f64() * 1e3;
            be.queue().addResidencySet(&set);
            println!("residency set built+committed in {t1:.2} ms, requestResidency returned at {t2:.2} ms, added to queue at {:.2} ms", t.elapsed().as_secs_f64() * 1e3);
            // wait until resident (allocatedSize grows to the shards' total)
            let t = Instant::now();
            loop {
                std::thread::sleep(std::time::Duration::from_millis(5));
                if t.elapsed().as_secs() > 20 || set.allocatedSize() >= shards.iter().map(|s| s.1 as u64).sum::<u64>() / 2 {
                    break;
                }
            }
            println!("residency set allocatedSize {} MB after {:.0} ms", set.allocatedSize() >> 20, t.elapsed().as_secs_f64() * 1e3);
            std::mem::forget(set); // keep the set alive for the whole process
        }
        "touch" | "keep" => {
            let t = Instant::now();
            for (_, _, b) in &shards {
                use_once(b);
            }
            println!("touch pass: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        }
        "mlock" | "madvise" => {
            let t = Instant::now();
            for ((name, _, _), host) in shards.iter().zip(&mems) {
                let sl = host.as_slice();
                let (p, n) = (sl.as_ptr() as *mut libc::c_void, sl.len());
                let t1 = Instant::now();
                let rc = if mode == "mlock" {
                    unsafe { libc::mlock(p, n) }
                } else {
                    let rc = unsafe { libc::madvise(p, n, libc::MADV_WILLNEED) };
                    let mut sum = 0u64;
                    for i in (0..n).step_by(16384) {
                        sum = sum.wrapping_add(sl[i] as u64);
                    }
                    std::hint::black_box(sum);
                    rc
                };
                println!("{mode} {name}: rc {rc} errno {} in {:.1} ms", std::io::Error::last_os_error(), t1.elapsed().as_secs_f64() * 1e3);
            }
            println!("{mode} pass: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        }
        "copy" => {
            let t = Instant::now();
            let mut copies = Vec::new();
            for ((name, len, _), host) in shards.iter().zip(&mems) {
                let src = host.as_slice();
                let t1 = Instant::now();
                let dst = be.alloc(*len, MemKind::Device).unwrap();
                be.upload(&dst, 0, src, 0).unwrap();
                println!("copy {name}: alloc+upload {:.1} ms", t1.elapsed().as_secs_f64() * 1e3);
                copies.push((name.clone(), *len, dst));
            }
            be.synchronize(0).unwrap();
            println!("copy pass: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
            shards = copies;
        }
        _ => {}
    }
    if mode == "keep" {
        let period: u64 = std::env::var("KEEP_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
        let what = std::env::var("KEEP_WHAT").unwrap_or_else(|_| "all".into());
        let small2 = be.alloc(4096, MemKind::Device).unwrap();
        let t = Instant::now();
        let (mut n, mut busy) = (0u32, 0.0f64);
        while (t.elapsed().as_millis() as u64) + period <= gap_ms {
            std::thread::sleep(std::time::Duration::from_millis(period));
            let t1 = Instant::now();
            match what.as_str() {
                "all" => shards.iter().for_each(|(_, _, b)| be.copy(&small, 0, b, 0, 16, 0).unwrap()),
                "first" => be.copy(&small, 0, &shards[0].2, 0, 16, 0).unwrap(),
                _ => be.copy(&small, 0, &small2, 0, 16, 0).unwrap(),
            }
            be.synchronize(0).unwrap();
            busy += t1.elapsed().as_secs_f64() * 1e3;
            n += 1;
        }
        println!("keep-warm ({what}, every {period} ms): {n} touches over {gap_ms} ms, {:.2} ms mean round trip", busy / n.max(1) as f64);
        let rest = gap_ms.saturating_sub(t.elapsed().as_millis() as u64);
        std::thread::sleep(std::time::Duration::from_millis(rest));
    } else if gap_ms > 0 {
        std::thread::sleep(std::time::Duration::from_millis(gap_ms));
    }
    for round in 0..2 {
        for (name, len, b) in &shards {
            println!("use {round} {name} ({:.2} GB): {:.2} ms", *len as f64 / 1e9, use_once(b));
        }
    }
}

/// One 1-byte blit out of `b` in a command buffer of its own on the backend's queue, committed and waited for; ms.
#[cfg(target_os = "macos")]
fn blit_once(
    be: &intarsia_backend_metal::MetalBackend,
    b: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
    scratch: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
) -> f64 {
    use objc2_metal::{MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};
    let t = std::time::Instant::now();
    let cb = be.queue().commandBuffer().unwrap();
    let enc = cb.blitCommandEncoder().unwrap();
    unsafe { enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(b, 0, scratch, 0, 1) };
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
    t.elapsed().as_secs_f64() * 1e3
}

#[cfg(target_os = "macos")]
fn raw_modes(
    be: &intarsia_backend_metal::MetalBackend,
    mode: &str,
    mems: &[std::sync::Arc<dyn intarsia_core::backend::HostMemory>],
    shards: &[(String, usize, intarsia_core::backend::Buffer)],
    sleep_ms: u64,
    gap_ms: u64,
) {
    use intarsia_backend_metal::MetalBuffer;
    use intarsia_core::backend::{Backend, MemKind};
    use objc2::runtime::ProtocolObject;
    use objc2_metal::{
        MTLAllocation, MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLDevice,
        MTLResidencySet, MTLResidencySetDescriptor, MTLResourceOptions,
    };
    use std::time::Instant;
    let scratch = be.new_mtl_buffer(16, true).unwrap();
    let wrapped: Vec<_> = shards.iter().map(|(_, _, b)| objc2::rc::Retained::from(b.downcast::<MetalBuffer>().unwrap().mtl())).collect();
    let mut owned_shared = Vec::new(); // keeps backend allocations alive
    let t = Instant::now();
    let bufs: Vec<objc2::rc::Retained<ProtocolObject<dyn MTLBuffer>>> = match mode {
        "copyshared" => {
            let mut v = Vec::new();
            for ((name, len, _), host) in shards.iter().zip(mems) {
                let t1 = Instant::now();
                let dst = be.alloc(*len, MemKind::Device).unwrap();
                be.upload(&dst, 0, host.as_slice(), 0).unwrap();
                v.push(objc2::rc::Retained::from(dst.downcast::<MetalBuffer>().unwrap().mtl()));
                println!("copyshared {name}: alloc+memcpy {:.1} ms", t1.elapsed().as_secs_f64() * 1e3);
                owned_shared.push(dst);
            }
            v
        }
        "copyprivate" => {
            let mut v = Vec::new();
            for ((name, len, _), src) in shards.iter().zip(&wrapped) {
                let t1 = Instant::now();
                let dst = be.device().newBufferWithLength_options(*len, MTLResourceOptions::StorageModePrivate).expect("private buffer");
                let cb = be.queue().commandBuffer().unwrap();
                let enc = cb.blitCommandEncoder().unwrap();
                unsafe { enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(src, 0, &dst, 0, *len) };
                enc.endEncoding();
                cb.commit();
                cb.waitUntilCompleted();
                println!("copyprivate {name}: alloc+blit {:.1} ms", t1.elapsed().as_secs_f64() * 1e3);
                v.push(dst);
            }
            v
        }
        _ => wrapped.clone(),
    };
    println!("{mode}: prepared {} buffers in {:.1} ms", bufs.len(), t.elapsed().as_secs_f64() * 1e3);
    std::thread::sleep(std::time::Duration::from_millis(sleep_ms));
    let set = (mode == "residencycpu").then(|| {
        let desc = MTLResidencySetDescriptor::new();
        let set = be.device().newResidencySetWithDescriptor_error(&desc).expect("residency set");
        for b in &bufs {
            set.addAllocation(ProtocolObject::<dyn MTLAllocation>::from_ref(&**b));
        }
        set.commit();
        let t = Instant::now();
        set.requestResidency();
        be.queue().addResidencySet(&set);
        println!("residency set requested+added in {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        set
    });
    // touch pass: every buffer used once by the GPU before the gap (this is the wiring cost, once)
    let tp: Vec<f64> = bufs.iter().map(|b| blit_once(be, b, &scratch)).collect();
    println!("touch pass ms: {}", tp.iter().map(|x| format!("{x:.1}")).collect::<Vec<_>>().join(" / "));
    let t = Instant::now();
    let period: u64 = std::env::var("KEEP_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(500);
    if let Some(set) = &set {
        let mut n = 0;
        while (t.elapsed().as_millis() as u64) + period <= gap_ms {
            std::thread::sleep(std::time::Duration::from_millis(period));
            set.requestResidency();
            n += 1;
        }
        println!("requestResidency x{n} (CPU only, every {period} ms)");
    }
    let rest = gap_ms.saturating_sub(t.elapsed().as_millis() as u64);
    std::thread::sleep(std::time::Duration::from_millis(rest));
    for round in 0..2 {
        for ((name, len, _), b) in shards.iter().zip(&bufs) {
            println!("use {round} {name} ({:.2} GB): {:.2} ms", *len as f64 / 1e9, blit_once(be, b, &scratch));
        }
    }
    if let Some(set) = set {
        std::mem::forget(set);
    }
}
