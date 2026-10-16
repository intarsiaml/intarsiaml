//! First-use latency of mmap'd weight shards on the GPU. A cold process pays about 140 ms before a Qwen3-8B prefill gets
//! going (command buffer 0 starts ~100 ms after its commit, buffers 1 and 2 ~25 ms later each): this wraps the shards of a
//! model directory the way the loader does and times one 16-byte GPU copy out of each, twice, after an optional
//! pre-wiring step.
//!
//!   wire_probe <model dir> [none|residency|touch] [sleep ms]
//!
//!   none       time the first and second use of every shard
//!   residency  an MTLResidencySet holding every shard (requestResidency, queue.addResidencySet) before the timed uses
//!   touch      one GPU copy per shard (not timed) before the timed uses
//!
//! `sleep ms` waits that long after wrapping (a loader doing other work) before the timed uses; the probe pre-wires wrapped
//! files (`MetalBackend::set_prewire`) only with `INTARSIA_METAL_PREWIRE=1` in its environment.
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
    let shards: Vec<_> = files
        .iter()
        .map(|p| {
            let mem: Arc<dyn HostMemory> = Arc::new(MmapHost(unsafe { Mmap::map(&std::fs::File::open(p).unwrap()).unwrap() }));
            let len = mem.as_slice().len();
            let b = be.wrap_host(mem, 0, len).expect("wrap");
            (p.file_name().unwrap().to_string_lossy().to_string(), len, b)
        })
        .collect();
    println!("mapped and wrapped {} shards in {:.1} ms (mode {mode}, sleep {sleep_ms} ms)", shards.len(), t0.elapsed().as_secs_f64() * 1e3);
    std::thread::sleep(std::time::Duration::from_millis(sleep_ms));
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
            let set = be.device().newResidencySetWithDescriptor_error(&desc).expect("residency set");
            for (_, _, b) in &shards {
                let mb = b.downcast::<MetalBuffer>().unwrap();
                set.addAllocation(objc2::runtime::ProtocolObject::<dyn MTLAllocation>::from_ref(mb.mtl()));
            }
            set.commit();
            let t1 = t.elapsed().as_secs_f64() * 1e3;
            set.requestResidency();
            be.queue().addResidencySet(&set);
            println!("residency set built+committed in {t1:.2} ms, requested, added to queue at {:.2} ms", t.elapsed().as_secs_f64() * 1e3);
            // wait until resident (allocatedSize grows to the shards' total)
            let t = Instant::now();
            loop {
                std::thread::sleep(std::time::Duration::from_millis(5));
                if t.elapsed().as_secs() > 20 || set.allocatedSize() >= shards.iter().map(|s| s.1 as u64).sum::<u64>() / 2 {
                    break;
                }
            }
            println!("residency set allocatedSize {} MB after {:.0} ms", set.allocatedSize() >> 20, t.elapsed().as_secs_f64() * 1e3);
        }
        "touch" => {
            let t = Instant::now();
            for (_, _, b) in &shards {
                use_once(b);
            }
            println!("touch pass: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        }
        _ => {}
    }
    for round in 0..2 {
        for (name, len, b) in &shards {
            println!("use {round} {name} ({:.2} GB): {:.2} ms", *len as f64 / 1e9, use_once(b));
        }
    }
}
