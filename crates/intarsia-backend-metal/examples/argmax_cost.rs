//! Lane ink-hostgap: GPU time of one f32 argmax dispatch over a logits row, the scalar 256-thread `argmax_f32`, `argmax_f32_v4`
//! (V % 4 == 0 only) and `argmax_f32_v4t` (any V), at Inkling-2-Mini's unpadded head (V = 200058) and Qwen3's 151936. `N`
//! dependent-free dispatches in one serial command buffer, GPU timestamps / N, median of `REPS`; "hot" reads one row
//! every time, "rotating" walks 48 rows (about 38 MB, so most reads miss the caches the previous dispatch filled).
//!
//!   cargo run --release -p intarsia-backend-metal --example argmax_cost
//!
//! Run it inside a fleetq slot (`fleetq run --host endor --quiet gpu,cpu ...`).

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("argmax_cost needs macOS / Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    imp::main()
}

#[cfg(target_os = "macos")]
mod imp {
    use intarsia_backend_metal::bench::{gpu_seconds, stats, Rng};
    use intarsia_backend_metal::{set_bytes, MetalBackend};
    use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLDevice, MTLDispatchType, MTLSize};

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct OpP {
        dims: [u32; 8],
        skips: [u32; 5],
        f0: f32,
        f1: f32,
        pad: u32,
    }

    const N: usize = 200;
    const REPS: usize = 15;
    const ROT: usize = 48;

    pub fn main() {
        let be = MetalBackend::new().expect("metal backend");
        println!("device: {}; {N} dispatches per command buffer, median of {REPS} (us per dispatch)", be.device().name());
        println!("{:<34} {:>8} {:>10} {:>10}", "kernel", "V", "hot", "rotating");
        let mut rng = Rng(5);
        for v in [200058usize, 151936] {
            let rows: Vec<_> = (0..ROT)
                .map(|_| {
                    let b = be.new_mtl_buffer(v * 4, true).expect("buffer");
                    let p = b.contents().as_ptr() as *mut f32;
                    for i in 0..v {
                        // SAFETY: the buffer holds v floats.
                        unsafe { *p.add(i) = rng.unit() };
                    }
                    b
                })
                .collect();
            let out = be.new_mtl_buffer(64, true).expect("buffer");
            let p = OpP { dims: [v as u32, 0, 0, 0, 0, 0, 0, 0], ..Default::default() };
            let run = |name: &str, threads: usize, rot: usize| -> f64 {
                let pso = be.pso(name).expect("pso");
                let mut t = Vec::new();
                for rep in 0..REPS + 2 {
                    let cb = be.queue().commandBuffer().unwrap();
                    let enc = cb.computeCommandEncoderWithDispatchType(MTLDispatchType::Serial).unwrap();
                    enc.setComputePipelineState(pso);
                    for i in 0..N {
                        // SAFETY: the buffers cover the row and the output.
                        unsafe {
                            enc.setBuffer_offset_atIndex(Some(&rows[i % rot]), 0, 0);
                            enc.setBuffer_offset_atIndex(Some(&out), 0, 1);
                        }
                        set_bytes(&enc, &p, 2);
                        enc.dispatchThreadgroups_threadsPerThreadgroup(
                            MTLSize { width: 1, height: 1, depth: 1 },
                            MTLSize { width: threads, height: 1, depth: 1 },
                        );
                    }
                    enc.endEncoding();
                    cb.commit();
                    cb.waitUntilCompleted();
                    if rep >= 2 {
                        t.push(gpu_seconds(&cb) * 1e6 / N as f64);
                    }
                }
                stats(&mut t).median
            };
            let mut kernels = vec![("argmax_f32 (scalar)", "argmax_f32", 256usize), ("argmax_f32_v4t", "argmax_f32_v4t", 1024)];
            if v % 4 == 0 {
                kernels.insert(1, ("argmax_f32_v4", "argmax_f32_v4", 1024));
            }
            for (label, name, threads) in kernels {
                println!("{label:<34} {v:>8} {:>8.2}us {:>8.2}us", run(name, threads, 1), run(name, threads, ROT));
            }
        }
    }
}
