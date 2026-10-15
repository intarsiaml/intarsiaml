//! TEMPORARY: simdgroup_matrix throughput microbenchmark (kernels/metal/attention_flash_micro.metal).
#[cfg(not(target_os = "macos"))]
fn main() {}

#[cfg(target_os = "macos")]
fn main() {
    use intarsia_backend_metal::bench;
    use intarsia_backend_metal::flash::MICRO_SHADERS;
    use intarsia_backend_metal::MetalBackend;
    use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};
    use std::ptr::NonNull;

    let be = MetalBackend::new().expect("metal");
    let sink = be.new_mtl_buffer(1 << 24, true).unwrap();
    let iters: u32 = 20000;
    println!("{:<16} {:>8} {:>10} {:>10}", "kernel", "tg/thr", "ms", "TFLOPS");
    for &name in MICRO_SHADERS {
        let nacc: usize = if name.ends_with("16") { 16 } else { 8 };
        for (tgs, thr) in [(40usize * 4, 128usize), (40 * 8, 256), (40 * 16, 256)] {
            let pso = be.pso(name).unwrap();
            let mut p = intarsia_backend_metal::OpP::default();
            p.dims[0] = iters;
            let mut times = Vec::new();
            for _ in 0..5 {
                let cb = be
                    .encode_and_commit(|enc| {
                        enc.setComputePipelineState(pso);
                        unsafe {
                            enc.setBuffer_offset_atIndex(Some(&sink), 0, 0);
                            enc.setBytes_length_atIndex(
                                NonNull::new(&mut p as *mut _ as *mut std::ffi::c_void).unwrap(),
                                std::mem::size_of::<intarsia_backend_metal::OpP>(),
                                1,
                            );
                        }
                        enc.dispatchThreadgroups_threadsPerThreadgroup(
                            MTLSize { width: tgs, height: 1, depth: 1 },
                            MTLSize { width: thr, height: 1, depth: 1 },
                        );
                    })
                    .unwrap();
                MetalBackend::wait(&cb).unwrap();
                times.push(bench::gpu_seconds(&cb));
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let t = times[2];
            let mmas = (tgs * thr / 32) as f64 * iters as f64 * nacc as f64;
            let _ = pso.maxTotalThreadsPerThreadgroup();
            println!("{:<16} {:>4}/{:<4} {:>10.2} {:>10.2}", name, tgs, thr, t * 1e3, mmas * 1024.0 / t / 1e12);
        }
    }
}
