//! P1 spike: NAX bf16 GEMM probe / microbench / error table. See docs/results/2026-10-14-metal-prefill-p1-spike.md.
#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("gemm_nax needs macOS / Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    imp::main()
}

#[cfg(target_os = "macos")]
mod imp {
    use intarsia_backend_metal::bench::{self, Rng};
    use intarsia_backend_metal::{set_bytes, MetalBackend};
    use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

    pub fn main() {
        let be = MetalBackend::new().expect("metal backend");
        println!("has_metal4 = {}", be.has_metal4());
        let pso = be.pso("gemm_nax_probe").expect("gemm_nax_probe pipeline");
        println!("probe pipeline: max threads {} width {}", pso.maxTotalThreadsPerThreadgroup(), pso.threadExecutionWidth());
        let (m, n, k) = (100usize, 96usize, 64usize);
        let a = be.new_mtl_buffer(m * k * 2, true).unwrap();
        let w = be.new_mtl_buffer(n * k * 2, true).unwrap();
        let c = be.new_mtl_buffer(m * n * 2, true).unwrap();
        let mut r = Rng(12345);
        let av: Vec<u16> = (0..m * k).map(|_| bench::f32_to_bf16(r.unit())).collect();
        let wv: Vec<u16> = (0..n * k).map(|_| bench::f32_to_bf16(r.unit())).collect();
        unsafe {
            std::ptr::copy_nonoverlapping(av.as_ptr(), a.contents().as_ptr() as *mut u16, av.len());
            std::ptr::copy_nonoverlapping(wv.as_ptr(), w.contents().as_ptr() as *mut u16, wv.len());
            std::ptr::write_bytes(c.contents().as_ptr() as *mut u8, 0, m * n * 2);
        }
        let cb = be
            .encode_and_commit(|enc| {
                enc.setComputePipelineState(pso);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&a), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&w), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(&c), 0, 2);
                }
                set_bytes(enc, &[m as u32, n as u32, k as u32, 0u32], 3);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: n.div_ceil(32), height: m.div_ceil(64), depth: 1 },
                    MTLSize { width: 128, height: 1, depth: 1 },
                );
            })
            .unwrap();
        MetalBackend::wait(&cb).unwrap();
        let out: Vec<u16> = unsafe { std::slice::from_raw_parts(c.contents().as_ptr() as *const u16, m * n).to_vec() };
        let mut max_abs = 0f64;
        let mut max_ref = 0f64;
        for i in 0..m {
            for j in 0..n {
                let mut s = 0f64;
                for kk in 0..k {
                    s += bench::bf16_to_f32(av[i * k + kk]) as f64 * bench::bf16_to_f32(wv[j * k + kk]) as f64;
                }
                let g = bench::bf16_to_f32(out[i * n + j]) as f64;
                max_abs = max_abs.max((g - s).abs());
                max_ref = max_ref.max(s.abs());
            }
        }
        println!("probe M={m} N={n} K={k}: max_abs {max_abs:.4e} (max |ref| {max_ref:.3})");
    }
}
