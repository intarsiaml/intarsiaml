//! Metal phase B2b: what does one dispatch of a tiny kernel cost inside a command buffer? Chains `N` dependent
//! dispatches of (a) a near-empty kernel (`add_bf16` over 32 elements), (b) `rmsnorm_add_bf16` over an 8 KB row
//! (H = 4096, the Qwen3-8B decode shape) at several threadgroup widths, and (c) `rmsnorm_bf16`, on
//!   * a serial encoder (`MTLDispatchTypeSerial`, what the backend stream uses),
//!   * a concurrent encoder with a buffer-scope memory barrier after every dispatch (the safe baseline),
//!   * a concurrent encoder without barriers (lower bound: dispatches may overlap, results are not ordered).
//! Times are GPU timestamps of the command buffer divided by `N`; the median of the repetitions is printed.
//!
//!   cargo run --release -p intarsia-backend-metal --example dispatch_cost
//!
//! Run it inside a fleetq slot (`fleetq run --host endor --exclusive ...`).

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("dispatch_cost needs macOS / Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    imp::main()
}

#[cfg(target_os = "macos")]
mod imp {
    use intarsia_backend_metal::bench::{gpu_seconds, stats};
    use intarsia_backend_metal::{set_bytes, MetalBackend};
    use objc2_metal::{
        MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
        MTLDevice, MTLDispatchType, MTLSize,
    };

    /// `OpP` of kernels/metal/common.h.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct OpP {
        dims: [u32; 8],
        skips: [u32; 5],
        f0: f32,
        f1: f32,
        pad: u32,
    }

    #[derive(Clone, Copy, PartialEq)]
    enum Mode {
        Serial,
        ConcBarrier,
        ConcNone,
    }
    const MODES: [(Mode, &str); 3] = [
        (Mode::Serial, "serial"),
        (Mode::ConcBarrier, "concurrent+barrier"),
        (Mode::ConcNone, "concurrent, no barrier"),
    ];

    const N: usize = 2000;
    const REPS: usize = 15;
    const H: usize = 4096;

    pub fn main() {
        let be = MetalBackend::new().expect("metal backend");
        println!("device: {}; {N} chained dispatches per command buffer, median of {REPS}", be.device().name());
        let mk = |bytes: usize| be.new_mtl_buffer(bytes, true).expect("buffer");
        let (a, b, c, d, w) = (mk(H * 2), mk(H * 2), mk(H * 2), mk(H * 2), mk(H * 2));
        // sane data: bf16 1.0 everywhere
        for buf in [&a, &b, &c, &d, &w] {
            let p = buf.contents().as_ptr() as *mut u16;
            for i in 0..H {
                unsafe { *p.add(i) = 0x3f80 };
            }
        }
        let run = |name: &str, bufs: &[&_], threads: usize, groups: usize, p: &OpP, mode: Mode, n: usize| -> f64 {
            let pso = be.pso(name).expect("pso");
            let mut v = Vec::new();
            for rep in 0..REPS + 2 {
                let cb = be.queue().commandBuffer().unwrap();
                let enc = match mode {
                    Mode::Serial => cb.computeCommandEncoderWithDispatchType(MTLDispatchType::Serial),
                    _ => cb.computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent),
                }
                .unwrap();
                enc.setComputePipelineState(pso);
                for i in 0..n {
                    // ping-pong the data input/output so consecutive dispatches are dependent
                    for (j, bf) in bufs.iter().enumerate() {
                        unsafe { enc.setBuffer_offset_atIndex(Some(*bf), 0, j) };
                    }
                    let _ = i;
                    set_bytes(&enc, p, bufs.len());
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize { width: groups, height: 1, depth: 1 },
                        MTLSize { width: threads, height: 1, depth: 1 },
                    );
                    if mode == Mode::ConcBarrier {
                        enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
                    }
                }
                enc.endEncoding();
                cb.commit();
                cb.waitUntilCompleted();
                if rep >= 2 {
                    v.push(gpu_seconds(&cb) * 1e6 / n as f64);
                }
            }
            stats(&mut v).median
        };
        let p_add = OpP { dims: [32, 0, 0, 0, 0, 0, 0, 0], ..Default::default() };
        let p_norm = OpP { dims: [H as u32, 0, 0, 0, 0, 0, 0, 0], f0: 1e-6, ..Default::default() };
        println!("\n{:<44} {:>10} {:>20} {:>24}", "kernel", "serial", "concurrent+barrier", "concurrent, no barrier");
        let row = |label: &str, f: &dyn Fn(Mode) -> f64| {
            let r: Vec<f64> = MODES.iter().map(|&(m, _)| f(m)).collect();
            println!("{label:<44} {:>8.2}us {:>18.2}us {:>22.2}us", r[0], r[1], r[2]);
        };
        // add_bf16: grid of 1 threadgroup x 32 threads, 32 elements: as close to empty as a real kernel gets
        row("add_bf16, 1 tg x 32 thr (near-empty)", &|m| run("add_bf16", &[&a, &b, &c], 32, 1, &p_add, m, N));
        for threads in [64usize, 128, 256, 512, 1024] {
            row(&format!("rmsnorm_add_bf16 H=4096, {threads} thr"), &|m| {
                run("rmsnorm_add_bf16", &[&a, &b, &w, &c, &d], threads, 1, &p_norm, m, N)
            });
        }
        for threads in [128usize, 256, 512, 1024] {
            row(&format!("rmsnorm_bf16 H=4096, {threads} thr"), &|m| {
                run("rmsnorm_bf16", &[&a, &w, &c], threads, 1, &p_norm, m, N)
            });
        }
        // the phase B2b forms: same element-to-thread assignment and result bits, all loads issued before the reduction
        row("rmsnorm_add_bf16_ex16 H=4096, 256 thr (new)", &|m| {
            run("rmsnorm_add_bf16_ex16", &[&a, &b, &w, &c, &d], 256, 1, &p_norm, m, N)
        });
        row("rmsnorm_bf16_ex16 H=4096, 256 thr (new)", &|m| {
            run("rmsnorm_bf16_ex16", &[&a, &w, &c], 256, 1, &p_norm, m, N)
        });
        let p_qk8 = OpP { dims: [128, 0, 0, 0, 0, 0, 0, 0], f0: 1e-6, ..Default::default() };
        row("rmsnorm_bf16_ex1 (qk_norm) 32 rows x 128, 128 thr (new)", &|m| {
            run("rmsnorm_bf16_ex1", &[&a, &w, &c], 128, 32, &p_qk8, m, N)
        });
        // rope at the decode q shape [1, 32, 128] (already at the dispatch floor; a pair-per-thread form measured 2.0 us)
        let p_rope = OpP { dims: [1, 32, 128, 128, 0, 130, 0, 0], f0: 1e6, ..Default::default() };
        let big = mk(32 * 128 * 2);
        let big2 = mk(32 * 128 * 2);
        row("rope_bf16 q [1,32,128], 16 tg x 256 thr", &|m| {
            run("rope_bf16", &[&big, &big2, &a], 256, 16, &p_rope, m, N)
        });
        // argmax over the Qwen3 vocabulary (f32 logits)
        let logits = mk(151936 * 4);
        let outi = mk(64);
        let p_am = OpP { dims: [151936, 0, 0, 0, 0, 0, 0, 0], ..Default::default() };
        row("argmax_f32 V=151936, 256 thr", &|m| run("argmax_f32", &[&logits, &outi], 256, 1, &p_am, m, 200));
        row("argmax_f32_v4 V=151936, 1024 thr (new)", &|m| {
            run("argmax_f32_v4", &[&logits, &outi], 1024, 1, &p_am, m, 200)
        });
        // qk-norm shaped: 32 rows of 128
        let p_qk = OpP { dims: [128, 0, 0, 0, 0, 0, 0, 0], f0: 1e-6, ..Default::default() };
        for threads in [32usize, 128, 256] {
            row(&format!("rmsnorm_bf16 (qk_norm) 32 rows x 128, {threads} thr"), &|m| {
                run("rmsnorm_bf16", &[&a, &w, &c], threads, 32, &p_qk, m, N)
            });
        }
    }
}
