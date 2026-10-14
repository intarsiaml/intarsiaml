//! Phase B part 3 GEMV efficiency bench: the Qwen3-8B decode GEMV shapes in a chain-serial stream (one command buffer,
//! serial dispatch, distinct weights per dispatch, i.e. what the decode stream does), several kernel candidates per shape
//! measured interleaved round by round (A B C A B C ...) so drift hits all candidates alike.
//!
//!   cargo run --release -p intarsia-backend-metal --example gemv_b3 -- [case ...]
//!   cases: qo (4096x4096), down (K=12288, N=4096), kv (pair 1024+1024, K=4096), gu (pair 12288+12288, K=4096), lm
//!   env: B3_ROUNDS (default 30), B3_POOL_MIB (default 3072), B3_ONLY=<name suffixes, comma separated> (candidate filter; first candidate always kept), B3_A8=1 (weights
//!   bound at byte offset 8, 8-byte-load kernels only)
//!
//! Each round samples the GPU's `Device Utilization %` (ioreg) before measuring; a round whose idle reading is above
//! 10 is discarded and redone. Run it inside a fleetq slot.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("gemv_b3 needs macOS / Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    imp::main()
}

#[cfg(target_os = "macos")]
mod imp {
    use intarsia_backend_metal::bench::{self, stats};
    use intarsia_backend_metal::{encode_gemv, set_bytes, GemvParams, MetalBackend, MtlBuffer};
    use objc2::runtime::ProtocolObject;
    use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize};

    const PEAK_GBS: f64 = 614.0;

    #[derive(Clone, Copy, PartialEq)]
    enum Kind {
        /// `gemv_bf16_*`: `rows_per_tg` rows per threadgroup (`GemvParams`).
        Single,
        /// `gemv_pair_bf16_*`: `rows_per_tg` rows per threadgroup per matrix, rounded up per matrix (`PairParams`).
        Pair,
    }

    #[derive(Clone, Copy)]
    struct Cand {
        name: &'static str,
        kind: Kind,
        rows_per_tg: u32,
        threads: u32,
        /// Byte offset the weights are bound at (8 for the 8-byte-load kernels).
        off: usize,
    }

    const fn single(name: &'static str, rows_per_tg: u32, sg: u32) -> Cand {
        Cand { name, kind: Kind::Single, rows_per_tg, threads: 32 * sg, off: 0 }
    }
    const fn pair(name: &'static str, rows_per_tg: u32, sg: u32) -> Cand {
        Cand { name, kind: Kind::Pair, rows_per_tg, threads: 32 * sg, off: 0 }
    }

    fn single_cands() -> Vec<Cand> {
        vec![
            single("gemv_bf16_r1u4s4", 4, 4), // current default
            single("gemv_bf16_r1u4s2", 2, 2),
            single("gemv_bf16_r1u4s1", 1, 1),
            single("gemv_bf16_r1u2s4", 4, 4),
            single("gemv_bf16_r1u8s4", 4, 4),
            single("gemv_bf16_r1u8s2", 2, 2),
            single("gemv_bf16_r1u2s2", 2, 2),
            single("gemv_bf16_rt_u1s4", 1, 4),
            single("gemv_bf16_rt_u2s4", 1, 4),
            single("gemv_bf16_rt_u4s4", 1, 4),
            single("gemv_bf16_rt_u1s8", 1, 8),
            single("gemv_bf16_rt_u2s8", 1, 8),
            single("gemv_bf16_rt_u1s2", 1, 2),
            single("gemv_bf16_rt_u2s2", 1, 2),
            single("gemv_bf16_rt_u4s2", 1, 2),
            single("gemv_bf16_rt_u8s2", 1, 2),
            single("gemv_bf16_rt_u1s16", 1, 16),
            single("gemv_bf16_rt_u2s16", 1, 16),
            single("gemv_bf16_rt_u3s16", 1, 16),
            single("gemv_bf16_rt_u3s8", 1, 8),
            single("gemv_bf16_rt_u6s8", 1, 8),
            single("gemv_bf16_rt_u3s4", 1, 4),
            single("gemv_bf16_rt_u4s8", 1, 8),
        ]
    }

    fn pair_cands() -> Vec<Cand> {
        vec![
            pair("gemv_pair_bf16", 4, 4), // current
            pair("gemv_pair_bf16_u4s2", 2, 2),
            pair("gemv_pair_bf16_u4s1", 1, 1),
            pair("gemv_pair_bf16_u2s4", 4, 4),
            pair("gemv_pair_bf16_u8s4", 4, 4),
            pair("gemv_pair_bf16_rt_u1s4", 1, 4),
            pair("gemv_pair_bf16_rt_u2s4", 1, 4),
            pair("gemv_pair_bf16_rt_u4s4", 1, 4),
            pair("gemv_pair_bf16_rt_u1s8", 1, 8),
            pair("gemv_pair_bf16_rt_u2s8", 1, 8),
            pair("gemv_pair_bf16_rt_u2s2", 1, 2),
            pair("gemv_pair_bf16_rt_u4s2", 1, 2),
            pair("gemv_pair_bf16_rt_u1s16", 1, 16),
            pair("gemv_pair_bf16_rt_u2s16", 1, 16),
            pair("gemv_pair_bf16_rt_u4s8", 1, 8),
        ]
    }

    struct Case {
        label: &'static str,
        k: usize,
        n0: usize,
        n1: usize, // 0 for single
    }

    const CASES: [Case; 5] = [
        Case { label: "qo", k: 4096, n0: 4096, n1: 0 },
        Case { label: "down", k: 12288, n0: 4096, n1: 0 },
        Case { label: "kv", k: 4096, n0: 1024, n1: 1024 },
        Case { label: "gu", k: 4096, n0: 12288, n1: 12288 },
        Case { label: "lm", k: 4096, n0: 151936, n1: 0 },
    ];

    struct Pool {
        w0: Vec<MtlBuffer>,
        w1: Vec<MtlBuffer>,
        x: MtlBuffer,
        y0: MtlBuffer,
        y1: MtlBuffer,
    }

    impl Pool {
        fn new(be: &MetalBackend, c: &Case, pool_bytes: usize, off: usize) -> Pool {
            let wb0 = c.n0 * c.k * 2;
            let wb1 = c.n1 * c.k * 2;
            let count = pool_bytes.div_ceil(wb0 + wb1).max(2);
            let (mut w0, mut w1): (Vec<MtlBuffer>, Vec<MtlBuffer>) = (Vec::new(), Vec::new());
            for i in 0..count {
                for (wb, w) in [(wb0, &mut w0), (wb1, &mut w1)] {
                    if wb == 0 {
                        continue;
                    }
                    let b = be.new_mtl_buffer(wb + 64, true).unwrap();
                    // SAFETY: wb bytes inside the allocation (64 spare bytes for the offset).
                    unsafe {
                        let dst = (b.contents().as_ptr() as *mut u8).add(off);
                        if i == 0 {
                            bench::fill_bf16_random(dst as *mut u16, wb / 2, 42 + w.len() as u64);
                        } else {
                            std::ptr::copy_nonoverlapping((w[0].contents().as_ptr() as *const u8).add(off), dst, wb);
                        }
                    }
                    w.push(b);
                }
            }
            let x = be.new_mtl_buffer(c.k * 2, true).unwrap();
            let y0 = be.new_mtl_buffer(c.n0 * 2, true).unwrap();
            let y1 = be.new_mtl_buffer((c.n1 * 2).max(16), true).unwrap();
            // SAFETY: k*2 bytes.
            unsafe { bench::fill_bf16_random(x.contents().as_ptr() as *mut u16, c.k, 7) };
            Pool { w0, w1, x, y0, y1 }
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct PairP {
        k: u32,
        n0: u32,
        n1: u32,
        y0_skip: u32,
        y1_skip: u32,
        pad: [u32; 3],
    }

    fn dispatch(
        be: &MetalBackend,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        c: &Case,
        cand: &Cand,
        pool: &Pool,
        i: usize,
    ) {
        let pso = be.pso(cand.name).unwrap_or_else(|| panic!("no pso {}", cand.name));
        match cand.kind {
            Kind::Single => {
                let p = GemvParams { k: c.k as u32, n: c.n0 as u32, ..Default::default() };
                encode_gemv(
                    enc,
                    pso,
                    (&pool.w0[i], cand.off),
                    (&pool.x, 0),
                    (&pool.y0, 0),
                    &p,
                    (c.n0 as u32).div_ceil(cand.rows_per_tg),
                    cand.threads,
                );
            }
            Kind::Pair => {
                let p = PairP { k: c.k as u32, n0: c.n0 as u32, n1: c.n1 as u32, ..Default::default() };
                let groups = (c.n0 as u32).div_ceil(cand.rows_per_tg) + (c.n1 as u32).div_ceil(cand.rows_per_tg);
                enc.setComputePipelineState(pso);
                // SAFETY: buffers sized by Pool::new; offsets as the kernel's alignment contract.
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&pool.w0[i]), cand.off, 0);
                    enc.setBuffer_offset_atIndex(Some(&pool.w1[i]), cand.off, 1);
                    enc.setBuffer_offset_atIndex(Some(&pool.x), 0, 2);
                    enc.setBuffer_offset_atIndex(Some(&pool.y0), 0, 3);
                    enc.setBuffer_offset_atIndex(Some(&pool.y1), 0, 4);
                }
                set_bytes(enc, &p, 5);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: groups as usize, height: 1, depth: 1 },
                    MTLSize { width: cand.threads as usize, height: 1, depth: 1 },
                );
            }
        }
    }

    fn chain_seconds(be: &MetalBackend, c: &Case, cand: &Cand, pool: &Pool, count: usize) -> f64 {
        let cb = be.queue().commandBuffer().unwrap();
        let enc = cb.computeCommandEncoder().unwrap(); // serial dispatch
        for i in 0..count {
            dispatch(be, &enc, c, cand, pool, i % pool.w0.len());
        }
        enc.endEncoding();
        be.commit(&cb);
        MetalBackend::wait(&cb).unwrap();
        bench::gpu_seconds(&cb)
    }

    /// `Device Utilization %` of the GPU right now (ioreg), or None.
    fn gpu_util() -> Option<u32> {
        let out = std::process::Command::new("ioreg").args(["-r", "-c", "IOAccelerator", "-d", "1"]).output().ok()?;
        let s = String::from_utf8_lossy(&out.stdout);
        let key = "\"Device Utilization %\"=";
        let at = s.find(key)? + key.len();
        let digits: String = s[at..].chars().take_while(|c| c.is_ascii_digit()).collect();
        digits.parse().ok()
    }

    fn bf16_max_diff(a: &[u16], b: &[u16]) -> (usize, f32) {
        let mut nd = 0;
        let mut mx = 0f32;
        for (x, y) in a.iter().zip(b) {
            if x != y {
                nd += 1;
                mx = mx.max((bench::bf16_to_f32(*x) - bench::bf16_to_f32(*y)).abs());
            }
        }
        (nd, mx)
    }

    pub fn main() {
        let be = MetalBackend::new().expect("metal backend");
        let want: Vec<String> = std::env::args().skip(1).collect();
        let rounds: usize = std::env::var("B3_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(30);
        let pool_bytes: usize =
            std::env::var("B3_POOL_MIB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(3072) << 20;
        let only = std::env::var("B3_ONLY").ok();
        let a8 = std::env::var("B3_A8").is_ok();
        println!("rounds {rounds}, pool {} MiB, a8 {a8}", pool_bytes >> 20);
        for c in CASES.iter().filter(|c| want.is_empty() || want.iter().any(|w| w == c.label)) {
            let pool = Pool::new(&be, c, pool_bytes, 0);
            let is_pair = c.n1 > 0;
            let mut cands: Vec<Cand> = if is_pair { pair_cands() } else { single_cands() };
            if let Some(o) = &only {
                let base = cands[0];
                cands.retain(|x| o.split(',').any(|t| x.name.ends_with(t) || x.name == t));
                if !cands.iter().any(|x| x.name == base.name) {
                    cands.insert(0, base);
                }
            }
            if a8 {
                cands = a8_cands(is_pair);
            }
            let step_bytes = ((c.n0 + c.n1) * c.k * 2 + c.k * 2 + (c.n0 + c.n1) * 2) as f64;
            let count = pool.w0.len();
            println!(
                "\n== case {} K={} N={}+{} : {:.1} MiB per dispatch x {} dispatches per chain ==",
                c.label,
                c.k,
                c.n0,
                c.n1,
                step_bytes / 1048576.0,
                count
            );
            // correctness vs the first candidate, on pool entry 0
            let mut outs: Vec<(Vec<u16>, Vec<u16>)> = Vec::new();
            for cand in &cands {
                // SAFETY: y buffers hold n0 / n1 u16.
                unsafe {
                    std::ptr::write_bytes(pool.y0.contents().as_ptr() as *mut u8, 0, c.n0 * 2);
                    std::ptr::write_bytes(pool.y1.contents().as_ptr() as *mut u8, 0, c.n1 * 2);
                }
                chain_seconds(&be, c, cand, &pool, 1);
                // SAFETY: as above.
                unsafe {
                    let y0 = std::slice::from_raw_parts(pool.y0.contents().as_ptr() as *const u16, c.n0).to_vec();
                    let y1 = std::slice::from_raw_parts(pool.y1.contents().as_ptr() as *const u16, c.n1).to_vec();
                    outs.push((y0, y1));
                }
            }
            let mut samples: Vec<Vec<f64>> = vec![Vec::new(); cands.len()];
            let mut utils: Vec<u32> = Vec::new();
            let mut discarded = 0;
            // warm-up
            for cand in &cands {
                for _ in 0..2 {
                    chain_seconds(&be, c, cand, &pool, count.min(40));
                }
            }
            let mut round = 0;
            let mut order: Vec<usize> = (0..cands.len()).collect();
            while round < rounds && discarded < 200 {
                let u = gpu_util().unwrap_or(999);
                if u > 10 {
                    discarded += 1;
                    std::thread::sleep(std::time::Duration::from_millis(300));
                    continue;
                }
                utils.push(u);
                order.rotate_left(1);
                for &ci in &order {
                    let secs = chain_seconds(&be, c, &cands[ci], &pool, count);
                    samples[ci].push(count as f64 * step_bytes / secs / 1e9);
                }
                round += 1;
            }
            println!("idle utilization readings per round: {:?}; waited-for-idle polls (reading >10%): {discarded}", utils);
            let base_med = stats(&mut samples[0].clone()).median;
            for (ci, cand) in cands.iter().enumerate() {
                let s = stats(&mut samples[ci]);
                let us = step_bytes / (s.median * 1e9) * 1e6;
                let (nd, mx) = bf16_max_diff(&outs[ci].0, &outs[0].0);
                let (nd1, mx1) = bf16_max_diff(&outs[ci].1, &outs[0].1);
                println!(
                    "RESULT case={} cand={} median={:.1} p10={:.1} p90={:.1} pct_peak={:.1} us_per_dispatch={:.2} vs_first={:+.1}% diff_elems={} max_abs_diff={:.3e}",
                    c.label,
                    cand.name,
                    s.median,
                    s.p10,
                    s.p90,
                    100.0 * s.median / PEAK_GBS,
                    us,
                    100.0 * (s.median / base_med - 1.0),
                    nd + nd1,
                    mx.max(mx1)
                );
            }
        }
    }

    fn a8_cands(is_pair: bool) -> Vec<Cand> {
        let c = |name: &'static str, kind, rows_per_tg, sg: u32, off| Cand { name, kind, rows_per_tg, threads: 32 * sg, off };
        if is_pair {
            vec![
                c("gemv_pair_bf16", Kind::Pair, 4, 4, 0), // aligned 16-byte reference
                c("gemv_pair_bf16_a8", Kind::Pair, 4, 4, 8), // current a8
                c("gemv_pair_bf16_rt_u2s8", Kind::Pair, 1, 8, 0), // aligned rt reference
                c("gemv_pair_bf16_rt_a8u4s8", Kind::Pair, 1, 8, 8),
                c("gemv_pair_bf16_rt_a8u8s4", Kind::Pair, 1, 4, 8),
                c("gemv_pair_bf16_rt_a8u2s8", Kind::Pair, 1, 8, 8),
                c("gemv_pair_bf16_rt_a8u4s4", Kind::Pair, 1, 4, 8),
                c("gemv_pair_bf16_rt_a8u8s8", Kind::Pair, 1, 8, 8),
            ]
        } else {
            vec![
                c("gemv_bf16_r1u4s4", Kind::Single, 4, 4, 0), // aligned 16-byte reference
                c("gemv_bf16_a8u8s4", Kind::Single, 4, 4, 8), // current a8
                c("gemv_bf16_rt_u2s8", Kind::Single, 1, 8, 0), // aligned rt reference
                c("gemv_bf16_rt_a8u4s8", Kind::Single, 1, 8, 8),
                c("gemv_bf16_rt_a8u8s4", Kind::Single, 1, 4, 8),
                c("gemv_bf16_rt_a8u2s8", Kind::Single, 1, 8, 8),
                c("gemv_bf16_rt_a8u4s4", Kind::Single, 1, 4, 8),
                c("gemv_bf16_rt_a8u8s8", Kind::Single, 1, 8, 8),
            ]
        }
    }
}
