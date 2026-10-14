//! Metal spike bandwidth benchmark: copy-kernel ceiling, bf16 GEMV variants at the Qwen3-8B shapes, and a
//! decode-shaped GEMV-only sweep. All times are GPU timestamps (`GPUEndTime - GPUStartTime` of the command buffer).
//!
//!   cargo run --release -p intarsia-backend-metal --example gemv_bw [copy] [gemv] [decode]
//!
//! Run it inside a fleetq slot (`fleetq run --host endor --exclusive ...`), it needs the GPU and ~16 GB for the sweep.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("gemv_bw needs macOS / Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    imp::main()
}

#[cfg(target_os = "macos")]
mod imp {
    use intarsia_backend_metal::bench::{self, stats, Stats};
    use intarsia_backend_metal::decode::{DecodeSeq, HIDDEN, KV};
    use intarsia_backend_metal::kernels::{GemvVariant, GEMV_VARIANTS};
    use intarsia_backend_metal::{encode_gemv, GemvParams, MetalBackend, MtlBuffer};
    use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLDevice, MTLDispatchType};
    use std::time::Instant;

    const PEAK_GBS: f64 = 614.0;
    const GIB: usize = 1 << 30;
    /// The four GEMV shapes of the go/no-go criterion (K, N): Qwen3-8B attention/MLP/lm_head projections.
    const SHAPES: [(usize, usize); 4] = [(4096, 4096), (4096, 12288), (12288, 4096), (4096, 151936)];
    const WARMUP: usize = 5;
    const REPS: usize = 100;
    /// Distinct weight bytes cycled through per measurement, >> any cache.
    const POOL_BYTES: usize = 4 * GIB;

    pub fn main() {
        let be = MetalBackend::new().expect("metal backend");
        let want: Vec<String> = std::env::args().skip(1).collect();
        let run = |s: &str| want.is_empty() || want.iter().any(|w| w == s);
        println!("device: {}", be.device().name());
        println!(
            "page size {} B, maxBufferLength {} B, recommendedMaxWorkingSetSize {} B, peak assumed {PEAK_GBS} GB/s",
            be.page_size(),
            be.device().maxBufferLength(),
            be.device().recommendedMaxWorkingSetSize()
        );
        println!(
            "warm-up {WARMUP}, reps {REPS}, weight pool >= {} GiB per shape, GB/s = 1e9 bytes/s",
            POOL_BYTES >> 30
        );
        let copy_med = if run("copy") { copy_section(&be) } else { f64::NAN };
        if run("gemv") {
            gemv_section(
                &be,
                if copy_med.is_nan() { bench::measure_copy_gbs(&be, GIB, 5, 21).unwrap() } else { copy_med },
            );
        }
        if want.iter().any(|w| w == "align") {
            align_section(&be);
        }
        if run("decode") {
            decode_section(&be);
        }
    }

    fn fmt(s: Stats) -> String {
        format!("median {:7.1}  p10 {:7.1}  p90 {:7.1}", s.median, s.p10, s.p90)
    }

    fn copy_section(be: &MetalBackend) -> f64 {
        println!("\n== copy kernel (copy_u4: 16 B loads/stores, 4 per thread); GB/s = (read+write bytes)/time ==");
        let mut med1 = f64::NAN;
        for gib in [1usize, 2] {
            let mut v = bench::copy_gbs_samples(be, gib * GIB, WARMUP, REPS).unwrap();
            let s = stats(&mut v);
            println!("copy {gib} GiB: {}  ({:.1}% of {PEAK_GBS})", fmt(s), 100.0 * s.median / PEAK_GBS);
            if gib == 1 {
                med1 = s.median;
            }
        }
        println!("-- read-only ceiling (read_xor_u4, writes 4 B per 64 B read); GB/s = read bytes/time --");
        for gib in [1usize, 4] {
            let mut v = read_samples(be, gib * GIB);
            let s = stats(&mut v);
            println!("read {gib} GiB: {}  ({:.1}% of {PEAK_GBS})", fmt(s), 100.0 * s.median / PEAK_GBS);
        }
        med1
    }

    fn read_samples(be: &MetalBackend, bytes: usize) -> Vec<f64> {
        let src = be.new_mtl_buffer(bytes, true).unwrap();
        let out = be.new_mtl_buffer(bench::read_out_bytes(bytes), true).unwrap();
        // SAFETY: freshly allocated; touching the pages keeps first-touch faults out of the timing.
        unsafe { std::ptr::write_bytes(src.contents().as_ptr() as *mut u8, 0x5a, bytes) };
        let mut r = Vec::new();
        for i in 0..WARMUP + REPS {
            let cb = be
                .encode_and_commit(|enc| {
                    bench::encode_read(be, enc, &src, &out, bytes).unwrap();
                })
                .unwrap();
            MetalBackend::wait(&cb).unwrap();
            if i >= WARMUP {
                r.push(bytes as f64 / bench::gpu_seconds(&cb) / 1e9);
            }
        }
        r
    }

    #[derive(Clone, Copy, PartialEq)]
    enum Mode {
        /// One dispatch per command buffer, rotating through the pool.
        Solo,
        /// The whole pool in one command buffer, serial dispatch type (each dispatch waits for the previous).
        ChainSerial,
        /// Same, concurrent dispatch type (no dependency between dispatches: tail/head overlap allowed).
        ChainConcurrent,
    }

    fn cb_with(
        be: &MetalBackend,
        concurrent: bool,
        f: impl FnOnce(&objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>),
    ) -> f64 {
        let cb = be.queue().commandBuffer().unwrap();
        let enc = cb
            .computeCommandEncoderWithDispatchType(if concurrent {
                MTLDispatchType::Concurrent
            } else {
                MTLDispatchType::Serial
            })
            .unwrap();
        f(&enc);
        enc.endEncoding();
        be.commit(&cb);
        MetalBackend::wait(&cb).unwrap();
        bench::gpu_seconds(&cb)
    }

    struct Pool {
        k: usize,
        n: usize,
        /// Byte offset the weights are bound at inside each buffer (allocation has 64 spare bytes).
        w_off: usize,
        w: Vec<MtlBuffer>,
        x: MtlBuffer,
        y: MtlBuffer,
    }

    impl Pool {
        fn new(be: &MetalBackend, k: usize, n: usize, w_off: usize) -> Pool {
            let wb = n * k * 2;
            let count = POOL_BYTES.div_ceil(wb).max(2);
            let mut w: Vec<MtlBuffer> = Vec::new();
            for i in 0..count {
                let b = be.new_mtl_buffer(wb + 64, true).unwrap();
                // SAFETY: wb bytes.
                unsafe {
                    if i == 0 {
                        bench::fill_bf16_random((b.contents().as_ptr() as *mut u8).add(w_off) as *mut u16, n * k, 42);
                    } else {
                        std::ptr::copy_nonoverlapping(
                            (w[0].contents().as_ptr() as *const u8).add(w_off),
                            (b.contents().as_ptr() as *mut u8).add(w_off),
                            wb,
                        );
                    }
                }
                w.push(b);
            }
            let x = be.new_mtl_buffer(k * 2, true).unwrap();
            let y = be.new_mtl_buffer(n * 2, true).unwrap();
            // SAFETY: k*2 bytes.
            unsafe { bench::fill_bf16_random(x.contents().as_ptr() as *mut u16, k, 7) };
            Pool { k, n, w_off, w, x, y }
        }

        fn bytes_per_gemv(&self) -> f64 {
            (self.n * self.k * 2 + self.k * 2 + self.n * 2) as f64
        }

        fn dispatch(
            &self,
            enc: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
            be: &MetalBackend,
            v: GemvVariant,
            i: usize,
        ) {
            let p = GemvParams { k: self.k as u32, n: self.n as u32, ..Default::default() };
            encode_gemv(
                enc,
                be.pso(v.name).unwrap(),
                (&self.w[i], self.w_off),
                (&self.x, 0),
                (&self.y, 0),
                &p,
                (self.n as u32).div_ceil(v.rows_per_tg()),
                v.threads_per_tg(),
            );
        }

        /// GB/s per rep.
        fn measure(&self, be: &MetalBackend, v: GemvVariant, mode: Mode, reps: usize) -> Vec<f64> {
            let mut out = Vec::new();
            for rep in 0..WARMUP + reps {
                let (secs, gemvs) = match mode {
                    Mode::Solo => (cb_with(be, false, |enc| self.dispatch(enc, be, v, rep % self.w.len())), 1),
                    Mode::ChainSerial | Mode::ChainConcurrent => (
                        cb_with(be, mode == Mode::ChainConcurrent, |enc| {
                            for i in 0..self.w.len() {
                                self.dispatch(enc, be, v, i);
                            }
                        }),
                        self.w.len(),
                    ),
                };
                if rep >= WARMUP {
                    out.push(gemvs as f64 * self.bytes_per_gemv() / secs / 1e9);
                }
            }
            out
        }
    }

    fn gemv_section(be: &MetalBackend, copy_med: f64) {
        println!("\n== GEMV variants; GB/s = (weight + x + y bytes)/GPU time; copy reference = {copy_med:.1} GB/s ==");
        println!("modes: solo = 1 dispatch per command buffer (includes launch overhead); chain-serial = whole pool in one command buffer, serial dispatch; chain-conc = same, concurrent dispatch");
        for (k, n) in SHAPES {
            let pool = Pool::new(be, k, n, 0);
            println!(
                "\n-- K={k} N={n}: {:.1} MiB weights x {} buffers = {:.2} GiB per chain rep --",
                (n * k * 2) as f64 / 1048576.0,
                pool.w.len(),
                (pool.w.len() * n * k * 2) as f64 / GIB as f64
            );
            // burn-in: let the GPU clocks ramp and the pipelines warm before the first timed variant
            let _ = pool.measure(be, GEMV_VARIANTS[0], Mode::ChainSerial, 30);
            for v in GEMV_VARIANTS {
                for (label, mode) in
                    [("solo", Mode::Solo), ("chain-serial", Mode::ChainSerial), ("chain-conc", Mode::ChainConcurrent)]
                {
                    let s = stats(&mut pool.measure(be, *v, mode, REPS));
                    println!(
                        "RESULT gemv K={k} N={n} variant={} mode={label} median={:.1} p10={:.1} p90={:.1} pct_peak={:.1} pct_copy={:.1}",
                        v.name,
                        s.median,
                        s.p10,
                        s.p90,
                        100.0 * s.median / PEAK_GBS,
                        100.0 * s.median / copy_med
                    );
                }
            }
        }
    }

    /// What a misaligned weight file costs: the aligned fast path against the 8-byte and scalar fallbacks, with weights
    /// bound at the offset each kernel can accept. Only runs when asked for (`align`).
    fn align_section(be: &MetalBackend) {
        println!("\n== alignment fallbacks; chain-serial GB/s, weights bound at the given byte offset ==");
        let cases: [(&str, GemvVariant, usize); 5] = [
            ("aligned16 (default)", intarsia_backend_metal::DEFAULT_GEMV, 0),
            ("a8 kernel at offset 8", intarsia_backend_metal::kernels::DEFAULT_GEMV_A8, 8),
            ("a8 kernel at offset 0", intarsia_backend_metal::kernels::DEFAULT_GEMV_A8, 0),
            ("scalar kernel at offset 0", intarsia_backend_metal::kernels::GEMV_SCALAR, 0),
            ("scalar kernel at offset 8", intarsia_backend_metal::kernels::GEMV_SCALAR, 8),
        ];
        for (k, n) in SHAPES {
            for (label, v, off) in cases {
                let pool = Pool::new(be, k, n, off);
                let _ = pool.measure(be, v, Mode::ChainSerial, 20);
                let s = stats(&mut pool.measure(be, v, Mode::ChainSerial, REPS));
                println!(
                    "RESULT align K={k} N={n} case=\"{label}\" variant={} median={:.1} p10={:.1} p90={:.1} pct_peak={:.1}",
                    v.name,
                    s.median,
                    s.p10,
                    s.p90,
                    100.0 * s.median / PEAK_GBS
                );
            }
        }
    }

    /// Fastest variant per distinct (K, N) of the sequence, by chain-serial median over that shape's own buffers.
    fn best_per_shape(be: &MetalBackend, seq: &DecodeSeq) -> Vec<((usize, usize), GemvVariant)> {
        let mut shapes: Vec<(usize, usize)> = Vec::new();
        for o in &seq.ops {
            if !shapes.contains(&(o.k, o.n)) {
                shapes.push((o.k, o.n));
            }
        }
        let mut best = Vec::new();
        for (k, n) in shapes {
            let idx: Vec<usize> =
                seq.ops.iter().enumerate().filter(|(_, o)| o.k == k && o.n == n).map(|(i, _)| i).collect();
            let mut top: Option<(f64, GemvVariant)> = None;
            for v in GEMV_VARIANTS {
                let mut ts = Vec::new();
                for rep in 0..12 {
                    let t = cb_with(be, false, |enc| {
                        for &i in &idx {
                            let p = GemvParams { k: k as u32, n: n as u32, ..Default::default() };
                            encode_gemv(
                                enc,
                                be.pso(v.name).unwrap(),
                                (&seq.w[i], 0),
                                (seq.x_for(k), 0),
                                (&seq.y[i], 0),
                                &p,
                                (n as u32).div_ceil(v.rows_per_tg()),
                                v.threads_per_tg(),
                            );
                        }
                    });
                    if rep >= 2 {
                        ts.push(t / idx.len() as f64);
                    }
                }
                let med = stats(&mut ts).median;
                if top.is_none_or(|(m, _)| med < m) {
                    top = Some((med, *v));
                }
            }
            best.push(((k, n), top.unwrap().1));
        }
        best
    }

    fn decode_section(be: &MetalBackend) {
        println!("\n== Qwen3-8B decode-shaped GEMV sweep: 36 x (q,k,v,o,gate,up,down) + lm_head, one command buffer, serial dispatch ==");
        let copy = bench::measure_copy_gbs(be, GIB, 5, 21).unwrap();
        let mut rd = read_samples(be, 4 * GIB);
        let read = stats(&mut rd).median;
        let seq =
            DecodeSeq::qwen3_8b(be, intarsia_backend_metal::decode::LAYERS, true).expect("allocate decode weights");
        println!(
            "{} GEMVs, weights {:.2} GB ({:.2} GiB), step bytes {:.2} GB; copy ceiling {copy:.1} GB/s, read ceiling {read:.1} GB/s, peak {PEAK_GBS}",
            seq.ops.len(),
            seq.weight_bytes() as f64 / 1e9,
            seq.weight_bytes() as f64 / GIB as f64,
            seq.step_bytes() as f64 / 1e9
        );
        println!(
            "dims: hidden {HIDDEN}, kv {KV}, ffn {}, vocab {}",
            intarsia_backend_metal::decode::FFN,
            intarsia_backend_metal::decode::VOCAB
        );
        let best = best_per_shape(be, &seq);
        for ((k, n), v) in &best {
            println!("best variant for K={k} N={n}: {}", v.name);
        }
        let mut configs: Vec<(String, Box<dyn Fn(usize, usize) -> GemvVariant>)> = Vec::new();
        for v in GEMV_VARIANTS {
            let v = *v;
            configs.push((v.name.to_string(), Box::new(move |_, _| v)));
        }
        let best2 = best.clone();
        configs.push((
            "best-per-shape".to_string(),
            Box::new(move |k, n| best2.iter().find(|(s, _)| *s == (k, n)).unwrap().1),
        ));
        for (name, pick) in &configs {
            let (mut gpu, mut wall, mut enc_cpu) = (Vec::new(), Vec::new(), Vec::new());
            for rep in 0..WARMUP + 50 {
                let t0 = Instant::now();
                let cb = be.queue().commandBuffer().unwrap();
                let enc = cb.computeCommandEncoder().unwrap();
                seq.encode_direct(be, &enc, pick.as_ref());
                enc.endEncoding();
                be.commit(&cb);
                let t1 = Instant::now();
                MetalBackend::wait(&cb).unwrap();
                let t2 = Instant::now();
                if rep >= WARMUP {
                    gpu.push(bench::gpu_seconds(&cb));
                    wall.push((t2 - t0).as_secs_f64());
                    enc_cpu.push((t1 - t0).as_secs_f64());
                }
            }
            let g = stats(&mut gpu);
            let w = stats(&mut wall);
            let e = stats(&mut enc_cpu);
            let gbs = |t: f64| seq.step_bytes() as f64 / t / 1e9;
            println!(
                "RESULT decode config={name} gpu_ms median={:.3} p10={:.3} p90={:.3} | tok_s median={:.1} (p10 {:.1}, p90 {:.1}) | GB/s median={:.1} ({:.1}% of {PEAK_GBS}, {:.1}% of copy, {:.1}% of read) | wall_ms median={:.3} | encode+commit_ms median={:.3}",
                g.median * 1e3,
                g.p10 * 1e3,
                g.p90 * 1e3,
                1.0 / g.median,
                1.0 / g.p90,
                1.0 / g.p10,
                gbs(g.median),
                100.0 * gbs(g.median) / PEAK_GBS,
                100.0 * gbs(g.median) / copy,
                100.0 * gbs(g.median) / read,
                w.median * 1e3,
                e.median * 1e3
            );
        }
    }
}
