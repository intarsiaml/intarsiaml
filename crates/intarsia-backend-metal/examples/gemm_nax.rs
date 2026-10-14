//! P1 spike: NAX bf16 GEMM microbench and error table (docs/results/2026-10-14-metal-prefill-p1-spike.md).
//!
//!   cargo run --release -p intarsia-backend-metal --example gemm_nax -- bench|err [shape ...]
//!   shapes: qo (4096x4096), kv (1024x4096), gu (12288x4096), down (4096x12288); default all
//!   env: NAX_M=256,1024,2048  NAX_ONLY=<shader substrings, comma separated>  NAX_SWZ=0,2 (swizzle log2 candidates)
//!        NAX_REPS=<dispatches per chain> NAX_ROUNDS=<interleaved rounds> NAX_POOL_MIB=<rotating distinct weights>
//!        NAX_WOFF / NAX_AOFF / NAX_COFF=<byte offsets of W, A, C inside their buffers, even>
//!
//! bench: the shapes as GEMMs `C[M,N] = A[M,K] W[N,K]^T`, candidates run in chains of back-to-back dispatches in one
//! serial command buffer (GPU timestamps), interleaved round by round; `RESULT` lines carry TFLOPS (2*M*N*K / time).
//! err: each candidate against an f64 CPU reference on sampled output rows; `ERR` lines. Run inside a fleetq slot.

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
    use intarsia_backend_metal::bench::{self, stats, Rng};
    use intarsia_backend_metal::gemm_nax::{encode_nax, NaxParams, NaxVariant, NAX_VARIANTS};
    use intarsia_backend_metal::{MetalBackend, MtlBuffer};
    use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

    struct Shape {
        label: &'static str,
        n: usize,
        k: usize,
    }

    const SHAPES: [Shape; 4] = [
        Shape { label: "qo", n: 4096, k: 4096 },
        Shape { label: "kv", n: 1024, k: 4096 },
        Shape { label: "gu", n: 12288, k: 4096 },
        Shape { label: "down", n: 4096, k: 12288 },
    ];

    fn env_list<T: std::str::FromStr>(name: &str, default: &[T]) -> Vec<T>
    where
        T: Clone,
    {
        match std::env::var(name) {
            Ok(v) => v.split(',').filter_map(|s| s.trim().parse().ok()).collect(),
            Err(_) => default.to_vec(),
        }
    }

    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
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

    struct Operands {
        a: MtlBuffer,
        ws: Vec<MtlBuffer>,
        c: MtlBuffer,
        aoff: usize,
        woff: usize,
        coff: usize,
    }

    impl Operands {
        fn new(be: &MetalBackend, m: usize, n: usize, k: usize, pool_bytes: usize, offs: (usize, usize, usize)) -> Operands {
            let (aoff, woff, coff) = offs;
            let a = be.new_mtl_buffer(m * k * 2 + aoff + 64, true).unwrap();
            let c = be.new_mtl_buffer(m * n * 2 + coff + 64, true).unwrap();
            // x ~ U[-1,1), w ~ U[-1,1) / sqrt(K) (the admission sweep's scales), distinct seeds
            let wb = n * k * 2;
            let count = if pool_bytes == 0 { 1 } else { pool_bytes.div_ceil(wb).max(2) };
            let mut ws: Vec<MtlBuffer> = Vec::new();
            for i in 0..count {
                let w = be.new_mtl_buffer(wb + woff + 64, true).unwrap();
                // SAFETY: wb bytes at woff inside the allocation.
                unsafe {
                    let dst = (w.contents().as_ptr() as *mut u8).add(woff) as *mut u16;
                    if i == 0 {
                        fill_scaled(dst, n * k, 99, 1.0 / (k as f32).sqrt());
                    } else {
                        let src = (ws[0].contents().as_ptr() as *const u8).add(woff);
                        std::ptr::copy_nonoverlapping(src, dst as *mut u8, wb);
                    }
                }
                ws.push(w);
            }
            // SAFETY: m*k u16 at aoff inside the allocation; c zeroed up front.
            unsafe {
                fill_scaled((a.contents().as_ptr() as *mut u8).add(aoff) as *mut u16, m * k, 7, 1.0);
                std::ptr::write_bytes(c.contents().as_ptr() as *mut u8, 0, m * n * 2 + coff);
            }
            if std::env::var_os("NAX_OUTLIERS").is_some() {
                // heavy-tailed data: every 97th activation x100, every 89th weight x30 (still bf16 values)
                // SAFETY: the same ranges as filled above.
                unsafe {
                    let pa = (a.contents().as_ptr() as *mut u8).add(aoff) as *mut u16;
                    for i in (0..m * k).step_by(97) {
                        let v = bench::bf16_to_f32(std::ptr::read_unaligned(pa.add(i)));
                        std::ptr::write_unaligned(pa.add(i), bench::f32_to_bf16(v * 100.0));
                    }
                    let pw = (ws[0].contents().as_ptr() as *mut u8).add(woff) as *mut u16;
                    for i in (0..n * k).step_by(89) {
                        let v = bench::bf16_to_f32(std::ptr::read_unaligned(pw.add(i)));
                        std::ptr::write_unaligned(pw.add(i), bench::f32_to_bf16(v * 30.0));
                    }
                }
            }
            Operands { a, ws, c, aoff, woff, coff }
        }
    }

    unsafe fn fill_scaled(p: *mut u16, n: usize, seed: u64, scale: f32) {
        let mut r = Rng(seed | 1);
        for i in 0..n {
            std::ptr::write_unaligned(p.add(i), bench::f32_to_bf16(r.unit() * scale));
        }
    }

    fn params(m: usize, n: usize, k: usize, swz: u32, op: &Operands) -> NaxParams {
        NaxParams {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            swz_log: swz,
            s0: ((op.aoff & 3) / 2) as u32,
            s1: ((op.woff & 3) / 2) as u32,
            s2: ((op.coff & 3) / 2) as u32,
            pad: 0,
        }
    }

    fn dispatch_chain(be: &MetalBackend, v: &NaxVariant, m: usize, n: usize, k: usize, swz: u32, op: &Operands, count: usize) -> f64 {
        let pso = be.pso(v.shader).unwrap_or_else(|| panic!("no pipeline {}", v.shader));
        let p = params(m, n, k, swz, op);
        let cb = be.queue().commandBuffer().unwrap();
        let enc = cb.computeCommandEncoder().unwrap();
        for i in 0..count {
            let w = &op.ws[i % op.ws.len()];
            encode_nax(
                &enc,
                pso,
                v,
                (&op.a, op.aoff & !3),
                (w, op.woff & !3),
                (&op.c, op.coff & !3),
                &p,
            );
        }
        enc.endEncoding();
        be.commit(&cb);
        MetalBackend::wait(&cb).unwrap();
        bench::gpu_seconds(&cb)
    }

    fn candidates() -> Vec<NaxVariant> {
        let only = std::env::var("NAX_ONLY").ok();
        NAX_VARIANTS
            .iter()
            .filter(|v| only.as_ref().is_none_or(|o| o.split(',').any(|t| v.shader.contains(t.trim()))))
            .copied()
            .collect()
    }

    fn bench_mode(be: &MetalBackend, shapes: &[&Shape]) {
        let ms: Vec<usize> = env_list("NAX_M", &[256, 1024, 2048]);
        let swzs: Vec<u32> = env_list("NAX_SWZ", &[0]);
        let rounds = env_usize("NAX_ROUNDS", 15);
        let reps = env_usize("NAX_REPS", 10);
        let pool = env_usize("NAX_POOL_MIB", 0) << 20;
        let cands = candidates();
        std::thread::sleep(std::time::Duration::from_secs(4));
        let utils: Vec<Option<u32>> = (0..3)
            .map(|_| {
                let u = gpu_util();
                std::thread::sleep(std::time::Duration::from_millis(400));
                u
            })
            .collect();
        println!("gpu idle (ioreg Device Utilization %, 3 samples): {utils:?}");
        println!("rounds {rounds}, reps {reps}, pool {} MiB, swizzles {swzs:?}", pool >> 20);
        for s in shapes {
            for &m in &ms {
                let op = Operands::new(be, m, s.n, s.k, pool, (0, 0, 0));
                let flop = 2.0 * m as f64 * s.n as f64 * s.k as f64;
                let grid: Vec<(NaxVariant, u32)> = cands.iter().flat_map(|c| swzs.iter().map(move |&z| (*c, z))).collect();
                let mut samples: Vec<Vec<f64>> = vec![Vec::new(); grid.len()];
                for (v, z) in &grid {
                    dispatch_chain(be, v, m, s.n, s.k, *z, &op, 3);
                }
                let mut order: Vec<usize> = (0..grid.len()).collect();
                for _ in 0..rounds {
                    order.rotate_left(1);
                    for &gi in &order {
                        let (v, z) = &grid[gi];
                        let secs = dispatch_chain(be, v, m, s.n, s.k, *z, &op, reps);
                        samples[gi].push(flop * reps as f64 / secs / 1e12);
                    }
                }
                for (gi, (v, z)) in grid.iter().enumerate() {
                    let st = stats(&mut samples[gi]);
                    println!(
                        "RESULT shape={} M={} N={} K={} cand={} swz={} relaxed={} median_tflops={:.2} p10={:.2} p90={:.2} us_per_call={:.1}",
                        s.label, m, s.n, s.k, v.shader, z, v.relaxed, st.median, st.p10, st.p90,
                        flop / (st.median * 1e12) * 1e6
                    );
                }
            }
        }
    }

    /// Output rows checked against the f64 reference: spread over [0, m) plus the last rows (the M edge).
    fn sample_rows(m: usize) -> Vec<usize> {
        let mut rows: Vec<usize> = (0..m).step_by((m / 24).max(1)).collect();
        rows.extend(m.saturating_sub(5)..m);
        rows.sort();
        rows.dedup();
        rows
    }

    fn ref_rows(a: &[u16], w: &[u16], n: usize, k: usize, rows: &[usize]) -> Vec<f64> {
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let wf: Vec<f32> = w.iter().map(|&x| bench::bf16_to_f32(x)).collect();
        let mut out = vec![0f64; rows.len() * n];
        let chunk = n.div_ceil(threads);
        std::thread::scope(|sc| {
            let mut rest: Vec<&mut [f64]> = Vec::new();
            // split each row's output by column chunk: do it per thread over columns instead
            let _ = &mut rest;
            let out_ptr = out.as_mut_ptr() as usize;
            for t in 0..threads {
                let (n0, n1) = (t * chunk, ((t + 1) * chunk).min(n));
                if n0 >= n1 {
                    continue;
                }
                let wf = &wf;
                sc.spawn(move || {
                    for (ri, &r) in rows.iter().enumerate() {
                        let ar: Vec<f64> = a[r * k..(r + 1) * k].iter().map(|&x| bench::bf16_to_f32(x) as f64).collect();
                        for j in n0..n1 {
                            let wr = &wf[j * k..(j + 1) * k];
                            let mut s = 0f64;
                            for kk in 0..k {
                                s += ar[kk] * wr[kk] as f64;
                            }
                            // SAFETY: columns n0..n1 of row ri are written by this thread only.
                            unsafe { *(out_ptr as *mut f64).add(ri * n + j) = s };
                        }
                    }
                });
            }
        });
        out
    }

    /// bf16 ulp at |x| (spacing of bf16 numbers around x), for error-in-ulps.
    fn bf16_ulp(x: f64) -> f64 {
        let x = x.abs().max(1e-30);
        let e = x.log2().floor();
        2f64.powf(e - 7.0)
    }

    fn err_mode(be: &MetalBackend, shapes: &[&Shape]) {
        let ms: Vec<usize> = env_list("NAX_M", &[256, 1024, 2048]);
        let swzs: Vec<u32> = env_list("NAX_SWZ", &[0]);
        let offs = (env_usize("NAX_AOFF", 0), env_usize("NAX_WOFF", 0), env_usize("NAX_COFF", 0));
        let cands = candidates();
        for s in shapes {
            for &m in &ms {
                let op = Operands::new(be, m, s.n, s.k, 0, offs);
                // host copies of the bf16 inputs
                let (a, w) = unsafe {
                    (
                        std::slice::from_raw_parts((op.a.contents().as_ptr() as *const u8).add(op.aoff) as *const u16, m * s.k).to_vec(),
                        std::slice::from_raw_parts((op.ws[0].contents().as_ptr() as *const u8).add(op.woff) as *const u16, s.n * s.k)
                            .to_vec(),
                    )
                };
                let rows = sample_rows(m);
                let reference = ref_rows(&a, &w, s.n, s.k, &rows);
                let rms = (reference.iter().map(|x| x * x).sum::<f64>() / reference.len() as f64).sqrt();
                let max_ref = reference.iter().fold(0f64, |a, &b| a.max(b.abs()));
                for v in &cands {
                    for &z in &swzs {
                        // SAFETY: c holds m*n u16 past coff.
                        unsafe { std::ptr::write_bytes(op.c.contents().as_ptr() as *mut u8, 0xff, m * s.n * 2 + op.coff) };
                        dispatch_chain(be, v, m, s.n, s.k, z, &op, 1);
                        let c = unsafe {
                            std::slice::from_raw_parts((op.c.contents().as_ptr() as *const u8).add(op.coff) as *const u16, m * s.n)
                        };
                        let (mut max_abs, mut max_ulp, mut num, mut den, mut nonfinite) = (0f64, 0f64, 0f64, 0f64, 0usize);
                        let (mut bias, mut cnt, mut hash) = (0f64, 0usize, 0xcbf2_9ce4_8422_2325u64);
                        let mut max_rel_big = 0f64;
                        for (ri, &r) in rows.iter().enumerate() {
                            for j in 0..s.n {
                                let g = bench::bf16_to_f32(c[r * s.n + j]) as f64;
                                if !g.is_finite() {
                                    nonfinite += 1;
                                    continue;
                                }
                                let rf = reference[ri * s.n + j];
                                hash = (hash ^ c[r * s.n + j] as u64).wrapping_mul(0x0000_0100_0000_01b3);
                                bias += (g.abs() - rf.abs()) / bf16_ulp(rf.abs().max(0.1 * rms));
                                cnt += 1;
                                let e = (g - rf).abs();
                                max_abs = max_abs.max(e);
                                max_ulp = max_ulp.max(e / bf16_ulp(rf.abs().max(0.1 * rms)));
                                num += e * e;
                                den += rf * rf;
                                if rf.abs() >= 0.1 * rms {
                                    max_rel_big = max_rel_big.max(e / rf.abs());
                                }
                            }
                        }
                        println!(
                            "ERR shape={} M={} N={} K={} cand={} swz={} relaxed={} rows={} max_abs={:.4e} max_abs_over_rms={:.3e} max_rel(|ref|>=0.1rms)={:.3e} rel_l2={:.3e} max_ulp(of max(|ref|,0.1rms))={:.2} bias_ulp={:+.3} nonfinite={} ref_rms={:.3} ref_max={:.2} hash={:016x}",
                            s.label, m, s.n, s.k, v.shader, z, v.relaxed, rows.len(), max_abs, max_abs / rms, max_rel_big,
                            (num / den).sqrt(), max_ulp, bias / cnt.max(1) as f64, nonfinite, rms, max_ref, hash
                        );
                    }
                }
            }
        }
    }

    pub fn main() {
        let be = MetalBackend::new().expect("metal backend");
        assert!(be.has_metal4(), "metal4.0 library not loaded");
        let mut args: Vec<String> = std::env::args().skip(1).collect();
        let mode = if args.is_empty() { "bench".to_string() } else { args.remove(0) };
        let shapes: Vec<&Shape> = SHAPES.iter().filter(|s| args.is_empty() || args.iter().any(|a| a == s.label)).collect();
        match mode.as_str() {
            "bench" => bench_mode(&be, &shapes),
            "err" => err_mode(&be, &shapes),
            other => panic!("mode '{other}': bench or err"),
        }
    }
}
