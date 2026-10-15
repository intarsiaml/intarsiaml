//! Decode attention microbenchmark (metal phase B2a): per-call GPU time and K+V bandwidth of the single-pass and the
//! split-KV `AttnDecode` kernels at a given context length, with one KV cache per layer so the reads come from DRAM like a
//! decode step's do. GPU time is the command buffers' `GPUStartTime` / `GPUEndTime` from the backend's own
//! `INTARSIA_METAL_PROFILE=cb:<file>` trace (this example sets it), divided by the dispatches in the buffer.
//!
//!   cargo run --release -p intarsia-backend-metal --example attn_bw -- [--heads N] [--kv-heads N] [--dim D]
//!       [--layers L] [--reps R] [kernel...] [ctx...]
//!
//! `kernel` is `single` (attn_decode_bf16), `splitkv` (default: both) or `f8` (attn_decode_f8_splitkv on an fp8 e4m3 cache, half
//! the bytes); `ctx` the number of visible keys (default
//! 128 1024 4096 16384). The split policy can be set with `INTARSIA_METAL_SPLITKV=<min_keys>,<target_tgs>`.
//! Run inside a fleetq slot (it streams up to ~2.5 GB).

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("attn_bw needs macOS / Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    imp::main()
}

#[cfg(target_os = "macos")]
mod imp {
    use intarsia_backend_metal::bench::{f32_to_bf16, stats, Rng};
    use intarsia_backend_metal::kernels::{ATTN_DECODE_BF16, ATTN_DECODE_BF16_SPLITKV, ATTN_DECODE_F8_SPLITKV};
    use intarsia_backend_metal::MetalBackend;
    use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
    use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};

    const PEAK_GBS: f64 = 614.0;

    fn rand_buf(be: &MetalBackend, n: usize, rng: &mut Rng) -> Buffer {
        let b = be.alloc(n * 2, MemKind::Device).unwrap();
        let chunk = 1 << 22;
        let mut off = 0;
        while off < n {
            let m = chunk.min(n - off);
            let v: Vec<u16> = (0..m).map(|_| f32_to_bf16(rng.unit())).collect();
            // SAFETY: u16 -> u8 view of the same memory.
            let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, m * 2) };
            be.upload(&b, off * 2, bytes, 0).unwrap();
            off += m;
        }
        b
    }

    fn rand_buf8(be: &MetalBackend, n: usize, rng: &mut Rng) -> Buffer {
        let b = be.alloc(n, MemKind::Device).unwrap();
        let chunk = 1 << 22;
        let mut off = 0;
        while off < n {
            let m = chunk.min(n - off);
            let v: Vec<u8> = (0..m).map(|_| intarsia_core::dtype::f32_to_fp8_e4m3(rng.unit() * 4.0)).collect();
            be.upload(&b, off, &v, 0).unwrap();
            off += m;
        }
        b
    }

    pub fn main() {
        let mut heads = 32usize;
        let mut kv_heads = 8usize;
        let mut dim = 128usize;
        let mut layers = 36usize;
        let mut reps = 30usize;
        let mut kernels: Vec<&'static str> = Vec::new();
        let mut ctxs: Vec<usize> = Vec::new();
        let mut args = std::env::args().skip(1);
        while let Some(a) = args.next() {
            match a.as_str() {
                "--heads" => heads = args.next().unwrap().parse().unwrap(),
                "--kv-heads" => kv_heads = args.next().unwrap().parse().unwrap(),
                "--dim" => dim = args.next().unwrap().parse().unwrap(),
                "--layers" => layers = args.next().unwrap().parse().unwrap(),
                "--reps" => reps = args.next().unwrap().parse().unwrap(),
                "single" => kernels.push(ATTN_DECODE_BF16),
                "splitkv" => kernels.push(ATTN_DECODE_BF16_SPLITKV),
                "f8" => kernels.push(ATTN_DECODE_F8_SPLITKV),
                n => ctxs.push(n.parse().expect("context length")),
            }
        }
        if kernels.is_empty() {
            kernels = vec![ATTN_DECODE_BF16, ATTN_DECODE_BF16_SPLITKV];
        }
        if ctxs.is_empty() {
            ctxs = vec![128, 1024, 4096, 16384];
        }
        let trace = std::env::temp_dir().join(format!("attn_bw-{}.jsonl", std::process::id()));
        std::env::set_var("INTARSIA_METAL_PROFILE", format!("cb:{}", trace.display()));
        let be = MetalBackend::new().expect("metal backend");
        println!(
            "heads {heads}/{kv_heads} head_dim {dim}, {layers} layers (one KV cache each), {reps} reps; split policy {}",
            std::env::var("INTARSIA_METAL_SPLITKV").unwrap_or_else(|_| "default".into())
        );
        println!("{:>8} {:<26} {:>9} {:>9} {:>9} {:>8}", "visible", "kernel", "us/call", "p10", "p90", "GB/s");
        let mut rng = Rng(0x9e3779b97f4a7c15);
        for &ctx in &ctxs {
            let cap = ctx.next_multiple_of(64);
            let t = |b: &Buffer, shape: &[usize]| TensorRef { buf: b.clone(), offset: 0, dtype: DType::Bf16, shape: shape.to_vec() };
            let q = rand_buf(&be, heads * dim, &mut rng);
            let out = be.alloc(heads * dim * 2, MemKind::Device).unwrap();
            let caches: Vec<(Buffer, Buffer)> = (0..layers)
                .map(|_| (rand_buf(&be, kv_heads * cap * dim, &mut rng), rand_buf(&be, kv_heads * cap * dim, &mut rng)))
                .collect();
            let attn = OpAttrs {
                pos0: ctx - 1,
                scale: 1.0 / (dim as f32).sqrt(),
                n_heads: heads,
                n_kv_heads: kv_heads,
                head_dim: dim,
                kv_cap: cap,
                ..Default::default()
            };
            let calls: Vec<OpCall> = caches
                .iter()
                .map(|(k, v)| {
                    OpCall::new(
                        Op::AttnDecode,
                        vec![t(&q, &[1, heads, dim]), t(k, &[kv_heads, cap, dim]), t(v, &[kv_heads, cap, dim])],
                        vec![t(&out, &[1, heads, dim])],
                        attn.clone(),
                    )
                })
                .collect();
            let caches8: Vec<(Buffer, Buffer)> = if kernels.contains(&ATTN_DECODE_F8_SPLITKV) {
                (0..layers).map(|_| (rand_buf8(&be, kv_heads * cap * dim, &mut rng), rand_buf8(&be, kv_heads * cap * dim, &mut rng))).collect()
            } else {
                Vec::new()
            };
            let calls8: Vec<OpCall> = caches8
                .iter()
                .map(|(k, v)| {
                    let t8 = |b: &Buffer, shape: &[usize]| TensorRef { buf: b.clone(), offset: 0, dtype: DType::Fp8E4m3, shape: shape.to_vec() };
                    OpCall::new(
                        Op::AttnDecode,
                        vec![t(&q, &[1, heads, dim]), t8(k, &[kv_heads, cap, dim]), t8(v, &[kv_heads, cap, dim])],
                        vec![t(&out, &[1, heads, dim])],
                        attn.clone(),
                    )
                })
                .collect();
            for &kernel in &kernels {
                let is8 = kernel == ATTN_DECODE_F8_SPLITKV;
                let calls = if is8 { &calls8 } else { &calls };
                let bytes = (2 * kv_heads * ctx * dim * if is8 { 1 } else { 2 }) as f64;
                let need = be
                    .kernels()
                    .iter()
                    .find(|k| k.name == kernel)
                    .map_or(0, |k| k.scratch_bytes(&ShapeInfo::from_call(&calls[0])));
                let scratch = (need > 0).then(|| {
                    let b = be.alloc(need, MemKind::Device).unwrap();
                    TensorRef { buf: b, offset: 0, dtype: DType::U8, shape: vec![need] }
                });
                let mut samples = Vec::new();
                let mut seen = 0usize;
                for rep in 0..reps + 3 {
                    for c in calls.iter() {
                        let mut c = c.clone();
                        c.scratch = scratch.clone();
                        be.launch(kernel, &c, 0).unwrap();
                    }
                    be.synchronize(0).unwrap();
                    // the trace is flushed at synchronize; the newest record is this rep's command buffer
                    let text = std::fs::read_to_string(&trace).unwrap_or_default();
                    let lines: Vec<&str> = text.lines().collect();
                    if lines.len() > seen {
                        let r: serde_json::Value = serde_json::from_str(lines[lines.len() - 1]).unwrap();
                        let (s, e) = (r["s"].as_f64().unwrap(), r["e"].as_f64().unwrap());
                        if rep >= 3 {
                            samples.push((e - s) / layers as f64 * 1e6);
                        }
                    }
                    seen = lines.len();
                }
                let st = stats(&mut samples);
                println!(
                    "{:>8} {:<26} {:>9.1} {:>9.1} {:>9.1} {:>8.0}   ({:.0}% of {PEAK_GBS} GB/s)",
                    ctx,
                    kernel,
                    st.median,
                    st.p10,
                    st.p90,
                    bytes / (st.median * 1e-6) / 1e9,
                    100.0 * bytes / (st.median * 1e-6) / 1e9 / PEAK_GBS
                );
            }
        }
        let _ = std::fs::remove_file(&trace);
    }
}
