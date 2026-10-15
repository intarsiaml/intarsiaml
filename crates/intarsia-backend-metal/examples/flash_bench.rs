//! Prefill attention microbenchmark (metal prefill track P2): GPU time per call and effective TFLOPS of the
//! `AttnPrefill` kernels for a chunk of T queries after `pos0` cached positions, one KV cache per layer. GPU time is the
//! command buffers' `GPUStartTime` / `GPUEndTime` from the backend's own `INTARSIA_METAL_PROFILE=cb:<file>` trace (this
//! example sets it), divided by the dispatches in the buffer.
//!
//!   cargo run --release -p intarsia-backend-metal --example flash_bench -- [--heads N] [--kv-heads N] [--dim D]
//!       [--layers L] [--reps R] [--chunk T] [kernel-or-variant...] [pos0...]
//!
//! `kernel` is `old` (attn_prefill_bf16), `flash` (attn_prefill_bf16_flash, default variant) or the name of a flash
//! variant (`attn_flash_*`, see flash.rs). Defaults: chunk 256, pos0 0 1024 3840 15872. FLOPs are counted causally:
//! 4 * D * H * (T * pos0 + T * (T + 1) / 2) (the convention of docs/results/raw/2026-10-14-metal-phase-c/analyze_prefill.py).
//! Run inside a fleetq slot.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("flash_bench needs macOS / Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    imp::main()
}

#[cfg(target_os = "macos")]
mod imp {
    use intarsia_backend_metal::bench::{f32_to_bf16, stats, Rng};
    use intarsia_backend_metal::flash::{select_variant, FLASH_VARIANTS};
    use intarsia_backend_metal::flash_nax::{select_variant as select_nax, NAX_FLASH_VARIANTS};
    use intarsia_backend_metal::kernels::{ATTN_PREFILL_BF16, ATTN_PREFILL_BF16_FLASH, ATTN_PREFILL_BF16_FLASH_NAX};
    use intarsia_backend_metal::MetalBackend;
    use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
    use intarsia_core::{DType, Op, OpAttrs, OpCall};

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

    pub fn main() {
        let (mut heads, mut kv_heads, mut dim, mut layers, mut reps, mut chunk) = (32usize, 8usize, 128usize, 36usize, 10usize, 256usize);
        let mut kernels: Vec<(String, &'static str, Option<&'static str>)> = Vec::new();
        let mut total = 0usize;
        let mut pos: Vec<usize> = Vec::new();
        let mut args = std::env::args().skip(1);
        while let Some(a) = args.next() {
            match a.as_str() {
                "--heads" => heads = args.next().unwrap().parse().unwrap(),
                "--kv-heads" => kv_heads = args.next().unwrap().parse().unwrap(),
                "--dim" => dim = args.next().unwrap().parse().unwrap(),
                "--layers" => layers = args.next().unwrap().parse().unwrap(),
                "--reps" => reps = args.next().unwrap().parse().unwrap(),
                "--chunk" => chunk = args.next().unwrap().parse().unwrap(),
                "--total" => total = args.next().unwrap().parse().unwrap(),
                "nax" => kernels.push(("nax".into(), ATTN_PREFILL_BF16_FLASH_NAX, None)),
                n if n.starts_with("attn_nax_") => {
                    let v = NAX_FLASH_VARIANTS.iter().find(|v| v.name == n).expect("unknown nax flash variant");
                    kernels.push((n.into(), ATTN_PREFILL_BF16_FLASH_NAX, Some(v.name)));
                }
                "old" => kernels.push(("old".into(), ATTN_PREFILL_BF16, None)),
                "flash" => kernels.push(("flash".into(), ATTN_PREFILL_BF16_FLASH, None)),
                n if n.starts_with("attn_flash_") => {
                    let v = FLASH_VARIANTS.iter().find(|v| v.name == n).expect("unknown flash variant");
                    kernels.push((n.into(), ATTN_PREFILL_BF16_FLASH, Some(v.name)));
                }
                n => pos.push(n.parse().expect("pos0")),
            }
        }
        if kernels.is_empty() {
            kernels = vec![
                ("old".into(), ATTN_PREFILL_BF16, None),
                ("flash".into(), ATTN_PREFILL_BF16_FLASH, None),
                ("nax".into(), ATTN_PREFILL_BF16_FLASH_NAX, None),
            ];
        }
        if total > 0 {
            pos = (0..total).step_by(chunk).collect();
        } else if pos.is_empty() {
            pos = vec![0, 1024, 3840, 15872];
        }
        let mut agg: Vec<(f64, f64)> = vec![(0.0, 0.0); kernels.len()];
        let trace = std::env::temp_dir().join(format!("flash_bench-{}.jsonl", std::process::id()));
        std::env::set_var("INTARSIA_METAL_PROFILE", format!("cb:{}", trace.display()));
        let be = MetalBackend::new().expect("metal backend");
        println!("heads {heads}/{kv_heads} head_dim {dim}, chunk {chunk}, {layers} layers (one KV cache each), {reps} reps");
        println!("{:>7} {:<28} {:>9} {:>9} {:>9} {:>8}", "pos0", "kernel", "us/call", "p10", "p90", "TFLOPS");
        let mut rng = Rng(0x9e3779b97f4a7c15);
        let t = |b: &Buffer, shape: &[usize]| TensorRef { buf: b.clone(), offset: 0, dtype: DType::Bf16, shape: shape.to_vec() };
        // --total N: one cache set at the final capacity, every chunk of an N-token prompt in turn
        let fixed_cap = if total > 0 { total.next_multiple_of(64) } else { 0 };
        let mut fixed: Option<(Buffer, Vec<(Buffer, Buffer)>)> = None;
        for &p0 in &pos {
            let cap = if total > 0 { fixed_cap } else { (p0 + chunk).next_multiple_of(64) };
            let tq = if total > 0 { chunk.min(total - p0) } else { chunk };
            let (q, caches) = match (total > 0, fixed.take()) {
                (true, Some(f)) => {
                    fixed = Some(f.clone());
                    f
                }
                _ => {
                    let q = rand_buf(&be, chunk * heads * dim, &mut rng);
                    let caches: Vec<(Buffer, Buffer)> = (0..layers)
                        .map(|_| (rand_buf(&be, kv_heads * cap * dim, &mut rng), rand_buf(&be, kv_heads * cap * dim, &mut rng)))
                        .collect();
                    if total > 0 {
                        fixed = Some((q.clone(), caches.clone()));
                    }
                    (q, caches)
                }
            };
            let out = be.alloc(chunk * heads * dim * 2, MemKind::Device).unwrap();
            let attn = OpAttrs {
                pos0: p0,
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
                        Op::AttnPrefill,
                        vec![t(&q, &[tq, heads, dim]), t(k, &[kv_heads, cap, dim]), t(v, &[kv_heads, cap, dim])],
                        vec![t(&out, &[tq, heads, dim])],
                        attn.clone(),
                    )
                })
                .collect();
            let flops = 4.0 * dim as f64 * heads as f64 * (tq * p0) as f64 + 4.0 * dim as f64 * heads as f64 * (tq * (tq + 1)) as f64 / 2.0;
            for (ki, (label, kernel, variant)) in kernels.iter().enumerate() {
                match *kernel {
                    ATTN_PREFILL_BF16_FLASH_NAX => select_nax(*variant),
                    _ => select_variant(*variant),
                }
                let mut samples = Vec::new();
                let mut seen = 0usize;
                for rep in 0..reps + 2 {
                    for c in &calls {
                        be.launch(kernel, c, 0).unwrap();
                    }
                    be.synchronize(0).unwrap();
                    let text = std::fs::read_to_string(&trace).unwrap_or_default();
                    let lines: Vec<&str> = text.lines().collect();
                    if lines.len() > seen {
                        let r: serde_json::Value = serde_json::from_str(lines[lines.len() - 1]).unwrap();
                        let (s, e) = (r["s"].as_f64().unwrap(), r["e"].as_f64().unwrap());
                        if rep >= 2 {
                            samples.push((e - s) / layers as f64 * 1e6);
                        }
                    }
                    seen = lines.len();
                }
                let st = stats(&mut samples);
                agg[ki].0 += flops;
                agg[ki].1 += st.median;
                if total == 0 {
                    println!(
                        "{:>7} {:<28} {:>9.1} {:>9.1} {:>9.1} {:>8.2}",
                        p0,
                        label,
                        st.median,
                        st.p10,
                        st.p90,
                        flops / (st.median * 1e-6) / 1e12
                    );
                }
            }
        }
        if total > 0 {
            for (ki, (label, _, _)) in kernels.iter().enumerate() {
                println!(
                    "total {total} chunk {chunk}: {:<28} {:>10.1} us per layer ({:.1} ms over {layers} layers) {:>7.2} TFLOPS",
                    label,
                    agg[ki].1,
                    agg[ki].1 * layers as f64 / 1e3,
                    agg[ki].0 / (agg[ki].1 * 1e-6) / 1e12
                );
            }
        }
        let _ = std::fs::remove_file(&trace);
    }
}
