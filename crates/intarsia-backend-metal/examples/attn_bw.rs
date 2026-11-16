//! Decode attention microbenchmark (metal phase B2a): per-call GPU time and K+V bandwidth of the single-pass and the
//! split-KV `AttnDecode` kernels at a given context length, with one KV cache per layer so the reads come from DRAM like a
//! decode step's do. GPU time is the command buffers' `GPUStartTime` / `GPUEndTime` from the backend's own
//! `INTARSIA_METAL_PROFILE=cb:<file>` trace (this example sets it), divided by the dispatches in the buffer.
//!
//!   cargo run --release -p intarsia-backend-metal --example attn_bw -- [--heads N] [--kv-heads N] [--dim D]
//!       [--layers L] [--reps R] [--warm W] [--devpos] [--cap C] [kernel...] [ctx...]
//!
//! `--devpos` passes the position as a device tensor (token feedback, `use_dev = 1`: the split grid is sized for the cache
//! capacity), `--cap C` fixes the cache capacity (default: `ctx` rounded up to 64; the 35B decode runs at 8192).
//! `kernel` is `single` (attn_decode_bf16) or `splitkv` (default: both); `ctx` the number of visible keys (default
//! 128 1024 4096 16384). The split policy can be set with `INTARSIA_METAL_SPLITKV=<min_keys>,<target_tgs>`.
//!
//! Lane mka additions: `rowtab` = the row-table kernel (`attn_decode_bf16_splitkv_rows`) with `--rows M[,M..]` rows that all
//! point at the SAME cache at positions `ctx - M .. ctx - 1` (what an MTP verify pass launches), `rowtabd` = the same with one
//! cache per row (a batched step). `--hot` reuses one cache for every launch of a buffer (reads come from the GPU caches where
//! they fit: the compute side of the kernel). GB/s is unique K + V bytes of the launch over its time; `--roof G` sets the
//! roofline the percentage is against (default 550).
//! `--gap MB`: a GPU copy of MB megabytes runs between the attention launches of a buffer (another kernel that streams from
//! memory, as the projections between two attention layers of a decode step do); its own time, measured in a buffer of copies
//! alone, is subtracted. Chained attention launches overlap each other and flatter the kernel by up to 1.6x against the
//! engine's per-dispatch trace; `--gap` is the in-engine-like number.
//! Run inside a fleetq slot (it streams up to ~2.5 GB; `rowtabd` M times that).

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
    use intarsia_backend_metal::kernels::{
        ATTN_DECODE_BF16, ATTN_DECODE_BF16_SPLITKV, ATTN_PREFILL_BF16_FLASH, ATTN_PREFILL_BF16_FLASH_NAX,
        ATTN_PREFILL_BF16_SPLITKV_ROWS as ROWSX,
    };
    use intarsia_backend_metal::MetalBackend;
    use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
    use intarsia_core::ops::RowTab;
    use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};

    const ROWTAB: &str = "attn_decode_bf16_splitkv_rows";

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
        let mut heads = 32usize;
        let mut kv_heads = 8usize;
        let mut dim = 128usize;
        let mut layers = 36usize;
        let mut reps = 30usize;
        let mut devpos = false;
        let mut warm = 3usize;
        let mut cap_fixed = 0usize;
        let mut kernels: Vec<(&'static str, bool)> = Vec::new();
        let mut rows_list: Vec<usize> = vec![1];
        let mut hot = false;
        let mut roof = 550.0f64;
        let mut gap_mb = 0usize;
        let mut ctxs: Vec<usize> = Vec::new();
        let mut args = std::env::args().skip(1);
        while let Some(a) = args.next() {
            match a.as_str() {
                "--heads" => heads = args.next().unwrap().parse().unwrap(),
                "--kv-heads" => kv_heads = args.next().unwrap().parse().unwrap(),
                "--dim" => dim = args.next().unwrap().parse().unwrap(),
                "--layers" => layers = args.next().unwrap().parse().unwrap(),
                "--reps" => reps = args.next().unwrap().parse().unwrap(),
                "--devpos" => devpos = true,
                "--warm" => warm = args.next().unwrap().parse().unwrap(),
                "--cap" => cap_fixed = args.next().unwrap().parse().unwrap(),
                "--rows" => {
                    rows_list = args.next().unwrap().split(',').map(|v| v.parse().unwrap()).collect();
                }
                "--hot" => hot = true,
                "--roof" => roof = args.next().unwrap().parse().unwrap(),
                "--gap" => gap_mb = args.next().unwrap().parse().unwrap(),
                "single" => kernels.push((ATTN_DECODE_BF16, false)),
                "splitkv" => kernels.push((ATTN_DECODE_BF16_SPLITKV, false)),
                "rowtab" => kernels.push((ROWTAB, false)),
                "rows" => kernels.push((ROWSX, false)),
                // the prefill kernels on M rows at the end of the cache (numerics-changing references, not bitwise the decode)
                "flash" => kernels.push((ATTN_PREFILL_BF16_FLASH, false)),
                "flashnax" => kernels.push((ATTN_PREFILL_BF16_FLASH_NAX, false)),
                "rowtabd" => kernels.push((ROWTAB, true)),
                n => ctxs.push(n.parse().expect("context length")),
            }
        }
        if kernels.is_empty() {
            kernels = vec![(ATTN_DECODE_BF16, false), (ATTN_DECODE_BF16_SPLITKV, false)];
        }
        if ctxs.is_empty() {
            ctxs = vec![128, 1024, 4096, 16384];
        }
        let trace = std::env::temp_dir().join(format!("attn_bw-{}.jsonl", std::process::id()));
        std::env::set_var("INTARSIA_METAL_PROFILE", format!("cb:{}", trace.display()));
        let be = MetalBackend::new().expect("metal backend");
        println!(
            "heads {heads}/{kv_heads} head_dim {dim}, {layers} layers ({}), {reps} reps, {}; split policy {}; kv-once {}",
            if hot { "ONE cache for all: hot" } else { "one KV cache each" },
            if devpos { "device position" } else { "host position" },
            std::env::var("INTARSIA_METAL_SPLITKV").unwrap_or_else(|_| "default".into()),
            intarsia_backend_metal::kv_once()
        );
        {
            use objc2_metal::MTLComputePipelineState;
            let extra = ["attn_decode_rowtab_g2", "attn_pack_g2_f2048", "attn_pack_g2_f4096", "attn_pack_g2_f7680"];
            for n in intarsia_backend_metal::kernels::ATTN_DECODE_SPLITKV_SHADERS.into_iter().chain(extra) {
                if let Some(pso) = be.pso(n) {
                    println!(
                        "pso {n}: max threads/tg {}, static tg mem {}",
                        pso.maxTotalThreadsPerThreadgroup(),
                        pso.staticThreadgroupMemoryLength()
                    );
                }
            }
        }
        // `--gap`: two buffers per layer slot, and the time of a buffer of the copies alone
        let gap_bytes = gap_mb << 20;
        let gaps: Vec<(Buffer, Buffer)> = if gap_bytes > 0 {
            let mut r = Rng(7);
            (0..layers).map(|_| (rand_buf(&be, gap_bytes / 2, &mut r), rand_buf(&be, gap_bytes / 2, &mut r))).collect()
        } else {
            Vec::new()
        };
        let read_last = |seen: &mut usize| -> Option<f64> {
            let text = std::fs::read_to_string(&trace).unwrap_or_default();
            let lines: Vec<&str> = text.lines().collect();
            let r = (lines.len() > *seen).then(|| {
                let r: serde_json::Value = serde_json::from_str(lines[lines.len() - 1]).unwrap();
                r["e"].as_f64().unwrap() - r["s"].as_f64().unwrap()
            });
            *seen = lines.len();
            r
        };
        let mut gap_s = 0.0f64;
        if gap_bytes > 0 {
            let mut v = Vec::new();
            let mut seen = 0usize;
            for rep in 0..reps + warm {
                for (a, b) in &gaps {
                    be.copy(a, 0, b, 0, gap_bytes, 0).unwrap();
                }
                be.synchronize(0).unwrap();
                if let Some(t) = read_last(&mut seen) {
                    if rep >= warm {
                        v.push(t);
                    }
                }
            }
            let st = stats(&mut v);
            gap_s = st.median;
            println!("gap: {gap_mb} MB copy between launches, {:.1} us each (subtracted)", gap_s / layers as f64 * 1e6);
        }
        println!(
            "{:>8} {:<32} {:>4} {:>9} {:>9} {:>9} {:>8}",
            "visible", "kernel", "M", "us/call", "p10", "p90", "GB/s"
        );
        let mut rng = Rng(0x9e3779b97f4a7c15);
        for &ctx in &ctxs {
            let cap = if cap_fixed > 0 { cap_fixed.max(ctx) } else { ctx.next_multiple_of(64) };
            let pos_buf = be.alloc(4, MemKind::Device).unwrap();
            be.upload(&pos_buf, 0, &((ctx - 1) as i32).to_le_bytes(), 0).unwrap();
            let t = |b: &Buffer, shape: &[usize]| TensorRef {
                buf: b.clone(),
                offset: 0,
                dtype: DType::Bf16,
                shape: shape.to_vec(),
            };
            let max_rows = rows_list.iter().copied().max().unwrap_or(1).max(1);
            let need_d = kernels.iter().any(|k| k.1);
            let q = rand_buf(&be, max_rows * heads * dim, &mut rng);
            let out = be.alloc(max_rows * heads * dim * 2, MemKind::Device).unwrap();
            // caches[layer][row]: row > 0 only for `rowtabd`
            let per_layer = if need_d { max_rows } else { 1 };
            let caches: Vec<Vec<(Buffer, Buffer)>> = (0..if hot { 1 } else { layers })
                .map(|_| {
                    (0..per_layer)
                        .map(|_| {
                            (
                                rand_buf(&be, kv_heads * cap * dim, &mut rng),
                                rand_buf(&be, kv_heads * cap * dim, &mut rng),
                            )
                        })
                        .collect()
                })
                .collect();
            let attn = |pos0: usize| OpAttrs {
                pos0,
                scale: 1.0 / (dim as f32).sqrt(),
                n_heads: heads,
                n_kv_heads: kv_heads,
                head_dim: dim,
                kv_cap: cap,
                ..Default::default()
            };
            for &(kernel, distinct) in &kernels {
                for &m in &rows_list {
                    let rowtab = kernel == ROWTAB;
                    let g0b = kernel == ROWSX;
                    let rowsx = g0b || kernel == ATTN_PREFILL_BF16_FLASH || kernel == ATTN_PREFILL_BF16_FLASH_NAX;
                    if (!rowtab && !rowsx && m != 1) || (g0b && !(2..=8).contains(&m)) {
                        continue;
                    }
                    if ctx < m {
                        continue;
                    }
                    let kshape = [kv_heads, cap, dim];
                    let calls: Vec<OpCall> = (0..layers)
                        .map(|l| {
                            let cl = &caches[if hot { 0 } else { l }];
                            if rowsx {
                                // G0b: M rows adjacent in the grid, row r at position ctx - M + r on the layer's cache
                                return OpCall::new(
                                    Op::AttnPrefill,
                                    vec![t(&q, &[m, heads, dim]), t(&cl[0].0, &kshape), t(&cl[0].1, &kshape)],
                                    vec![t(&out, &[m, heads, dim])],
                                    attn(ctx - m),
                                );
                            }
                            if !rowtab {
                                return OpCall::new(
                                    Op::AttnDecode,
                                    vec![t(&q, &[1, heads, dim]), t(&cl[0].0, &kshape), t(&cl[0].1, &kshape)],
                                    vec![t(&out, &[1, heads, dim])],
                                    attn(ctx - 1),
                                );
                            }
                            let mut bytes = vec![0u8; m * 2 * 8];
                            let mut pos = Vec::new();
                            for r in 0..m {
                                let c = &cl[if distinct { r } else { 0 }];
                                for (j, b) in [&c.0, &c.1].into_iter().enumerate() {
                                    let a = be.device_address(b).expect("device address");
                                    bytes[(r * 2 + j) * 8..(r * 2 + j) * 8 + 8].copy_from_slice(&a.to_le_bytes());
                                }
                                // same cache: the verify rows' positions; distinct caches: every row at the full length
                                let p = if distinct { ctx - 1 } else { ctx - m + r };
                                pos.extend((p as i32).to_le_bytes());
                            }
                            let tb = be.alloc(bytes.len().max(16), MemKind::Device).unwrap();
                            be.upload(&tb, 0, &bytes, 0).unwrap();
                            let pb = be.alloc(pos.len().max(16), MemKind::Device).unwrap();
                            be.upload(&pb, 0, &pos, 0).unwrap();
                            let mut c = OpCall::new(
                                Op::AttnDecode,
                                vec![t(&q, &[m, heads, dim]), t(&cl[0].0, &kshape), t(&cl[0].1, &kshape)],
                                vec![t(&out, &[m, heads, dim])],
                                attn(0),
                            );
                            c.rows = Some(RowTab {
                                ptrs: TensorRef { buf: tb, offset: 0, dtype: DType::U8, shape: vec![m, 2, 8] },
                                rows: m,
                                stride: 2,
                                index: 0,
                                byte_off: 0,
                            });
                            c.pos = Some(TensorRef { buf: pb, offset: 0, dtype: DType::I32, shape: vec![m] });
                            c
                        })
                        .collect();
                    // unique K + V bytes one launch reads
                    let bytes = (2 * kv_heads * ctx * dim * 2) as f64 * if distinct { m as f64 } else { 1.0 };
                    let need = be
                        .kernels()
                        .iter()
                        .find(|k| k.name == kernel && k.op == calls[0].op)
                        .map_or(0, |k| k.scratch_bytes(&ShapeInfo::from_call(&calls[0])));
                    let scratch = (need > 0).then(|| {
                        let b = be.alloc(need, MemKind::Device).unwrap();
                        TensorRef { buf: b, offset: 0, dtype: DType::U8, shape: vec![need] }
                    });
                    let mut samples = Vec::new();
                    let mut seen = std::fs::read_to_string(&trace).unwrap_or_default().lines().count();
                    for rep in 0..reps + warm {
                        for (li, c) in calls.iter().enumerate() {
                            if gap_bytes > 0 {
                                be.copy(&gaps[li].0, 0, &gaps[li].1, 0, gap_bytes, 0).unwrap();
                            }
                            let mut c = c.clone();
                            c.scratch = scratch.clone();
                            if devpos && !rowtab && !rowsx {
                                c.pos = Some(TensorRef {
                                    buf: pos_buf.clone(),
                                    offset: 0,
                                    dtype: DType::I32,
                                    shape: vec![1],
                                });
                            }
                            be.launch(kernel, &c, 0).unwrap();
                        }
                        be.synchronize(0).unwrap();
                        // the trace is flushed at synchronize; the newest record is this rep's command buffer
                        if let Some(t) = read_last(&mut seen) {
                            if rep >= warm {
                                samples.push((t - gap_s) / layers as f64 * 1e6);
                            }
                        }
                    }
                    let st = stats(&mut samples);
                    let name = if distinct { format!("{kernel}/distinct") } else { kernel.to_string() };
                    println!(
                        "{:>8} {:<32} {:>4} {:>9.1} {:>9.1} {:>9.1} {:>8.0}   ({:.0}% of {roof} GB/s; floor {:.1} us)",
                        ctx,
                        name,
                        m,
                        st.median,
                        st.p10,
                        st.p90,
                        bytes / (st.median * 1e-6) / 1e9,
                        100.0 * bytes / (st.median * 1e-6) / 1e9 / roof,
                        bytes / roof / 1e3
                    );
                }
            }
        }
        let _ = std::fs::remove_file(&trace);
    }
}
