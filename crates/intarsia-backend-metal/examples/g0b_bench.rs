//! G0b bench (docs/results/2026-10-15-metal-g0b-attn.md): T(k)/T(1) of the multi-row verification attention and of the small
//! decode ops at Qwen3-8B shapes.
//!
//!   cargo run --release -p intarsia-backend-metal --example g0b_bench -- [attn|small]... [ctx...]
//!
//! env: G0B_ROUNDS (20), G0B_OUT=<csv>, G0B_POOL_MIB (1536, K+V per pool), G0B_MIN_MS (20, shortest chain).
//!
//! Method (as G0 / gemv_b3): a timed serial chain of N dispatches in ONE command buffer (INTARSIA_METAL_COMMIT_EVERY is raised so
//! nothing commits early), time = the buffer's GPUEndTime - GPUStartTime (the backend's own `cb:` trace) / N. Attention: the
//! chain walks a pool of distinct per-layer KV caches (>= 1.5 GiB, so every call reads K / V from DRAM like a decode step
//! does after streaming 15 GB of weights). Candidates are interleaved, once per round, in rotated order; the wall-clock epoch
//! of each chain is recorded so the analysis can join it with a macmon GPU-clock trace and drop rounds below 1450 MHz.
//! Run inside an exclusive fleetq slot.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("g0b_bench needs macOS / Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    imp::main()
}

#[cfg(target_os = "macos")]
mod imp {
    use intarsia_backend_metal::bench::{f32_to_bf16, Rng};
    use intarsia_backend_metal::kernels::*;
    use intarsia_backend_metal::MetalBackend;
    use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
    use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};
    use std::io::Write;

    fn tref(b: &Buffer, off: usize, dt: DType, shape: &[usize]) -> TensorRef {
        TensorRef { buf: b.clone(), offset: off, dtype: dt, shape: shape.to_vec() }
    }

    fn rand_buf(be: &MetalBackend, bytes: usize, rng: &mut Rng, signed: bool) -> Buffer {
        let b = be.alloc(bytes, MemKind::Device).unwrap();
        let n = bytes / 2;
        let chunk = 1 << 22;
        let mut off = 0;
        while off < n {
            let m = chunk.min(n - off);
            let v: Vec<u16> = (0..m).map(|_| f32_to_bf16(if signed { rng.unit() * 2.0 - 1.0 } else { rng.unit() })).collect();
            // SAFETY: u16 -> u8 view of the same memory.
            let by = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, m * 2) };
            be.upload(&b, off * 2, by, 0).unwrap();
            off += m;
        }
        b
    }

    fn scratch_for(be: &MetalBackend, kernel: &str, cl: &OpCall) -> Option<TensorRef> {
        let need = be.kernels().iter().find(|k| k.name == kernel).map_or(0, |k| k.scratch_bytes(&ShapeInfo::from_call(cl)));
        (need > 0).then(|| {
            let b = be.alloc(need, MemKind::Device).unwrap();
            TensorRef { buf: b, offset: 0, dtype: DType::U8, shape: vec![need] }
        })
    }

    /// One candidate: a label, a row count, and the launches of ONE pass over its pool (kernel, call).
    struct Cand {
        label: String,
        k: usize,
        launches: Vec<(&'static str, OpCall)>,
    }

    struct Timer {
        trace: std::path::PathBuf,
        seen: usize,
    }

    impl Timer {
        /// Run `n` launches (cycling over the candidate's pass) as one command buffer; (seconds per dispatch, epoch ms).
        fn chain(&mut self, be: &MetalBackend, c: &Cand, n: usize) -> (f64, u128) {
            self.seen = std::fs::read_to_string(&self.trace).map_or(0, |t| t.lines().count());
            for i in 0..n {
                let (k, call) = &c.launches[i % c.launches.len()];
                be.launch(k, call, 0).unwrap();
            }
            be.synchronize(0).unwrap();
            let epoch = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis();
            let text = std::fs::read_to_string(&self.trace).unwrap_or_default();
            let lines: Vec<&str> = text.lines().collect();
            assert!(lines.len() > self.seen, "no trace record for the chain");
            assert_eq!(lines.len(), self.seen + 1, "the chain was split into several command buffers");
            self.seen = lines.len();
            let r: serde_json::Value = serde_json::from_str(lines[lines.len() - 1]).unwrap();
            ((r["e"].as_f64().unwrap() - r["s"].as_f64().unwrap()) / n as f64, epoch)
        }
    }

    pub fn main() {
        let rounds: usize = std::env::var("G0B_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
        let pool_mib: usize = std::env::var("G0B_POOL_MIB").ok().and_then(|v| v.parse().ok()).unwrap_or(1536);
        let min_ms: f64 = std::env::var("G0B_MIN_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(20.0);
        let out_path = std::env::var("G0B_OUT").unwrap_or_else(|_| "g0b.csv".into());
        let mut benches: Vec<String> = Vec::new();
        let mut ctxs: Vec<usize> = Vec::new();
        for a in std::env::args().skip(1) {
            match a.as_str() {
                "attn" | "small" => benches.push(a),
                n => ctxs.push(n.parse().expect("context length")),
            }
        }
        if benches.is_empty() {
            benches = vec!["attn".into(), "small".into()];
        }
        if ctxs.is_empty() {
            ctxs = vec![128, 1024, 4096, 16384];
        }
        let trace = std::env::temp_dir().join(format!("g0b_bench-{}.jsonl", std::process::id()));
        std::env::set_var("INTARSIA_METAL_PROFILE", format!("cb:{}", trace.display()));
        std::env::set_var("INTARSIA_METAL_COMMIT_EVERY", "1000000");
        let be = MetalBackend::new().expect("metal backend");
        let mut timer = Timer { trace, seen: 0 };
        let mut csv = std::fs::File::create(&out_path).unwrap();
        writeln!(csv, "epoch_ms,bench,ctx,cand,k,round,sec_per_dispatch,n_dispatch").unwrap();
        let mut rng = Rng(0x9e3779b97f4a7c15);
        let (heads, nkv, d) = (32usize, 8usize, 128usize);

        let mut run_set = |be: &MetalBackend, bench: &str, ctx: usize, cands: Vec<Cand>, csv: &mut std::fs::File| {
            // chain length: the fastest candidate (k = 1) must run >= min_ms; same N for every candidate of the set
            let c1 = &cands[0];
            let probe = c1.launches.len().clamp(8, 64);
            timer.chain(be, c1, probe);
            let (t1, _) = timer.chain(be, c1, probe);
            let n = ((min_ms * 1e-3 / t1).ceil() as usize).max(c1.launches.len().min(256)).max(64);
            // warm the clocks for ~0.5 s
            let warm_end = std::time::Instant::now() + std::time::Duration::from_millis(500);
            while std::time::Instant::now() < warm_end {
                timer.chain(be, &cands[cands.len() - 1], n);
            }
            let nc = cands.len();
            for round in 0..rounds {
                for j in 0..nc {
                    let c = &cands[(j + round) % nc];
                    let (t, ep) = timer.chain(be, c, n);
                    writeln!(csv, "{ep},{bench},{ctx},{},{},{round},{t:.9},{n}", c.label, c.k).unwrap();
                }
            }
            csv.flush().unwrap();
            eprintln!("{bench} ctx {ctx}: {nc} candidates x {rounds} rounds, chain of {n} dispatches (T(1) probe {:.1} us)", t1 * 1e6);
        };

        for bench in &benches {
            if bench == "attn" {
                for &ctx in &ctxs {
                    // cache layout [Nkv, cap, D]; pool of per-layer caches in two big buffers (K, V)
                    let cap = (ctx + 8).next_multiple_of(64);
                    let layer_bytes = nkv * cap * d * 2;
                    let layers = ((pool_mib << 20) / (2 * layer_bytes)).max(36);
                    let (kb, vb) = (rand_buf(&be, layers * layer_bytes, &mut rng, true), rand_buf(&be, layers * layer_bytes, &mut rng, true));
                    let qb = rand_buf(&be, 8 * heads * d * 2, &mut rng, true);
                    let ob = be.alloc(8 * heads * d * 2 + 64, MemKind::Device).unwrap();
                    let at = OpAttrs {
                        pos0: ctx - 1,
                        scale: 1.0 / (d as f32).sqrt(),
                        n_heads: heads,
                        n_kv_heads: nkv,
                        head_dim: d,
                        kv_cap: cap,
                        ..Default::default()
                    };
                    let mut cands: Vec<Cand> = Vec::new();
                    // (label, k, kernel, op, rows per launch group, launches per layer)
                    let mut mk = |label: &str, k: usize, kernel: &'static str, op: Op, per_layer: usize| {
                        let mut launches = Vec::new();
                        let mut sc: Option<TensorRef> = None;
                        for l in 0..layers {
                            let ko = l * layer_bytes;
                            let kv = |b: &Buffer| tref(b, ko, DType::Bf16, &[nkv, cap, d]);
                            for r in 0..per_layer {
                                // per_layer > 1: k separate 1-row launches (the sequential fallback), row r at pos0 + r
                                let (rows, qoff, pos0) = if per_layer == 1 { (k, 0, ctx - 1) } else { (1, r * heads * d * 2, ctx - 1 + r) };
                                let mut cl = OpCall::new(
                                    op,
                                    vec![tref(&qb, qoff, DType::Bf16, &[rows, heads, d]), kv(&kb), kv(&vb)],
                                    vec![tref(&ob, qoff, DType::Bf16, &[rows, heads, d])],
                                    OpAttrs { pos0, ..at.clone() },
                                );
                                if sc.is_none() {
                                    sc = scratch_for(&be, kernel, &cl);
                                }
                                cl.scratch = sc.clone();
                                launches.push((kernel, cl));
                            }
                        }
                        // a kernel that cannot take the call (error on the first launch) is skipped, not fatal
                        let (kn, c0) = &launches[0];
                        if be.launch(kn, c0, 0).and_then(|_| be.synchronize(0)).is_err() {
                            eprintln!("skip candidate {label}: first launch failed");
                            return;
                        }
                        cands.push(Cand { label: label.to_string(), k, launches });
                    };
                    mk("serving1", 1, ATTN_DECODE_BF16_SPLITKV, Op::AttnDecode, 1);
                    for k in 2..=8 {
                        mk(&format!("rows{k}"), k, ATTN_PREFILL_BF16_SPLITKV_ROWS, Op::AttnPrefill, 1);
                    }
                    for k in [2, 4, 6, 8] {
                        mk(&format!("seq{k}"), k, ATTN_DECODE_BF16_SPLITKV, Op::AttnDecode, k);
                    }
                    for k in [2, 4, 8] {
                        mk(&format!("flash{k}"), k, ATTN_PREFILL_BF16_FLASH, Op::AttnPrefill, 1);
                    }
                    // what the registry picks for AttnPrefill T = 2..8 today (priority 12, head_dim 128)
                    for k in [2, 4, 8] {
                        mk(&format!("nax{k}"), k, ATTN_PREFILL_BF16_FLASH_NAX, Op::AttnPrefill, 1);
                    }
                    run_set(&be, "attn", ctx, cands, &mut csv);
                }
            } else {
                // small ops at the decode shapes; k = 1..=8
                let bf = DType::Bf16;
                let h = 4096usize;
                let mut mkset = |name: &str, build: &dyn Fn(usize) -> (&'static str, OpCall), csv: &mut std::fs::File| {
                    let cands: Vec<Cand> = (1..=8)
                        .map(|k| {
                            let (kernel, cl) = build(k);
                            let mut cl = cl;
                            cl.scratch = scratch_for(&be, kernel, &cl);
                            Cand { label: format!("{name}{k}"), k, launches: vec![(kernel, cl)] }
                        })
                        .collect();
                    run_set(&be, "small", 0, cands, csv);
                };
                let x = rand_buf(&be, 8 * 12288 * 2, &mut rng, true);
                let y = rand_buf(&be, 8 * 12288 * 2, &mut rng, true);
                let w = rand_buf(&be, 12288 * 2, &mut rng, true);
                let n1 = be.alloc(8 * 12288 * 2, MemKind::Device).unwrap();
                let n2 = be.alloc(8 * 12288 * 2, MemKind::Device).unwrap();
                mkset("norm_add", &|k| {
                    (
                        RMSNORM_ADD_BF16,
                        OpCall::new(
                            Op::RmsNormAdd,
                            vec![tref(&x, 0, bf, &[k, h]), tref(&y, 0, bf, &[k, h]), tref(&w, 0, bf, &[h])],
                            vec![tref(&n1, 0, bf, &[k, h]), tref(&n2, 0, bf, &[k, h])],
                            OpAttrs { eps: 1e-6, ..Default::default() },
                        ),
                    )
                }, &mut csv);
                mkset("rmsnorm", &|k| {
                    (
                        RMSNORM_BF16,
                        OpCall::new(
                            Op::RmsNorm,
                            vec![tref(&x, 0, bf, &[k, h]), tref(&w, 0, bf, &[h])],
                            vec![tref(&n1, 0, bf, &[k, h])],
                            OpAttrs { eps: 1e-6, ..Default::default() },
                        ),
                    )
                }, &mut csv);
                mkset("silu", &|k| {
                    (
                        SILU_GLU_BF16,
                        OpCall::new(
                            Op::SiluGlu,
                            vec![tref(&x, 0, bf, &[k, 12288]), tref(&y, 0, bf, &[k, 12288])],
                            vec![tref(&n1, 0, bf, &[k, 12288])],
                            OpAttrs::default(),
                        ),
                    )
                }, &mut csv);
                mkset("add", &|k| {
                    (
                        ADD_BF16,
                        OpCall::new(
                            Op::Add,
                            vec![tref(&x, 0, bf, &[k, h]), tref(&y, 0, bf, &[k, h])],
                            vec![tref(&n1, 0, bf, &[k, h])],
                            OpAttrs::default(),
                        ),
                    )
                }, &mut csv);
                let cap = 4096usize;
                let (q, kk, vv) = (
                    rand_buf(&be, 8 * heads * d * 2, &mut rng, true),
                    rand_buf(&be, 8 * nkv * d * 2, &mut rng, true),
                    rand_buf(&be, 8 * nkv * d * 2, &mut rng, true),
                );
                let (wq, wk) = (rand_buf(&be, d * 2, &mut rng, true), rand_buf(&be, d * 2, &mut rng, true));
                let qo = be.alloc(8 * heads * d * 2, MemKind::Device).unwrap();
                let (kc, vc) = (be.alloc(nkv * cap * d * 2, MemKind::Device).unwrap(), be.alloc(nkv * cap * d * 2, MemKind::Device).unwrap());
                mkset("qkrope", &|k| {
                    (
                        QK_NORM_ROPE_KV_BF16,
                        OpCall::new(
                            Op::QkNormRopeKv,
                            vec![
                                tref(&q, 0, bf, &[k, heads, d]),
                                tref(&kk, 0, bf, &[k, nkv, d]),
                                tref(&vv, 0, bf, &[k, nkv, d]),
                                tref(&wq, 0, bf, &[d]),
                                tref(&wk, 0, bf, &[d]),
                            ],
                            vec![tref(&qo, 0, bf, &[k, heads, d]), tref(&kc, 0, bf, &[nkv, cap, d]), tref(&vc, 0, bf, &[nkv, cap, d])],
                            OpAttrs {
                                pos0: 1000,
                                theta: 1_000_000.0,
                                eps: 1e-6,
                                head_dim: d,
                                rotary_dim: d,
                                n_heads: heads,
                                n_kv_heads: nkv,
                                kv_cap: cap,
                                ..Default::default()
                            },
                        ),
                    )
                }, &mut csv);
                let v = 151936usize;
                let lg = be.alloc(8 * v * 4, MemKind::Device).unwrap();
                {
                    let mut vals = vec![0u8; 8 * v * 4];
                    for (i, ch) in vals.chunks_exact_mut(4).enumerate() {
                        ch.copy_from_slice(&(((i as u64 * 2654435761) % 1000) as f32 * 0.01).to_le_bytes());
                    }
                    be.upload(&lg, 0, &vals, 0).unwrap();
                }
                let ids = be.alloc(64, MemKind::Device).unwrap();
                be.upload(&ids, 0, &(0..16).flat_map(|i: i32| (i * 7).to_le_bytes()).collect::<Vec<u8>>(), 0).unwrap();
                let am = be.alloc(64, MemKind::Device).unwrap();
                mkset("argmax", &|k| {
                    (
                        ARGMAX_F32,
                        OpCall::new(
                            Op::Argmax,
                            vec![tref(&lg, 0, DType::F32, &[k, v])],
                            vec![tref(&am, 0, DType::I32, &[k])],
                            OpAttrs::default(),
                        ),
                    )
                }, &mut csv);
                let table = rand_buf(&be, 4096 * h * 2, &mut rng, true);
                mkset("embedding", &|k| {
                    (
                        EMBEDDING_BF16,
                        OpCall::new(
                            Op::Embedding,
                            vec![tref(&ids, 0, DType::I32, &[k]), tref(&table, 0, bf, &[4096, h])],
                            vec![tref(&n1, 0, bf, &[k, h])],
                            OpAttrs::default(),
                        ),
                    )
                }, &mut csv);
            }
        }
        let _ = std::fs::remove_file(&timer.trace);
    }
}
