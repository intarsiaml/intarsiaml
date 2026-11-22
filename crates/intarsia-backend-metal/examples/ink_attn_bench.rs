//! Inkling-2-Mini attention microbenchmark (lane ink-m-attn): per-launch GPU time of `ink_kv_prologue_bf16` and
//! `rel_sink_attn_bf16` at the real shapes (24 / 8 heads, head_dim 128, d_rel 16, extent 1024, K 4), `L` dependent launches
//! per command buffer, each on its own operand set (like the visits of a decode step). GPU time is the command buffer's
//! `GPUStartTime` / `GPUEndTime` from the backend's `INTARSIA_METAL_PROFILE=cb:<file>` trace, divided by the launches in it.
//! A near-empty `add_bf16` over 32 elements gives the launch floor in the same harness.
//!
//!   cargo run --release -p intarsia-backend-metal --example ink_attn_bench -- [--layers L] [--reps R] [--prefill] [--fused]
//!
//! Run it inside a fleetq slot with `--quiet gpu,cpu`.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("ink_attn_bench needs macOS / Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    imp::main()
}

#[cfg(target_os = "macos")]
mod imp {
    use intarsia_backend_metal::bench::{f32_to_bf16, stats, Rng};
    use intarsia_backend_metal::MetalBackend;
    use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
    use intarsia_core::inkling2::ops::nq;
    use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};

    const H: usize = 24;
    const HK: usize = 8;
    const DH: usize = 128;
    const DR: usize = 16;
    const E: usize = 1024;
    const K: usize = 4;
    const RR: usize = K - 1 + 16;

    fn t(b: &Buffer, shape: &[usize]) -> TensorRef {
        TensorRef { buf: b.clone(), offset: 0, dtype: DType::Bf16, shape: shape.to_vec() }
    }

    fn buf(be: &MetalBackend, n: usize, rng: &mut Rng, s: f32) -> Buffer {
        let v: Vec<u16> = (0..n).map(|_| f32_to_bf16((rng.unit() * 2.0 - 1.0) * s)).collect();
        let b = be.alloc(n * 2 + 64, MemKind::Device).unwrap();
        // SAFETY: u16 -> u8 view of the same memory.
        be.upload(&b, 0, unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }, 0).unwrap();
        b
    }

    pub struct Case {
        pub kernel: &'static str,
        pub call: OpCall,
    }

    fn attrs(win: usize, ring: bool, full: bool, pos0: usize) -> OpAttrs {
        let mut a = OpAttrs { pos0, n_heads: H, n_kv_heads: HK, head_dim: DH, window: win, scale: (DH as f32).powf(-0.5), eps: 1e-6, ..OpAttrs::default() };
        a.ink.qk_clip = 22.6;
        a.ink.rel_clip = 2.0;
        a.ink.d_rel = DR;
        a.ink.kv_ring = ring;
        a.ink.dyn_conv = full;
        if full {
            (a.ink.tau_floor, a.ink.tau_alpha) = (1024, 0.1);
        }
        a
    }

    /// The prologue + attention pair of one visit (shared intermediates), `full`: full-attention layer.
    pub fn visit(be: &MetalBackend, rng: &mut Rng, rows: usize, ctx: usize, full: bool) -> (Case, Case) {
        let ckv = HK * DH;
        let pw = if full { ckv + HK * K } else { ckv };
        let nqw = nq(H, DH, DR);
        let pos0 = ctx;
        let (win, ring, cr) = if full { (0, false, ctx + rows + 16) } else { (1024, true, 1040) };
        let a = attrs(win, ring, full, pos0);
        let (pq, pk, pv) = (buf(be, rows * nqw, rng, 1.0), buf(be, rows * pw, rng, 1.0), buf(be, rows * pw, rng, 1.0));
        let w: Vec<Buffer> = [K * ckv, K * ckv, H * DH, H * DR, DR * E].iter().map(|&n| buf(be, n, rng, 1.0)).collect();
        let (kr, vr) = (buf(be, RR * ckv, rng, 1.0), buf(be, RR * ckv, rng, 1.0));
        let (q, rel, k, v, o) = (buf(be, rows * H * DH, rng, 1.0), buf(be, rows * H * E, rng, 1.0), buf(be, rows * ckv, rng, 1.0), buf(be, rows * ckv, rng, 1.0), buf(be, rows * H * DH, rng, 1.0));
        let (kc, vc) = (buf(be, HK * cr * DH, rng, 1.0), buf(be, HK * cr * DH, rng, 1.0));
        let pro = OpCall::new(
            Op::InkKvPrologue,
            vec![t(&pq, &[rows, nqw]), t(&pk, &[rows, pw]), t(&pv, &[rows, pw]), t(&w[0], &[K, ckv]), t(&w[1], &[K, ckv]), t(&w[2], &[H, DH]), t(&w[3], &[H, DR]), t(&w[4], &[DR, E])],
            vec![t(&q, &[rows, H, DH]), t(&rel, &[rows, H, E]), t(&k, &[rows, HK, DH]), t(&v, &[rows, HK, DH]), t(&kr, &[RR, ckv]), t(&vr, &[RR, ckv])],
            a.clone(),
        );
        let mut att = OpCall::new(
            Op::RelSinkAttn,
            vec![t(&q, &[rows, H, DH]), t(&k, &[rows, HK, DH]), t(&v, &[rows, HK, DH]), t(&kc, &[HK, cr, DH]), t(&vc, &[HK, cr, DH]), t(&rel, &[rows, H, E]), t(&pq, &[rows, nqw])],
            vec![t(&o, &[rows, H * DH]), t(&kc, &[HK, cr, DH]), t(&vc, &[HK, cr, DH])],
            a,
        );
        let need = be.kernels().iter().find(|k| k.name == "rel_sink_attn_bf16").unwrap().scratch_bytes(&ShapeInfo::from_call(&att));
        let sb = be.alloc(need.max(16), MemKind::Device).unwrap();
        att.scratch = Some(TensorRef { buf: sb, offset: 0, dtype: DType::U8, shape: vec![need.max(16)] });
        (Case { kernel: "ink_kv_prologue_bf16", call: pro }, Case { kernel: "rel_sink_attn_bf16", call: att })
    }

    fn floor_case(be: &MetalBackend, rng: &mut Rng) -> Case {
        let (a, b, c) = (buf(be, 32, rng, 1.0), buf(be, 32, rng, 1.0), buf(be, 32, rng, 1.0));
        Case { kernel: "add_bf16", call: OpCall::new(Op::Add, vec![t(&a, &[1, 32]), t(&b, &[1, 32])], vec![t(&c, &[1, 32])], OpAttrs::default()) }
    }

    /// Median / p10 / p90 GPU µs per launch of the case list (each command buffer runs the whole list once).
    pub fn time(be: &MetalBackend, trace: &std::path::Path, cases: &[&Case], reps: usize) -> (f64, f64, f64) {
        let mut samples = Vec::new();
        let mut seen = std::fs::read_to_string(trace).unwrap_or_default().lines().count();
        for rep in 0..reps + 3 {
            for c in cases {
                be.launch(c.kernel, &c.call, 0).unwrap();
            }
            be.synchronize(0).unwrap();
            let text = std::fs::read_to_string(trace).unwrap_or_default();
            let lines: Vec<&str> = text.lines().collect();
            if lines.len() > seen && rep >= 3 {
                // every command buffer of the rep (the stream may commit more than one): the sum of their GPU spans
                let mut gpu = 0.0;
                for l in &lines[seen..] {
                    let r: serde_json::Value = serde_json::from_str(l).unwrap();
                    gpu += r["e"].as_f64().unwrap() - r["s"].as_f64().unwrap();
                }
                samples.push(gpu / cases.len() as f64 * 1e6);
            }
            seen = lines.len();
        }
        let st = stats(&mut samples);
        (st.median, st.p10, st.p90)
    }

    pub fn main() {
        let mut layers = 16usize;
        let mut reps = 60usize;
        let mut prefill = false;
        let mut args = std::env::args().skip(1);
        while let Some(a) = args.next() {
            match a.as_str() {
                "--layers" => layers = args.next().unwrap().parse().unwrap(),
                "--reps" => reps = args.next().unwrap().parse().unwrap(),
                "--prefill" => prefill = true,
                o => panic!("unknown arg {o}"),
            }
        }
        let trace = std::env::temp_dir().join(format!("ink_attn_bench-{}.jsonl", std::process::id()));
        std::env::set_var("INTARSIA_METAL_PROFILE", format!("cb:{}", trace.display()));
        let be = MetalBackend::new().expect("metal backend");
        let mut rng = Rng(0x9e3779b97f4a7c15);
        println!("{layers} launches (distinct operands) per command buffer, {reps} reps; GPU us per launch: median p10 p90");
        let fl: Vec<Case> = (0..layers).map(|_| floor_case(&be, &mut rng)).collect();
        for pass in 0..2 {
            let (m, lo, hi) = time(&be, &trace, &fl.iter().collect::<Vec<_>>(), reps);
            if pass == 1 {
                println!("{:<44} {m:>8.2} {lo:>8.2} {hi:>8.2}", "floor: add_bf16 x32");
            }
        }
        let rows_set: &[usize] = if prefill { &[2048] } else { &[1, 4, 16] };
        let ctxs: &[usize] = if prefill { &[0, 2048, 30720] } else { &[512, 4096, 32768] };
        for &full in &[false, true] {
            for &ctx in ctxs {
                for &rows in rows_set {
                    let n = if prefill { layers.min(4) } else { layers };
                    let vs: Vec<(Case, Case)> = (0..n).map(|_| visit(&be, &mut rng, rows, ctx, full)).collect();
                    let pro: Vec<&Case> = vs.iter().map(|v| &v.0).collect();
                    let att: Vec<&Case> = vs.iter().map(|v| &v.1).collect();
                    let both: Vec<&Case> = vs.iter().flat_map(|v| [&v.0, &v.1]).collect();
                    let r = if prefill { reps.min(10) } else { reps };
                    let _ = time(&be, &trace, &both, 2);
                    let (pm, _, _) = time(&be, &trace, &pro, r);
                    let (am, alo, ahi) = time(&be, &trace, &att, r);
                    let (bm, _, _) = time(&be, &trace, &both, r);
                    let kind = if full { "full" } else { "swa " };
                    println!(
                        "{kind} ctx {ctx:>6} rows {rows:>4}: prologue {pm:>8.2}  attn {am:>8.2} ({alo:.2}..{ahi:.2})  pair/visit {:>8.2}",
                        2.0 * bm
                    );
                }
            }
        }
        let _ = std::fs::remove_file(&trace);
    }
}
