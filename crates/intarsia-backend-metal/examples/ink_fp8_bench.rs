//! Microbenchmark of the Inkling-2-Mini dense projections in bf16 against the opt-in quantised forms (lane ink-fp8,
//! docs/results/2026-11-21-inkling2-fp8.md): the stacked in-projection (one bf16 `InkInProj`, one e4m3 `InkInProj`, three
//! `MatmulFp8` / `MatmulMxfp4` packs), `wo` and the unembed (200058 rows) as `Logits` / `MatmulFp8` / `MatmulMxfp4`. GPU time
//! per call from the backend's `INTARSIA_METAL_PROFILE=cb:<file>` trace, weights rotating through more than the system cache.
//!
//!   cargo run --release -p intarsia-backend-metal --example ink_fp8_bench -- [--reps N] [--rows 1,4,16] [proj|wo|head ...]
//!
//! The e4m3 `InkInProj` family is `INTARSIA_INK_FP8_PROJ_CFG` (one per process). Run inside a fleetq slot (`--quiet gpu,cpu`).

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("ink_fp8_bench needs macOS / Metal");
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
    use intarsia_core::{DType, Op, OpAttrs, OpCall};

    const D: usize = 2304;
    const NS: [usize; 3] = [6552, 1056, 1056];
    const VOCAB: usize = 200058;

    fn t(b: &Buffer, dt: DType, shape: &[usize]) -> TensorRef {
        TensorRef { buf: b.clone(), offset: 0, dtype: dt, shape: shape.to_vec() }
    }

    fn up(be: &MetalBackend, bytes: &[u8]) -> Buffer {
        let b = be.alloc(bytes.len() + 64, MemKind::Device).unwrap();
        be.upload(&b, 0, bytes, 0).unwrap();
        b
    }

    fn bf(be: &MetalBackend, n: usize, scale: f32, rng: &mut Rng) -> Buffer {
        let v: Vec<u8> = (0..n).flat_map(|_| f32_to_bf16(rng.unit() * scale).to_le_bytes()).collect();
        up(be, &v)
    }

    /// e4m3 codes without the NaN codes
    fn e4m3(be: &MetalBackend, n: usize, rng: &mut Rng) -> Buffer {
        let v: Vec<u8> = (0..n)
            .map(|_| {
                let b = ((rng.unit() * 0.5 + 0.5) * 255.0) as u8;
                if b & 0x7F == 0x7F { b ^ 1 } else { b }
            })
            .collect();
        up(be, &v)
    }

    fn bytes_u8(be: &MetalBackend, n: usize, lo: u8, span: u8, rng: &mut Rng) -> Buffer {
        let v: Vec<u8> = (0..n).map(|_| lo + (((rng.unit() * 0.5 + 0.5) * span as f32) as u8).min(span.saturating_sub(1))).collect();
        up(be, &v)
    }

    fn f32s(be: &MetalBackend, v: &[f32]) -> Buffer {
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        up(be, &b)
    }

    struct Case {
        kernel: &'static str,
        call: OpCall,
    }

    /// Median / p10 / p90 GPU microseconds per case (each rep: the case list back to back for about 30 ms of GPU work).
    fn bench(be: &MetalBackend, trace: &std::path::Path, cases: &[Case], reps: usize) -> (f64, f64, f64) {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = std::fs::OpenOptions::new().read(true).open(trace).ok();
        let mut pos = f.as_mut().map_or(0, |f| f.seek(SeekFrom::End(0)).unwrap());
        let mut inner = 1usize;
        let mut samples = Vec::new();
        for rep in 0..reps + 3 {
            for _ in 0..inner {
                for c in cases {
                    be.launch(c.kernel, &c.call, 0).unwrap();
                }
            }
            be.synchronize(0).unwrap();
            if f.is_none() {
                f = std::fs::OpenOptions::new().read(true).open(trace).ok();
            }
            let Some(fh) = f.as_mut() else { continue };
            fh.seek(SeekFrom::Start(pos)).unwrap();
            let mut text = String::new();
            fh.read_to_string(&mut text).unwrap();
            pos += text.len() as u64;
            let mut tot = 0.0;
            for l in text.lines() {
                let r: serde_json::Value = serde_json::from_str(l).unwrap();
                tot += r["e"].as_f64().unwrap() - r["s"].as_f64().unwrap();
            }
            let per = tot / (cases.len() * inner) as f64;
            if rep == 0 {
                inner = ((0.03 / (per * cases.len() as f64).max(1e-7)) as usize).clamp(1, 4000);
            } else if rep >= 3 {
                samples.push(per * 1e6);
            }
        }
        let st = stats(&mut samples);
        (st.median, st.p10, st.p90)
    }

    #[derive(Clone, Copy, PartialEq)]
    enum Fmt {
        Bf16,
        Fp8,
        Mxfp4,
    }

    /// One `[n, k]` weight in `fmt` and its op's weight inputs; `f32_out` for the head.
    fn matmul_case(be: &MetalBackend, rng: &mut Rng, fmt: Fmt, rows: usize, n: usize, k: usize, f32_out: bool) -> Case {
        let x = bf(be, rows * k, 1.0, rng);
        let ydt = if f32_out { DType::F32 } else { DType::Bf16 };
        let y = be.alloc(rows * n * ydt.size_bytes() + 64, MemKind::Device).unwrap();
        let xs = t(&x, DType::Bf16, &[rows, k]);
        let ys = t(&y, ydt, &[rows, n]);
        let multi = rows > 1;
        let (kernel, op, ins, attrs): (&'static str, Op, Vec<TensorRef>, OpAttrs) = match fmt {
            Fmt::Bf16 => {
                let w = t(&bf(be, n * k, 0.02, rng), DType::Bf16, &[n, k]);
                match (f32_out, multi) {
                    (true, false) => (intarsia_backend_metal::kernels::LOGITS_BF16_GEMV, Op::Logits, vec![xs, w], OpAttrs::default()),
                    (true, true) => (intarsia_backend_metal::multirow::LOGITS_BF16_GEMV_ROWS, Op::Logits, vec![xs, w], OpAttrs::default()),
                    (false, false) => ("matmul_bf16_gemv", Op::Matmul, vec![xs, w], OpAttrs::default()),
                    (false, true) => ("matmul_bf16_gemv_rows", Op::Matmul, vec![xs, w], OpAttrs::default()),
                }
            }
            Fmt::Fp8 => {
                let w = t(&e4m3(be, n * k, rng), DType::Fp8E4m3, &[n, k]);
                let s = t(&f32s(be, &[1e-4, 0.0]), DType::F32, &[2]);
                let kn = if multi { intarsia_backend_metal::fp8::MATMUL_FP8_GEMV_ROWS } else { intarsia_backend_metal::fp8::MATMUL_FP8_GEMV };
                (kn, Op::MatmulFp8, vec![xs, w, s], OpAttrs::default())
            }
            Fmt::Mxfp4 => {
                let w = t(&bytes_u8(be, n * k / 2, 0, 255, rng), DType::U8, &[n, k / 2]);
                let s = t(&bytes_u8(be, n * k / 32, 118, 6, rng), DType::U8, &[n, k / 32]);
                let a = t(&f32s(be, &[0.0]), DType::F32, &[1]);
                let kn = if multi { intarsia_backend_metal::mxfp4::MATMUL_MXFP4_GEMV_ROWS } else { intarsia_backend_metal::mxfp4::MATMUL_MXFP4_GEMV };
                (kn, Op::MatmulMxfp4, vec![xs, w, s, a], OpAttrs { mxfp4_w4a16: true, ..OpAttrs::default() })
            }
        };
        let mut call = OpCall::new(op, ins, vec![ys], attrs);
        call.exact_rows = multi;
        call.check().unwrap();
        Case { kernel, call }
    }

    fn proj_case(be: &MetalBackend, rng: &mut Rng, fp8: bool, rows: usize, s: usize) -> Case {
        let n: usize = NS.iter().sum();
        let x = bf(be, s * rows * D, 1.0, rng);
        let ys: Vec<Buffer> = NS.iter().map(|&k| be.alloc(rows * k * 2 + 64, MemKind::Device).unwrap()).collect();
        let mut ins = vec![t(&x, DType::Bf16, &[s, rows, D])];
        let kernel = if fp8 {
            ins.push(t(&e4m3(be, n * D, rng), DType::Fp8E4m3, &[n, D]));
            ins.push(t(&f32s(be, &[1e-4, 3e-5, 2e-4]), DType::F32, &[3]));
            if rows == 1 { intarsia_backend_metal::inkmix::INK_IN_PROJ_F8_GEMV } else { intarsia_backend_metal::inkmix::INK_IN_PROJ_F8_GEMV_ROWS }
        } else {
            ins.push(t(&bf(be, n * D, 0.02, rng), DType::Bf16, &[n, D]));
            if rows == 1 { intarsia_backend_metal::inkmix::INK_IN_PROJ_BF16_GEMV } else { intarsia_backend_metal::inkmix::INK_IN_PROJ_BF16_GEMV_ROWS }
        };
        let mut call =
            OpCall::new(Op::InkInProj, ins, NS.iter().zip(&ys).map(|(&k, y)| t(y, DType::Bf16, &[rows, k])).collect(), OpAttrs::default());
        call.exact_rows = rows > 1;
        call.check().unwrap();
        Case { kernel, call }
    }

    pub fn main() {
        let mut reps = 40usize;
        let mut rows_list = vec![1usize, 4, 16];
        let mut what: Vec<String> = Vec::new();
        let mut args = std::env::args().skip(1);
        while let Some(a) = args.next() {
            match a.as_str() {
                "--reps" => reps = args.next().unwrap().parse().unwrap(),
                "--rows" => rows_list = args.next().unwrap().split(',').map(|v| v.parse().unwrap()).collect(),
                o => what.push(o.to_string()),
            }
        }
        if what.is_empty() {
            what = vec!["proj".into(), "wo".into(), "head".into()];
        }
        let trace = std::env::temp_dir().join(format!("ink_fp8_bench-{}.jsonl", std::process::id()));
        std::env::set_var("INTARSIA_METAL_PROFILE", format!("cb:{}", trace.display()));
        let be = MetalBackend::new().expect("metal backend");
        let mut rng = Rng(0x5eed_f8);
        let nw: usize = NS.iter().sum();
        println!("{:<52} {:>9} {:>9} {:>9} {:>8}", "case", "us/call", "p10", "p90", "GB/s");
        let line = |name: String, (m, lo, hi): (f64, f64, f64), bytes: f64| {
            println!("{name:<52} {m:>9.2} {lo:>9.2} {hi:>9.2} {:>8.1}", bytes / (m * 1e-6) / 1e9);
        };
        let wbytes = |fmt: Fmt, n: usize, k: usize| -> f64 {
            match fmt {
                Fmt::Bf16 => (n * k * 2) as f64,
                Fmt::Fp8 => (n * k) as f64,
                Fmt::Mxfp4 => (n * k / 2 + n * k / 32) as f64,
            }
        };
        let fname = |f: Fmt| match f {
            Fmt::Bf16 => "bf16",
            Fmt::Fp8 => "fp8",
            Fmt::Mxfp4 => "mxfp4",
        };
        for w in &what {
            match w.as_str() {
                "proj" => {
                    for &r in &rows_list {
                        for fp8 in [false, true] {
                            let sets = if fp8 { 24 } else { 12 };
                            let cases: Vec<Case> = (0..sets).map(|_| proj_case(&be, &mut rng, fp8, r, 3)).collect();
                            let b = wbytes(if fp8 { Fmt::Fp8 } else { Fmt::Bf16 }, nw, D);
                            line(format!("InkInProj {} R={r} S=3", if fp8 { "e4m3" } else { "bf16" }), bench(&be, &trace, &cases, reps), b);
                        }
                        for fmt in [Fmt::Fp8, Fmt::Mxfp4] {
                            // three packs per visit, 16 visits' worth of weights
                            let cases: Vec<Case> =
                                (0..16).flat_map(|_| NS.map(|n| n)).map(|n| matmul_case(&be, &mut rng, fmt, r, n, D, false)).collect();
                            let (m, lo, hi) = bench(&be, &trace, &cases, reps);
                            line(format!("3 x {} packs (per visit) R={r}", fname(fmt)), (m * 3.0, lo * 3.0, hi * 3.0), wbytes(fmt, nw, D));
                        }
                    }
                }
                "wo" => {
                    for &r in &rows_list {
                        for fmt in [Fmt::Bf16, Fmt::Fp8, Fmt::Mxfp4] {
                            let cases: Vec<Case> = (0..32).map(|_| matmul_case(&be, &mut rng, fmt, r, 2304, 3072, false)).collect();
                            line(format!("wo [2304,3072] {} R={r}", fname(fmt)), bench(&be, &trace, &cases, reps), wbytes(fmt, 2304, 3072));
                        }
                    }
                }
                "head" => {
                    for &r in &rows_list {
                        for fmt in [Fmt::Bf16, Fmt::Fp8, Fmt::Mxfp4] {
                            let cases: Vec<Case> = (0..2).map(|_| matmul_case(&be, &mut rng, fmt, r, VOCAB, D, true)).collect();
                            line(format!("unembed [{VOCAB},{D}] {} R={r}", fname(fmt)), bench(&be, &trace, &cases, reps.min(20)), wbytes(fmt, VOCAB, D));
                        }
                    }
                }
                o => panic!("unknown benchmark {o}"),
            }
        }
        let _ = std::fs::remove_file(&trace);
    }
}
