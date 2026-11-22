//! Microbenchmark of the lossless P12 GEMVs against their bf16 twins on the Inkling-2-Mini dense projections (lane
//! ink-pack12, docs/results/2026-11-21-inkling2-pack12.md): the stacked in-projection `InkInProj` (S = 1 and 3), `wo`
//! [2304, 3072] and the unembed [200058, 2304] (`Logits`), rows 1 / 4 / 16. Same method as ink_fp8_bench: GPU time per call
//! from the backend's `INTARSIA_METAL_PROFILE=cb:<file>` trace, weight sets rotating through more than the system cache so
//! every call reads its weights from DRAM. Weights are weight-like (normal rows, so each has its natural escapes).
//!
//!   cargo run --release -p intarsia-backend-metal --example ink_p12_bench -- [--reps N] [--rows 1,4,16] [proj|wo|head ...]
//!
//! Run inside a fleetq slot (`--quiet gpu,cpu`).

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("ink_p12_bench needs macOS / Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    imp::main()
}

#[cfg(target_os = "macos")]
mod imp {
    use intarsia_backend_metal::bench::stats;
    use intarsia_backend_metal::MetalBackend;
    use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
    use intarsia_core::quant::p12;
    use intarsia_core::rng::SplitMix64;
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

    fn b16(v: &[u16]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// A weight-like `[n, k]` weight as bf16 and as P12 (codes, aux, bytes).
    struct W {
        bf: TensorRef,
        codes: TensorRef,
        aux: TensorRef,
        p12_bytes: usize,
    }

    fn weight(be: &MetalBackend, rng: &mut SplitMix64, n: usize, k: usize) -> W {
        let w = p12::random_weight(rng, n, k, (k as f32).powf(-0.5), false);
        let p = p12::pack(&w, n, k, 0).unwrap();
        let aux: Vec<u8> = p.aux.iter().flat_map(|v| v.to_le_bytes()).collect();
        W {
            bf: t(&up(be, &b16(&w)), DType::Bf16, &[n, k]),
            codes: t(&up(be, &p.codes), DType::U8, &[n, p12::row_bytes(k)]),
            aux: t(&up(be, &aux), DType::I32, &[p.aux.len()]),
            p12_bytes: p.bytes(),
        }
    }

    fn x_rows(be: &MetalBackend, rng: &mut SplitMix64, n: usize) -> Buffer {
        let v: Vec<u16> = rng.normal_vec(n, 1.0).iter().map(|x| half::bf16::from_f32(*x).to_bits()).collect();
        up(be, &b16(&v))
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

    /// `Matmul` (bf16 out) or `Logits` (f32 out) of `rows` rows on weight `w`, bf16 or P12.
    fn gemv_case(be: &MetalBackend, rng: &mut SplitMix64, w: &W, p12: bool, rows: usize, n: usize, k: usize, f32o: bool) -> Case {
        let x = t(&x_rows(be, rng, rows * k), DType::Bf16, &[rows, k]);
        let ydt = if f32o { DType::F32 } else { DType::Bf16 };
        let y = t(&be.alloc(rows * n * ydt.size_bytes() + 64, MemKind::Device).unwrap(), ydt, &[rows, n]);
        let multi = rows > 1;
        use intarsia_backend_metal::p12::*;
        let (kernel, op, ins) = match (p12, f32o, multi) {
            (false, true, false) => ("logits_bf16_gemv", Op::Logits, vec![x, w.bf.clone()]),
            (false, true, true) => ("logits_bf16_gemv_rows", Op::Logits, vec![x, w.bf.clone()]),
            (false, false, false) => ("matmul_bf16_gemv", Op::Matmul, vec![x, w.bf.clone()]),
            (false, false, true) => ("matmul_bf16_gemv_rows", Op::Matmul, vec![x, w.bf.clone()]),
            (true, true, false) => (MATMUL_P12_LOGITS_GEMV, Op::MatmulP12, vec![x, w.codes.clone(), w.aux.clone()]),
            (true, true, true) => (MATMUL_P12_LOGITS_GEMV_ROWS, Op::MatmulP12, vec![x, w.codes.clone(), w.aux.clone()]),
            (true, false, false) => (MATMUL_P12_GEMV, Op::MatmulP12, vec![x, w.codes.clone(), w.aux.clone()]),
            (true, false, true) => (MATMUL_P12_GEMV_ROWS, Op::MatmulP12, vec![x, w.codes.clone(), w.aux.clone()]),
        };
        let mut call = OpCall::new(op, ins, vec![y], OpAttrs::default());
        call.exact_rows = multi;
        call.check().unwrap();
        Case { kernel, call }
    }

    fn proj_case(be: &MetalBackend, rng: &mut SplitMix64, w: &W, p12: bool, rows: usize, s: usize) -> Case {
        use intarsia_backend_metal::inkmix::{INK_IN_PROJ_BF16_GEMV, INK_IN_PROJ_BF16_GEMV_ROWS};
        use intarsia_backend_metal::p12::{INK_IN_PROJ_P12_GEMV, INK_IN_PROJ_P12_GEMV_ROWS};
        let x = t(&x_rows(be, rng, s * rows * D), DType::Bf16, &[s, rows, D]);
        let ys: Vec<TensorRef> = NS.iter().map(|&k| t(&be.alloc(rows * k * 2 + 64, MemKind::Device).unwrap(), DType::Bf16, &[rows, k])).collect();
        let (kernel, ins) = match (p12, rows > 1) {
            (false, false) => (INK_IN_PROJ_BF16_GEMV, vec![x, w.bf.clone()]),
            (false, true) => (INK_IN_PROJ_BF16_GEMV_ROWS, vec![x, w.bf.clone()]),
            (true, false) => (INK_IN_PROJ_P12_GEMV, vec![x, w.codes.clone(), w.aux.clone()]),
            (true, true) => (INK_IN_PROJ_P12_GEMV_ROWS, vec![x, w.codes.clone(), w.aux.clone()]),
        };
        let mut call = OpCall::new(Op::InkInProj, ins, ys, OpAttrs::default());
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
        let trace = std::env::temp_dir().join(format!("ink_p12_bench-{}.jsonl", std::process::id()));
        std::env::set_var("INTARSIA_METAL_PROFILE", format!("cb:{}", trace.display()));
        let be = MetalBackend::new().expect("metal backend");
        let mut rng = SplitMix64::new(0x5eed_12);
        println!("{:<44} {:>9} {:>9} {:>9} {:>8} {:>8}", "case", "us/call", "p10", "p90", "GB/s", "x bf16");
        let line = |name: String, (m, lo, hi): (f64, f64, f64), bytes: f64, base: Option<f64>| {
            let sp = base.map_or(String::new(), |b| format!("{:.3}", b / m));
            println!("{name:<44} {m:>9.2} {lo:>9.2} {hi:>9.2} {:>8.1} {sp:>8}", bytes / (m * 1e-6) / 1e9);
        };
        // weight sets per shape: more bytes than the system cache, so every call streams its weight from DRAM
        let run = |name: &str, n: usize, k: usize, sets: usize, f32o: bool, proj: bool, rng: &mut SplitMix64| {
            let ws: Vec<W> = (0..sets).map(|_| weight(&be, rng, n, k)).collect();
            let pb = ws[0].p12_bytes as f64;
            for &r in &rows_list {
                for s in if proj { vec![1usize, 3] } else { vec![1] } {
                    let mut base = None;
                    for p12 in [false, true] {
                        let cases: Vec<Case> = ws
                            .iter()
                            .map(|w| if proj { proj_case(&be, rng, w, p12, r, s) } else { gemv_case(&be, rng, w, p12, r, n, k, f32o) })
                            .collect();
                        let res = bench(&be, &trace, &cases, if f32o { reps.min(20) } else { reps });
                        let tag = if proj { format!("{name} S={s} R={r}") } else { format!("{name} R={r}") };
                        line(format!("{tag} {}", if p12 { "p12" } else { "bf16" }), res, if p12 { pb } else { (n * k * 2) as f64 }, base);
                        base.get_or_insert(res.0);
                    }
                }
            }
        };
        for w in &what {
            match w.as_str() {
                "proj" => run("InkInProj [8664,2304]", NS.iter().sum(), D, 12, false, true, &mut rng),
                "wo" => run("wo [2304,3072]", 2304, 3072, 32, false, false, &mut rng),
                "head" => run("unembed [200058,2304]", VOCAB, D, 2, true, false, &mut rng),
                o => panic!("unknown benchmark {o}"),
            }
        }
        let _ = std::fs::remove_file(&trace);
    }
}
