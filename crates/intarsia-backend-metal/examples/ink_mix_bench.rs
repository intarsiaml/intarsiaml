//! Microbenchmark of the Inkling-2-Mini mix / conv / in-projection kernels (lane ink-m-mix,
//! docs/results/2026-11-21-inkling2-metal-mix.md). GPU time per call from the backend's `INTARSIA_METAL_PROFILE=cb:<file>`
//! trace (command-buffer GPU start / end), `L` dependent launches per command buffer, each on its own operand set (weights
//! rotate through more than the system cache holds, like the 48 visits of a decode step).
//!
//!   cargo run --release -p intarsia-backend-metal --example ink_mix_bench -- [--reps N] [--rows 1,4,16] [what...]
//!
//! `what`: `proj` (InkInProj, real shape, S = 1 and 3), `proj3mm` (the same as three Matmul GEMVs), `wo` (the `wo_ud`
//! Matmul, [2304, 3072]), `prefill` (InkInProj and wo at 2048 rows), `mix` (ConvMix / AttnResMix / LoopBlend / SconvRes).
//! Run inside a fleetq slot (`--quiet gpu,cpu`).

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("ink_mix_bench needs macOS / Metal");
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

    pub const D: usize = 2304;
    pub const NS: [usize; 3] = [6552, 1056, 1056];

    pub fn t(b: &Buffer, dt: DType, shape: &[usize]) -> TensorRef {
        TensorRef { buf: b.clone(), offset: 0, dtype: dt, shape: shape.to_vec() }
    }

    pub fn bf_buf(be: &MetalBackend, n: usize, scale: f32, rng: &mut Rng) -> Buffer {
        let v: Vec<u16> = (0..n).map(|_| f32_to_bf16(rng.unit() * scale)).collect();
        let b = be.alloc(n * 2 + 64, MemKind::Device).unwrap();
        // SAFETY: u16 -> u8 view of the same memory.
        be.upload(&b, 0, unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }, 0).unwrap();
        b
    }

    pub fn f32_buf(be: &MetalBackend, n: usize, scale: f32, rng: &mut Rng) -> Buffer {
        let v: Vec<f32> = (0..n).map(|_| rng.unit() * scale).collect();
        let b = be.alloc(n * 4 + 64, MemKind::Device).unwrap();
        // SAFETY: f32 -> u8 view of the same memory.
        be.upload(&b, 0, unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n * 4) }, 0).unwrap();
        b
    }

    pub struct Case {
        pub kernel: &'static str,
        pub call: OpCall,
    }

    /// Median / p10 / p90 GPU microseconds per case: each rep launches the case list `inner` times back to back (so the GPU
    /// stays busy for tens of milliseconds and runs at its working clock, as in a decode stream), then sums the GPU time of
    /// the rep's command buffers. 3 warm-up reps.
    pub fn bench(be: &MetalBackend, trace: &std::path::Path, cases: &[Case], reps: usize) -> (f64, f64, f64) {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = std::fs::OpenOptions::new().read(true).open(trace).ok();
        let mut pos = f.as_mut().map_or(0, |f| f.seek(SeekFrom::End(0)).unwrap());
        // aim at about 30 ms of GPU work per rep: estimate from one pass
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

    fn proj_cases(be: &MetalBackend, rng: &mut Rng, sets: usize, rows: usize, s: usize, kernel: &'static str) -> Vec<Case> {
        let n: usize = NS.iter().sum();
        (0..sets)
            .map(|_| {
                let w = bf_buf(be, n * D, 0.02, rng);
                let x = bf_buf(be, s * rows * D, 1.0, rng);
                let ys: Vec<Buffer> = NS.iter().map(|&k| bf_buf(be, rows * k, 0.0, rng)).collect();
                let mut call = OpCall::new(
                    Op::InkInProj,
                    vec![t(&x, DType::Bf16, &[s, rows, D]), t(&w, DType::Bf16, &[n, D])],
                    NS.iter().zip(&ys).map(|(&k, y)| t(y, DType::Bf16, &[rows, k])).collect(),
                    OpAttrs::default(),
                );
                call.exact_rows = rows > 1;
                Case { kernel, call }
            })
            .collect()
    }

    /// The three-Matmul form of InkInProj (what a port without the stacked op launches): per segment one GEMV.
    fn proj3mm_cases(be: &MetalBackend, rng: &mut Rng, sets: usize, rows: usize) -> Vec<Case> {
        let kernel = if rows == 1 { "matmul_bf16_gemv" } else { "matmul_bf16_gemv_rows" };
        let mut v = Vec::new();
        for _ in 0..sets {
            for &k in &NS {
                let w = bf_buf(be, k * D, 0.02, rng);
                let x = bf_buf(be, rows * D, 1.0, rng);
                let y = bf_buf(be, rows * k, 0.0, rng);
                let mut call = OpCall::new(
                    Op::Matmul,
                    vec![t(&x, DType::Bf16, &[rows, D]), t(&w, DType::Bf16, &[k, D])],
                    vec![t(&y, DType::Bf16, &[rows, k])],
                    OpAttrs::default(),
                );
                call.exact_rows = rows > 1;
                v.push(Case { kernel, call });
            }
        }
        v
    }

    fn mm_cases(be: &MetalBackend, rng: &mut Rng, sets: usize, rows: usize, n: usize, k: usize, kernel: &'static str) -> Vec<Case> {
        (0..sets)
            .map(|_| {
                let w = bf_buf(be, n * k, 0.02, rng);
                let x = bf_buf(be, rows * k, 1.0, rng);
                let y = bf_buf(be, rows * n, 0.0, rng);
                let mut call = OpCall::new(
                    Op::Matmul,
                    vec![t(&x, DType::Bf16, &[rows, k]), t(&w, DType::Bf16, &[n, k])],
                    vec![t(&y, DType::Bf16, &[rows, n])],
                    OpAttrs::default(),
                );
                call.exact_rows = rows > 1 && kernel.ends_with("_rows");
                Case { kernel, call }
            })
            .collect()
    }

    pub fn main() {
        let mut reps = 60usize;
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
            what = vec!["proj".into(), "proj3mm".into(), "wo".into()];
        }
        let trace = std::env::temp_dir().join(format!("ink_mix_bench-{}.jsonl", std::process::id()));
        std::env::set_var("INTARSIA_METAL_PROFILE", format!("cb:{}", trace.display()));
        let be = MetalBackend::new().expect("metal backend");
        let mut rng = Rng(0x9e3779b97f4a7c15);
        let nw: usize = NS.iter().sum();
        let wbytes = (nw * D * 2) as f64;
        // 12 weight sets of 40 MB: 480 MB rotate through, far more than the system cache
        let sets = 12;
        println!("{:<44} {:>9} {:>9} {:>9} {:>8}", "case", "us/call", "p10", "p90", "GB/s");
        let line = |name: String, (m, lo, hi): (f64, f64, f64), bytes: f64| {
            println!("{name:<44} {m:>9.2} {lo:>9.2} {hi:>9.2} {:>8.1}", bytes / (m * 1e-6) / 1e9);
        };
        for w in &what {
            match w.as_str() {
                "proj" => {
                    for &r in &rows_list {
                        for s in [1usize, 3] {
                            let k = if r == 1 { "ink_in_proj_bf16_gemv" } else { "ink_in_proj_bf16_gemv_rows" };
                            let cases = proj_cases(&be, &mut rng, sets, r, s, k);
                            let _ = bench(&be, &trace, &cases, 5);
                            let b = wbytes + (s * r * D * 2 + r * nw * 2) as f64;
                            line(format!("InkInProj R={r} S={s}"), bench(&be, &trace, &cases, reps), b);
                        }
                    }
                }
                "proj3mm" => {
                    for &r in &rows_list {
                        let cases = proj3mm_cases(&be, &mut rng, sets, r);
                        let _ = bench(&be, &trace, &cases, 5);
                        let (m, lo, hi) = bench(&be, &trace, &cases, reps);
                        let b = wbytes + (3 * r * D * 2 + r * nw * 2) as f64;
                        // per visit = three launches
                        line(format!("3 x Matmul GEMV (per visit) R={r}"), (m * 3.0, lo * 3.0, hi * 3.0), b);
                    }
                }
                "wo" => {
                    for &r in &rows_list {
                        let k = if r == 1 { "matmul_bf16_gemv" } else { "matmul_bf16_gemv_rows" };
                        let cases = mm_cases(&be, &mut rng, 24, r, 2304, 3072, k);
                        let _ = bench(&be, &trace, &cases, 5);
                        let b = (2304 * 3072 * 2 + r * (3072 + 2304) * 2) as f64;
                        line(format!("wo Matmul [2304,3072] R={r} ({k})"), bench(&be, &trace, &cases, reps), b);
                    }
                }
                "mmfull" => {
                    for &r in &rows_list {
                        let k = if r == 1 { "matmul_bf16_gemv" } else { "matmul_bf16_gemv_rows" };
                        let cases = mm_cases(&be, &mut rng, sets, r, nw, D, k);
                        let _ = bench(&be, &trace, &cases, 5);
                        let b = wbytes + (r * (D + nw) * 2) as f64;
                        line(format!("Matmul [8664,2304] R={r} ({k})"), bench(&be, &trace, &cases, reps), b);
                    }
                }
                "nsweep" => {
                    for n in [1056usize, 3072, 6552, 8664, 13104] {
                        let sets_n = (480usize << 20) / (n * D * 2);
                        let cases = mm_cases(&be, &mut rng, sets_n, 1, n, D, "matmul_bf16_gemv");
                        let _ = bench(&be, &trace, &cases, 5);
                        let b = (n * D * 2 + (D + n) * 2) as f64;
                        line(format!("Matmul GEMV [{n},2304] R=1 x{sets_n}"), bench(&be, &trace, &cases, reps), b);
                    }
                }
                "naxsmall" => {
                    for &r in &rows_list {
                        let cases = mm_cases(&be, &mut rng, sets, r, nw, D, "matmul_bf16_nax");
                        let _ = bench(&be, &trace, &cases, 5);
                        let b = wbytes + (r * (D + nw) * 2) as f64;
                        line(format!("Matmul NAX [8664,2304] R={r}"), bench(&be, &trace, &cases, reps), b);
                    }
                }
                "prefill" => {
                    let r = 2048;
                    for s in [1usize, 3] {
                        let cases = proj_cases(&be, &mut rng, 4, r, s, "ink_in_proj_bf16_nax");
                        let _ = bench(&be, &trace, &cases, 3);
                        let (m, lo, hi) = bench(&be, &trace, &cases, reps.min(20));
                        let flops = 2.0 * (r * nw * D) as f64;
                        println!(
                            "{:<44} {m:>9.1} {lo:>9.1} {hi:>9.1} {:>8.1} TFLOP/s",
                            format!("InkInProj NAX T={r} S={s}"),
                            flops / (m * 1e-6) / 1e12
                        );
                    }
                    let cases = mm_cases(&be, &mut rng, 4, r, 2304, 3072, "matmul_bf16_nax");
                    let _ = bench(&be, &trace, &cases, 3);
                    let (m, lo, hi) = bench(&be, &trace, &cases, reps.min(20));
                    let flops = 2.0 * (r * 2304 * 3072) as f64;
                    println!("{:<44} {m:>9.1} {lo:>9.1} {hi:>9.1} {:>8.1} TFLOP/s", "wo Matmul NAX T=2048", flops / (m * 1e-6) / 1e12);
                }
                "mix" => crate::imp_mix::run(&be, &trace, &mut rng, &rows_list, reps),
                o => panic!("unknown benchmark {o}"),
            }
        }
        let _ = std::fs::remove_file(&trace);
    }
}

#[cfg(target_os = "macos")]
mod imp_mix {
    use intarsia_backend_metal::bench::Rng;
    use intarsia_backend_metal::MetalBackend;

    pub fn run(_be: &MetalBackend, _trace: &std::path::Path, _rng: &mut Rng, _rows: &[usize], _reps: usize) {
        println!("mix: not built yet");
    }
}
