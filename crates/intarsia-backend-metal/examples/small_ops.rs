//! Small decode-op microbenchmark and output fingerprint (metal phase B6): per-call GPU time of `RmsNormAdd`,
//! `QkNormRopeKv`, `SiluGlu` and `Argmax` at the Qwen3-8B decode shapes, `layers` dependent launches per command buffer
//! (the serial encoder puts a barrier between them, as in a decode step). GPU time is the command buffers'
//! `GPUStartTime` / `GPUEndTime` from the backend's `INTARSIA_METAL_PROFILE=cb:<file>` trace (this example sets it),
//! divided by the dispatches in the buffer. Then an FNV-1a hash of each op's output on fixed random data over several
//! shapes and element offsets, so a pre and a new build can be compared bit for bit.
//!
//!   cargo run --release -p intarsia-backend-metal --example small_ops -- [--layers L] [--reps R] [--hash-only] [op...]
//!
//! `op` is `norm_add`, `qkrope`, `silu` or `argmax` (default: all). Run inside a fleetq slot.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("small_ops needs macOS / Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    imp::main()
}

#[cfg(target_os = "macos")]
mod imp {
    use intarsia_backend_metal::bench::{f32_to_bf16, stats, Rng};
    use intarsia_backend_metal::kernels::{ARGMAX_F32, QK_NORM_ROPE_KV_BF16, RMSNORM_ADD_BF16, SILU_GLU_BF16};
    use intarsia_backend_metal::MetalBackend;
    use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
    use intarsia_core::{DType, Op, OpAttrs, OpCall};

    fn t(b: &Buffer, off_el: usize, dt: DType, shape: &[usize]) -> TensorRef {
        TensorRef { buf: b.clone(), offset: off_el * dt.size_bytes(), dtype: dt, shape: shape.to_vec() }
    }

    fn bf_buf(be: &MetalBackend, off: usize, n: usize, rng: &mut Rng) -> Buffer {
        let v: Vec<u16> = (0..off + n).map(|_| f32_to_bf16(rng.unit() * 2.0 - 1.0)).collect();
        let b = be.alloc((off + n) * 2 + 64, MemKind::Device).unwrap();
        // SAFETY: u16 -> u8 view of the same memory.
        be.upload(&b, 0, unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }, 0).unwrap();
        b
    }

    fn f32_buf(be: &MetalBackend, n: usize, rng: &mut Rng) -> Buffer {
        let v: Vec<f32> = (0..n).map(|_| rng.unit() * 20.0 - 10.0).collect();
        let b = be.alloc(n * 4 + 64, MemKind::Device).unwrap();
        // SAFETY: f32 -> u8 view of the same memory.
        be.upload(&b, 0, unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n * 4) }, 0).unwrap();
        b
    }

    fn fnv(h: &mut u64, be: &MetalBackend, b: &Buffer, bytes: usize) {
        let mut raw = vec![0u8; bytes];
        be.download(b, 0, &mut raw, 0).unwrap();
        for x in raw {
            *h ^= x as u64;
            *h = h.wrapping_mul(0x100000001b3);
        }
    }

    /// One op instance: the kernel, its call, and the output buffers (with their byte sizes) to fingerprint.
    struct Case {
        kernel: &'static str,
        call: OpCall,
        outs: Vec<(Buffer, usize)>,
    }

    fn norm_add(be: &MetalBackend, rng: &mut Rng, rows: usize, h: usize, off: usize) -> Case {
        let (x, r, w) = (bf_buf(be, off, rows * h, rng), bf_buf(be, off, rows * h, rng), bf_buf(be, off, h, rng));
        let (n, s) = (bf_buf(be, off, rows * h, rng), bf_buf(be, off, rows * h, rng));
        let bf = DType::Bf16;
        let call = OpCall::new(
            Op::RmsNormAdd,
            vec![t(&x, off, bf, &[rows, h]), t(&r, off, bf, &[rows, h]), t(&w, off, bf, &[h])],
            vec![t(&n, off, bf, &[rows, h]), t(&s, off, bf, &[rows, h])],
            OpAttrs { eps: 1e-6, w_offset: 0.0, ..Default::default() },
        );
        Case { kernel: RMSNORM_ADD_BF16, call, outs: vec![(n, (off + rows * h) * 2), (s, (off + rows * h) * 2)] }
    }

    #[allow(clippy::too_many_arguments)]
    fn qkrope(
        be: &MetalBackend,
        rng: &mut Rng,
        tk: usize,
        n: usize,
        nkv: usize,
        d: usize,
        rd: usize,
        cap: usize,
        pos0: usize,
        off: usize,
    ) -> Case {
        let bf = DType::Bf16;
        let (q, k, v) =
            (bf_buf(be, off, tk * n * d, rng), bf_buf(be, off, tk * nkv * d, rng), bf_buf(be, off, tk * nkv * d, rng));
        let (wq, wk) = (bf_buf(be, off, d, rng), bf_buf(be, off, d, rng));
        let qo = bf_buf(be, off, tk * n * d, rng);
        let (kc, vc) = (bf_buf(be, off, nkv * cap * d, rng), bf_buf(be, off, nkv * cap * d, rng));
        let call = OpCall::new(
            Op::QkNormRopeKv,
            vec![
                t(&q, off, bf, &[tk, n, d]),
                t(&k, off, bf, &[tk, nkv, d]),
                t(&v, off, bf, &[tk, nkv, d]),
                t(&wq, off, bf, &[d]),
                t(&wk, off, bf, &[d]),
            ],
            vec![t(&qo, off, bf, &[tk, n, d]), t(&kc, off, bf, &[nkv, cap, d]), t(&vc, off, bf, &[nkv, cap, d])],
            OpAttrs {
                pos0,
                theta: 1_000_000.0,
                eps: 1e-6,
                w_offset: 0.0,
                head_dim: d,
                rotary_dim: rd,
                n_heads: n,
                n_kv_heads: nkv,
                kv_cap: cap,
                ..Default::default()
            },
        );
        let cb = (off + nkv * cap * d) * 2;
        Case { kernel: QK_NORM_ROPE_KV_BF16, call, outs: vec![(qo, (off + tk * n * d) * 2), (kc, cb), (vc, cb)] }
    }

    fn silu(be: &MetalBackend, rng: &mut Rng, n: usize, off: usize) -> Case {
        let bf = DType::Bf16;
        let (g, u, o) = (bf_buf(be, off, n, rng), bf_buf(be, off, n, rng), bf_buf(be, off, n, rng));
        let call = OpCall::new(
            Op::SiluGlu,
            vec![t(&g, off, bf, &[1, n]), t(&u, off, bf, &[1, n])],
            vec![t(&o, off, bf, &[1, n])],
            OpAttrs::default(),
        );
        Case { kernel: SILU_GLU_BF16, call, outs: vec![(o, (off + n) * 2)] }
    }

    fn argmax(be: &MetalBackend, rng: &mut Rng, rows: usize, v: usize) -> Case {
        let l = f32_buf(be, rows * v, rng);
        let o = be.alloc(rows * 4 + 64, MemKind::Device).unwrap();
        let call = OpCall::new(
            Op::Argmax,
            vec![t(&l, 0, DType::F32, &[rows, v])],
            vec![t(&o, 0, DType::I32, &[rows])],
            OpAttrs::default(),
        );
        Case { kernel: ARGMAX_F32, call, outs: vec![(o, rows * 4)] }
    }

    fn bench(be: &MetalBackend, trace: &std::path::Path, label: &str, cases: &[Case], reps: usize, print: bool) {
        let mut samples = Vec::new();
        let mut seen = std::fs::read_to_string(trace).unwrap_or_default().lines().count();
        for rep in 0..reps + 3 {
            for c in cases {
                be.launch(c.kernel, &c.call, 0).unwrap();
            }
            be.synchronize(0).unwrap();
            let text = std::fs::read_to_string(trace).unwrap_or_default();
            let lines: Vec<&str> = text.lines().collect();
            if lines.len() > seen {
                let r: serde_json::Value = serde_json::from_str(lines[lines.len() - 1]).unwrap();
                let (s, e) = (r["s"].as_f64().unwrap(), r["e"].as_f64().unwrap());
                if rep >= 3 {
                    samples.push((e - s) / cases.len() as f64 * 1e6);
                }
            }
            seen = lines.len();
        }
        let st = stats(&mut samples);
        if !print {
            return;
        }
        println!("{:<34} {:>8.2} {:>8.2} {:>8.2}", label, st.median, st.p10, st.p90);
    }

    pub fn main() {
        let mut layers = 72usize;
        let mut reps = 400usize;
        let mut hash_only = false;
        let mut ops: Vec<String> = Vec::new();
        let mut args = std::env::args().skip(1);
        while let Some(a) = args.next() {
            match a.as_str() {
                "--layers" => layers = args.next().unwrap().parse().unwrap(),
                "--reps" => reps = args.next().unwrap().parse().unwrap(),
                "--hash-only" => hash_only = true,
                o => ops.push(o.to_string()),
            }
        }
        if ops.is_empty() {
            ops = ["norm_add", "qkrope", "silu", "argmax"].iter().map(|s| s.to_string()).collect();
        }
        let trace = std::env::temp_dir().join(format!("small_ops-{}.jsonl", std::process::id()));
        std::env::set_var("INTARSIA_METAL_PROFILE", format!("cb:{}", trace.display()));
        let be = MetalBackend::new().expect("metal backend");
        let mut rng = Rng(0x9e3779b97f4a7c15);

        if !hash_only {
            println!("{layers} launches per command buffer, {reps} reps");
            println!("{:<34} {:>8} {:>8} {:>8}", "op", "us/call", "p10", "p90");
            // two passes over the ops; only the second is printed (the first brings the GPU off its idle clock)
            for pass in 0..2 {
                for op in &ops {
                    // one operand set per launch, like the per-layer tensors of a decode step
                    let cases: Vec<Case> = (0..layers)
                        .map(|l| match op.as_str() {
                            "norm_add" => norm_add(&be, &mut rng, 1, 4096, 0),
                            "qkrope" => qkrope(&be, &mut rng, 1, 32, 8, 128, 128, 4160, 4000 + l, 0),
                            "silu" => silu(&be, &mut rng, 12288, 0),
                            "argmax" => argmax(&be, &mut rng, 1, 151936),
                            o => panic!("unknown op {o}"),
                        })
                        .collect();
                    bench(&be, &trace, op, &cases, if pass == 0 { reps / 4 } else { reps }, pass == 1);
                }
            }
        }

        // fingerprints: fixed seeds per case, so pre and new builds see the same data
        println!("fingerprints (FNV-1a of every output buffer, guards included)");
        type Mk = Box<dyn Fn(&MetalBackend, &mut Rng) -> Case>;
        let shapes: Vec<(String, Mk)> = vec![
            ("norm_add 1x4096 off0".into(), Box::new(|b, r| norm_add(b, r, 1, 4096, 0))),
            ("norm_add 3x4096 off1".into(), Box::new(|b, r| norm_add(b, r, 3, 4096, 1))),
            ("norm_add 2x128 off0".into(), Box::new(|b, r| norm_add(b, r, 2, 128, 0))),
            ("norm_add 1x100 off1".into(), Box::new(|b, r| norm_add(b, r, 1, 100, 1))),
            ("norm_add 2x5120 off0".into(), Box::new(|b, r| norm_add(b, r, 2, 5120, 0))),
            ("norm_add 1x4097 off1".into(), Box::new(|b, r| norm_add(b, r, 1, 4097, 1))),
            ("norm_add 2x8192 off0".into(), Box::new(|b, r| norm_add(b, r, 2, 8192, 0))),
            ("norm_add 1x12288 off0".into(), Box::new(|b, r| norm_add(b, r, 1, 12288, 0))),
            ("qkrope t1 32/8 d128 pos4000".into(), Box::new(|b, r| qkrope(b, r, 1, 32, 8, 128, 128, 4160, 4000, 0))),
            ("qkrope t5 32/8 d128 pos7 off1".into(), Box::new(|b, r| qkrope(b, r, 5, 32, 8, 128, 128, 64, 7, 1))),
            ("qkrope t3 16/4 d64 rd32 pos0".into(), Box::new(|b, r| qkrope(b, r, 3, 16, 4, 64, 32, 16, 0, 0))),
            ("qkrope t2 8/2 d256 pos61 cap63".into(), Box::new(|b, r| qkrope(b, r, 2, 8, 2, 256, 256, 63, 61, 0))),
            ("qkrope t4 12/3 d96 rd64 pos30 off1".into(), Box::new(|b, r| qkrope(b, r, 4, 12, 3, 96, 64, 40, 30, 1))),
            ("silu 12288 off0".into(), Box::new(|b, r| silu(b, r, 12288, 0))),
            ("silu 12289 off1".into(), Box::new(|b, r| silu(b, r, 12289, 1))),
            ("argmax 1x151936".into(), Box::new(|b, r| argmax(b, r, 1, 151936))),
            ("argmax 3x1000".into(), Box::new(|b, r| argmax(b, r, 3, 1000))),
            ("argmax 2x1001".into(), Box::new(|b, r| argmax(b, r, 2, 1001))),
        ];
        for (i, (name, mk)) in shapes.iter().enumerate() {
            let mut r = Rng(0x51ed270b + i as u64 * 7919);
            let c = mk(&be, &mut r);
            be.launch(c.kernel, &c.call, 0).unwrap();
            be.synchronize(0).unwrap();
            let mut h = 0xcbf29ce484222325u64;
            for (b, n) in &c.outs {
                fnv(&mut h, &be, b, *n);
            }
            println!("{name:<40} {h:016x}");
        }
        let _ = std::fs::remove_file(&trace);
    }
}
