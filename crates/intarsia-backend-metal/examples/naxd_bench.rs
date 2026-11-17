//! Lane mprefill: `matmul_mxfp4_nax` per call, the scratch path (activation pre-pass + `mxfp4_w_to_bf16` + the fp8 NAX GEMM)
//! against the in-kernel decode forms (`mxfp4::NAXD_FORMS`, kernels/metal/mxfp4_naxd.metal), at the Qwen3.8-27B prefill
//! shapes and M = 256 .. 2048 (docs/results/2026-11-16-mprefill.md).
//!
//!   cargo run --release -p intarsia-backend-metal --example naxd_bench
//!   env: NAXD_M=256,512,1024,2048  NAXD_SHAPES=17408x5120,...  NAXD_ROUNDS=7  NAXD_ACT=0|1  NAXD_POOL_MIB=1024
//!        INTARSIA_METAL_MXFP4_NAXD_SWZ=<log2> (grid swizzle of the forms)
//!
//! Each round runs every arm once, in an order that rotates per round; an arm = `reps` launches over a rotating pool of
//! distinct weights (so the weights stream from memory as in a prefill pass), timed host-side around `synchronize`. Prints
//! per-arm median us per call, TFLOPS (2*M*N*K / time), the speedup against the scratch path, and an output hash (all
//! arms of one cell must agree: the forms are bitwise the scratch path). Run inside a fleetq `--quiet gpu,cpu` slot.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("naxd_bench needs macOS / Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    imp::main()
}

#[cfg(target_os = "macos")]
mod imp {
    use intarsia_backend_metal::bench::Rng;
    use intarsia_backend_metal::mxfp4::{with_naxd, NAXD_FORMS};
    use intarsia_backend_metal::MetalBackend;
    use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
    use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};
    use std::time::Instant;

    const NAX: &str = "matmul_mxfp4_nax";

    fn list(var: &str, dflt: &str) -> Vec<String> {
        std::env::var(var).unwrap_or_else(|_| dflt.to_string()).split(',').map(|s| s.trim().to_string()).collect()
    }

    pub fn main() {
        let be = MetalBackend::new().expect("metal backend");
        let t = |b: &Buffer, dt: DType, shape: &[usize]| TensorRef {
            buf: b.clone(),
            offset: 0,
            dtype: dt,
            shape: shape.to_vec(),
        };
        let fill = |bytes: usize, seed: u64, f: &dyn Fn(u8) -> u8| -> Buffer {
            let mut r = Rng(seed | 1);
            let v: Vec<u8> = (0..bytes).map(|_| f((r.next_u64() >> 40) as u8)).collect();
            let b = be.alloc(bytes + 16, MemKind::Device).unwrap();
            be.upload(&b, 0, &v, 0).unwrap();
            b
        };
        let ms: Vec<usize> = list("NAXD_M", "256,512,1024,2048").iter().map(|v| v.parse().unwrap()).collect();
        let shapes: Vec<(usize, usize)> = list("NAXD_SHAPES", "17408x5120,5120x17408,10240x5120,5120x6144")
            .iter()
            .map(|s| {
                let (a, b) = s.split_once('x').unwrap();
                (a.parse().unwrap(), b.parse().unwrap())
            })
            .collect();
        let rounds: usize = std::env::var("NAXD_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(7);
        let actv: f32 = std::env::var("NAXD_ACT").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
        let pool_bytes: usize =
            std::env::var("NAXD_POOL_MIB").ok().and_then(|v| v.parse().ok()).unwrap_or(1024usize) << 20;
        let arms: Vec<Option<usize>> = std::iter::once(None).chain((0..NAXD_FORMS.len()).map(Some)).collect();
        let name = |a: &Option<usize>| a.map_or("scratch (3 dispatches)".to_string(), |f| NAXD_FORMS[f].0.to_string());
        for &(n, k) in &shapes {
            let (wbytes, sbytes) = (n * k / 2, n * k / 32);
            let pool = (pool_bytes / (wbytes + sbytes)).clamp(2, 64);
            let ws: Vec<_> = (0..pool).map(|i| fill(wbytes, 10 + i as u64, &|b| b)).collect();
            let ss: Vec<_> = (0..pool).map(|i| fill(sbytes, 400 + i as u64, &|b| 119 + b % 12)).collect();
            let act = be.alloc(16, MemKind::Device).unwrap();
            be.upload(&act, 0, &actv.to_le_bytes(), 0).unwrap();
            for &m in &ms {
                let x = fill(m * k * 2, 7, &|b| b & 0x3f | 0x3c);
                let y = be.alloc(m * n * 2 + 16, MemKind::Device).unwrap();
                let mk = |i: usize, scratch: Option<(&Buffer, usize)>| {
                    let mut c = OpCall::new(
                        Op::MatmulMxfp4,
                        vec![
                            t(&x, DType::Bf16, &[m, k]),
                            t(&ws[i], DType::U8, &[n, k / 2]),
                            t(&ss[i], DType::U8, &[n, k / 32]),
                            t(&act, DType::F32, &[1]),
                        ],
                        vec![t(&y, DType::Bf16, &[m, n])],
                        OpAttrs { mxfp4_w4a16: actv == 0.0, ..OpAttrs::default() },
                    );
                    if let Some((sb, need)) = scratch {
                        c.scratch = Some(t(sb, DType::U8, &[need]));
                    }
                    c
                };
                let need = be
                    .kernels()
                    .iter()
                    .find(|q| q.name == NAX)
                    .unwrap()
                    .scratch_bytes(&ShapeInfo::from_call(&mk(0, None)));
                let sb = be.alloc(need + 16, MemKind::Device).unwrap();
                let calls: Vec<OpCall> = (0..pool).map(|i| mk(i, Some((&sb, need)))).collect();
                // per launch about 2*m*n*k / 40e12 s; aim at about 60 ms per arm and round
                let reps = ((0.06 / (2.0 * (m * n * k) as f64 / 40e12)) as usize).clamp(4, 400);
                let run = |a: Option<usize>, reps: usize| {
                    with_naxd(a, || {
                        for r in 0..reps {
                            be.launch(NAX, &calls[r % pool], 0).unwrap();
                        }
                        be.synchronize(0).unwrap();
                    })
                };
                let mut hashes = vec![];
                for &a in &arms {
                    with_naxd(a, || {
                        be.launch(NAX, &calls[0], 0).unwrap();
                        be.synchronize(0).unwrap();
                    });
                    let mut raw = vec![0u8; m * n * 2];
                    be.download(&y, 0, &mut raw, 0).unwrap();
                    hashes.push(
                        raw.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)),
                    );
                    run(a, reps.min(8));
                }
                let mut us: Vec<Vec<f64>> = vec![vec![]; arms.len()];
                for rd in 0..rounds {
                    for j in 0..arms.len() {
                        let ai = (j + rd) % arms.len();
                        let t0 = Instant::now();
                        run(arms[ai], reps);
                        us[ai].push(t0.elapsed().as_secs_f64() / reps as f64 * 1e6);
                    }
                }
                let med = |v: &mut Vec<f64>| {
                    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    (v[v.len() / 2], v[0], v[v.len() - 1])
                };
                let base = med(&mut us[0].clone()).0;
                // lane msm192: the per-round times (round order), for the every-round ratio
                for (ai, a) in arms.iter().enumerate() {
                    let v: Vec<String> = us[ai].iter().map(|x| format!("{x:.1}")).collect();
                    println!("ROUNDS n={n} k={k} m={m} arm={} us {}", name(a), v.join(" "));
                }
                for (ai, a) in arms.iter().enumerate() {
                    let (md, lo, hi) = med(&mut us[ai]);
                    println!(
                        "RESULT n={n} k={k} m={m} act={actv} arm={:40} med {md:8.1} us [{lo:8.1} .. {hi:8.1}] {:5.1} TFLOPS  x{:.3} vs scratch  h={:016x}{}",
                        name(a),
                        2.0 * (m * n * k) as f64 / (md * 1e-6) / 1e12,
                        base / md,
                        hashes[ai],
                        if hashes[ai] != hashes[0] { "  HASH DIFFERS" } else { "" }
                    );
                }
            }
        }
    }
}
