//! Decode GEMV bandwidth of `matmul_mxfp4_gemv` (M = 1) at the Qwen3.8-27B shapes: `launch` of a pool of distinct weight
//! buffers (>> cache), wall time of a synced batch (so it includes the host encoding and the activation pre-pass dispatch).
//! Reports weight bytes (codes + scales) per second.
//!
//!   cargo run --release -p intarsia-backend-metal --example mxfp4_bw            all shapes
//!   MXFP4_BW_SHAPE=17408x5120 ... mxfp4_bw                                      one shape (NxK)
//!   MXFP4_BW_M=4                                                                 activation rows (default 1)
//!   MXFP4_BW_ACT=1                                                               W4A4 (activations QDQ'd per group of 32; default 0 = W4A16)
//!
//! Run it inside a fleetq slot (`fleetq run --host endor --quiet gpu,cpu --mem 8 ...`).

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("mxfp4_bw needs macOS / Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    use intarsia_backend_metal::bench::Rng;
    use intarsia_backend_metal::MetalBackend;
    use intarsia_core::backend::{Backend, MemKind, TensorRef};
    use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};
    use objc2_metal::MTLDevice;
    use std::time::Instant;

    let be = MetalBackend::new().expect("metal backend");
    println!("device: {}", be.device().name());
    let t = |b: &intarsia_core::backend::Buffer, dt: DType, shape: &[usize]| TensorRef {
        buf: b.clone(),
        offset: 0,
        dtype: dt,
        shape: shape.to_vec(),
    };
    let fill = |bytes: usize, seed: u64, f: &dyn Fn(u8) -> u8| -> intarsia_core::backend::Buffer {
        let mut r = Rng(seed | 1);
        let v: Vec<u8> = (0..bytes).map(|_| f((r.next_u64() >> 40) as u8)).collect();
        let b = be.alloc(bytes + 16, MemKind::Device).unwrap();
        be.upload(&b, 0, &v, 0).unwrap();
        b
    };
    let m: usize = std::env::var("MXFP4_BW_M").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let all: Vec<(&str, usize, usize)> = vec![
        ("mlp gate/up", 17408, 5120),
        ("mlp down", 5120, 17408),
        ("gdn in_proj_qkv", 10240, 5120),
        ("gdn in_proj_z", 6144, 5120),
        ("gdn/attn out_proj", 5120, 6144),
        ("attn q_proj (gated)", 12288, 5120),
        ("attn k/v proj", 1024, 5120),
    ];
    let only: Option<(usize, usize)> = std::env::var("MXFP4_BW_SHAPE").ok().and_then(|v| {
        let (a, b) = v.split_once('x')?;
        Some((a.parse().ok()?, b.parse().ok()?))
    });
    let kernel = std::env::var("MXFP4_BW_KERNEL").unwrap_or_else(|_| "matmul_mxfp4_gemv".to_string());
    const POOL_BYTES: usize = 1 << 30;
    for (what, n, k) in all {
        if let Some((on, ok)) = only {
            if (on, ok) != (n, k) {
                continue;
            }
        }
        let wbytes = n * k / 2;
        let sbytes = n * k / 32;
        let pool = ((POOL_BYTES / (wbytes + sbytes)).clamp(2, 256)).max(2);
        let ws: Vec<_> = (0..pool).map(|i| fill(wbytes, 10 + i as u64, &|b| b)).collect();
        // scales around 2^-8 .. 2^3
        let ss: Vec<_> = (0..pool).map(|i| fill(sbytes, 400 + i as u64, &|b| 119 + b % 12)).collect();
        let x = fill(m * k * 2, 7, &|b| b & 0x3f | 0x3c);
        let act = be.alloc(16, MemKind::Device).unwrap();
        let actv: f32 = std::env::var("MXFP4_BW_ACT").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
        be.upload(&act, 0, &actv.to_le_bytes(), 0).unwrap();
        let y = be.alloc(m * n * 2 + 16, MemKind::Device).unwrap();
        let calls: Vec<OpCall> = (0..pool)
            .map(|i| {
                let mut c = OpCall::new(
                    Op::MatmulMxfp4,
                    vec![
                        t(&x, DType::Bf16, &[m, k]),
                        t(&ws[i], DType::U8, &[n, k / 2]),
                        t(&ss[i], DType::U8, &[n, k / 32]),
                        t(&act, DType::F32, &[1]),
                    ],
                    vec![t(&y, DType::Bf16, &[m, n])],
                    // the model marks a W4A16 call (act 0); such calls take the fused GEMV
                    OpAttrs { mxfp4_w4a16: actv == 0.0, ..OpAttrs::default() },
                );
                let need =
                    be.kernels().iter().find(|q| q.name == kernel).unwrap().scratch_bytes(&ShapeInfo::from_call(&c));
                let sb = be.alloc(need + 8, MemKind::Device).unwrap();
                c.scratch = Some(t(&sb, DType::U8, &[need + 8]));
                c
            })
            .collect();
        let run = |reps: usize| {
            for r in 0..reps {
                be.launch(&kernel, &calls[r % pool], 0).unwrap();
            }
            be.synchronize(0).unwrap();
        };
        // output hash of call 0 (bitwise comparison across variants of the same shape)
        be.launch(&kernel, &calls[0], 0).unwrap();
        be.synchronize(0).unwrap();
        let mut raw = vec![0u8; m * n * 2];
        be.download(&y, 0, &mut raw, 0).unwrap();
        let hash =
            raw.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3));
        run(pool.min(20));
        let reps = (pool * 4).max(400);
        let mut best = f64::MAX;
        let mut all_us = vec![];
        for _ in 0..5 {
            let t0 = Instant::now();
            run(reps);
            let us = t0.elapsed().as_secs_f64() / reps as f64 * 1e6;
            all_us.push(us);
            best = best.min(us);
        }
        all_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = all_us[all_us.len() / 2];
        let bytes = (wbytes + sbytes) as f64;
        println!(
            "{what:22} n={n:6} k={k:5} m={m} h={hash:016x}: median {med:7.1} us ({:6.1} GB/s)  best {best:7.1} us ({:6.1} GB/s)",
            bytes / (med * 1e-6) / 1e9,
            bytes / (best * 1e-6) / 1e9
        );
    }
}
