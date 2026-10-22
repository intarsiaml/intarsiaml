//! Decode GEMV bandwidth of `matmul_fp8_gemv` and `matmul_nvfp4_gemv` at the Qwen3.6-35B shapes (and the NVFP4 lm_head):
//! `launch` of a pool of distinct weight buffers (>> cache), wall time of a synced batch, so it includes the host encoding.
//!
//!   cargo run --release -p intarsia-backend-metal --example quant_bw
//!
//! Run it inside a fleetq slot (`fleetq run --host endor --mem 8 ...`).

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("quant_bw needs macOS / Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    use intarsia_backend_metal::bench::Rng;
    use intarsia_backend_metal::MetalBackend;
    use intarsia_core::backend::{Backend, MemKind, TensorRef};
    use intarsia_core::{DType, Op, OpAttrs, OpCall};
    use objc2_metal::MTLDevice;
    use std::time::Instant;

    let be = MetalBackend::new().expect("metal backend");
    println!("device: {}", be.device().name());
    for mb in [8usize, 16, 64, 512] {
        let g = intarsia_backend_metal::bench::measure_copy_gbs(&be, mb << 20, 5, 30).unwrap();
        println!("copy ceiling, one dispatch of {mb} MB: {g:.0} GB/s (read + write)");
    }
    let t = |b: &intarsia_core::backend::Buffer, dt: DType, shape: &[usize]| TensorRef { buf: b.clone(), offset: 0, dtype: dt, shape: shape.to_vec() };
    let fill = |bytes: usize, seed: u64| -> intarsia_core::backend::Buffer {
        let mut r = Rng(seed | 1);
        let v: Vec<u8> = (0..bytes).map(|_| (r.next_u64() >> 40) as u8 & 0x77).collect();
        let b = be.alloc(bytes, MemKind::Device).unwrap();
        be.upload(&b, 0, &v, 0).unwrap();
        b
    };
    const POOL_BYTES: usize = 1 << 30;
    for (what, op, n, k, m) in [
        ("fp8 in_proj_qkv", Op::MatmulFp8, 8192usize, 2048usize, 1usize),
        ("fp8 in_proj_z", Op::MatmulFp8, 4096, 2048, 1),
        ("fp8 out_proj", Op::MatmulFp8, 2048, 4096, 1),
        ("fp8 k/v proj", Op::MatmulFp8, 512, 2048, 1),
        ("fp8 large (65536 rows)", Op::MatmulFp8, 65536, 2048, 1),
        ("fp8 in_proj_qkv M=4", Op::MatmulFp8, 8192, 2048, 4),
        ("nvfp4 8192x2048", Op::MatmulNvfp4, 8192, 2048, 1),
        ("fp8 16384 rows", Op::MatmulFp8, 16384, 2048, 1),
        ("fp8 32768 rows", Op::MatmulFp8, 32768, 2048, 1),
        ("nvfp4 shared gate", Op::MatmulNvfp4, 512, 2048, 1),
        ("nvfp4 shared down", Op::MatmulNvfp4, 2048, 512, 1),
        ("nvfp4 lm_head", Op::MatmulNvfp4, 248320, 2048, 1),
    ] {
        let wbytes = if op == Op::MatmulFp8 { n * k } else { n * k / 2 };
        let pool = std::env::var("QBW_POOL").ok().and_then(|v| v.parse().ok()).unwrap_or((POOL_BYTES / wbytes).clamp(2, 256));
        let ws: Vec<_> = (0..pool).map(|i| fill(wbytes, 10 + i as u64)).collect();
        let bs: Vec<_> = (0..pool).map(|i| fill(n * k / 16, 400 + i as u64)).collect();
        let x = fill(m * k * 2, 7);
        let sc = be.alloc(8, MemKind::Device).unwrap();
        let mut sb = 0.02f32.to_le_bytes().to_vec();
        sb.extend(0.5f32.to_le_bytes());
        be.upload(&sc, 0, &sb, 0).unwrap();
        let y = be.alloc(m * n * 4, MemKind::Device).unwrap();
        let (kernel, f32_out) = if op == Op::MatmulFp8 { ("matmul_fp8_gemv", false) } else { ("matmul_nvfp4_gemv", true) };
        let calls: Vec<OpCall> = (0..pool)
            .map(|i| {
                if op == Op::MatmulFp8 {
                    OpCall::new(op, vec![t(&x, DType::Bf16, &[m, k]), t(&ws[i], DType::Fp8E4m3, &[n, k]), t(&sc, DType::F32, &[2])], vec![t(&y, DType::Bf16, &[m, n])], OpAttrs::default())
                } else {
                    OpCall::new(
                        op,
                        vec![t(&x, DType::Bf16, &[m, k]), t(&ws[i], DType::U8, &[n, k / 2]), t(&bs[i], DType::Fp8E4m3, &[n, k / 16]), t(&sc, DType::F32, &[1])],
                        vec![t(&y, if f32_out { DType::F32 } else { DType::Bf16 }, &[m, n])],
                        OpAttrs::default(),
                    )
                }
            })
            .collect();
        let run = |reps: usize| {
            for r in 0..reps {
                be.launch(kernel, &calls[r % pool], 0).unwrap();
            }
            be.synchronize(0).unwrap();
        };
        run(pool.min(20));
        let reps = (pool * 4).max(200);
        let t0 = Instant::now();
        run(reps);
        let us = t0.elapsed().as_secs_f64() / reps as f64 * 1e6;
        let bytes = wbytes + if op == Op::MatmulNvfp4 { n * k / 16 } else { 0 };
        println!("{what:22} {kernel:18} n={n:6} k={k:5} m={m}: {us:8.1} us/call  {:6.1} GB/s weight bytes", bytes as f64 / (us * 1e-6) / 1e9);
    }
}
