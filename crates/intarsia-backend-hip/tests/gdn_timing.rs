//! Timing of the GatedDeltaNet kernels at the Qwen3.6-35B-A3B shapes (16 key / 32 value heads of 128, conv width
//! 8192, kernel 4). Prints microseconds per launch; only asserts that the launches succeed. Skips without a GPU.
#![cfg(ik_hip)]

use intarsia_core::{DType, MemKind, Op, OpAttrs, OpCall, TensorRef};
use std::time::Instant;

fn t(b: &dyn intarsia_core::Backend, dt: DType, shape: &[usize]) -> TensorRef {
    let n: usize = shape.iter().product();
    let buf = b.alloc(n * dt.size_bytes().max(1), MemKind::Device).unwrap();
    b.zero(&buf, 0).unwrap();
    TensorRef::new(buf, dt, shape)
}

#[test]
fn gdn_kernels_timing_at_35b_shapes() {
    let Ok(b) = intarsia_backend_hip::create(0) else {
        eprintln!("skipping: no usable CUDA device");
        return;
    };
    let (hk, hv, dk, dv, ks, c) = (16usize, 32usize, 128usize, 128usize, 4usize, 8192usize);
    let state = t(&*b, DType::F32, &[hv, dv, dk]);
    let cstate = t(&*b, DType::F32, &[c, ks - 1]);
    let taps = t(&*b, DType::Bf16, &[c, ks]);
    let a_log = t(&*b, DType::Bf16, &[hv]);
    let dtb = t(&*b, DType::Bf16, &[hv]);
    assert_eq!(2 * hk * dk + hv * dv, c);
    for tokens in [1usize, 16, 512, 2048] {
        let x = t(&*b, DType::Bf16, &[tokens, c]);
        let y = t(&*b, DType::Bf16, &[tokens, c]);
        let a = t(&*b, DType::Bf16, &[tokens, hv]);
        let bb = t(&*b, DType::Bf16, &[tokens, hv]);
        let o = t(&*b, DType::Bf16, &[tokens, hv, dv]);
        let conv = OpCall::new(Op::GdnConv, vec![x.clone(), taps.clone()], vec![y.clone(), cstate.clone()], OpAttrs::default());
        let delta = OpCall::new(
            Op::GdnDeltaRule,
            vec![y.clone(), a, bb, a_log.clone(), dtb.clone()],
            vec![o, state.clone()],
            OpAttrs { eps: 1e-6, ..Default::default() },
        );
        for (name, call) in [("gdn_conv_bf16", &conv), ("gdn_delta_rule_bf16", &delta)] {
            b.launch(name, call, 0).unwrap();
            b.synchronize(0).unwrap();
            let iters = if tokens >= 512 { 5 } else { 50 };
            let t0 = Instant::now();
            for _ in 0..iters {
                b.launch(name, call, 0).unwrap();
            }
            b.synchronize(0).unwrap();
            let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            println!("{name:<22} T={tokens:<5} {us:>10.1} us/launch");
        }
    }
}

/// Lane gdn-chunk (docs/results/2026-11-14-hip-gdn-chunk.md): the chunked delta rule against the sequential kernel the registry
/// picks at that T (v2 below `GDN_V3_MIN_TOKENS`, v3 from there), per launch, at the 35B shape and its tp2 shard. The crossover
/// is where `GDN_CHUNK_MIN_TOKENS` comes from. Launches by name, so it needs no knob. Prints only.
#[test]
fn gdn_chunk_vs_sequential_timing() {
    let Ok(b) = intarsia_backend_hip::create(0) else {
        eprintln!("skipping: no usable HIP device");
        return;
    };
    for (hk, hv) in [(16usize, 32usize), (8, 16)] {
        let (dk, dv) = (128usize, 128usize);
        let c = 2 * hk * dk + hv * dv;
        let state = t(&*b, DType::F32, &[hv, dv, dk]);
        let a_log = t(&*b, DType::Bf16, &[hv]);
        let dtb = t(&*b, DType::Bf16, &[hv]);
        for tokens in [33usize, 48, 64, 96, 128, 192, 256, 384, 512, 1024, 2048, 4096] {
            let y = t(&*b, DType::Bf16, &[tokens, c]);
            let a = t(&*b, DType::Bf16, &[tokens, hv]);
            let bb = t(&*b, DType::Bf16, &[tokens, hv]);
            let o = t(&*b, DType::Bf16, &[tokens, hv, dv]);
            let delta = OpCall::new(
                Op::GdnDeltaRule,
                vec![y.clone(), a, bb, a_log.clone(), dtb.clone()],
                vec![o, state.clone()],
                OpAttrs { eps: 1e-6, ..Default::default() },
            );
            let shape = intarsia_core::ShapeInfo { m: tokens, n_heads: hv, n_kv_heads: hk, head_dim: dk, k: dk, n: dv, ..Default::default() };
            let mut chunk = delta.clone();
            chunk.scratch = Some(t(&*b, DType::U8, &[intarsia_backend_hip::kernels::gdn_chunk_scratch(&shape)]));
            let seq = if tokens >= intarsia_backend_hip::kernels::GDN_V3_MIN_TOKENS { "gdn_delta_rule_bf16_v3" } else { "gdn_delta_rule_bf16_v2" };
            let mut us = [0f64; 2];
            for (i, (name, call)) in [(seq, &delta), ("gdn_delta_rule_bf16_chunk", &chunk)].into_iter().enumerate() {
                b.launch(name, call, 0).unwrap();
                b.synchronize(0).unwrap();
                let iters = if tokens >= 512 { 20 } else { 100 };
                let mut best = f64::MAX; // best of 3 batches
                for _ in 0..3 {
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        b.launch(name, call, 0).unwrap();
                    }
                    b.synchronize(0).unwrap();
                    best = best.min(t0.elapsed().as_secs_f64() * 1e6 / iters as f64);
                }
                us[i] = best;
            }
            println!("Hk={hk:<2} Hv={hv:<2} T={tokens:<5} {seq:<22} {:>9.1} us   chunk {:>9.1} us   seq/chunk {:.2}", us[0], us[1], us[0] / us[1]);
        }
    }
}
