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
