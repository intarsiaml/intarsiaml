//! `gdn_delta_rule_bf16_v2` (kernels/hip/gdn.hip: the rows kernel without its stalls) must give every bit of
//! `gdn_delta_rule_bf16`: the output rows and the f32 recurrent state, on the 35B's head layout and on small odd ones (a
//! value-row count that leaves lanes idle, token counts around the prefetch depth and the 32-token gate batch, a single
//! decode token). The prefetch depth is `INTARSIA_HIP_GDN_PD` (1 | 2 | 4, read once per process); run the binary once per
//! value. `gdn_v2_timing` (ignored) prints both kernels at the 35B's decode and prefill shapes.
#![cfg(ik_hip)]

use intarsia_core::backend::Backend;
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, MemKind, Op, OpAttrs, OpCall, TensorRef};
use std::sync::Arc;
use std::time::Instant;

fn device() -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_HIP_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_hip::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping: no usable HIP device ({e})");
            None
        }
    }
}

fn rand(rng: &mut SplitMix64, dt: DType, shape: &[usize], scale: f32) -> HostTensor {
    HostTensor::from_f32_as(dt, shape, &rng.normal_vec(shape.iter().product(), scale))
}

/// A delta-rule call: `t` tokens, `hk` key heads, `hv` value heads of `dv` rows, key width 128, a random incoming state.
fn case(rng: &mut SplitMix64, t: usize, hk: usize, hv: usize, dv: usize) -> HostCall {
    let dk = 128;
    let c = 2 * hk * dk + hv * dv;
    let state = rand(rng, DType::F32, &[hv, dv, dk], 0.3);
    HostCall {
        op: Op::GdnDeltaRule,
        inputs: vec![
            rand(rng, DType::Bf16, &[t, c], 1.0),
            rand(rng, DType::Bf16, &[t, hv], 1.0),
            rand(rng, DType::Bf16, &[t, hv], 1.5),
            rand(rng, DType::Bf16, &[hv], 0.5),
            rand(rng, DType::Bf16, &[hv], 0.5),
        ],
        outputs: vec![OutSpec::new(DType::Bf16, &[t, hv, dv]), OutSpec { dtype: DType::F32, shape: vec![hv, dv, dk], init: Some(state) }],
        attrs: OpAttrs { eps: 1e-6, ..Default::default() },
    }
}

#[test]
fn gdn_v2_is_bitwise_the_rows_kernel() {
    let Some(hip) = device() else { return };
    let chunk = std::env::var("INTARSIA_HIP_GDN_PD").unwrap_or_else(|_| "default".into());
    let mut rng = SplitMix64::new(4242);
    let mut shapes: Vec<(usize, usize, usize, usize)> = Vec::new();
    // the 35B: 16 key heads, 32 value heads of 128 rows; token counts around the chunk sizes and a prefill tail
    for t in [1usize, 2, 3, 4, 5, 7, 8, 9, 31, 32, 33, 63, 64, 65, 312, 1000] {
        shapes.push((t, 16, 32, 128));
    }
    // a value-row count that is not a multiple of the 16 rows of a wave (idle lanes), one key head per value head, and
    // four value heads per key head
    shapes.push((37, 4, 4, 40));
    shapes.push((100, 2, 8, 128));
    shapes.push((19, 1, 1, 8));
    let mut nonzero = 0usize;
    for &(t, hk, hv, dv) in &shapes {
        let hc = case(&mut rng, t, hk, hv, dv);
        std::env::set_var("INTARSIA_HIP_SPLIT_US", "0");
        let a = run_host(&*hip, "gdn_delta_rule_bf16", &hc).unwrap();
        let b = run_host(&*hip, "gdn_delta_rule_bf16_v2", &hc).unwrap();
        let what = format!("pd {chunk}: t={t} hk={hk} hv={hv} dv={dv}");
        assert!(a[0].data == b[0].data, "{what}: output rows differ");
        assert!(a[1].data == b[1].data, "{what}: final state differs");
        // round 8: v2 as consecutive token ranges, each continuing the state the previous one stored (INTARSIA_HIP_SPLIT_US,
        // kernels/hip/ik_hip.h): 1 = one token per launch, 7 / 40 = ranges that are no multiple of the prefetch depth or
        // of the 32-token gate batch
        for split in ["1", "7", "40"] {
            std::env::set_var("INTARSIA_HIP_SPLIT_US", split);
            std::env::set_var("INTARSIA_HIP_SPLIT_MIN_US", "0"); // round 9 trigger off: pieces at every size
            let c = run_host(&*hip, "gdn_delta_rule_bf16_v2", &hc).unwrap();
            assert!(a[0].data == c[0].data, "{what}: INTARSIA_HIP_SPLIT_US={split}: output rows differ");
            assert!(a[1].data == c[1].data, "{what}: INTARSIA_HIP_SPLIT_US={split}: final state differs");
        }
        std::env::remove_var("INTARSIA_HIP_SPLIT_US");
        std::env::remove_var("INTARSIA_HIP_SPLIT_MIN_US");
        assert!(a[1].data != hc.outputs[1].init.as_ref().unwrap().data, "{what}: the state did not move");
        nonzero += a[0].to_f32().iter().filter(|v| **v != 0.0).count();
    }
    assert!(nonzero > 0, "all outputs zero: nothing was computed");
    println!("gdn v2 (pd {chunk}): {} shapes bitwise", shapes.len());
}

#[test]
#[ignore]
fn gdn_v2_timing() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(7);
    let (hk, hv, dv, dk) = (16usize, 32usize, 128usize, 128usize);
    let put = |t: &HostTensor| -> TensorRef {
        let buf = hip.alloc(t.nbytes().max(16), MemKind::Device).unwrap();
        hip.upload(&buf, 0, &t.data, 0).unwrap();
        TensorRef::new(buf, t.dtype, &t.shape)
    };
    for t in [1usize, 312, 1024, 2048, 4096] {
        let hc = case(&mut rng, t, hk, hv, dv);
        let ins: Vec<TensorRef> = hc.inputs.iter().map(put).collect();
        let o = put(&HostTensor::zeros(DType::Bf16, &[t, hv, dv]));
        let st = put(&HostTensor::zeros(DType::F32, &[hv, dv, dk]));
        let call = OpCall::new(Op::GdnDeltaRule, ins, vec![o, st], hc.attrs.clone());
        let mut line = format!("gdn T={t:<5}");
        for k in ["gdn_delta_rule_bf16", "gdn_delta_rule_bf16_v2"] {
            hip.launch(k, &call, 0).unwrap();
            hip.synchronize(0).unwrap();
            let iters = if t == 1 { 200 } else { 5 };
            let t0 = Instant::now();
            for _ in 0..iters {
                hip.launch(k, &call, 0).unwrap();
            }
            hip.synchronize(0).unwrap();
            let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            line += &format!("  {k} {us:>9.1} us ({:.2} us/token)", us / t as f64);
        }
        println!("{line}");
    }
}

