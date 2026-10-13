//! Device-side position on a real GPU (graph capture itself is CUDA-only, see crates/intarsia-backend-cuda/tests/graph_pos.rs).
//!
//! * rope / kv_append / attn_decode (SIMT and split-KV) read their position from an `i32 [1]` tensor when they
//!   run; the result must be *bit-identical* to launching with the same value as the host `pos0` (admission's
//!   `run_host` only exercises the host path, so this is where the device path is checked).
//! * split-KV decode (max-grid launch) still matches the SIMT kernel at every position.
//!
//! Needs a GPU: skips (passes) without one. `INTARSIA_HIP_DEVICE` picks the ordinal.
#![cfg(ik_hip)]

use intarsia_core::rng::SplitMix64;
use intarsia_core::{Backend, DType, HostTensor, MemKind, Op, OpAttrs, OpCall, ShapeInfo, TensorRef};
use std::sync::Arc;

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

fn put(be: &dyn Backend, t: &HostTensor) -> TensorRef {
    let buf = be.alloc(t.nbytes().max(16), MemKind::Device).unwrap();
    be.zero(&buf, 0).unwrap();
    be.upload(&buf, 0, &t.data, 0).unwrap();
    TensorRef::new(buf, t.dtype, &t.shape)
}

fn get(be: &dyn Backend, t: &TensorRef) -> Vec<u8> {
    let mut d = vec![0u8; t.nbytes()];
    be.synchronize(0).unwrap();
    be.download(&t.buf, t.offset, &mut d, 0).unwrap();
    d
}

fn pos_tensor(be: &dyn Backend, p: usize) -> TensorRef {
    put(be, &HostTensor::from_i32(&[1], &[p as i32]))
}

fn rand_bf16(rng: &mut SplitMix64, shape: &[usize], scale: f32) -> HostTensor {
    HostTensor::from_f32_as(DType::Bf16, shape, &rng.normal_vec(shape.iter().product(), scale))
}

/// Launch `kernel` once, with the position given on the host (`pos = None`, `attrs.pos0 = p`) or on the device.
fn launch(
    be: &dyn Backend,
    kernel: &str,
    op: Op,
    inputs: &[TensorRef],
    outputs: &[TensorRef],
    attrs: &OpAttrs,
    p: usize,
    pos: Option<&TensorRef>,
) {
    let mut a = attrs.clone();
    a.pos0 = if pos.is_some() { 0 } else { p };
    let mut call = OpCall::new(op, inputs.to_vec(), outputs.to_vec(), a);
    call.pos = pos.cloned();
    let k = be.kernels().into_iter().find(|k| k.name == kernel).expect("registered kernel");
    let need = k.scratch_bytes(&ShapeInfo::from_call(&call));
    if need > 0 {
        call.scratch = Some(TensorRef::new(be.alloc(need.max(16), MemKind::Device).unwrap(), DType::U8, &[need]));
    }
    be.launch(kernel, &call, 0).unwrap();
}

const POSITIONS: [usize; 9] = [0, 1, 63, 64, 65, 127, 1000, 2047, 4090];

#[test]
fn rope_and_kv_append_device_position_is_bitwise_the_host_position() {
    let Some(be) = device() else { return };
    let mut rng = SplitMix64::new(5);
    for &(n, d, rot) in &[(16usize, 256usize, 64usize), (8, 128, 0), (6, 64, 0)] {
        let x = rand_bf16(&mut rng, &[1, n, d], 1.0);
        for &p in &POSITIONS {
            let attrs = OpAttrs { theta: 1e6, head_dim: d, rotary_dim: rot, ..OpAttrs::default() };
            let run = |dev: bool| {
                let (xi, out) = (put(&*be, &x), put(&*be, &HostTensor::zeros(DType::Bf16, &[1, n, d])));
                let pos = pos_tensor(&*be, p);
                launch(&*be, "rope_bf16", Op::Rope, &[xi], &[out.clone()], &attrs, p, dev.then_some(&pos));
                get(&*be, &out)
            };
            assert!(run(false) == run(true), "rope n={n} d={d} rot={rot} p={p}: device position differs from host");
        }
    }
    for &(nkv, d) in &[(2usize, 256usize), (8, 128)] {
        let cap = 4096 + 8;
        let new = rand_bf16(&mut rng, &[1, nkv, d], 1.0);
        let cache0 = rand_bf16(&mut rng, &[nkv, cap, d], 1.0);
        for &p in &POSITIONS {
            let attrs = OpAttrs { kv_cap: cap, n_kv_heads: nkv, head_dim: d, ..OpAttrs::default() };
            let run = |dev: bool| {
                let (ni, cache) = (put(&*be, &new), put(&*be, &cache0));
                let pos = pos_tensor(&*be, p);
                launch(&*be, "kv_append_bf16", Op::KvAppend, &[ni], &[cache.clone()], &attrs, p, dev.then_some(&pos));
                get(&*be, &cache)
            };
            assert!(run(false) == run(true), "kv_append nkv={nkv} d={d} p={p}: device position differs from host");
        }
    }
}

#[test]
fn attention_decode_device_position_is_bitwise_the_host_position() {
    let Some(be) = device() else { return };
    let mut rng = SplitMix64::new(9);
    // 35B attention (16 heads, 2 kv heads, 256), a 8B-like GQA (32/8/128) and a group of 5
    for &(n, nkv, d) in &[(16usize, 2usize, 256usize), (32, 8, 128), (10, 2, 64)] {
        let cap = 4096 + 8;
        let q = rand_bf16(&mut rng, &[1, n, d], 2.0);
        let kc = rand_bf16(&mut rng, &[nkv, cap, d], 1.0);
        let vc = rand_bf16(&mut rng, &[nkv, cap, d], 1.0);
        let attrs = OpAttrs {
            scale: 1.0 / (d as f32).sqrt(),
            n_heads: n,
            n_kv_heads: nkv,
            head_dim: d,
            kv_cap: cap,
            ..OpAttrs::default()
        };
        for kernel in ["attn_decode_bf16", "attn_decode_bf16_splitkv"] {
            if kernel.ends_with("splitkv") && n / nkv > 8 {
                continue;
            }
            for &p in &POSITIONS {
                let run = |dev: bool| {
                    let (qi, ki, vi) = (put(&*be, &q), put(&*be, &kc), put(&*be, &vc));
                    let out = put(&*be, &HostTensor::zeros(DType::Bf16, &[1, n, d]));
                    let pos = pos_tensor(&*be, p);
                    launch(&*be, kernel, Op::AttnDecode, &[qi, ki, vi], &[out.clone()], &attrs, p, dev.then_some(&pos));
                    get(&*be, &out)
                };
                assert!(run(false) == run(true), "{kernel} n={n} nkv={nkv} d={d} p={p}: device position differs from host");
            }
        }
    }
}

#[test]
fn splitkv_decode_matches_the_simt_kernel_at_every_position() {
    // the max-grid launch (blocks past the active split count exit at once) must still be the same attention
    let Some(be) = device() else { return };
    let mut rng = SplitMix64::new(13);
    let (n, nkv, d, cap) = (16usize, 2usize, 256usize, 4096 + 8);
    let q = rand_bf16(&mut rng, &[1, n, d], 2.0);
    let kc = rand_bf16(&mut rng, &[nkv, cap, d], 1.0);
    let vc = rand_bf16(&mut rng, &[nkv, cap, d], 1.0);
    let attrs = OpAttrs { scale: 1.0 / (d as f32).sqrt(), n_heads: n, n_kv_heads: nkv, head_dim: d, kv_cap: cap, ..OpAttrs::default() };
    for &p in &POSITIONS {
        let run = |kernel: &str| {
            let (qi, ki, vi) = (put(&*be, &q), put(&*be, &kc), put(&*be, &vc));
            let out = put(&*be, &HostTensor::zeros(DType::Bf16, &[1, n, d]));
            launch(&*be, kernel, Op::AttnDecode, &[qi, ki, vi], &[out.clone()], &attrs, p, None);
            HostTensor::new(DType::Bf16, vec![1, n, d], get(&*be, &out)).unwrap().to_f32()
        };
        let (a, b) = (run("attn_decode_bf16"), run("attn_decode_bf16_splitkv"));
        let max = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        assert!(max < 2e-2, "p={p}: split-KV differs from the SIMT kernel by {max}");
    }
}
