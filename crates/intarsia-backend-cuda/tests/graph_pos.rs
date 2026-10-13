//! Device-side position and CUDA graph capture/replay on a real GPU.
//!
//! * rope / kv_append / attn_decode (SIMT and split-KV) read their position from an `i32 [1]` tensor when they
//!   run; the result must be *bit-identical* to launching with the same value as the host `pos0` (admission's
//!   `run_host` only exercises the host path, so this is where the device path is checked).
//! * a graph capturing kv_append + split-KV attention is replayed at many positions by rewriting the position
//!   tensor between replays, and each replay must equal the eager launch at that position.
//! * a failed capture leaves the stream usable.
//!
//! Needs a GPU: without one the tests print SKIPPED (and fail under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_cuda)]

mod common;

use intarsia_core::rng::SplitMix64;
use intarsia_core::{Backend, DType, Error, HostTensor, MemKind, Op, OpAttrs, OpCall, ShapeInfo, TensorRef};
use std::sync::Arc;

fn device() -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_cuda::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            common::skip("graph_pos", &format!("no usable CUDA device ({e})"));
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

#[test]
fn graph_replay_follows_the_device_position() {
    let Some(be) = device() else { return };
    assert!(be.supports_graphs());
    let mut rng = SplitMix64::new(21);
    let (n, nkv, d, cap) = (16usize, 2usize, 256usize, 4096 + 8);
    let q = put(&*be, &rand_bf16(&mut rng, &[1, n, d], 2.0));
    let knew = put(&*be, &rand_bf16(&mut rng, &[1, nkv, d], 1.0));
    let vnew = put(&*be, &rand_bf16(&mut rng, &[1, nkv, d], 1.0));
    let kc = put(&*be, &rand_bf16(&mut rng, &[nkv, cap, d], 1.0));
    let vc = put(&*be, &rand_bf16(&mut rng, &[nkv, cap, d], 1.0));
    let out = put(&*be, &HostTensor::zeros(DType::Bf16, &[1, n, d]));
    let out_ref = put(&*be, &HostTensor::zeros(DType::Bf16, &[1, n, d]));
    let pos = pos_tensor(&*be, 0);
    let append = OpAttrs { kv_cap: cap, n_kv_heads: nkv, head_dim: d, ..OpAttrs::default() };
    let attn = OpAttrs { scale: 1.0 / (d as f32).sqrt(), n_heads: n, n_kv_heads: nkv, head_dim: d, kv_cap: cap, ..OpAttrs::default() };
    let step = |p: usize, dev: Option<&TensorRef>, o: &TensorRef| {
        launch(&*be, "kv_append_bf16", Op::KvAppend, &[knew.clone()], &[kc.clone()], &append, p, dev);
        launch(&*be, "kv_append_bf16", Op::KvAppend, &[vnew.clone()], &[vc.clone()], &append, p, dev);
        launch(&*be, "attn_decode_bf16_splitkv", Op::AttnDecode, &[q.clone(), kc.clone(), vc.clone()], &[o.clone()], &attn, p, dev);
    };
    // the scratch of each launch is allocated per call above; a captured graph keeps its address alive only
    // while the buffer lives, so give the capture one that outlives it
    let k = be.kernels().into_iter().find(|k| k.name == "attn_decode_bf16_splitkv").unwrap();
    let probe = OpCall::new(Op::AttnDecode, vec![q.clone(), kc.clone(), vc.clone()], vec![out.clone()], attn.clone());
    let need = k.scratch_bytes(&ShapeInfo::from_call(&probe));
    let scratch = TensorRef::new(be.alloc(need, MemKind::Device).unwrap(), DType::U8, &[need]);
    let graph = be
        .graph_capture(0, &mut || {
            let mut a = attn.clone();
            a.pos0 = 0;
            for (new, cache) in [(&knew, &kc), (&vnew, &vc)] {
                let mut c = OpCall::new(Op::KvAppend, vec![new.clone()], vec![cache.clone()], append.clone());
                c.pos = Some(pos.clone());
                be.launch("kv_append_bf16", &c, 0)?;
            }
            let mut c = OpCall::new(Op::AttnDecode, vec![q.clone(), kc.clone(), vc.clone()], vec![out.clone()], a);
            c.pos = Some(pos.clone());
            c.scratch = Some(scratch.clone());
            be.launch("attn_decode_bf16_splitkv", &c, 0)
        })
        .unwrap();
    for &p in &POSITIONS {
        be.upload(&pos.buf, 0, &(p as i32).to_le_bytes(), 0).unwrap();
        be.graph_replay(&graph, 0).unwrap();
        let got = get(&*be, &out);
        step(p, None, &out_ref); // eager, host position, same caches (the append is idempotent)
        assert!(got == get(&*be, &out_ref), "graph replay at p={p} differs from the eager launch");
    }
    // replaying again at an earlier position (position moves backwards after a truncate)
    be.upload(&pos.buf, 0, &(3i32).to_le_bytes(), 0).unwrap();
    be.graph_replay(&graph, 0).unwrap();
    let got = get(&*be, &out);
    step(3, None, &out_ref);
    assert!(got == get(&*be, &out_ref), "graph replay at p=3 after longer positions");
}

#[test]
fn a_failed_capture_leaves_the_stream_usable() {
    let Some(be) = device() else { return };
    let r = be.graph_capture(0, &mut || Err(Error::backend("body failed (test)")));
    assert!(r.is_err());
    // forbidden inside a capture: a synchronous copy -> the capture is invalidated, the error surfaces
    let a = put(&*be, &HostTensor::zeros(DType::Bf16, &[64]));
    let r = be.graph_capture(0, &mut || {
        let mut host = vec![0u8; 128];
        be.download(&a.buf, 0, &mut host, 0)
    });
    assert!(r.is_err(), "a synchronising call inside a capture must fail it");
    // and the stream still works
    let b = put(&*be, &rand_bf16(&mut SplitMix64::new(1), &[64], 1.0));
    be.copy(&a.buf, 0, &b.buf, 0, 128, 0).unwrap();
    assert!(get(&*be, &a) == get(&*be, &b));
    let _ = Op::Rope;
}
