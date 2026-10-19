//! The pipelined split-KV attention kernels (`attn_decode_bf16_splitkv_pipe`, `attn_prefill_bf16_splitkv_pipe`) against
//! the split-KV kernels they replace: every output bit must be the same (same partition, same fp32 operations in the same
//! order; only the loads move), for decode and short prefill / verify chunks, host and device position, eager and CUDA
//! graph replay, every GQA group / head-dim instantiation, at short and long contexts. Run once per ring depth:
//! `INTARSIA_SPLITKV_STAGES=2|3|4` (the default when unset). `splitkv_timing` (ignored) times both at serving shapes over
//! ten rotating layer caches (as decode reads them, cold in L2).
//! Needs a GPU: without one the tests print SKIPPED (and fail under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_cuda)]

mod common;

use intarsia_core::rng::SplitMix64;
use intarsia_core::{Backend, DType, HostTensor, MemKind, Op, OpAttrs, OpCall, ShapeInfo, TensorRef};
use std::sync::Arc;

fn device() -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_cuda::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            common::skip("splitkv_pipe", &format!("no usable CUDA device ({e})"));
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

fn rand_bf16(rng: &mut SplitMix64, shape: &[usize], scale: f32) -> HostTensor {
    HostTensor::from_f32_as(DType::Bf16, shape, &rng.normal_vec(shape.iter().product(), scale))
}

fn attrs(n: usize, nkv: usize, d: usize, cap: usize) -> OpAttrs {
    OpAttrs { scale: 1.0 / (d as f32).sqrt(), n_heads: n, n_kv_heads: nkv, head_dim: d, kv_cap: cap, ..OpAttrs::default() }
}

fn scratch_for(be: &dyn Backend, kernel: &str, call: &OpCall) -> Option<TensorRef> {
    let k = be.kernels().into_iter().find(|k| k.name == kernel).unwrap_or_else(|| panic!("{kernel} registered"));
    let need = k.scratch_bytes(&ShapeInfo::from_call(call));
    (need > 0).then(|| TensorRef::new(be.alloc(need.max(16), MemKind::Device).unwrap(), DType::U8, &[need]))
}

/// One launch; the output buffer is pre-filled with a pattern so that an unwritten element cannot pass as equal.
#[allow(clippy::too_many_arguments)]
fn run(be: &dyn Backend, kernel: &str, op: Op, q: &TensorRef, k: &TensorRef, v: &TensorRef, a: &OpAttrs, t: usize, p: usize, dev_pos: bool) -> Vec<u8> {
    let (n, d) = (a.n_heads, a.head_dim);
    let fill = HostTensor::new(DType::Bf16, vec![t, n, d], vec![0x5au8; t * n * d * 2]).unwrap();
    let out = put(be, &fill);
    let mut at = a.clone();
    at.pos0 = if dev_pos { 0 } else { p };
    let mut call = OpCall::new(op, vec![q.clone(), k.clone(), v.clone()], vec![out.clone()], at);
    if dev_pos {
        call.pos = Some(put(be, &HostTensor::from_i32(&[1], &[p as i32])));
    }
    call.scratch = scratch_for(be, kernel, &call);
    be.launch(kernel, &call, 0).unwrap_or_else(|e| panic!("{kernel} T={t} p={p}: {e}"));
    get(be, &out)
}

fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    a.chunks(2).zip(b.chunks(2)).position(|(x, y)| x != y)
}

#[test]
fn splitkv_pipe_is_bitwise_the_splitkv_kernels() {
    let Some(be) = device() else { return };
    let mut rng = SplitMix64::new(0x5eed);
    let mut cases = 0usize;
    // (n_heads, n_kv_heads, head_dim): the 35B (16/2/256) and 8B (32/8/128) shapes, then every (GP, TPK) instantiation:
    // groups 1, 2, 3, 4, 5, 8 at head dims 32, 64, 128, 256
    let heads = [
        (16, 2, 256), (32, 8, 128), (4, 4, 64), (6, 2, 32), (12, 4, 256), (10, 2, 128), (8, 8, 128), (6, 2, 128),
        (16, 2, 64), (4, 2, 32), (5, 5, 256), (40, 8, 128),
    ];
    // (rows, pos0): decode at split / chunk boundaries and long contexts, verify and short prefill chunks
    let tp: &[(usize, usize)] = &[
        (1, 0), (1, 1), (1, 15), (1, 63), (1, 64), (1, 65), (1, 127), (1, 1033), (1, 4095), (1, 8191), (1, 16025),
        (1, 16383), (2, 700), (3, 4093), (4, 33), (5, 1), (8, 1999), (16, 5), (16, 16000), (7, 4624), (13, 63),
    ];
    for &(n, nkv, d) in &heads {
        for &(t, p) in tp {
            if n * d * (t + p) > 80_000_000 {
                continue; // keep the host-side random fill affordable
            }
            let cap = p + t + 3;
            let a = attrs(n, nkv, d, cap);
            // q scale 4 makes the softmax peaky (stress the max / rescale path), 1 keeps it flat
            let qs = if (t + p) % 2 == 0 { 1.0 } else { 4.0 };
            let q = put(&*be, &rand_bf16(&mut rng, &[t, n, d], qs));
            let k = put(&*be, &rand_bf16(&mut rng, &[nkv, cap, d], 1.0));
            let v = put(&*be, &rand_bf16(&mut rng, &[nkv, cap, d], 1.0));
            let mut pairs = vec![(Op::AttnPrefill, "attn_prefill_bf16_splitkv", "attn_prefill_bf16_splitkv_pipe", false)];
            if t == 1 {
                pairs.push((Op::AttnDecode, "attn_decode_bf16_splitkv", "attn_decode_bf16_splitkv_pipe", false));
                pairs.push((Op::AttnDecode, "attn_decode_bf16_splitkv", "attn_decode_bf16_splitkv_pipe", true));
            }
            for (op, base, pipe, dev) in pairs {
                let want = run(&*be, base, op, &q, &k, &v, &a, t, p, false);
                let got = run(&*be, pipe, op, &q, &k, &v, &a, t, p, dev);
                assert!(
                    got == want,
                    "{pipe} (device pos {dev}) N={n} Nkv={nkv} D={d} T={t} pos0={p}: differs from {base} at element {:?}",
                    first_diff(&got, &want)
                );
                cases += 1;
            }
        }
    }
    println!("bitwise cases: {cases} (INTARSIA_SPLITKV_STAGES={:?})", std::env::var("INTARSIA_SPLITKV_STAGES").ok());
    assert!(cases > 400, "too few cases ran: {cases}");
}

/// Non-finite and extreme inputs: an all -inf / huge-score row, and keys that are zero past the visible end (the base
/// kernel zeroes the rows it does not load; the ring zero-fills them). NaN in the cache must propagate the same way too.
#[test]
fn splitkv_pipe_matches_on_extreme_values() {
    let Some(be) = device() else { return };
    let mut rng = SplitMix64::new(77);
    let (n, nkv, d) = (16usize, 2usize, 256usize);
    for &(t, p) in &[(1usize, 200usize), (1, 4100), (4, 1000)] {
        let cap = p + t + 64; // rows past the end hold values the kernels must not read
        let a = attrs(n, nkv, d, cap);
        for case in 0..3 {
            let mut kf = rng.normal_vec(nkv * cap * d, 1.0);
            let mut vf = rng.normal_vec(nkv * cap * d, 1.0);
            let qs = if case == 1 { 60.0 } else { 1.0 }; // scores far beyond exp2's range: the max path decides
            for h in 0..nkv {
                for j in p + t..cap {
                    for e in 0..d {
                        let i = (h * cap + j) * d + e;
                        kf[i] = f32::NAN; // never visible
                        vf[i] = f32::INFINITY;
                    }
                }
            }
            if case == 2 {
                vf[(cap / 3) * d + 7] = f32::NAN; // a visible NaN: both must give the same NaN pattern
            }
            let q = put(&*be, &rand_bf16(&mut rng, &[t, n, d], qs));
            let k = put(&*be, &HostTensor::from_f32_as(DType::Bf16, &[nkv, cap, d], &kf));
            let v = put(&*be, &HostTensor::from_f32_as(DType::Bf16, &[nkv, cap, d], &vf));
            let want = run(&*be, "attn_prefill_bf16_splitkv", Op::AttnPrefill, &q, &k, &v, &a, t, p, false);
            let got = run(&*be, "attn_prefill_bf16_splitkv_pipe", Op::AttnPrefill, &q, &k, &v, &a, t, p, false);
            assert!(got == want, "case {case} T={t} p={p}: differs at {:?}", first_diff(&got, &want));
        }
    }
}

/// A CUDA graph capturing the pipelined decode kernel (first launch inside the capture: the dynamic shared-memory opt-in
/// must work there) replayed at many positions equals the base kernel launched eagerly at each.
#[test]
fn splitkv_pipe_graph_replay_is_bitwise_the_eager_base() {
    let Some(be) = device() else { return };
    assert!(be.supports_graphs());
    let mut rng = SplitMix64::new(21);
    for &(n, nkv, d) in &[(16usize, 2usize, 256usize), (32, 8, 128)] {
        let cap = 20000;
        let a = attrs(n, nkv, d, cap);
        let q = put(&*be, &rand_bf16(&mut rng, &[1, n, d], 2.0));
        let k = put(&*be, &rand_bf16(&mut rng, &[nkv, cap, d], 1.0));
        let v = put(&*be, &rand_bf16(&mut rng, &[nkv, cap, d], 1.0));
        let out = put(&*be, &HostTensor::zeros(DType::Bf16, &[1, n, d]));
        let pos = put(&*be, &HostTensor::from_i32(&[1], &[0]));
        let mut call = OpCall::new(Op::AttnDecode, vec![q.clone(), k.clone(), v.clone()], vec![out.clone()], a.clone());
        call.pos = Some(pos.clone());
        call.scratch = scratch_for(&*be, "attn_decode_bf16_splitkv_pipe", &call);
        let graph = be.graph_capture(0, &mut || be.launch("attn_decode_bf16_splitkv_pipe", &call, 0)).unwrap();
        for &p in &[0usize, 1, 63, 64, 65, 1000, 4095, 16025, 19990, 7] {
            be.upload(&pos.buf, 0, &(p as i32).to_le_bytes(), 0).unwrap();
            be.graph_replay(&graph, 0).unwrap();
            let got = get(&*be, &out);
            let want = run(&*be, "attn_decode_bf16_splitkv", Op::AttnDecode, &q, &k, &v, &a, 1, p, false);
            assert!(got == want, "{n}/{nkv}/{d}: graph replay at p={p} differs from the eager base kernel");
        }
    }
}

/// Time of the split-KV kernels and their pipelined twins at decode shapes, ten rotating layer caches per context (as a
/// decode step reads them: each layer's cache is cold in L2). `cargo test --release -p intarsia-backend-cuda --test
/// splitkv_pipe -- --ignored --nocapture splitkv_timing`. KV GB/s counts K + V bytes of the visible keys once.
#[test]
#[ignore]
fn splitkv_timing() {
    use std::time::Instant;
    let Some(be) = device() else { return };
    let mut rng = SplitMix64::new(3);
    const LAYERS: usize = 10;
    // (label, n_heads, n_kv_heads, head_dim, rows, kv_len)
    let shapes: &[(&str, usize, usize, usize, usize, usize)] = &[
        ("35B decode", 16, 2, 256, 1, 1034),
        ("35B decode", 16, 2, 256, 1, 4096),
        ("35B decode", 16, 2, 256, 1, 16026),
        ("35B decode", 16, 2, 256, 1, 32768),
        ("35B verify", 16, 2, 256, 4, 16026),
        ("8B decode ", 32, 8, 128, 1, 1024),
        ("8B decode ", 32, 8, 128, 1, 4096),
        ("8B decode ", 32, 8, 128, 1, 16384),
    ];
    for &(label, n, nkv, d, t, kv) in shapes {
        let p = kv - t;
        let cap = kv + 8;
        let a = attrs(n, nkv, d, cap);
        // one random cache uploaded into ten buffers (the time does not depend on the values)
        let kh = rand_bf16(&mut rng, &[nkv, cap, d], 1.0);
        let vh = rand_bf16(&mut rng, &[nkv, cap, d], 1.0);
        let q = put(&*be, &rand_bf16(&mut rng, &[t, n, d], 1.0));
        let layers: Vec<(TensorRef, TensorRef)> = (0..LAYERS).map(|_| (put(&*be, &kh), put(&*be, &vh))).collect();
        let out = put(&*be, &HostTensor::zeros(DType::Bf16, &[t, n, d]));
        let mut line = format!("{label} N={n} Nkv={nkv} D={d} T={t} kv={kv:6}:");
        for kernel in ["splitkv", "splitkv_pipe"] {
            let (op, name) = if t == 1 {
                (Op::AttnDecode, format!("attn_decode_bf16_{kernel}"))
            } else {
                (Op::AttnPrefill, format!("attn_prefill_bf16_{kernel}"))
            };
            let mut at = a.clone();
            at.pos0 = p;
            let calls: Vec<OpCall> = layers
                .iter()
                .map(|(k, v)| {
                    let mut c = OpCall::new(op, vec![q.clone(), k.clone(), v.clone()], vec![out.clone()], at.clone());
                    c.scratch = scratch_for(&*be, &name, &c);
                    c
                })
                .collect();
            for c in &calls {
                be.launch(&name, c, 0).unwrap();
            }
            be.synchronize(0).unwrap();
            let reps = 30;
            let t0 = Instant::now();
            for _ in 0..reps {
                for c in &calls {
                    be.launch(&name, c, 0).unwrap();
                }
            }
            be.synchronize(0).unwrap();
            let us = t0.elapsed().as_secs_f64() * 1e6 / (reps * LAYERS) as f64;
            let gbs = (2 * nkv * kv * d * 2) as f64 / us / 1e3;
            line += &format!("  {kernel} {us:7.1} us {gbs:6.1} GB/s");
        }
        println!("{line}");
    }
}
