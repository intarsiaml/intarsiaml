//! The fp8_e4m3 KV-cache attention kernels (`attn_decode_f8kv_splitkv`, `attn_prefill_f8kv_splitkv`,
//! `attn_prefill_f8kv_flash`) against the bf16 kernels they are built from: on an fp8 cache their output must be BITWISE the
//! bf16 kernel's on the same cache upcast to bf16 (every e4m3 value is a bf16 value, the conversion is exact, and the
//! arithmetic after it is the bf16 kernels' code), for decode, short prefill / verify and flash prefill chunks, host and
//! device position, eager and CUDA graph replay, every GQA group / head-dim instantiation. The accuracy cost of an fp8
//! cache is therefore only the rounding of K / V to e4m3 (measured by the accuracy gates, not here).
//! `f8kv_timing` (ignored) times bf16 vs fp8 caches at serving shapes over ten rotating layer caches (cold in L2).
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
            common::skip("attn_f8kv", &format!("no usable CUDA device ({e})"));
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

/// A cache as fp8 (what the f8kv kernels read) and the same values as bf16 (exact upcast, what the bf16 kernels read).
fn cache_pair(vals: &[f32], shape: &[usize]) -> (HostTensor, HostTensor) {
    let f8 = HostTensor::from_f32_as(DType::Fp8E4m3, shape, vals);
    let up = HostTensor::from_f32_as(DType::Bf16, shape, &f8.to_f32());
    (f8, up)
}

fn attrs(n: usize, nkv: usize, d: usize, cap: usize) -> OpAttrs {
    OpAttrs { scale: 1.0 / (d as f32).sqrt(), n_heads: n, n_kv_heads: nkv, head_dim: d, kv_cap: cap, ..OpAttrs::default() }
}

/// One launch; the output buffer is pre-filled with a pattern so that an unwritten element cannot pass as equal.
#[allow(clippy::too_many_arguments)]
fn run(be: &dyn Backend, kernel: &str, op: Op, q: &TensorRef, k: &TensorRef, v: &TensorRef, a: &OpAttrs, t: usize, p: usize, dev_pos: bool) -> Vec<u8> {
    let (n, d) = (a.n_heads, a.head_dim);
    let out = put(be, &HostTensor::new(DType::Bf16, vec![t, n, d], vec![0x5au8; t * n * d * 2]).unwrap());
    let mut at = a.clone();
    at.pos0 = if dev_pos { 0 } else { p };
    let mut call = OpCall::new(op, vec![q.clone(), k.clone(), v.clone()], vec![out.clone()], at);
    if dev_pos {
        call.pos = Some(put(be, &HostTensor::from_i32(&[1], &[p as i32])));
    }
    let kimpl = be.kernels().into_iter().find(|x| x.name == kernel).unwrap_or_else(|| panic!("{kernel} registered"));
    let need = kimpl.scratch_bytes(&ShapeInfo::from_call(&call));
    if need > 0 {
        call.scratch = Some(TensorRef::new(be.alloc(need.max(16), MemKind::Device).unwrap(), DType::U8, &[need]));
    }
    be.launch(kernel, &call, 0).unwrap_or_else(|e| panic!("{kernel} T={t} p={p}: {e}"));
    get(be, &out)
}

fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    a.chunks(2).zip(b.chunks(2)).position(|(x, y)| x != y)
}

/// (op, f8kv kernel, bf16 kernel, device position?) for a call of `t` rows
fn pairs(t: usize) -> Vec<(Op, &'static str, &'static str, bool)> {
    if t == 1 {
        vec![
            (Op::AttnDecode, "attn_decode_f8kv_splitkv", "attn_decode_bf16_splitkv", false),
            (Op::AttnDecode, "attn_decode_f8kv_splitkv", "attn_decode_bf16_splitkv", true),
            (Op::AttnPrefill, "attn_prefill_f8kv_splitkv", "attn_prefill_bf16_splitkv", false),
        ]
    } else if t <= 16 {
        vec![(Op::AttnPrefill, "attn_prefill_f8kv_splitkv", "attn_prefill_bf16_splitkv", false)]
    } else {
        vec![(Op::AttnPrefill, "attn_prefill_f8kv_flash", "attn_prefill_bf16_flash", false)]
    }
}

#[test]
fn f8kv_kernels_are_bitwise_the_bf16_kernels_on_the_upcast_cache() {
    let Some(be) = device() else { return };
    let mut rng = SplitMix64::new(0xf8);
    let mut cases = std::collections::BTreeMap::<&str, usize>::new();
    // (n_heads, n_kv_heads, head_dim): the 35B (16/2/256) and 8B (32/8/128) shapes, then the other instantiations
    let heads = [(16, 2, 256), (32, 8, 128), (4, 4, 64), (6, 2, 32), (12, 4, 256), (10, 2, 128), (8, 8, 128), (16, 2, 64)];
    // (rows, pos0): decode at split / chunk boundaries and long contexts, verify / short chunks, flash prefill chunks
    let tp: &[(usize, usize)] = &[
        (1, 0), (1, 1), (1, 63), (1, 64), (1, 65), (1, 1033), (1, 4095), (1, 16025), (2, 700), (4, 33), (16, 5),
        (16, 16000), (7, 4624), (17, 0), (33, 129), (64, 64), (65, 3000), (130, 77), (300, 0), (23, 4608), (256, 15872),
    ];
    for &(n, nkv, d) in &heads {
        for &(t, p) in tp {
            if d == 32 && t > 16 {
                continue; // no flash kernel for head_dim 32 (the SIMT prefill kernel serves it; bf16 cache only)
            }
            if n * d * (t + p) > 80_000_000 {
                continue; // keep the host-side random fill affordable
            }
            let cap = p + t + 3;
            let a = attrs(n, nkv, d, cap);
            let qs = if (t + p) % 2 == 0 { 1.0 } else { 4.0 }; // peaky / flat softmax
            let q = put(&*be, &HostTensor::from_f32_as(DType::Bf16, &[t, n, d], &rng.normal_vec(t * n * d, qs)));
            let (k8, kb) = cache_pair(&rng.normal_vec(nkv * cap * d, 1.0), &[nkv, cap, d]);
            let (v8, vb) = cache_pair(&rng.normal_vec(nkv * cap * d, 1.0), &[nkv, cap, d]);
            let (k8, kb, v8, vb) = (put(&*be, &k8), put(&*be, &kb), put(&*be, &v8), put(&*be, &vb));
            for (op, f8, bf, dev) in pairs(t) {
                let want = run(&*be, bf, op, &q, &kb, &vb, &a, t, p, false);
                let got = run(&*be, f8, op, &q, &k8, &v8, &a, t, p, dev);
                assert!(
                    got == want,
                    "{f8} (device pos {dev}) N={n} Nkv={nkv} D={d} T={t} pos0={p}: differs from {bf} on the upcast cache at {:?}",
                    first_diff(&got, &want)
                );
                *cases.entry(f8).or_default() += 1;
            }
        }
    }
    println!("bitwise cases: {cases:?}");
    for name in ["attn_decode_f8kv_splitkv", "attn_prefill_f8kv_splitkv", "attn_prefill_f8kv_flash"] {
        assert!(cases.get(name).copied().unwrap_or(0) >= 20, "{name}: too few cases");
    }
}

/// Small shapes with NaN in the cache rows past the visible keys (never read: the split-KV kernel zeroes rows it does not
/// load, the flash tiles zero-fill them) and a visible NaN in V (both kernels give the same NaN pattern). Also the race
/// check's target for the fp8 flash tile stores (scripts/cuda-checks/race-targets.txt).
#[test]
fn f8kv_matches_with_nan_past_the_end() {
    let Some(be) = device() else { return };
    let mut rng = SplitMix64::new(77);
    let (n, nkv, d) = (16usize, 2usize, 256usize);
    for &(t, p) in &[(1usize, 200usize), (4, 1000), (40, 100), (70, 0)] {
        let cap = p + t + 64;
        let a = attrs(n, nkv, d, cap);
        for case in 0..2 {
            let mut kf = rng.normal_vec(nkv * cap * d, 1.0);
            let mut vf = rng.normal_vec(nkv * cap * d, 1.0);
            for h in 0..nkv {
                for j in p + t..cap {
                    for e in 0..d {
                        kf[(h * cap + j) * d + e] = f32::NAN;
                        vf[(h * cap + j) * d + e] = f32::NAN;
                    }
                }
            }
            if case == 1 {
                vf[(p / 2) * d + 7] = f32::NAN;
            }
            let q = put(&*be, &HostTensor::from_f32_as(DType::Bf16, &[t, n, d], &rng.normal_vec(t * n * d, 1.0)));
            let (k8, kb) = cache_pair(&kf, &[nkv, cap, d]);
            let (v8, vb) = cache_pair(&vf, &[nkv, cap, d]);
            let (k8, kb, v8, vb) = (put(&*be, &k8), put(&*be, &kb), put(&*be, &v8), put(&*be, &vb));
            for (op, f8, bf, dev) in pairs(t) {
                let want = run(&*be, bf, op, &q, &kb, &vb, &a, t, p, false);
                let got = run(&*be, f8, op, &q, &k8, &v8, &a, t, p, dev);
                assert!(got == want, "{f8} case {case} T={t} p={p}: differs at {:?}", first_diff(&got, &want));
            }
        }
    }
}

/// A CUDA graph capturing the fp8 decode kernel, replayed at many positions, equals the bf16 kernel launched eagerly on
/// the upcast cache at each.
#[test]
fn f8kv_graph_replay_is_bitwise_the_eager_bf16_kernel() {
    let Some(be) = device() else { return };
    assert!(be.supports_graphs());
    let mut rng = SplitMix64::new(21);
    for &(n, nkv, d) in &[(16usize, 2usize, 256usize), (32, 8, 128)] {
        let cap = 20000;
        let a = attrs(n, nkv, d, cap);
        let q = put(&*be, &HostTensor::from_f32_as(DType::Bf16, &[1, n, d], &rng.normal_vec(n * d, 2.0)));
        let (k8, kb) = cache_pair(&rng.normal_vec(nkv * cap * d, 1.0), &[nkv, cap, d]);
        let (v8, vb) = cache_pair(&rng.normal_vec(nkv * cap * d, 1.0), &[nkv, cap, d]);
        let (k8, kb, v8, vb) = (put(&*be, &k8), put(&*be, &kb), put(&*be, &v8), put(&*be, &vb));
        let out = put(&*be, &HostTensor::zeros(DType::Bf16, &[1, n, d]));
        let pos = put(&*be, &HostTensor::from_i32(&[1], &[0]));
        let mut call = OpCall::new(Op::AttnDecode, vec![q.clone(), k8.clone(), v8.clone()], vec![out.clone()], a.clone());
        call.pos = Some(pos.clone());
        let kimpl = be.kernels().into_iter().find(|x| x.name == "attn_decode_f8kv_splitkv").unwrap();
        let need = kimpl.scratch_bytes(&ShapeInfo::from_call(&call));
        call.scratch = Some(TensorRef::new(be.alloc(need, MemKind::Device).unwrap(), DType::U8, &[need]));
        let graph = be.graph_capture(0, &mut || be.launch("attn_decode_f8kv_splitkv", &call, 0)).unwrap();
        for &p in &[0usize, 1, 63, 64, 65, 1000, 4095, 16025, 19990, 7] {
            be.upload(&pos.buf, 0, &(p as i32).to_le_bytes(), 0).unwrap();
            be.graph_replay(&graph, 0).unwrap();
            let got = get(&*be, &out);
            let want = run(&*be, "attn_decode_bf16_splitkv", Op::AttnDecode, &q, &kb, &vb, &a, 1, p, false);
            assert!(got == want, "{n}/{nkv}/{d}: graph replay at p={p} differs from the eager bf16 kernel");
        }
    }
}

/// Time of bf16 vs fp8 caches at decode / prefill-chunk shapes, ten rotating layer caches per context (cold in L2, as a
/// decode step reads them). `cargo test --release -p intarsia-backend-cuda --test attn_f8kv -- --ignored --nocapture
/// f8kv_timing`. GB/s counts the cache bytes of the visible keys once (half for fp8).
#[test]
#[ignore]
fn f8kv_timing() {
    use std::time::Instant;
    let Some(be) = device() else { return };
    let mut rng = SplitMix64::new(3);
    const LAYERS: usize = 10;
    // (label, n_heads, n_kv_heads, head_dim, rows, kv_len)
    let shapes: &[(&str, usize, usize, usize, usize, usize)] = &[
        ("35B decode ", 16, 2, 256, 1, 1034),
        ("35B decode ", 16, 2, 256, 1, 4096),
        ("35B decode ", 16, 2, 256, 1, 16026),
        ("35B decode ", 16, 2, 256, 1, 32768),
        ("35B chunk  ", 16, 2, 256, 2048, 2048),
        ("35B chunk  ", 16, 2, 256, 2048, 16384),
        ("8B decode  ", 32, 8, 128, 1, 16384),
    ];
    for &(label, n, nkv, d, t, kv) in shapes {
        let p = kv - t;
        let cap = kv + 8;
        let a = attrs(n, nkv, d, cap);
        let (k8h, kbh) = cache_pair(&rng.normal_vec(nkv * cap * d, 1.0), &[nkv, cap, d]);
        let (v8h, vbh) = cache_pair(&rng.normal_vec(nkv * cap * d, 1.0), &[nkv, cap, d]);
        let q = put(&*be, &HostTensor::from_f32_as(DType::Bf16, &[t, n, d], &rng.normal_vec(t * n * d, 1.0)));
        let out = put(&*be, &HostTensor::zeros(DType::Bf16, &[t, n, d]));
        let mut line = format!("{label} N={n} Nkv={nkv} D={d} T={t:4} kv={kv:6}:");
        for (kname, kh, vh, bytes) in [("bf16", &kbh, &vbh, 2usize), ("f8kv", &k8h, &v8h, 1)] {
            let layers: Vec<(TensorRef, TensorRef)> = (0..LAYERS).map(|_| (put(&*be, kh), put(&*be, vh))).collect();
            let (op, name) = match (t, kname) {
                (1, "bf16") => (Op::AttnDecode, "attn_decode_bf16_splitkv"),
                (1, _) => (Op::AttnDecode, "attn_decode_f8kv_splitkv"),
                (_, "bf16") => (Op::AttnPrefill, "attn_prefill_bf16_flash"),
                _ => (Op::AttnPrefill, "attn_prefill_f8kv_flash"),
            };
            let mut at = a.clone();
            at.pos0 = p;
            let kimpl = be.kernels().into_iter().find(|x| x.name == name).unwrap();
            let calls: Vec<OpCall> = layers
                .iter()
                .map(|(k, v)| {
                    let mut c = OpCall::new(op, vec![q.clone(), k.clone(), v.clone()], vec![out.clone()], at.clone());
                    let need = kimpl.scratch_bytes(&ShapeInfo::from_call(&c));
                    if need > 0 {
                        c.scratch = Some(TensorRef::new(be.alloc(need, MemKind::Device).unwrap(), DType::U8, &[need]));
                    }
                    c
                })
                .collect();
            for c in &calls {
                be.launch(name, c, 0).unwrap();
            }
            be.synchronize(0).unwrap();
            let reps = if t == 1 { 30 } else { 3 };
            let t0 = Instant::now();
            for _ in 0..reps {
                for c in &calls {
                    be.launch(name, c, 0).unwrap();
                }
            }
            be.synchronize(0).unwrap();
            let us = t0.elapsed().as_secs_f64() * 1e6 / (reps * LAYERS) as f64;
            let gbs = (2 * nkv * kv * d * bytes) as f64 / us / 1e3;
            line += &format!("  {kname} {us:8.1} us {gbs:6.1} GB/s");
        }
        println!("{line}");
    }
}
