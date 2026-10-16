//! Randomised shape sweep of the cuda attention kernels (SIMT, split-KV, flash) against the cpu
//! reference: GQA group sizes 1..8 (incl. non power of two), head_dim 32/64/128/256, query rows 1..300,
//! cache offsets up to a long context, odd sizes that do not align to any tile. Needs a GPU: without one the
//! tests print SKIPPED (and fail under CI=1 / INTARSIA_REQUIRE_GPU=1). `INTARSIA_CUDA_DEVICE` picks the ordinal.
#![cfg(ik_cuda)]

mod common;

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::admission::judge;
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{Backend, DType, HostTensor, Op, OpAttrs, ShapeInfo};

fn device() -> Option<std::sync::Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_cuda::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            common::skip("attention", &format!("no usable CUDA device ({e})"));
            None
        }
    }
}

fn rand_t(rng: &mut SplitMix64, shape: &[usize], scale: f32) -> HostTensor {
    let n: usize = shape.iter().product();
    HostTensor::from_f32_as(DType::Bf16, shape, &rng.normal_vec(n, scale))
}

fn case(op: Op, rng: &mut SplitMix64, n: usize, nkv: usize, d: usize, t: usize, p: usize, qscale: f32) -> HostCall {
    let cap = p + t + 3;
    HostCall {
        op,
        inputs: vec![rand_t(rng, &[t, n, d], qscale), rand_t(rng, &[nkv, cap, d], 1.0), rand_t(rng, &[nkv, cap, d], 1.0)],
        outputs: vec![OutSpec::new(DType::Bf16, &[t, n, d])],
        attrs: OpAttrs {
            pos0: p,
            scale: 1.0 / (d as f32).sqrt(),
            n_heads: n,
            n_kv_heads: nkv,
            head_dim: d,
            kv_cap: cap,
            ..OpAttrs::default()
        },
    }
}

#[test]
fn attention_kernels_match_reference_on_random_shapes() {
    let Some(cuda) = device() else { return };
    let cpu = CpuBackend::arc();
    let mut rng = SplitMix64::new(77);
    let ckernels = cuda.kernels();
    let mut runs = std::collections::BTreeMap::<String, usize>::new();
    // (n_heads, n_kv_heads, head_dim): groups of 1, 2, 3, 4, 5, 8 (3 and 5 take the GP=4 / GP=8 paths)
    let heads = [(4, 4, 64), (6, 2, 128), (6, 2, 32), (12, 4, 256), (10, 2, 128), (16, 2, 256), (8, 8, 128), (32, 8, 128)];
    // (rows, pos0): decode and verify sizes, tile-unaligned prefill chunks, long caches
    let tp = [
        (1, 0), (1, 1), (1, 63), (1, 64), (1, 4095), (2, 700), (4, 33), (8, 1999), (16, 5),
        (17, 0), (33, 129), (63, 1), (64, 64), (65, 3000), (130, 77), (300, 0),
        // the shapes of the 4631-token chat prompt (chunk 256: 18 full chunks, then 23 rows at pos0 4608), and a
        // non-aligned cache offset with a non-multiple-of-16 row count
        (23, 4608), (7, 4624), (1, 4630),
    ];
    for &(n, nkv, d) in &heads {
        for &(t, p) in &tp {
            if n * d * (t + p) > 6_000_000 && t > 64 {
                continue; // keep the cpu reference affordable
            }
            // q scale 4 makes the softmax peaky (stress the max/rescale path), 1 keeps it flat
            let qscale = if (t + p) % 2 == 0 { 1.0 } else { 4.0 };
            for op in [Op::AttnPrefill, Op::AttnDecode] {
                if op == Op::AttnDecode && t != 1 {
                    continue;
                }
                let hc = case(op, &mut rng, n, nkv, d, t, p, qscale);
                let call_shape = {
                    let mut a = ShapeInfo::default();
                    a.m = t;
                    a.n_heads = n;
                    a.n_kv_heads = nkv;
                    a.head_dim = d;
                    a.k = d;
                    a.kv_len = p + t;
                    a.act = Some(DType::Bf16);
                    a.weight = Some(DType::Bf16);
                    a
                };
                let rname = cpu.kernels().into_iter().find(|k| k.op == op && k.is_reference).unwrap().name;
                let want = run_host(&*cpu, &rname, &hc).unwrap();
                for k in ckernels.iter().filter(|k| k.op == op && k.accepts(&call_shape)) {
                    let got = run_host(&*cuda, &k.name, &hc)
                        .unwrap_or_else(|e| panic!("{} N={n} Nkv={nkv} D={d} T={t} pos0={p}: {e}", k.name));
                    let (ok, ma, rl) = judge(k, &got, &want);
                    assert!(ok, "{} N={n} Nkv={nkv} D={d} T={t} pos0={p} qscale={qscale}: max_abs {ma:.3e} rel_l2 {rl:.3e}", k.name);
                    // deterministic kernels: a second launch is bit-identical
                    let again = run_host(&*cuda, &k.name, &hc).unwrap();
                    assert_eq!(got[0].data, again[0].data, "{} is not repeatable at T={t} pos0={p}", k.name);
                    *runs.entry(k.name.clone()).or_default() += 1;
                }
            }
        }
    }
    println!("cases per kernel: {runs:?}");
    for name in ["attn_prefill_bf16", "attn_decode_bf16", "attn_prefill_bf16_flash", "attn_prefill_bf16_splitkv", "attn_decode_bf16_splitkv"] {
        assert!(runs.get(name).copied().unwrap_or(0) > 0, "{name} ran no cases");
    }
}

/// Throughput of the attention kernels at serving shapes. `cargo test --release -p intarsia-backend-cuda
/// --test attention -- --ignored --nocapture`. Prints achieved KV-read bandwidth (decode) and TFLOPS (prefill);
/// compare with `intarsia doctor` (copy bandwidth) for the roofline.
#[test]
#[ignore]
fn attention_throughput() {
    use intarsia_core::{MemKind, OpCall, TensorRef};
    use std::time::Instant;
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(5);
    let put = |t: &HostTensor| -> TensorRef {
        let b = cuda.alloc(t.nbytes().max(16), MemKind::Device).unwrap();
        cuda.upload(&b, 0, &t.data, 0).unwrap();
        TensorRef::new(b, t.dtype, &t.shape)
    };
    // (name, n_heads, n_kv_heads, head_dim, rows, kv_len)
    let runs: &[(&str, usize, usize, usize, usize, usize)] = &[
        ("decode 8B   ", 32, 8, 128, 1, 1024),
        ("decode 8B   ", 32, 8, 128, 1, 4096),
        ("decode 8B   ", 32, 8, 128, 1, 16384),
        ("decode 35B  ", 16, 2, 256, 1, 4096),
        ("decode 35B  ", 16, 2, 256, 1, 16384),
        ("decode 35B  ", 16, 2, 256, 1, 65536),
        ("verify 8B  4", 32, 8, 128, 4, 4096),
        ("verify 8B 16", 32, 8, 128, 16, 4096),
        ("prefill 8B  ", 32, 8, 128, 256, 256),
        ("prefill 8B  ", 32, 8, 128, 256, 4096),
        ("prefill 8B  ", 32, 8, 128, 256, 16384),
        ("prefill 35B ", 16, 2, 256, 256, 4096),
        ("prefill 35B ", 16, 2, 256, 256, 16384),
        ("prefill 8B 1024", 32, 8, 128, 1024, 4096),
    ];
    for &(name, n, nkv, d, t, kv) in runs {
        let p = kv - t;
        let hc = case(if t == 1 { Op::AttnDecode } else { Op::AttnPrefill }, &mut rng, n, nkv, d, t, p, 1.0);
        let inputs: Vec<TensorRef> = hc.inputs.iter().map(&put).collect();
        let out = put(&HostTensor::zeros(DType::Bf16, &[t, n, d]));
        let mut call = OpCall::new(hc.op, inputs, vec![out], hc.attrs.clone());
        let mut s = ShapeInfo::default();
        s.m = t;
        s.n_heads = n;
        s.n_kv_heads = nkv;
        s.head_dim = d;
        s.k = d;
        s.kv_len = kv;
        s.act = Some(DType::Bf16);
        s.weight = Some(DType::Bf16);
        let mut best: Option<intarsia_core::KernelImpl> = None;
        for k in cuda.kernels().into_iter().filter(|k| k.op == call.op && k.accepts(&s)) {
            if best.as_ref().is_none_or(|b| k.priority > b.priority) {
                best = Some(k);
            }
        }
        let k = best.unwrap();
        let need = k.scratch_bytes(&s);
        if need > 0 {
            call.scratch = Some(TensorRef::new(cuda.alloc(need, MemKind::Device).unwrap(), DType::U8, &[need]));
        }
        for _ in 0..3 {
            cuda.launch(&k.name, &call, 0).unwrap();
        }
        cuda.synchronize(0).unwrap();
        let iters = 100;
        let t0 = Instant::now();
        for _ in 0..iters {
            cuda.launch(&k.name, &call, 0).unwrap();
        }
        cuda.synchronize(0).unwrap();
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        // KV bytes touched once (K + V), and causal flops
        let kv_bytes = (2 * nkv * kv * d * 2) as f64;
        let flops = 4.0 * (n * d) as f64 * (t as f64 * p as f64 + (t * (t + 1)) as f64 / 2.0);
        println!(
            "{name} N={n} Nkv={nkv} D={d} T={t:4} kv={kv:6}: {:9.1} us  {:7.1} GB/s KV  {:6.1} TFLOPS  [{}]",
            us,
            kv_bytes / us / 1e3,
            flops / us / 1e6,
            k.name
        );
    }
}
