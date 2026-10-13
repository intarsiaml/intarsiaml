//! Randomised shape sweep of the hip attention kernels (SIMT, split-KV, flash) against the cpu
//! reference: GQA group sizes 1..8 (incl. non power of two), head_dim 32/64/128/256, query rows 1..300,
//! cache offsets up to a long context, odd sizes that do not align to any tile. Needs a GPU: skips
//! (passes) without one. `INTARSIA_HIP_DEVICE` picks the ordinal.
#![cfg(ik_hip)]

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::admission::judge;
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{Backend, DType, HostTensor, Op, OpAttrs, ShapeInfo};

fn device() -> Option<std::sync::Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_HIP_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_hip::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping: no usable HIP device ({e})");
            None
        }
    }
}

fn rand_t(rng: &mut SplitMix64, shape: &[usize], scale: f32) -> HostTensor {
    let n: usize = shape.iter().product();
    HostTensor::from_f32_as(DType::Bf16, shape, &rng.normal_vec(n, scale))
}

/// Slots of K/V cache past the causal edge in the poisoned variants: more than one tile of any kernel, so a leak
/// that reads a whole tile past the visible keys still lands on poison.
const POISON_TAIL: usize = 130;

#[derive(Clone, Copy)]
enum Poison {
    /// K = +-1e4 (alternating by slot), V = 1e4 past the causal edge: a leaked key dominates or wrecks the softmax.
    Big,
    /// K = 0 (a leaked key gets the weight of a typical valid one), V = 1e4 past the causal edge.
    Neutral,
}

fn bf16_bytes(v: f32) -> [u8; 2] {
    let t = HostTensor::from_f32_as(DType::Bf16, &[1], &[v]);
    [t.data[0], t.data[1]]
}

/// The same call with a cache of `kv_len + POISON_TAIL` slots per head: slots `< kv_len` (kv_len = pos0 + rows) are
/// copied from `hc`, every slot at or past it is poison. The cpu reference never reads past `kv_len`, so the
/// reference output of `hc` stays the expected one; a kernel that reads a slot it must not does not.
fn poisoned(hc: &HostCall, poison: Poison) -> HostCall {
    let a = &hc.attrs;
    let (nkv, d, p) = (a.n_kv_heads, a.head_dim, a.pos0);
    let t = hc.inputs[0].shape[0];
    let (cap, kv_len) = (a.kv_cap, p + t);
    let cap2 = kv_len + POISON_TAIL;
    let big = bf16_bytes(1.0e4);
    let nbig = bf16_bytes(-1.0e4);
    let zero = [0u8, 0u8];
    let relayout = |src: &HostTensor, is_k: bool| {
        let mut data = Vec::with_capacity(nkv * cap2 * d * 2);
        for h in 0..nkv {
            for slot in 0..cap2 {
                if slot < kv_len {
                    let off = (h * cap + slot) * d * 2;
                    data.extend_from_slice(&src.data[off..off + d * 2]);
                } else {
                    let v = match (is_k, poison) {
                        (true, Poison::Big) => if slot % 2 == 0 { big } else { nbig },
                        (true, Poison::Neutral) => zero,
                        (false, _) => big,
                    };
                    for _ in 0..d {
                        data.extend_from_slice(&v);
                    }
                }
            }
        }
        HostTensor::new(DType::Bf16, vec![nkv, cap2, d], data).unwrap()
    };
    let mut out = hc.clone();
    out.inputs[1] = relayout(&hc.inputs[1], true);
    out.inputs[2] = relayout(&hc.inputs[2], false);
    out.attrs.kv_cap = cap2;
    out
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
    let Some(hip) = device() else { return };
    let cpu = CpuBackend::arc();
    let mut rng = SplitMix64::new(77);
    let ckernels = hip.kernels();
    let mut runs = std::collections::BTreeMap::<String, usize>::new();
    // (n_heads, n_kv_heads, head_dim): groups of 1, 2, 3, 4, 5, 8 (3 and 5 take the GP=4 / GP=8 paths)
    let heads = [(4, 4, 64), (6, 2, 128), (6, 2, 32), (12, 4, 256), (10, 2, 128), (16, 2, 256), (8, 8, 128), (32, 8, 128)];
    // (rows, pos0): decode and verify sizes, tile-unaligned prefill chunks, long caches
    let tp = [
        (1, 0), (1, 1), (1, 63), (1, 64), (1, 4095), (2, 700), (4, 33), (8, 1999), (16, 5),
        (17, 0), (33, 129), (63, 1), (64, 64), (65, 3000), (130, 77), (300, 0),
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
                    let got = run_host(&*hip, &k.name, &hc)
                        .unwrap_or_else(|e| panic!("{} N={n} Nkv={nkv} D={d} T={t} pos0={p}: {e}", k.name));
                    let (ok, ma, rl) = judge(k, &got, &want);
                    assert!(ok, "{} N={n} Nkv={nkv} D={d} T={t} pos0={p} qscale={qscale}: max_abs {ma:.3e} rel_l2 {rl:.3e}", k.name);
                    // deterministic kernels: a second launch is bit-identical
                    let again = run_host(&*hip, &k.name, &hc).unwrap();
                    assert_eq!(got[0].data, again[0].data, "{} is not repeatable at T={t} pos0={p}", k.name);
                    // poisoned tail: every K/V slot at or past the causal edge holds garbage; any read of it fails
                    for pz in [Poison::Big, Poison::Neutral] {
                        let phc = poisoned(&hc, pz);
                        let pgot = run_host(&*hip, &k.name, &phc)
                            .unwrap_or_else(|e| panic!("{} (poisoned) N={n} Nkv={nkv} D={d} T={t} pos0={p}: {e}", k.name));
                        let (ok, ma, rl) = judge(k, &pgot, &want);
                        assert!(ok, "{} leaks past the causal edge: N={n} Nkv={nkv} D={d} T={t} pos0={p}: max_abs {ma:.3e} rel_l2 {rl:.3e}", k.name);
                    }
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

/// Exact causal / chunk-edge checks. K = 0 makes every visible key weigh the same, and V is one-hot in the key index
/// (V[key] = 1 for one key, 0 elsewhere; every slot past the causal edge is K = 0, V = 1 as well), so row r outputs
/// exactly 1/(pos0 + r + 1) if key <= pos0 + r and exactly 0 otherwise. A kernel that drops a key, counts it twice
/// (split / tile boundary) or lets a row see key `pos0 + r + 1` (causal mask) is off by 50-100% in that row,
/// far above the bf16 admission tolerance that hid such errors in the random sweep.
#[test]
fn attention_causal_and_chunk_edges_are_exact() {
    let Some(hip) = device() else { return };
    let cpu = CpuBackend::arc();
    let mut rng = SplitMix64::new(4242);
    let ckernels = hip.kernels();
    let one = bf16_bytes(1.0);
    let mut runs = std::collections::BTreeMap::<String, usize>::new();
    let heads = [(6usize, 2usize, 64usize), (8, 2, 128), (12, 4, 256)];
    // (rows, pos0): q-block starts (64k, 64k+1), key-tile edges (32/64 multiples +-1), split edges
    let mut cases: Vec<(usize, usize)> = Vec::new();
    for t in [1usize, 2, 3, 16, 17, 33, 64, 65, 66, 129] {
        for base in [0usize, 32, 64, 128, 256, 1024] {
            for dp in [-2i64, -1, 0, 1, 2] {
                let p = base as i64 + dp;
                if p >= 0 {
                    cases.push((t, p as usize));
                }
            }
        }
    }
    // kv_len = pos0 + rows on a chunk edge, with the one-hot key on the edge itself
    for t in [1usize, 2, 17] {
        for edge in [64usize, 128, 192, 256, 512, 1024, 4096] {
            for dl in [-1i64, 0, 1] {
                let kv = edge as i64 + dl;
                if kv > t as i64 {
                    cases.push((t, kv as usize - t));
                }
            }
        }
    }
    cases.sort();
    cases.dedup();
    for &(n, nkv, d) in &heads {
        for &(t, p) in &cases {
            if n * d * (t + p) > 3_000_000 && t > 1 {
                continue; // keep the cpu reference affordable
            }
            let kv_len = p + t;
            // one-hot keys worth testing: the last two rows' own keys (causal edge, q-block starts) and every
            // 64-multiple chunk edge +-1 inside the visible range (split boundaries)
            let mut keys: Vec<usize> = vec![kv_len - 1];
            if t >= 2 {
                keys.push(kv_len - 2);
            }
            let mut e = 64;
            while e < kv_len + 1 {
                for b in [e - 1, e, e + 1] {
                    if b < kv_len {
                        keys.push(b);
                    }
                }
                e += 64;
            }
            // thin out long lists: first, last and a stride, so a 4k context does not run ~190 cases per kernel
            keys.sort();
            keys.dedup();
            if keys.len() > 8 {
                let stride = keys.len().div_ceil(8);
                let last = *keys.last().unwrap();
                keys = keys.iter().copied().step_by(stride).chain(std::iter::once(last)).collect();
                keys.dedup();
            }
            for op in [Op::AttnPrefill, Op::AttnDecode] {
                if op == Op::AttnDecode && t != 1 {
                    continue;
                }
                let mut shape = ShapeInfo::default();
                shape.m = t;
                shape.n_heads = n;
                shape.n_kv_heads = nkv;
                shape.head_dim = d;
                shape.k = d;
                shape.kv_len = kv_len;
                shape.act = Some(DType::Bf16);
                shape.weight = Some(DType::Bf16);
                let cap = kv_len + POISON_TAIL;
                let rname = cpu.kernels().into_iter().find(|k| k.op == op && k.is_reference).unwrap().name;
                for &b in &keys {
                    let kd = vec![0u8; nkv * cap * d * 2];
                    let mut vd = vec![0u8; nkv * cap * d * 2];
                    for h in 0..nkv {
                        for slot in 0..cap {
                            if slot == b || slot >= kv_len {
                                for dd in 0..d {
                                    let off = ((h * cap + slot) * d + dd) * 2;
                                    vd[off..off + 2].copy_from_slice(&one);
                                }
                            }
                        }
                    }
                    let hc = HostCall {
                        op,
                        inputs: vec![
                            rand_t(&mut rng, &[t, n, d], 1.0),
                            HostTensor::new(DType::Bf16, vec![nkv, cap, d], kd).unwrap(),
                            HostTensor::new(DType::Bf16, vec![nkv, cap, d], vd).unwrap(),
                        ],
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
                    };
                    // the reference sees the same cache (slots past kv_len are never read by it)
                    let want = run_host(&*cpu, &rname, &hc).unwrap()[0].to_f32();
                    for r in 0..t {
                        let expect = if b <= p + r { 1.0 / (p + r + 1) as f32 } else { 0.0 };
                        let row = &want[r * n * d..(r + 1) * n * d];
                        assert!(
                            row.iter().all(|x| (x - expect).abs() <= 4e-3 * expect.max(1e-9) + 1e-9),
                            "cpu reference row {r} is not the closed form (key {b}, T={t}, pos0={p})"
                        );
                    }
                    for k in ckernels.iter().filter(|k| k.op == op && k.accepts(&shape)) {
                        let got = run_host(&*hip, &k.name, &hc)
                            .unwrap_or_else(|e| panic!("{} N={n} Nkv={nkv} D={d} T={t} pos0={p}: {e}", k.name))[0]
                            .to_f32();
                        let (mut num, mut den) = (0f64, 0f64);
                        for (g, w) in got.iter().zip(&want) {
                            num += ((g - w) as f64).powi(2);
                            den += (*w as f64).powi(2);
                        }
                        let rel = (num / den.max(1e-30)).sqrt();
                        assert!(
                            got.iter().all(|x| x.is_finite()) && rel < 2e-2,
                            "{}: N={n} Nkv={nkv} D={d} T={t} pos0={p} one-hot key {b}: rel_l2 {rel:.3e} (a dropped, doubled or leaked key)",
                            k.name
                        );
                        *runs.entry(k.name.clone()).or_default() += 1;
                    }
                }
            }
        }
    }
    println!("edge cases per kernel: {runs:?}");
    for name in ["attn_prefill_bf16", "attn_decode_bf16", "attn_prefill_bf16_flash", "attn_prefill_bf16_splitkv", "attn_decode_bf16_splitkv"] {
        assert!(runs.get(name).copied().unwrap_or(0) > 0, "{name} ran no edge cases");
    }
}

/// Throughput of the attention kernels at serving shapes. `cargo test --release -p intarsia-backend-hip
/// --test attention -- --ignored --nocapture`. Prints achieved KV-read bandwidth (decode) and TFLOPS (prefill);
/// compare with `intarsia doctor` (copy bandwidth) for the roofline.
#[test]
#[ignore]
fn attention_throughput() {
    use intarsia_core::{MemKind, OpCall, TensorRef};
    use std::time::Instant;
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(5);
    let put = |t: &HostTensor| -> TensorRef {
        let b = hip.alloc(t.nbytes().max(16), MemKind::Device).unwrap();
        hip.upload(&b, 0, &t.data, 0).unwrap();
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
        for k in hip.kernels().into_iter().filter(|k| k.op == call.op && k.accepts(&s)) {
            if best.as_ref().is_none_or(|b| k.priority > b.priority) {
                best = Some(k);
            }
        }
        let k = best.unwrap();
        let need = k.scratch_bytes(&s);
        if need > 0 {
            call.scratch = Some(TensorRef::new(hip.alloc(need, MemKind::Device).unwrap(), DType::U8, &[need]));
        }
        for _ in 0..3 {
            hip.launch(&k.name, &call, 0).unwrap();
        }
        hip.synchronize(0).unwrap();
        let iters = 100;
        let t0 = Instant::now();
        for _ in 0..iters {
            hip.launch(&k.name, &call, 0).unwrap();
        }
        hip.synchronize(0).unwrap();
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
