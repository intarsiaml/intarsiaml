//! Round 8 (docs/results/2026-10-19-hip-round8.md): the split-KV attention's round-8 combine (kernels/hip/
//! attention_splitkv.hip combine_kernel<D, 16>: the partials loaded 16 splits per batch) must give every bit of the
//! pre-round-8 combine (INTARSIA_HIP_SPLITKV_R8=0, one split at a time, read at every launch). Decode (one query row) and
//! the verify / short-prefill path (2..16 rows), on GQA groups 1..8 and head dims 32..256, split counts from 1 to the
//! 64-split cap (key counts at one key, inside a split, at 4k and past 16k), so the last batch is full, partial or alone,
//! with a peaky and a flat softmax and random keys past the causal edge.
//!
//! Its own test binary with a single test: it sets an environment variable, which no other test may race with.
#![cfg(ik_hip)]

use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{Backend, DType, HostTensor, Op, OpAttrs};

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

/// `t` query rows at positions p .. p + t - 1 over a cache of cap = p + t + 70 slots (the 70 past the edge are random too)
fn case(op: Op, rng: &mut SplitMix64, n: usize, nkv: usize, d: usize, t: usize, p: usize, qscale: f32) -> HostCall {
    let cap = p + t + 70;
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
fn splitkv_round8_loops_are_bitwise_the_old_ones() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(808);
    // (n_heads, n_kv_heads, head_dim): the 35B (16 / 2 / 256), the 8B (32 / 8 / 128), groups 1, 3, 5
    let heads = [(16usize, 2usize, 256usize), (8, 1, 256), (32, 8, 128), (4, 4, 64), (6, 2, 32), (12, 4, 256), (10, 2, 128)];
    // (rows, pos0)
    let short = [(1usize, 0usize), (1, 1), (1, 15), (1, 63), (1, 64), (1, 65), (1, 700), (2, 700), (4, 33), (8, 1999), (16, 5)];
    let long = [(1usize, 4095usize), (1, 4096), (1, 9000), (1, 17631), (1, 20000), (16, 9000)];
    let mut cases = 0usize;
    for (hi, &(n, nkv, d)) in heads.iter().enumerate() {
        let tps: Vec<(usize, usize)> = short.iter().copied().chain(if hi < 2 { long.to_vec() } else { vec![] }).collect();
        for (t, p) in tps {
            let qscale = if (t + p) % 2 == 0 { 1.0 } else { 4.0 };
            for (op, name) in [(Op::AttnDecode, "attn_decode_bf16_splitkv"), (Op::AttnPrefill, "attn_prefill_bf16_splitkv")] {
                if op == Op::AttnDecode && t != 1 {
                    continue;
                }
                let hc = case(op, &mut rng, n, nkv, d, t, p, qscale);
                std::env::set_var("INTARSIA_HIP_SPLITKV_R8", "0");
                let old = run_host(&*hip, name, &hc).unwrap_or_else(|e| panic!("{name} N={n} Nkv={nkv} D={d} T={t} pos0={p}: {e}"));
                std::env::remove_var("INTARSIA_HIP_SPLITKV_R8");
                let new = run_host(&*hip, name, &hc).unwrap();
                assert!(
                    old[0].data == new[0].data,
                    "{name} N={n} Nkv={nkv} D={d} T={t} pos0={p} qscale={qscale}: the round-8 loops changed the output"
                );
                assert!(new[0].to_f32().iter().any(|v| *v != 0.0), "{name} T={t} pos0={p}: all zero");
                cases += 1;
            }
        }
    }
    println!("splitkv round 8: {cases} cases bitwise");
}
