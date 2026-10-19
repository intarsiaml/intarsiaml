//! Round 9 (docs/results/2026-10-19-hip-round9.md): split-KV attention stage 1 at 8 waves per SIMD (kernels/hip/
//! attention_splitkv.hip `splitkv_kernel_o8`, the same body under `amdgpu_waves_per_eu(8)`, opt-in with INTARSIA_HIP_SPLITKV_OCC=1)
//! must give every bit of the 7-wave kernel (the default; read at every launch). Decode and the verify / short-prefill path
//! (2..16 rows) on GQA groups 1..8 and head dims 32..256, key counts from one key to past the 64-split cap, peaky and flat
//! softmax, random keys past the causal edge. Also pins the selection (`splitkv_occ_selected`): off by default, `1` on, `0` off.
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
fn splitkv_8_waves_is_bitwise_the_7_wave_kernel() {
    std::env::remove_var("INTARSIA_HIP_SPLITKV_OCC");
    assert!(!intarsia_backend_hip::splitkv_occ_selected(), "the 7-wave stage 1 must be the default");
    std::env::set_var("INTARSIA_HIP_SPLITKV_OCC", "0");
    assert!(!intarsia_backend_hip::splitkv_occ_selected(), "INTARSIA_HIP_SPLITKV_OCC=0 must select the 7-wave kernel");
    std::env::set_var("INTARSIA_HIP_SPLITKV_OCC", "1");
    assert!(intarsia_backend_hip::splitkv_occ_selected(), "INTARSIA_HIP_SPLITKV_OCC=1 must select the 8-wave kernel");
    std::env::remove_var("INTARSIA_HIP_SPLITKV_OCC");
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(909);
    // (n_heads, n_kv_heads, head_dim): the 35B (16 / 2 / 256), the 8B (32 / 8 / 128), groups 1, 3, 5
    let heads = [(16usize, 2usize, 256usize), (32, 8, 128), (4, 4, 64), (6, 2, 32), (12, 4, 256), (10, 2, 128)];
    // (rows, pos0)
    let short = [(1usize, 0usize), (1, 1), (1, 63), (1, 64), (1, 700), (2, 700), (4, 33), (8, 1999), (16, 5)];
    let long = [(1usize, 4095usize), (1, 4408), (1, 9000), (1, 17631), (1, 20000), (16, 9000)];
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
                std::env::remove_var("INTARSIA_HIP_SPLITKV_OCC");
                let old = run_host(&*hip, name, &hc).unwrap_or_else(|e| panic!("{name} N={n} Nkv={nkv} D={d} T={t} pos0={p}: {e}"));
                std::env::set_var("INTARSIA_HIP_SPLITKV_OCC", "1");
                let new = run_host(&*hip, name, &hc).unwrap();
                std::env::remove_var("INTARSIA_HIP_SPLITKV_OCC");
                assert!(
                    old[0].data == new[0].data,
                    "{name} N={n} Nkv={nkv} D={d} T={t} pos0={p} qscale={qscale}: the 8-wave stage 1 changed the output"
                );
                assert!(new[0].to_f32().iter().any(|v| *v != 0.0), "{name} T={t} pos0={p}: all zero");
                cases += 1;
            }
        }
    }
    println!("splitkv 8 waves: {cases} cases bitwise");
}
