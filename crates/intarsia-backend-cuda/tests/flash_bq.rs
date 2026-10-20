//! `attn_prefill_bf16_flash` at head_dim 256 runs 128 query rows per block by default (kernels/cuda/attention_flash.cu);
//! `INTARSIA_CUDA_FLASH_BQ=64` runs the 64-row blocks it replaced. The outputs must be bit-identical for every query count
//! and cache offset (the fully masked key tiles of the lower warps must not change the running max, sum or O). The
//! launcher reads the variable per launch, so this single-test binary flips it between calls. Needs a GPU (SKIPPED
//! otherwise; fails under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_cuda)]

mod common;

use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{Backend, DType, HostTensor, Op, OpAttrs};

fn device() -> Option<std::sync::Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_cuda::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            common::skip("flash_bq", &format!("no usable CUDA device ({e})"));
            None
        }
    }
}

fn rand_t(rng: &mut SplitMix64, shape: &[usize], scale: f32) -> HostTensor {
    let n: usize = shape.iter().product();
    HostTensor::from_f32_as(DType::Bf16, shape, &rng.normal_vec(n, scale))
}

fn case(rng: &mut SplitMix64, n: usize, nkv: usize, t: usize, p: usize, qscale: f32) -> HostCall {
    let d = 256;
    let cap = p + t + 3;
    HostCall {
        op: Op::AttnPrefill,
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
fn flash_d256_bq128_is_bitwise_bq64() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(91);
    let mut n = 0;
    // (rows, pos0): one row, block-edge row counts (63..129, 191, 192), 2048-token chunks with and without a long prefix
    let tp = [
        (1, 0), (17, 0), (63, 0), (64, 0), (65, 3), (127, 0), (128, 0), (129, 31), (191, 0), (192, 64), (300, 1999),
        (2048, 0), (2048, 6000),
    ];
    for &(nh, nkv) in &[(16usize, 2usize), (12, 4), (8, 8)] {
        for &(t, p) in &tp {
            for qscale in [1.0f32, 8.0] {
                let hc = case(&mut rng, nh, nkv, t, p, qscale);
                std::env::set_var("INTARSIA_CUDA_FLASH_BQ", "64");
                let a = run_host(&*cuda, "attn_prefill_bf16_flash", &hc).unwrap();
                std::env::remove_var("INTARSIA_CUDA_FLASH_BQ");
                let b = run_host(&*cuda, "attn_prefill_bf16_flash", &hc).unwrap();
                let first = a[0].data.iter().zip(&b[0].data).position(|(u, v)| u != v);
                assert!(
                    first.is_none(),
                    "heads {nh}/{nkv} T={t} pos0={p} qscale={qscale}: BQ 128 differs from BQ 64 at byte {first:?}"
                );
                n += 1;
            }
        }
    }
    println!("flash_bq: {n} cases bit-identical");
}
