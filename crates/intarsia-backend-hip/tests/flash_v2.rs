//! `attn_prefill_bf16_flash_v2` (kernels/hip/attention_flash.hip, `flash_hpb_kernel`): the WMMA flash prefill kernel with
//! INTARSIA_HIP_FLASH_HPB (1, 2 or 4; default 2) query heads of one kv head per block, so a K / V tile is staged into LDS
//! once for all of them. Each wave runs `flash_kernel`'s instructions for its 16 rows, so:
//!
//! * `flash_v2_is_bitwise_v1`: every output byte equals `attn_prefill_bf16_flash`'s at the 35B's (16 / 2 / 256) and the 8B's
//!   (32 / 8 / 128) shapes, at groups of 1, 2, 3 (HPB falls back to 1), 4 and 8, head dims 64 / 128 / 256, chunks that are
//!   and are not a multiple of the 64-row query tile, from position 0 and from a cached prefix (the second prefill chunk);
//! * the cpu reference within the admission budget, and causal-edge poisoning, come from tests/attention.rs, whose sweeps
//!   require v2 to run.
//!
//! remote.sh r5test runs this file at HPB 1, 2 and 4 (each value is read once per process). Needs a GPU: skips without one.
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

fn case(rng: &mut SplitMix64, n: usize, nkv: usize, d: usize, t: usize, p: usize) -> HostCall {
    let cap = p + t + 5;
    HostCall {
        op: Op::AttnPrefill,
        inputs: vec![rand_t(rng, &[t, n, d], 1.0), rand_t(rng, &[nkv, cap, d], 1.0), rand_t(rng, &[nkv, cap, d], 1.0)],
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
fn flash_v2_is_bitwise_v1() {
    let Some(hip) = device() else { return };
    eprintln!("INTARSIA_HIP_FLASH_HPB={:?}", std::env::var("INTARSIA_HIP_FLASH_HPB").ok());
    let mut rng = SplitMix64::new(61);
    let mut n_cases = 0;
    for (n, nkv, d) in
        [(16usize, 2usize, 256usize), (32, 8, 128), (8, 8, 64), (6, 2, 128), (8, 2, 64), (16, 2, 128), (4, 1, 256)]
    {
        for (t, p) in [(17usize, 0usize), (64, 0), (100, 0), (300, 0), (129, 37), (64, 4096), (257, 1000)] {
            let hc = case(&mut rng, n, nkv, d, t, p);
            let a = run_host(&*hip, "attn_prefill_bf16_flash", &hc).unwrap();
            let b = run_host(&*hip, "attn_prefill_bf16_flash_v2", &hc).unwrap();
            assert!(a[0].data == b[0].data, "flash v2 differs from v1: n={n} nkv={nkv} d={d} t={t} pos0={p}");
            n_cases += 1;
        }
    }
    // a full 35B prefill chunk, from the start and after a 4096-token prefix
    for p in [0usize, 4096] {
        let hc = case(&mut rng, 16, 2, 256, 4096, p);
        let a = run_host(&*hip, "attn_prefill_bf16_flash", &hc).unwrap();
        let b = run_host(&*hip, "attn_prefill_bf16_flash_v2", &hc).unwrap();
        assert!(a[0].data == b[0].data, "flash v2 differs from v1 on a 4096-row 35B chunk at pos0={p}");
        n_cases += 1;
    }
    eprintln!("flash v2 bitwise v1 on {n_cases} cases");
}
