//! `rope_bf16` rotates all heads of a token in one thread (`rope_multi_kernel`) for calls of 5 or more rows with one position per
//! row; `INTARSIA_CUDA_ROPE_MULTI=0` runs the one-launch-per-group-of-4-heads `rope_kernel` it replaced. The outputs must be
//! bit-identical: same per-element expressions, the double-precision pow / sincos of a (token, frequency) is only computed once
//! instead of once per head group. The launcher reads the variable per launch, so the test flips it between calls
//! (`ENV_LOCK`). Needs a GPU (SKIPPED otherwise; fails under CI=1 / INTARSIA_REQUIRE_GPU=1).
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
            common::skip("rope_multi", &format!("no usable CUDA device ({e})"));
            None
        }
    }
}

fn case(rng: &mut SplitMix64, t: usize, n: usize, d: usize, rdim: usize, pos0: usize, theta: f32) -> HostCall {
    let x = HostTensor::from_f32_as(DType::Bf16, &[t, n, d], &rng.normal_vec(t * n * d, 2.0));
    HostCall {
        op: Op::Rope,
        inputs: vec![x],
        outputs: vec![OutSpec::new(DType::Bf16, &[t, n, d])],
        attrs: OpAttrs { pos0, theta, head_dim: d, rotary_dim: rdim, ..OpAttrs::default() },
    }
}

/// The tests of this binary flip a process-wide variable: one at a time, also under a parallel test runner.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Every head layout of the registered models (Qwen3 8B 32/8 x 128, Qwen3.6-35B 16/2 x 256 with a quarter rotary dim, Gemma-style
/// 16 x 256, a head count that is not a multiple of the 4-head group, a partial-rotary 64), the row counts around the 5-row switch
/// and a 4096-token chunk, positions from 0 to a long prefix; theta 1e4 / 1e6 / 1e7.
#[test]
fn rope_multi_is_bitwise_the_group_kernel() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(0x5eed);
    let layouts = [(32usize, 128usize, 0usize), (8, 128, 0), (16, 256, 64), (2, 256, 64), (6, 128, 0), (3, 64, 0), (5, 64, 32), (1, 128, 0)];
    let mut n = 0;
    for &(heads, d, rdim) in &layouts {
        for &t in &[1usize, 4, 5, 6, 33, 64, 257, 1000, 4096] {
            for &(pos0, theta) in &[(0usize, 1.0e4f32), (37, 1.0e6), (16001, 1.0e7)] {
                let hc = case(&mut rng, t, heads, d, rdim, pos0, theta);
                std::env::set_var("INTARSIA_CUDA_ROPE_MULTI", "0");
                let a = run_host(&*cuda, "rope_bf16", &hc).unwrap();
                std::env::remove_var("INTARSIA_CUDA_ROPE_MULTI");
                let b = run_host(&*cuda, "rope_bf16", &hc).unwrap();
                let first = a[0].data.iter().zip(&b[0].data).position(|(u, v)| u != v);
                assert!(
                    first.is_none(),
                    "T={t} heads={heads} D={d} rot={rdim} pos0={pos0} theta={theta}: the multi-head kernel differs from the group kernel at byte {first:?}"
                );
                n += 1;
            }
        }
    }
    assert!(n >= 200, "cases ran: {n}");
}
