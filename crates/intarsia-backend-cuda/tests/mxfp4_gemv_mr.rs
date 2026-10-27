//! `matmul_mxfp4_gemv_mr` and `matmul_mxfp4_gemv_mr_rows` (kernels/cuda/mxfp4.cu: the multi-row W4A16 GEMV for M = 1..4): both are
//! the one-row GEMV (`matmul_mxfp4_gemv`) byte for byte, at every M, for N that is and is not a multiple of the rows per warp (2 and
//! 4), K of one group to the 27B's 17408, bf16 and f32 outputs, and for rows alone against the same rows in a larger call.
//! Needs a GPU: without one the tests print SKIPPED (and fail under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_cuda)]

mod common;

use intarsia_core::backend::Backend;
use intarsia_core::exec::run_host;
use intarsia_core::quant::cases::matmul_mxfp4_w4a16_case;
use intarsia_core::rng::SplitMix64;
use intarsia_core::DType;
use std::sync::Arc;

fn device() -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_cuda::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            common::skip("mxfp4_gemv_mr", &format!("no usable CUDA device ({e})"));
            None
        }
    }
}

#[test]
fn mr_is_bitwise_the_one_row_gemv() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(0x6d91);
    // (n, k): N odd, 1, 2, 3, 5 (tail warps), a ragged lane tail (33 groups), one group, the 27B's k = 5120 / 6144 / 17408
    let shapes = [(1usize, 64usize), (2, 32), (3, 1056), (5, 160), (9, 5120), (135, 256), (1001, 5120), (520, 6144), (2048, 17408), (5120, 5120)];
    for m in 1..=4usize {
        for &(n, k) in &shapes {
            for odt in [DType::Bf16, DType::F32] {
                let hc = matmul_mxfp4_w4a16_case(&mut rng, m, n, k, DType::Bf16, odt);
                let want = run_host(&*cuda, "matmul_mxfp4_gemv", &hc).unwrap().remove(0);
                for name in ["matmul_mxfp4_gemv_mr", "matmul_mxfp4_gemv_mr_rows"] {
                    let got = run_host(&*cuda, name, &hc).unwrap().remove(0);
                    assert!(got.data == want.data, "M={m} N={n} K={k} {odt:?}: {name} differs from the one-row GEMV");
                }
            }
        }
    }
}

#[test]
fn mr_rows_do_not_depend_on_m_or_position() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(0x6d92);
    let (n, k) = (1000usize, 5120usize);
    // the 4-row call is the reference; every shorter call and every single row must reproduce its rows byte for byte
    let hc4 = matmul_mxfp4_w4a16_case(&mut rng, 4, n, k, DType::Bf16, DType::Bf16);
    let full = run_host(&*cuda, "matmul_mxfp4_gemv_mr", &hc4).unwrap().remove(0);
    let osz = full.dtype.size_bytes();
    for m in 1..=3usize {
        let mut hc = hc4.clone();
        let xb = hc.inputs[0].dtype.size_bytes();
        hc.inputs[0].shape = vec![m, k];
        hc.inputs[0].data.truncate(m * k * xb);
        hc.outputs[0].shape = vec![m, n];
        let got = run_host(&*cuda, "matmul_mxfp4_gemv_mr", &hc).unwrap().remove(0);
        assert!(got.data[..] == full.data[..m * n * osz], "the first {m} rows of a 4-row call differ from the {m}-row call");
    }
}
