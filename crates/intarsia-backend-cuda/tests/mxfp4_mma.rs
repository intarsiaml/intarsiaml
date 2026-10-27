//! `matmul_mxfp4_mma` (kernels/cuda/mxfp4_gemm.cu: the W4A16 tensor-core GEMM of the 27B's MXFP4 projections):
//! 1. a row's output bytes do not depend on M (tile shapes of 16, 32 and 64 rows), on the other rows or on its position in the
//!    call (the engine's batched == solo rule inside one call): rows of an M-row call equal the same rows of a 300-row call, and a
//!    call with the rows reversed gives the reversed rows;
//! 2. it agrees with the GEMV (`matmul_mxfp4_gemv`, f32 FMA per 32-group) up to accumulation order: both graded against each other
//!    here in rel_l2 and by the kernel admission sweep against the CPU reference;
//! 3. `matmul_mxfp4_gemv_rows` (the rows_exact twin) is the GEMV, byte for byte, on a W4A16 and a W4A4 call.
//! Needs a GPU: without one the tests print SKIPPED (and fail under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_cuda)]

mod common;

use intarsia_core::backend::Backend;
use intarsia_core::exec::{run_host, HostCall};
use intarsia_core::quant::cases::{matmul_mxfp4_case, matmul_mxfp4_w4a16_case};
use intarsia_core::quant::Mxfp4Act;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor};
use std::sync::Arc;

fn device() -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_cuda::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            common::skip("mxfp4_mma", &format!("no usable CUDA device ({e})"));
            None
        }
    }
}

/// `hc` with its activations replaced by `rows` (each `k` long), same weights and output dtype.
fn with_rows(hc: &HostCall, rows: &[Vec<f32>], k: usize) -> HostCall {
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let mut out = hc.clone();
    out.inputs[0] = HostTensor::from_f32_as(DType::Bf16, &[rows.len(), k], &flat);
    out.outputs[0].shape = vec![rows.len(), hc.outputs[0].shape[1]];
    out
}

fn row_bytes(t: &HostTensor, r: usize, n: usize) -> &[u8] {
    let osz = t.dtype.size_bytes();
    &t.data[r * n * osz..(r + 1) * n * osz]
}

fn rel_l2(a: &HostTensor, b: &HostTensor) -> f64 {
    let (a, b) = (a.to_f32(), b.to_f32());
    let (mut d, mut s) = (0f64, 0f64);
    for (x, y) in a.iter().zip(&b) {
        d += ((x - y) as f64).powi(2);
        s += (*x as f64).powi(2);
    }
    (d / s.max(1e-30)).sqrt()
}

#[test]
fn mxfp4_mma_rows_do_not_depend_on_m_or_position() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(0x6d78);
    // (n, k): a tail in N (not a multiple of the 128-row weight tile), one and several 128-wide k units, the 27B's hidden size
    for (n, k) in [(136usize, 256usize), (1000, 512), (520, 5120), (5120, 640)] {
        for odt in [DType::Bf16, DType::F32] {
            let big = matmul_mxfp4_w4a16_case(&mut rng, 300, n, k, DType::Bf16, odt);
            let rows_all: Vec<Vec<f32>> = big.inputs[0].to_f32().chunks(k).map(|c| c.to_vec()).collect();
            let want = run_host(&*cuda, "matmul_mxfp4_mma", &big).unwrap().remove(0);
            for m in [1usize, 2, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 255] {
                let part = with_rows(&big, &rows_all[..m], k);
                let got = run_host(&*cuda, "matmul_mxfp4_mma", &part).unwrap().remove(0);
                for r in 0..m {
                    assert!(row_bytes(&got, r, n) == row_bytes(&want, r, n), "n={n} k={k} {odt:?}: row {r} of an M={m} call differs from the 300-row call");
                }
            }
            // a window of rows that does not start at 0 and the same rows in reversed order
            let (lo, hi) = (37usize, 37 + 70);
            let rev: Vec<Vec<f32>> = rows_all[lo..hi].iter().rev().cloned().collect();
            let got = run_host(&*cuda, "matmul_mxfp4_mma", &with_rows(&big, &rev, k)).unwrap().remove(0);
            for r in 0..(hi - lo) {
                assert!(row_bytes(&got, r, n) == row_bytes(&want, hi - 1 - r, n), "n={n} k={k} {odt:?}: reversed window, row {r}");
            }
            let again = run_host(&*cuda, "matmul_mxfp4_mma", &big).unwrap().remove(0);
            assert!(again.data == want.data, "n={n} k={k} {odt:?}: not repeatable");
        }
    }
}

#[test]
fn mxfp4_mma_agrees_with_the_gemv() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(0x6d79);
    for (m, n, k) in [(9usize, 136usize, 256usize), (40, 1000, 512), (300, 1000, 512), (70, 5120, 5120), (257, 17408, 5120)] {
        for odt in [DType::Bf16, DType::F32] {
            let hc = matmul_mxfp4_w4a16_case(&mut rng, m, n, k, DType::Bf16, odt);
            let gemv = run_host(&*cuda, "matmul_mxfp4_gemv", &hc).unwrap().remove(0);
            let mma = run_host(&*cuda, "matmul_mxfp4_mma", &hc).unwrap().remove(0);
            let e = rel_l2(&gemv, &mma);
            // same products (exact in bf16 x bf16 -> f32), a different order of the f32 sums; an f32 output differs at 1e-6
            let bound = if odt == DType::F32 { 1e-5 } else { 4e-3 };
            assert!(e < bound, "M={m} N={n} K={k} {odt:?}: rel_l2 {e:.3e} against the GEMV (bound {bound:.0e})");
        }
    }
}

#[test]
fn mxfp4_gemv_rows_is_the_gemv() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(0x6d7a);
    for (m, n, k) in [(1usize, 136usize, 256usize), (3, 1000, 512), (16, 520, 5120), (40, 300, 160)] {
        for odt in [DType::Bf16, DType::F32] {
            for hc in [
                matmul_mxfp4_w4a16_case(&mut rng, m, n, k, DType::Bf16, odt),
                matmul_mxfp4_case(&mut rng, m, n, k, Mxfp4Act::Dynamic, DType::Bf16, odt),
            ] {
                let a = run_host(&*cuda, "matmul_mxfp4_gemv", &hc).unwrap().remove(0);
                let b = run_host(&*cuda, "matmul_mxfp4_gemv_rows", &hc).unwrap().remove(0);
                assert!(a.data == b.data, "M={m} N={n} K={k} {odt:?}: gemv_rows differs from gemv");
            }
        }
    }
}
