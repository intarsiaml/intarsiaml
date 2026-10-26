//! `matmul_mxfp4_mma` v2 (lane mx, kernels/hip/mxfp4.hip `Mxfp4PolicyV2` + `mx_gemm_run`) must give the output of the
//! kernel it replaced, `matmul_mxfp4_mma_v1`, bit for bit.
//!
//! 1. `v2_is_bitwise_v1`: the 27B's projection shapes at prefill M (every tile of `mx_gemm_run`'s table), partial M and N
//!    tiles, N not a multiple of 8, both activation modes (W4A4 QDQ, W4A16) and both output dtypes.
//! 2. `v2_is_bitwise_v1_on_every_scale_byte`: weight scale bytes 0..=255 (v2's direct bf16 build for 3..=250, v1's path for
//!    the rest: subnormal 2^-127.., NaN at 255), every code, activations with outlier and all-zero groups so the f32 sums
//!    round. Negative controls in the same test: the outputs really round (f64 exact sum != output on many elements, so a
//!    changed summation order would show), and one flipped weight code changes the output (the comparison is live).
//! 3. `v2_rows_do_not_depend_on_m`: a row's output is the same at M = 128, 300 and 2100 (different tiles and blocks), so
//!    a batched prefill chunk and a solo one give the same bits.
//!
//! Tiles outside the table (INTARSIA_HIP_MXFP4_TILE) are checked by scripts/mx/mxbench.hip. Needs a GPU: skips without.
#![cfg(ik_hip)]

use intarsia_core::backend::Backend;
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::quant::cases::matmul_mxfp4_case;
use intarsia_core::quant::mxfp4::{dequant_rows, qdq_rows};
use intarsia_core::quant::Mxfp4Act;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor};
use std::sync::Arc;

fn device() -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_HIP_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_hip::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            assert!(std::env::var("INTARSIA_REQUIRE_GPU").is_err(), "INTARSIA_REQUIRE_GPU set but no HIP device: {e}");
            eprintln!("skipping: no usable HIP device ({e})");
            None
        }
    }
}

fn count_diff(got: &[u8], want: &[u8], dt: DType) -> usize {
    let w = dt.size_bytes();
    got.chunks(w).zip(want.chunks(w)).filter(|(a, b)| a != b).count()
}

fn bits_eq(label: &str, got: &[u8], want: &[u8], dt: DType) {
    assert_eq!(got.len(), want.len(), "{label}: lengths");
    let bad = count_diff(got, want, dt);
    if bad > 0 {
        let w = dt.size_bytes();
        let first = got.chunks(w).zip(want.chunks(w)).position(|(a, b)| a != b).unwrap();
        panic!("{label}: v2 differs from v1 in {bad} of {} elements (first at {first})", got.len() / w);
    }
}

fn v2_vs_v1(hip: &dyn Backend, hc: &HostCall, label: &str) -> Vec<HostTensor> {
    let got = run_host(hip, "matmul_mxfp4_mma", hc).unwrap();
    let want = run_host(hip, "matmul_mxfp4_mma_v1", hc).unwrap();
    bits_eq(label, &got[0].data, &want[0].data, hc.outputs[0].dtype);
    want
}

// (m, n, k): 27B shapes and edges
const SHAPES: [(usize, usize, usize); 12] = [
    (4096, 17408, 5120), // ffn gate / up
    (2048, 5120, 17408), // ffn down
    (4096, 10240, 5120), // gdn in_qkv
    (300, 6144, 5120),   // gdn in_z, partial M
    (2049, 1024, 5120),  // attn k / v, one row past
    (200, 48, 5120),     // gdn in_b / in_a: N below one tile
    (128, 5120, 6144),   // gdn out / attn o
    (64, 12288, 5120),   // the 128 x 64 tile
    (20, 640, 512),      // the 128 x 32 tile
    (9, 256, 256),       // the 128 x 16 tile
    (257, 130, 1088),    // N not a multiple of 8 (scalar stores)
    (513, 17, 64),       // one k stage pair
];

#[test]
fn v2_is_bitwise_v1() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(91);
    for (m, n, k) in SHAPES {
        for act in [Mxfp4Act::Dynamic, Mxfp4Act::None] {
            for odt in [DType::Bf16, DType::F32] {
                let hc = matmul_mxfp4_case(&mut rng, m, n, k, act, DType::Bf16, odt);
                v2_vs_v1(&*hip, &hc, &format!("m={m} n={n} k={k} act={act:?} out={odt:?}"));
            }
        }
    }
}

/// Activations whose 32-groups span a wide exponent range (outliers, tiny and zero groups); weight scale bytes cycle
/// through every value 0..=255, some groups zeroed.
fn adversarial(rng: &mut SplitMix64, hc: &mut HostCall, m: usize, n: usize, k: usize, all_bytes: bool) {
    let mut x = hc.inputs[0].to_f32();
    for r in 0..m {
        for g in 0..k / 32 {
            let f = match rng.next_u64() % 9 {
                0 => 16384.0,
                1 => 1.0 / 16384.0,
                2 => 0.0,
                3 => 64.0,
                _ => 1.0,
            };
            for e in 0..32 {
                x[r * k + g * 32 + e] *= f;
            }
        }
    }
    hc.inputs[0] = HostTensor::from_f32_as(DType::Bf16, &[m, k], &x);
    let mut w = hc.inputs[1].data.clone();
    let mut s = hc.inputs[2].data.clone();
    let mut i = 0usize;
    for r in 0..n {
        for g in 0..k / 32 {
            if all_bytes {
                s[r * k / 32 + g] = (i % 256) as u8;
                i += 1;
            } else {
                s[r * k / 32 + g] = 112 + (rng.next_u64() % 16) as u8;
            }
            if rng.next_u64() % 11 == 0 {
                w[r * k / 2 + g * 16..r * k / 2 + g * 16 + 16].fill(0);
            }
        }
    }
    hc.inputs[1].data = w;
    hc.inputs[2].data = s;
}

#[test]
fn v2_is_bitwise_v1_on_every_scale_byte() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(92);
    // every scale byte, f32 and bf16 out (outputs with inf / NaN compare as bits too)
    for (m, n, k) in [(130usize, 256usize, 1024usize), (2048, 512, 2048)] {
        for act in [Mxfp4Act::Dynamic, Mxfp4Act::None] {
            let mut hc = matmul_mxfp4_case(&mut rng, m, n, k, act, DType::Bf16, DType::F32);
            adversarial(&mut rng, &mut hc, m, n, k, true);
            v2_vs_v1(&*hip, &hc, &format!("all scale bytes m={m} n={n} k={k} act={act:?} f32"));
            let mut hb = hc.clone();
            hb.outputs = vec![OutSpec::new(DType::Bf16, &[m, n])];
            v2_vs_v1(&*hip, &hb, &format!("all scale bytes m={m} n={n} k={k} act={act:?} bf16"));
        }
    }
    // model-range scales with rounding sums, and the negative controls
    let mut inexact_total = 0usize;
    for (m, n, k) in [(256usize, 512usize, 5120usize), (300, 640, 17408), (2048, 1024, 6144)] {
        let mut hc = matmul_mxfp4_case(&mut rng, m, n, k, Mxfp4Act::Dynamic, DType::Bf16, DType::F32);
        adversarial(&mut rng, &mut hc, m, n, k, false);
        let want = v2_vs_v1(&*hip, &hc, &format!("adversarial m={m} n={n} k={k} f32"));
        let xq = qdq_rows(&hc.inputs[0].to_f32(), m, k);
        let wd = dequant_rows(&hc.inputs[1].data, &hc.inputs[2].data, n, k);
        let got = want[0].to_f32();
        let rows = m.min(32);
        let mut inexact = 0usize;
        for r in 0..rows {
            for c in 0..n {
                let ex: f64 = (0..k).map(|i| xq[r * k + i] as f64 * wd[c * k + i] as f64).sum();
                if (ex as f32).to_bits() != got[r * n + c].to_bits() {
                    inexact += 1;
                }
            }
        }
        eprintln!("adversarial m={m} n={n} k={k}: {inexact} of {} checked outputs round (f32 chain != f64 exact)", rows * n);
        inexact_total += inexact;
    }
    assert!(inexact_total > 100, "negative control: the adversarial inputs must make the f32 sums round ({inexact_total})");
    let (m, n, k) = (256usize, 256usize, 1024usize);
    let hc = matmul_mxfp4_case(&mut rng, m, n, k, Mxfp4Act::Dynamic, DType::Bf16, DType::F32);
    let a = run_host(&*hip, "matmul_mxfp4_mma", &hc).unwrap();
    let mut hm = hc.clone();
    hm.inputs[1].data[7 * k / 2 + 100] ^= 0x07;
    let b = run_host(&*hip, "matmul_mxfp4_mma_v1", &hm).unwrap();
    assert!(count_diff(&a[0].data, &b[0].data, DType::F32) > 0, "negative control: a flipped weight code must change the output");
}

#[test]
fn v2_rows_do_not_depend_on_m() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(93);
    let (n, k) = (10240usize, 5120usize);
    let big = matmul_mxfp4_case(&mut rng, 2100, n, k, Mxfp4Act::Dynamic, DType::Bf16, DType::Bf16);
    let full = run_host(&*hip, "matmul_mxfp4_mma", &big).unwrap();
    let x = &big.inputs[0];
    let (row_x, row_o) = (k * 2, n * 2);
    for m in [128usize, 300] {
        let mut hc = big.clone();
        hc.inputs[0] = HostTensor::new(x.dtype, vec![m, k], x.data[1000 * row_x..(1000 + m) * row_x].to_vec()).unwrap();
        hc.outputs = vec![OutSpec::new(DType::Bf16, &[m, n])];
        let part = run_host(&*hip, "matmul_mxfp4_mma", &hc).unwrap();
        bits_eq(
            &format!("rows 1000..{} at M={m} vs M=2100", 1000 + m),
            &part[0].data,
            &full[0].data[1000 * row_o..(1000 + m) * row_o],
            DType::Bf16,
        );
    }
}
