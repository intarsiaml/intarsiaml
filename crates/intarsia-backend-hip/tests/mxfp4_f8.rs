//! `matmul_mxfp4_mma_f8` (lane mx, kernels/hip/mxfp4.hip `launch_f8`): the MXFP4 prefill GEMM on the gfx12 fp8 x fp8 WMMA
//! must give `matmul_mxfp4_mma`'s output bit for bit (the bf16-dequant WMMA chain it replaces).
//!
//! 1. `f8_is_bitwise_mma`: the 27B's projection shapes at prefill M, every tile path of `mxf8_gemm_run` (128 x 128,
//!    128 x 256, 256 x 128, the M <= 64 tile), partial M and N tiles, N not a multiple of 8, K % 64 == 32 (the bf16 fallback
//!    inside), both activation modes (W4A4 runs the fp8 kernel, W4A16 the bf16 one inside) and both output dtypes.
//! 2. `f8_is_bitwise_mma_on_adversarial_scales`: activations with outlier groups (2^+-14 against the rest), all-zero groups,
//!    weight groups zeroed with scale byte 0, and wide weight-scale spans: data on which the f32 sums round. Negative
//!    controls in the same test: the f64 exact sum differs from the mma output on many elements (so the inputs do round,
//!    and an order change would show), and a single flipped weight code changes the f8 output (the comparison is live).
//! 3. `f8_rows_do_not_depend_on_m`: a row's output is the same at M = 128, 300 and 2100 (different tiles and blocks):
//!    batched prefill chunks and solo ones give the same bits.
//!
//! Needs a GPU: skips (passes) without one.
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
        panic!("{label}: f8 differs from mma in {bad} of {} elements (first at {first})", got.len() / w);
    }
}

fn f8_vs_mma(hip: &dyn Backend, hc: &HostCall, label: &str) -> Vec<HostTensor> {
    let got = run_host(hip, "matmul_mxfp4_mma_f8", hc).unwrap();
    let want = run_host(hip, "matmul_mxfp4_mma", hc).unwrap();
    bits_eq(label, &got[0].data, &want[0].data, hc.outputs[0].dtype);
    want
}

// (m, n, k): 27B shapes and edges
const SHAPES: [(usize, usize, usize); 12] = [
    (2048, 17408, 5120), // ffn gate / up, 256 x 128 tile
    (2048, 5120, 17408), // ffn down, 128 x 256 tile
    (300, 10240, 5120),  // gdn in_qkv, 128 x 128, partial M
    (129, 6144, 5120),   // gdn in_z, one row past a tile
    (2049, 1024, 5120),  // attn k / v (128 x 256 needs one block per CU)
    (200, 48, 5120),     // gdn in_b / in_a: N below one tile
    (128, 5120, 6144),   // gdn out / attn o at the smallest M served
    (64, 12288, 5120),   // the M <= 64 tile (served only by a forced call; the registry starts at 128)
    (257, 130, 1088),    // N not a multiple of 8 (scalar stores), K = 17 stages
    (130, 131, 1056),    // K % 64 == 32: the bf16 kernel inside
    (513, 17, 64),       // one stage
    (1000, 8, 128),
];

#[test]
fn f8_is_bitwise_mma() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(91);
    for (m, n, k) in SHAPES {
        for act in [Mxfp4Act::Dynamic, Mxfp4Act::None] {
            for odt in [DType::Bf16, DType::F32] {
                let hc = matmul_mxfp4_case(&mut rng, m, n, k, act, DType::Bf16, odt);
                f8_vs_mma(&*hip, &hc, &format!("m={m} n={n} k={k} act={act:?} out={odt:?}"));
            }
        }
    }
}

/// Make activations whose 32-groups span a wide exponent range (outliers, tiny groups, zero groups) and weights with
/// zeroed groups (scale byte 0) and wide scale spans.
fn adversarial(rng: &mut SplitMix64, hc: &mut HostCall, m: usize, n: usize, k: usize) {
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
    for r in 0..n {
        for g in 0..k / 32 {
            let roll = rng.next_u64() % 11;
            if roll == 0 {
                w[r * k / 2 + g * 16..r * k / 2 + g * 16 + 16].fill(0);
                s[r * k / 32 + g] = 0;
            } else {
                s[r * k / 32 + g] = 112 + (rng.next_u64() % 16) as u8;
            }
        }
    }
    hc.inputs[1].data = w;
    hc.inputs[2].data = s;
}

#[test]
fn f8_is_bitwise_mma_on_adversarial_scales() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(92);
    let mut inexact_total = 0usize;
    for (m, n, k) in [(256usize, 512usize, 5120usize), (300, 640, 17408), (2048, 1024, 6144)] {
        let mut hc = matmul_mxfp4_case(&mut rng, m, n, k, Mxfp4Act::Dynamic, DType::Bf16, DType::F32);
        adversarial(&mut rng, &mut hc, m, n, k);
        let want = f8_vs_mma(&*hip, &hc, &format!("adversarial m={m} n={n} k={k} f32"));
        let mut hb = hc.clone();
        hb.outputs = vec![OutSpec::new(DType::Bf16, &[m, n])];
        f8_vs_mma(&*hip, &hb, &format!("adversarial m={m} n={n} k={k} bf16"));
        // negative control 1: the f64 exact sum of the same (exact) products differs from the f32 chain on many outputs,
        // so these inputs do round and a different summation order would change bits
        let xq = qdq_rows(&hc.inputs[0].to_f32(), m, k);
        let wd = dequant_rows(&hc.inputs[1].data, &hc.inputs[2].data, n, k);
        let got = want[0].to_f32();
        let rows = m.min(64);
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
    assert!(inexact_total > 100, "the adversarial inputs must make the f32 sums round ({inexact_total})");
    // negative control 2: one flipped weight code changes the f8 output (the comparison sees the kernel)
    let (m, n, k) = (256usize, 256usize, 1024usize);
    let hc = matmul_mxfp4_case(&mut rng, m, n, k, Mxfp4Act::Dynamic, DType::Bf16, DType::F32);
    let a = run_host(&*hip, "matmul_mxfp4_mma_f8", &hc).unwrap();
    let mut hm = hc.clone();
    hm.inputs[1].data[7 * k / 2 + 100] ^= 0x07;
    let b = run_host(&*hip, "matmul_mxfp4_mma_f8", &hm).unwrap();
    assert!(count_diff(&a[0].data, &b[0].data, DType::F32) > 0, "a flipped weight code must change the output");
}

#[test]
fn f8_rows_do_not_depend_on_m() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(93);
    let (n, k) = (10240usize, 5120usize);
    let big = matmul_mxfp4_case(&mut rng, 2100, n, k, Mxfp4Act::Dynamic, DType::Bf16, DType::Bf16);
    let full = run_host(&*hip, "matmul_mxfp4_mma_f8", &big).unwrap();
    let x = &big.inputs[0];
    let row_x = k * 2;
    let row_o = n * 2;
    for m in [128usize, 300] {
        // rows 1000 .. 1000 + m of the big call, as their own call
        let mut hc = big.clone();
        hc.inputs[0] = HostTensor::new(x.dtype, vec![m, k], x.data[1000 * row_x..(1000 + m) * row_x].to_vec()).unwrap();
        hc.outputs = vec![OutSpec::new(DType::Bf16, &[m, n])];
        let part = run_host(&*hip, "matmul_mxfp4_mma_f8", &hc).unwrap();
        bits_eq(&format!("rows 1000..{} at M={m} vs M=2100", 1000 + m), &part[0].data, &full[0].data[1000 * row_o..(1000 + m) * row_o], DType::Bf16);
    }
}
