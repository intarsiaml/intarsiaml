//! `matmul_mxfp4_mma_fold` (lane mx, the default W4A4 prefill GEMM since 2026-10-25, INTARSIA_HIP_MXFP4_FOLD=0 turns it
//! off; kernels/hip/mxfp4.hip `launch_fold`): MXFP4 W4A4 with
//! both operands folded into e4m3 against the row's largest group scale, one fp8 WMMA GEMM. It changes numerics (the fp8
//! WMMA's sums differ from the bf16 chain's in the last bits), so it is checked against `matmul_mxfp4_mma` by error, and
//! bit for bit only where it must be:
//!
//! 1. `fold_matches_mma_closely`: 27B shapes at prefill M, W4A4, f32 out: relative L2 error vs the f64 exact sum of the same
//!    exact products no worse than 2x the mma kernel's (both are f32 accumulations of the same products), and rel L2 vs
//!    mma < 1e-5. Negative control: one flipped weight code moves the output far past that bound.
//! 2. `fold_falls_back_bitwise`: W4A16 (act = 0) and K % 64 == 32 run the bf16 kernel inside: bitwise `matmul_mxfp4_mma`.
//! 3. `fold_rows_do_not_depend_on_m`: on every 27B shape a row's output is the same at 13 M from 9 to 4096 as at M = 4100
//!    (every tile of the table, the 256-row chunk): batched = solo; negative controls (one flipped code moves only its
//!    column; one changed row moves only itself).
//! 4. `fold_wide_activation_span`: rows whose activation groups span more than 14 octaves round only their tiny groups:
//!    error vs exact stays small relative to the row's output norm.
//!
//! Needs a GPU: skips without one.
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

fn rel_l2(a: &[f32], b: &[f64]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        num += (*x as f64 - y) * (*x as f64 - y);
        den += y * y;
    }
    (num / den.max(1e-300)).sqrt()
}

fn rel_l2_f(a: &[f32], b: &[f32]) -> f64 {
    let bd: Vec<f64> = b.iter().map(|v| *v as f64).collect();
    rel_l2(a, &bd)
}

/// exact f64 sums of the first `rows` rows
fn exact(hc: &HostCall, m: usize, n: usize, k: usize, rows: usize) -> Vec<f64> {
    let xq = qdq_rows(&hc.inputs[0].to_f32(), m, k);
    let wd = dequant_rows(&hc.inputs[1].data, &hc.inputs[2].data, n, k);
    let mut out = vec![0f64; rows * n];
    for r in 0..rows {
        for c in 0..n {
            out[r * n + c] = (0..k).map(|i| xq[r * k + i] as f64 * wd[c * k + i] as f64).sum();
        }
    }
    out
}

#[test]
fn fold_matches_mma_closely() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(81);
    for (m, n, k) in [(256usize, 1024usize, 5120usize), (300, 640, 17408), (2048, 512, 6144), (130, 48, 5120)] {
        let hc = matmul_mxfp4_case(&mut rng, m, n, k, Mxfp4Act::Dynamic, DType::Bf16, DType::F32);
        let f = run_host(&*hip, "matmul_mxfp4_mma_fold", &hc).unwrap()[0].to_f32();
        let g = run_host(&*hip, "matmul_mxfp4_mma", &hc).unwrap()[0].to_f32();
        let rows = m.min(16);
        let ex = exact(&hc, m, n, k, rows);
        let ef = rel_l2(&f[..rows * n], &ex);
        let eg = rel_l2(&g[..rows * n], &ex);
        let fg = rel_l2_f(&f, &g);
        let same = f.iter().zip(&g).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
        eprintln!("m={m} n={n} k={k}: rel L2 fold {ef:.2e}, mma {eg:.2e} (vs f64 exact); fold vs mma {fg:.2e}; {same} of {} outputs bitwise", f.len());
        assert!(ef <= 2.0 * eg + 1e-9, "fold error {ef:.3e} > 2x mma {eg:.3e}");
        assert!(fg < 1e-5, "fold vs mma rel L2 {fg:.3e}");
        // bf16 out: the same kernel, rounded at the end
        let mut hb = hc.clone();
        hb.outputs = vec![OutSpec::new(DType::Bf16, &[m, n])];
        let fb = run_host(&*hip, "matmul_mxfp4_mma_fold", &hb).unwrap()[0].to_f32();
        assert!(rel_l2_f(&fb, &g) < 1e-2, "bf16 out");
    }
    // negative control: one flipped weight code is far past the bound
    let (m, n, k) = (256usize, 256usize, 1024usize);
    let hc = matmul_mxfp4_case(&mut rng, m, n, k, Mxfp4Act::Dynamic, DType::Bf16, DType::F32);
    let a = run_host(&*hip, "matmul_mxfp4_mma_fold", &hc).unwrap()[0].to_f32();
    let mut hm = hc.clone();
    for r in 0..n {
        hm.inputs[1].data[r * k / 2 + 100] ^= 0x07;
    }
    let b = run_host(&*hip, "matmul_mxfp4_mma", &hm).unwrap()[0].to_f32();
    let d = rel_l2_f(&a, &b);
    assert!(d > 1e-3, "negative control: flipped codes must show ({d:.3e})");
}

#[test]
fn fold_falls_back_bitwise() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(82);
    for (m, n, k, act) in [(300usize, 640usize, 5120usize, Mxfp4Act::None), (2048, 1024, 5120, Mxfp4Act::None), (130, 131, 1056, Mxfp4Act::Dynamic)] {
        for odt in [DType::Bf16, DType::F32] {
            let hc = matmul_mxfp4_case(&mut rng, m, n, k, act, DType::Bf16, odt);
            let f = run_host(&*hip, "matmul_mxfp4_mma_fold", &hc).unwrap();
            let g = run_host(&*hip, "matmul_mxfp4_mma", &hc).unwrap();
            assert!(f[0].data == g[0].data, "m={m} n={n} k={k} act={act:?} {odt:?}: fallback must be bitwise matmul_mxfp4_mma");
        }
    }
}

/// rows r0..r0+m of `big` as an M = m call (same weights, scales, act flag)
fn rows_of(big: &HostCall, r0: usize, m: usize, k: usize, n: usize) -> HostCall {
    let x = &big.inputs[0];
    let mut hc = big.clone();
    hc.inputs[0] = HostTensor::new(x.dtype, vec![m, k], x.data[r0 * k * 2..(r0 + m) * k * 2].to_vec()).unwrap();
    hc.outputs = vec![OutSpec::new(big.outputs[0].dtype, &[m, n])];
    hc
}

/// Batched = solo at every prefill M (lane mx fold-default): for every 27B projection shape, rows of an M = 4100 call are
/// bitwise the same rows run at M in {9, 16, 17, 63, 64, 65, 255, 256, 257, 2047, 2048, 2049, 4096} (the GEMV / fold
/// boundary, the fold tile table's M <= 64 and M >= 2048 switches, the 256-row chunk). Negative controls: one flipped weight
/// code changes its column and no other; one changed activation row changes that row and no other. How often v2 differs
/// from the fold on the same rows is printed (context).
#[test]
fn fold_rows_do_not_depend_on_m() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(83);
    const BIG: usize = 4100;
    let shapes = [(17408usize, 5120usize), (5120, 17408), (10240, 5120), (5120, 6144), (6144, 5120), (12288, 5120), (1024, 5120), (48, 5120)];
    for (si, (n, k)) in shapes.into_iter().enumerate() {
        let odt = if si == 6 { DType::F32 } else { DType::Bf16 };
        let big = matmul_mxfp4_case(&mut rng, BIG, n, k, Mxfp4Act::Dynamic, DType::Bf16, odt);
        let full = run_host(&*hip, "matmul_mxfp4_mma_fold", &big).unwrap();
        let ob = odt.size_bytes() * n;
        for m in [9usize, 16, 17, 63, 64, 65, 255, 256, 257, 2047, 2048, 2049, 4096] {
            let r0 = 1000usize.min(BIG - m) - (m % 3);
            let part = run_host(&*hip, "matmul_mxfp4_mma_fold", &rows_of(&big, r0, m, k, n)).unwrap();
            assert!(part[0].data == full[0].data[r0 * ob..(r0 + m) * ob], "n={n} k={k}: rows {r0}..{} at M={m} differ from M={BIG}", r0 + m);
        }
        // context, not asserted: how many outputs of v2 (another chain) differ from the fold on 300 of these rows
        let v2 = run_host(&*hip, "matmul_mxfp4_mma", &rows_of(&big, 7, 300, k, n)).unwrap();
        let es = odt.size_bytes();
        let differ = v2[0].data.chunks(es).zip(full[0].data[7 * ob..307 * ob].chunks(es)).filter(|(a, b)| a != b).count();
        eprintln!("n={n} k={k} {odt:?}: rows independent of M at 13 sizes; v2 differs from the fold on {differ} of {} outputs", 300 * n);
    }
    // negative control 1: one flipped weight code (column 5) moves column 5 and nothing else, so the bitwise comparison
    // above sees a one-code change
    {
        let (n, k, m) = (64usize, 5120usize, 130usize);
        let hc = matmul_mxfp4_case(&mut rng, m, n, k, Mxfp4Act::Dynamic, DType::Bf16, DType::F32);
        let a = run_host(&*hip, "matmul_mxfp4_mma_fold", &hc).unwrap()[0].to_f32();
        let mut hm = hc.clone();
        hm.inputs[1].data[5 * k / 2 + 40] ^= 0x07;
        let b = run_host(&*hip, "matmul_mxfp4_mma_fold", &hm).unwrap()[0].to_f32();
        let moved: Vec<usize> = (0..m * n).filter(|&i| a[i].to_bits() != b[i].to_bits()).map(|i| i % n).collect();
        assert!(!moved.is_empty() && moved.iter().all(|&c| c == 5), "flipped code in column 5: changed columns {moved:?}");
    }
    // negative control 2: one changed activation row moves that row only
    let (n, k, m) = (1024usize, 5120usize, 300usize);
    let hc = matmul_mxfp4_case(&mut rng, m, n, k, Mxfp4Act::Dynamic, DType::Bf16, DType::Bf16);
    let a = run_host(&*hip, "matmul_mxfp4_mma_fold", &hc).unwrap()[0].data.clone();
    let mut x = hc.inputs[0].to_f32();
    for e in 0..k {
        x[77 * k + e] *= -1.5;
    }
    let mut hm = hc.clone();
    hm.inputs[0] = HostTensor::from_f32_as(DType::Bf16, &[m, k], &x);
    let b = run_host(&*hip, "matmul_mxfp4_mma_fold", &hm).unwrap()[0].data.clone();
    let rb = n * 2;
    for r in 0..m {
        let same = a[r * rb..(r + 1) * rb] == b[r * rb..(r + 1) * rb];
        assert_eq!(same, r != 77, "row {r}: only row 77 was changed");
    }
}

#[test]
fn fold_wide_activation_span() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(84);
    let (m, n, k) = (128usize, 256usize, 5120usize);
    let mut hc = matmul_mxfp4_case(&mut rng, m, n, k, Mxfp4Act::Dynamic, DType::Bf16, DType::F32);
    let mut x = hc.inputs[0].to_f32();
    for r in 0..m {
        // one outlier group 2^18 above the rest, every 7th group 2^-6 below: spans of 18 and 24 octaves
        for e in 0..32 {
            x[r * k + (r % 160) * 32 + e] *= 262144.0;
        }
        for g in (3..k / 32).step_by(7) {
            for e in 0..32 {
                x[r * k + g * 32 + e] *= 1.0 / 64.0;
            }
        }
    }
    hc.inputs[0] = HostTensor::from_f32_as(DType::Bf16, &[m, k], &x);
    let f = run_host(&*hip, "matmul_mxfp4_mma_fold", &hc).unwrap()[0].to_f32();
    let g = run_host(&*hip, "matmul_mxfp4_mma", &hc).unwrap()[0].to_f32();
    let rows = 16;
    let ex = exact(&hc, m, n, k, rows);
    let ef = rel_l2(&f[..rows * n], &ex);
    let eg = rel_l2(&g[..rows * n], &ex);
    eprintln!("wide span: rel L2 fold {ef:.2e}, mma {eg:.2e}");
    assert!(ef < 1e-3, "wide-span rows: fold rel L2 {ef:.3e}");
}
