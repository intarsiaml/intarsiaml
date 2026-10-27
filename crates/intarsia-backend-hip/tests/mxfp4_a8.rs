//! W4A8 (MXFP4 weights x dynamic per-token fp8 e4m3 activations, `OpAttrs::mxfp4_a8`; lane W4A8, kernels/hip/mxfp4_a8.hip and
//! mxfp4_a8_gemm.hip): the decode kernels and the prefill GEMM against the f64 reference of the same math
//! (`quant::reference::matmul_mxfp4_a8`), and the batch-invariance the engine's exactness rests on.
//!
//! 1. `a8_kernels_match_the_reference`: every W4A8 kernel (`matmul_mxfp4_a8_gemv`, `_gemv_rows`, `_mma`) at decode and prefill row
//!    counts, f32 out, rel L2 vs the f64 sum < 3e-5, bf16 out compared with the bf16-rounded reference.
//! 2. `a8_mma_wide_scale_span_takes_the_group_path`: a weight row whose scale bytes span more than 14 octaves (outside the exact
//!    fold) still matches the reference (the kernel falls back to the per-group rescale per call).
//! 3. `a8_mma_rows_do_not_depend_on_m`: rows of an M = 4100 call are bitwise the rows run at other M (chunked prefill, tails).
//! 4. `a8_rows_are_bitwise_the_solo_kernel`: `_gemv_rows` at M = 1..16 == `_gemv` M = 1 per row; the decode kernels agree with the
//!    prefill GEMM to 1e-5 (another summation order).
//! 5. Negative control: one flipped weight code moves the output.
//!
//! Needs a GPU: skips without one. INTARSIA_HIP_MXFP4_A8_PERGROUP=1 reruns 1-3 on the per-group path.
#![cfg(ik_hip)]

use intarsia_core::backend::Backend;
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::quant::cases::matmul_mxfp4_case;
use intarsia_core::quant::reference::matmul_mxfp4_a8;
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

fn want(hc: &HostCall, m: usize, n: usize, k: usize) -> Vec<f64> {
    matmul_mxfp4_a8(&hc.inputs[0].to_f32(), m, k, &hc.inputs[1].data, &hc.inputs[2].data, n)
}

fn rows_of(big: &HostCall, r0: usize, m: usize, k: usize, n: usize) -> HostCall {
    let x = &big.inputs[0];
    let mut hc = big.clone();
    hc.inputs[0] = HostTensor::new(x.dtype, vec![m, k], x.data[r0 * k * 2..(r0 + m) * k * 2].to_vec()).unwrap();
    hc.outputs = vec![OutSpec::new(big.outputs[0].dtype, &[m, n])];
    hc
}

#[test]
fn a8_kernels_match_the_reference() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(91);
    for (m, n, k) in [(1usize, 640usize, 5120usize), (3, 1024, 5120), (8, 5120, 6144), (16, 301, 2080), (17, 640, 5120), (64, 1024, 5120), (300, 1000, 512), (300, 5120, 5120), (130, 131, 1056), (1, 17408, 5120)] {
        let mut hc = matmul_mxfp4_case(&mut rng, m, n, k, Mxfp4Act::Fp8, DType::Bf16, DType::F32);
        if m >= 3 {
            // an all-zero row (scale clamps to the minimum, outputs +0) and a tiny one (every element quantises to 0 or a subnormal)
            let mut x = hc.inputs[0].to_f32();
            x[k..2 * k].fill(0.0);
            x[2 * k..3 * k].iter_mut().for_each(|v| *v *= 1e-35);
            hc.inputs[0] = HostTensor::from_f32_as(DType::Bf16, &[m, k], &x);
        }
        let w = want(&hc, m, n, k);
        let kernels: &[&str] = if m <= 8 {
            &["matmul_mxfp4_a8_gemv", "matmul_mxfp4_a8_gemv_rows", "matmul_mxfp4_a8_mma"]
        } else if m <= 16 {
            &["matmul_mxfp4_a8_gemv_rows", "matmul_mxfp4_a8_mma"]
        } else {
            &["matmul_mxfp4_a8_mma"]
        };
        for kernel in kernels {
            let got = run_host(&*hip, kernel, &hc).unwrap().remove(0).to_f32();
            let r = rel_l2(&got, &w);
            assert!(r < 3e-5, "{kernel} M={m} N={n} K={k} f32 out: rel_l2 {r:.3e}");
        }
        // bf16 out: the reference rounded to bf16 (a one-ulp flip near a rounding boundary is allowed: <= 1% of outputs differ)
        let mut hb = hc.clone();
        hb.outputs = vec![OutSpec::new(DType::Bf16, &[m, n])];
        for kernel in kernels {
            let got = run_host(&*hip, kernel, &hb).unwrap().remove(0).to_f32();
            let wf: Vec<f32> = w.iter().map(|v| *v as f32).collect();
            let wr: Vec<f64> = HostTensor::from_f32_as(DType::Bf16, &[wf.len()], &wf).to_f32().iter().map(|v| *v as f64).collect();
            let differ = got.iter().zip(&wr).filter(|(g, r)| **g as f64 != **r).count();
            assert!(differ * 100 <= got.len(), "{kernel} M={m} N={n} K={k} bf16 out: {differ} of {} outputs differ from the rounded reference", got.len());
        }
    }
}

#[test]
fn a8_mma_wide_scale_span_takes_the_group_path() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(92);
    let (m, n, k) = (40usize, 96usize, 1024usize);
    let mut hc = matmul_mxfp4_case(&mut rng, m, n, k, Mxfp4Act::Fp8, DType::Bf16, DType::F32);
    // row 5: one group 20 octaves above the others (and one 20 below), beyond the 14-octave fold bound
    hc.inputs[2].data[5 * (k / 32) + 3] = hc.inputs[2].data[5 * (k / 32)].saturating_add(20).min(250);
    hc.inputs[2].data[5 * (k / 32) + 9] = hc.inputs[2].data[5 * (k / 32)].saturating_sub(20).max(4);
    let w = want(&hc, m, n, k);
    for kernel in ["matmul_mxfp4_a8_mma", "matmul_mxfp4_a8_gemv_rows"] {
        let mut h = hc.clone();
        if kernel.ends_with("_rows") {
            h = rows_of(&hc, 0, 16, k, n);
        }
        let mm = h.outputs[0].shape[0];
        let got = run_host(&*hip, kernel, &h).unwrap().remove(0).to_f32();
        let r = rel_l2(&got, &w[..mm * n]);
        assert!(r < 3e-5, "{kernel}: rel_l2 {r:.3e}");
    }
}

#[test]
fn a8_mma_rows_do_not_depend_on_m() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(93);
    const BIG: usize = 4100;
    for (n, k, odt) in [(17408usize, 5120usize, DType::Bf16), (5120, 17408, DType::Bf16), (10240, 5120, DType::Bf16), (5120, 6144, DType::F32), (1024, 5120, DType::Bf16), (48, 5120, DType::Bf16)] {
        let big = matmul_mxfp4_case(&mut rng, BIG, n, k, Mxfp4Act::Fp8, DType::Bf16, odt);
        let full = run_host(&*hip, "matmul_mxfp4_a8_mma", &big).unwrap();
        let ob = odt.size_bytes() * n;
        for m in [1usize, 9, 16, 17, 63, 64, 65, 255, 256, 257, 2047, 2048, 2049, 4096] {
            let r0 = 1000usize.min(BIG - m) - (m % 3);
            let part = run_host(&*hip, "matmul_mxfp4_a8_mma", &rows_of(&big, r0, m, k, n)).unwrap();
            assert!(part[0].data == full[0].data[r0 * ob..(r0 + m) * ob], "n={n} k={k}: rows {r0}..{} at M={m} differ from M={BIG}", r0 + m);
        }
    }
}

#[test]
fn a8_rows_are_bitwise_the_solo_kernel() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(94);
    for (n, k) in [(5120usize, 5120usize), (1024, 5120), (301, 2080), (6144, 5120)] {
        for odt in [DType::Bf16, DType::F32] {
            let big = matmul_mxfp4_case(&mut rng, 16, n, k, Mxfp4Act::Fp8, DType::Bf16, odt);
            let ob = odt.size_bytes() * n;
            let batch = run_host(&*hip, "matmul_mxfp4_a8_gemv_rows", &big).unwrap();
            for r in 0..16 {
                let solo = run_host(&*hip, "matmul_mxfp4_a8_gemv", &rows_of(&big, r, 1, k, n)).unwrap();
                assert!(solo[0].data == batch[0].data[r * ob..(r + 1) * ob], "n={n} k={k} {odt:?}: row {r} of the 16-row call differs from the solo kernel");
                for m in [2usize, 5, 11] {
                    if r + m > 16 {
                        continue;
                    }
                    let part = run_host(&*hip, "matmul_mxfp4_a8_gemv_rows", &rows_of(&big, r, m, k, n)).unwrap();
                    assert!(part[0].data == batch[0].data[r * ob..(r + m) * ob], "n={n} k={k} {odt:?}: rows {r}..{} at M={m} differ from M=16", r + m);
                }
            }
            // another summation order, same math
            let mma = run_host(&*hip, "matmul_mxfp4_a8_mma", &big).unwrap();
            if odt == DType::F32 {
                let (a, b) = (batch[0].to_f32(), mma[0].to_f32());
                let bd: Vec<f64> = b.iter().map(|v| *v as f64).collect();
                assert!(rel_l2(&a, &bd) < 1e-5, "n={n} k={k}: decode kernel vs prefill GEMM {:.3e}", rel_l2(&a, &bd));
            }
        }
    }
}

#[test]
fn a8_flipped_weight_code_is_visible() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(95);
    let (m, n, k) = (24usize, 128usize, 1024usize);
    let hc = matmul_mxfp4_case(&mut rng, m, n, k, Mxfp4Act::Fp8, DType::Bf16, DType::F32);
    let mut bad = hc.clone();
    bad.inputs[1].data[7 * (k / 2) + 11] ^= 0x04;
    for kernel in ["matmul_mxfp4_a8_mma", "matmul_mxfp4_a8_gemv_rows"] {
        let (a, b) = if kernel.ends_with("_rows") {
            (rows_of(&hc, 0, 16, k, n), rows_of(&bad, 0, 16, k, n))
        } else {
            (hc.clone(), bad.clone())
        };
        let ga = run_host(&*hip, kernel, &a).unwrap().remove(0).to_f32();
        let gb = run_host(&*hip, kernel, &b).unwrap().remove(0).to_f32();
        let col7 = (0..ga.len() / n).filter(|r| ga[r * n + 7] != gb[r * n + 7]).count();
        let other = ga.iter().zip(&gb).enumerate().filter(|(i, (x, y))| i % n != 7 && x != y).count();
        assert!(col7 > 0 && other == 0, "{kernel}: flipped code changed column 7 in {col7} rows and {other} other outputs");
    }
}

/// Prefill timing of the three MXFP4 GEMMs on the 27B's projection shapes: W4A4 `matmul_mxfp4_mma_fold` (the former default),
/// W4A16 `matmul_mxfp4_mma` (act = none, bf16 WMMA, the current 27B default), W4A8 `matmul_mxfp4_a8_mma`. Prints `TIME` lines.
///   cargo test --release -p intarsia-backend-hip --test mxfp4_a8 -- --ignored --nocapture timing (quiet on the GPU)
#[test]
#[ignore = "micro-timing"]
fn timing_mxfp4_prefill() {
    use intarsia_core::backend::MemKind;
    use intarsia_core::{OpCall, ShapeInfo};
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(96);
    let time = |kernel: &str, hc: &HostCall| -> f64 {
        let put = |t: &HostTensor| {
            let buf = hip.alloc(t.nbytes().max(16), MemKind::Device).unwrap();
            hip.upload(&buf, 0, &t.data, 0).unwrap();
            intarsia_core::backend::TensorRef::new(buf, t.dtype, &t.shape)
        };
        let ins: Vec<_> = hc.inputs.iter().map(&put).collect();
        let outs: Vec<_> = hc.outputs.iter().map(|o| put(&HostTensor::zeros(o.dtype, &o.shape))).collect();
        let mut call = OpCall::new(hc.op, ins, outs, hc.attrs.clone());
        call.check().unwrap();
        let k = hip.kernels().into_iter().find(|k| k.name == kernel).unwrap_or_else(|| panic!("no kernel {kernel}"));
        let need = k.scratch_bytes(&ShapeInfo::from_call(&call));
        let buf = hip.alloc(need.max(16), MemKind::Device).unwrap();
        call.scratch = Some(intarsia_core::backend::TensorRef::new(buf, DType::U8, &[need.max(16)]));
        for _ in 0..5 {
            hip.launch(kernel, &call, 0).unwrap();
        }
        hip.synchronize(0).unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..20 {
            hip.launch(kernel, &call, 0).unwrap();
        }
        hip.synchronize(0).unwrap();
        t0.elapsed().as_secs_f64() * 1e6 / 20.0
    };
    for (n, k) in [(17408usize, 5120usize), (5120, 17408), (10240, 5120), (6144, 5120), (5120, 6144), (12288, 5120)] {
        for m in [256usize, 1024, 4096] {
            let flops = 2.0 * m as f64 * n as f64 * k as f64;
            for (kernel, act) in [("matmul_mxfp4_mma_fold", Mxfp4Act::Dynamic), ("matmul_mxfp4_mma", Mxfp4Act::None), ("matmul_mxfp4_a8_mma", Mxfp4Act::Fp8)] {
                let hc = matmul_mxfp4_case(&mut rng, m, n, k, act, DType::Bf16, DType::Bf16);
                let us = time(kernel, &hc);
                println!("TIME mxfp4 prefill {n}x{k:<6} M={m:<5} {kernel:<24} {us:10.1} us  {:6.1} TFLOPS", flops / us / 1e6);
            }
        }
    }
}
