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
//! 6. `a8_mma_rows_do_not_depend_on_position` (lane GLR): the same activation row at other positions of a call (first, tile edges, last, M tails),
//!    for calls of M on both sides of every tile cutover, gives the same bits as the solo row.
//! 7. `a8_mma_hashes` (ignored; prints `HASH` lines): FNV-1a of the output bytes over shapes with every K and M tail; run once per
//!    `INTARSIA_HIP_MXFP4_A8_PD` value (1 = the previous kernel and tile table; the kernel reads it once per process) and diff the lines: prefetch
//!    distance 1 and 2, and the tile table of each, must agree bitwise.
//!
//! 8. `a8_span_cache_reuses_and_refills` (lane pf27c): calls that reuse one weight's buffers (the backend's span cache: the fold flag read back
//!    after the first call, later calls launch only the selected path's kernel) give the bits of fresh-buffer calls, foldable and wide-span
//!    weights, single and pair; a weight rewritten in place gives the new weight's bits.
//!
//! Needs a GPU: skips without one. INTARSIA_HIP_MXFP4_A8_PERGROUP=1 reruns 1-3 on the per-group path.
#![cfg(ik_hip)]

use intarsia_core::backend::{Backend, MemKind};
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::quant::cases::{matmul_mxfp4_case, matmul_mxfp4_pair_case};
use intarsia_core::quant::reference::matmul_mxfp4_a8;
use intarsia_core::quant::Mxfp4Act;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, OpCall, ShapeInfo, TensorRef};
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
fn a8_mma_rows_do_not_depend_on_position() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(98);
    for (n, k, odt) in [(17408usize, 5120usize, DType::Bf16), (5120, 8704, DType::Bf16), (300, 4128, DType::F32), (200, 4000, DType::Bf16)] {
        let big = matmul_mxfp4_case(&mut rng, 4100, n, k, Mxfp4Act::Fp8, DType::Bf16, odt);
        let ob = odt.size_bytes() * n;
        // row 7 of the big call as the probe: its solo result (M = 1, the small tile) is the reference
        let probe = rows_of(&big, 7, 1, k, n);
        let want = run_host(&*hip, "matmul_mxfp4_a8_mma", &probe).unwrap().remove(0).data;
        let xrow = big.inputs[0].data[7 * k * 2..8 * k * 2].to_vec();
        for m in [17usize, 65, 255, 257, 1000, 2047, 2048, 2049, 4100] {
            for pos in [0usize, 1, 15, 16, 31, 63, 64, 127, 128, 255, 256, 257, 1023, 2047, 2048, m - 1] {
                if pos >= m {
                    continue;
                }
                let mut hc = rows_of(&big, 100usize.min(4100 - m), m, k, n);
                hc.inputs[0].data[pos * k * 2..(pos + 1) * k * 2].copy_from_slice(&xrow);
                let got = run_host(&*hip, "matmul_mxfp4_a8_mma", &hc).unwrap().remove(0).data;
                assert!(got[pos * ob..(pos + 1) * ob] == want[..], "n={n} k={k} {odt:?}: the probe row at position {pos} of an M={m} call differs from its solo result");
            }
        }
    }
}

/// FNV-1a 64 over the output bytes of the W4A8 prefill GEMM on shapes with every K tail (K % 256 = 0, 32, 64, 128, 160, 224, < 256), N not
/// a multiple of the tile, and M on both sides of the tile thresholds. Prints `HASH m n k out hash` lines.
#[test]
#[ignore = "prints hashes for a cross-process diff"]
fn a8_mma_hashes() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(99);
    let fnv = |d: &[u8]| d.iter().fold(0xcbf29ce484222325u64, |h, b| (h ^ *b as u64).wrapping_mul(0x100000001b3));
    for (m, n, k) in [
        (2048usize, 17408usize, 5120usize), (2048, 5120, 8704), (2048, 6144, 5120), (2048, 3072, 5120), (2048, 5120, 3072), (4096, 17408, 5120),
        (1024, 17408, 5120), (1024, 5120, 8704), (512, 6144, 5120), (1536, 5120, 3072), (256, 3072, 5120), (130, 5120, 5120),
        (2049, 8200, 4128), (2100, 777, 4160), (1000, 4100, 4224), (777, 300, 4000), (300, 200, 4064), (65, 100, 96), (257, 129, 224),
        (2300, 1000, 160), (2048, 8192, 1056), (3000, 12288, 5120), (64, 1024, 5120), (1, 640, 5120),
    ] {
        for odt in [DType::Bf16, DType::F32] {
            let hc = matmul_mxfp4_case(&mut rng, m, n, k, Mxfp4Act::Fp8, DType::Bf16, odt);
            let out = run_host(&*hip, "matmul_mxfp4_a8_mma", &hc).unwrap().remove(0);
            println!("HASH {m} {n} {k} {odt:?} {:016x}", fnv(&out.data));
        }
    }
}

/// Lane pf27b: `matmul_mxfp4_pair_a8_mma` (one quant pre-pass for both GEMMs) is bitwise two `matmul_mxfp4_a8_mma` calls, at the 27B tp2
/// pair shapes (GDN qkv | z and b | a, attention k | v, FFN gate | up, the 2048- and 1810-row chunks), both kernels' M ranges (staged
/// M <= 64, direct above), N1 above N0 (the shared wref region sized by the larger), f32 and bf16 out; and the pair with the per-group
/// path (INTARSIA_HIP_MXFP4_A8_PERGROUP=1 reruns this test on it).
#[test]
fn a8_pair_is_bitwise_two_single_calls() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(97);
    for (m, n0, n1, k) in [
        (2048usize, 5120usize, 3072usize, 5120usize), (2048, 24, 24, 5120), (1810, 8704, 8704, 5120), (2048, 512, 512, 5120),
        (1810, 5120, 3072, 5120), (130, 512, 512, 5120), (17, 640, 640, 5120), (64, 1024, 300, 5120), (65, 100, 1000, 512), (300, 777, 2049, 4160),
    ] {
        for odt in [DType::Bf16, DType::F32] {
            let hc = matmul_mxfp4_pair_case(&mut rng, m, n0, n1, k, Mxfp4Act::Fp8, DType::Bf16, odt);
            let pair = run_host(&*hip, "matmul_mxfp4_pair_a8_mma", &hc).unwrap();
            for (j, (wi, n)) in [(1usize, n0), (3, n1)].into_iter().enumerate() {
                let mut one = hc.clone();
                one.op = intarsia_core::Op::MatmulMxfp4;
                one.inputs = vec![hc.inputs[0].clone(), hc.inputs[wi].clone(), hc.inputs[wi + 1].clone(), hc.inputs[5].clone()];
                one.outputs = vec![OutSpec::new(odt, &[m, n])];
                let single = run_host(&*hip, "matmul_mxfp4_a8_mma", &one).unwrap();
                assert!(pair[j].data == single[0].data, "M={m} N0={n0} N1={n1} K={k} {odt:?}: pair output {j} differs from the single call");
            }
        }
    }
}

/// Lane pf27c: the backend's span cache (device.rs `SpanCache`, one span pass per weight instead of per call). Calls on the same device buffers
/// (round 0 fills the entry with both fold paths launched, later rounds run after the flag has been read back and launch only the selected
/// path's kernel) give the bits of `run_host` calls (fresh buffers: a new entry each), for foldable weights and a wide-span one (the per-group
/// path), single and pair; then every input is rewritten in place (upload into the same buffers, the wide-span case flipping to the fold
/// path) and the calls give the new weights' bits.
#[test]
fn a8_span_cache_reuses_and_refills() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(101);
    let put = |t: &HostTensor| {
        let b = hip.alloc(t.nbytes().max(16), MemKind::Device).unwrap();
        hip.upload(&b, 0, &t.data, 0).unwrap();
        TensorRef::new(b, t.dtype, &t.shape)
    };
    let launch = |kernel: &str, hc: &HostCall, ins: &[TensorRef]| -> Vec<Vec<u8>> {
        let outs: Vec<TensorRef> = hc.outputs.iter().map(|o| put(&HostTensor::zeros(o.dtype, &o.shape))).collect();
        let mut call = OpCall::new(hc.op, ins.to_vec(), outs.clone(), hc.attrs.clone());
        let kern = hip.kernels().into_iter().find(|x| x.name == kernel).unwrap();
        let need = kern.scratch_bytes(&ShapeInfo::from_call(&call)).max(16);
        call.scratch = Some(TensorRef::new(hip.alloc(need, MemKind::Device).unwrap(), DType::U8, &[need]));
        hip.launch(kernel, &call, 0).unwrap();
        hip.synchronize(0).unwrap();
        outs.iter()
            .map(|t| {
                let mut v = vec![0u8; t.nbytes()];
                hip.download(&t.buf, t.offset, &mut v, 0).unwrap();
                v
            })
            .collect()
    };
    // row 5 of the scales [N, K/32] at `si`: one group 20 octaves above the others, beyond the fold bound (test 2's construction)
    let widen = |hc: &mut HostCall, si: usize, k: usize| {
        let g = k / 32;
        hc.inputs[si].data[5 * g + 3] = hc.inputs[si].data[5 * g].saturating_add(20).min(250);
    };
    let check = |kernel: &str, a: &HostCall, b: &HostCall, what: &str| {
        let want_a = run_host(&*hip, kernel, a).unwrap();
        let want_b = run_host(&*hip, kernel, b).unwrap();
        let ins: Vec<TensorRef> = a.inputs.iter().map(put).collect();
        for round in 0..3 {
            let got = launch(kernel, a, &ins);
            for (j, w) in want_a.iter().enumerate() {
                assert!(got[j] == w.data, "{kernel} {what}: round {round}, output {j} differs from the fresh-buffer call");
            }
        }
        for (t, h) in ins.iter().zip(&b.inputs) {
            hip.upload(&t.buf, t.offset, &h.data, 0).unwrap();
        }
        for round in 0..2 {
            let got = launch(kernel, b, &ins);
            for (j, w) in want_b.iter().enumerate() {
                assert!(got[j] == w.data, "{kernel} {what}: after the in-place rewrite, round {round}, output {j} differs from the new weight's call");
            }
        }
    };
    for (m, n, k, wide) in [(2048usize, 5120usize, 3072usize, false), (300, 777, 4160, false), (1810, 512, 5120, false), (130, 512, 1024, true)] {
        let mut a = matmul_mxfp4_case(&mut rng, m, n, k, Mxfp4Act::Fp8, DType::Bf16, DType::Bf16);
        let b = matmul_mxfp4_case(&mut rng, m, n, k, Mxfp4Act::Fp8, DType::Bf16, DType::Bf16);
        if wide {
            widen(&mut a, 2, k);
        }
        check("matmul_mxfp4_a8_mma", &a, &b, &format!("M={m} N={n} K={k} wide={wide}"));
    }
    for (m, n0, n1, k, wide) in [(2048usize, 8704usize, 8704usize, 5120usize, false), (2048, 24, 24, 5120, false), (300, 777, 1000, 4160, true)] {
        let mut a = matmul_mxfp4_pair_case(&mut rng, m, n0, n1, k, Mxfp4Act::Fp8, DType::Bf16, DType::Bf16);
        let b = matmul_mxfp4_pair_case(&mut rng, m, n0, n1, k, Mxfp4Act::Fp8, DType::Bf16, DType::Bf16);
        if wide {
            widen(&mut a, 4, k); // the second weight takes the per-group path, the first the fold path
        }
        check("matmul_mxfp4_pair_a8_mma", &a, &b, &format!("M={m} N0={n0} N1={n1} K={k} wide={wide}"));
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
/// Mean microseconds of one launch of `kernel` on `hc` (5 warm-up launches, then `iters`).
fn time_launch(hip: &Arc<dyn Backend>, kernel: &str, hc: &HostCall, iters: usize) -> f64 {
    use intarsia_core::backend::MemKind;
    use intarsia_core::{OpCall, ShapeInfo};
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
    for _ in 0..iters {
        hip.launch(kernel, &call, 0).unwrap();
    }
    hip.synchronize(0).unwrap();
    t0.elapsed().as_secs_f64() * 1e6 / iters as f64
}

/// Prefill timing of the three MXFP4 GEMMs on the 27B's projection shapes: W4A4 `matmul_mxfp4_mma_fold` (the former default),
/// W4A16 `matmul_mxfp4_mma` (act = none, bf16 WMMA, the current 27B default), W4A8 `matmul_mxfp4_a8_mma`. Prints `TIME` lines.
///   cargo test --release -p intarsia-backend-hip --test mxfp4_a8 -- --ignored --nocapture timing (quiet on the GPU)
#[test]
#[ignore = "micro-timing"]
fn timing_mxfp4_prefill() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(96);
    for (n, k) in [(17408usize, 5120usize), (5120, 17408), (10240, 5120), (6144, 5120), (5120, 6144), (12288, 5120)] {
        for m in [256usize, 1024, 4096] {
            let flops = 2.0 * m as f64 * n as f64 * k as f64;
            for (kernel, act) in [("matmul_mxfp4_mma_fold", Mxfp4Act::Dynamic), ("matmul_mxfp4_mma", Mxfp4Act::None), ("matmul_mxfp4_a8_mma", Mxfp4Act::Fp8)] {
                let hc = matmul_mxfp4_case(&mut rng, m, n, k, act, DType::Bf16, DType::Bf16);
                let us = time_launch(&hip, kernel, &hc, 20);
                println!("TIME mxfp4 prefill {n}x{k:<6} M={m:<5} {kernel:<24} {us:10.1} us  {:6.1} TFLOPS", flops / us / 1e6);
            }
        }
    }
}

/// `matmul_mxfp4_a8_mma` only, at the Qwen3.8-27B tp2 per-rank projection shapes (gate+up pair, down, q+gate, z, o) and the prefill chunk
/// rows 2048 / 4096, 100 launches each, bf16 out. Prints `TIME a8tp2` lines (microseconds include the activation pre-pass and span check).
/// Env INTARSIA_HIP_MXFP4_A8_* of the kernel apply, so one run per knob value sweeps a variant.
#[test]
#[ignore = "micro-timing"]
fn timing_a8_tp2() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(97);
    let ms: Vec<usize> = std::env::var("GLR_TIMING_M").ok().map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_else(|| vec![2048, 4096]);
    // GLR_TIMING_ONLY=q+gate,o restricts the shapes, GLR_TIMING_ITERS the launches per case (profiler passes)
    let only: Vec<String> = std::env::var("GLR_TIMING_ONLY").ok().map(|v| v.split(',').map(str::to_string).collect()).unwrap_or_default();
    let iters: usize = std::env::var("GLR_TIMING_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(100);
    let mut shapes = vec![("gate+up", 17408usize, 5120usize), ("down", 5120, 8704), ("q+gate", 6144, 5120), ("z", 3072, 5120), ("o", 5120, 3072)];
    // PF27B_SMALL=1 (lane pf27b): the small-N projections instead (attention k / v, GDN b / a), whose grids under-fill the card
    if std::env::var("PF27B_SMALL").is_ok_and(|v| v == "1") {
        shapes = vec![("k/v", 512, 5120), ("b/a", 24, 5120), ("gate", 8704, 5120), ("qkv", 5120, 5120)];
    }
    for (name, n, k) in shapes {
        if !only.is_empty() && !only.iter().any(|o| o == name) {
            continue;
        }
        for &m in &ms {
            let flops = 2.0 * m as f64 * n as f64 * k as f64;
            let hc = matmul_mxfp4_case(&mut rng, m, n, k, Mxfp4Act::Fp8, DType::Bf16, DType::Bf16);
            let us = time_launch(&hip, "matmul_mxfp4_a8_mma", &hc, iters);
            println!("TIME a8tp2 {name:<8} {n}x{k:<5} M={m:<5} {us:10.1} us  {:6.1} TFLOPS", flops / us / 1e6);
        }
    }
}
