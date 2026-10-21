//! cuda `matmul_exl3_gemv` (kernels/cuda/exl3.cu): accuracy against the cpu reference over the op's sweep, what the fp16
//! rounding of the rotated input accounts for, batch independence of a row's bits, a bad codebook, and an ignored decode
//! microbenchmark (`INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-cuda --test exl3 -- --ignored
//! --nocapture bench_decode_shapes`).
//! Needs a GPU: without one the tests print SKIPPED (and fail under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_cuda)]

mod common;

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::admission::{judge, sweep, SweepConfig};
use intarsia_core::backend::{upload_new, Backend, MemKind, TensorRef};
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::ops::OpAttrs;
use intarsia_core::quant::cases::matmul_nvfp4_case;
use intarsia_core::quant::exl3::{self, Codebook, HAD};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, Op, OpCall, ShapeInfo};
use std::sync::Arc;
use std::time::Instant;

const KERNEL: &str = "matmul_exl3_gemv";

fn device() -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_cuda::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            common::skip("exl3", &format!("no usable CUDA device ({e})"));
            None
        }
    }
}

/// (rel_l2, max_abs scaled by max(1, max|ref|)), the admission harness's measures.
fn measure(got: &[f32], want: &[f32]) -> (f64, f64) {
    let (mut num, mut den, mut ma, mut rmax) = (0f64, 0f64, 0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        let d = (*g as f64 - *w as f64).abs();
        num += d * d;
        den += (*w as f64) * (*w as f64);
        ma = ma.max(d);
        rmax = rmax.max((*w as f64).abs());
    }
    ((num / den.max(1e-30)).sqrt(), ma / rmax.max(1.0))
}

/// Every case of the op's sweep (quick and full) against `ref_matmul_exl3`, judged by the kernel's own tolerance. Prints
/// the worst rel_l2 / max_abs per output dtype (the numbers the tolerance constants in `exl3.rs` are set from).
#[test]
fn matches_the_cpu_reference_over_the_sweep() {
    let Some(cuda) = device() else { return };
    let cpu = CpuBackend::new();
    let kern = cuda.kernels().into_iter().find(|k| k.name == KERNEL).unwrap();
    let mut worst: std::collections::BTreeMap<String, (f64, f64, String)> = Default::default();
    let mut n = 0;
    let mut failed = Vec::new();
    for quick in [true, false] {
        for case in sweep(Op::MatmulExl3, &SweepConfig { quick, ..SweepConfig::default() }) {
            let want = run_host(&cpu, "ref_matmul_exl3", &case.call).unwrap();
            let got = run_host(&*cuda, KERNEL, &case.call).unwrap();
            let (ok, ma, rl) = judge(&kern, &got, &want);
            let key = format!("{}", got[0].dtype);
            let w = worst.entry(key).or_insert((0.0, 0.0, String::new()));
            if rl > w.0 {
                *w = (rl, w.1, case.label.clone());
            }
            w.1 = w.1.max(ma);
            if !ok {
                failed.push(format!("{}: max_abs {ma:.3e} rel_l2 {rl:.3e}", case.label));
            }
            n += 1;
        }
    }
    for (dt, (rl, ma, label)) in &worst {
        println!("worst {dt} output: rel_l2 {rl:.3e} (at {label}), max_abs {ma:.3e}");
    }
    println!("{n} cases, {} outside the kernel's tolerance", failed.len());
    assert!(failed.is_empty(), "outside the kernel's tolerance:\n{}", failed.join("\n"));
}

/// The only gap between the kernel and the f64 reference is meant to be the fp16 rounding of the rotated input (the mma's
/// operand): against a reference that rounds `H (x * suh)` to fp16 the residual is accumulation order only.
#[test]
fn only_the_input_rounding_separates_it_from_the_reference() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(404);
    println!("{:<28} {:>14} {:>14}", "case", "vs f64 ref", "vs f16-xh ref");
    for (k2, cb) in [(8usize, Codebook::Inst3), (6, Codebook::Mcg), (5, Codebook::Mul1), (4, Codebook::Inst3), (16, Codebook::Mul1)] {
        let (m, n, k) = (3usize, 384usize, 512usize);
        let hc = exl3::case(&mut rng, m, n, k, k2, cb, DType::F32, DType::F32);
        let got = run_host(&*cuda, KERNEL, &hc).unwrap()[0].to_f32();
        let x = hc.inputs[0].to_f32();
        let (suh, svh) = (exl3::f16_bytes(&hc.inputs[2].data), exl3::f16_bytes(&hc.inputs[3].data));
        let xh = exl3::rotate_input(&x, m, k, &suh);
        // the same rounding the kernel applies (round to nearest even), through the host fp16 conversion
        let xr: Vec<f64> = HostTensor::from_f32_as(DType::F16, &[xh.len()], &xh.iter().map(|v| *v as f32).collect::<Vec<_>>())
            .to_f32()
            .into_iter()
            .map(f64::from)
            .collect();
        let run = |xh: &[f64]| -> Vec<f32> {
            let mut y = vec![0f32; m * n];
            for nb in 0..n / HAD {
                let blk = exl3::forward_block(xh, m, k, n, &hc.inputs[1].data, &svh, nb, k2, cb);
                for mi in 0..m {
                    for j in 0..HAD {
                        y[mi * n + nb * HAD + j] = blk[mi * HAD + j] as f32;
                    }
                }
            }
            y
        };
        let (exact, rounded) = (run(&xh), run(&xr));
        let (a, _) = measure(&got, &exact);
        let (b, _) = measure(&got, &rounded);
        println!("{:<28} {:>14.3e} {:>14.3e}", format!("bits={} cb={}", k2 as f64 / 2.0, cb.name()), a, b);
        assert!(b < a / 2.0, "k2={k2} {}: the fp16 rounding should dominate the gap to the f64 reference ({a:e} vs {b:e})", cb.name());
    }
}

/// A row's bits do not depend on the batch (the block's K split is a function of (K, N) only, and the launcher's 128-row
/// slices change nothing): in a 300-row call (slices of 128, 128 and 44 rows) row 0 is bit for bit the 1-row call, rows
/// 8..13 the 5-row call, rows 127..129 a call across the slice edge, and so on.
#[test]
fn rows_do_not_depend_on_the_batch() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(505);
    for (k2, cb, n, k) in [(8usize, Codebook::Inst3, 512usize, 1024usize), (5, Codebook::Mul1, 256, 384), (4, Codebook::Mcg, 1024, 128)] {
        let full = exl3::case(&mut rng, 300, n, k, k2, cb, DType::Bf16, DType::Bf16);
        let all = run_host(&*cuda, KERNEL, &full).unwrap();
        let rows = |hc: &HostCall, r0: usize, r1: usize| -> HostCall {
            let mut h = hc.clone();
            let t = &hc.inputs[0];
            h.inputs[0] = HostTensor { shape: vec![r1 - r0, k], data: t.data[r0 * k * 2..r1 * k * 2].to_vec(), ..t.clone() };
            h.outputs = vec![OutSpec::new(DType::Bf16, &[r1 - r0, n])];
            h
        };
        for (r0, r1) in [(0usize, 1usize), (8, 13), (16, 17), (0, 16), (127, 129), (128, 300), (256, 300)] {
            let part = run_host(&*cuda, KERNEL, &rows(&full, r0, r1)).unwrap();
            assert_eq!(part[0].data, all[0].data[r0 * n * 2..r1 * n * 2], "k2={k2} rows {r0}..{r1}");
        }
        let again = run_host(&*cuda, KERNEL, &full).unwrap();
        assert_eq!(again[0].data, all[0].data, "not repeatable");
    }
}

/// f16 activations and outputs (not in the sweep).
#[test]
fn f16_activations_and_outputs() {
    let Some(cuda) = device() else { return };
    let cpu = CpuBackend::new();
    let kern = cuda.kernels().into_iter().find(|k| k.name == KERNEL).unwrap();
    let mut rng = SplitMix64::new(606);
    for (xdt, odt) in [(DType::F16, DType::F16), (DType::F16, DType::F32), (DType::Bf16, DType::F16), (DType::F32, DType::Bf16)] {
        let hc = exl3::case(&mut rng, 5, 256, 384, 6, Codebook::Mcg, xdt, odt);
        let want = run_host(&cpu, "ref_matmul_exl3", &hc).unwrap();
        let got = run_host(&*cuda, KERNEL, &hc).unwrap();
        let (ok, ma, rl) = judge(&kern, &got, &want);
        println!("x={xdt} out={odt}: max_abs {ma:.3e} rel_l2 {rl:.3e}");
        assert!(ok, "x={xdt} out={odt}: max_abs {ma:.3e} rel_l2 {rl:.3e}");
    }
}

/// A codebook value the kernel cannot decode (outside 0..2, or mcg / 3inst at a half-integer width) gives NaN, not
/// uninitialised memory; the launch itself succeeds.
#[test]
fn a_bad_codebook_gives_nan() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(707);
    for (k2, cbv) in [(8usize, 7i32), (8, -1), (5, 0), (5, 1)] {
        let mut hc = exl3::case(&mut rng, 3, 256, 256, k2, Codebook::Mul1, DType::Bf16, DType::F32);
        hc.inputs[4] = HostTensor { data: cbv.to_le_bytes().to_vec(), ..hc.inputs[4].clone() };
        let got = run_host(&*cuda, KERNEL, &hc).unwrap();
        assert!(got[0].to_f32().iter().all(|v| v.is_nan()), "k2={k2} codebook {cbv}");
    }
}

fn dev_tensor(b: &dyn Backend, t: &HostTensor) -> TensorRef {
    TensorRef::new(upload_new(b, &t.data).unwrap(), t.dtype, &t.shape)
}

/// Cold-weight time of one launch of `kernel` on copies of the weight parts `weights` (input indices `wi` of `hc`):
/// `copies` calls, each on its own device copy of the weights, are captured in one CUDA graph (so the host launch cost
/// does not count and consecutive calls miss the L2) and replayed `reps` times; the result is the median replay time per
/// call in seconds. (The Backend trait has no events; a replay is timed host-side between two synchronisations.)
fn time_cold(b: &dyn Backend, kernel: &str, hc: &HostCall, wi: &[usize], copies: usize, reps: usize) -> f64 {
    let shared: Vec<TensorRef> = hc.inputs.iter().map(|t| dev_tensor(b, t)).collect();
    let out = dev_tensor(b, &HostTensor::zeros(hc.outputs[0].dtype, &hc.outputs[0].shape));
    let mut calls = Vec::new();
    for _ in 0..copies {
        let mut inputs = shared.clone();
        for &i in wi {
            inputs[i] = dev_tensor(b, &hc.inputs[i]);
        }
        calls.push(OpCall::new(hc.op, inputs, vec![out.clone()], hc.attrs.clone()));
    }
    let need = b.kernels().iter().find(|k| k.name == kernel).map_or(0, |k| k.scratch_bytes(&ShapeInfo::from_call(&calls[0])));
    if need > 0 {
        let s = TensorRef::new(b.alloc(need.max(16), MemKind::Device).unwrap(), DType::U8, &[need]);
        for c in &mut calls {
            c.scratch = Some(s.clone());
        }
    }
    let g = b
        .graph_capture(0, &mut || {
            for c in &calls {
                b.launch(kernel, c, 0)?;
            }
            Ok(())
        })
        .unwrap();
    for _ in 0..3 {
        b.graph_replay(&g, 0).unwrap();
    }
    b.synchronize(0).unwrap();
    let mut t: Vec<f64> = (0..reps)
        .map(|_| {
            let t0 = Instant::now();
            b.graph_replay(&g, 0).unwrap();
            b.synchronize(0).unwrap();
            t0.elapsed().as_secs_f64() / copies as f64
        })
        .collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t[t.len() / 2]
}

/// Copies of a weight of `bytes` so that the graph reads at least 192 MB (more than any cache), at least 3, at most 48.
fn copies_for(bytes: usize) -> usize {
    (192usize << 20).div_ceil(bytes.max(1)).clamp(3, 48)
}

fn bf16_matmul_case(rng: &mut SplitMix64, m: usize, n: usize, k: usize) -> HostCall {
    HostCall {
        op: Op::Matmul,
        inputs: vec![
            HostTensor::from_f32_as(DType::Bf16, &[m, k], &rng.normal_vec(m * k, 1.0)),
            HostTensor::from_f32_as(DType::Bf16, &[n, k], &rng.normal_vec(n * k, 0.02)),
        ],
        outputs: vec![OutSpec::new(DType::Bf16, &[m, n])],
        attrs: OpAttrs::default(),
    }
}

/// Decode microbenchmark at the Qwen3-8B projection shapes (M = 1) plus widths / codebooks / rows at 4096 x 4096, with the
/// bf16 GEMV and the NVFP4 GEMV on the same shapes. Prints us per call and the weight bytes read per second.
#[test]
#[ignore]
fn bench_decode_shapes() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(808);
    println!("{:<44} {:>9} {:>10} {:>9}", "case (M=1 unless noted)", "us/call", "weight MB", "GB/s");
    let row = |label: &str, us: f64, bytes: usize| {
        println!("{:<44} {:>9.1} {:>10.2} {:>9.0}", label, us * 1e6, bytes as f64 / 1e6, bytes as f64 / 1e9 / us);
    };
    let mut cases: Vec<(usize, usize, usize, usize, Codebook)> = Vec::new(); // (m, n, k, k2, cb)
    for &(n, k) in &[(4096usize, 4096usize), (1024, 4096), (12288, 4096), (4096, 12288)] {
        cases.push((1, n, k, 8, Codebook::Inst3));
    }
    for k2 in [4usize, 5, 6, 10, 16] {
        cases.push((1, 4096, 4096, k2, if k2 == 5 { Codebook::Mul1 } else { Codebook::Inst3 }));
    }
    cases.push((1, 4096, 4096, 8, Codebook::Mcg));
    cases.push((1, 4096, 4096, 8, Codebook::Mul1));
    cases.push((1, 4096, 4096, 7, Codebook::Mul1));
    for m in [2usize, 4, 8, 16, 32, 128, 256, 300] {
        cases.push((m, 4096, 4096, 8, Codebook::Inst3));
    }
    for (m, n, k, k2, cb) in cases {
        let hc = exl3::case(&mut rng, m, n, k, k2, cb, DType::Bf16, DType::Bf16);
        let bytes = hc.inputs[1].data.len();
        let us = time_cold(&*cuda, KERNEL, &hc, &[1], copies_for(bytes), 15);
        row(&format!("exl3 {n}x{k} bits={} {} M={m}", k2 as f64 / 2.0, cb.name()), us, bytes);
    }
    println!();
    for &(n, k) in &[(4096usize, 4096usize), (1024, 4096), (12288, 4096), (4096, 12288)] {
        let hc = bf16_matmul_case(&mut rng, 1, n, k);
        let bytes = hc.inputs[1].data.len();
        let us = time_cold(&*cuda, "matmul_bf16_gemv", &hc, &[1], copies_for(bytes), 15);
        row(&format!("bf16 matmul_bf16_gemv {n}x{k}"), us, bytes);
        let hc = matmul_nvfp4_case(&mut rng, 1, n, k, DType::Bf16, DType::Bf16);
        let bytes = hc.inputs[1].data.len() + hc.inputs[2].data.len();
        let us = time_cold(&*cuda, "matmul_nvfp4_gemv", &hc, &[1, 2], copies_for(bytes), 15);
        row(&format!("nvfp4 matmul_nvfp4_gemv {n}x{k}"), us, bytes);
    }
}
