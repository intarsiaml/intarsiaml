//! hip quantised-weight / MoE kernels: edge cases beyond the admission sweep (skewed and invalid routing,
//! multi-chunk experts) and an ignored microbenchmark on the real Qwen3.6-35B-A3B shapes
//! (`cargo test --release -p intarsia-backend-hip --test quantmoe -- --ignored --nocapture`).
//! Needs a GPU: skips (passes) without one.
#![cfg(ik_hip)]

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::backend::{upload_new, Backend, MemKind, TensorRef};
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::quant::cases::{experts_case, matmul_nvfp4_case};
use intarsia_core::quant::ops::moe_experts_scratch;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, Op, OpCall, ShapeInfo};
use std::sync::Arc;
use std::time::Instant;

fn device() -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_HIP_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_hip::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping: no usable CUDA device ({e})");
            None
        }
    }
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

fn check_experts(hip: &dyn Backend, hc: &HostCall, label: &str) {
    let cpu = CpuBackend::new();
    let want = run_host(&cpu, "ref_moe_experts", hc).unwrap();
    let got = run_host(hip, "moe_experts_bf16", hc).unwrap();
    let r = rel_l2(&got[0].to_f32(), &want[0].to_f32());
    assert!(r < 1e-3, "{label}: rel_l2 {r}");
    let again = run_host(hip, "moe_experts_bf16", hc).unwrap();
    assert_eq!(got[0].data, again[0].data, "{label}: not repeatable");
}

#[test]
fn moe_experts_skewed_and_invalid_routing() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(11);
    let (t, e, k, h, i) = (40usize, 16usize, 4usize, 256usize, 64usize);
    let base = experts_case(&mut rng, t, e, k, h, i, DType::Bf16);
    // every token picks experts 0..k: 40 pairs per expert, five 8-pair chunks each
    let mut hot = base.clone();
    let ids: Vec<i32> = (0..t).flat_map(|_| (0..k as i32)).collect();
    hot.inputs[1] = HostTensor::from_i32(&[t, k], &ids);
    check_experts(&*hip, &hot, "hot experts");
    // prefill-sized and skewed: 1024 pairs on each of four experts is 32 pair tiles, the launch grid (INTARSIA_HIP_MOE_Z, default 2 x the
    // average of 256 pairs per expert) holds 16, so each block steps through a second tile
    let mut big = experts_case(&mut rng, 1024, e, k, h, i, DType::Bf16);
    let ids: Vec<i32> = (0..1024).flat_map(|_| (0..k as i32)).collect();
    big.inputs[1] = HostTensor::from_i32(&[1024, k], &ids);
    check_experts(&*hip, &big, "hot experts, more pair tiles than the grid");
    // one expert receives every pair of token 0 (duplicate ids are legal input) and some ids are out of range
    let mut odd = base.clone();
    let mut ids = odd.inputs[1].to_i32().unwrap();
    for j in 0..k {
        ids[j] = 3;
    }
    ids[k] = -1;
    ids[k + 1] = e as i32 + 5;
    odd.inputs[1] = HostTensor::from_i32(&[t, k], &ids);
    check_experts(&*hip, &odd, "duplicate and invalid ids");
    // a single token
    let one = experts_case(&mut rng, 1, 64, 8, 2048, 512, DType::Bf16);
    check_experts(&*hip, &one, "T=1 real shape");
}

fn cpu_ref(cpu: &CpuBackend, op: Op, hc: &HostCall) -> Vec<HostTensor> {
    let name = cpu.kernels().into_iter().find(|k| k.op == op && k.is_reference).unwrap().name;
    run_host(cpu, &name, hc).unwrap()
}

/// (max_abs scaled by max(|ref|max, 1), rel_l2): the measures `intarsia_core::admission::judge` uses.
fn errors(got: &[f32], want: &[f32]) -> (f64, f64) {
    let ref_max = want.iter().fold(0f64, |m, y| m.max(y.abs() as f64));
    let max_abs = got.iter().zip(want).fold(0f64, |m, (x, y)| m.max((x - y).abs() as f64));
    (max_abs / ref_max.max(1.0), rel_l2(got, want))
}

/// f32 outputs (the lm_head logits) are graded at the f32 budget (`F32_OUT_TOL`, the dense `logits_bf16` one). The
/// registration's shared `TOL` (1.5e-2 / 1e-3, was 1.5e-2 / 6e-3) is looser; a scale-indexing or K-tail bug that moves
/// logits by ~1e-3 would pass it. Shapes include K and N tails.
#[test]
fn quant_f32_outputs_meet_the_f32_tolerance() {
    let Some(hip) = device() else { return };
    let cpu = CpuBackend::new();
    let mut rng = SplitMix64::new(81);
    let tol = intarsia_backend_hip::quantmoe::F32_OUT_TOL;
    let (f32_max_abs, f32_rel_l2) = (tol.max_abs as f64, tol.rel_l2 as f64);
    let shapes = [(1usize, 1000usize, 2048usize), (1, 4099, 2064), (3, 517, 48), (8, 1031, 4112), (9, 130, 2096), (33, 257, 80), (256, 1001, 2064)];
    let mut worst = (0f64, 0f64);
    let mut ran = 0;
    for &(m, n, k) in &shapes {
        for op in [Op::MatmulNvfp4, Op::MatmulFp8] {
            let hc = if op == Op::MatmulNvfp4 {
                matmul_nvfp4_case(&mut rng, m, n, k, DType::Bf16, DType::F32)
            } else {
                intarsia_core::quant::cases::matmul_fp8_case(&mut rng, m, n, k, true, DType::Bf16, DType::F32)
            };
            let want = cpu_ref(&cpu, op, &hc)[0].to_f32();
            let mut shape = ShapeInfo::default();
            shape.m = m;
            shape.n = n;
            shape.k = k;
            shape.act = Some(DType::Bf16);
            shape.weight = Some(if op == Op::MatmulNvfp4 { DType::U8 } else { DType::Fp8E4m3 });
            for kern in hip.kernels().into_iter().filter(|c| c.op == op && c.accepts(&shape)) {
                let got = run_host(&*hip, &kern.name, &hc).unwrap()[0].to_f32();
                let (ma, rl) = errors(&got, &want);
                worst = (worst.0.max(ma), worst.1.max(rl));
                assert!(
                    got.iter().all(|x| x.is_finite()) && ma <= f32_max_abs && rl <= f32_rel_l2,
                    "{} f32 out M={m} N={n} K={k}: max_abs {ma:.3e} rel_l2 {rl:.3e}",
                    kern.name
                );
                ran += 1;
            }
        }
    }
    assert!(ran >= shapes.len() * 2, "only {ran} kernel runs");
    println!("f32-output quant matmuls: {ran} runs, worst max_abs {:.3e}, worst rel_l2 {:.3e}", worst.0, worst.1);
}

fn token_rows(t: &HostTensor, a: usize, b: usize) -> HostTensor {
    let row = t.nbytes() / t.shape[0];
    let mut shape = t.shape.clone();
    shape[0] = b - a;
    HostTensor::new(t.dtype, shape, t.data[a * row..b * row].to_vec()).unwrap()
}

/// `moe_experts` and batch size. A token's result must not depend on how many other tokens are in the call, except
/// where documented: with P = T*K <= 16 (decode) and T < 16 (grouped GEMV) the accumulation order is the one a lone
/// token gets, so those are bit-identical to T = 1; from T = 16 the tensor-core path accumulates in another order
/// (docs/backends/hip.md "MoE and batch size") and only matches within tolerance, unless `INTARSIA_HIP_MOE_WMMA=0`
/// selects the batch-invariant mode, in which case every T is bit-identical to T = 1. The knob is read once per
/// process, so the default-mode run re-executes this test with the knob off.
#[test]
fn moe_experts_batch_invariance() {
    let Some(hip) = device() else { return };
    let strict = std::env::var("INTARSIA_HIP_MOE_WMMA").map(|v| v == "0").unwrap_or(false);
    let mut rng = SplitMix64::new(33);
    let (e, k, h, i) = (32usize, 4usize, 512usize, 256usize);
    let mut differing_tokens = 0usize;
    for t in [2usize, 4, 7, 15, 16, 17, 48, 130] {
        let hc = experts_case(&mut rng, t, e, k, h, i, DType::Bf16);
        let batch = run_host(&*hip, "moe_experts_bf16", &hc).unwrap()[0].clone();
        let row = batch.nbytes() / t;
        let mut diff = 0usize;
        let (mut all_b, mut all_s) = (Vec::new(), Vec::new());
        for tok in 0..t {
            let mut one = hc.clone();
            for idx in [0usize, 1, 2] {
                one.inputs[idx] = token_rows(&hc.inputs[idx], tok, tok + 1);
            }
            one.outputs = vec![OutSpec::new(DType::Bf16, &[1, h])];
            let single = run_host(&*hip, "moe_experts_bf16", &one).unwrap()[0].clone();
            let b = &batch.data[tok * row..(tok + 1) * row];
            if b != &single.data[..] {
                diff += 1;
            }
            all_b.extend(HostTensor::new(DType::Bf16, vec![1, h], b.to_vec()).unwrap().to_f32());
            all_s.extend(single.to_f32());
        }
        let must_match = strict || t < 16;
        if must_match {
            assert_eq!(diff, 0, "T={t}: {diff} of {t} tokens differ from their T=1 result (strict={strict})");
        } else {
            let r = rel_l2(&all_b, &all_s);
            assert!(r < 1e-3, "T={t}: batched vs T=1 rel_l2 {r}");
            differing_tokens += diff;
        }
    }
    println!("batch invariance: strict={strict}, tokens differing from T=1 in the tensor-core path: {differing_tokens}");
    if !strict && std::env::var_os("INTARSIA_TEST_CHILD").is_none() {
        let st = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "moe_experts_batch_invariance", "--nocapture"])
            .env("INTARSIA_HIP_MOE_WMMA", "0")
            .env("INTARSIA_TEST_CHILD", "1")
            .status()
            .unwrap();
        assert!(st.success(), "batch-invariant mode (INTARSIA_HIP_MOE_WMMA=0) is not bit-identical to T=1");
    }
}

fn dev_tensor(b: &dyn Backend, t: &HostTensor) -> TensorRef {
    TensorRef::new(upload_new(b, &t.data).unwrap(), t.dtype, &t.shape)
}

/// Bytes of weight data that must lie between two reads of the same byte, so the 64 MB Infinity Cache cannot serve
/// it (matmul_bench.rs rotates over the same amount).
const COLD_BYTES: usize = 768 * 1024 * 1024;

fn new_call(b: &dyn Backend, kernel: &str, hc: &HostCall, inputs: Vec<TensorRef>) -> OpCall {
    let outputs: Vec<TensorRef> =
        hc.outputs.iter().map(|o| dev_tensor(b, &HostTensor::zeros(o.dtype, &o.shape))).collect();
    let mut call = OpCall::new(hc.op, inputs, outputs, hc.attrs.clone());
    let need = b.kernels().iter().find(|k| k.name == kernel).map_or(0, |k| k.scratch_bytes(&ShapeInfo::from_call(&call)));
    if need > 0 {
        call.scratch = Some(TensorRef::new(b.alloc(need, MemKind::Device).unwrap(), DType::U8, &[need]));
    }
    call
}

/// Seconds per launch with the same buffers every time (cache-hot for anything below ~64 MB): the old method.
fn bench_hot(b: &dyn Backend, kernel: &str, hc: &HostCall, reps: usize) -> f64 {
    let inputs: Vec<TensorRef> = hc.inputs.iter().map(|t| dev_tensor(b, t)).collect();
    let call = new_call(b, kernel, hc, inputs);
    for _ in 0..10 {
        b.launch(kernel, &call, 0).unwrap();
    }
    b.synchronize(0).unwrap();
    let t0 = Instant::now();
    for _ in 0..reps {
        b.launch(kernel, &call, 0).unwrap();
    }
    b.synchronize(0).unwrap();
    t0.elapsed().as_secs_f64() / reps as f64
}

/// Seconds per launch with DRAM-cold weights: every input except input 0 (the activations) and `ids_idx` (the
/// routing) exists in enough distinct device copies (>= COLD_BYTES in total) that successive launches rotate over
/// them, so a launch never finds its weights in the cache. For `moe_experts` (`ids_idx`) the routing also advances
/// each launch to the next block of experts, so T*K touched experts never repeat within `E / (T*K)` launches
/// (>= COLD_BYTES of weights travelled between two reads of an expert). Returns (seconds, copies, MB per copy).
fn bench_cold(b: &dyn Backend, kernel: &str, hc: &HostCall, reps: usize, ids_idx: Option<usize>) -> (f64, usize, f64) {
    let wbytes: usize =
        hc.inputs.iter().enumerate().filter(|(i, _)| *i != 0 && Some(*i) != ids_idx).map(|(_, t)| t.nbytes()).sum();
    let copies = COLD_BYTES.div_ceil(wbytes.max(1)).max(2);
    let x = dev_tensor(b, &hc.inputs[0]);
    // routing variants (moe): launch r uses experts (r*P + j) % E for pair j
    let id_sets: Vec<TensorRef> = match ids_idx {
        Some(ii) => {
            let (t, k) = (hc.inputs[ii].shape[0], hc.inputs[ii].shape[1]);
            let e = hc.inputs[3].shape[0];
            let p = t * k;
            let period = e / gcd(e, p % e.max(1)).max(1);
            (0..period.max(1))
                .map(|r| {
                    let ids: Vec<i32> = (0..t * k).map(|j| ((r * p + j) % e) as i32).collect();
                    dev_tensor(b, &HostTensor::from_i32(&[t, k], &ids))
                })
                .collect()
        }
        None => Vec::new(),
    };
    let weights: Vec<Vec<TensorRef>> = (0..copies)
        .map(|_| {
            hc.inputs
                .iter()
                .enumerate()
                .map(|(i, t)| if i == 0 { x.clone() } else { dev_tensor(b, t) })
                .collect()
        })
        .collect();
    // one prepared call per (weight copy, routing variant) combination, so the timed loop only launches
    let period = copies * id_sets.len().max(1) / gcd(copies, id_sets.len().max(1));
    let calls: Vec<OpCall> = (0..period)
        .map(|r| {
            let mut inputs = weights[r % copies].clone();
            if let Some(ii) = ids_idx {
                inputs[ii] = id_sets[r % id_sets.len()].clone();
            }
            new_call(b, kernel, hc, inputs)
        })
        .collect();
    let reps = reps.max(period * 2);
    for r in 0..period.min(8) {
        b.launch(kernel, &calls[r], 0).unwrap();
    }
    b.synchronize(0).unwrap();
    let t0 = Instant::now();
    for r in 0..reps {
        b.launch(kernel, &calls[r % period], 0).unwrap();
    }
    b.synchronize(0).unwrap();
    (t0.elapsed().as_secs_f64() / reps as f64, copies, wbytes as f64 / 1e6)
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

#[test]
#[ignore]
fn bench_real_shapes() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(5);
    println!("hot = same buffers every launch (cache-assisted); cold = weights rotate over >= {} MB of distinct copies (DRAM)", COLD_BYTES >> 20);
    println!("{:<34} {:>9} {:>9} {:>11} {:>9} {:>9} {:>7}", "case", "hot us", "cold us", "weight MB", "hot GB/s", "cold GB/s", "cold %627");
    let row = |name: &str, hot: f64, cold: f64, mb: f64| {
        println!(
            "{:<34} {:>9.1} {:>9.1} {:>11.2} {:>9.0} {:>9.0} {:>6.0}%",
            name, hot * 1e6, cold * 1e6, mb, mb / 1e3 / hot, mb / 1e3 / cold, 100.0 * mb / 1e3 / cold / 627.0
        );
    };
    // NVFP4 bytes: 0.5 B/weight + 1/16 B scale
    let nv = |n: usize, k: usize| (n * k) as f64 * (0.5 + 1.0 / 16.0) / 1e6;
    for (name, m, n, k, out) in [
        ("lm_head M=1 (248320x2048)", 1usize, 248_320usize, 2048usize, DType::F32),
        ("shared gate/up M=1 (512x2048)", 1, 512, 2048, DType::Bf16),
        ("shared down M=1 (2048x512)", 1, 2048, 512, DType::Bf16),
        ("shared gate M=4", 4, 512, 2048, DType::Bf16),
        ("lm_head prefill M=256 mma", 256, 248_320, 2048, DType::F32),
    ] {
        let hc = matmul_nvfp4_case(&mut rng, m, n, k, DType::Bf16, out);
        let kern = if m <= 8 { "matmul_nvfp4_gemv" } else { "matmul_nvfp4_mma" };
        let reps = if m > 8 { 5 } else { 50 };
        let hot = bench_hot(&*hip, kern, &hc, reps);
        let (cold, copies, _) = bench_cold(&*hip, kern, &hc, reps, None);
        row(&format!("{name} x{copies}"), hot, cold, nv(n, k));
    }
    for (name, m, n, k) in [
        ("fp8 qkv M=1 (8192x2048)", 1usize, 8192usize, 2048usize),
        ("fp8 z M=1 (4096x2048)", 1, 4096, 2048),
        ("fp8 o_proj M=1 (2048x4096)", 1, 2048, 4096),
        ("fp8 k/v_proj M=1 (512x2048)", 1, 512, 2048),
        ("fp8 qkv M=256 mma", 256, 8192, 2048),
    ] {
        let hc = intarsia_core::quant::cases::matmul_fp8_case(&mut rng, m, n, k, true, DType::Bf16, DType::Bf16);
        let kern = if m <= 4 { "matmul_fp8_gemv" } else { "matmul_fp8_mma" };
        let reps = if m > 4 { 5 } else { 50 };
        let hot = bench_hot(&*hip, kern, &hc, reps);
        let (cold, copies, _) = bench_cold(&*hip, kern, &hc, reps, None);
        row(&format!("{name} x{copies}"), hot, cold, (n * k) as f64 / 1e6);
    }
    for (name, t, e) in [("moe_experts T=1 (8 of 256)", 1usize, 256usize), ("moe_experts T=4", 4, 256), ("moe_experts T=16", 16, 256), ("moe_experts T=64", 64, 256), ("moe_experts T=256", 256, 256)] {
        let hc = experts_case(&mut rng, t, e, 8, 2048, 512, DType::Bf16);
        let hot = bench_hot(&*hip, "moe_experts_bf16", &hc, 20);
        let (cold, copies, _) = bench_cold(&*hip, "moe_experts_bf16", &hc, 20, Some(1));
        let touched = (t * 8).min(e) as f64 * (nv(1024, 2048) + nv(2048, 512));
        row(&format!("{name} x{copies}"), hot, cold, touched);
    }
    let _ = (moe_experts_scratch, Op::MoeTopk, OutSpec::new(DType::F32, &[1]));
}

/// Prefill-shaped timings at the 35B's shapes (hot caches, so compute-side numbers): the fp8 projections on the bf16 and on the
/// e4m3 WMMA at M = 4096, the routed experts at the chunk sizes of a 4408-token prompt. Sweep INTARSIA_HIP_MOE_Z with one process
/// per value: `cargo test --release -p intarsia-backend-hip --test quantmoe -- --ignored --nocapture bench_prefill_shapes`
#[test]
#[ignore]
fn bench_prefill_shapes() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(5);
    println!("env: MOE_Z={:?} MOE_WMMA={:?}", std::env::var("INTARSIA_HIP_MOE_Z").ok(), std::env::var("INTARSIA_HIP_MOE_WMMA").ok());
    for (name, n, k) in [("qkv 8192x2048", 8192usize, 2048usize), ("z 4096x2048", 4096, 2048), ("o_proj 2048x4096", 2048, 4096), ("k 512x2048", 512, 2048)] {
        for (m, aq) in [(312usize, true), (4096, true), (4096, false)] {
            let hc = intarsia_core::quant::cases::matmul_fp8_case(&mut rng, m, n, k, aq, DType::Bf16, DType::Bf16);
            let flops = 2.0 * (m * n * k) as f64;
            let mut line = format!("fp8 {name} M={m}{}:", if aq { "" } else { " weight-only" });
            for kern in ["matmul_fp8_mma", "matmul_fp8_mma_v2", "matmul_fp8_mma_f8"] {
                let t = bench_hot(&*hip, kern, &hc, 20);
                line += &format!("  {kern} {:.0} us {:.1} TFLOPS", t * 1e6, flops / t / 1e12);
            }
            println!("{line}");
        }
    }
    // INTARSIA_HIP_MOE_PAIRS is read at every launch: both pair tiles in this process (the test is ignored, so nothing races)
    for t in [312usize, 1024, 2048, 4096] {
        let hc = experts_case(&mut rng, t, 256, 8, 2048, 512, DType::Bf16);
        let flops = (t * 8) as f64 * (2.0 * 2.0 * 512.0 * 2048.0 + 2.0 * 2048.0 * 512.0);
        let mut line = format!("moe_experts T={t}:");
        for pairs in ["32", "64"] {
            std::env::set_var("INTARSIA_HIP_MOE_PAIRS", pairs);
            let secs = bench_hot(&*hip, "moe_experts_bf16_v2", &hc, 10);
            line += &format!("  pairs {pairs} {:.0} us {:.1} TFLOPS", secs * 1e6, flops / secs / 1e12);
        }
        std::env::remove_var("INTARSIA_HIP_MOE_PAIRS");
        println!("{line}");
        // projection variant bits (INTARSIA_HIP_MOE_MW, moe_wmma.h) at the auto tile, when INTARSIA_HIP_MW_SWEEP is set
        if std::env::var("INTARSIA_HIP_MW_SWEEP").is_ok() {
            let mut line = format!("moe_mw T={t}:");
            for mw in [0, 1, 4, 5, 13, 25, 29, 41, 45] {
                std::env::set_var("INTARSIA_HIP_MOE_MW", mw.to_string());
                let secs = bench_hot(&*hip, "moe_experts_bf16_v2", &hc, 10);
                line += &format!("  mw{mw} {:.0} us", secs * 1e6);
            }
            std::env::remove_var("INTARSIA_HIP_MOE_MW");
            println!("{line}");
        }
    }
}

/// Decode-path kernels, v1 against v2 (and the env-selected variants) at the real Qwen3.6-35B-A3B decode shapes, DRAM-cold:
/// `cargo test --release -p intarsia-backend-hip --test quantmoe -- --ignored --nocapture bench_decode_variants`
/// (run it once per INTARSIA_HIP_FP8_ROWS / INTARSIA_HIP_FP8_Q setting to sweep those knobs).
#[test]
#[ignore]
fn bench_decode_variants() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(5);
    println!(
        "env: FP8_ROWS={:?} FP8_Q={:?}",
        std::env::var("INTARSIA_HIP_FP8_ROWS").ok(),
        std::env::var("INTARSIA_HIP_FP8_Q").ok()
    );
    println!("{:<44} {:>9} {:>9} {:>9}", "case / kernel", "hot us", "cold us", "cold GB/s");
    let nv = |n: usize, k: usize| (n * k) as f64 * (0.5 + 1.0 / 16.0) / 1e6;
    let line = |name: &str, kern: &str, hot: f64, cold: f64, mb: f64| {
        println!("{:<44} {:>9.1} {:>9.1} {:>9.0}", format!("{name} {kern}"), hot * 1e6, cold * 1e6, mb / 1e3 / cold);
    };
    for (name, m, n, k, out) in [
        ("nvfp4 lm_head (248320x2048)", 1usize, 248_320usize, 2048usize, DType::F32),
        ("nvfp4 shared gate/up (512x2048)", 1, 512, 2048, DType::Bf16),
        ("nvfp4 shared down (2048x512)", 1, 2048, 512, DType::Bf16),
    ] {
        let hc = matmul_nvfp4_case(&mut rng, m, n, k, DType::Bf16, out);
        for kern in ["matmul_nvfp4_gemv", "matmul_nvfp4_gemv_v2"] {
            let hot = bench_hot(&*hip, kern, &hc, 100);
            let (cold, _, _) = bench_cold(&*hip, kern, &hc, 100, None);
            line(name, kern, hot, cold, nv(n, k));
        }
    }
    for (name, n, k) in [
        ("fp8 qkv/q (8192x2048)", 8192usize, 2048usize),
        ("fp8 z (4096x2048)", 4096, 2048),
        ("fp8 o_proj (2048x4096)", 2048, 4096),
        ("fp8 k/v (512x2048)", 512, 2048),
    ] {
        let hc = intarsia_core::quant::cases::matmul_fp8_case(&mut rng, 1, n, k, true, DType::Bf16, DType::Bf16);
        for kern in ["matmul_fp8_gemv", "matmul_fp8_gemv_v2"] {
            let hot = bench_hot(&*hip, kern, &hc, 100);
            let (cold, _, _) = bench_cold(&*hip, kern, &hc, 100, None);
            line(name, kern, hot, cold, (n * k) as f64 / 1e6);
        }
    }
    let hc = experts_case(&mut rng, 1, 256, 8, 2048, 512, DType::Bf16);
    let touched = 8.0 * (nv(1024, 2048) + nv(2048, 512));
    for kern in ["moe_experts_bf16", "moe_experts_bf16_v2"] {
        let hot = bench_hot(&*hip, kern, &hc, 100);
        let (cold, _, _) = bench_cold(&*hip, kern, &hc, 100, Some(1));
        line("moe_experts T=1 (8 of 256)", kern, hot, cold, touched);
    }
    let hc = intarsia_core::quant::cases::topk_case(&mut rng, 1, 256, 8, false, DType::Bf16, true);
    for kern in ["moe_topk_bf16", "moe_topk_bf16_v2"] {
        let hot = bench_hot(&*hip, kern, &hc, 200);
        println!("{:<44} {:>9.1}", format!("moe_topk T=1 E=256 {kern}"), hot * 1e6);
    }
}

/// fp8 GEMV at the 35B's decode shapes, DRAM-cold, GB/s against 627: v2 on every single projection and on a size scale
/// (K = 2048, N = 16 .. 32768: time against bytes gives the per-kernel fixed cost and the streaming rate), and every (R, W) of
/// the pair kernel on the two pairs. The pair's configuration travels in INTARSIA_HIP_FP8_PAIR (read at each launch).
/// `cargo test --release -p intarsia-backend-hip --test quantmoe -- --ignored --nocapture bench_fp8_sweep`
#[test]
#[ignore]
fn bench_fp8_sweep() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(5);
    println!("{:<34} {:<14} {:>9} {:>9} {:>7}", "case", "config", "hot us", "cold us", "GB/s");
    let line = |name: &str, cfg: &str, hot: f64, cold: f64, mb: f64| {
        println!("{:<34} {:<14} {:>9.1} {:>9.1} {:>7.0}", name, cfg, hot * 1e6, cold * 1e6, mb / 1e3 / cold);
    };
    let reps = 100;
    for (name, n, k) in [
        ("o_proj/out_proj 2048x4096", 2048usize, 4096usize),
        ("attn q 8192x2048", 8192, 2048),
        ("k or v 512x2048", 512, 2048),
    ] {
        let hc = intarsia_core::quant::cases::matmul_fp8_case(&mut rng, 1, n, k, true, DType::Bf16, DType::Bf16);
        let hot = bench_hot(&*hip, "matmul_fp8_gemv_v2", &hc, reps);
        let (cold, _, _) = bench_cold(&*hip, "matmul_fp8_gemv_v2", &hc, reps, None);
        line(name, "v2", hot, cold, (n * k) as f64 / 1e6);
    }
    for n in [16usize, 256, 1024, 2048, 4096, 8192, 16384, 32768] {
        let hc = intarsia_core::quant::cases::matmul_fp8_case(&mut rng, 1, n, 2048, true, DType::Bf16, DType::Bf16);
        let mb = (n * 2048) as f64 / 1e6;
        let hot = bench_hot(&*hip, "matmul_fp8_gemv_v2", &hc, reps);
        let (cold, _, _) = bench_cold(&*hip, "matmul_fp8_gemv_v2", &hc, reps, None);
        line(&format!("scale N={n} ({mb:.2} MB)"), "v2", hot, cold, mb);
    }
    for (name, n0, n1) in [("pair qkv+z 12288x2048", 8192usize, 4096usize), ("pair k+v 1024x2048", 512, 512)] {
        let hc = intarsia_core::quant::cases::matmul_fp8_pair_case(&mut rng, 1, n0, n1, 2048, true, true, DType::Bf16, DType::Bf16);
        let mb = ((n0 + n1) * 2048) as f64 / 1e6;
        for r in [1, 2, 4] {
            for w in [4, 8, 16] {
                let cfg = format!("{r},{w}");
                std::env::set_var("INTARSIA_HIP_FP8_PAIR", &cfg);
                let hot = bench_hot(&*hip, "matmul_fp8_pair_gemv", &hc, reps);
                let (cold, _, _) = bench_cold(&*hip, "matmul_fp8_pair_gemv", &hc, reps, None);
                line(name, &format!("pair {cfg}"), hot, cold, mb);
            }
        }
        std::env::remove_var("INTARSIA_HIP_FP8_PAIR");
    }
}
