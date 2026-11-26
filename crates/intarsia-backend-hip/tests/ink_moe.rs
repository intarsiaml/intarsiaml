//! Inkling-2-Mini routed experts on HIP (lane ink-hip-moe; kernels/hip/ink_moe.hip): `moe_experts_situ_bf16[_rows]` (NVFP4, SiTU)
//! and `moe_experts_mxfp8_bf16[_rows]` (72) against the cpu reference at the registered tolerances (the admission's own `judge`),
//! over the admission sweep plus pair-table shapes (more than 512 pairs) and odd routing; rows-exactness (a T-row call bitwise the
//! one-row calls, on the shared-memory path and on the pair-table path, at every load window); and an env-gated microbenchmark at
//! the real expert shape (128 experts, top-8, H 2304, I 1152):
//!   INK_MOE_BENCH=1 cargo test --release -p intarsia-backend-hip --test ink_moe -- --nocapture --test-threads=1
//! `INK_MOE_BIG=1` adds the sweep's T >= 1024 cases (the cpu reference takes a while). Needs a GPU: skips without one.
#![cfg(ik_hip)]

mod common;

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::admission::{judge, shape_of, sweep, SweepConfig};
use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::quant::cases::{experts_case, random_routing};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, KernelImpl, Op, OpAttrs, OpCall, ShapeInfo};
use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Instant;

const NAMES: [&str; 4] = ["moe_experts_situ_bf16", "moe_experts_situ_bf16_rows", "moe_experts_mxfp8_bf16", "moe_experts_mxfp8_bf16_rows"];

fn device(test: &str) -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_HIP_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_hip::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            common::skip(test, &format!("no usable HIP device ({e})"));
            None
        }
    }
}

fn kernel(hip: &dyn Backend, name: &str) -> KernelImpl {
    hip.kernels().into_iter().find(|k| k.name == name).unwrap_or_else(|| panic!("hip kernel {name} is not registered"))
}

fn cpu_ref(cpu: &CpuBackend, hc: &HostCall) -> Vec<HostTensor> {
    let name = cpu.kernels().into_iter().find(|k| k.op == hc.op && k.is_reference).expect("cpu reference").name;
    run_host(cpu, &name, hc).unwrap()
}

/// A situ NVFP4 case as the admission sweep builds it (routing weights around 8 / K, caps 4 / 25).
fn situ_case(rng: &mut SplitMix64, t: usize, e: usize, k: usize, h: usize, i: usize) -> HostCall {
    let mut c = experts_case(rng, t, e, k, h, i, DType::Bf16);
    let w: Vec<f32> = c.inputs[2].to_f32().iter().map(|v| v * 8.0).collect();
    c.inputs[2] = HostTensor::from_f32(&[t, k], &w);
    (c.attrs.ink.situ, c.attrs.ink.gate_cap, c.attrs.ink.up_cap) = (true, 4.0, 25.0);
    c
}

/// An MXFP8 case as the admission sweep builds it: e4m3 bytes with exponents 4..=10, E8M0 scales 116..120; `epi` silu | gelu | situ.
fn mxfp8_case(rng: &mut SplitMix64, t: usize, e: usize, k: usize, h: usize, i: usize, epi: &str) -> HostCall {
    let fp8 = |rng: &mut SplitMix64, shape: &[usize]| {
        let n: usize = shape.iter().product();
        let data = (0..n)
            .map(|_| {
                let r = rng.next_u64();
                ((r & 0x80) as u8) | ((4 + (r >> 8) % 7) as u8) << 3 | ((r >> 16) & 7) as u8
            })
            .collect();
        HostTensor::new(DType::Fp8E4m3, shape.to_vec(), data).expect("sized")
    };
    let sc = |rng: &mut SplitMix64, shape: &[usize]| {
        let n: usize = shape.iter().product();
        HostTensor::new(DType::U8, shape.to_vec(), (0..n).map(|_| 116 + (rng.next_u64() % 4) as u8).collect()).expect("sized")
    };
    let (ids, ws) = random_routing(rng, t, e, k);
    let ws = if epi == "situ" { HostTensor::from_f32(&[t, k], &ws.to_f32().iter().map(|v| v * 8.0).collect::<Vec<_>>()) } else { ws };
    let inputs = vec![
        HostTensor::from_f32_as(DType::Bf16, &[t, h], &rng.normal_vec(t * h, 1.0)),
        ids,
        ws,
        fp8(rng, &[e, 2 * i, h]),
        sc(rng, &[e, 2 * i, h / 32]),
        fp8(rng, &[e, h, i]),
        sc(rng, &[e, h, i / 32]),
    ];
    let mut a = OpAttrs { top_k: k, gelu: epi == "gelu", ..OpAttrs::default() };
    if epi == "situ" {
        (a.ink.situ, a.ink.gate_cap, a.ink.up_cap) = (true, 4.0, 25.0);
    }
    HostCall { op: Op::MoeExpertsMxfp8, inputs, outputs: vec![OutSpec::new(DType::Bf16, &[t, h])], attrs: a }
}

/// Runs every kernel of this lane that accepts `hc` and judges it against the cpu reference; returns how many kernels ran.
fn check(hip: &dyn Backend, cpu: &CpuBackend, hc: &HostCall, label: &str, fails: &mut Vec<String>) -> usize {
    let s = shape_of(hc).unwrap();
    let ks: Vec<KernelImpl> = NAMES.iter().map(|n| kernel(hip, n)).filter(|k| k.op == hc.op && k.accepts(&s)).collect();
    if ks.is_empty() {
        return 0;
    }
    let want = cpu_ref(cpu, hc);
    for k in &ks {
        let got = match run_host(hip, &k.name, hc) {
            Ok(g) => g,
            Err(e) => {
                fails.push(format!("{} {label}: launch error {e}", k.name));
                continue;
            }
        };
        let (pass, max_abs, rel) = judge(k, &got, &want);
        println!("{:<28} {label:<52} max_abs {max_abs:.3e} rel_l2 {rel:.3e} {}", k.name, if pass { "ok" } else { "FAIL" });
        if !pass {
            fails.push(format!("{} {label}: max_abs {max_abs:.3e} rel_l2 {rel:.3e}", k.name));
        }
    }
    ks.len()
}

#[test]
fn ink_moe_matches_cpu_reference() {
    let Some(hip) = device("ink_moe_matches_cpu_reference") else { return };
    let cpu = CpuBackend::new();
    let big = std::env::var_os("INK_MOE_BIG").is_some();
    let mut fails = Vec::new();
    let mut hits = [0usize; 2];
    for (n, op) in [Op::MoeExperts, Op::MoeExpertsMxfp8].into_iter().enumerate() {
        for case in sweep(op, &SweepConfig { quick: false, ..SweepConfig::default() }) {
            let s = shape_of(&case.call).unwrap();
            if s.m >= 1024 && !big {
                continue;
            }
            hits[n] += check(&*hip, &cpu, &case.call, &case.label, &mut fails);
        }
    }
    assert!(hits[0] > 0 && hits[1] > 0, "no sweep case reached the kernels: {hits:?}");
    // the pair-table path (more than 512 pairs) at small shapes, and 512 / 513+ pairs around the switch
    let mut rng = SplitMix64::new(0x1A4B_0001);
    for (t, e, k, h, i) in [(150usize, 16usize, 4usize, 64usize, 32usize), (70, 32, 8, 256, 128), (64, 32, 8, 96, 160), (65, 32, 8, 96, 160)] {
        let c = situ_case(&mut rng, t, e, k, h, i);
        check(&*hip, &cpu, &c, &format!("own T={t} E={e} K={k} H={h} I={i} situ"), &mut fails);
        for epi in ["silu", "gelu", "situ"] {
            let c = mxfp8_case(&mut rng, t, e, k, h, i, epi);
            check(&*hip, &cpu, &c, &format!("own T={t} E={e} K={k} H={h} I={i} mxfp8 {epi}"), &mut fails);
        }
    }
    // odd routing: every token on the same experts (40 pairs per expert: five batches), duplicate ids in a token, ids out of range
    for (t, tag) in [(40usize, "decode path"), (140, "table path")] {
        let (e, k, h, i) = (16usize, 4usize, 128usize, 64usize);
        for mx in [false, true] {
            let base = if mx { mxfp8_case(&mut rng, t, e, k, h, i, "situ") } else { situ_case(&mut rng, t, e, k, h, i) };
            let mut hot = base.clone();
            let ids: Vec<i32> = (0..t).flat_map(|_| 0..k as i32).collect();
            hot.inputs[1] = HostTensor::from_i32(&[t, k], &ids);
            check(&*hip, &cpu, &hot, &format!("hot experts T={t} mx={mx} ({tag})"), &mut fails);
            let mut odd = base.clone();
            let mut ids = odd.inputs[1].to_i32().unwrap();
            for j in 0..k {
                ids[j] = 3;
            }
            ids[k] = -1;
            ids[k + 1] = e as i32 + 5;
            odd.inputs[1] = HostTensor::from_i32(&[t, k], &ids);
            check(&*hip, &cpu, &odd, &format!("duplicate and invalid ids T={t} mx={mx} ({tag})"), &mut fails);
        }
    }
    assert!(fails.is_empty(), "{} failures:\n{}", fails.len(), fails.join("\n"));
}

/// Row `r` of `hc` as a one-row call.
fn one_row(hc: &HostCall, r: usize) -> HostCall {
    let mut c = hc.clone();
    for n in 0..3 {
        let t = &hc.inputs[n];
        let w = t.nbytes() / t.shape[0];
        c.inputs[n] = HostTensor::new(t.dtype, vec![1, t.shape[1]], t.data[r * w..(r + 1) * w].to_vec()).unwrap();
    }
    c.outputs = vec![OutSpec::new(hc.outputs[0].dtype, &[1, hc.outputs[0].shape[1]])];
    c
}

/// A T-row call is bitwise its one-row calls: on the shared-memory path (T 5 and 16), on the pair-table path (T 70 at top-8 = 560
/// pairs; rows 0, 33, 69), at every load window and in both step forms; the `_rows` twins give the ordinary kernels' bits.
#[test]
fn ink_moe_rows_exact() {
    let Some(hip) = device("ink_moe_rows_exact") else { return };
    let mut rng = SplitMix64::new(0x1A4B_0002);
    let (e, k, h, i) = (32usize, 8usize, 288usize, 160usize);
    for mx in [false, true] {
        let (ord, rows) = if mx { (NAMES[2], NAMES[3]) } else { (NAMES[0], NAMES[1]) };
        for t in [5usize, 16, 70] {
            let hc = if mx { mxfp8_case(&mut rng, t, e, k, h, i, "situ") } else { situ_case(&mut rng, t, e, k, h, i) };
            // the reference: the first form of the step at window 3; every other form / window gives its bits
            intarsia_backend_hip::inkmoe::set_lean(0);
            intarsia_backend_hip::inkmoe::set_pre(3, 3);
            let first = run_host(&*hip, ord, &hc).unwrap();
            intarsia_backend_hip::inkmoe::set_lean(1);
            for nb in [2, 4] {
                intarsia_backend_hip::inkmoe::set_nb(nb);
                for pre in [1, 2, 3, 4, 6] {
                    intarsia_backend_hip::inkmoe::set_pre(pre, pre);
                    let alt = run_host(&*hip, ord, &hc).unwrap();
                    assert_eq!(first[0].data, alt[0].data, "mx={mx} T={t}: lean, {nb} pairs per batch, load window {pre} changes bits");
                }
            }
            intarsia_backend_hip::inkmoe::set_nb(-1);
            intarsia_backend_hip::inkmoe::set_lean(-1);
            intarsia_backend_hip::inkmoe::set_pre(-1, -1);
            let full = run_host(&*hip, ord, &hc).unwrap();
            assert_eq!(first[0].data, full[0].data, "mx={mx} T={t}: the default form changes bits");
            if t <= 16 {
                let tw = run_host(&*hip, rows, &hc).unwrap();
                assert_eq!(full[0].data, tw[0].data, "mx={mx} T={t}: the _rows twin differs");
            }
            let rb = full[0].nbytes() / t;
            let picks: Vec<usize> = if t <= 16 { (0..t).collect() } else { vec![0, 33, t - 1] };
            for r in picks {
                let one = run_host(&*hip, rows, &one_row(&hc, r)).unwrap();
                assert_eq!(&full[0].data[r * rb..(r + 1) * rb], &one[0].data[..], "mx={mx} T={t}: row {r} is not the one-row call");
            }
        }
    }
}

fn put(b: &dyn Backend, t: &HostTensor) -> TensorRef {
    let buf = b.alloc(t.nbytes().max(16), MemKind::Device).unwrap();
    b.zero(&buf, 0).unwrap();
    b.upload(&buf, 0, &t.data, 0).unwrap();
    TensorRef::new(buf, t.dtype, &t.shape)
}

/// Microbenchmark at the real expert shape. One weight set on the device; `NR` calls with different routings (so successive calls
/// read different experts: 8 of 128 each at T = 1) run back to back, best of 5 passes. Prints us per call and GB/s over the bytes
/// of the distinct experts a call reads (payload + scales), for both launches and for gate_up / down alone: the default form, then
/// (T <= 16) each step form and load window.
#[test]
fn ink_moe_bench() {
    if std::env::var_os("INK_MOE_BENCH").is_none() {
        eprintln!("skipping: INK_MOE_BENCH unset");
        return;
    }
    let Some(hip) = device("ink_moe_bench") else { return };
    let b = &*hip;
    let (k, h, i) = (8usize, 2304usize, 1152usize);
    let ts: Vec<usize> =
        std::env::var("INK_MOE_BENCH_T").ok().map(|s| s.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![1, 2, 4, 16, 64, 256]);
    const NR: usize = 12;
    for mx in [false, true] {
        let e = if mx { 64usize } else { 128 };
        let name = if mx { NAMES[2] } else { NAMES[0] };
        let mut rng = SplitMix64::new(0xBE7C + mx as u64);
        let base = if mx { mxfp8_case(&mut rng, 1, e, k, h, i, "situ") } else { situ_case(&mut rng, 1, e, k, h, i) };
        let weights: Vec<TensorRef> = base.inputs[3..].iter().map(|t| put(b, t)).collect();
        // bytes one expert costs: every weight / scale tensor is [E, ..]
        let per_expert: usize = base.inputs[3..].iter().filter(|t| t.shape.len() == 3).map(|t| t.nbytes() / e).sum();
        // of which gate_up (inputs 3, 4) and down (5, 6)
        let gu_expert: usize = base.inputs[3..5].iter().map(|t| t.nbytes() / e).sum();
        let dn_expert: usize = base.inputs[5..7].iter().map(|t| t.nbytes() / e).sum();
        let kern = kernel(b, name);
        for &t in &ts {
            let mut calls = Vec::new();
            let mut bytes = 0usize;
            for _ in 0..NR {
                let (ids, ws) = random_routing(&mut rng, t, e, k);
                let distinct: BTreeSet<i32> = ids.to_i32().unwrap().into_iter().collect();
                bytes += distinct.len() * per_expert;
                let ws = HostTensor::from_f32(&[t, k], &ws.to_f32().iter().map(|v| v * 8.0).collect::<Vec<_>>());
                let x = HostTensor::from_f32_as(DType::Bf16, &[t, h], &rng.normal_vec(t * h, 1.0));
                let mut inputs = vec![put(b, &x), put(b, &ids), put(b, &ws)];
                inputs.extend(weights.iter().cloned());
                let out = put(b, &HostTensor::zeros(DType::Bf16, &[t, h]));
                let mut call = OpCall::new(base.op, inputs, vec![out], base.attrs.clone());
                call.check().unwrap();
                let need = kern.scratch_bytes(&ShapeInfo::from_call(&call));
                let buf = b.alloc(need.max(16), MemKind::Device).unwrap();
                call.scratch = Some(TensorRef::new(buf, DType::U8, &[need]));
                calls.push(call);
            }
            let avg_bytes = bytes as f64 / NR as f64;
            // (lean, window): the default (-1, -1) first, then the sweep at decode sizes
            // lean = -1: the default; 0: the first form (the reference of window 1); 2 / 4: lean with that many pairs per batch
            let mut cfgs: Vec<(i32, i32)> = vec![(-1, -1)];
            if t <= 16 && std::env::var_os("INK_MOE_BENCH_NOSWEEP").is_none() {
                cfgs.push((0, 3));
                for nb in [2, 4] {
                    cfgs.extend([1, 2, 3, 4, 6].map(|p| (nb, p)));
                }
            }
            for (lean, pre) in cfgs {
                intarsia_backend_hip::inkmoe::set_lean(if lean > 0 { 1 } else { lean });
                intarsia_backend_hip::inkmoe::set_nb(lean);
                intarsia_backend_hip::inkmoe::set_pre(pre, pre);
                let mut us = [0f64; 3];
                for probe in [0, 1, 2] {
                    // down alone reads the h of the full run before it
                    intarsia_backend_hip::inkmoe::set_probe(0);
                    for c in &calls {
                        b.launch(name, c, 0).unwrap();
                    }
                    b.synchronize(0).unwrap();
                    intarsia_backend_hip::inkmoe::set_probe(probe);
                    let reps = if t >= 64 { 1 } else { 4 };
                    let mut best = f64::MAX;
                    for _ in 0..7 {
                        let t0 = Instant::now();
                        for _ in 0..reps {
                            for c in &calls {
                                b.launch(name, c, 0).unwrap();
                            }
                        }
                        b.synchronize(0).unwrap();
                        best = best.min(t0.elapsed().as_secs_f64() * 1e6 / (reps * NR) as f64);
                    }
                    us[probe as usize] = best;
                }
                intarsia_backend_hip::inkmoe::set_probe(0);
                let (gu_b, dn_b) = (gu_expert as f64 * avg_bytes / per_expert as f64, dn_expert as f64 * avg_bytes / per_expert as f64);
                println!(
                    "INKMOEBENCH {} T={t:<4} lean={lean:<2} pre={pre:<2} both {:>9.1} us {:>6.0} GB/s | gate_up {:>9.1} us {:>5.0} GB/s | down {:>9.1} us {:>5.0} GB/s | {:.1} MB / call",
                    if mx { "mxfp8" } else { "nvfp4" },
                    us[0],
                    avg_bytes / us[0] / 1e3,
                    us[1],
                    gu_b / us[1] / 1e3,
                    us[2],
                    dn_b / us[2] / 1e3,
                    avg_bytes / 1e6
                );
            }
            intarsia_backend_hip::inkmoe::set_lean(-1);
            intarsia_backend_hip::inkmoe::set_nb(-1);
            intarsia_backend_hip::inkmoe::set_pre(-1, -1);
        }
    }
}
