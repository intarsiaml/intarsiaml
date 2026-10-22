//! The core-op fusion phase-2 kernels (docs/results/2026-10-19-core-op-fusion-p2.md) against the launches they replace on
//! the same device, bitwise, and against the cpu reference (admission budget):
//!
//! * `gate_split_qk_norm_rope_kv_bf16` (P4a) = `gate_split_bf16` -> `qk_norm_rope_kv_bf16` of the split-off query; all
//!   four outputs (q, gate, k cache, v cache) compared, bf16 and fp8 caches, partial rotary, both norm offsets.
//! * `gdn_conv_delta_bf16` (P3) = `gdn_conv_bf16` -> `gdn_delta_rule_bf16_v2`; the output, the delta-rule state and the conv
//!   state compared after every launch of a chain of launches that carries both states (chunk boundaries: T = 5, 1, 32, 1,
//!   17, ...), with the race seam on (every other block reads the old conv state late, so a block that stored the new q / k
//!   state before a sibling read it would show) and with the two-launch form (no counter slot).
//!
//! The shapes include the 35B's (16 / 2 heads of 256; 16 key / 32 value heads of 128) and the edges: value rows that do not
//! fill a block (idle lanes), one block per key head (the first ticket is the last), several value heads per key head.
//! scripts/hip-decode/mutate.py has the mutants (`p2-*`) this file must fail on.
//!
//! Needs a GPU: skips (passes) without one.
#![cfg(ik_hip)]

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::admission::{gdn_conv_delta_case, sweep, SweepConfig};
use intarsia_core::backend::Backend;
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, Op, OpAttrs};
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

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

fn bits_eq(label: &str, what: &str, got: &HostTensor, want: &HostTensor) {
    assert_eq!(got.shape, want.shape, "{label}: {what} shape");
    if got.data != want.data {
        let (g, w) = (got.to_f32(), want.to_f32());
        let bad = g.iter().zip(&w).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let first = g.iter().zip(&w).position(|(a, b)| a.to_bits() != b.to_bits()).unwrap_or(0);
        panic!(
            "{label}: {what} differs from the unfused launches in {bad} of {} elements (first at {first}: {} vs {})",
            g.len(),
            g[first],
            w[first]
        );
    }
}

fn near_reference(op: Op, hc: &HostCall, got: &[HostTensor], rel: f64, label: &str) {
    let cpu = CpuBackend::new();
    let reference = cpu.kernels().into_iter().find(|k| k.op == op && k.is_reference).unwrap().name;
    let want = run_host(&cpu, &reference, hc).unwrap();
    for (o, (g, w)) in got.iter().zip(&want).enumerate() {
        let r = rel_l2(&g.to_f32(), &w.to_f32());
        assert!(r < rel, "{label}: output {o} vs {reference}: rel_l2 {r}");
    }
}

fn call(op: Op, inputs: Vec<HostTensor>, outputs: Vec<OutSpec>, attrs: OpAttrs) -> HostCall {
    HostCall { op, inputs, outputs, attrs }
}

fn with_init(t: &HostTensor) -> OutSpec {
    OutSpec { dtype: t.dtype, shape: t.shape.clone(), init: Some(t.clone()) }
}

/// The unfused P4a chain on `dev`: gate_split_bf16, then qk_norm_rope_kv_bf16 of the split-off query.
fn p4a_unfused(dev: &dyn Backend, hc: &HostCall) -> Vec<HostTensor> {
    let (t, n, d2) = (hc.inputs[0].shape[0], hc.inputs[0].shape[1], hc.inputs[0].shape[2]);
    let split = call(
        Op::GateSplit,
        vec![hc.inputs[0].clone()],
        vec![OutSpec::new(DType::Bf16, &[t, n, d2 / 2]), OutSpec::new(DType::Bf16, &[t, n, d2 / 2])],
        OpAttrs::default(),
    );
    let qg = run_host(dev, "gate_split_bf16", &split).unwrap();
    let mut inputs = hc.inputs.clone();
    inputs[0] = qg[0].clone();
    let pro = call(
        Op::QkNormRopeKv,
        inputs,
        vec![hc.outputs[0].clone(), hc.outputs[2].clone(), hc.outputs[3].clone()],
        hc.attrs.clone(),
    );
    let r = run_host(dev, "qk_norm_rope_kv_bf16", &pro).unwrap();
    vec![r[0].clone(), qg[1].clone(), r[1].clone(), r[2].clone()]
}

#[test]
fn gate_split_qk_norm_rope_kv_is_bitwise_split_then_prologue() {
    let Some(hip) = device() else { return };
    let mut n = 0;
    // the admission sweep's cases (incl. the 35B's heads), bf16 activations, then each with an fp8 cache
    for (nh, nkv, d) in [(16usize, 2usize, 256usize), (8, 1, 256), (32, 8, 128), (8, 8, 128)] {
        let cfg = SweepConfig { n_heads: nh, n_kv_heads: nkv, head_dim: d, quick: false, seed: 91, ..SweepConfig::default() };
        for case in sweep(Op::GateSplitQkNormRopeKv, &cfg) {
            if case.call.inputs[0].dtype != DType::Bf16 {
                continue;
            }
            for fp8 in [false, true] {
                let mut hc = case.call.clone();
                if fp8 {
                    for o in &mut hc.outputs[2..4] {
                        let init = o.init.as_ref().unwrap().to_f32();
                        o.dtype = DType::Fp8E4m3;
                        o.init = Some(HostTensor::from_f32_as(DType::Fp8E4m3, &o.shape, &init));
                    }
                }
                let label = format!("P4a {} fp8 cache {fp8}", case.label);
                let got = run_host(&*hip, "gate_split_qk_norm_rope_kv_bf16", &hc).unwrap_or_else(|e| panic!("{label}: {e}"));
                let want = p4a_unfused(&*hip, &hc);
                for (i, what) in ["q", "gate", "k cache", "v cache"].iter().enumerate() {
                    bits_eq(&label, what, &got[i], &want[i]);
                }
                if !fp8 {
                    near_reference(Op::GateSplitQkNormRopeKv, &hc, &got, 2e-2, &label);
                }
                n += 1;
            }
        }
    }
    assert!(n >= 100, "only {n} cases ran");
}

/// The unfused P3 chain on `dev`: gdn_conv_bf16, then gdn_delta_rule_bf16_v2 of its output. Returns (o, state, conv state).
fn p3_unfused(dev: &dyn Backend, hc: &HostCall) -> Vec<HostTensor> {
    let x = &hc.inputs[0];
    let conv = call(Op::GdnConv, vec![x.clone(), hc.inputs[1].clone()], vec![OutSpec::new(DType::Bf16, &x.shape), hc.outputs[2].clone()], OpAttrs::default());
    let c = run_host(dev, "gdn_conv_bf16", &conv).unwrap();
    let mut inputs = vec![c[0].clone()];
    inputs.extend(hc.inputs[2..].iter().cloned());
    let delta = call(Op::GdnDeltaRule, inputs, vec![hc.outputs[0].clone(), hc.outputs[1].clone()], hc.attrs.clone());
    let d = run_host(dev, "gdn_delta_rule_bf16_v2", &delta).unwrap();
    vec![d[0].clone(), d[1].clone(), c[1].clone()]
}

/// (elements differing in bits, max |difference|, first differing index) of two tensors of one shape.
fn diff(got: &HostTensor, want: &HostTensor) -> (usize, f32, usize) {
    assert_eq!(got.shape, want.shape);
    let (g, w) = (got.to_f32(), want.to_f32());
    let mut out = (0usize, 0f32, usize::MAX);
    for (i, (a, b)) in g.iter().zip(&w).enumerate() {
        if a.to_bits() != b.to_bits() {
            out.0 += 1;
            out.1 = out.1.max((a - b).abs());
            out.2 = out.2.min(i);
        }
    }
    out
}

const P3_OUTS: [&str; 3] = ["o", "delta state", "conv state"];

/// A chain of P3 launches with the token counts `ts`, both paths carrying their own states; every launch compared, all
/// three tensors reported before the step fails. Under the race seam (INTARSIA_HIP_GDN_CD_SKEW set) each step also runs
/// the fused kernel without the seam on the same inputs: fused(skew) == fused(no skew) != unfused is arithmetic,
/// fused(skew) != fused(no skew) is timing.
fn p3_chain(hip: &dyn Backend, rng: &mut SplitMix64, shape: (usize, usize, usize, usize), ts: &[usize], label: &str) {
    let (hk, hv, dk, dv) = shape;
    let skew = std::env::var("INTARSIA_HIP_GDN_CD_SKEW").ok();
    let mut fused_states: Option<(HostTensor, HostTensor)> = None;
    let mut plain_states: Option<(HostTensor, HostTensor)> = None;
    for (step, &t) in ts.iter().enumerate() {
        let mut hc = gdn_conv_delta_case(rng, t, hk, hv, dk, dv, 4, DType::Bf16);
        let lab = format!("{label} step {step} T={t} Hk={hk} Hv={hv} dv={dv}");
        // the first launch starts from the case's random states, every later one from the states each path produced
        let mut f = hc.clone();
        if let Some((s, c)) = &fused_states {
            f.outputs[1] = with_init(s);
            f.outputs[2] = with_init(c);
        }
        if let Some((s, c)) = &plain_states {
            hc.outputs[1] = with_init(s);
            hc.outputs[2] = with_init(c);
        }
        let got = run_host(hip, "gdn_conv_delta_bf16", &f).unwrap_or_else(|e| panic!("{lab}: {e}"));
        let want = p3_unfused(hip, &hc);
        let mut bad = Vec::new();
        for (i, what) in P3_OUTS.iter().enumerate() {
            let (n, m, first) = diff(&got[i], &want[i]);
            if n > 0 {
                bad.push(format!("{what}: {n} of {} differ, max |d| {m:e}, first at {first}", got[i].shape.iter().product::<usize>()));
            }
        }
        if let Some(sk) = &skew {
            std::env::remove_var("INTARSIA_HIP_GDN_CD_SKEW");
            let calm = run_host(hip, "gdn_conv_delta_bf16", &f).unwrap();
            std::env::set_var("INTARSIA_HIP_GDN_CD_SKEW", sk);
            for (i, what) in P3_OUTS.iter().enumerate() {
                let (n, m, first) = diff(&got[i], &calm[i]);
                if n > 0 {
                    bad.push(format!("TIMING: {what} skewed vs unskewed fused: {n} differ, max |d| {m:e}, first at {first}"));
                }
            }
            if !bad.is_empty() {
                for (i, what) in P3_OUTS.iter().enumerate() {
                    let (n, m, _) = diff(&calm[i], &want[i]);
                    bad.push(format!("unskewed fused vs unfused: {what}: {n} differ, max |d| {m:e}"));
                }
            }
        }
        assert!(bad.is_empty(), "{lab}: {}", bad.join("; "));
        if step == 0 {
            near_reference(Op::GdnConvDelta, &f, &got, 2e-2, &lab);
        }
        fused_states = Some((got[1].clone(), got[2].clone()));
        plain_states = Some((want[1].clone(), want[2].clone()));
    }
}

/// Many fresh draws at the 35B's shape (one token: the decode step; 32: the longest chunk), no seam: a rounding difference
/// that only some inputs expose (one bf16 rounding of a conv output flipping) shows here.
#[test]
fn gdn_conv_delta_many_draws_bitwise() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(95);
    for d in 0..200 {
        p3_chain(&*hip, &mut rng, P3_SHAPES[0], &[1], &format!("P3 draw {d}"));
    }
    for d in 0..20 {
        p3_chain(&*hip, &mut rng, P3_SHAPES[0], &[32], &format!("P3 draw32 {d}"));
    }
    eprintln!("P3: 220 draws bitwise");
}

const P3_SHAPES: [(usize, usize, usize, usize); 6] = [
    (16, 32, 128, 128), // the 35B: 8 row blocks per value head, 16 blocks per key head
    (8, 16, 128, 128),  // the 35B tp=2 shard (lane H1)
    (1, 2, 128, 32),    // the admission sweep's small one
    (2, 4, 128, 40),    // value rows that do not fill the last block (idle lanes)
    (1, 1, 128, 16),    // one block per key head
    (2, 6, 128, 24),    // three value heads per key head, a partial block
];
const P3_CHAIN: [usize; 9] = [5, 1, 32, 1, 17, 2, 1, 31, 3];

#[test]
fn gdn_conv_delta_is_bitwise_conv_then_delta_rule() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(92);
    for shape in P3_SHAPES {
        p3_chain(&*hip, &mut rng, shape, &P3_CHAIN, "P3");
    }
    // every prefetch depth (gdn_delta_rule_bf16_v2's knob, read once per process: the same value on both paths)
    let pd = std::env::var("INTARSIA_HIP_GDN_PD").unwrap_or_else(|_| "2".into());
    eprintln!("INTARSIA_HIP_GDN_PD={pd}");
}

/// The race seam: every other block reads its old conv state late. A kernel that stored a key head's new q / k state
/// before every block of that head had read the old one fails here (and the counter must still reset between launches).
#[test]
fn gdn_conv_delta_conv_state_survives_skewed_blocks() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(93);
    std::env::set_var("INTARSIA_HIP_GDN_CD_SKEW", "40");
    for shape in [P3_SHAPES[0], P3_SHAPES[2], P3_SHAPES[4]] {
        p3_chain(&*hip, &mut rng, shape, &[1, 1, 5, 1, 32, 1], "P3 skewed");
    }
    std::env::remove_var("INTARSIA_HIP_GDN_CD_SKEW");
}

/// The two-launch form (no counter slot): the fused kernel stores no conv state, a second launch stores it.
#[test]
fn gdn_conv_delta_without_a_slot_is_bitwise_too() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(94);
    std::env::set_var("INTARSIA_HIP_GDN_CD_NOSLOT", "1");
    for shape in [P3_SHAPES[0], P3_SHAPES[2]] {
        p3_chain(&*hip, &mut rng, shape, &[5, 1, 32, 1], "P3 no slot");
    }
    std::env::remove_var("INTARSIA_HIP_GDN_CD_NOSLOT");
}
