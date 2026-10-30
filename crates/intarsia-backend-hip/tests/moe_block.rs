//! `moe_block_bf16` (kernels/hip/moe_block.hip, `Op::MoeBlock`, P2', docs/results/2026-10-19-p2prime-moeblock.md): the
//! MoE block after the router logits in one launch. Its output must be bitwise the default HIP chain it replaces:
//! `moe_topk_bf16_v2` -> `moe_experts_bf16_v2` -> `matmul_nvfp4_glu_gemv` (T = 1; at T > 1 the gate and up
//! `matmul_nvfp4_gemv_v2` + `silu_glu_bf16`) -> `shared_down_combine_nvfp4_gemv`.
//!
//! * `moe_block_is_bitwise_the_chain`: every output byte equals the chain's at the 35B's decode shapes (T = 1, 2), at small /
//!   odd ones (H past the glu window, IS past the shared-down window, row tails, K = E), both top-k normalisations, logits
//!   with ties; plus the cpu reference (when registered) within the admission budget.
//! * `moe_block_counters_reset_between_launches`: 200 back-to-back launches without a host sync on one scratch, two shapes.
//! * `moe_block_graph_replay_is_bitwise_eager`: one captured launch replayed 20 times with new activations.
//! * `moe_block_dispatch_order`: INTARSIA_HIP_MOE_BLOCK_SCRAMBLE delays blocks by blockIdx before they draw their ticket, so
//!   tickets come in an order unrelated to blockIdx (deadlock-freedom for any dispatch order; bitwise every launch).
//! * `moe_block_sentinel_stress`: both scratch edges (routed h, shared sa) poisoned with a bf16 NaN before every launch,
//!   the down waves pre-reading their h and sa lines before the wait, one producer role late (the last routed role, the last
//!   shared role, token 0's last routed / shared role, or every producer), T = 1 and T = 2 shapes, with and without a second
//!   instance on stream 1. Every launch bitwise the chain (the mutants of every edge must fail it).
//! * `moe_block_timing` (only with INTARSIA_HIP_MOE_BLOCK_BENCH=1): the M1 microbench, graph replay of L layers of the chain
//!   (v2, and v3) against L fused launches at the 35B decode shape.
//!
//! Needs a GPU: skips (passes) without one.
#![cfg(ik_hip)]

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::quant::cases::{random_experts, topk_case};
use intarsia_core::quant::nvfp4;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{Backend, DType, HostTensor, MemKind, Op, OpAttrs, OpCall, ShapeInfo, TensorRef};
use std::sync::Arc;
use std::time::Instant;

const K_NAME: &str = "moe_block_bf16";

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

/// A MoeBlock call: n, logits, sl, the five routed expert tensors, shared gate / up / down (three nvfp4 parts each).
fn block_case(rng: &mut SplitMix64, t: usize, e: usize, k: usize, h: usize, i: usize, is: usize, norm: bool, ties: bool) -> HostCall {
    let ex = random_experts(rng, e, h, i);
    let x = HostTensor::from_f32_as(DType::Bf16, &[t, h], &rng.normal_vec(t * h, 1.0));
    let logits = topk_case(rng, t, e, k, ties, DType::Bf16, norm).inputs.remove(0);
    let sl = HostTensor::from_f32_as(DType::Bf16, &[t, 1], &rng.normal_vec(t, 2.0));
    let g = nvfp4::random_rows(rng, is, h);
    let u = nvfp4::random_rows(rng, is, h);
    let d = nvfp4::random_rows(rng, h, is);
    let mut inputs = vec![x, logits, sl, ex.gate_up_w, ex.gate_up_s, ex.down_w, ex.down_s, ex.gscales];
    inputs.extend(g.parts);
    inputs.extend(u.parts);
    inputs.extend(d.parts);
    HostCall {
        op: Op::MoeBlock,
        inputs,
        outputs: vec![OutSpec::new(DType::Bf16, &[t, h])],
        attrs: OpAttrs { top_k: k, norm_topk: norm, ..OpAttrs::default() },
    }
}

fn dims(hc: &HostCall) -> (usize, usize, usize, usize, usize, usize) {
    let (t, h) = (hc.inputs[0].shape[0], hc.inputs[0].shape[1]);
    let e = hc.inputs[1].shape[1];
    let i = hc.inputs[3].shape[1] / 2;
    let is = hc.inputs[8].shape[0];
    (t, e, hc.attrs.top_k, h, i, is)
}

/// The default HIP chain on the device, launch by launch (each op's output downloaded and handed to the next).
fn chain(be: &dyn Backend, hc: &HostCall) -> Vec<u8> {
    let (t, _e, k, h, _i, is) = dims(hc);
    let inp = &hc.inputs;
    let call = |op: Op, inputs: Vec<HostTensor>, outputs: Vec<OutSpec>, attrs: OpAttrs| HostCall { op, inputs, outputs, attrs };
    let tk = run_host(
        be,
        "moe_topk_bf16_v2",
        &call(Op::MoeTopk, vec![inp[1].clone()], vec![OutSpec::new(DType::F32, &[t, k]), OutSpec::new(DType::I32, &[t, k])], hc.attrs.clone()),
    )
    .unwrap();
    let routed = run_host(
        be,
        "moe_experts_bf16_v2",
        &call(
            Op::MoeExperts,
            vec![inp[0].clone(), tk[1].clone(), tk[0].clone(), inp[3].clone(), inp[4].clone(), inp[5].clone(), inp[6].clone(), inp[7].clone()],
            vec![OutSpec::new(DType::Bf16, &[t, h])],
            hc.attrs.clone(),
        ),
    )
    .unwrap()
    .remove(0);
    let a = if t == 1 {
        run_host(
            be,
            "matmul_nvfp4_glu_gemv",
            &call(Op::MatmulNvfp4Glu, vec![inp[0].clone(), inp[8].clone(), inp[9].clone(), inp[10].clone(), inp[11].clone(), inp[12].clone(), inp[13].clone()], vec![OutSpec::new(DType::Bf16, &[t, is])], OpAttrs::default()),
        )
        .unwrap()
        .remove(0)
    } else {
        let mm = |w: usize| {
            run_host(
                be,
                "matmul_nvfp4_gemv_v2",
                &call(Op::MatmulNvfp4, vec![inp[0].clone(), inp[w].clone(), inp[w + 1].clone(), inp[w + 2].clone()], vec![OutSpec::new(DType::Bf16, &[t, is])], OpAttrs::default()),
            )
            .unwrap()
            .remove(0)
        };
        let (g, u) = (mm(8), mm(11));
        run_host(be, "silu_glu_bf16", &call(Op::SiluGlu, vec![g, u], vec![OutSpec::new(DType::Bf16, &[t, is])], OpAttrs::default())).unwrap().remove(0)
    };
    run_host(
        be,
        "shared_down_combine_nvfp4_gemv",
        &call(Op::SharedDownCombine, vec![a, inp[14].clone(), inp[15].clone(), inp[16].clone(), routed, inp[2].clone()], vec![OutSpec::new(DType::Bf16, &[t, h])], OpAttrs::default()),
    )
    .unwrap()
    .remove(0)
    .data
}

fn same_as_chain(hip: &dyn Backend, hc: &HostCall, label: &str) {
    let want = chain(hip, hc);
    let got = run_host(hip, K_NAME, hc).unwrap_or_else(|e| panic!("{label}: moe_block: {e}"));
    if got[0].data != want {
        let n = got[0].data.chunks(2).zip(want.chunks(2)).filter(|(a, b)| a != b).count();
        panic!("{label}: moe_block differs from the chain in {n} of {} outputs", want.len() / 2);
    }
    let cpu = CpuBackend::new();
    if let Some(reference) = cpu.kernels().into_iter().find(|k| k.op == Op::MoeBlock && k.is_reference) {
        let r = rel_l2(&got[0].to_f32(), &run_host(&cpu, &reference.name, hc).unwrap()[0].to_f32());
        assert!(r < 2e-2, "{label}: moe_block vs {}: rel_l2 {r}", reference.name);
    }
}

#[test]
fn moe_block_is_bitwise_the_chain() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(71);
    let mut n = 0;
    for (t, e, k, h, i, is) in [
        (1usize, 256usize, 8usize, 2048usize, 512usize, 512usize), // the 35B's decode step
        (2, 256, 8, 2048, 512, 512),
        (1, 256, 8, 2048, 256, 256), // the 35B tp=2 shard (lane H1)
        (2, 256, 8, 2048, 256, 256),
        (1, 64, 8, 1024, 256, 256),
        (1, 32, 4, 256, 48, 64),
        (1, 16, 2, 4112, 160, 32), // H / 16 past the 4-iteration windows (glu, gate_up)
        (1, 16, 2, 256, 32, 2064), // IS / 16 past the shared-down window
        (2, 16, 8, 64, 16, 16),
        (1, 16, 8, 208, 64, 16), // K = 8 of 16 experts, row tails (208 = 26 x 8)
        (4, 16, 4, 128, 32, 48), // T * K = 16
        (1, 8, 8, 256, 32, 32), // K = E
        (3, 512, 4, 256, 32, 32), // E = 512 (the top-k limit)
    ] {
        for norm in [true, false] {
            for ties in [false, true] {
                let hc = block_case(&mut rng, t, e, k, h, i, is, norm, ties);
                same_as_chain(&*hip, &hc, &format!("t={t} e={e} k={k} h={h} i={i} is={is} norm={norm} ties={ties}"));
                n += 1;
            }
        }
    }
    eprintln!("moe_block_is_bitwise_the_chain: {n} cases bitwise");
}

/// Token `t` of a MoeBlock call as a T = 1 call (x, logits and the shared gate logit sliced; the weights are shared).
fn token_of(hc: &HostCall, t: usize) -> HostCall {
    let row = |x: &HostTensor| {
        let w = x.shape[1];
        let bytes = w * x.dtype.size_bytes();
        HostTensor { dtype: x.dtype, shape: vec![1, w], data: x.data[t * bytes..(t + 1) * bytes].to_vec() }
    };
    let mut inputs = hc.inputs.clone();
    for j in 0..3 {
        inputs[j] = row(&hc.inputs[j]);
    }
    let h = hc.inputs[0].shape[1];
    HostCall { op: hc.op, inputs, outputs: vec![OutSpec::new(DType::Bf16, &[1, h])], attrs: hc.attrs.clone() }
}

/// G-B of a row-exact MoE block kernel (`moe_block_bf16_rows`, lane hr3, batched decode; `moe_block_bf16_grouped`, lane l7): token t of a T-token
/// launch is BITWISE the T = 1 launch of the solo step (`moe_block_bf16`) on token t alone, T = 1..=16, at the 35B's decode shape (top-8 of 256
/// experts: T * K far past 16) and on smaller ones, norm on and off, with ties in the routing. `ts_big` / `ts_small`: the row counts tried at the
/// 35B shape and at the others (the bigger shapes at every row count would be minutes of random weights).
fn rows_are_the_solo_block_per_token(kernel: &str, seed: u64, ts_big: &[usize], ts_small: &[usize]) {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(seed);
    let (mut cases, mut tokens, mut controls) = (0, 0, 0);
    for (e, k, h, i, is) in [
        (256usize, 8usize, 2048usize, 512usize, 512usize), // the 35B's decode step
        (256, 8, 2048, 256, 256),                        // its tp=2 shard
        (64, 8, 1024, 256, 256),
        (32, 4, 256, 48, 64),
        (16, 2, 4112, 160, 32),
        (512, 4, 256, 32, 32), // E = 512
    ] {
        for norm in [true, false] {
            for ties in [false, true] {
                for t in 1..=16usize {
                    if e == 256 && i == 512 && !ts_big.contains(&t) {
                        continue;
                    }
                    if !(e == 256 && i == 512) && !ts_small.contains(&t) {
                        continue;
                    }
                    let hc = block_case(&mut rng, t, e, k, h, i, is, norm, ties);
                    let got = run_host(&*hip, kernel, &hc).unwrap_or_else(|er| panic!("{kernel} t={t} e={e}: {er}"));
                    let hb = h * 2;
                    for r in 0..t {
                        let want = run_host(&*hip, K_NAME, &token_of(&hc, r)).unwrap();
                        assert!(
                            got[0].data[r * hb..(r + 1) * hb] == want[0].data[..],
                            "{kernel} t={t} e={e} k={k} h={h} i={i} is={is} norm={norm} ties={ties}: token {r} differs from the solo T = 1 block"
                        );
                        tokens += 1;
                    }
                    // negative control: the comparison discriminates (token r against the solo block of the NEXT token differs)
                    if t > 1 {
                        let other = run_host(&*hip, K_NAME, &token_of(&hc, 1)).unwrap();
                        assert!(got[0].data[..hb] != other[0].data[..], "control: token 0 equals the solo block of token 1 (t={t} e={e}): the comparison cannot fail");
                        controls += 1;
                    }
                    cases += 1;
                }
            }
        }
    }
    eprintln!("{kernel}: {cases} launches, {tokens} tokens bitwise the T = 1 block, {controls} negative controls differ");
}

#[test]
fn moe_block_rows_are_the_solo_block_per_token() {
    rows_are_the_solo_block_per_token("moe_block_bf16_rows", 91, &[1, 2, 3, 4, 7, 8, 9, 12, 15, 16], &[1, 3, 5, 16]);
}

/// Lane l7: the grouped block (kernels/hip/moe_block_grouped.hip), every row count 1..=16 at the small shapes and a spread at the 35B's (lane hgv:
/// every residue of T mod 4, the shared-expert kernels run groups of 4 tokens per block).
#[test]
fn moe_block_grouped_is_the_solo_block_per_token() {
    rows_are_the_solo_block_per_token("moe_block_bf16_grouped", 92, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 16], &[1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 16]);
}

/// Lane l7 negative controls for the grouped block: a block assembled the same way but with one stage on an inexact (tensor-core) kernel must NOT be
/// the solo block per token, so the comparison of `moe_block_grouped_is_the_solo_block_per_token` has something to catch. (a) the routed experts through
/// `moe_experts_bf16_v2` at T = 16 (tensor-core projections, another accumulation order); (b) the shared gate / up / down through `matmul_nvfp4_mma`
/// (reported only, see below).
#[test]
fn moe_block_grouped_negative_controls() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(93);
    let (t, e, k, h, i, is) = (16usize, 256usize, 8usize, 2048usize, 512usize, 512usize);
    let hc = block_case(&mut rng, t, e, k, h, i, is, true, false);
    let inp = &hc.inputs;
    let call = |op: Op, inputs: Vec<HostTensor>, outputs: Vec<OutSpec>, attrs: OpAttrs| HostCall { op, inputs, outputs, attrs };
    let tk = run_host(&*hip, "moe_topk_bf16_v2", &call(Op::MoeTopk, vec![inp[1].clone()], vec![OutSpec::new(DType::F32, &[t, k]), OutSpec::new(DType::I32, &[t, k])], hc.attrs.clone())).unwrap();
    let assemble = |experts: &str, mm: &str| -> Vec<u8> {
        let routed = run_host(
            &*hip,
            experts,
            &call(
                Op::MoeExperts,
                vec![inp[0].clone(), tk[1].clone(), tk[0].clone(), inp[3].clone(), inp[4].clone(), inp[5].clone(), inp[6].clone(), inp[7].clone()],
                vec![OutSpec::new(DType::Bf16, &[t, h])],
                hc.attrs.clone(),
            ),
        )
        .unwrap()
        .remove(0);
        let mmul = |x: &HostTensor, w: usize, n: usize| {
            run_host(&*hip, mm, &call(Op::MatmulNvfp4, vec![x.clone(), inp[w].clone(), inp[w + 1].clone(), inp[w + 2].clone()], vec![OutSpec::new(DType::Bf16, &[t, n])], OpAttrs::default()))
                .unwrap()
                .remove(0)
        };
        let (g, u) = (mmul(&inp[0], 8, is), mmul(&inp[0], 11, is));
        let a = run_host(&*hip, "silu_glu_bf16", &call(Op::SiluGlu, vec![g, u], vec![OutSpec::new(DType::Bf16, &[t, is])], OpAttrs::default())).unwrap().remove(0);
        let so = mmul(&a, 14, h);
        run_host(&*hip, "moe_combine_bf16", &call(Op::MoeCombine, vec![routed, so, inp[2].clone()], vec![OutSpec::new(DType::Bf16, &[t, h])], OpAttrs::default())).unwrap().remove(0).data
    };
    let hb = h * 2;
    let solo: Vec<Vec<u8>> = (0..t).map(|r| run_host(&*hip, K_NAME, &token_of(&hc, r)).unwrap().remove(0).data).collect();
    let differs = |got: &[u8]| (0..t).filter(|r| got[r * hb..(r + 1) * hb] != solo[*r][..]).count();
    // the exact assembly of the same stages IS the solo block (the control's baseline: everything but the mutated stage is the chain)
    let exact = assemble("moe_experts_bf16_rows", "matmul_nvfp4_gemv_rows");
    assert_eq!(differs(&exact), 0, "the unfused row-exact chain must be the solo block per token");
    let a = differs(&assemble("moe_experts_bf16_v2", "matmul_nvfp4_gemv_rows"));
    let b = differs(&assemble("moe_experts_bf16_rows", "matmul_nvfp4_mma"));
    eprintln!("moe_block_grouped_negative_controls: tensor-core experts differ in {a} of {t} tokens, tensor-core shared GEMVs in {b} of {t}");
    assert!(a > 0, "control (a): the tensor-core experts gave the solo block in every token, the comparison cannot fail");
    // (b) is reported, not asserted: on these activations the tensor-core shared GEMVs (exact products, f32 accumulation in another order) can round to
    // the same bf16 everywhere, so it is no control; (a) is the control of the routed path and the "next token's solo block" control of
    // `moe_block_grouped_is_the_solo_block_per_token` covers the whole block
}

/// Lane l7: the grouped block replayed from a captured graph (7 launches) is the eager launch, on new activations each replay.
#[test]
fn moe_block_grouped_graph_replay_is_bitwise_eager() {
    let Some(be) = device() else { return };
    assert!(be.supports_graphs());
    let mut rng = SplitMix64::new(94);
    let (t, name) = (9usize, "moe_block_bf16_grouped");
    let hc = block_case(&mut rng, t, 256, 8, 2048, 512, 512, true, false);
    let d = Dev::new(&*be, name, &hc);
    let graph = be.graph_capture(0, &mut || be.launch(name, &d.call, 0)).unwrap();
    for rep in 0..12 {
        let mut v = hc.clone();
        v.inputs[0] = HostTensor::from_f32_as(DType::Bf16, &v.inputs[0].shape, &rng.normal_vec(v.inputs[0].numel(), 1.0));
        v.inputs[1] = HostTensor::from_f32_as(DType::Bf16, &v.inputs[1].shape, &rng.normal_vec(v.inputs[1].numel(), 2.0));
        d.set(&*be, 0, &v.inputs[0], 0);
        d.set(&*be, 1, &v.inputs[1], 0);
        be.graph_replay(&graph, 0).unwrap();
        let got = d.out_on(&*be, 0);
        let hb = 2048 * 2;
        for r in 0..t {
            let want = run_host(&*be, K_NAME, &token_of(&v, r)).unwrap().remove(0).data;
            assert!(got[r * hb..(r + 1) * hb] == want[..], "replay {rep}: token {r} differs from the solo T = 1 block");
        }
    }
}

/// Device copies of a host call (inputs, a zeroed output, the declared scratch).
struct Dev {
    call: OpCall,
}

impl Dev {
    fn new(be: &dyn Backend, kernel: &str, hc: &HostCall) -> Dev {
        let put = |t: &HostTensor| {
            let buf = be.alloc(t.nbytes().max(16), MemKind::Device).unwrap();
            be.zero(&buf, 0).unwrap();
            be.upload(&buf, 0, &t.data, 0).unwrap();
            TensorRef::new(buf, t.dtype, &t.shape)
        };
        let inputs: Vec<TensorRef> = hc.inputs.iter().map(put).collect();
        let outputs: Vec<TensorRef> = hc.outputs.iter().map(|o| put(&HostTensor::zeros(o.dtype, &o.shape))).collect();
        let mut call = OpCall::new(hc.op, inputs, outputs, hc.attrs.clone());
        let k = be.kernels().into_iter().find(|k| k.name == kernel).unwrap();
        let need = k.scratch_bytes(&ShapeInfo::from_call(&call));
        if need > 0 {
            call.scratch = Some(TensorRef::new(be.alloc(need.max(16), MemKind::Device).unwrap(), DType::U8, &[need.max(16)]));
        }
        Dev { call }
    }
    fn out_on(&self, be: &dyn Backend, s: u32) -> Vec<u8> {
        let o = &self.call.outputs[0];
        let mut d = vec![0u8; o.nbytes()];
        be.synchronize(s).unwrap();
        be.download(&o.buf, o.offset, &mut d, s).unwrap();
        d
    }
    fn set(&self, be: &dyn Backend, j: usize, t: &HostTensor, s: u32) {
        be.upload(&self.call.inputs[j].buf, self.call.inputs[j].offset, &t.data, s).unwrap();
    }
}

/// `n` activation sets for a case (new x and logits each, so routing changes too), with the chain's output for each.
fn variants(be: &dyn Backend, hc: &HostCall, rng: &mut SplitMix64, n: usize) -> Vec<(HostTensor, HostTensor, Vec<u8>)> {
    (0..n)
        .map(|_| {
            let mut v = hc.clone();
            v.inputs[0] = HostTensor::from_f32_as(DType::Bf16, &v.inputs[0].shape, &rng.normal_vec(v.inputs[0].numel(), 1.0));
            v.inputs[1] = HostTensor::from_f32_as(DType::Bf16, &v.inputs[1].shape, &rng.normal_vec(v.inputs[1].numel(), 2.0));
            let want = chain(be, &v);
            (v.inputs[0].clone(), v.inputs[1].clone(), want)
        })
        .collect()
}

#[test]
fn moe_block_counters_reset_between_launches() {
    let Some(be) = device() else { return };
    let mut rng = SplitMix64::new(72);
    let a = block_case(&mut rng, 1, 256, 8, 2048, 512, 512, true, false);
    let b = block_case(&mut rng, 2, 64, 8, 1024, 256, 256, true, false);
    let (mut da, mut db) = (Dev::new(&*be, K_NAME, &a), Dev::new(&*be, K_NAME, &b));
    let big = if da.call.scratch.as_ref().unwrap().nbytes() >= db.call.scratch.as_ref().unwrap().nbytes() { &da } else { &db };
    let shared = big.call.scratch.clone();
    da.call.scratch = shared.clone();
    db.call.scratch = shared;
    let (va, vb) = (variants(&*be, &a, &mut rng, 3), variants(&*be, &b, &mut rng, 3));
    let mut n = 0usize;
    for round in 0..4 {
        for _ in 0..25 {
            da.set(&*be, 0, &va[n % 3].0, 0);
            da.set(&*be, 1, &va[n % 3].1, 0);
            be.launch(K_NAME, &da.call, 0).unwrap();
            db.set(&*be, 0, &vb[n % 3].0, 0);
            db.set(&*be, 1, &vb[n % 3].1, 0);
            be.launch(K_NAME, &db.call, 0).unwrap();
            n += 1;
        }
        let last = (n - 1) % 3;
        assert!(da.out_on(&*be, 0) == va[last].2, "round {round}: T=1 after back-to-back launches differs from the chain");
        assert!(db.out_on(&*be, 0) == vb[last].2, "round {round}: T=2 after back-to-back launches differs from the chain");
    }
}

#[test]
fn moe_block_graph_replay_is_bitwise_eager() {
    let Some(be) = device() else { return };
    assert!(be.supports_graphs());
    let mut rng = SplitMix64::new(73);
    let hc = block_case(&mut rng, 1, 256, 8, 2048, 512, 512, true, false);
    let d = Dev::new(&*be, K_NAME, &hc);
    let graph = be.graph_capture(0, &mut || be.launch(K_NAME, &d.call, 0)).unwrap();
    let vs = variants(&*be, &hc, &mut rng, 20);
    for (rep, v) in vs.iter().enumerate() {
        d.set(&*be, 0, &v.0, 0);
        d.set(&*be, 1, &v.1, 0);
        be.graph_replay(&graph, 0).unwrap();
        assert!(d.out_on(&*be, 0) == v.2, "replay {rep} differs from the chain run eagerly");
    }
}

/// Tickets drawn in an order unrelated to blockIdx (blocks sleep (blockIdx % 7) * SKEW before drawing): no hang, and every
/// launch bitwise the chain. With SKEW_ALL the producers are late as well.
#[test]
fn moe_block_dispatch_order() {
    let Some(be) = device() else { return };
    let mut rng = SplitMix64::new(74);
    let mut n = 0;
    for (t, e, k, h, i, is) in [(1usize, 256usize, 8usize, 2048usize, 512usize, 512usize), (2, 256, 8, 2048, 512, 512), (1, 16, 2, 256, 16, 32)] {
        let hc = block_case(&mut rng, t, e, k, h, i, is, true, false);
        let d = Dev::new(&*be, K_NAME, &hc);
        let vs = variants(&*be, &hc, &mut rng, 2);
        std::env::set_var("INTARSIA_HIP_MOE_BLOCK_SCRAMBLE", "1");
        for j in 0..30 {
            std::env::set_var("INTARSIA_HIP_MOE_BLOCK_SKEW", ["1", "4", "16"][j % 3]);
            if j % 2 == 1 {
                std::env::set_var("INTARSIA_HIP_MOE_BLOCK_SKEW_ALL", "1");
            } else {
                std::env::remove_var("INTARSIA_HIP_MOE_BLOCK_SKEW_ALL");
            }
            let v = &vs[j % 2];
            d.set(&*be, 0, &v.0, 0);
            d.set(&*be, 1, &v.1, 0);
            be.launch(K_NAME, &d.call, 0).unwrap();
            assert!(d.out_on(&*be, 0) == v.2, "t={t} e={e} launch {j}: scrambled tickets give a different output");
            n += 1;
        }
        for v in ["INTARSIA_HIP_MOE_BLOCK_SCRAMBLE", "INTARSIA_HIP_MOE_BLOCK_SKEW", "INTARSIA_HIP_MOE_BLOCK_SKEW_ALL"] {
            std::env::remove_var(v);
        }
    }
    eprintln!("moe_block_dispatch_order: {n} launches bitwise");
}

fn stress_rounds() -> usize {
    std::env::var("INTARSIA_HIP_MOE_BLOCK_STRESS_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(120)
}

/// One stress configuration: which producer is late (INTARSIA_HIP_MOE_BLOCK_SKEW_ROLE, or every producer), the skews cycled
/// per launch, the side instance on stream 1.
struct Stress {
    name: &'static str,
    role: Option<&'static str>,
    skews: [&'static str; 3],
    side: bool,
}

/// `rounds` launches per shape and configuration, h and sa poisoned with the bf16 NaN 0x7fc0 before every launch, PREREAD
/// on: returns (launches, failures); prints per configuration and shape the launches that differ (and carry NaN).
fn sentinel_stress(be: &dyn Backend, rng: &mut SplitMix64, cfgs: &[Stress], shapes: &[(usize, usize, usize, usize, usize, usize)], rounds: usize) -> (usize, Vec<String>) {
    let poison = |d: &Dev, s: u32| {
        let sc = d.call.scratch.as_ref().unwrap();
        let pat: Vec<u8> = (0..sc.nbytes()).map(|i| if i % 2 == 0 { 0xc0 } else { 0x7f }).collect();
        be.upload(&sc.buf, sc.offset, &pat, s).unwrap();
    };
    let nans = |v: &[u8]| v.chunks(2).filter(|c| (u16::from_le_bytes([c[0], c[1]]) & 0x7fff) > 0x7f80).count();
    let side = block_case(rng, 2, 256, 8, 2048, 512, 512, true, false);
    let side_want = chain(be, &side);
    let ds = Dev::new(be, K_NAME, &side);
    std::env::set_var("INTARSIA_HIP_MOE_BLOCK_PREREAD", "1");
    let (mut launches, mut bad) = (0usize, Vec::new());
    for &(t, e, k, h, i, is) in shapes {
        let hc = block_case(rng, t, e, k, h, i, is, true, false);
        let d = Dev::new(be, K_NAME, &hc);
        let vs = variants(be, &hc, rng, 2);
        for cfg in cfgs {
            match cfg.role {
                Some(r) => {
                    std::env::set_var("INTARSIA_HIP_MOE_BLOCK_SKEW_ROLE", r);
                    std::env::remove_var("INTARSIA_HIP_MOE_BLOCK_SKEW_ALL");
                }
                None => {
                    std::env::remove_var("INTARSIA_HIP_MOE_BLOCK_SKEW_ROLE");
                    std::env::set_var("INTARSIA_HIP_MOE_BLOCK_SKEW_ALL", "1");
                }
            }
            let mut tally = [(0usize, 0usize); 3];
            let mut side_bad = 0usize;
            for j in 0..rounds {
                std::env::set_var("INTARSIA_HIP_MOE_BLOCK_SKEW", cfg.skews[j % 3]);
                let v = &vs[j % 2];
                poison(&d, 0);
                d.set(be, 0, &v.0, 0);
                d.set(be, 1, &v.1, 0);
                if cfg.side {
                    poison(&ds, 1);
                    be.launch(K_NAME, &ds.call, 1).unwrap();
                    launches += 1;
                }
                be.launch(K_NAME, &d.call, 0).unwrap();
                launches += 1;
                let got = d.out_on(be, 0);
                if got != v.2 {
                    let n = nans(&got);
                    tally[j % 3].0 += 1;
                    tally[j % 3].1 += (n > 0) as usize;
                    bad.push(format!("[{}] t={t} e={e} k={k} h={h} i={i} is={is} launch {j}: differs ({n} NaN outputs)", cfg.name));
                }
                if cfg.side && ds.out_on(be, 1) != side_want {
                    side_bad += 1;
                    bad.push(format!("[{}] stream 1 launch {j}: differs", cfg.name));
                }
            }
            let per: Vec<String> = (0..3).map(|s| format!("skew {}: {} differ ({} NaN)", cfg.skews[s], tally[s].0, tally[s].1)).collect();
            eprintln!("  [{}] t={t} e={e} k={k} h={h} i={i} is={is}, {rounds} rounds: {}; stream 1: {side_bad} differ", cfg.name, per.join(", "));
        }
    }
    for v in ["INTARSIA_HIP_MOE_BLOCK_SKEW", "INTARSIA_HIP_MOE_BLOCK_SKEW_ALL", "INTARSIA_HIP_MOE_BLOCK_PREREAD", "INTARSIA_HIP_MOE_BLOCK_SKEW_ROLE"] {
        std::env::remove_var(v);
    }
    (launches, bad)
}

/// The sentinel stress of BOTH cross-block edges (routed gate_up -> down via h, shared glu -> down via sa), at the shapes and
/// skews that kill the v3 ordering mutants (step 0: T = 1, last role late; M0: T = 2, token 0's last role late).
#[test]
fn moe_block_sentinel_stress() {
    let Some(be) = device() else { return };
    let mut rng = SplitMix64::new(75);
    let sk = ["0", "20", "40"];
    let cfgs = [
        Stress { name: "routed last late", role: Some("0"), skews: sk, side: false },
        Stress { name: "shared last late", role: Some("1"), skews: sk, side: false },
        Stress { name: "routed tok0 last late", role: Some("2"), skews: sk, side: false },
        Stress { name: "shared tok0 last late", role: Some("3"), skews: sk, side: false },
        Stress { name: "routed tok0 last late, side", role: Some("2"), skews: sk, side: true },
        Stress { name: "shared tok0 last late, side", role: Some("3"), skews: sk, side: true },
        Stress { name: "every producer late, side", role: None, skews: ["0", "2", "6"], side: true },
    ];
    let shapes = [(1usize, 256usize, 8usize, 2048usize, 512usize, 512usize), (2, 256, 8, 2048, 512, 512), (1, 64, 8, 1024, 256, 256), (1, 16, 2, 256, 16, 32), (4, 16, 4, 128, 32, 48)];
    let (launches, bad) = sentinel_stress(&*be, &mut rng, &cfgs, &shapes, stress_rounds());
    eprintln!("moe_block_sentinel_stress: {launches} launches, {} differ from the chain", bad.len());
    assert!(bad.is_empty(), "{} of {launches} launches read h / sa early or stale:\n{}", bad.len(), bad[..bad.len().min(12)].join("\n"));
}

/// M1 microbench (INTARSIA_HIP_MOE_BLOCK_BENCH=1, else skipped): L layers (distinct weights, routing and activations) at the
/// 35B decode shape, captured as one graph of the default chain (5 launches per layer), of the v3 chain (4), and of the fused
/// op (1); also one graph per chain kernel alone and one of L tiny launches (the launch-boundary price inside a replay).
/// Prints us per layer, median of `reps` replays, rounds interleaved.
#[test]
fn moe_block_timing() {
    if std::env::var("INTARSIA_HIP_MOE_BLOCK_BENCH").as_deref() != Ok("1") {
        return;
    }
    let Some(be) = device() else { return };
    let layers: usize = std::env::var("INTARSIA_HIP_MOE_BLOCK_BENCH_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    let reps = 50usize;
    let mut rng = SplitMix64::new(76);
    let (t, e, k, h, i, is) = (1usize, 256usize, 8usize, 2048usize, 512usize, 512usize);
    let base = block_case(&mut rng, t, e, k, h, i, is, true, false);
    let dev_t = |t: &HostTensor| {
        let buf = be.alloc(t.nbytes().max(16), MemKind::Device).unwrap();
        be.upload(&buf, 0, &t.data, 0).unwrap();
        TensorRef::new(buf, t.dtype, &t.shape)
    };
    let out_t = |dt: DType, shape: &[usize]| dev_t(&HostTensor::zeros(dt, shape));
    let scratch_for = |kernel: &str, call: &OpCall| -> Option<TensorRef> {
        let kk = be.kernels().into_iter().find(|x| x.name == kernel).unwrap();
        let need = kk.scratch_bytes(&ShapeInfo::from_call(call));
        (need > 0).then(|| TensorRef::new(be.alloc(need.max(16), MemKind::Device).unwrap(), DType::U8, &[need.max(16)]))
    };
    // per layer: device inputs (weights re-uploaded per layer, so no layer's weights stay in a cache), the chain's calls
    struct Layer {
        calls: Vec<(&'static str, OpCall)>,    // the default chain
        calls_v3: Vec<(&'static str, OpCall)>, // the v3 chain
        fused: OpCall,
    }
    let mut ls = Vec::new();
    for _ in 0..layers {
        let mut hc = base.clone();
        hc.inputs[0] = HostTensor::from_f32_as(DType::Bf16, &[t, h], &rng.normal_vec(t * h, 1.0));
        hc.inputs[1] = HostTensor::from_f32_as(DType::Bf16, &[t, e], &rng.normal_vec(t * e, 2.0));
        let inp: Vec<TensorRef> = hc.inputs.iter().map(dev_t).collect();
        let (w, ids) = (out_t(DType::F32, &[t, k]), out_t(DType::I32, &[t, k]));
        let (routed, a, out, out2) = (out_t(DType::Bf16, &[t, h]), out_t(DType::Bf16, &[t, is]), out_t(DType::Bf16, &[t, h]), out_t(DType::Bf16, &[t, h]));
        let at = hc.attrs.clone();
        let topk = OpCall::new(Op::MoeTopk, vec![inp[1].clone()], vec![w.clone(), ids.clone()], at.clone());
        let mut ex = OpCall::new(
            Op::MoeExperts,
            vec![inp[0].clone(), ids.clone(), w.clone(), inp[3].clone(), inp[4].clone(), inp[5].clone(), inp[6].clone(), inp[7].clone()],
            vec![routed.clone()],
            at.clone(),
        );
        ex.scratch = scratch_for("moe_experts_bf16_v2", &ex);
        let glu = OpCall::new(Op::MatmulNvfp4Glu, vec![inp[0].clone(), inp[8].clone(), inp[9].clone(), inp[10].clone(), inp[11].clone(), inp[12].clone(), inp[13].clone()], vec![a.clone()], OpAttrs::default());
        let sdc = OpCall::new(Op::SharedDownCombine, vec![a.clone(), inp[14].clone(), inp[15].clone(), inp[16].clone(), routed.clone(), inp[2].clone()], vec![out.clone()], OpAttrs::default());
        let mut fused = OpCall::new(Op::MoeBlock, inp.clone(), vec![out2.clone()], at.clone());
        fused.scratch = scratch_for(K_NAME, &fused);
        ls.push(Layer {
            calls: vec![("moe_topk_bf16_v2", topk.clone()), ("moe_experts_bf16_v2", ex.clone()), ("matmul_nvfp4_glu_gemv", glu.clone()), ("shared_down_combine_nvfp4_gemv", sdc.clone())],
            calls_v3: vec![("moe_topk_bf16_v2", topk), ("moe_experts_bf16_v3", ex), ("matmul_nvfp4_glu_gemv", glu), ("shared_down_combine_nvfp4_gemv", sdc)],
            fused,
        });
    }
    // a tiny launch: moe_combine on one 256-element row (one block), for the price of a launch boundary in a replay
    let tiny = {
        let r = out_t(DType::Bf16, &[1, 256]);
        OpCall::new(Op::MoeCombine, vec![r.clone(), r.clone(), out_t(DType::Bf16, &[1, 1])], vec![out_t(DType::Bf16, &[1, 256])], OpAttrs::default())
    };
    let capture = |f: &dyn Fn() -> intarsia_core::Result<()>| be.graph_capture(0, &mut || f()).unwrap();
    let g_chain = capture(&|| {
        for l in &ls {
            for (n, c) in &l.calls {
                be.launch(n, c, 0)?;
            }
        }
        Ok(())
    });
    let g_v3 = capture(&|| {
        for l in &ls {
            for (n, c) in &l.calls_v3 {
                be.launch(n, c, 0)?;
            }
        }
        Ok(())
    });
    let g_fused = capture(&|| {
        for l in &ls {
            be.launch(K_NAME, &l.fused, 0)?;
        }
        Ok(())
    });
    let g_tiny = capture(&|| {
        for _ in 0..layers {
            be.launch("moe_combine_bf16", &tiny, 0)?;
        }
        Ok(())
    });
    let g_one: Vec<_> = (0..4)
        .map(|j| {
            capture(&|| {
                for l in &ls {
                    let (n, c) = &l.calls[j];
                    be.launch(n, c, 0)?;
                }
                Ok(())
            })
        })
        .collect();
    let time = |g: &intarsia_core::backend::GraphHandle| -> f64 {
        be.graph_replay(g, 0).unwrap();
        be.synchronize(0).unwrap();
        let mut v: Vec<f64> = (0..reps)
            .map(|_| {
                let t0 = Instant::now();
                be.graph_replay(g, 0).unwrap();
                be.synchronize(0).unwrap();
                t0.elapsed().as_secs_f64() * 1e6 / layers as f64
            })
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[reps / 2]
    };
    let names = ["chain (v2 default)", "chain (v3)", "moe_block (fused)", "tiny launch", "topk_v2 alone", "experts_v2 alone", "glu alone", "sdc alone"];
    let mut acc = vec![Vec::new(); names.len()];
    for _round in 0..3 {
        acc[0].push(time(&g_chain));
        acc[1].push(time(&g_v3));
        acc[2].push(time(&g_fused));
        acc[3].push(time(&g_tiny));
        for j in 0..4 {
            acc[4 + j].push(time(&g_one[j]));
        }
    }
    eprintln!("moe_block_timing: {layers} layers, {reps} replays per round, median us per layer, 3 rounds:");
    for (n, v) in names.iter().zip(&acc) {
        eprintln!("  {n:22} {}", v.iter().map(|x| format!("{x:8.2}")).collect::<Vec<_>>().join(" "));
    }
    // the fused output equals the chain's in every layer (one replay each)
    be.graph_replay(&g_chain, 0).unwrap();
    be.graph_replay(&g_fused, 0).unwrap();
    be.synchronize(0).unwrap();
    for (j, l) in ls.iter().enumerate() {
        let rd = |t: &TensorRef| {
            let mut d = vec![0u8; t.nbytes()];
            be.download(&t.buf, t.offset, &mut d, 0).unwrap();
            d
        };
        assert!(rd(&l.calls[3].1.outputs[0]) == rd(&l.fused.outputs[0]), "layer {j}: fused differs from the chain");
    }
}

/// l7 microbench (INTARSIA_HIP_MOE_ROWS_BENCH=1, else skipped): the batched MoE block at T = 5..16 rows (INTARSIA_HIP_MOE_ROWS_BENCH_T, default
/// "8,16"), L layers (INTARSIA_HIP_MOE_BLOCK_BENCH_LAYERS, default 16) with distinct weights, routing and activations, each variant one HIP graph,
/// median of 50 replays, 3 interleaved rounds, us per layer. Variants: (a) the unfused chain exactly as the model runs it after the gate logit
/// (topk, `moe_experts_bf16_rows`, shared gate and up `matmul_nvfp4_gemv_rows`, `silu_glu`, shared down, `moe_combine`); (b) topk + experts;
/// (c) the shared pieces + combine; (d) one `matmul_nvfp4_gemv_v2` launch at M = 4 per shared matrix, the optimistic floor of one weight pass;
/// (e) a tiny launch (the price of a launch boundary in a replay); (f) the fused `moe_block_bf16_rows`; (g) `moe_block_bf16_grouped` (lane l7). Also logs the distinct experts per layer.
#[test]
fn moe_rows_timing() {
    if std::env::var("INTARSIA_HIP_MOE_ROWS_BENCH").as_deref() != Ok("1") {
        return;
    }
    let Some(be) = device() else { return };
    let layers: usize = std::env::var("INTARSIA_HIP_MOE_BLOCK_BENCH_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    let ts: Vec<usize> = std::env::var("INTARSIA_HIP_MOE_ROWS_BENCH_T")
        .unwrap_or_else(|_| "8,16".into())
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect();
    let reps = 50usize;
    let mut rng = SplitMix64::new(77);
    let (e, k, h, i, is) = (256usize, 8usize, 2048usize, 512usize, 512usize);
    // the standard table runs the old grouped projections (INTARSIA_HIP_MOE_G2=0); the several-rows-per-warp configs follow (INTARSIA_HIP_MOE_ROWS_BENCH_G2)
    std::env::set_var("INTARSIA_HIP_MOE_G2", "0");
    let g2_cfgs: Vec<String> = std::env::var("INTARSIA_HIP_MOE_ROWS_BENCH_G2").unwrap_or_default().split(',').filter(|v| !v.is_empty()).map(String::from).collect();
    let base = block_case(&mut rng, 1, e, k, h, i, is, true, false);
    let dev_t = |t: &HostTensor| {
        let buf = be.alloc(t.nbytes().max(16), MemKind::Device).unwrap();
        be.upload(&buf, 0, &t.data, 0).unwrap();
        TensorRef::new(buf, t.dtype, &t.shape)
    };
    let out_t = |dt: DType, shape: &[usize]| dev_t(&HostTensor::zeros(dt, shape));
    let scratch_for = |kernel: &str, call: &OpCall| -> Option<TensorRef> {
        let kk = be.kernels().into_iter().find(|x| x.name == kernel).unwrap();
        let need = kk.scratch_bytes(&ShapeInfo::from_call(call));
        (need > 0).then(|| TensorRef::new(be.alloc(need.max(16), MemKind::Device).unwrap(), DType::U8, &[need.max(16)]))
    };
    // the weights of every layer are uploaded again, so no layer's weights stay in a cache
    for &t in &ts {
        struct Layer {
            full: Vec<(&'static str, OpCall)>,
            topk_ex: Vec<(&'static str, OpCall)>,
            shared: Vec<(&'static str, OpCall)>,
            floor: Vec<(&'static str, OpCall)>,
            fused: OpCall,
            grouped: OpCall,
            topk_ids: TensorRef,
        }
        let mut ls = Vec::new();
        for _ in 0..layers {
            let mut hc = base.clone();
            hc.inputs[0] = HostTensor::from_f32_as(DType::Bf16, &[t, h], &rng.normal_vec(t * h, 1.0));
            hc.inputs[1] = HostTensor::from_f32_as(DType::Bf16, &[t, e], &rng.normal_vec(t * e, 2.0));
            hc.inputs[2] = HostTensor::from_f32_as(DType::Bf16, &[t, 1], &rng.normal_vec(t, 2.0));
            let inp: Vec<TensorRef> = hc.inputs.iter().map(dev_t).collect();
            let (w, ids) = (out_t(DType::F32, &[t, k]), out_t(DType::I32, &[t, k]));
            let (routed, g, u, a, so, out, out2) = (
                out_t(DType::Bf16, &[t, h]),
                out_t(DType::Bf16, &[t, is]),
                out_t(DType::Bf16, &[t, is]),
                out_t(DType::Bf16, &[t, is]),
                out_t(DType::Bf16, &[t, h]),
                out_t(DType::Bf16, &[t, h]),
                out_t(DType::Bf16, &[t, h]),
            );
            let at = hc.attrs.clone();
            let topk = OpCall::new(Op::MoeTopk, vec![inp[1].clone()], vec![w.clone(), ids.clone()], at.clone());
            let mut ex = OpCall::new(
                Op::MoeExperts,
                vec![inp[0].clone(), ids.clone(), w.clone(), inp[3].clone(), inp[4].clone(), inp[5].clone(), inp[6].clone(), inp[7].clone()],
                vec![routed.clone()],
                at.clone(),
            );
            ex.scratch = scratch_for("moe_experts_bf16_rows", &ex);
            let mm = |x: &TensorRef, wi: usize, o: &TensorRef| OpCall::new(Op::MatmulNvfp4, vec![x.clone(), inp[wi].clone(), inp[wi + 1].clone(), inp[wi + 2].clone()], vec![o.clone()], OpAttrs::default());
            let (cg, cu) = (mm(&inp[0], 8, &g), mm(&inp[0], 11, &u));
            let silu = OpCall::new(Op::SiluGlu, vec![g.clone(), u.clone()], vec![a.clone()], OpAttrs::default());
            let cd = mm(&a, 14, &so);
            let comb = OpCall::new(Op::MoeCombine, vec![routed.clone(), so.clone(), inp[2].clone()], vec![out.clone()], OpAttrs::default());
            // the floor: the same three matrices at M = 4 through the v2 GEMV (one weight pass for 4 rows)
            let (x4, a4) = (out_t(DType::Bf16, &[4, h]), out_t(DType::Bf16, &[4, is]));
            let (g4, u4, d4) = (out_t(DType::Bf16, &[4, is]), out_t(DType::Bf16, &[4, is]), out_t(DType::Bf16, &[4, h]));
            let f = |x: &TensorRef, wi: usize, o: &TensorRef| (("matmul_nvfp4_gemv_v2"), mm(x, wi, o));
            let mut fused = OpCall::new(Op::MoeBlock, inp.clone(), vec![out2.clone()], at.clone());
            fused.scratch = scratch_for("moe_block_bf16_rows", &fused);
            let mut grouped = OpCall::new(Op::MoeBlock, inp.clone(), vec![out_t(DType::Bf16, &[t, h])], at.clone());
            grouped.scratch = scratch_for("moe_block_bf16_grouped", &grouped);
            let full = vec![
                ("moe_topk_bf16_v2", topk.clone()),
                ("moe_experts_bf16_rows", ex.clone()),
                ("matmul_nvfp4_gemv_rows", cg.clone()),
                ("matmul_nvfp4_gemv_rows", cu.clone()),
                ("silu_glu_bf16", silu.clone()),
                ("matmul_nvfp4_gemv_rows", cd.clone()),
                ("moe_combine_bf16", comb.clone()),
            ];
            ls.push(Layer {
                topk_ex: full[..2].to_vec(),
                shared: full[2..].to_vec(),
                full,
                floor: vec![f(&x4, 8, &g4), f(&x4, 11, &u4), f(&a4, 14, &d4)],
                fused,
                grouped,
                topk_ids: ids,
            });
        }
        let tiny = {
            let r = out_t(DType::Bf16, &[1, 256]);
            OpCall::new(Op::MoeCombine, vec![r.clone(), r.clone(), out_t(DType::Bf16, &[1, 1])], vec![out_t(DType::Bf16, &[1, 256])], OpAttrs::default())
        };
        let capture = |f: &dyn Fn() -> intarsia_core::Result<()>| be.graph_capture(0, &mut || f()).unwrap();
        let cap_all = |pick: &dyn Fn(&Layer) -> &Vec<(&'static str, OpCall)>| {
            capture(&|| {
                for l in &ls {
                    for (n, c) in pick(l) {
                        be.launch(n, c, 0)?;
                    }
                }
                Ok(())
            })
        };
        let g_full = cap_all(&|l| &l.full);
        let g_topk_ex = cap_all(&|l| &l.topk_ex);
        let g_shared = cap_all(&|l| &l.shared);
        let g_floor = cap_all(&|l| &l.floor);
        let g_tiny = capture(&|| {
            for _ in 0..layers {
                be.launch("moe_combine_bf16", &tiny, 0)?;
            }
            Ok(())
        });
        let g_fused = capture(&|| {
            for l in &ls {
                be.launch("moe_block_bf16_rows", &l.fused, 0)?;
            }
            Ok(())
        });
        let g_grouped = capture(&|| {
            for l in &ls {
                be.launch("moe_block_bf16_grouped", &l.grouped, 0)?;
            }
            Ok(())
        });
        let time = |g: &intarsia_core::backend::GraphHandle| -> f64 {
            be.graph_replay(g, 0).unwrap();
            be.synchronize(0).unwrap();
            let mut v: Vec<f64> = (0..reps)
                .map(|_| {
                    let t0 = Instant::now();
                    be.graph_replay(g, 0).unwrap();
                    be.synchronize(0).unwrap();
                    t0.elapsed().as_secs_f64() * 1e6 / layers as f64
                })
                .collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[reps / 2]
        };
        let names = ["(a) unfused chain", "(b) topk + experts", "(c) shared + combine", "(d) 3 x gemv_v2 M=4", "(e) tiny launch", "(f) fused rows block", "(g) grouped block"];
        let mut acc = vec![Vec::new(); names.len()];
        for _round in 0..3 {
            acc[0].push(time(&g_full));
            acc[1].push(time(&g_topk_ex));
            acc[2].push(time(&g_shared));
            acc[3].push(time(&g_floor));
            acc[4].push(time(&g_tiny));
            acc[5].push(time(&g_fused));
            acc[6].push(time(&g_grouped));
        }
        // distinct experts per layer (the routing of the topk graph's last replay)
        be.graph_replay(&g_topk_ex, 0).unwrap();
        be.synchronize(0).unwrap();
        let mut counts = Vec::new();
        for l in &ls {
            let mut d = vec![0u8; l.topk_ids.nbytes()];
            be.download(&l.topk_ids.buf, l.topk_ids.offset, &mut d, 0).unwrap();
            let mut seen = std::collections::BTreeSet::new();
            for c in d.chunks(4) {
                seen.insert(i32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            counts.push(seen.len());
        }
        let mean = counts.iter().sum::<usize>() as f64 / counts.len() as f64;
        eprintln!("moe_rows_timing T={t}: {layers} layers, {reps} replays per round, median us per layer, 3 rounds; distinct experts per layer mean {mean:.1} (min {} max {})", counts.iter().min().unwrap(), counts.iter().max().unwrap());
        for (n, v) in names.iter().zip(&acc) {
            eprintln!("  {n:24} {}", v.iter().map(|x| format!("{x:8.2}")).collect::<Vec<_>>().join(" "));
        }
        drop((g_full, g_topk_ex, g_shared, g_floor, g_tiny, g_fused, g_grouped));
        // the G2 configs: topk + experts (b) and the grouped block (g) again with another rows-per-warp mapping; the grouped output must stay bit for bit
        if !g2_cfgs.is_empty() {
            let rd = |t: &TensorRef| {
                let mut d = vec![0u8; t.nbytes()];
                be.download(&t.buf, t.offset, &mut d, 0).unwrap();
                d
            };
            let g_ref = capture(&|| {
                for l in &ls {
                    be.launch("moe_block_bf16_grouped", &l.grouped, 0)?;
                }
                Ok(())
            });
            be.graph_replay(&g_ref, 0).unwrap();
            be.synchronize(0).unwrap();
            let want: Vec<Vec<u8>> = ls.iter().map(|l| rd(&l.grouped.outputs[0])).collect();
            for cfg in &g2_cfgs {
                std::env::set_var("INTARSIA_HIP_MOE_G2", cfg);
                let g_b = cap_all(&|l| &l.topk_ex);
                let g_g = capture(&|| {
                    for l in &ls {
                        be.launch("moe_block_bf16_grouped", &l.grouped, 0)?;
                    }
                    Ok(())
                });
                let g_a = cap_all(&|l| &l.full);
                be.graph_replay(&g_g, 0).unwrap();
                be.synchronize(0).unwrap();
                for (j, l) in ls.iter().enumerate() {
                    assert!(rd(&l.grouped.outputs[0]) == want[j], "G2={cfg} T={t} layer {j}: the grouped block changed bits");
                }
                let (mut tb, mut tg, mut ta) = (Vec::new(), Vec::new(), Vec::new());
                for _round in 0..3 {
                    ta.push(time(&g_a));
                    tb.push(time(&g_b));
                    tg.push(time(&g_g));
                }
                let f = |v: &Vec<f64>| v.iter().map(|x| format!("{x:8.2}")).collect::<Vec<_>>().join(" ");
                eprintln!("  G2={cfg:<5} (a) unfused {}  (b) topk+experts {}  (g) grouped {}", f(&ta), f(&tb), f(&tg));
            }
            std::env::set_var("INTARSIA_HIP_MOE_G2", "0");
        }
    }
}
