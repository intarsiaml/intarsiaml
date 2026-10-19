//! `moe_experts_bf16_v3` (kernels/hip/moe.hip, `moe_experts_p1_v3_kernel`): the decode path of `moe_experts_bf16_v2` in one
//! launch (gate_up blocks and down + reduce blocks in one grid; the down blocks wait on a per-stream counter that the last
//! of them resets). The arithmetic is v2's, expression for expression, so:
//!
//! * `v3_is_bitwise_v2`: every output byte equals v2's at the 35B's decode shapes (T = 1, 2), at small / odd ones (I / 16
//!   below the 32 lanes, H / 16 past the prefetch window, row tails), with duplicated and invalid expert ids, and on the
//!   grouped prefill shapes v3 hands to v2 unchanged; plus the cpu reference within the admission budget.
//! * `v3_counters_reset_between_launches`: 200 back-to-back launches without a host sync, alternating two shapes (NG and
//!   ND differ), each output bitwise v2's: a counter left non-zero would let a later launch's down blocks read h early.
//! * `v3_waits_for_every_gate_up_role`: with INTARSIA_HIP_MOE_V3_SKEW the last gate_up role starts ~0.1 ms late; the output
//!   must still be v2's (a down role that waited for all but one gate_up role would read that role's h rows unwritten);
//! * `v3_graph_replay_is_bitwise_eager`: one captured launch replayed 20 times with new activations uploaded between
//!   replays (the decode graph's use) equals v2 run eagerly every time.
//!
//! Needs a GPU: skips (passes) without one.
#![cfg(ik_hip)]

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::exec::{run_host, HostCall};
use intarsia_core::quant::cases::experts_case;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{Backend, DType, HostTensor, MemKind, Op, OpCall, ShapeInfo, TensorRef};
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

fn same_as_v2(hip: &dyn Backend, hc: &HostCall, label: &str) {
    let cpu = CpuBackend::new();
    let reference = cpu.kernels().into_iter().find(|k| k.op == Op::MoeExperts && k.is_reference).unwrap().name;
    let a = run_host(hip, "moe_experts_bf16_v2", hc).unwrap_or_else(|e| panic!("{label}: v2: {e}"));
    let b = run_host(hip, "moe_experts_bf16_v3", hc).unwrap_or_else(|e| panic!("{label}: v3: {e}"));
    assert!(a[0].data == b[0].data, "{label}: v3 differs from v2");
    let want = run_host(&cpu, &reference, hc).unwrap();
    let r = rel_l2(&b[0].to_f32(), &want[0].to_f32());
    assert!(r < 1e-3, "{label}: v3 vs {reference}: rel_l2 {r}");
}

#[test]
fn v3_is_bitwise_v2() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(51);
    for (t, e, k, h, i) in [
        (1usize, 256usize, 8usize, 2048usize, 512usize),
        (2, 256, 8, 2048, 512),
        (2, 64, 8, 2048, 512),
        (1, 32, 4, 256, 48),
        (1, 16, 2, 4112, 160),
        (2, 16, 8, 64, 16),
        (1, 16, 8, 208, 64),
        (3, 16, 5, 128, 32),
        (4, 32, 8, 512, 128),
        (40, 16, 4, 256, 64),
        (64, 256, 8, 2048, 512),
    ] {
        let hc = experts_case(&mut rng, t, e, k, h, i, DType::Bf16);
        same_as_v2(&*hip, &hc, &format!("experts t={t} e={e} k={k} h={h} i={i}"));
    }
    // routing the sweep does not draw: both tokens on the same experts, a duplicated expert, out-of-range ids
    let base = experts_case(&mut rng, 2, 16, 4, 256, 64, DType::Bf16);
    let mut odd = base.clone();
    let mut ids = odd.inputs[1].to_i32().unwrap();
    ids[0] = 3;
    ids[1] = 3;
    ids[2] = -1;
    ids[3] = 16 + 5;
    ids[4] = 3;
    odd.inputs[1] = HostTensor::from_i32(&[2, 4], &ids);
    same_as_v2(&*hip, &odd, "duplicate and invalid ids");
}

/// Device copies of a host call (inputs, a zeroed output, the declared scratch) and the launch of one kernel on stream 0.
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
        call.scratch =
            Some(TensorRef::new(be.alloc(need.max(16), MemKind::Device).unwrap(), DType::U8, &[need.max(16)]));
        Dev { call }
    }
    fn out(&self, be: &dyn Backend) -> Vec<u8> {
        let o = &self.call.outputs[0];
        let mut d = vec![0u8; o.nbytes()];
        be.synchronize(0).unwrap();
        be.download(&o.buf, o.offset, &mut d, 0).unwrap();
        d
    }
}

/// Two calls of different shapes (NG and ND differ) share ONE scratch, as the model's 40 MoE layers share the MoE
/// workspace, and every launch gets new activations: the h a down block would read early (a counter not reset, a wait
/// threshold short by one) is the previous launch's, of the other call, so it gives wrong values, not the right ones.
/// 4 rounds of 25 launch pairs without a host sync in between (only the last pair of a round is compared) and 20 pairs
/// with a sync and a comparison after each.
#[test]
fn v3_counters_reset_between_launches() {
    let Some(be) = device() else { return };
    let mut rng = SplitMix64::new(52);
    let a = experts_case(&mut rng, 1, 256, 8, 2048, 512, DType::Bf16);
    let b = experts_case(&mut rng, 2, 64, 8, 1024, 256, DType::Bf16);
    let (mut da, mut db) = (Dev::new(&*be, "moe_experts_bf16_v3", &a), Dev::new(&*be, "moe_experts_bf16_v3", &b));
    // one scratch for both: the larger of the two
    let big = if da.call.scratch.as_ref().unwrap().nbytes() >= db.call.scratch.as_ref().unwrap().nbytes() {
        &da
    } else {
        &db
    };
    let shared = big.call.scratch.clone();
    da.call.scratch = shared.clone();
    db.call.scratch = shared;
    // a few activation sets per call, with v2's outputs for each
    let variants = |hc: &HostCall, rng: &mut SplitMix64| -> Vec<(HostTensor, Vec<u8>)> {
        (0..3)
            .map(|_| {
                let mut h = hc.clone();
                h.inputs[0] =
                    HostTensor::from_f32_as(DType::Bf16, &h.inputs[0].shape, &rng.normal_vec(h.inputs[0].numel(), 1.0));
                let want = run_host(&*be, "moe_experts_bf16_v2", &h).unwrap().remove(0).data;
                (h.inputs[0].clone(), want)
            })
            .collect()
    };
    let (va, vb) = (variants(&a, &mut rng), variants(&b, &mut rng));
    let set_x =
        |d: &Dev, x: &HostTensor| be.upload(&d.call.inputs[0].buf, d.call.inputs[0].offset, &x.data, 0).unwrap();
    let mut n = 0usize;
    for round in 0..4 {
        for _ in 0..25 {
            set_x(&da, &va[n % 3].0);
            be.launch("moe_experts_bf16_v3", &da.call, 0).unwrap();
            set_x(&db, &vb[n % 3].0);
            be.launch("moe_experts_bf16_v3", &db.call, 0).unwrap();
            n += 1;
        }
        let last = (n - 1) % 3;
        assert!(da.out(&*be) == va[last].1, "round {round}: T=1 output after back-to-back launches differs from v2");
        assert!(db.out(&*be) == vb[last].1, "round {round}: T=2 output after back-to-back launches differs from v2");
    }
    for j in 0..20 {
        set_x(&da, &va[j % 3].0);
        be.launch("moe_experts_bf16_v3", &da.call, 0).unwrap();
        assert!(da.out(&*be) == va[j % 3].1, "launch {j}: T=1 differs from v2");
        set_x(&db, &vb[(j + 1) % 3].0);
        be.launch("moe_experts_bf16_v3", &db.call, 0).unwrap();
        assert!(db.out(&*be) == vb[(j + 1) % 3].1, "launch {j}: T=2 differs from v2");
    }
}

#[test]
fn v3_waits_for_every_gate_up_role() {
    let Some(be) = device() else { return };
    let mut rng = SplitMix64::new(54);
    for (t, e, k, h, i) in [(1usize, 256usize, 8usize, 2048usize, 512usize), (2, 64, 8, 2048, 512), (1, 16, 2, 256, 16)]
    {
        let a = experts_case(&mut rng, t, e, k, h, i, DType::Bf16);
        let mut b = a.clone();
        b.inputs[0] =
            HostTensor::from_f32_as(DType::Bf16, &b.inputs[0].shape, &rng.normal_vec(b.inputs[0].numel(), 1.0));
        let want = run_host(&*be, "moe_experts_bf16_v2", &b).unwrap().remove(0).data;
        // one scratch: the launch before leaves its own h behind, so an early read gives the wrong values, not zeros
        let d = Dev::new(&*be, "moe_experts_bf16_v3", &a);
        be.launch("moe_experts_bf16_v3", &d.call, 0).unwrap();
        be.upload(&d.call.inputs[0].buf, d.call.inputs[0].offset, &b.inputs[0].data, 0).unwrap();
        std::env::set_var("INTARSIA_HIP_MOE_V3_SKEW", "40");
        be.launch("moe_experts_bf16_v3", &d.call, 0).unwrap();
        let got = d.out(&*be);
        std::env::remove_var("INTARSIA_HIP_MOE_V3_SKEW");
        assert!(got == want, "t={t} e={e} k={k} h={h} i={i}: with the last gate_up role late, v3 differs from v2");
    }
}

#[test]
fn v3_graph_replay_is_bitwise_eager() {
    let Some(be) = device() else { return };
    assert!(be.supports_graphs());
    let mut rng = SplitMix64::new(53);
    let hc = experts_case(&mut rng, 1, 256, 8, 2048, 512, DType::Bf16);
    let d = Dev::new(&*be, "moe_experts_bf16_v3", &hc);
    let graph = be.graph_capture(0, &mut || be.launch("moe_experts_bf16_v3", &d.call, 0)).unwrap();
    for rep in 0..20 {
        // new activations (and routing weights) for this replay, as the decode graph gets a new token every step
        let mut h = hc.clone();
        h.inputs[0] =
            HostTensor::from_f32_as(DType::Bf16, &h.inputs[0].shape, &rng.normal_vec(h.inputs[0].numel(), 1.0));
        be.upload(&d.call.inputs[0].buf, d.call.inputs[0].offset, &h.inputs[0].data, 0).unwrap();
        be.graph_replay(&graph, 0).unwrap();
        let got = d.out(&*be);
        let want = run_host(&*be, "moe_experts_bf16_v2", &h).unwrap().remove(0).data;
        assert!(got == want, "replay {rep} differs from v2 run eagerly");
    }
}
