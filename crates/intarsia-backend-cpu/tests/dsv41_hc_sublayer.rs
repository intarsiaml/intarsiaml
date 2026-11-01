//! `Op::HcSublayer` is exactly the op sequence it fuses: `[HcPost ->] HcMixes, HcPre (previous pre-mix), RmsNorm`, every
//! intermediate stored in the dtype the graph stores it in. The cpu reference of the fused op must match running the separate
//! cpu references in sequence bit for bit (the graph switches between the two, docs/design/dsv41-ops.md).

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, Op, OpAttrs};

fn rand_t(rng: &mut SplitMix64, dt: DType, shape: &[usize], scale: f32) -> HostTensor {
    let n: usize = shape.iter().product();
    HostTensor::from_f32_as(dt, shape, &rng.normal_vec(n, scale))
}

#[test]
fn hc_sublayer_equals_the_separate_ops_bit_for_bit() {
    let cpu = CpuBackend::new();
    let run = |op: Op, inputs: Vec<HostTensor>, outputs: Vec<OutSpec>, attrs: &OpAttrs| {
        run_host(&cpu, &format!("ref_{}", op.as_str()), &HostCall { op, inputs, outputs, attrs: attrs.clone() }).unwrap()
    };
    for dt in [DType::Bf16, DType::F32] {
        for (t, s, d) in [(1usize, 4usize, 256usize), (5, 4, 64), (3, 2, 96)] {
            for fold in [false, true] {
                let mut rng = SplitMix64::new(0x5b1 ^ (t * 31 + d) as u64 ^ fold as u64);
                let r = &mut rng;
                let m = (2 + s) * s;
                let x = rand_t(r, dt, &[t, s, d], 1.0);
                let (f, sc, base) = (rand_t(r, DType::F32, &[m, s * d], 0.05), rand_t(r, DType::F32, &[3], 0.5), rand_t(r, DType::F32, &[m], 0.5));
                let pre_in = rand_t(r, DType::F32, &[t, s], 0.7);
                let w = rand_t(r, dt, &[d], 1.0);
                let (px, pm, pc) = (rand_t(r, dt, &[t, d], 1.0), rand_t(r, DType::F32, &[t, s], 1.0), rand_t(r, DType::F32, &[t, s, s], 0.5));
                let mut a = OpAttrs { eps: 1e-6, ..OpAttrs::default() };
                (a.dsv.iters, a.dsv.hc_eps) = (20, 1e-6);

                // separate ops, as the graph runs them
                let h = if fold { run(Op::HcPost, vec![px.clone(), x.clone(), pm.clone(), pc.clone()], vec![OutSpec::new(dt, &[t, s, d])], &a).remove(0) } else { x.clone() };
                let mix = run(Op::HcMixes, vec![h.clone(), f.clone(), sc.clone(), base.clone()], vec![OutSpec::new(DType::F32, &[t, s]), OutSpec::new(DType::F32, &[t, s]), OutSpec::new(DType::F32, &[t, s, s])], &a);
                let y = run(Op::HcPre, vec![h.clone(), pre_in.clone()], vec![OutSpec::new(dt, &[t, d])], &a).remove(0);
                let n = run(Op::RmsNorm, vec![y, w.clone()], vec![OutSpec::new(dt, &[t, d])], &a).remove(0);

                // fused
                let (fold_in, h_out) = if fold {
                    (vec![px, pm, pc], OutSpec::new(dt, &[t, s, d]))
                } else {
                    (vec![HostTensor::zeros(dt, &[0]), HostTensor::zeros(DType::F32, &[0]), HostTensor::zeros(DType::F32, &[0])], OutSpec::new(dt, &[0]))
                };
                let mut inputs = vec![x, f, sc, base, pre_in, w];
                inputs.extend(fold_in);
                let outs = vec![OutSpec::new(dt, &[t, d]), OutSpec::new(DType::F32, &[t, s]), OutSpec::new(DType::F32, &[t, s]), OutSpec::new(DType::F32, &[t, s, s]), h_out];
                let got = run(Op::HcSublayer, inputs, outs, &a);

                let what = format!("{dt} T={t} S={s} D={d} fold={fold}");
                assert_eq!(got[0].data, n.data, "n {what}");
                assert_eq!(got[1].data, mix[0].data, "pre {what}");
                assert_eq!(got[2].data, mix[1].data, "post {what}");
                assert_eq!(got[3].data, mix[2].data, "comb {what}");
                if fold {
                    assert_eq!(got[4].data, h.data, "h_out {what}");
                }
            }
        }
    }
}
