//! `InkRoute` (op 76, lane ink-dfuse) is defined as `MoeRoute(Logits(x, w), bias)`: on both cpu kernel sets the fused op
//! equals the pair run in sequence, bit for bit, on every admission case of the op (weights, ids and the logits when given).

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::admission::SweepConfig;
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::{DType, Op};

#[test]
fn ink_route_is_logits_then_moe_route_bitwise() {
    let be = CpuBackend::arc();
    let cases = intarsia_core::inkling2::cases::sweep(Op::InkRoute, &SweepConfig { quick: false, ..Default::default() });
    assert!(cases.len() > 40, "{} cases", cases.len());
    for set in ["ref", "cpu_f32"] {
        for c in &cases {
            let hc = &c.call;
            let fused = run_host(&*be, &format!("{set}_ink_route"), hc).unwrap();
            let (t, e) = (hc.inputs[0].shape[0], hc.inputs[1].shape[0]);
            let lg = HostCall {
                op: Op::Logits,
                inputs: hc.inputs[..2].to_vec(),
                outputs: vec![OutSpec::new(DType::F32, &[t, e])],
                attrs: Default::default(),
            };
            let l = run_host(&*be, &format!("{set}_logits"), &lg).unwrap();
            let rt = HostCall { op: Op::MoeRoute, inputs: vec![l[0].clone(), hc.inputs[2].clone()], outputs: hc.outputs[..2].to_vec(), attrs: hc.attrs.clone() };
            let r = run_host(&*be, &format!("{set}_moe_route"), &rt).unwrap();
            assert_eq!(fused[0].data, r[0].data, "{set} {}: weights", c.label);
            assert_eq!(fused[1].data, r[1].data, "{set} {}: ids", c.label);
            if hc.outputs[2].shape != [0] {
                assert_eq!(fused[2].data, l[0].data, "{set} {}: logits", c.label);
            }
        }
    }
    eprintln!("{} cases bitwise on both kernel sets", cases.len());
}
