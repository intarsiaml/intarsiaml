//! `InkAttnDecode` (lane ink-m-attn) is defined as `RelSinkAttn(InkKvPrologue(..))`: on the cpu reference the fused op equals
//! the pair run in sequence, bit for bit, on every admission case of the op (outputs o, both caches and both conv rings).

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::admission::SweepConfig;
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::{DType, Op};

#[test]
fn fused_decode_attention_is_the_pair_bitwise() {
    let be = CpuBackend::arc();
    let cases = intarsia_core::inkling2::cases::sweep(Op::InkAttnDecode, &SweepConfig { quick: false, ..Default::default() });
    assert!(cases.len() > 40, "{} cases", cases.len());
    for c in &cases {
        let hc = &c.call;
        let fused = run_host(&*be, "ref_ink_attn_decode", hc).unwrap();
        let (t, a) = (hc.inputs[0].shape[0], &hc.attrs);
        let e = hc.inputs[7].shape[1];
        let pro = HostCall {
            op: Op::InkKvPrologue,
            inputs: hc.inputs[..8].to_vec(),
            outputs: vec![
                OutSpec::new(DType::Bf16, &[t, a.n_heads, a.head_dim]),
                OutSpec::new(DType::Bf16, &[t, a.n_heads, e]),
                OutSpec::new(DType::Bf16, &[t, a.n_kv_heads, a.head_dim]),
                OutSpec::new(DType::Bf16, &[t, a.n_kv_heads, a.head_dim]),
                hc.outputs[3].clone(),
                hc.outputs[4].clone(),
            ],
            attrs: a.clone(),
        };
        let p = run_host(&*be, "ref_ink_kv_prologue", &pro).unwrap();
        let att = HostCall {
            op: Op::RelSinkAttn,
            inputs: vec![p[0].clone(), p[2].clone(), p[3].clone(), hc.inputs[8].clone(), hc.inputs[9].clone(), p[1].clone(), hc.inputs[0].clone()],
            outputs: hc.outputs[..3].to_vec(),
            attrs: a.clone(),
        };
        let q = run_host(&*be, "ref_rel_sink_attn", &att).unwrap();
        let pair = [&q[0], &q[1], &q[2], &p[4], &p[5]];
        for (j, (f, w)) in fused.iter().zip(pair).enumerate() {
            assert_eq!(f.data, w.data, "{}: output {j}", c.label);
        }
    }
    eprintln!("{} cases bitwise", cases.len());
}
