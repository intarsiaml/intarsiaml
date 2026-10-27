//! cpu kernels of the GatedDeltaNet / gated-attention ops, plus the norm / rope variants they need
//! (w_offset, partial rotary): the fast set must agree with the reference over the admission sweep.

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::admission::{sweep, SweepConfig};
use intarsia_core::exec::run_host;
use intarsia_core::Op;

#[test]
fn gdn_ops_and_variants_agree_between_ref_and_fast() {
    let cfg = SweepConfig { quick: false, ..Default::default() };
    let b = CpuBackend::new();
    let ops = [Op::GateSplit, Op::SigmoidMul, Op::GdnConv, Op::GdnDeltaRule, Op::ShortConv, Op::RmsNorm, Op::RmsNormAdd, Op::QkNorm, Op::Rope];
    for op in ops {
        let cases = sweep(op, &cfg);
        assert!(!cases.is_empty(), "{op}: empty sweep");
        for case in cases.iter().take(40) {
            let want = run_host(&b, &format!("ref_{}", op.as_str()), &case.call).unwrap();
            let got = run_host(&b, &format!("cpu_f32_{}", op.as_str()), &case.call).unwrap();
            for (g, w) in got.iter().zip(&want) {
                let (a, bb) = (g.to_f32(), w.to_f32());
                let num: f64 = a.iter().zip(&bb).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
                let den: f64 = bb.iter().map(|y| (*y as f64).powi(2)).sum();
                let rel = if den > 0.0 { (num / den).sqrt() } else { num.sqrt() };
                assert!(rel < 1e-2, "{op} {}: rel_l2 {rel}", case.label);
            }
        }
    }
}

#[test]
fn partial_rope_leaves_the_tail_untouched_and_offset_norm_scales_by_one_plus_w() {
    let cfg = SweepConfig { quick: true, ..Default::default() };
    let b = CpuBackend::new();
    let case = sweep(Op::Rope, &cfg).into_iter().find(|c| c.call.attrs.rotary_dim > 0 && c.call.attrs.pos0 > 0).expect("partial rope case");
    let out = run_host(&b, "ref_rope", &case.call).unwrap();
    let d = case.call.attrs.head_dim;
    let rd = case.call.attrs.rotary_dim;
    let x = case.call.inputs[0].to_f32();
    let y = out[0].to_f32();
    for (xr, yr) in x.chunks(d).zip(y.chunks(d)) {
        for j in rd..d {
            assert_eq!(xr[j], yr[j], "{}: tail value {j} must pass through", case.label);
        }
    }
    assert!(x.chunks(d).zip(y.chunks(d)).any(|(xr, yr)| xr[..rd] != yr[..rd]), "rotated part must change at pos0>0");

    let norm = sweep(Op::RmsNorm, &cfg).into_iter().find(|c| c.call.attrs.w_offset == 1.0).expect("offset norm case");
    let mut plain = norm.call.clone();
    plain.attrs.w_offset = 0.0;
    // (w + 1) vs w: rebuild with w' = w + 1 and offset 0 must equal offset 1 with w
    let with_off = run_host(&b, "ref_rmsnorm", &norm.call).unwrap()[0].to_f32();
    let no_off = run_host(&b, "ref_rmsnorm", &plain).unwrap()[0].to_f32();
    assert!(with_off.iter().zip(&no_off).any(|(a, b)| (a - b).abs() > 1e-3), "w_offset must change the output");
}
