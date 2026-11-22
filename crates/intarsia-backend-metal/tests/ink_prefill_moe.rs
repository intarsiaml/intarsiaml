#![cfg(target_os = "macos")]
//! Lane ink-prefill: the second / third-generation grouped prefill experts (inkprefill.rs, `INTARSIA_INK_MOE_NAX2`) against the
//! admitted first generation at the real shape (H 2304, I 1152, 128 experts, top-8, SiTU) on a 2048-row prefill chunk, NVFP4
//! and MXFP8. The admission sweep's MoE cases are at most 40 rows, so this is where the 64-row tiles run at a real shape. Bound:
//! the Metal MoE budget (`MOE_TOL`, max_abs 1.5e-2, rel_l2 1e-3) of each form's output against the first generation's.
//! `INTARSIA_INK_PREFILL_TRACE=1` prints the kernels that ran.

use intarsia_backend_metal::bench::Rng;
use intarsia_backend_metal::inkprefill::{set_form, NAX2, NAX3};
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::{DType, HostTensor, Op, OpAttrs, OpCall, ShapeInfo};

const E: usize = 128;
const K: usize = 8;
const H: usize = 2304;
const I: usize = 1152;
const T: usize = 2048;

fn tr(b: &Buffer, dt: DType, shape: &[usize]) -> TensorRef {
    TensorRef { buf: b.clone(), offset: 0, dtype: dt, shape: shape.to_vec() }
}

fn fill(be: &MetalBackend, bytes: usize, seed: u64, f: &dyn Fn(u64) -> u8) -> Buffer {
    let mut r = Rng(seed | 1);
    let v: Vec<u8> = (0..bytes).map(|_| f(r.next_u64())).collect();
    let b = be.alloc(bytes, MemKind::Device).unwrap();
    be.upload(&b, 0, &v, 0).unwrap();
    b
}

fn read_bf16(be: &MetalBackend, b: &Buffer, n: usize) -> Vec<f32> {
    let mut raw = vec![0u8; n * 2];
    be.download(b, 0, &mut raw, 0).unwrap();
    raw.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect()
}

#[test]
fn prefill_forms_match_the_first_generation_at_2048_rows() {
    let be = MetalBackend::new().expect("metal");
    if !be.has_metal4() {
        eprintln!("no metal4.0 library: skipped");
        return;
    }
    let nib = |r: u64| (r >> 40) as u8;
    let nvsc = |r: u64| 0x28 + ((r >> 40) % 8) as u8;
    let e4 = |r: u64| ((r >> 33) & 0x80) as u8 | ((4 + (r >> 41) % 7) as u8) << 3 | ((r >> 50) & 7) as u8;
    let e8 = |r: u64| 116 + ((r >> 40) % 4) as u8;
    let mut rng = intarsia_core::rng::SplitMix64::new(0x9a11);
    let xb = be.alloc(T * H * 2, MemKind::Device).unwrap();
    be.upload(&xb, 0, &HostTensor::from_f32_as(DType::Bf16, &[T, H], &rng.normal_vec(T * H, 1.0)).data, 0).unwrap();
    let (ids, ws) = intarsia_core::quant::cases::random_routing(&mut rng, T, E, K);
    let ib = be.alloc(T * K * 4, MemKind::Device).unwrap();
    be.upload(&ib, 0, &ids.data, 0).unwrap();
    let wb = be.alloc(T * K * 4, MemKind::Device).unwrap();
    let w8: Vec<f32> = ws.to_f32().iter().map(|v| v * 8.0).collect();
    be.upload(&wb, 0, &HostTensor::from_f32(&[T, K], &w8).data, 0).unwrap();
    let out = be.alloc(T * H * 2, MemKind::Device).unwrap();
    let mut failures = Vec::new();
    for mx in [false, true] {
        let w: Vec<TensorRef> = if mx {
            vec![
                tr(&fill(&be, E * 2 * I * H, 1, &e4), DType::Fp8E4m3, &[E, 2 * I, H]),
                tr(&fill(&be, E * 2 * I * H / 32, 2, &e8), DType::U8, &[E, 2 * I, H / 32]),
                tr(&fill(&be, E * H * I, 3, &e4), DType::Fp8E4m3, &[E, H, I]),
                tr(&fill(&be, E * H * I / 32, 4, &e8), DType::U8, &[E, H, I / 32]),
            ]
        } else {
            let gs = be.alloc(E * 12, MemKind::Device).unwrap();
            be.upload(&gs, 0, &vec![0.05f32; E * 3].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(), 0).unwrap();
            vec![
                tr(&fill(&be, E * 2 * I * H / 2, 1, &nib), DType::U8, &[E, 2 * I, H / 2]),
                tr(&fill(&be, E * 2 * I * H / 16, 2, &nvsc), DType::Fp8E4m3, &[E, 2 * I, H / 16]),
                tr(&fill(&be, E * H * I / 2, 3, &nib), DType::U8, &[E, H, I / 2]),
                tr(&fill(&be, E * H * I / 16, 4, &nvsc), DType::Fp8E4m3, &[E, H, I / 16]),
                tr(&gs, DType::F32, &[E, 3]),
            ]
        };
        let kname = if mx { "moe_experts_mxfp8_bf16" } else { "moe_experts_situ_bf16" };
        let kern = be.kernels().into_iter().find(|k| k.name == kname).unwrap();
        let mut ins = vec![tr(&xb, DType::Bf16, &[T, H]), tr(&ib, DType::I32, &[T, K]), tr(&wb, DType::F32, &[T, K])];
        ins.extend(w);
        let mut a = OpAttrs { top_k: K, ..OpAttrs::default() };
        (a.ink.situ, a.ink.gate_cap, a.ink.up_cap) = (true, 4.0, 25.0);
        let mut call =
            OpCall::new(if mx { Op::MoeExpertsMxfp8 } else { Op::MoeExperts }, ins, vec![tr(&out, DType::Bf16, &[T, H])], a);
        call.check().unwrap();
        let need = kern.scratch_bytes(&ShapeInfo::from_call(&call));
        call.scratch = Some(TensorRef::new(be.alloc(need.max(16), MemKind::Device).unwrap(), DType::U8, &[need]));
        let run = |form: Option<&str>| -> Vec<f32> {
            set_form(form);
            // poison the output first (bf16 0xFFFF = NaN), so a form that writes nothing cannot pass as the last one's result
            be.upload(&out, 0, &vec![0xFFu8; T * H * 2], 0).unwrap();
            be.launch(kname, &call, 0).unwrap();
            be.synchronize(0).unwrap();
            read_bf16(&be, &out, T * H)
        };
        let base = run(None);
        let f = mx as usize;
        let forms: Vec<String> = NAX2
            .iter()
            .filter(|&&(ff, _, s)| ff == f && (s.ends_with("_v0") || s.ends_with("_v1") || s.ends_with("_v3")))
            .map(|&(_, _, s)| s.to_string())
            .chain(NAX3.iter().filter(|&&(ff, ..)| ff == f).map(|&(_, _, _, s)| format!("nax3_{s}")))
            .collect();
        let nb: f64 = base.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        for form in &forms {
            let y = run(Some(form));
            let mut max_abs = 0f64;
            let mut d2 = 0f64;
            let mut nonfinite = 0usize;
            for (a, b) in y.iter().zip(&base) {
                if !a.is_finite() {
                    nonfinite += 1;
                    continue;
                }
                let d = (*a as f64 - *b as f64).abs();
                max_abs = max_abs.max(d);
                d2 += d * d;
            }
            let rel = d2.sqrt() / nb;
            let ok = nonfinite == 0 && max_abs <= 1.5e-2 && rel <= 1e-3;
            eprintln!("{} {form:20} max_abs {max_abs:.3e} rel_l2 {rel:.3e} nonfinite {nonfinite} {}", if mx { "mx" } else { "nv" }, if ok { "ok" } else { "FAIL" });
            if !ok {
                failures.push(format!("{} {form}", if mx { "mx" } else { "nv" }));
            }
        }
    }
    set_form(None);
    assert!(failures.is_empty(), "forms off the first generation: {failures:?}");
}
