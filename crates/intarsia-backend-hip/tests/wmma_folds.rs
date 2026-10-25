//! Lane wg: the WMMA family's decode folds against the chain they replace. With INTARSIA_HIP_WMMA_ROWS=1 a solo step runs the `*_wmma`
//! folds at M = 1 while a batched step runs the unfused chain through `matmul_*_wmma_rows`; the two must agree bit for bit or a
//! batched row differs from its solo decode. Each fold here is compared with its prologue launch followed by `matmul_fp8_wmma` /
//! `matmul_bf16_wmma` (the same launchers the rows registrations use), W8A8 and W8A16, with segments of different scales.
//! Run it three times: default, `INTARSIA_HIP_WMMA_DIRECT_MIN=1` and `=65` (staged vs direct activations).
//! Needs a GPU: skips (passes) without one.
#![cfg(ik_hip)]

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::backend::Backend;
use intarsia_core::exec::{run_host as run_host_plain, run_host_frag, HostCall, OutSpec};
use intarsia_core::quant::cases::{
    gated_rmsnorm_matmul_fp8_case, matmul_fp8_pair_case, matmul_pair_case, rmsnorm_add_matmul_fp8_pair_case, rmsnorm_add_matmul_pair_case,
    sigmoid_mul_matmul_fp8_case,
};
use intarsia_core::quant::fp8::scales_tensor;
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

/// Every launch of this file. With INTARSIA_WG_FRAG=1 a HIP launch also runs on weights rearranged into fragment order (`Layout::Fp8Frag`) and
/// must give the bytes of the row-major run (same sums, other addresses); the tagged outputs are returned. Run the file once with it set.
fn run_host(be: &dyn Backend, kernel: &str, hc: &HostCall) -> intarsia_core::Result<Vec<HostTensor>> {
    let plain = run_host_plain(be, kernel, hc)?;
    if be.name() != "hip" || std::env::var("INTARSIA_WG_FRAG").as_deref() != Ok("1") {
        return Ok(plain);
    }
    let tagged = run_host_frag(be, kernel, hc)?;
    for (i, (t, p)) in tagged.iter().zip(&plain).enumerate() {
        assert!(t.data == p.data, "{kernel}: output {i} on fragment-order weights differs from the row-major run");
    }
    Ok(tagged)
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

fn call(op: Op, inputs: Vec<HostTensor>, outputs: Vec<OutSpec>, attrs: OpAttrs) -> HostCall {
    HostCall { op, inputs, outputs, attrs }
}

fn bits_eq(label: &str, what: &str, got: &HostTensor, want: &HostTensor) {
    assert_eq!(got.shape, want.shape, "{label}: {what} shape");
    if got.data != want.data {
        let (g, w) = (got.to_f32(), want.to_f32());
        let bad = g.iter().zip(&w).enumerate().filter(|(_, (a, b))| a.to_bits() != b.to_bits()).count();
        let first = g.iter().zip(&w).position(|(a, b)| a.to_bits() != b.to_bits()).unwrap();
        panic!("{label}: {what} differs from the unfused launches in {bad} of {} elements (first at {first}: {} vs {})", g.len(), g[first], w[first]);
    }
}

/// The fused kernel's outputs are within `rel` of the cpu reference kernel (admission-style check).
fn near_reference(op: Op, hc: &HostCall, got: &[HostTensor], rel: f64, label: &str) {
    let cpu = CpuBackend::new();
    let reference = cpu.kernels().into_iter().find(|k| k.op == op && k.is_reference).unwrap().name;
    let want = run_host(&cpu, &reference, hc).unwrap();
    for (o, (g, w)) in got.iter().zip(&want).enumerate() {
        let r = rel_l2(&g.to_f32(), &w.to_f32());
        assert!(r < rel, "{label}: output {o} vs {reference}: rel_l2 {r}");
    }
}

fn out_bf16(shape: &[usize]) -> OutSpec {
    OutSpec::new(DType::Bf16, shape)
}


#[test]
fn matmul_fp8_pair_wmma_is_bitwise_two_launches() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(91);
    for (n0, n1, k) in [(8192usize, 4096usize, 2048usize), (512, 512, 2048), (64, 48, 64), (17, 33, 48), (130, 7, 1040), (1, 1, 16), (4096, 2048, 4096), (16, 16, 16)] {
        for (aq0, aq1, is1) in [(true, true, 0.125f32), (true, true, 0.3), (false, false, 0.0), (true, false, 0.0), (false, true, 0.25)] {
            let mut hc = matmul_fp8_pair_case(&mut rng, 1, n0, n1, k, aq0, aq1, DType::Bf16, DType::Bf16);
            let s = hc.inputs[4].to_f32();
            hc.inputs[4] = scales_tensor(s[0] * 1.7, if aq1 { is1 } else { 0.0 });
            let label = format!("wmma fp8 pair n0={n0} n1={n1} k={k} aq={aq0}/{aq1} is1={is1}");
            let got = run_host(&*hip, "matmul_fp8_pair_wmma", &hc).unwrap();
            for (j, w, sc) in [(0usize, 1usize, 2usize), (1, 3, 4)] {
                let one = call(Op::MatmulFp8, vec![hc.inputs[0].clone(), hc.inputs[w].clone(), hc.inputs[sc].clone()], vec![hc.outputs[j].clone()], OpAttrs::default());
                let want = run_host(&*hip, "matmul_fp8_wmma", &one).unwrap();
                bits_eq(&label, &format!("output {j}"), &got[j], &want[0]);
            }
            near_reference(Op::MatmulFp8Pair, &hc, &got, 6e-3, &label);
        }
    }
}

#[test]
fn rmsnorm_add_matmul_fp8_pair_wmma_is_bitwise_norm_then_two_launches() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(92);
    for (n0, n1, k) in [(512usize, 512usize, 2048usize), (24, 40, 1040), (64, 48, 64), (17, 33, 48), (9, 300, 1024), (1024, 1024, 2048)] {
        for (aq0, aq1, w_offset, second) in [(true, true, 1.0f32, None), (false, false, 0.0, None), (true, false, 1.0, None), (true, true, 1.0, Some((3.0f32, 0.02f32)))] {
            let mut hc = rmsnorm_add_matmul_fp8_pair_case(&mut rng, 1, n0, n1, k, aq0, aq1, DType::Bf16, w_offset);
            if let Some((f, is)) = second {
                let s = hc.inputs[6].to_f32();
                hc.inputs[6] = scales_tensor(s[0] * f, is);
            }
            let label = format!("wmma P5 n0={n0} n1={n1} k={k} aq={aq0}/{aq1} w_offset={w_offset} second={second:?}");
            let got = run_host(&*hip, "rmsnorm_add_matmul_fp8_pair_wmma", &hc).unwrap_or_else(|e| panic!("{label}: {e}"));
            let norm = call(Op::RmsNormAdd, hc.inputs[..3].to_vec(), hc.outputs[..2].to_vec(), hc.attrs.clone());
            let nw = run_host(&*hip, "rmsnorm_add_bf16", &norm).unwrap();
            bits_eq(&label, "normed", &got[0], &nw[0]);
            bits_eq(&label, "sum", &got[1], &nw[1]);
            for (j, w, sc) in [(2usize, 3usize, 4usize), (3, 5, 6)] {
                let one = call(Op::MatmulFp8, vec![nw[0].clone(), hc.inputs[w].clone(), hc.inputs[sc].clone()], vec![hc.outputs[j].clone()], OpAttrs::default());
                let want = run_host(&*hip, "matmul_fp8_wmma", &one).unwrap();
                bits_eq(&label, &format!("output {j}"), &got[j], &want[0]);
            }
            near_reference(Op::RmsNormAddMatmulFp8Pair, &hc, &got, 6e-3, &label);
        }
    }
}

#[test]
fn sigmoid_mul_matmul_fp8_wmma_is_bitwise_gate_then_launch() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(93);
    for (nn, k) in [(2048usize, 4096usize), (64, 64), (40, 96), (17, 48), (520, 2064), (8192, 512), (3, 4096)] {
        for aq in [true, false] {
            let hc = sigmoid_mul_matmul_fp8_case(&mut rng, 1, nn, k, aq, DType::Bf16, DType::Bf16);
            let label = format!("wmma P4b n={nn} k={k} aq={aq}");
            let got = run_host(&*hip, "sigmoid_mul_matmul_fp8_wmma", &hc).unwrap_or_else(|e| panic!("{label}: {e}"));
            let gate = call(Op::SigmoidMul, hc.inputs[..2].to_vec(), vec![OutSpec::new(DType::Bf16, &[1, k])], OpAttrs::default());
            let g = run_host(&*hip, "sigmoid_mul_bf16", &gate).unwrap();
            let mm = call(Op::MatmulFp8, vec![g[0].clone(), hc.inputs[2].clone(), hc.inputs[3].clone()], hc.outputs.clone(), OpAttrs::default());
            let want = run_host(&*hip, "matmul_fp8_wmma", &mm).unwrap();
            bits_eq(&label, "out", &got[0], &want[0]);
            near_reference(Op::SigmoidMulMatmulFp8, &hc, &got, 6e-3, &label);
        }
    }
}

#[test]
fn gated_rmsnorm_matmul_fp8_wmma_is_bitwise_norm_then_launch() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(94);
    for (r, n) in [(32usize, 2048usize), (32, 1), (4, 40), (1, 64), (7, 33), (16, 520), (32, 8192), (24, 4096), (3, 2049)] {
        for aq in [true, false] {
            for w_offset in [0.0f32, 1.0] {
                let mut hc = gated_rmsnorm_matmul_fp8_case(&mut rng, r, n, aq, DType::Bf16);
                hc.attrs.w_offset = w_offset;
                let label = format!("wmma gated norm + fp8 r={r} n={n} act_quant={aq} w_offset={w_offset}");
                let got = run_host(&*hip, "gated_rmsnorm_matmul_fp8_wmma", &hc).unwrap_or_else(|e| panic!("{label}: {e}"));
                let norm = call(Op::GatedRmsnorm, vec![hc.inputs[0].clone(), hc.inputs[1].clone(), hc.inputs[2].clone()], vec![out_bf16(&[r, 128])], hc.attrs.clone());
                let mut normed = run_host(&*hip, "gated_rmsnorm_bf16", &norm).unwrap().remove(0);
                normed.shape = vec![1, r * 128];
                let mm = call(Op::MatmulFp8, vec![normed, hc.inputs[3].clone(), hc.inputs[4].clone()], vec![out_bf16(&[1, n])], OpAttrs::default());
                let want = run_host(&*hip, "matmul_fp8_wmma", &mm).unwrap();
                bits_eq(&label, "output", &got[0], &want[0]);
                near_reference(Op::GatedRmsnormMatmulFp8, &hc, &got, 6e-3, &label);
            }
        }
    }
}

#[test]
fn matmul_pair_bf16_wmma_is_bitwise_two_launches() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(95);
    for (n0, n1, k) in [(32usize, 32usize, 2048usize), (16, 16, 2048), (256, 1, 2048), (33, 1, 96), (300, 5, 528), (9, 17, 48), (1, 1, 16), (7, 4097, 128), (2048, 2048, 2048)] {
        let hc = matmul_pair_case(&mut rng, 1, n0, n1, k, DType::Bf16);
        let label = format!("wmma bf16 pair n0={n0} n1={n1} k={k}");
        let got = run_host(&*hip, "matmul_pair_bf16_wmma", &hc).unwrap_or_else(|e| panic!("{label}: {e}"));
        for (j, w) in [(0usize, 1usize), (1, 2)] {
            let one = call(Op::Matmul, vec![hc.inputs[0].clone(), hc.inputs[w].clone()], vec![hc.outputs[j].clone()], OpAttrs::default());
            let want = run_host(&*hip, "matmul_bf16_wmma", &one).unwrap();
            bits_eq(&label, &format!("output {j}"), &got[j], &want[0]);
        }
        near_reference(Op::MatmulPair, &hc, &got, 1e-3, &label);
    }
}

#[test]
fn rmsnorm_add_matmul_pair_bf16_wmma_is_bitwise_norm_then_two_launches() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(96);
    for (n0, n1, k) in [(32usize, 32usize, 2048usize), (16, 16, 2048), (256, 1, 2048), (33, 1, 96), (9, 17, 48), (1, 1, 16), (300, 5, 528), (2048, 2048, 2048), (64, 1, 1024)] {
        for w_offset in [0.0f32, 1.0] {
            let hc = rmsnorm_add_matmul_pair_case(&mut rng, 1, n0, n1, k, DType::Bf16, w_offset);
            let label = format!("wmma norm+bf16 pair n0={n0} n1={n1} k={k} w_offset={w_offset}");
            let got = run_host(&*hip, "rmsnorm_add_matmul_pair_bf16_wmma", &hc).unwrap_or_else(|e| panic!("{label}: {e}"));
            let norm = call(Op::RmsNormAdd, hc.inputs[..3].to_vec(), vec![out_bf16(&[1, k]), out_bf16(&[1, k])], hc.attrs.clone());
            let want = run_host(&*hip, "rmsnorm_add_bf16", &norm).unwrap();
            bits_eq(&label, "normed", &got[0], &want[0]);
            bits_eq(&label, "sum", &got[1], &want[1]);
            for (j, w) in [(2usize, 3usize), (3, 4)] {
                let one = call(Op::Matmul, vec![want[0].clone(), hc.inputs[w].clone()], vec![hc.outputs[j].clone()], OpAttrs::default());
                let mm = run_host(&*hip, "matmul_bf16_wmma", &one).unwrap();
                bits_eq(&label, &format!("output {j}"), &got[j], &mm[0]);
            }
            near_reference(Op::RmsNormAddMatmulPair, &hc, &got, 1e-3, &label);
        }
    }
}
