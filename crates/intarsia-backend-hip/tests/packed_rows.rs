//! Lane hg (packed multi-prompt prefill): are the row-wise prefill kernels of the 35B independent of how many other rows share the call?
//!
//! A packed prefill pass runs the GEMMs, the router, the MoE and the elementwise ops once over the concatenated rows of several
//! prompts, and attention / GDN per prompt. Each prompt's result is bitwise its solo prefill only if every row-wise kernel gives a
//! row the same bits whatever else is in the call. For each kernel the prefill selects at M >= 16 this test runs one call over the
//! packed rows of several segments and one call per segment, and compares every row byte for byte.
//! `f8_rows_do_not_depend_on_m` (fp8_f8.rs) covers `matmul_fp8_mma_f8`; this file covers the NVFP4 GEMM, the bf16 GEMM (router, shared-expert
//! gate), the MoE top-k and experts and the combine, at the 35B's real shapes. Needs a GPU: skips (passes) without one.
#![cfg(ik_hip)]

use intarsia_core::backend::Backend;
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::quant::cases::{experts_case, matmul_nvfp4_case, topk_case};
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

fn rows(t: &HostTensor, a: usize, b: usize) -> HostTensor {
    let row = t.nbytes() / t.shape[0];
    let mut shape = t.shape.clone();
    shape[0] = b - a;
    HostTensor::new(t.dtype, shape, t.data[a * row..b * row].to_vec()).unwrap()
}

/// Segment lengths of a packed pass: prompt-sized, odd, and the smallest the packing rule admits (16 rows).
const SEGS: [usize; 6] = [1023, 999, 64, 16, 2048, 17];

/// Run `kernel` once over all rows and once per segment; every row of every output must match bit for bit. `row_inputs`: the inputs
/// whose leading dimension is the row count; every output is row-wise.
fn check(hip: &dyn Backend, kernel: &str, hc: &HostCall, row_inputs: &[usize], segs: &[usize], label: &str) {
    let total: usize = segs.iter().sum();
    let full = run_host(hip, kernel, hc).unwrap();
    let again = run_host(hip, kernel, hc).unwrap();
    for (a, b) in full.iter().zip(&again) {
        assert_eq!(a.data, b.data, "{label}: not repeatable");
    }
    let mut lo = 0;
    let mut checked = 0usize;
    for &len in segs {
        let mut one = hc.clone();
        for &i in row_inputs {
            one.inputs[i] = rows(&hc.inputs[i], lo, lo + len);
        }
        one.outputs = hc
            .outputs
            .iter()
            .map(|o| {
                let mut shape = o.shape.clone();
                shape[0] = len;
                OutSpec::new(o.dtype, &shape)
            })
            .collect();
        let single = run_host(hip, kernel, &one).unwrap();
        for (k, (f, s)) in full.iter().zip(&single).enumerate() {
            let row = f.nbytes() / total;
            let want = &f.data[lo * row..(lo + len) * row];
            if want != &s.data[..] {
                let bad = (0..len).filter(|r| want[r * row..(r + 1) * row] != s.data[r * row..(r + 1) * row]).count();
                panic!("{label}: output {k}, segment of {len} rows at row {lo}: {bad} rows differ from the same rows alone");
            }
            checked += len;
        }
        lo += len;
    }
    eprintln!("{label}: {kernel} rows identical, {} segments, {total} rows, {checked} row-outputs", segs.len());
}

#[test]
fn moe_experts_rows_do_not_depend_on_t() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(71);
    let total: usize = SEGS.iter().sum();
    // the 35B: 256 experts, top-8, hidden 2048, expert width 512
    let hc = experts_case(&mut rng, total, 256, 8, 2048, 512, DType::Bf16);
    check(&*hip, "moe_experts_bf16_v2", &hc, &[0, 1, 2], &SEGS, "moe_experts 35B shape");
}

#[test]
fn moe_experts_small_segments_cross_the_gemv_wmma_threshold() {
    // Documents the packing rule: a segment of < 16 rows takes the GEMV-style path alone (T < 16) but the tensor-core path inside a
    // packed call, so its rows may differ; the test only requires segments >= 16 to match and reports the short one.
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(72);
    let segs = [200usize, 15, 300];
    let total: usize = segs.iter().sum();
    let hc = experts_case(&mut rng, total, 256, 8, 2048, 512, DType::Bf16);
    let full = run_host(&*hip, "moe_experts_bf16_v2", &hc).unwrap();
    let row = full[0].nbytes() / total;
    let mut lo = 0;
    for &len in &segs {
        let mut one = hc.clone();
        for i in [0usize, 1, 2] {
            one.inputs[i] = rows(&hc.inputs[i], lo, lo + len);
        }
        one.outputs = vec![OutSpec::new(DType::Bf16, &[len, 2048])];
        let single = run_host(&*hip, "moe_experts_bf16_v2", &one).unwrap();
        let same = full[0].data[lo * row..(lo + len) * row] == single[0].data[..];
        eprintln!("segment of {len} rows: {}", if same { "identical" } else { "DIFFERS from the same rows alone" });
        if len >= 16 {
            assert!(same, "a segment of {len} rows must be identical");
        }
        lo += len;
    }
}

#[test]
fn moe_topk_rows_do_not_depend_on_t() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(73);
    let total: usize = SEGS.iter().sum();
    let hc = topk_case(&mut rng, total, 256, 8, false, DType::Bf16, true);
    check(&*hip, "moe_topk_bf16_v2", &hc, &[0], &SEGS, "moe_topk");
    let hc = topk_case(&mut rng, total, 256, 8, true, DType::Bf16, true);
    check(&*hip, "moe_topk_bf16_v2", &hc, &[0], &SEGS, "moe_topk with ties");
}

#[test]
fn nvfp4_gemm_rows_do_not_depend_on_m() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(74);
    let total: usize = SEGS.iter().sum();
    // the shared expert: gate / up (N 512, K 2048) and down (N 2048, K 512)
    for (n, k) in [(512usize, 2048usize), (2048, 512)] {
        let hc = matmul_nvfp4_case(&mut rng, total, n, k, DType::Bf16, DType::Bf16);
        check(&*hip, "matmul_nvfp4_mma", &hc, &[0], &SEGS, &format!("matmul_nvfp4_mma N{n} K{k}"));
    }
}

#[test]
fn bf16_gemm_rows_do_not_depend_on_m() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(75);
    let total: usize = SEGS.iter().sum();
    // the router (N 256, K 2048) and the shared-expert gate logit (N 1, K 2048)
    for (n, k) in [(256usize, 2048usize), (1, 2048)] {
        let x = HostTensor::from_f32_as(DType::Bf16, &[total, k], &rng.normal_vec(total * k, 1.0));
        let w = HostTensor::from_f32_as(DType::Bf16, &[n, k], &rng.normal_vec(n * k, 0.05));
        let hc = HostCall { op: Op::Matmul, inputs: vec![x, w], outputs: vec![OutSpec::new(DType::Bf16, &[total, n])], attrs: OpAttrs::default() };
        check(&*hip, "matmul_bf16", &hc, &[0], &SEGS, &format!("matmul_bf16 N{n} K{k}"));
    }
}
