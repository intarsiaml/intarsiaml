//! `matmul_fp8_mma_v2` (kernels/hip/fp8.hip, `Fp8PolicyQ`): `matmul_fp8_mma` with the W8A8 activations quantised once per call
//! into scratch and the weights decoded by the gfx12 fp8 conversion instruction. It must give `matmul_fp8_mma`'s output bit for
//! bit: the bf16 tiles in LDS are the same values and the WMMAs run in the same order.
//!
//! * `v2_is_bitwise_mma`: random W8A8 (three weight / input scale pairs) and weight-only cases, f32 and bf16 outputs, shapes from
//!   M = 5 to 4096 with M, N and K tails (K not a multiple of the 32-element k tile, N not of the 128-row tile) and the model's
//!   projections: every output bit equal to `matmul_fp8_mma`'s.
//! * `v2_decodes_every_e4m3_code`: weights that run through all 254 non-NaN e4m3 codes in every k position of a tile, and
//!   activations that quantise to every code (subnormals, both signs, saturation past 448 * input_scale, jitter around the
//!   rounding boundaries): bitwise `matmul_fp8_mma`, whose decode is the software one, so the conversion instruction agrees
//!   with it on every code that can occur.
//! * `v2_close_to_cpu_reference`: v2 is close to the cpu reference (not a copy of a wrong v1): W8A8 within 2e-6 relative L2
//!   (fp8_f8.rs's bound: only the order of the additions differs), weight-only within the admission's f32 budget (2e-4).
//!
//! scripts/hip-decode/mutate.py (`v2-*` mutations) breaks the kernel in ways these must catch. Needs a GPU: skips (passes)
//! without one.
#![cfg(ik_hip)]

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::backend::Backend;
use intarsia_core::dtype::fp8_e4m3_to_f32;
use intarsia_core::exec::{run_host, HostCall};
use intarsia_core::quant::cases::matmul_fp8_case;
use intarsia_core::quant::fp8::scales_tensor;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, Op};
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

/// Case with the weight scale scaled by `ws` and input scale `is` (0 = weight-only activations).
fn case(rng: &mut SplitMix64, m: usize, n: usize, k: usize, odt: DType, ws: f32, is: f32) -> HostCall {
    let mut hc = matmul_fp8_case(rng, m, n, k, is > 0.0, DType::Bf16, odt);
    let s = hc.inputs[2].to_f32();
    hc.inputs[2] = scales_tensor(s[0] * ws, is);
    hc
}

fn assert_bitwise(hip: &dyn Backend, hc: &HostCall, what: &str) {
    let v1 = run_host(hip, "matmul_fp8_mma", hc).unwrap().remove(0);
    let v2 = run_host(hip, "matmul_fp8_mma_v2", hc).unwrap().remove(0);
    assert_eq!(v1.shape, v2.shape);
    if v1.data != v2.data {
        let (a, b) = (v1.to_f32(), v2.to_f32());
        let bad = a.iter().zip(&b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        let first = a.iter().zip(&b).position(|(x, y)| x.to_bits() != y.to_bits()).unwrap();
        panic!("{what}: {bad} of {} outputs differ from matmul_fp8_mma (first at {first}: {} vs {})", a.len(), a[first], b[first]);
    }
}

const SHAPES: &[(usize, usize, usize)] = &[
    (5, 512, 2048),
    (16, 64, 32),
    (17, 130, 48),
    (33, 259, 1040),
    (64, 4096, 2048),
    (65, 2048, 4096),
    (100, 8192, 2048),
    (129, 2048, 4096),
    (257, 4096, 2048),
    (300, 17, 96),
    (130, 131, 4112),
    (1024, 512, 2048),
    (4096, 1024, 2048),
];

#[test]
fn v2_is_bitwise_mma() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(91);
    let mut n = 0;
    for &(m, nn, k) in SHAPES {
        for (ws, is) in [(1.0f32, 0.31f32), (2.5, 0.07), (0.4, 4.0), (1.7, 0.0)] {
            for odt in [DType::F32, DType::Bf16] {
                let hc = case(&mut rng, m, nn, k, odt, ws, is);
                assert_bitwise(&*hip, &hc, &format!("m={m} n={nn} k={k} ws={ws} is={is} {odt:?}"));
                n += 1;
            }
        }
    }
    eprintln!("v2 == matmul_fp8_mma bit for bit on {n} cases");
}

/// The 254 e4m3 codes that are not NaN (0x7f, 0xff).
fn codes() -> Vec<u8> {
    (0u16..256).map(|c| c as u8).filter(|c| c & 0x7f != 0x7f).collect()
}

#[test]
fn v2_decodes_every_e4m3_code() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(92);
    let cs = codes();
    assert_eq!(cs.len(), 254);
    for &(m, n, k) in &[(37usize, 254usize, 2032usize), (130, 127, 1024), (5, 256, 4064)] {
        for is in [0.0f32, 0.05, 1.0, 3.5] {
            let mut hc = case(&mut rng, m, n, k, DType::F32, 1.0, is);
            // weights: row r, column c holds code (r + c) mod 254, so each code meets every k position of a tile, every row
            let w: Vec<u8> = (0..n * k).map(|i| cs[(i / k + i % k) % cs.len()]).collect();
            hc.inputs[1] = HostTensor::new(DType::Fp8E4m3, vec![n, k], w).unwrap();
            // activations: around every code times the input scale (jitter up to +-6 % crosses the rounding boundaries), with a
            // saturating value (up to 5x past 448) and an exact zero now and then; weight-only: the same values times 1
            let s = if is > 0.0 { is } else { 1.0 };
            let x: Vec<f32> = (0..m * k)
                .map(|i| {
                    let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
                    match i % 61 {
                        0 => s * 448.0 * (1.0 + 4.0 * u as f32) * if i % 2 == 0 { 1.0 } else { -1.0 },
                        1 => 0.0,
                        _ => fp8_e4m3_to_f32(cs[rng.next_u64() as usize % cs.len()]) * s * (0.94 + 0.12 * u as f32),
                    }
                })
                .collect();
            hc.inputs[0] = HostTensor::from_f32_as(DType::Bf16, &[m, k], &x);
            assert_bitwise(&*hip, &hc, &format!("all codes m={m} n={n} k={k} is={is}"));
            let mut hb = hc.clone();
            hb.outputs[0].dtype = DType::Bf16;
            assert_bitwise(&*hip, &hb, &format!("all codes m={m} n={n} k={k} is={is} bf16 out"));
        }
    }
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

#[test]
fn v2_close_to_cpu_reference() {
    let Some(hip) = device() else { return };
    let cpu = CpuBackend::new();
    let reference = cpu.kernels().into_iter().find(|k| k.op == Op::MatmulFp8 && k.is_reference).unwrap().name;
    let mut rng = SplitMix64::new(93);
    // W8A8: every product e4m3 x e4m3 is exact, only the order of the f32 additions differs (fp8_f8.rs uses the same 2e-6);
    // weight-only: bf16 activations, judged by the admission budget of f32 outputs (2e-4)
    let mut worst = [0f64; 2];
    for &(m, n, k) in &[(5usize, 512usize, 2048usize), (33, 259, 1040), (129, 2048, 4096), (300, 17, 96)] {
        for (mode, is, bound) in [(0usize, 0.31f32, 2e-6f64), (1, 0.0, 2e-4)] {
            let hc = case(&mut rng, m, n, k, DType::F32, 1.3, is);
            let want = run_host(&cpu, &reference, &hc).unwrap().remove(0).to_f32();
            let got = run_host(&*hip, "matmul_fp8_mma_v2", &hc).unwrap().remove(0).to_f32();
            let r = rel_l2(&got, &want);
            worst[mode] = worst[mode].max(r);
            assert!(r < bound, "m={m} n={n} k={k} is={is}: rel_l2 {r} vs {reference} (bound {bound})");
        }
    }
    eprintln!("v2 vs cpu reference: worst rel_l2 (f32 out) W8A8 {:.3e}, weight-only {:.3e}", worst[0], worst[1]);
}
