//! W4A8 (`Mxfp4Act::Fp8`): MXFP4 weights x dynamic per-token fp8 e4m3 activations, cpu reference. The oracle here is written
//! independently of `quant::mxfp4::a8_quant_rows` / `reference::matmul_mxfp4_a8`: a brute-force nearest-code e4m3 encoder
//! and a plain f64 triple loop.

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::dtype::fp8_e4m3_to_f32;
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::quant::cases::{matmul_mxfp4_case, matmul_mxfp4_pair_case};
use intarsia_core::quant::mxfp4::{self, Mxfp4Act};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, Op, OpAttrs};

const K1: &str = "ref_matmul_mxfp4";
const K2: &str = "cpu_f32_matmul_mxfp4";
const KP: &str = "ref_matmul_mxfp4_pair";

/// RNE e4m3fn by search over the 126 finite non-negative codes; ties go to the even code. Saturates at 448.
fn e4m3_value(x: f32) -> f32 {
    let a = x.abs().min(448.0);
    let vals: Vec<f32> = (0u8..127).map(fp8_e4m3_to_f32).collect();
    let mut best = 0usize;
    for i in 1..vals.len() {
        let (db, di) = ((a - vals[best]).abs(), (a - vals[i]).abs());
        if di < db || (di == db && i % 2 == 0 && best % 2 == 1) {
            best = i;
        }
    }
    vals[best].copysign(x)
}

fn oracle(x: &[f32], m: usize, k: usize, w: &[u8], sc: &[u8], n: usize) -> Vec<f64> {
    let wd = mxfp4::dequant_rows(w, sc, n, k);
    let mut out = vec![0f64; m * n];
    for mi in 0..m {
        let row = &x[mi * k..(mi + 1) * k];
        let amax = row.iter().fold(0f32, |a, v| a.max(v.abs()));
        let s = (amax / 448.0).max(1.0 / (448.0 * 512.0));
        let q: Vec<f64> = row.iter().map(|v| e4m3_value((v / s).clamp(-448.0, 448.0)) as f64).collect();
        for ni in 0..n {
            let mut acc = 0f64;
            for ki in 0..k {
                acc += q[ki] * wd[ni * k + ki] as f64;
            }
            out[mi * n + ni] = acc * s as f64;
        }
    }
    out
}

fn rel_l2(a: &[f32], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64 - y).powi(2)).sum();
    let den: f64 = b.iter().map(|y| y * y).sum();
    (num / den.max(1e-30)).sqrt()
}

#[test]
fn matmul_fp8_matches_the_independent_oracle() {
    let b = CpuBackend::new();
    let mut rng = SplitMix64::new(21);
    for (m, n, k) in [(1, 64, 64), (5, 40, 96), (3, 17, 256), (9, 33, 160)] {
        let call = matmul_mxfp4_case(&mut rng, m, n, k, Mxfp4Act::Fp8, DType::Bf16, DType::F32);
        assert!(call.attrs.mxfp4_a8);
        let x = call.inputs[0].to_f32();
        let want = oracle(&x, m, k, &call.inputs[1].data, &call.inputs[2].data, n);
        for kernel in [K1, K2] {
            let got = run_host(&b, kernel, &call).unwrap().remove(0).to_f32();
            assert!(rel_l2(&got, &want) < 1e-6, "{kernel} M={m} N={n} K={k}: {}", rel_l2(&got, &want));
        }
    }
}

#[test]
fn rows_are_independent_of_the_batch() {
    let b = CpuBackend::new();
    let mut rng = SplitMix64::new(22);
    let (m, n, k) = (6, 48, 128);
    let call = matmul_mxfp4_case(&mut rng, m, n, k, Mxfp4Act::Fp8, DType::Bf16, DType::Bf16);
    let all = run_host(&b, K1, &call).unwrap().remove(0);
    for r in 0..m {
        let mut one = HostCall {
            op: Op::MatmulMxfp4,
            inputs: call.inputs.clone(),
            outputs: vec![OutSpec::new(DType::Bf16, &[1, n])],
            attrs: call.attrs.clone(),
        };
        let x = &call.inputs[0];
        let rb = k * x.dtype.size_bytes();
        one.inputs[0] = HostTensor { dtype: x.dtype, shape: vec![1, k], data: x.data[r * rb..(r + 1) * rb].to_vec() };
        let got = run_host(&b, K1, &one).unwrap().remove(0);
        assert_eq!(got.data, all.data[r * n * 2..(r + 1) * n * 2], "row {r}");
    }
}

#[test]
fn pair_equals_two_singles_bitwise() {
    let b = CpuBackend::new();
    let mut rng = SplitMix64::new(23);
    let pair = matmul_mxfp4_pair_case(&mut rng, 4, 40, 24, 96, Mxfp4Act::Fp8, DType::Bf16, DType::Bf16);
    let got = run_host(&b, KP, &pair).unwrap();
    let i = &pair.inputs;
    for (j, n) in [40usize, 24].into_iter().enumerate() {
        let single = HostCall {
            op: Op::MatmulMxfp4,
            inputs: vec![i[0].clone(), i[1 + 2 * j].clone(), i[2 + 2 * j].clone(), i[5].clone()],
            outputs: vec![OutSpec::new(DType::Bf16, &[4, n])],
            attrs: OpAttrs { mxfp4_a8: true, ..OpAttrs::default() },
        };
        assert_eq!(got[j].data, run_host(&b, K1, &single).unwrap().remove(0).data, "out{j}");
    }
}

#[test]
fn fp8_is_closer_to_w4a16_than_w4a4_is() {
    let b = CpuBackend::new();
    let (m, n, k) = (8, 96, 512);
    let mut rng = SplitMix64::new(24);
    let base = matmul_mxfp4_case(&mut rng, m, n, k, Mxfp4Act::None, DType::Bf16, DType::F32);
    let run = |act: Mxfp4Act| {
        let mut c = base.clone();
        c.inputs[3] = mxfp4::act_tensor(act);
        c.attrs.mxfp4_a8 = act == Mxfp4Act::Fp8;
        run_host(&b, K1, &c).unwrap().remove(0).to_f32()
    };
    let (a16, a4, a8) = (run(Mxfp4Act::None), run(Mxfp4Act::Dynamic), run(Mxfp4Act::Fp8));
    let d = |x: &[f32]| rel_l2(x, &a16.iter().map(|v| *v as f64).collect::<Vec<_>>());
    assert!(d(&a8) < d(&a4) * 0.5, "fp8 {} vs w4a4 {}", d(&a8), d(&a4));
}

#[test]
fn an_unknown_act_flag_is_refused() {
    let b = CpuBackend::new();
    let mut rng = SplitMix64::new(25);
    let mut c = matmul_mxfp4_case(&mut rng, 2, 32, 64, Mxfp4Act::Dynamic, DType::Bf16, DType::Bf16);
    c.inputs[3] = intarsia_core::HostTensor::from_f32(&[1], &[3.0]);
    assert!(run_host(&b, K1, &c).is_err());
}

#[test]
fn scale_and_modes() {
    assert_eq!(mxfp4::a8_scale(448.0), 1.0);
    assert_eq!(mxfp4::a8_scale(0.0), mxfp4::A8_MIN_SCALE);
    assert_eq!(Mxfp4Act::parse("fp8").unwrap(), Mxfp4Act::Fp8);
    assert_eq!(Mxfp4Act::parse("w4a8").unwrap(), Mxfp4Act::Fp8);
    assert_eq!(Mxfp4Act::from_flag(Mxfp4Act::Fp8.flag()), Some(Mxfp4Act::Fp8));
    assert_eq!(Mxfp4Act::from_flag(0.5), None);
    // quantised values are the oracle's
    let x: Vec<f32> = (0..64).map(|i| (i as f32 - 31.5) * 0.37).collect();
    let (q, s) = mxfp4::a8_quant_rows(&x, 2, 32);
    for r in 0..2 {
        for j in 0..32 {
            let want = e4m3_value((x[r * 32 + j] / s[r]).clamp(-448.0, 448.0));
            assert_eq!(fp8_e4m3_to_f32(q[r * 32 + j]), want);
        }
    }
}
