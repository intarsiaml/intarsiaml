//! MXFP4 (AMD Quark OCP MX) cpu kernels and loader against numbers computed by independent Python
//! (tools/gen_mxfp4_fixture.py: torch bit-level e2m1 decode, exp2 e8m0, Quark's own `even_round` for the activation
//! scale) from real row slices of amd/Qwen3.8-27B-Quark-AWQ-MXFP4 (tests/fixtures/qwen38_mxfp4_slices.safetensors).

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::quant::{self, mxfp4, HostQTensor, Mxfp4Act, QuantLoadOpts, WeightMeta, MXFP4};
use intarsia_core::weights::MemWeights;
use intarsia_core::{DType, HostTensor, Op, OpAttrs};
use std::collections::BTreeMap;

fn load_st(name: &str) -> BTreeMap<String, HostTensor> {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    let b = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let n = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
    let hdr: serde_json::Value = serde_json::from_slice(&b[8..8 + n]).unwrap();
    let mut out = BTreeMap::new();
    for (k, v) in hdr.as_object().unwrap() {
        if k == "__metadata__" {
            continue;
        }
        let dt = DType::from_safetensors(v["dtype"].as_str().unwrap()).unwrap();
        let shape: Vec<usize> = v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        let (s, e) = (v["data_offsets"][0].as_u64().unwrap() as usize, v["data_offsets"][1].as_u64().unwrap() as usize);
        out.insert(k.clone(), HostTensor::new(dt, shape, b[8 + n + s..8 + n + e].to_vec()).unwrap());
    }
    out
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

const KEYS: [&str; 5] = ["gate", "down", "in_a", "q", "o"];

fn mx(s: &BTreeMap<String, HostTensor>, p: &str, act: Mxfp4Act) -> HostQTensor {
    let w = &s[&format!("{p}.weight")];
    HostQTensor {
        format: MXFP4,
        n: w.shape[0],
        k: w.shape[1] * 2,
        parts: vec![w.clone(), s[&format!("{p}.weight_scale")].clone(), mxfp4::act_tensor(act)],
        act_quant: act == Mxfp4Act::Dynamic,
    }
}

#[test]
fn dequant_matches_independent_decode_bitwise() {
    let (s, e) = (load_st("qwen38_mxfp4_slices.safetensors"), load_st("qwen38_mxfp4_expected.safetensors"));
    for key in KEYS {
        let q = mx(&s, key, Mxfp4Act::None);
        let want = e[&format!("{key}.W")].to_f32();
        let got = q.dequant();
        assert_eq!(got.len(), want.len(), "{key}");
        let bad = got.iter().zip(&want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        assert_eq!(bad, 0, "{key}: {bad} dequantised values differ from the torch decode");
    }
}

#[test]
fn activation_qdq_matches_quark_even_round_bitwise() {
    let e = load_st("qwen38_mxfp4_expected.safetensors");
    for key in KEYS.iter().copied().chain(["edge"]) {
        let x = e[&format!("{key}.x")].to_f32();
        let want = e[&format!("{key}.xq")].to_f32();
        let k = e[&format!("{key}.x")].shape[1];
        let got = mxfp4::qdq_rows(&x, x.len() / k, k);
        let bad: Vec<usize> = (0..got.len()).filter(|&i| got[i].to_bits() != want[i].to_bits()).collect();
        assert!(bad.is_empty(), "{key}: {} QDQ values differ, first at {}: x {} got {} want {}", bad.len(), bad[0], x[bad[0]], got[bad[0]], want[bad[0]]);
    }
}

#[test]
fn matmul_w4a16_and_w4a4_match_torch() {
    let (s, e) = (load_st("qwen38_mxfp4_slices.safetensors"), load_st("qwen38_mxfp4_expected.safetensors"));
    let b = CpuBackend::new();
    for key in KEYS {
        let x = e[&format!("{key}.x")].clone();
        for (act, tag) in [(Mxfp4Act::None, "w4a16"), (Mxfp4Act::Dynamic, "w4a4")] {
            let q = mx(&s, key, act);
            let want = e[&format!("{key}.y_{tag}")].to_f32();
            for kernel in ["ref_matmul_mxfp4", "cpu_f32_matmul_mxfp4"] {
                let mut inputs = vec![x.clone()];
                inputs.extend(q.parts.iter().cloned());
                let hc = HostCall { op: Op::MatmulMxfp4, inputs, outputs: vec![OutSpec::new(DType::F32, &[x.shape[0], q.n])], attrs: OpAttrs::default() };
                let y = run_host(&b, kernel, &hc).unwrap().remove(0).to_f32();
                let r = rel_l2(&y, &want);
                assert!(r < 1e-6, "{key} {tag} {kernel} rel_l2 {r}");
            }
        }
        // the two modes really differ on real data (the QDQ is not a no-op)
        let (a, d) = (e[&format!("{key}.y_w4a16")].to_f32(), e[&format!("{key}.y_w4a4")].to_f32());
        assert!(rel_l2(&d, &a) > 1e-3, "{key}: W4A4 and W4A16 outputs are suspiciously equal");
    }
}

#[test]
fn loader_detects_and_loads_the_real_layout() {
    let s = load_st("qwen38_mxfp4_slices.safetensors");
    let mut w = MemWeights::new();
    for key in KEYS {
        w.insert(&format!("{key}.weight"), s[&format!("{key}.weight")].clone());
        w.insert(&format!("{key}.weight_scale"), s[&format!("{key}.weight_scale")].clone());
    }
    let b = CpuBackend::new();
    for key in KEYS {
        let meta = WeightMeta::new(&w, key);
        let f = quant::detect(&meta).unwrap();
        assert_eq!(f.id(), MXFP4, "{key}");
        for act in [Mxfp4Act::Dynamic, Mxfp4Act::None] {
            let q = f.load(&meta, &b, &QuantLoadOpts { mxfp4_act: act, ..Default::default() }).unwrap();
            assert_eq!(q.op(false), Op::MatmulMxfp4);
            assert_eq!(q.act_quant, act == Mxfp4Act::Dynamic);
            let h = q.download(&b).unwrap();
            assert_eq!(h.parts[2].to_f32(), vec![act.flag()]);
            assert_eq!(h.dequant(), mx(&s, key, act).dequant(), "{key}");
        }
    }
}

/// `MatmulMxfp4Pair` is exactly two `MatmulMxfp4` calls of the same x and act: bitwise, in both cpu implementations.
#[test]
fn pair_equals_two_matmul_mxfp4_bitwise() {
    use intarsia_core::quant::cases::matmul_mxfp4_pair_case;
    let b = CpuBackend::new();
    let mut rng = intarsia_core::rng::SplitMix64::new(11);
    for (m, n0, n1, k) in [(1usize, 64usize, 48usize, 64usize), (3, 40, 96, 96), (8, 17, 33, 64), (1, 48, 48, 5120)] {
        for act in [Mxfp4Act::Dynamic, Mxfp4Act::None] {
            for (xdt, odt) in [(DType::Bf16, DType::Bf16), (DType::F32, DType::F32), (DType::Bf16, DType::F32)] {
                let pair = matmul_mxfp4_pair_case(&mut rng, m, n0, n1, k, act, xdt, odt);
                for (pk, sk) in [("ref_matmul_mxfp4_pair", "ref_matmul_mxfp4"), ("cpu_f32_matmul_mxfp4_pair", "cpu_f32_matmul_mxfp4")] {
                    let got = run_host(&b, pk, &pair).unwrap();
                    assert_eq!(got.len(), 2);
                    let i = &pair.inputs;
                    for (j, n) in [n0, n1].into_iter().enumerate() {
                        let single = HostCall {
                            op: Op::MatmulMxfp4,
                            inputs: vec![i[0].clone(), i[1 + 2 * j].clone(), i[2 + 2 * j].clone(), i[5].clone()],
                            outputs: vec![OutSpec::new(odt, &[m, n])],
                            attrs: OpAttrs::default(),
                        };
                        let want = run_host(&b, sk, &single).unwrap().remove(0);
                        assert_eq!(got[j].shape, want.shape);
                        assert_eq!(got[j].data, want.data, "{pk} out{j} M={m} N={n} K={k} act={} x={xdt} out={odt}", act.name());
                    }
                }
            }
        }
    }
}
