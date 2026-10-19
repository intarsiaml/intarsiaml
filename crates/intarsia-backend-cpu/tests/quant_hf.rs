//! Quantised-weight and MoE cpu kernels against numbers computed by independent Python
//! (tools/gen_quantmoe_fixture.py: numpy/torch dequantisation + HF `Qwen3_5MoeSparseMoeBlock`) from real
//! slices of the Qwen3.6-35B-A3B-NVFP4 checkpoint (tests/fixtures/qwen36_quant_slices.safetensors).

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::backend::Backend;
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::quant::moe::{MoeConfig, MoeScratch, MoeWeights, OpRunner};
use intarsia_core::quant::{fp8, nvfp4, HostQTensor, QuantLoadOpts, FP8_E4M3, NVFP4};
use intarsia_core::weights::MemWeights;
use intarsia_core::{DType, HostTensor, Op, OpAttrs, OpCall, Result, TensorRef};
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

fn nv(s: &BTreeMap<String, HostTensor>, p: &str) -> HostQTensor {
    let w = &s[&format!("{p}.weight")];
    HostQTensor {
        format: NVFP4,
        n: w.shape[0],
        k: w.shape[1] * 2,
        parts: vec![w.clone(), s[&format!("{p}.weight_scale")].clone(), HostTensor::new(DType::F32, vec![1], s[&format!("{p}.weight_scale_2")].data.clone()).unwrap()],
        act_quant: false,
    }
}

fn fp(s: &BTreeMap<String, HostTensor>, p: &str, act: bool) -> HostQTensor {
    let w = &s[&format!("{p}.weight")];
    let ws = s[&format!("{p}.weight_scale")].to_f32()[0];
    let is = s[&format!("{p}.input_scale")].to_f32()[0];
    HostQTensor {
        format: FP8_E4M3,
        n: w.shape[0],
        k: w.shape[1],
        parts: vec![w.clone(), fp8::scales_tensor(ws, if act { is } else { 0.0 })],
        act_quant: act,
    }
}

fn matmul(b: &CpuBackend, kernel: &str, op: Op, x: &HostTensor, q: &HostQTensor) -> Vec<f32> {
    let mut inputs = vec![x.clone()];
    inputs.extend(q.parts.iter().cloned());
    let hc = HostCall { op, inputs, outputs: vec![OutSpec::new(DType::F32, &[x.shape[0], q.n])], attrs: OpAttrs::default() };
    run_host(b, kernel, &hc).unwrap().remove(0).to_f32()
}

#[test]
fn nvfp4_dequant_and_matmul_match_numpy() {
    let (s, e) = (load_st("qwen36_quant_slices.safetensors"), load_st("qwen36_quant_expected.safetensors"));
    let b = CpuBackend::new();
    for key in ["nv_gate", "nv_down", "nv_head"] {
        let q = nv(&s, key);
        let want = e[&format!("{key}.W_rows0")].to_f32();
        let got = q.dequant();
        let r = rel_l2(&got[..4 * q.k], &want);
        assert!(r < 1e-6, "{key} dequant rel_l2 {r}");
        let x = e[&format!("{key}.x")].clone();
        let want = e[&format!("{key}.y")].to_f32();
        for kernel in ["ref_matmul_nvfp4", "cpu_f32_matmul_nvfp4"] {
            let y = matmul(&b, kernel, Op::MatmulNvfp4, &x, &q);
            let r = rel_l2(&y, &want);
            assert!(r < 1e-5, "{key} {kernel} rel_l2 {r}");
        }
    }
}

#[test]
fn fp8_w8a16_and_w8a8_match_torch() {
    let (s, e) = (load_st("qwen36_quant_slices.safetensors"), load_st("qwen36_quant_expected.safetensors"));
    let b = CpuBackend::new();
    for key in ["fp8_z", "fp8_o"] {
        let x = e[&format!("{key}.x")].clone();
        for (act, tag, tol) in [(false, "w8a16", 1e-5), (true, "w8a8", 1e-5)] {
            let q = fp(&s, key, act);
            let want = e[&format!("{key}.y_{tag}")].to_f32();
            for kernel in ["ref_matmul_fp8", "cpu_f32_matmul_fp8"] {
                let y = matmul(&b, kernel, Op::MatmulFp8, &x, &q);
                let r = rel_l2(&y, &want);
                assert!(r < tol, "{key} {tag} {kernel} rel_l2 {r}");
            }
        }
    }
    // the two modes differ measurably (activation rounding is real), so the option is not a no-op
    let x = e["fp8_z.x"].clone();
    let a = matmul(&b, "ref_matmul_fp8", Op::MatmulFp8, &x, &fp(&s, "fp8_z", false));
    let c = matmul(&b, "ref_matmul_fp8", Op::MatmulFp8, &x, &fp(&s, "fp8_z", true));
    assert!(rel_l2(&a, &c) > 1e-3);
}

struct Runner<'a> {
    b: &'a CpuBackend,
    prefix: &'a str,
}

impl OpRunner for Runner<'_> {
    fn run(&self, op: Op, inputs: Vec<TensorRef>, outputs: Vec<TensorRef>, attrs: OpAttrs) -> Result<()> {
        let call = OpCall::new(op, inputs, outputs, attrs);
        call.check()?;
        self.b.launch(&format!("{}{}", self.prefix, op.as_str()), &call, 0)?;
        self.b.synchronize(0)
    }
}

fn moe_weights(s: &BTreeMap<String, HostTensor>) -> MemWeights {
    let mut w = MemWeights::new();
    let p = "model.language_model.layers.1.mlp";
    w.insert(&format!("{p}.gate.weight"), s["moe.router"].clone());
    w.insert(&format!("{p}.shared_expert_gate.weight"), s["moe.shared_gate"].clone());
    let mut add = |dst: String, src: &str| {
        for suf in ["weight", "weight_scale", "weight_scale_2", "input_scale"] {
            w.insert(&format!("{dst}.{suf}"), s[&format!("{src}.{suf}")].clone());
        }
    };
    for proj in ["gate_proj", "up_proj", "down_proj"] {
        add(format!("{p}.shared_expert.{proj}"), &format!("moe.shared.{proj}"));
        for e in 0..8 {
            add(format!("{p}.experts.{e}.{proj}"), &format!("moe.e{e}.{proj}"));
        }
    }
    w
}

fn to_dev(b: &CpuBackend, t: &HostTensor) -> TensorRef {
    let buf = intarsia_core::backend::upload_new(b, &t.data).unwrap();
    TensorRef::new(buf, t.dtype, &t.shape)
}

fn dl(b: &CpuBackend, t: &TensorRef) -> HostTensor {
    let mut d = vec![0u8; t.nbytes()];
    b.download(&t.buf, t.offset, &mut d, 0).unwrap();
    HostTensor::new(t.dtype, t.shape.clone(), d).unwrap()
}

#[test]
fn topk_matches_hf_router() {
    let e = load_st("qwen36_quant_expected.safetensors");
    let b = CpuBackend::new();
    for (dt, sfx, tol) in [(DType::F32, "", 1e-6), (DType::Bf16, "_bf16", 1e-2)] {
        let logits = HostTensor::from_f32_as(dt, &[5, 8], &e[&format!("moe.logits{sfx}")].to_f32());
        let hc = HostCall {
            op: Op::MoeTopk,
            inputs: vec![logits],
            outputs: vec![OutSpec::new(DType::F32, &[5, 2]), OutSpec::new(DType::I32, &[5, 2])],
            attrs: OpAttrs { top_k: 2, norm_topk: true, ..OpAttrs::default() },
        };
        for kernel in ["ref_moe_topk", "cpu_f32_moe_topk"] {
            let out = run_host(&b, kernel, &hc).unwrap();
            assert_eq!(out[1].to_i32().unwrap(), e[&format!("moe.topk_id{sfx}")].to_i32().unwrap(), "{kernel} ids");
            let r = rel_l2(&out[0].to_f32(), &e[&format!("moe.topk_w{sfx}")].to_f32());
            assert!(r < tol, "{kernel} {dt} weights rel_l2 {r}");
        }
    }
}

#[test]
fn moe_block_matches_hf() {
    let s = load_st("qwen36_quant_slices.safetensors");
    let e = load_st("qwen36_quant_expected.safetensors");
    let b = CpuBackend::new();
    let cfg = MoeConfig { hidden: 256, experts: 8, top_k: 2, inter: 64, shared_inter: 64, norm_topk: true };
    let w = moe_weights(&s);
    let moe = MoeWeights::load(&b, &w, "model.language_model.layers.1.mlp", &cfg, &QuantLoadOpts::default()).unwrap();
    assert!(moe.weight_bytes() > 0);
    for (dt, want_key, tol) in [(DType::F32, "moe.y_f32", 2e-5), (DType::Bf16, "moe.y_bf16", 4e-2)] {
        let x = HostTensor::from_f32_as(dt, &[5, 256], &e["moe.x"].to_f32());
        let ws = MoeScratch::new(&b, &cfg, 8, dt).unwrap();
        let (xt, out) = (to_dev(&b, &x), to_dev(&b, &HostTensor::zeros(dt, &[5, 256])));
        for prefix in ["ref_", "cpu_f32_"] {
            moe.forward(&Runner { b: &b, prefix }, &ws, &xt, &out).unwrap();
            let got = dl(&b, &out).to_f32();
            let r = rel_l2(&got, &e[want_key].to_f32());
            assert!(r < tol, "{prefix} {dt}: rel_l2 {r} vs HF");
        }
    }
}

/// A runner that also runs the fused ops (the cpu backend registers every one) and records the ops it ran.
struct FusingRunner<'a> {
    b: &'a CpuBackend,
    prefix: &'a str,
    ops: std::cell::RefCell<Vec<Op>>,
}

impl OpRunner for FusingRunner<'_> {
    fn run(&self, op: Op, inputs: Vec<TensorRef>, outputs: Vec<TensorRef>, attrs: OpAttrs) -> Result<()> {
        self.ops.borrow_mut().push(op);
        Runner { b: self.b, prefix: self.prefix }.run(op, inputs, outputs, attrs)
    }
    fn run_fused(&self, op: Op, inputs: Vec<TensorRef>, outputs: Vec<TensorRef>, attrs: OpAttrs) -> Result<bool> {
        self.run(op, inputs, outputs, attrs)?;
        Ok(true)
    }
}

/// The MoE block's fused path (router + shared-gate pair, shared gate/up glu, and the phase-1 `SharedDownCombine`: the shared
/// down projection with the combine as its epilogue) gives the bits of the unfused ops, on both cpu executors, and runs
/// `SharedDownCombine` in place of the shared-down matmul and the combine.
#[test]
fn moe_block_fused_path_is_bitwise_the_unfused_one() {
    let s = load_st("qwen36_quant_slices.safetensors");
    let e = load_st("qwen36_quant_expected.safetensors");
    let b = CpuBackend::new();
    let cfg = MoeConfig { hidden: 256, experts: 8, top_k: 2, inter: 64, shared_inter: 64, norm_topk: true };
    let moe = MoeWeights::load(&b, &moe_weights(&s), "model.language_model.layers.1.mlp", &cfg, &QuantLoadOpts::default()).unwrap();
    for dt in [DType::F32, DType::Bf16] {
        for t in [1usize, 5] {
            let x = HostTensor::from_f32_as(dt, &[t, 256], &e["moe.x"].to_f32()[..t * 256]);
            let ws = MoeScratch::new(&b, &cfg, 8, dt).unwrap();
            let xt = to_dev(&b, &x);
            for prefix in ["ref_", "cpu_f32_"] {
                let (o1, o2) = (to_dev(&b, &HostTensor::zeros(dt, &[t, 256])), to_dev(&b, &HostTensor::zeros(dt, &[t, 256])));
                moe.forward(&Runner { b: &b, prefix }, &ws, &xt, &o1).unwrap();
                let fr = FusingRunner { b: &b, prefix, ops: Default::default() };
                moe.forward(&fr, &ws, &xt, &o2).unwrap();
                assert_eq!(dl(&b, &o1).data, dl(&b, &o2).data, "{prefix} {dt} t={t}: fused MoE block differs from unfused");
                let ops = fr.ops.borrow();
                let n = |op: Op| ops.iter().filter(|&&o| o == op).count();
                assert_eq!(n(Op::SharedDownCombine), 1, "{prefix}: {ops:?}");
                assert_eq!((n(Op::MoeCombine), n(Op::MatmulNvfp4)), (0, 0), "{prefix}: {ops:?}");
            }
        }
    }
}

#[test]
fn kernels_pass_admission_sweep_and_are_registered() {
    use intarsia_core::admission::{run_admission, AdmissionOpts, SweepConfig};
    let b = CpuBackend::new();
    let ops = [Op::MatmulFp8, Op::MatmulNvfp4, Op::MoeTopk, Op::MoeExperts, Op::MoeCombine];
    for op in ops {
        let names: Vec<_> = b.kernels().into_iter().filter(|k| k.op == op).map(|k| k.name.to_string()).collect();
        assert!(names.iter().any(|n| n.starts_with("ref_")) && names.iter().any(|n| n.starts_with("cpu_f32_")), "{op}: {names:?}");
    }
    let opts = AdmissionOpts {
        sweep: Some(SweepConfig { quick: true, ..Default::default() }),
        only_ops: Some(ops.to_vec()),
        ..Default::default()
    };
    let report = run_admission(&b, &b, &opts).unwrap();
    assert!(!report.rows.is_empty());
    assert!(report.all_pass(), "{}", report.render(true));
    let _ = nvfp4::GROUP;
}

#[test]
fn plan_time_cases_are_built_for_every_new_op() {
    use intarsia_core::admission::{case_from_call, case_label, sweep, ShapeCaps, SweepConfig};
    let b = CpuBackend::new();
    let caps = ShapeCaps { rows: 4, n: 512, pos0: 8, emb_vocab: 512 };
    let mut rng = intarsia_core::rng::SplitMix64::new(3);
    for op in [
        Op::MatmulFp8,
        Op::MatmulNvfp4,
        Op::MoeTopk,
        Op::MoeExperts,
        Op::MoeCombine,
        Op::RmsNormAddMatmulFp8Pair,
        Op::SigmoidMulMatmulFp8,
        Op::SharedDownCombine,
    ] {
        let case = sweep(op, &SweepConfig { quick: true, ..Default::default() }).remove(0);
        // a planned call: device tensors with the case's shapes and dtypes
        let inputs: Vec<TensorRef> = case.call.inputs.iter().map(|t| to_dev(&b, t)).collect();
        let outputs: Vec<TensorRef> = case.call.outputs.iter().map(|o| to_dev(&b, &HostTensor::zeros(o.dtype, &o.shape))).collect();
        let call = OpCall::new(op, inputs, outputs, case.call.attrs.clone());
        let label = case_label(&call, &caps).unwrap_or_else(|| panic!("{op}: no case label"));
        let (hc, l2) = case_from_call(&call, &caps, &mut rng).unwrap_or_else(|| panic!("{op}: no case builder"));
        assert_eq!(label, l2);
        assert_eq!(hc.op, op);
        run_host(&b, &format!("ref_{}", op.as_str()), &hc).unwrap();
        run_host(&b, &format!("cpu_f32_{}", op.as_str()), &hc).unwrap();
    }
}
