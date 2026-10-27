//! The hybrid decoder with MXFP4 (AMD Quark OCP MX) projections, through the VL-wrapper recipe (recipes/qwen3_5_vl.toml).
//!
//! The tiny random `Qwen3_5ForCausalLM` of `tests/fixtures/tiny_qwen3_5/` is renamed to the `model.language_model.*`
//! layout of `Qwen3_5ForConditionalGeneration` (Qwen3.8-27B) and every projection (gated attention q/k/v/o, GatedDeltaNet
//! in_proj_* / out_proj, dense FFN gate/up/down) is quantised to MXFP4. With 16-bit activations (W4A16) the model must
//! reproduce, on the cpu backend, the same model whose weights are the dequantised MXFP4 values in f32: this covers
//! the MXFP4 loader and matmul in all three block kinds, including the dense FFN, which loaded bf16 only before the M1
//! drill. With dynamic MXFP4 activations (W4A4, what Quark declares) the logits must stay finite and differ (the random
//! tiny model is too sensitive for a closeness bound: MXFP4 weights alone move its logits by rel l2 ~1.5).

use intarsia_core::model::{HeadMode, LoadOptions, Model};
use intarsia_core::quant::{mxfp4::Mxfp4, Mxfp4Act, QuantCfg, QuantFormat};
use intarsia_core::recipe::ModelDims;
use intarsia_core::weights::MemWeights;
use intarsia_core::{DType, HostTensor, Telemetry};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

fn read_f32_safetensors(path: &Path) -> BTreeMap<String, (Vec<usize>, Vec<f32>)> {
    let data = std::fs::read(path).unwrap();
    let hlen = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&data[8..8 + hlen]).unwrap();
    let mut out = BTreeMap::new();
    for (name, v) in header.as_object().unwrap() {
        if name == "__metadata__" {
            continue;
        }
        assert_eq!(v["dtype"], "F32", "{name}");
        let shape: Vec<usize> = v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        let off = v["data_offsets"].as_array().unwrap();
        let (b, e) = (8 + hlen + off[0].as_u64().unwrap() as usize, 8 + hlen + off[1].as_u64().unwrap() as usize);
        out.insert(name.clone(), (shape, data[b..e].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()));
    }
    out
}

const PROJ: [&str; 12] = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.in_proj_a",
    "linear_attn.in_proj_b",
    "linear_attn.out_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

/// (MXFP4 checkpoint, f32 checkpoint with the dequantised MXFP4 values), both in the VL layout; number of quantised
/// projections.
fn checkpoints() -> (MemWeights, MemWeights, usize) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_qwen3_5");
    let st = read_f32_safetensors(&dir.join("model.safetensors"));
    let (mut mx, mut deq, mut nq) = (MemWeights::new(), MemWeights::new(), 0);
    for (name, (shape, vals)) in st {
        let vl = name.strip_prefix("model.").map(|r| format!("model.language_model.{r}")).unwrap_or(name.clone());
        let module = vl.strip_suffix(".weight").unwrap_or(&vl);
        if PROJ.iter().any(|p| module.ends_with(p)) {
            let q = Mxfp4.quantize(&HostTensor::from_f32(&shape, &vals), &QuantCfg::default()).expect("K % 32 == 0");
            deq.insert(&vl, HostTensor::from_f32(&shape, &q.dequant()));
            mx.insert(&format!("{module}.weight"), q.parts[0].clone());
            mx.insert(&format!("{module}.weight_scale"), q.parts[1].clone());
            nq += 1;
        } else {
            let t = HostTensor::from_f32(&shape, &vals);
            mx.insert(&vl, t.clone());
            deq.insert(&vl, t);
        }
    }
    (mx, deq, nq)
}

fn logits(src: &MemWeights, act: Option<Mxfp4Act>) -> Vec<Vec<f32>> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_qwen3_5");
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/qwen3_5_vl.toml")).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    let m = Model::load(
        intarsia_backend_cpu::CpuBackend::arc(),
        &recipe,
        &dims,
        src,
        &LoadOptions {
            max_seq: 64,
            prefill_chunk: 7, // chunked prefill: several QDQ'd row blocks
            act_dtype: Some(DType::F32),
            kv_dtype: Some(DType::F32),
            mxfp4_act: act,
            ..Default::default()
        },
        Arc::new(Telemetry::new()),
    )
    .unwrap();
    let ids: Vec<u32> = (0..26u32).map(|i| (i * 37 + 5) % 96).collect();
    let mut st = m.new_state().unwrap();
    let mut rows = Vec::new();
    let mut cb = |_p: usize, r: &[f32]| {
        rows.push(r.to_vec());
        Ok(())
    };
    m.forward(&mut st, &ids, HeadMode::AllLogits, Some(&mut cb)).unwrap();
    rows
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

#[test]
fn mxfp4_w4a16_matches_the_dequantised_model_and_w4a4_runs() {
    let (mx, deq, nq) = checkpoints();
    assert_eq!(nq, 4 * 8 + 2 * 7, "every projection of the 6 layers is MXFP4 (4 GDN + 2 attention layers, each with an FFN)");
    let want = logits(&deq, None);
    let w4a16 = logits(&mx, Some(Mxfp4Act::None));
    assert_eq!(w4a16.len(), want.len());
    for (i, (g, w)) in w4a16.iter().zip(&want).enumerate() {
        let r = rel_l2(g, w);
        assert!(r < 1e-5, "W4A16 position {i}: rel l2 {r} against the dequantised f32 model");
    }
    let w4a4 = logits(&mx, Some(Mxfp4Act::Dynamic));
    let mut worst = 0f64;
    for (g, w) in w4a4.iter().zip(&want) {
        assert!(g.iter().all(|v| v.is_finite()));
        worst = worst.max(rel_l2(g, w));
    }
    println!("W4A4 vs W4A16 worst rel l2 {worst:.3e}");
    // scale of a 4-bit perturbation on this random tiny model: its original f32 weights vs their MXFP4 rounding
    let w_only = want.iter().zip(&logits(&original(), None)).map(|(a, b)| rel_l2(a, b)).fold(0f64, f64::max);
    println!("MXFP4 weights vs f32 weights worst rel l2 {w_only:.3e}");
    assert!(worst > 1e-4, "W4A4 must differ from W4A16 (the activation QDQ ran)");
    // W4A8 (per-token fp8 activations) runs through the whole hybrid model (GDN + attention + FFN, the pair fusions included) and sits
    // closer to W4A16 than W4A4 does: e4m3 keeps 3 mantissa bits of every element, e2m1 one
    let w4a8 = logits(&mx, Some(Mxfp4Act::Fp8));
    let mut worst8 = 0f64;
    for (g, w) in w4a8.iter().zip(&want) {
        assert!(g.iter().all(|v| v.is_finite()));
        worst8 = worst8.max(rel_l2(g, w));
    }
    println!("W4A8 vs W4A16 worst rel l2 {worst8:.3e}");
    assert!(worst8 > 1e-6, "W4A8 must differ from W4A16 (the fp8 quantisation ran)");
    assert!(worst8 < worst, "W4A8 ({worst8:.3e}) must be closer to W4A16 than W4A4 ({worst:.3e})");
}

/// The fixture's original f32 weights in the VL layout.
fn original() -> MemWeights {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_qwen3_5");
    let mut w = MemWeights::new();
    for (name, (shape, vals)) in read_f32_safetensors(&dir.join("model.safetensors")) {
        let vl = name.strip_prefix("model.").map(|r| format!("model.language_model.{r}")).unwrap_or(name.clone());
        w.insert(&vl, HostTensor::from_f32(&shape, &vals));
    }
    w
}
