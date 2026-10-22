//! A Qwen3-shaped model whose attention projections and lm_head are EXL3 (ExLlamaV3 trellis), loaded through the
//! qwen3 recipe on the cpu backend, against the same model with the dequantised weights stored as plain f32: the EXL3
//! loader, `Op::MatmulExl3`, the EXL3 lm_head (no `lm_head.weight`, only `lm_head.trellis`) and the declared
//! `quant_method = "exl3"` scheme together. Dims are multiples of 128 (the Hadamard blocks); the dense FFN stays
//! plain (it loads plain tensors on main).

use intarsia_core::model::{HeadMode, LoadOptions, Model};
use intarsia_core::quant::exl3::{self, Codebook};
use intarsia_core::quant::scheme::QuantScheme;
use intarsia_core::recipe::ModelDims;
use intarsia_core::rng::SplitMix64;
use intarsia_core::weights::{MemWeights, WeightInfo, WeightSource};
use intarsia_core::{DType, HostTensor, Result, Telemetry};
use std::sync::Arc;

const H: usize = 256;
const NH: usize = 2;
const NKV: usize = 2;
const HD: usize = 128;
const I: usize = 384;
const V: usize = 384;
const LAYERS: usize = 2;

fn dims() -> ModelDims {
    ModelDims {
        hidden: H,
        n_layers: LAYERS,
        n_heads: NH,
        n_kv_heads: NKV,
        head_dim: HD,
        intermediate: I,
        vocab: V,
        rms_eps: 1e-6,
        rope_theta: 10000.0,
        partial_rotary_factor: 1.0,
        tie_embeddings: false,
        max_positions: 512,
        layer_types: Vec::new(),
        linear: None,
        moe: None,
        gemma: None,
    }
}

struct Declared {
    inner: MemWeights,
    scheme: Option<QuantScheme>,
}

impl WeightSource for Declared {
    fn names(&self) -> Vec<String> {
        self.inner.names()
    }
    fn info(&self, name: &str) -> Option<WeightInfo> {
        self.inner.info(name)
    }
    fn bytes(&self, name: &str) -> Result<&[u8]> {
        self.inner.bytes(name)
    }
    fn quant_scheme(&self) -> Option<&QuantScheme> {
        self.scheme.as_ref()
    }
}

/// (EXL3 checkpoint, dequantised f32 checkpoint, number of EXL3 modules). Widths and codebooks vary per module.
fn checkpoints(ffn_exl3: bool) -> (MemWeights, MemWeights, usize) {
    let mut rng = SplitMix64::new(21);
    let (mut ex, mut deq) = (MemWeights::new(), MemWeights::new());
    let plain = |name: &str, shape: &[usize], vals: Vec<f32>, ex: &mut MemWeights, deq: &mut MemWeights| {
        let t = HostTensor::from_f32(shape, &vals);
        ex.insert(name, t.clone());
        deq.insert(name, t);
    };
    let norm = |n: usize, rng: &mut SplitMix64| (0..n).map(|_| 1.0 + 0.1 * rng.normal()).collect::<Vec<f32>>();
    plain("model.embed_tokens.weight", &[V, H], rng.normal_vec(V * H, 0.5), &mut ex, &mut deq);
    plain("model.norm.weight", &[H], norm(H, &mut rng), &mut ex, &mut deq);
    let widths = [(8usize, Codebook::Inst3), (4, Codebook::Mcg), (5, Codebook::Mul1), (6, Codebook::Inst3), (12, Codebook::Mcg)];
    let mut nq = 0;
    let mut quant = |module: &str, n: usize, k: usize, rng: &mut SplitMix64, ex: &mut MemWeights, deq: &mut MemWeights| {
        let (k2, cb) = widths[nq % widths.len()];
        let q = exl3::random(rng, n, k, k2, cb);
        deq.insert(&format!("{module}.weight"), HostTensor::from_f32(&[n, k], &q.dequant()));
        ex.insert(&format!("{module}.trellis"), q.parts[0].clone());
        ex.insert(&format!("{module}.suh"), q.parts[1].clone());
        ex.insert(&format!("{module}.svh"), q.parts[2].clone());
        if let Some((suffix, v)) = cb.marker() {
            ex.insert(&format!("{module}.{suffix}"), HostTensor { dtype: DType::I32, shape: vec![], data: v.to_le_bytes().to_vec() });
        }
        nq += 1;
    };
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}");
        plain(&format!("{p}.input_layernorm.weight"), &[H], norm(H, &mut rng), &mut ex, &mut deq);
        plain(&format!("{p}.post_attention_layernorm.weight"), &[H], norm(H, &mut rng), &mut ex, &mut deq);
        plain(&format!("{p}.self_attn.q_norm.weight"), &[HD], norm(HD, &mut rng), &mut ex, &mut deq);
        plain(&format!("{p}.self_attn.k_norm.weight"), &[HD], norm(HD, &mut rng), &mut ex, &mut deq);
        quant(&format!("{p}.self_attn.q_proj"), NH * HD, H, &mut rng, &mut ex, &mut deq);
        quant(&format!("{p}.self_attn.k_proj"), NKV * HD, H, &mut rng, &mut ex, &mut deq);
        quant(&format!("{p}.self_attn.v_proj"), NKV * HD, H, &mut rng, &mut ex, &mut deq);
        quant(&format!("{p}.self_attn.o_proj"), H, NH * HD, &mut rng, &mut ex, &mut deq);
        for (m, n, k) in [("gate_proj", I, H), ("up_proj", I, H), ("down_proj", H, I)] {
            if ffn_exl3 {
                quant(&format!("{p}.mlp.{m}"), n, k, &mut rng, &mut ex, &mut deq);
            } else {
                plain(&format!("{p}.mlp.{m}.weight"), &[n, k], rng.normal_vec(n * k, 0.1), &mut ex, &mut deq);
            }
        }
    }
    quant("lm_head", V, H, &mut rng, &mut ex, &mut deq);
    (ex, deq, nq)
}

fn logits(src: &dyn WeightSource) -> Vec<Vec<f32>> {
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/qwen3.toml")).unwrap();
    let m = Model::load(
        intarsia_backend_cpu::CpuBackend::arc(),
        &recipe,
        &dims(),
        src,
        &LoadOptions { max_seq: 64, prefill_chunk: 5, act_dtype: Some(DType::F32), kv_dtype: Some(DType::F32), ..Default::default() },
        Arc::new(Telemetry::new()),
    )
    .unwrap();
    let ids: Vec<u32> = (0..13u32).map(|i| (i * 37 + 5) % V as u32).collect();
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
fn exl3_model_matches_the_dequantised_model() {
    for ffn_exl3 in [false, true] {
        exl3_model_case(ffn_exl3);
    }
}

/// `ffn_exl3`: the dense FFN projections are EXL3 too (every linear layer, as in an ExLlamaV3 Qwen3 checkpoint; needs
/// the dense FFN to load quantised projections, M1's `e6bc0380`).
fn exl3_model_case(ffn_exl3: bool) {
    let (ex, deq, nq) = checkpoints(ffn_exl3);
    assert_eq!(nq, if ffn_exl3 { 7 } else { 4 } * LAYERS + 1);
    let scheme = QuantScheme::parse_exl3(&serde_json::json!({"quant_method":"exl3","bits":4.0,"head_bits":6})).unwrap();
    let want = logits(&deq);
    let got = logits(&Declared { inner: ex, scheme: Some(scheme) });
    assert_eq!(got.len(), want.len());
    let worst = got.iter().zip(&want).map(|(g, w)| rel_l2(g, w)).fold(0f64, f64::max);
    println!("EXL3 model (ffn exl3: {ffn_exl3}) vs dequantised f32 model: worst rel l2 {worst:.3e} over {} positions", got.len());
    assert!(worst < 1e-5, "worst rel l2 {worst}");
}

#[test]
fn a_declared_exl3_scheme_refuses_other_quantised_layers() {
    let (_, deq, _) = checkpoints(false);
    let mut w = deq;
    // an fp8 projection in an EXL3 checkpoint is not what exllamav3 writes
    w.insert("model.layers.0.self_attn.q_proj.weight", HostTensor::from_f32_as(DType::Fp8E4m3, &[NH * HD, H], &vec![0.0; NH * HD * H]));
    w.insert("model.layers.0.self_attn.q_proj.weight_scale", HostTensor::from_f32(&[], &[1.0]));
    w.insert("model.layers.0.self_attn.q_proj.input_scale", HostTensor::from_f32(&[], &[1.0]));
    let scheme = QuantScheme::parse_exl3(&serde_json::json!({"quant_method":"exl3"})).unwrap();
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/qwen3.toml")).unwrap();
    let e = Model::load(
        intarsia_backend_cpu::CpuBackend::arc(),
        &recipe,
        &dims(),
        &Declared { inner: w, scheme: Some(scheme) },
        &LoadOptions { max_seq: 64, ..Default::default() },
        Arc::new(Telemetry::new()),
    )
    .err()
    .expect("fp8 layer under an EXL3 scheme must not load")
    .to_string();
    assert!(e.contains("EXL3") && e.contains("q_proj"), "{e}");
}
