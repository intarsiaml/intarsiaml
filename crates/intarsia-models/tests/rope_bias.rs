//! Scaled rotary embeddings (llama3, yarn) and q / k / v projection biases end to end on the CPU backend: a tiny
//! Llama / Qwen2-shaped decoder (no qk-norm, biases) and a Qwen3-shaped one with scaling, against a naive f64 decoder
//! that evaluates the same formulas (`rope::RopeScaling::freqs` for the table, which `rope.rs` checks against
//! transformers' values) without ops, caches or fusion.

mod common;
use common::*;
use intarsia_core::model::{HeadMode, LoadOptions, Model};
use intarsia_core::recipe::ModelDims;
use intarsia_core::rng::SplitMix64;
use intarsia_core::rope::RopeScaling;
use intarsia_core::weights::MemWeights;
use intarsia_core::{DType, HostTensor, Telemetry};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Clone)]
struct Variant {
    bias: bool,
    qk_norm: bool,
    rope: RopeScaling,
}

fn llama3_small() -> RopeScaling {
    // original 16: pair 0 (wavelength 6.3) is in the blend zone, pairs >= 1 (wavelength >= 19.9) are divided by 8
    RopeScaling::Llama3 { factor: 8.0, low_freq_factor: 1.0, high_freq_factor: 4.0, original_max: 16.0 }
}

fn yarn_small() -> RopeScaling {
    RopeScaling::Yarn {
        factor: 4.0,
        original_max: 32.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        truncate: true,
        attention_factor: None,
        mscale: None,
        mscale_all_dim: None,
    }
}

fn recipe_for(v: &Variant) -> intarsia_core::recipe::Recipe {
    let mut t = include_str!("../../../recipes/qwen3.toml").to_string();
    if !v.qk_norm {
        t = t.replace("q_norm = \"model.layers.{i}.self_attn.q_norm\"\n", "");
        t = t.replace("k_norm = \"model.layers.{i}.self_attn.k_norm\"\n", "");
    }
    if v.bias {
        t = t.replace(
            "o = \"model.layers.{i}.self_attn.o_proj\"\n",
            "o = \"model.layers.{i}.self_attn.o_proj\"\n\
             q_bias = \"model.layers.{i}.self_attn.q_proj.bias\"\n\
             k_bias = \"model.layers.{i}.self_attn.k_proj.bias\"\n\
             v_bias = \"model.layers.{i}.self_attn.v_proj.bias\"\n",
        );
    }
    intarsia_models::parse_recipe(&t).unwrap()
}

struct Raw {
    weights: MemWeights,
    raw: BTreeMap<String, Vec<f64>>,
}

fn weights_for(v: &Variant, dt: DType, seed: u64) -> Raw {
    let mut rng = SplitMix64::new(seed);
    let mut w = MemWeights::new();
    let mut raw = BTreeMap::new();
    let mut add = |name: &str, shape: &[usize], scale: f32, norm: bool| {
        let n: usize = shape.iter().product();
        let vals: Vec<f32> = (0..n).map(|_| if norm { 1.0 + 0.1 * rng.normal() } else { scale * rng.normal() }).collect();
        let t = HostTensor::from_f32_as(dt, shape, &vals);
        raw.insert(name.to_string(), t.to_f32().iter().map(|x| *x as f64).collect::<Vec<f64>>());
        w.insert(name, t);
    };
    add("model.embed_tokens.weight", &[V, H], 0.5, false);
    add("model.norm.weight", &[H], 0.0, true);
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}");
        add(&format!("{p}.input_layernorm.weight"), &[H], 0.0, true);
        add(&format!("{p}.post_attention_layernorm.weight"), &[H], 0.0, true);
        add(&format!("{p}.self_attn.q_proj.weight"), &[NH * HD, H], 0.15, false);
        add(&format!("{p}.self_attn.k_proj.weight"), &[NKV * HD, H], 0.15, false);
        add(&format!("{p}.self_attn.v_proj.weight"), &[NKV * HD, H], 0.15, false);
        add(&format!("{p}.self_attn.o_proj.weight"), &[H, NH * HD], 0.15, false);
        if v.bias {
            // large enough that a dropped bias moves the logits far beyond the tolerance
            add(&format!("{p}.self_attn.q_proj.bias"), &[NH * HD], 0.5, false);
            add(&format!("{p}.self_attn.k_proj.bias"), &[NKV * HD], 0.5, false);
            add(&format!("{p}.self_attn.v_proj.bias"), &[NKV * HD], 0.5, false);
        }
        if v.qk_norm {
            add(&format!("{p}.self_attn.q_norm.weight"), &[HD], 0.0, true);
            add(&format!("{p}.self_attn.k_norm.weight"), &[HD], 0.0, true);
        }
        add(&format!("{p}.mlp.gate_proj.weight"), &[I, H], 0.15, false);
        add(&format!("{p}.mlp.up_proj.weight"), &[I, H], 0.15, false);
        add(&format!("{p}.mlp.down_proj.weight"), &[H, I], 0.15, false);
    }
    Raw { weights: w, raw }
}

fn dims_for(v: &Variant) -> ModelDims {
    ModelDims { rope_scaling: v.rope, ..dims() }
}

fn load_for(v: &Variant, w: &Raw, kv: DType) -> Model {
    Model::load(
        intarsia_backend_cpu::CpuBackend::arc(),
        &recipe_for(v),
        &dims_for(v),
        &w.weights,
        &LoadOptions { kv_dtype: Some(kv), ..Default::default() },
        Arc::new(Telemetry::new()),
    )
    .unwrap()
}

// ---- naive f64 decoder -------------------------------------------------------------------------------------------

fn rms(x: &[f64], w: Option<&[f64]>) -> Vec<f64> {
    let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let r = 1.0 / (ms + 1e-6).sqrt();
    x.iter().enumerate().map(|(i, a)| a * r * w.map_or(1.0, |w| w[i])).collect()
}

fn linear(w: &[f64], rows: usize, cols: usize, x: &[f64], bias: Option<&[f64]>) -> Vec<f64> {
    (0..rows).map(|n| (0..cols).map(|k| w[n * cols + k] * x[k]).sum::<f64>() + bias.map_or(0.0, |b| b[n])).collect()
}

fn rope(x: &mut [f64], pos: usize, rope: &RopeScaling) {
    let half = HD / 2;
    let orig = x.to_vec();
    let (inv, m): (Vec<f64>, f64) = match rope.freqs(10000.0, HD) {
        Some(f) => (f.inv_freq.iter().map(|v| *v as f64).collect(), f.attention_factor as f64),
        None => ((0..half).map(|i| 10000f64.powf(-2.0 * i as f64 / HD as f64)).collect(), 1.0),
    };
    for i in 0..half {
        let ang = pos as f64 * inv[i];
        let (s, c) = ang.sin_cos();
        let (s, c) = (s * m, c * m);
        x[i] = orig[i] * c - orig[i + half] * s;
        x[i + half] = orig[i + half] * c + orig[i] * s;
    }
}

fn naive(v: &Variant, w: &Raw, ids: &[u32]) -> Vec<Vec<f64>> {
    let g = |name: &str| &w.raw[name];
    let emb = g("model.embed_tokens.weight");
    let mut hs: Vec<Vec<f64>> = ids.iter().map(|&i| emb[i as usize * H..(i as usize + 1) * H].to_vec()).collect();
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}");
        let nw = g(&format!("{p}.input_layernorm.weight"));
        let (wq, wk, wv, wo) = (
            g(&format!("{p}.self_attn.q_proj.weight")),
            g(&format!("{p}.self_attn.k_proj.weight")),
            g(&format!("{p}.self_attn.v_proj.weight")),
            g(&format!("{p}.self_attn.o_proj.weight")),
        );
        let b = |n: &str| v.bias.then(|| g(&format!("{p}.self_attn.{n}_proj.bias")).as_slice());
        let norm = |n: &str| v.qk_norm.then(|| g(&format!("{p}.self_attn.{n}_norm.weight")).as_slice());
        let (mut qs, mut ks, mut vs): (Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>) = (vec![], vec![], vec![]);
        for (t, h) in hs.iter().enumerate() {
            let n = rms(h, Some(nw));
            let q = linear(wq, NH * HD, H, &n, b("q"));
            let k = linear(wk, NKV * HD, H, &n, b("k"));
            let vv = linear(wv, NKV * HD, H, &n, b("v"));
            let heads = |x: &[f64], nm: Option<&[f64]>| -> Vec<Vec<f64>> {
                x.chunks(HD)
                    .map(|c| {
                        let mut c = if nm.is_some() { rms(c, nm) } else { c.to_vec() };
                        rope(&mut c, t, &v.rope);
                        c
                    })
                    .collect()
            };
            qs.push(heads(&q, norm("q")));
            ks.push(heads(&k, norm("k")));
            vs.push(vv.chunks(HD).map(|c| c.to_vec()).collect());
        }
        let group = NH / NKV;
        for t in 0..hs.len() {
            let mut cat = vec![0f64; NH * HD];
            for h in 0..NH {
                let kvh = h / group;
                let sc: Vec<f64> = (0..=t)
                    .map(|j| (0..HD).map(|e| qs[t][h][e] * ks[j][kvh][e]).sum::<f64>() / (HD as f64).sqrt())
                    .collect();
                let mx = sc.iter().cloned().fold(f64::MIN, f64::max);
                let ex: Vec<f64> = sc.iter().map(|s| (s - mx).exp()).collect();
                let den: f64 = ex.iter().sum();
                for j in 0..=t {
                    for e in 0..HD {
                        cat[h * HD + e] += ex[j] / den * vs[j][kvh][e];
                    }
                }
            }
            let o = linear(wo, H, NH * HD, &cat, None);
            for (a, b) in hs[t].iter_mut().zip(o) {
                *a += b;
            }
        }
        let nw2 = g(&format!("{p}.post_attention_layernorm.weight"));
        let (gp, up, dp) = (g(&format!("{p}.mlp.gate_proj.weight")), g(&format!("{p}.mlp.up_proj.weight")), g(&format!("{p}.mlp.down_proj.weight")));
        for h in hs.iter_mut() {
            let n = rms(h, Some(nw2));
            let gv = linear(gp, I, H, &n, None);
            let uv = linear(up, I, H, &n, None);
            let a: Vec<f64> = gv.iter().zip(&uv).map(|(g, u)| g / (1.0 + (-g).exp()) * u).collect();
            let o = linear(dp, H, I, &a, None);
            for (x, y) in h.iter_mut().zip(o) {
                *x += y;
            }
        }
    }
    let fw = g("model.norm.weight");
    hs.iter().map(|h| linear(emb, V, H, &rms(h, Some(fw)), None)).collect()
}

fn ids(n: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * 37 + 11) % V) as u32).collect()
}

fn all_logits(m: &Model, ids: &[u32]) -> Vec<Vec<f32>> {
    let mut st = m.new_state().unwrap();
    let mut rows = vec![];
    let mut cb = |_p: usize, r: &[f32]| {
        rows.push(r.to_vec());
        Ok(())
    };
    m.forward(&mut st, ids, HeadMode::AllLogits, Some(&mut cb)).unwrap();
    rows
}

fn rel_l2(got: &[Vec<f32>], want: &[Vec<f64>]) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for (g, w) in got.iter().zip(want) {
        num += g.iter().zip(w).map(|(a, b)| (*a as f64 - b).powi(2)).sum::<f64>();
        den += w.iter().map(|b| b * b).sum::<f64>();
    }
    (num / den).sqrt()
}

fn check(v: &Variant, seed: u64, n: usize) -> f64 {
    let w = weights_for(v, DType::F32, seed);
    let m = load_for(v, &w, DType::F32);
    let input = ids(n);
    let got = all_logits(&m, &input);
    let want = naive(v, &w, &input);
    assert_eq!(got.len(), want.len());
    let e = rel_l2(&got, &want);
    assert!(e < 1e-4, "rel_l2 {e}");
    e
}

#[test]
fn llama_shaped_llama3_rope_no_norm_no_bias() {
    check(&Variant { bias: false, qk_norm: false, rope: llama3_small() }, 11, 20);
}

#[test]
fn qwen2_shaped_biases_with_plain_rope() {
    check(&Variant { bias: true, qk_norm: false, rope: RopeScaling::None }, 12, 12);
}

#[test]
fn qwen2_shaped_biases_with_llama3_and_yarn_rope() {
    check(&Variant { bias: true, qk_norm: false, rope: llama3_small() }, 13, 20);
    check(&Variant { bias: true, qk_norm: false, rope: yarn_small() }, 14, 20);
}

#[test]
fn qk_norm_model_with_scaling_skips_the_fused_prologue() {
    // the CPU backend has the fused QkNormRopeKv; a scaled embedding must still be applied (the fold bails)
    check(&Variant { bias: false, qk_norm: true, rope: llama3_small() }, 15, 20);
    check(&Variant { bias: false, qk_norm: true, rope: yarn_small() }, 16, 20);
}

#[test]
fn the_scaling_and_the_biases_are_not_vacuous() {
    // same weights, scaled vs plain rope: the logits differ by far more than the tolerance
    let v = Variant { bias: true, qk_norm: false, rope: llama3_small() };
    let w = weights_for(&v, DType::F32, 21);
    let input = ids(20);
    let scaled = naive(&v, &w, &input);
    let plain = naive(&Variant { rope: RopeScaling::None, ..v.clone() }, &w, &input);
    let d: f64 = scaled.iter().zip(&plain).map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>()).sum::<f64>().sqrt();
    let n: f64 = plain.iter().map(|a| a.iter().map(|x| x * x).sum::<f64>()).sum::<f64>().sqrt();
    assert!(d / n > 1e-2, "scaling moved the logits by {}", d / n);
    // and the engine, loaded with plain rope on the same weights, is NOT the scaled reference
    let m = load_for(&Variant { rope: RopeScaling::None, ..v.clone() }, &w, DType::F32);
    let got = all_logits(&m, &input);
    assert!(rel_l2(&got, &scaled) > 1e-2);
}

#[test]
fn decode_steps_equal_prefill_with_scaled_rope_and_bias() {
    let v = Variant { bias: true, qk_norm: false, rope: yarn_small() };
    let w = weights_for(&v, DType::F32, 17);
    let m = load_for(&v, &w, DType::F32);
    let input = ids(14);
    let full = all_logits(&m, &input);
    // one token at a time: the single-row decode path (position from the cache length)
    let mut st = m.new_state().unwrap();
    let mut rows = vec![];
    for tok in &input {
        let mut cb = |_p: usize, r: &[f32]| {
            rows.push(r.to_vec());
            Ok(())
        };
        m.forward(&mut st, std::slice::from_ref(tok), HeadMode::AllLogits, Some(&mut cb)).unwrap();
    }
    assert_eq!(rows.len(), full.len());
    for (p, (a, b)) in rows.iter().zip(&full).enumerate() {
        let d: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>().sqrt();
        let n: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum::<f64>().sqrt();
        assert!(d / n < 1e-4, "pos {p}: {}", d / n);
    }
}

#[test]
fn a_checkpoint_bias_the_recipe_does_not_map_is_a_load_error() {
    let v = Variant { bias: true, qk_norm: false, rope: RopeScaling::None };
    let w = weights_for(&v, DType::F32, 22);
    let unmapped = Variant { bias: false, ..v };
    let r = Model::load(
        intarsia_backend_cpu::CpuBackend::arc(),
        &recipe_for(&unmapped),
        &dims_for(&unmapped),
        &w.weights,
        &LoadOptions::default(),
        Arc::new(Telemetry::new()),
    );
    let e = r.err().expect("must refuse").to_string();
    assert!(e.contains("q_proj.bias") && e.contains("recipe maps no bias"), "{e}");
}
