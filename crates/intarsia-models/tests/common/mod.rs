#![allow(dead_code)]
//! A tiny synthetic Qwen3-shaped model for end-to-end tests without a checkpoint.

use intarsia_core::engine::{ChatOpts, TextCodec};
use intarsia_core::model::{LoadOptions, Model};
use intarsia_core::recipe::{ModelDims, Recipe};
use intarsia_core::rng::SplitMix64;
use intarsia_core::weights::MemWeights;
use intarsia_core::{DType, HostTensor, Result, Telemetry};
use std::sync::Arc;

pub const H: usize = 32;
pub const NH: usize = 4;
pub const NKV: usize = 2;
pub const HD: usize = 16; // head_dim != hidden / heads on purpose (Qwen3 style)
pub const I: usize = 48;
pub const V: usize = 300;
pub const LAYERS: usize = 3;

pub fn dims() -> ModelDims {
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
        tie_embeddings: true,
        max_positions: 512,
    }
}

pub fn recipe() -> Recipe {
    intarsia_models::parse_recipe(include_str!("../../../../recipes/qwen3.toml")).unwrap()
}

pub struct Tiny {
    pub weights: MemWeights,
    /// name -> f32 values, for the naive reference
    pub raw: std::collections::BTreeMap<String, (Vec<usize>, Vec<f32>)>,
}

pub fn tiny_weights(dt: DType, seed: u64) -> Tiny {
    let mut rng = SplitMix64::new(seed);
    let mut w = MemWeights::new();
    let mut raw = std::collections::BTreeMap::new();
    let mut add = |name: &str, shape: &[usize], scale: f32, norm: bool, rng: &mut SplitMix64| {
        let n: usize = shape.iter().product();
        let vals: Vec<f32> =
            (0..n).map(|_| if norm { 1.0 + 0.1 * rng.normal() } else { scale * rng.normal() }).collect();
        let t = HostTensor::from_f32_as(dt, shape, &vals);
        raw.insert(name.to_string(), (shape.to_vec(), t.to_f32()));
        w.insert(name, t);
    };
    add("model.embed_tokens.weight", &[V, H], 0.5, false, &mut rng);
    add("model.norm.weight", &[H], 0.0, true, &mut rng);
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}");
        add(&format!("{p}.input_layernorm.weight"), &[H], 0.0, true, &mut rng);
        add(&format!("{p}.post_attention_layernorm.weight"), &[H], 0.0, true, &mut rng);
        add(&format!("{p}.self_attn.q_proj.weight"), &[NH * HD, H], 0.15, false, &mut rng);
        add(&format!("{p}.self_attn.k_proj.weight"), &[NKV * HD, H], 0.15, false, &mut rng);
        add(&format!("{p}.self_attn.v_proj.weight"), &[NKV * HD, H], 0.15, false, &mut rng);
        add(&format!("{p}.self_attn.o_proj.weight"), &[H, NH * HD], 0.15, false, &mut rng);
        add(&format!("{p}.self_attn.q_norm.weight"), &[HD], 0.0, true, &mut rng);
        add(&format!("{p}.self_attn.k_norm.weight"), &[HD], 0.0, true, &mut rng);
        add(&format!("{p}.mlp.gate_proj.weight"), &[I, H], 0.15, false, &mut rng);
        add(&format!("{p}.mlp.up_proj.weight"), &[I, H], 0.15, false, &mut rng);
        add(&format!("{p}.mlp.down_proj.weight"), &[H, I], 0.15, false, &mut rng);
    }
    Tiny { weights: w, raw }
}

pub fn load(t: &Tiny, opts: &LoadOptions) -> Model {
    Model::load(
        intarsia_backend_cpu::CpuBackend::arc(),
        &recipe(),
        &dims(),
        &t.weights,
        opts,
        Arc::new(Telemetry::new()),
    )
    .unwrap()
}

/// id = byte value (+0), vocab 300; ids >= 256 render as "<N>".
pub struct ByteCodec;

impl TextCodec for ByteCodec {
    fn encode(&self, text: &str, _s: bool) -> Result<Vec<u32>> {
        Ok(text.bytes().map(|b| b as u32).collect())
    }
    fn decode(&self, ids: &[u32], _s: bool) -> Result<String> {
        let mut bytes = Vec::new();
        for &i in ids {
            if i < 256 {
                bytes.push(i as u8);
            } else {
                bytes.extend_from_slice(format!("<{i}>").as_bytes());
            }
        }
        Ok(String::from_utf8_lossy(&bytes).to_string())
    }
    fn apply_chat_template(&self, _m: &[serde_json::Value], _o: &ChatOpts) -> Result<String> {
        Ok(String::new())
    }
    fn has_chat_template(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// naive f64 reference decoder over the raw weights (no ops, no cache)
// ---------------------------------------------------------------------------

fn mat(raw: &Tiny, name: &str) -> (usize, usize, Vec<f64>) {
    let (s, v) = &raw.raw[name];
    (s[0], s[1], v.iter().map(|x| *x as f64).collect())
}

fn vecw(raw: &Tiny, name: &str) -> Vec<f64> {
    raw.raw[name].1.iter().map(|x| *x as f64).collect()
}

fn rms(x: &[f64], w: &[f64]) -> Vec<f64> {
    let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let r = 1.0 / (ms + 1e-6).sqrt();
    x.iter().zip(w).map(|(a, b)| a * r * b).collect()
}

fn linear(w: &(usize, usize, Vec<f64>), x: &[f64]) -> Vec<f64> {
    (0..w.0).map(|n| (0..w.1).map(|k| w.2[n * w.1 + k] * x[k]).sum()).collect()
}

fn rope(x: &mut [f64], pos: usize) {
    let half = HD / 2;
    let orig = x.to_vec();
    for i in 0..half {
        let ang = pos as f64 * 10000f64.powf(-2.0 * i as f64 / HD as f64);
        let (s, c) = ang.sin_cos();
        x[i] = orig[i] * c - orig[i + half] * s;
        x[i + half] = orig[i + half] * c + orig[i] * s;
    }
}

/// Logits `[len][V]` for every position.
pub fn naive_logits(raw: &Tiny, ids: &[u32]) -> Vec<Vec<f64>> {
    let emb = mat(raw, "model.embed_tokens.weight");
    let mut hs: Vec<Vec<f64>> = ids.iter().map(|&i| emb.2[i as usize * H..(i as usize + 1) * H].to_vec()).collect();
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}");
        // attention
        let nw = vecw(raw, &format!("{p}.input_layernorm.weight"));
        let (wq, wk, wv, wo) = (
            mat(raw, &format!("{p}.self_attn.q_proj.weight")),
            mat(raw, &format!("{p}.self_attn.k_proj.weight")),
            mat(raw, &format!("{p}.self_attn.v_proj.weight")),
            mat(raw, &format!("{p}.self_attn.o_proj.weight")),
        );
        let (qn, kn) =
            (vecw(raw, &format!("{p}.self_attn.q_norm.weight")), vecw(raw, &format!("{p}.self_attn.k_norm.weight")));
        let mut ks: Vec<Vec<Vec<f64>>> = vec![]; // [t][kvh][hd]
        let mut vs: Vec<Vec<Vec<f64>>> = vec![];
        let mut qs: Vec<Vec<Vec<f64>>> = vec![]; // [t][h][hd]
        for (t, h) in hs.iter().enumerate() {
            let n = rms(h, &nw);
            let q = linear(&wq, &n);
            let k = linear(&wk, &n);
            let v = linear(&wv, &n);
            let mut qh: Vec<Vec<f64>> = q
                .chunks(HD)
                .map(|c| {
                    let mut c = rms(c, &qn);
                    rope(&mut c, t);
                    c
                })
                .collect();
            let kh: Vec<Vec<f64>> = k
                .chunks(HD)
                .map(|c| {
                    let mut c = rms(c, &kn);
                    rope(&mut c, t);
                    c
                })
                .collect();
            let vh: Vec<Vec<f64>> = v.chunks(HD).map(|c| c.to_vec()).collect();
            qs.push(std::mem::take(&mut qh));
            ks.push(kh);
            vs.push(vh);
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
            let o = linear(&wo, &cat);
            for (a, b) in hs[t].iter_mut().zip(o) {
                *a += b;
            }
        }
        // ffn
        let nw2 = vecw(raw, &format!("{p}.post_attention_layernorm.weight"));
        let (g, u, d) = (
            mat(raw, &format!("{p}.mlp.gate_proj.weight")),
            mat(raw, &format!("{p}.mlp.up_proj.weight")),
            mat(raw, &format!("{p}.mlp.down_proj.weight")),
        );
        for h in hs.iter_mut() {
            let n = rms(h, &nw2);
            let gv = linear(&g, &n);
            let uv = linear(&u, &n);
            let a: Vec<f64> = gv.iter().zip(&uv).map(|(g, u)| g / (1.0 + (-g).exp()) * u).collect();
            let o = linear(&d, &a);
            for (x, y) in h.iter_mut().zip(o) {
                *x += y;
            }
        }
    }
    let fw = vecw(raw, "model.norm.weight");
    hs.iter()
        .map(|h| {
            let n = rms(h, &fw);
            linear(&emb, &n)
        })
        .collect()
}
