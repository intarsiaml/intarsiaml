//! `--spec mtp` (the checkpoint's multi-token-prediction head as the speculative drafter) on the cpu backend.
//!
//! The model is the tiny hybrid MoE of `tp_hybrid.rs` (`tests/fixtures/tiny_qwen3_5`: 4 GatedDeltaNet and 2 gated
//! full-attention layers, the `qwen3_5_moe` recipe, NVFP4 experts, FP8 attention / GDN projections) plus a random MTP
//! head in the 35B checkpoint's layout (`mtp.*`, bf16, fused `experts.gate_up_proj` / `experts.down_proj`).
//!
//! * The head's forward matches an f64 host implementation of vLLM's `Qwen3NextMultiTokenPredictor` (concat order,
//!   norms, gated attention, MoE with the load-time NVFP4 experts, the model's lm_head).
//! * Greedy output with MTP on is token-identical to MTP off over 256 tokens, with the head's own drafts and with
//!   scripted drafts that hit every acceptance pattern (all, some, none accepted) through the GatedDeltaNet rollback,
//!   at k = 1..3, across `max_tokens` and stop ids landing inside an accepted run, and for a follow-up request that
//!   reuses the prefix. The cpu kernels run every row of a multi-row call with the single-row loop, so verify rows are
//!   the decode rows here; this proves the engine logic, not a GPU backend's numerics.

mod common;
use common::ByteCodec;
use intarsia_backend_cpu::CpuBackend;
use intarsia_core::dtype::f32_to_fp8_e4m3;
use intarsia_core::model::{DraftScript, HeadMode, LoadOptions, Model, StepOut};
use intarsia_core::quant::{nvfp4, Fp8Act};
use intarsia_core::recipe::ModelDims;
use intarsia_core::rng::SplitMix64;
use intarsia_core::spec::{SpecConfig, SpecMode};
use intarsia_core::weights::{MemWeights, WeightSource};
use intarsia_core::{
    DType, Engine, EngineMeta, FinishReason, GenEvent, GenRequest, HostTensor, LocalEngine, Telemetry,
};
use intarsia_models::SafeTensors;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const E: usize = 8;
const TOP_K: usize = 2;
const I: usize = 32;
const IS: usize = 32;
const H: usize = 64;
const LAYERS: usize = 6;
const V: u32 = 96;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_qwen3_5")
}

fn put_nvfp4(w: &mut MemWeights, rng: &mut SplitMix64, prefix: &str, n: usize, k: usize) {
    let q = nvfp4::random_rows(rng, n, k);
    w.insert(&format!("{prefix}.weight"), q.parts[0].clone());
    w.insert(&format!("{prefix}.weight_scale"), q.parts[1].clone());
    w.insert(&format!("{prefix}.weight_scale_2"), q.parts[2].clone());
}

fn bf16(rng: &mut SplitMix64, shape: &[usize], scale: f32) -> HostTensor {
    HostTensor::from_f32_as(DType::Bf16, shape, &rng.normal_vec(shape.iter().product(), scale))
}

fn to_fp8(w: &mut MemWeights, prefix: &str) {
    let name = format!("{prefix}.weight");
    let info = w.info(&name).unwrap_or_else(|| panic!("{name} missing"));
    let t = HostTensor::new(info.dtype, info.shape.clone(), w.bytes(&name).unwrap().to_vec()).unwrap();
    let v = t.to_f32();
    let amax = v.iter().fold(0f32, |m, x| m.max(x.abs()));
    let ws = if amax > 0.0 { amax / 448.0 } else { 1.0 };
    let data: Vec<u8> = v.iter().map(|x| f32_to_fp8_e4m3(x / ws)).collect();
    w.insert(&name, HostTensor::new(DType::Fp8E4m3, info.shape, data).unwrap());
    w.insert(&format!("{prefix}.weight_scale"), HostTensor::from_f32(&[], &[ws]));
}

/// Head dimensions of the fixture (config.json): 4 query heads, 2 KV heads, head_dim 32.
const NH: usize = 4;
const NKV: usize = 2;
const HD: usize = 32;

/// The tiny hybrid MoE (as in `tp_hybrid.rs`) plus an MTP head: `mtp_scale` scales the head's weights (small values keep
/// its drafts close to "copy the embedding", which a random model then sometimes accepts).
fn weights(seed: u64) -> MemWeights {
    let src = SafeTensors::open_dir(&fixture_dir()).unwrap();
    let mut w = MemWeights::new();
    for name in src.names() {
        if name.contains(".mlp.") {
            continue;
        }
        let info = src.info(&name).unwrap();
        let t = HostTensor::new(info.dtype, info.shape.clone(), src.bytes(&name).unwrap().to_vec()).unwrap();
        let renamed = if let Some(rest) = name.strip_prefix("model.") {
            format!("model.language_model.{rest}")
        } else {
            name.clone()
        };
        w.insert(&renamed, t);
    }
    let mut rng = SplitMix64::new(0x7b_35);
    for l in 0..LAYERS {
        let p = format!("model.language_model.layers.{l}");
        let attn = [".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj", ".self_attn.o_proj"];
        let gdn = [".linear_attn.in_proj_qkv", ".linear_attn.in_proj_z", ".linear_attn.out_proj"];
        for s in attn.iter().chain(&gdn) {
            if w.info(&format!("{p}{s}.weight")).is_some() {
                to_fp8(&mut w, &format!("{p}{s}"));
            }
        }
        let p = format!("{p}.mlp");
        w.insert(&format!("{p}.gate.weight"), bf16(&mut rng, &[E, H], 0.3));
        w.insert(&format!("{p}.shared_expert_gate.weight"), bf16(&mut rng, &[1, H], 0.3));
        for x in 0..E {
            put_nvfp4(&mut w, &mut rng, &format!("{p}.experts.{x}.gate_proj"), I, H);
            put_nvfp4(&mut w, &mut rng, &format!("{p}.experts.{x}.up_proj"), I, H);
            put_nvfp4(&mut w, &mut rng, &format!("{p}.experts.{x}.down_proj"), H, I);
        }
        put_nvfp4(&mut w, &mut rng, &format!("{p}.shared_expert.gate_proj"), IS, H);
        put_nvfp4(&mut w, &mut rng, &format!("{p}.shared_expert.up_proj"), IS, H);
        put_nvfp4(&mut w, &mut rng, &format!("{p}.shared_expert.down_proj"), H, IS);
    }
    // the MTP head (bf16, the 35B checkpoint's names and layouts)
    let mut rng = SplitMix64::new(seed);
    let m = "mtp";
    let l = "mtp.layers.0";
    w.insert(&format!("{m}.fc.weight"), bf16(&mut rng, &[H, 2 * H], 0.15));
    for n in [
        format!("{m}.pre_fc_norm_embedding"),
        format!("{m}.pre_fc_norm_hidden"),
        format!("{m}.norm"),
        format!("{l}.input_layernorm"),
        format!("{l}.post_attention_layernorm"),
    ] {
        w.insert(&format!("{n}.weight"), bf16(&mut rng, &[H], 0.2));
    }
    w.insert(&format!("{l}.self_attn.q_proj.weight"), bf16(&mut rng, &[NH * HD * 2, H], 0.15));
    w.insert(&format!("{l}.self_attn.k_proj.weight"), bf16(&mut rng, &[NKV * HD, H], 0.15));
    w.insert(&format!("{l}.self_attn.v_proj.weight"), bf16(&mut rng, &[NKV * HD, H], 0.15));
    w.insert(&format!("{l}.self_attn.o_proj.weight"), bf16(&mut rng, &[H, NH * HD], 0.15));
    w.insert(&format!("{l}.self_attn.q_norm.weight"), bf16(&mut rng, &[HD], 0.2));
    w.insert(&format!("{l}.self_attn.k_norm.weight"), bf16(&mut rng, &[HD], 0.2));
    let p = format!("{l}.mlp");
    w.insert(&format!("{p}.gate.weight"), bf16(&mut rng, &[E, H], 0.3));
    w.insert(&format!("{p}.shared_expert_gate.weight"), bf16(&mut rng, &[1, H], 0.3));
    w.insert(&format!("{p}.experts.gate_up_proj"), bf16(&mut rng, &[E, 2 * I, H], 0.15));
    w.insert(&format!("{p}.experts.down_proj"), bf16(&mut rng, &[E, H, I], 0.15));
    w.insert(&format!("{p}.shared_expert.gate_proj.weight"), bf16(&mut rng, &[IS, H], 0.15));
    w.insert(&format!("{p}.shared_expert.up_proj.weight"), bf16(&mut rng, &[IS, H], 0.15));
    w.insert(&format!("{p}.shared_expert.down_proj.weight"), bf16(&mut rng, &[H, IS], 0.15));
    w
}

fn dims() -> (intarsia_core::recipe::Recipe, ModelDims) {
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/qwen3_5_moe.toml")).unwrap();
    let mut cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture_dir().join("config.json")).unwrap()).unwrap();
    cfg["architectures"] = serde_json::json!(["Qwen3_5MoeForCausalLM"]);
    cfg["num_experts"] = E.into();
    cfg["num_experts_per_tok"] = TOP_K.into();
    cfg["moe_intermediate_size"] = I.into();
    cfg["shared_expert_intermediate_size"] = IS.into();
    cfg["norm_topk_prob"] = true.into();
    cfg.as_object_mut().unwrap().remove("intermediate_size");
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    assert_eq!((dims.n_heads, dims.n_kv_heads, dims.head_dim, dims.hidden), (NH, NKV, HD, H), "fixture dimensions");
    (recipe, dims)
}

fn load(w: &MemWeights, act: DType, mtp: bool, k: Option<usize>) -> Model {
    let (recipe, dims) = dims();
    let opts = LoadOptions {
        max_seq: 512,
        prefill_chunk: 8,
        act_dtype: Some(act),
        kv_dtype: Some(act),
        fp8_act: Some(Fp8Act::None),
        mtp,
        ..Default::default()
    };
    let mut m = Model::load(CpuBackend::arc(), &recipe, &dims, w, &opts, Arc::new(Telemetry::new())).unwrap();
    if let Some(k) = k {
        m.enable_verify(k + 1).unwrap();
        m.configure_plan(&Default::default(), None).unwrap();
    }
    m
}

// ------------------------------------------------------------------------------------------------------------------
// f64 host reference of the head
// ------------------------------------------------------------------------------------------------------------------

fn f(w: &MemWeights, name: &str) -> Vec<f64> {
    let info = w.info(name).unwrap_or_else(|| panic!("{name}"));
    HostTensor::new(info.dtype, info.shape.clone(), w.bytes(name).unwrap().to_vec())
        .unwrap()
        .to_f32()
        .iter()
        .map(|&x| x as f64)
        .collect()
}

/// Zero-centred RMSNorm (`x * rsqrt(mean(x^2) + eps) * (1 + w)`, recipe `norm_offset = 1`).
fn rms(x: &[f64], w: &[f64]) -> Vec<f64> {
    let r = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + 1e-6).sqrt();
    x.iter().zip(w).map(|(v, w)| v * r * (1.0 + w)).collect()
}

/// `y = W x`, `W` `[n, k]` row-major.
fn mv(wm: &[f64], x: &[f64]) -> Vec<f64> {
    let k = x.len();
    wm.chunks(k).map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum()).collect()
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// HF rotate-half rope on the first `rd` values of a head at position `pos`.
fn rope(x: &mut [f64], pos: usize, rd: usize, theta: f64) {
    let half = rd / 2;
    for i in 0..half {
        let inv = theta.powf(-((2 * i) as f64) / rd as f64);
        let (c, s) = ((pos as f64 * inv).cos(), (pos as f64 * inv).sin());
        let (a, b) = (x[i], x[i + half]);
        x[i] = a * c - b * s;
        x[i + half] = b * c + a * s;
    }
}

/// NVFP4 round trip of an `[n, k]` matrix (what the loader stores for the head's routed experts).
fn nvfp4_rt(v: &[f64], n: usize, k: usize) -> Vec<f64> {
    let vf: Vec<f32> = v.iter().map(|&x| x as f32).collect();
    let (w, s, gs) = nvfp4::quantize_rows(&vf, n, k);
    nvfp4::dequant_rows(&w, &s, gs, n, k).iter().map(|&x| x as f64).collect()
}

/// vLLM `Qwen3NextMultiTokenPredictor` over positions `0..t` (empty cache), then the model's lm_head: logits per row.
fn reference(w: &MemWeights, ids: &[u32], hidden: &[f32], theta: f64, rd: usize) -> Vec<Vec<f64>> {
    let t = ids.len();
    let emb = f(w, "model.language_model.embed_tokens.weight");
    let lm = f(w, "lm_head.weight");
    let l = "mtp.layers.0";
    let fc = f(w, "mtp.fc.weight");
    let (ne_w, nh_w, norm_w) =
        (f(w, "mtp.pre_fc_norm_embedding.weight"), f(w, "mtp.pre_fc_norm_hidden.weight"), f(w, "mtp.norm.weight"));
    let (ln1, ln2) =
        (f(w, &format!("{l}.input_layernorm.weight")), f(w, &format!("{l}.post_attention_layernorm.weight")));
    let (wq, wk, wv, wo) = (
        f(w, &format!("{l}.self_attn.q_proj.weight")),
        f(w, &format!("{l}.self_attn.k_proj.weight")),
        f(w, &format!("{l}.self_attn.v_proj.weight")),
        f(w, &format!("{l}.self_attn.o_proj.weight")),
    );
    let (qn, kn) = (f(w, &format!("{l}.self_attn.q_norm.weight")), f(w, &format!("{l}.self_attn.k_norm.weight")));
    let p = format!("{l}.mlp");
    let router = f(w, &format!("{p}.gate.weight"));
    let sg = f(w, &format!("{p}.shared_expert_gate.weight"));
    let gu = f(w, &format!("{p}.experts.gate_up_proj"));
    let dn = f(w, &format!("{p}.experts.down_proj"));
    let (sgp, sup, sdp) = (
        f(w, &format!("{p}.shared_expert.gate_proj.weight")),
        f(w, &format!("{p}.shared_expert.up_proj.weight")),
        f(w, &format!("{p}.shared_expert.down_proj.weight")),
    );
    let experts: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = (0..E)
        .map(|x| {
            let g = nvfp4_rt(&gu[x * 2 * I * H..(x * 2 * I + I) * H], I, H);
            let u = nvfp4_rt(&gu[(x * 2 * I + I) * H..(x + 1) * 2 * I * H], I, H);
            let d = nvfp4_rt(&dn[x * H * I..(x + 1) * H * I], H, I);
            (g, u, d)
        })
        .collect();
    // per row: residual after fc, q / gate per head, k / v per KV head
    let mut res = Vec::new();
    let (mut qs, mut gates, mut ks, mut vs) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (pos, &id) in ids.iter().enumerate() {
        let e = &emb[id as usize * H..(id as usize + 1) * H];
        let hrow: Vec<f64> = hidden[pos * H..(pos + 1) * H].iter().map(|&x| x as f64).collect();
        let mut cat = rms(e, &ne_w);
        cat.extend(rms(&hrow, &nh_w));
        let x = mv(&fc, &cat);
        let n = rms(&x, &ln1);
        let q = mv(&wq, &n);
        let (mut qh, mut gh) = (Vec::new(), Vec::new());
        for h in 0..NH {
            let mut qq = rms(&q[h * 2 * HD..h * 2 * HD + HD], &qn);
            rope(&mut qq, pos, rd, theta);
            qh.push(qq);
            gh.push(q[h * 2 * HD + HD..(h + 1) * 2 * HD].to_vec());
        }
        let (k, v) = (mv(&wk, &n), mv(&wv, &n));
        let mut kh = Vec::new();
        let mut vh = Vec::new();
        for g in 0..NKV {
            let mut kk = rms(&k[g * HD..(g + 1) * HD], &kn);
            rope(&mut kk, pos, rd, theta);
            kh.push(kk);
            vh.push(v[g * HD..(g + 1) * HD].to_vec());
        }
        res.push(x);
        qs.push(qh);
        gates.push(gh);
        ks.push(kh);
        vs.push(vh);
    }
    let mut out = Vec::new();
    for pos in 0..t {
        let mut o = Vec::with_capacity(NH * HD);
        for h in 0..NH {
            let g = h / (NH / NKV);
            let sc: Vec<f64> = (0..=pos)
                .map(|j| qs[pos][h].iter().zip(&ks[j][g]).map(|(a, b)| a * b).sum::<f64>() / (HD as f64).sqrt())
                .collect();
            let mx = sc.iter().cloned().fold(f64::MIN, f64::max);
            let ex: Vec<f64> = sc.iter().map(|s| (s - mx).exp()).collect();
            let z: f64 = ex.iter().sum();
            for d in 0..HD {
                let a: f64 = (0..=pos).map(|j| ex[j] / z * vs[j][g][d]).sum();
                o.push(a * sigmoid(gates[pos][h][d]));
            }
        }
        let r1: Vec<f64> = res[pos].iter().zip(mv(&wo, &o)).map(|(a, b)| a + b).collect();
        let n2 = rms(&r1, &ln2);
        let lg = mv(&router, &n2);
        let mx = lg.iter().cloned().fold(f64::MIN, f64::max);
        let pr: Vec<f64> = lg.iter().map(|x| (x - mx).exp()).collect();
        let mut idx: Vec<usize> = (0..E).collect();
        idx.sort_by(|a, b| pr[*b].partial_cmp(&pr[*a]).unwrap());
        let top = &idx[..TOP_K];
        let ws: f64 = top.iter().map(|&x| pr[x]).sum();
        let mut moe = vec![0f64; H];
        for &x in top {
            let (g, u, d) = &experts[x];
            let a: Vec<f64> = mv(g, &n2).iter().zip(mv(u, &n2)).map(|(g, u)| silu(*g) * u).collect();
            for (m, y) in moe.iter_mut().zip(mv(d, &a)) {
                *m += pr[x] / ws * y;
            }
        }
        let a: Vec<f64> = mv(&sgp, &n2).iter().zip(mv(&sup, &n2)).map(|(g, u)| silu(*g) * u).collect();
        let gate = sigmoid(mv(&sg, &n2)[0]);
        for (m, y) in moe.iter_mut().zip(mv(&sdp, &a)) {
            *m += gate * y;
        }
        let r2: Vec<f64> = r1.iter().zip(&moe).map(|(a, b)| a + b).collect();
        out.push(mv(&lm, &rms(&r2, &norm_w)));
    }
    out
}

#[test]
fn mtp_head_matches_a_host_reference() {
    let w = weights(11);
    let m = load(&w, DType::F32, true, None);
    let ids: Vec<u32> = vec![5, 17, 88, 3, 41, 41, 9];
    let mut rng = SplitMix64::new(99);
    let hidden = rng.normal_vec(ids.len() * H, 1.0);
    let got = m.mtp_reference_pass(&ids, &hidden).unwrap();
    let want = reference(&w, &ids, &hidden, 10000.0, 8);
    let mut worst = 0f64;
    for (r, (g, w)) in got.iter().zip(&want).enumerate() {
        let scale = w.iter().fold(0f64, |a, x| a.max(x.abs()));
        let diff = g.iter().zip(w).fold(0f64, |a, (g, w)| a.max((*g as f64 - w).abs()));
        worst = worst.max(diff / scale);
        let am = |v: &[f64]| v.iter().enumerate().fold((0, f64::MIN), |b, (i, x)| if *x > b.1 { (i, *x) } else { b }).0;
        assert_eq!(am(&g.iter().map(|&x| x as f64).collect::<Vec<_>>()), am(w), "row {r} argmax");
    }
    assert!(worst < 2e-4, "max relative logit difference {worst:e}");
    // the hidden input matters (concat order / hidden norm wired): a different hidden row changes the logits
    let mut h2 = hidden.clone();
    for x in h2[..H].iter_mut() {
        *x = -*x;
    }
    let got2 = m.mtp_reference_pass(&ids, &h2).unwrap();
    assert!(got2[0].iter().zip(&got[0]).any(|(a, b)| (a - b).abs() > 1e-3));
}

#[test]
fn mtp_is_not_loaded_without_the_flag() {
    let w = weights(11);
    let off = load(&w, DType::Bf16, false, None);
    assert!(!off.has_mtp());
    let on = load(&w, DType::Bf16, true, None);
    assert!(on.has_mtp() && on.weights_bytes > off.weights_bytes);
    // the head's weights alone are absent: a model without them loads without the flag and refuses with it
    let mut bare = MemWeights::new();
    for n in w.names() {
        if !n.starts_with("mtp.") {
            let info = w.info(&n).unwrap();
            bare.insert(&n, HostTensor::new(info.dtype, info.shape.clone(), w.bytes(&n).unwrap().to_vec()).unwrap());
        }
    }
    let b = load(&bare, DType::Bf16, false, None);
    let (recipe, dims) = dims();
    assert!(Model::load(
        CpuBackend::arc(),
        &recipe,
        &dims,
        &bare,
        &LoadOptions { max_seq: 64, prefill_chunk: 8, mtp: true, ..Default::default() },
        Arc::new(Telemetry::new())
    )
    .is_err());
    // and the plain decode of a model loaded with the head is the same as without it
    let run = |m: &Model| {
        let mut st = m.new_state().unwrap();
        let mut ids = vec![];
        let mut tok = match m.forward(&mut st, &[3, 9, 27, 81, 50, 7, 7, 1, 60, 2], HeadMode::LastArgmax, None).unwrap()
        {
            StepOut::Token(t) => t,
            o => panic!("{o:?}"),
        };
        for _ in 0..20 {
            ids.push(tok);
            tok = match m.forward(&mut st, &[tok], HeadMode::LastArgmax, None).unwrap() {
                StepOut::Token(t) => t,
                o => panic!("{o:?}"),
            };
        }
        ids
    };
    assert_eq!(run(&on), run(&off));
    assert_eq!(run(&b), run(&off));
}

// ------------------------------------------------------------------------------------------------------------------
// engine: token identity
// ------------------------------------------------------------------------------------------------------------------

fn engine(w: &MemWeights, k: Option<usize>) -> (LocalEngine, Arc<Telemetry>) {
    engine_with(w, k, None)
}

fn engine_with(w: &MemWeights, k: Option<usize>, script: Option<DraftScript>) -> (LocalEngine, Arc<Telemetry>) {
    let tel = Arc::new(Telemetry::new());
    let (recipe, dims) = dims();
    let opts = LoadOptions {
        max_seq: 512,
        prefill_chunk: 8,
        fp8_act: Some(Fp8Act::None),
        mtp: k.is_some(),
        ..Default::default()
    };
    let mut m = Model::load(CpuBackend::arc(), &recipe, &dims, w, &opts, tel.clone()).unwrap();
    let spec = match k {
        Some(k) => {
            m.enable_verify(k + 1).unwrap();
            m.configure_plan(&Default::default(), None).unwrap();
            if script.is_some() {
                m.mtp_script(script).unwrap();
            }
            SpecConfig { mode: SpecMode::Mtp, k }
        }
        None => SpecConfig::default(),
    };
    let e = LocalEngine::new(
        m,
        Arc::new(ByteCodec),
        EngineMeta { id: "tiny-mtp".into(), eos_ids: vec![], spec, ..Default::default() },
        tel.clone(),
    )
    .unwrap();
    (e, tel)
}

#[derive(Debug, PartialEq)]
struct Out {
    ids: Vec<u32>,
    reason: FinishReason,
    completion_tokens: usize,
    cached_tokens: usize,
}

fn run(e: &LocalEngine, req: GenRequest) -> Out {
    let (mut ids, mut done) = (vec![], None);
    e.generate(req, &mut |ev| match ev {
        GenEvent::Token { id, .. } => ids.push(id),
        GenEvent::Done { finish_reason, completion_tokens, cached_tokens, .. } => {
            done = Some((finish_reason, completion_tokens, cached_tokens))
        }
        GenEvent::Error { message } => panic!("engine error: {message}"),
    })
    .unwrap();
    let (reason, completion_tokens, cached_tokens) = done.unwrap();
    Out { ids, reason, completion_tokens, cached_tokens }
}

fn req(prompt: &[u32], n: usize) -> GenRequest {
    GenRequest { prompt: prompt.to_vec(), max_tokens: n, temperature: 0.0, ignore_eos: true, ..Default::default() }
}

const PROMPTS: [&[u32]; 3] = [&[3, 9, 27, 81, 50, 7, 7, 1, 60, 2, 33, 33, 12], &[90, 1, 2, 3, 4, 5], &[44; 21]];

#[test]
fn mtp_head_drafts_keep_greedy_output_identical_over_256_tokens() {
    let w = weights(11);
    let (off, _) = engine(&w, None);
    for k in 1..=3 {
        let (on, tel) = engine(&w, Some(k));
        for p in PROMPTS {
            let (a, b) = (run(&off, req(p, 256)), run(&on, req(p, 256)));
            assert_eq!(a, b, "k {k} prompt {p:?}");
        }
        let t = tel.snapshot();
        assert!(t.spec_steps > 0 && t.spec_proposed > 0, "k {k}: the head drafted");
        eprintln!("k {k}: {} verify passes, {} drafted, {} accepted", t.spec_steps, t.spec_proposed, t.spec_accepted);
    }
}

/// Drafts from the true greedy continuation with a mistake at a pseudo-random offset (or none): every pass accepts
/// all, some or none of its drafts.
fn scripted(truth: Arc<Mutex<Vec<Vec<u32>>>>, seed: u64) -> DraftScript {
    Box::new(move |ctx: &[u32], k: usize| {
        let streams = truth.lock().unwrap();
        let s = streams.iter().find(|s| s.len() > ctx.len() && s[..ctx.len()] == *ctx);
        let Some(s) = s else { return vec![7; k] };
        let mut d: Vec<u32> = s[ctx.len()..].iter().copied().take(k).collect();
        while d.len() < k {
            d.push(1);
        }
        let h = (ctx.len() as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ seed;
        let wrong = (h >> 7) % (k as u64 + 2); // k + 1 or k: all right; else draft `wrong` is wrong
        if (wrong as usize) < k {
            d[wrong as usize] = (d[wrong as usize] + 1 + (h % 5) as u32) % V;
        }
        d
    })
}

#[test]
fn scripted_drafts_hit_every_acceptance_pattern_and_stay_identical() {
    let w = weights(11);
    let (off, _) = engine(&w, None);
    let truth: Arc<Mutex<Vec<Vec<u32>>>> = Arc::new(Mutex::new(Vec::new()));
    for p in PROMPTS {
        let o = run(&off, req(p, 300));
        let mut s = p.to_vec();
        s.extend(&o.ids);
        truth.lock().unwrap().push(s);
    }
    for k in 1..=3 {
        let (on, tel) = engine_with(&w, Some(k), Some(scripted(truth.clone(), k as u64)));
        for p in PROMPTS {
            let (a, b) = (run(&off, req(p, 256)), run(&on, req(p, 256)));
            assert_eq!(a, b, "k {k} prompt {p:?}");
            // max_tokens landing anywhere inside a run, and a follow-up request reusing the prefix
            for n in [1usize, 2, 3, 5, 8, 13] {
                let (a, b) = (run(&off, req(p, n)), run(&on, req(p, n)));
                assert_eq!(a, b, "k {k} max_tokens {n}");
                let mut next = p.to_vec();
                next.extend(&a.ids);
                next.push(11);
                let (a2, b2) = (run(&off, req(&next, 9)), run(&on, req(&next, 9)));
                assert_eq!(a2, b2, "k {k} follow-up after max_tokens {n}");
            }
            // a stop id inside an accepted run
            let full = run(&off, req(p, 40));
            for at in [3usize, 10, 21] {
                let stop = full.ids[at];
                let r = || GenRequest { stop_ids: vec![stop], ..req(p, 40) };
                assert_eq!(run(&off, r()), run(&on, r()), "k {k} stop id {stop}");
            }
        }
        let t = tel.snapshot();
        eprintln!("scripted k {k}: {} passes, {} drafted, {} accepted", t.spec_steps, t.spec_proposed, t.spec_accepted);
        assert!(t.spec_accepted > 0 && t.spec_accepted < t.spec_proposed, "k {k}: partial acceptance exercised");
    }
}

#[test]
fn min_tokens_requests_stay_identical_and_still_speculate_without_stop_ids() {
    let w = weights(11);
    let (off, _) = engine(&w, None);
    let truth: Arc<Mutex<Vec<Vec<u32>>>> = Arc::new(Mutex::new(Vec::new()));
    for p in PROMPTS {
        let o = run(&off, req(p, 80));
        let mut s = p.to_vec();
        s.extend(&o.ids);
        truth.lock().unwrap().push(s);
    }
    for k in 1..=3 {
        // record the context length of every draft (prompt + generated tokens, the next one included)
        let seen: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
        let inner = scripted(truth.clone(), 7 + k as u64);
        let rec = seen.clone();
        let script: DraftScript = Box::new(move |ctx: &[u32], k: usize| {
            rec.lock().unwrap().push(ctx.len());
            inner(ctx, k)
        });
        let (on, tel) = engine_with(&w, Some(k), Some(script));
        for p in PROMPTS {
            // localeval speed's request shape: ignore_eos + min_tokens = max_tokens; nothing to mask, so spec runs
            let before = tel.snapshot().spec_steps;
            let r = || GenRequest { min_tokens: 64, ..req(p, 64) };
            assert_eq!(run(&off, r()), run(&on, r()), "k {k} min_tokens, no stop ids");
            assert!(tel.snapshot().spec_steps > before, "k {k}: speculation ran inside min_tokens without stop ids");
            // stop ids inside the min_tokens window are masked: spec waits for the window to end
            let full = run(&off, req(p, 60));
            let r = || GenRequest { min_tokens: 12, stop_ids: vec![full.ids[5], full.ids[20]], ..req(p, 60) };
            seen.lock().unwrap().clear();
            assert_eq!(run(&off, r()), run(&on, r()), "k {k} min_tokens with stop ids");
            let gens: Vec<usize> = seen.lock().unwrap().iter().map(|&n| n - p.len()).collect();
            assert!(!gens.is_empty(), "k {k}: speculation resumed after the min_tokens window");
            assert!(
                gens.iter().all(|&g| g >= 12),
                "k {k}: drafted inside the min_tokens window with stop ids: {gens:?}"
            );
        }
    }
}

#[test]
fn non_drafting_requests_skip_the_head_and_leave_greedy_ones_identical() {
    let w = weights(11);
    let (off, _) = engine(&w, None);
    let (on, tel) = engine(&w, Some(2));
    let p = PROMPTS[0];
    // a sampled request never drafts: same output as off with the same seed, and no verify pass
    let sampled = || GenRequest { temperature: 0.8, seed: Some(5), ..req(p, 40) };
    let before = tel.snapshot().spec_steps;
    assert_eq!(run(&off, sampled()), run(&on, sampled()), "sampled request");
    assert_eq!(tel.snapshot().spec_steps, before, "a sampled request ran no verify pass");
    // a greedy follow-up that would reuse the sampled request's prefix: the head did not follow that request, so the
    // engine starts it from an empty sequence; output identical to off, and it drafts again
    let s = run(&off, sampled());
    let mut next = p.to_vec();
    next.extend(&s.ids);
    next.push(9);
    let (a, b) = (run(&off, req(&next, 48)), run(&on, req(&next, 48)));
    assert_eq!(a.ids, b.ids, "greedy follow-up after a sampled request");
    assert!(tel.snapshot().spec_steps > before, "the greedy follow-up drafted");
    assert_eq!(b.cached_tokens, 0, "the follow-up did not reuse a prefix the head never saw");
    // and a second greedy follow-up reuses the prefix again (the head followed the first one)
    let mut next2 = next.clone();
    next2.extend(&b.ids);
    next2.push(3);
    let (a2, b2) = (run(&off, req(&next2, 32)), run(&on, req(&next2, 32)));
    assert_eq!(a2, b2, "second greedy follow-up, prefix reused on both");
}
