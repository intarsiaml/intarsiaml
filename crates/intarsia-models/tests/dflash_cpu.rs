//! `--spec dflash` (the DFlash parallel block drafter, `crates/intarsia-core/src/model/dflash.rs`) on the cpu backend, gate G2
//! of lane dc.
//!
//! The target is a tiny dense Qwen3 (6 layers, hidden 64, 4 / 2 heads of 16, qk-norm, SwiGLU, untied `lm_head`) with random
//! bf16 weights; the draft is a random 3-layer checkpoint in the DCOCO layout (`layers.N.*`, `fc`, `hidden_norm`, `norm`)
//! tapping target layers 0, 2 and 4, block 7.
//!
//! * Greedy output with DFlash on is token-identical to `--spec off` (several prompts, `--spec-k` 1 / 3 / 7, odd
//!   `max_tokens`, a follow-up request that reuses the prefix), and drafting really happened (`spec_proposed > 0`). With
//!   random weights almost no draft is accepted, which is fine: the rollback of every rejected row is the point.
//! * Without `LoadOptions::dflash` nothing is loaded and no sequence carries draft state.
//! * The draft's context ingest and block pass match an f64 host implementation of the architecture documented in the
//!   module doc (fc over the taps, hidden_norm, k_norm + neox rope on the cached context keys, non-causal block attention
//!   over context + block, final norm, lm_head argmax).

mod common;
use common::{recipe, ByteCodec};
use intarsia_backend_cpu::CpuBackend;
use intarsia_core::model::{DflashConfig, DflashLoad, LoadOptions, Model};
use intarsia_core::recipe::ModelDims;
use intarsia_core::rng::SplitMix64;
use intarsia_core::spec::{SpecConfig, SpecMode};
use intarsia_core::weights::MemWeights;
use intarsia_core::{DType, Engine, EngineMeta, FinishReason, GenEvent, GenRequest, HostTensor, LocalEngine, Telemetry};
use std::collections::BTreeMap;
use std::sync::Arc;

const H: usize = 64;
const NH: usize = 4;
const NKV: usize = 2;
const HD: usize = 16;
const I: usize = 96;
const V: usize = 256;
const LAYERS: usize = 6;
const THETA: f64 = 10000.0;
/// Draft: layers, tapped target layers, mask token, block.
const DLAYERS: usize = 3;
const TAPS: [usize; 3] = [0, 2, 4];
const MASK: u32 = 250;
const BLOCK: usize = 7;

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
        rope_theta: THETA as f32,
        partial_rotary_factor: 1.0,
        rope_scaling: Default::default(),
        tie_embeddings: false,
        max_positions: 512,
        layer_types: Vec::new(),
        linear: None,
        moe: None,
        gemma: None,
    }
}

/// Random bf16 tensors in a `MemWeights`, with the (bf16-rounded) values kept as f64 for the host reference.
struct Gen {
    w: MemWeights,
    raw: BTreeMap<String, Vec<f64>>,
    rng: SplitMix64,
}

impl Gen {
    fn new(seed: u64) -> Gen {
        Gen { w: MemWeights::new(), raw: BTreeMap::new(), rng: SplitMix64::new(seed) }
    }
    /// `norm`: a norm weight around 1; else a normal matrix of std `scale`.
    fn put(&mut self, name: &str, shape: &[usize], scale: f32, norm: bool) {
        let n: usize = shape.iter().product();
        let vals: Vec<f32> =
            (0..n).map(|_| if norm { 1.0 + 0.1 * self.rng.normal() } else { scale * self.rng.normal() }).collect();
        let t = HostTensor::from_f32_as(DType::Bf16, shape, &vals);
        self.raw.insert(name.to_string(), t.to_f32().iter().map(|&x| x as f64).collect());
        self.w.insert(name, t);
    }
    fn get(&self, name: &str) -> &[f64] {
        &self.raw[name]
    }
}

fn layer_tensors(g: &mut Gen, p: &str, ls: f32) {
    g.put(&format!("{p}.input_layernorm.weight"), &[H], 0.0, true);
    g.put(&format!("{p}.post_attention_layernorm.weight"), &[H], 0.0, true);
    g.put(&format!("{p}.self_attn.q_proj.weight"), &[NH * HD, H], ls, false);
    g.put(&format!("{p}.self_attn.k_proj.weight"), &[NKV * HD, H], ls, false);
    g.put(&format!("{p}.self_attn.v_proj.weight"), &[NKV * HD, H], ls, false);
    g.put(&format!("{p}.self_attn.o_proj.weight"), &[H, NH * HD], ls, false);
    g.put(&format!("{p}.self_attn.q_norm.weight"), &[HD], 0.0, true);
    g.put(&format!("{p}.self_attn.k_norm.weight"), &[HD], 0.0, true);
    g.put(&format!("{p}.mlp.gate_proj.weight"), &[I, H], ls, false);
    g.put(&format!("{p}.mlp.up_proj.weight"), &[I, H], ls, false);
    g.put(&format!("{p}.mlp.down_proj.weight"), &[H, I], ls, false);
}

/// `ls`: std of the layer matrices (0.15 normal; tiny makes every layer nearly the identity on the residual stream).
fn target_weights(seed: u64, ls: f32) -> Gen {
    let mut g = Gen::new(seed);
    g.put("model.embed_tokens.weight", &[V, H], 0.5, false);
    g.put("model.norm.weight", &[H], 0.0, true);
    g.put("lm_head.weight", &[V, H], 0.3, false);
    for l in 0..LAYERS {
        layer_tensors(&mut g, &format!("model.layers.{l}"), ls);
    }
    g
}

/// The draft checkpoint's tensors, named as in the checkpoint (no `model.` prefix); embedding and lm_head are the target's.
/// `tie`: the target, whose final norm the draft's `norm` then copies (with near-identity layers on both sides the draft then
/// computes the target's own token map, so some drafts are accepted).
fn draft_weights(seed: u64, ls: f32, tie: Option<&Gen>) -> Gen {
    let mut g = Gen::new(seed);
    for l in 0..DLAYERS {
        layer_tensors(&mut g, &format!("layers.{l}"), ls);
    }
    g.put("fc.weight", &[H, TAPS.len() * H], 0.1, false);
    g.put("hidden_norm.weight", &[H], 0.0, true);
    g.put("norm.weight", &[H], 0.0, true);
    if let Some(t) = tie {
        let vals: Vec<f32> = t.get("model.norm.weight").iter().map(|&x| x as f32).collect();
        g.w.insert("norm.weight", HostTensor::from_f32_as(DType::Bf16, &[H], &vals));
    }
    g
}

fn dflash_load(draft: Gen) -> DflashLoad {
    DflashLoad {
        src: Arc::new(draft.w),
        dcoco: false,
        config: DflashConfig {
            layer_ids: TAPS.to_vec(),
            mask_id: MASK,
            block: BLOCK,
            hidden: H,
            n_layers: DLAYERS,
            intermediate: I,
            n_heads: NH,
            n_kv_heads: NKV,
            head_dim: HD,
            vocab: V,
            topk: 0,
            emb_dim: 0,
            state_dim: 0,
            mlp_dim: 0,
        },
    }
}

fn opts(dflash: Option<DflashLoad>) -> LoadOptions {
    LoadOptions { max_seq: 512, prefill_chunk: 16, profile_every: 0, dflash, ..Default::default() }
}

/// A model with the draft loaded (or not), prepared for `--spec-k k` verify passes when `k` is given.
fn model(tseed: u64, dseed: u64, with_draft: bool, k: Option<usize>) -> (Model, Arc<Telemetry>) {
    model_ls(tseed, dseed, with_draft, k, 0.15, false)
}

fn model_ls(tseed: u64, dseed: u64, with_draft: bool, k: Option<usize>, ls: f32, tie: bool) -> (Model, Arc<Telemetry>) {
    let tel = Arc::new(Telemetry::new());
    let tw = target_weights(tseed, ls);
    let dl = with_draft.then(|| dflash_load(draft_weights(dseed, ls, tie.then_some(&tw))));
    let mut m = Model::load(CpuBackend::arc(), &recipe(), &dims(), &tw.w, &opts(dl), tel.clone()).unwrap();
    if let Some(k) = k {
        m.enable_verify(k + 1).unwrap();
        m.configure_plan(&Default::default(), None).unwrap();
    }
    (m, tel)
}

/// `k`: DFlash with `--spec-k k`; None: `--spec off` (same weights, no draft loaded).
fn engine(k: Option<usize>) -> (LocalEngine, Arc<Telemetry>) {
    engine_ls(k, 0.15, false)
}

fn engine_ls(k: Option<usize>, ls: f32, tie: bool) -> (LocalEngine, Arc<Telemetry>) {
    let (m, tel) = model_ls(1, 2, k.is_some(), k, ls, tie);
    let spec = match k {
        Some(k) => SpecConfig { mode: SpecMode::Dflash, k },
        None => SpecConfig::default(),
    };
    let e = LocalEngine::new(
        m,
        Arc::new(ByteCodec),
        EngineMeta { id: "tiny-dflash".into(), eos_ids: vec![], spec, ..Default::default() },
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

const PROMPTS: [&[u32]; 4] = [
    &[3, 9, 27, 81, 50, 7, 7, 1, 60, 2, 33, 33, 12],
    &[90, 1, 2, 3, 4, 5],
    &[44; 21],
    &[
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 11, 21, 31, 41, 51, 61, 71,
        81, 91, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191, 201,
    ],
];

/// Greedy output with DFlash on equals `--spec off`, token for token, at several `--spec-k`; the draft really ran.
#[test]
fn dflash_greedy_ids_equal_spec_off() {
    for k in [1usize, 3, 7] {
        // fresh engines per k: both start with an empty prefix cache, so `cached_tokens` is comparable too
        let (off, _) = engine(None);
        let (on, tel) = engine(Some(k));
        for p in PROMPTS {
            let (a, b) = (run(&off, req(p, 200)), run(&on, req(p, 200)));
            eprintln!("{} ids, completion_tokens {}", a.ids.len(), a.completion_tokens);
            assert_eq!(a, b, "k {k} prompt {p:?}");
        }
        let t = tel.snapshot();
        assert!(t.spec_proposed > 0, "k {k}: the draft never proposed anything (vacuous test)");
        eprintln!("k {k}: proposed {} accepted {}", t.spec_proposed, t.spec_accepted);
    }
}

/// Short and odd generation lengths (the rollback lands on every position of a block), and two consecutive requests on
/// the same engine, the second reusing the first's prefix, stay identical to `--spec off`.
#[test]
fn dflash_rollback_odd_lengths_and_prefix_reuse() {
    for k in [1usize, 3, 7] {
        let (off, _) = engine(None);
        let (on, tel) = engine(Some(k));
        for p in PROMPTS {
            for n in [1usize, 3, 8, 13] {
                assert_eq!(run(&off, req(p, n)), run(&on, req(p, n)), "k {k} max_tokens {n} prompt {p:?}");
            }
        }
        // the second prompt extends the first one's prompt + completion: its prefix comes from the cache
        let first = PROMPTS[0];
        let a1 = run(&off, req(first, 20));
        let b1 = run(&on, req(first, 20));
        assert_eq!(a1, b1, "k {k} first request");
        let mut second = first.to_vec();
        second.extend(&a1.ids);
        second.extend([5u32, 6, 7]);
        let (a2, b2) = (run(&off, req(&second, 30)), run(&on, req(&second, 30)));
        assert_eq!(a2, b2, "k {k} prefix-reusing request");
        assert!(a2.cached_tokens > 0, "k {k}: the second request did not reuse the prefix ({})", a2.cached_tokens);
        // and a request that diverges inside the cached prefix (a rollback of the cache and the draft's context)
        let mut third = second[..first.len() + 5].to_vec();
        third.extend([99u32, 98, 97, 96]);
        let (a3, b3) = (run(&off, req(&third, 25)), run(&on, req(&third, 25)));
        assert_eq!(a3, b3, "k {k} request diverging inside the prefix");
        assert!(tel.snapshot().spec_proposed > 0, "k {k}: the draft never proposed anything");
    }
}

/// A draft that is right some of the time: with near-identity layers (and the draft's final norm tied to the target's) both
/// compute the same token map from the residual stream, so the anchor row's draft is accepted and later rows sometimes are.
/// The accepted rows stay in the draft's context cache, the rejected ones roll back, and the output is still `--spec off`'s.
#[test]
fn dflash_partial_acceptance_keeps_ids_identical() {
    for k in [1usize, 3, 7] {
        let (off, _) = engine_ls(None, 0.02, true);
        let (on, tel) = engine_ls(Some(k), 0.02, true);
        for p in PROMPTS {
            assert_eq!(run(&off, req(p, 200)), run(&on, req(p, 200)), "k {k} prompt {p:?}");
            for n in [1usize, 3, 8] {
                assert_eq!(run(&off, req(p, n)), run(&on, req(p, n)), "k {k} max_tokens {n} prompt {p:?}");
            }
        }
        let t = tel.snapshot();
        eprintln!("near-identity k {k}: proposed {} accepted {}", t.spec_proposed, t.spec_accepted);
        assert!(t.spec_accepted > 0, "k {k}: no draft was ever accepted");
        if k > 1 {
            assert!(t.spec_accepted < t.spec_proposed, "k {k}: every draft was accepted (no rollback exercised)");
        }
    }
}

/// Without `LoadOptions::dflash` nothing is loaded and sequences carry no draft state; asking for the mode fails loudly.
#[test]
fn dflash_off_loads_nothing() {
    let (m, _) = model(1, 2, false, Some(3));
    assert!(!m.has_dflash());
    assert_eq!(m.dflash_block(), 0);
    let mut st = m.new_state().unwrap();
    assert!(st.dflash.is_none(), "a sequence carries draft state without the draft");
    assert!(m.dflash_draft(&mut st, 1, 3).is_err());

    let (m, _) = model(1, 2, true, Some(3));
    assert!(m.has_dflash());
    assert_eq!(m.dflash_block(), BLOCK);
    assert!(m.new_state().unwrap().dflash.is_some());
    // the measured-cost gate is on with the draft (built-in values, then the recipe's [dflash] table), off without it
    let c = m.spec_cost_params();
    assert!(c.enabled(), "the DFlash cost gate is off");
    assert_eq!((c.min_speedup, c.window, c.probe_every, c.reprobe_tokens), (1.03, 8, 128, 64), "recipe [dflash] values not applied");
    assert!(!model(1, 2, false, Some(3)).0.spec_cost_params().enabled());

    // --spec dflash on a model without the draft is refused at engine construction
    let (m, tel) = model(1, 2, false, Some(3));
    let r = LocalEngine::new(
        m,
        Arc::new(ByteCodec),
        EngineMeta { id: "x".into(), eos_ids: vec![], spec: SpecConfig { mode: SpecMode::Dflash, k: 3 }, ..Default::default() },
        tel,
    );
    assert!(r.is_err(), "dflash mode without the draft was accepted");
}

/// A draft whose geometry or taps do not fit the target is refused at load.
#[test]
fn dflash_bad_geometry_is_refused() {
    let tw = target_weights(1, 0.15);
    for bad in 0..4 {
        let mut dl = dflash_load(draft_weights(2, 0.15, None));
        match bad {
            0 => dl.config.layer_ids = vec![0, 2, LAYERS], // tap beyond the target
            1 => dl.config.block = Model::VERIFY_MAX_ROWS, // block + anchor over the verify rows
            2 => dl.config.mask_id = V as u32,
            _ => dl.config.intermediate = I + 1,
        }
        let r = Model::load(CpuBackend::arc(), &recipe(), &dims(), &tw.w, &opts(Some(dl)), Arc::new(Telemetry::new()));
        assert!(r.is_err(), "case {bad}: a draft that does not fit was loaded");
    }
}

// ------------------------------------------------------------------------------------------------------------------
// numerics against an f64 host implementation of the documented architecture
// ------------------------------------------------------------------------------------------------------------------

fn matvec(w: &[f64], n: usize, k: usize, x: &[f64]) -> Vec<f64> {
    assert_eq!(w.len(), n * k);
    (0..n).map(|r| (0..k).map(|c| w[r * k + c] * x[c]).sum()).collect()
}

fn rms(x: &[f64], w: &[f64], eps: f64, off: f64) -> Vec<f64> {
    let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let r = 1.0 / (ms + eps).sqrt();
    x.iter().zip(w).map(|(a, b)| a * r * (b + off)).collect()
}

/// neox / rotate-half rope over the full head, in place.
fn rope(x: &mut [f64], pos: usize) {
    let half = HD / 2;
    let orig = x.to_vec();
    for i in 0..half {
        let ang = pos as f64 * THETA.powf(-2.0 * i as f64 / HD as f64);
        let (s, c) = ang.sin_cos();
        x[i] = orig[i] * c - orig[i + half] * s;
        x[i + half] = orig[i + half] * c + orig[i] * s;
    }
}

/// Per-head qk-norm then rope at `pos`, over `heads` heads of `x`.
fn head_norm_rope(x: &[f64], heads: usize, w: &[f64], pos: usize, eps: f64, off: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(heads * HD);
    for h in 0..heads {
        let mut v = rms(&x[h * HD..(h + 1) * HD], w, eps, off);
        rope(&mut v, pos);
        out.extend(v);
    }
    out
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

fn round_bf16(x: &[f32]) -> Vec<f64> {
    HostTensor::from_f32_as(DType::Bf16, &[x.len()], x).to_f32().iter().map(|&v| v as f64).collect()
}

/// The draft over `rows` context positions (features `[rows, taps * H]`) and the block `ids`: the block's final normed hidden
/// states, `[ids.len()][H]`.
fn reference(tw: &Gen, dw: &Gen, feat: &[f32], rows: usize, ids: &[u32], eps: f64, off: f64) -> Vec<Vec<f64>> {
    let nt = TAPS.len();
    let feat = round_bf16(feat);
    let fc = dw.get("fc.weight");
    // context: h_ctx = hidden_norm(fc(feat))
    let hctx: Vec<Vec<f64>> = (0..rows)
        .map(|p| {
            let f = &feat[p * nt * H..(p + 1) * nt * H];
            rms(&matvec(fc, H, nt * H, f), dw.get("hidden_norm.weight"), eps, off)
        })
        .collect();
    let b = ids.len();
    let embed = tw.get("model.embed_tokens.weight");
    let mut x: Vec<Vec<f64>> = ids.iter().map(|&i| embed[i as usize * H..(i as usize + 1) * H].to_vec()).collect();
    let group = NH / NKV;
    for l in 0..DLAYERS {
        let p = format!("layers.{l}");
        let g = |n: &str| dw.get(&format!("{p}.{n}.weight"));
        // keys / values: the cached context (from h_ctx, no input_layernorm), then the block rows (from the normed rows)
        let mut keys: Vec<Vec<f64>> = Vec::new();
        let mut vals: Vec<Vec<f64>> = Vec::new();
        for (pos, hc) in hctx.iter().enumerate() {
            let k = matvec(g("self_attn.k_proj"), NKV * HD, H, hc);
            keys.push(head_norm_rope(&k, NKV, g("self_attn.k_norm"), pos, eps, off));
            vals.push(matvec(g("self_attn.v_proj"), NKV * HD, H, hc));
        }
        let mut qs = Vec::new();
        for (r, xr) in x.iter().enumerate() {
            let n = rms(xr, g("input_layernorm"), eps, off);
            let q = matvec(g("self_attn.q_proj"), NH * HD, H, &n);
            qs.push(head_norm_rope(&q, NH, g("self_attn.q_norm"), rows + r, eps, off));
            let k = matvec(g("self_attn.k_proj"), NKV * HD, H, &n);
            keys.push(head_norm_rope(&k, NKV, g("self_attn.k_norm"), rows + r, eps, off));
            vals.push(matvec(g("self_attn.v_proj"), NKV * HD, H, &n));
        }
        for r in 0..b {
            // non-causal: every row sees every key
            let mut attn = vec![0f64; NH * HD];
            for h in 0..NH {
                let kv = h / group;
                let q = &qs[r][h * HD..(h + 1) * HD];
                let s: Vec<f64> = keys
                    .iter()
                    .map(|k| q.iter().zip(&k[kv * HD..(kv + 1) * HD]).map(|(a, b)| a * b).sum::<f64>() / (HD as f64).sqrt())
                    .collect();
                let m = s.iter().cloned().fold(f64::MIN, f64::max);
                let e: Vec<f64> = s.iter().map(|v| (v - m).exp()).collect();
                let z: f64 = e.iter().sum();
                for (w, v) in e.iter().zip(&vals) {
                    for d in 0..HD {
                        attn[h * HD + d] += w / z * v[kv * HD + d];
                    }
                }
            }
            let o = matvec(g("self_attn.o_proj"), H, NH * HD, &attn);
            for (a, b) in x[r].iter_mut().zip(&o) {
                *a += b;
            }
        }
        for xr in x.iter_mut() {
            let n = rms(xr, g("post_attention_layernorm"), eps, off);
            let gate = matvec(g("mlp.gate_proj"), I, H, &n);
            let up = matvec(g("mlp.up_proj"), I, H, &n);
            let a: Vec<f64> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
            let d = matvec(g("mlp.down_proj"), H, I, &a);
            for (a, b) in xr.iter_mut().zip(&d) {
                *a += b;
            }
        }
    }
    x.iter().map(|xr| rms(xr, dw.get("norm.weight"), eps, off)).collect()
}

/// Greedy token and the gap to the runner-up of `lm_head · h`.
fn argmax_gap(lm: &[f64], h: &[f64]) -> (u32, f64, Vec<f64>) {
    let logits = matvec(lm, V, H, h);
    let (mut best, mut second) = (0usize, f64::MIN);
    for i in 1..V {
        if logits[i] > logits[best] {
            second = logits[best];
            best = i;
        } else if logits[i] > second {
            second = logits[i];
        }
    }
    if second == f64::MIN || V == 1 {
        second = logits[0].min(logits[1]);
    }
    (best as u32, logits[best] - second, logits)
}

/// The draft's ingest (`dflash_ingest_host`: fc over the taps, hidden_norm, k-norm + rope, cache) and block pass
/// (`dflash_draft`) against the host implementation of the module doc's architecture.
#[test]
fn dflash_draft_matches_host_reference() {
    let rows = 21; // more than one prefill chunk (16): the ingest runs in two parts
    let (m, _) = model(1, 2, true, Some(BLOCK));
    let (tw, dw) = (target_weights(1, 0.15), draft_weights(2, 0.15, None));
    let eps = dims().rms_eps as f64;
    let off = recipe().model.norm_offset as f64;
    let mut rng = SplitMix64::new(77);
    let feat: Vec<f32> = (0..rows * TAPS.len() * H).map(|_| rng.normal()).collect();
    let next = 42u32;

    let mut st = m.new_state().unwrap();
    m.dflash_ingest_host(&mut st, &feat, rows).unwrap();
    assert_eq!(st.len, rows);
    let drafts = m.dflash_draft(&mut st, next, BLOCK).unwrap();
    assert_eq!(drafts.len(), BLOCK);
    let dev = m.dflash_last_hidden(BLOCK).unwrap();

    let mut ids = vec![MASK; BLOCK];
    ids[0] = next;
    let want = reference(&tw, &dw, &feat, rows, &ids, eps, off);

    // hidden states
    let (mut num, mut den) = (0f64, 0f64);
    for (r, w) in want.iter().enumerate() {
        for (c, x) in w.iter().enumerate() {
            let d = dev[r * H + c] as f64;
            num += (d - x).powi(2);
            den += x * x;
        }
    }
    let rel = (num / den).sqrt();
    eprintln!("draft block hidden: rel-L2 {rel:.4e}");
    assert!(rel < 0.03, "draft block hidden states differ from the host reference: rel-L2 {rel}");

    // drafts: the device's argmax is the argmax of lm_head over ITS hidden (up to a 1e-2 tie), and equals the reference's
    // wherever the reference's top-2 gap exceeds what the hidden-state error can move
    let lm = tw.get("lm_head.weight");
    let (mut flips, mut skipped) = (0, 0);
    for r in 0..BLOCK {
        let dh: Vec<f64> = dev[r * H..(r + 1) * H].iter().map(|&x| x as f64).collect();
        let (own, own_gap, own_logits) = argmax_gap(lm, &dh);
        if own != drafts[r] {
            assert!(own_gap < 1e-2, "row {r}: device draft {} is not the argmax of its own hidden ({own}, gap {own_gap})", drafts[r]);
        }
        let (refd, gap, ref_logits) = argmax_gap(lm, &want[r]);
        let err = own_logits.iter().zip(&ref_logits).map(|(a, b)| (a - b).abs()).fold(0f64, f64::max);
        if drafts[r] != refd {
            if gap < 2.0 * err + 1e-2 {
                skipped += 1;
            } else {
                flips += 1;
                eprintln!("row {r}: device {} host {refd}, host gap {gap:.4} logit err {err:.4}", drafts[r]);
            }
        }
    }
    eprintln!("drafts {drafts:?}; undecidable ties skipped: {skipped}");
    assert_eq!(flips, 0, "draft tokens differ from the host reference beyond the numeric noise");
}
