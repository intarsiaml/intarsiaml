//! `--spec dflash` with a DFlash2 draft (`crates/intarsia-core/src/model/dflash2.rs`, lane dfh) on the cpu backend.
//!
//! Target: the tiny hybrid `Qwen3_5ForCausalLM` of `tests/fixtures/tiny_qwen3_5` (6 layers: GatedDeltaNet x3, gated attention,
//! GatedDeltaNet, gated attention; f32), whose HF per-layer hidden states are in `golden.safetensors`. Draft: a random 2-layer
//! DFlash2 checkpoint with its own geometry (4 / 2 heads of 16, the target has 4 / 2 of 32), tapping target layers 1
//! (GatedDeltaNet) and 3 (gated attention), block 4, sliding window 6.
//!
//! * The taps are the target's residual stream after each tapped layer: equal to HF's `hidden_states[l + 1]`, and far from
//!   the neighbouring layers' (a tap one layer off fails here; nothing else would catch it, drafts never change the output).
//! * The draft's ingest and block pass equal an f64 host implementation of the architecture in the module doc (z-lab's
//!   `model_mlx.py` semantics): standard RMSNorm, grouped dynamic conv in two branches around attention and MLP, the sliding
//!   window per block row (context shorter and longer than the window), non-causal block attention, rows 1.. drafting.
//! * Greedy output with DFlash2 on equals `--spec off` on the hybrid target (recurrent state rolled back through
//!   `SpecRows`): random drafts, and scripted drafts that hit every acceptance pattern, odd `max_tokens` and follow-ups.

mod common;
use common::ByteCodec;
use intarsia_backend_cpu::CpuBackend;
use intarsia_core::model::{DflashConfig, DflashLoad, DraftScript, HeadMode, LoadOptions, Model};
use intarsia_core::recipe::ModelDims;
use intarsia_core::rng::SplitMix64;
use intarsia_core::spec::{SpecConfig, SpecMode};
use intarsia_core::weights::MemWeights;
use intarsia_core::{DType, Engine, EngineMeta, FinishReason, GenEvent, GenRequest, HostTensor, LocalEngine, Telemetry};
use intarsia_models::SafeTensors;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const H: usize = 64;
const V: usize = 96;
/// Draft geometry (its own, not the target's).
const DNH: usize = 4;
const DNKV: usize = 2;
const DHD: usize = 16;
const DI: usize = 48;
const DLAYERS: usize = 2;
const CONV_K: usize = 2;
const CONV_GS: usize = 16;
const G: usize = H / CONV_GS;
const TAPS: [usize; 2] = [1, 3];
const MASK: u32 = 95;
const BLOCK: usize = 4;
const WINDOW: usize = 6;
const DTHETA: f64 = 1_000_000.0;
const DEPS: f64 = 1e-6;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_qwen3_5")
}

/// name -> values of a safetensors file with f32 / i32 tensors.
fn read_f32(path: &Path) -> BTreeMap<String, (Vec<usize>, Vec<f32>)> {
    let data = std::fs::read(path).unwrap();
    let hlen = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&data[8..8 + hlen]).unwrap();
    let mut out = BTreeMap::new();
    for (name, v) in header.as_object().unwrap() {
        if name == "__metadata__" {
            continue;
        }
        let shape: Vec<usize> = v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        let off = v["data_offsets"].as_array().unwrap();
        let (b, e) = (8 + hlen + off[0].as_u64().unwrap() as usize, 8 + hlen + off[1].as_u64().unwrap() as usize);
        let vals: Vec<f32> = match v["dtype"].as_str().unwrap() {
            "F32" => data[b..e].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            "I32" => data[b..e].chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
            other => panic!("dtype {other}"),
        };
        out.insert(name.clone(), (shape, vals));
    }
    out
}

fn target_spec() -> (intarsia_core::recipe::Recipe, ModelDims) {
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/qwen3_5.toml")).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fixture_dir().join("config.json")).unwrap()).unwrap();
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    assert_eq!((dims.hidden, dims.vocab, dims.n_layers), (H, V, 6));
    (recipe, dims)
}

/// Random bf16 tensors in a `MemWeights`, with the (bf16-rounded) values kept as f64 for the host reference.
struct Gen {
    w: MemWeights,
    raw: BTreeMap<String, Vec<f64>>,
    rng: SplitMix64,
}

impl Gen {
    fn put_with(&mut self, name: &str, shape: &[usize], f: &mut dyn FnMut(usize, &mut SplitMix64) -> f32) {
        let n: usize = shape.iter().product();
        let vals: Vec<f32> = (0..n).map(|i| f(i, &mut self.rng)).collect();
        let t = HostTensor::from_f32_as(DType::Bf16, shape, &vals);
        self.raw.insert(name.to_string(), t.to_f32().iter().map(|&x| x as f64).collect());
        self.w.insert(name, t);
    }
    fn mat(&mut self, name: &str, shape: &[usize], scale: f32) {
        self.put_with(name, shape, &mut |_, r| scale * r.normal());
    }
    /// A standard-RMSNorm weight (scale `w`, not `1 + w`): around 1.
    fn norm(&mut self, name: &str, n: usize) {
        self.put_with(name, &[n], &mut |_, r| 1.0 + 0.1 * r.normal());
    }
    fn get(&self, name: &str) -> &[f64] {
        &self.raw[name]
    }
}

fn draft_weights(seed: u64) -> Gen {
    let mut g = Gen { w: MemWeights::new(), raw: BTreeMap::new(), rng: SplitMix64::new(seed) };
    let ls = 0.15;
    for l in 0..DLAYERS {
        let p = format!("layers.{l}");
        g.norm(&format!("{p}.input_layernorm.weight"), H);
        g.norm(&format!("{p}.post_attention_layernorm.weight"), H);
        g.mat(&format!("{p}.self_attn.q_proj.weight"), &[DNH * DHD, H], ls);
        g.mat(&format!("{p}.self_attn.k_proj.weight"), &[DNKV * DHD, H], ls);
        g.mat(&format!("{p}.self_attn.v_proj.weight"), &[DNKV * DHD, H], ls);
        g.mat(&format!("{p}.self_attn.o_proj.weight"), &[H, DNH * DHD], ls);
        g.norm(&format!("{p}.self_attn.q_norm.weight"), DHD);
        g.norm(&format!("{p}.self_attn.k_norm.weight"), DHD);
        g.mat(&format!("{p}.mlp.gate_proj.weight"), &[DI, H], ls);
        g.mat(&format!("{p}.mlp.up_proj.weight"), &[DI, H], ls);
        g.mat(&format!("{p}.mlp.down_proj.weight"), &[H, DI], ls);
        for c in ["attention_conv", "mlp_conv"] {
            // base [2 branches, K taps, H]: tap 0 near 1, tap 1 small (a near-identity conv with some mixing of the previous row)
            g.put_with(&format!("{p}.{c}.base_kernel"), &[2, CONV_K, H], &mut |i, r| {
                let tap = (i / H) % CONV_K;
                if tap == 0 { 1.0 + 0.1 * r.normal() } else { 0.3 * r.normal() }
            });
            g.mat(&format!("{p}.{c}.kernel_projection.weight"), &[2 * CONV_K * G, H], 0.1);
        }
    }
    g.mat("fc.weight", &[H, TAPS.len() * H], 0.1);
    g.norm("hidden_norm.weight", H);
    g.norm("norm.weight", H);
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
            intermediate: DI,
            n_heads: DNH,
            n_kv_heads: DNKV,
            head_dim: DHD,
            vocab: V,
            v2: true,
            rms_eps: DEPS as f32,
            rope_theta: DTHETA as f32,
            sliding_window: WINDOW,
            conv_kernel: CONV_K,
            conv_group: CONV_GS,
            ..Default::default()
        },
    }
}

/// The hybrid target (f32) with the DFlash2 draft (seed `dseed`) or without it; verify passes for `--spec-k k`.
fn model(dseed: Option<u64>, k: Option<usize>, chunk: usize) -> (Model, Arc<Telemetry>) {
    let (recipe, dims) = target_spec();
    let src = SafeTensors::open_dir(&fixture_dir()).unwrap();
    let tel = Arc::new(Telemetry::new());
    let opts = LoadOptions {
        max_seq: 512,
        prefill_chunk: chunk,
        act_dtype: Some(DType::F32),
        kv_dtype: Some(DType::F32),
        profile_every: 0,
        dflash: dseed.map(|s| dflash_load(draft_weights(s))),
        ..Default::default()
    };
    let mut m = Model::load(CpuBackend::arc(), &recipe, &dims, &src, &opts, tel.clone()).unwrap();
    if let Some(k) = k {
        m.enable_verify(k + 1).unwrap();
        m.configure_plan(&Default::default(), None).unwrap();
    }
    (m, tel)
}

fn rel_l2(got: &[f32], want: &[f64]) -> f64 {
    let num: f64 = got.iter().zip(want).map(|(a, b)| (*a as f64 - b).powi(2)).sum();
    let den: f64 = want.iter().map(|b| b.powi(2)).sum();
    (num / den).sqrt()
}

// ------------------------------------------------------------------------------------------------------------------
// taps against HF's per-layer hidden states
// ------------------------------------------------------------------------------------------------------------------

#[test]
fn dflash2_taps_are_the_target_layer_outputs() {
    let gold = read_f32(&fixture_dir().join("golden.safetensors"));
    let ids: Vec<u32> = gold["ids"].1.iter().map(|&x| x as u32).collect();
    let t = ids.len();
    let (m, _) = model(Some(5), Some(3), 64);
    let mut st = m.new_state().unwrap();
    m.forward(&mut st, &ids, HeadMode::None, None).unwrap();
    let taps = m.dflash_taps(t).unwrap();
    for (i, &l) in TAPS.iter().enumerate() {
        let hf = |j: usize| -> Vec<f64> { gold[&format!("hidden_{j}")].1.iter().map(|&x| x as f64).collect() };
        // HF hidden_states[0] is the embedding, [j] the output of layer j - 1
        let own = rel_l2(&taps[i], &hf(l + 1));
        let before = rel_l2(&taps[i], &hf(l));
        let after = rel_l2(&taps[i], &hf(l + 2));
        eprintln!("tap layer {l}: rel l2 vs HF out(l) {own:.2e}, out(l-1) {before:.2e}, out(l+1) {after:.2e}");
        assert!(own < 1e-4, "tap of layer {l} is not HF's output of layer {l}: rel l2 {own}");
        assert!(before > 1e-2 && after > 1e-2, "the neighbouring layers must be far for the check to mean anything");
    }
}

// ------------------------------------------------------------------------------------------------------------------
// draft forward against an f64 host reference
// ------------------------------------------------------------------------------------------------------------------

fn mv(w: &[f64], n: usize, k: usize, x: &[f64]) -> Vec<f64> {
    (0..n).map(|r| (0..k).map(|c| w[r * k + c] * x[c]).sum()).collect()
}

/// Standard RMSNorm (`x * w`).
fn rms(x: &[f64], w: &[f64]) -> Vec<f64> {
    let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let r = 1.0 / (ms + DEPS).sqrt();
    x.iter().zip(w).map(|(v, w)| v * r * w).collect()
}

/// Per head: RMSNorm over the head then neox rope (full head) at `pos`.
fn head_norm_rope(x: &[f64], heads: usize, w: &[f64], pos: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len());
    for hh in 0..heads {
        let mut v = rms(&x[hh * DHD..(hh + 1) * DHD], w);
        let half = DHD / 2;
        for j in 0..half {
            let inv = DTHETA.powf(-2.0 * j as f64 / DHD as f64);
            let (s, c) = (pos as f64 * inv).sin_cos();
            let (a, b) = (v[j], v[j + half]);
            v[j] = a * c - b * s;
            v[j + half] = b * c + a * s;
        }
        out.extend(v);
    }
    out
}

/// `conv(x, dyn, base)` over the block rows: `dyn` is row r's `[K * G]` slice of one branch, `base` that branch's `[K, H]`.
fn conv(x: &[Vec<f64>], dynr: &[Vec<f64>], base: &[f64]) -> Vec<Vec<f64>> {
    (0..x.len())
        .map(|r| {
            (0..H)
                .map(|c| {
                    let mut acc = 0.0;
                    for o in 0..CONV_K.min(r + 1) {
                        acc += (base[o * H + c] + dynr[r][o * G + c / CONV_GS]) * x[r - o][c];
                    }
                    acc
                })
                .collect()
        })
        .collect()
}

/// The draft over a context of `feat.len()` rows (`[taps * H]` each) and the block `ids` at positions `C ..`: the final normed
/// rows (`[t][H]`).
fn reference(dw: &Gen, embed: &[f32], feat: &[Vec<f64>], ids: &[u32]) -> Vec<Vec<f64>> {
    let c = feat.len();
    let t = ids.len();
    let ctx: Vec<Vec<f64>> = feat.iter().map(|f| rms(&mv(dw.get("fc.weight"), H, TAPS.len() * H, f), dw.get("hidden_norm.weight"))).collect();
    let mut x: Vec<Vec<f64>> = ids.iter().map(|&i| embed[i as usize * H..(i as usize + 1) * H].iter().map(|&v| v as f64).collect()).collect();
    let (qd, kvd) = (DNH * DHD, DNKV * DHD);
    for l in 0..DLAYERS {
        let p = |s: &str| format!("layers.{l}.{s}");
        let w = |s: &str| dw.get(&p(s));
        // context keys / values of this layer
        let ck: Vec<Vec<f64>> =
            (0..c).map(|pos| head_norm_rope(&mv(w("self_attn.k_proj.weight"), kvd, H, &ctx[pos]), DNKV, w("self_attn.k_norm.weight"), pos)).collect();
        let cv: Vec<Vec<f64>> = (0..c).map(|pos| mv(w("self_attn.v_proj.weight"), kvd, H, &ctx[pos])).collect();
        let sub = |x: &[Vec<f64>], norm: &str, cname: &str, body: &dyn Fn(&[Vec<f64>]) -> Vec<Vec<f64>>| -> Vec<Vec<f64>> {
            let n: Vec<Vec<f64>> = x.iter().map(|r| rms(r, w(norm))).collect();
            let dynall: Vec<Vec<f64>> = n.iter().map(|r| mv(w(&format!("{cname}.kernel_projection.weight")), 2 * CONV_K * G, H, r)).collect();
            let d0: Vec<Vec<f64>> = dynall.iter().map(|d| d[..CONV_K * G].to_vec()).collect();
            let d1: Vec<Vec<f64>> = dynall.iter().map(|d| d[CONV_K * G..].to_vec()).collect();
            let base = w(&format!("{cname}.base_kernel"));
            let cin = conv(&n, &d0, &base[..CONV_K * H]);
            let out = body(&cin);
            let f = conv(&out, &d1, &base[CONV_K * H..]);
            x.iter().zip(&f).map(|(a, b)| a.iter().zip(b).map(|(u, v)| u + v).collect()).collect()
        };
        let attn = |cin: &[Vec<f64>]| -> Vec<Vec<f64>> {
            let q: Vec<Vec<f64>> =
                (0..t).map(|r| head_norm_rope(&mv(w("self_attn.q_proj.weight"), qd, H, &cin[r]), DNH, w("self_attn.q_norm.weight"), c + r)).collect();
            let bk: Vec<Vec<f64>> =
                (0..t).map(|r| head_norm_rope(&mv(w("self_attn.k_proj.weight"), kvd, H, &cin[r]), DNKV, w("self_attn.k_norm.weight"), c + r)).collect();
            let bv: Vec<Vec<f64>> = (0..t).map(|r| mv(w("self_attn.v_proj.weight"), kvd, H, &cin[r])).collect();
            (0..t)
                .map(|r| {
                    // row r: the last `WINDOW - 1 - r` context keys, then every block key
                    let keep = (WINDOW - 1 - r).min(c);
                    let keys: Vec<(&Vec<f64>, &Vec<f64>)> = (c - keep..c).map(|j| (&ck[j], &cv[j])).chain((0..t).map(|j| (&bk[j], &bv[j]))).collect();
                    let mut o = vec![0.0; qd];
                    for hh in 0..DNH {
                        let kh = hh / (DNH / DNKV);
                        let qh = &q[r][hh * DHD..(hh + 1) * DHD];
                        let sc: Vec<f64> = keys
                            .iter()
                            .map(|(k, _)| qh.iter().zip(&k[kh * DHD..(kh + 1) * DHD]).map(|(a, b)| a * b).sum::<f64>() / (DHD as f64).sqrt())
                            .collect();
                        let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let e: Vec<f64> = sc.iter().map(|s| (s - mx).exp()).collect();
                        let z: f64 = e.iter().sum();
                        for (wgt, (_, v)) in e.iter().zip(&keys) {
                            for d in 0..DHD {
                                o[hh * DHD + d] += wgt / z * v[kh * DHD + d];
                            }
                        }
                    }
                    mv(w("self_attn.o_proj.weight"), H, qd, &o)
                })
                .collect()
        };
        let mlp = |cin: &[Vec<f64>]| -> Vec<Vec<f64>> {
            cin.iter()
                .map(|r| {
                    let g = mv(w("mlp.gate_proj.weight"), DI, H, r);
                    let u = mv(w("mlp.up_proj.weight"), DI, H, r);
                    let a: Vec<f64> = g.iter().zip(&u).map(|(g, u)| g / (1.0 + (-g).exp()) * u).collect();
                    mv(w("mlp.down_proj.weight"), H, DI, &a)
                })
                .collect()
        };
        x = sub(&x, "input_layernorm.weight", "attention_conv", &attn);
        x = sub(&x, "post_attention_layernorm.weight", "mlp_conv", &mlp);
    }
    x.iter().map(|r| rms(r, dw.get("norm.weight"))).collect()
}

#[test]
fn dflash2_draft_matches_host_reference() {
    let tw = read_f32(&fixture_dir().join("model.safetensors"));
    let embed = &tw["model.embed_tokens.weight"].1;
    let dw = draft_weights(5);
    // context shorter than the window (every row sees all of it) and longer (row r keeps the last WINDOW - 1 - r)
    for c in [3usize, 13] {
        let (m, _) = model(Some(5), Some(3), 64);
        let mut st = m.new_state().unwrap();
        let mut rng = SplitMix64::new(c as u64);
        let feat_f32: Vec<f32> = (0..c * TAPS.len() * H).map(|_| rng.normal()).collect();
        let feat: Vec<Vec<f64>> = feat_f32.chunks(TAPS.len() * H).map(|r| r.iter().map(|&v| v as f64).collect()).collect();
        st.dflash.as_mut().unwrap().active = true;
        m.dflash_ingest_host(&mut st, &feat_f32, c).unwrap();
        let next = 17u32;
        let drafts = m.dflash_draft(&mut st, next, BLOCK - 1).unwrap();
        assert_eq!(drafts.len(), BLOCK - 1, "rows 1.. draft");
        let got = m.dflash_last_hidden(BLOCK).unwrap();
        let mut ids = vec![MASK; BLOCK];
        ids[0] = next;
        let want = reference(&dw, embed, &feat, &ids);
        for r in 0..BLOCK {
            let e = rel_l2(&got[r * H..(r + 1) * H], &want[r]);
            eprintln!("context {c} row {r}: rel l2 {e:.2e}");
            assert!(e < 1e-4, "context {c} row {r}: rel l2 {e}");
        }
        // drafts are the target lm_head's argmax of rows 1..
        let lm = &tw["lm_head.weight"].1;
        for r in 1..BLOCK {
            let logits: Vec<f64> = (0..V).map(|v| (0..H).map(|j| lm[v * H + j] as f64 * want[r][j]).sum()).collect();
            let mut s = logits.clone();
            s.sort_by(|a, b| b.partial_cmp(a).unwrap());
            if s[0] - s[1] > 1e-3 {
                let am = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
                assert_eq!(drafts[r - 1], am, "context {c}: draft of row {r}");
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------
// engine: token identity on the hybrid target
// ------------------------------------------------------------------------------------------------------------------

fn engine(k: Option<usize>, script: Option<DraftScript>) -> (LocalEngine, Arc<Telemetry>) {
    let (m, tel) = model(k.map(|_| 5), k, 8);
    if script.is_some() {
        m.dflash_script(script).unwrap();
    }
    let spec = match k {
        Some(k) => SpecConfig { mode: SpecMode::Dflash, k },
        None => SpecConfig::default(),
    };
    let e = LocalEngine::new(m, Arc::new(ByteCodec), EngineMeta { id: "tiny-dflash2".into(), eos_ids: vec![], spec, ..Default::default() }, tel.clone())
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
        GenEvent::Done { finish_reason, completion_tokens, cached_tokens, .. } => done = Some((finish_reason, completion_tokens, cached_tokens)),
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
fn dflash2_greedy_ids_equal_spec_off_on_the_hybrid() {
    let (off, _) = engine(None, None);
    for k in [1usize, 3] {
        let (on, tel) = engine(Some(k), None);
        for p in PROMPTS {
            assert_eq!(run(&off, req(p, 96)), run(&on, req(p, 96)), "k {k} prompt {p:?}");
        }
        let t = tel.snapshot();
        assert!(t.spec_steps > 0 && t.spec_proposed > 0, "k {k}: the draft drafted");
        eprintln!("k {k}: {} verify passes, {} drafted, {} accepted", t.spec_steps, t.spec_proposed, t.spec_accepted);
    }
}

/// Drafts from the true greedy continuation with a mistake at a pseudo-random offset (or none): every pass accepts all, some
/// or none of its drafts, so the recurrent rollback runs to every length.
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
        let wrong = (h >> 7) % (k as u64 + 2);
        if (wrong as usize) < k {
            d[wrong as usize] = (d[wrong as usize] + 1 + (h % 5) as u32) % V as u32;
        }
        d
    })
}

#[test]
fn dflash2_scripted_drafts_hit_every_acceptance_pattern_and_stay_identical() {
    let (off, _) = engine(None, None);
    let truth: Arc<Mutex<Vec<Vec<u32>>>> = Arc::new(Mutex::new(Vec::new()));
    for p in PROMPTS {
        let o = run(&off, req(p, 140));
        let mut s = p.to_vec();
        s.extend(&o.ids);
        truth.lock().unwrap().push(s);
    }
    for k in [1usize, 3] {
        let (on, tel) = engine(Some(k), Some(scripted(truth.clone(), k as u64)));
        for p in PROMPTS {
            assert_eq!(run(&off, req(p, 120)), run(&on, req(p, 120)), "k {k} prompt {p:?}");
            for n in [1usize, 2, 3, 5, 8] {
                let (a, b) = (run(&off, req(p, n)), run(&on, req(p, n)));
                assert_eq!(a, b, "k {k} max_tokens {n}");
                let mut next = p.to_vec();
                next.extend(&a.ids);
                next.push(11);
                assert_eq!(run(&off, req(&next, 9)), run(&on, req(&next, 9)), "k {k} follow-up after max_tokens {n}");
            }
        }
        let t = tel.snapshot();
        eprintln!("scripted k {k}: {} passes, {} drafted, {} accepted", t.spec_steps, t.spec_proposed, t.spec_accepted);
        assert!(t.spec_accepted > 0 && t.spec_accepted < t.spec_proposed, "k {k}: partial acceptance exercised");
    }
}
