//! Lane ink-dflash2: a DFlash2 draft (`DFlash2DraftModel`, `crates/intarsia-core/src/model/dflash2.rs`) on an Inkling-2 target, as
//! `incoai/Inkling-2-Mini-DFlash2` drafts for Inkling-2-Mini. Target: `golden/inkling2-tiny` (16 visits, vocab 320 padded / 300
//! real); draft: a random 2-layer DFlash2 checkpoint written here with the real draft's config layout (`dflash_config` taps at visits
//! 2, 6, 10 and the final norm 16, block 8, sliding window 10, a candidate selector with the "after_correction" softcap 30, vocab 320).
//!
//! * Routing by config: a `dflash_config` directory is a DFlash2 draft, the model directory (`mtp_config`) the DSpark draft.
//! * The loader's Inkling checks (visit taps ending at the final norm, the padded vocab) and the softcap modes.
//! * The taps: the DFlash2 draft's per-tap staging rows are bitwise the DSpark draft's tap-major rows of the same target pass (the
//!   reference aliases its DFlash capture to the DSpark one; a wrong tap would only lower acceptance, nothing else catches it).
//! * The draft pass on the Inkling target equals an f64 host reference (block rows `rms(embed[id], embed_norm)`, mask, positions, window, the
//!   unembed's real rows as the head, the selector with its softcap) at the existing DFlash2 test's tolerance.
//! * Greedy output with the draft on equals `--spec off` (recurrent state rolled back through `SpecRows`), random and scripted
//!   drafts, `--spec-k` 1 / 3 / 7, prompts longer than the prefill chunk and the draft's window; with admitted kernels only.
//!   (`--max-batch` above 1 refuses any `--spec dflash` draft at `intarsia_models::load`, unchanged; the tiny fixture has no
//!   tokenizer for `load_engine`.) Backend: cpu, or HIP with `INTARSIA_INKLING2_BACKEND=hip` (the engine tests).

mod common;
use common::ByteCodec;
use intarsia_core::model::{DraftScript, HeadMode, LoadOptions, Model};
use intarsia_core::recipe::ModelDims;
use intarsia_core::rng::SplitMix64;
use intarsia_core::spec::{SpecConfig, SpecMode};
use intarsia_core::weights::WeightSource;
use intarsia_core::{DType, Engine, EngineMeta, FinishReason, GenEvent, GenRequest, HostTensor, LocalEngine, Telemetry};
use intarsia_models::safetensors::SafeTensors;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const H: usize = 64;
const V_PAD: usize = 320;
const V: usize = 300;
const DNH: usize = 4;
const DNKV: usize = 2;
const DHD: usize = 16;
const DI: usize = 48;
const DLAYERS: usize = 2;
const CONV_K: usize = 2;
const CONV_GS: usize = 16;
const G: usize = H / CONV_GS;
const TAPS: [usize; 4] = [2, 6, 10, 16];
const MASK: u32 = 298;
const BLOCK: usize = 8;
const WINDOW: usize = 10;
const DTHETA: f64 = 10_000.0;
const DEPS: f64 = 1e-6;
const SEL_RANK: usize = 8;
const SEL_TOP: usize = 4;
const CAP: f64 = 30.0;
/// The tiny target's `rms_norm_eps` (its `embed_norm`).
const TEPS: f64 = 1e-6;

fn golden() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden")
}

fn target_dir() -> Option<PathBuf> {
    let t = golden().join("inkling2-tiny");
    t.join("config.json").exists().then_some(t)
}

/// Random bf16 tensors, kept as tensors (to write) and as their bf16-rounded f64 values (for the host reference).
struct Gen {
    t: Vec<(String, HostTensor)>,
    raw: BTreeMap<String, Vec<f64>>,
    rng: SplitMix64,
}

impl Gen {
    fn put_with(&mut self, name: &str, shape: &[usize], f: &mut dyn FnMut(usize, &mut SplitMix64) -> f32) {
        let n: usize = shape.iter().product();
        let vals: Vec<f32> = (0..n).map(|i| f(i, &mut self.rng)).collect();
        let t = HostTensor::from_f32_as(DType::Bf16, shape, &vals);
        self.raw.insert(name.to_string(), t.to_f32().iter().map(|&x| x as f64).collect());
        self.t.push((name.to_string(), t));
    }
    fn mat(&mut self, name: &str, shape: &[usize], scale: f32) {
        self.put_with(name, shape, &mut |_, r| scale * r.normal());
    }
    fn norm(&mut self, name: &str, n: usize) {
        self.put_with(name, &[n], &mut |_, r| 1.0 + 0.1 * r.normal());
    }
    fn get(&self, name: &str) -> &[f64] {
        &self.raw[name]
    }
}

fn draft_weights(seed: u64) -> Gen {
    let mut g = Gen { t: Vec::new(), raw: BTreeMap::new(), rng: SplitMix64::new(seed) };
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
    // the selector's codebooks over the padded vocabulary, as the real draft stores them
    g.mat("candidate_selector.hidden_projection.weight", &[SEL_RANK, H], 0.3);
    g.mat("candidate_selector.predecessor_codebook", &[V_PAD, SEL_RANK], 1.0);
    g.mat("candidate_selector.successor_codebook", &[V_PAD, SEL_RANK], 1.0);
    g
}

/// The real draft's `config.json` layout at the tiny geometry; `patch` edits it.
fn draft_config(patch: &dyn Fn(&mut Value)) -> Value {
    let mut c = json!({
        "architectures": ["DFlash2DraftModel"],
        "dflash_config": {
            "block_size": BLOCK, "conv_group_size": CONV_GS, "conv_kernel_size": CONV_K, "final_logit_softcapping": CAP,
            "mask_token_id": MASK, "selector_rank": SEL_RANK, "selector_softcap_mode": "after_correction", "selector_top_k": SEL_TOP,
            "target_layer_ids": TAPS
        },
        "head_dim": DHD, "hidden_act": "silu", "hidden_size": H, "intermediate_size": DI, "is_causal": false,
        "layer_types": ["sliding_attention", "sliding_attention"], "model_type": "qwen3", "num_attention_heads": DNH,
        "num_hidden_layers": DLAYERS, "num_key_value_heads": DNKV, "num_target_layers": 16, "rms_norm_eps": DEPS,
        "rope_parameters": {"rope_theta": DTHETA, "rope_type": "default"}, "sliding_window": WINDOW, "tie_word_embeddings": false,
        "use_sliding_window": true, "vocab_size": V_PAD
    });
    patch(&mut c);
    c
}

/// A draft checkpoint directory (`config.json` + `model.safetensors`) under the test target's tmp dir.
fn write_draft(name: &str, g: &Gen, cfg: &Value) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("inkling2_dflash2").join(name);
    std::fs::create_dir_all(&d).unwrap();
    std::fs::write(d.join("config.json"), serde_json::to_vec_pretty(cfg).unwrap()).unwrap();
    let ts: Vec<(&str, &HostTensor)> = g.t.iter().map(|(n, t)| (n.as_str(), t)).collect();
    intarsia_models::safetensors::write_safetensors(&d.join("model.safetensors"), &ts).unwrap();
    d
}

/// The default draft (seed 5) as a checkpoint directory.
fn draft_dir() -> PathBuf {
    static DIR: Mutex<Option<PathBuf>> = Mutex::new(None);
    let mut d = DIR.lock().unwrap();
    d.get_or_insert_with(|| write_draft("seed5", &draft_weights(5), &draft_config(&|_| {}))).clone()
}

/// cpu, or HIP with `INTARSIA_INKLING2_BACKEND=hip` (one R9700; `HIP_VISIBLE_DEVICES` picks the card). The f32 reference test and
/// the loader tests stay on cpu.
fn backend(f32: bool) -> Arc<dyn intarsia_core::Backend> {
    match std::env::var("INTARSIA_INKLING2_BACKEND").as_deref() {
        Ok("hip") if !f32 => intarsia_backend_hip::create(0).unwrap(),
        Ok("cpu") | Ok("hip") | Err(_) => intarsia_backend_cpu::CpuBackend::arc(),
        Ok(other) => panic!("INTARSIA_INKLING2_BACKEND={other}: cpu or hip"),
    }
}

fn recipe_dims() -> (intarsia_core::recipe::Recipe, ModelDims) {
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/inkling2.toml")).unwrap();
    let cfg: Value = serde_json::from_str(&std::fs::read_to_string(target_dir().unwrap().join("config.json")).unwrap()).unwrap();
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    assert_eq!((dims.hidden, dims.vocab), (H, V));
    (recipe, dims)
}

/// The tiny target with the draft of `ddir` (or none); verify passes for `--spec-k k`. `f32`: f32 activations (the reference test).
fn try_model(ddir: Option<&Path>, k: usize, f32: bool, plan: &intarsia_core::plan::PlanOptions) -> intarsia_core::Result<(Model, Arc<Telemetry>)> {
    let (recipe, dims) = recipe_dims();
    let src = SafeTensors::open_dir(&target_dir().unwrap()).unwrap();
    let dflash = match ddir {
        Some(d) => Some(intarsia_models::load::open_dflash(Some(d), false)?),
        None => None,
    };
    let draft = dflash.is_some();
    let tel = Arc::new(Telemetry::new());
    let mut opts = LoadOptions { max_seq: 256, prefill_chunk: 16, graphs: false, dflash, ..Default::default() };
    if f32 {
        opts.act_dtype = Some(DType::F32);
        opts.kv_dtype = Some(DType::F32);
    }
    let mut m = Model::load(backend(f32), &recipe, &dims, &src, &opts, tel.clone())?;
    if draft {
        m.enable_verify(k + 1)?;
        m.configure_plan(plan, Some(&*intarsia_backend_cpu::CpuBackend::arc()))?;
    }
    Ok((m, tel))
}

fn model(ddir: Option<&Path>, k: usize) -> (Model, Arc<Telemetry>) {
    try_model(ddir, k, false, &Default::default()).unwrap()
}

fn engine(k: Option<usize>, script: Option<DraftScript>) -> (LocalEngine, Arc<Telemetry>) {
    let dd = draft_dir();
    let (m, tel) = model(k.map(|_| dd.as_path()), k.unwrap_or(1));
    if script.is_some() {
        m.dflash_script(script).unwrap();
    }
    let spec = match k {
        Some(k) => SpecConfig { mode: SpecMode::Dflash, k, k_min: None },
        None => SpecConfig::default(),
    };
    let e = LocalEngine::new(m, Arc::new(ByteCodec), EngineMeta { id: "tiny-ink-dflash2".into(), eos_ids: vec![], spec, ..Default::default() }, tel.clone())
        .unwrap();
    (e, tel)
}

#[derive(Debug, PartialEq)]
struct Out {
    ids: Vec<u32>,
    reason: FinishReason,
}

fn run(e: &dyn Engine, prompt: &[u32], n: usize) -> Out {
    let (mut ids, mut reason) = (vec![], None);
    let req = GenRequest { prompt: prompt.to_vec(), max_tokens: n, temperature: 0.0, ignore_eos: true, ..Default::default() };
    e.generate(req, &mut |ev| match ev {
        GenEvent::Token { id, .. } => ids.push(id),
        GenEvent::Done { finish_reason, .. } => reason = Some(finish_reason),
        GenEvent::Error { message } => panic!("engine error: {message}"),
    })
    .unwrap();
    Out { ids, reason: reason.unwrap() }
}

/// Prompts: short, one past the prefill chunk (16) and the draft's window, one past two chunks.
fn prompts() -> Vec<Vec<u32>> {
    vec![
        vec![3, 9, 27, 81, 50, 7, 7, 1, 60, 2, 33, 33, 12],
        vec![90, 1, 2, 3, 4, 5],
        (0..24u32).map(|i| (i * 37 + 11) % 290 + 5).collect(),
        (0..40u32).map(|i| (i * 53 + 7) % 280 + 3).collect(),
    ]
}

// ------------------------------------------------------------------------------------------------------------------
// loader
// ------------------------------------------------------------------------------------------------------------------

#[test]
fn routes_by_config_and_parses_the_softcap() {
    if target_dir().is_none() {
        eprintln!("skipped: no tiny fixtures");
        return;
    }
    let dl = intarsia_models::load::open_dflash(Some(&draft_dir()), false).unwrap();
    let c = &dl.config;
    assert!(c.v2 && c.dspark.is_none() && c.spine.is_none(), "a dflash_config directory is a DFlash2 draft");
    assert_eq!((c.block, c.mask_id, c.vocab, c.layer_ids.clone()), (BLOCK, MASK, V_PAD, TAPS.to_vec()));
    assert_eq!((c.selector_rank, c.selector_top_k, c.selector_post_softcap), (SEL_RANK, SEL_TOP, CAP as f32));
    // the model directory (mtp_config + mtp.safetensors) is the DSpark draft
    let ds = golden().join("inkling2-tiny-dspark");
    if ds.join("mtp.safetensors").exists() {
        assert!(intarsia_models::load::open_dflash(Some(&ds), false).unwrap().config.dspark.is_some());
    }
    let g = draft_weights(5);
    let with = |name: &str, f: &dyn Fn(&mut Value)| intarsia_models::load::open_dflash(Some(&write_draft(name, &g, &draft_config(f))), false);
    // no softcap: none applied
    assert_eq!(with("nocap", &|c| c["dflash_config"].as_object_mut().unwrap().retain(|k, _| !k.contains("softcap"))).unwrap().config.selector_post_softcap, 0.0);
    // the reference's default mode (before_correction) with a selector, explicit or implied: not implemented, refused
    assert!(with("before", &|c| c["dflash_config"]["selector_softcap_mode"] = json!("before_correction")).is_err());
    assert!(with("implied", &|c| { c["dflash_config"].as_object_mut().unwrap().remove("selector_softcap_mode"); }).is_err());
    assert!(with("bogus", &|c| c["dflash_config"]["selector_softcap_mode"] = json!("sideways")).is_err());
    // without a selector the drafts are the rows' argmax, which no softcap moves
    let nosel = with("nosel", &|c| {
        c["dflash_config"]["selector_softcap_mode"] = json!("before_correction");
        c["dflash_config"]["selector_rank"] = json!(0);
    })
    .unwrap();
    assert_eq!(nosel.config.selector_post_softcap, 0.0);
}

#[test]
fn inkling_load_checks_visit_taps_and_padded_vocab() {
    if target_dir().is_none() {
        eprintln!("skipped: no tiny fixtures");
        return;
    }
    let g = draft_weights(5);
    let load = |name: &str, f: &dyn Fn(&mut Value)| try_model(Some(&write_draft(name, &g, &draft_config(f))), 7, false, &Default::default()).map(|_| ());
    assert!(load("ok", &|_| {}).is_ok());
    // visit ids: sorted, unique, the last the final norm (16 = the tiny target's visit count)
    for (name, taps) in [("nofinal", json!([2, 6, 10])), ("short", json!([2, 6, 10, 15])), ("unsorted", json!([6, 2, 10, 16])), ("past", json!([2, 6, 10, 17]))] {
        let r = load(name, &|c| c["dflash_config"]["target_layer_ids"] = taps.clone());
        assert!(r.is_err(), "taps {taps}: loaded");
    }
    // the vocab must be the target's padded one (or its real one)
    assert!(load("vocab", &|c| c["vocab_size"] = json!(310)).is_err());
    // a mask token in the padding is refused
    assert!(load("mask", &|c| c["dflash_config"]["mask_token_id"] = json!(305)).is_err());
}

// ------------------------------------------------------------------------------------------------------------------
// taps
// ------------------------------------------------------------------------------------------------------------------

#[test]
fn taps_equal_the_dspark_drafts_taps() {
    let ds = golden().join("inkling2-tiny-dspark");
    if target_dir().is_none() || !ds.join("mtp.safetensors").exists() {
        eprintln!("skipped: no tiny fixtures");
        return;
    }
    let dsc = intarsia_models::load::open_dflash(Some(&ds), false).unwrap();
    assert_eq!(dsc.config.layer_ids, TAPS.to_vec(), "the tiny DSpark draft taps the same visits");
    for p in prompts().into_iter().filter(|p| p.len() <= 16) {
        let t = p.len();
        let taps = |dir: &Path| {
            let (m, _) = model(Some(dir), 3);
            let mut st = m.new_state().unwrap();
            m.forward(&mut st, &p, HeadMode::None, None).unwrap();
            m.dflash_taps(t).unwrap()
        };
        let (a, b) = (taps(&draft_dir()), taps(&ds));
        assert_eq!(a.len(), TAPS.len());
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert!(x.iter().any(|v| *v != 0.0), "tap {i} is empty");
            assert!(x.iter().map(|v| v.to_bits()).eq(y.iter().map(|v| v.to_bits())), "prompt of {t}: tap {i} (visit {}) differs from DSpark's", TAPS[i]);
        }
        // (visits 2, 6 and 10 are the tiny loop's pass starts, which all enter with the prelude's partial, so those three taps are
        // one value by design; the next test taps visits whose partials differ)
        assert_ne!(a[0], a[3], "the final-norm tap is the partial entering visit 2");
    }
}

/// Taps at visits that are not loop restarts: tap `v` is the partial visit `v - 1` leaves (`vNN.partial_out` of `Model::ink_dump`,
/// the oracle's names), the last tap the final-norm output (`final_hidden`), bitwise, and no two taps alike.
#[test]
fn taps_are_the_visit_partials_and_the_final_norm() {
    if target_dir().is_none() {
        eprintln!("skipped: no tiny fixtures");
        return;
    }
    let taps = [3usize, 7, 13, 16];
    let dir = write_draft("taps3713", &draft_weights(5), &draft_config(&|c| c["dflash_config"]["target_layer_ids"] = json!(taps)));
    for p in prompts().into_iter().filter(|p| p.len() <= 16) {
        let t = p.len();
        let (m, _) = model(Some(&dir), 3);
        let mut st = m.new_state().unwrap();
        m.forward(&mut st, &p, HeadMode::None, None).unwrap();
        let got = m.dflash_taps(t).unwrap();
        let (m2, _) = model(None, 3);
        let mut st2 = m2.new_state().unwrap();
        let dump = m2.ink_dump(&mut st2, &p, None).unwrap();
        for (i, &v) in taps.iter().enumerate() {
            let key = if v == 16 { "final_hidden".to_string() } else { format!("v{:02}.partial_out", v - 1) };
            let want = &dump.get(&key).unwrap_or_else(|| panic!("dump has no {key}")).1;
            assert_eq!(want.len(), t * H, "{key}");
            assert!(got[i].iter().map(|x| x.to_bits()).eq(want.iter().map(|x| x.to_bits())), "prompt of {t}: tap {i} (visit {v}) is not {key}");
        }
        for i in 0..taps.len() {
            for j in 0..i {
                assert_ne!(got[i], got[j], "taps {j} and {i} are the same rows");
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------
// the draft pass against an f64 host reference
// ------------------------------------------------------------------------------------------------------------------

fn mv(w: &[f64], n: usize, k: usize, x: &[f64]) -> Vec<f64> {
    (0..n).map(|r| (0..k).map(|c| w[r * k + c] * x[c]).sum()).collect()
}

fn rms(x: &[f64], w: &[f64]) -> Vec<f64> {
    rms_eps(x, w, DEPS)
}

fn rms_eps(x: &[f64], w: &[f64], eps: f64) -> Vec<f64> {
    let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let r = 1.0 / (ms + eps).sqrt();
    x.iter().zip(w).map(|(v, w)| v * r * w).collect()
}

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

/// The draft (as `tests/dflash2_cpu.rs`'s reference, z-lab's semantics) over a context of `feat.len()` rows and the block `ids`
/// at positions `C ..`, the rows the target's embedding output `rms(embed[id], embed_norm)` (eps `TEPS`): the final normed rows.
fn reference(dw: &Gen, embed: &[f64], embed_norm: &[f64], feat: &[Vec<f64>], ids: &[u32]) -> Vec<Vec<f64>> {
    let c = feat.len();
    let t = ids.len();
    let ctx: Vec<Vec<f64>> = feat.iter().map(|f| rms(&mv(dw.get("fc.weight"), H, TAPS.len() * H, f), dw.get("hidden_norm.weight"))).collect();
    let mut x: Vec<Vec<f64>> = ids.iter().map(|&i| rms_eps(&embed[i as usize * H..(i as usize + 1) * H], embed_norm, TEPS)).collect();
    let (qd, kvd) = (DNH * DHD, DNKV * DHD);
    for l in 0..DLAYERS {
        let p = |s: &str| format!("layers.{l}.{s}");
        let w = |s: &str| dw.get(&p(s));
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

/// A bf16 tensor of the target checkpoint as f64.
fn target_f64(name: &str) -> Vec<f64> {
    let src = SafeTensors::open_dir(&target_dir().unwrap()).unwrap();
    let t = HostTensor::new(DType::Bf16, src.info(name).unwrap().shape.clone(), src.bytes(name).unwrap().to_vec()).unwrap();
    t.to_f32().iter().map(|&x| x as f64).collect()
}

#[test]
fn draft_pass_matches_host_reference() {
    if target_dir().is_none() {
        eprintln!("skipped: no tiny fixtures");
        return;
    }
    // a context shorter than the window (every row sees all of it) and longer (row r keeps the last WINDOW - 1 - r)
    let embed = target_f64("model.llm.embed.weight");
    let embed_norm = target_f64("model.llm.embed_norm.weight");
    let unembed = target_f64("model.llm.unembed.weight");
    assert_eq!((embed.len(), unembed.len()), (V_PAD * H, V_PAD * H));
    let dw = draft_weights(5);
    let mut checked = 0;
    for c in [3usize, 13] {
        let (m, _) = try_model(Some(&draft_dir()), 7, true, &Default::default()).unwrap();
        let mut st = m.new_state().unwrap();
        let mut rng = SplitMix64::new(c as u64);
        let feat_f32: Vec<f32> = (0..c * TAPS.len() * H).map(|_| rng.normal()).collect();
        let feat: Vec<Vec<f64>> = feat_f32.chunks(TAPS.len() * H).map(|r| r.iter().map(|&v| v as f64).collect()).collect();
        m.dflash_ingest_host(&mut st, &feat_f32, c).unwrap();
        let next = 17u32;
        let drafts = m.dflash_draft(&mut st, next, BLOCK - 1).unwrap();
        assert_eq!(drafts.len(), BLOCK - 1, "rows 1.. draft");
        assert!(drafts.iter().all(|&d| (d as usize) < V), "a draft in the unembed's padding: {drafts:?}");
        let got = m.dflash_last_hidden(BLOCK).unwrap();
        let mut ids = vec![MASK; BLOCK];
        ids[0] = next;
        let want = reference(&dw, &embed, &embed_norm, &feat, &ids);
        for r in 0..BLOCK {
            let num: f64 = got[r * H..(r + 1) * H].iter().zip(&want[r]).map(|(a, b)| (*a as f64 - b).powi(2)).sum();
            let e = (num / want[r].iter().map(|b| b * b).sum::<f64>()).sqrt();
            eprintln!("context {c} row {r}: rel l2 {e:.2e}");
            assert!(e < 1e-4, "context {c} row {r}: rel l2 {e}");
        }
        // the selector over the real rows of the unembed: walk from the anchor, top SEL_TOP logits of each row, the score
        // logit + pred[prev] . proj(h) . succ[cand], softcapped after the correction; checked while the walk's margins are clear
        let (proj, pred, succ) =
            (dw.get("candidate_selector.hidden_projection.weight"), dw.get("candidate_selector.predecessor_codebook"), dw.get("candidate_selector.successor_codebook"));
        let mut prev = next as usize;
        for r in 1..BLOCK {
            let logits: Vec<f64> = (0..V).map(|v| (0..H).map(|j| unembed[v * H + j] * want[r][j]).sum()).collect();
            let mut order: Vec<usize> = (0..V).collect();
            order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
            if logits[order[SEL_TOP - 1]] - logits[order[SEL_TOP]] < 1e-3 {
                break; // the candidate set itself is a near tie
            }
            let hp = mv(proj, SEL_RANK, H, &want[r]);
            let mut sc: Vec<(f64, usize)> = order[..SEL_TOP]
                .iter()
                .map(|&cand| {
                    let e: f64 = (0..SEL_RANK).map(|j| pred[prev * SEL_RANK + j] * hp[j] * succ[cand * SEL_RANK + j]).sum();
                    (CAP * ((logits[cand] + e) / CAP).tanh(), cand)
                })
                .collect();
            sc.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            if sc[0].0 - sc[1].0 < 1e-3 {
                break;
            }
            assert_eq!(drafts[r - 1] as usize, sc[0].1, "context {c}: draft of row {r}");
            checked += 1;
            prev = sc[0].1;
        }
    }
    assert!(checked >= 4, "only {checked} drafts were checked against the host selector");
}

// ------------------------------------------------------------------------------------------------------------------
// engine: token identity
// ------------------------------------------------------------------------------------------------------------------

#[test]
fn greedy_ids_equal_spec_off() {
    if target_dir().is_none() {
        eprintln!("skipped: no tiny fixtures");
        return;
    }
    let (off, _) = engine(None, None);
    for k in [1usize, 3, 7] {
        let (on, tel) = engine(Some(k), None);
        for p in prompts() {
            for n in [1usize, 2, 5, 40] {
                assert_eq!(run(&off, &p, n), run(&on, &p, n), "k {k} max_tokens {n} prompt {p:?}");
            }
        }
        let t = tel.snapshot();
        assert!(t.spec_proposed > 0, "k {k}: the draft never proposed anything (vacuous test)");
        eprintln!("k {k}: proposed {} accepted {}", t.spec_proposed, t.spec_accepted);
    }
}

#[test]
fn partial_acceptance_keeps_ids_identical() {
    if target_dir().is_none() {
        eprintln!("skipped: no tiny fixtures");
        return;
    }
    let (off, _) = engine(None, None);
    for k in [2usize, 3, 7] {
        let mut acc = 0;
        for p in prompts() {
            let reference = run(&off, &p, 48);
            let mut full = p.clone();
            full.extend(&reference.ids);
            let step = Mutex::new(0usize);
            let script: DraftScript = Box::new(move |seq: &[u32], kk: usize| {
                let mut s = step.lock().unwrap();
                *s += 1;
                let right = *s % (kk + 1);
                let at = seq.len();
                let mut d: Vec<u32> = (0..kk).filter_map(|j| full.get(at + j).copied()).collect();
                if right < d.len() {
                    d[right] = (d[right] + 1) % 300;
                }
                d
            });
            let (on, tel) = engine(Some(k), Some(script));
            assert_eq!(reference, run(&on, &p, 48), "k {k} prompt {p:?}");
            acc += tel.snapshot().spec_accepted;
        }
        assert!(acc > 0, "k {k}: nothing accepted (vacuous test)");
    }
}

#[test]
fn runs_with_admitted_kernels_only() {
    if target_dir().is_none() {
        eprintln!("skipped: no tiny fixtures");
        return;
    }
    let plan = intarsia_core::plan::PlanOptions { admit: true, require_admitted: true, no_cache: true, ..Default::default() };
    let (off, _) = engine(None, None);
    let (m, tel) = try_model(Some(&draft_dir()), 7, false, &plan).unwrap();
    assert!(m.registry.require_admitted);
    let spec = SpecConfig { mode: SpecMode::Dflash, k: 7, k_min: None };
    let on = LocalEngine::new(m, Arc::new(ByteCodec), EngineMeta { id: "tiny-ink-dflash2".into(), eos_ids: vec![], spec, ..Default::default() }, tel.clone())
        .unwrap();
    for p in prompts() {
        assert_eq!(run(&off, &p, 40), run(&on, &p, 40), "prompt {p:?}");
    }
    assert!(tel.snapshot().spec_proposed > 0, "the draft never proposed anything (vacuous test)");
}
