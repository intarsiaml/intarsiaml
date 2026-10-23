//! Continuous batching of a hybrid model (GatedDeltaNet recurrent state + gated attention) on the cpu backend, behind the
//! opt-in `Model::allow_recurrent_batch` (`INTARSIA_RECURRENT_BATCH=1` in `load_engine`): the prototype of
//! docs/results/2026-10-23-rb-recurrent-batching.md. Model: `tests/fixtures/tiny_qwen3_5/` (6 layers: linear x3, full,
//! linear, full; f32; the recipe of `hybrid.rs`).
//!
//! The cpu kernels compute every row of a multi-row call with the one-row loop, so these tests check the engine logic and
//! the batch plumbing (state slots, per-row recurrent updates, snapshots, restorable prefix reuse, the solo prefill
//! schedule), not GPU numerics. What they pin down:
//!
//! * `Model::forward_batch` returns, for every row, bit for bit the logits of the solo decode step and leaves the
//!   recurrent state as the solo step does (model level, `common/batch_identity.rs`, >= 128 steps, N = 2..4);
//! * a `BatchedEngine` with N slots serves requests token for token (and text, finish reason and counts) as a
//!   `LocalEngine` one at a time, over >= 128 generated tokens, with more requests than slots (slot reuse, so a
//!   recycled slot starts from zero state), prefix reuse off and on (restore from a snapshot);
//! * with the opt-in off, nothing changed: `enable_batch` still refuses the hybrid model.
mod common;
use common::ByteCodec;
use intarsia_core::dtype::f32_to_fp8_e4m3;
use intarsia_core::model::{LoadOptions, Model};
use intarsia_core::quant::{nvfp4, Fp8Act};
use intarsia_core::recipe::ModelDims;
use intarsia_core::rng::SplitMix64;
use intarsia_core::weights::{MemWeights, WeightSource};
use intarsia_core::HostTensor;
use intarsia_core::{
    BatchedEngine, BatchedOptions, DType, Engine, EngineMeta, FinishReason, GenEvent, GenRequest, LocalEngine,
    Telemetry,
};
use intarsia_models::SafeTensors;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::Duration;

#[path = "common/batch_identity.rs"]
mod batch_identity;

const CHUNK: usize = 16;
const MAX_SEQ: usize = 512;
/// The fixture's vocabulary (config.json).
const VOCAB: u32 = 96;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_qwen3_5")
}

/// Which tiny hybrid: the checked-in dense-FFN fixture, or the same fixture with a mixture-of-experts FFN built the way the
/// 35B's is (fp8 attention / GatedDeltaNet projections, NVFP4 routed and shared experts, a shared-expert gate).
#[derive(Clone, Copy, Debug)]
enum Variant {
    Dense,
    Moe,
}

const VARIANTS: [Variant; 2] = [Variant::Dense, Variant::Moe];

const E: usize = 8;
const TOP_K: usize = 2;
const MOE_I: usize = 32;
const MOE_H: usize = 64;
const LAYERS: usize = 6;

fn put_nvfp4(w: &mut MemWeights, rng: &mut SplitMix64, prefix: &str, n: usize, k: usize) {
    let q = nvfp4::random_rows(rng, n, k);
    w.insert(&format!("{prefix}.weight"), q.parts[0].clone());
    w.insert(&format!("{prefix}.weight_scale"), q.parts[1].clone());
    w.insert(&format!("{prefix}.weight_scale_2"), q.parts[2].clone());
}

fn bf16(rng: &mut SplitMix64, shape: &[usize], scale: f32) -> HostTensor {
    HostTensor::from_f32_as(DType::Bf16, shape, &rng.normal_vec(shape.iter().product(), scale))
}

/// `{prefix}.weight` as e4m3 with a per-tensor `{prefix}.weight_scale` (amax / 448).
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

/// The fixture's attention-free MLP tensors dropped, random fp8 projections and NVFP4 experts in their place.
fn moe_weights() -> MemWeights {
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
        w.insert(&format!("{p}.gate.weight"), bf16(&mut rng, &[E, MOE_H], 0.3));
        w.insert(&format!("{p}.shared_expert_gate.weight"), bf16(&mut rng, &[1, MOE_H], 0.3));
        for x in 0..E {
            put_nvfp4(&mut w, &mut rng, &format!("{p}.experts.{x}.gate_proj"), MOE_I, MOE_H);
            put_nvfp4(&mut w, &mut rng, &format!("{p}.experts.{x}.up_proj"), MOE_I, MOE_H);
            put_nvfp4(&mut w, &mut rng, &format!("{p}.experts.{x}.down_proj"), MOE_H, MOE_I);
        }
        put_nvfp4(&mut w, &mut rng, &format!("{p}.shared_expert.gate_proj"), MOE_I, MOE_H);
        put_nvfp4(&mut w, &mut rng, &format!("{p}.shared_expert.up_proj"), MOE_I, MOE_H);
        put_nvfp4(&mut w, &mut rng, &format!("{p}.shared_expert.down_proj"), MOE_H, MOE_I);
    }
    w
}

/// The tiny hybrid model, loaded and not yet planned.
fn load_model(v: Variant, tel: &Arc<Telemetry>) -> Model {
    let dir = fixture_dir();
    let mut cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let recipe_src = match v {
        Variant::Dense => include_str!("../../../recipes/qwen3_5.toml"),
        Variant::Moe => {
            cfg["architectures"] = serde_json::json!(["Qwen3_5MoeForCausalLM"]);
            cfg["num_experts"] = E.into();
            cfg["num_experts_per_tok"] = TOP_K.into();
            cfg["moe_intermediate_size"] = MOE_I.into();
            cfg["shared_expert_intermediate_size"] = MOE_I.into();
            cfg["norm_topk_prob"] = true.into();
            cfg.as_object_mut().unwrap().remove("intermediate_size");
            include_str!("../../../recipes/qwen3_5_moe.toml")
        }
    };
    let recipe = intarsia_models::parse_recipe(recipe_src).unwrap();
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    assert_eq!(dims.vocab as u32, VOCAB);
    let opts = LoadOptions {
        max_seq: MAX_SEQ,
        // a budget that holds MAX_SEQ tokens for every batched slot (the plan otherwise shrinks the context to share the KV budget)
        mem_budget: Some(4 << 30),
        prefill_chunk: CHUNK,
        act_dtype: Some(DType::F32),
        kv_dtype: Some(DType::F32),
        fp8_act: matches!(v, Variant::Moe).then_some(Fp8Act::None),
        ..Default::default()
    };
    let backend = intarsia_backend_cpu::CpuBackend::arc();
    match v {
        Variant::Dense => {
            Model::load(backend, &recipe, &dims, &SafeTensors::open_dir(&dir).unwrap(), &opts, tel.clone())
        }
        Variant::Moe => Model::load(backend, &recipe, &dims, &moe_weights(), &opts, tel.clone()),
    }
    .unwrap()
}

/// The planned tiny hybrid model; `rows >= 2` also enables batching (with the opt-in when `allow`).
fn model(v: Variant, rows: usize, allow: bool, tel: &Arc<Telemetry>) -> Model {
    let mut m = load_model(v, tel);
    if allow {
        m.allow_recurrent_batch().unwrap();
    }
    if rows > 1 {
        m.enable_batch(rows).unwrap();
    }
    m.configure_plan(&Default::default(), None).unwrap();
    m
}

fn meta() -> EngineMeta {
    // no EOS id: every request runs to max_tokens, so 128+ tokens are compared
    EngineMeta { id: "tiny-hybrid".into(), eos_ids: vec![], ..Default::default() }
}

fn solo(v: Variant) -> LocalEngine {
    let tel = Arc::new(Telemetry::new());
    LocalEngine::new(model(v, 1, false, &tel), Arc::new(ByteCodec), meta(), tel).unwrap()
}

fn batched(v: Variant, rows: usize, reuse: bool) -> (BatchedEngine, Arc<Telemetry>) {
    let tel = Arc::new(Telemetry::new());
    let m = model(v, rows, true, &tel);
    let mut o = BatchedOptions::new(rows, 64);
    o.prefix_reuse = reuse;
    (BatchedEngine::new(m, Arc::new(ByteCodec), meta(), tel.clone(), o).unwrap(), tel)
}

#[derive(Debug, PartialEq)]
struct Out {
    ids: Vec<u32>,
    text: String,
    reason: Option<FinishReason>,
    prompt_tokens: usize,
    completion_tokens: usize,
    cached_tokens: usize,
}

fn collect(rx: &Receiver<GenEvent>) -> Result<Out, String> {
    let (mut ids, mut text) = (vec![], String::new());
    loop {
        match rx.recv_timeout(Duration::from_secs(300)).expect("engine stalled") {
            GenEvent::Token { id, text: t } => {
                ids.push(id);
                text.push_str(&t);
            }
            GenEvent::Done { finish_reason, prompt_tokens, completion_tokens, cached_tokens, .. } => {
                return Ok(Out {
                    ids,
                    text,
                    reason: Some(finish_reason),
                    prompt_tokens,
                    completion_tokens,
                    cached_tokens,
                })
            }
            GenEvent::Error { message } => return Err(message),
        }
    }
}

/// Run `req` alone on the solo engine, cold: a throw-away request first so no state prefix of the previous case is reused.
fn run_solo(e: &LocalEngine, req: GenRequest) -> Out {
    let flush = GenRequest { prompt: vec![95, 94, 93], max_tokens: 1, ignore_eos: true, ..Default::default() };
    collect(&e.submit(flush).unwrap()).unwrap();
    collect(&e.submit(req).unwrap()).unwrap()
}

fn prompt(seed: u64, len: usize) -> Vec<u32> {
    let mut rng = intarsia_core::rng::SplitMix64::new(0xBA7C + seed);
    (0..len).map(|_| 1 + (rng.next_u64() % (VOCAB as u64 - 2)) as u32).collect()
}

/// Greedy requests of several prompt lengths, 130 generated tokens each (`ignore_eos`), some seeded-sampled.
fn cases(n: usize) -> Vec<GenRequest> {
    let lens = [5, 17, 33, 9, 48, 16, 40, 3, 25, 64];
    (0..n)
        .map(|i| {
            let mut r = GenRequest {
                prompt: prompt(i as u64, lens[i % lens.len()]),
                max_tokens: 130 + (i % 3) * 7,
                ignore_eos: true,
                ..Default::default()
            };
            if i % 4 == 3 {
                r.temperature = 0.9;
                r.top_k = 20;
                r.top_p = 0.95;
                r.seed = Some(7 + i as u64);
            }
            r
        })
        .collect()
}

#[test]
fn hybrid_batching_is_refused_without_the_opt_in() {
    let tel = Arc::new(Telemetry::new());
    // the default path: exactly the old refusal
    let mut m = load_model(Variant::Dense, &tel);
    let err = m.enable_batch(2).unwrap_err().to_string();
    assert!(err.contains("dense attention-only"), "{err}");
    // the opt-in is for the cpu backend and must come before enable_batch
    let mut m = load_model(Variant::Dense, &tel);
    m.allow_recurrent_batch().unwrap();
    assert!(m.recurrent_batch());
    m.enable_batch(2).unwrap();
    assert!(m.allow_recurrent_batch().is_err(), "after enable_batch");
    // a model without the opt-in has no batch rows, so the engine cannot be built over it
    let m = model(Variant::Dense, 1, false, &tel);
    assert!(!m.recurrent_batch());
    assert_eq!(m.batch_rows(), 0);
    assert!(BatchedEngine::new(m, Arc::new(ByteCodec), meta(), tel, BatchedOptions::new(2, 4)).is_err());
}

#[test]
fn batched_hybrid_decode_is_bitwise_the_solo_step() {
    for v in VARIANTS {
        let tel = Arc::new(Telemetry::new());
        let m = model(v, 4, true, &tel);
        // N = 2, 3 and 4 sequences of different prompt lengths, 130 consecutive batched steps each (the fixture's context is
        // 256 tokens, so each batch size gets fresh sequences); greedy rows alternate onto the on-device argmax path, odd
        // rows sample
        for lens in [vec![3usize, 9], vec![5, 11, 20], vec![4, 8, 13, 22]] {
            let n = lens.len();
            let cfg = batch_identity::Cfg { lens, batches: vec![n], steps: 130, vocab_range: (1, VOCAB - 1) };
            assert_eq!(batch_identity::run(&m, &cfg), n * 130, "{v:?}");
        }
    }
}

#[test]
fn batched_hybrid_decode_beside_prefill_chunks() {
    for v in VARIANTS {
        let tel = Arc::new(Telemetry::new());
        let m = model(v, 4, true, &tel);
        // 3 decoding sequences with a 120-token prompt prefilled in chunks of 16 beside them
        let n = batch_identity::run_with_prefill(&m, &[5, 20, 33], 120, 16, (1, VOCAB - 1));
        assert_eq!(n, 3 * 2 * 120usize.div_ceil(16), "{v:?}");
    }
}

fn check_engine(v: Variant, rows: usize, n_cases: usize, reuse: bool) {
    let reference = solo(v);
    let want: Vec<Out> = cases(n_cases).into_iter().map(|r| run_solo(&reference, r)).collect();
    assert!(want.iter().all(|w| w.completion_tokens >= 128), "the comparison needs >= 128 generated tokens");
    let (eng, tel) = batched(v, rows, reuse);
    // all at once: more requests than slots, so they queue, overlap, finish at different times, and slots are recycled
    let rxs: Vec<_> = cases(n_cases).into_iter().map(|r| eng.submit(r).unwrap()).collect();
    for (i, (rx, w)) in rxs.iter().zip(&want).enumerate() {
        let mut got = collect(rx).unwrap();
        if !reuse {
            assert_eq!(got.cached_tokens, 0, "case {i}: prefix reuse is off");
        }
        // with reuse on, the cached length depends on which idle slot a request got; the tokens must not
        got.cached_tokens = w.cached_tokens;
        assert_eq!(&got, w, "case {i} ({v:?}, rows {rows}, reuse {reuse})");
    }
    let m = eng.metrics();
    assert!(m.batch_hist[2..].iter().sum::<u64>() > 0, "no batched decode pass ran: {:?}", m.batch_hist);
    let snap = tel.snapshot();
    assert_eq!((snap.running, snap.waiting), (0, 0), "{snap:?}");
    assert_eq!(snap.requests_total as usize, n_cases);
}

#[test]
fn batched_engine_equals_solo_over_128_tokens() {
    for v in VARIANTS {
        for rows in [2usize, 3, 4] {
            check_engine(v, rows, 9, false);
        }
    }
}

#[test]
fn batched_engine_with_prefix_reuse_equals_solo() {
    // reuse on: a recycled slot restores a snapshot (or its own held sequence) instead of starting from zero
    for v in VARIANTS {
        for rows in [2usize, 4] {
            check_engine(v, rows, 9, true);
        }
    }
}

#[test]
fn prefix_reuse_restores_recurrent_state_from_a_snapshot() {
    for v in VARIANTS {
        prefix_reuse_case(v);
    }
}

fn prefix_reuse_case(v: Variant) {
    let reference = solo(v);
    let base = prompt(900, 40);
    let mut b = base[..33].to_vec();
    b.extend(prompt(901, 12));
    let mut c = base.clone();
    c.extend(prompt(902, 9));
    let req = |p: &Vec<u32>| GenRequest { prompt: p.clone(), max_tokens: 130, ignore_eos: true, ..Default::default() };
    // the prompts in an order that makes each reuse a different kind: the whole prompt again (snapshot at plen-1), a shared
    // prefix that ends between chunk boundaries (snapshot at 32), a prompt that extends an earlier one, a short prefix
    let seq = [base.clone(), b, c, base.clone(), base[..16].to_vec()];
    let want: Vec<Out> = seq.iter().map(|p| run_solo(&reference, req(p))).collect();
    // two slots make `pick_slot` choose between a reusable and a cold slot; three leave one cold
    for rows in [2usize, 3] {
        let (eng, _) = batched(v, rows, true);
        let mut reused = 0;
        for (i, p) in seq.iter().enumerate() {
            let got = collect(&eng.submit(req(p)).unwrap()).unwrap();
            reused += usize::from(got.cached_tokens > 0);
            let w = &want[i];
            assert_eq!(
                (&got.ids, &got.text, got.reason, got.completion_tokens),
                (&w.ids, &w.text, w.reason, w.completion_tokens),
                "request {i}, {v:?}, rows {rows}"
            );
        }
        assert!(reused >= 2, "{v:?}, rows {rows}: only {reused} of {} requests reused a prefix", seq.len());
    }
}
