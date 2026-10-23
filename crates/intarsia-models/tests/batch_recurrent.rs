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
//!
//! The same file is the CUDA gate G-C on the tiny hybrid (docs/results/2026-10-23-rb2-cuda-batching.md): with
//! `INTARSIA_BATCH_TEST_BACKEND=cuda` (or `hip`, lane hb) the model runs on the CUDA (HIP) backend in bf16 (activations and KV), and the batched
//! step and engine must still be bitwise the solo step with the real `_rows` kernels (G-B) and the unfused chains of the
//! solo step's fusions in the multi-row plan; N goes up to 16. Without a GPU the CUDA run skips.
mod common;
use common::ByteCodec;
use intarsia_core::dtype::f32_to_fp8_e4m3;
use intarsia_core::model::{HeadMode, LoadOptions, Model};
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
    /// fp8 projections without activation quantisation (W8A16)
    Moe,
    /// the 35B recipe: W8A8, activations quantised with the static `input_scale` of every fp8 projection
    MoeA8,
}

const VARIANTS: [Variant; 3] = [Variant::Dense, Variant::Moe, Variant::MoeA8];

/// `INTARSIA_BATCH_TEST_BACKEND=cuda` / `hip` runs the file on that GPU backend (bf16), anything else on the cpu backend (f32).
fn gpu() -> Option<&'static str> {
    match std::env::var("INTARSIA_BATCH_TEST_BACKEND").as_deref() {
        Ok("cuda") => Some("cuda"),
        Ok("hip") => Some("hip"),
        _ => None,
    }
}

/// On a GPU backend (cuda or hip) rather than the cpu reference.
fn on_cuda() -> bool {
    gpu().is_some()
}

/// The backend every model of the process runs on (one GPU context for all of them); `None`: no GPU, the test skips.
fn backend() -> Option<Arc<dyn intarsia_core::backend::Backend>> {
    static B: std::sync::OnceLock<Option<Arc<dyn intarsia_core::backend::Backend>>> = std::sync::OnceLock::new();
    B.get_or_init(|| {
        if let Some(name) = gpu() {
            let made = if name == "hip" { intarsia_backend_hip::create(0) } else { intarsia_backend_cuda::create(0) };
            match made {
                Ok(b) => Some(b),
                Err(e) => {
                    eprintln!("SKIPPED: no usable {name} device ({e})");
                    if std::env::var("INTARSIA_REQUIRE_GPU").is_ok_and(|v| v == "1") {
                        panic!("INTARSIA_REQUIRE_GPU=1 but no {name} device: {e}");
                    }
                    None
                }
            }
        } else {
            Some(intarsia_backend_cpu::CpuBackend::arc())
        }
    })
    .clone()
}

/// Most batched sequences the file exercises: 4 on the cpu reference, the CUDA cap on the GPU.
fn max_rows() -> usize {
    if on_cuda() {
        16
    } else {
        4
    }
}

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
fn to_fp8(w: &mut MemWeights, prefix: &str, input_scale: Option<f32>) {
    let name = format!("{prefix}.weight");
    let info = w.info(&name).unwrap_or_else(|| panic!("{name} missing"));
    let t = HostTensor::new(info.dtype, info.shape.clone(), w.bytes(&name).unwrap().to_vec()).unwrap();
    let v = t.to_f32();
    let amax = v.iter().fold(0f32, |m, x| m.max(x.abs()));
    let ws = if amax > 0.0 { amax / 448.0 } else { 1.0 };
    let data: Vec<u8> = v.iter().map(|x| f32_to_fp8_e4m3(x / ws)).collect();
    w.insert(&name, HostTensor::new(DType::Fp8E4m3, info.shape, data).unwrap());
    w.insert(&format!("{prefix}.weight_scale"), HostTensor::from_f32(&[], &[ws]));
    if let Some(s) = input_scale {
        w.insert(&format!("{prefix}.input_scale"), HostTensor::from_f32(&[], &[s]));
    }
}

/// The checked-in f32 fixture, as the CUDA kernels take it: every float tensor in bf16 (they register no f32 weights).
/// `keep`: which tensors to take. The cpu run keeps the f32 tensors as they are.
fn fixture_tensors(keep: impl Fn(&str) -> bool) -> Vec<(String, HostTensor)> {
    let src = SafeTensors::open_dir(&fixture_dir()).unwrap();
    src.names()
        .into_iter()
        .filter(|n| keep(n))
        .map(|name| {
            let info = src.info(&name).unwrap();
            let t = HostTensor::new(info.dtype, info.shape.clone(), src.bytes(&name).unwrap().to_vec()).unwrap();
            let t = if on_cuda() && t.dtype == DType::F32 {
                HostTensor::from_f32_as(DType::Bf16, &t.shape.clone(), &t.to_f32())
            } else {
                t
            };
            (name, t)
        })
        .collect()
}

/// The fixture with its tensors in the dtype the backend takes (cuda: bf16), not renamed.
fn dense_weights() -> MemWeights {
    let mut w = MemWeights::new();
    for (name, t) in fixture_tensors(|_| true) {
        w.insert(&name, t);
    }
    w
}

/// The fixture's attention-free MLP tensors dropped, random fp8 projections and NVFP4 experts in their place.
fn moe_weights(input_scale: Option<f32>) -> MemWeights {
    let mut w = MemWeights::new();
    for (name, t) in fixture_tensors(|n| !n.contains(".mlp.")) {
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
                to_fp8(&mut w, &format!("{p}{s}"), input_scale);
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
    let backend = backend().expect("a backend (tests skip first when there is none)");
    let dir = fixture_dir();
    let mut cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let recipe_src = match v {
        Variant::Dense => include_str!("../../../recipes/qwen3_5.toml"),
        Variant::Moe | Variant::MoeA8 => {
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
        // the cpu reference runs in f32; the CUDA kernels are bf16
        act_dtype: Some(if on_cuda() { DType::Bf16 } else { DType::F32 }),
        kv_dtype: Some(if on_cuda() { DType::Bf16 } else { DType::F32 }),
        fp8_act: match v {
            Variant::Dense => None,
            Variant::Moe => Some(Fp8Act::None),
            Variant::MoeA8 => Some(Fp8Act::Static),
        },
        ..Default::default()
    };
    match v {
        Variant::Dense if !on_cuda() => {
            Model::load(backend, &recipe, &dims, &SafeTensors::open_dir(&dir).unwrap(), &opts, tel.clone())
        }
        Variant::Dense => Model::load(backend, &recipe, &dims, &dense_weights(), &opts, tel.clone()),
        Variant::Moe => Model::load(backend, &recipe, &dims, &moe_weights(None), &opts, tel.clone()),
        Variant::MoeA8 => Model::load(backend, &recipe, &dims, &moe_weights(Some(0.35)), &opts, tel.clone()),
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
    batched_overlap(v, rows, reuse, false)
}

fn batched_overlap(v: Variant, rows: usize, reuse: bool, overlap: bool) -> (BatchedEngine, Arc<Telemetry>) {
    let tel = Arc::new(Telemetry::new());
    let m = model(v, rows, true, &tel);
    let mut o = BatchedOptions::new(rows, 64);
    o.prefix_reuse = reuse;
    o.overlap = overlap;
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
    if backend().is_none() {
        return;
    }
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

/// Distinct kernels `rec` picked for `op`, sorted.
fn picks(rec: &intarsia_core::plan::Recorder, op: intarsia_core::Op) -> Vec<String> {
    let mut v: Vec<String> = rec.calls.iter().filter(|c| c.call.op == op).map(|c| c.kernel.clone()).collect();
    v.sort();
    v.dedup();
    v
}

/// On CUDA the batched plan must really run the row-exact `_rows` kernels (and the solo step's own for the per-row ops):
/// a green identity test over a plan that silently picked something else would prove nothing.
fn check_cuda_batch_picks(m: &Model, v: Variant) {
    use intarsia_core::plan::Phase;
    use intarsia_core::Op;
    let (rec, dec) = (m.dry_run(Phase::Batch, 4, 10).unwrap(), m.dry_run(Phase::Decode, 1, 10).unwrap());
    for op in [Op::Matmul, Op::MatmulFp8, Op::MatmulNvfp4, Op::MoeExperts, Op::Logits, Op::GdnDeltaRule, Op::AttnDecode] {
        eprintln!("{v:?} {op}: decode {:?} | batch(4) {:?}", picks(&dec, op), picks(&rec, op));
    }
    for op in [Op::Matmul, Op::MatmulFp8, Op::MatmulNvfp4, Op::MoeExperts] {
        for k in picks(&rec, op) {
            assert!(k.contains("rows"), "{v:?}: batched {op} runs {k}, not a row-exact kernel");
        }
    }
    if matches!(v, Variant::Moe | Variant::MoeA8) {
        assert!(picks(&rec, Op::MoeExperts).iter().any(|k| k.contains("rows")), "{:?}", picks(&rec, Op::MoeExperts));
        assert!(!picks(&rec, Op::MatmulFp8).is_empty() && !picks(&rec, Op::MatmulNvfp4).is_empty());
    }
    // the per-sequence ops are the solo step's own kernels, or on CUDA (row tables, R3a) the row-table twin of one: the same
    // kernel's one-row call on every row in one launch (`<solo kernel>_rows`, bitwise the solo kernel, tests/row_tables.rs)
    for op in [Op::GdnDeltaRule, Op::AttnDecode] {
        let solo = picks(&dec, op);
        for k in picks(&rec, op) {
            let twin = k.strip_suffix("_rows").is_some_and(|base| solo.iter().any(|s| s == base));
            assert!(solo.contains(&k) || twin, "{v:?} {op}: batched {k}, solo {solo:?}");
        }
    }
}

/// R3c / Fable m3: decode graphs are recorded per row count (batched) and per sequence state (solo), not per step. Batches of
/// changing sizes over a rotating window of sequences, and single rows alternating between sequences, record each shape once.
/// (CUDA only: other backends replay no graphs, and the check is vacuous there.)
#[test]
fn alternating_rows_and_batches_do_not_recapture() {
    if backend().is_none() || !on_cuda() {
        return;
    }
    let tel = Arc::new(Telemetry::new());
    let m = model(VARIANTS[VARIANTS.len() - 1], max_rows(), true, &tel);
    if !m.graph_stats().enabled {
        return;
    }
    let lens: Vec<usize> = (0..4).map(|i| 4 + 3 * i).collect();
    let cfg = batch_identity::Cfg { lens, batches: vec![2, 3, 4, 2, 3, 4, 3, 2], steps: 6, vocab_range: (1, VOCAB - 1) };
    let before = m.graph_stats();
    let n = batch_identity::run(&m, &cfg);
    assert_eq!(n, (2 + 3 + 4 + 2 + 3 + 4 + 3 + 2) * 6);
    let st = m.graph_stats();
    // three row counts: at most three recordings however the sequences rotate (the eager comparison steps in the harness are solo)
    assert!(st.batch_captures - before.batch_captures <= 3, "batched recordings {st:?}");
    // solo steps of four states, interleaved one by one: one recording per state, however often they alternate
    let mut states: Vec<_> = (0..4).map(|_| m.new_state().unwrap()).collect();
    for s in states.iter_mut() {
        m.forward(s, &[1, 2, 3], HeadMode::None, None).unwrap();
    }
    let c0 = m.graph_stats().captures;
    for step in 0..12u32 {
        for s in states.iter_mut() {
            m.forward(s, &[1 + step % 7], HeadMode::LastArgmax, None).unwrap();
        }
    }
    let c1 = m.graph_stats().captures;
    assert!(c1 - c0 <= 4, "solo recordings for 4 alternating states: {}", c1 - c0);
}

#[test]
fn batched_hybrid_decode_is_bitwise_the_solo_step() {
    if backend().is_none() {
        return;
    }
    for v in VARIANTS {
        let tel = Arc::new(Telemetry::new());
        let m = model(v, max_rows(), true, &tel);
        if on_cuda() {
            check_cuda_batch_picks(&m, v);
        }
        // N = 2, 3 and 4 sequences of different prompt lengths (on the GPU also 8 and 16), 130 consecutive batched steps each
        // (the fixture's context is 256 tokens, so each batch size gets fresh sequences); greedy rows alternate onto the
        // on-device argmax path, odd rows sample
        let mut sets = vec![vec![3usize, 9], vec![5, 11, 20], vec![4, 8, 13, 22]];
        if on_cuda() {
            sets.push((0..8).map(|i| 3 + 5 * i).collect());
            sets.push((0..16).map(|i| 2 + 2 * i + i % 3).collect());
        }
        for lens in sets {
            let n = lens.len();
            let cfg = batch_identity::Cfg { lens, batches: vec![n], steps: 130, vocab_range: (1, VOCAB - 1) };
            assert_eq!(batch_identity::run(&m, &cfg), n * 130, "{v:?}");
        }
    }
}

#[test]
fn batched_hybrid_decode_beside_prefill_chunks() {
    if backend().is_none() {
        return;
    }
    for v in VARIANTS {
        let tel = Arc::new(Telemetry::new());
        let m = model(v, max_rows().min(8), true, &tel);
        // 3 decoding sequences with a 120-token prompt prefilled in chunks of 16 beside them
        let n = batch_identity::run_with_prefill(&m, &[5, 20, 33], 120, 16, (1, VOCAB - 1));
        assert_eq!(n, 3 * 2 * 120usize.div_ceil(16), "{v:?}");
    }
}

/// R3d: decode steps on the decode twin (own stream and workspace) beside a prefill chunk on another thread, in the same GPU
/// at the same time: every decode row bitwise its solo step, the chunked prefill bitwise the prefill alone. CUDA only.
#[test]
fn decode_twin_beside_a_running_prefill_is_bitwise_solo() {
    if backend().is_none() || !on_cuda() {
        return;
    }
    for v in VARIANTS {
        let tel = Arc::new(Telemetry::new());
        let m = model(v, max_rows().min(8), true, &tel);
        let twin = m.decode_twin(m.batch_rows()).unwrap();
        let (compared, during) = batch_identity::run_overlapped(&m, &twin, &[5, 20, 33], 120, 16, (1, VOCAB - 1), 2);
        assert!(compared >= 3 * 2 * 120usize.div_ceil(16), "{v:?}: {compared}");
        eprintln!("{v:?}: {compared} decode rows bitwise solo, {during} batched steps begun while a chunk was running");
    }
}

/// The batched engine with overlap on equals the solo engine over 128 tokens (CUDA only).
#[test]
fn batched_engine_with_overlap_equals_solo_over_128_tokens() {
    if backend().is_none() || !on_cuda() {
        return;
    }
    for v in VARIANTS {
        for rows in [2usize, 4, 8] {
            check_engine_overlap(v, rows, 2 * rows + 3, false, true);
        }
    }
}

fn check_engine(v: Variant, rows: usize, n_cases: usize, reuse: bool) {
    check_engine_overlap(v, rows, n_cases, reuse, false)
}

fn check_engine_overlap(v: Variant, rows: usize, n_cases: usize, reuse: bool, overlap: bool) {
    let reference = solo(v);
    let want: Vec<Out> = cases(n_cases).into_iter().map(|r| run_solo(&reference, r)).collect();
    assert!(want.iter().all(|w| w.completion_tokens >= 128), "the comparison needs >= 128 generated tokens");
    let (eng, tel) = batched_overlap(v, rows, reuse, overlap);
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
    if backend().is_none() {
        return;
    }
    for v in VARIANTS {
        for rows in [2usize, 3, 4] {
            check_engine(v, rows, 9, false);
        }
        if on_cuda() {
            // more slots, more requests than slots
            for rows in [8usize, 16] {
                check_engine(v, rows, 2 * rows + 3, false);
            }
        }
    }
}

#[test]
fn batched_engine_with_prefix_reuse_equals_solo() {
    if backend().is_none() {
        return;
    }
    // reuse on: a recycled slot restores a snapshot (or its own held sequence) instead of starting from zero
    for v in VARIANTS {
        for rows in [2usize, 4] {
            check_engine(v, rows, 9, true);
        }
        if on_cuda() {
            check_engine(v, 8, 19, true);
        }
    }
}

#[test]
fn prefix_reuse_restores_recurrent_state_from_a_snapshot() {
    if backend().is_none() {
        return;
    }
    for v in VARIANTS {
        prefix_reuse_case(v);
    }
}

fn prefix_reuse_case(v: Variant) {
    if on_cuda() {
        // the prefix store stays at batch 1: the batched engine never reuses a recurrent slot (checked in `check_engine`:
        // cached_tokens is 0 with reuse requested)
        return;
    }
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

/// m4 of the rb review: the recurrent state of every batched slot (live copy + snapshots) is counted in the memory plan and
/// allocated when the slot is created, not lazily inside the scheduler loop.
#[test]
fn memory_plan_counts_recurrent_state_and_snapshots_and_slots_preallocate_them() {
    if backend().is_none() {
        return;
    }
        for v in VARIANTS {
        let tel = Arc::new(Telemetry::new());
        let solo = model(v, 1, false, &tel);
        let rec = solo.recurrent_bytes_per_state();
        assert!(rec > 0, "{v:?}: the hybrid has recurrent state");
        let rows = 4;
        let m = model(v, rows, true, &tel);
        let keep = m.batch_snapshots();
        assert_eq!(keep, if on_cuda() { 0 } else { 4 }, "the prefix store stays at batch 1 on cuda");
        let extra = (rows as u64 - 1) * rec * (1 + keep as u64);
        // the recurrent states of the extra slots, plus (row tables) the kernel scratch of the multi-row attention launch, which is
        // sized for `rows` rows of split-KV partials instead of one
        let ws = m.plan.workspace_bytes;
        assert!(ws >= solo.plan.workspace_bytes + extra && ws - (solo.plan.workspace_bytes + extra) < (64 << 20), "{v:?}: workspace {ws}");
        assert!(m.plan.notes.iter().any(|n| n.contains("recurrent state of 4 batched sequences")), "{:?}", m.plan.notes);
        // a slot holds its snapshot buffers from the start ...
        let mut st = m.new_state().unwrap();
        assert_eq!(st.snap_pool.len(), keep, "{v:?}");
        // ... and a long run of snapshots (more than it keeps) reuses them: pool + snapshots never grow past `keep`
        let be = backend().unwrap();
        for i in 0..3 * keep.max(1) {
            m.forward(&mut st, &[1 + (i as u32 % 90)], intarsia_core::model::HeadMode::None, None).unwrap();
            st.snapshot(&*be, keep).unwrap();
            assert_eq!(st.snap_pool.len() + st.snapshots.len(), keep, "{v:?}: after snapshot {i}");
            assert!(st.snapshots.len() <= keep);
        }
        if keep == 0 {
            // cuda: no snapshots at all, and the engine never reuses a recurrent slot's prefix
            assert!(st.snapshots.is_empty() && st.snap_pool.is_empty());
            continue;
        }
        // truncating below the snapshots returns their buffers to the pool
        let keep_len = st.snapshots[0].len;
        st.truncate(keep_len, &*be).unwrap();
        assert_eq!(st.snap_pool.len() + st.snapshots.len(), keep, "{v:?}: after truncate");
        // a solo (non-batching) model allocates lazily, as before
        let lazy = solo.new_state().unwrap();
        assert!(lazy.snap_pool.is_empty());
    }
}
