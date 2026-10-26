//! `--spec mtp` under tensor parallelism with in-process ranks (lane dfh), on the cpu backend: the dense tiny hybrid target
//! (`tests/fixtures/tiny_qwen3_5`: 4 GatedDeltaNet and 2 gated full-attention layers, dense SwiGLU, the `qwen3_5` recipe) plus a
//! random bf16 MTP head in the Qwen3.5 / 3.8 dense layout (`mtp.*`), at tp=2 over `HostCommGroup`.
//!
//! * Every rank holds its shard of the head (q / k / v / gate / up by rows, o / down by columns) and runs it in lockstep with
//!   rank 0, which drafts: the drafts at tp=2 equal the drafts at tp=1, with the replicated head and with the vocab-parallel
//!   one (rank 0 drafts from its own share of the head when `--mtp-draft-vocab` fits inside it, else from a whole copy).
//! * Greedy output with MTP on at tp=2 equals MTP off at tp=2, with the head's own drafts and with scripted drafts that hit
//!   every acceptance pattern through the per-rank GatedDeltaNet rollback, at k = 1..3, for odd `max_tokens`.
//! * The cpu kernels run every row of a multi-row call with the single-row loop: this proves the engine and rank logic, not a
//!   GPU backend's numerics (the HIP runs are in docs/results/2026-10-26-dfh-hip-dflash.md).

mod common;
use common::ByteCodec;
use intarsia_backend_cpu::CpuBackend;
use intarsia_core::model::{DraftScript, HeadMode, LoadOptions, Model, StepOut};
use intarsia_core::recipe::ModelDims;
use intarsia_core::rng::SplitMix64;
use intarsia_core::spec::{SpecConfig, SpecMode};
use intarsia_core::weights::{MemWeights, WeightSource};
use intarsia_core::{DType, Engine, EngineMeta, FinishReason, GenEvent, GenRequest, HostTensor, LocalEngine, Telemetry};
use intarsia_models::SafeTensors;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const H: usize = 64;
const I: usize = 96;
const NH: usize = 4;
const NKV: usize = 2;
const HD: usize = 32;
const V: usize = 96;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_qwen3_5")
}

fn spec() -> (intarsia_core::recipe::Recipe, ModelDims) {
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/qwen3_5.toml")).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fixture_dir().join("config.json")).unwrap()).unwrap();
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    assert_eq!((dims.hidden, dims.vocab, dims.n_heads, dims.n_kv_heads, dims.head_dim, dims.intermediate), (H, V, NH, NKV, HD, I));
    (recipe, dims)
}

fn bf16(rng: &mut SplitMix64, shape: &[usize], scale: f32) -> HostTensor {
    HostTensor::from_f32_as(DType::Bf16, shape, &rng.normal_vec(shape.iter().product(), scale))
}

/// The fixture's weights plus a random MTP head (`scale` keeps its drafts near "copy the embedding", which the random target then
/// sometimes accepts).
fn weights() -> MemWeights {
    let src = SafeTensors::open_dir(&fixture_dir()).unwrap();
    let mut w = MemWeights::new();
    for name in src.names() {
        let info = src.info(&name).unwrap();
        w.insert(&name, HostTensor::new(info.dtype, info.shape.clone(), src.bytes(&name).unwrap().to_vec()).unwrap());
    }
    let mut rng = SplitMix64::new(0xd_f4_17);
    let (m, l) = ("mtp", "mtp.layers.0");
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
    w.insert(&format!("{l}.mlp.gate_proj.weight"), bf16(&mut rng, &[I, H], 0.15));
    w.insert(&format!("{l}.mlp.up_proj.weight"), bf16(&mut rng, &[I, H], 0.15));
    w.insert(&format!("{l}.mlp.down_proj.weight"), bf16(&mut rng, &[H, I], 0.15));
    w
}

/// `k`: MTP on with the verify pass sized for k drafts; `draft_vocab`: `--mtp-draft-vocab` (0 = the full head).
fn opts(k: Option<usize>, vocab_par: bool, draft_vocab: usize) -> LoadOptions {
    LoadOptions {
        max_seq: 512,
        prefill_chunk: 8,
        act_dtype: Some(DType::F32),
        kv_dtype: Some(DType::F32),
        profile_every: 0,
        tp_vocab: vocab_par,
        mtp: k.is_some(),
        mtp_k: k.unwrap_or(0),
        mtp_draft_vocab: Some(draft_vocab),
        ..Default::default()
    }
}

fn finish(mut m: Model, k: Option<usize>) -> Model {
    if let Some(k) = k {
        m.enable_verify(k + 1).unwrap();
        m.configure_plan(&Default::default(), None).unwrap();
    }
    m
}

fn model_tp1(k: Option<usize>, draft_vocab: usize) -> (Model, Arc<Telemetry>) {
    let (recipe, dims) = spec();
    let tel = Arc::new(Telemetry::new());
    let m = Model::load(CpuBackend::arc(), &recipe, &dims, &weights(), &opts(k, false, draft_vocab), tel.clone()).unwrap();
    (finish(m, k), tel)
}

fn model_tp2(k: Option<usize>, vocab_par: bool, draft_vocab: usize) -> (Model, Arc<Telemetry>) {
    let (recipe, dims) = spec();
    let tel = Arc::new(Telemetry::new());
    let backends: Vec<Arc<dyn intarsia_core::backend::Backend>> = (0..2).map(|_| CpuBackend::arc()).collect();
    let comms = intarsia_core::HostCommGroup::new(backends.clone());
    let m = Model::load_tp(backends, comms, &recipe, &dims, &weights(), &opts(k, vocab_par, draft_vocab), tel.clone()).unwrap();
    assert_eq!(m.vocab_parallel(), vocab_par, "the head split as asked");
    (finish(m, k), tel)
}

const PROMPTS: [&[u32]; 3] = [&[3, 9, 27, 81, 50, 7, 7, 1, 60, 2, 33, 33, 12], &[90, 1, 2, 3, 4, 5], &[44; 21]];

/// Prefill `p`, then the head's drafts after the target's next token, then once more after two plain decode steps (the head's
/// catch-up over queued rows).
fn drafts(m: &Model, p: &[u32], k: usize) -> Vec<Vec<u32>> {
    let mut st = m.new_state().unwrap();
    let mut next = match m.forward(&mut st, p, HeadMode::LastArgmax, None).unwrap() {
        StepOut::Token(t) => t,
        o => panic!("prefill gave {o:?}"),
    };
    let mut out = vec![m.mtp_draft(&mut st, next, k).unwrap()];
    for _ in 0..2 {
        next = match m.forward(&mut st, &[next], HeadMode::LastArgmax, None).unwrap() {
            StepOut::Token(t) => t,
            o => panic!("decode gave {o:?}"),
        };
    }
    out.push(m.mtp_draft(&mut st, next, k).unwrap());
    out
}

#[test]
fn mtp_tp2_drafts_equal_tp1_drafts() {
    // draft vocab 40 fits inside rank 0's share of a vocab-parallel head (48 rows); 0 (the full head) needs rank 0's whole copy
    for dv in [0usize, 40] {
        let (m1, _) = model_tp1(Some(3), dv);
        for vocab_par in [false, true] {
            let (m2, _) = model_tp2(Some(3), vocab_par, dv);
            for p in PROMPTS {
                let (a, b) = (drafts(&m1, p, 3), drafts(&m2, p, 3));
                eprintln!("draft vocab {dv} vocab_par {vocab_par} prompt {p:?}: tp1 {a:?} tp2 {b:?}");
                assert_eq!(a, b, "tp=2 drafts (draft vocab {dv}, vocab_par {vocab_par}) differ from tp=1 for {p:?}");
                assert!(a.iter().all(|d| d.len() == 3));
            }
        }
    }
}

#[derive(Debug, PartialEq)]
struct Out {
    ids: Vec<u32>,
    reason: FinishReason,
    completion_tokens: usize,
}

fn run(e: &LocalEngine, req: GenRequest) -> Out {
    let (mut ids, mut done) = (vec![], None);
    e.generate(req, &mut |ev| match ev {
        GenEvent::Token { id, .. } => ids.push(id),
        GenEvent::Done { finish_reason, completion_tokens, .. } => done = Some((finish_reason, completion_tokens)),
        GenEvent::Error { message } => panic!("engine error: {message}"),
    })
    .unwrap();
    let (reason, completion_tokens) = done.unwrap();
    Out { ids, reason, completion_tokens }
}

fn req(prompt: &[u32], n: usize) -> GenRequest {
    GenRequest { prompt: prompt.to_vec(), max_tokens: n, temperature: 0.0, ignore_eos: true, ..Default::default() }
}

fn engine(k: Option<usize>, vocab_par: bool, script: Option<DraftScript>) -> (LocalEngine, Arc<Telemetry>) {
    let (m, tel) = model_tp2(k, vocab_par, 40);
    if script.is_some() {
        m.mtp_script(script).unwrap();
    }
    let spec = match k {
        Some(k) => SpecConfig { mode: SpecMode::Mtp, k },
        None => SpecConfig::default(),
    };
    let e = LocalEngine::new(m, Arc::new(ByteCodec), EngineMeta { id: "tiny-mtp-tp2".into(), eos_ids: vec![], spec, ..Default::default() }, tel.clone())
        .unwrap();
    (e, tel)
}

/// Drafts from the true continuation with one wrong id at a position that depends on the context (or none): every acceptance
/// pattern, all accepted, some, none.
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
fn mtp_tp2_greedy_ids_equal_spec_off() {
    for vocab_par in [true, false] {
        let truth: Arc<Mutex<Vec<Vec<u32>>>> = Arc::new(Mutex::new(Vec::new()));
        let (off, _) = engine(None, vocab_par, None);
        for p in PROMPTS {
            let o = run(&off, req(p, 100));
            let mut s = p.to_vec();
            s.extend(&o.ids);
            truth.lock().unwrap().push(s);
        }
        for (k, script) in [(1usize, false), (3, false), (2, true), (3, true)] {
            let (on, tel) = engine(Some(k), vocab_par, script.then(|| scripted(truth.clone(), 11)));
            for p in PROMPTS {
                assert_eq!(run(&off, req(p, 90)), run(&on, req(p, 90)), "tp2 vocab_par {vocab_par} k {k} scripted {script} prompt {p:?}");
                for n in [1usize, 4, 7] {
                    assert_eq!(run(&off, req(p, n)), run(&on, req(p, n)), "tp2 vocab_par {vocab_par} k {k} max_tokens {n}");
                }
            }
            let t = tel.snapshot();
            eprintln!("tp2 vocab_par {vocab_par} k {k} scripted {script}: {} passes, {} drafted, {} accepted", t.spec_steps, t.spec_proposed, t.spec_accepted);
            assert!(t.spec_steps > 0, "the verify path ran at tp=2");
            if script {
                assert!(t.spec_accepted > 0 && t.spec_accepted < t.spec_proposed, "partial acceptance exercised at tp=2");
            }
        }
    }
}
