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
use common::{ByteCodec, Flaky, FAIL_MEMORY, OK};
use intarsia_backend_cpu::CpuBackend;
use intarsia_core::model::{DraftScript, HeadMode, LoadOptions, Model, StepOut};
use intarsia_core::recipe::ModelDims;
use intarsia_core::rng::SplitMix64;
use intarsia_core::spec::{SpecConfig, SpecMode};
use intarsia_core::weights::{MemWeights, WeightSource};
use intarsia_core::{DType, Engine, EngineMeta, Error, FinishReason, GenEvent, GenRequest, HostTensor, LocalEngine, Telemetry};
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

fn run(e: &dyn Engine, req: GenRequest) -> Out {
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
        Some(k) => SpecConfig { mode: SpecMode::Mtp, k, k_min: None },
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

// ---- the dense head whole on rank 0 (lane d2r: a rank in another process cannot run the lockstep head) ----

/// `opts` with the head whole on rank 0 (`LoadOptions::mtp_whole`, what the engine loader sets for a group with a remote rank).
fn opts_whole(k: Option<usize>, vocab_par: bool, draft_vocab: usize, act: DType) -> LoadOptions {
    LoadOptions { mtp_whole: true, act_dtype: Some(act), kv_dtype: Some(act), ..opts(k, vocab_par, draft_vocab) }
}

/// In-process tp=2 with the dense head whole on rank 0.
fn model_tp2_whole(k: Option<usize>, vocab_par: bool, draft_vocab: usize, act: DType) -> (Model, Arc<Telemetry>) {
    let (recipe, dims) = spec();
    let tel = Arc::new(Telemetry::new());
    let backends: Vec<Arc<dyn intarsia_core::backend::Backend>> = (0..2).map(|_| CpuBackend::arc()).collect();
    let comms = intarsia_core::HostCommGroup::new(backends.clone());
    let m = Model::load_tp(backends, comms, &recipe, &dims, &weights(), &opts_whole(k, vocab_par, draft_vocab, act), tel.clone()).unwrap();
    (finish(m, k), tel)
}

/// Rank 0 here, rank 1 behind a localhost TCP connection running `tp_remote::serve` on another thread (as `intarsia tp-worker` does),
/// joined by the host communicator; `--spec mtp` on both when `k` is given (rank 1 loads no head and enables its verify rows).
fn model_tp2_remote(k: Option<usize>, draft_vocab: usize, act: DType) -> (Model, Arc<Telemetry>, std::thread::JoinHandle<u64>) {
    use intarsia_core::model::TpRank;
    use intarsia_core::tp_remote;
    let (recipe, dims) = spec();
    let tel = Arc::new(Telemetry::new());
    let backends: Vec<Arc<dyn intarsia_core::backend::Backend>> = (0..2).map(|_| CpuBackend::arc()).collect();
    let comms = intarsia_core::HostCommGroup::new(backends.clone());
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (b1, c1, r1, d1) = (backends[1].clone(), comms[1].clone(), recipe.clone(), dims.clone());
    let server = std::thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        let mut m = Model::load_shard(
            b1,
            &r1,
            &d1,
            &weights(),
            &opts_whole(k, false, draft_vocab, act),
            Arc::new(Telemetry::new()),
            TpRank { rank: 1, world: 2, comm: Some(c1) },
        )
        .unwrap();
        assert!(!m.has_mtp(), "rank 1 keeps no head");
        if let Some(k) = k {
            m.enable_verify_served(k + 1).unwrap();
        }
        let cap = tp_remote::handshake(&mut s, "fixture", m.plan.kv_cap_tokens).unwrap();
        m.clamp_kv_cap(cap).unwrap();
        m.configure_plan(&Default::default(), None).unwrap();
        tp_remote::serve(&m, s).unwrap()
    });
    let mut s = std::net::TcpStream::connect(addr).unwrap();
    let tp = TpRank { rank: 0, world: 2, comm: Some(comms[0].clone()) };
    let mut m = Model::load_shard(backends[0].clone(), &recipe, &dims, &weights(), &opts_whole(k, false, draft_vocab, act), tel.clone(), tp).unwrap();
    let cap = tp_remote::handshake(&mut s, "fixture", m.plan.kv_cap_tokens).unwrap();
    m.clamp_kv_cap(cap).unwrap();
    m.attach_remote(tp_remote::RemotePeer::new(s, comms[0].clone()).unwrap()).unwrap();
    (finish(m, k), tel, server)
}

/// The whole dense head on rank 0 computes what the tp=1 head computes, bit for bit, from the same hidden rows (same weights, same
/// kernels, no collective), up to the cap of a whole head's pass (`prefill_chunk / world` rows).
#[test]
fn dense_mtp_head_whole_on_rank0_is_bitwise_the_tp1_head() {
    for act in [DType::F32, DType::Bf16] {
        let (recipe, dims) = spec();
        let o1 = LoadOptions { act_dtype: Some(act), kv_dtype: Some(act), ..opts(Some(2), false, 0) };
        let m1 = finish(Model::load(CpuBackend::arc(), &recipe, &dims, &weights(), &o1, Arc::new(Telemetry::new())).unwrap(), Some(2));
        let (m2, _) = model_tp2_whole(Some(2), false, 0, act);
        assert!(m2.has_mtp());
        let mut rng = SplitMix64::new(99);
        for t in 1..=4usize {
            let ids: Vec<u32> = (0..t as u32).map(|i| (i * 31 + 5) % V as u32).collect();
            let hidden = rng.normal_vec(t * H, 1.0);
            let (a, b) = (m1.mtp_reference_pass(&ids, &hidden).unwrap(), m2.mtp_reference_pass(&ids, &hidden).unwrap());
            for (r, (x, y)) in a.iter().zip(&b).enumerate() {
                assert!(x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()), "act {act} t {t}: row {r} differs from tp=1");
            }
        }
        assert!(m2.mtp_reference_pass(&[1; 5], &vec![0.5; 5 * H]).is_err(), "a pass over the whole head's row cap is refused");
    }
}

/// Through the model (prefill, the head's catch-up, chained drafts): the whole head at tp=2, in-process and with the other rank in
/// another "process", drafts what tp=1 drafts on this fixture (f32: the trunk's two-way sums move the hidden rows by rounding only).
#[test]
fn dense_mtp_whole_head_tp2_drafts_equal_tp1_drafts() {
    for dv in [0usize, 40] {
        let (m1, _) = model_tp1(Some(3), dv);
        let (mi, _) = model_tp2_whole(Some(3), false, dv, DType::F32);
        let (mv, _) = model_tp2_whole(Some(3), true, dv, DType::F32);
        let (mr, _, server) = model_tp2_remote(Some(3), dv, DType::F32);
        for p in PROMPTS {
            let a = drafts(&m1, p, 3);
            let (b, v, c) = (drafts(&mi, p, 3), drafts(&mv, p, 3), drafts(&mr, p, 3));
            eprintln!("draft vocab {dv} prompt {p:?}: tp1 {a:?} in-process {b:?} vocab-par {v:?} remote {c:?}");
            assert_eq!(a, b, "in-process whole-head tp=2 drafts (draft vocab {dv}) differ from tp=1 for {p:?}");
            assert_eq!(a, v, "vocab-parallel whole-head tp=2 drafts (draft vocab {dv}) differ from tp=1 for {p:?}");
            assert_eq!(a, c, "remote whole-head tp=2 drafts (draft vocab {dv}) differ from tp=1 for {p:?}");
            assert!(a.iter().all(|d| d.len() == 3));
        }
        drop(mr);
        server.join().unwrap();
    }
}

fn engine_whole(k: Option<usize>, remote: bool, script: Option<DraftScript>) -> (LocalEngine, Arc<Telemetry>, Option<std::thread::JoinHandle<u64>>) {
    let (m, tel, server) = if remote {
        let (m, tel, s) = model_tp2_remote(k, 40, DType::F32);
        (m, tel, Some(s))
    } else {
        let (m, tel) = model_tp2_whole(k, false, 40, DType::F32);
        (m, tel, None)
    };
    if script.is_some() {
        m.mtp_script(script).unwrap();
    }
    let spec = match k {
        Some(k) => SpecConfig { mode: SpecMode::Mtp, k, k_min: None },
        None => SpecConfig::default(),
    };
    let e = LocalEngine::new(m, Arc::new(ByteCodec), EngineMeta { id: "tiny-mtp-whole-tp2".into(), eos_ids: vec![], spec, ..Default::default() }, tel.clone())
        .unwrap();
    (e, tel, server)
}

/// Greedy output with the whole dense head at tp=2 (`--spec mtp`) equals `--spec off` at tp=2, in-process and with the other rank in
/// another "process" (the verify pass and the rollback over the connection), with the head's own drafts and with scripted drafts that
/// hit every acceptance pattern, at k = 1..3, 256 tokens and short budgets.
#[test]
fn dense_mtp_whole_head_tp2_greedy_ids_equal_spec_off() {
    for remote in [false, true] {
        let (off, _, off_server) = engine_whole(None, remote, None);
        let truth: Arc<Mutex<Vec<Vec<u32>>>> = Arc::new(Mutex::new(Vec::new()));
        let mut want = Vec::new();
        for p in PROMPTS {
            let o = run(&off, req(p, 256));
            let mut s = p.to_vec();
            s.extend(&o.ids);
            truth.lock().unwrap().push(s);
            want.push(o);
        }
        for (k, script) in [(1usize, false), (2, false), (3, false), (2, true), (3, true)] {
            let (on, tel, server) = engine_whole(Some(k), remote, script.then(|| scripted(truth.clone(), 11)));
            for (p, o) in PROMPTS.iter().zip(&want) {
                assert_eq!(&run(&on, req(p, 256)), o, "remote {remote} k {k} scripted {script} prompt {p:?}");
                for n in [1usize, 4, 7] {
                    assert_eq!(run(&off, req(p, n)), run(&on, req(p, n)), "remote {remote} k {k} scripted {script} max_tokens {n}");
                }
            }
            let t = tel.snapshot();
            eprintln!("remote {remote} k {k} scripted {script}: {} passes, {} drafted, {} accepted", t.spec_steps, t.spec_proposed, t.spec_accepted);
            assert!(t.spec_steps > 0, "the verify path ran at tp=2");
            if script {
                assert!(t.spec_accepted > 0 && t.spec_accepted < t.spec_proposed, "partial acceptance exercised at tp=2");
            }
            if remote && k == 2 && !script {
                // a sampled request on a tp=2 MTP group takes the plain path (sampled speculation is single-rank)
                let r = GenRequest { prompt: PROMPTS[0].to_vec(), max_tokens: 24, temperature: 0.8, ignore_eos: true, ..Default::default() };
                assert_eq!(run(&on, r).completion_tokens, 24, "a sampled request on the remote tp=2 MTP group");
            }
            drop(on);
            if let Some(s) = server {
                assert!(s.join().unwrap() > 0, "the remote rank served passes");
            }
        }
        drop(off);
        if let Some(s) = off_server {
            s.join().unwrap();
        }
    }
}

/// The model as a checkpoint directory (config, weights with the MTP head, a word-level tokenizer) for the engine loader.
///
/// Written once per process: tests run in parallel threads, and rewriting `model.safetensors` under an engine that has it mapped is a SIGBUS.
fn model_dir() -> PathBuf {
    static DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    DIR.get_or_init(write_model_dir).clone()
}

fn write_model_dir() -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("tp_mtp_model");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(fixture_dir().join("config.json"), dir.join("config.json")).unwrap();
    let w = weights();
    let ts: Vec<(String, HostTensor)> = w
        .names()
        .into_iter()
        .map(|n| {
            let info = w.info(&n).unwrap();
            let t = HostTensor::new(info.dtype, info.shape.clone(), w.bytes(&n).unwrap().to_vec()).unwrap();
            (n, t)
        })
        .collect();
    let refs: Vec<(&str, &HostTensor)> = ts.iter().map(|(n, t)| (n.as_str(), t)).collect();
    intarsia_models::safetensors::write_safetensors(&dir.join("model.safetensors"), &refs).unwrap();
    let vocab: serde_json::Map<String, serde_json::Value> = (0..V).map(|i| (format!("t{i}"), i.into())).collect();
    let tok = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [], "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "t0"}
    });
    std::fs::write(dir.join("tokenizer.json"), tok.to_string()).unwrap();
    dir
}

/// Through the engine loader (`load_engine_tp`, what `generate` / `serve` call): `--spec mtp` loads on an in-process tp=2 group
/// (window 6 found the loader's backend gate still refusing it) and its greedy ids equal `--spec off`'s on the same group.
#[test]
fn mtp_tp2_loads_through_the_engine_loader_and_matches_spec_off() {
    let dir = model_dir();
    for tp_vocab in [true, false] {
        let load = |spec: SpecMode| {
            let backends: Vec<Arc<dyn intarsia_core::backend::Backend>> = (0..2).map(|_| CpuBackend::arc() as Arc<dyn intarsia_core::backend::Backend>).collect();
            let comms = intarsia_core::HostCommGroup::new(backends.clone());
            let o = intarsia_models::EngineOptions {
                load: LoadOptions { max_seq: 256, act_dtype: Some(DType::F32), kv_dtype: Some(DType::F32), tp_vocab, ..Default::default() },
                spec: SpecConfig { mode: spec, k: 2, k_min: None },
                ..Default::default()
            };
            intarsia_models::load_engine_tp(&dir, backends, comms, &o).unwrap_or_else(|e| panic!("tp=2 load with --spec {spec:?} (tp_vocab {tp_vocab}): {e}"))
        };
        let (off, on) = (load(SpecMode::Off), load(SpecMode::Mtp));
        for p in PROMPTS {
            assert_eq!(run(&*off.engine, req(p, 60)), run(&*on.engine, req(p, 60)), "loader tp2 tp_vocab {tp_vocab} prompt {p:?}");
        }
    }
}

/// A lone `--spec-k 0` (no `--spec`) through the engine loader: refused with a message that names it, at tp=1 and tp=2 alike; the
/// CLI's parse refuses a draft length outside 1..=max for every mode (review of PR #185, 2026-10-26: it was read as the recipe default
/// with k = 0).
#[test]
fn a_lone_spec_k_zero_is_refused_by_the_loader() {
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = model_dir();
    assert!(SpecConfig::parse("off", 0).is_err() && SpecConfig::parse("mtp", 0).is_err(), "the CLI parse refuses k = 0");
    let o = intarsia_models::EngineOptions {
        load: LoadOptions { max_seq: 256, act_dtype: Some(DType::F32), kv_dtype: Some(DType::F32), ..Default::default() },
        spec_auto: true,
        spec_k_given: true,
        spec: SpecConfig { mode: SpecMode::Off, k: 0, k_min: None },
        ..Default::default()
    };
    let one = intarsia_models::load_engine(&dir, CpuBackend::arc(), &o);
    assert!(one.is_err_and(|e| e.to_string().contains("--spec-k 0")), "tp=1 load of a lone --spec-k 0");
    let backends: Vec<Arc<dyn intarsia_core::backend::Backend>> = (0..2).map(|_| CpuBackend::arc() as Arc<dyn intarsia_core::backend::Backend>).collect();
    let comms = intarsia_core::HostCommGroup::new(backends.clone());
    let two = intarsia_models::load_engine_tp(&dir, backends, comms, &o);
    assert!(two.is_err_and(|e| e.to_string().contains("--spec-k 0")), "tp=2 load of a lone --spec-k 0");
}

/// `INTARSIA_SPEC_DEFAULT` is process-wide: the tests that read or set it take this lock for their whole body.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The recipe default's adaptive depth (`[tuning.<backend>] mtp_default_k_min`, `intarsia_models::load::default_spec`): a floor below
/// k makes the default adaptive on that backend only; a lone `--spec-k` and a floor not below k keep a fixed length.
#[test]
fn default_spec_takes_the_recipe_floor_unless_spec_k_is_given() {
    use intarsia_models::load::default_spec;
    let r = intarsia_models::parse_recipe(&format!("{}\n[tuning.hip]\nmtp_default_tp_k = 4\nmtp_default_k_min = 2\n", include_str!("../../../recipes/qwen3_5.toml"))).unwrap();
    let auto = intarsia_models::EngineOptions { spec_auto: true, ..Default::default() };
    let mtp = |k, k_min| SpecConfig { mode: SpecMode::Mtp, k, k_min };
    assert_eq!(default_spec(&auto, &r, "hip", 4), mtp(4, Some(2)), "recipe floor 2 under k 4");
    assert_eq!(default_spec(&auto, &r, "cuda", 4), mtp(4, None), "another backend: fixed");
    assert_eq!(default_spec(&auto, &r, "hip", 2), mtp(2, None), "floor not below k: fixed");
    let lone = intarsia_models::EngineOptions { spec_auto: true, spec_k_given: true, ..Default::default() };
    assert_eq!(default_spec(&lone, &r, "hip", 3), mtp(3, None), "a lone --spec-k fixes the length");
}

/// The recipe's tensor-parallel MTP default (`[tuning.<backend>] mtp_default_tp_k`, `intarsia_models::load::tp_mtp_default`): on only
/// without --spec, at tp > 1 on in-process ranks of that backend, one sequence, and not with INTARSIA_SPEC_DEFAULT=0.
#[test]
fn tp_mtp_default_needs_no_spec_tp_the_backend_one_sequence_and_no_opt_out() {
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    use intarsia_models::load::tp_mtp_default;
    let with = intarsia_models::parse_recipe(&format!("{}\n[tuning.hip]\nmtp_default_tp_k = 2\n", include_str!("../../../recipes/qwen3_5.toml"))).unwrap();
    let without = intarsia_models::parse_recipe(include_str!("../../../recipes/qwen3_5.toml")).unwrap();
    let auto = intarsia_models::EngineOptions { spec_auto: true, ..Default::default() };
    assert_eq!(tp_mtp_default(&auto, &with, "hip", 2, true), Some(2), "on: no --spec, hip, tp 2, in-process");
    assert_eq!(tp_mtp_default(&auto, &without, "hip", 2, true), None, "a recipe without the key");
    assert_eq!(tp_mtp_default(&auto, &with, "hip", 1, true), None, "tp 1 unchanged");
    assert_eq!(tp_mtp_default(&auto, &with, "cuda", 2, true), None, "another backend unchanged");
    assert_eq!(tp_mtp_default(&auto, &with, "hip", 2, false), None, "a rank in another process");
    let explicit = intarsia_models::EngineOptions { spec_auto: false, ..Default::default() };
    assert_eq!(tp_mtp_default(&explicit, &with, "hip", 2, true), None, "--spec off given (or a library caller)");
    let batch = intarsia_models::EngineOptions { spec_auto: true, max_batch: Some(4), ..Default::default() };
    assert_eq!(tp_mtp_default(&batch, &with, "hip", 2, true), None, "--max-batch 4");
    let dfl = intarsia_models::EngineOptions { spec_auto: true, spec: SpecConfig { mode: SpecMode::Dflash, k: 7, k_min: None }, ..Default::default() };
    assert_eq!(tp_mtp_default(&dfl, &with, "hip", 2, true), None, "another --spec");
    // a lone --spec-k N (spec_auto, no --spec) keeps the default and sets only its draft length
    let lone = intarsia_models::EngineOptions { spec_auto: true, spec_k_given: true, spec: SpecConfig { mode: SpecMode::Off, k: 3, k_min: None }, ..Default::default() };
    assert_eq!(tp_mtp_default(&lone, &with, "hip", 2, true), Some(3), "lone --spec-k 3: default applies with k = 3");
    assert_eq!(tp_mtp_default(&lone, &without, "hip", 2, true), None, "lone --spec-k, no recipe key: off (the loader warns)");
    assert_eq!(tp_mtp_default(&lone, &with, "cuda", 2, true), None, "lone --spec-k, another backend: off");
    assert_eq!(tp_mtp_default(&lone, &with, "hip", 1, true), None, "lone --spec-k, tp 1: off");
    let lone0 = intarsia_models::EngineOptions { spec_auto: true, spec_k_given: true, spec: SpecConfig { mode: SpecMode::Off, k: 0, k_min: None }, ..Default::default() };
    assert_eq!(tp_mtp_default(&lone0, &with, "hip", 2, true), None, "lone --spec-k 0: no drafting, the default is not applied (not SpecConfig{{Mtp, k: 0}})");
    let explicit_k = intarsia_models::EngineOptions { spec_auto: false, spec_k_given: true, spec: SpecConfig { mode: SpecMode::Off, k: 3, k_min: None }, ..Default::default() };
    assert_eq!(tp_mtp_default(&explicit_k, &with, "hip", 2, true), None, "--spec off --spec-k 3: unchanged, off");
    // a prefix store (explicit caching intent; a drafting request bypasses it) skips the default; an explicit --spec mtp is not this function's
    let stored = intarsia_models::EngineOptions {
        spec_auto: true,
        prefix_store: intarsia_core::prefix_store::PrefixStoreConfig::parse("on").unwrap(),
        ..Default::default()
    };
    assert_eq!(tp_mtp_default(&stored, &with, "hip", 2, true), None, "--prefix-store on: no default at hip tp 2");
    let off_store = intarsia_models::EngineOptions { spec_auto: true, prefix_store: intarsia_core::prefix_store::PrefixStoreConfig::parse("off").unwrap(), ..Default::default() };
    assert_eq!(tp_mtp_default(&off_store, &with, "hip", 2, true), Some(2), "--prefix-store off is no store");
    std::env::set_var("INTARSIA_PREFIX_STORE", "on");
    let env_store = tp_mtp_default(&auto, &with, "hip", 2, true);
    std::env::remove_var("INTARSIA_PREFIX_STORE");
    assert_eq!(env_store, None, "INTARSIA_PREFIX_STORE=on: no default");
    assert_eq!(tp_mtp_default(&auto, &with, "hip", 2, true), Some(2), "and back on once it is unset");
    // the default through the engine loader in the same test (the env var below is process-wide): runs spec, ids equal off
    loader_default_matches_off();
    std::env::set_var("INTARSIA_SPEC_DEFAULT", "0");
    let off = tp_mtp_default(&auto, &with, "hip", 2, true);
    std::env::remove_var("INTARSIA_SPEC_DEFAULT");
    assert_eq!(off, None, "INTARSIA_SPEC_DEFAULT=0");
}

/// The recipe's single-rank MTP default (`[tuning.<backend>] mtp_default_k`, `intarsia_models::load::mtp_default`, lane cmtp): on only
/// without --spec, at one rank on that backend, one sequence, and not with INTARSIA_SPEC_DEFAULT=0; the tp key and this key do not
/// answer for each other's world size. Through the engine loader: runs spec, ids equal off; a checkpoint without the head falls back.
#[test]
fn single_rank_mtp_default_needs_no_spec_one_rank_the_backend_one_sequence_and_no_opt_out() {
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    use intarsia_models::load::{mtp_default, tp_mtp_default};
    let base = include_str!("../../../recipes/qwen3_5.toml");
    let with = intarsia_models::parse_recipe(&format!("{base}\n[tuning.cuda]\nmtp_default_k = 2\n")).unwrap();
    let tp_only = intarsia_models::parse_recipe(&format!("{base}\n[tuning.cuda]\nmtp_default_tp_k = 2\n")).unwrap();
    let without = intarsia_models::parse_recipe(base).unwrap();
    let auto = intarsia_models::EngineOptions { spec_auto: true, ..Default::default() };
    assert_eq!(mtp_default(&auto, &with, "cuda", 1, true), Some(2), "on: no --spec, cuda, one rank");
    assert_eq!(mtp_default(&auto, &without, "cuda", 1, true), None, "a recipe without the key");
    assert_eq!(mtp_default(&auto, &with, "hip", 1, true), None, "another backend unchanged");
    assert_eq!(mtp_default(&auto, &with, "cuda", 2, true), None, "tp 2 is mtp_default_tp_k's");
    assert_eq!(tp_mtp_default(&auto, &with, "cuda", 2, true), None, "the one-rank key does not turn tp on");
    assert_eq!(mtp_default(&auto, &tp_only, "cuda", 1, true), None, "the tp key does not turn one rank on");
    let explicit = intarsia_models::EngineOptions { spec_auto: false, ..Default::default() };
    assert_eq!(mtp_default(&explicit, &with, "cuda", 1, true), None, "--spec given (or a library caller)");
    let batch = intarsia_models::EngineOptions { spec_auto: true, max_batch: Some(4), ..Default::default() };
    assert_eq!(mtp_default(&batch, &with, "cuda", 1, true), None, "--max-batch 4");
    let one = intarsia_models::EngineOptions { spec_auto: true, max_batch: Some(1), ..Default::default() };
    assert_eq!(mtp_default(&one, &with, "cuda", 1, true), Some(2), "--max-batch 1 keeps it");
    let dfl = intarsia_models::EngineOptions { spec_auto: true, spec: SpecConfig { mode: SpecMode::Dflash, k: 7, k_min: None }, ..Default::default() };
    assert_eq!(mtp_default(&dfl, &with, "cuda", 1, true), None, "another --spec");
    // a lone `--spec-k N` (no `--spec`) keeps the default and sets only its draft length
    let lone = intarsia_models::EngineOptions { spec_auto: true, spec_k_given: true, spec: SpecConfig { mode: SpecMode::Off, k: 3, k_min: None }, ..Default::default() };
    assert_eq!(mtp_default(&lone, &with, "cuda", 1, true), Some(3), "a lone --spec-k 3 sets k");
    assert_eq!(mtp_default(&lone, &without, "cuda", 1, true), None, "a lone --spec-k where no default applies stays off");
    assert_eq!(mtp_default(&lone, &with, "hip", 1, true), None, "a lone --spec-k on another backend stays off");
    // a lone `--spec-k 0` is no draft length: the default is not applied with k = 0 (the loader refuses it outright, next test)
    let lone0 = intarsia_models::EngineOptions { spec_auto: true, spec_k_given: true, spec: SpecConfig { mode: SpecMode::Off, k: 0, k_min: None }, ..Default::default() };
    assert_eq!(mtp_default(&lone0, &with, "cuda", 1, true), None, "a lone --spec-k 0 is not SpecConfig {{ Mtp, k: 0 }}");
    let zero = intarsia_models::parse_recipe(&format!("{base}\n[tuning.cuda]\nmtp_default_k = 0\n")).unwrap();
    assert_eq!(mtp_default(&auto, &zero, "cuda", 1, true), None, "k = 0 is unset");
    let stored = intarsia_models::EngineOptions {
        spec_auto: true,
        prefix_store: intarsia_core::prefix_store::PrefixStoreConfig::parse("on").unwrap(),
        ..Default::default()
    };
    assert_eq!(mtp_default(&stored, &with, "cuda", 1, true), None, "--prefix-store on: no default at cuda tp 1");
    let lone_stored = intarsia_models::EngineOptions { spec_k_given: true, spec: SpecConfig { mode: SpecMode::Off, k: 3, k_min: None }, ..stored.clone() };
    assert_eq!(mtp_default(&lone_stored, &with, "cuda", 1, true), None, "a lone --spec-k does not bring it back");
    std::env::set_var("INTARSIA_PREFIX_STORE", "on");
    let env_store = mtp_default(&auto, &with, "cuda", 1, true);
    std::env::remove_var("INTARSIA_PREFIX_STORE");
    assert_eq!(env_store, None, "INTARSIA_PREFIX_STORE=on: no default");
    assert_eq!(mtp_default(&auto, &with, "cuda", 1, true), Some(2), "and back on once it is unset");
    single_rank_loader_default_matches_off();
    std::env::set_var("INTARSIA_SPEC_DEFAULT", "0");
    let off = mtp_default(&auto, &with, "cuda", 1, true);
    std::env::remove_var("INTARSIA_SPEC_DEFAULT");
    assert_eq!(off, None, "INTARSIA_SPEC_DEFAULT=0");
}

/// G0 for the shipped recipes: exactly these (recipe, backend) pairs make `--spec mtp` the default, so every other model and backend
/// (the 8B, CUDA and Metal, the 35B on HIP and CPU) runs without --spec as before.
#[test]
fn only_the_measured_recipe_backend_pairs_default_to_mtp() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../recipes");
    let mut got = Vec::new();
    for e in std::fs::read_dir(&dir).unwrap() {
        let path = e.unwrap().path();
        if path.extension().is_none_or(|x| x != "toml") {
            continue;
        }
        let recipe = intarsia_models::parse_recipe(&std::fs::read_to_string(&path).unwrap()).unwrap();
        for backend in ["cpu", "cuda", "hip", "metal"] {
            if let Some(k) = recipe.resolve_mtp_default_tp(backend) {
                got.push(format!("{} {backend} tp {k}", recipe.model.family));
            }
            if let Some(k) = recipe.resolve_mtp_default(backend) {
                got.push(format!("{} {backend} tp1 {k}", recipe.model.family));
            }
        }
    }
    got.sort();
    assert_eq!(got, ["qwen3_5_moe cuda tp1 2", "qwen3_5_moe metal tp1 2", "qwen3_5_vl cuda tp1 2", "qwen3_5_vl hip tp 4", "qwen3_5_vl hip tp1 4", "qwen3_5_vl metal tp1 2"], "the recipes that turn MTP on by default");
}

fn single_rank_loader_default_matches_off() {
    let dir = model_dir();
    let recipe = dir.join("qwen3_5_mtp_default_k.toml");
    std::fs::write(&recipe, format!("{}\n[tuning.cpu]\nmtp_default_k = 2\n", include_str!("../../../recipes/qwen3_5.toml"))).unwrap();
    let load = |auto: bool, dir: &Path| {
        let o = intarsia_models::EngineOptions {
            load: LoadOptions { max_seq: 256, act_dtype: Some(DType::F32), kv_dtype: Some(DType::F32), ..Default::default() },
            recipe: Some(recipe.clone()),
            spec_auto: auto,
            ..Default::default()
        };
        intarsia_models::load_engine(dir, CpuBackend::arc(), &o).unwrap()
    };
    let (off, on) = (load(false, &dir), load(true, &dir));
    for p in PROMPTS {
        assert_eq!(run(&*off.engine, req(p, 40)), run(&*on.engine, req(p, 40)), "one-rank default-on vs off, prompt {p:?}");
    }
    assert!(on.engine.telemetry().spec_steps > 0, "the one-rank default actually ran --spec mtp");
    assert_eq!(off.engine.telemetry().spec_steps, 0);
    // with a prefix store the default is skipped: plain decode, same ids
    let stored = intarsia_models::EngineOptions {
        load: LoadOptions { max_seq: 256, act_dtype: Some(DType::F32), kv_dtype: Some(DType::F32), ..Default::default() },
        recipe: Some(recipe.clone()),
        spec_auto: true,
        prefix_store: intarsia_core::prefix_store::PrefixStoreConfig::parse("on").unwrap(),
        ..Default::default()
    };
    let st = intarsia_models::load_engine(&dir, CpuBackend::arc(), &stored).unwrap();
    assert_eq!(run(&*off.engine, req(PROMPTS[0], 40)), run(&*st.engine, req(PROMPTS[0], 40)));
    assert_eq!(st.engine.telemetry().spec_steps, 0, "a prefix store: the default is not applied");
    let bare = Path::new(env!("CARGO_TARGET_TMPDIR")).join("tp_mtp_model_bare_k");
    std::fs::create_dir_all(&bare).unwrap();
    for f in ["config.json", "tokenizer.json"] {
        std::fs::copy(dir.join(f), bare.join(f)).unwrap();
    }
    std::fs::copy(fixture_dir().join("model.safetensors"), bare.join("model.safetensors")).unwrap();
    let b = load(true, &bare);
    assert_eq!(run(&*b.engine, req(PROMPTS[0], 20)), run(&*off.engine, req(PROMPTS[0], 20)));
    assert_eq!(b.engine.telemetry().spec_steps, 0, "no head: the default is not applied");
}

/// The default through the engine loader on the cpu group (a recipe file with `[tuning.cpu] mtp_default_tp_k = 2`): it loads the head,
/// verify passes run, and its greedy ids equal --spec off's. A checkpoint without the head's weights falls back to plain decode.
fn loader_default_matches_off() {
    let dir = model_dir();
    let recipe = dir.join("qwen3_5_mtp_default.toml");
    std::fs::write(&recipe, format!("{}\n[tuning.cpu]\nmtp_default_tp_k = 2\n", include_str!("../../../recipes/qwen3_5.toml"))).unwrap();
    let load = |auto: bool, k_given: bool, dir: &Path| {
        let backends: Vec<Arc<dyn intarsia_core::backend::Backend>> = (0..2).map(|_| CpuBackend::arc() as Arc<dyn intarsia_core::backend::Backend>).collect();
        let comms = intarsia_core::HostCommGroup::new(backends.clone());
        let o = intarsia_models::EngineOptions {
            load: LoadOptions { max_seq: 256, act_dtype: Some(DType::F32), kv_dtype: Some(DType::F32), ..Default::default() },
            recipe: Some(recipe.clone()),
            spec_auto: auto,
            spec_k_given: k_given,
            spec: SpecConfig { mode: SpecMode::Off, k: if k_given { 3 } else { SpecConfig::default().k }, k_min: None },
            ..Default::default()
        };
        intarsia_models::load_engine_tp(dir, backends, comms, &o).unwrap()
    };
    let (off, on) = (load(false, false, &dir), load(true, false, &dir));
    for p in PROMPTS {
        assert_eq!(run(&*off.engine, req(p, 40)), run(&*on.engine, req(p, 40)), "default-on vs off, prompt {p:?}");
    }
    assert!(on.engine.telemetry().spec_steps > 0, "the default actually ran --spec mtp");
    // a lone --spec-k 3 keeps the default (verify passes run, ids equal off); with INTARSIA_SPEC_DEFAULT=0 it stays off
    let lone = load(true, true, &dir);
    for p in PROMPTS {
        assert_eq!(run(&*off.engine, req(p, 40)), run(&*lone.engine, req(p, 40)), "lone --spec-k 3 vs off, prompt {p:?}");
    }
    assert!(lone.engine.telemetry().spec_steps > 0, "a lone --spec-k ran the default --spec mtp");
    std::env::set_var("INTARSIA_SPEC_DEFAULT", "0");
    let lone_off = load(true, true, &dir);
    std::env::remove_var("INTARSIA_SPEC_DEFAULT");
    assert_eq!(run(&*lone_off.engine, req(PROMPTS[0], 20)), run(&*off.engine, req(PROMPTS[0], 20)));
    assert_eq!(lone_off.engine.telemetry().spec_steps, 0, "INTARSIA_SPEC_DEFAULT=0 wins over a lone --spec-k");
    // the same recipe on a checkpoint without the head: no error, plain decode
    let bare = Path::new(env!("CARGO_TARGET_TMPDIR")).join("tp_mtp_model_bare");
    std::fs::create_dir_all(&bare).unwrap();
    for f in ["config.json", "tokenizer.json"] {
        std::fs::copy(dir.join(f), bare.join(f)).unwrap();
    }
    std::fs::copy(fixture_dir().join("model.safetensors"), bare.join("model.safetensors")).unwrap();
    let b = load(true, false, &bare);
    assert_eq!(run(&*b.engine, req(PROMPTS[0], 20)), run(&*off.engine, req(PROMPTS[0], 20)));
    assert_eq!(b.engine.telemetry().spec_steps, 0, "no head: the default is not applied");
}

/// A failed MTP pass aborts the communicator (`Model::mtp_draft_tp`, `Model::mtp_catch_up`): the group is dead from then on,
/// and the engine learns it from the model, whatever the error that ended the pass (an allocation failure on rank 0 here).
#[test]
fn a_failed_mtp_pass_marks_the_group_aborted() {
    let (recipe, dims) = spec();
    for catch_up in [false, true] {
        let be = Flaky::new();
        let backends: Vec<Arc<dyn intarsia_core::backend::Backend>> = vec![be.clone(), CpuBackend::arc()];
        let comms = intarsia_core::HostCommGroup::new(backends.clone());
        let tel = Arc::new(Telemetry::new());
        let m = Model::load_tp(backends, comms, &recipe, &dims, &weights(), &opts(Some(3), false, 40), tel).unwrap();
        let m = finish(m, Some(3));
        let mut st = m.new_state().unwrap();
        st.mtp.as_mut().expect("MTP state").active = true;
        let p = PROMPTS[0];
        let next = match m.forward(&mut st, p, HeadMode::LastArgmax, None).unwrap() {
            StepOut::Token(t) => t,
            o => panic!("prefill gave {o:?}"),
        };
        // the head's first draft syncs its cache; a verify pass queues its rows without running the head over them
        m.mtp_draft(&mut st, next, 3).unwrap();
        let next = *m.forward_verify(&mut st, &[next, 5, 6]).unwrap().last().unwrap();
        assert!(!m.tp_group_aborted());
        assert!(st.mtp.as_ref().unwrap().pending_rows >= 2, "rows are queued for the catch-up");
        be.set(FAIL_MEMORY);
        let err = if catch_up { m.mtp_catch_up(&mut st).unwrap_err() } else { m.mtp_draft(&mut st, next, 3).unwrap_err() };
        be.set(OK);
        assert!(matches!(err, Error::Memory(_)), "catch_up {catch_up}: {err}");
        assert!(m.tp_group_aborted(), "catch_up {catch_up}: the failed pass aborted the group and the model does not know");
    }
}

/// Batched MTP default (lane bm, user decision 2026-11-05 ~19:45 EST "N=1 default"): `[tuning.<backend>] batched_mtp` gives N at tp = 1
/// in-process only; `INTARSIA_BATCHED_MTP` wins (`0` off); a batched server takes the recipe's `mtp_default_k` only where N > 0. The
/// shipped recipes set it for exactly the measured pairs (the 35B on CUDA and Metal): HIP and every tp > 1 stay off.
#[test]
fn batched_mtp_default_only_where_measured_and_env_wins() {
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    use intarsia_models::load::{batched_mtp_for, mtp_default, tp_mtp_default};
    let moe = intarsia_models::parse_recipe(include_str!("../../../recipes/qwen3_5_moe.toml")).unwrap();
    std::env::remove_var("INTARSIA_BATCHED_MTP");
    assert_eq!(batched_mtp_for(&moe, "cuda", 1, true).unwrap(), 1, "35B cuda tp1");
    assert_eq!(batched_mtp_for(&moe, "metal", 1, true).unwrap(), 1, "35B metal tp1");
    assert_eq!(batched_mtp_for(&moe, "hip", 1, true).unwrap(), 0, "35B hip stays off");
    assert_eq!(batched_mtp_for(&moe, "cuda", 2, true).unwrap(), 0, "tp 2 stays off");
    assert_eq!(batched_mtp_for(&moe, "cuda", 1, false).unwrap(), 0, "a rank in another process stays off");
    let batch = intarsia_models::EngineOptions { spec_auto: true, max_batch: Some(16), ..Default::default() };
    assert_eq!(mtp_default(&batch, &moe, "cuda", 1, true), Some(2), "a --max-batch 16 server on CUDA takes the MTP default");
    assert_eq!(mtp_default(&batch, &moe, "hip", 1, true), None, "not on HIP");
    let vl = intarsia_models::parse_recipe(include_str!("../../../recipes/qwen3_5_vl.toml")).unwrap();
    assert_eq!(tp_mtp_default(&batch, &vl, "hip", 2, true), None, "HIP tp 2 batched server: unchanged, off");
    assert_eq!(mtp_default(&batch, &vl, "metal", 1, true), None, "a recipe without batched_mtp keeps its MTP default single-sequence");
    std::env::set_var("INTARSIA_BATCHED_MTP", "0");
    let off = (batched_mtp_for(&moe, "cuda", 1, true).unwrap(), mtp_default(&batch, &moe, "cuda", 1, true));
    std::env::set_var("INTARSIA_BATCHED_MTP", "3");
    let three = (batched_mtp_for(&moe, "hip", 1, true).unwrap(), batched_mtp_for(&moe, "cuda", 2, true).unwrap());
    std::env::remove_var("INTARSIA_BATCHED_MTP");
    assert_eq!(off, (0, None), "INTARSIA_BATCHED_MTP=0 turns it and the batched MTP default off");
    assert_eq!(three, (3, 0), "the env var sets N on any backend at tp 1, never at tp 2");
    // the shipped recipes: exactly the measured pairs
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../recipes");
    let mut got = Vec::new();
    for e in std::fs::read_dir(&dir).unwrap() {
        let path = e.unwrap().path();
        if path.extension().is_none_or(|x| x != "toml") {
            continue;
        }
        let r = intarsia_models::parse_recipe(&std::fs::read_to_string(&path).unwrap()).unwrap();
        for backend in ["cpu", "cuda", "hip", "metal"] {
            if let Some(n) = r.resolve_batched_mtp(backend) {
                got.push(format!("{} {backend} {n}", r.model.family));
            }
        }
    }
    got.sort();
    assert_eq!(got, ["qwen3_5_moe cuda 1", "qwen3_5_moe metal 1"], "the recipes that turn batched MTP on by default");
}
