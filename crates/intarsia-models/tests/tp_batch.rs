//! Continuous-batching decode under in-process tensor parallelism (`Model::forward_batch` with `--tp 2`), on the cpu backend
//! with the host-staged communicator: every row of a batched step at tp=2 must be BIT FOR BIT the solo (one-sequence) step of
//! that sequence at tp=2 (logits and argmax ids), for the dense tiny model and the tiny hybrid (GatedDeltaNet + gated
//! attention + NVFP4 MoE) model, 2..=8 rows at staggered context lengths, with the vocab-parallel head on and off (and the two
//! heads give the same bits), next to the fused all-reduce + rmsnorm, and through the batched engine with prefix reuse.
//!
//! The step protocol between rank 0 and its peer is pinned by mutants (`Model::set_tp_batch_mutant`): a peer that runs one
//! row fewer, its rows in another order, a row at a stale position, skips a commit or drops a collective must make the
//! comparison fail or the step error, never pass and never hang (the communicator waits a few seconds at most here).

mod common;
use common::arnorm_comm::ArNormComm;
use common::ByteCodec;
use intarsia_backend_cpu::CpuBackend;
use intarsia_core::backend::Backend;
use intarsia_core::comm::{Comm, HostCommGroup};
use intarsia_core::dtype::f32_to_fp8_e4m3;
use intarsia_core::model::{HeadMode, LoadOptions, Model, StepOut, TpBatchMutant};
use intarsia_core::quant::{nvfp4, Fp8Act};
use intarsia_core::recipe::{ModelDims, Recipe};
use intarsia_core::rng::SplitMix64;
use intarsia_core::state::SequenceState;
use intarsia_core::weights::{MemWeights, WeightSource};
use intarsia_core::{
    BatchedEngine, BatchedOptions, DType, Engine, EngineMeta, GenEvent, GenRequest, HostTensor, LocalEngine, Telemetry,
};
use intarsia_models::SafeTensors;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    /// `tests/common`: the dense Qwen3-style tiny model (vocab 300)
    Dense,
    /// `tests/fixtures/tiny_qwen3_5` under the `qwen3_5_moe` recipe with fp8 projections and NVFP4 experts (vocab 96)
    Hybrid,
}

// ---- the tiny hybrid (as tests/tp_hybrid.rs builds it) ----

const E: usize = 8;
const TOP_K: usize = 2;
const MI: usize = 32;
const MH: usize = 64;
const HY_LAYERS: usize = 6;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_qwen3_5")
}

fn put_nvfp4(w: &mut MemWeights, rng: &mut SplitMix64, prefix: &str, n: usize, k: usize) {
    let q = nvfp4::random_rows(rng, n, k);
    w.insert(&format!("{prefix}.weight"), q.parts[0].clone());
    w.insert(&format!("{prefix}.weight_scale"), q.parts[1].clone());
    w.insert(&format!("{prefix}.weight_scale_2"), q.parts[2].clone());
}

fn to_fp8(w: &mut MemWeights, prefix: &str) {
    let name = format!("{prefix}.weight");
    let info = w.info(&name).unwrap();
    let t = HostTensor::new(info.dtype, info.shape.clone(), w.bytes(&name).unwrap().to_vec()).unwrap();
    let v = t.to_f32();
    let amax = v.iter().fold(0f32, |m, x| m.max(x.abs()));
    let ws = if amax > 0.0 { amax / 448.0 } else { 1.0 };
    let data: Vec<u8> = v.iter().map(|x| f32_to_fp8_e4m3(x / ws)).collect();
    w.insert(&name, HostTensor::new(DType::Fp8E4m3, info.shape, data).unwrap());
    w.insert(&format!("{prefix}.weight_scale"), HostTensor::from_f32(&[], &[ws]));
}

fn hybrid_weights() -> MemWeights {
    let src = SafeTensors::open_dir(&fixture_dir()).unwrap();
    let mut w = MemWeights::new();
    for name in src.names() {
        if name.contains(".mlp.") {
            continue;
        }
        let info = src.info(&name).unwrap();
        let t = HostTensor::new(info.dtype, info.shape.clone(), src.bytes(&name).unwrap().to_vec()).unwrap();
        let renamed = if let Some(rest) = name.strip_prefix("model.") { format!("model.language_model.{rest}") } else { name.clone() };
        w.insert(&renamed, t);
    }
    let mut rng = SplitMix64::new(0x7b_35);
    let bf16 = |rng: &mut SplitMix64, shape: &[usize]| HostTensor::from_f32_as(DType::Bf16, shape, &rng.normal_vec(shape.iter().product(), 0.3));
    for l in 0..HY_LAYERS {
        let p = format!("model.language_model.layers.{l}");
        let attn = [".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj", ".self_attn.o_proj"];
        let gdn = [".linear_attn.in_proj_qkv", ".linear_attn.in_proj_z", ".linear_attn.out_proj"];
        for s in attn.iter().chain(&gdn) {
            if w.info(&format!("{p}{s}.weight")).is_some() {
                to_fp8(&mut w, &format!("{p}{s}"));
            }
        }
        let p = format!("{p}.mlp");
        w.insert(&format!("{p}.gate.weight"), bf16(&mut rng, &[E, MH]));
        w.insert(&format!("{p}.shared_expert_gate.weight"), bf16(&mut rng, &[1, MH]));
        for x in 0..E {
            put_nvfp4(&mut w, &mut rng, &format!("{p}.experts.{x}.gate_proj"), MI, MH);
            put_nvfp4(&mut w, &mut rng, &format!("{p}.experts.{x}.up_proj"), MI, MH);
            put_nvfp4(&mut w, &mut rng, &format!("{p}.experts.{x}.down_proj"), MH, MI);
        }
        put_nvfp4(&mut w, &mut rng, &format!("{p}.shared_expert.gate_proj"), MI, MH);
        put_nvfp4(&mut w, &mut rng, &format!("{p}.shared_expert.up_proj"), MI, MH);
        put_nvfp4(&mut w, &mut rng, &format!("{p}.shared_expert.down_proj"), MH, MI);
    }
    w
}

fn hybrid_spec() -> (Recipe, ModelDims) {
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/qwen3_5_moe.toml")).unwrap();
    let mut cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture_dir().join("config.json")).unwrap()).unwrap();
    cfg["architectures"] = serde_json::json!(["Qwen3_5MoeForCausalLM"]);
    cfg["num_experts"] = E.into();
    cfg["num_experts_per_tok"] = TOP_K.into();
    cfg["moe_intermediate_size"] = MI.into();
    cfg["shared_expert_intermediate_size"] = MI.into();
    cfg["norm_topk_prob"] = true.into();
    cfg.as_object_mut().unwrap().remove("intermediate_size");
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    (recipe, dims)
}

// ---- loading ----

const CHUNK: usize = 8;
const MAX_SEQ: usize = 256;
const ROWS: usize = 8;

#[derive(Clone, Copy)]
struct Setup {
    kind: Kind,
    world: usize,
    vocab: bool,
    arnorm: bool,
    rows: usize,
    /// The host communicator's wait for the other rank (mutant tests: seconds, so a dropped collective fails fast).
    timeout: Option<Duration>,
}

fn setup(kind: Kind) -> Setup {
    Setup { kind, world: 2, vocab: false, arnorm: false, rows: ROWS, timeout: None }
}

fn vocab_of(kind: Kind) -> u32 {
    match kind {
        Kind::Dense => common::V as u32,
        Kind::Hybrid => 96,
    }
}

fn load(s: Setup, tel: Arc<Telemetry>) -> Model {
    let (recipe, dims, weights): (Recipe, ModelDims, MemWeights) = match s.kind {
        Kind::Dense => (common::recipe(), common::dims(), common::tiny_weights(DType::F32, 7).weights),
        Kind::Hybrid => {
            let (r, d) = hybrid_spec();
            (r, d, hybrid_weights())
        }
    };
    let opts = LoadOptions {
        max_seq: MAX_SEQ,
        mem_budget: Some(4 << 30),
        prefill_chunk: CHUNK,
        act_dtype: Some(DType::F32),
        kv_dtype: Some(DType::F32),
        fp8_act: (s.kind == Kind::Hybrid).then_some(Fp8Act::None),
        tp_vocab: s.vocab,
        tp_fusion: s.arnorm,
        tp_arnorm: s.arnorm,
        ..Default::default()
    };
    let backends: Vec<Arc<dyn Backend>> = (0..s.world).map(|_| CpuBackend::arc()).collect();
    let mut m = if s.world == 1 {
        Model::load(backends[0].clone(), &recipe, &dims, &weights, &opts, tel).unwrap()
    } else {
        let comms: Vec<Arc<dyn Comm>> = if s.arnorm {
            ArNormComm::group(backends.clone(), usize::MAX).0
        } else if let Some(t) = s.timeout {
            HostCommGroup::with_timeout(backends.clone(), t)
        } else {
            HostCommGroup::new(backends.clone())
        };
        Model::load_tp(backends, comms, &recipe, &dims, &weights, &opts, tel).unwrap()
    };
    assert_eq!(m.vocab_parallel(), s.vocab && s.world > 1, "the head split must be what the test asks for");
    if s.kind == Kind::Hybrid {
        m.allow_recurrent_batch().unwrap();
    }
    if s.rows > 1 {
        m.enable_batch(s.rows).unwrap();
    }
    m.configure_plan(&Default::default(), None).unwrap();
    m
}

fn model(s: Setup) -> Model {
    load(s, Arc::new(Telemetry::new()))
}

// ---- the batched-vs-solo harness ----

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn host_argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best as u32
}

fn prompt(i: usize, len: usize, vocab: u32) -> Vec<u32> {
    let mut rng = SplitMix64::new(0x7b_ba7c + i as u64);
    (0..len).map(|_| 1 + (rng.next_u64() % (vocab as u64 - 2)) as u32).collect()
}

struct Row {
    solo: SequenceState,
    batch: SequenceState,
    tok: u32,
}

/// Prefill one state pair per entry of `lens` (distinct prompts, so a row mix-up shows), then `steps` batched steps of all
/// rows against the solo steps of each row. Row `r` asks for the logits on even `r + step` and for the argmax otherwise.
/// Returns every batched row's output as bits (logits rows, and the ids), or the first difference / error.
fn run(m: &Model, lens: &[usize], steps: usize) -> Result<Vec<Vec<u32>>, String> {
    let vocab = m.dims.vocab as u32;
    let mut rows: Vec<Row> = Vec::new();
    for (i, &len) in lens.iter().enumerate() {
        let p = prompt(i, len, vocab);
        let mut solo = m.new_state().map_err(|e| e.to_string())?;
        let mut batch = m.new_state().map_err(|e| e.to_string())?;
        let a = m.forward(&mut solo, &p, HeadMode::LastArgmax, None).map_err(|e| e.to_string())?;
        let b = m.forward(&mut batch, &p, HeadMode::LastArgmax, None).map_err(|e| e.to_string())?;
        let (StepOut::Token(a), StepOut::Token(b)) = (a, b) else { return Err("prefill gave no token".into()) };
        if a != b {
            return Err(format!("row {i}: the two prefills differ"));
        }
        rows.push(Row { solo, batch, tok: a });
    }
    let mut out = Vec::new();
    for step in 0..steps {
        let modes: Vec<HeadMode> =
            (0..rows.len()).map(|r| if (r + step) % 2 == 0 { HeadMode::LastLogits } else { HeadMode::LastArgmax }).collect();
        let ids: Vec<u32> = rows.iter().map(|r| r.tok).collect();
        let got = {
            let mut refs: Vec<&mut SequenceState> = rows.iter_mut().map(|r| &mut r.batch).collect();
            m.forward_batch(&mut refs, &ids, &modes).map_err(|e| format!("step {step}: forward_batch: {e}"))?
        };
        let mut toks = Vec::new();
        for (r, row) in rows.iter_mut().enumerate() {
            let want = m.forward(&mut row.solo, &[row.tok], modes[r], None).map_err(|e| format!("step {step} row {r}: solo: {e}"))?;
            let next = match (&got[r], &want) {
                (StepOut::Logits(g), StepOut::Logits(w)) => {
                    if bits(g) != bits(w) {
                        let i = g.iter().zip(w).position(|(a, b)| a.to_bits() != b.to_bits()).unwrap_or(0);
                        return Err(format!("step {step} row {r}: logits differ from the solo step (first at {i}: {} vs {})", g[i], w[i]));
                    }
                    out.push(bits(g));
                    host_argmax(g)
                }
                (StepOut::Token(g), StepOut::Token(w)) => {
                    if g != w {
                        return Err(format!("step {step} row {r}: argmax {g} vs solo {w}"));
                    }
                    *g
                }
                (g, w) => return Err(format!("step {step} row {r}: {g:?} vs {w:?}")),
            };
            if row.batch.len != row.solo.len {
                return Err(format!("step {step} row {r}: batched state at {} tokens, solo at {}", row.batch.len, row.solo.len));
            }
            toks.push(next);
            row.tok = next;
        }
        out.push(toks);
    }
    Ok(out)
}

/// Staggered context lengths (several prefill chunks, 1-token prompts, different per row).
fn lens(b: usize) -> Vec<usize> {
    [3usize, 17, 9, 1, 26, 12, 5, 33][..b].to_vec()
}

const STEPS: usize = 10;

/// Every batch size 2..=8 on fresh states; returns all batched outputs (for the cross-model comparisons).
fn check_all_sizes(m: &Model, what: &str) -> Vec<Vec<u32>> {
    let mut all = Vec::new();
    for b in 2..=ROWS {
        match run(m, &lens(b), STEPS) {
            Ok(o) => all.extend(o),
            Err(e) => panic!("{what}, {b} rows: {e}"),
        }
    }
    all
}

#[test]
fn tp2_batched_dense_is_bitwise_the_tp2_solo_step_with_and_without_the_vocab_split() {
    let s = setup(Kind::Dense);
    let split = model(Setup { vocab: true, ..s });
    let replicated = model(s);
    let a = check_all_sizes(&split, "dense, vocab-parallel head");
    let b = check_all_sizes(&replicated, "dense, replicated head");
    assert_eq!(a.len(), b.len());
    assert!(a == b, "batched steps: the vocab-parallel head is not bitwise the replicated head");
    // the batched steps took the batched path on both ranks, and the peer ran its head slice and gather
    assert!(split.rank_model(1).unwrap().batch_rows() == ROWS);
}

#[test]
fn tp2_batched_hybrid_is_bitwise_the_tp2_solo_step_with_and_without_the_vocab_split() {
    let s = setup(Kind::Hybrid);
    let split = model(Setup { vocab: true, ..s });
    let replicated = model(s);
    let a = check_all_sizes(&split, "hybrid, vocab-parallel head");
    let b = check_all_sizes(&replicated, "hybrid, replicated head");
    assert!(a == b, "batched hybrid steps: the vocab-parallel head is not bitwise the replicated head");
}

#[test]
fn tp2_batched_next_to_the_fused_all_reduce_rmsnorm_is_bitwise_solo() {
    for kind in [Kind::Dense, Kind::Hybrid] {
        let m = model(Setup { arnorm: true, ..setup(kind) });
        for b in [2usize, 5, 8] {
            run(&m, &lens(b), 6).unwrap_or_else(|e| panic!("{kind:?} with tp_arnorm, {b} rows: {e}"));
        }
    }
}

#[test]
fn tp2_batch_memory_plan_is_agreed_by_both_ranks() {
    for vocab in [false, true] {
        let m = model(Setup { vocab, ..setup(Kind::Hybrid) });
        let p = m.rank_model(1).unwrap();
        assert_eq!(p.batch_rows(), m.batch_rows());
        assert_eq!(p.plan.kv_cap_tokens, m.plan.kv_cap_tokens, "every rank holds the same context per sequence");
        assert!(m.plan.kv_cap_tokens >= MAX_SEQ.min(m.plan.kv_cap_tokens));
        // a fresh state's peer sub-state has the rank-0 capacity and the peer's recurrent snapshots
        let st = m.new_state().unwrap();
        assert_eq!(st.peers.len(), 1);
        assert_eq!(st.peers[0].st.cap, st.cap);
    }
}

#[test]
fn tp1_batched_is_unchanged() {
    // the tp=1 batched path (no peers) still gives the solo bits, for both models
    for kind in [Kind::Dense, Kind::Hybrid] {
        let m = model(Setup { world: 1, ..setup(kind) });
        for b in [2usize, 4, 8] {
            run(&m, &lens(b), STEPS).unwrap_or_else(|e| panic!("{kind:?} tp1, {b} rows: {e}"));
        }
    }
}

#[test]
fn tp2_batched_step_issues_the_solo_steps_collectives_once_with_b_rows_of_payload() {
    for (kind, vocab) in [(Kind::Dense, false), (Kind::Dense, true), (Kind::Hybrid, true)] {
        let m = model(Setup { vocab, ..setup(kind) });
        let v = m.dims.vocab as u32;
        let mut sts: Vec<SequenceState> = (0..5).map(|_| m.new_state().unwrap()).collect();
        for (i, st) in sts.iter_mut().enumerate() {
            m.forward(st, &prompt(i, 3 + 4 * i, v), HeadMode::LastArgmax, None).unwrap();
        }
        let c0 = m.comm_stats().unwrap();
        m.forward(&mut sts[0], &[1], HeadMode::LastArgmax, None).unwrap();
        let c1 = m.comm_stats().unwrap();
        let mut refs: Vec<&mut SequenceState> = sts.iter_mut().collect();
        m.forward_batch(&mut refs, &[2, 3, 4, 5, 6], &[HeadMode::LastArgmax; 5]).unwrap();
        let c2 = m.comm_stats().unwrap();
        let (solo, batch) = (c1.calls - c0.calls, c2.calls - c1.calls);
        assert!(solo > 0);
        assert_eq!(batch, solo, "{kind:?} vocab {vocab}: one batched step makes the solo step's collectives, once");
        assert_eq!(c2.bytes - c1.bytes, 5 * (c1.bytes - c0.bytes), "{kind:?} vocab {vocab}: with 5 rows of payload each");
        let p = m.rank_model(1).unwrap().comm_stats().unwrap();
        assert_eq!(p.calls, c2.calls, "both ranks issued the same collectives");
    }
}

#[test]
fn tp2_batch_refuses_a_state_of_another_model_per_row() {
    // a tp=1 state into a tp=2 batch is rejected by the per-row check, before any rank starts
    let m2 = model(setup(Kind::Dense));
    let m1 = model(Setup { world: 1, ..setup(Kind::Dense) });
    let foreign = m1.new_state().unwrap();
    let e = m2.check_batch_row(&foreign, 1, HeadMode::LastArgmax).unwrap_err().to_string();
    assert!(e.contains("does not belong"), "{e}");
}

// ---- the step protocol: mutants ----

/// A mutant must make the batched steps fail (an error, or bits that differ from the solo step), never pass.
fn mutant_is_caught(kind: Kind, vocab: bool, mutant: TpBatchMutant) {
    let m = model(Setup { vocab, timeout: Some(Duration::from_secs(3)), ..setup(kind) });
    // control: the same harness passes without the mutant (so a caught mutant is not a broken harness)
    run(&m, &lens(4), 3).unwrap_or_else(|e| panic!("{kind:?} control run failed: {e}"));
    m.set_tp_batch_mutant(1, Some(mutant)).unwrap();
    let r = run(&m, &lens(4), 3);
    assert!(r.is_err(), "{kind:?} (vocab split {vocab}): mutant {mutant:?} was not caught");
    eprintln!("{kind:?} vocab={vocab} {mutant:?} caught: {}", r.unwrap_err());
}

#[test]
fn tp2_batch_peer_running_fewer_rows_is_caught() {
    mutant_is_caught(Kind::Dense, false, TpBatchMutant::DropRow);
    mutant_is_caught(Kind::Hybrid, true, TpBatchMutant::DropRow);
}

#[test]
fn tp2_batch_peer_with_its_rows_in_another_order_is_caught() {
    mutant_is_caught(Kind::Dense, true, TpBatchMutant::ReverseRows);
    mutant_is_caught(Kind::Hybrid, false, TpBatchMutant::ReverseRows);
}

#[test]
fn tp2_batch_peer_at_a_stale_position_is_caught() {
    mutant_is_caught(Kind::Dense, false, TpBatchMutant::StalePosition);
    mutant_is_caught(Kind::Hybrid, true, TpBatchMutant::StalePosition);
}

#[test]
fn tp2_batch_peer_skipping_a_commit_is_caught() {
    mutant_is_caught(Kind::Dense, false, TpBatchMutant::SkipCommit);
    mutant_is_caught(Kind::Hybrid, false, TpBatchMutant::SkipCommit);
}

#[test]
fn tp2_batch_peer_dropping_a_collective_is_caught_without_hanging() {
    let t0 = std::time::Instant::now();
    mutant_is_caught(Kind::Dense, false, TpBatchMutant::DropCollective);
    mutant_is_caught(Kind::Hybrid, true, TpBatchMutant::DropCollective);
    assert!(t0.elapsed() < Duration::from_secs(120), "a dropped collective must fail within the communicator's wait");
}

// ---- the batched engine at tp=2 ----

fn collect(rx: &Receiver<GenEvent>) -> Result<(Vec<u32>, usize), String> {
    let mut ids = Vec::new();
    loop {
        match rx.recv_timeout(Duration::from_secs(300)).expect("engine stalled") {
            GenEvent::Token { id, .. } => ids.push(id),
            GenEvent::Done { completion_tokens, .. } => return Ok((ids, completion_tokens)),
            GenEvent::Error { message } => return Err(message),
        }
    }
}

fn meta() -> EngineMeta {
    EngineMeta { id: "tiny-tp2".into(), eos_ids: vec![], ..Default::default() }
}

fn cases(n: usize, vocab: u32) -> Vec<GenRequest> {
    let lens = [5usize, 17, 33, 9, 48, 16, 40, 3, 25];
    (0..n)
        .map(|i| {
            let mut r = GenRequest { prompt: prompt(100 + i, lens[i % lens.len()], vocab), max_tokens: 130, ignore_eos: true, ..Default::default() };
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
fn tp2_batched_engine_equals_the_tp2_solo_engine_over_128_tokens() {
    for (kind, vocab, reuse) in [(Kind::Hybrid, true, true), (Kind::Dense, false, true), (Kind::Hybrid, false, false)] {
        let v = vocab_of(kind);
        let tel = Arc::new(Telemetry::new());
        let solo = LocalEngine::new(load(Setup { vocab, rows: 1, ..setup(kind) }, tel.clone()), Arc::new(ByteCodec), meta(), tel).unwrap();
        let want: Vec<Vec<u32>> = cases(7, v)
            .into_iter()
            .map(|r| {
                // cold: a throw-away request first, so no prefix of the previous case is reused
                let flush = GenRequest { prompt: vec![v - 1, v - 2, v - 3], max_tokens: 1, ignore_eos: true, ..Default::default() };
                collect(&solo.submit(flush).unwrap()).unwrap();
                collect(&solo.submit(r).unwrap()).unwrap().0
            })
            .collect();
        let tel = Arc::new(Telemetry::new());
        let m = load(Setup { vocab, rows: 3, ..setup(kind) }, tel.clone());
        let mut o = BatchedOptions::new(3, 64);
        o.prefix_reuse = reuse;
        let eng = BatchedEngine::new(m, Arc::new(ByteCodec), meta(), tel, o).unwrap();
        let rxs: Vec<_> = cases(7, v).into_iter().map(|r| eng.submit(r).unwrap()).collect();
        for (i, (rx, w)) in rxs.iter().zip(&want).enumerate() {
            let (got, n) = collect(rx).unwrap();
            assert!(n >= 128);
            assert_eq!(&got, w, "case {i} ({kind:?}, vocab split {vocab}, reuse {reuse})");
        }
        let mx = eng.metrics();
        assert!(mx.batch_hist[2..].iter().sum::<u64>() > 0, "no batched decode pass ran: {:?}", mx.batch_hist);
    }
}
