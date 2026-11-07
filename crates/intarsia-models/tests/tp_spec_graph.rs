//! The captured speculative MTP step (`Model::spec_step`) at tensor parallelism with in-process ranks (lane gt2,
//! `INTARSIA_SPEC_GRAPH_TP=1`), on the cpu backend: the dense tiny hybrid target of `tests/tp_mtp.rs` (4 GatedDeltaNet and 2
//! gated full-attention layers, a random MTP head sharded across the ranks) at tp=2.
//!
//! The ranks run on a cpu backend that records graphs the way a GPU does ([`RecCpu`]: while capturing, launches and copies are
//! recorded, host transfers are errors) and a communicator whose collectives are recorded too ([`RecComm`]: the host
//! rendezvous runs at replay, on each rank's own thread), so the step really runs as one replay per rank, the drafts reaching
//! the peers through the recorded id gather.
//!
//! * One step's drafts and verify outputs at tp=2 equal tp=1's eager draft and verify pass (replicated and vocab-parallel head).
//! * Greedy output with the captured step on equals `--spec off`, with the head's own drafts and with scripted drafts that hit
//!   every acceptance pattern through the per-rank GatedDeltaNet rollback, at k = 1..3, for odd `max_tokens`, and the step
//!   replayed (rank 0's replays need the peers' replays: the recorded gathers rendezvous).
//!
//! The head whole on rank 0 (an MoE target, lane mt2) has no tiny fixture: that path is checked on the 35B (HIP ids gate in
//! docs/results/2026-11-07-spec-graph-tp2.md).

mod common;
use common::ByteCodec;
use intarsia_backend_cpu::CpuBackend;
use intarsia_core::backend::{Backend, Buffer, DeviceInfo, GraphHandle, GraphImpl, MemKind, StreamId, TensorRef};
use intarsia_core::model::{DraftScript, HeadMode, LoadOptions, Model, StepOut};
use intarsia_core::recipe::ModelDims;
use intarsia_core::rng::SplitMix64;
use intarsia_core::spec::{SpecConfig, SpecMode};
use intarsia_core::weights::{MemWeights, WeightSource};
use intarsia_core::{
    Comm, CommStats, DType, Engine, EngineMeta, Error, FinishReason, GenEvent, GenRequest, HostCommGroup, HostTensor, KernelImpl,
    LocalEngine, OpCall, Result, Telemetry,
};
use intarsia_models::SafeTensors;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const H: usize = 64;
const I: usize = 96;
const NH: usize = 4;
const NKV: usize = 2;
const HD: usize = 32;
const V: usize = 96;

fn knob_on() {
    std::env::set_var("INTARSIA_SPEC_GRAPH_TP", "1");
}

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

/// The fixture's weights plus the random MTP head of `tests/tp_mtp.rs`.
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

fn opts(k: Option<usize>, vocab_par: bool) -> LoadOptions {
    LoadOptions {
        max_seq: 512,
        prefill_chunk: 8,
        act_dtype: Some(DType::F32),
        kv_dtype: Some(DType::F32),
        profile_every: 0,
        tp_vocab: vocab_par,
        mtp: k.is_some(),
        mtp_k: k.unwrap_or(0),
        mtp_draft_vocab: Some(40),
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

fn model_tp1(k: Option<usize>) -> Model {
    let (recipe, dims) = spec();
    let m = Model::load(CpuBackend::arc(), &recipe, &dims, &weights(), &opts(k, false), Arc::new(Telemetry::new())).unwrap();
    finish(m, k)
}

/// tp=2; `captured`: on [`RecCpu`] ranks over [`RecComm`] (graphs usable, the captured step runs), else plain cpu ranks over the
/// host communicator (not capturable: every pass eager).
fn model_tp2(k: Option<usize>, vocab_par: bool, captured: bool, tel: Arc<Telemetry>) -> (Model, Vec<Arc<RecCpu>>) {
    let (recipe, dims) = spec();
    let recs: Vec<Arc<RecCpu>> = if captured { (0..2).map(|_| RecCpu::arc()).collect() } else { Vec::new() };
    let (backends, comms): (Vec<Arc<dyn Backend>>, Vec<Arc<dyn Comm>>) = if captured {
        let inner: Vec<Arc<dyn Backend>> = recs.iter().map(|r| r.inner.clone()).collect();
        let host = HostCommGroup::new(inner);
        let comms = host.into_iter().zip(&recs).map(|(c, r)| Arc::new(RecComm { inner: c, rec: r.clone() }) as Arc<dyn Comm>).collect();
        (recs.iter().map(|r| r.clone() as Arc<dyn Backend>).collect(), comms)
    } else {
        let b: Vec<Arc<dyn Backend>> = (0..2).map(|_| CpuBackend::arc()).collect();
        (b.clone(), HostCommGroup::new(b))
    };
    let m = Model::load_tp(backends, comms, &recipe, &dims, &weights(), &opts(k, vocab_par), tel).unwrap();
    assert_eq!(m.vocab_parallel(), vocab_par, "the head split as asked");
    (finish(m, k), recs)
}

const PROMPTS: [&[u32]; 3] = [&[3, 9, 27, 81, 50, 7, 7, 1, 60, 2, 33, 33, 12], &[90, 1, 2, 3, 4, 5], &[44; 21]];

fn token(o: StepOut) -> u32 {
    match o {
        StepOut::Token(t) => t,
        o => panic!("expected a token, got {o:?}"),
    }
}

/// Captured steps at tp=2 draft and verify what tp=1's eager draft and verify passes give, after the prompt (one queued head
/// row) and again after a pass whose rows were all kept (the head catches up over k + 1 rows), with the replicated and the
/// vocab-parallel head.
#[test]
fn captured_tp2_step_equals_tp1_eager_draft_and_verify() {
    knob_on();
    for k in [1usize, 2, 3] {
        let m1 = model_tp1(Some(k));
        for vocab_par in [false, true] {
            let (m2, recs) = model_tp2(Some(k), vocab_par, true, Arc::new(Telemetry::new()));
            for p in PROMPTS {
                let (mut s1, mut s2) = (m1.new_state().unwrap(), m2.new_state().unwrap());
                let mut n1 = token(m1.forward(&mut s1, p, HeadMode::LastArgmax, None).unwrap());
                let mut n2 = token(m2.forward(&mut s2, p, HeadMode::LastArgmax, None).unwrap());
                for round in 0..3 {
                    assert_eq!(n1, n2, "k {k} vocab_par {vocab_par} round {round}: target tokens");
                    let d1 = m1.mtp_draft(&mut s1, n1, k).unwrap();
                    let mut ids = vec![n1];
                    ids.extend(&d1);
                    let o1 = m1.forward_verify(&mut s1, &ids).unwrap();
                    assert!(m2.spec_step_ok(&s2, n2, k), "k {k} vocab_par {vocab_par} round {round}: the captured step applies at tp=2");
                    let pass = m2.spec_step(&mut s2, n2, k).unwrap().expect("captured step ran");
                    eprintln!("k {k} vocab_par {vocab_par} round {round} prompt {p:?}: tp1 drafts {d1:?} out {o1:?}, tp2 drafts {:?} out {:?}", pass.drafts, pass.out);
                    assert_eq!(pass.drafts, d1, "k {k} vocab_par {vocab_par} round {round}: drafts");
                    assert_eq!(pass.out, o1, "k {k} vocab_par {vocab_par} round {round}: verify outputs");
                    assert_eq!(s1.len, s2.len);
                    // every row kept (no rollback: the engine test covers it); the next step's head catches up over all q rows
                    n1 = *o1.last().unwrap();
                    n2 = *pass.out.last().unwrap();
                }
            }
            let rp: Vec<usize> = recs.iter().map(|r| r.replays.load(std::sync::atomic::Ordering::Relaxed)).collect();
            assert!(rp.iter().all(|&r| r > 0), "both ranks replayed: {rp:?}");
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

fn engine(k: Option<usize>, vocab_par: bool, captured: bool, script: Option<DraftScript>) -> (LocalEngine, Arc<Telemetry>) {
    let tel = Arc::new(Telemetry::new());
    let (m, _) = model_tp2(k, vocab_par, captured, tel.clone());
    if script.is_some() {
        m.mtp_script(script).unwrap();
    }
    let spec = match k {
        Some(k) => SpecConfig { mode: SpecMode::Mtp, k },
        None => SpecConfig::default(),
    };
    let e = LocalEngine::new(m, Arc::new(ByteCodec), EngineMeta { id: "tiny-mtp-tp2-graph".into(), eos_ids: vec![], spec, ..Default::default() }, tel.clone())
        .unwrap();
    (e, tel)
}

/// Drafts from the true continuation with one wrong id at a position that depends on the context (or none), as `tests/tp_mtp.rs`.
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

/// Greedy output with the captured step at tp=2 equals `--spec off`, and the replayed steps draft and accept exactly as the
/// eager tp=2 passes do (same counts), with the head's own drafts and scripted ones.
#[test]
fn captured_tp2_step_keeps_greedy_ids_equal_spec_off() {
    knob_on();
    for vocab_par in [true, false] {
        let truth: Arc<Mutex<Vec<Vec<u32>>>> = Arc::new(Mutex::new(Vec::new()));
        let (off, _) = engine(None, vocab_par, false, None);
        for p in PROMPTS {
            let o = run(&off, req(p, 100));
            let mut s = p.to_vec();
            s.extend(&o.ids);
            truth.lock().unwrap().push(s);
        }
        for (k, script) in [(1usize, false), (2, false), (3, false), (2, true), (3, true)] {
            let (on, tel) = engine(Some(k), vocab_par, true, script.then(|| scripted(truth.clone(), 11)));
            let (eager, etel) = engine(Some(k), vocab_par, false, script.then(|| scripted(truth.clone(), 11)));
            for p in PROMPTS {
                let o = run(&off, req(p, 90));
                assert_eq!(o, run(&on, req(p, 90)), "vocab_par {vocab_par} k {k} scripted {script} prompt {p:?}");
                assert_eq!(o, run(&eager, req(p, 90)), "eager: vocab_par {vocab_par} k {k} scripted {script} prompt {p:?}");
                for n in [1usize, 4, 7] {
                    let a = run(&off, req(p, n));
                    assert_eq!(a, run(&on, req(p, n)), "vocab_par {vocab_par} k {k} max_tokens {n}");
                    assert_eq!(a, run(&eager, req(p, n)), "eager: vocab_par {vocab_par} k {k} max_tokens {n}");
                    let mut next = p.to_vec();
                    next.extend(&a.ids);
                    next.push(11);
                    assert_eq!(run(&off, req(&next, 9)), run(&on, req(&next, 9)), "vocab_par {vocab_par} k {k} follow-up after {n}");
                    run(&eager, req(&next, 9));
                }
            }
            let (t, te) = (tel.snapshot(), etel.snapshot());
            let (captures, replays) = on.spec_graph_counts();
            eprintln!(
                "tp2 vocab_par {vocab_par} k {k} scripted {script}: {} passes, {} drafted, {} accepted; {captures} spec captures, {replays} spec replays",
                t.spec_steps, t.spec_proposed, t.spec_accepted
            );
            assert!(t.spec_steps > 0 && replays > 0, "the captured step ran at tp=2");
            assert_eq!(
                (t.spec_steps, t.spec_proposed, t.spec_accepted),
                (te.spec_steps, te.spec_proposed, te.spec_accepted),
                "vocab_par {vocab_par} k {k} scripted {script}: replayed and eager steps draft and accept alike"
            );
            if script {
                assert!(t.spec_accepted > 0 && t.spec_accepted < t.spec_proposed, "partial acceptance exercised");
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------
// a cpu backend and a communicator that record graphs
// ------------------------------------------------------------------------------------------------------------------

type CommStep = Arc<dyn Fn() -> Result<()> + Send + Sync>;

enum Step {
    Launch(String, OpCall),
    Copy(Buffer, usize, Buffer, usize, usize),
    Comm(CommStep),
}
struct Recording(Vec<Step>);
impl GraphImpl for Recording {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// The cpu backend with graph capture the way a GPU does it (as in `tests/mtp.rs`): while capturing, launches, device copies and
/// (through [`RecComm`]) collectives are recorded, not executed; allocation and host transfers are errors. It claims the HIP
/// verify-attention contract (`verify_attn_per_row`), so the step takes the path HIP takes.
struct RecCpu {
    inner: Arc<dyn Backend>,
    recording: Mutex<Option<Vec<Step>>>,
    replays: std::sync::atomic::AtomicUsize,
}
impl RecCpu {
    fn arc() -> Arc<RecCpu> {
        Arc::new(RecCpu { inner: CpuBackend::arc(), recording: Mutex::new(None), replays: Default::default() })
    }
    fn forbid(&self, what: &str) -> Result<()> {
        if self.recording.lock().unwrap().is_some() {
            return Err(Error::backend(format!("{what} while capturing: a graph capture would reject it")));
        }
        Ok(())
    }
    /// Record `f` if capturing (true), else leave it to the caller.
    fn record_comm(&self, f: CommStep) -> bool {
        if let Some(rec) = self.recording.lock().unwrap().as_mut() {
            rec.push(Step::Comm(f));
            return true;
        }
        false
    }
}
impl Backend for RecCpu {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn info(&self) -> DeviceInfo {
        self.inner.info()
    }
    fn alloc(&self, bytes: usize, kind: MemKind) -> Result<Buffer> {
        self.forbid("alloc")?;
        self.inner.alloc(bytes, kind)
    }
    fn upload(&self, dst: &Buffer, o: usize, src: &[u8], s: StreamId) -> Result<()> {
        self.forbid("upload")?;
        self.inner.upload(dst, o, src, s)
    }
    fn download(&self, src: &Buffer, o: usize, dst: &mut [u8], s: StreamId) -> Result<()> {
        self.forbid("download")?;
        self.inner.download(src, o, dst, s)
    }
    fn copy(&self, d: &Buffer, d_o: usize, s: &Buffer, s_o: usize, n: usize, st: StreamId) -> Result<()> {
        if let Some(rec) = self.recording.lock().unwrap().as_mut() {
            rec.push(Step::Copy(d.clone(), d_o, s.clone(), s_o, n));
            return Ok(());
        }
        self.inner.copy(d, d_o, s, s_o, n, st)
    }
    fn kernels(&self) -> Vec<KernelImpl> {
        self.inner.kernels()
    }
    fn launch(&self, kernel: &str, call: &OpCall, stream: StreamId) -> Result<()> {
        if let Some(rec) = self.recording.lock().unwrap().as_mut() {
            call.check()?;
            rec.push(Step::Launch(kernel.to_string(), call.clone()));
            return Ok(());
        }
        self.inner.launch(kernel, call, stream)
    }
    fn synchronize(&self, s: StreamId) -> Result<()> {
        self.forbid("synchronize")?;
        self.inner.synchronize(s)
    }
    fn supports_graphs(&self) -> bool {
        true
    }
    fn verify_attn_per_row(&self) -> bool {
        true
    }
    fn graph_capture(&self, _stream: StreamId, body: &mut dyn FnMut() -> Result<()>) -> Result<GraphHandle> {
        *self.recording.lock().unwrap() = Some(Vec::new());
        let ran = body();
        let rec = self.recording.lock().unwrap().take().unwrap();
        ran?;
        Ok(GraphHandle::new(Recording(rec)))
    }
    fn graph_replay(&self, g: &GraphHandle, stream: StreamId) -> Result<()> {
        let r = g.downcast::<Recording>().expect("our graph");
        self.replays.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        for step in &r.0 {
            match step {
                Step::Launch(k, c) => self.inner.launch(k, c, stream)?,
                Step::Copy(d, d_o, s, s_o, n) => self.inner.copy(d, *d_o, s, *s_o, *n, stream)?,
                Step::Comm(f) => f()?,
            }
        }
        Ok(())
    }
}

/// The host communicator with capturable collectives: while its rank's [`RecCpu`] records, a collective is recorded (the
/// rendezvous with the other rank happens when the recording is replayed, on each rank's own thread), else it runs now.
struct RecComm {
    inner: Arc<dyn Comm>,
    rec: Arc<RecCpu>,
}
impl RecComm {
    fn go(&self, f: CommStep) -> Result<()> {
        if self.rec.record_comm(f.clone()) {
            Ok(())
        } else {
            f()
        }
    }
}
impl Comm for RecComm {
    fn name(&self) -> &str {
        "rec-host"
    }
    fn rank(&self) -> usize {
        self.inner.rank()
    }
    fn world(&self) -> usize {
        self.inner.world()
    }
    fn all_reduce_sum(&self, x: &TensorRef, stream: StreamId) -> Result<()> {
        let (c, x) = (self.inner.clone(), x.clone());
        self.go(Arc::new(move || c.all_reduce_sum(&x, stream)))
    }
    fn all_reduce_sum_into(&self, partial: &TensorRef, out: &TensorRef, stream: StreamId) -> Result<()> {
        let (c, p, o) = (self.inner.clone(), partial.clone(), out.clone());
        self.go(Arc::new(move || c.all_reduce_sum_into(&p, &o, stream)))
    }
    fn all_gather(&self, send: &TensorRef, recv: &TensorRef, stream: StreamId) -> Result<()> {
        let (c, s, r) = (self.inner.clone(), send.clone(), recv.clone());
        self.go(Arc::new(move || c.all_gather(&s, &r, stream)))
    }
    fn all_gather_quantum(&self) -> Option<usize> {
        self.inner.all_gather_quantum()
    }
    fn abort(&self) {
        self.inner.abort()
    }
    fn stats(&self) -> CommStats {
        self.inner.stats()
    }
    fn note_timed(&self, d: Duration) {
        self.inner.note_timed(d)
    }
    fn capture_safe(&self) -> bool {
        true
    }
}
