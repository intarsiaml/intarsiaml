//! The captured decode step (CUDA graph) against the eager path, on the tiny hybrid model (GatedDeltaNet
//! layers with in-place recurrent state + gated attention with a KV cache).
//!
//! `Rec` wraps the cpu backend and implements graph capture the way a GPU does it: while capturing, launches
//! are *recorded, not executed* (and any host I/O or synchronisation is an error, because a real capture would
//! reject it); replay re-launches the recording. Position-dependent ops carry their device-side position
//! (`OpCall::pos`), which the cpu backend resolves when the launch runs, so a step that still baked `pos0`
//! into its recording would diverge from the eager path after the first replay, and one that ran the step
//! eagerly before capturing would apply the recurrent update twice.

use intarsia_core::backend::{Backend, Buffer, DeviceInfo, GraphHandle, GraphImpl, MemKind, StreamId};
use intarsia_core::model::{HeadMode, LoadOptions, Model, StepOut};
use intarsia_core::recipe::ModelDims;
use intarsia_core::state::SequenceState;
use intarsia_core::{DType, Error, KernelImpl, Op, OpCall, Result, Telemetry};
use intarsia_models::SafeTensors;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

struct Recording(Vec<(String, OpCall)>);
impl GraphImpl for Recording {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

struct Rec {
    inner: Arc<dyn Backend>,
    recording: Mutex<Option<Vec<(String, OpCall)>>>,
    graphs: bool,
    fail_capture: bool,
    captures: AtomicUsize,
    replays: AtomicUsize,
    /// launches that actually executed
    executed: AtomicUsize,
    /// (op, carries a device position) of the last recording
    last: Mutex<Vec<(Op, bool)>>,
}

impl Rec {
    fn new(graphs: bool, fail_capture: bool) -> Arc<Rec> {
        Arc::new(Rec {
            inner: intarsia_backend_cpu::CpuBackend::arc(),
            recording: Mutex::new(None),
            graphs,
            fail_capture,
            captures: AtomicUsize::new(0),
            replays: AtomicUsize::new(0),
            executed: AtomicUsize::new(0),
            last: Mutex::new(Vec::new()),
        })
    }
    fn capturing(&self) -> bool {
        self.recording.lock().unwrap().is_some()
    }
    fn forbid(&self, what: &str) -> Result<()> {
        if self.capturing() {
            return Err(Error::backend(format!("{what} while capturing: a graph capture would reject it")));
        }
        Ok(())
    }
}

impl Backend for Rec {
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
        self.forbid("copy")?;
        self.inner.copy(d, d_o, s, s_o, n, st)
    }
    fn kernels(&self) -> Vec<KernelImpl> {
        self.inner.kernels()
    }
    fn launch(&self, kernel: &str, call: &OpCall, stream: StreamId) -> Result<()> {
        if let Some(rec) = self.recording.lock().unwrap().as_mut() {
            call.check()?;
            rec.push((kernel.to_string(), call.clone()));
            return Ok(());
        }
        self.executed.fetch_add(1, Ordering::Relaxed);
        self.inner.launch(kernel, call, stream)
    }
    fn synchronize(&self, s: StreamId) -> Result<()> {
        self.forbid("synchronize")?;
        self.inner.synchronize(s)
    }
    fn supports_graphs(&self) -> bool {
        self.graphs
    }
    fn graph_capture(&self, _stream: StreamId, body: &mut dyn FnMut() -> Result<()>) -> Result<GraphHandle> {
        if self.fail_capture {
            return Err(Error::backend("capture refused (test)"));
        }
        *self.recording.lock().unwrap() = Some(Vec::new());
        let ran = body();
        let rec = self.recording.lock().unwrap().take().unwrap();
        ran?;
        *self.last.lock().unwrap() = rec.iter().map(|(_, c)| (c.op, c.pos.is_some())).collect();
        self.captures.fetch_add(1, Ordering::Relaxed);
        Ok(GraphHandle::new(Recording(rec)))
    }
    fn graph_replay(&self, g: &GraphHandle, stream: StreamId) -> Result<()> {
        let r = g.downcast::<Recording>().expect("our graph");
        self.replays.fetch_add(1, Ordering::Relaxed);
        for (kernel, call) in &r.0 {
            self.inner.launch(kernel, call, stream)?;
        }
        Ok(())
    }
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_qwen3_5")
}

fn load(backend: Arc<dyn Backend>, graphs: bool, profile_every: u32) -> Model {
    let dir = fixture_dir();
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/qwen3_5.toml")).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    let src = SafeTensors::open_dir(&dir).unwrap();
    Model::load(
        backend,
        &recipe,
        &dims,
        &src,
        &LoadOptions {
            max_seq: 64,
            prefill_chunk: 8,
            act_dtype: Some(DType::F32),
            kv_dtype: Some(DType::F32),
            profile_every,
            graphs,
            ..Default::default()
        },
        Arc::new(Telemetry::new()),
    )
    .unwrap()
}

const PROMPT: [u32; 11] = [3, 17, 5, 9, 31, 2, 8, 44, 7, 13, 21];

/// Greedy decode `n` tokens after `PROMPT`; every step through `LastLogits` (so the logits are comparable bit for
/// bit) and its argmax checked against `LastArgmax` on a second state.
fn decode(m: &Model, st: &mut SequenceState, first: u32, n: usize) -> (Vec<u32>, Vec<Vec<f32>>) {
    let (mut toks, mut rows) = (vec![], vec![]);
    let mut cur = first;
    for _ in 0..n {
        let StepOut::Logits(l) = m.forward(st, &[cur], HeadMode::LastLogits, None).unwrap() else { panic!() };
        cur = l.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 as u32;
        toks.push(cur);
        rows.push(l);
    }
    (toks, rows)
}

fn start(m: &Model) -> (SequenceState, u32) {
    let mut st = m.new_state().unwrap();
    let StepOut::Token(t) = m.forward(&mut st, &PROMPT, HeadMode::LastArgmax, None).unwrap() else { panic!() };
    (st, t)
}

#[test]
fn graph_decode_is_bitwise_the_eager_decode() {
    let eager = load(Rec::new(false, false), true, 0);
    assert!(!eager.graph_stats().enabled, "a backend without graphs never uses them");
    let rec = Rec::new(true, false);
    let graphed = load(rec.clone(), true, 0);
    assert!(graphed.graph_stats().enabled);
    let (mut se, fe) = start(&eager);
    let (mut sg, fg) = start(&graphed);
    assert_eq!(fe, fg);
    let n = 40; // 11 + 1 + 40 < 64
    let (te, le) = decode(&eager, &mut se, fe, n);
    let (tg, lg) = decode(&graphed, &mut sg, fg, n);
    assert_eq!(te, tg, "greedy tokens");
    for (i, (a, b)) in le.iter().zip(&lg).enumerate() {
        assert!(a == b, "logits of step {i} differ between eager and graph decode");
    }
    assert_eq!(rec.captures.load(Ordering::Relaxed), 1, "one capture for one sequence state");
    assert_eq!(rec.replays.load(Ordering::Relaxed), n);
    assert_eq!(graphed.graph_stats().replays, n as u64);
    assert_eq!(se.len, sg.len);
    // the recording is the whole step: one embedding, both heads, no host work
    // (the recording itself is checked by `forbid` above: any upload / download / sync inside it would have failed)
}

#[test]
fn argmax_head_matches_through_the_graph() {
    let eager = load(Rec::new(false, false), true, 0);
    let graphed = load(Rec::new(true, false), true, 0);
    let (mut se, mut ce) = start(&eager);
    let (mut sg, mut cg) = start(&graphed);
    for step in 0..30 {
        let StepOut::Token(a) = eager.forward(&mut se, &[ce], HeadMode::LastArgmax, None).unwrap() else { panic!() };
        let StepOut::Token(b) = graphed.forward(&mut sg, &[cg], HeadMode::LastArgmax, None).unwrap() else { panic!() };
        assert_eq!(a, b, "step {step}");
        ce = a;
        cg = b;
    }
}

#[test]
fn profiled_steps_interleave_with_replays() {
    // profile_every 1 -> every 8th decode step runs eagerly with per-op timing, the others replay
    let rec = Rec::new(true, false);
    let graphed = load(rec.clone(), true, 1);
    let eager = load(Rec::new(false, false), true, 0);
    let (mut se, fe) = start(&eager);
    let (mut sg, fg) = start(&graphed);
    let (te, le) = decode(&eager, &mut se, fe, 40);
    let (tg, lg) = decode(&graphed, &mut sg, fg, 40);
    assert_eq!(te, tg);
    assert!(le == lg);
    let replays = rec.replays.load(Ordering::Relaxed);
    assert!(replays > 0 && replays < 40, "some steps replay, the profiled ones do not (replays = {replays})");
}

#[test]
fn a_new_state_gets_its_own_recording_and_truncate_keeps_replaying() {
    let rec = Rec::new(true, false);
    let m = load(rec.clone(), true, 0);
    let (mut s1, f1) = start(&m);
    let (t1, _) = decode(&m, &mut s1, f1, 10);
    assert_eq!(rec.captures.load(Ordering::Relaxed), 1);
    // a second sequence state: its buffers are different, so the recording of the first must not be replayed
    let (mut s2, f2) = start(&m);
    assert_eq!(f1, f2);
    let (t2, _) = decode(&m, &mut s2, f2, 10);
    assert_eq!(t1, t2);
    assert_eq!(rec.captures.load(Ordering::Relaxed), 2);
    // reset the state in place and decode again: same buffers, same recording, same tokens
    s2.reset(&*m.backend).unwrap();
    let StepOut::Token(f3) = m.forward(&mut s2, &PROMPT, HeadMode::LastArgmax, None).unwrap() else { panic!() };
    let (t3, _) = decode(&m, &mut s2, f3, 10);
    assert_eq!(t1, t3);
    assert_eq!(rec.captures.load(Ordering::Relaxed), 2);
}

#[test]
fn no_graphs_flag_and_failed_capture_fall_back_to_eager() {
    let rec = Rec::new(true, false);
    let m = load(rec.clone(), false, 0); // --no-graphs
    assert!(!m.graph_stats().enabled);
    let (mut st, f) = start(&m);
    let (t_off, _) = decode(&m, &mut st, f, 12);
    assert_eq!(rec.replays.load(Ordering::Relaxed), 0);
    assert_eq!(rec.captures.load(Ordering::Relaxed), 0);

    // toggled at run time
    let m2 = load(Rec::new(true, false), true, 0);
    m2.set_graphs(false);
    let (mut s2, f2) = start(&m2);
    assert_eq!(decode(&m2, &mut s2, f2, 12).0, t_off);

    // a backend whose capture fails: decoding continues eagerly with the right tokens
    let bad = Rec::new(true, true);
    let m3 = load(bad.clone(), true, 0);
    let (mut s3, f3) = start(&m3);
    assert_eq!(decode(&m3, &mut s3, f3, 12).0, t_off);
    assert!(!m3.graph_stats().enabled, "a failed capture switches graphs off");
    assert_eq!(bad.replays.load(Ordering::Relaxed), 0);
}

#[test]
fn the_recording_contains_the_position_dependent_ops_with_a_device_position() {
    let rec = Rec::new(true, false);
    let m = load(rec.clone(), true, 0);
    let (mut st, f) = start(&m);
    decode(&m, &mut st, f, 2);
    let last = rec.last.lock().unwrap();
    let count = |op: Op| last.iter().filter(|(o, _)| *o == op).count();
    // 2 full-attention layers, one decode attention each. The cpu backend registers the fused `QkNormRopeKv`, which takes
    // over q and k rope and the k and v appends of a layer (a backend without it runs 4 rope + 4 kv_append here).
    assert_eq!((count(Op::Rope), count(Op::KvAppend), count(Op::QkNormRopeKv), count(Op::AttnDecode)), (0, 0, 2, 2));
    assert_eq!(count(Op::AttnPrefill), 0);
    for (op, has_pos) in last.iter() {
        let needs = matches!(op, Op::Rope | Op::KvAppend | Op::QkNormRopeKv | Op::AttnDecode);
        assert_eq!(*has_pos, needs, "{op}: device position expected exactly on rope / kv_append / qk_norm_rope_kv / attn_decode");
    }
    assert_eq!(count(Op::Embedding), 1);
    assert_eq!(count(Op::Argmax), 1);
    let snap = m.graph_stats();
    assert_eq!((snap.captures, snap.replays), (1, 2));
}

/// Prefill `prompt[pos..]` the way the engine does (`generate_one`): chunks (balanced for an attention-only state), a snapshot after each, a snapshot one
/// token before the end, then the last token through `LastLogits` (a single-token step, so it takes the graph path).
/// Returns that last step's logits.
fn prefill_like_the_engine(m: &Model, st: &mut SequenceState, prompt: &[u32], mut pos: usize) -> Vec<f32> {
    let max_chunk = m.prefill_chunk();
    let balance = !st.has_recurrent();
    while prompt.len() - pos > max_chunk {
        let chunk = if balance { intarsia_core::engine::balanced_chunk_len(prompt.len() - pos, max_chunk) } else { max_chunk };
        m.forward(st, &prompt[pos..pos + chunk], HeadMode::None, None).unwrap();
        pos += chunk;
        st.snapshot(&*m.backend, intarsia_core::state::RECURRENT_SNAPSHOTS).unwrap();
    }
    if prompt.len() - pos > 1 {
        m.forward(st, &prompt[pos..prompt.len() - 1], HeadMode::None, None).unwrap();
        pos = prompt.len() - 1;
        st.snapshot(&*m.backend, intarsia_core::state::RECURRENT_SNAPSHOTS).unwrap();
    }
    let StepOut::Logits(l) = m.forward(st, &prompt[pos..], HeadMode::LastLogits, None).unwrap() else { panic!() };
    l
}

fn argmax(l: &[f32]) -> u32 {
    l.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 as u32
}

/// What one scenario run produced (for comparing graph, eager and from-scratch runs bit for bit).
struct Run {
    /// token count held when the restore happened, and the length restored to
    len_before_restore: usize,
    restored_len: usize,
    /// first-token logits and decoded tokens / logits after the (re)prefill of prompt B
    first_logits: Vec<f32>,
    tokens: Vec<u32>,
    rows: Vec<Vec<f32>>,
}

const PROMPT_A: [u32; 20] = [3, 17, 5, 9, 31, 2, 8, 44, 7, 13, 21, 4, 6, 12, 19, 23, 27, 30, 11, 15];
/// Shares the first 16 tokens (two prefill chunks) with `PROMPT_A`, then differs.
const PROMPT_B: [u32; 22] = [3, 17, 5, 9, 31, 2, 8, 44, 7, 13, 21, 4, 6, 12, 19, 23, 40, 41, 42, 43, 10, 25];
const DECODE_A: usize = 6;
const DECODE_B: usize = 24;

/// Serve A (prefill + a few decode steps, which captures the graph), then B through the engine's prefix-reuse path:
/// roll the sequence back to the recurrent snapshot at the shared prefix, prefill B's suffix, decode.
fn serve_a_then_b_with_a_restore(m: &Model) -> Run {
    let mut st = m.new_state().unwrap();
    let l = prefill_like_the_engine(m, &mut st, &PROMPT_A, 0);
    decode(m, &mut st, argmax(&l), DECODE_A);
    let len_before_restore = st.len;
    // `generate_one`: reuse what the state can return to exactly, then truncate
    let common = st.tokens.iter().zip(PROMPT_B.iter()).take_while(|(a, b)| a == b).count();
    let p = st.restorable_len(common.min(PROMPT_B.len() - 1));
    st.truncate(p, &*m.backend).unwrap();
    assert_eq!(st.len, p);
    let first_logits = prefill_like_the_engine(m, &mut st, &PROMPT_B, p);
    let (tokens, rows) = decode(m, &mut st, argmax(&first_logits), DECODE_B);
    Run { len_before_restore, restored_len: p, first_logits, tokens, rows }
}

#[test]
fn replay_after_a_mid_sequence_gdn_restore_is_bitwise_the_eager_decode() {
    // The GPU part (the real 35B, 256-step greedy decode with graphs against `--no-graphs`, after a prefix-reuse
    // restore on a server) is covered by the acceptance run, scripts/phase1b-acceptance.sh stage "graphs".
    // This test covers the restore-with-graphs mechanics on the tiny hybrid model: the restore copies the snapshot
    // into the same recurrent buffers the captured graph reads and writes, so the capture must stay valid.
    let eager = load(Rec::new(false, false), true, 0);
    let rec = Rec::new(true, false);
    let graphed = load(rec.clone(), true, 0);
    assert!(graphed.graph_stats().enabled && !eager.graph_stats().enabled);

    let g = serve_a_then_b_with_a_restore(&graphed);
    let e = serve_a_then_b_with_a_restore(&eager);

    // the restore really happened: back to the shared two-chunk prefix, strictly inside the held sequence
    assert_eq!(g.restored_len, 16, "restored to the snapshot at the end of the second prefill chunk");
    assert!(g.restored_len > 0 && g.restored_len < g.len_before_restore, "{} -> {}", g.len_before_restore, g.restored_len);
    assert_eq!(g.len_before_restore, PROMPT_A.len() + DECODE_A);
    assert_eq!((e.len_before_restore, e.restored_len), (g.len_before_restore, g.restored_len));

    // one capture for the whole run: the restore did not invalidate or recreate the recording
    let stats = graphed.graph_stats();
    let single_token_steps = (1 + DECODE_A) + (1 + DECODE_B); // the last prompt token goes through the graph too
    assert_eq!(stats.captures, 1, "no recapture after the restore");
    assert_eq!(stats.replays, single_token_steps as u64, "every single-token step replayed, before and after the restore");
    assert_eq!(rec.captures.load(Ordering::Relaxed), 1);
    assert_eq!(rec.replays.load(Ordering::Relaxed), single_token_steps);

    // graph == eager, bit for bit, across the restore
    assert!(g.first_logits == e.first_logits, "logits of the first step after the restore differ");
    assert_eq!(g.tokens, e.tokens, "greedy tokens after the restore");
    assert_eq!(g.rows.len(), DECODE_B);
    for (i, (a, b)) in g.rows.iter().zip(&e.rows).enumerate() {
        assert!(a == b, "logits of post-restore step {i} differ between graph and eager decode");
    }

    // and the restore itself is right: B prefilled from scratch (no restore) on the eager model gives the same bits
    let mut st = eager.new_state().unwrap();
    let l = prefill_like_the_engine(&eager, &mut st, &PROMPT_B, 0);
    assert!(l == g.first_logits, "restored prefill differs from a from-scratch prefill");
    let (t, r) = decode(&eager, &mut st, argmax(&l), DECODE_B);
    assert_eq!(t, g.tokens);
    assert!(r == g.rows);
}
