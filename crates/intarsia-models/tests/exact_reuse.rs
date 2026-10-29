//! Exact prefix reuse (lane bi8, docs/results/2026-10-29-bi8-serve-batch-invariance.md): on a backend whose dense prefill
//! bits depend on a pass's row count (`Backend::prefix_reuse_any_cut` false: CUDA, and Metal since lane prb) a request resumes from a held KV prefix
//! only where its own cold chunk schedule cuts, and only over rows the held sequence wrote with exactly those passes, in the
//! single-sequence engine and in the batched engine's slot choice alike. The cpu kernels are chunk-invariant, so here the
//! bits agree either way: these tests pin the policy (which prefix is reused), the CUDA gates in `batch_cuda_real.rs` the bits.
//! Lane prc (docs/results/2026-10-29-prc-metal-cheap-reuse.md): a backend that also says passes of >= 2 rows agree
//! (`Backend::prefix_reuse_min_pass_rows`, Metal) resumes at any row of a prefix written by such passes, `Mode::Wide` below.
//!
//! Lane rip (docs/results/2026-10-29-rip-row-invariant-prefill.md): on a backend with row-invariant prefill
//! (`Backend::row_invariant_prefill`; CUDA with `INTARSIA_ROW_INVARIANT_PREFILL=1`) a request resumes at any row a row-invariant prefill pass wrote, never inside rows
//! a decode step wrote (`Mode::Invariant` below: the cpu backend with every kernel marked `row_invariant`).
mod common;
use common::*;
use intarsia_core::backend::{Backend, Buffer, DeviceInfo, MemKind, StreamId};
use intarsia_core::model::{LoadOptions, Model};
use intarsia_core::{
    BatchedEngine, BatchedOptions, DType, Engine, EngineMeta, GenEvent, GenRequest, KernelImpl, LocalEngine, Op, OpCall, Result,
    Telemetry,
};
use std::sync::Arc;
use std::time::Duration;

const CHUNK: usize = 8;

/// The cpu backend, reporting a chunk-sensitive prefill as CUDA did before lane rip (`inv` false), or a chunk-sensitive ordinary
/// prefill with row-invariant prefill passes, as CUDA with `INTARSIA_ROW_INVARIANT_PREFILL=1` (`inv` true: every cpu kernel is marked `row_invariant`; the cpu
/// kernels compute each row alone); with `min_rows` what Metal reports (passes of that many rows or more agree, a 1-row pass
/// is kept out of prefill chunking).
struct Strict {
    inner: Arc<dyn Backend>,
    inv: bool,
    /// with `inv`: leave the matmuls untagged, as a CUDA checkpoint whose GEMMs have no `row_invariant` kernel (fp8, NVFP4, MoE)
    partial: bool,
    min_rows: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    /// the cpu backend: any cut
    Any,
    /// chunk-sensitive, no row-invariant prefill: the BI8 ladder
    Ladder,
    /// chunk-sensitive with row-invariant prefill passes: any cut inside the rows those wrote
    Invariant,
    /// the backend asks for row-invariant prefill but an op of the pass (the matmul) has no `row_invariant` kernel: the plan
    /// switches it off and reuse falls back to the ladder (Gemma-4 / quantised checkpoints on CUDA)
    Partial,
    /// chunk-sensitive, passes of >= 2 rows agree (Metal, lane prc): any cut inside rows such passes wrote
    Wide,
}

impl Backend for Strict {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn info(&self) -> DeviceInfo {
        self.inner.info()
    }
    fn alloc(&self, bytes: usize, kind: MemKind) -> Result<Buffer> {
        self.inner.alloc(bytes, kind)
    }
    fn upload(&self, dst: &Buffer, o: usize, src: &[u8], s: StreamId) -> Result<()> {
        self.inner.upload(dst, o, src, s)
    }
    fn download(&self, src: &Buffer, o: usize, dst: &mut [u8], s: StreamId) -> Result<()> {
        self.inner.download(src, o, dst, s)
    }
    fn copy(&self, d: &Buffer, d_o: usize, s: &Buffer, s_o: usize, n: usize, st: StreamId) -> Result<()> {
        self.inner.copy(d, d_o, s, s_o, n, st)
    }
    fn kernels(&self) -> Vec<KernelImpl> {
        let mut v = self.inner.kernels();
        if self.inv {
            v = v.into_iter().map(|k| if self.partial && k.op == Op::Matmul { k } else { k.row_invariant() }).collect();
        }
        v
    }
    fn launch(&self, kernel: &str, call: &OpCall, stream: StreamId) -> Result<()> {
        self.inner.launch(kernel, call, stream)
    }
    fn synchronize(&self, s: StreamId) -> Result<()> {
        self.inner.synchronize(s)
    }
    fn prefix_reuse_any_cut(&self) -> bool {
        false
    }
    fn row_invariant_prefill(&self) -> bool {
        self.inv
    }
    fn avoid_single_row_prefill(&self) -> bool {
        self.min_rows.is_some()
    }
    fn prefix_reuse_min_pass_rows(&self) -> Option<usize> {
        self.min_rows
    }
}

fn model(mode: Mode, rows: usize) -> (Model, Arc<Telemetry>) {
    let cpu = intarsia_backend_cpu::CpuBackend::arc();
    let strict = |inv, partial, min_rows| -> Arc<dyn Backend> { Arc::new(Strict { inner: cpu.clone(), inv, partial, min_rows }) };
    let backend: Arc<dyn Backend> = match mode {
        Mode::Any => cpu.clone(),
        Mode::Ladder => strict(false, false, None),
        Mode::Invariant => strict(true, false, None),
        Mode::Partial => strict(true, true, None),
        Mode::Wide => strict(false, false, Some(2)),
    };
    let t = tiny_weights(DType::Bf16, 5);
    let tel = Arc::new(Telemetry::new());
    let mut m = Model::load(
        backend,
        &recipe(),
        &dims(),
        &t.weights,
        &LoadOptions { max_seq: 160, prefill_chunk: CHUNK, ..Default::default() },
        tel.clone(),
    )
    .unwrap();
    if rows > 1 {
        m.enable_batch(rows).unwrap();
    }
    m.configure_plan(&Default::default(), None).unwrap();
    assert_eq!(m.row_invariant_prefill(), mode == Mode::Invariant, "{mode:?}");
    (m, tel)
}

fn meta() -> EngineMeta {
    EngineMeta { id: "tiny".into(), eos_ids: vec![299], ..Default::default() }
}

fn req(prompt: &str) -> GenRequest {
    GenRequest { prompt: prompt.bytes().map(|b| b as u32).collect(), max_tokens: 6, ignore_eos: true, ..Default::default() }
}

/// (generated ids, cached prompt tokens)
fn collect(rx: &std::sync::mpsc::Receiver<GenEvent>) -> (Vec<u32>, usize) {
    let mut ids = vec![];
    loop {
        match rx.recv_timeout(Duration::from_secs(120)).expect("engine stalled") {
            GenEvent::Token { id, .. } => ids.push(id),
            GenEvent::Done { cached_tokens, .. } => return (ids, cached_tokens),
            GenEvent::Error { message } => panic!("request failed: {message}"),
        }
    }
}

// a 30-byte prompt: cold chunks (0, 8) (8, 16) (16, 24) (24, 30), so 8, 16 and 24 are the reuse points
const LONG: &str = "abcdefghijklmnopqrstuvwxyz0123";
// shares 20 bytes with LONG: the largest point it may resume from is 16
const SHARED20: &str = "abcdefghijklmnopqrstQQQQQQQQQQQQ";
// a short prompt (one cold chunk): never resumed
const SHORT: &str = "hey you";

/// The sequence of requests both engines run, with the prefix each must reuse on a chunk-sensitive backend and on cpu.
fn cases() -> Vec<(&'static str, usize, usize)> {
    vec![
        // (prompt, cached when strict, cached on cpu: any cut, plen - 1 for a repeat)
        (SHORT, 0, 0),
        (SHORT, 0, SHORT.len() - 1),
        (LONG, 0, 0),
        (LONG, 24, LONG.len() - 1),
        (SHARED20, 16, 20),
    ]
}

/// The prefix a request of `cases()` reuses in `mode` (row-invariant: the cpu figure, every reused row a prefill row).
fn want(mode: Mode, strict: usize, any: usize) -> usize {
    if matches!(mode, Mode::Ladder | Mode::Partial) { strict } else { any }
}

#[test]
fn single_sequence_engine_resumes_only_at_the_cold_schedule() {
    for mode in [Mode::Ladder, Mode::Any, Mode::Invariant, Mode::Partial] {
        let (m, tel) = model(mode, 1);
        let e = LocalEngine::new(m, Arc::new(ByteCodec), meta(), tel).unwrap();
        let cold = |p: &str| {
            let (m, tel) = model(mode, 1);
            collect(&LocalEngine::new(m, Arc::new(ByteCodec), meta(), tel).unwrap().submit(req(p)).unwrap()).0
        };
        for (p, want_strict, want_any) in cases() {
            let (ids, cached) = collect(&e.submit(req(p)).unwrap());
            assert_eq!(cached, want(mode, want_strict, want_any), "{p:?} {mode:?}");
            assert_eq!(ids, cold(p), "{p:?} {mode:?}: the warm request computes its cold output");
        }
        // a follow-up turn: LONG plus its generated tokens plus more. Rows 24..30 were the final chunk (6 rows), rows from 30
        // on were written by decode steps: only the 8-row ladder up to 24 matches the new prompt's own cold chunks
        let (first, _) = collect(&e.submit(req(LONG)).unwrap());
        let mut follow = req(LONG);
        follow.prompt.extend(&first);
        follow.prompt.extend("and more text here".bytes().map(|b| b as u32));
        let (_, cached) = collect(&e.submit(follow.clone()).unwrap());
        // the held sequence: the prompt and every generated token but the last (never fed back). Row-invariant: the whole
        // prompt (its rows were all prefill passes), not the decode-written rows after it
        let any_cut = LONG.len() + req(LONG).max_tokens - 1;
        let w = match mode {
            Mode::Ladder | Mode::Partial => 24,
            Mode::Any => any_cut,
            Mode::Invariant => LONG.len(),
            Mode::Wide => unreachable!(),
        };
        assert_eq!(cached, w, "follow-up turn {mode:?}");
        // the next turn's prompt holds the follow-up turn whole: every row of it a prefill row now
        let (second, _) = collect(&e.submit(follow.clone()).unwrap());
        let mut third = follow.clone();
        third.prompt.extend(&second);
        third.prompt.extend(" and the third turn".bytes().map(|b| b as u32));
        let (ids, cached) = collect(&e.submit(third.clone()).unwrap());
        if mode == Mode::Invariant {
            assert_eq!(cached, follow.prompt.len(), "third turn");
        }
        let (m2, tel2) = model(mode, 1);
        let cold3 = collect(&LocalEngine::new(m2, Arc::new(ByteCodec), meta(), tel2).unwrap().submit(third).unwrap()).0;
        assert_eq!(ids, cold3, "third turn {mode:?}: the warm request computes its cold output");
    }
}

#[test]
fn batched_engine_picks_slots_by_the_same_rule() {
    for mode in [Mode::Ladder, Mode::Any, Mode::Invariant, Mode::Partial] {
        let strict = matches!(mode, Mode::Ladder | Mode::Partial);
        let (m, tel) = model(mode, 2);
        let mut o = BatchedOptions::new(2, 16);
        o.prefix_reuse = true;
        let e = BatchedEngine::new(m, Arc::new(ByteCodec), meta(), tel, o).unwrap();
        // one at a time, so each request finds the previous ones' slots idle
        for (p, want_strict, want_any) in cases() {
            let (_, cached) = collect(&e.submit(req(p)).unwrap());
            if strict {
                assert_eq!(cached, want_strict, "{p:?} strict");
            } else {
                // two slots: the least recently used one may hold the other prompt, so at most the single-sequence figure
                assert!(cached <= want_any, "{p:?}: cached {cached} > {want_any}");
            }
        }
    }
}

/// Lane rip: the batched engine writes a prompt's last row in a prefill pass too, so with row-invariant prefill turn 2 of a chat
/// resumes from turn 1's whole prompt (not from turn 1's reply, which decode steps wrote), and computes its cold output.
#[test]
fn batched_engine_multi_turn_resumes_the_whole_prompt() {
    for mode in [Mode::Invariant, Mode::Ladder] {
        let fresh = || {
            let (m, tel) = model(mode, 2);
            let mut o = BatchedOptions::new(2, 16);
            o.prefix_reuse = true;
            BatchedEngine::new(m, Arc::new(ByteCodec), meta(), tel, o).unwrap()
        };
        let e = fresh();
        let (first, cached) = collect(&e.submit(req(LONG)).unwrap());
        assert_eq!(cached, 0);
        let mut follow = req(LONG);
        follow.prompt.extend(&first);
        follow.prompt.extend("and more text here".bytes().map(|b| b as u32));
        let (ids, cached) = collect(&e.submit(follow.clone()).unwrap());
        let w = if mode == Mode::Invariant { LONG.len() } else { 24 };
        assert_eq!(cached, w, "turn 2 {mode:?}");
        let (cold, c0) = collect(&fresh().submit(follow).unwrap());
        assert_eq!(c0, 0);
        assert_eq!(ids, cold, "turn 2 {mode:?}: the warm request computes its cold output");
    }
}

/// Lane prb (docs/results/2026-10-29-prb-prefix-reuse-probe.md): the cpu backend's prefill is row-count invariant, which is why
/// it keeps `prefix_reuse_any_cut` (default true): every output element is one fixed-order dot product of a row of activations
/// with a row of weights (and attention is per (row, head)), whatever the number of rows or the first row of a pass. Pinned bit
/// for bit: a 30-token prompt prefilled in one `forward` (chunks 8, 8, 8, 6) and split anywhere (k, 30 - k), the 1-row tail
/// (k = 29) included, gives the same logits for the last prompt row and for the decode steps after it, in bf16 and f32.
#[test]
fn cpu_prefill_is_bitwise_row_count_invariant() {
    use intarsia_core::model::{HeadMode, StepOut};
    let logits = |o: StepOut| match o {
        StepOut::Logits(l) => l,
        o => panic!("expected logits, got {o:?}"),
    };
    for dt in [DType::Bf16, DType::F32] {
        let t = tiny_weights(dt, 5);
        let m = Model::load(
            intarsia_backend_cpu::CpuBackend::arc(),
            &recipe(),
            &dims(),
            &t.weights,
            &LoadOptions { max_seq: 160, prefill_chunk: CHUNK, ..Default::default() },
            Arc::new(Telemetry::new()),
        )
        .unwrap();
        assert!(m.backend.prefix_reuse_any_cut());
        let p: Vec<u32> = LONG.bytes().map(|b| b as u32).collect();
        let steps = [7u32, 8, 9];
        let run = |k: usize| {
            let mut st = m.new_state().unwrap();
            let mut rows = vec![];
            if k == 0 {
                rows.push(logits(m.forward(&mut st, &p, HeadMode::LastLogits, None).unwrap()));
            } else {
                m.forward(&mut st, &p[..k], HeadMode::None, None).unwrap();
                rows.push(logits(m.forward(&mut st, &p[k..], HeadMode::LastLogits, None).unwrap()));
            }
            for &s in &steps {
                rows.push(logits(m.forward(&mut st, &[s], HeadMode::LastLogits, None).unwrap()));
            }
            rows
        };
        let cold = run(0);
        for k in 1..p.len() {
            assert!(run(k) == cold, "{dt:?}: split ({k}, {}) differs from the one-pass prefill", p.len() - k);
        }
    }
}

/// What `Wide` (Metal's report) resumes at, per request of `cases()`: a repeat cuts two rows before its end (the remainder is not
/// a 1-row pass), a shared prefix at the shared row.
const WIDE_CACHED: [usize; 5] = [0, SHORT.len() - 2, 0, LONG.len() - 2, 20];

#[test]
fn single_sequence_engine_resumes_at_any_row_of_wide_passes() {
    let (m, tel) = model(Mode::Wide, 1);
    let e = LocalEngine::new(m, Arc::new(ByteCodec), meta(), tel).unwrap();
    let cold = |p: &str| {
        let (m, tel) = model(Mode::Wide, 1);
        collect(&LocalEngine::new(m, Arc::new(ByteCodec), meta(), tel).unwrap().submit(req(p)).unwrap()).0
    };
    for ((p, ..), want) in cases().into_iter().zip(WIDE_CACHED) {
        let (ids, cached) = collect(&e.submit(req(p)).unwrap());
        assert_eq!(cached, want, "{p:?}");
        assert_eq!(ids, cold(p), "{p:?}: the warm request computes its cold output");
    }
    // a follow-up turn: LONG, its generated tokens, more text. Rows from 30 on were written by decode steps (1-row passes):
    // the resume point is the end of the prefill's rows, not the end of the shared tokens
    let (first, _) = collect(&e.submit(req(LONG)).unwrap());
    let mut follow = req(LONG);
    follow.prompt.extend(&first);
    follow.prompt.extend("and more text here".bytes().map(|b| b as u32));
    let (_, cached) = collect(&e.submit(follow).unwrap());
    assert_eq!(cached, LONG.len());
}

#[test]
fn batched_engine_picks_slots_by_wide_passes() {
    let (m, tel) = model(Mode::Wide, 2);
    let mut o = BatchedOptions::new(2, 16);
    o.prefix_reuse = true;
    let e = BatchedEngine::new(m, Arc::new(ByteCodec), meta(), tel, o).unwrap();
    for ((p, ..), want) in cases().into_iter().zip(WIDE_CACHED) {
        let (_, cached) = collect(&e.submit(req(p)).unwrap());
        assert_eq!(cached, want, "{p:?}");
    }
}

/// `Model::reuse_len` and `SequenceState::wide_reuse_len` on hand-built histories (Wide: passes of >= 2 rows agree).
#[test]
fn wide_reuse_len_reads_the_pass_history() {
    use intarsia_core::model::HeadMode;
    let (m, _) = model(Mode::Wide, 1);
    let p: Vec<u32> = LONG.bytes().map(|b| b as u32).collect();
    let mut st = m.new_state().unwrap();
    m.forward(&mut st, &p, HeadMode::None, None).unwrap();
    for t in [7u32, 8, 9] {
        m.forward(&mut st, &[t], HeadMode::None, None).unwrap();
    }
    let n = p.len();
    // the prompt again: rows 0..30 are wide, the decode rows are not; the last token alone would be a 1-row pass
    assert_eq!(st.wide_reuse_len(n - 1, 2), n - 1);
    assert_eq!(m.reuse_len(&st, n, n - 1), n - 2);
    // a longer prompt sharing everything held: stops where the decode rows begin
    assert_eq!(st.wide_reuse_len(n + 3, 2), n);
    assert_eq!(m.reuse_len(&st, n + 20, n + 3), n);
    // a truncate that cuts through the prompt's last pass keeps it as a wide pass: the next repeat reuses just as much
    st.truncate(n - 2, &*m.backend).unwrap();
    m.forward(&mut st, &p[n - 2..], HeadMode::None, None).unwrap();
    assert_eq!(st.passes.iter().map(|q| q.rows).collect::<Vec<_>>(), [8, 8, 8, 6, 2]);
    assert_eq!(m.reuse_len(&st, n, n - 1), n - 2);
    // a row-exact verify pass is 1-row work whatever its row count
    st.commit_verify(&[1, 2, 3]);
    assert_eq!(st.wide_reuse_len(n + 3, 2), n);
    // a 2-token prompt repeated: its last token alone would be a 1-row pass, so nothing
    let mut s2 = m.new_state().unwrap();
    m.forward(&mut s2, &p[..2], HeadMode::None, None).unwrap();
    assert_eq!(m.reuse_len(&s2, 2, 1), 0);
    // rows restored from a prefix store were written by no pass of this state
    s2.load_prefix(&p[..10]);
    assert_eq!(m.reuse_len(&s2, 20, 10), 0);
    // CUDA's report (no min) keeps the cold-schedule rule, cpu's any cut
    let (c, _) = model(Mode::Ladder, 1);
    let mut s3 = c.new_state().unwrap();
    c.forward(&mut s3, &p, HeadMode::None, None).unwrap();
    assert_eq!(c.reuse_len(&s3, n + 20, n), 24);
    let (cpu, _) = model(Mode::Any, 1);
    let mut s4 = cpu.new_state().unwrap();
    cpu.forward(&mut s4, &p, HeadMode::None, None).unwrap();
    assert_eq!(cpu.reuse_len(&s4, n + 20, n), n);
}

/// The one pass record serves every rule: on `Invariant` the row-invariant prefix is the unbroken run of row-invariant prefill
/// passes from row 0, a truncate through one clips it (the rows below the cut stay row-invariant), and a decode step, a verify
/// pass or an ordinary pass ends it.
#[test]
fn pass_record_carries_the_row_invariant_prefix() {
    use intarsia_core::model::HeadMode;
    use intarsia_core::state::PassKind;
    let (m, _) = model(Mode::Invariant, 1);
    let p: Vec<u32> = LONG.bytes().map(|b| b as u32).collect();
    let n = p.len();
    let mut st = m.new_state().unwrap();
    m.forward_prefill(&mut st, &p[..20], HeadMode::None, None).unwrap();
    m.forward_prefill(&mut st, &p[20..], HeadMode::None, None).unwrap();
    assert!(st.passes.iter().all(|q| q.kind == PassKind::RowInvariant), "{:?}", st.passes);
    assert_eq!(st.inv_len(), n);
    assert_eq!(m.reuse_len(&st, n + 5, n), n);
    // a truncate through the last pass clips it: rows 0..25 are still row-invariant
    st.truncate(25, &*m.backend).unwrap();
    assert_eq!(st.inv_len(), 25);
    // a decode step ends the run; a row-invariant pass after it does not join
    st.commit_decode(&[1]);
    m.forward_prefill(&mut st, &p[26..], HeadMode::None, None).unwrap();
    assert_eq!(st.inv_len(), 25);
    assert_eq!(m.reuse_len(&st, n + 5, n), 25);
    assert_eq!(st.passes.iter().map(|q| q.kind).collect::<Vec<_>>()[st.passes.len() - 2..], [PassKind::Decode, PassKind::RowInvariant]);
    // a verify pass, an ordinary pass: the same
    for f in [|st: &mut intarsia_core::state::SequenceState| st.commit_verify(&[1, 2]), |st: &mut intarsia_core::state::SequenceState| st.commit(&[1, 2])] {
        let mut s2 = m.new_state().unwrap();
        m.forward_prefill(&mut s2, &p[..10], HeadMode::None, None).unwrap();
        f(&mut s2);
        m.forward_prefill(&mut s2, &p[12..], HeadMode::None, None).unwrap();
        assert_eq!(s2.inv_len(), 10);
    }
    // a prefix-store restore: no pass wrote the rows
    st.load_prefix(&p[..10]);
    assert_eq!(st.inv_len(), 0);
}
