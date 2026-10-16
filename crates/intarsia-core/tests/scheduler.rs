//! Continuous-batching scheduler tests without a GPU (crates/intarsia-core/src/scheduler.rs).
//!
//! The mock model's output for a position is a rolling hash over the slot's KV entries `0..=pos`, each entry hashing
//! its token, its position and who wrote it (the prefill chunk `(start, len)` that wrote it, or a decode step). So a
//! request's output changes if its slot is truncated wrongly, if it reads another slot, if its prefill is chunked
//! differently, or if a reused prefix was written differently; it does not depend on anything else in the batch
//! (unless `leak` is set, the negative control). The identity check compares every batched request with its solo
//! reference: one slot preloaded with the exact KV entries the batched run reused, prefilled with the chunk size the
//! batched run recorded, nothing else submitted.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use intarsia_core::rng::SplitMix64;
use intarsia_core::scheduler::{
    chunk_schedule, next_prefill_chunk, BatchEngine, BatchModel, BatchSeq, EndKind, RequestId, Row, Scheduler, SchedulerConfig, SeqStep,
    TraceEvent,
};
use intarsia_core::{Error, FinishReason, Result};

thread_local! {
    /// `BatchModel::avoid_single_row_prefill` of every `MockModel` created on this test thread (tests run one thread each).
    static AVOID: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn avoid() -> bool {
    AVOID.with(|a| a.get())
}

/// Run `f` with the mock models' flag set to `v`.
fn with_avoid<R>(v: bool, f: impl FnOnce() -> R) -> R {
    let old = AVOID.with(|a| a.replace(v));
    let r = f();
    AVOID.with(|a| a.set(old));
    r
}

const V: u64 = 50;
const EOS: u32 = 2;

fn mix(a: u64, b: u64) -> u64 {
    SplitMix64::new(a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15)).next_u64()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Argmax,
    Logits,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Writer {
    Prefill(usize, usize),
    Decode,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Entry {
    tok: u32,
    writer: Writer,
    h: u64,
}

type Out = (u64, Mode);

#[derive(Default)]
struct Rec {
    /// KV entries of the slot right after admission (the reused prefix), per request.
    snapshots: HashMap<RequestId, Vec<Entry>>,
    cached: HashMap<RequestId, usize>,
    /// Prefill chunks `(start, len)` as the model ran them, per request.
    chunks: HashMap<RequestId, Vec<(usize, usize)>>,
    batch_sizes: Vec<usize>,
}

struct MockModel {
    kv: Vec<Vec<Entry>>,
    toks: Vec<Vec<u32>>,
    cap: usize,
    max_rows: usize,
    owner: Vec<Option<RequestId>>,
    decodes: HashMap<RequestId, usize>,
    /// Row failure: the `n`-th decode row (1-based) of request `id` returns an inner error.
    fail_row: Vec<(RequestId, usize)>,
    /// Prefill failure of request `id` (non-fatal error).
    fail_prefill: Vec<RequestId>,
    /// The whole decode pass number `n` (1-based) fails (fatal).
    fail_step_at: Option<usize>,
    steps: usize,
    /// Negative control: the output depends on the number of rows.
    leak: bool,
    rec: Arc<Mutex<Rec>>,
    /// `BatchModel::avoid_single_row_prefill`.
    avoid: bool,
    /// While `false`, every `prefill_chunk` and `decode_step` blocks (a long step, for the live-gauge test).
    gate: Option<Arc<(Mutex<bool>, Condvar)>>,
}

impl MockModel {
    fn new(slots: usize, cap: usize) -> Self {
        MockModel {
            kv: vec![Vec::new(); slots],
            toks: vec![Vec::new(); slots],
            cap,
            max_rows: slots,
            owner: vec![None; slots],
            decodes: HashMap::new(),
            fail_row: Vec::new(),
            fail_prefill: Vec::new(),
            fail_step_at: None,
            steps: 0,
            leak: false,
            rec: Arc::new(Mutex::new(Rec::default())),
            avoid: avoid(),
            gate: None,
        }
    }
    fn wait_gate(&self) {
        if let Some(g) = &self.gate {
            let mut open = g.0.lock().unwrap();
            while !*open {
                open = g.1.wait(open).unwrap();
            }
        }
    }
    fn preload(&mut self, slot: usize, entries: Vec<Entry>) {
        self.toks[slot] = entries.iter().map(|e| e.tok).collect();
        self.kv[slot] = entries;
    }
    fn push(&mut self, slot: usize, tok: u32, writer: Writer) -> u64 {
        let pos = self.kv[slot].len();
        assert!(pos < self.cap, "slot {slot} overflows its capacity");
        let prev = self.kv[slot].last().map_or(0x1234, |e| e.h);
        let w = match writer {
            Writer::Prefill(s, l) => mix(s as u64 + 1, l as u64 + 7),
            Writer::Decode => 0xDEC0DE,
        };
        let h = mix(mix(prev, tok as u64), mix(pos as u64, w));
        self.kv[slot].push(Entry { tok, writer, h });
        self.toks[slot].push(tok);
        h
    }
}

fn mode_code(m: Mode) -> u64 {
    match m {
        Mode::Argmax => 11,
        Mode::Logits => 13,
    }
}

impl BatchModel for MockModel {
    type Mode = Mode;
    type Out = Out;
    fn slots(&self) -> usize {
        self.kv.len()
    }
    fn slot_cap(&self) -> usize {
        self.cap
    }
    fn avoid_single_row_prefill(&self) -> bool {
        self.avoid
    }
    fn slot_tokens(&self, slot: usize) -> &[u32] {
        &self.toks[slot]
    }
    fn truncate(&mut self, slot: usize, len: usize) -> Result<()> {
        assert!(len <= self.kv[slot].len());
        self.kv[slot].truncate(len);
        self.toks[slot].truncate(len);
        Ok(())
    }
    fn prefill_chunk(&mut self, slot: usize, tokens: &[u32], mode: Option<Mode>) -> Result<Option<Out>> {
        self.wait_gate();
        let id = self.owner[slot].expect("prefill into an unowned slot");
        if self.fail_prefill.contains(&id) {
            return Err(Error::Other("injected prefill failure".into()));
        }
        let start = self.kv[slot].len();
        self.rec.lock().unwrap().chunks.entry(id).or_default().push((start, tokens.len()));
        let mut h = 0;
        for &t in tokens {
            h = self.push(slot, t, Writer::Prefill(start, tokens.len()));
        }
        Ok(mode.map(|m| (mix(h, mode_code(m)), m)))
    }
    fn decode_step(&mut self, rows: &[Row<Mode>]) -> Result<Vec<Result<Out>>> {
        self.wait_gate();
        assert!(!rows.is_empty() && rows.len() <= self.max_rows);
        let mut seen: Vec<usize> = rows.iter().map(|r| r.slot).collect();
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), rows.len(), "two rows on one slot");
        self.steps += 1;
        self.rec.lock().unwrap().batch_sizes.push(rows.len());
        if self.fail_step_at == Some(self.steps) {
            return Err(Error::Backend("injected device failure".into()));
        }
        let b = rows.len() as u64;
        let mut outs = Vec::new();
        for r in rows {
            let id = self.owner[r.slot].expect("decode on an unowned slot");
            let n = self.decodes.entry(id).or_default();
            *n += 1;
            let n = *n;
            let h = self.push(r.slot, r.token, Writer::Decode);
            if self.fail_row.contains(&(id, n)) {
                outs.push(Err(Error::Other("injected row failure".into())));
                continue;
            }
            let mut o = mix(h, mode_code(r.mode));
            if self.leak {
                o = mix(o, b);
            }
            outs.push(Ok((o, r.mode)));
        }
        Ok(outs)
    }
    fn on_admit(&mut self, slot: usize, id: RequestId, cached: usize) {
        self.owner[slot] = Some(id);
        let mut rec = self.rec.lock().unwrap();
        rec.snapshots.insert(id, self.kv[slot].clone());
        rec.cached.insert(id, cached);
    }
}

#[derive(Clone, Debug, Default)]
struct Params {
    prompt: Vec<u32>,
    max_tokens: usize,
    min_tokens: usize,
    stop_seq: Vec<u32>,
    ignore_eos: bool,
    seed: Option<u64>,
    /// The sequence errors at its `n`-th token (a codec error, say).
    fail_at: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
enum Ev {
    Tok(u32),
    Done { reason: FinishReason, completion: usize, cached: usize },
    Err(String),
}

struct MockSeq {
    p: Params,
    tx: Sender<Ev>,
    cancel: Arc<AtomicBool>,
    rng: Option<SplitMix64>,
    stop_ids: Vec<u32>,
    tok: u32,
    generated: usize,
    max_new: usize,
    cached: usize,
    reason: Option<FinishReason>,
    out: Vec<u32>,
    spec_log: Arc<Mutex<Vec<bool>>>,
    on_tokens: Arc<AtomicUsize>,
}

impl MockSeq {
    fn new(p: Params) -> (MockSeq, Receiver<Ev>, Arc<AtomicBool>) {
        let (tx, rx) = channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let stop_ids = if p.ignore_eos { Vec::new() } else { vec![EOS] };
        let seq = MockSeq {
            rng: p.seed.map(SplitMix64::new),
            p,
            tx,
            cancel: cancel.clone(),
            stop_ids,
            tok: 0,
            generated: 0,
            max_new: 0,
            cached: 0,
            reason: None,
            out: Vec::new(),
            spec_log: Arc::new(Mutex::new(Vec::new())),
            on_tokens: Arc::new(AtomicUsize::new(0)),
        };
        (seq, rx, cancel)
    }
    fn head_mode(&self, k: usize) -> Mode {
        if self.rng.is_some() || (k <= self.p.min_tokens && !self.stop_ids.is_empty()) {
            Mode::Logits
        } else {
            Mode::Argmax
        }
    }
    fn token_from(&mut self, out: Out, k: usize) -> Result<u32> {
        if out.1 != self.head_mode(k) {
            return Err(Error::Other("wrong head mode".into()));
        }
        let mut t = match &mut self.rng {
            Some(r) => (mix(out.0, r.next_u64()) % V) as u32,
            None => (out.0 % V) as u32,
        };
        if k <= self.p.min_tokens {
            while self.stop_ids.contains(&t) {
                t = (t + 1) % V as u32;
            }
        }
        Ok(t)
    }
}

impl BatchSeq<MockModel> for MockSeq {
    fn prompt(&self) -> &[u32] {
        &self.p.prompt
    }
    fn admit(&mut self, cached: usize, cap: usize) -> Result<bool> {
        let plen = self.p.prompt.len();
        if plen == 0 {
            return Err(Error::BadRequest("empty prompt".into()));
        }
        if plen >= cap {
            return Err(Error::BadRequest("prompt does not fit".into()));
        }
        self.cached = cached;
        self.max_new = self.p.max_tokens.min(cap - plen);
        if self.max_new == 0 {
            self.reason = Some(FinishReason::Length);
        }
        Ok(self.max_new > 0)
    }
    fn first_mode(&self) -> Mode {
        self.head_mode(1)
    }
    fn prepare(&mut self, _m: &mut MockModel, _slot: usize, spec_allowed: bool) -> Result<SeqStep<Mode>> {
        self.spec_log.lock().unwrap().push(spec_allowed);
        Ok(SeqStep::Plain(self.tok, self.head_mode(self.generated + 1)))
    }
    fn accept(&mut self, out: Out) -> Result<()> {
        self.tok = self.token_from(out, self.generated + 1)?;
        Ok(())
    }
    fn on_token(&mut self) -> Result<bool> {
        self.on_tokens.fetch_add(1, Ordering::SeqCst);
        let tok = self.tok;
        self.generated += 1;
        if self.stop_ids.contains(&tok) {
            self.reason = Some(FinishReason::Stop);
            return Ok(true);
        }
        if self.p.fail_at == Some(self.generated) {
            return Err(Error::Text("injected decode failure".into()));
        }
        self.out.push(tok);
        if self.tx.send(Ev::Tok(tok)).is_err() {
            self.reason = Some(FinishReason::Cancelled);
            return Ok(true);
        }
        if self.generated > self.p.min_tokens && !self.p.stop_seq.is_empty() && self.out.ends_with(&self.p.stop_seq) {
            self.reason = Some(FinishReason::Stop);
            return Ok(true);
        }
        if self.generated >= self.max_new {
            self.reason = Some(FinishReason::Length);
            return Ok(true);
        }
        if self.cancel.load(Ordering::SeqCst) {
            self.reason = Some(FinishReason::Cancelled);
            return Ok(true);
        }
        Ok(false)
    }
    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
    fn finish(&mut self, _m: &mut MockModel, _slot: Option<usize>, cancelled: bool) -> Result<()> {
        // the scheduler only finishes a request uncancelled once it ended on its own (as engine::Seq::finish expects)
        let reason =
            if cancelled { FinishReason::Cancelled } else { self.reason.expect("finish before the sequence ended") };
        let _ = self.tx.send(Ev::Done { reason, completion: self.generated, cached: self.cached });
        Ok(())
    }
    fn fail(&mut self, err: &Error) {
        let _ = self.tx.send(Ev::Err(err.to_string()));
    }
}

fn collect(rx: &Receiver<Ev>) -> Vec<Ev> {
    rx.try_iter().collect()
}

fn tokens(evs: &[Ev]) -> Vec<u32> {
    evs.iter().filter_map(|e| if let Ev::Tok(t) = e { Some(*t) } else { None }).collect()
}

fn run_to_end(s: &mut Scheduler<MockModel, MockSeq>) {
    let mut n = 0;
    while s.has_work() {
        s.step();
        n += 1;
        assert!(n < 1_000_000, "scheduler does not terminate");
    }
}

/// Independent statement of the schedule rule (not `chunk_schedule`): from `cached`, chunks of `c` while more than `c`
/// tokens remain, then the rest; with `avoid` (and `c >= 3`) a `k * c + 1` remainder with `k >= 1` ends `c - 1, 2`.
fn expected_chunks(cached: usize, plen: usize, c: usize, avoid: bool) -> Vec<(usize, usize)> {
    let rem = plen - cached;
    let n_full = (rem - 1) / c; // the final chunk holds 1..=c tokens
    let split = avoid && c >= 3 && n_full >= 1 && rem - n_full * c == 1;
    let n_full = if split { n_full - 1 } else { n_full };
    let mut v: Vec<(usize, usize)> = (0..n_full).map(|i| (cached + i * c, c)).collect();
    let start = cached + n_full * c;
    if split {
        v.push((start, c - 1));
        v.push((start + c - 1, 2));
    } else {
        v.push((start, plen - start));
    }
    v
}

struct SoloRun {
    events: Vec<Ev>,
    chunks: Vec<(usize, usize)>,
}

/// The solo reference: one slot holding `snapshot` (the exact reused prefix KV), chunk `c`, nothing else running.
fn solo(
    p: &Params,
    snapshot: Vec<Entry>,
    c: usize,
    cap: usize,
    fail_row_n: Option<usize>,
    fail_prefill: bool,
) -> SoloRun {
    let mut m = MockModel::new(1, cap);
    m.preload(0, snapshot);
    if let Some(n) = fail_row_n {
        m.fail_row.push((0, n));
    }
    if fail_prefill {
        m.fail_prefill.push(0);
    }
    let rec = m.rec.clone();
    let cfg = SchedulerConfig { max_batch: 1, solo_chunk: c, busy_chunk: c, ..Default::default() };
    let mut s = Scheduler::new(m, cfg).unwrap();
    let (seq, rx, _c) = MockSeq::new(p.clone());
    s.submit(seq).unwrap();
    run_to_end(&mut s);
    let chunks = rec.lock().unwrap().chunks.get(&0).cloned().unwrap_or_default();
    SoloRun { events: collect(&rx), chunks }
}

struct Req {
    p: Params,
    arrive: u64,
    cancel_at: Option<u64>,
    drop_rx_at: Option<u64>,
    fail_row_n: Option<usize>,
    fail_prefill: bool,
}

struct BatchRun {
    events: Vec<Vec<Ev>>,
    ids: Vec<Option<RequestId>>,
    cancelled: Vec<bool>,
    trace: Vec<TraceEvent>,
    rec: Arc<Mutex<Rec>>,
    metrics: intarsia_core::scheduler::BatchMetrics,
    on_tokens: usize,
}

fn run_batched(cfg: SchedulerConfig, cap: usize, reqs: &[Req], leak: bool) -> BatchRun {
    let mut m = MockModel::new(cfg.max_batch, cap);
    m.leak = leak;
    let rec = m.rec.clone();
    let mut s = Scheduler::new(m, SchedulerConfig { trace: true, ..cfg }).unwrap();
    let on_tokens = Arc::new(AtomicUsize::new(0));
    let mut rxs: Vec<Option<Receiver<Ev>>> = (0..reqs.len()).map(|_| None).collect();
    let mut flags: Vec<Option<Arc<AtomicBool>>> = vec![None; reqs.len()];
    let mut ids = vec![None; reqs.len()];
    let mut events: Vec<Vec<Ev>> = vec![Vec::new(); reqs.len()];
    let mut cancelled = vec![false; reqs.len()];
    let mut it: u64 = 0;
    loop {
        for (i, r) in reqs.iter().enumerate() {
            if r.arrive == it {
                let (mut seq, rx, flag) = MockSeq::new(r.p.clone());
                seq.on_tokens = on_tokens.clone();
                // ids are assigned in submit order; the injected failures are keyed by id
                let id = s.iteration_next_id_hint();
                if let Some(n) = r.fail_row_n {
                    s.model_mut().fail_row.push((id, n));
                }
                if r.fail_prefill {
                    s.model_mut().fail_prefill.push(id);
                }
                ids[i] = Some(s.submit(seq).unwrap());
                assert_eq!(ids[i], Some(id));
                rxs[i] = Some(rx);
                flags[i] = Some(flag);
            }
            if r.cancel_at == Some(it) {
                if let Some(f) = &flags[i] {
                    f.store(true, Ordering::SeqCst);
                    cancelled[i] = true;
                }
            }
            if r.drop_rx_at == Some(it) {
                if let Some(rx) = rxs[i].take() {
                    events[i].extend(collect(&rx));
                    cancelled[i] = true;
                }
            }
        }
        for (i, rx) in rxs.iter().enumerate() {
            if let Some(rx) = rx {
                events[i].extend(collect(rx));
            }
        }
        let pending = reqs.iter().any(|r| r.arrive > it);
        if !s.has_work() && !pending {
            break;
        }
        s.step();
        it += 1;
        assert!(it < 1_000_000);
    }
    for (i, rx) in rxs.iter().enumerate() {
        if let Some(rx) = rx {
            events[i].extend(collect(rx));
        }
    }
    let metrics = s.metrics();
    BatchRun {
        events,
        ids,
        cancelled,
        trace: s.take_trace(),
        rec,
        metrics,
        on_tokens: on_tokens.load(Ordering::SeqCst),
    }
}

/// Helper on the scheduler for the tests: the id the next submit gets (ids count up from 0 per scheduler).
trait NextId {
    fn iteration_next_id_hint(&self) -> RequestId;
}
impl NextId for Scheduler<MockModel, MockSeq> {
    fn iteration_next_id_hint(&self) -> RequestId {
        self.metrics().submitted
    }
}

/// What a randomized run exercised (so a passing run cannot be a vacuous one).
#[derive(Debug, Default)]
struct Cov {
    exact: usize,
    prefix: usize,
    reused: usize,
    busy_chunk: usize,
    multi_chunk: usize,
    errors: usize,
    sampled: usize,
    min_tokens: usize,
    stopped: usize,
}

/// Compare every request of a batched run with its solo reference. Returns the number of mismatches.
fn check_identity(run: &BatchRun, reqs: &[Req], cfg: &SchedulerConfig, cap: usize) -> usize {
    check_identity_cov(run, reqs, cfg, cap, &mut Cov::default())
}

fn check_identity_cov(run: &BatchRun, reqs: &[Req], cfg: &SchedulerConfig, cap: usize, cov: &mut Cov) -> usize {
    let rec = run.rec.lock().unwrap();
    let mut start_chunk: HashMap<RequestId, usize> = HashMap::new();
    for e in &run.trace {
        if let TraceEvent::PrefillStart { id, chunk, decoding, .. } = e {
            let want = if *decoding > 0 { cfg.busy_chunk } else { cfg.solo_chunk };
            assert_eq!(*chunk, want, "request {id}: chunk size choice");
            start_chunk.insert(*id, *chunk);
        }
    }
    let mut mismatches = 0;
    for (i, r) in reqs.iter().enumerate() {
        let id = run.ids[i].unwrap();
        let got = &run.events[i];
        let Some(snapshot) = rec.snapshots.get(&id).cloned() else {
            // never admitted: cancelled while queued (no tokens) or rejected before admission
            assert!(tokens(got).is_empty(), "request {id} produced tokens without a slot");
            continue;
        };
        let cached = rec.cached[&id];
        assert_eq!(snapshot.len(), cached);
        assert_eq!(snapshot.iter().map(|e| e.tok).collect::<Vec<_>>(), r.p.prompt[..cached].to_vec());
        let c = start_chunk.get(&id).copied().unwrap_or(cfg.solo_chunk);
        let reference = solo(&r.p, snapshot, c, cap, r.fail_row_n, r.fail_prefill);
        let chunks = rec.chunks.get(&id).cloned().unwrap_or_default();
        let plen = r.p.prompt.len();
        cov.reused += (cached > 0) as usize;
        cov.busy_chunk += (c == cfg.busy_chunk && chunks.len() > 1) as usize;
        cov.multi_chunk += (chunks.len() > 1) as usize;
        cov.errors += matches!(reference.events.last(), Some(Ev::Err(_))) as usize;
        cov.sampled += r.p.seed.is_some() as usize;
        cov.min_tokens += (r.p.min_tokens > 0) as usize;
        cov.stopped += matches!(reference.events.last(), Some(Ev::Done { reason: FinishReason::Stop, .. })) as usize;
        if run.cancelled[i] {
            cov.prefix += 1;
            // a cancelled request is a prefix of its reference: tokens, and the chunks that ran
            let t = tokens(got);
            let rt = tokens(&reference.events);
            let chunks_prefix =
                chunks.len() <= reference.chunks.len() && chunks[..] == reference.chunks[..chunks.len()];
            if t.len() > rt.len() || t[..] != rt[..t.len()] || !chunks_prefix {
                mismatches += 1;
            }
        } else {
            if r.max_tokens_nonzero() && !r.fail_prefill && plen < cap && plen > 0 {
                assert_eq!(chunks, expected_chunks(cached, plen, c, avoid()), "request {id}: chunk schedule vs the rule");
            }
            cov.exact += 1;
            if *got != reference.events || chunks != reference.chunks {
                mismatches += 1;
            }
        }
    }
    mismatches
}

impl Req {
    fn max_tokens_nonzero(&self) -> bool {
        self.p.max_tokens > 0
    }
}

/// Scheduler invariants visible in the trace.
fn check_trace(run: &BatchRun, cfg: &SchedulerConfig) {
    let mut admitted = Vec::new();
    let mut started = Vec::new();
    let mut decode_iters: HashMap<RequestId, Vec<u64>> = HashMap::new();
    let mut prefill_tokens_in_iter: HashMap<u64, usize> = HashMap::new();
    let mut decode_in_iter: HashMap<u64, Vec<RequestId>> = HashMap::new();
    let mut chunk_of: HashMap<RequestId, usize> = HashMap::new();
    let mut n_decode = 0u64;
    for e in &run.trace {
        match e {
            TraceEvent::Admitted { id, .. } => admitted.push(*id),
            TraceEvent::PrefillStart { id, chunk, .. } => {
                started.push(*id);
                chunk_of.insert(*id, *chunk);
            }
            TraceEvent::Chunk { iter, id, len, last, .. } => {
                *prefill_tokens_in_iter.entry(*iter).or_default() += len;
                if !last {
                    // only the final chunk may be short, except the (C - 1, 2) split before a 2-row final chunk
                    let c = chunk_of[id];
                    assert!(*len == c || (avoid() && c >= 3 && *len == c - 1), "only the final chunk may be short");
                }
            }
            TraceEvent::Decode { iter, ids, spec_allowed } => {
                n_decode += 1;
                assert!(ids.len() <= cfg.max_batch);
                assert_eq!(*spec_allowed, ids.len() < cfg.spec_off_at, "speculation gate");
                decode_in_iter.insert(*iter, ids.clone());
                for id in ids {
                    decode_iters.entry(*id).or_default().push(*iter);
                }
            }
            TraceEvent::Finished { .. } => {}
        }
    }
    // FIFO: admission and prefill start in submit (= id) order
    assert!(admitted.windows(2).all(|w| w[0] < w[1]), "admission not FIFO: {admitted:?}");
    assert!(started.windows(2).all(|w| w[0] < w[1]), "prefill start not FIFO: {started:?}");
    // decode-first: once decoding, a request gets a row every iteration until it ends
    for (id, its) in &decode_iters {
        assert!(its.windows(2).all(|w| w[1] == w[0] + 1), "request {id} skipped a decode iteration: {its:?}");
    }
    // prefill budget while others decode: an iteration whose prefill stalls a stream that decodes before and after it
    for (iter, tok) in &prefill_tokens_in_iter {
        let (Some(now), Some(next)) = (decode_in_iter.get(iter), decode_in_iter.get(&(iter + 1))) else { continue };
        if now.iter().any(|id| next.contains(id)) {
            assert!(
                *tok <= cfg.busy_chunk * cfg.busy_chunks_per_iter,
                "iter {iter}: {tok} prefill tokens beside decode"
            );
        }
    }
    // metrics agree with the trace and the model
    let m = &run.metrics;
    assert_eq!(m.decode_steps, n_decode);
    assert_eq!(m.batch_hist.iter().sum::<u64>(), m.decode_steps);
    let sizes = run.rec.lock().unwrap().batch_sizes.clone();
    for b in 1..=8 {
        assert_eq!(m.batch_hist[b], sizes.iter().filter(|&&s| s == b).count() as u64);
    }
    assert_eq!(m.ttft.count + m.itl.count, run.on_tokens as u64);
    assert_eq!(m.running, 0);
    assert_eq!(m.queue_depth, 0);
}

fn random_prompt(rng: &mut SplitMix64, len: usize) -> Vec<u32> {
    (0..len).map(|_| 3 + (rng.next_u64() % (V - 3)) as u32).collect()
}

fn random_reqs(rng: &mut SplitMix64, n: usize, cap: usize) -> Vec<Req> {
    let families: Vec<Vec<u32>> = (0..3)
        .map(|_| {
            let l = 50 + (rng.next_u64() % 600) as usize;
            random_prompt(rng, l)
        })
        .collect();
    let mut out = Vec::new();
    for _ in 0..n {
        let r = rng.next_u64() % 100;
        let mut prompt = if r < 40 {
            // shared prefix (prefix reuse from idle slots)
            let f = &families[(rng.next_u64() % 3) as usize];
            let mut p = f[..1 + (rng.next_u64() as usize % f.len())].to_vec();
            let extra = (rng.next_u64() % 400) as usize;
            p.extend(random_prompt(rng, extra));
            p
        } else if r < 50 {
            let l = 2049 + (rng.next_u64() % 1500) as usize; // several solo chunks
            random_prompt(rng, l)
        } else {
            let l = 1 + (rng.next_u64() % 1200) as usize;
            random_prompt(rng, l)
        };
        let special = rng.next_u64() % 100;
        if special < 2 {
            prompt.clear(); // rejected: empty
        } else if special < 4 {
            prompt = random_prompt(rng, cap + 5); // rejected: too long
        }
        let mut p = Params {
            prompt,
            max_tokens: (rng.next_u64() % 80) as usize,
            min_tokens: if rng.next_u64() % 4 == 0 { (rng.next_u64() % 12) as usize } else { 0 },
            ignore_eos: rng.next_u64() % 5 == 0,
            seed: if rng.next_u64() % 3 == 0 { Some(rng.next_u64()) } else { None },
            ..Default::default()
        };
        if rng.next_u64() % 4 == 0 {
            p.stop_seq = vec![3 + (rng.next_u64() % (V - 3)) as u32];
        }
        if rng.next_u64() % 25 == 0 {
            p.fail_at = Some(1 + (rng.next_u64() % 10) as usize);
        }
        let arrive = rng.next_u64() % 120;
        let cancel_at = if rng.next_u64() % 7 == 0 { Some(arrive + rng.next_u64() % 60) } else { None };
        let drop_rx_at = if rng.next_u64() % 15 == 0 { Some(arrive + rng.next_u64() % 60) } else { None };
        let fail_row_n = if rng.next_u64() % 25 == 0 { Some(1 + (rng.next_u64() % 8) as usize) } else { None };
        let fail_prefill = rng.next_u64() % 40 == 0;
        out.push(Req { p, arrive, cancel_at, drop_rx_at, fail_row_n, fail_prefill });
    }
    out
}

#[test]
fn chunk_schedule_rule() {
    // flag off: today's schedule, bit for bit
    assert_eq!(chunk_schedule(0, 1, 2048, false), vec![(0, 1)]);
    assert_eq!(chunk_schedule(0, 2048, 2048, false), vec![(0, 2048)]);
    assert_eq!(chunk_schedule(0, 2049, 2048, false), vec![(0, 2048), (2048, 1)]);
    assert_eq!(chunk_schedule(100, 700, 256, false), vec![(100, 256), (356, 256), (612, 88)]);
    assert_eq!(chunk_schedule(99, 100, 256, false), vec![(99, 1)]);
    for cached in [0, 1, 7, 300] {
        for plen in [cached + 1, cached + 2, cached + 3, cached + 255, cached + 256, cached + 257, cached + 258, cached + 3000] {
            for c in [1, 2, 3, 4, 256, 2048] {
                let plain = {
                    let mut v = Vec::new();
                    let mut pos = cached;
                    while plen - pos > c {
                        v.push((pos, c));
                        pos += c;
                    }
                    v.push((pos, plen - pos));
                    v
                };
                assert_eq!(chunk_schedule(cached, plen, c, false), plain, "flag off, {cached} {plen} {c}");
                assert_eq!(chunk_schedule(cached, plen, c, false), expected_chunks(cached, plen, c, false));
                let on = chunk_schedule(cached, plen, c, true);
                assert_eq!(on, expected_chunks(cached, plen, c, true), "flag on, {cached} {plen} {c}");
                // always a partition of cached..plen with chunks <= c
                let mut pos = cached;
                for &(s, l) in &on {
                    assert!(s == pos && l >= 1 && l <= c);
                    pos += l;
                }
                assert_eq!(pos, plen);
                // a 1-row chunk only where unavoidable: c <= 2, or the whole remainder is one row
                let ones = on.iter().filter(|x| x.1 == 1).count();
                if c >= 3 && plen - cached > 1 {
                    assert_eq!(ones, 0, "flag on, {cached} {plen} {c}: {on:?}");
                }
            }
        }
    }
}

#[test]
fn next_prefill_chunk_rule() {
    // (remaining, chunk) -> flag off, flag on
    let cases: &[(usize, usize, usize, usize)] = &[
        (1, 256, 1, 1),         // whole remainder is one row: unavoidable
        (256, 256, 256, 256),   // exactly one chunk
        (257, 256, 256, 255),   // L = C + 1
        (258, 256, 256, 256),   // 2-row tail is fine
        (513, 256, 256, 256),   // 2C + 1: first chunk is a plain C, the split happens one step later
        (2049, 2048, 2048, 2047),
        (2049, 256, 256, 256),  // 2049 @ 256: eight full chunks, then 257 left
        (257, 2048, 257, 257),  // fits one chunk
        (3, 2, 2, 2),           // C = 2: (1, 2) would itself be a 1-row chunk, keep (2, 1)
        (2, 2, 2, 2),
        (4, 3, 3, 2),           // C = 3: 4 -> (2, 2)
        (5, 1, 1, 1),           // C = 1: unavoidable
        (1, 1, 1, 1),
    ];
    for &(rem, c, off, on) in cases {
        assert_eq!(next_prefill_chunk(rem, c, false), off, "off {rem} {c}");
        assert_eq!(next_prefill_chunk(rem, c, true), on, "on {rem} {c}");
    }
    // the named schedules
    let sch = |cached, plen, c| chunk_schedule(cached, plen, c, true);
    assert_eq!(sch(0, 257, 256), vec![(0, 255), (255, 2)]);
    assert_eq!(sch(0, 513, 256), vec![(0, 256), (256, 255), (511, 2)]);
    assert_eq!(sch(0, 2049, 2048), vec![(0, 2047), (2047, 2)]);
    assert_eq!(sch(0, 4097, 2048), vec![(0, 2048), (2048, 2047), (4095, 2)]);
    assert_eq!(sch(0, 2049, 256), {
        let mut v: Vec<_> = (0..7).map(|i| (i * 256, 256)).collect();
        v.push((1792, 255));
        v.push((2047, 2));
        v
    });
    // cached offset: 257 rows left after 100 cached
    assert_eq!(sch(100, 357, 256), vec![(100, 255), (355, 2)]);
    assert_eq!(chunk_schedule(100, 357, 256, false), vec![(100, 256), (356, 1)]);
    // prefix hit capped at prompt_len - 1 leaves one row: unavoidable
    assert_eq!(sch(99, 100, 256), vec![(99, 1)]);
    // C = 2 and C = 1
    assert_eq!(sch(0, 3, 2), vec![(0, 2), (2, 1)]);
    assert_eq!(sch(0, 5, 2), vec![(0, 2), (2, 2), (4, 1)]);
    assert_eq!(sch(0, 3, 1), vec![(0, 1), (1, 1), (2, 1)]);
    assert_eq!(sch(0, 4, 3), vec![(0, 2), (2, 2)]);
}

#[test]
fn config_limits() {
    let m = || MockModel::new(8, 64);
    assert!(Scheduler::<MockModel, MockSeq>::new(m(), SchedulerConfig { max_batch: 0, ..Default::default() }).is_err());
    assert!(Scheduler::<MockModel, MockSeq>::new(
        MockModel::new(9, 64),
        SchedulerConfig { max_batch: 9, ..Default::default() }
    )
    .is_err());
    let d = SchedulerConfig::default();
    assert_eq!((d.max_batch, d.solo_chunk, d.busy_chunk, d.busy_chunks_per_iter, d.spec_off_at), (1, 2048, 256, 1, 4));
    assert!(!d.trace && !d.trace_stderr, "tracing is off by default");
    let ok = |n| SchedulerConfig { spec_off_at: n, ..Default::default() }.validate();
    assert!(ok(0).is_ok() && ok(9).is_ok() && ok(10).is_err());
}

/// The main property test: staggered arrivals, cancellations at random iterations, client disconnects, mixed lengths,
/// greedy / sampled / min_tokens / stop sequences / EOS, rejected prompts, injected per-request failures; every request
/// equals its solo reference (cancelled ones: a prefix of it), for B in {1, 2, 3, 4, 6, 8}.
#[test]
fn batched_equals_solo_randomized() {
    for flag in [false, true] {
        with_avoid(flag, batched_equals_solo_randomized_body);
    }
}

fn batched_equals_solo_randomized_body() {
    let cap = 4096;
    let mut total = 0;
    let mut cov = Cov::default();
    for &b in &[1usize, 2, 3, 4, 6, 8] {
        let mut max_rows = 0;
        for seed in 0..6u64 {
            let mut rng = SplitMix64::new(seed * 1000 + b as u64);
            let reqs = random_reqs(&mut rng, 24, cap);
            let cfg = SchedulerConfig { max_batch: b, max_queue: 1000, ..Default::default() };
            let run = run_batched(cfg.clone(), cap, &reqs, false);
            check_trace(&run, &cfg);
            let mm = check_identity_cov(&run, &reqs, &cfg, cap, &mut cov);
            assert_eq!(mm, 0, "B = {b}, seed {seed}: {mm} requests differ from their solo reference");
            total += reqs.len();
            max_rows = max_rows.max((1..=8).rev().find(|&k| run.metrics.batch_hist[k] > 0).unwrap_or(0));
        }
        assert_eq!(max_rows, b, "B = {b} never reached");
    }
    assert_eq!(total, 6 * 6 * 24);
    eprintln!("coverage: {cov:?}");
    assert!(cov.exact >= 500 && cov.prefix >= 50 && cov.reused >= 50 && cov.busy_chunk >= 50, "{cov:?}");
    assert!(cov.multi_chunk >= 50 && cov.errors >= 20 && cov.sampled >= 100 && cov.min_tokens >= 50, "{cov:?}");
    assert!(cov.stopped >= 50, "{cov:?}");
}

/// Same with two chunks per iteration under load and a small busy chunk (budget 2 x 64).
#[test]
fn batched_equals_solo_other_budget() {
    for flag in [false, true] {
        with_avoid(flag, batched_equals_solo_other_budget_body);
    }
}

fn batched_equals_solo_other_budget_body() {
    let cap = 4096;
    for &b in &[2usize, 4, 8] {
        let mut rng = SplitMix64::new(77 + b as u64);
        let reqs = random_reqs(&mut rng, 20, cap);
        let cfg = SchedulerConfig {
            max_batch: b,
            max_queue: 1000,
            busy_chunk: 64,
            busy_chunks_per_iter: 2,
            spec_off_at: 2,
            ..Default::default()
        };
        let run = run_batched(cfg.clone(), cap, &reqs, false);
        check_trace(&run, &cfg);
        assert_eq!(check_identity(&run, &reqs, &cfg, cap), 0);
    }
}

/// Negative control 1: a model whose rows depend on the batch size is caught.
#[test]
fn leaky_model_is_detected() {
    let cap = 4096;
    let mut rng = SplitMix64::new(5);
    let mut reqs = random_reqs(&mut rng, 16, cap);
    for (i, r) in reqs.iter_mut().enumerate() {
        r.arrive = i as u64; // overlap guaranteed
        r.cancel_at = None;
        r.drop_rx_at = None;
        r.p.max_tokens = 40;
        r.p.ignore_eos = true;
        r.p.fail_at = None;
        r.fail_row_n = None;
        r.fail_prefill = false;
        if r.p.prompt.is_empty() || r.p.prompt.len() >= cap {
            r.p.prompt = vec![5; 10];
        }
    }
    let cfg = SchedulerConfig { max_batch: 4, max_queue: 1000, ..Default::default() };
    let run = run_batched(cfg.clone(), cap, &reqs, true);
    assert!(check_identity(&run, &reqs, &cfg, cap) > 0, "a batch-size-dependent model went unnoticed");
}

/// Negative control 2: the mock is sensitive to the chunk schedule, so a reference with another chunk size differs.
#[test]
fn chunk_size_changes_output() {
    let mut rng = SplitMix64::new(9);
    let p = Params { prompt: random_prompt(&mut rng, 1000), max_tokens: 20, ignore_eos: true, ..Default::default() };
    let a = solo(&p, Vec::new(), 2048, 4096, None, false);
    let b = solo(&p, Vec::new(), 256, 4096, None, false);
    assert_eq!(a.chunks, vec![(0, 1000)]);
    assert_eq!(b.chunks, vec![(0, 256), (256, 256), (512, 256), (768, 232)]);
    assert_ne!(tokens(&a.events), tokens(&b.events));
}

fn submit(s: &mut Scheduler<MockModel, MockSeq>, p: Params) -> (RequestId, Receiver<Ev>, Arc<AtomicBool>) {
    let (seq, rx, f) = MockSeq::new(p);
    (s.submit(seq).unwrap(), rx, f)
}

fn greedy(prompt: Vec<u32>, max_tokens: usize) -> Params {
    Params { prompt, max_tokens, ignore_eos: true, ..Default::default() }
}

/// max_batch = 1 (the default) always prefills at the solo chunk: today's behaviour.
#[test]
fn max_batch_one_uses_solo_chunk() {
    let mut s = Scheduler::new(MockModel::new(1, 8192), SchedulerConfig { trace: true, ..Default::default() }).unwrap();
    let mut rng = SplitMix64::new(1);
    let rxs: Vec<_> = (0..5).map(|i| submit(&mut s, greedy(random_prompt(&mut rng, 1500 + 700 * i), 10))).collect();
    run_to_end(&mut s);
    let chunks: Vec<usize> = s
        .take_trace()
        .iter()
        .filter_map(|e| if let TraceEvent::PrefillStart { chunk, .. } = e { Some(*chunk) } else { None })
        .collect();
    assert_eq!(chunks, vec![2048; 5]);
    for (_, rx, _) in rxs {
        assert_eq!(tokens(&collect(&rx)).len(), 10);
    }
}

/// Run the same small workload under `cfg`; the scheduler is returned for its trace.
fn traced_run(cfg: SchedulerConfig) -> Scheduler<MockModel, MockSeq> {
    let mut s = Scheduler::new(MockModel::new(2, 8192), SchedulerConfig { max_batch: 2, ..cfg }).unwrap();
    let mut rng = SplitMix64::new(7);
    for _ in 0..3 {
        submit(&mut s, greedy(random_prompt(&mut rng, 300), 20));
    }
    run_to_end(&mut s);
    s
}

/// Tracing off (the default and production): no event is stored. Print-only tracing stores nothing either. With the
/// store on it is a ring of at most `trace_cap` events, newest kept.
#[test]
fn trace_store_is_off_by_default_and_bounded() {
    let mut off = traced_run(SchedulerConfig::default());
    assert!(off.take_trace().is_empty(), "no trace store when tracing is off");

    let mut print_only = traced_run(SchedulerConfig { trace_stderr: true, ..Default::default() });
    assert!(print_only.take_trace().is_empty(), "stderr tracing must not store events");

    let full = traced_run(SchedulerConfig { trace: true, ..Default::default() }).take_trace();
    assert!(full.len() > 8, "workload should produce more events than the small cap below");
    let cap = 5;
    let ring = traced_run(SchedulerConfig { trace: true, trace_cap: cap, ..Default::default() }).take_trace();
    assert_eq!(ring.len(), cap, "store stays at trace_cap");
    assert_eq!(ring[..], full[full.len() - cap..], "the newest events are the ones kept");
}

/// A request that starts its prefill while another decodes takes 256-token chunks, and keeps them after the other one
/// ends mid-prefill; the decoding stream advances one token per iteration meanwhile.
#[test]
fn busy_chunk_is_sticky() {
    let cfg = SchedulerConfig { max_batch: 2, trace: true, ..Default::default() };
    let mut s = Scheduler::new(MockModel::new(2, 8192), cfg).unwrap();
    let mut rng = SplitMix64::new(2);
    let (a, rxa, _) = submit(&mut s, greedy(random_prompt(&mut rng, 100), 3));
    s.step(); // A prefilled, first token
    let (b, rxb, _) = submit(&mut s, greedy(random_prompt(&mut rng, 1000), 5));
    run_to_end(&mut s);
    let tr = s.take_trace();
    let starts: Vec<(RequestId, usize)> = tr
        .iter()
        .filter_map(|e| if let TraceEvent::PrefillStart { id, chunk, .. } = e { Some((*id, *chunk)) } else { None })
        .collect();
    assert_eq!(starts, vec![(a, 2048), (b, 256)]);
    let b_chunks: Vec<(usize, usize)> = tr
        .iter()
        .filter_map(|e| {
            if let TraceEvent::Chunk { id, start, len, .. } = e {
                (*id == b).then_some((*start, *len))
            } else {
                None
            }
        })
        .collect();
    assert_eq!(b_chunks, chunk_schedule(0, 1000, 256, false));
    // A ended (3 tokens) while B was still prefilling: B's later chunks stay 256
    let a_end = tr
        .iter()
        .find_map(|e| if let TraceEvent::Finished { iter, id, .. } = e { (*id == a).then_some(*iter) } else { None })
        .unwrap();
    let b_last = tr
        .iter()
        .filter_map(|e| if let TraceEvent::Chunk { iter, id, .. } = e { (*id == b).then_some(*iter) } else { None })
        .max()
        .unwrap();
    assert!(a_end < b_last);
    assert_eq!(tokens(&collect(&rxa)).len(), 3);
    assert_eq!(tokens(&collect(&rxb)).len(), 5);
}

/// Idle-slot prefix reuse: longest common prefix wins, running slots are never taken, LRU breaks ties.
#[test]
fn prefix_reuse_from_idle_slots() {
    let cfg = SchedulerConfig { max_batch: 3, trace: true, ..Default::default() };
    let mut s = Scheduler::new(MockModel::new(3, 8192), cfg).unwrap();
    let mut rng = SplitMix64::new(3);
    let p1 = random_prompt(&mut rng, 300);
    let p2 = random_prompt(&mut rng, 200);
    let (_, _r1, _) = submit(&mut s, greedy(p1.clone(), 4)); // slot 0
    run_to_end(&mut s);
    let (_, _r2, _) = submit(&mut s, greedy(p2.clone(), 4)); // slot 1 (no prefix anywhere: LRU, slot 1 never used)
    run_to_end(&mut s);
    // a long-running request on p2's prefix takes slot 1 (longest prefix)
    let mut p2b = p2[..150].to_vec();
    p2b.extend(random_prompt(&mut rng, 20));
    let (d, _rd, fd) = submit(&mut s, greedy(p2b.clone(), 1000));
    s.step();
    // same prompt again while D runs: slot 1 is busy, so it reuses only what idle slots offer (nothing) -> LRU slot 2
    let (e, _re, _) = submit(&mut s, greedy(p2b.clone(), 2));
    // then a continuation of p1: slot 0
    let mut p1b = p1.clone();
    p1b.extend(random_prompt(&mut rng, 10));
    let (f, _rf, _) = submit(&mut s, greedy(p1b, 2));
    for _ in 0..50 {
        s.step();
    }
    fd.store(true, Ordering::SeqCst);
    run_to_end(&mut s);
    let adm: HashMap<RequestId, (usize, usize)> = s
        .take_trace()
        .iter()
        .filter_map(|e| {
            if let TraceEvent::Admitted { id, slot, cached, .. } = e {
                Some((*id, (*slot, *cached)))
            } else {
                None
            }
        })
        .collect();
    assert_eq!(adm[&0], (0, 0));
    assert_eq!(adm[&1], (1, 0));
    assert_eq!(adm[&d], (1, 150));
    assert_eq!(adm[&e], (2, 0));
    assert_eq!(adm[&f], (0, 300));
}

/// Cancellation in all three phases: queued (never takes a slot), between prefill chunks, decoding. The others go on
/// unchanged; a cancelled slot holds exactly its tokens' KV and can be reused.
#[test]
fn cancel_in_every_phase() {
    let cfg = SchedulerConfig { max_batch: 3, trace: true, ..Default::default() };
    let mut s = Scheduler::new(MockModel::new(3, 8192), cfg).unwrap();
    let mut rng = SplitMix64::new(4);
    let pa = random_prompt(&mut rng, 50);
    let (a, rxa, fa) = submit(&mut s, greedy(pa.clone(), 100)); // decoding, cancelled later
    s.step();
    let pb = random_prompt(&mut rng, 3000);
    let (b, rxb, fb) = submit(&mut s, greedy(pb.clone(), 5)); // cancelled between prefill chunks
    let (_d, rxd, _) = submit(&mut s, greedy(random_prompt(&mut rng, 10), 5)); // runs normally
    let (c, rxc, fc) = submit(&mut s, greedy(random_prompt(&mut rng, 10), 5)); // cancelled while queued
    s.step(); // b: first 256 chunk
    s.step(); // b: second chunk (d waits behind b: FIFO prefill)
    fc.store(true, Ordering::SeqCst);
    fb.store(true, Ordering::SeqCst);
    s.step();
    for _ in 0..3 {
        s.step();
    }
    fa.store(true, Ordering::SeqCst);
    run_to_end(&mut s);
    let tr = s.take_trace();
    assert!(!tr.iter().any(|e| matches!(e, TraceEvent::Admitted { id, .. } if *id == c)), "queued cancel took a slot");
    let ends: HashMap<RequestId, EndKind> = tr
        .iter()
        .filter_map(|e| if let TraceEvent::Finished { id, how, .. } = e { Some((*id, *how)) } else { None })
        .collect();
    assert_eq!((ends[&a], ends[&b], ends[&c]), (EndKind::Cancelled, EndKind::Cancelled, EndKind::Cancelled));
    let last = |rx: &Receiver<Ev>| collect(rx).pop().unwrap();
    assert!(matches!(last(&rxa), Ev::Done { reason: FinishReason::Cancelled, .. }));
    assert!(matches!(last(&rxb), Ev::Done { reason: FinishReason::Cancelled, completion: 0, .. }));
    assert!(matches!(last(&rxc), Ev::Done { reason: FinishReason::Cancelled, completion: 0, .. }));
    assert_eq!(tokens(&collect(&rxd)).len(), 5);
    // b's slot holds exactly the prompt prefix it prefilled (two 256-token chunks), reusable by a later request
    let m = s.model();
    let slot_b = (0..3).find(|&i| m.slot_tokens(i).len() > 100 && m.slot_tokens(i)[..10] == pb[..10]).unwrap();
    assert_eq!(m.slot_tokens(slot_b), &pb[..512]);
}

/// Failure isolation: a row error, a sequence error and a prefill error fail only their request and invalidate its
/// slot; a whole-step failure fails everything in flight and refuses later submits.
#[test]
fn failure_isolation_and_fatal() {
    let cfg = SchedulerConfig { max_batch: 3, trace: true, ..Default::default() };
    let mut m = MockModel::new(3, 8192);
    m.fail_row.push((0, 3));
    m.fail_prefill.push(2);
    let mut s = Scheduler::new(m, cfg).unwrap();
    let mut rng = SplitMix64::new(6);
    let (_, rx0, _) = submit(&mut s, greedy(random_prompt(&mut rng, 30), 20));
    let mut p1 = greedy(random_prompt(&mut rng, 30), 20);
    p1.fail_at = Some(4);
    let (_, rx1, _) = submit(&mut s, p1);
    let (_, rx2, _) = submit(&mut s, greedy(random_prompt(&mut rng, 30), 20));
    let (_, rx3, _) = submit(&mut s, greedy(random_prompt(&mut rng, 30), 20));
    run_to_end(&mut s);
    for rx in [&rx0, &rx1, &rx2] {
        assert!(matches!(collect(rx).pop(), Some(Ev::Err(_))));
    }
    let ev3 = collect(&rx3);
    assert_eq!(tokens(&ev3).len(), 20);
    assert!(matches!(ev3.last(), Some(Ev::Done { reason: FinishReason::Length, completion: 20, .. })));
    // failed slots were invalidated; only the survivor's slot holds tokens
    let held: Vec<usize> = (0..3).map(|i| s.model().slot_tokens(i).len()).collect();
    assert_eq!(held.iter().filter(|&&l| l > 0).count(), 1, "{held:?}");
    assert_eq!(s.metrics().failed, 3);

    // fatal
    let mut m = MockModel::new(3, 8192);
    m.fail_step_at = Some(5);
    let mut s = Scheduler::new(m, SchedulerConfig { max_batch: 3, ..Default::default() }).unwrap();
    let rxs: Vec<_> = (0..5).map(|_| submit(&mut s, greedy(random_prompt(&mut rng, 30), 50)).1).collect();
    run_to_end(&mut s);
    assert!(s.failed().is_some());
    for rx in &rxs {
        assert!(matches!(collect(rx).pop(), Some(Ev::Err(_))));
    }
    let (seq, rx, _) = MockSeq::new(greedy(vec![4, 5], 3));
    assert!(s.submit(seq).is_err());
    assert!(matches!(collect(&rx).pop(), Some(Ev::Err(_))));
}

/// Queue overflow: beyond max_queue waiting requests, submit fails at once (with an Error event); the rest complete.
#[test]
fn queue_overflow() {
    let cfg = SchedulerConfig { max_batch: 1, max_queue: 3, ..Default::default() };
    let mut s = Scheduler::new(MockModel::new(1, 8192), cfg).unwrap();
    let mut ok = Vec::new();
    let mut refused = 0;
    for i in 0..6 {
        let (seq, rx, _) = MockSeq::new(greedy(vec![5 + i; 8], 3));
        match s.submit(seq) {
            Ok(_) => ok.push(rx),
            Err(_) => {
                refused += 1;
                assert!(matches!(collect(&rx).pop(), Some(Ev::Err(e)) if e.contains("queue full")));
            }
        }
    }
    assert_eq!((ok.len(), refused), (3, 3));
    s.step(); // one admitted: room for one more
    let (seq, rx, _) = MockSeq::new(greedy(vec![9; 8], 3));
    assert!(s.submit(seq).is_ok());
    ok.push(rx);
    run_to_end(&mut s);
    for rx in ok {
        assert_eq!(tokens(&collect(&rx)).len(), 3);
    }
    let m = s.metrics();
    assert_eq!((m.rejected, m.queue_depth_max, m.finished), (3, 3, 4));
}

/// Speculation hook: offered while fewer than spec_off_at sequences decode, never at or above it.
#[test]
fn spec_gate_follows_batch_size() {
    for off_at in [4usize, 2, 0] {
        let cfg = SchedulerConfig { max_batch: 6, spec_off_at: off_at, trace: true, ..Default::default() };
        let mut s = Scheduler::new(MockModel::new(6, 8192), cfg).unwrap();
        let mut logs = Vec::new();
        let mut rxs = Vec::new();
        for i in 0..6u32 {
            let (seq, rx, _) = MockSeq::new(greedy(vec![7 + i; 5], 12 + 3 * i as usize));
            logs.push(seq.spec_log.clone());
            s.submit(seq).unwrap();
            rxs.push(rx);
        }
        run_to_end(&mut s);
        let mut seen = (false, false);
        for e in s.take_trace() {
            if let TraceEvent::Decode { ids, spec_allowed, .. } = e {
                assert_eq!(spec_allowed, ids.len() < off_at);
                if spec_allowed {
                    seen.0 = true
                } else {
                    seen.1 = true
                }
            }
        }
        if off_at == 4 {
            assert!(seen.0 && seen.1, "both gate values exercised");
        }
        if off_at == 0 {
            assert!(!seen.0);
        }
        assert!(logs.iter().all(|l| !l.lock().unwrap().is_empty()));
    }
}

/// Real threads: several submitters and cancellers against the engine thread; every request still equals its solo
/// reference (cancelled ones a prefix of it), and nothing hangs.
#[test]
fn engine_concurrent_submit_and_cancel() {
    let cap = 4096;
    let model = MockModel::new(4, cap);
    let rec = model.rec.clone();
    let cfg = SchedulerConfig { max_batch: 4, max_queue: 1000, ..Default::default() };
    let engine = Arc::new(BatchEngine::<MockModel, MockSeq>::spawn(model, cfg.clone()).unwrap());
    let results: Arc<Mutex<Vec<(Params, RequestId, Vec<Ev>, bool)>>> = Arc::new(Mutex::new(Vec::new()));
    let mut handles = Vec::new();
    for t in 0..4u64 {
        let engine = engine.clone();
        let results = results.clone();
        handles.push(std::thread::spawn(move || {
            let mut rng = SplitMix64::new(100 + t);
            let mut mine = Vec::new();
            for _ in 0..10 {
                let len = 1 + (rng.next_u64() % 2500) as usize;
                let p = Params {
                    prompt: random_prompt(&mut rng, len),
                    max_tokens: 1 + (rng.next_u64() % 60) as usize,
                    seed: if rng.next_u64() % 2 == 0 { Some(rng.next_u64()) } else { None },
                    ..Default::default()
                };
                let (seq, rx, flag) = MockSeq::new(p.clone());
                let id = engine.submit(seq).unwrap();
                let kind = rng.next_u64() % 6;
                if kind == 0 {
                    // cancel from another thread after a while
                    let f = flag.clone();
                    let d = rng.next_u64() % 3;
                    std::thread::spawn(move || {
                        std::thread::sleep(Duration::from_millis(d));
                        f.store(true, Ordering::SeqCst);
                    });
                }
                mine.push((p, id, rx, kind == 0, kind == 1));
                std::thread::sleep(Duration::from_micros(rng.next_u64() % 500));
            }
            for (p, id, rx, cancelled, drop_early) in mine {
                let mut evs = Vec::new();
                loop {
                    if drop_early && evs.len() >= 2 {
                        break; // client disconnect: drop the receiver
                    }
                    match rx.recv_timeout(Duration::from_secs(20)) {
                        Ok(e) => {
                            let end = matches!(e, Ev::Done { .. } | Ev::Err(_));
                            evs.push(e);
                            if end {
                                break;
                            }
                        }
                        Err(e) => panic!("request {id}: no event within 20 s ({e:?})"),
                    }
                }
                results.lock().unwrap().push((p, id, evs, cancelled || drop_early));
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    // wait for the engine to settle (dropped receivers finish at their next token)
    let t0 = std::time::Instant::now();
    while engine.metrics().running > 0 || engine.metrics().queue_depth > 0 {
        assert!(t0.elapsed() < Duration::from_secs(20), "engine did not drain");
        std::thread::sleep(Duration::from_millis(5));
    }
    let m = engine.metrics();
    assert_eq!(m.submitted, 40);
    assert_eq!(m.batch_hist.iter().sum::<u64>(), m.decode_steps);
    assert!(engine.failed().is_none());
    let rec = rec.lock().unwrap();
    let results = results.lock().unwrap();
    let mut mismatches = 0;
    for (p, id, evs, cut) in results.iter() {
        let Some(snapshot) = rec.snapshots.get(id).cloned() else {
            assert!(*cut && tokens(evs).is_empty());
            continue;
        };
        let chunks = rec.chunks.get(id).cloned().unwrap_or_default();
        // C as recorded: the first chunk's length (any C >= the length for a single chunk gives the same schedule)
        let c = chunks.first().map_or(2048, |c| c.1);
        let reference = solo(p, snapshot, c, cap, None, false);
        if *cut {
            let (t, rt) = (tokens(evs), tokens(&reference.events));
            if t.len() > rt.len() || t[..] != rt[..t.len()] {
                mismatches += 1;
            }
        } else if *evs != reference.events || chunks != reference.chunks {
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, 0);
    drop(results);
    drop(rec);
    // shutdown with work in flight: the drop cancels and joins
    let (seq, rx, _) = MockSeq::new(greedy(vec![5; 100], 4000 - 101));
    engine.submit(seq).unwrap();
    std::thread::sleep(Duration::from_millis(2));
    drop(Arc::try_unwrap(engine).ok().expect("sole owner"));
    // the engine thread is joined: the request ended (cancelled by the shutdown, or done if it was that fast)
    let evs: Vec<Ev> = rx.try_iter().collect();
    assert!(matches!(evs.last(), Some(Ev::Done { .. })), "{:?}", evs.last());
}

/// Engine-level queue bound and fatal failure.
#[test]
fn engine_queue_bound_and_fatal() {
    let mut model = MockModel::new(1, 4096);
    model.fail_step_at = Some(3);
    let cfg = SchedulerConfig { max_batch: 1, max_queue: 2, ..Default::default() };
    let engine = BatchEngine::<MockModel, MockSeq>::spawn(model, cfg).unwrap();
    let mut rxs = Vec::new();
    let mut refused = 0;
    for _ in 0..50 {
        let (seq, rx, _) = MockSeq::new(greedy(vec![6; 3000], 50));
        match engine.submit(seq) {
            Ok(_) => rxs.push(rx),
            Err(_) => refused += 1,
        }
    }
    assert!(refused > 0, "queue bound never hit");
    for rx in rxs {
        let last = loop {
            match rx.recv_timeout(Duration::from_secs(20)).expect("event") {
                e @ (Ev::Done { .. } | Ev::Err(_)) => break e,
                Ev::Tok(_) => {}
            }
        };
        assert!(matches!(last, Ev::Err(_)), "{last:?}");
    }
    assert!(engine.failed().is_some());
    let (seq, rx, _) = MockSeq::new(greedy(vec![6; 3], 5));
    assert!(engine.submit(seq).is_err());
    assert!(matches!(rx.try_iter().last(), Some(Ev::Err(_))));
}

// ---------------------------------------------------------------------------
// FIFO fairness (design 2.2 "Fairness", operator decision 6, M2 gate "no admitted request waits behind a later one")
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum St {
    Queued,
    Prefilling,
    Decoding,
    Gone,
}

#[derive(Default, Debug)]
struct FifoCov {
    admitted: usize,
    reused: usize,
    /// Chunks run while a younger request was prefilling too (the oldest-first choice mattered).
    contested_chunks: usize,
    /// Requests admitted at least one iteration after they were submitted (they queued behind a full house or a
    /// predecessor); needs `arrive` to be filled in.
    waited: usize,
    cancel_queued: usize,
    cancel_prefill: usize,
    cancel_decode: usize,
    failed_prefill: usize,
}

/// Replay a scheduler trace and check the fairness properties. Ids are assigned in submit order, so "older" is "lower id".
///
/// 1. A request is admitted only when every older request has left the queue (admitted, cancelled while queued, or
///    rejected): nothing admitted waits behind a later one, with or without prefix reuse and cancellations.
/// 2. A prefill chunk goes to the oldest prefilling request: no chunk while an older request is still prefilling.
/// 3. First tokens (final chunks) come in submit order.
fn check_fifo(trace: &[TraceEvent], n: usize, arrive: &[u64]) -> FifoCov {
    let mut st = vec![St::Queued; n];
    let mut cov = FifoCov::default();
    let mut last_first: Option<RequestId> = None;
    for e in trace {
        match e {
            TraceEvent::Admitted { iter, id, cached, .. } => {
                let i = *id as usize;
                assert_eq!(st[i], St::Queued, "request {id} admitted twice");
                if let Some(j) = (0..i).find(|&j| st[j] == St::Queued) {
                    panic!("request {id} admitted while older request {j} still waits");
                }
                if arrive.get(i).is_some_and(|&a| *iter > a + 1) {
                    cov.waited += 1;
                }
                st[i] = St::Prefilling;
                cov.admitted += 1;
                cov.reused += usize::from(*cached > 0);
            }
            TraceEvent::Chunk { id, last, .. } => {
                let i = *id as usize;
                assert_eq!(st[i], St::Prefilling, "chunk for request {id} that is not prefilling");
                if let Some(j) = (0..i).find(|&j| st[j] == St::Prefilling) {
                    panic!("chunk for request {id} while older request {j} is still prefilling");
                }
                if (i + 1..n).any(|j| st[j] == St::Prefilling) {
                    cov.contested_chunks += 1;
                }
                if *last {
                    assert!(last_first.map_or(true, |p| p < *id), "first tokens out of order: {last_first:?} then {id}");
                    last_first = Some(*id);
                    st[i] = St::Decoding;
                }
            }
            TraceEvent::Finished { id, how, .. } => {
                let i = *id as usize;
                if *how == EndKind::Cancelled {
                    match st[i] {
                        St::Queued => cov.cancel_queued += 1,
                        St::Prefilling => cov.cancel_prefill += 1,
                        St::Decoding => cov.cancel_decode += 1,
                        St::Gone => panic!("request {id} finished twice"),
                    }
                }
                if *how == EndKind::Failed && st[i] == St::Prefilling {
                    cov.failed_prefill += 1;
                }
                st[i] = St::Gone;
            }
            TraceEvent::PrefillStart { .. } | TraceEvent::Decode { .. } => {}
        }
    }
    assert!(st.iter().all(|&s| s == St::Gone), "requests left unfinished: {st:?}");
    cov
}

/// Pressure: a short arrival window so that several requests are queued and prefilling at once, and cancels that land
/// while a request is queued, prefilling and decoding.
fn pressure_reqs(rng: &mut SplitMix64, n: usize, cap: usize) -> Vec<Req> {
    let mut reqs = random_reqs(rng, n, cap);
    for r in &mut reqs {
        let a = r.arrive;
        r.arrive = a / 4;
        r.cancel_at = r.cancel_at.map(|c| r.arrive + (c - a) / 3);
        r.drop_rx_at = r.drop_rx_at.map(|c| r.arrive + (c - a) / 3);
    }
    reqs
}

#[test]
fn fifo_fairness_randomized() {
    let cap = 4096;
    let mut total = FifoCov::default();
    let mut runs = 0;
    for avoid_flag in [false, true] {
        with_avoid(avoid_flag, || {
            for b in 1..=8usize {
                for per_iter in 1..=3usize {
                    for reuse in [true, false] {
                        for seed in 0..2u64 {
                            let mut rng = SplitMix64::new(9000 + seed * 100 + (b * 10 + per_iter) as u64);
                            let reqs = pressure_reqs(&mut rng, 30, cap);
                            let cfg = SchedulerConfig {
                                max_batch: b,
                                max_queue: 1000,
                                busy_chunks_per_iter: per_iter,
                                prefix_reuse: reuse,
                                ..Default::default()
                            };
                            let run = run_batched(cfg.clone(), cap, &reqs, false);
                            check_trace(&run, &cfg);
                            let mut arrive = vec![0u64; reqs.len()];
                            for (r, id) in reqs.iter().zip(&run.ids) {
                                arrive[id.unwrap() as usize] = r.arrive;
                            }
                            let c = check_fifo(&run.trace, reqs.len(), &arrive);
                            assert_eq!(c.admitted as u64, run.metrics.admitted, "B = {b}, seed {seed}");
                            assert!(reuse || c.reused == 0, "prefix reuse off but a request reused KV");
                            total.admitted += c.admitted;
                            total.reused += c.reused;
                            total.contested_chunks += c.contested_chunks;
                            total.waited += c.waited;
                            total.cancel_queued += c.cancel_queued;
                            total.cancel_prefill += c.cancel_prefill;
                            total.cancel_decode += c.cancel_decode;
                            total.failed_prefill += c.failed_prefill;
                            runs += 1;
                        }
                    }
                }
            }
        });
    }
    eprintln!("fifo coverage over {runs} runs: {total:?}");
    // the properties were exercised, not vacuous
    assert!(total.contested_chunks >= 200, "{total:?}");
    assert!(total.waited >= 200, "{total:?}");
    assert!(total.reused >= 100, "{total:?}");
    assert!(total.cancel_queued >= 30 && total.cancel_prefill >= 30 && total.cancel_decode >= 30, "{total:?}");
    assert!(total.failed_prefill >= 5, "{total:?}");
}

/// Negative control for the checker itself: traces that overtake are rejected.
#[test]
fn fifo_checker_rejects_overtaking() {
    use std::panic::catch_unwind;
    let adm = |id, slot| TraceEvent::Admitted { iter: 1, id, slot, cached: 0 };
    let chunk = |id, last| TraceEvent::Chunk { iter: 1, id, start: 0, len: 10, last };
    let fin = |id, how| TraceEvent::Finished { iter: 2, id, how };
    // fine: 0 cancelled while queued, 1 and 2 in order
    let ok = vec![
        fin(0, EndKind::Cancelled),
        adm(1, 0),
        adm(2, 1),
        chunk(1, true),
        chunk(2, true),
        fin(1, EndKind::Done),
        fin(2, EndKind::Done),
    ];
    let c = check_fifo(&ok, 3, &[]);
    assert_eq!((c.cancel_queued, c.admitted), (1, 2));
    // 1 admitted while 0 still waits
    let t = vec![adm(1, 0), adm(0, 1), fin(0, EndKind::Done), fin(1, EndKind::Done)];
    assert!(catch_unwind(|| check_fifo(&t, 2, &[])).is_err(), "admission overtake not caught");
    // a younger request's chunk while an older one is prefilling
    let t = vec![adm(0, 0), adm(1, 1), chunk(1, false), chunk(0, true), chunk(1, true)];
    assert!(catch_unwind(|| check_fifo(&t, 2, &[])).is_err(), "prefill overtake not caught");
    // a chunk after the request's final chunk
    let t = vec![adm(0, 0), adm(1, 1), fin(0, EndKind::Cancelled), chunk(1, true), adm(2, 0), chunk(2, true), chunk(1, true)];
    assert!(catch_unwind(|| check_fifo(&t, 3, &[])).is_err(), "chunk after the final chunk not caught");
}

/// The queue-depth gauge is live: requests submitted while the scheduler thread is inside a long step (a big prefill
/// chunk) are counted at once, they do not appear only when the thread next drains its channel (soak2 shedding burst:
/// `intarsia_requests_waiting` 7 against a sampled `intarsia_batch_queue_depth` 4).
#[test]
fn queue_depth_is_live_during_a_long_step() {
    /// Opens the gate when dropped, so a failing assert does not leave the scheduler thread blocked in `Drop`.
    struct OpenOnDrop(Arc<(Mutex<bool>, Condvar)>);
    impl Drop for OpenOnDrop {
        fn drop(&mut self) {
            *self.0 .0.lock().unwrap() = true;
            self.0 .1.notify_all();
        }
    }
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let mut model = MockModel::new(2, 4096);
    model.gate = Some(gate.clone());
    let cfg = SchedulerConfig { max_batch: 2, max_queue: 5, ..Default::default() };
    let engine = BatchEngine::<MockModel, MockSeq>::spawn(model, cfg).unwrap();
    let _guard = OpenOnDrop(gate.clone()); // declared after `engine`, so it drops first
    let mut rxs = Vec::new();
    let mut go = |engine: &BatchEngine<MockModel, MockSeq>, n: usize| {
        let (seq, rx, _) = MockSeq::new(greedy(vec![6; n], 3));
        let r = engine.submit(seq);
        rxs.push(rx);
        r
    };
    go(&engine, 10).unwrap();
    // the scheduler thread admits it (the queue counter drops) and then blocks in its prefill chunk
    let t0 = std::time::Instant::now();
    while engine.metrics().queue_depth != 0 || engine.metrics().admitted == 0 {
        assert!(t0.elapsed() < Duration::from_secs(10), "request never admitted");
        std::thread::sleep(Duration::from_millis(1));
    }
    std::thread::sleep(Duration::from_millis(20)); // the thread is now inside the gated chunk
    for k in 1..=5 {
        go(&engine, 10).unwrap();
        assert_eq!(engine.metrics().queue_depth, k, "gauge lags behind the {k} submits");
    }
    assert!(go(&engine, 10).is_err(), "bound is max_queue = 5");
    let m = engine.metrics();
    assert_eq!((m.queue_depth, m.queue_depth_max, m.rejected), (5, 5, 1), "a refused request is never counted");
    {
        let mut open = gate.0.lock().unwrap();
        *open = true;
        gate.1.notify_all();
    }
    for rx in rxs.iter().take(6) {
        loop {
            match rx.recv_timeout(Duration::from_secs(20)).expect("event") {
                Ev::Done { .. } => break,
                Ev::Err(e) => panic!("{e}"),
                Ev::Tok(_) => {}
            }
        }
    }
    let m = engine.metrics();
    assert_eq!((m.queue_depth, m.queue_depth_max), (0, 5));
    assert_eq!(m.mixed_iterations > 0 || m.prefill_chunks >= 6, true);
    assert!(m.chunk.count >= 6, "chunk time recorded: {}", m.chunk.count);
}
