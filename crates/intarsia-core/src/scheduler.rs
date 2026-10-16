//! Continuous-batching scheduler core (docs/research/continuous-batching-design.md sections 2, 3.5, 6, 8.1): slots,
//! FIFO admission, decode-first iteration-level batching, chunked-prefill interleave, idle-slot prefix reuse,
//! per-request cancellation and failure isolation, the speculation on/off hook and per-iteration metrics.
//!
//! The model and the per-request logic sit behind two small traits so the policy is testable without a GPU:
//!
//! * [`BatchModel`]: owns `max_batch` KV slots. Prefill one chunk into one slot, one batched decode step over several
//!   slots, truncate a slot, read the tokens a slot's KV holds. The real adapter wraps `Model` + one `SequenceState`
//!   per slot (`Model::forward` for chunks, `Model::forward_batch` for steps).
//! * [`BatchSeq`]: one request's state machine (sampling, `min_tokens`, stop ids / strings, EOS, `max_tokens`,
//!   streaming, cancellation, its own `Done` / `Error` event). The real adapter wraps `engine::Seq`.
//!
//! # Exactness contract (operator decision 8.1: exact only)
//!
//! A request's token stream must not depend on what else runs. The scheduler guarantees its half by construction:
//!
//! 1. A sequence only ever sees its own rows: [`BatchSeq::accept`] gets exactly the output of the row (or final
//!    chunk) that fed that sequence's own input, and nothing else about the batch (not `B`, not the row index).
//! 2. Its slot is truncated to `cached` at admission and is written by nobody else while the sequence runs.
//! 3. Its prefill chunk schedule is a function of `(cached, prompt_len, C)` and the backend's
//!    `avoid_single_row_prefill` flag only, where `C` is chosen once, when its first chunk runs (see [`chunk_schedule`]
//!    and [`SchedulerConfig::busy_chunk`]), and never changes afterwards. The flag (Metal: true) only turns a
//!    `.., C, 1` tail into `.., C - 1, 2` ([`next_prefill_chunk`]), so that a request's last row, which picks the first
//!    token, is never computed as a 1-row chunk (decode kernels) in one schedule and inside a bigger chunk in another.
//!    Decode rows and prefill chunks never share a forward pass (design 2.4, 4.5 item 3).
//!
//! The model's half ([`BatchModel`] docs): the output of row `r` of [`BatchModel::decode_step`] is a function of slot
//! `rows[r].slot`'s KV contents, `rows[r].token` and `rows[r].mode` only, bitwise; likewise a chunk's output.
//!
//! Consequently the **solo reference** of a batched request is: the same request run alone (one slot, nothing else
//! submitted), starting from a slot whose KV holds exactly the `cached`-token prefix the batched run reused (same
//! values, including whether those rows were prefill- or decode-written, design 4.5 items 2 and 4), prefilled with
//! `--prefill-chunk C` where `C` is the chunk size the batched run recorded for it ([`TraceEvent::PrefillStart`]).
//! With the default policy `C` is 2048 (`solo_chunk`) when no other sequence was decoding as its prefill began and
//! 256 (`busy_chunk`) otherwise; a request with `max_batch = 1` therefore always gets today's 2048 schedule.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::error::{Error, Result};

/// Hard cap on `max_batch` (operator decision 8.1: exact kernels only, 8 rows; `Model::BATCH_MAX_ROWS`).
pub const MAX_BATCH_CAP: usize = 8;

/// Identifies one submitted request (assigned at submit, increasing).
pub type RequestId = u64;

/// One row of a batched decode step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Row<Mode> {
    /// Slot whose KV the row reads and appends to (at that slot's own current length).
    pub slot: usize,
    /// Token fed at that position.
    pub token: u32,
    /// What the row returns (real model: `HeadMode::LastArgmax` or `HeadMode::LastLogits`).
    pub mode: Mode,
}

/// The model side of the scheduler: `max_batch` KV slots and the passes over them.
///
/// Contract (exactness): every output is a function of the inputs named here and of the addressed slot's KV contents,
/// bitwise, and of nothing else: not of other slots, the number of rows, the row order or the call history of other
/// slots. A slot's KV contents are only changed by calls that name that slot.
pub trait BatchModel {
    /// Per-row head mode (real model: `HeadMode`).
    type Mode: Copy + std::fmt::Debug;
    /// Per-row output (real model: `StepOut`).
    type Out;

    /// Number of slots (fixed for the model's lifetime, `>= max_batch`).
    fn slots(&self) -> usize;
    /// Token capacity of each slot (uniform, design 3.3).
    fn slot_cap(&self) -> usize;
    /// The tokens whose KV slot `slot` holds, positions `0..len` (the prefix-reuse candidate).
    fn slot_tokens(&self, slot: usize) -> &[u32];
    /// Shorten slot `slot` to its first `len` tokens (`SequenceState::truncate`). `len = 0` invalidates it.
    fn truncate(&mut self, slot: usize, len: usize) -> Result<()>;
    /// Prefill `tokens` into slot `slot` at its current length, as one forward pass of exactly `tokens.len()` rows
    /// (the chunk size is part of numerical identity, design 4.5 item 1). `mode = None`: KV only, returns `None`;
    /// `Some(m)`: the last position's head output in mode `m` (the request's first token).
    fn prefill_chunk(&mut self, slot: usize, tokens: &[u32], mode: Option<Self::Mode>) -> Result<Option<Self::Out>>;
    /// One decode token for each row in one batched pass (`1..=max_batch` rows, distinct slots); each slot grows by
    /// one. Inner `Err`: that row failed alone (the request fails, the batch goes on). Outer `Err`: the pass failed as
    /// a whole; the scheduler treats it as fatal for every in-flight request (design 2.5).
    fn decode_step(&mut self, rows: &[Row<Self::Mode>]) -> Result<Vec<Result<Self::Out>>>;
    /// Whether an error from a single-slot call ([`BatchModel::prefill_chunk`], an inner row error) means the device
    /// is unusable, so every in-flight request must fail (default: backend errors).
    fn is_fatal(&self, e: &Error) -> bool {
        matches!(e, Error::Backend(_))
    }
    /// Whether prefill chunking avoids a 1-row tail ([`next_prefill_chunk`]); default false (plain `C, C, .., rest`).
    /// The real adapter returns `Backend::avoid_single_row_prefill()` and false for models with recurrent state (their
    /// last token deliberately runs alone, see `engine::Seq::prefill_step`).
    fn avoid_single_row_prefill(&self) -> bool {
        false
    }
    /// Trace hook: request `id` was admitted to `slot`, which now holds `cached` reused tokens. No effect on outputs.
    fn on_admit(&mut self, _slot: usize, _id: RequestId, _cached: usize) {}
}

/// What a decoding sequence needs this iteration ([`BatchSeq::prepare`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeqStep<Mode> {
    /// A plain decode row: feed `token`, run the head in `mode`, hand the row's output to [`BatchSeq::accept`].
    Plain(u32, Mode),
    /// The next token is already known (a verified speculative run): no row, go straight to [`BatchSeq::on_token`].
    Ready,
}

/// The per-request side of the scheduler. One value per submitted request; it streams its own tokens and sends its own
/// final `Done` / `Error` event.
///
/// Contract (exactness): its behaviour is a function of its own request, the `cached` length it is admitted with and
/// the outputs it is handed; it never sees which rows or slots run beside it.
pub trait BatchSeq<M: BatchModel>: Send {
    /// The prompt tokens.
    fn prompt(&self) -> &[u32];
    /// Admission into a slot with `cached` reused tokens (`< prompt.len()`) and capacity `cap`. `Err`: rejected (the
    /// scheduler calls [`BatchSeq::fail`], the slot is untouched). `Ok(false)`: nothing to generate (no room or
    /// `max_tokens = 0`; the sequence sets its finish reason to `Length`, the scheduler calls [`BatchSeq::finish`]
    /// at once). `Ok(true)`: prefill follows.
    fn admit(&mut self, cached: usize, cap: usize) -> Result<bool>;
    /// Head mode of the final prefill chunk (the step that samples generated token 1).
    fn first_mode(&self) -> M::Mode;
    /// The scheduler chose prefill chunk size `chunk` for this request (once, at its first chunk) while `decoding`
    /// other sequences were decoding. Informational (the harness reads it back to pick the matching solo reference).
    fn on_prefill_start(&mut self, _chunk: usize, _decoding: usize) {}
    /// Plan this iteration's step. `spec_allowed` is the scheduler's speculation gate (false when `B >= spec_off_at`).
    /// A speculating sequence may run its verify pass here on its own slot (single-slot, exact by the G1 argument)
    /// and return `Ready`; it must not draft when `spec_allowed` is false.
    fn prepare(&mut self, model: &mut M, slot: usize, spec_allowed: bool) -> Result<SeqStep<M::Mode>>;
    /// The output of this sequence's row or final chunk: becomes the next token.
    fn accept(&mut self, out: M::Out) -> Result<()>;
    /// Account for the next token (stop ids / EOS, stop strings, streaming, `max_tokens`, client gone). `true`: done.
    fn on_token(&mut self) -> Result<bool>;
    /// Cancellation requested (flag set, or the client went away).
    fn cancelled(&self) -> bool;
    /// The request ended: roll back anything unaccounted in `slot` (speculative rows), send `Done`. `slot` is `None`
    /// when it never got one (cancelled while queued). `cancelled`: ended by the scheduler's cancellation check (the
    /// sequence may be in any phase). With `cancelled == false` it is only called after [`BatchSeq::on_token`]
    /// returned true or [`BatchSeq::admit`] returned `Ok(false)`, so the sequence has its finish reason set.
    fn finish(&mut self, model: &mut M, slot: Option<usize>, cancelled: bool) -> Result<()>;
    /// The request failed: send `Error`.
    fn fail(&mut self, err: &Error);
}

/// Scheduler policy. Defaults are the operator decisions (design 8.1 and the 2026-10-15 scheduler confirmation).
#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    /// Slots / most rows per decode step (`--max-batch`), `1..=MAX_BATCH_CAP`. Default 1: batching is opt-in.
    pub max_batch: usize,
    /// Most requests waiting for a slot; a submit beyond it is rejected.
    pub max_queue: usize,
    /// Prefill chunk of a sequence that starts its prefill while no other sequence is decoding (model chunk, 2048 on
    /// Metal): today's solo behaviour and golden baselines.
    pub solo_chunk: usize,
    /// Prefill chunk of a sequence that starts its prefill while at least one other sequence is decoding (256). The
    /// choice is made once per sequence, at its first chunk, and kept to its last chunk.
    pub busy_chunk: usize,
    /// Prefill chunks per iteration while sequences are decoding (prefill budget = `busy_chunk` x this). Only the
    /// final chunk of a prompt is shorter than its `C`; the budget never splits a chunk.
    pub busy_chunks_per_iter: usize,
    /// Speculation is offered to sequences only while fewer than this many sequences decode (design 6: off at B >= 4).
    /// 0 disables it entirely.
    pub spec_off_at: usize,
    /// Keep a [`TraceEvent`] log (tests, the identity harness).
    pub trace: bool,
    /// Offer idle slots' KV prefixes for reuse at admission (default). Off: every request starts from an empty slot
    /// (the concurrent-identity harness compares against cold-prefill solo runs).
    pub prefix_reuse: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        SchedulerConfig {
            max_batch: 1,
            max_queue: 64,
            solo_chunk: 2048,
            busy_chunk: 256,
            busy_chunks_per_iter: 1,
            spec_off_at: 4,
            trace: false,
            prefix_reuse: true,
        }
    }
}

impl SchedulerConfig {
    pub fn validate(&self) -> Result<()> {
        if self.max_batch == 0 || self.max_batch > MAX_BATCH_CAP {
            return Err(Error::BadRequest(format!("max_batch {} outside 1..={MAX_BATCH_CAP}", self.max_batch)));
        }
        if self.solo_chunk == 0 || self.busy_chunk == 0 || self.busy_chunks_per_iter == 0 {
            return Err(Error::BadRequest("prefill chunk sizes and chunks per iteration must be >= 1".into()));
        }
        Ok(())
    }
}

/// Length of the next prefill chunk when `remaining` (>= 1) prompt tokens are left and chunks are at most `chunk`.
///
/// The rule: `chunk` while more than `chunk` tokens remain, then the rest (1..=`chunk`) as the final chunk. With
/// `avoid_one_row` (backends where a 1-row chunk is not bitwise equal to the same row inside a bigger chunk, see
/// `Backend::avoid_single_row_prefill`) one exception: when exactly `chunk + 1` tokens remain, take `chunk - 1` so the
/// final chunk has 2 rows instead of 1, i.e. a prompt of `k * chunk + 1` tokens (`k >= 1`) ends `.., chunk - 1, 2`
/// instead of `.., chunk, 1`. Every chunk stays <= `chunk` (workspace is sized for it). Unavoidable cases keep the
/// plain rule: `chunk <= 2` (the split would itself make a 1-row chunk) and a whole remainder of 1 row (nothing precedes
/// it, e.g. a prefix hit capped at `prompt_len - 1`). With `avoid_one_row == false` the result is the plain rule.
///
/// Shared by `engine::Seq::prefill_step` and [`chunk_schedule`] so solo and batched schedules cannot drift.
pub fn next_prefill_chunk(remaining: usize, chunk: usize, avoid_one_row: bool) -> usize {
    debug_assert!(remaining >= 1 && chunk >= 1);
    if remaining <= chunk {
        remaining
    } else if avoid_one_row && chunk >= 3 && remaining == chunk + 1 {
        chunk - 1
    } else {
        chunk
    }
}

/// The prefill chunk schedule of a request: `(start, len)` of every forward pass, the last one carrying the head.
/// Same rule as `engine::Seq::prefill_step` ([`next_prefill_chunk`]): from `cached`, full chunks of `chunk` while more
/// than `chunk` tokens remain, then the rest (1..=`chunk` tokens) as the final chunk; with `avoid_one_row` a
/// `chunk + 1` remainder is split `(chunk - 1, 2)` instead of `(chunk, 1)`.
pub fn chunk_schedule(cached: usize, prompt_len: usize, chunk: usize, avoid_one_row: bool) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut pos = cached;
    while pos < prompt_len {
        let n = next_prefill_chunk(prompt_len - pos, chunk, avoid_one_row);
        out.push((pos, n));
        pos += n;
    }
    out
}

fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Scheduler trace (only with [`SchedulerConfig::trace`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TraceEvent {
    Admitted {
        iter: u64,
        id: RequestId,
        slot: usize,
        cached: usize,
    },
    /// The request's first chunk is about to run with chunk size `chunk`; `decoding` sequences were decoding.
    PrefillStart {
        iter: u64,
        id: RequestId,
        chunk: usize,
        decoding: usize,
    },
    Chunk {
        iter: u64,
        id: RequestId,
        start: usize,
        len: usize,
        last: bool,
    },
    /// One batched decode step over these requests (row order), with the speculation gate value offered.
    Decode {
        iter: u64,
        ids: Vec<RequestId>,
        spec_allowed: bool,
    },
    Finished {
        iter: u64,
        id: RequestId,
        how: EndKind,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EndKind {
    Done,
    Cancelled,
    Rejected,
    Failed,
    /// Failed because the engine failed (fatal step error).
    Fatal,
}

/// Fixed-bucket latency histogram (milliseconds).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LatencyHist {
    /// Counts per bucket: `<= BOUNDS_MS[i]`, the last entry is `> BOUNDS_MS[last]`.
    pub counts: Vec<u64>,
    pub count: u64,
    pub sum_ms: f64,
    pub max_ms: f64,
}

impl LatencyHist {
    pub const BOUNDS_MS: [f64; 13] = [1.0, 2.0, 5.0, 10.0, 20.0, 35.0, 50.0, 75.0, 100.0, 150.0, 250.0, 500.0, 1000.0];
    pub fn record(&mut self, d: Duration) {
        if self.counts.is_empty() {
            self.counts = vec![0; Self::BOUNDS_MS.len() + 1];
        }
        let ms = d.as_secs_f64() * 1e3;
        let i = Self::BOUNDS_MS.iter().position(|&b| ms <= b).unwrap_or(Self::BOUNDS_MS.len());
        self.counts[i] += 1;
        self.count += 1;
        self.sum_ms += ms;
        self.max_ms = self.max_ms.max(ms);
    }
}

/// Per-iteration scheduler metrics (design 2.5, M2).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BatchMetrics {
    pub iterations: u64,
    /// Batched decode passes; `batch_hist[b]` counts passes of `b` rows (index 0 unused).
    pub decode_steps: u64,
    pub batch_hist: [u64; MAX_BATCH_CAP + 1],
    pub prefill_chunks: u64,
    pub prefill_tokens: u64,
    /// Requests waiting for a slot now / at most.
    pub queue_depth: usize,
    pub queue_depth_max: usize,
    /// Requests holding a slot now.
    pub running: usize,
    pub submitted: u64,
    pub admitted: u64,
    pub finished: u64,
    pub cancelled: u64,
    pub rejected: u64,
    pub failed: u64,
    /// Tokens of reused prefixes at admission.
    pub cached_tokens: u64,
    /// Submit to first token.
    pub ttft: LatencyHist,
    /// Between consecutive tokens of one request.
    pub itl: LatencyHist,
    /// Wall time of one iteration with a decode step, and its EWMA (alpha 0.1).
    pub step: LatencyHist,
    pub step_ewma_ms: f64,
}

enum Phase {
    /// `pos`: prompt tokens in the slot; `chunk`: the size chosen at the first chunk.
    Prefilling {
        pos: usize,
        chunk: Option<usize>,
    },
    Decoding,
}

struct Active<S> {
    id: RequestId,
    seq: S,
    slot: usize,
    phase: Phase,
    t_submit: Instant,
    t_last: Option<Instant>,
}

struct Waiting<S> {
    id: RequestId,
    seq: S,
    t_submit: Instant,
}

/// The deterministic scheduler: [`Scheduler::step`] runs one iteration. [`BatchEngine`] drives it from a thread;
/// tests drive it directly.
pub struct Scheduler<M: BatchModel, S: BatchSeq<M>> {
    cfg: SchedulerConfig,
    model: M,
    waiting: VecDeque<Waiting<S>>,
    /// Admission order (oldest first).
    active: Vec<Active<S>>,
    busy: Vec<bool>,
    last_used: Vec<u64>,
    iter: u64,
    next_id: RequestId,
    failed: Option<String>,
    /// The same, published before any request hears of the failure ([`BatchEngine::failed`] reads it).
    failed_shared: Arc<Mutex<Option<String>>>,
    metrics: Arc<Mutex<BatchMetrics>>,
    /// Requests submitted but not yet admitted (shared with [`BatchEngine::submit`] for the queue bound).
    queued: Arc<AtomicUsize>,
    trace: Vec<TraceEvent>,
}

impl<M: BatchModel, S: BatchSeq<M>> Scheduler<M, S> {
    pub fn new(model: M, cfg: SchedulerConfig) -> Result<Self> {
        cfg.validate()?;
        if model.slots() < cfg.max_batch {
            return Err(Error::BadRequest(format!(
                "model has {} slots, max_batch is {}",
                model.slots(),
                cfg.max_batch
            )));
        }
        let n = cfg.max_batch;
        Ok(Scheduler {
            cfg,
            model,
            waiting: VecDeque::new(),
            active: Vec::new(),
            busy: vec![false; n],
            last_used: vec![0; n],
            iter: 0,
            next_id: 0,
            failed: None,
            failed_shared: Arc::new(Mutex::new(None)),
            metrics: Arc::new(Mutex::new(BatchMetrics::default())),
            queued: Arc::new(AtomicUsize::new(0)),
            trace: Vec::new(),
        })
    }

    pub fn config(&self) -> &SchedulerConfig {
        &self.cfg
    }
    pub fn model(&self) -> &M {
        &self.model
    }
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }
    pub fn metrics(&self) -> BatchMetrics {
        self.metrics.lock().unwrap().clone()
    }
    pub fn take_trace(&mut self) -> Vec<TraceEvent> {
        std::mem::take(&mut self.trace)
    }
    /// The fatal error that stopped the scheduler, if any.
    pub fn failed(&self) -> Option<&str> {
        self.failed.as_deref()
    }
    /// Anything admitted or waiting.
    pub fn has_work(&self) -> bool {
        !self.waiting.is_empty() || !self.active.is_empty()
    }
    pub fn iteration(&self) -> u64 {
        self.iter
    }

    /// Submit a request (FIFO). Rejected (and `seq.fail` called) when the queue is full or the scheduler failed.
    pub fn submit(&mut self, mut seq: S) -> Result<RequestId> {
        if let Some(f) = &self.failed {
            let e = Error::Other(format!("engine failed: {f}"));
            seq.fail(&e);
            return Err(e);
        }
        if self.queued.load(Ordering::SeqCst) >= self.cfg.max_queue {
            let e = Error::QueueFull(self.cfg.max_queue);
            seq.fail(&e);
            self.metrics.lock().unwrap().rejected += 1;
            return Err(e);
        }
        self.queued.fetch_add(1, Ordering::SeqCst);
        let id = self.next_id;
        self.next_id += 1;
        self.enqueue(id, seq, Instant::now());
        Ok(id)
    }

    /// Queue a request already counted in `queued` (the engine's submit path).
    fn enqueue(&mut self, id: RequestId, seq: S, t_submit: Instant) {
        self.waiting.push_back(Waiting { id, seq, t_submit });
        let mut m = self.metrics.lock().unwrap();
        m.submitted += 1;
        m.queue_depth = self.waiting.len();
        m.queue_depth_max = m.queue_depth_max.max(self.waiting.len());
    }

    fn ev(&mut self, e: TraceEvent) {
        if self.cfg.trace {
            self.trace.push(e);
        }
    }

    fn free_slot(&mut self, slot: usize) {
        self.busy[slot] = false;
        self.last_used[slot] = self.iter;
    }

    /// End active request `i`: `finish` (or `fail` if finishing errs), free its slot.
    fn end_active(&mut self, i: usize, cancelled: bool) {
        let mut a = self.active.remove(i);
        let res = a.seq.finish(&mut self.model, Some(a.slot), cancelled);
        let how = match res {
            Ok(()) if cancelled => EndKind::Cancelled,
            Ok(()) => EndKind::Done,
            Err(e) => {
                a.seq.fail(&e);
                let _ = self.model.truncate(a.slot, 0);
                EndKind::Failed
            }
        };
        self.free_slot(a.slot);
        self.ev(TraceEvent::Finished { iter: self.iter, id: a.id, how });
        let mut m = self.metrics.lock().unwrap();
        match how {
            EndKind::Cancelled => m.cancelled += 1,
            EndKind::Failed => m.failed += 1,
            _ => m.finished += 1,
        }
    }

    /// Fail active request `i` with `e`; its slot's KV may be ahead of what the sequence accounted for, so the slot
    /// is invalidated (never a reuse candidate). A fatal error fails everything.
    fn fail_active(&mut self, i: usize, e: Error) {
        if self.model.is_fatal(&e) {
            self.fail_all(&e.to_string());
            return;
        }
        let mut a = self.active.remove(i);
        a.seq.fail(&e);
        let _ = self.model.truncate(a.slot, 0);
        self.free_slot(a.slot);
        self.ev(TraceEvent::Finished { iter: self.iter, id: a.id, how: EndKind::Failed });
        self.metrics.lock().unwrap().failed += 1;
    }

    /// Fatal: every in-flight and waiting request gets `Error`; later submits are refused.
    pub fn fail_all(&mut self, msg: &str) {
        if self.failed.is_none() {
            self.failed = Some(msg.to_string());
            *self.failed_shared.lock().unwrap() = Some(msg.to_string());
        }
        let e = Error::Other(format!("engine failed: {msg}"));
        let n = self.active.len() + self.waiting.len();
        for mut a in std::mem::take(&mut self.active) {
            a.seq.fail(&e);
            self.busy[a.slot] = false;
            self.ev(TraceEvent::Finished { iter: self.iter, id: a.id, how: EndKind::Fatal });
        }
        for mut w in std::mem::take(&mut self.waiting) {
            self.queued.fetch_sub(1, Ordering::SeqCst);
            w.seq.fail(&e);
            self.ev(TraceEvent::Finished { iter: self.iter, id: w.id, how: EndKind::Fatal });
        }
        let mut m = self.metrics.lock().unwrap();
        m.failed += n as u64;
        m.running = 0;
        m.queue_depth = 0;
    }

    /// Cancel everything (shutdown): each request finishes as cancelled.
    pub fn cancel_all(&mut self) {
        while !self.active.is_empty() {
            self.end_active(0, true);
        }
        for mut w in std::mem::take(&mut self.waiting) {
            self.queued.fetch_sub(1, Ordering::SeqCst);
            if w.seq.finish(&mut self.model, None, true).is_err() {
                w.seq.fail(&Error::Other("cancelled".into()));
            }
        }
    }

    /// The slot for `prompt` among the free ones: longest reusable prefix (`min(common, plen - 1)`), ties to the least
    /// recently used, then the lowest index. Running slots are never considered.
    fn pick_slot(&self, prompt: &[u32]) -> Option<(usize, usize)> {
        let reuse_max = prompt.len().saturating_sub(1);
        (0..self.cfg.max_batch)
            .filter(|&s| !self.busy[s])
            .map(|s| {
                let cached = if self.cfg.prefix_reuse { common_prefix(self.model.slot_tokens(s), prompt).min(reuse_max) } else { 0 };
                (s, cached)
            })
            .max_by(|a, b| a.1.cmp(&b.1).then(self.last_used[b.0].cmp(&self.last_used[a.0])).then(b.0.cmp(&a.0)))
    }

    /// One scheduler iteration: reap, admit, one batched decode step, prefill chunk(s). Returns whether anything ran.
    pub fn step(&mut self) -> bool {
        if self.failed.is_some() {
            return false;
        }
        self.iter += 1;
        let t_iter = Instant::now();
        let mut did = false;

        // 1. reap cancellations: queued requests leave the queue, running ones leave their slot
        let mut i = 0;
        while i < self.waiting.len() {
            if self.waiting[i].seq.cancelled() {
                let mut w = self.waiting.remove(i).unwrap();
                self.queued.fetch_sub(1, Ordering::SeqCst);
                if let Err(e) = w.seq.finish(&mut self.model, None, true) {
                    w.seq.fail(&e);
                }
                self.ev(TraceEvent::Finished { iter: self.iter, id: w.id, how: EndKind::Cancelled });
                self.metrics.lock().unwrap().cancelled += 1;
                did = true;
            } else {
                i += 1;
            }
        }
        let mut i = 0;
        while i < self.active.len() {
            if self.active[i].seq.cancelled() {
                self.end_active(i, true);
                did = true;
            } else {
                i += 1;
            }
        }

        // 2. admit, FIFO: the head of the queue takes a free slot or everybody waits
        while !self.waiting.is_empty() {
            let Some((slot, cached)) = self.pick_slot(self.waiting[0].seq.prompt()) else { break };
            let mut w = self.waiting.pop_front().unwrap();
            self.queued.fetch_sub(1, Ordering::SeqCst);
            did = true;
            let cap = self.model.slot_cap();
            match w.seq.admit(cached, cap) {
                Err(e) => {
                    w.seq.fail(&e);
                    self.ev(TraceEvent::Finished { iter: self.iter, id: w.id, how: EndKind::Rejected });
                    self.metrics.lock().unwrap().rejected += 1;
                    continue;
                }
                Ok(go) => {
                    if let Err(e) = self.model.truncate(slot, cached) {
                        w.seq.fail(&e);
                        if self.model.is_fatal(&e) {
                            self.fail_all(&e.to_string());
                            return true;
                        }
                        let _ = self.model.truncate(slot, 0);
                        self.metrics.lock().unwrap().failed += 1;
                        continue;
                    }
                    self.busy[slot] = true;
                    self.model.on_admit(slot, w.id, cached);
                    self.ev(TraceEvent::Admitted { iter: self.iter, id: w.id, slot, cached });
                    {
                        let mut m = self.metrics.lock().unwrap();
                        m.admitted += 1;
                        m.cached_tokens += cached as u64;
                    }
                    self.active.push(Active {
                        id: w.id,
                        seq: w.seq,
                        slot,
                        phase: Phase::Prefilling { pos: cached, chunk: None },
                        t_submit: w.t_submit,
                        t_last: None,
                    });
                    if !go {
                        let i = self.active.len() - 1;
                        self.end_active(i, false);
                    }
                }
            }
        }

        // 3. decode: every decoding sequence advances one token (decode-first)
        let decoding: Vec<usize> =
            (0..self.active.len()).filter(|&i| matches!(self.active[i].phase, Phase::Decoding)).collect();
        let mut stepped = false;
        if !decoding.is_empty() {
            did = true;
            let b = decoding.len();
            let spec_allowed = b < self.cfg.spec_off_at;
            // plan every row; a planning error fails that request only
            let mut rows: Vec<Row<M::Mode>> = Vec::with_capacity(b);
            let mut row_of: Vec<RequestId> = Vec::with_capacity(b);
            let mut ready: Vec<RequestId> = Vec::new();
            let mut failed: Vec<(RequestId, Error)> = Vec::new();
            for &i in &decoding {
                let a = &mut self.active[i];
                match a.seq.prepare(&mut self.model, a.slot, spec_allowed) {
                    Ok(SeqStep::Plain(token, mode)) => {
                        rows.push(Row { slot: a.slot, token, mode });
                        row_of.push(a.id);
                    }
                    Ok(SeqStep::Ready) => ready.push(a.id),
                    Err(e) => failed.push((a.id, e)),
                }
            }
            for (id, e) in failed {
                if let Some(i) = self.index_of(id) {
                    self.fail_active(i, e);
                    if self.failed.is_some() {
                        return true;
                    }
                }
            }
            if !rows.is_empty() {
                self.ev(TraceEvent::Decode { iter: self.iter, ids: row_of.clone(), spec_allowed });
                let outs = match self.model.decode_step(&rows) {
                    Ok(o) if o.len() == rows.len() => o,
                    Ok(o) => {
                        self.fail_all(&format!("decode step returned {} rows for {}", o.len(), rows.len()));
                        return true;
                    }
                    Err(e) => {
                        self.fail_all(&e.to_string());
                        return true;
                    }
                };
                stepped = true;
                {
                    let mut m = self.metrics.lock().unwrap();
                    m.decode_steps += 1;
                    m.batch_hist[rows.len()] += 1;
                }
                for (id, out) in row_of.iter().zip(outs) {
                    let i = self.index_of(*id).expect("row of a live request");
                    let res = out.and_then(|o| self.active[i].seq.accept(o));
                    match res {
                        Ok(()) => ready.push(*id),
                        Err(e) => {
                            self.fail_active(i, e);
                            if self.failed.is_some() {
                                return true;
                            }
                        }
                    }
                }
            }
            // account the new tokens in admission order
            let now = Instant::now();
            let order: Vec<RequestId> = self.active.iter().map(|a| a.id).filter(|id| ready.contains(id)).collect();
            for id in order {
                let i = self.index_of(id).expect("ready request is live");
                self.token_done(i, now);
                if self.failed.is_some() {
                    return true;
                }
            }
        }

        // 4. prefill: the oldest prefilling sequence gets its next chunk(s)
        let n_decoding = self.active.iter().filter(|a| matches!(a.phase, Phase::Decoding)).count();
        let per_iter = if n_decoding > 0 { self.cfg.busy_chunks_per_iter } else { 1 };
        for _ in 0..per_iter {
            let Some(i) = self.active.iter().position(|a| matches!(a.phase, Phase::Prefilling { .. })) else { break };
            did = true;
            if self.prefill_one(i, n_decoding) {
                break;
            }
            if self.failed.is_some() {
                return true;
            }
        }

        let mut m = self.metrics.lock().unwrap();
        m.iterations += 1;
        m.running = self.active.len();
        m.queue_depth = self.waiting.len();
        if stepped {
            let d = t_iter.elapsed();
            m.step.record(d);
            let ms = d.as_secs_f64() * 1e3;
            m.step_ewma_ms = if m.decode_steps == 1 { ms } else { 0.9 * m.step_ewma_ms + 0.1 * ms };
        }
        did
    }

    fn index_of(&self, id: RequestId) -> Option<usize> {
        self.active.iter().position(|a| a.id == id)
    }

    /// Account one new token of active request `i` (`on_token`), with TTFT / ITL; finish it if done.
    fn token_done(&mut self, i: usize, now: Instant) {
        {
            let a = &mut self.active[i];
            let mut m = self.metrics.lock().unwrap();
            match a.t_last {
                None => m.ttft.record(now - a.t_submit),
                Some(t) => m.itl.record(now - t),
            }
            a.t_last = Some(now);
        }
        match self.active[i].seq.on_token() {
            Ok(false) => {}
            Ok(true) => self.end_active(i, false),
            Err(e) => self.fail_active(i, e),
        }
    }

    /// Run the next chunk of active request `i`. Returns true when this ended its prefill (first token taken or
    /// the request ended), so a following chunk of this iteration goes to the next prefilling request only from the
    /// next iteration on (one request's prefill completion per iteration).
    fn prefill_one(&mut self, i: usize, n_decoding: usize) -> bool {
        let (pos, chunk) = match self.active[i].phase {
            Phase::Prefilling { pos, chunk } => (pos, chunk),
            Phase::Decoding => unreachable!(),
        };
        let chunk = match chunk {
            Some(c) => c,
            None => {
                let c = if n_decoding > 0 { self.cfg.busy_chunk } else { self.cfg.solo_chunk };
                let id = self.active[i].id;
                self.ev(TraceEvent::PrefillStart { iter: self.iter, id, chunk: c, decoding: n_decoding });
                self.active[i].seq.on_prefill_start(c, n_decoding);
                c
            }
        };
        let plen = self.active[i].seq.prompt().len();
        let len = next_prefill_chunk(plen - pos, chunk, self.model.avoid_single_row_prefill());
        let last = len == plen - pos;
        let mode = if last { Some(self.active[i].seq.first_mode()) } else { None };
        let slot = self.active[i].slot;
        let id = self.active[i].id;
        self.ev(TraceEvent::Chunk { iter: self.iter, id, start: pos, len, last });
        let res = {
            let tokens = &self.active[i].seq.prompt()[pos..pos + len];
            self.model.prefill_chunk(slot, tokens, mode)
        };
        {
            let mut m = self.metrics.lock().unwrap();
            m.prefill_chunks += 1;
            m.prefill_tokens += len as u64;
        }
        match res {
            Err(e) => {
                self.fail_active(i, e);
                true
            }
            Ok(out) if !last => {
                if out.is_some() {
                    self.fail_active(i, Error::Other("model returned output for a non-final prefill chunk".into()));
                    return true;
                }
                self.active[i].phase = Phase::Prefilling { pos: pos + len, chunk: Some(chunk) };
                false
            }
            Ok(None) => {
                self.fail_active(i, Error::Other("model returned no output for the final prefill chunk".into()));
                true
            }
            Ok(Some(out)) => {
                if let Err(e) = self.active[i].seq.accept(out) {
                    self.fail_active(i, e);
                    return true;
                }
                self.active[i].phase = Phase::Decoding;
                self.token_done(i, Instant::now());
                true
            }
        }
    }
}

enum Msg<S> {
    Job { id: RequestId, seq: S, t_submit: Instant },
}

struct EngineShared {
    queued: Arc<AtomicUsize>,
    max_queue: usize,
    next_id: AtomicU64,
    /// Set by the scheduler when it fails, before any request hears of it; `submit` holds it across "check + send", the
    /// thread across draining the channel after a failure, so no submitted request is lost between the two.
    failed: Arc<Mutex<Option<String>>>,
    shutdown: AtomicBool,
    metrics: Arc<Mutex<BatchMetrics>>,
}

/// The scheduler on its own thread, which owns the model (design 2.1). `submit` only queues.
pub struct BatchEngine<M, S> {
    tx: Option<Sender<Msg<S>>>,
    shared: Arc<EngineShared>,
    thread: Option<JoinHandle<()>>,
    _model: std::marker::PhantomData<fn() -> M>,
}

impl<M, S> BatchEngine<M, S>
where
    M: BatchModel + Send + 'static,
    S: BatchSeq<M> + 'static,
{
    pub fn spawn(model: M, cfg: SchedulerConfig) -> Result<Self> {
        let sched: Scheduler<M, S> = Scheduler::new(model, cfg)?;
        let shared = Arc::new(EngineShared {
            queued: sched.queued.clone(),
            max_queue: sched.cfg.max_queue,
            next_id: AtomicU64::new(0),
            failed: sched.failed_shared.clone(),
            shutdown: AtomicBool::new(false),
            metrics: sched.metrics.clone(),
        });
        let (tx, rx) = mpsc::channel();
        let sh = shared.clone();
        let thread = std::thread::Builder::new()
            .name("intarsia-batch".into())
            .spawn(move || run_scheduler(sched, rx, sh))
            .map_err(Error::Io)?;
        Ok(BatchEngine { tx: Some(tx), shared, thread: Some(thread), _model: std::marker::PhantomData })
    }

    /// Queue a request. Refused (with `seq.fail` called) when the queue is full or the engine failed.
    pub fn submit(&self, mut seq: S) -> Result<RequestId> {
        let failed = self.shared.failed.lock().unwrap();
        if let Some(f) = failed.as_ref() {
            let e = Error::Other(format!("engine failed: {f}"));
            seq.fail(&e);
            return Err(e);
        }
        if self.shared.queued.fetch_add(1, Ordering::SeqCst) >= self.shared.max_queue {
            self.shared.queued.fetch_sub(1, Ordering::SeqCst);
            let e = Error::QueueFull(self.shared.max_queue);
            seq.fail(&e);
            self.shared.metrics.lock().unwrap().rejected += 1;
            return Err(e);
        }
        let id = self.shared.next_id.fetch_add(1, Ordering::SeqCst);
        let msg = Msg::Job { id, seq, t_submit: Instant::now() };
        match self.tx.as_ref().expect("engine running").send(msg) {
            Ok(()) => Ok(id),
            Err(mpsc::SendError(Msg::Job { mut seq, .. })) => {
                self.shared.queued.fetch_sub(1, Ordering::SeqCst);
                let e = Error::Other("engine stopped".into());
                seq.fail(&e);
                Err(e)
            }
        }
    }

    pub fn metrics(&self) -> BatchMetrics {
        self.shared.metrics.lock().unwrap().clone()
    }

    pub fn failed(&self) -> Option<String> {
        self.shared.failed.lock().unwrap().clone()
    }
}

impl<M, S> Drop for BatchEngine<M, S> {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::SeqCst);
        drop(self.tx.take());
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn panic_message(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_default()
}

fn run_scheduler<M: BatchModel, S: BatchSeq<M>>(
    mut sched: Scheduler<M, S>,
    rx: Receiver<Msg<S>>,
    sh: Arc<EngineShared>,
) {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    let mut open = true;
    loop {
        if sh.shutdown.load(Ordering::SeqCst) {
            sched.cancel_all();
            break;
        }
        // block only when there is nothing to run
        if !sched.has_work() {
            if !open {
                break;
            }
            match rx.recv() {
                Ok(Msg::Job { id, seq, t_submit }) => sched.enqueue(id, seq, t_submit),
                Err(_) => {
                    open = false;
                    continue;
                }
            }
        }
        while open {
            match rx.try_recv() {
                Ok(Msg::Job { id, seq, t_submit }) => sched.enqueue(id, seq, t_submit),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => open = false,
            }
        }
        if let Err(p) = catch_unwind(AssertUnwindSafe(|| sched.step())) {
            let msg = format!("scheduler panicked: {}", panic_message(&*p));
            let _ = catch_unwind(AssertUnwindSafe(|| sched.fail_all(&msg)));
            if sched.failed.is_none() {
                *sched.failed_shared.lock().unwrap() = Some(msg.clone());
                sched.failed = Some(msg);
            }
        }
        if let Some(f) = sched.failed() {
            let _held = sh.failed.lock().unwrap();
            // refuse whatever is still in the channel; later submits see `failed` first (under the same lock)
            let e = Error::Other(format!("engine failed: {f}"));
            while let Ok(Msg::Job { mut seq, .. }) = rx.try_recv() {
                sh.queued.fetch_sub(1, Ordering::SeqCst);
                seq.fail(&e);
            }
            break;
        }
    }
}
