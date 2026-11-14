//! The continuous-batching engine: `scheduler::BatchEngine` wired to the real [`Model`] (`--max-batch N`, 2..=8).
//!
//! Two adapters sit between the scheduler's traits and the existing machinery:
//!
//! * [`SlotModel`] (`BatchModel`): the model plus one [`SequenceState`] per slot. Prefill chunks are
//!   `Model::forward` on the slot's state, a decode step is one `Model::forward_batch` over the slots' states (split
//!   borrows over the slot vector), truncation is `SequenceState::truncate`.
//! * [`SeqAdapter`] (`BatchSeq`): one request, wrapping `engine::Seq` (the same per-request state machine the
//!   single-sequence engine drives), so sampling, stop handling, streaming and speculation are the code that
//!   `LocalEngine` runs. `Seq::build` + `Seq::admit` are the construction half of `Seq::new`; `report_done` /
//!   `report_error` are the shared outcome helpers.
//!
//! `--max-batch 1` never gets here: it keeps the `LocalEngine` path untouched.

use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::Instant;

use serde_json::Value;

use super::{
    Prep,
    build_info, report_done, report_error, ChatOpts, Engine, EngineMeta, FinishReason, GenEvent, GenRequest, ModelInfo,
    ScriptStep, Seq, SeqEnv, SeqStore, StorePolicy, TextCodec,
};
use crate::error::{Error, Result};
use crate::model::{HeadMode, Model, StepOut};
use crate::scheduler::{
    BatchEngine, BatchMetrics, BatchModel, BatchSeq, Housekeep, RequestId, Row, SchedulerConfig, SeqStep, Verified, VerifyRows,
    MAX_BATCH_CAP, SPEC_OFF_AT_DEFAULT,
};
use crate::state::SequenceState;
use crate::telemetry::{Telemetry, TelemetrySnapshot};

/// Prefill chunk used while other sequences decode (design 8.1: 256 while others decode, the model's chunk alone).
pub const BUSY_PREFILL_CHUNK: usize = 256;

/// `INTARSIA_BATCHED_MTP=N` (lane bm, opt-in): `--spec mtp` with `--max-batch` > 1, drafting while at most `N` sequences decode
/// ([`BatchedOptions::mtp_max_b`]). Unset, empty or `0`: off (`--spec mtp` stays refused with the batching engine).
pub fn batched_mtp_env() -> Result<usize> {
    Ok(batched_mtp_env_opt()?.unwrap_or(0))
}

/// [`batched_mtp_env`], `None` when the variable is unset or empty (the recipe's `[tuning.<backend>] batched_mtp` default then applies).
pub fn batched_mtp_env_opt() -> Result<Option<usize>> {
    match std::env::var("INTARSIA_BATCHED_MTP") {
        Err(_) => Ok(None),
        Ok(v) if v.is_empty() => Ok(None),
        Ok(v) => v.parse::<usize>().ok().filter(|&n| n <= MAX_BATCH_CAP).map(Some).ok_or_else(|| {
            Error::invalid(format!("INTARSIA_BATCHED_MTP={v}: a sequence count 0..={MAX_BATCH_CAP} (draft while at most that many decode)"))
        }),
    }
}

/// `INTARSIA_BATCHED_MTP_SAMPLED` (lane mtp-sampled): `0` keeps sampled requests (temperature > 0) on plain steps in the
/// batching engine; unset, empty or anything else: they draft with the MTP head like greedy ones ([`BatchedOptions::mtp_sampled`]).
pub fn batched_mtp_sampled_env() -> bool {
    std::env::var("INTARSIA_BATCHED_MTP_SAMPLED").map(|v| v != "0").unwrap_or(true)
}

/// Knobs of [`BatchedEngine::new`].
#[derive(Clone, Debug)]
pub struct BatchedOptions {
    /// `--max-batch`, `2..=MAX_BATCH_CAP`.
    pub max_batch: usize,
    /// Requests that may wait for a slot (the scheduler's queue bound).
    pub max_queue: usize,
    /// Reuse the KV prefix an idle slot holds (default on). `INTARSIA_CB_NO_REUSE=1` switches it off, which the
    /// concurrent-identity harness uses to compare against a cold-prefill solo reference.
    pub prefix_reuse: bool,
    /// Log scheduler events to stderr (`INTARSIA_CB_TRACE=1`). Nothing is stored.
    pub trace: bool,
    /// Speculation drafts only while fewer than this many sequences decode (`--spec-off-at`, default
    /// [`SPEC_OFF_AT_DEFAULT`]). No effect unless speculation is enabled.
    pub spec_off_at: usize,
    /// Prefill chunk of a sequence that starts its prefill while others decode (`--busy-chunk`, default
    /// [`BUSY_PREFILL_CHUNK`]); capped at the model's chunk.
    pub busy_chunk: usize,
    /// Test seam: feed `token` once instead of the real one in the decode row of slot `.0` whose slot holds `.1` tokens
    /// (error-isolation tests put an invalid id into one row). Always `None` outside tests.
    #[doc(hidden)]
    pub row_fault: Option<(usize, usize, u32)>,
    /// Prefill/decode overlap (R3d, `INTARSIA_OVERLAP`): batched decode runs on a decode twin of the model (own stream, own
    /// workspace) so a prefill chunk can be in flight beside the decode steps. Output is unchanged, bit for bit; CUDA only.
    pub overlap: bool,
    /// Packed prefill (`INTARSIA_PACK_PREFILL=1` on, `=0` off): the next chunks of several waiting prompts run in one pass
    /// (`Model::forward_packed`) on a hybrid (recurrent) model; each prompt's state is its solo prefill's when every row-wise kernel is
    /// independent of the other rows of the call. `None` (unset): the backend's default (`Backend::pack_prefill_default`: on for
    /// CUDA, lane srv16, and HIP, lane fh; off elsewhere).
    pub pack_prefill: Option<bool>,
    /// Most body rows of one packed pass (`INTARSIA_PACK_ROWS`, default [`PACK_ROWS_DEFAULT`], 0: the workspace bound only); see
    /// [`SchedulerConfig::pack_rows`].
    pub pack_rows: usize,
    /// A packed pass takes at most half of the waiting prompts (default on, `INTARSIA_PACK_HALF=0` off); see
    /// [`SchedulerConfig::pack_half`].
    pub pack_half: bool,
    /// Batched MTP speculation (lane bm, opt-in, [`batched_mtp_env`]): `--spec mtp` drafts while at most this many sequences
    /// decode (and `spec_off_at` allows it); wider steps run plain batched decode. 0: `--spec mtp` is refused here (the default).
    pub mtp_max_b: usize,
    /// Lane bm phase B (`INTARSIA_BATCHED_VERIFY=1`, opt-in): with batched MTP, the sequences that draft in a step verify in ONE
    /// ragged pass (`Model::forward_verify_batch`) instead of one solo pass each.
    pub batched_verify: bool,
    /// Lane mtp-sampled (`INTARSIA_BATCHED_MTP_SAMPLED`, [`batched_mtp_sampled_env`]): with batched MTP, sampled requests
    /// (temperature > 0) draft too, with seed-exact acceptance (`engine/keyed.rs`): a seeded request's ids are the same with it on
    /// or off. Off: sampled requests take plain steps.
    pub mtp_sampled: bool,
    /// Lane mtp2s (`INTARSIA_VERIFY_TOPK=1`, opt-in): in a batched verify pass, a sampled sequence's rows come back as the
    /// device-side top-k of each row (`HeadMode::LastTopK`, the candidates a plain batched decode step samples from) instead of
    /// whole logits rows, when its sampler takes the top-k in a plain step for every row of the pass (`Seq::verify_topk`). The
    /// draws are the same (`Sampler::sample_topk` is exact); a pass downloads `rows x k` values instead of `rows x vocab`.
    pub verify_topk: bool,
    /// Lane interleave (`INTARSIA_PREFILL_PAUSE_ROWS`, default [`PAUSE_ROWS_DEFAULT`] = 768, 0 off): while other sequences decode, a
    /// prefill chunk of more than this many rows parks between layers about every this many rows and their decode step runs
    /// there ([`BatchModel::pause_rows`]). The chunk schedule and every output are unchanged, bit for bit. Not with `overlap`.
    pub pause_rows: usize,
    /// Lane interleave (`INTARSIA_SETTLE_SLICE_MB`, default [`SETTLE_SLICE_MB_DEFAULT`] = 0, off): > 0, a prefix-store settle after a
    /// request's `Done` moves at most about this many MiB per scheduler iteration, between decode steps, instead of all of it at once
    /// right after the `Done`. Meant with [`BatchedOptions::pause_rows`]: without pauses a slice runs only in an iteration where nothing
    /// prefills, so a long prefill beside a pending settle has no relief (captures past the deferral cap then download inside the chunk).
    pub settle_slice_mb: usize,
    /// Tests only: start the scheduler thread held, so that a whole workload is submitted before its first iteration admits
    /// anything (see [`BatchedEngine::release`]). Always false from [`BatchedOptions::new`].
    #[doc(hidden)]
    pub hold_start: bool,
}

/// Default [`BatchedOptions::pause_rows`] (lane interleave): 768. On the Spark 35B, 2 streams with the store (window 2 of
/// docs/results/2026-11-13-interleave.md, pause alone vs both switches off): median ITL p99 370 -> 180 ms, engine gaps above
/// 0.25 s 104 -> 3, TTFT sum -0.6%, decode per stream -1.3%. Outputs are bitwise unchanged. `INTARSIA_PREFILL_PAUSE_ROWS=0`: off.
pub const PAUSE_ROWS_DEFAULT: usize = 768;
/// Default [`BatchedOptions::settle_slice_mb`]: 0, off (the settle runs whole right after the `Done`). Measured opt-in value: 48
/// (~30-45 ms of demotes per slice on the Spark).
pub const SETTLE_SLICE_MB_DEFAULT: usize = 0;

/// Default row cap of a packed prefill pass (lane srv16b): a decoding stream waits for the whole pass, ~0.14 ms per row on the Spark
/// 35B, so 3072 rows (three ~1k-token prompts) keep that wait under ~0.5 s, and keep most of the 8192-row pass's 16-stream gain
/// (docs/results/2026-10-29-srv16b-pack-latency.md).
pub const PACK_ROWS_DEFAULT: usize = 3072;

impl BatchedOptions {
    /// The scheduler policy these options select, for a model whose prefill chunk is `solo_chunk`.
    pub fn scheduler_config(&self, solo_chunk: usize) -> SchedulerConfig {
        SchedulerConfig {
            max_batch: self.max_batch,
            max_queue: self.max_queue,
            solo_chunk,
            busy_chunk: self.busy_chunk.min(solo_chunk),
            prefix_reuse: self.prefix_reuse,
            trace_stderr: self.trace,
            spec_off_at: self.spec_off_at,
            pack_prefill: self.pack_prefill.unwrap_or(false),
            pack_rows: self.pack_rows,
            pack_half: self.pack_half,
            ..SchedulerConfig::default()
        }
    }

    pub fn new(max_batch: usize, max_queue: usize) -> Self {
        let env = |k: &str| std::env::var(k).map(|v| v == "1").unwrap_or(false);
        BatchedOptions {
            max_batch,
            max_queue,
            prefix_reuse: !env("INTARSIA_CB_NO_REUSE"),
            trace: env("INTARSIA_CB_TRACE"),
            spec_off_at: SPEC_OFF_AT_DEFAULT,
            busy_chunk: BUSY_PREFILL_CHUNK,
            row_fault: None,
            overlap: std::env::var("INTARSIA_OVERLAP").map(|v| v == "1").unwrap_or(false),
            pack_prefill: std::env::var("INTARSIA_PACK_PREFILL").ok().map(|v| v == "1"),
            pack_rows: std::env::var("INTARSIA_PACK_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(PACK_ROWS_DEFAULT),
            pack_half: std::env::var("INTARSIA_PACK_HALF").map(|v| v != "0").unwrap_or(true),
            mtp_max_b: 0,
            pause_rows: std::env::var("INTARSIA_PREFILL_PAUSE_ROWS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(PAUSE_ROWS_DEFAULT),
            settle_slice_mb: std::env::var("INTARSIA_SETTLE_SLICE_MB")
                .ok()
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(SETTLE_SLICE_MB_DEFAULT),
            batched_verify: env("INTARSIA_BATCHED_VERIFY"),
            mtp_sampled: batched_mtp_sampled_env(),
            verify_topk: env("INTARSIA_VERIFY_TOPK"),
            hold_start: false,
        }
    }
}

/// The model side of the scheduler: one KV slot per `SequenceState`.
pub(crate) struct SlotModel {
    model: Arc<Model>,
    /// The decode twin (R3d, `BatchedOptions::overlap`): batched decode steps run on it, on their own stream with their own
    /// workspace, so a prefill chunk of `model` can be in flight beside them. `None`: decode runs on `model`.
    twin: Option<Model>,
    states: Vec<SequenceState>,
    /// [`BatchedOptions::row_fault`], cleared once fired.
    row_fault: Option<(usize, usize, u32)>,
    /// The prefill chunk running in the background (`BatchModel::prefill_begin`), if any.
    inflight: Option<Inflight>,
    /// Lane bm phase B: drafting sequences verify in one batched pass ([`BatchModel::verify_step`]).
    batched_verify: bool,
    /// Lane mtp2s: [`BatchedOptions::verify_topk`] (with batched verify only).
    verify_topk: bool,
    /// Lane interleave: [`BatchedOptions::pause_rows`] (0 with the decode twin: overlap runs chunks beside decode instead).
    pause_rows: usize,
    /// A prefix-store settle left to do after a request's `Done` ([`BatchModel::housekeep`]), with the telemetry to report it to.
    settle: Option<(Arc<std::sync::Mutex<crate::prefix_store::PrefixStore>>, Arc<Telemetry>)>,
    /// The engine's prefix store (with the telemetry to report to), if on: the idle settle ([`Housekeep::Idle`]) settles whatever is
    /// deferred, also captures of a request that failed and never reached `finish`.
    store: Option<(Arc<std::sync::Mutex<crate::prefix_store::PrefixStore>>, Arc<Telemetry>)>,
    /// [`BatchedOptions::settle_slice_mb`] in bytes.
    settle_slice: u64,
    /// Per slot: the request it holds runs the prefix store's schedule (set at admission, [`BatchSeq::restore`]), so its final
    /// prefill chunk runs whole on a recurrent state, as `Seq::prefill_step` runs it with the store on.
    store_slot: Vec<bool>,
}

/// A prefill chunk on its own thread. The slot's state travels with it (a placeholder holds the slot meanwhile) and comes back
/// at `prefill_end`, so no state is shared between the threads.
struct Inflight {
    slot: usize,
    done: Arc<std::sync::atomic::AtomicBool>,
    /// A chunk that parks between layers ([`BatchModel::prefill_begin_paused`]): it sends on the first when it parks and waits
    /// on the second (`false`: go on, `true`: abort).
    park: Option<(Receiver<()>, Sender<bool>)>,
    handle: std::thread::JoinHandle<(SequenceState, Result<Option<StepOut>>)>,
}

/// One prefill chunk of `st` (the body of [`BatchModel::prefill_chunk`], also what the background thread runs). `whole_last`: a
/// request on the prefix store's schedule, whose final chunk runs whole (no lone last token), as on the single-sequence engine.
fn run_prefill(
    model: &Model,
    st: &mut SequenceState,
    tokens: &[u32],
    mode: Option<HeadMode>,
    whole_last: bool,
) -> Result<Option<StepOut>> {
    // the store schedule's final chunk is one pass (`Seq::prefill_step` with the store on): the plain one below
    if st.has_recurrent() && !(whole_last && mode.is_some()) {
        // the solo schedule of a recurrent state (`Seq::prefill_step`): a snapshot after every non-final chunk, and
        // before the final chunk's last token, which runs alone (so a repeated prompt reuses all but that token)
        let keep = model.batch_snapshots();
        return Ok(match mode {
            None => {
                if !matches!(model.forward_prefill(st, tokens, HeadMode::None, None)?, StepOut::None) {
                    return Err(Error::Other("a headless prefill pass returned an output".into()));
                }
                st.snapshot(&*model.backend, keep)?;
                None
            }
            Some(m) => {
                if tokens.len() > 1 {
                    // nothing is asked of this pass: a head output here would be a bug, not a result to drop
                    if !matches!(model.forward_prefill(st, &tokens[..tokens.len() - 1], HeadMode::None, None)?, StepOut::None) {
                        return Err(Error::Other("a headless prefill pass returned an output".into()));
                    }
                    st.snapshot(&*model.backend, keep)?;
                }
                match model.forward_prefill(st, &tokens[tokens.len() - 1..], m, None)? {
                    StepOut::None => None,
                    o => Some(o),
                }
            }
        });
    }
    let out = model.forward_prefill(st, tokens, mode.unwrap_or(HeadMode::None), None)?;
    Ok(match out {
        StepOut::None => None,
        o => Some(o),
    })
}

impl SlotModel {
    /// Sum of the slots' KV usage for telemetry: (used bytes, total bytes, used tokens, capacity tokens).
    fn kv_totals(&self) -> (u64, u64, u64, u64) {
        self.states.iter().fold((0, 0, 0, 0), |a, s| {
            (a.0 + s.used_bytes() as u64, a.1 + s.total_bytes() as u64, a.2 + s.len as u64, a.3 + s.cap as u64)
        })
    }
}

impl BatchModel for SlotModel {
    type Mode = HeadMode;
    type Out = StepOut;

    fn macro_step(&self, slot: usize, remaining: usize, chunk: usize) -> Option<usize> {
        let m = &self.model;
        if m.macro_rows() == 0 || chunk != m.prefill_chunk() || !self.states[slot].has_recurrent() || remaining < 2 {
            return None;
        }
        // the plain schedule runs the last token alone, so the body is everything before it; a step that takes the whole body
        // is the final call (which then ends with that token)
        let body = remaining - 1;
        let n = crate::scheduler::macro_step_rows(body, chunk, m.macro_rows())?;
        Some(if n == body { remaining } else { n })
    }

    fn slots(&self) -> usize {
        self.states.len()
    }
    fn slot_cap(&self) -> usize {
        self.states[0].cap
    }
    fn slot_tokens(&self, slot: usize) -> &[u32] {
        &self.states[slot].tokens
    }
    fn truncate(&mut self, slot: usize, len: usize) -> Result<()> {
        self.states[slot].truncate(len, &*self.model.backend)
    }
    fn restorable_len(&self, slot: usize, want: usize) -> usize {
        self.states[slot].restorable_len(want)
    }
    fn reuse_len(&self, slot: usize, plen: usize, want: usize) -> usize {
        // the solo engine's rule (`Seq::new`): a slot's history never changes a request's bits
        self.model.reuse_len(&self.states[slot], plen, want)
    }
    fn prefill_chunk(&mut self, slot: usize, tokens: &[u32], mode: Option<HeadMode>) -> Result<Option<StepOut>> {
        run_prefill(&self.model, &mut self.states[slot], tokens, mode, self.store_slot[slot])
    }
    fn overlap(&self) -> bool {
        self.twin.is_some()
    }
    fn prefill_begin(&mut self, slot: usize, tokens: &[u32], mode: Option<HeadMode>) -> Result<()> {
        if self.inflight.is_some() {
            return Err(Error::Other("a prefill chunk is already in flight".into()));
        }
        let placeholder = self.model.placeholder_state();
        let mut st = std::mem::replace(&mut self.states[slot], placeholder);
        let model = self.model.clone();
        let tokens = tokens.to_vec();
        let whole_last = self.store_slot[slot];
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = done.clone();
        let handle = std::thread::Builder::new()
            .name("intarsia-prefill".into())
            .spawn(move || {
                let r = run_prefill(&model, &mut st, &tokens, mode, whole_last);
                flag.store(true, std::sync::atomic::Ordering::Release);
                (st, r)
            })
            .map_err(|e| Error::Other(format!("cannot start the prefill thread: {e}")))?;
        self.inflight = Some(Inflight { slot, done, handle, park: None });
        Ok(())
    }
    fn prefill_done(&self) -> bool {
        self.inflight.as_ref().is_none_or(|f| f.done.load(std::sync::atomic::Ordering::Acquire))
    }
    fn prefill_end(&mut self) -> Result<Option<StepOut>> {
        let f = self.inflight.take().ok_or_else(|| Error::Other("no prefill chunk in flight".into()))?;
        match f.handle.join() {
            Ok((st, r)) => {
                self.states[f.slot] = st;
                r
            }
            // the slot's state died with the thread: the engine goes down with it, as for any panic of its own thread
            Err(panic) => std::panic::resume_unwind(panic),
        }
    }
    fn pause_rows(&self) -> usize {
        if self.twin.is_some() {
            0
        } else {
            self.pause_rows
        }
    }
    fn prefill_begin_paused(&mut self, slot: usize, tokens: &[u32], mode: Option<HeadMode>) -> Result<()> {
        if self.inflight.is_some() {
            return Err(Error::Other("a prefill chunk is already in flight".into()));
        }
        let placeholder = self.model.placeholder_state();
        let mut st = std::mem::replace(&mut self.states[slot], placeholder);
        let model = self.model.clone();
        let tokens = tokens.to_vec();
        let whole_last = self.store_slot[slot];
        let rows = self.pause_rows;
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = done.clone();
        let (park_tx, park_rx) = channel::<()>();
        let (go_tx, go_rx) = channel::<bool>();
        let spawned = std::thread::Builder::new().name("intarsia-prefill".into()).spawn(move || {
            // park: tell the scheduler, wait for its go (a dropped sender is an abort: the scheduler went away)
            let hook: crate::model::PauseFn = Box::new(move || {
                let _ = park_tx.send(());
                match go_rx.recv() {
                    Ok(false) => Ok(()),
                    _ => Err(Error::Other("prefill chunk aborted while parked".into())),
                }
            });
            let r = model.with_layer_pause(rows, hook, || run_prefill(&model, &mut st, &tokens, mode, whole_last));
            flag.store(true, std::sync::atomic::Ordering::Release);
            (st, r)
        });
        match spawned {
            Ok(handle) => {
                self.inflight = Some(Inflight { slot, done, handle, park: Some((park_rx, go_tx)) });
                Ok(())
            }
            Err(e) => Err(Error::Other(format!("cannot start the prefill thread: {e}"))),
        }
    }
    fn prefill_park(&mut self) -> bool {
        // the hook (and its sender) is dropped when the chunk's pass ends, however it ends
        self.inflight.as_ref().and_then(|f| f.park.as_ref()).is_some_and(|(rx, _)| rx.recv().is_ok())
    }
    fn prefill_resume(&mut self, abort: bool) {
        if let Some((_, go)) = self.inflight.as_ref().and_then(|f| f.park.as_ref()) {
            let _ = go.send(abort);
        }
    }
    fn housekeep(&mut self, at: Housekeep) {
        if at == Housekeep::Idle {
            // nothing runs or waits: settle whatever is deferred, pending slices or not (a failed request's captures too)
            if let Some((store, tel)) = &self.store {
                let mut s = store.lock().unwrap_or_else(|e| e.into_inner());
                if s.deferring() || self.settle.is_some() {
                    s.settle(&*self.model.backend, "idle");
                    tel.set_prefix_store(s.snapshot());
                }
            }
            self.settle = None;
            return;
        }
        let Some((store, tel)) = &self.settle else { return };
        let backend = &*self.model.backend;
        let mut s = store.lock().unwrap_or_else(|e| e.into_inner());
        let done = match at {
            Housekeep::Idle => unreachable!("handled above"),
            Housekeep::Step => s.settle_some(backend, self.settle_slice, "after done"),
            // inside a parked chunk only when the deferred captures near their cap, past which a capture downloads inside the chunk
            Housekeep::Pause if s.deferred_bytes() > crate::prefix_store::DEFER_CAP / 2 => {
                s.settle_some(backend, self.settle_slice, "in a paused chunk")
            }
            Housekeep::Pause => false,
        };
        tel.set_prefix_store(s.snapshot());
        drop(s);
        if done {
            self.settle = None;
        }
    }
    fn avoid_single_row_prefill(&self) -> bool {
        // the backend flag (Metal), except for recurrent state (never batched; mirrors `Seq::prefill_step`)
        self.model.backend.avoid_single_row_prefill() && !self.states[0].has_recurrent()
    }
    fn tail_merge(&self) -> usize {
        self.model.tail_merge()
    }
    fn pack_rows_max(&self) -> usize {
        if self.twin.is_none() && self.model.pack_capable() && self.states.iter().all(|s| s.has_recurrent()) {
            self.model.pack_rows_max()
        } else {
            0
        }
    }
    fn pack_min_rows(&self) -> usize {
        self.model.pack_min_rows()
    }
    fn pack_whole_prompts(&self) -> bool {
        self.model.backend.pack_split_gemms()
    }
    fn prefill_pack(&mut self, items: &[(usize, &[u32], Option<HeadMode>)]) -> Result<Vec<Option<StepOut>>> {
        // the schedule of `run_prefill` for a recurrent state, with the bodies of all chunks in one pass: the body is the chunk
        // (non-final) or the chunk without its last token (final); a snapshot follows each body; the last token runs alone
        let keep = self.model.batch_snapshots();
        let uniform = {
            let rows: Vec<(&SequenceState, usize)> =
                items.iter().map(|&(slot, tokens, mode)| (&self.states[slot], tokens.len() - usize::from(mode.is_some()))).collect();
            self.model.pack_uniform(&rows)
        };
        if !uniform {
            // the segments would take different attention paths: each prefills alone, as without the knob
            return items.iter().map(|&(slot, tokens, mode)| self.prefill_chunk(slot, tokens, mode)).collect();
        }
        {
            let mut by_slot: Vec<Option<&mut SequenceState>> = self.states.iter_mut().map(Some).collect();
            let mut packed: Vec<(&mut SequenceState, &[u32])> = Vec::with_capacity(items.len());
            for &(slot, tokens, mode) in items {
                let st = by_slot
                    .get_mut(slot)
                    .and_then(Option::take)
                    .ok_or_else(|| Error::Other(format!("packed prefill names slot {slot} twice or out of range")))?;
                let body = if mode.is_some() { &tokens[..tokens.len() - 1] } else { tokens };
                packed.push((st, body));
            }
            self.model.forward_packed(&mut packed)?;
            for (st, _) in packed.iter_mut() {
                st.snapshot(&*self.model.backend, keep)?;
            }
        }
        let mut out = Vec::with_capacity(items.len());
        for &(slot, tokens, mode) in items {
            out.push(match mode {
                None => None,
                Some(m) => match self.model.forward_prefill(&mut self.states[slot], &tokens[tokens.len() - 1..], m, None)? {
                    StepOut::None => None,
                    o => Some(o),
                },
            });
        }
        Ok(out)
    }
    fn batched_verify(&self) -> bool {
        self.batched_verify
    }

    fn verify_step(&mut self, rows: &[VerifyRows<HeadMode>]) -> Result<Vec<Result<Verified<StepOut>>>> {
        let m: &Model = &self.model;
        let mut by_slot: Vec<Option<&mut SequenceState>> = self.states.iter_mut().map(Some).collect();
        let mut refs: Vec<&mut SequenceState> = Vec::with_capacity(rows.len());
        for (slot, _, want) in rows {
            if want.is_some_and(|m| !matches!(m, HeadMode::LastLogits | HeadMode::LastTopK(_))) {
                return Err(Error::Unsupported(format!("verify rows in head mode {want:?}")));
            }
            refs.push(
                by_slot
                    .get_mut(*slot)
                    .and_then(Option::take)
                    .ok_or_else(|| Error::Other(format!("verify row names slot {slot} twice or out of range")))?,
            );
        }
        // passes of at most `cap` rows (the workspace and the head's logits rows), sequences in order
        let cap = Model::VERIFY_MAX_ROWS.min(m.ws_rows());
        let mut out: Vec<Result<Verified<StepOut>>> = Vec::with_capacity(rows.len());
        let mut i = 0;
        while i < rows.len() {
            let mut j = i;
            let mut n = 0;
            while j < rows.len() && n + rows[j].1.len() <= cap {
                n += rows[j].1.len();
                j += 1;
            }
            if j == i {
                return Err(Error::invalid(format!("a verify segment of {} rows exceeds the pass's {cap}", rows[i].1.len())));
            }
            let ids: Vec<Vec<u32>> = rows[i..j].iter().map(|(_, v, _)| v.clone()).collect();
            let want: Vec<Option<HeadMode>> = rows[i..j].iter().map(|(_, _, w)| *w).collect();
            // an error here is a device / pass failure: fatal for the whole batch, as a decode step's
            if want.iter().any(Option::is_some) {
                // lane mtp-sampled: the sampled sequences' rows as logits (they draw their own tokens), the others' greedy tokens;
                // lane mtp2s: or as each row's device-side top-k (`LastTopK`, `INTARSIA_VERIFY_TOPK`)
                let (toks, rws) = m.forward_verify_batch_rows(&mut refs[i..j], &ids, &want)?;
                for ((t, r), w) in toks.into_iter().zip(rws).zip(want) {
                    out.push(Ok(if w.is_some() { Verified::Rows(r) } else { Verified::Greedy(t) }));
                }
            } else {
                let res = m.forward_verify_batch(&mut refs[i..j], &ids)?;
                out.extend(res.into_iter().map(|t| Ok(Verified::Greedy(t))));
            }
            i = j;
        }
        Ok(out)
    }

    fn decode_step(&mut self, rows: &[Row<HeadMode>]) -> Result<Vec<Result<StepOut>>> {
        let dm: &Model = self.twin.as_ref().unwrap_or(&self.model);
        // split borrows: one `&mut SequenceState` per row, in row order, from the slot vector
        let mut by_slot: Vec<Option<&mut SequenceState>> = self.states.iter_mut().map(Some).collect();
        let mut refs: Vec<&mut SequenceState> = Vec::with_capacity(rows.len());
        for r in rows {
            let st = by_slot
                .get_mut(r.slot)
                .and_then(Option::take)
                .ok_or_else(|| Error::Other(format!("decode row names slot {} twice or out of range", r.slot)))?;
            refs.push(st);
        }
        // per-row validation first, so a bad row fails its own request and the rest of the batch goes on
        let mut res: Vec<Option<Result<StepOut>>> = (0..rows.len()).map(|_| None).collect();
        let mut good: Vec<usize> = Vec::with_capacity(rows.len());
        let mut tokens: Vec<u32> = Vec::with_capacity(rows.len());
        for (r, row) in rows.iter().enumerate() {
            let mut token = row.token;
            if let Some((slot, len, bad)) = self.row_fault {
                if row.slot == slot && refs[r].len == len {
                    token = bad;
                    self.row_fault = None;
                }
            }
            tokens.push(token);
            match dm.check_batch_row(&*refs[r], token, row.mode) {
                Ok(()) => good.push(r),
                Err(e) => res[r] = Some(Err(e)),
            }
        }
        if !good.is_empty() {
            let ids: Vec<u32> = good.iter().map(|&r| tokens[r]).collect();
            let modes: Vec<HeadMode> = good.iter().map(|&r| rows[r].mode).collect();
            let mut slots: Vec<Option<&mut SequenceState>> = refs.into_iter().map(Some).collect();
            let mut grefs: Vec<&mut SequenceState> = good.iter().map(|&r| slots[r].take().expect("row once")).collect();
            // an error here is a device / pass failure: fatal for the whole batch (the scheduler fails every request)
            let outs = dm.forward_batch(&mut grefs, &ids, &modes)?;
            for (&r, o) in good.iter().zip(outs) {
                res[r] = Some(Ok(o));
            }
        }
        Ok(res.into_iter().map(|o| o.expect("every row has a result")).collect())
    }
}

/// One request on the scheduler.
pub(crate) struct SeqAdapter {
    /// `None` once [`BatchSeq::finish`] consumed it.
    seq: Option<Seq>,
    tx: Sender<GenEvent>,
    telemetry: Arc<Telemetry>,
    plen: usize,
    vocab: usize,
    t_new: Instant,
    /// `request_started` was reported (the request got a slot).
    started: bool,
    /// The scheduler accepted the request into its queue ([`BatchSeq::on_queued`]).
    queued: bool,
    /// `Done` or `Error` was sent.
    ended: bool,
    /// Prefill chunk size the scheduler picked for this request (0: none ran).
    chunk: usize,
    /// The request ended as cancelled (reported `Done` with `Cancelled`), see [`BatchSeq::ended_cancelled`].
    was_cancelled: bool,
    /// Prefix store: when the admission lookup started (the time to first token includes the lookup and the restore).
    t_lookup: Option<Instant>,
}

impl Drop for SeqAdapter {
    /// A request dropped without its `Done` / `Error` (the scheduler panicked while it was detached from the queue and
    /// the slots) still hears of it, instead of finding its channel closed.
    fn drop(&mut self) {
        if !self.ended {
            self.fail(&Error::Other("request aborted: the scheduler failed".into()));
        }
    }
}

impl SeqAdapter {
    fn seq(&mut self) -> Result<&mut Seq> {
        self.seq.as_mut().ok_or_else(|| Error::Other("request already finished".into()))
    }
}

impl BatchSeq<SlotModel> for SeqAdapter {
    fn prompt(&self) -> &[u32] {
        self.seq.as_ref().map_or(&[], |s| &s.req.prompt)
    }

    fn admit(&mut self, cached: usize, cap: usize) -> Result<bool> {
        let (vocab, t_lookup) = (self.vocab, self.t_lookup);
        let seq = self.seq()?;
        if let Some(&bad) = seq.req.prompt.iter().find(|&&t| t as usize >= vocab) {
            return Err(Error::BadRequest(format!("token id {bad} >= vocab {vocab}")));
        }
        let go = seq.admit(cached, cap)?;
        if let Some(t) = t_lookup {
            seq.t0 = t;
        }
        self.started = true;
        self.telemetry.request_started();
        Ok(go)
    }

    fn validate(&self) -> Result<()> {
        let Some(seq) = self.seq.as_ref() else { return Ok(()) };
        if let Some(&bad) = seq.req.prompt.iter().find(|&&t| t as usize >= self.vocab) {
            return Err(Error::BadRequest(format!("token id {bad} >= vocab {}", self.vocab)));
        }
        seq.check_constraint()
    }

    fn first_mode(&self) -> HeadMode {
        self.seq.as_ref().map_or(HeadMode::LastArgmax, |s| s.head_mode(1))
    }

    fn on_prefill_start(&mut self, chunk: usize, _decoding: usize) {
        // a request on the prefix store's schedule runs the model's chunk whatever the scheduler picked
        self.chunk = self.seq.as_ref().and_then(|s| s.ps.as_ref()).map_or(chunk, |ps| ps.chunk);
    }

    fn restore(&mut self, model: &mut SlotModel, slot: usize, local: usize, idle: bool) -> Result<usize> {
        let on = self.seq.as_ref().is_some_and(|s| s.ps.is_some());
        model.store_slot[slot] = on;
        if !on {
            return Ok(local);
        }
        let t = Instant::now();
        self.t_lookup = Some(t);
        let SlotModel { model: m, states, .. } = model;
        let st = &mut states[slot];
        let seq = self.seq.as_mut().expect("checked above");
        let ps = seq.ps.as_ref().expect("checked above");
        // the solo engine's rule (`Seq::new`): bitwise reuse only, what prefill wrote at a point of this prompt's own schedule
        let mut p = ps.local_reuse(st, local.min(st.clean));
        {
            let mut store = ps.pol.store.lock().unwrap_or_else(|e| e.into_inner());
            if idle && store.deferring() {
                // a backstop (the single-sequence engine's "prefill start" settle): captures a failed request left on the device,
                // which no `finish` settled. Only with no other request in flight, whose decode would wait for it
                store.settle(&*m.backend, "admission");
                seq.telemetry.set_prefix_store(store.snapshot());
            }
            // inserts during this prefill copy device-to-device and leave the host copy for `finish` (after the `Done`)
            store.set_defer(true);
        }
        let max_reuse = seq.plen.saturating_sub(1);
        match ps.restore(m, st, &seq.req.prompt, p, max_reuse, &seq.telemetry) {
            Ok(Some((len, took))) => {
                p = len;
                seq.restore_time = took;
            }
            Ok(None) => {}
            Err(e) => {
                // a rejected entry (corrupt file, layout mismatch): run cold
                eprintln!("prefix store: restore failed ({e}); prefilling from scratch");
                st.truncate(0, &*m.backend)?;
                p = 0;
            }
        }
        if let Some(ps) = seq.ps.as_mut() {
            ps.last_insert = p;
        }
        Ok(p)
    }

    fn chunk_len(&self, pos: usize, remaining: usize) -> Option<usize> {
        let ps = self.seq.as_ref()?.ps.as_ref()?;
        // the single-sequence engine's store schedule (`Seq::prefill_step`): the model's chunk cut at the next message boundary
        Some(SeqStore::chunk_end(&ps.boundaries, pos, remaining.min(ps.chunk), ps.chunk) - pos)
    }

    fn own_chunk(&self) -> Option<usize> {
        self.seq.as_ref()?.ps.as_ref().map(|ps| ps.chunk)
    }

    fn on_chunk(&mut self, model: &mut SlotModel, slot: usize, start: usize, end: usize, last: bool) -> Result<()> {
        let Some(seq) = self.seq.as_mut() else { return Ok(()) };
        let Some(ps) = seq.ps.as_ref() else { return Ok(()) };
        let SlotModel { model: m, states, .. } = model;
        let st = &mut states[slot];
        // the prefill-clean watermark, as `Seq::prefill_step` keeps it (only store requests raise it: another request's rows
        // come from another schedule)
        if st.clean == start {
            st.clean = st.len;
        }
        {
            // a finished request's settle ended the deferral: this request's captures still wait for its own `finish`
            let mut store = ps.pol.store.lock().unwrap_or_else(|e| e.into_inner());
            if !store.deferring() {
                store.set_defer(true);
            }
        }
        if !last {
            seq.pos = end;
            return seq.store_at(m, st, false);
        }
        let plen = seq.plen;
        if ps.headers.last() == Some(&plen) {
            // the prompt ends at its last message header: the final pass is that cut's chunk (see `Seq::prefill_step`)
            let at = std::mem::replace(&mut seq.pos, plen);
            let stored = seq.store_at(m, st, false);
            seq.pos = at;
            stored?;
        }
        if let Some(ps) = &seq.ps {
            let mut store = ps.pol.store.lock().unwrap_or_else(|e| e.into_inner());
            store.note_prefill(plen - seq.cached, seq.t0.elapsed().saturating_sub(seq.restore_time));
            seq.telemetry.set_prefix_store(store.snapshot());
        }
        Ok(())
    }

    fn prepare(
        &mut self,
        model: &mut SlotModel,
        slot: usize,
        spec_allowed: bool,
        busy: bool,
    ) -> Result<SeqStep<HeadMode>> {
        let SlotModel { model, twin, states, batched_verify, verify_topk, .. } = model;
        let model: &Model = twin.as_ref().unwrap_or(&**model);
        let defer = *batched_verify && busy;
        let seq = self.seq()?;
        // the busy-batch drafter check exists because a solo verify pass beside a batched step is a whole extra weight pass; a
        // batched verify pass shares the weights with the other drafting sequences, so it does not apply there
        seq.set_spec_allowed(spec_allowed, busy && !defer);
        Ok(match seq.prepare_with(model, &mut states[slot], defer)? {
            Prep::Plain(tok, mode) => {
                // device-side sampling: a sampled row asks for the top-k candidates instead of the whole logits row
                let mode = match (mode, model.device_sampling()) {
                    (HeadMode::LastLogits, true) => seq.topk_for(seq.generated + 1).map_or(mode, HeadMode::LastTopK),
                    _ => mode,
                };
                SeqStep::Plain(tok, mode)
            }
            Prep::Ready => SeqStep::Ready,
            // a sampled sequence draws its own token from every row: its rows come back as logits
            Prep::Verify(ids) => {
                // lane mtp2s (`INTARSIA_VERIFY_TOPK=1`, opt-in): a sampled sequence's rows as the device-side top-k when every row's
                // draw is one a plain batched step would make from it (`Seq::verify_topk`), else whole logits rows
                let mode = match (*verify_topk && model.device_sampling(), seq.verify_topk(ids.len())) {
                    (true, Some(k)) => HeadMode::LastTopK(k),
                    _ => HeadMode::LastLogits,
                };
                SeqStep::Verify(ids, seq.keyed().then_some(mode))
            }
        })
    }

    fn next_ready(&mut self) -> bool {
        self.seq.as_mut().is_some_and(|s| s.next_pending())
    }

    fn accept_verify(&mut self, model: &mut SlotModel, slot: usize, out: Verified<StepOut>) -> Result<()> {
        let SlotModel { model, states, .. } = model;
        let seq = self.seq()?;
        match out {
            Verified::Greedy(out) => seq.accept_verify(model, &mut states[slot], out),
            Verified::Rows(rows) => seq.accept_verify_rows(model, &mut states[slot], rows),
        }
    }

    fn accept(&mut self, out: StepOut) -> Result<()> {
        let seq = self.seq()?;
        if seq.awaiting_first() {
            seq.accept_first(out)
        } else {
            seq.accept_plain(out)
        }
    }

    fn on_token(&mut self) -> Result<bool> {
        self.seq()?.on_token()
    }

    fn cancelled(&self) -> bool {
        self.seq.as_ref().is_some_and(|s| s.cancelled())
    }

    fn ended_cancelled(&self) -> bool {
        self.was_cancelled
    }

    fn finish(&mut self, model: &mut SlotModel, slot: Option<usize>, cancelled: bool) -> Result<()> {
        let mut seq = self.seq.take().ok_or_else(|| Error::Other("request already finished".into()))?;
        if cancelled {
            seq.cancel();
        }
        let store = seq.ps.as_ref().map(|ps| ps.pol.store.clone());
        let outcome = match slot {
            Some(s) => seq.finish(&model.model, &mut model.states[s])?,
            None => seq.queued_outcome(FinishReason::Cancelled),
        };
        if !self.started {
            // never held a slot: balance the counters (enqueued -> started -> finished)
            self.started = true;
            self.telemetry.request_started();
        }
        let kv = model.kv_totals();
        self.telemetry.set_kv(kv.0, kv.1, kv.2, kv.3);
        self.ended = true;
        self.was_cancelled = outcome.reason == FinishReason::Cancelled;
        report_done(&self.telemetry, &self.tx, outcome, Some(self.chunk));
        if let (Some(store), Some(_)) = (store, slot) {
            // after the reply (`settle_after_done` of the single-sequence engine): the captures this request (and any other
            // still prefilling) left on the device move to their tier. By default here, in one piece, on the scheduler thread (the
            // other sequences' next decode step waits for it); with `INTARSIA_SETTLE_SLICE_MB` > 0 (lane interleave, opt-in) in
            // slices between later decode steps instead, and whole once the engine is idle (`SlotModel::housekeep`). A request
            // that never held a slot (cancelled while queued) captured nothing and settles nothing.
            let mut s = store.lock().unwrap_or_else(|e| e.into_inner());
            if s.deferring() {
                if model.settle_slice == 0 {
                    s.settle(&*model.model.backend, "after done");
                    self.telemetry.set_prefix_store(s.snapshot());
                } else if model.settle.is_none() {
                    drop(s);
                    model.settle = Some((store, self.telemetry.clone()));
                }
            }
        }
        Ok(())
    }

    fn on_queued(&mut self) {
        self.queued = true;
    }

    fn fail(&mut self, err: &Error) {
        if self.ended {
            return;
        }
        self.ended = true;
        if self.queued && !self.started {
            // accepted, then failed before it held a slot (engine died, admission refused): stays in `requests_total`
            self.telemetry.request_dropped_queued("error");
            let _ = self.tx.send(GenEvent::Error { message: err.to_string() });
        } else if self.started {
            let generated = self.seq.as_ref().map_or(0, |s| s.generated);
            report_error(&self.telemetry, &self.tx, self.plen, generated, self.t_new.elapsed(), err);
        } else {
            // refused at submit (never queued): undo `request_enqueued`
            self.telemetry.request_rejected();
            let _ = self.tx.send(GenEvent::Error { message: err.to_string() });
        }
    }
}

/// Engine that decodes up to `max_batch` requests together (continuous batching, Metal only).
pub struct BatchedEngine {
    env: SeqEnv,
    engine: BatchEngine<SlotModel, SeqAdapter>,
    info: ModelInfo,
    plan: Value,
    vocab: usize,
    verify_rows: usize,
    /// The slots hold recurrent state (hybrid opt-in): no sequence drafts (`Seq::build`).
    recurrent: bool,
    /// Lane mtp-sampled: sampled requests draft too ([`BatchedOptions::mtp_sampled`] under `--spec mtp`).
    mtp_sampled: bool,
    /// The prefix store (`EngineMeta::prefix_store`), plain decode only (`BatchedEngine::new` refuses speculation).
    store: Option<StorePolicy>,
    /// The model's prefill chunk: the store schedule's chunk.
    solo_chunk: usize,
}

impl BatchedEngine {
    /// `model` must have been prepared with `Model::enable_batch(opts.max_batch)` (and `enable_verify` if `meta.spec`
    /// is on) before `configure_plan`; `enable_batch` refuses what cannot batch exactly, and `intarsia_models::load_engine`
    /// additionally refuses backends without row-exact kernels (the cpu backend, exempt there so tests can exercise the
    /// scheduler, serves one row at a time).
    pub fn new(
        mut model: Model,
        codec: Arc<dyn TextCodec>,
        meta: EngineMeta,
        telemetry: Arc<Telemetry>,
        opts: BatchedOptions,
    ) -> Result<BatchedEngine> {
        if opts.max_batch < 2 || opts.max_batch > MAX_BATCH_CAP {
            return Err(Error::invalid(format!(
                "the batched engine serves --max-batch 2..={MAX_BATCH_CAP} (got {}); 1 uses the single-sequence engine",
                opts.max_batch
            )));
        }
        if model.batch_rows() < opts.max_batch {
            return Err(Error::invalid(format!(
                "model prepared for {} batched rows, --max-batch is {} (Model::enable_batch before configure_plan)",
                model.batch_rows(),
                opts.max_batch
            )));
        }
        if meta.prefix_store.is_some() && meta.spec.enabled() {
            // slice 1 of the store here is plain decode: batched MTP turns slot reuse off (a drafting request would get no reuse at
            // all), and a restore into a slot holding a drafter's cache is not synced the way `Seq::new` syncs it
            return Err(Error::Unsupported(format!(
                "the prefix store with --max-batch > 1 and speculative decoding ({:?}; --spec off, or use --max-batch 1)",
                meta.spec.mode
            )));
        }
        let mtp = meta.spec.mode == crate::spec::SpecMode::Mtp;
        if mtp && opts.mtp_max_b == 0 {
            return Err(Error::Unsupported(
                "--spec mtp with the batching engine (use --max-batch 1, or opt in with INTARSIA_BATCHED_MTP=N)".into(),
            ));
        }
        if mtp && !model.has_mtp() {
            return Err(Error::invalid("--spec mtp needs the model loaded with its MTP head (LoadOptions::mtp)"));
        }
        if meta.spec.enabled() && model.verify_rows() < meta.spec.verify_rows() {
            return Err(Error::invalid(format!(
                "speculative decoding with k = {} needs verify passes of {} rows; the model was prepared for {} \
                 (Model::enable_verify before configure_plan)",
                meta.spec.k,
                meta.spec.verify_rows(),
                model.verify_rows()
            )));
        }
        let mut states = Vec::with_capacity(opts.max_batch);
        for _ in 0..opts.max_batch {
            states.push(model.new_state()?);
        }
        let recurrent = states[0].has_recurrent();
        if recurrent {
            // opt-in cpu prototype (`Model::allow_recurrent_batch`, docs/results/2026-10-23-rb-recurrent-batching.md)
            if !model.recurrent_batch() {
                return Err(Error::Unsupported("continuous batching on a model with recurrent state".into()));
            }
            // batched MTP rolls a sequence's recurrent state back through its verify pass's per-row copies (lane bm)
            if meta.spec.enabled() && !mtp {
                return Err(Error::Unsupported(
                    "speculative decoding on a model with recurrent state needs the single-sequence engine (--max-batch 1)".into(),
                ));
            }
        }
        let store = StorePolicy::new(&model, &*codec, &meta, &states[0])?;
        if store.is_some() {
            store_combinations(&mut model, &opts)?;
        }
        let cap = states[0].cap;
        let total_kv: u64 = states.iter().map(|s| s.total_bytes() as u64).sum();
        let dinfo = model.backend.info();
        let info = build_info(&model, &dinfo, &meta, &*codec, cap);
        let mut plan = crate::plan::plan_json(&model, &meta.id);
        if let Some(o) = plan.as_object_mut() {
            o.insert("max_batch".into(), opts.max_batch.into());
        }
        telemetry.set_model(&meta.id, dinfo, model.weights_bytes);
        let backend = model.backend.clone();
        telemetry.set_device_probe(Box::new(move || backend.info()));
        telemetry.set_kv(0, total_kv, 0, cap as u64 * opts.max_batch as u64);

        let solo_chunk = model.prefill_chunk().max(1);
        let mut cfg = opts.scheduler_config(solo_chunk);
        if opts.pack_prefill.is_none() {
            // the prefix store turns the backend's default off (its requests never pack; `store_combinations` refuses the opt-in)
            cfg.pack_prefill = model.backend.pack_prefill_default();
            if cfg.pack_prefill && store.is_some() {
                eprintln!("prefix store: the backend's packed prefill is off with --max-batch > 1");
                cfg.pack_prefill = false;
            }
        }
        if mtp {
            // lane bm: the per-step choice. A sequence drafts (its own draft and verify pass on its slot, `Seq::prepare`) only while
            // at most `mtp_max_b` sequences decode; a wider step is plain batched decode, which only queues the heads' rows
            cfg.spec_off_at = cfg.spec_off_at.min(opts.mtp_max_b + 1);
            // a reused prefix would need the head's cache to cover it: every request starts its slot empty (which also syncs the head)
            cfg.prefix_reuse = false;
        }
        if recurrent && model.batch_snapshots() == 0 {
            // no batched snapshots on this backend: a recurrent slot cannot be rewound, so no slot is a reuse candidate (every
            // request starts from an empty slot; with the prefix store on, its admission restores from the store instead)
            cfg.prefix_reuse = false;
        }
        let vocab = model.dims.vocab;
        let verify_rows = model.verify_rows();
        if let Some(s) = &store {
            s.announce(&telemetry);
        }
        let env = SeqEnv { codec, telemetry, eos: meta.eos_ids, spec: meta.spec };
        let twin = if opts.overlap {
            match model.decode_twin(opts.max_batch) {
                Ok(t) => Some(t),
                Err(e) => {
                    eprintln!("prefill/decode overlap unavailable ({e}); chunks run to completion before decode steps");
                    None
                }
            }
        } else {
            None
        };
        // lane bm phase B: `INTARSIA_BATCHED_VERIFY=1` (opt-in, with batched MTP) verifies the drafting sequences in one pass
        let batched_verify = mtp && twin.is_none() && opts.batched_verify;
        let mtp_sampled = mtp && opts.mtp_sampled;
        if batched_verify && opts.verify_topk {
            eprintln!("batched verify: sampled rows as the device-side top-k when the request samples from it (INTARSIA_VERIFY_TOPK=1)");
        }
        if mtp {
            eprintln!(
                "batched MTP: sampled requests (temperature > 0) {}",
                if mtp_sampled { "draft too, seed-exact (INTARSIA_BATCHED_MTP_SAMPLED=0: plain steps)" } else { "take plain steps (INTARSIA_BATCHED_MTP_SAMPLED=0)" }
            );
        }
        let store_slot = vec![false; states.len()];
        let slots =
            SlotModel {
                model: Arc::new(model),
                twin,
                states,
                row_fault: opts.row_fault,
                inflight: None,
                batched_verify,
                verify_topk: batched_verify && opts.verify_topk,
                pause_rows: opts.pause_rows,
                settle: None,
                store: store.as_ref().map(|p| (p.store.clone(), env.telemetry.clone())),
                settle_slice: (opts.settle_slice_mb as u64) << 20,
                store_slot,
            };
        let engine = if opts.hold_start { BatchEngine::spawn_held(slots, cfg)? } else { BatchEngine::spawn(slots, cfg)? };
        Ok(BatchedEngine { env, engine, info, plan, vocab, verify_rows, recurrent, store, solo_chunk, mtp_sampled })
    }

    /// Let an engine built with [`BatchedOptions::hold_start`] start scheduling (no effect on any other engine).
    #[doc(hidden)]
    pub fn release(&self) {
        self.engine.release();
    }

    /// The scheduler's metrics (batch-size histogram, queue depth, TTFT / inter-token latency, ...).
    pub fn metrics(&self) -> BatchMetrics {
        self.engine.metrics()
    }
}

/// The prefix store with the batching engine, slice 1 (docs/tuning.md, `--prefix-store`): plain decode only. Speculative
/// decoding is refused, and so are the opt-ins that change how a prompt is chunked or where its chunk runs (`INTARSIA_OVERLAP=1`,
/// `INTARSIA_PACK_PREFILL=1`, `INTARSIA_PREFILL_MACRO` > 0; tensor parallelism is refused by `StorePolicy::new`); a recipe's macro
/// prefill default is turned off here, the backend's packed-prefill default in [`BatchedEngine::new`].
fn store_combinations(model: &mut Model, opts: &BatchedOptions) -> Result<()> {
    let refuse = |what: &str| Err(Error::Unsupported(format!("the prefix store with --max-batch > 1 and {what} (unset it, or use --max-batch 1)")));
    if opts.overlap {
        return refuse("prefill/decode overlap (INTARSIA_OVERLAP=1)");
    }
    if opts.pack_prefill == Some(true) {
        return refuse("packed prefill (INTARSIA_PACK_PREFILL=1)");
    }
    let env = crate::scheduler::parse_macro_env(std::env::var(crate::scheduler::MACRO_ENV).ok().as_deref())?;
    if env.is_some_and(|n| n > 0) {
        return refuse("macro prefill (INTARSIA_PREFILL_MACRO)");
    }
    if model.macro_rows() > 0 {
        // `load_engine` turns the recipe's macro prefill off before the weights are read (the workspace is then planned without it);
        // a model loaded another way keeps the workspace sized for the macro pass (`Model::set_macro_rows` only changes the schedule)
        eprintln!("prefix store: the recipe's macro prefill ({} rows) is off with --max-batch > 1", model.macro_rows());
        model.set_macro_rows(0);
    }
    eprintln!("prefix store with --max-batch > 1: workspace {} MiB", model.plan.workspace_bytes >> 20);
    Ok(())
}

fn needs_solo(what: &str) -> Error {
    Error::Unsupported(format!("{what} needs the single-sequence engine (--max-batch 1)"))
}

impl Engine for BatchedEngine {
    fn info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn tokenize(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        self.env.codec.encode(text, add_special_tokens)
    }

    fn detokenize(&self, ids: &[u32]) -> Result<String> {
        self.env.codec.decode(ids, false)
    }

    fn apply_chat_template(&self, messages: &[Value], opts: &ChatOpts) -> Result<String> {
        self.env.codec.apply_chat_template(messages, opts)
    }

    fn tokenizer_json(&self) -> Option<String> {
        self.env.codec.tokenizer_json()
    }

    fn submit(&self, req: GenRequest) -> Result<Receiver<GenEvent>> {
        if let Some(why) = self.engine.failed() {
            return Err(Error::Other(format!("engine failed: {why}")));
        }
        // refused here, so the server answers 400: from `admit` it reached the client as a 500 (a 200 stream with an error
        // event when streaming); `admit` keeps its check for the slot path
        super::check_prompt_ids(&req.prompt, self.vocab)?;
        if req.prewarm.is_some() {
            return Err(Error::BadRequest("prewarm needs the single-sequence engine with the prefix store (--max-batch 1)".into()));
        }
        let (etx, erx) = channel();
        let plen = req.prompt.len();
        let mut seq = Seq::build(&self.env, req, etx.clone(), self.verify_rows, self.recurrent);
        if self.mtp_sampled {
            // lane mtp-sampled: a sampled request drafts too, seed-exact (`engine/keyed.rs`)
            seq.enable_keyed(self.env.spec.mode, self.verify_rows);
        }
        if let Some(pol) = &self.store {
            // plain decode only: no request drafts with the store on (`BatchedEngine::new` refuses speculation)
            seq.ps = Some(SeqStore::new(pol.clone(), &seq.req.prompt, self.recurrent, self.solo_chunk));
        }
        let adapter = SeqAdapter {
            seq: Some(seq),
            tx: etx,
            telemetry: self.env.telemetry.clone(),
            plen,
            vocab: self.vocab,
            t_new: Instant::now(),
            started: false,
            queued: false,
            ended: false,
            chunk: 0,
            was_cancelled: false,
            t_lookup: None,
        };
        self.env.telemetry.request_enqueued();
        // a refusal calls `adapter.fail`, which undoes `request_enqueued`
        let _id: RequestId = self.engine.submit(adapter)?;
        Ok(erx)
    }

    fn failed(&self) -> Option<String> {
        self.engine.failed()
    }

    fn forward_logits_each(&self, _ids: &[u32], _cb: &mut dyn FnMut(usize, &[f32]) -> Result<()>) -> Result<()> {
        Err(needs_solo("forward_logits"))
    }

    fn forward_logits_decode(
        &self,
        _ids: &[u32],
        _n_decode: usize,
        _cb: &mut dyn FnMut(usize, &[f32]) -> Result<()>,
    ) -> Result<()> {
        Err(needs_solo("forward_logits_decode"))
    }

    fn forward_script(&self, _steps: &[ScriptStep]) -> Result<Vec<Option<Vec<f32>>>> {
        Err(needs_solo("forward_script"))
    }

    fn plan(&self) -> Value {
        self.plan.clone()
    }

    fn telemetry(&self) -> TelemetrySnapshot {
        self.env.telemetry.snapshot()
    }

    fn batch_metrics(&self) -> Option<BatchMetrics> {
        Some(self.engine.metrics())
    }

    fn prefix_store(&self) -> Value {
        self.store.as_ref().map_or(Value::Null, StorePolicy::info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoCodec;
    impl TextCodec for NoCodec {
        fn encode(&self, _: &str, _: bool) -> Result<Vec<u32>> {
            Ok(vec![])
        }
        fn decode(&self, _: &[u32], _: bool) -> Result<String> {
            Ok(String::new())
        }
        fn apply_chat_template(&self, _: &[Value], _: &ChatOpts) -> Result<String> {
            Ok(String::new())
        }
        fn has_chat_template(&self) -> bool {
            false
        }
    }

    fn adapter(started: bool) -> (SeqAdapter, Receiver<GenEvent>, Arc<Telemetry>) {
        let tel = Arc::new(Telemetry::new());
        let env = SeqEnv { codec: Arc::new(NoCodec), telemetry: tel.clone(), eos: vec![], spec: Default::default() };
        let (tx, rx) = channel();
        let seq = Seq::build(&env, GenRequest { prompt: vec![1, 2], ..Default::default() }, tx.clone(), 0, false);
        tel.request_enqueued();
        if started {
            tel.request_started();
        }
        let a = SeqAdapter {
            seq: Some(seq),
            tx,
            telemetry: tel.clone(),
            plen: 2,
            vocab: 10,
            t_new: Instant::now(),
            started,
            queued: false,
            ended: false,
            chunk: 0,
            was_cancelled: false,
            t_lookup: None,
        };
        (a, rx, tel)
    }

    /// A request dropped by an unwinding scheduler (detached from the queue and the slots) gets an `Error`, and its
    /// telemetry counters balance, instead of a silently closed channel.
    #[test]
    fn dropped_request_hears_an_error() {
        for started in [false, true] {
            let (a, rx, tel) = adapter(started);
            drop(a);
            match rx.try_recv() {
                Ok(GenEvent::Error { message }) => assert!(message.contains("aborted"), "{message}"),
                other => panic!("expected an error event, got {other:?}"),
            }
            let s = tel.snapshot();
            assert_eq!((s.running, s.waiting), (0, 0), "started {started}: {s:?}");
        }
    }

    /// Tokens a request generated (and streamed) before it failed count in `generation_tokens_total`; a request that
    /// never held a slot adds none.
    #[test]
    fn failed_request_counts_its_generated_tokens() {
        let (mut a, _rx, tel) = adapter(true);
        a.seq.as_mut().unwrap().generated = 5;
        a.fail(&Error::Backend("injected".into()));
        let s = tel.snapshot();
        assert_eq!(s.generation_tokens_total, 5, "{s:?}");
        let (mut q, _rx, tel) = adapter(false);
        q.fail(&Error::Backend("injected".into()));
        assert_eq!(tel.snapshot().generation_tokens_total, 0);
    }

    /// A request the scheduler queued and then failed before it started (engine died, admission refused) stays in
    /// `requests_total` and is counted as an error; a refusal at submit (never queued) is undone.
    #[test]
    fn queued_then_failed_request_stays_counted() {
        let (mut q, rx, tel) = adapter(false);
        q.on_queued();
        q.fail(&Error::Other("engine failed: boom".into()));
        assert!(matches!(rx.try_recv(), Ok(GenEvent::Error { .. })));
        let s = tel.snapshot();
        assert_eq!((s.requests_total, s.waiting, s.running), (1, 0, 0), "{s:?}");
        assert_eq!(s.finished["error"], 1);

        let (mut r, rx, tel) = adapter(false);
        r.fail(&Error::QueueFull(1));
        assert!(matches!(rx.try_recv(), Ok(GenEvent::Error { .. })));
        let s = tel.snapshot();
        assert_eq!((s.requests_total, s.waiting), (0, 0), "{s:?}");
        assert!(s.finished.is_empty());
    }

    #[test]
    fn finished_request_is_not_reported_twice() {
        let (mut a, rx, _) = adapter(true);
        a.fail(&Error::Other("x".into()));
        drop(a);
        assert!(matches!(rx.try_recv(), Ok(GenEvent::Error { .. })));
        assert!(rx.try_recv().is_err(), "a second event after the first Error");
    }

    #[test]
    fn spec_off_at_reaches_the_scheduler_config() {
        let mut o = BatchedOptions::new(4, 16);
        assert_eq!(o.scheduler_config(2048).spec_off_at, SPEC_OFF_AT_DEFAULT);
        o.spec_off_at = 7;
        let c = o.scheduler_config(2048);
        assert_eq!((c.spec_off_at, c.max_batch, c.max_queue), (7, 4, 16));
        assert!(!c.trace, "production never stores the trace");
        assert!(c.validate().is_ok());
    }

    #[test]
    fn busy_chunk_reaches_the_scheduler_config() {
        let mut o = BatchedOptions::new(4, 16);
        assert_eq!(o.scheduler_config(2048).busy_chunk, BUSY_PREFILL_CHUNK);
        assert_eq!(BUSY_PREFILL_CHUNK, 256, "the default decode-time prefill chunk stays 256");
        o.busy_chunk = 320;
        let c = o.scheduler_config(2048);
        assert_eq!((c.busy_chunk, c.solo_chunk), (320, 2048));
        assert!(c.validate().is_ok());
        // never above the model's chunk
        assert_eq!(o.scheduler_config(128).busy_chunk, 128);
    }
}
