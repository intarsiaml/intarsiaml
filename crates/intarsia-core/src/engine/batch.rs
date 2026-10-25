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
    build_info, report_done, report_error, ChatOpts, Engine, EngineMeta, FinishReason, GenEvent, GenRequest, ModelInfo,
    ScriptStep, Seq, SeqEnv, TextCodec,
};
use crate::error::{Error, Result};
use crate::model::{HeadMode, Model, StepOut};
use crate::scheduler::{
    BatchEngine, BatchMetrics, BatchModel, BatchSeq, RequestId, Row, SchedulerConfig, SeqStep, MAX_BATCH_CAP,
    SPEC_OFF_AT_DEFAULT,
};
use crate::state::SequenceState;
use crate::telemetry::{Telemetry, TelemetrySnapshot};

/// Prefill chunk used while other sequences decode (design 8.1: 256 while others decode, the model's chunk alone).
pub const BUSY_PREFILL_CHUNK: usize = 256;

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
    /// Packed prefill (`INTARSIA_PACK_PREFILL=1`, opt-in): the next chunks of several waiting prompts run in one pass
    /// (`Model::forward_packed`) on a hybrid (recurrent) model; each prompt's state is its solo prefill's when every row-wise kernel is
    /// independent of the other rows of the call.
    pub pack_prefill: bool,
}

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
            pack_prefill: self.pack_prefill,
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
            pack_prefill: std::env::var("INTARSIA_PACK_PREFILL").map(|v| v == "1").unwrap_or(false),
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
}

/// A prefill chunk on its own thread. The slot's state travels with it (a placeholder holds the slot meanwhile) and comes back
/// at `prefill_end`, so no state is shared between the threads.
struct Inflight {
    slot: usize,
    done: Arc<std::sync::atomic::AtomicBool>,
    handle: std::thread::JoinHandle<(SequenceState, Result<Option<StepOut>>)>,
}

/// One prefill chunk of `st` (the body of [`BatchModel::prefill_chunk`], also what the background thread runs).
fn run_prefill(model: &Model, st: &mut SequenceState, tokens: &[u32], mode: Option<HeadMode>) -> Result<Option<StepOut>> {
    if st.has_recurrent() {
        // the solo schedule of a recurrent state (`Seq::prefill_step`): a snapshot after every non-final chunk, and
        // before the final chunk's last token, which runs alone (so a repeated prompt reuses all but that token)
        let keep = model.batch_snapshots();
        return Ok(match mode {
            None => {
                if !matches!(model.forward(st, tokens, HeadMode::None, None)?, StepOut::None) {
                    return Err(Error::Other("a headless prefill pass returned an output".into()));
                }
                st.snapshot(&*model.backend, keep)?;
                None
            }
            Some(m) => {
                if tokens.len() > 1 {
                    // nothing is asked of this pass: a head output here would be a bug, not a result to drop
                    if !matches!(model.forward(st, &tokens[..tokens.len() - 1], HeadMode::None, None)?, StepOut::None) {
                        return Err(Error::Other("a headless prefill pass returned an output".into()));
                    }
                    st.snapshot(&*model.backend, keep)?;
                }
                match model.forward(st, &tokens[tokens.len() - 1..], m, None)? {
                    StepOut::None => None,
                    o => Some(o),
                }
            }
        });
    }
    let out = model.forward(st, tokens, mode.unwrap_or(HeadMode::None), None)?;
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
    fn prefill_chunk(&mut self, slot: usize, tokens: &[u32], mode: Option<HeadMode>) -> Result<Option<StepOut>> {
        run_prefill(&self.model, &mut self.states[slot], tokens, mode)
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
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = done.clone();
        let handle = std::thread::Builder::new()
            .name("intarsia-prefill".into())
            .spawn(move || {
                let r = run_prefill(&model, &mut st, &tokens, mode);
                flag.store(true, std::sync::atomic::Ordering::Release);
                (st, r)
            })
            .map_err(|e| Error::Other(format!("cannot start the prefill thread: {e}")))?;
        self.inflight = Some(Inflight { slot, done, handle });
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
        Model::PACK_MIN_ROWS
    }
    fn prefill_pack(&mut self, items: &[(usize, &[u32], Option<HeadMode>)]) -> Result<Vec<Option<StepOut>>> {
        // the schedule of `run_prefill` for a recurrent state, with the bodies of all chunks in one pass: the body is the chunk
        // (non-final) or the chunk without its last token (final); a snapshot follows each body; the last token runs alone
        let keep = self.model.batch_snapshots();
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
                Some(m) => match self.model.forward(&mut self.states[slot], &tokens[tokens.len() - 1..], m, None)? {
                    StepOut::None => None,
                    o => Some(o),
                },
            });
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
        let vocab = self.vocab;
        let seq = self.seq()?;
        if let Some(&bad) = seq.req.prompt.iter().find(|&&t| t as usize >= vocab) {
            return Err(Error::BadRequest(format!("token id {bad} >= vocab {vocab}")));
        }
        let go = seq.admit(cached, cap)?;
        self.started = true;
        self.telemetry.request_started();
        Ok(go)
    }

    fn first_mode(&self) -> HeadMode {
        self.seq.as_ref().map_or(HeadMode::LastArgmax, |s| s.head_mode(1))
    }

    fn on_prefill_start(&mut self, chunk: usize, _decoding: usize) {
        self.chunk = chunk;
    }

    fn prepare(
        &mut self,
        model: &mut SlotModel,
        slot: usize,
        spec_allowed: bool,
        busy: bool,
    ) -> Result<SeqStep<HeadMode>> {
        let SlotModel { model, twin, states, .. } = model;
        let model: &Model = twin.as_ref().unwrap_or(&**model);
        let seq = self.seq()?;
        seq.set_spec_allowed(spec_allowed, busy);
        Ok(match seq.prepare(model, &mut states[slot])? {
            Some((tok, mode)) => {
                // device-side sampling: a sampled row asks for the top-k candidates instead of the whole logits row
                let mode = match (mode, model.device_sampling()) {
                    (HeadMode::LastLogits, true) => seq.topk_for(seq.generated + 1).map_or(mode, HeadMode::LastTopK),
                    _ => mode,
                };
                SeqStep::Plain(tok, mode)
            }
            None => SeqStep::Ready,
        })
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
}

impl BatchedEngine {
    /// `model` must have been prepared with `Model::enable_batch(opts.max_batch)` (and `enable_verify` if `meta.spec`
    /// is on) before `configure_plan`; `enable_batch` refuses what cannot batch exactly, and `intarsia_models::load_engine`
    /// additionally refuses backends without row-exact kernels (the cpu backend, exempt there so tests can exercise the
    /// scheduler, serves one row at a time).
    pub fn new(
        model: Model,
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
        if meta.spec.mode == crate::spec::SpecMode::Mtp {
            return Err(Error::Unsupported("--spec mtp with the batching engine (use --max-batch 1)".into()));
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
            if meta.spec.enabled() {
                return Err(Error::Unsupported(
                    "speculative decoding on a model with recurrent state needs the single-sequence engine (--max-batch 1)".into(),
                ));
            }
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
        if recurrent && model.batch_snapshots() == 0 {
            // the prefix store stays at batch 1 on this backend: a recurrent slot cannot be rewound, so no slot is a reuse
            // candidate (every request starts from an empty slot)
            cfg.prefix_reuse = false;
        }
        let vocab = model.dims.vocab;
        let verify_rows = model.verify_rows();
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
        let engine = BatchEngine::spawn(
            SlotModel { model: Arc::new(model), twin, states, row_fault: opts.row_fault, inflight: None },
            cfg,
        )?;
        Ok(BatchedEngine { env, engine, info, plan, vocab, verify_rows, recurrent })
    }

    /// The scheduler's metrics (batch-size histogram, queue depth, TTFT / inter-token latency, ...).
    pub fn metrics(&self) -> BatchMetrics {
        self.engine.metrics()
    }
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
        let (etx, erx) = channel();
        let plen = req.prompt.len();
        let seq = Seq::build(&self.env, req, etx.clone(), self.verify_rows, self.recurrent);
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
