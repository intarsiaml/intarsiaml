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
    /// Log scheduler events to stderr (`INTARSIA_CB_TRACE=1`).
    pub trace: bool,
    /// Test seam: feed `token` once instead of the real one in the decode row of slot `.0` whose slot holds `.1` tokens
    /// (error-isolation tests put an invalid id into one row). Always `None` outside tests.
    #[doc(hidden)]
    pub row_fault: Option<(usize, usize, u32)>,
}

impl BatchedOptions {
    pub fn new(max_batch: usize, max_queue: usize) -> Self {
        let env = |k: &str| std::env::var(k).map(|v| v == "1").unwrap_or(false);
        BatchedOptions {
            max_batch,
            max_queue,
            prefix_reuse: !env("INTARSIA_CB_NO_REUSE"),
            trace: env("INTARSIA_CB_TRACE"),
            row_fault: None,
        }
    }
}

/// The model side of the scheduler: one KV slot per `SequenceState`.
pub(crate) struct SlotModel {
    model: Model,
    states: Vec<SequenceState>,
    /// [`BatchedOptions::row_fault`], cleared once it fired.
    row_fault: Option<(usize, usize, u32)>,
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
    fn prefill_chunk(&mut self, slot: usize, tokens: &[u32], mode: Option<HeadMode>) -> Result<Option<StepOut>> {
        let out = self.model.forward(&mut self.states[slot], tokens, mode.unwrap_or(HeadMode::None), None)?;
        Ok(match out {
            StepOut::None => None,
            o => Some(o),
        })
    }
    fn avoid_single_row_prefill(&self) -> bool {
        // the backend flag (Metal), except for recurrent state (never batched; mirrors `Seq::prefill_step`)
        self.model.backend.avoid_single_row_prefill() && !self.states[0].has_recurrent()
    }
    fn decode_step(&mut self, rows: &[Row<HeadMode>]) -> Result<Vec<Result<StepOut>>> {
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
            match self.model.check_batch_row(&*refs[r], token, row.mode) {
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
            let outs = self.model.forward_batch(&mut grefs, &ids, &modes)?;
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
    /// `Done` or `Error` was sent.
    ended: bool,
    /// Prefill chunk size the scheduler picked for this request (0: none ran).
    chunk: usize,
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

    fn prepare(&mut self, model: &mut SlotModel, slot: usize, spec_allowed: bool) -> Result<SeqStep<HeadMode>> {
        let SlotModel { model, states, .. } = model;
        let seq = self.seq()?;
        seq.set_spec_allowed(spec_allowed);
        Ok(match seq.prepare(model, &mut states[slot])? {
            Some((tok, mode)) => SeqStep::Plain(tok, mode),
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
        report_done(&self.telemetry, &self.tx, outcome, self.chunk);
        Ok(())
    }

    fn fail(&mut self, err: &Error) {
        if self.ended {
            return;
        }
        self.ended = true;
        if self.started {
            report_error(&self.telemetry, &self.tx, self.plen, self.t_new.elapsed(), err);
        } else {
            // refused before it held a slot: undo `request_enqueued`
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
        if states[0].has_recurrent() {
            return Err(Error::Unsupported("continuous batching on a model with recurrent state".into()));
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
        let cfg = SchedulerConfig {
            max_batch: opts.max_batch,
            max_queue: opts.max_queue,
            solo_chunk,
            busy_chunk: BUSY_PREFILL_CHUNK.min(solo_chunk),
            prefix_reuse: opts.prefix_reuse,
            trace: opts.trace,
            ..SchedulerConfig::default()
        };
        let vocab = model.dims.vocab;
        let verify_rows = model.verify_rows();
        let env = SeqEnv { codec, telemetry, eos: meta.eos_ids, spec: meta.spec };
        let engine = BatchEngine::spawn(SlotModel { model, states, row_fault: opts.row_fault }, cfg)?;
        Ok(BatchedEngine { env, engine, info, plan, vocab, verify_rows })
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

    fn submit(&self, req: GenRequest) -> Result<Receiver<GenEvent>> {
        if let Some(why) = self.engine.failed() {
            return Err(Error::Other(format!("engine failed: {why}")));
        }
        let (etx, erx) = channel();
        let plen = req.prompt.len();
        let seq = Seq::build(&self.env, req, etx.clone(), self.verify_rows, false);
        let adapter = SeqAdapter {
            seq: Some(seq),
            tx: etx,
            telemetry: self.env.telemetry.clone(),
            plen,
            vocab: self.vocab,
            t_new: Instant::now(),
            started: false,
            ended: false,
            chunk: 0,
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
