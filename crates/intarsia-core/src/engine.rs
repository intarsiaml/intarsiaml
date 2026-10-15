//! Engine API consumed by the server, CLI, accuracy and bench tools, plus the
//! Phase 0 implementation: one worker thread, FIFO queue, a single sequence.
//!
//! `Engine` is object safe: hold it as `Arc<dyn Engine>`.

use crate::error::{Error, Result};
use crate::model::{HeadMode, Model, StepOut};
use crate::sampler::{Sampler, SamplingParams};
use crate::state::SequenceState;
use crate::telemetry::{Telemetry, TelemetrySnapshot};
use serde::Serialize;
use serde_json::Value;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// public data types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct GenRequest {
    /// Prompt token ids (already tokenized / templated by the caller).
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    /// EOS (and `stop_ids`) are suppressed until this many tokens exist.
    pub min_tokens: usize,
    /// Keep generating past the model's EOS tokens (forced-length benchmarks).
    pub ignore_eos: bool,
    /// <= 0 means greedy.
    pub temperature: f32,
    pub top_p: f32,
    /// 0 = disabled.
    pub top_k: usize,
    pub seed: Option<u64>,
    /// Extra stop token ids (not emitted).
    pub stop_ids: Vec<u32>,
    /// Stop strings; generation ends before the first match, which is not emitted.
    pub stop: Vec<String>,
    /// Set by the caller to cancel the request: checked while queued, between
    /// prefill chunks and between decode steps.
    pub cancel: Option<Arc<AtomicBool>>,
    /// Accuracy tooling: every sampling step appends its full logits row (before the `min_tokens` mask) here. The
    /// engine then takes the host-logits path for every step instead of the on-device argmax.
    pub trace_logits: Option<Arc<Mutex<Vec<Vec<f32>>>>>,
}

impl Default for GenRequest {
    fn default() -> Self {
        GenRequest {
            prompt: Vec::new(),
            max_tokens: 256,
            min_tokens: 0,
            ignore_eos: false,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: None,
            stop_ids: Vec::new(),
            stop: Vec::new(),
            cancel: None,
            trace_logits: None,
        }
    }
}

/// One step of [`Engine::forward_script`].
#[derive(Clone, Debug)]
pub struct ScriptStep {
    pub ids: Vec<u32>,
    /// Return the logits row of the last token of `ids`.
    pub logits: bool,
    /// Roll the sequence back to this many tokens first (what `generate` does to reuse a prefix).
    pub truncate: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// EOS, a stop id or a stop string.
    Stop,
    /// `max_tokens` (or the context window) reached.
    Length,
    /// The consumer dropped the receiver.
    Cancelled,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::Cancelled => "cancelled",
        }
    }
}

#[derive(Clone, Debug)]
pub enum GenEvent {
    /// One generated token. `text` is the newly released text, which can be
    /// empty (partial UTF-8 sequence or held back as a possible stop-string
    /// prefix). When text is still held at the end, one extra `Token` that
    /// repeats the last id carries the remainder. The EOS token is counted in
    /// `completion_tokens` but not emitted.
    Token { id: u32, text: String },
    Done {
        finish_reason: FinishReason,
        prompt_tokens: usize,
        completion_tokens: usize,
        /// Prompt tokens served from the reused KV prefix.
        cached_tokens: usize,
        /// The stop string that ended the generation (`finish_reason` is `Stop`), if one did.
        stop_sequence: Option<String>,
        /// Time to first token (prompt processing + first sample).
        prefill_ms: f64,
        /// Time spent producing tokens after the first.
        decode_ms: f64,
    },
    /// The request failed (bad request, backend error); no `Done` follows.
    Error { message: String },
}

#[derive(Clone, Debug, Default)]
pub struct ChatOpts {
    pub add_generation_prompt: bool,
    /// Qwen3-style `enable_thinking` template variable.
    pub enable_thinking: Option<bool>,
    /// OpenAI-style tools array, passed to the template as `tools`.
    pub tools: Option<Value>,
    /// Further template variables.
    pub extra: serde_json::Map<String, Value>,
}

/// Tokenizer + chat template. Implemented by `intarsia-models`.
pub trait TextCodec: Send + Sync {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>>;
    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String>;
    /// `messages` are OpenAI-style JSON objects (`role`, `content`, optional `tool_calls`, ...).
    fn apply_chat_template(&self, messages: &[Value], opts: &ChatOpts) -> Result<String>;
    fn has_chat_template(&self) -> bool;
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub family: String,
    pub backend: String,
    pub device: String,
    pub arch: String,
    pub n_layers: usize,
    pub hidden: usize,
    pub vocab: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    /// Planned context capacity in tokens (from measured free memory).
    pub context_len: usize,
    pub act_dtype: String,
    pub kv_dtype: String,
    pub weights_bytes: u64,
    pub eos_ids: Vec<u32>,
    pub has_chat_template: bool,
    pub reasoning_parser: Option<String>,
    pub tool_parser: Option<String>,
    pub memory_plan: crate::memplan::MemoryPlan,
}

pub trait Engine: Send + Sync {
    fn info(&self) -> ModelInfo;

    fn tokenize(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>>;
    fn detokenize(&self, ids: &[u32]) -> Result<String>;
    fn apply_chat_template(&self, messages: &[Value], opts: &ChatOpts) -> Result<String>;

    /// Queue a generation. Events arrive on the returned channel, ending with
    /// `Done` or `Error`. Dropping the receiver cancels the request (client disconnect).
    fn submit(&self, req: GenRequest) -> Result<Receiver<GenEvent>>;

    /// Blocking convenience over [`Engine::submit`].
    fn generate(&self, req: GenRequest, sink: &mut dyn FnMut(GenEvent)) -> Result<()> {
        let rx = self.submit(req)?;
        for ev in rx.iter() {
            let end = matches!(ev, GenEvent::Done { .. } | GenEvent::Error { .. });
            sink(ev);
            if end {
                break;
            }
        }
        Ok(())
    }

    /// Full-vocabulary logits for every position of `ids` (accuracy tooling).
    /// Memory is `len * vocab * 4` bytes; prefer [`Engine::forward_logits_each`] for long inputs.
    fn forward_logits(&self, ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(ids.len());
        self.forward_logits_each(ids, &mut |_, row| {
            out.push(row.to_vec());
            Ok(())
        })?;
        Ok(out)
    }

    /// Streaming variant: `(position, logits row)` in order; the row is only valid during the call.
    fn forward_logits_each(&self, ids: &[u32], cb: &mut dyn FnMut(usize, &[f32]) -> Result<()>) -> Result<()>;

    /// Teacher-forced decode (accuracy tooling): prefill `ids[..n - n_decode]`, then feed the
    /// remaining ids one at a time through the single-token decode path (`T = 1`: the decode
    /// attention and GEMV kernels that serving uses) and report the logits row of each. `cb` gets
    /// `(absolute position, logits row)` for the last `n_decode` positions only.
    fn forward_logits_decode(
        &self,
        _ids: &[u32],
        _n_decode: usize,
        _cb: &mut dyn FnMut(usize, &[f32]) -> Result<()>,
    ) -> Result<()> {
        Err(Error::Unsupported("this engine has no teacher-forced decode path".into()))
    }

    /// Scripted forward passes on a fresh sequence (accuracy tooling): each step runs `ids` through the model
    /// as one forward pass (the caller picks the chunk shapes) and yields the logits row of its last token when
    /// `logits` is set (a one-token step then takes the serving decode path). Replays what `generate` does to
    /// the sequence, e.g. prefill chunks, one-token decode steps, prefill chunks at other boundaries.
    fn forward_script(&self, _steps: &[ScriptStep]) -> Result<Vec<Option<Vec<f32>>>> {
        Err(Error::Unsupported("this engine has no scripted forward path".into()))
    }

    fn telemetry(&self) -> TelemetrySnapshot;

    /// `Some(reason)` once the engine can no longer serve requests (the worker
    /// panicked); `/health` then answers 503 and the supervisor restarts the process.
    fn failed(&self) -> Option<String> {
        None
    }

    /// The execution plan (served at `GET /v1/intarsia/plan`); `Null` when the engine has none.
    fn plan(&self) -> Value {
        Value::Null
    }
}

// ---------------------------------------------------------------------------
// incremental detokenisation + stop strings
// ---------------------------------------------------------------------------

/// Turns a growing token list into text deltas without emitting half UTF-8
/// characters (byte-level BPE can split one character across tokens).
pub struct IncrementalDecoder {
    ids: Vec<u32>,
    prefix: usize,
    read: usize,
}

impl Default for IncrementalDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl IncrementalDecoder {
    pub fn new() -> Self {
        IncrementalDecoder { ids: Vec::new(), prefix: 0, read: 0 }
    }

    pub fn push(&mut self, codec: &dyn TextCodec, id: u32) -> Result<String> {
        self.ids.push(id);
        let prefix_text = codec.decode(&self.ids[self.prefix..self.read], true)?;
        let new_text = codec.decode(&self.ids[self.prefix..], true)?;
        if new_text.len() > prefix_text.len() && !new_text.ends_with('\u{FFFD}') {
            if let Some(delta) = new_text.get(prefix_text.len()..) {
                let delta = delta.to_string();
                self.prefix = self.read;
                self.read = self.ids.len();
                return Ok(delta);
            }
        }
        Ok(String::new())
    }

    /// Remaining text (lossy) at the end of the stream.
    pub fn flush(&mut self, codec: &dyn TextCodec) -> Result<String> {
        let prefix_text = codec.decode(&self.ids[self.prefix..self.read], true)?;
        let new_text = codec.decode(&self.ids[self.prefix..], true)?;
        self.prefix = self.ids.len();
        self.read = self.ids.len();
        Ok(new_text.get(prefix_text.len()..).unwrap_or("").to_string())
    }
}

/// Holds back text that could still become a stop string.
pub struct StopMatcher {
    stops: Vec<String>,
    pending: String,
    matched: Option<String>,
}

impl StopMatcher {
    pub fn new(stops: &[String]) -> Self {
        StopMatcher { stops: stops.iter().filter(|s| !s.is_empty()).cloned().collect(), pending: String::new(), matched: None }
    }

    /// The stop string that triggered the last `push` (if it stopped).
    pub fn matched(&self) -> Option<&str> {
        self.matched.as_deref()
    }

    /// Returns (text safe to release, stopped).
    pub fn push(&mut self, text: &str) -> (String, bool) {
        if self.stops.is_empty() {
            return (text.to_string(), false);
        }
        self.pending.push_str(text);
        let hit = self.stops.iter().filter_map(|s| self.pending.find(s.as_str()).map(|i| (i, s))).min_by_key(|h| h.0);
        if let Some((i, s)) = hit {
            self.matched = Some(s.clone());
            let out = self.pending[..i].to_string();
            self.pending.clear();
            return (out, true);
        }
        let mut keep = 0usize;
        for s in &self.stops {
            for k in (1..s.len()).rev() {
                if k <= keep {
                    break;
                }
                if s.is_char_boundary(k) && self.pending.ends_with(&s[..k]) {
                    keep = k;
                    break;
                }
            }
        }
        let rel = self.pending.len() - keep;
        let out: String = self.pending.drain(..rel).collect();
        (out, false)
    }

    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

// ---------------------------------------------------------------------------
// LocalEngine
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct EngineMeta {
    pub id: String,
    pub eos_ids: Vec<u32>,
    pub reasoning_parser: Option<String>,
    pub tool_parser: Option<String>,
}

struct Inner {
    model: Model,
    state: SequenceState,
}

struct Shared {
    inner: Mutex<Inner>,
    telemetry: Arc<Telemetry>,
    codec: Arc<dyn TextCodec>,
    eos: Vec<u32>,
    /// Set when the worker panicked: the engine accepts no more work.
    failed: Mutex<Option<String>>,
}

impl Shared {
    fn failure(&self) -> Option<String> {
        self.failed.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

/// Debug aid: `INTARSIA_DEBUG_STALL_S=n` makes the worker sleep n seconds at the start of every
/// job (a wedged-GPU stand-in for testing the watchdog and supervisor);
/// `INTARSIA_DEBUG_PANIC=1` makes it panic instead.
fn debug_hooks() {
    if let Some(s) = std::env::var("INTARSIA_DEBUG_STALL_S").ok().and_then(|v| v.parse::<f64>().ok()) {
        std::thread::sleep(Duration::from_secs_f64(s.max(0.0)));
    }
    if std::env::var("INTARSIA_DEBUG_PANIC").map(|v| v == "1").unwrap_or(false) {
        panic!("INTARSIA_DEBUG_PANIC set");
    }
}

fn panic_message(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".into())
}

struct Job {
    req: GenRequest,
    tx: Sender<GenEvent>,
}

/// Single-sequence engine: a worker thread draining a FIFO queue.
pub struct LocalEngine {
    shared: Arc<Shared>,
    tx: Mutex<Option<Sender<Job>>>,
    info: ModelInfo,
    plan: Value,
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl LocalEngine {
    pub fn new(
        model: Model,
        codec: Arc<dyn TextCodec>,
        meta: EngineMeta,
        telemetry: Arc<Telemetry>,
    ) -> Result<LocalEngine> {
        let state = model.new_state()?;
        let dinfo = model.backend.info();
        let info = ModelInfo {
            id: meta.id.clone(),
            family: model.recipe.model.family.clone(),
            backend: dinfo.backend.clone(),
            device: if model.tp_world() > 1 {
                format!("{} x{} (tp={}, {} comm)", dinfo.name, model.tp_world(), model.tp_world(), model.comm_name().unwrap_or_default())
            } else {
                dinfo.name.clone()
            },
            arch: dinfo.arch.clone(),
            n_layers: model.dims.n_layers,
            hidden: model.dims.hidden,
            vocab: model.dims.vocab,
            n_heads: model.dims.n_heads,
            n_kv_heads: model.dims.n_kv_heads,
            head_dim: model.dims.head_dim,
            max_position_embeddings: model.dims.max_positions,
            context_len: state.cap,
            act_dtype: model.act.name().into(),
            kv_dtype: model.kv.name().into(),
            weights_bytes: model.weights_bytes,
            eos_ids: meta.eos_ids.clone(),
            has_chat_template: codec.has_chat_template(),
            reasoning_parser: meta.reasoning_parser.clone(),
            tool_parser: meta.tool_parser.clone(),
            memory_plan: model.plan.clone(),
        };
        let plan = crate::plan::plan_json(&model, &meta.id);
        telemetry.set_model(&meta.id, dinfo, model.weights_bytes);
        let backend = model.backend.clone();
        telemetry.set_device_probe(Box::new(move || backend.info()));
        telemetry.set_kv(0, state.total_bytes() as u64, 0, state.cap as u64);

        let shared = Arc::new(Shared {
            inner: Mutex::new(Inner { model, state }),
            telemetry,
            codec,
            eos: meta.eos_ids,
            failed: Mutex::new(None),
        });
        let (tx, rx) = channel::<Job>();
        let sh = shared.clone();
        let worker = std::thread::Builder::new()
            .name("intarsia-engine".into())
            .spawn(move || {
                for job in rx.iter() {
                    if let Some(why) = sh.failure() {
                        sh.telemetry.request_rejected();
                        let _ = job.tx.send(GenEvent::Error { message: format!("engine failed: {why}") });
                        continue;
                    }
                    let (tx, plen) = (job.tx.clone(), job.req.prompt.len());
                    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_job(&sh, job)));
                    if let Err(p) = r {
                        // A panic can leave the device or the KV state inconsistent: fail the
                        // engine for good (the supervisor restarts the process).
                        let why = format!("worker panicked: {}", panic_message(&*p));
                        *sh.failed.lock().unwrap_or_else(|e| e.into_inner()) = Some(why.clone());
                        sh.telemetry.request_finished("error", plen, 0, 0, Duration::ZERO, Duration::ZERO, Duration::ZERO);
                        let _ = tx.send(GenEvent::Error { message: why });
                    }
                }
            })
            .map_err(|e| Error::Other(format!("spawn engine worker: {e}")))?;
        Ok(LocalEngine { shared, tx: Mutex::new(Some(tx)), info, plan, worker: Mutex::new(Some(worker)) })
    }
}

impl Drop for LocalEngine {
    fn drop(&mut self) {
        // closing the channel ends the worker loop
        self.tx.lock().unwrap().take();
        if let Some(h) = self.worker.lock().unwrap().take() {
            let _ = h.join();
        }
    }
}

impl Engine for LocalEngine {
    fn info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn tokenize(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        self.shared.codec.encode(text, add_special_tokens)
    }

    fn detokenize(&self, ids: &[u32]) -> Result<String> {
        self.shared.codec.decode(ids, false)
    }

    fn apply_chat_template(&self, messages: &[Value], opts: &ChatOpts) -> Result<String> {
        self.shared.codec.apply_chat_template(messages, opts)
    }

    fn submit(&self, req: GenRequest) -> Result<Receiver<GenEvent>> {
        if let Some(why) = self.shared.failure() {
            return Err(Error::Other(format!("engine failed: {why}")));
        }
        let (etx, erx) = channel();
        self.shared.telemetry.request_enqueued();
        let guard = self.tx.lock().unwrap();
        let sent = match guard.as_ref() {
            Some(tx) => tx.send(Job { req, tx: etx }).map_err(|_| Error::Other("engine worker has stopped".into())),
            None => Err(Error::Other("engine is shut down".into())),
        };
        if let Err(e) = sent {
            self.shared.telemetry.request_rejected();
            return Err(e);
        }
        Ok(erx)
    }

    fn failed(&self) -> Option<String> {
        self.shared.failure()
    }

    fn forward_logits_each(&self, ids: &[u32], cb: &mut dyn FnMut(usize, &[f32]) -> Result<()>) -> Result<()> {
        if let Some(why) = self.shared.failure() {
            return Err(Error::Other(format!("engine failed: {why}")));
        }
        let mut g = self.shared.inner.lock().unwrap();
        let Inner { model, state } = &mut *g;
        state.reset(&*model.backend)?;
        model.forward(state, ids, HeadMode::AllLogits, Some(cb))?;
        self.shared.telemetry.set_kv(
            state.used_bytes() as u64,
            state.total_bytes() as u64,
            state.len as u64,
            state.cap as u64,
        );
        Ok(())
    }

    fn forward_logits_decode(
        &self,
        ids: &[u32],
        n_decode: usize,
        cb: &mut dyn FnMut(usize, &[f32]) -> Result<()>,
    ) -> Result<()> {
        if n_decode == 0 || n_decode >= ids.len() {
            return Err(Error::invalid(format!("n_decode {n_decode} must be in 1..{}", ids.len())));
        }
        let mut g = self.shared.inner.lock().unwrap();
        let Inner { model, state } = &mut *g;
        state.reset(&*model.backend)?;
        let split = ids.len() - n_decode;
        model.forward(state, &ids[..split], HeadMode::None, None)?;
        for (j, id) in ids[split..].iter().enumerate() {
            match model.forward(state, std::slice::from_ref(id), HeadMode::LastLogits, None)? {
                StepOut::Logits(row) => cb(split + j, &row)?,
                _ => return Err(Error::Other("backend returned no logits".into())),
            }
        }
        Ok(())
    }

    fn forward_script(&self, steps: &[ScriptStep]) -> Result<Vec<Option<Vec<f32>>>> {
        if let Some(why) = self.shared.failure() {
            return Err(Error::Other(format!("engine failed: {why}")));
        }
        let mut g = self.shared.inner.lock().unwrap();
        let Inner { model, state } = &mut *g;
        state.reset(&*model.backend)?;
        let mut out = Vec::with_capacity(steps.len());
        for s in steps {
            if let Some(n) = s.truncate {
                state.truncate(n, &*model.backend)?;
            }
            let mode = if s.logits { HeadMode::LastLogits } else { HeadMode::None };
            out.push(match model.forward(state, &s.ids, mode, None)? {
                StepOut::Logits(row) => Some(row),
                _ if s.logits => return Err(Error::Other("backend returned no logits".into())),
                _ => None,
            });
        }
        Ok(out)
    }

    fn plan(&self) -> Value {
        self.plan.clone()
    }

    fn telemetry(&self) -> TelemetrySnapshot {
        self.shared.telemetry.snapshot()
    }
}

struct Outcome {
    reason: FinishReason,
    stop_sequence: Option<String>,
    prompt_tokens: usize,
    completion_tokens: usize,
    cached_tokens: usize,
    prefill: Duration,
    decode: Duration,
}

fn run_job(sh: &Shared, job: Job) {
    sh.telemetry.request_started();
    let t_total = Instant::now();
    debug_hooks();
    let mut g = sh.inner.lock().unwrap();
    let res = generate_one(sh, &mut g, &job.req, &job.tx);
    let kv = (g.state.used_bytes() as u64, g.state.total_bytes() as u64, g.state.len as u64, g.state.cap as u64);
    drop(g);
    sh.telemetry.set_kv(kv.0, kv.1, kv.2, kv.3);
    match res {
        Ok(o) => {
            sh.telemetry.request_finished(
                o.reason.as_str(),
                o.prompt_tokens,
                o.completion_tokens,
                o.cached_tokens,
                o.prefill,
                o.decode,
                o.prefill,
            );
            let _ = job.tx.send(GenEvent::Done {
                finish_reason: o.reason,
                prompt_tokens: o.prompt_tokens,
                completion_tokens: o.completion_tokens,
                cached_tokens: o.cached_tokens,
                stop_sequence: o.stop_sequence,
                prefill_ms: o.prefill.as_secs_f64() * 1e3,
                decode_ms: o.decode.as_secs_f64() * 1e3,
            });
        }
        Err(e) => {
            sh.telemetry.request_finished(
                "error",
                job.req.prompt.len(),
                0,
                0,
                t_total.elapsed(),
                Duration::ZERO,
                t_total.elapsed(),
            );
            let _ = job.tx.send(GenEvent::Error { message: e.to_string() });
        }
    }
}

fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Rows of the next prefill chunk when `remaining` prompt tokens are left to compute and one forward takes at most
/// `max` rows. A prompt is split into the fewest chunks that fit (`ceil(remaining / max)`) of sizes as equal as
/// possible, instead of full `max` chunks plus a short tail: a tail of a few rows takes the slow small-M GEMM path on
/// Metal and re-streams all weights for little work. Calling it again on what is left gives the next chunk, so the
/// sizes run `ceil` first, then `floor` (they differ by at most one). A `remaining` of at most `max` is one chunk.
/// `remaining` is what follows the reused prefix; the chunk starts wherever the reuse ended, so no alignment is
/// needed (the KV cache is contiguous).
pub fn balanced_chunk_len(remaining: usize, max: usize) -> usize {
    let max = max.max(1);
    if remaining <= max {
        return remaining;
    }
    remaining.div_ceil(remaining.div_ceil(max))
}

fn generate_one(sh: &Shared, g: &mut Inner, req: &GenRequest, tx: &Sender<GenEvent>) -> Result<Outcome> {
    let Inner { model, state: st } = g;
    let plen = req.prompt.len();
    if plen == 0 {
        return Err(Error::BadRequest("empty prompt".into()));
    }
    if plen >= st.cap {
        return Err(Error::BadRequest(format!(
            "prompt of {plen} tokens does not fit the context window of {} tokens",
            st.cap
        )));
    }
    let max_new = req.max_tokens.min(st.cap - plen);

    // KV prefix reuse (contiguous cache of the previous request; always keep >= 1 token to compute)
    let mut p = common_prefix(&st.tokens, &req.prompt).min(plen - 1);
    if st.has_recurrent() {
        // Recurrent state (GatedDeltaNet / Mamba) cannot be shortened: reuse only what the state can return
        // to exactly: the whole held sequence, a snapshot taken during an earlier prefill, or nothing.
        p = st.restorable_len(p);
    }
    st.truncate(p, &*model.backend)?;
    let cached = p;

    let params = SamplingParams { temperature: req.temperature, top_k: req.top_k, top_p: req.top_p };
    let greedy = req.temperature <= 0.0;
    let mut sampler = Sampler::new(params, req.seed);
    let mut stop_ids: Vec<u32> = req.stop_ids.clone();
    if !req.ignore_eos {
        stop_ids.extend_from_slice(&sh.eos);
    }

    // choose how to get the next token: argmax on device when nothing needs the logits
    let pick = |model: &Model, st: &mut SequenceState, ids: &[u32], k: usize, sampler: &mut Sampler| -> Result<u32> {
        // The logits only travel to the host when something uses them: sampling, or the min_tokens
        // mask on stop ids. With `ignore_eos` (no stop ids) a forced-length greedy run keeps the
        // on-device argmax, so forced-length speed benchmarks measure the real decode path.
        let masks_stop_ids = k <= req.min_tokens && !stop_ids.is_empty();
        let need_logits = !greedy || masks_stop_ids || req.trace_logits.is_some();
        if !need_logits {
            return match model.forward(st, ids, HeadMode::LastArgmax, None)? {
                StepOut::Token(t) => Ok(t),
                _ => Err(Error::Other("backend returned no token".into())),
            };
        }
        match model.forward(st, ids, HeadMode::LastLogits, None)? {
            StepOut::Logits(mut l) => {
                if let Some(t) = &req.trace_logits {
                    t.lock().unwrap().push(l.clone());
                }
                if k <= req.min_tokens {
                    for &s in &stop_ids {
                        if let Some(x) = l.get_mut(s as usize) {
                            *x = f32::NEG_INFINITY;
                        }
                    }
                }
                Ok(sampler.sample(&l) as u32)
            }
            _ => Err(Error::Other("backend returned no logits".into())),
        }
    };

    let cancelled = || req.cancel.as_ref().is_some_and(|c| c.load(Ordering::Relaxed));
    let t0 = Instant::now();
    let cancel_outcome = |p: usize| Outcome {
        reason: FinishReason::Cancelled,
        stop_sequence: None,
        prompt_tokens: plen,
        completion_tokens: 0,
        cached_tokens: p,
        prefill: t0.elapsed(),
        decode: Duration::ZERO,
    };
    if max_new == 0 {
        return Ok(Outcome {
            reason: FinishReason::Length,
            stop_sequence: None,
            prompt_tokens: plen,
            completion_tokens: 0,
            cached_tokens: cached,
            prefill: Duration::ZERO,
            decode: Duration::ZERO,
        });
    }
    // Prefill all but the last chunk here so a cancel is noticed between chunks.
    let max_chunk = model.prefill_chunk().max(1);
    // Attention-only states balance the chunk sizes (see `balanced_chunk_len`). A recurrent state keeps the fixed
    // grid of full chunks: its snapshots sit on chunk boundaries, and a run restored from one must see the same
    // chunk shapes as a from-scratch run, which only a boundary that does not depend on the prompt length gives.
    let balance = !st.has_recurrent();
    let mut pos = p;
    while plen - pos > max_chunk {
        if cancelled() {
            return Ok(cancel_outcome(cached));
        }
        let chunk = if balance { balanced_chunk_len(plen - pos, max_chunk) } else { max_chunk };
        match model.forward(st, &req.prompt[pos..pos + chunk], HeadMode::None, None)? {
            StepOut::None => {}
            _ => return Err(Error::Other("backend returned output for a prefill chunk".into())),
        }
        pos += chunk;
        st.snapshot(&*model.backend, crate::state::RECURRENT_SNAPSHOTS)?;
    }
    if cancelled() {
        return Ok(cancel_outcome(cached));
    }
    if st.has_recurrent() && plen - pos > 1 {
        // snapshot one token before the end of the prompt: a repeated or regenerated request reuses everything
        // but its last token, as with an attention-only cache
        model.forward(st, &req.prompt[pos..plen - 1], HeadMode::None, None)?;
        pos = plen - 1;
        st.snapshot(&*model.backend, crate::state::RECURRENT_SNAPSHOTS)?;
    }
    let mut tok = pick(model, st, &req.prompt[pos..], 1, &mut sampler)?;
    let prefill = t0.elapsed();
    let t_dec = Instant::now();

    let codec = &*sh.codec;
    let mut dec = IncrementalDecoder::new();
    // Like vLLM, stop strings are not checked until `min_tokens` tokens exist.
    let mut stops = StopMatcher::new(if req.min_tokens > 0 { &[] } else { &req.stop });
    let mut generated = 0usize;
    let mut reason;
    let mut string_stop = false;
    let mut last_id = tok;
    loop {
        generated += 1;
        if stop_ids.contains(&tok) {
            reason = FinishReason::Stop;
            break;
        }
        last_id = tok;
        let delta = dec.push(codec, tok)?;
        if req.min_tokens > 0 && generated == req.min_tokens + 1 {
            stops = StopMatcher::new(&req.stop);
        }
        let (text, stopped) = stops.push(&delta);
        if tx.send(GenEvent::Token { id: tok, text }).is_err() {
            reason = FinishReason::Cancelled;
            break;
        }
        if stopped {
            string_stop = true;
            reason = FinishReason::Stop;
            break;
        }
        if generated >= max_new {
            reason = FinishReason::Length;
            break;
        }
        if cancelled() {
            reason = FinishReason::Cancelled;
            break;
        }
        tok = pick(model, st, &[tok], generated + 1, &mut sampler)?;
    }
    if reason != FinishReason::Cancelled && !string_stop {
        // release anything still held back: partial UTF-8 and possible stop-string prefixes
        let mut tail = String::new();
        if let Ok(t) = dec.flush(codec) {
            let (rel, hit) = stops.push(&t);
            tail.push_str(&rel);
            if hit {
                string_stop = true;
                reason = FinishReason::Stop;
            }
        }
        if !string_stop {
            tail.push_str(&stops.flush());
        }
        if !tail.is_empty() {
            let _ = tx.send(GenEvent::Token { id: last_id, text: tail });
        }
    }
    Ok(Outcome {
        reason,
        stop_sequence: if string_stop { stops.matched().map(String::from) } else { None },
        prompt_tokens: plen,
        completion_tokens: generated,
        cached_tokens: cached,
        prefill,
        decode: t_dec.elapsed(),
    })
}

#[cfg(test)]
mod tests {
    use super::balanced_chunk_len;

    /// The whole plan: chunk sizes until nothing is left, the way `generate_one` walks it.
    fn plan(remaining: usize, max: usize) -> Vec<usize> {
        let mut v = Vec::new();
        let mut left = remaining;
        while left > 0 {
            let c = balanced_chunk_len(left, max);
            assert!(c >= 1 && c <= max.max(1) && c <= left);
            v.push(c);
            left -= c;
        }
        v
    }

    #[test]
    fn at_most_one_chunk_is_unchanged() {
        assert_eq!(plan(1, 2048), [1]);
        assert_eq!(plan(100, 2048), [100]);
        assert_eq!(plan(2048, 2048), [2048]);
        assert!(plan(0, 2048).is_empty());
    }

    #[test]
    fn exact_multiples_stay_full() {
        assert_eq!(plan(4096, 2048), [2048, 2048]);
        assert_eq!(plan(3 * 256, 256), [256, 256, 256]);
    }

    #[test]
    fn one_over_a_multiple_is_split_evenly() {
        // used to be 2048 + 2048 + 1
        assert_eq!(plan(4097, 2048), [1366, 1366, 1365]);
        assert_eq!(plan(2049, 2048), [1025, 1024]);
        assert_eq!(plan(257, 256), [129, 128]);
        assert_eq!(plan(4096 + 11, 2048), [1369, 1369, 1369]);
    }

    #[test]
    fn tiny_tails_never_appear() {
        for max in [1usize, 2, 7, 64, 256, 1024, 2048] {
            for rem in 1..=(5 * max).min(10_000) {
                let p = plan(rem, max);
                assert_eq!(p.iter().sum::<usize>(), rem);
                assert_eq!(p.len(), rem.div_ceil(max), "fewest chunks: rem {rem} max {max}");
                let (lo, hi) = (*p.iter().min().unwrap(), *p.iter().max().unwrap());
                assert!(hi - lo <= 1, "equal sizes: {p:?}");
                assert!(p.windows(2).all(|w| w[0] >= w[1]), "non-increasing: {p:?}");
                if p.len() > 1 {
                    // every chunk is at least half of max (rem > max): no sliver after the full chunks
                    assert!(lo * 2 >= max, "rem {rem} max {max}: {p:?}");
                }
            }
        }
    }

    #[test]
    fn after_prefix_reuse_only_the_rest_is_planned() {
        // 4100-token prompt: 3 reused tokens leave 4097 rows, 1000 reused leave 3100; the plan covers exactly the rest,
        // from wherever the reuse ended
        let (plen, max) = (4100usize, 2048usize);
        for reused in [0usize, 3, 1000, 2048, 2049, 4099] {
            let p = plan(plen - reused, max);
            assert_eq!(p.iter().sum::<usize>() + reused, plen);
            assert!(p.len() == (plen - reused).div_ceil(max));
        }
        assert_eq!(plan(4100 - 1000, 2048), [1550, 1550]);
        assert_eq!(plan(4100 - 4099, 2048), [1]);
    }
}
