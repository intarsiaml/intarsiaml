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
    /// Speculative decoding (`--spec`; default off). On needs `Model::enable_verify` before the plan is configured
    /// (`intarsia_models::load_engine` does both, and only on the Metal backend).
    pub spec: crate::spec::SpecConfig,
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
    spec: crate::spec::SpecConfig,
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
        if meta.spec.enabled() {
            if model.verify_rows() < meta.spec.verify_rows() {
                return Err(Error::invalid(format!(
                    "speculative decoding with k = {} needs verify passes of {} rows; the model was prepared for {} \
                     (Model::enable_verify before configure_plan)",
                    meta.spec.k,
                    meta.spec.verify_rows(),
                    model.verify_rows()
                )));
            }
            if state.has_recurrent() {
                return Err(Error::Unsupported("speculative decoding on a model with recurrent state".into()));
            }
        }
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
            spec: meta.spec,
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

pub(crate) struct Outcome {
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
    let Job { req, tx } = job;
    let plen = req.prompt.len();
    let mut g = sh.inner.lock().unwrap();
    let res = generate_one(sh, &mut g, req, tx.clone());
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
            let _ = tx.send(GenEvent::Done {
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
                plen,
                0,
                0,
                t_total.elapsed(),
                Duration::ZERO,
                t_total.elapsed(),
            );
            let _ = tx.send(GenEvent::Error { message: e.to_string() });
        }
    }
}

fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// What one speculative verify pass yields.
struct SpecStep {
    /// Tokens to emit, in order (`d_1..d_j` and the token greedy decode picks after them, or only `d_1..d_j` when
    /// `plain_next`).
    emit: Vec<u32>,
    /// Drafts accepted.
    accepted: usize,
    /// The token after `emit` must come from a plain (masked) step.
    plain_next: bool,
}

/// Most sequences `--max-batch` may ask for: `LocalEngine` decodes one sequence at a time. The batching scheduler
/// (docs/research/continuous-batching-design.md, M1) raises it; the model side (`Model::enable_batch`) is ready for
/// up to `Model::BATCH_MAX_ROWS`.
pub const SERVE_MAX_BATCH: usize = 1;

/// One verify pass over `[last, drafts...]` at `st.len` (token `generated + 1` is row 0's), greedy acceptance, KV
/// rollback of the rejected rows. Inside the `min_tokens` window the sequential path masks the stop ids before its
/// (host) argmax; a row there whose device argmax is not a stop id has the same argmax under the mask (both take the
/// lowest index of the maximum), one whose argmax is a stop id ends the pass before that row and leaves its token to
/// a plain step.
fn spec_verify(
    model: &Model,
    st: &mut SequenceState,
    last: u32,
    drafts: &[u32],
    generated: usize,
    min_tokens: usize,
    stop_ids: &[u32],
) -> Result<SpecStep> {
    let len0 = st.len;
    let mut ids = Vec::with_capacity(1 + drafts.len());
    ids.push(last);
    ids.extend_from_slice(drafts);
    let out = model.forward_verify(st, &ids)?;
    let j = crate::spec::accept(drafts, &out);
    let masked_stop =
        (0..=j).find(|&i| generated + 1 + i <= min_tokens && !stop_ids.is_empty() && stop_ids.contains(&out[i]));
    let step = match masked_stop {
        // rows 0..i emitted d_1..d_i; row i (fed d_i) is recomputed by the plain step, so its KV row goes too
        Some(i) => SpecStep { emit: out[..i].to_vec(), accepted: i, plain_next: true },
        None => SpecStep { emit: out[..=j].to_vec(), accepted: j, plain_next: false },
    };
    let keep = len0 + if step.plain_next { step.accepted } else { step.accepted + 1 };
    st.truncate(keep, &*model.backend)?;
    Ok(step)
}

/// Result of one [`Seq::prefill_step`].
pub(crate) enum PrefillStep {
    /// A prompt chunk went through the model; call again.
    Chunk,
    /// The prompt is in and the first token is sampled (`Seq::tok`): start the decode loop.
    Started,
    /// The request ended without generating (cancelled during prefill, or no room for a token).
    Finished(Outcome),
}

/// What [`Seq::plan_step`] wants next for a decoding sequence.
pub(crate) enum Next {
    /// A token already known (a verified speculative run): no forward pass.
    Emit(u32),
    /// A verify pass over `[tok, drafts...]` ([`Seq::verify`]).
    Verify(Vec<u32>),
    /// One plain decode step of `[tok]` ([`Seq::head_mode`], [`Seq::token_from`]).
    Plain,
}

/// One request's generation state machine: sampling, `min_tokens` masking, stop ids / strings, streaming, `max_tokens`,
/// cancellation and speculative drafting, independent of how the forward passes are scheduled.
/// `generate_one` drives one `Seq` to completion; a batching scheduler can interleave several, each against its own
/// [`SequenceState`].
///
/// Lifecycle: [`Seq::new`] (validates, reuses the cached KV prefix) -> [`Seq::prefill_step`] until `Started` ->
/// loop { [`Seq::on_token`] (true = done) ; [`Seq::advance`] } -> [`Seq::finish`].
pub(crate) struct Seq {
    req: GenRequest,
    tx: Sender<GenEvent>,
    codec: Arc<dyn TextCodec>,
    telemetry: Arc<Telemetry>,
    plen: usize,
    cached: usize,
    max_new: usize,
    greedy: bool,
    sampler: Sampler,
    /// `req.stop_ids` plus the model's EOS ids (unless `ignore_eos`).
    stop_ids: Vec<u32>,
    t0: Instant,
    /// Prompt tokens in the KV cache.
    pos: usize,
    prefill: Duration,
    t_dec: Instant,
    /// The token just sampled, not yet through [`Seq::on_token`].
    tok: u32,
    /// The last token that was emitted (carries the final held-back text).
    last_id: u32,
    generated: usize,
    reason: Option<FinishReason>,
    string_stop: bool,
    dec: IncrementalDecoder,
    stops: StopMatcher,
    // Speculative decoding (greedy only: sampled requests, logit tracing and recurrent state take plain steps). A
    // verify pass yields several tokens; they wait in `pending` and go through the same per-token checks as sequential
    // tokens, so EOS / stop ids, stop strings, min_tokens and max_tokens act on them exactly the same way.
    spec_on: bool,
    spec_k: usize,
    pending: std::collections::VecDeque<u32>,
    /// The next token must come from a plain step (a verify row inside the min_tokens mask chose a stop id).
    plain_next: bool,
    /// Prompt + tokens so far: the drafter's lookup context.
    ctx: Vec<u32>,
    /// Stops drafting where it does not pay (open-ended text), see `spec::GateParams`.
    gate: crate::spec::SpecGate,
}

impl Seq {
    /// Validate the request and reuse the KV prefix `st` holds (truncating it to what is reused).
    fn new(sh: &Shared, model: &Model, st: &mut SequenceState, req: GenRequest, tx: Sender<GenEvent>) -> Result<Seq> {
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

        let params = SamplingParams { temperature: req.temperature, top_k: req.top_k, top_p: req.top_p };
        let greedy = req.temperature <= 0.0;
        let sampler = Sampler::new(params, req.seed);
        let mut stop_ids: Vec<u32> = req.stop_ids.clone();
        if !req.ignore_eos {
            stop_ids.extend_from_slice(&sh.eos);
        }
        // Like vLLM, stop strings are not checked until `min_tokens` tokens exist.
        let stops = StopMatcher::new(if req.min_tokens > 0 { &[] } else { &req.stop });
        let spec_on =
            sh.spec.enabled() && greedy && req.trace_logits.is_none() && !st.has_recurrent() && model.verify_rows() >= 2;
        let spec_k = sh.spec.k.min(model.verify_rows().saturating_sub(1));
        let ctx: Vec<u32> = if spec_on { req.prompt.clone() } else { Vec::new() };
        let now = Instant::now();
        Ok(Seq {
            plen,
            cached: p,
            max_new,
            greedy,
            sampler,
            stop_ids,
            t0: now,
            pos: p,
            prefill: Duration::ZERO,
            t_dec: now,
            tok: 0,
            last_id: 0,
            generated: 0,
            reason: None,
            string_stop: false,
            dec: IncrementalDecoder::new(),
            stops,
            spec_on,
            spec_k,
            pending: std::collections::VecDeque::new(),
            plain_next: false,
            ctx,
            gate: crate::spec::SpecGate::new(crate::spec::GateParams::from_env()),
            codec: sh.codec.clone(),
            telemetry: sh.telemetry.clone(),
            req,
            tx,
        })
    }

    fn cancelled(&self) -> bool {
        self.req.cancel.as_ref().is_some_and(|c| c.load(Ordering::Relaxed))
    }

    fn cancel_outcome(&self) -> Outcome {
        Outcome {
            reason: FinishReason::Cancelled,
            stop_sequence: None,
            prompt_tokens: self.plen,
            completion_tokens: 0,
            cached_tokens: self.cached,
            prefill: self.t0.elapsed(),
            decode: Duration::ZERO,
        }
    }

    /// How the head of a step that samples the `k`-th generated token (1-based) must run: the logits only travel to
    /// the host when something uses them (sampling, the `min_tokens` mask on stop ids, logit tracing); otherwise the
    /// argmax stays on device. With `ignore_eos` (no stop ids) a forced-length greedy run keeps the on-device argmax,
    /// so forced-length speed benchmarks measure the real decode path.
    pub(crate) fn head_mode(&self, k: usize) -> HeadMode {
        let masks_stop_ids = k <= self.req.min_tokens && !self.stop_ids.is_empty();
        if !self.greedy || masks_stop_ids || self.req.trace_logits.is_some() {
            HeadMode::LastLogits
        } else {
            HeadMode::LastArgmax
        }
    }

    /// The `k`-th generated token from the output of a step run with [`Seq::head_mode`]`(k)`.
    pub(crate) fn token_from(&mut self, out: StepOut, k: usize) -> Result<u32> {
        let logits = self.head_mode(k) == HeadMode::LastLogits;
        match out {
            StepOut::Token(t) if !logits => Ok(t),
            StepOut::Logits(mut l) if logits => {
                if let Some(t) = &self.req.trace_logits {
                    t.lock().unwrap().push(l.clone());
                }
                if k <= self.req.min_tokens {
                    for &s in &self.stop_ids {
                        if let Some(x) = l.get_mut(s as usize) {
                            *x = f32::NEG_INFINITY;
                        }
                    }
                }
                Ok(self.sampler.sample(&l) as u32)
            }
            _ => Err(Error::Other(if logits { "backend returned no logits" } else { "backend returned no token" }.into())),
        }
    }

    /// Run the next prompt chunk (all but the last chunk first, so a cancel is noticed between chunks), then the final
    /// one with the first token sampled from it.
    pub(crate) fn prefill_step(&mut self, model: &Model, st: &mut SequenceState) -> Result<PrefillStep> {
        if self.max_new == 0 {
            return Ok(PrefillStep::Finished(Outcome {
                reason: FinishReason::Length,
                stop_sequence: None,
                prompt_tokens: self.plen,
                completion_tokens: 0,
                cached_tokens: self.cached,
                prefill: Duration::ZERO,
                decode: Duration::ZERO,
            }));
        }
        let chunk = model.prefill_chunk().max(1);
        if self.cancelled() {
            return Ok(PrefillStep::Finished(self.cancel_outcome()));
        }
        if self.plen - self.pos > chunk {
            match model.forward(st, &self.req.prompt[self.pos..self.pos + chunk], HeadMode::None, None)? {
                StepOut::None => {}
                _ => return Err(Error::Other("backend returned output for a prefill chunk".into())),
            }
            self.pos += chunk;
            st.snapshot(&*model.backend, crate::state::RECURRENT_SNAPSHOTS)?;
            return Ok(PrefillStep::Chunk);
        }
        if st.has_recurrent() && self.plen - self.pos > 1 {
            // snapshot one token before the end of the prompt: a repeated or regenerated request reuses everything
            // but its last token, as with an attention-only cache
            model.forward(st, &self.req.prompt[self.pos..self.plen - 1], HeadMode::None, None)?;
            self.pos = self.plen - 1;
            st.snapshot(&*model.backend, crate::state::RECURRENT_SNAPSHOTS)?;
        }
        let mode = self.head_mode(1);
        let out = model.forward(st, &self.req.prompt[self.pos..], mode, None)?;
        self.tok = self.token_from(out, 1)?;
        self.prefill = self.t0.elapsed();
        self.t_dec = Instant::now();
        Ok(PrefillStep::Started)
    }

    /// Account for `self.tok`: EOS / stop ids, incremental detokenisation, stop strings, streaming, `max_tokens`,
    /// cancellation. Returns true when the sequence is finished (call [`Seq::finish`]).
    pub(crate) fn on_token(&mut self) -> Result<bool> {
        let tok = self.tok;
        self.generated += 1;
        if self.spec_on {
            self.ctx.push(tok);
        }
        if self.stop_ids.contains(&tok) {
            self.reason = Some(FinishReason::Stop);
            return Ok(true);
        }
        self.last_id = tok;
        let delta = self.dec.push(&*self.codec, tok)?;
        if self.req.min_tokens > 0 && self.generated == self.req.min_tokens + 1 {
            self.stops = StopMatcher::new(&self.req.stop);
        }
        let (text, stopped) = self.stops.push(&delta);
        if self.tx.send(GenEvent::Token { id: tok, text }).is_err() {
            self.reason = Some(FinishReason::Cancelled);
            return Ok(true);
        }
        if stopped {
            self.string_stop = true;
            self.reason = Some(FinishReason::Stop);
            return Ok(true);
        }
        if self.generated >= self.max_new {
            self.reason = Some(FinishReason::Length);
            return Ok(true);
        }
        if self.cancelled() {
            self.reason = Some(FinishReason::Cancelled);
            return Ok(true);
        }
        Ok(false)
    }

    /// What the sequence needs to produce its next token.
    pub(crate) fn plan_step(&mut self, st: &SequenceState) -> Next {
        if let Some(t) = self.pending.pop_front() {
            return Next::Emit(t);
        }
        if self.spec_on && !std::mem::take(&mut self.plain_next) && self.gate.allow() {
            // verify feeds 1 + d tokens: keep them inside the context window
            let room = st.cap.saturating_sub(st.len + 1);
            let drafts = crate::spec::propose_min(&self.ctx, self.spec_k.min(room), self.gate.min_n());
            if !drafts.is_empty() {
                return Next::Verify(drafts);
            }
        }
        Next::Plain
    }

    /// Run the verify pass for `drafts` ([`Next::Verify`]); the verified tokens queue up for [`Seq::plan_step`].
    pub(crate) fn verify(&mut self, model: &Model, st: &mut SequenceState, drafts: &[u32]) -> Result<()> {
        let pos = st.len;
        let step = spec_verify(model, st, self.tok, drafts, self.generated, self.req.min_tokens, &self.stop_ids)?;
        self.telemetry.spec_step(drafts.len(), step.accepted);
        self.gate.record(step.accepted);
        if crate::spec::trace_enabled() {
            eprintln!(
                "spec: token {} pos {} drafted {} accepted {} emits {}{}",
                self.generated + 1,
                pos,
                drafts.len(),
                step.accepted,
                step.emit.len(),
                if step.plain_next { " (min_tokens cut: next token by a plain step)" } else { "" }
            );
        }
        self.pending.extend(step.emit);
        self.plain_next = step.plain_next;
        Ok(())
    }

    /// The input of a plain decode step: the token to feed and how the head of that step must run
    /// ([`Seq::head_mode`] for the token the step samples). Marks the sequence as taking a plain step now, so a pending
    /// "plain next" (a min_tokens cut that emitted nothing) is served by this one. Pair with [`Seq::accept_plain`]; a
    /// solo step ([`Seq::advance`]) and a batched step run exactly this pair around their forward pass.
    pub(crate) fn plain_input(&mut self) -> (u32, HeadMode) {
        self.plain_next = false;
        (self.tok, self.head_mode(self.generated + 1))
    }

    /// Take the output of the plain step started with [`Seq::plain_input`] as the next `self.tok`.
    pub(crate) fn accept_plain(&mut self, out: StepOut) -> Result<()> {
        self.tok = self.token_from(out, self.generated + 1)?;
        Ok(())
    }

    /// Everything of [`Seq::advance`] up to the forward pass of a plain step: a token that is already known (a verified
    /// speculative run) becomes `self.tok`, a verify pass runs; `Some((token, mode))` when a plain step is still needed
    /// ([`Seq::plain_input`]), which the caller runs alone or inside a batch and hands to [`Seq::accept_plain`].
    pub(crate) fn prepare(&mut self, model: &Model, st: &mut SequenceState) -> Result<Option<(u32, HeadMode)>> {
        match self.plan_step(st) {
            Next::Emit(t) => {
                self.tok = t;
                return Ok(None);
            }
            Next::Verify(drafts) => {
                self.verify(model, st, &drafts)?;
                if let Some(t) = self.pending.pop_front() {
                    self.tok = t;
                    return Ok(None);
                }
            }
            Next::Plain => {}
        }
        Ok(Some(self.plain_input()))
    }

    /// Produce the next `self.tok` on this sequence alone (plan, run the forward pass, sample).
    pub(crate) fn advance(&mut self, model: &Model, st: &mut SequenceState) -> Result<()> {
        if let Some((tok, mode)) = self.prepare(model, st)? {
            let out = model.forward(st, &[tok], mode, None)?;
            self.accept_plain(out)?;
        }
        Ok(())
    }

    /// Roll the KV back, release held-back text and report. Call once [`Seq::on_token`] returned true.
    pub(crate) fn finish(mut self, model: &Model, st: &mut SequenceState) -> Result<Outcome> {
        let mut reason = self.reason.expect("finish before the sequence ended");
        if self.spec_on {
            // A stop inside an accepted run leaves KV rows of tokens that were never emitted: roll back to what sequential
            // decode holds at this point (every emitted token but the last fed), so a following request reuses the same
            // prefix and prefills the same chunks.
            let want = self.plen + self.generated - 1;
            if crate::spec::trace_enabled() && (!self.pending.is_empty() || st.len > want) {
                eprintln!(
                    "spec: finished at token {} with {} verified tokens unemitted; KV {} -> {want}",
                    self.generated,
                    self.pending.len(),
                    st.len
                );
            }
            if st.len > want {
                st.truncate(want, &*model.backend)?;
            }
        }
        if reason != FinishReason::Cancelled && !self.string_stop {
            // release anything still held back: partial UTF-8 and possible stop-string prefixes
            let mut tail = String::new();
            if let Ok(t) = self.dec.flush(&*self.codec) {
                let (rel, hit) = self.stops.push(&t);
                tail.push_str(&rel);
                if hit {
                    self.string_stop = true;
                    reason = FinishReason::Stop;
                }
            }
            if !self.string_stop {
                tail.push_str(&self.stops.flush());
            }
            if !tail.is_empty() {
                let _ = self.tx.send(GenEvent::Token { id: self.last_id, text: tail });
            }
        }
        Ok(Outcome {
            reason,
            stop_sequence: if self.string_stop { self.stops.matched().map(String::from) } else { None },
            prompt_tokens: self.plen,
            completion_tokens: self.generated,
            cached_tokens: self.cached,
            prefill: self.prefill,
            decode: self.t_dec.elapsed(),
        })
    }
}

/// Single-sequence driver: prefill, then one token at a time until the sequence ends.
fn generate_one(sh: &Shared, g: &mut Inner, req: GenRequest, tx: Sender<GenEvent>) -> Result<Outcome> {
    let Inner { model, state: st } = g;
    let mut seq = Seq::new(sh, model, st, req, tx)?;
    loop {
        match seq.prefill_step(model, st)? {
            PrefillStep::Chunk => {}
            PrefillStep::Started => break,
            PrefillStep::Finished(o) => return Ok(o),
        }
    }
    while !seq.on_token()? {
        seq.advance(model, st)?;
    }
    seq.finish(model, st)
}
