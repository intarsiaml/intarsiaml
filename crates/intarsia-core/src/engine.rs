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
        }
    }
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

    fn telemetry(&self) -> TelemetrySnapshot;
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
}

impl StopMatcher {
    pub fn new(stops: &[String]) -> Self {
        StopMatcher { stops: stops.iter().filter(|s| !s.is_empty()).cloned().collect(), pending: String::new() }
    }

    /// Returns (text safe to release, stopped).
    pub fn push(&mut self, text: &str) -> (String, bool) {
        if self.stops.is_empty() {
            return (text.to_string(), false);
        }
        self.pending.push_str(text);
        let hit = self.stops.iter().filter_map(|s| self.pending.find(s.as_str())).min();
        if let Some(i) = hit {
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
            device: dinfo.name.clone(),
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
        telemetry.set_model(&meta.id, dinfo, model.weights_bytes);
        let backend = model.backend.clone();
        telemetry.set_device_probe(Box::new(move || backend.info()));
        telemetry.set_kv(0, state.total_bytes() as u64, 0, state.cap as u64);

        let shared =
            Arc::new(Shared { inner: Mutex::new(Inner { model, state }), telemetry, codec, eos: meta.eos_ids });
        let (tx, rx) = channel::<Job>();
        let sh = shared.clone();
        let worker = std::thread::Builder::new()
            .name("intarsia-engine".into())
            .spawn(move || {
                for job in rx.iter() {
                    run_job(&sh, job);
                }
            })
            .map_err(|e| Error::Other(format!("spawn engine worker: {e}")))?;
        Ok(LocalEngine { shared, tx: Mutex::new(Some(tx)), info, worker: Mutex::new(Some(worker)) })
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

    fn forward_logits_each(&self, ids: &[u32], cb: &mut dyn FnMut(usize, &[f32]) -> Result<()>) -> Result<()> {
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

    fn telemetry(&self) -> TelemetrySnapshot {
        self.shared.telemetry.snapshot()
    }
}

struct Outcome {
    reason: FinishReason,
    prompt_tokens: usize,
    completion_tokens: usize,
    cached_tokens: usize,
    prefill: Duration,
    decode: Duration,
}

fn run_job(sh: &Shared, job: Job) {
    sh.telemetry.request_started();
    let t_total = Instant::now();
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
        p = 0;
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
        let need_logits = !greedy || k <= req.min_tokens;
        if !need_logits {
            return match model.forward(st, ids, HeadMode::LastArgmax, None)? {
                StepOut::Token(t) => Ok(t),
                _ => Err(Error::Other("backend returned no token".into())),
            };
        }
        match model.forward(st, ids, HeadMode::LastLogits, None)? {
            StepOut::Logits(mut l) => {
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

    let t0 = Instant::now();
    if max_new == 0 {
        return Ok(Outcome {
            reason: FinishReason::Length,
            prompt_tokens: plen,
            completion_tokens: 0,
            cached_tokens: cached,
            prefill: Duration::ZERO,
            decode: Duration::ZERO,
        });
    }
    let mut tok = pick(model, st, &req.prompt[p..], 1, &mut sampler)?;
    let prefill = t0.elapsed();
    let t_dec = Instant::now();

    let codec = &*sh.codec;
    let mut dec = IncrementalDecoder::new();
    let mut stops = StopMatcher::new(&req.stop);
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
        prompt_tokens: plen,
        completion_tokens: generated,
        cached_tokens: cached,
        prefill,
        decode: t_dec.elapsed(),
    })
}
