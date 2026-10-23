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
    /// Structured output (docs/structured-output.md): every sampled token is masked to what this constraint allows.
    /// The constrained steps take the host-logits path; speculative decoding is off for the request. `None` (the
    /// default) leaves the request on the unconstrained path, unchanged.
    pub constraint: Option<Arc<dyn crate::constrain::ConstraintSpec>>,
    /// Prefix store prewarm (`POST /v1/intarsia/prefill`): prefill the prompt into the store and generate nothing. `None`
    /// (the default): a normal request.
    pub prewarm: Option<Prewarm>,
}

/// A prewarm request ([`GenRequest::prewarm`]).
#[derive(Clone, Debug, Default)]
pub struct Prewarm {
    /// Pin the stored prefix under this name (never evicted).
    pub pin: Option<String>,
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
            constraint: None,
            prewarm: None,
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
        /// Prefill chunk size the prompt was processed with. Only the batched engine reports it (it picks 256 or the
        /// model's chunk per request, see `scheduler::SchedulerConfig`); the single-sequence engine sends `None`
        /// (its chunk is the plan's `prefill_chunk`) so CUDA / HIP responses are unchanged.
        prefill_chunk: Option<usize>,
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
    /// The HF `tokenizer.json` text, from which structured output builds its token table; `None` when the codec has
    /// none (structured output is then unavailable).
    fn tokenizer_json(&self) -> Option<String> {
        None
    }
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
    /// The tokenizer's HF `tokenizer.json` text (structured output builds its token table from it); `None` when the
    /// engine has none.
    fn tokenizer_json(&self) -> Option<String> {
        None
    }

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

    /// Scheduler metrics of a batching engine (`None` for the single-sequence engine).
    fn batch_metrics(&self) -> Option<crate::scheduler::BatchMetrics> {
        None
    }

    /// The prefix store's counters and entries (`GET /v1/intarsia/prefix-store`); `Null` when it is off.
    fn prefix_store(&self) -> Value {
        Value::Null
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

mod batch;
pub use batch::{BatchedEngine, BatchedOptions, BUSY_PREFILL_CHUNK};

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
    /// Content-addressed prefix store (`crate::prefix_store`); `None` (the default): off. Single-sequence engine only.
    pub prefix_store: Option<crate::prefix_store::PrefixStoreConfig>,
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
    /// The prefix store and how requests use it (`EngineMeta::prefix_store`).
    store: Option<StorePolicy>,
}

/// How the single-sequence engine uses its prefix store.
#[derive(Clone)]
pub(crate) struct StorePolicy {
    pub(crate) store: Arc<Mutex<crate::prefix_store::PrefixStore>>,
    /// Token ids that start a message boundary (each position holding one is a chunk end and an entry).
    pub(crate) boundary_ids: Vec<u32>,
    /// Token ids that end a message header: a cut right after the first one following a boundary token.
    pub(crate) header_end_ids: Vec<u32>,
    /// `PrefixStoreConfig::cut_roles`, tokenized (empty: every header is a cut).
    pub(crate) cut_roles: Vec<Vec<u32>>,
    /// Prefill is bitwise the same however the prompt is chunked on this backend (cpu, hip): any prefill-written length
    /// is a valid restore point. Otherwise only points of the prompt's own chunk schedule are.
    pub(crate) chunk_invariant: bool,
    pub(crate) min_tokens: usize,
    pub(crate) grid_tokens: usize,
}

impl StorePolicy {
    fn new(model: &Model, codec: &dyn TextCodec, meta: &EngineMeta, st: &SequenceState) -> Result<Option<StorePolicy>> {
        let Some(cfg) = meta.prefix_store.clone() else { return Ok(None) };
        if model.is_tp() {
            return Err(Error::Unsupported("the prefix store with tensor parallelism".into()));
        }
        let one = |list: &[String]| -> Vec<u32> {
            let mut out = Vec::new();
            for t in list {
                match codec.encode(t, false) {
                    Ok(ids) if ids.len() == 1 => out.push(ids[0]),
                    _ => eprintln!("prefix store: {t:?} is not one token of this tokenizer; ignored"),
                }
            }
            out
        };
        let boundary_ids = one(&cfg.boundary_tokens);
        let header_end_ids = one(&cfg.header_end_tokens);
        let cut_roles: Vec<Vec<u32>> = cfg.cut_roles.iter().filter_map(|r| codec.encode(r, false).ok()).collect();
        // chunk invariance is a property of backend and model (the hybrid's GatedDeltaNet is chunk-sensitive even on
        // cpu): restore only at schedule points unless the operator says otherwise
        let chunk_invariant = cfg.any_cut;
        let desc = crate::prefix_store::describe_build(model, &meta.id, &cfg);
        let (min_tokens, grid_tokens) = (cfg.min_tokens, cfg.grid_tokens);
        let store = crate::prefix_store::PrefixStore::new(cfg, model.slot_specs(), &desc)?;
        let _ = st;
        Ok(Some(StorePolicy { store: Arc::new(Mutex::new(store)), boundary_ids, header_end_ids, cut_roles, chunk_invariant, min_tokens, grid_tokens }))
    }
}

/// A request's view of the prefix store ([`Seq`]): its prompt's boundaries and schedule points.
pub(crate) struct SeqStore {
    pol: StorePolicy,
    /// Positions `i` (`0 < i < plen`) where the prompt holds a boundary token: the state for `prompt[..i]` is stored.
    boundaries: Vec<usize>,
    /// The cuts right after a message header (`<|im_start|>role\n`), and the prompt's end when a header ends it: where
    /// entries are stored. The cut before the header is a chunk end only (an entry there would hold nearly the same
    /// tokens for another 63 MiB of recurrent state).
    headers: Vec<usize>,
    /// Message starts in the prompt (a pinned prewarm of one message ends where the next request cuts: message 2).
    messages: usize,
    /// The prompt's cold chunk schedule (every chunk end; `plen - 1` for a recurrent state): the lengths a restore may
    /// return to when prefill is chunk-sensitive.
    points: std::collections::BTreeSet<usize>,
    /// Length of the last entry this request stored (grid spacing).
    last_insert: usize,
}

impl SeqStore {
    fn new(pol: StorePolicy, prompt: &[u32], recurrent: bool, chunk: usize) -> SeqStore {
        let plen = prompt.len();
        // cuts: after the end of every message header (`<|im_start|>role\n`), and before the start of the second
        // message (where a pinned system prompt ends). Not before every message start: on a MoE every chunk reads all
        // experts once, so a 3-token chunk per message costs a decode step each (measured on CUDA, results doc).
        let mut boundaries = Vec::new();
        let mut headers = Vec::new();
        let mut in_header = false;
        let mut starts = 0usize;
        let mut hdr_start = 0usize;
        for (i, t) in prompt.iter().enumerate() {
            if pol.boundary_ids.contains(t) {
                starts += 1;
                hdr_start = i;
                if starts == 2 && i > 0 {
                    boundaries.push(i);
                }
                in_header = true;
            } else if in_header && pol.header_end_ids.contains(t) {
                let role = &prompt[hdr_start + 1..i];
                let wanted = pol.cut_roles.is_empty() || starts == 2 || pol.cut_roles.iter().any(|r| r[..] == *role);
                if wanted && i + 1 < plen {
                    boundaries.push(i + 1);
                }
                if wanted {
                    // a header ending the prompt is no cut (the last pass ends there anyway) but still an entry point
                    headers.push(i + 1);
                }
                in_header = false;
            }
        }
        let mut points = std::collections::BTreeSet::new();
        let mut pos = 0;
        while pos < plen {
            let rem = plen - pos;
            let n = Self::chunk_end(&boundaries, pos, rem.min(chunk), chunk) - pos;
            if n < rem {
                pos += n;
                points.insert(pos);
            } else {
                // no `plen - 1` point: with the store on the last chunk runs whole (a regenerate restores the entry
                // after the last header instead), which saves a pass per request; on a MoE a pass reads every expert
                let _ = recurrent;
                break;
            }
        }
        SeqStore { pol, boundaries, headers, messages: starts, points, last_insert: 0 }
    }

    /// A length `st` can return to bitwise, at most `cap` (already capped at the prefill-clean watermark): on a
    /// chunk-sensitive backend a point of this prompt's schedule only.
    fn local_reuse(&self, st: &SequenceState, cap: usize) -> usize {
        let ok = |l: usize| l == 0 || self.pol.chunk_invariant || self.points.contains(&l);
        if !st.has_recurrent() {
            return if self.pol.chunk_invariant { cap } else { self.points.range(..=cap).next_back().copied().unwrap_or(0) };
        }
        let mut best = 0;
        for l in st.snapshots.iter().map(|s| s.len).chain(std::iter::once(st.len)) {
            if l <= cap && l > best && ok(l) && st.restorable_len(l) == l {
                best = l;
            }
        }
        best
    }

    /// Restore the longest stored prefix of `prompt` (at most `max_len` tokens) when it beats the `local` length the
    /// state already holds: `Some((length, time))`, or `None` (nothing restored, the caller truncates to `local`).
    fn restore(
        &self,
        model: &Model,
        st: &mut SequenceState,
        prompt: &[u32],
        local: usize,
        max_len: usize,
        tel: &Telemetry,
    ) -> Result<Option<(usize, Duration)>> {
        let mut store = self.pol.store.lock().unwrap_or_else(|e| e.into_inner());
        let hit = store.lookup_filtered(prompt, max_len, |l| self.pol.chunk_invariant || self.points.contains(&l) || l == prompt.len());
        let out = match hit {
            Some(h) if h.len > local => {
                let r = store.restore(&*model.backend, st, h);
                tel.set_prefix_store(store.snapshot());
                let r = r?;
                Some((r.len, r.elapsed))
            }
            Some(_) => {
                store.note_local_win();
                None
            }
            None => None,
        };
        tel.set_prefix_store(store.snapshot());
        Ok(out)
    }

    /// End of the next prefill chunk from `pos` (the engine's rule, cut at the next boundary): a function of the prompt
    /// tokens and `pos` only, so a restored run and a cold run cut the same chunks after any schedule point.
    fn chunk_end(boundaries: &[usize], pos: usize, n: usize, chunk: usize) -> usize {
        let i = boundaries.partition_point(|&b| b <= pos);
        match boundaries.get(i) {
            Some(&b) if b < pos + n => pos + (b - pos).min(chunk),
            _ => pos + n,
        }
    }
}


impl Shared {
    fn failure(&self) -> Option<String> {
        self.failed.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn env(&self) -> SeqEnv {
        SeqEnv {
            codec: self.codec.clone(),
            telemetry: self.telemetry.clone(),
            eos: self.eos.clone(),
            spec: self.spec,
        }
    }
}

/// What a [`Seq`] needs from its engine besides the model: shared by the single-sequence engine and the batched one.
#[derive(Clone)]
pub(crate) struct SeqEnv {
    pub(crate) codec: Arc<dyn TextCodec>,
    pub(crate) telemetry: Arc<Telemetry>,
    pub(crate) eos: Vec<u32>,
    pub(crate) spec: crate::spec::SpecConfig,
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

/// The [`ModelInfo`] an engine over `model` reports; `context_len` is the token capacity of one sequence.
pub(crate) fn build_info(
    model: &Model,
    dinfo: &crate::backend::DeviceInfo,
    meta: &EngineMeta,
    codec: &dyn TextCodec,
    context_len: usize,
) -> ModelInfo {
    ModelInfo {
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
        context_len,
        act_dtype: model.act.name().into(),
        kv_dtype: model.kv.name().into(),
        weights_bytes: model.weights_bytes,
        eos_ids: meta.eos_ids.clone(),
        has_chat_template: codec.has_chat_template(),
        reasoning_parser: meta.reasoning_parser.clone(),
        tool_parser: meta.tool_parser.clone(),
        memory_plan: model.plan.clone(),
    }
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
            let mtp = meta.spec.mode == crate::spec::SpecMode::Mtp;
            if mtp && (!model.has_mtp() || state.mtp.is_none()) {
                return Err(Error::invalid("--spec mtp needs the model loaded with its MTP head (LoadOptions::mtp)"));
            }
            // the MTP path keeps per-row recurrent states in its verify passes; n-gram drafting does not
            if state.has_recurrent() && !mtp {
                return Err(Error::Unsupported("speculative decoding on a model with recurrent state".into()));
            }
        }
        let dinfo = model.backend.info();
        let info = build_info(&model, &dinfo, &meta, &*codec, state.cap);
        let plan = crate::plan::plan_json(&model, &meta.id);
        telemetry.set_model(&meta.id, dinfo, model.weights_bytes);
        let backend = model.backend.clone();
        telemetry.set_device_probe(Box::new(move || backend.info()));
        telemetry.set_kv(0, state.total_bytes() as u64, 0, state.cap as u64);

        let store = StorePolicy::new(&model, &*codec, &meta, &state)?;
        if let Some(s) = &store {
            let mut g = s.store.lock().unwrap();
            telemetry.set_prefix_store(g.snapshot());
            eprintln!(
                "prefix store: on (build {}, {} entries loaded, boundaries {:?}, chunk-invariant backend: {})",
                g.snapshot().build,
                g.snapshot().entries,
                s.boundary_ids,
                s.chunk_invariant
            );
            drop(g);
        }
        let shared = Arc::new(Shared {
            inner: Mutex::new(Inner { model, state }),
            telemetry,
            codec,
            eos: meta.eos_ids,
            spec: meta.spec,
            failed: Mutex::new(None),
            store,
        });
        let (tx, rx) = channel::<Job>();
        let sh = shared.clone();
        let worker = std::thread::Builder::new()
            .name("intarsia-engine".into())
            .spawn(move || {
                for job in rx.iter() {
                    if let Some(why) = sh.failure() {
                        // queued, then the engine died: still an accepted request, counted as an error
                        sh.telemetry.request_dropped_queued("error");
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
    fn tokenizer_json(&self) -> Option<String> {
        self.shared.codec.tokenizer_json()
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

    fn prefix_store(&self) -> Value {
        let Some(pol) = &self.shared.store else { return Value::Null };
        let mut s = pol.store.lock().unwrap_or_else(|e| e.into_inner());
        serde_json::json!({
            "stats": s.snapshot(),
            "boundary_ids": pol.boundary_ids,
            "chunk_invariant": pol.chunk_invariant,
            "entries": s.entries(),
        })
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

/// Report a finished request: telemetry, then the `Done` event. Shared by the single-sequence worker and the batched
/// engine so the two cannot drift. `chunk` is the prefill chunk size the request's prompt was processed with, when the
/// engine reports it (batched engine only).
pub(crate) fn report_done(telemetry: &Telemetry, tx: &Sender<GenEvent>, o: Outcome, chunk: Option<usize>) {
    telemetry.request_finished(
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
        prefill_chunk: chunk,
    });
}

/// Report a failed request (one that had started): telemetry, then the `Error` event. `generated` is the number of
/// tokens the request had generated before it failed (already streamed to the client), so the generated-token counter
/// covers failed requests too.
pub(crate) fn report_error(
    telemetry: &Telemetry,
    tx: &Sender<GenEvent>,
    plen: usize,
    generated: usize,
    elapsed: Duration,
    e: &Error,
) {
    telemetry.request_finished("error", plen, generated, 0, elapsed, Duration::ZERO, elapsed);
    let _ = tx.send(GenEvent::Error { message: e.to_string() });
}

fn run_job(sh: &Shared, job: Job) {
    sh.telemetry.request_started();
    let t_total = Instant::now();
    debug_hooks();
    let Job { req, tx } = job;
    let plen = req.prompt.len();
    let mut g = sh.inner.lock().unwrap();
    let mut generated = 0;
    let res = generate_one(sh, &mut g, req, tx.clone(), &mut generated);
    let kv = (g.state.used_bytes() as u64, g.state.total_bytes() as u64, g.state.len as u64, g.state.cap as u64);
    drop(g);
    sh.telemetry.set_kv(kv.0, kv.1, kv.2, kv.3);
    match res {
        // the chunk size is reported only by the batched engine (its scheduler picks it per request)
        Ok(o) => report_done(&sh.telemetry, &tx, o, None),
        Err(e) => report_error(&sh.telemetry, &tx, plen, generated, t_total.elapsed(), &e),
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

/// Most sequences `--max-batch` may ask for on a backend that does not state its own cap (`Backend::batch_row_cap`
/// default; Metal's): the batching scheduler (docs/research/continuous-batching-design.md, M1) decodes up to that many
/// together on backends with row-exact multi-row kernels (Metal, CUDA); `--max-batch 1` keeps the single-sequence
/// `LocalEngine`. The CUDA backend states 16.
pub const SERVE_MAX_BATCH: usize = 8;

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
    // A masked stop on row 0 keeps nothing of the pass (`keep == len0`). A recurrent or MTP state cannot roll back to
    // before the pass (`SequenceState::spec_rollback` keeps at least one row), so `Seq::plan_step` never drafts with
    // `--spec mtp` inside a min_tokens window that masks stop ids. Relaxing that guard needs a pre-pass state copy.
    debug_assert!(
        masked_stop != Some(0) || !(st.has_recurrent() || st.mtp.is_some()),
        "masked stop on verify row 0 of a recurrent/MTP state: plan_step's min_tokens guard was bypassed"
    );
    let step = match masked_stop {
        // rows 0..i emitted d_1..d_i; row i (fed d_i) is recomputed by the plain step, so its KV row goes too
        Some(i) => SpecStep { emit: out[..i].to_vec(), accepted: i, plain_next: true },
        None => SpecStep { emit: out[..=j].to_vec(), accepted: j, plain_next: false },
    };
    let keep = len0 + if step.plain_next { step.accepted } else { step.accepted + 1 };
    spec_cut(model, st, keep)?;
    Ok(step)
}

/// `INTARSIA_MTP_VERIFY_CHECK=1`: before every `--spec mtp` verify pass, run its rows as single decode steps too and print
/// the max |logit difference| and whether the argmax agrees, per row ([`Model::verify_check`]). Diagnostic, slow.
fn mtp_verify_check() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INTARSIA_MTP_VERIFY_CHECK").is_ok_and(|v| !v.is_empty() && v != "0"))
}

/// Cut the state back to `keep` tokens after (or inside) the last verify pass: a recurrent or MTP state through the
/// verify pass's per-row record ([`SequenceState::spec_rollback`]), an attention-only one by truncation.
fn spec_cut(model: &Model, st: &mut SequenceState, keep: usize) -> Result<()> {
    if st.has_recurrent() || st.mtp.is_some() {
        st.spec_rollback(keep, &*model.backend)
    } else {
        st.truncate(keep, &*model.backend)
    }
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
    /// Draft up to this many tokens with the MTP head after `tok`, then verify them ([`Seq::prepare`]).
    Draft(usize),
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
    /// Drafts from the model's MTP head (`--spec mtp`) instead of prompt lookup.
    spec_mtp: bool,
    spec_k: usize,
    /// The batching scheduler withholds drafting (batch too large); set per step by [`Seq::set_spec_allowed`].
    spec_paused: bool,
    /// Other sequences decode in the same scheduler step; set per step by [`Seq::set_spec_allowed`] (solo: false).
    spec_busy: bool,
    pending: std::collections::VecDeque<u32>,
    /// The next token must come from a plain step (a verify row inside the min_tokens mask chose a stop id).
    plain_next: bool,
    /// Prompt + tokens so far: the drafter's lookup context.
    ctx: Vec<u32>,
    /// Stops drafting where it does not pay (open-ended text), see `spec::GateParams`.
    gate: crate::spec::SpecGate,
    /// The request's constraint state (`GenRequest::constraint`), or why it could not start (reported by `admit`).
    cons: Option<std::result::Result<crate::constrain::ConstraintRun, String>>,
    /// The prefix store's view of this request (single-sequence engine with the store on); `None`: the store is off and
    /// prefill runs exactly as without it.
    ps: Option<SeqStore>,
    /// Time spent restoring a stored prefix (part of the time to first token).
    restore_time: Duration,
}

/// Reject prompts a sequence of capacity `cap` cannot take.
fn check_prompt(plen: usize, cap: usize) -> Result<()> {
    if plen == 0 {
        return Err(Error::BadRequest("empty prompt".into()));
    }
    if plen >= cap {
        return Err(Error::BadRequest(format!(
            "prompt of {plen} tokens does not fit the context window of {cap} tokens"
        )));
    }
    Ok(())
}

impl Seq {
    /// Validate the request and reuse the KV prefix `st` holds (truncating it to what is reused).
    fn new(sh: &Shared, model: &Model, st: &mut SequenceState, req: GenRequest, tx: Sender<GenEvent>) -> Result<Seq> {
        let t_start = Instant::now();
        let plen = req.prompt.len();
        check_prompt(plen, st.cap)?;
        let recurrent = st.has_recurrent();
        // a prewarm computes the whole prompt (no token to sample), a request keeps >= 1 token to compute
        let max_reuse = if req.prewarm.is_some() { plen } else { plen - 1 };
        let mut seq = Seq::build(&sh.env(), req, tx, model.verify_rows(), recurrent);
        // `--spec mtp`: only a drafting request runs the head. After a request that did not, the head's cache no
        // longer covers the sequence, so a drafting request then starts from an empty sequence (no prefix reuse).
        let drafting = seq.spec_on && seq.spec_mtp;
        // the prefix store restores the main model's state only, never the MTP head's cache: a drafting request on a
        // model with an MTP head bypasses the store (the store stays off for it, prefill runs exactly as without it)
        let ps = sh
            .store
            .as_ref()
            .filter(|_| !(drafting && st.mtp.is_some()))
            .map(|pol| SeqStore::new(pol.clone(), &seq.req.prompt, recurrent, model.prefill_chunk().max(1)));

        // KV prefix reuse (contiguous cache of the previous request; always keep >= 1 token to compute)
        let mut p = common_prefix(&st.tokens, &seq.req.prompt).min(plen - 1);
        if let Some(ps) = &ps {
            // bitwise reuse only: what prefill wrote, at a point of this prompt's own chunk schedule
            p = ps.local_reuse(st, p.min(st.clean));
        } else if recurrent {
            // Recurrent state (GatedDeltaNet / Mamba) cannot be shortened: reuse only what the state can return
            // to exactly: the whole held sequence, a snapshot taken during an earlier prefill, or nothing.
            p = st.restorable_len(p);
        }
        if st.mtp.as_ref().is_some_and(|m| drafting && !m.synced) {
            p = 0;
        }
        let mut restored = Duration::ZERO;
        if let Some(ps) = &ps {
            // inserts during this prefill copy device-to-device and leave the host copy for later
            let mut store = ps.pol.store.lock().unwrap_or_else(|e| e.into_inner());
            store.settle(&*model.backend);
            store.set_defer(true);
        }
        if let Some(ps) = &ps {
            match ps.restore(model, st, &seq.req.prompt, p, max_reuse, &sh.telemetry) {
                Ok(Some((len, took))) => {
                    p = len;
                    restored = took;
                }
                Ok(None) => st.truncate(p, &*model.backend)?,
                Err(e) => {
                    // a rejected entry (corrupt file, layout mismatch): run cold
                    eprintln!("prefix store: restore failed ({e}); prefilling from scratch");
                    st.truncate(0, &*model.backend)?;
                    p = 0;
                }
            }
        } else {
            st.truncate(p, &*model.backend)?;
        }
        if let Some(m) = st.mtp.as_mut() {
            m.active = drafting;
        }
        if seq.req.prewarm.is_some() {
            // a prewarm is admitted for its prefill only
            seq.req.max_tokens = 1;
        }
        seq.admit(p, st.cap)?;
        seq.ps = ps;
        if let Some(ps) = &mut seq.ps {
            ps.last_insert = p;
        }
        seq.restore_time = restored;
        if seq.ps.is_some() {
            // the time to first token includes the lookup and the restore
            seq.t0 = t_start;
        }
        Ok(seq)
    }

    /// Construct the per-request machinery (sampler, stop ids, stop matcher, speculation switches) without touching
    /// any KV state. `verify_rows` and `recurrent` describe the model and state the sequence will run on. Pair with
    /// [`Seq::admit`] once the slot and its reusable prefix are known.
    pub(crate) fn build(env: &SeqEnv, req: GenRequest, tx: Sender<GenEvent>, verify_rows: usize, recurrent: bool) -> Seq {
        let plen = req.prompt.len();
        let params = SamplingParams { temperature: req.temperature, top_k: req.top_k, top_p: req.top_p };
        let greedy = req.temperature <= 0.0;
        let sampler = Sampler::new(params, req.seed);
        let mut stop_ids: Vec<u32> = req.stop_ids.clone();
        if !req.ignore_eos {
            stop_ids.extend_from_slice(&env.eos);
        }
        // Like vLLM, stop strings are not checked until `min_tokens` tokens exist.
        let stops = StopMatcher::new(if req.min_tokens > 0 { &[] } else { &req.stop });
        let cons = req
            .constraint
            .as_ref()
            .map(|c| c.start().map(crate::constrain::ConstraintRun::new).map_err(|e| e.to_string()));
        // constrained requests take plain steps only (drafts would bypass the mask)
        let spec_mtp = env.spec.mode == crate::spec::SpecMode::Mtp;
        let spec_on = env.spec.enabled()
            && greedy
            && req.trace_logits.is_none()
            && cons.is_none()
            && (!recurrent || spec_mtp)
            && verify_rows >= 2;
        let spec_k = env.spec.k.min(verify_rows.saturating_sub(1));
        let ctx: Vec<u32> = if spec_on { req.prompt.clone() } else { Vec::new() };
        let now = Instant::now();
        Seq {
            plen,
            cached: 0,
            max_new: 0,
            greedy,
            sampler,
            stop_ids,
            t0: now,
            pos: 0,
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
            spec_mtp,
            spec_k,
            spec_paused: false,
            spec_busy: false,
            pending: std::collections::VecDeque::new(),
            plain_next: false,
            ctx,
            gate: crate::spec::SpecGate::new(crate::spec::GateParams::from_env()),
            cons,
            codec: env.codec.clone(),
            telemetry: env.telemetry.clone(),
            req,
            tx,
            ps: None,
            restore_time: Duration::ZERO,
        }
    }

    /// Take a slot of capacity `cap` whose KV holds `cached` reusable prompt tokens (`< prompt.len()`): validate the
    /// prompt against `cap`, set what `max_tokens` leaves room for. `Ok(false)`: nothing to generate (`max_tokens = 0`
    /// or no room), the finish reason is set to `Length` and no prefill is needed. Errors reject the request.
    pub(crate) fn admit(&mut self, cached: usize, cap: usize) -> Result<bool> {
        check_prompt(self.plen, cap)?;
        if let Some(Err(e)) = &self.cons {
            return Err(Error::BadRequest(format!("structured output: {e}")));
        }
        debug_assert!(cached < self.plen || (self.req.prewarm.is_some() && cached == self.plen));
        self.max_new = self.req.max_tokens.min(cap - self.plen);
        self.cached = cached;
        self.pos = cached;
        let now = Instant::now();
        self.t0 = now;
        self.t_dec = now;
        if self.max_new == 0 {
            self.reason = Some(FinishReason::Length);
            return Ok(false);
        }
        // the first token's mask is computed while the prompt is prefilled
        self.prefetch_mask();
        Ok(true)
    }

    /// End the sequence as cancelled (the scheduler's cancellation check). [`Seq::finish`] then reports it.
    pub(crate) fn cancel(&mut self) {
        self.reason = Some(FinishReason::Cancelled);
    }

    /// Whether speculation may draft on the next step (the batching scheduler turns it off at larger batch sizes), and
    /// whether other sequences decode in that step (`busy`: the drafter gate then also stops a request whose drafts
    /// rarely pay, see [`crate::spec::SpecGate::allow`]).
    pub(crate) fn set_spec_allowed(&mut self, allowed: bool, busy: bool) {
        self.spec_paused = !allowed;
        self.spec_busy = busy;
    }

    /// The outcome of a request that ended before it held any state (cancelled while queued).
    pub(crate) fn queued_outcome(&self, reason: FinishReason) -> Outcome {
        Outcome {
            reason,
            stop_sequence: None,
            prompt_tokens: self.plen,
            completion_tokens: 0,
            cached_tokens: 0,
            prefill: Duration::ZERO,
            decode: Duration::ZERO,
        }
    }

    /// Whether the next token to take is the first (it comes out of the final prefill chunk).
    pub(crate) fn awaiting_first(&self) -> bool {
        self.generated == 0
    }

    /// The final prefill chunk's output: generated token 1. Starts the decode clock.
    pub(crate) fn accept_first(&mut self, out: StepOut) -> Result<()> {
        self.tok = self.token_from(out, 1)?;
        self.prefill = self.t0.elapsed();
        self.t_dec = Instant::now();
        Ok(())
    }

    /// The next token is masked by an active constraint.
    fn constrained(&self) -> bool {
        matches!(&self.cons, Some(Ok(c)) if c.active())
    }

    /// Start the next step's constraint mask on a helper thread, overlapping the forward pass about to run.
    fn prefetch_mask(&mut self) {
        if let Some(Ok(c)) = &mut self.cons {
            c.prefetch();
        }
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
        if !self.greedy || masks_stop_ids || self.req.trace_logits.is_some() || self.constrained() {
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
                if let Some(Ok(c)) = &mut self.cons {
                    if c.active() {
                        let m = c.mask()?;
                        let t = crate::constrain::prof::t0();
                        let ok = m.apply(&mut l);
                        crate::constrain::prof::add(&crate::constrain::prof::APPLY, t);
                        if !ok {
                            return Err(Error::Other("structured output: the constraint allows no token here".into()));
                        }
                        let t = crate::constrain::prof::t0();
                        let tok = self.sampler.sample(&l) as u32;
                        crate::constrain::prof::add(&crate::constrain::prof::SAMPLE, t);
                        crate::constrain::prof::step();
                        return Ok(tok);
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
        let prewarm = self.req.prewarm.is_some();
        if prewarm && self.ps.is_none() {
            return Err(Error::BadRequest("prewarm needs the prefix store (--prefix-store)".into()));
        }
        if self.max_new == 0 && !prewarm {
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
        if prewarm && self.pos == self.plen {
            // the whole prompt was already stored (restored): only pin it
            self.store_at(model, st, true)?;
            return Ok(PrefillStep::Finished(self.prewarm_outcome()));
        }
        let remaining = self.plen - self.pos;
        let n = if let Some(ps) = &self.ps {
            // prefix store on: the plain chunk cut at the next message boundary, a function of the prompt and the
            // position only (no tail merge, no 1-row avoidance: both depend on the prompt's length), so the state at
            // any chunk end is the state any other prompt with the same prefix computes there
            SeqStore::chunk_end(&ps.boundaries, self.pos, remaining.min(chunk), chunk) - self.pos
        } else if st.has_recurrent() {
            // recurrent states keep the plain chunking (their last token runs alone below, for the snapshot rule)
            // (a tail merge folds a short remainder into the last chunk: `remaining` runs in one pass)
            if crate::scheduler::next_prefill_chunk_tm(remaining, chunk, false, model.tail_merge()) == remaining {
                remaining
            } else {
                chunk
            }
        } else {
            crate::scheduler::next_prefill_chunk_tm(
                remaining,
                chunk,
                model.backend.avoid_single_row_prefill(),
                model.tail_merge(),
            )
        };
        if n < remaining || prewarm {
            let start = self.pos;
            match model.forward(st, &self.req.prompt[self.pos..self.pos + n], HeadMode::None, None)? {
                StepOut::None => {}
                _ => return Err(Error::Other("backend returned output for a prefill chunk".into())),
            }
            self.pos += n;
            st.snapshot(&*model.backend, crate::state::RECURRENT_SNAPSHOTS)?;
            if st.clean == start {
                st.clean = st.len;
            }
            let last = self.pos == self.plen;
            self.store_at(model, st, last)?;
            if last {
                return Ok(PrefillStep::Finished(self.prewarm_outcome()));
            }
            return Ok(PrefillStep::Chunk);
        }
        if st.has_recurrent() && self.plen - self.pos > 1 && self.ps.is_none() {
            // snapshot one token before the end of the prompt: a repeated or regenerated request reuses everything
            // but its last token, as with an attention-only cache
            let start = self.pos;
            model.forward(st, &self.req.prompt[self.pos..self.plen - 1], HeadMode::None, None)?;
            self.pos = self.plen - 1;
            st.snapshot(&*model.backend, crate::state::RECURRENT_SNAPSHOTS)?;
            if st.clean == start {
                st.clean = st.len;
            }
        }
        let start = self.pos;
        let mode = self.head_mode(1);
        let out = model.forward(st, &self.req.prompt[self.pos..], mode, None)?;
        if st.clean == start {
            st.clean = st.len;
        }
        if self.ps.as_ref().is_some_and(|ps| ps.headers.last() == Some(&self.plen)) {
            // the prompt ends at its last message header (Qwen3's generation prompt `<|im_start|>assistant\n`, no
            // `<think>` after it): this final pass is that cut's chunk, so store its state (the next turn restores it).
            // The head adds ops after the last layer and writes no state, so the state is the one a plain chunk
            // ending here writes.
            let at = std::mem::replace(&mut self.pos, self.plen);
            let stored = self.store_at(model, st, false);
            self.pos = at;
            stored?;
        }
        self.tok = self.token_from(out, 1)?;
        self.prefill = self.t0.elapsed();
        self.t_dec = Instant::now();
        if let Some(ps) = &self.ps {
            let mut store = ps.pol.store.lock().unwrap_or_else(|e| e.into_inner());
            store.note_prefill(self.plen - self.cached, self.prefill.saturating_sub(self.restore_time));
            self.telemetry.set_prefix_store(store.snapshot());
        }
        Ok(PrefillStep::Started)
    }

    /// After a prefill chunk ended at `self.pos` (prefix store on): store the state there when it is a message boundary,
    /// a grid point (`grid_tokens` past the last entry) or the end of a prewarm, and only if it is prefill-clean and
    /// newly computed. `last`: the end of a prewarm (pinned when asked).
    fn store_at(&mut self, model: &Model, st: &mut SequenceState, last: bool) -> Result<()> {
        let Some(ps) = &mut self.ps else { return Ok(()) };
        let pos = self.pos;
        let pin = if last { self.req.prewarm.as_ref().and_then(|p| p.pin.clone()) } else { None };
        let fresh = pos > self.cached || pin.is_some();
        if pos < ps.pol.min_tokens || st.clean < pos || st.len != pos || !fresh {
            return Ok(());
        }
        // after a message header: the last one of the prompt always (the next turn of a chat restores it), the others
        // when at least `min_tokens` past the previous entry (an agent loop's short tool messages)
        let header = ps.headers.binary_search(&pos).is_ok();
        let last_header = ps.headers.last() == Some(&pos);
        let spaced = pos - ps.last_insert >= ps.pol.min_tokens;
        if !(last || last_header || (header && spaced) || pos - ps.last_insert >= ps.pol.grid_tokens) {
            return Ok(());
        }
        let mut store = ps.pol.store.lock().unwrap_or_else(|e| e.into_inner());
        // the chunk ends that wrote rows 0..pos: this prompt's schedule points (they are the same for every prompt that
        // shares the prefix up to a point, which is what makes a stored page reusable)
        let ends: Vec<usize> = ps.points.range(..pos).copied().chain(std::iter::once(pos)).collect();
        // a failed copy (out of memory) costs the entry, never the request
        if let Err(e) = store.insert(&*model.backend, st, pos, pin.as_deref(), &ends) {
            eprintln!("prefix store: insert at {pos} tokens failed: {e}");
        }
        if let (Some(name), true) = (&pin, ps.messages >= 2 && ps.last_insert > 0 && ps.last_insert < pos) {
            // a pinned prewarm also pins its last header entry: a later request restores that one when the prewarm's
            // end is not a cut of its own schedule (a prefix of several messages)
            store.pin(&st.tokens[..ps.last_insert], name);
        }
        ps.last_insert = pos;
        self.telemetry.set_prefix_store(store.snapshot());
        Ok(())
    }

    /// Move the entries this request captured on the device to their tier ([`crate::prefix_store::PrefixStore::settle`]).
    pub(crate) fn settle_store(&self, model: &Model) {
        if let Some(ps) = &self.ps {
            let mut store = ps.pol.store.lock().unwrap_or_else(|e| e.into_inner());
            if store.deferring() {
                store.settle(&*model.backend);
                self.telemetry.set_prefix_store(store.snapshot());
            }
        }
    }

    /// The end of a prewarm request: no token generated, the prompt counted as processed.
    fn prewarm_outcome(&mut self) -> Outcome {
        self.prefill = self.t0.elapsed();
        Outcome {
            reason: FinishReason::Length,
            stop_sequence: None,
            prompt_tokens: self.plen,
            completion_tokens: 0,
            cached_tokens: self.cached,
            prefill: self.prefill,
            decode: Duration::ZERO,
        }
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
        if let Some(Ok(c)) = &mut self.cons {
            c.commit(tok).map_err(|e| Error::Other(format!("structured output: {e}")))?;
        }
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
        if self.spec_on && !self.spec_paused && !std::mem::take(&mut self.plain_next) && self.gate.allow(self.spec_busy) {
            // verify feeds 1 + d tokens: keep them inside the context window
            let room = st.cap.saturating_sub(st.len + 1);
            if self.spec_mtp {
                // MTP drafts are always available. Not inside a min_tokens window that masks stop ids (a masked stop on
                // row 0 would have to roll the recurrent state back to before the pass; with no stop ids, e.g.
                // ignore_eos, nothing is masked), and no drafts past max_tokens (a pass emits up to k + 1 tokens).
                let k = self.spec_k.min(room).min(self.max_new.saturating_sub(self.generated + 1));
                if k > 0 && (self.generated >= self.req.min_tokens || self.stop_ids.is_empty()) {
                    return Next::Draft(k);
                }
                return Next::Plain;
            }
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
        self.prefetch_mask();
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
            Next::Draft(k) => {
                let t0 = Instant::now();
                let drafts = model.mtp_draft(st, self.tok, k)?;
                if mtp_verify_check() {
                    let mut ids = vec![self.tok];
                    ids.extend_from_slice(&drafts);
                    let rows = model.verify_check(st, &ids)?;
                    // a row decides an emitted token when every draft before it was accepted
                    let decides = |r: usize| (0..r).all(|i| drafts[i] == rows[i].2);
                    let line: Vec<String> = rows
                        .iter()
                        .enumerate()
                        .map(|(r, (d, a, b, m))| format!("{d:.3e}:{}:{m:.3e}:{}", u8::from(a == b), u8::from(decides(r))))
                        .collect();
                    eprintln!("spec-check: rows {} {}", rows.len(), line.join(" "));
                }
                let t1 = Instant::now();
                self.verify(model, st, &drafts)?;
                if crate::spec::trace_enabled() {
                    // both phases end in a token download, so host time is device time here
                    eprintln!(
                        "spec: mtp draft {:.3} ms, verify + rollback {:.3} ms",
                        (t1 - t0).as_secs_f64() * 1e3,
                        t1.elapsed().as_secs_f64() * 1e3
                    );
                }
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

    /// Whether the rest of this request can decode with device-side token feedback (`model/feedback.rs`): every
    /// remaining token is an on-device argmax ([`Seq::head_mode`] is `LastArgmax` from here on: it only changes
    /// across the `min_tokens` boundary, which lies behind), no speculative decoding (drafting needs the tokens on the
    /// host), no structured-output constraint (a lazy one activates on a token the loop would only see after it
    /// speculated the next step unmasked), and the model allows it for `st`.
    pub(crate) fn feedback_ok(&self, model: &Model, st: &SequenceState) -> bool {
        !self.spec_on
            && self.cons.is_none()
            && self.pending.is_empty()
            && self.head_mode(self.generated + 1) == HeadMode::LastArgmax
            && model.feedback_ok(st)
    }

    /// Decode the rest of the request with token feedback: step N+1 is submitted before step N's token is read. Call
    /// after [`Seq::on_token`] returned false and [`Seq::feedback_ok`]; returns once `on_token` returned true (then
    /// [`Seq::finish`]). A stop is seen one step late, so the step submitted after the last token is discarded: the
    /// state is cut back to what the plain loop holds, every emitted token but the last fed (also after an error).
    pub(crate) fn run_feedback(&mut self, model: &Model, st: &mut SequenceState) -> Result<()> {
        let r = self.feedback_loop(model, st);
        if r.is_err() {
            // a held step must not stay open in the stream
            let _ = model.feedback_discard();
        }
        let want = self.plen + self.generated - 1;
        if st.len > want {
            let t = st.truncate(want, &*model.backend);
            if r.is_ok() {
                t?;
            }
        }
        r
    }

    fn feedback_loop(&mut self, model: &Model, st: &mut SequenceState) -> Result<()> {
        use crate::model::feedback::{Feed, RING, UNREAD};
        let (tok, _) = self.plain_input();
        let mut cur = model.feedback_step(st, Feed::Host(tok), RING[0], false)?;
        loop {
            // `cur` samples token generated + 1; unless max_tokens ends the request there, feed it on to the next step
            // before reading it (its start queued behind `cur`, the rest held until the token says it is needed)
            let next = if self.generated + 1 < self.max_new {
                let out = if cur.slot == RING[0] { RING[1] } else { RING[0] };
                Some(model.feedback_step(st, Feed::Slot(cur.slot), out, true)?)
            } else {
                None
            };
            let tok = model.feedback_read(cur)?;
            if next.is_some() {
                // the next step was committed with the token unread: it sits at position plen + generated
                let i = self.plen + self.generated;
                debug_assert_eq!(st.tokens.get(i), Some(&UNREAD));
                st.tokens[i] = tok;
            }
            self.tok = tok;
            if self.on_token()? {
                if next.is_some() {
                    // the stop came one step late: drop the held rest of the step (the caller rolls its position back)
                    model.feedback_discard()?;
                }
                return Ok(());
            }
            cur = match next {
                Some(p) => model.feedback_release(p)?,
                // not reached (on_token ends the request at max_new); stay correct anyway
                None => model.feedback_step(st, Feed::Host(self.tok), RING[0], false)?,
            };
        }
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
                spec_cut(model, st, want)?;
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
/// `generated` receives the number of tokens generated before a failure (left at 0 on success).
fn generate_one(
    sh: &Shared,
    g: &mut Inner,
    req: GenRequest,
    tx: Sender<GenEvent>,
    generated: &mut usize,
) -> Result<Outcome> {
    let Inner { model, state: st } = g;
    let mut seq = Seq::new(sh, model, st, req, tx)?;
    let r = run_seq(&mut seq, model, st);
    // entries captured on the device during this request go to their tier now (a no-op when already settled)
    seq.settle_store(model);
    match r {
        Ok(Some(o)) => Ok(o),
        Ok(None) => {
            let n = seq.generated;
            let res = seq.finish(model, st);
            if res.is_err() {
                *generated = n;
            }
            res
        }
        Err(e) => {
            *generated = seq.generated;
            Err(e)
        }
    }
}

/// Prefill and decode until the request stops: `Some` when it finished during prefill, `None` when it is ready for
/// [`Seq::finish`].
fn run_seq(seq: &mut Seq, model: &mut Model, st: &mut SequenceState) -> Result<Option<Outcome>> {
    loop {
        match seq.prefill_step(model, st)? {
            PrefillStep::Chunk => {}
            PrefillStep::Started => break,
            PrefillStep::Finished(o) => return Ok(Some(o)),
        }
    }
    let mut first = true;
    loop {
        let done = seq.on_token()?;
        if first {
            // after the first token went out: the prefix store's device-to-host copies stay off the time to first token
            seq.settle_store(model);
            first = false;
        }
        if done {
            break;
        }
        if seq.feedback_ok(model, st) {
            seq.run_feedback(model, st)?;
            break;
        }
        seq.advance(model, st)?;
    }
    Ok(None)
}
