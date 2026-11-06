//! Shared plumbing: app state, errors, the engine bridge, SSE helper, sampling parameters.

use crate::health::Progress;
use crate::metrics::Metrics;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use futures_util::Stream;
use intarsia_core::engine::{GenEvent, GenRequest};
use intarsia_core::Engine;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver};

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<dyn Engine>,
    pub model_name: String,
    pub metrics: Arc<Metrics>,
    pub created: u64,
    /// Requests that may wait behind the running ones (`max_running` at a time, see `ServerConfig`).
    pub max_queue: usize,
    /// Requests the engine runs at once (`--max-batch`).
    pub max_running: usize,
    /// `max_tokens` when the client sends none (0 = as much as the context allows).
    pub default_max_tokens: usize,
    /// Sampling for the fields a request leaves out (the checkpoint's `generation_config.json`, see [`SamplingDefaults`]).
    pub sampling_defaults: Arc<SamplingDefaults>,
    pub progress: Arc<Progress>,
    /// Bounded `source` metric label of the request this clone serves (see [`AppState::for_request`]).
    pub source: Arc<str>,
    /// Structured output's token table and parser factory, built on the first constrained request.
    pub grammar: Arc<std::sync::OnceLock<Result<Arc<crate::structured::GrammarEnv>, String>>>,
    /// Flips to true when the server is shutting down: the streams that never end on their own (`/events`) end then, so
    /// the graceful shutdown that waits for every open response can finish ([`crate::Shutdown`]).
    pub shutdown: Arc<tokio::sync::watch::Sender<bool>>,
    inflight: Arc<AtomicUsize>,
    counter: Arc<AtomicU64>,
    salt: u64,
}

/// One admitted request; releases its queue place when dropped.
struct Slot(Arc<AtomicUsize>);

impl Drop for Slot {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Cancels the engine request when the HTTP side goes away (response dropped).
struct CancelGuard(Arc<AtomicBool>);

impl Drop for CancelGuard {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

impl AppState {
    pub fn new(engine: Arc<dyn Engine>, model_name: String, cfg: &crate::ServerConfig) -> AppState {
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
        AppState {
            engine,
            model_name,
            metrics: Arc::new(Metrics::new()),
            created: now.as_secs(),
            max_queue: cfg.max_queue,
            max_running: cfg.max_running.max(1),
            default_max_tokens: cfg.default_max_tokens,
            sampling_defaults: Arc::new(cfg.sampling_defaults.clone()),
            progress: Arc::new(Progress::new(Duration::from_secs_f64(cfg.watchdog_s.max(0.0)))),
            source: Arc::from("unknown"),
            grammar: Arc::new(std::sync::OnceLock::new()),
            shutdown: Arc::new(tokio::sync::watch::channel(false).0),
            inflight: Arc::new(AtomicUsize::new(0)),
            counter: Arc::new(AtomicU64::new(0)),
            salt: now.as_nanos() as u64 & 0xffff_ffff,
        }
    }

    /// This state tagged with the client named by the request headers: `X-Intarsia-Client`, else the product token of
    /// `User-Agent` (`curl/8.4` -> `curl`). Cardinality is bounded, see [`Metrics::source_label`].
    pub fn for_request(mut self, headers: &axum::http::HeaderMap) -> AppState {
        let get = |n: &str| headers.get(n).and_then(|v| v.to_str().ok()).map(str::trim).filter(|s| !s.is_empty());
        let raw = get("x-intarsia-client").or_else(|| get("user-agent").map(|ua| ua.split(['/', ' ']).next().unwrap_or(ua)));
        self.source = Arc::from(self.metrics.source_label(raw));
        self
    }

    fn acquire(&self) -> Option<Slot> {
        let limit = self.max_queue.saturating_add(self.max_running);
        let mut n = self.inflight.load(Ordering::Acquire);
        loop {
            if n >= limit {
                return None;
            }
            match self.inflight.compare_exchange_weak(n, n + 1, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return Some(Slot(self.inflight.clone())),
                Err(cur) => n = cur,
            }
        }
    }

    /// Unique request id such as `chatcmpl-0000a1b2c3d4-1`.
    pub fn new_id(&self, prefix: &str) -> String {
        format!("{prefix}-{:08x}{:04x}", self.salt, self.counter.fetch_add(1, Ordering::Relaxed))
    }
}

// ---------------------------------------------------------------------------
// errors
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Flavor {
    OpenAi,
    Anthropic,
}

#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub flavor: Flavor,
    pub message: String,
    /// Seconds for the `Retry-After` header (429 / 503).
    pub retry_after: Option<u32>,
}

impl ApiError {
    pub fn new(status: StatusCode, flavor: Flavor, msg: impl std::fmt::Display) -> ApiError {
        ApiError { status, flavor, message: msg.to_string(), retry_after: None }
    }
    pub fn bad_request(flavor: Flavor, msg: impl std::fmt::Display) -> ApiError {
        ApiError::new(StatusCode::BAD_REQUEST, flavor, msg)
    }
    pub fn internal(flavor: Flavor, msg: impl std::fmt::Display) -> ApiError {
        ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, flavor, msg)
    }
    pub fn not_found(flavor: Flavor, msg: impl std::fmt::Display) -> ApiError {
        ApiError::new(StatusCode::NOT_FOUND, flavor, msg)
    }
    pub fn unavailable(flavor: Flavor, msg: impl std::fmt::Display) -> ApiError {
        ApiError::new(StatusCode::SERVICE_UNAVAILABLE, flavor, msg)
    }
    pub fn too_many_requests(flavor: Flavor, msg: impl std::fmt::Display) -> ApiError {
        ApiError { retry_after: Some(1), ..ApiError::new(StatusCode::TOO_MANY_REQUESTS, flavor, msg) }
    }
    pub fn body(&self) -> Value {
        match self.flavor {
            Flavor::OpenAi => json!({"error": {
                "message": self.message,
                "type": match self.status {
                    StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
                    s if s.is_client_error() => "invalid_request_error",
                    _ => "server_error",
                },
                "param": null,
                "code": self.status.as_u16(),
            }}),
            Flavor::Anthropic => json!({"type": "error", "error": {
                "type": match self.status {
                    StatusCode::NOT_FOUND => "not_found_error",
                    StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
                    StatusCode::SERVICE_UNAVAILABLE => "overloaded_error",
                    s if s.is_client_error() => "invalid_request_error",
                    _ => "api_error",
                },
                "message": self.message,
            }}),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let retry = self.retry_after;
        let mut r = (self.status, Json(self.body())).into_response();
        if let Some(s) = retry {
            r.headers_mut().insert(axum::http::header::RETRY_AFTER, axum::http::HeaderValue::from(s));
        }
        r
    }
}

// ---------------------------------------------------------------------------
// engine bridge
// ---------------------------------------------------------------------------

/// Run a blocking engine call (tokenizer, template, telemetry) off the async threads.
pub async fn blocking<T, F>(engine: &Arc<dyn Engine>, f: F) -> Result<T, ApiError>
where
    T: Send + 'static,
    F: FnOnce(&dyn Engine) -> intarsia_core::Result<T> + Send + 'static,
{
    let e = engine.clone();
    match tokio::task::spawn_blocking(move || f(&*e)).await {
        Ok(Ok(v)) => Ok(v),
        // Tokenizer / template failures are caused by the request content.
        Ok(Err(err)) => Err(ApiError::bad_request(Flavor::OpenAi, err)),
        Err(join) => Err(ApiError::internal(Flavor::OpenAi, format!("worker panicked: {join}"))),
    }
}

/// Like [`blocking`] but errors carry the given flavor.
pub async fn blocking_as<T, F>(engine: &Arc<dyn Engine>, flavor: Flavor, f: F) -> Result<T, ApiError>
where
    T: Send + 'static,
    F: FnOnce(&dyn Engine) -> intarsia_core::Result<T> + Send + 'static,
{
    blocking(engine, f).await.map_err(|mut e| {
        e.flavor = flavor;
        e
    })
}

/// Events of one admitted request. Dropping it (the HTTP response went away) sets the
/// request's cancel flag, which the engine checks between prefill chunks and decode steps.
pub struct GenRx {
    rx: UnboundedReceiver<GenEvent>,
    _cancel: CancelGuard,
}

impl GenRx {
    pub async fn recv(&mut self) -> Option<GenEvent> {
        self.rx.recv().await
    }
}

/// Admit a request (429 when the queue is full, 503 when the engine has failed), submit it to
/// the engine and forward its events.
pub fn start(st: &AppState, flavor: Flavor, mut req: GenRequest) -> Result<GenRx, ApiError> {
    if let Some(why) = st.engine.failed() {
        return Err(ApiError::unavailable(flavor, format!("engine failed: {why}")));
    }
    let Some(slot) = st.acquire() else {
        st.metrics.rejected.fetch_add(1, Ordering::Relaxed);
        return Err(ApiError::too_many_requests(
            flavor,
            format!("the request queue is full ({} waiting); retry shortly", st.max_queue),
        ));
    };
    let cancel = Arc::new(AtomicBool::new(false));
    req.cancel = Some(cancel.clone());
    let rx = st.engine.submit(req).map_err(|e| {
        let status = match e {
            intarsia_core::Error::BadRequest(_) => StatusCode::BAD_REQUEST,
            // the engine's own queue is full: the same answer as the server's limit
            intarsia_core::Error::QueueFull(n) => {
                st.metrics.rejected.fetch_add(1, Ordering::Relaxed);
                return ApiError::too_many_requests(flavor, format!("the request queue is full ({n} waiting); retry shortly"));
            }
            _ => StatusCode::SERVICE_UNAVAILABLE,
        };
        ApiError::new(status, flavor, e)
    })?;
    let (tx, out) = unbounded_channel();
    let metrics = st.metrics.clone();
    let source = st.source.clone();
    metrics.observe_admitted(&source);
    std::thread::Builder::new()
        .name("intarsia-bridge".into())
        .spawn(move || {
            let _slot = slot;
            for ev in rx.iter() {
                let last = matches!(ev, GenEvent::Done { .. } | GenEvent::Error { .. });
                if let GenEvent::Done { prompt_tokens, completion_tokens, prefill_ms, decode_ms, .. } = &ev {
                    metrics.observe_done(&source, *prompt_tokens, *completion_tokens, *prefill_ms, *decode_ms);
                }
                // Client gone: stop reading, which drops `rx` and cancels the generation.
                if tx.send(ev).is_err() || last {
                    break;
                }
            }
        })
        .map_err(|e| ApiError::internal(flavor, format!("spawn bridge: {e}")))?;
    Ok(GenRx { rx: out, _cancel: CancelGuard(cancel) })
}

/// Final result of a non-streaming generation.
pub struct Collected {
    pub text: String,
    pub finish: intarsia_core::engine::FinishReason,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub cached_tokens: usize,
    pub prefill_chunk: Option<usize>,
    pub stop_sequence: Option<String>,
    /// Generated token ids (the stop/EOS token is not included), see `real_id_count`.
    pub ids: Vec<u32>,
}

/// How many of the `Token` events of a generation carry a real token. The engine counts the stop/EOS
/// token in `completion_tokens` without emitting it, and may append one extra `Token` that repeats the
/// last id to release held-back text; that tail always comes last, so truncating to this count drops it.
pub fn real_id_count(finish: intarsia_core::engine::FinishReason, string_stop: bool, completion_tokens: usize) -> usize {
    let stop_id = finish == intarsia_core::engine::FinishReason::Stop && !string_stop;
    completion_tokens - usize::from(stop_id && completion_tokens > 0)
}

/// Streaming side of `return_token_ids`: holds back the newest `Token` so the engine's text-only tail
/// event (which repeats the last id) can be told apart once `Done` arrives.
#[derive(Default)]
pub struct IdHold {
    held: Option<(u32, String)>,
    seen: usize,
}

impl IdHold {
    /// Take a token; returns the previously held one, which is certainly a real token.
    pub fn push(&mut self, id: u32, text: String) -> Option<(u32, String)> {
        self.seen += 1;
        self.held.replace((id, text))
    }
    /// At `Done`: the held event, with its id unless it is the text-only tail.
    pub fn finish(
        &mut self,
        finish: intarsia_core::engine::FinishReason,
        string_stop: bool,
        completion_tokens: usize,
    ) -> Option<(Option<u32>, String)> {
        let (id, text) = self.held.take()?;
        let tail = self.seen > real_id_count(finish, string_stop, completion_tokens);
        Some(((!tail).then_some(id), text))
    }
}

pub async fn collect(mut rx: GenRx, flavor: Flavor) -> Result<Collected, ApiError> {
    let mut text = String::new();
    let mut ids = Vec::new();
    while let Some(ev) = rx.recv().await {
        match ev {
            GenEvent::Token { id, text: t } => {
                text.push_str(&t);
                ids.push(id);
            }
            GenEvent::Done {
                finish_reason, prompt_tokens, completion_tokens, cached_tokens, stop_sequence, prefill_chunk, ..
            } => {
                ids.truncate(real_id_count(finish_reason, stop_sequence.is_some(), completion_tokens));
                return Ok(Collected {
                    ids,
                    text,
                    finish: finish_reason,
                    prompt_tokens,
                    completion_tokens,
                    cached_tokens,
                    prefill_chunk,
                    stop_sequence,
                })
            }
            GenEvent::Error { message } => return Err(ApiError::internal(flavor, message)),
        }
    }
    Err(ApiError::internal(flavor, "engine stopped before finishing the request"))
}

/// Turn engine events into an SSE response. `prelude` is sent first; `f` maps
/// each event (and `None` once the channel closes) to zero or more SSE events.
pub fn sse_from<F>(
    rx: GenRx,
    prelude: Vec<Event>,
    f: F,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>>
where
    F: FnMut(Option<GenEvent>) -> Vec<Event> + Send + 'static,
{
    let init = (rx, VecDeque::from(prelude), f, false);
    let stream = futures_util::stream::unfold(init, |(mut rx, mut q, mut f, mut fin)| async move {
        loop {
            if let Some(e) = q.pop_front() {
                return Some((Ok(e), (rx, q, f, fin)));
            }
            if fin {
                return None;
            }
            match rx.recv().await {
                Some(ev) => q.extend(f(Some(ev))),
                None => {
                    q.extend(f(None));
                    fin = true;
                }
            }
        }
    });
    Sse::new(stream).keep_alive(KeepAlive::default())
}

// ---------------------------------------------------------------------------
// request helpers
// ---------------------------------------------------------------------------

/// Sampling / limit fields shared by all the endpoints.
#[derive(Default)]
pub struct Sampling {
    pub max_tokens: Option<usize>,
    pub min_tokens: usize,
    pub ignore_eos: bool,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i64>,
    pub seed: Option<u64>,
    pub stop: Vec<String>,
    pub stop_ids: Vec<u32>,
}

/// What a request that leaves a sampling field out gets: the checkpoint's own `generation_config.json`, as vLLM does with
/// `--generation-config auto` (the default there). A field the request sends always wins; a field the file does not have
/// falls back to the built-in (temperature 1.0, `top_p` 1.0, `top_k` 0 = off). `SamplingDefaults::default()` is the built-ins
/// alone (`--generation-config none`). `min_p` and `repetition_penalty` are not supported by the engine and are not applied.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SamplingDefaults {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    /// 0 = off.
    pub top_k: Option<usize>,
    /// The file these came from (`None`: the built-ins only).
    pub source: Option<String>,
}

impl SamplingDefaults {
    /// Read `<dir>/generation_config.json`. A missing file gives the built-ins; the second value lists what was ignored or wrong.
    pub fn from_dir(dir: &std::path::Path) -> (SamplingDefaults, Vec<String>) {
        let path = dir.join("generation_config.json");
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return (SamplingDefaults::default(), Vec::new()),
            Err(e) => return (SamplingDefaults::default(), vec![format!("{}: {e}", path.display())]),
        };
        match serde_json::from_str::<Value>(&text) {
            Ok(v) => {
                let (mut d, warns) = SamplingDefaults::from_json(&v);
                d.source = Some(path.display().to_string());
                (d, warns)
            }
            Err(e) => (SamplingDefaults::default(), vec![format!("{}: {e}", path.display())]),
        }
    }

    /// The sampling fields of a `generation_config.json` document. A value out of the range the API accepts is dropped, with a warning.
    pub fn from_json(v: &Value) -> (SamplingDefaults, Vec<String>) {
        let mut d = SamplingDefaults::default();
        let mut warns = Vec::new();
        if let Some(t) = v.get("temperature").and_then(Value::as_f64) {
            if (0.0..=10.0).contains(&t) {
                d.temperature = Some(t as f32);
            } else {
                warns.push(format!("temperature {t} outside [0, 10], ignored"));
            }
        }
        if let Some(p) = v.get("top_p").and_then(Value::as_f64) {
            if p > 0.0 && p <= 1.0 {
                d.top_p = Some(p as f32);
            } else {
                warns.push(format!("top_p {p} outside (0, 1], ignored"));
            }
        }
        if let Some(k) = v.get("top_k").and_then(Value::as_i64) {
            d.top_k = Some(k.max(0) as usize);
        }
        if v.get("min_p").and_then(Value::as_f64).is_some_and(|m| m > 0.0) {
            warns.push("min_p is not supported, ignored".into());
        }
        if v.get("repetition_penalty").and_then(Value::as_f64).is_some_and(|r| r != 1.0) {
            warns.push("repetition_penalty is not supported, ignored".into());
        }
        (d, warns)
    }

    /// The values a request that names nothing is sampled with.
    pub fn effective(&self) -> (f32, f32, usize) {
        (self.temperature.unwrap_or(1.0), self.top_p.unwrap_or(1.0), self.top_k.unwrap_or(0))
    }

    /// One line for the startup log.
    pub fn describe(&self) -> String {
        let (t, p, k) = self.effective();
        match &self.source {
            Some(s) => format!("temperature={t} top_p={p} top_k={k} (from {s}; a request's own values win)"),
            None => format!("temperature={t} top_p={p} top_k={k} (built-in; no checkpoint generation_config applied)"),
        }
    }

    /// The `intarsia_sampling_defaults` metric line (value 1), labelled with the effective values.
    pub fn metric(&self) -> String {
        let (t, p, k) = self.effective();
        let src = self.source.as_deref().unwrap_or("builtin").replace('\\', "\\\\").replace('"', "\\\"");
        format!(
            "# HELP intarsia_sampling_defaults Sampling a request without its own values gets (the checkpoint's generation_config.json, or the built-ins).\n# TYPE intarsia_sampling_defaults gauge\nintarsia_sampling_defaults{{source=\"{src}\",temperature=\"{t}\",top_p=\"{p}\",top_k=\"{k}\"}} 1\n"
        )
    }
}

impl Sampling {
    pub fn into_request(
        self,
        prompt: Vec<u32>,
        flavor: Flavor,
        context_len: usize,
        default_max_tokens: usize,
        defaults: &SamplingDefaults,
    ) -> Result<GenRequest, ApiError> {
        let bad = |m: &str| ApiError::bad_request(flavor, m);
        let temperature = self.temperature.or(defaults.temperature).unwrap_or(1.0);
        if !(temperature >= 0.0) || temperature > 10.0 {
            return Err(bad("temperature must be in [0, 10]"));
        }
        let top_p = self.top_p.or(defaults.top_p).unwrap_or(1.0);
        if !(top_p > 0.0 && top_p <= 1.0) {
            return Err(bad("top_p must be in (0, 1]"));
        }
        if self.max_tokens == Some(0) {
            return Err(bad("max_tokens must be at least 1"));
        }
        if prompt.is_empty() {
            return Err(bad("the prompt is empty"));
        }
        if prompt.len() >= context_len {
            return Err(bad(&format!(
                "the prompt has {} tokens but the context window is {} tokens",
                prompt.len(),
                context_len
            )));
        }
        let prompt_len = prompt.len();
        // a `top_k` the request sends wins, 0 and negative (off) included
        let top_k = match self.top_k {
            Some(k) => k.max(0) as usize,
            None => defaults.top_k.unwrap_or(0),
        };
        Ok(GenRequest {
            prompt,
            // Default: what is left of the context window, bounded by `default_max_tokens`
            // (0 = no extra bound) so one client cannot hold the worker for minutes.
            max_tokens: self.max_tokens.unwrap_or_else(|| {
                let left = context_len.saturating_sub(prompt_len).max(1);
                if default_max_tokens == 0 { left } else { left.min(default_max_tokens) }
            }),
            min_tokens: self.min_tokens,
            ignore_eos: self.ignore_eos,
            temperature,
            top_p,
            top_k,
            seed: self.seed,
            stop_ids: self.stop_ids,
            stop: self.stop,
            cancel: None,
            trace_logits: None,
            constraint: None,
            prewarm: None,
        })
    }
}

/// `stop` as a string, an array of strings or null.
pub fn parse_stop(v: &Option<Value>, flavor: Flavor) -> Result<Vec<String>, ApiError> {
    match v {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::String(s)) => Ok(vec![s.clone()]),
        Some(Value::Array(a)) => a
            .iter()
            .map(|x| x.as_str().map(String::from).ok_or_else(|| ApiError::bad_request(flavor, "stop must be strings")))
            .collect(),
        _ => Err(ApiError::bad_request(flavor, "stop must be a string or an array of strings")),
    }
}

/// Flatten message `content` to a plain string: the chat templates call string
/// methods on it. Text parts are concatenated; other part types are rejected.
pub fn flatten_content(v: &Value, flavor: Flavor) -> Result<String, ApiError> {
    match v {
        Value::Null => Ok(String::new()),
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                match p {
                    Value::String(s) => out.push_str(s),
                    Value::Object(o) => match o.get("type").and_then(Value::as_str) {
                        Some("text") | Some("input_text") | None => {
                            out.push_str(o.get("text").and_then(Value::as_str).unwrap_or(""))
                        }
                        Some(other) => {
                            return Err(ApiError::bad_request(
                                flavor,
                                format!("unsupported content part type '{other}' (text only)"),
                            ))
                        }
                    },
                    _ => return Err(ApiError::bad_request(flavor, "invalid content part")),
                }
            }
            Ok(out)
        }
        _ => Err(ApiError::bad_request(flavor, "content must be a string or an array of parts")),
    }
}

pub fn sse_json(name: Option<&str>, v: &Value) -> Event {
    // `event:` line first, as the Anthropic wire format shows it.
    let e = match name {
        Some(n) => Event::default().event(n),
        None => Event::default(),
    };
    e.data(v.to_string())
}

#[cfg(test)]
mod sampling_defaults_tests {
    use super::*;

    fn qwen() -> SamplingDefaults {
        SamplingDefaults::from_json(&json!({"temperature": 1.0, "top_k": 20, "top_p": 0.95, "do_sample": true})).0
    }

    fn req(s: Sampling, d: &SamplingDefaults) -> GenRequest {
        s.into_request(vec![1, 2, 3], Flavor::OpenAi, 64, 0, d).unwrap()
    }

    #[test]
    fn checkpoint_beats_built_in() {
        let r = req(Sampling::default(), &qwen());
        assert_eq!((r.temperature, r.top_p, r.top_k), (1.0, 0.95, 20));
        let r = req(Sampling::default(), &SamplingDefaults::from_json(&json!({"temperature": 0.6, "top_k": 20, "top_p": 0.95})).0);
        assert_eq!(r.temperature, 0.6);
    }

    #[test]
    fn built_in_without_a_file_or_a_field() {
        let r = req(Sampling::default(), &SamplingDefaults::default());
        assert_eq!((r.temperature, r.top_p, r.top_k), (1.0, 1.0, 0));
        // a file with only some fields leaves the others at the built-in
        let r = req(Sampling::default(), &SamplingDefaults::from_json(&json!({"top_k": 20})).0);
        assert_eq!((r.temperature, r.top_p, r.top_k), (1.0, 1.0, 20));
    }

    #[test]
    fn request_beats_checkpoint() {
        let s = Sampling { temperature: Some(0.0), top_p: Some(1.0), top_k: Some(0), ..Default::default() };
        let r = req(s, &qwen());
        assert_eq!((r.temperature, r.top_p, r.top_k), (0.0, 1.0, 0), "greedy, top_p 1 and top_k off sent by the request stay");
        let s = Sampling { temperature: Some(0.3), top_k: Some(-1), ..Default::default() };
        let r = req(s, &qwen());
        assert_eq!((r.temperature, r.top_p, r.top_k), (0.3, 0.95, 0), "top_k -1 is off, and the unsent top_p is the checkpoint's");
        let s = Sampling { top_k: Some(5), ..Default::default() };
        assert_eq!(req(s, &qwen()).top_k, 5);
    }

    #[test]
    fn off_switch_is_todays_behaviour() {
        // `--generation-config none` is `SamplingDefaults::default()`: what into_request gave before the defaults existed
        let r = req(Sampling::default(), &SamplingDefaults::default());
        assert_eq!((r.temperature, r.top_p, r.top_k), (1.0, 1.0, 0));
        assert!(SamplingDefaults::default().source.is_none());
    }

    #[test]
    fn bad_and_unsupported_values_are_dropped_with_a_warning() {
        let (d, w) = SamplingDefaults::from_json(&json!({"temperature": 11.0, "top_p": 0.0, "top_k": -1, "min_p": 0.05, "repetition_penalty": 1.1}));
        assert_eq!((d.temperature, d.top_p, d.top_k), (None, None, Some(0)));
        assert_eq!(w.len(), 4, "{w:?}");
        let (d, w) = SamplingDefaults::from_json(&json!({"min_p": 0.0, "repetition_penalty": 1.0}));
        assert_eq!(d, SamplingDefaults::default());
        assert!(w.is_empty());
    }

    #[test]
    fn missing_file_is_the_built_in_and_a_present_one_is_read() {
        let dir = std::env::temp_dir().join(format!("intarsia-gencfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (d, w) = SamplingDefaults::from_dir(&dir);
        assert_eq!(d, SamplingDefaults::default());
        assert!(w.is_empty());
        std::fs::write(dir.join("generation_config.json"), r#"{"temperature": 0.6, "top_k": 20, "top_p": 0.95}"#).unwrap();
        let (d, w) = SamplingDefaults::from_dir(&dir);
        assert!(w.is_empty());
        assert_eq!(d.effective(), (0.6, 0.95, 20));
        assert!(d.source.as_deref().unwrap().ends_with("generation_config.json"));
        assert!(d.describe().contains("top_k=20") && d.metric().contains("top_k=\"20\""));
        std::fs::write(dir.join("generation_config.json"), "{not json").unwrap();
        let (d, w) = SamplingDefaults::from_dir(&dir);
        assert_eq!(d, SamplingDefaults::default());
        assert_eq!(w.len(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }
}
