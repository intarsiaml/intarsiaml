//! OpenAI-compatible endpoints.

use crate::api::*;
use crate::structured::{self, Part, Piece, ReasoningParser, ToolParser};
use axum::extract::{Path, State};
use axum::response::sse::Event;
use axum::response::{IntoResponse, Json, Response};
use intarsia_core::engine::{ChatOpts, FinishReason, GenEvent};
use serde::Deserialize;
use serde_json::{json, Map, Value};

const F: Flavor = Flavor::OpenAi;

#[derive(Deserialize, Default)]
pub struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

#[derive(Deserialize, Default)]
#[serde(default)]
pub struct CommonParams {
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
    min_tokens: Option<usize>,
    ignore_eos: bool,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i64>,
    seed: Option<u64>,
    stop: Option<Value>,
    stop_token_ids: Vec<u32>,
    n: Option<u32>,
    stream: bool,
    stream_options: Option<StreamOptions>,
    /// intarsia extension (named as in vLLM): return the generated token ids as `token_ids`
    /// (choice when not streaming, delta/choice per chunk when streaming). Off by default.
    return_token_ids: bool,
}

impl CommonParams {
    fn sampling(&self) -> Result<Sampling, ApiError> {
        if self.n.unwrap_or(1) != 1 {
            return Err(ApiError::bad_request(F, "only n=1 is supported"));
        }
        Ok(Sampling {
            max_tokens: self.max_completion_tokens.or(self.max_tokens),
            min_tokens: self.min_tokens.unwrap_or(0),
            ignore_eos: self.ignore_eos,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            seed: self.seed,
            stop: parse_stop(&self.stop, F)?,
            stop_ids: self.stop_token_ids.clone(),
        })
    }
    fn include_usage(&self) -> bool {
        self.stream_options.as_ref().map(|o| o.include_usage).unwrap_or(false)
    }
}

/// `return_token_ids` (the opt-in intarsia extension) also reports the prefill chunk size the prompt was processed
/// with under `prompt_tokens_details.prefill_chunk`: with continuous batching it depends on what else was decoding
/// (the concurrent-identity harness picks the matching solo reference by it).
fn with_chunk(mut u: Value, rid: bool, chunk: Option<usize>) -> Value {
    if let (true, Some(c)) = (rid, chunk) {
        u["prompt_tokens_details"]["prefill_chunk"] = json!(c);
    }
    u
}

fn usage(prompt: usize, completion: usize, cached: usize) -> Value {
    json!({
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": cached},
    })
}

fn finish_str(f: FinishReason) -> &'static str {
    match f {
        FinishReason::Length => "length",
        // `cancelled` never reaches a live client; report it as a normal stop.
        _ => "stop",
    }
}

fn error_event(message: &str) -> Event {
    sse_json(None, &json!({"error": {"message": message, "type": "server_error", "param": null, "code": 500}}))
}

// ---------------------------------------------------------------------------
// /v1/chat/completions
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct ChatReq {
    messages: Vec<Value>,
    #[serde(default)]
    tools: Option<Value>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    parallel_tool_calls: Option<bool>,
    #[serde(default)]
    response_format: Option<Value>,
    #[serde(default)]
    chat_template_kwargs: Option<Map<String, Value>>,
    #[serde(flatten)]
    common: CommonParams,
}

fn tool_call(id: String, name: &str, arguments: &Value) -> Value {
    json!({"id": id, "type": "function", "function": {"name": name, "arguments": arguments.to_string()}})
}

/// Streaming deltas for parsed output pieces; `n` counts the tool calls so far (their `index`).
fn tool_deltas(st: &AppState, pieces: Vec<Piece>, n: &mut usize) -> Vec<Value> {
    let mut out = Vec::new();
    for p in pieces {
        match p {
            Piece::Text(t) if t.is_empty() => {}
            Piece::Text(t) => out.push(json!({"content": t})),
            Piece::Call { name, arguments } => {
                let mut c = tool_call(st.new_id("call"), &name, &arguments);
                c["index"] = json!(*n);
                *n += 1;
                out.push(json!({"tool_calls": [c]}));
            }
        }
    }
    out
}

/// Streaming deltas for output split by the reasoning parser: thinking goes to `reasoning_content`, the answer to
/// `content` (or, when tool calls are parsed, through the tool parser).
fn think_deltas(st: &AppState, parts: Vec<Part>, tp: &mut Option<ToolParser>, n: &mut usize) -> Vec<Value> {
    let mut out = Vec::new();
    for p in parts {
        match (p, tp.as_mut()) {
            (Part::Reasoning(t), _) => out.push(json!({"reasoning_content": t})),
            (Part::Content(t), Some(tp)) => out.extend(tool_deltas(st, tp.push(&t), n)),
            (Part::Content(t), None) => out.push(json!({"content": t})),
        }
    }
    out
}

pub async fn chat_completions(
    State(st): State<AppState>,
    headers: axum::http::HeaderMap,
    body: Result<Json<ChatReq>, axum::extract::rejection::JsonRejection>,
) -> Response {
    match chat_inner(st.for_request(&headers), body).await {
        Ok(r) => r,
        Err(e) => e.into_response(),
    }
}

async fn chat_inner(
    st: AppState,
    body: Result<Json<ChatReq>, axum::extract::rejection::JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(req) = body.map_err(|e| ApiError::bad_request(F, e.body_text()))?;
    let info = st.engine.info();
    if !info.has_chat_template {
        return Err(ApiError::bad_request(F, "this model has no chat template; use /v1/completions"));
    }
    if req.messages.is_empty() {
        return Err(ApiError::bad_request(F, "messages must not be empty"));
    }
    let sampling = req.common.sampling()?;
    let bad = |m: String| ApiError::bad_request(F, m);
    let tools = structured::openai_tools(&req.tools).map_err(bad)?;
    let choice = structured::openai_choice(&req.tool_choice, &tools).map_err(bad)?;
    let format = structured::response_format(&req.response_format).map_err(bad)?;
    let hermes = info.tool_parser.as_deref() == Some("hermes");
    let parallel = req.parallel_tool_calls.unwrap_or(true);
    let (plan, parse_tools) = structured::choose(format, &tools, &choice, parallel, hermes).map_err(bad)?;

    // Content-part arrays become plain strings (templates call string methods).
    let mut messages = Vec::with_capacity(req.messages.len());
    for m in &req.messages {
        let Value::Object(o) = m else { return Err(ApiError::bad_request(F, "each message must be an object")) };
        let mut o = o.clone();
        let content = flatten_content(o.get("content").unwrap_or(&Value::Null), F)?;
        o.insert("content".into(), Value::String(content));
        messages.push(Value::Object(o));
    }
    let mut opts = ChatOpts { add_generation_prompt: true, tools: req.tools.clone(), ..Default::default() };
    if let Some(kw) = &req.chat_template_kwargs {
        for (k, v) in kw {
            if k == "enable_thinking" {
                opts.enable_thinking = v.as_bool();
            } else {
                opts.extra.insert(k.clone(), v.clone());
            }
        }
    }
    // Reasoning is parsed (and a constraint waits for `</think>`) only when the request turns thinking on and the model
    // has a reasoning parser; every other request is answered exactly as before.
    let think = opts.enable_thinking == Some(true) && info.reasoning_parser.is_some();
    let prompt = blocking_as(&st.engine, F, move |e| {
        let text = e.apply_chat_template(&messages, &opts)?;
        e.tokenize(&text, false)
    })
    .await?;
    let mut gen = sampling.into_request(prompt, F, info.context_len, st.default_max_tokens)?;
    structured::attach(&st, F, &mut gen, plan, think).await?;
    let rx = start(&st, F, gen)?;

    let id = st.new_id("chatcmpl");
    let created = now_secs();
    let model = st.model_name.clone();
    let chunk = move |delta: Value, finish: Option<&str>, usage_null: bool| {
        let mut v = json!({
            "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        });
        if usage_null {
            v["usage"] = Value::Null;
        }
        v
    };

    let rid = req.common.return_token_ids;
    if !req.common.stream {
        let c = collect(rx, F).await?;
        let mut message = json!({"role": "assistant", "content": c.text});
        let mut finish = finish_str(c.finish);
        let mut answer = c.text.clone();
        if think {
            let (reasoning, content) = ReasoningParser::parse_all(&c.text);
            message["reasoning_content"] = reasoning.map_or(Value::Null, Value::String);
            answer = content.unwrap_or_default();
            message["content"] = json!(answer);
        }
        if parse_tools {
            let (content, calls) = structured::split_calls(&answer);
            if !calls.is_empty() {
                message["content"] = content.map_or(Value::Null, Value::String);
                message["tool_calls"] = calls.iter().map(|(n, a)| tool_call(st.new_id("call"), n, a)).collect();
                if c.finish != FinishReason::Length {
                    finish = "tool_calls";
                }
            }
        }
        let v = json!({
            "id": chunk(json!({}), None, false)["id"].clone(), "object": "chat.completion", "created": created, "model": st.model_name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": with_chunk(usage(c.prompt_tokens, c.completion_tokens, c.cached_tokens), rid, c.prefill_chunk),
        });
        let mut v = v;
        if rid {
            v["choices"][0]["token_ids"] = json!(c.ids);
        }
        return Ok(Json(v).into_response());
    }

    let inc = req.common.include_usage();
    let first = sse_json(None, &chunk(json!({"role": "assistant", "content": ""}), None, inc));
    let mut done = false;
    let model2 = st.model_name.clone();
    let id2 = chunk(json!({}), None, false)["id"].clone();
    let mut hold = IdHold::default();
    let delta_of = |id: Option<u32>, text: String| match id {
        Some(id) => json!({"content": text, "token_ids": [id]}),
        None => json!({"content": text}),
    };
    // hermes tool calls are parsed out of the text (token ids are not reported per delta then)
    let mut tp = parse_tools.then(ToolParser::default);
    let mut rp = think.then(ReasoningParser::default);
    let mut ncalls = 0usize;
    let st2 = st.clone();
    let sse = sse_from(rx, vec![first], move |ev| match ev {
        Some(GenEvent::Token { id: tid, text }) => {
            if let Some(rp) = rp.as_mut() {
                think_deltas(&st2, rp.push(&text), &mut tp, &mut ncalls)
                    .into_iter()
                    .map(|d| sse_json(None, &chunk(d, None, inc)))
                    .collect()
            } else if let Some(tp) = tp.as_mut() {
                tool_deltas(&st2, tp.push(&text), &mut ncalls)
                    .into_iter()
                    .map(|d| sse_json(None, &chunk(d, None, inc)))
                    .collect()
            } else if rid {
                hold.push(tid, text)
                    .map(|(i, t)| sse_json(None, &chunk(delta_of(Some(i), t), None, inc)))
                    .into_iter()
                    .collect()
            } else if text.is_empty() {
                vec![]
            } else {
                vec![sse_json(None, &chunk(json!({"content": text}), None, inc))]
            }
        }
        Some(GenEvent::Done { finish_reason, prompt_tokens, completion_tokens, cached_tokens, stop_sequence, prefill_chunk, .. }) => {
            done = true;
            let mut out = Vec::new();
            if let Some(rp) = rp.as_mut() {
                let mut ds = think_deltas(&st2, rp.finish(), &mut tp, &mut ncalls);
                if let Some(tp) = tp.as_mut() {
                    ds.extend(tool_deltas(&st2, tp.finish(), &mut ncalls));
                }
                for d in ds {
                    out.push(sse_json(None, &chunk(d, None, inc)));
                }
            } else if let Some(tp) = tp.as_mut() {
                for d in tool_deltas(&st2, tp.finish(), &mut ncalls) {
                    out.push(sse_json(None, &chunk(d, None, inc)));
                }
            } else if let Some((i, t)) = hold.finish(finish_reason, stop_sequence.is_some(), completion_tokens) {
                out.push(sse_json(None, &chunk(delta_of(i, t), None, inc)));
            }
            let fr = if ncalls > 0 && finish_reason != FinishReason::Length { "tool_calls" } else { finish_str(finish_reason) };
            out.push(sse_json(None, &chunk(json!({}), Some(fr), inc)));
            if inc {
                out.push(sse_json(
                    None,
                    &json!({"id": id2, "object": "chat.completion.chunk", "created": created, "model": model2,
                            "choices": [], "usage": with_chunk(usage(prompt_tokens, completion_tokens, cached_tokens), rid, prefill_chunk)}),
                ));
            }
            out
        }
        Some(GenEvent::Error { message }) => {
            done = true;
            vec![error_event(&message)]
        }
        None => {
            let mut out = Vec::new();
            if !done {
                out.push(error_event("engine stopped before finishing the request"));
            }
            out.push(Event::default().data("[DONE]"));
            out
        }
    });
    Ok(sse.into_response())
}

// ---------------------------------------------------------------------------
// /v1/completions
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct CompletionReq {
    prompt: Value,
    #[serde(default)]
    echo: bool,
    #[serde(default)]
    response_format: Option<Value>,
    /// Not supported here (chat only); present so it is rejected instead of ignored.
    #[serde(default)]
    tools: Option<Value>,
    #[serde(flatten)]
    common: CommonParams,
}

pub async fn completions(
    State(st): State<AppState>,
    headers: axum::http::HeaderMap,
    body: Result<Json<CompletionReq>, axum::extract::rejection::JsonRejection>,
) -> Response {
    match completions_inner(st.for_request(&headers), body).await {
        Ok(r) => r,
        Err(e) => e.into_response(),
    }
}

enum PromptIn {
    Text(String),
    Ids(Vec<u32>),
}

fn parse_prompt(v: &Value) -> Result<PromptIn, ApiError> {
    let ids = |a: &Vec<Value>| -> Option<Vec<u32>> { a.iter().map(|x| x.as_u64().map(|n| n as u32)).collect() };
    match v {
        Value::String(s) => Ok(PromptIn::Text(s.clone())),
        Value::Array(a) if a.is_empty() => Err(ApiError::bad_request(F, "prompt must not be empty")),
        Value::Array(a) if a.iter().all(Value::is_u64) => Ok(PromptIn::Ids(ids(a).unwrap_or_default())),
        Value::Array(a) if a.len() == 1 => match &a[0] {
            Value::String(s) => Ok(PromptIn::Text(s.clone())),
            Value::Array(inner) => ids(inner)
                .map(PromptIn::Ids)
                .ok_or_else(|| ApiError::bad_request(F, "prompt token arrays must hold integers")),
            _ => Err(ApiError::bad_request(F, "invalid prompt")),
        },
        Value::Array(_) => Err(ApiError::bad_request(F, "batched prompts are not supported (one prompt per request)")),
        _ => Err(ApiError::bad_request(F, "prompt must be a string, token ids, or a one-element array")),
    }
}

async fn completions_inner(
    st: AppState,
    body: Result<Json<CompletionReq>, axum::extract::rejection::JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(req) = body.map_err(|e| ApiError::bad_request(F, e.body_text()))?;
    let info = st.engine.info();
    let sampling = req.common.sampling()?;
    if req.tools.as_ref().is_some_and(|t| !t.is_null()) {
        return Err(ApiError::bad_request(F, "tools are supported on /v1/chat/completions only"));
    }
    let plan = structured::response_format(&req.response_format).map_err(|m| ApiError::bad_request(F, m))?;
    let pin = parse_prompt(&req.prompt)?;
    let (prompt, prefix) = match pin {
        PromptIn::Ids(ids) => (ids, String::new()),
        PromptIn::Text(t) => {
            let t2 = t.clone();
            let ids = blocking_as(&st.engine, F, move |e| e.tokenize(&t2, true)).await?;
            (ids, if req.echo { t } else { String::new() })
        }
    };
    let mut gen = sampling.into_request(prompt, F, info.context_len, st.default_max_tokens)?;
    structured::attach(&st, F, &mut gen, plan, false).await?;
    let rx = start(&st, F, gen)?;

    let id = st.new_id("cmpl");
    let created = now_secs();
    let model = st.model_name.clone();
    let chunk = move |text: &str, finish: Option<&str>, usage_null: bool| {
        let mut v = json!({
            "id": id, "object": "text_completion", "created": created, "model": model,
            "choices": [{"index": 0, "text": text, "logprobs": null, "finish_reason": finish}],
        });
        if usage_null {
            v["usage"] = Value::Null;
        }
        v
    };

    let rid = req.common.return_token_ids;
    if !req.common.stream {
        let c = collect(rx, F).await?;
        let mut v = chunk(&format!("{prefix}{}", c.text), Some(finish_str(c.finish)), false);
        if rid {
            v["choices"][0]["token_ids"] = json!(c.ids);
        }
        v["usage"] = with_chunk(usage(c.prompt_tokens, c.completion_tokens, c.cached_tokens), rid, c.prefill_chunk);
        return Ok(Json(v).into_response());
    }

    let inc = req.common.include_usage();
    let mut pre = Vec::new();
    if !prefix.is_empty() {
        pre.push(sse_json(None, &chunk(&prefix, None, inc)));
    }
    let mut done = false;
    let (model2, id2) = (st.model_name.clone(), chunk("", None, false)["id"].clone());
    let mut hold = IdHold::default();
    let chunk_of = move |chunk: &dyn Fn(&str, Option<&str>, bool) -> Value, id: Option<u32>, text: &str, inc: bool| {
        let mut v = chunk(text, None, inc);
        if let Some(id) = id {
            v["choices"][0]["token_ids"] = json!([id]);
        }
        v
    };
    let sse = sse_from(rx, pre, move |ev| match ev {
        Some(GenEvent::Token { id: tid, text }) => {
            if rid {
                hold.push(tid, text)
                    .map(|(i, t)| sse_json(None, &chunk_of(&chunk, Some(i), &t, inc)))
                    .into_iter()
                    .collect()
            } else if text.is_empty() {
                vec![]
            } else {
                vec![sse_json(None, &chunk(&text, None, inc))]
            }
        }
        Some(GenEvent::Done { finish_reason, prompt_tokens, completion_tokens, cached_tokens, stop_sequence, prefill_chunk, .. }) => {
            done = true;
            let mut out = Vec::new();
            if let Some((i, t)) = hold.finish(finish_reason, stop_sequence.is_some(), completion_tokens) {
                out.push(sse_json(None, &chunk_of(&chunk, i, &t, inc)));
            }
            out.push(sse_json(None, &chunk("", Some(finish_str(finish_reason)), inc)));
            if inc {
                out.push(sse_json(
                    None,
                    &json!({"id": id2, "object": "text_completion", "created": created, "model": model2,
                            "choices": [], "usage": with_chunk(usage(prompt_tokens, completion_tokens, cached_tokens), rid, prefill_chunk)}),
                ));
            }
            out
        }
        Some(GenEvent::Error { message }) => {
            done = true;
            vec![error_event(&message)]
        }
        None => {
            let mut out = Vec::new();
            if !done {
                out.push(error_event("engine stopped before finishing the request"));
            }
            out.push(Event::default().data("[DONE]"));
            out
        }
    });
    Ok(sse.into_response())
}

// ---------------------------------------------------------------------------
// /v1/models
// ---------------------------------------------------------------------------

fn model_obj(st: &AppState) -> Value {
    let info = st.engine.info();
    json!({"id": st.model_name, "object": "model", "created": st.created, "owned_by": "intarsia",
           "max_model_len": info.context_len, "root": info.id})
}

pub async fn models(State(st): State<AppState>) -> Json<Value> {
    Json(json!({"object": "list", "data": [model_obj(&st)]}))
}

pub async fn model(State(st): State<AppState>, Path(id): Path<String>) -> Response {
    if id == st.model_name {
        Json(model_obj(&st)).into_response()
    } else {
        ApiError::not_found(F, format!("The model '{id}' does not exist")).into_response()
    }
}

pub fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}
