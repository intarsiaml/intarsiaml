//! Anthropic Messages API: text, and client tools (`tool_use` / `tool_result`) with schema-constrained arguments.

use crate::api::*;
use crate::structured::{self, Part, Piece, ReasoningParser, Tool, ToolParser};
use axum::extract::State;
use axum::response::sse::Event;
use axum::response::{IntoResponse, Json, Response};
use intarsia_core::engine::{ChatOpts, FinishReason, GenEvent};
use serde::Deserialize;
use serde_json::{json, Value};

const F: Flavor = Flavor::Anthropic;

#[derive(Deserialize)]
pub struct MessagesReq {
    messages: Vec<Value>,
    #[serde(default)]
    system: Option<Value>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    stop_sequences: Vec<String>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<i64>,
    #[serde(default)]
    thinking: Option<Value>,
    #[serde(default)]
    tools: Option<Value>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    chat_template_kwargs: Option<serde_json::Map<String, Value>>,
}

/// Anthropic content (string or blocks) to a plain string. Only text blocks are
/// supported; `thinking` blocks from earlier turns are dropped.
fn blocks_to_text(v: &Value) -> Result<String, ApiError> {
    match v {
        Value::String(s) => Ok(s.clone()),
        Value::Array(blocks) => {
            let mut out = String::new();
            for b in blocks {
                match b.get("type").and_then(Value::as_str) {
                    Some("text") => out.push_str(b.get("text").and_then(Value::as_str).unwrap_or("")),
                    Some("thinking") | Some("redacted_thinking") => {}
                    other => {
                        return Err(ApiError::bad_request(
                            F,
                            format!("unsupported content block type '{}' (text only)", other.unwrap_or("?")),
                        ))
                    }
                }
            }
            Ok(out)
        }
        _ => Err(ApiError::bad_request(F, "content must be a string or an array of blocks")),
    }
}

/// Whether `v` (message content) holds `tool_use` / `tool_result` blocks.
fn has_tool_blocks(v: &Value) -> bool {
    v.as_array().is_some_and(|a| {
        a.iter().any(|b| matches!(b.get("type").and_then(Value::as_str), Some("tool_use") | Some("tool_result")))
    })
}

/// One Anthropic message with tool blocks, as chat-template messages: an assistant turn's `tool_use` blocks become
/// `tool_calls`; a user turn's `tool_result` blocks become `tool` messages (before the turn's text, if any).
fn tool_turn(role: &str, blocks: &[Value], out: &mut Vec<Value>) -> Result<(), ApiError> {
    let mut text = String::new();
    let mut calls = Vec::new();
    for b in blocks {
        match b.get("type").and_then(Value::as_str) {
            Some("tool_use") if role == "assistant" => {
                let name = b.get("name").and_then(Value::as_str).ok_or_else(|| ApiError::bad_request(F, "tool_use.name is required"))?;
                calls.push(json!({"id": b.get("id").cloned().unwrap_or(Value::Null), "type": "function",
                                  "function": {"name": name, "arguments": b.get("input").cloned().unwrap_or(json!({}))}}));
            }
            Some("tool_result") if role == "user" => {
                let content = match b.get("content") {
                    None | Some(Value::Null) => String::new(),
                    Some(c) => blocks_to_text(c)?,
                };
                out.push(json!({"role": "tool", "tool_call_id": b.get("tool_use_id").cloned().unwrap_or(Value::Null),
                                "content": content}));
            }
            Some(t @ ("tool_use" | "tool_result")) => {
                return Err(ApiError::bad_request(F, format!("a {t} block cannot be in a {role} message")))
            }
            _ => text.push_str(&blocks_to_text(&Value::Array(vec![b.clone()]))?),
        }
    }
    if role == "assistant" {
        let mut m = json!({"role": role, "content": text});
        if !calls.is_empty() {
            m["tool_calls"] = Value::Array(calls);
        }
        out.push(m);
    } else if !text.is_empty() {
        out.push(json!({"role": role, "content": text}));
    }
    Ok(())
}

fn to_chat_messages(system: &Option<Value>, messages: &[Value]) -> Result<Vec<Value>, ApiError> {
    let mut out = Vec::with_capacity(messages.len() + 1);
    if let Some(s) = system {
        let t = blocks_to_text(s)?;
        if !t.is_empty() {
            out.push(json!({"role": "system", "content": t}));
        }
    }
    for m in messages {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("");
        if role != "user" && role != "assistant" {
            return Err(ApiError::bad_request(F, format!("invalid message role '{role}'")));
        }
        let c = m.get("content").unwrap_or(&Value::Null);
        if has_tool_blocks(c) {
            tool_turn(role, c.as_array().map(Vec::as_slice).unwrap_or(&[]), &mut out)?;
            continue;
        }
        let content = blocks_to_text(c)?;
        out.push(json!({"role": role, "content": content}));
    }
    Ok(out)
}

fn tool_use_id(st: &AppState) -> String {
    st.new_id("toolu").replacen("toolu-", "toolu_", 1)
}

/// Streaming state of a response whose blocks are not known up front (tool calls parsed out of the text): text blocks
/// open lazily, each call is one `tool_use` block.
struct Blocks {
    next: usize,
    text_open: bool,
    think_open: bool,
    calls: usize,
}

impl Blocks {
    /// Output split by the reasoning parser: thinking goes to a `thinking` block, the answer through the tool parser
    /// (when tool calls are parsed) into text / `tool_use` blocks.
    fn parts(&mut self, st: &AppState, parts: Vec<Part>, tp: &mut Option<ToolParser>) -> Vec<Event> {
        let mut out = Vec::new();
        for p in parts {
            match (p, tp.as_mut()) {
                (Part::Reasoning(t), _) => out.extend(self.think(t)),
                (Part::Content(t), Some(tp)) => out.extend(self.events(st, tp.push(&t))),
                (Part::Content(t), None) => out.extend(self.events(st, vec![Piece::Text(t)])),
            }
        }
        out
    }

    fn think(&mut self, t: String) -> Vec<Event> {
        let mut out = self.close_text();
        if !self.think_open {
            out.push(sse_json(
                Some("content_block_start"),
                &json!({"type": "content_block_start", "index": self.next, "content_block": {"type": "thinking", "thinking": "", "signature": ""}}),
            ));
            self.think_open = true;
        }
        out.push(sse_json(
            Some("content_block_delta"),
            &json!({"type": "content_block_delta", "index": self.next, "delta": {"type": "thinking_delta", "thinking": t}}),
        ));
        out
    }

    /// Close an open `thinking` block (Anthropic ends one with a `signature_delta`; the signature is empty here, the
    /// thinking is not signed).
    fn close_think(&mut self) -> Vec<Event> {
        if !self.think_open {
            return Vec::new();
        }
        self.think_open = false;
        self.next += 1;
        vec![
            sse_json(
                Some("content_block_delta"),
                &json!({"type": "content_block_delta", "index": self.next - 1, "delta": {"type": "signature_delta", "signature": ""}}),
            ),
            sse_json(Some("content_block_stop"), &json!({"type": "content_block_stop", "index": self.next - 1})),
        ]
    }

    fn events(&mut self, st: &AppState, pieces: Vec<Piece>) -> Vec<Event> {
        let mut out = Vec::new();
        for p in pieces {
            match p {
                Piece::Text(t) if t.is_empty() => {}
                Piece::Text(t) => {
                    out.extend(self.close_think());
                    if !self.text_open {
                        out.push(sse_json(
                            Some("content_block_start"),
                            &json!({"type": "content_block_start", "index": self.next, "content_block": {"type": "text", "text": ""}}),
                        ));
                        self.text_open = true;
                    }
                    out.push(sse_json(
                        Some("content_block_delta"),
                        &json!({"type": "content_block_delta", "index": self.next, "delta": {"type": "text_delta", "text": t}}),
                    ));
                }
                Piece::Call { name, arguments } => {
                    out.extend(self.close_think());
                    out.extend(self.close_text());
                    let i = self.next;
                    out.push(sse_json(
                        Some("content_block_start"),
                        &json!({"type": "content_block_start", "index": i,
                                "content_block": {"type": "tool_use", "id": tool_use_id(st), "name": name, "input": {}}}),
                    ));
                    out.push(sse_json(
                        Some("content_block_delta"),
                        &json!({"type": "content_block_delta", "index": i,
                                "delta": {"type": "input_json_delta", "partial_json": arguments.to_string()}}),
                    ));
                    out.push(sse_json(Some("content_block_stop"), &json!({"type": "content_block_stop", "index": i})));
                    self.next += 1;
                    self.calls += 1;
                }
            }
        }
        out
    }

    fn close_text(&mut self) -> Vec<Event> {
        if !self.text_open {
            return Vec::new();
        }
        self.text_open = false;
        self.next += 1;
        vec![sse_json(Some("content_block_stop"), &json!({"type": "content_block_stop", "index": self.next - 1}))]
    }
}

fn chat_opts(thinking: &Option<Value>, kwargs: &Option<serde_json::Map<String, Value>>) -> ChatOpts {
    let mut opts = ChatOpts { add_generation_prompt: true, ..Default::default() };
    match thinking.as_ref().and_then(|t| t.get("type")).and_then(Value::as_str) {
        Some("enabled") => opts.enable_thinking = Some(true),
        Some("disabled") => opts.enable_thinking = Some(false),
        _ => {}
    }
    if let Some(kw) = kwargs {
        for (k, v) in kw {
            if k == "enable_thinking" {
                opts.enable_thinking = v.as_bool();
            } else {
                opts.extra.insert(k.clone(), v.clone());
            }
        }
    }
    opts
}

/// `(stop_reason, stop_sequence)`: a stop string reports `stop_sequence` and the matched value.
fn stop_reason(f: FinishReason, stop_sequence: Option<&str>) -> (&'static str, Option<String>) {
    match (f, stop_sequence) {
        (FinishReason::Length, _) => ("max_tokens", None),
        (FinishReason::Stop, Some(s)) => ("stop_sequence", Some(s.to_string())),
        _ => ("end_turn", None),
    }
}

/// Anthropic's `input_tokens` counts only the prompt tokens that were not read from the cache.
fn usage(prompt_tokens: usize, cached_tokens: usize, output_tokens: usize) -> Value {
    json!({"input_tokens": prompt_tokens.saturating_sub(cached_tokens), "output_tokens": output_tokens,
           "cache_read_input_tokens": cached_tokens, "cache_creation_input_tokens": 0})
}

fn err_event(message: &str) -> Event {
    sse_json(Some("error"), &json!({"type": "error", "error": {"type": "api_error", "message": message}}))
}

pub async fn messages(
    State(st): State<AppState>,
    headers: axum::http::HeaderMap,
    body: Result<Json<MessagesReq>, axum::extract::rejection::JsonRejection>,
) -> Response {
    match messages_inner(st.for_request(&headers), body).await {
        Ok(r) => r,
        Err(e) => e.into_response(),
    }
}

async fn messages_inner(
    st: AppState,
    body: Result<Json<MessagesReq>, axum::extract::rejection::JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(req) = body.map_err(|e| ApiError::bad_request(F, e.body_text()))?;
    let info = st.engine.info();
    if !info.has_chat_template {
        return Err(ApiError::bad_request(F, "this model has no chat template"));
    }
    let max_tokens = req.max_tokens.ok_or_else(|| ApiError::bad_request(F, "max_tokens is required"))?;
    if req.messages.is_empty() {
        return Err(ApiError::bad_request(F, "messages must not be empty"));
    }
    let bad = |m: String| ApiError::bad_request(F, m);
    let tools = structured::anthropic_tools(&req.tools).map_err(bad)?;
    let choice = structured::anthropic_choice(&req.tool_choice, &tools).map_err(bad)?;
    let parallel = !req
        .tool_choice
        .as_ref()
        .and_then(|c| c.get("disable_parallel_tool_use"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let hermes = info.tool_parser.as_deref() == Some("hermes");
    let (plan, parse_tools) = structured::choose(None, &tools, &choice, parallel, hermes).map_err(bad)?;
    let messages = to_chat_messages(&req.system, &req.messages)?;
    let mut opts = chat_opts(&req.thinking, &req.chat_template_kwargs);
    if !tools.is_empty() {
        opts.tools = Some(tools.iter().map(Tool::openai).collect());
    }
    // Thinking blocks are produced (and a constraint waits for `</think>`) only when the request turns thinking on and
    // the model has a reasoning parser; every other request is answered exactly as before.
    let think = opts.enable_thinking == Some(true) && info.reasoning_parser.is_some();
    let sampling = Sampling {
        max_tokens: Some(max_tokens),
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        stop: req.stop_sequences.clone(),
        ..Default::default()
    };
    let prompt = blocking_as(&st.engine, F, move |e| {
        let text = e.apply_chat_template(&messages, &opts)?;
        e.tokenize(&text, false)
    })
    .await?;
    let input_tokens = prompt.len();
    let mut gen = sampling.into_request(prompt, F, info.context_len, st.default_max_tokens)?;
    structured::attach(&st, F, &mut gen, plan, think).await?;
    let rx = start(&st, F, gen)?;
    let id = st.new_id("msg").replacen("msg-", "msg_", 1);
    let model = st.model_name.clone();

    if !req.stream {
        let c = collect(rx, F).await?;
        let (mut reason, seq) = stop_reason(c.finish, c.stop_sequence.as_deref());
        let mut content = vec![json!({"type": "text", "text": c.text})];
        let mut thinking = None;
        let mut answer = c.text.clone();
        if think {
            let (reasoning, text) = ReasoningParser::parse_all(&c.text);
            thinking = reasoning.map(|r| json!({"type": "thinking", "thinking": r, "signature": ""}));
            answer = text.unwrap_or_default();
            content = vec![json!({"type": "text", "text": answer})];
        }
        if parse_tools {
            let (text, calls) = structured::split_calls(&answer);
            if !calls.is_empty() {
                content = text.map(|t| json!({"type": "text", "text": t})).into_iter().collect();
                for (name, input) in calls {
                    content.push(json!({"type": "tool_use", "id": tool_use_id(&st), "name": name, "input": input}));
                }
                if c.finish != FinishReason::Length {
                    reason = "tool_use";
                }
            }
        }
        if let Some(t) = thinking {
            content.insert(0, t);
        }
        return Ok(Json(json!({
            "id": id, "type": "message", "role": "assistant", "model": model,
            "content": content,
            "stop_reason": reason, "stop_sequence": seq,
            "usage": usage(c.prompt_tokens, c.cached_tokens, c.completion_tokens),
        }))
        .into_response());
    }

    let start_ev = sse_json(
        Some("message_start"),
        &json!({"type": "message_start", "message": {
            "id": id, "type": "message", "role": "assistant", "model": model, "content": [],
            "stop_reason": null, "stop_sequence": null,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0}}}),
    );
    if parse_tools || think {
        return Ok(stream_with_tools(st, rx, start_ev, parse_tools, think).into_response());
    }
    let prelude = vec![
        start_ev,
        sse_json(
            Some("content_block_start"),
            &json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ),
    ];
    let mut done = false;
    let sse = sse_from(rx, prelude, move |ev| match ev {
        Some(GenEvent::Token { text, .. }) => {
            if text.is_empty() {
                vec![]
            } else {
                vec![sse_json(
                    Some("content_block_delta"),
                    &json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}),
                )]
            }
        }
        Some(GenEvent::Done { finish_reason, prompt_tokens, completion_tokens, cached_tokens, stop_sequence, .. }) => {
            done = true;
            let (reason, seq) = stop_reason(finish_reason, stop_sequence.as_deref());
            vec![
                sse_json(Some("content_block_stop"), &json!({"type": "content_block_stop", "index": 0})),
                sse_json(
                    Some("message_delta"),
                    &json!({"type": "message_delta",
                            "delta": {"stop_reason": reason, "stop_sequence": seq},
                            "usage": usage(prompt_tokens, cached_tokens, completion_tokens)}),
                ),
                sse_json(Some("message_stop"), &json!({"type": "message_stop"})),
            ]
        }
        Some(GenEvent::Error { message }) => {
            done = true;
            vec![err_event(&message)]
        }
        None if !done => vec![err_event("engine stopped before finishing the request")],
        None => vec![],
    });
    Ok(sse.into_response())
}

/// The streaming response when tool calls are parsed out of the output: blocks open as the pieces arrive.
fn stream_with_tools(
    st: AppState,
    rx: GenRx,
    start_ev: Event,
    parse_tools: bool,
    think: bool,
) -> axum::response::sse::Sse<impl futures_util::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let mut done = false;
    let mut tp = parse_tools.then(ToolParser::default);
    let mut rp = think.then(ReasoningParser::default);
    let mut b = Blocks { next: 0, text_open: false, think_open: false, calls: 0 };
    sse_from(rx, vec![start_ev], move |ev| match ev {
        Some(GenEvent::Token { text, .. }) => match rp.as_mut() {
            Some(rp) => b.parts(&st, rp.push(&text), &mut tp),
            None => b.parts(&st, vec![Part::Content(text)], &mut tp),
        },
        Some(GenEvent::Done { finish_reason, prompt_tokens, completion_tokens, cached_tokens, stop_sequence, .. }) => {
            done = true;
            let tail = rp.as_mut().map(|rp| rp.finish()).unwrap_or_default();
            let mut out = b.parts(&st, tail, &mut tp);
            if let Some(tp) = tp.as_mut() {
                out.extend(b.events(&st, tp.finish()));
            }
            out.extend(b.close_think());
            out.extend(b.close_text());
            if b.next == 0 {
                // nothing was generated: one empty text block, as without tools
                out.push(sse_json(
                    Some("content_block_start"),
                    &json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
                ));
                out.push(sse_json(Some("content_block_stop"), &json!({"type": "content_block_stop", "index": 0})));
            }
            let (mut reason, seq) = stop_reason(finish_reason, stop_sequence.as_deref());
            if b.calls > 0 && finish_reason != FinishReason::Length {
                reason = "tool_use";
            }
            out.push(sse_json(
                Some("message_delta"),
                &json!({"type": "message_delta", "delta": {"stop_reason": reason, "stop_sequence": seq},
                        "usage": usage(prompt_tokens, cached_tokens, completion_tokens)}),
            ));
            out.push(sse_json(Some("message_stop"), &json!({"type": "message_stop"})));
            out
        }
        Some(GenEvent::Error { message }) => {
            done = true;
            vec![err_event(&message)]
        }
        None if !done => vec![err_event("engine stopped before finishing the request")],
        None => vec![],
    })
}

#[derive(Deserialize)]
pub struct CountReq {
    messages: Vec<Value>,
    #[serde(default)]
    system: Option<Value>,
    #[serde(default)]
    thinking: Option<Value>,
    #[serde(default)]
    tools: Option<Value>,
}

pub async fn count_tokens(
    State(st): State<AppState>,
    body: Result<Json<CountReq>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let r = async {
        let Json(req) = body.map_err(|e| ApiError::bad_request(F, e.body_text()))?;
        if !st.engine.info().has_chat_template {
            return Err(ApiError::bad_request(F, "this model has no chat template"));
        }
        let messages = to_chat_messages(&req.system, &req.messages)?;
        let mut opts = chat_opts(&req.thinking, &None);
        let tools = structured::anthropic_tools(&req.tools).map_err(|m| ApiError::bad_request(F, m))?;
        if !tools.is_empty() {
            opts.tools = Some(tools.iter().map(Tool::openai).collect());
        }
        let ids = blocking_as(&st.engine, F, move |e| {
            let text = e.apply_chat_template(&messages, &opts)?;
            e.tokenize(&text, false)
        })
        .await?;
        Ok(Json(json!({"input_tokens": ids.len()})).into_response())
    }
    .await;
    r.unwrap_or_else(|e: ApiError| e.into_response())
}
