//! Anthropic Messages API (text only).

use crate::api::*;
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
        let content = blocks_to_text(m.get("content").unwrap_or(&Value::Null))?;
        out.push(json!({"role": role, "content": content}));
    }
    Ok(out)
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
    body: Result<Json<MessagesReq>, axum::extract::rejection::JsonRejection>,
) -> Response {
    match messages_inner(st, body).await {
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
    let messages = to_chat_messages(&req.system, &req.messages)?;
    let opts = chat_opts(&req.thinking, &req.chat_template_kwargs);
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
    let gen = sampling.into_request(prompt, F, info.context_len, st.default_max_tokens)?;
    let rx = start(&st, F, gen)?;
    let id = st.new_id("msg").replacen("msg-", "msg_", 1);
    let model = st.model_name.clone();

    if !req.stream {
        let c = collect(rx, F).await?;
        let (reason, seq) = stop_reason(c.finish, c.stop_sequence.as_deref());
        return Ok(Json(json!({
            "id": id, "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": c.text}],
            "stop_reason": reason, "stop_sequence": seq,
            "usage": usage(c.prompt_tokens, c.cached_tokens, c.completion_tokens),
        }))
        .into_response());
    }

    let prelude = vec![
        sse_json(
            Some("message_start"),
            &json!({"type": "message_start", "message": {
                "id": id, "type": "message", "role": "assistant", "model": model, "content": [],
                "stop_reason": null, "stop_sequence": null,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0}}}),
        ),
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

#[derive(Deserialize)]
pub struct CountReq {
    messages: Vec<Value>,
    #[serde(default)]
    system: Option<Value>,
    #[serde(default)]
    thinking: Option<Value>,
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
        let opts = chat_opts(&req.thinking, &None);
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
