//! OpenAI-compatible endpoints.

use crate::api::*;
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
    chat_template_kwargs: Option<Map<String, Value>>,
    #[serde(flatten)]
    common: CommonParams,
}

pub async fn chat_completions(
    State(st): State<AppState>,
    body: Result<Json<ChatReq>, axum::extract::rejection::JsonRejection>,
) -> Response {
    match chat_inner(st, body).await {
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
    let prompt = blocking_as(&st.engine, F, move |e| {
        let text = e.apply_chat_template(&messages, &opts)?;
        e.tokenize(&text, false)
    })
    .await?;
    let gen = sampling.into_request(prompt, F, info.context_len, st.default_max_tokens)?;
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

    if !req.common.stream {
        let c = collect(rx, F).await?;
        let v = json!({
            "id": chunk(json!({}), None, false)["id"].clone(), "object": "chat.completion", "created": created, "model": st.model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": c.text},
                         "finish_reason": finish_str(c.finish)}],
            "usage": usage(c.prompt_tokens, c.completion_tokens, c.cached_tokens),
        });
        return Ok(Json(v).into_response());
    }

    let inc = req.common.include_usage();
    let first = sse_json(None, &chunk(json!({"role": "assistant", "content": ""}), None, inc));
    let mut done = false;
    let model2 = st.model_name.clone();
    let id2 = chunk(json!({}), None, false)["id"].clone();
    let sse = sse_from(rx, vec![first], move |ev| match ev {
        Some(GenEvent::Token { text, .. }) => {
            if text.is_empty() {
                vec![]
            } else {
                vec![sse_json(None, &chunk(json!({"content": text}), None, inc))]
            }
        }
        Some(GenEvent::Done { finish_reason, prompt_tokens, completion_tokens, cached_tokens, .. }) => {
            done = true;
            let mut out = vec![sse_json(None, &chunk(json!({}), Some(finish_str(finish_reason)), inc))];
            if inc {
                out.push(sse_json(
                    None,
                    &json!({"id": id2, "object": "chat.completion.chunk", "created": created, "model": model2,
                            "choices": [], "usage": usage(prompt_tokens, completion_tokens, cached_tokens)}),
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
    #[serde(flatten)]
    common: CommonParams,
}

pub async fn completions(
    State(st): State<AppState>,
    body: Result<Json<CompletionReq>, axum::extract::rejection::JsonRejection>,
) -> Response {
    match completions_inner(st, body).await {
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
    let pin = parse_prompt(&req.prompt)?;
    let (prompt, prefix) = match pin {
        PromptIn::Ids(ids) => (ids, String::new()),
        PromptIn::Text(t) => {
            let t2 = t.clone();
            let ids = blocking_as(&st.engine, F, move |e| e.tokenize(&t2, true)).await?;
            (ids, if req.echo { t } else { String::new() })
        }
    };
    let gen = sampling.into_request(prompt, F, info.context_len, st.default_max_tokens)?;
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

    if !req.common.stream {
        let c = collect(rx, F).await?;
        let mut v = chunk(&format!("{prefix}{}", c.text), Some(finish_str(c.finish)), false);
        v["usage"] = usage(c.prompt_tokens, c.completion_tokens, c.cached_tokens);
        return Ok(Json(v).into_response());
    }

    let inc = req.common.include_usage();
    let mut pre = Vec::new();
    if !prefix.is_empty() {
        pre.push(sse_json(None, &chunk(&prefix, None, inc)));
    }
    let mut done = false;
    let (model2, id2) = (st.model_name.clone(), chunk("", None, false)["id"].clone());
    let sse = sse_from(rx, pre, move |ev| match ev {
        Some(GenEvent::Token { text, .. }) => {
            if text.is_empty() {
                vec![]
            } else {
                vec![sse_json(None, &chunk(&text, None, inc))]
            }
        }
        Some(GenEvent::Done { finish_reason, prompt_tokens, completion_tokens, cached_tokens, .. }) => {
            done = true;
            let mut out = vec![sse_json(None, &chunk("", Some(finish_str(finish_reason)), inc))];
            if inc {
                out.push(sse_json(
                    None,
                    &json!({"id": id2, "object": "text_completion", "created": created, "model": model2,
                            "choices": [], "usage": usage(prompt_tokens, completion_tokens, cached_tokens)}),
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
