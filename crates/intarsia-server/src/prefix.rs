//! Prefix store endpoints (an intarsia extension next to the OpenAI API; docs/results/2026-10-22-pc-prefix-store.md):
//!
//! - `POST /v1/intarsia/prefill`: prefill a prompt into the prefix store without generating, so a later request that
//!   starts with it restores it instead of computing it. A chat UI calls it while the user types (the conversation so
//!   far, `partial: true`); a deployment calls it at startup with `pin` for its system prompt and tool schema.
//! - `GET /v1/intarsia/prefix-store`: counters and entries.

use crate::api::{blocking, blocking_as, flatten_content, start, AppState, ApiError, Flavor};
use axum::extract::State;
use axum::response::{IntoResponse, Json, Response};
use intarsia_core::engine::{ChatOpts, GenEvent, GenRequest, Prewarm};
use serde::Deserialize;
use serde_json::{json, Map, Value};

const F: Flavor = Flavor::OpenAi;

#[derive(Deserialize, Default)]
#[serde(default)]
pub struct PrefillReq {
    /// Chat messages (rendered with the model's chat template), or
    messages: Option<Vec<Value>>,
    /// a raw prompt (no template).
    prompt: Option<String>,
    tools: Option<Value>,
    chat_template_kwargs: Option<Map<String, Value>>,
    /// Render the assistant generation prompt too (default false: the conversation as the next request will start it).
    add_generation_prompt: bool,
    /// The last message is still being typed: drop the template's end-of-message suffix and the last `margin` tokens
    /// (the tokenizer may merge them differently once more text follows), so the stored prefix is a prefix of the
    /// request that follows.
    partial: bool,
    margin: Option<usize>,
    /// Pin the stored prefix under this name (never evicted).
    pin: Option<String>,
}

pub async fn prefill(
    State(st): State<AppState>,
    body: Result<Json<PrefillReq>, axum::extract::rejection::JsonRejection>,
) -> Response {
    match prefill_inner(st, body).await {
        Ok(r) => r,
        Err(e) => e.into_response(),
    }
}

async fn prefill_inner(
    st: AppState,
    body: Result<Json<PrefillReq>, axum::extract::rejection::JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(req) = body.map_err(|e| ApiError::bad_request(F, e.body_text()))?;
    // an engine without a store (the batched engine, or the store off) would generate instead of prewarming
    if blocking(&st.engine, |e| Ok(e.prefix_store())).await?.is_null() {
        return Err(ApiError::bad_request(F, "prewarm needs the prefix store (serve --prefix-store on, --max-batch 1)"));
    }
    let partial = req.partial;
    let margin = req.margin.unwrap_or(if partial { 2 } else { 0 });
    let prompt: Vec<u32> = match (req.messages, req.prompt) {
        (Some(msgs), None) => {
            let mut messages = Vec::with_capacity(msgs.len());
            for m in &msgs {
                let Value::Object(o) = m else { return Err(ApiError::bad_request(F, "each message must be an object")) };
                let mut o = o.clone();
                let content = flatten_content(o.get("content").unwrap_or(&Value::Null), F)?;
                o.insert("content".into(), Value::String(content));
                messages.push(Value::Object(o));
            }
            let mut opts =
                ChatOpts { add_generation_prompt: req.add_generation_prompt, tools: req.tools.clone(), ..Default::default() };
            if let Some(kw) = &req.chat_template_kwargs {
                for (k, v) in kw {
                    if k == "enable_thinking" {
                        opts.enable_thinking = v.as_bool();
                    } else {
                        opts.extra.insert(k.clone(), v.clone());
                    }
                }
            }
            blocking_as(&st.engine, F, move |e| {
                let mut text = e.apply_chat_template(&messages, &opts)?;
                if partial {
                    // the message is not finished: cut the end-of-message suffix the template added (ChatML)
                    for suffix in ["<|im_end|>\n", "<|im_end|>"] {
                        if let Some(t) = text.strip_suffix(suffix) {
                            text = t.to_string();
                            break;
                        }
                    }
                }
                e.tokenize(&text, false)
            })
            .await?
        }
        (None, Some(p)) => blocking_as(&st.engine, F, move |e| e.tokenize(&p, false)).await?,
        _ => return Err(ApiError::bad_request(F, "give exactly one of messages or prompt")),
    };
    let keep = prompt.len().saturating_sub(margin);
    if keep == 0 {
        return Err(ApiError::bad_request(F, "nothing to prefill"));
    }
    let gen = GenRequest {
        prompt: prompt[..keep].to_vec(),
        max_tokens: 1,
        prewarm: Some(Prewarm { pin: req.pin.clone() }),
        ..Default::default()
    };
    let t0 = std::time::Instant::now();
    let mut rx = start(&st, F, gen)?;
    while let Some(ev) = rx.recv().await {
        match ev {
            GenEvent::Done { prompt_tokens, cached_tokens, prefill_ms, .. } => {
                return Ok(Json(json!({
                    "object": "intarsia.prefill",
                    "prompt_tokens": prompt_tokens,
                    "cached_tokens": cached_tokens,
                    "computed_tokens": prompt_tokens - cached_tokens,
                    "prefill_ms": prefill_ms,
                    "elapsed_ms": t0.elapsed().as_secs_f64() * 1e3,
                    "pinned": req.pin,
                }))
                .into_response());
            }
            GenEvent::Error { message } => return Err(ApiError::bad_request(F, message)),
            GenEvent::Token { .. } => {}
        }
    }
    Err(ApiError::internal(F, "engine stopped before finishing the prefill"))
}

/// `GET /v1/intarsia/prefix-store`: the store's counters and entries (`{"enabled": false}` when it is off).
pub async fn store_info(State(st): State<AppState>) -> Response {
    match blocking(&st.engine, |e| Ok(e.prefix_store())).await {
        Ok(Value::Null) => Json(json!({"enabled": false})).into_response(),
        Ok(mut v) => {
            v["enabled"] = json!(true);
            Json(v).into_response()
        }
        Err(e) => e.into_response(),
    }
}
