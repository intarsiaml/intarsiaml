//! Inkling-2 through the OpenAI endpoint (lane ink-serve): the chat prompt comes from the model's own renderer, thinking and
//! tool calls are read from the block markers by token id, a JSON response format is framed in blocks. A mock engine with a
//! synthetic byte-level vocabulary plus the Inkling framing tokens; the markers reach the server as `Token` events with empty
//! text, as the real engine's detokenizer releases them.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use http_body_util::BodyExt;
use intarsia_core::engine::{ChatOpts, FinishReason, GenEvent, GenRequest, ModelInfo};
use intarsia_core::memplan::MemoryPlan;
use intarsia_core::telemetry::TelemetrySnapshot;
use intarsia_core::{Engine, Result};
use intarsia_models::inkling_chat::InklingChat;
use serde_json::{json, Value};
use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Mutex};
use tower::ServiceExt;

const NAMES: [&str; 12] = [
    "<|message_user|>", "<|message_model|>", "<|message_system|>", "<|message_tool|>", "<|content_text|>", "<|content_thinking|>",
    "<|content_tool_error|>", "<|content_xml|>", "<|content_invoke_tool_json|>", "<|end_message|>", "<|content_model_end_sampling|>",
    "<|content_invoke_tool_text|>",
];
const fn id(i: usize) -> u32 {
    256 + i as u32
}
const USER: u32 = id(0);
const MODEL: u32 = id(1);
const TEXT: u32 = id(4);
const THINK: u32 = id(5);
const TOOL_JSON: u32 = id(8);
const END: u32 = id(9);
const EOS: u32 = id(10);
const VOCAB: usize = 256 + 12;

/// Byte-fallback BPE: `<0xNN>` tokens (id = the byte), the framing tokens as added special tokens.
fn tokenizer_json() -> String {
    let mut vocab = serde_json::Map::new();
    for b in 0..256 {
        vocab.insert(format!("<0x{b:02X}>"), json!(b));
    }
    let added: Vec<Value> = NAMES
        .iter()
        .enumerate()
        .map(|(i, n)| json!({"id": id(i), "content": n, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}))
        .collect();
    json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added, "normalizer": null, "pre_tokenizer": null,
        "post_processor": null, "decoder": {"type": "Sequence", "decoders": [{"type": "ByteFallback"}]},
        "model": {"type": "BPE", "vocab": vocab, "merges": [], "byte_fallback": true}
    })
    .to_string()
}

fn tokenize(text: &str) -> Vec<u32> {
    let mut out = Vec::new();
    let mut rest = text;
    'outer: while !rest.is_empty() {
        for (i, n) in NAMES.iter().enumerate() {
            if let Some(r) = rest.strip_prefix(n) {
                out.push(id(i));
                rest = r;
                continue 'outer;
            }
        }
        out.push(rest.as_bytes()[0] as u32);
        rest = &rest[1..];
    }
    out
}

fn text_of(i: u32) -> String {
    if i < 256 { (i as u8 as char).to_string() } else { String::new() }
}

/// What the mock "model" does: tokens it says (the markers and ASCII), or under a constraint the same where the mask lets it.
#[derive(Default)]
struct Ink {
    chat: Option<InklingChat>,
    say: Mutex<Vec<u32>>,
    reqs: Mutex<Vec<GenRequest>>,
    masks: Mutex<Vec<Vec<u32>>>,
}

fn say(s: &[&str]) -> Vec<u32> {
    // markers by name, anything else as bytes
    s.iter().flat_map(|p| tokenize(p)).collect()
}

impl Engine for Ink {
    fn info(&self) -> ModelInfo {
        ModelInfo {
            id: "ink-mock".into(),
            family: "inkling2".into(),
            backend: "cpu".into(),
            device: "test".into(),
            arch: "x".into(),
            n_layers: 1,
            hidden: 1,
            vocab: VOCAB,
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: 1,
            max_position_embeddings: 4096,
            context_len: 4096,
            act_dtype: "bf16".into(),
            kv_dtype: "bf16".into(),
            weights_bytes: 10,
            eos_ids: vec![EOS],
            has_chat_template: true,
            reasoning_parser: Some("inkling".into()),
            tool_parser: Some("inkling".into()),
            memory_plan: MemoryPlan {
                device: "test".into(), total_bytes: 0, free_bytes: 0, weights_bytes: 0, workspace_bytes: 0, kv_bytes_per_token: 0,
                kv_fraction: 0.5, kv_budget_bytes: 0, kv_cap_tokens: 4096, kv_total_bytes: 0, notes: vec![],
            },
            reuse_cold_cuts_only: false,
        }
    }
    fn tokenize(&self, text: &str, _add_special: bool) -> Result<Vec<u32>> {
        Ok(tokenize(text))
    }
    fn detokenize(&self, ids: &[u32]) -> Result<String> {
        Ok(ids.iter().map(|&i| text_of(i)).collect())
    }
    fn apply_chat_template(&self, _: &[Value], _: &ChatOpts) -> Result<String> {
        panic!("Inkling prompts are rendered to ids, not through the jinja template")
    }
    fn chat_ids(&self, messages: &[Value], opts: &ChatOpts) -> Result<Option<Vec<u32>>> {
        self.chat.as_ref().unwrap().render(messages, opts).map(Some)
    }
    fn tokenizer_json(&self) -> Option<String> {
        Some(tokenizer_json())
    }
    fn submit(&self, req: GenRequest) -> Result<Receiver<GenEvent>> {
        let (tx, rx) = channel();
        let plen = req.prompt.len();
        let want = self.say.lock().unwrap().clone();
        let mut out: Vec<u32> = Vec::new();
        let mut finish = FinishReason::Stop;
        if let Some(spec) = &req.constraint {
            let mut c = spec.start().unwrap();
            for step in 0..req.max_tokens {
                let allowed: Vec<u32> = if c.active() {
                    let m = c.compute_mask().unwrap();
                    let a: Vec<u32> = (0..VOCAB as u32).filter(|&t| m.allows(t)).collect();
                    self.masks.lock().unwrap().push(a.clone());
                    a
                } else {
                    (0..VOCAB as u32).collect()
                };
                let pick = want.get(step).copied().filter(|t| allowed.contains(t)).unwrap_or_else(|| if allowed.contains(&EOS) { EOS } else { allowed[0] });
                if pick == EOS {
                    break;
                }
                c.commit(pick).unwrap();
                out.push(pick);
            }
        } else {
            out = want.iter().copied().take_while(|&t| t != EOS).take(req.max_tokens).collect();
            if out.len() >= req.max_tokens && want.len() > out.len() {
                finish = FinishReason::Length;
            }
        }
        self.reqs.lock().unwrap().push(req);
        let n = out.len();
        std::thread::spawn(move || {
            for &t in &out {
                tx.send(GenEvent::Token { id: t, text: text_of(t) }).ok();
            }
            tx.send(GenEvent::Done {
                finish_reason: finish, stop_sequence: None, prompt_tokens: plen, completion_tokens: n + 1, cached_tokens: 0,
                prefill_ms: 1.0, decode_ms: 1.0, prefill_chunk: None,
            })
            .ok();
        });
        Ok(rx)
    }
    fn forward_logits_each(&self, _: &[u32], _: &mut dyn FnMut(usize, &[f32]) -> Result<()>) -> Result<()> {
        Ok(())
    }
    fn telemetry(&self) -> TelemetrySnapshot {
        TelemetrySnapshot { schema_version: 1, model: "ink-mock".into(), ..Default::default() }
    }
}

fn app(say: Vec<u32>) -> (Router, Arc<Ink>) {
    let m = Arc::new(Ink { chat: InklingChat::from_tokenizer_json(&tokenizer_json()).unwrap(), say: Mutex::new(say), ..Default::default() });
    (intarsia_server::router(m.clone(), None), m)
}

async fn chat(app: &Router, body: Value) -> (StatusCode, String) {
    let req = Request::builder().method("POST").uri("/v1/chat/completions").header("content-type", "application/json").body(Body::from(body.to_string())).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

fn chunks(body: &str) -> Vec<Value> {
    body.lines().filter_map(|l| l.strip_prefix("data: ")).filter(|d| *d != "[DONE]").map(|d| serde_json::from_str(d).unwrap()).collect()
}

const ANSWER: [&str; 9] = ["<|message_model|>", "<|content_thinking|>", "The user asks for the capital. ", "<|end_message|>", "<|message_model|>", "<|content_text|>", "Paris.", "<|end_message|>", "<|content_model_end_sampling|>"];

#[tokio::test]
async fn the_prompt_is_the_vllm_rendering_and_reasoning_is_split() {
    let (app, mock) = app(say(&ANSWER));
    let (st, body) = chat(&app, json!({"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Capital of France?"}], "reasoning_effort": "low", "max_tokens": 200})).await;
    assert_eq!(st, StatusCode::OK, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    let m = &v["choices"][0]["message"];
    assert_eq!(m["reasoning_content"], "The user asks for the capital.");
    assert_eq!(m["content"], "Paris.");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert!(m.get("tool_calls").is_none());
    // the prompt: no generation prefix, the effort block before the first non-system message
    let prompt = mock.reqs.lock().unwrap()[0].prompt.clone();
    let text: String = prompt.iter().map(|&i| if i >= 256 { NAMES[i as usize - 256].to_string() } else { (i as u8 as char).to_string() }).collect();
    assert_eq!(
        text,
        "<|message_system|><|content_text|>Be brief.<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|><|message_user|><|content_text|>Capital of France?<|end_message|>"
    );
    assert_eq!(prompt.last(), Some(&END));
}

#[tokio::test]
async fn kwargs_effort_wins_and_none_is_zero() {
    let (app, mock) = app(say(&ANSWER));
    let (st, body) = chat(&app, json!({"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "high", "chat_template_kwargs": {"reasoning_effort": "none"}})).await;
    assert_eq!(st, StatusCode::OK, "{body}");
    let prompt = mock.reqs.lock().unwrap()[0].prompt.clone();
    let text: String = prompt.iter().map(|&i| if i >= 256 { NAMES[i as usize - 256].to_string() } else { (i as u8 as char).to_string() }).collect();
    assert!(text.contains("Thinking effort level: 0<|end_message|>"), "{text}");
    let (st, body) = chat(&app, json!({"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "loud"})).await;
    assert_eq!(st, StatusCode::BAD_REQUEST, "{body}");
}

#[tokio::test]
async fn streaming_splits_reasoning_and_content_by_marker() {
    let (app, _) = app(say(&ANSWER));
    let (st, body) = chat(&app, json!({"messages": [{"role": "user", "content": "hi"}], "stream": true, "stream_options": {"include_usage": true}})).await;
    assert_eq!(st, StatusCode::OK);
    let cs = chunks(&body);
    let (mut reasoning, mut content) = (String::new(), String::new());
    for c in &cs {
        let d = &c["choices"].get(0).map(|c| c["delta"].clone()).unwrap_or(Value::Null);
        if let Some(r) = d["reasoning_content"].as_str() {
            reasoning.push_str(r);
        }
        if let Some(t) = d["content"].as_str() {
            content.push_str(t);
        }
    }
    assert_eq!((reasoning.as_str(), content.as_str()), ("The user asks for the capital.", "Paris."));
    let fr: Vec<&Value> = cs.iter().filter_map(|c| c["choices"].get(0)).map(|c| &c["finish_reason"]).filter(|f| !f.is_null()).collect();
    assert_eq!(fr, vec![&json!("stop")]);
    assert!(cs.last().unwrap()["usage"]["completion_tokens"].as_u64().unwrap() > 0);
}

fn call_reply() -> Vec<u32> {
    say(&["<|message_model|>", "<|content_thinking|>", "need weather", "<|end_message|>", "<|message_model|>", "get_weather", "<|content_invoke_tool_json|>", r#"{"name":"get_weather","args":{"city":"Paris","days":2}}"#, "<|end_message|>", "<|content_model_end_sampling|>"])
}

fn weather() -> Value {
    json!([{"type": "function", "function": {"name": "get_weather", "description": "Weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}])
}

#[tokio::test]
async fn a_tool_call_is_parsed_and_the_declaration_is_in_the_prompt() {
    let (app, mock) = app(call_reply());
    let (st, body) = chat(&app, json!({"messages": [{"role": "user", "content": "weather in Paris?"}], "tools": weather()})).await;
    assert_eq!(st, StatusCode::OK, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    let c = &v["choices"][0];
    assert_eq!(c["finish_reason"], "tool_calls");
    assert_eq!(c["message"]["reasoning_content"], "need weather");
    assert!(c["message"]["content"].is_null(), "{c}");
    let call = &c["message"]["tool_calls"][0];
    assert_eq!(call["function"]["name"], "get_weather");
    assert_eq!(serde_json::from_str::<Value>(call["function"]["arguments"].as_str().unwrap()).unwrap(), json!({"city": "Paris", "days": 2}));
    assert!(call["id"].as_str().unwrap().starts_with("call"));
    let prompt = mock.reqs.lock().unwrap()[0].prompt.clone();
    let text: String = prompt.iter().map(|&i| if i >= 256 { NAMES[i as usize - 256].to_string() } else { (i as u8 as char).to_string() }).collect();
    assert!(text.starts_with("<|message_system|>tool_declare<|content_xml|>[{\"description\":\"Weather\",\"name\":\"get_weather\""), "{text}");
}

#[tokio::test]
async fn a_streamed_tool_call_and_the_follow_up_turn() {
    let (app, mock) = app(call_reply());
    let (_, body) = chat(&app, json!({"messages": [{"role": "user", "content": "w?"}], "tools": weather(), "stream": true})).await;
    let cs = chunks(&body);
    let calls: Vec<&Value> = cs.iter().filter_map(|c| c["choices"][0]["delta"].get("tool_calls")).collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0][0]["index"], 0);
    assert_eq!(calls[0][0]["function"]["name"], "get_weather");
    let fr = cs.iter().filter_map(|c| c["choices"][0]["finish_reason"].as_str()).last();
    assert_eq!(fr, Some("tool_calls"));
    // the client sends the call back with string arguments and the tool's result
    *mock.say.lock().unwrap() = say(&["<|message_model|>", "<|content_text|>", "Sunny.", "<|end_message|>", "<|content_model_end_sampling|>"]);
    let (st, body) = chat(&app, json!({"messages": [
        {"role": "user", "content": "w?"},
        {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}], "tools": weather()})).await;
    assert_eq!(st, StatusCode::OK, "{body}");
    let prompt = mock.reqs.lock().unwrap().last().unwrap().prompt.clone();
    let text: String = prompt.iter().map(|&i| if i >= 256 { NAMES[i as usize - 256].to_string() } else { (i as u8 as char).to_string() }).collect();
    assert!(text.contains("<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\"}}<|end_message|><|content_model_end_sampling|><|message_tool|>get_weather<|content_text|>sunny<|end_message|>"), "{text}");
}

#[tokio::test]
async fn tool_choice_required_is_an_instruction_not_a_grammar() {
    let (app, mock) = app(call_reply());
    let (st, body) = chat(&app, json!({"messages": [{"role": "user", "content": "w?"}], "tools": weather(), "tool_choice": "required"})).await;
    assert_eq!(st, StatusCode::OK, "{body}");
    let r = mock.reqs.lock().unwrap()[0].clone();
    assert!(r.constraint.is_none());
    let text: String = r.prompt.iter().map(|&i| if i >= 256 { NAMES[i as usize - 256].to_string() } else { (i as u8 as char).to_string() }).collect();
    assert!(text.contains("Call at least one of the provided functions in your next response."), "{text}");
    // none: no declaration
    let (st, _) = chat(&app, json!({"messages": [{"role": "user", "content": "w?"}], "tools": weather(), "tool_choice": "none"})).await;
    assert_eq!(st, StatusCode::OK);
    let r = mock.reqs.lock().unwrap()[1].clone();
    let text: String = r.prompt.iter().map(|&i| if i >= 256 { NAMES[i as usize - 256].to_string() } else { (i as u8 as char).to_string() }).collect();
    assert!(!text.contains("tool_declare"), "{text}");
}

fn schema_format() -> Value {
    json!({"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": false}}})
}

#[tokio::test]
async fn a_json_schema_is_framed_in_blocks() {
    let want = say(&["<|message_model|>", "<|content_thinking|>", "hm", "<|end_message|>", "<|message_model|>", "<|content_text|>", r#"{"a": 1}"#, "<|end_message|>", "<|content_model_end_sampling|>"]);
    let (app, mock) = app(want);
    let (st, body) = chat(&app, json!({"messages": [{"role": "user", "content": "give a"}], "response_format": schema_format(), "max_tokens": 60})).await;
    assert_eq!(st, StatusCode::OK, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    let m = &v["choices"][0]["message"];
    assert_eq!(m["reasoning_content"], "hm");
    assert_eq!(m["content"], "{\"a\": 1}");
    assert_eq!(serde_json::from_str::<Value>(m["content"].as_str().unwrap()).unwrap(), json!({"a": 1}));
    let masks = mock.masks.lock().unwrap();
    // the model's first token must open a message: nothing else is allowed (a JSON object cannot start the reply)
    assert_eq!(masks[0], vec![MODEL]);
    // after `<|message_model|>`: a thinking block or the text block
    assert_eq!(masks[1], vec![TEXT, THINK]);
}

#[tokio::test]
async fn a_json_schema_cannot_be_escaped_into_free_text() {
    // the "model" answers in prose in the text block; the mask does not allow it, so the JSON comes out
    let want = say(&["<|message_model|>", "<|content_text|>", "Sure! here", "<|end_message|>"]);
    let (app, _) = app(want);
    let (st, body) = chat(&app, json!({"messages": [{"role": "user", "content": "give a"}], "response_format": schema_format(), "max_tokens": 60})).await;
    assert_eq!(st, StatusCode::OK, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    let content = v["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(serde_json::from_str::<Value>(content).is_ok(), "{content}");
}

#[tokio::test]
async fn images_and_the_anthropic_route_are_refused() {
    let (app, _) = app(say(&ANSWER));
    let (st, body) = chat(&app, json!({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]})).await;
    assert_eq!(st, StatusCode::BAD_REQUEST, "{body}");
    let req = Request::builder().method("POST").uri("/v1/messages").header("content-type", "application/json")
        .body(Body::from(json!({"model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}).to_string())).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let _ = USER;
    let _ = TOOL_JSON;
}
