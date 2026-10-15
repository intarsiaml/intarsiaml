//! Handler tests against a mock engine (no model needed).

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use http_body_util::BodyExt;
use intarsia_core::engine::{ChatOpts, FinishReason, GenEvent, GenRequest, ModelInfo};
use intarsia_core::memplan::MemoryPlan;
use intarsia_core::telemetry::{OpStat, TelemetrySnapshot};
use intarsia_core::{Engine, Result};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Mutex};
use tower::ServiceExt;

#[derive(Default)]
struct Mock {
    reqs: Mutex<Vec<GenRequest>>,
    templated: Mutex<Vec<(Vec<Value>, ChatOpts)>>,
    /// When set, generate slowly and forever so cancellation can be observed.
    slow: bool,
    cancelled: Arc<AtomicBool>,
    /// Engine reports itself failed.
    fail: bool,
    /// Scripted generation: token events, `completion_tokens`, finish reason (overrides the canned one).
    script: Option<(Vec<(u32, &'static str)>, usize, FinishReason)>,
    /// Finish with this stop sequence.
    stop_seq: Option<String>,
    /// Telemetry overrides.
    busy: (u32, u32),
}

impl Engine for Mock {
    fn info(&self) -> ModelInfo {
        ModelInfo {
            id: "mock-model".into(),
            family: "mock".into(),
            backend: "cpu".into(),
            device: "test".into(),
            arch: "x".into(),
            n_layers: 1,
            hidden: 1,
            vocab: 256,
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: 1,
            max_position_embeddings: 64,
            context_len: 64,
            act_dtype: "bf16".into(),
            kv_dtype: "bf16".into(),
            weights_bytes: 10,
            eos_ids: vec![0],
            has_chat_template: true,
            reasoning_parser: None,
            tool_parser: None,
            memory_plan: MemoryPlan {
                device: "test".into(),
                total_bytes: 0,
                free_bytes: 0,
                weights_bytes: 0,
                workspace_bytes: 0,
                kv_bytes_per_token: 0,
                kv_fraction: 0.5,
                kv_budget_bytes: 0,
                kv_cap_tokens: 64,
                kv_total_bytes: 0,
                notes: vec![],
            },
        }
    }
    fn tokenize(&self, text: &str, _add_special: bool) -> Result<Vec<u32>> {
        Ok(text.bytes().map(|b| b as u32).collect())
    }
    fn detokenize(&self, ids: &[u32]) -> Result<String> {
        Ok(ids.iter().map(|i| *i as u8 as char).collect())
    }
    fn apply_chat_template(&self, messages: &[Value], opts: &ChatOpts) -> Result<String> {
        self.templated.lock().unwrap().push((messages.to_vec(), opts.clone()));
        let mut s = String::new();
        for m in messages {
            // The server must hand over plain-string content.
            let c = m["content"].as_str().expect("content must be a string");
            s.push_str(&format!("{}:{}\n", m["role"].as_str().unwrap(), c));
        }
        s.push_str("assistant:");
        Ok(s)
    }
    fn submit(&self, req: GenRequest) -> Result<Receiver<GenEvent>> {
        let (tx, rx) = channel();
        let plen = req.prompt.len();
        let max = req.max_tokens;
        self.reqs.lock().unwrap().push(req);
        let slow = self.slow;
        let stop_seq = self.stop_seq.clone();
        let cancelled = self.cancelled.clone();
        let script = self.script.clone();
        std::thread::spawn(move || {
            if let Some((toks, completion_tokens, finish_reason)) = script {
                for (id, text) in toks {
                    tx.send(GenEvent::Token { id, text: text.to_string() }).ok();
                }
                tx.send(GenEvent::Done {
                    finish_reason,
                    stop_sequence: None,
                    prompt_tokens: plen,
                    completion_tokens,
                    cached_tokens: 0,
                    prefill_ms: 1.0,
                    decode_ms: 1.0,
                    prefill_chunk: None,
                })
                .ok();
                return;
            }
            if slow {
                loop {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                    if tx.send(GenEvent::Token { id: 65, text: "x".into() }).is_err() {
                        cancelled.store(true, Ordering::SeqCst);
                        return;
                    }
                }
            }
            let n = max.min(2);
            let parts = ["Hel", "lo"];
            for p in parts.iter().take(n) {
                tx.send(GenEvent::Token { id: 65, text: p.to_string() }).ok();
                tx.send(GenEvent::Token { id: 65, text: String::new() }).ok();
            }
            let finish = if max < 2 { FinishReason::Length } else { FinishReason::Stop };
            tx.send(GenEvent::Done {
                finish_reason: finish,
                stop_sequence: stop_seq,
                prompt_tokens: plen,
                completion_tokens: n,
                cached_tokens: 3,
                prefill_ms: 20.0,
                decode_ms: 10.0,
                prefill_chunk: None,
            })
            .ok();
        });
        Ok(rx)
    }
    fn forward_logits_each(&self, _ids: &[u32], _cb: &mut dyn FnMut(usize, &[f32]) -> Result<()>) -> Result<()> {
        Ok(())
    }
    fn failed(&self) -> Option<String> {
        self.failed_hook()
    }
    fn plan(&self) -> Value {
        json!({"model": "mock-model", "layers": [], "ops": []})
    }
    fn telemetry(&self) -> TelemetrySnapshot {
        let mut t =
            TelemetrySnapshot { steps_total: 42, schema_version: 1, model: "mock-model".into(), ..Default::default() };
        t.running = self.busy.0;
        t.waiting = self.busy.1;
        t.prompt_tokens_total = 7;
        t.generation_tokens_total = 9;
        t.cached_tokens_total = 3;
        t.kv_used_bytes = 100;
        t.graph_replays = 11;
        t.graph_captures = 2;
        t.decode_eager_steps = 5;
        t.decode_profiled_steps = 3;
        t.kv_total_bytes = 1000;
        t.finished.insert("stop".into(), 2);
        t.ops.insert("matmul".into(), OpStat { calls: 5, timed_calls: 5, time_ms: 1500.0 });
        t
    }
}

impl Mock {
    fn failed_hook(&self) -> Option<String> {
        self.fail.then(|| "worker panicked: boom".to_string())
    }
}

fn app_with(mock: Arc<Mock>) -> Router {
    intarsia_server::router(mock, None)
}

fn app() -> (Router, Arc<Mock>) {
    let m = Arc::new(Mock::default());
    (app_with(m.clone()), m)
}

async fn call(app: &Router, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, String, String) {
    let req = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json")
        .body(match body {
            Some(b) => Body::from(b.to_string()),
            None => Body::empty(),
        })
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let ct = resp.headers().get("content-type").map(|v| v.to_str().unwrap().to_string()).unwrap_or_default();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, ct, String::from_utf8(bytes.to_vec()).unwrap())
}

fn sse_data(body: &str) -> Vec<String> {
    body.lines().filter_map(|l| l.strip_prefix("data: ")).map(String::from).collect()
}

fn sse_events(body: &str) -> Vec<(String, Value)> {
    let mut out = Vec::new();
    let mut name = String::new();
    for l in body.lines() {
        if let Some(n) = l.strip_prefix("event: ") {
            name = n.to_string();
        } else if let Some(d) = l.strip_prefix("data: ") {
            out.push((std::mem::take(&mut name), serde_json::from_str(d).unwrap()));
        }
    }
    out
}

#[tokio::test]
async fn chat_non_stream() {
    let (app, mock) = app();
    let (st, _, body) = call(
        &app,
        "POST",
        "/v1/chat/completions",
        Some(json!({
            "model": "whatever",
            "messages": [{"role": "system", "content": "be nice"},
                         {"role": "user", "content": [{"type": "text", "text": "hi "}, {"type": "text", "text": "there"}]}],
            "max_tokens": 50, "max_completion_tokens": 9, "min_tokens": 1, "ignore_eos": true,
            "temperature": 0.5, "top_p": 0.9, "top_k": 7, "seed": 11, "stop": ["END", "STOP"],
            "chat_template_kwargs": {"enable_thinking": false, "foo": 1}
        })),
    )
    .await;
    assert_eq!(st, StatusCode::OK, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["model"], "mock-model");
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    assert_eq!(v["choices"][0]["message"]["content"], "Hello");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(v["usage"]["completion_tokens"], 2);
    assert_eq!(v["usage"]["prompt_tokens_details"]["cached_tokens"], 3);
    assert_eq!(v["usage"]["total_tokens"], v["usage"]["prompt_tokens"].as_u64().unwrap() + 2);

    let r = mock.reqs.lock().unwrap()[0].clone();
    assert_eq!(r.max_tokens, 9, "max_completion_tokens wins");
    assert_eq!((r.min_tokens, r.ignore_eos, r.top_k, r.seed), (1, true, 7, Some(11)));
    assert!((r.temperature - 0.5).abs() < 1e-6 && (r.top_p - 0.9).abs() < 1e-6);
    assert_eq!(r.stop, vec!["END", "STOP"]);
    let (msgs, opts) = mock.templated.lock().unwrap()[0].clone();
    assert_eq!(msgs[1]["content"], "hi there", "content parts flattened");
    assert_eq!(opts.enable_thinking, Some(false));
    assert_eq!(opts.extra.get("foo"), Some(&json!(1)));
    assert!(opts.add_generation_prompt);
}

#[tokio::test]
async fn chat_stream_with_usage() {
    let (app, _) = app();
    let (st, ct, body) = call(
        &app,
        "POST",
        "/v1/chat/completions",
        Some(json!({"messages": [{"role": "user", "content": "hi"}], "stream": true,
                    "stream_options": {"include_usage": true}})),
    )
    .await;
    assert_eq!(st, StatusCode::OK);
    assert!(ct.starts_with("text/event-stream"), "{ct}");
    let data = sse_data(&body);
    assert_eq!(data.last().unwrap(), "[DONE]");
    let chunks: Vec<Value> = data[..data.len() - 1].iter().map(|d| serde_json::from_str(d).unwrap()).collect();
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    let text: String = chunks.iter().filter_map(|c| c["choices"][0]["delta"]["content"].as_str()).collect();
    assert_eq!(text, "Hello");
    let fin = &chunks[chunks.len() - 2];
    assert_eq!(fin["choices"][0]["finish_reason"], "stop");
    let usage = chunks.last().unwrap();
    assert_eq!(usage["choices"], json!([]));
    assert_eq!(usage["usage"]["completion_tokens"], 2);
    assert_eq!(usage["usage"]["prompt_tokens_details"]["cached_tokens"], 3);
    // every chunk shares the id
    let id = &chunks[0]["id"];
    assert!(chunks.iter().all(|c| &c["id"] == id));
    assert!(chunks[0]["usage"].is_null());
}

#[tokio::test]
async fn chat_stream_without_usage_has_no_usage_chunk() {
    let (app, _) = app();
    let (_, _, body) = call(
        &app,
        "POST",
        "/v1/chat/completions",
        Some(json!({"messages": [{"role": "user", "content": "hi"}], "stream": true, "max_tokens": 1})),
    )
    .await;
    let data = sse_data(&body);
    assert!(!body.contains("\"usage\""));
    let fin: Value = serde_json::from_str(&data[data.len() - 2]).unwrap();
    assert_eq!(fin["choices"][0]["finish_reason"], "length");
}

#[tokio::test]
async fn completions_variants() {
    let (app, mock) = app();
    let (st, _, body) = call(&app, "POST", "/v1/completions", Some(json!({"prompt": "abc", "max_tokens": 5}))).await;
    assert_eq!(st, StatusCode::OK, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "text_completion");
    assert_eq!(v["choices"][0]["text"], "Hello");
    assert_eq!(v["usage"]["prompt_tokens"], 3);
    assert_eq!(v["usage"]["prompt_tokens_details"]["cached_tokens"], 3);

    // token id prompt
    let (st, _, _) =
        call(&app, "POST", "/v1/completions", Some(json!({"prompt": [5, 6, 7, 8], "max_tokens": 5}))).await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(mock.reqs.lock().unwrap()[1].prompt, vec![5, 6, 7, 8]);

    // stream
    let (_, _, body) = call(
        &app,
        "POST",
        "/v1/completions",
        Some(json!({"prompt": "abc", "stream": true, "stream_options": {"include_usage": true}})),
    )
    .await;
    let data = sse_data(&body);
    assert_eq!(data.last().unwrap(), "[DONE]");
    let text: String = data[..data.len() - 1]
        .iter()
        .filter_map(|d| serde_json::from_str::<Value>(d).ok())
        .filter_map(|c| c["choices"][0]["text"].as_str().map(String::from))
        .collect();
    assert_eq!(text, "Hello");
    assert!(body.contains("cached_tokens"));
}

#[tokio::test]
async fn bad_requests_use_the_right_error_shape() {
    let (app, _) = app();
    let (st, _, body) = call(&app, "POST", "/v1/chat/completions", Some(json!({"messages": [], }))).await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert!(v["error"]["message"].is_string() && v["error"]["type"] == "invalid_request_error");

    let (st, _, _) =
        call(&app, "POST", "/v1/chat/completions", Some(json!({"messages": [{"role":"user","content":"x"}], "n": 2})))
            .await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
    let (st, _, _) = call(
        &app,
        "POST",
        "/v1/chat/completions",
        Some(json!({"messages": [{"role":"user","content":"x"}], "temperature": -1})),
    )
    .await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
    let (st, _, _) = call(&app, "POST", "/v1/chat/completions", Some(json!({"nope": 1}))).await;
    assert!(st.is_client_error());
    // prompt longer than the 64-token context
    let (st, _, body) = call(&app, "POST", "/v1/completions", Some(json!({"prompt": "x".repeat(100)}))).await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
    assert!(body.contains("context window"));

    let (st, _, body) =
        call(&app, "POST", "/v1/messages", Some(json!({"messages": [{"role":"user","content":"x"}]}))).await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["type"], "error");
    assert_eq!(v["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn anthropic_non_stream() {
    let (app, mock) = app();
    let (st, _, body) = call(
        &app,
        "POST",
        "/v1/messages",
        Some(json!({
            "model": "claude-x", "max_tokens": 16, "system": [{"type": "text", "text": "sys prompt"}],
            "stop_sequences": ["###"], "temperature": 0.0,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        })),
    )
    .await;
    assert_eq!(st, StatusCode::OK, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["type"], "message");
    assert_eq!(v["role"], "assistant");
    assert_eq!(v["content"][0], json!({"type": "text", "text": "Hello"}));
    assert_eq!(v["stop_reason"], "end_turn");
    assert_eq!(v["usage"]["output_tokens"], 2);
    assert!(v["usage"]["input_tokens"].as_u64().unwrap() > 0);
    let (msgs, _) = mock.templated.lock().unwrap()[0].clone();
    assert_eq!(msgs[0], json!({"role": "system", "content": "sys prompt"}));
    assert_eq!(msgs[1]["content"], "hello");
    let r = mock.reqs.lock().unwrap()[0].clone();
    assert_eq!((r.max_tokens, r.stop.clone()), (16, vec!["###".to_string()]));
}

#[tokio::test]
async fn anthropic_stream_event_order() {
    let (app, _) = app();
    let (st, _, body) = call(
        &app,
        "POST",
        "/v1/messages",
        Some(json!({"max_tokens": 1, "stream": true, "messages": [{"role": "user", "content": "hello"}]})),
    )
    .await;
    assert_eq!(st, StatusCode::OK);
    let ev = sse_events(&body);
    let names: Vec<&str> = ev.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop"
        ]
    );
    assert!(ev[0].1["message"]["usage"]["input_tokens"].as_u64().unwrap() > 0);
    assert_eq!(ev[2].1["delta"], json!({"type": "text_delta", "text": "Hel"}));
    assert_eq!(ev[4].1["delta"]["stop_reason"], "max_tokens");
    assert_eq!(ev[4].1["usage"]["output_tokens"], 1);
    for (n, v) in &ev {
        assert_eq!(&v["type"], n);
    }
}

#[tokio::test]
async fn count_tokens_and_thinking_flag() {
    let (app, mock) = app();
    let (st, _, body) = call(
        &app,
        "POST",
        "/v1/messages/count_tokens",
        Some(json!({"system": "s", "messages": [{"role": "user", "content": "hello"}]})),
    )
    .await;
    assert_eq!(st, StatusCode::OK);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["input_tokens"], "system:s\nuser:hello\nassistant:".len());
    assert!(mock.reqs.lock().unwrap().is_empty(), "count_tokens must not generate");

    call(
        &app,
        "POST",
        "/v1/messages",
        Some(
            json!({"max_tokens": 4, "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": "x"}]}),
        ),
    )
    .await;
    assert_eq!(mock.templated.lock().unwrap().last().unwrap().1.enable_thinking, Some(false));
}

#[tokio::test]
async fn models_health_metrics() {
    let (app, _) = app();
    let (st, _, body) = call(&app, "GET", "/v1/models", None).await;
    assert_eq!(st, StatusCode::OK);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["data"][0]["id"], "mock-model");
    let (st, _, _) = call(&app, "GET", "/v1/models/mock-model", None).await;
    assert_eq!(st, StatusCode::OK);
    let (st, _, _) = call(&app, "GET", "/v1/models/other", None).await;
    assert_eq!(st, StatusCode::NOT_FOUND);

    let (st, _, body) = call(&app, "GET", "/health", None).await;
    assert_eq!(st, StatusCode::OK);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["status"], "ok");
    assert_eq!(v["steps_total"], 42);

    // generate once so the histograms have data
    call(&app, "POST", "/v1/completions", Some(json!({"prompt": "abc"}))).await;
    let (st, ct, body) = call(&app, "GET", "/metrics", None).await;
    assert_eq!(st, StatusCode::OK);
    assert!(ct.starts_with("text/plain"));
    for needle in [
        "intarsia_engine_steps_total 42",
        "intarsia_graph_replays_total 11",
        "intarsia_graph_captures_total 2",
        "intarsia_decode_eager_steps_total 5",
        "intarsia_decode_profiled_steps_total 3",
        "# TYPE intarsia_graph_replays_total counter",
        "intarsia_requests_running 0",
        "intarsia_requests_waiting 0",
        "intarsia_tokens_prompt_total 7",
        "intarsia_tokens_generated_total 9",
        "intarsia_tokens_cached_total 3",
        "intarsia_kv_bytes_used 100",
        "intarsia_kv_bytes_total 1000",
        "intarsia_op_seconds_total{op=\"matmul\"} 1.5",
        "intarsia_ttft_seconds_count 1",
        "intarsia_ttft_seconds_bucket{le=\"0.025\"} 1",
        "intarsia_ttft_seconds_bucket{le=\"+Inf\"} 1",
        "intarsia_itl_seconds_count 1",
        "# TYPE intarsia_itl_seconds histogram",
    ] {
        assert!(body.contains(needle), "missing `{needle}` in:\n{body}");
    }
}

#[tokio::test]
async fn events_stream_telemetry() {
    let (app, _) = app();
    let req = Request::builder().uri("/events").body(Body::empty()).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let mut body = resp.into_body();
    let mut text = String::new();
    while text.matches("data: ").count() < 2 {
        let frame =
            tokio::time::timeout(std::time::Duration::from_secs(3), body.frame()).await.unwrap().unwrap().unwrap();
        text.push_str(&String::from_utf8_lossy(frame.data_ref().unwrap()));
    }
    let first = sse_data(&text).remove(0);
    let v: Value = serde_json::from_str(&first).unwrap();
    assert_eq!(v["steps_total"], 42);
    assert_eq!(v["schema_version"], 1);
}

#[tokio::test]
async fn dropping_the_response_cancels_generation() {
    let mock = Arc::new(Mock { slow: true, ..Default::default() });
    let app = app_with(mock.clone());
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(json!({"messages": [{"role": "user", "content": "hi"}], "stream": true}).to_string()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let mut body = resp.into_body();
    let _ = body.frame().await; // first chunk, then the client disconnects
    drop(body);
    for _ in 0..200 {
        if mock.cancelled.load(Ordering::SeqCst) {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("the engine never saw the cancellation");
}

fn slow_post(uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

#[tokio::test]
async fn full_queue_answers_429() {
    let mock = Arc::new(Mock { slow: true, ..Default::default() });
    let cfg = intarsia_server::ServerConfig { max_queue: 1, ..Default::default() };
    let app = intarsia_server::router_with(mock.clone(), &cfg);
    let chat = json!({"messages": [{"role": "user", "content": "hi"}], "stream": true});
    // one running + one waiting fill the queue (the mock never finishes)
    let r1 = app.clone().oneshot(slow_post("/v1/chat/completions", chat.clone())).await.unwrap();
    let r2 = app.clone().oneshot(slow_post("/v1/messages", json!({"max_tokens": 4, "stream": true,
        "messages": [{"role": "user", "content": "hi"}]}))).await.unwrap();
    assert_eq!((r1.status(), r2.status()), (StatusCode::OK, StatusCode::OK));
    let r3 = app.clone().oneshot(slow_post("/v1/chat/completions", chat.clone())).await.unwrap();
    assert_eq!(r3.status(), StatusCode::TOO_MANY_REQUESTS);
    assert!(r3.headers().contains_key("retry-after"));
    let v: Value = serde_json::from_slice(&r3.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(v["error"]["type"], "rate_limit_error");
    let r4 = app.clone().oneshot(slow_post("/v1/messages", json!({"max_tokens": 4,
        "messages": [{"role": "user", "content": "hi"}]}))).await.unwrap();
    assert_eq!(r4.status(), StatusCode::TOO_MANY_REQUESTS);
    let v: Value = serde_json::from_slice(&r4.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(v["error"]["type"], "rate_limit_error");
    let (_, _, m) = call(&app, "GET", "/metrics", None).await;
    assert!(m.contains("intarsia_requests_rejected_total 2"), "{m}");

    // freeing a place (client drops the first response) admits the next request again
    drop(r1);
    for _ in 0..200 {
        let r = app.clone().oneshot(slow_post("/v1/chat/completions", chat.clone())).await.unwrap();
        if r.status() == StatusCode::OK {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("the slot was never released");
}

#[tokio::test]
async fn cancel_flag_is_set_when_the_response_is_dropped() {
    let (app, mock) = app();
    // a completed request must hand the engine a cancel flag
    call(&app, "POST", "/v1/completions", Some(json!({"prompt": "abc"}))).await;
    assert!(mock.reqs.lock().unwrap()[0].cancel.is_some());
}

#[tokio::test]
async fn health_503_when_failed_or_stalled() {
    let mock = Arc::new(Mock { fail: true, ..Default::default() });
    let app = app_with(mock);
    let (st, _, body) = call(&app, "GET", "/health", None).await;
    assert_eq!(st, StatusCode::SERVICE_UNAVAILABLE);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["status"], "failed");
    assert!(v["detail"].as_str().unwrap().contains("boom"));
    // new work is refused with 503 in both dialects
    let (st, _, _) = call(&app, "POST", "/v1/completions", Some(json!({"prompt": "abc"}))).await;
    assert_eq!(st, StatusCode::SERVICE_UNAVAILABLE);
    let (st, _, body) =
        call(&app, "POST", "/v1/messages", Some(json!({"max_tokens": 4, "messages": [{"role":"user","content":"x"}]})))
            .await;
    assert_eq!(st, StatusCode::SERVICE_UNAVAILABLE);
    assert!(body.contains("overloaded_error"));

    // busy engine whose step counter does not move
    let mock = Arc::new(Mock { busy: (1, 0), ..Default::default() });
    let cfg = intarsia_server::ServerConfig { watchdog_s: 0.2, ..Default::default() };
    let app = intarsia_server::router_with(mock, &cfg);
    let (st, _, _) = call(&app, "GET", "/health", None).await;
    assert_eq!(st, StatusCode::OK, "just became busy");
    tokio::time::sleep(std::time::Duration::from_millis(350)).await;
    let (st, _, body) = call(&app, "GET", "/health", None).await;
    assert_eq!(st, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert!(body.contains("\"stalled\""));
}

#[tokio::test]
async fn vllm_gauges_and_plan() {
    let mock = Arc::new(Mock { busy: (1, 2), ..Default::default() });
    let app = app_with(mock);
    let (_, _, m) = call(&app, "GET", "/metrics", None).await;
    assert!(m.contains("\nvllm:num_requests_running 1\n"), "{m}");
    assert!(m.contains("\nvllm:num_requests_waiting 2\n"), "{m}");
    assert!(m.contains("\nintarsia_requests_waiting 2\n"));
    let (st, _, body) = call(&app, "GET", "/v1/intarsia/plan", None).await;
    assert_eq!(st, StatusCode::OK);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["model"], "mock-model");
}

#[tokio::test]
async fn anthropic_usage_excludes_cache_reads_and_reports_stop_sequence() {
    let mock = Arc::new(Mock { stop_seq: Some("###".into()), ..Default::default() });
    let app = app_with(mock);
    let body = json!({"max_tokens": 16, "stop_sequences": ["###"], "messages": [{"role": "user", "content": "hello"}]});
    let (_, _, text) = call(&app, "POST", "/v1/messages", Some(body.clone())).await;
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["stop_reason"], "stop_sequence");
    assert_eq!(v["stop_sequence"], "###");
    // mock: prompt is the templated text, 3 of its tokens were cached
    let prompt = "user:hello\nassistant:".len() as u64;
    assert_eq!(v["usage"]["input_tokens"], prompt - 3);
    assert_eq!(v["usage"]["cache_read_input_tokens"], 3);

    let mut b = body;
    b["stream"] = json!(true);
    let (_, _, text) = call(&app, "POST", "/v1/messages", Some(b)).await;
    let ev = sse_events(&text);
    let d = &ev.iter().find(|(n, _)| n == "message_delta").unwrap().1;
    assert_eq!(d["delta"]["stop_reason"], "stop_sequence");
    assert_eq!(d["delta"]["stop_sequence"], "###");
    assert_eq!(d["usage"]["input_tokens"], prompt - 3);
    assert_eq!(d["usage"]["cache_read_input_tokens"], 3);
}

#[tokio::test]
async fn default_max_tokens_is_bounded() {
    let (app, mock) = app();
    // 64-token context, prompt "abc" (3 tokens): default = min(context - prompt, 4096) = 61
    call(&app, "POST", "/v1/completions", Some(json!({"prompt": "abc"}))).await;
    assert_eq!(mock.reqs.lock().unwrap()[0].max_tokens, 61);
    let mock = Arc::new(Mock::default());
    let cfg = intarsia_server::ServerConfig { default_max_tokens: 10, ..Default::default() };
    let app = intarsia_server::router_with(mock.clone(), &cfg);
    call(&app, "POST", "/v1/completions", Some(json!({"prompt": "abc"}))).await;
    assert_eq!(mock.reqs.lock().unwrap()[0].max_tokens, 10);
}


/// Engine-shaped scripted stream: an empty-text token, and a tail event that repeats the last id
/// (text held back at the end) after three real tokens plus the uncounted-in-events EOS.
fn scripted_mock() -> Arc<Mock> {
    Arc::new(Mock {
        script: Some((vec![(10, "He"), (11, ""), (12, "llo"), (12, "!")], 4, FinishReason::Stop)),
        ..Default::default()
    })
}

#[tokio::test]
async fn token_ids_non_stream_and_default() {
    let app = app_with(scripted_mock());
    for (uri, body) in [
        ("/v1/chat/completions", json!({"messages": [{"role": "user", "content": "hi"}]})),
        ("/v1/completions", json!({"prompt": "hi"})),
    ] {
        let (_, _, plain) = call(&app, "POST", uri, Some(body.clone())).await;
        assert!(!plain.contains("token_ids"), "{uri}: default response must not carry token_ids: {plain}");
        let mut b = body.clone();
        b["return_token_ids"] = json!(false);
        let (_, _, off) = call(&app, "POST", uri, Some(b)).await;
        assert!(!off.contains("token_ids"), "{uri}");
        let mut b = body;
        b["return_token_ids"] = json!(true);
        let (st, _, on) = call(&app, "POST", uri, Some(b)).await;
        assert_eq!(st, StatusCode::OK);
        let v: Value = serde_json::from_str(&on).unwrap();
        // the tail event's repeated id is dropped, the text is kept
        assert_eq!(v["choices"][0]["token_ids"], json!([10, 11, 12]), "{uri}");
        let text = v["choices"][0]["text"].as_str().or(v["choices"][0]["message"]["content"].as_str()).unwrap();
        assert_eq!(text, "Hello!");
        assert_eq!(v["usage"]["completion_tokens"], 4);
    }
}

#[tokio::test]
async fn token_ids_stream_and_default() {
    let app = app_with(scripted_mock());
    for (uri, body) in [
        ("/v1/chat/completions", json!({"messages": [{"role": "user", "content": "hi"}], "stream": true})),
        ("/v1/completions", json!({"prompt": "hi", "stream": true})),
    ] {
        let (_, _, plain) = call(&app, "POST", uri, Some(body.clone())).await;
        assert!(!plain.contains("token_ids"), "{uri}: default stream must not carry token_ids");
        let mut b = body;
        b["return_token_ids"] = json!(true);
        let (_, _, on) = call(&app, "POST", uri, Some(b)).await;
        let data = sse_data(&on);
        assert_eq!(data.last().unwrap(), "[DONE]");
        let chunks: Vec<Value> = data[..data.len() - 1].iter().map(|d| serde_json::from_str(d).unwrap()).collect();
        let mut ids = Vec::new();
        let mut text = String::new();
        for c in &chunks {
            let ch = &c["choices"][0];
            let carrier = if ch.get("delta").is_some() { &ch["delta"] } else { ch };
            if let Some(a) = carrier["token_ids"].as_array() {
                ids.extend(a.iter().map(|x| x.as_u64().unwrap()));
            }
            if let Some(t) = carrier["content"].as_str().or(carrier["text"].as_str()) {
                text.push_str(t);
            }
        }
        assert_eq!(ids, vec![10, 11, 12], "{uri}");
        assert_eq!(text, "Hello!", "{uri}");
        let fin = &chunks[chunks.len() - 1]["choices"][0];
        assert_eq!(fin["finish_reason"], "stop");
    }
}

#[tokio::test]
async fn token_ids_length_finish_keeps_all() {
    // no stop token, no tail: every event is a real token, even a repeated id
    let mock = Arc::new(Mock {
        script: Some((vec![(7, "a"), (7, "b")], 2, FinishReason::Length)),
        ..Default::default()
    });
    let app = app_with(mock);
    let (_, _, on) =
        call(&app, "POST", "/v1/completions", Some(json!({"prompt": "hi", "return_token_ids": true}))).await;
    let v: Value = serde_json::from_str(&on).unwrap();
    assert_eq!(v["choices"][0]["token_ids"], json!([7, 7]));
    let (_, _, on) = call(
        &app,
        "POST",
        "/v1/completions",
        Some(json!({"prompt": "hi", "return_token_ids": true, "stream": true})),
    )
    .await;
    assert_eq!(on.matches("\"token_ids\":[7]").count(), 2);
}
