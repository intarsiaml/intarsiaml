//! The prefix store through the HTTP API, end to end on the cpu backend: a real `LocalEngine` (the tiny hybrid fixture,
//! GatedDeltaNet + attention) behind the router, a ChatML codec. Covers `POST /v1/intarsia/prefill` (a prewarm while the
//! user types, `partial: true`), the following chat request reusing it (`usage.prompt_tokens_details.cached_tokens`),
//! `GET /v1/intarsia/prefix-store` and the `intarsia_prefix_store_*` families of `/metrics`.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use http_body_util::BodyExt;
use intarsia_core::engine::{ChatOpts, TextCodec};
use intarsia_core::model::{LoadOptions, Model};
use intarsia_core::prefix_store::PrefixStoreConfig;
use intarsia_core::recipe::ModelDims;
use intarsia_core::{DType, EngineMeta, LocalEngine, Telemetry};
use intarsia_models::SafeTensors;
use serde_json::{json, Value};
use std::path::Path;
use std::sync::Arc;
use tower::ServiceExt;

const SPECIALS: [(&str, u32); 4] = [("<|im_start|>", 3), ("<|im_end|>", 6), ("<think>", 4), ("\n", 5)];

/// ChatML specials as single tokens, every other byte one token in 10..90.
struct ChatMl;

impl TextCodec for ChatMl {
    fn encode(&self, text: &str, _s: bool) -> intarsia_core::Result<Vec<u32>> {
        let mut out = Vec::new();
        let mut rest = text;
        'outer: while !rest.is_empty() {
            for (s, id) in SPECIALS {
                if let Some(r) = rest.strip_prefix(s) {
                    out.push(id);
                    rest = r;
                    continue 'outer;
                }
            }
            out.push(10 + rest.as_bytes()[0] as u32 % 80);
            rest = &rest[1..];
        }
        Ok(out)
    }
    fn decode(&self, ids: &[u32], _s: bool) -> intarsia_core::Result<String> {
        Ok(ids.iter().map(|&i| if i >= 10 { (b'0' + (i % 40) as u8) as char } else { ' ' }).collect())
    }
    fn apply_chat_template(&self, messages: &[Value], opts: &ChatOpts) -> intarsia_core::Result<String> {
        let mut s = String::new();
        for m in messages {
            s.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", m["role"].as_str().unwrap_or(""), m["content"].as_str().unwrap_or("")));
        }
        if opts.add_generation_prompt {
            s.push_str("<|im_start|>assistant\n<think>\n");
        }
        Ok(s)
    }
    fn has_chat_template(&self) -> bool {
        true
    }
}

fn engine(store: bool) -> Arc<LocalEngine> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../intarsia-models/tests/fixtures/tiny_qwen3_5");
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/qwen3_5.toml")).unwrap();
    let cfg: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    let src = SafeTensors::open_dir(&dir).unwrap();
    let m = Model::load(
        intarsia_backend_cpu::CpuBackend::arc(),
        &recipe,
        &dims,
        &src,
        &LoadOptions { max_seq: 400, prefill_chunk: 16, act_dtype: Some(DType::F32), kv_dtype: Some(DType::F32), ..Default::default() },
        Arc::new(Telemetry::new()),
    )
    .unwrap();
    let ps = store.then(|| PrefixStoreConfig { min_tokens: 4, page_tokens: 16, ..Default::default() });
    Arc::new(
        LocalEngine::new(m, Arc::new(ChatMl), EngineMeta { id: "tiny".into(), eos_ids: vec![0], prefix_store: ps, ..Default::default() }, Arc::new(Telemetry::new()))
            .unwrap(),
    )
}

async fn call(app: &Router, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, String) {
    let req = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json")
        .body(body.map(|b| Body::from(b.to_string())).unwrap_or_else(Body::empty))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let st = resp.status();
    let b = resp.into_body().collect().await.unwrap().to_bytes();
    (st, String::from_utf8(b.to_vec()).unwrap())
}

fn history() -> Vec<Value> {
    vec![
        json!({"role": "system", "content": "be terse and cite the documents"}),
        json!({"role": "user", "content": "first question about the document"}),
        json!({"role": "assistant", "content": "a short answer from turn one"}),
    ]
}

async fn chat(app: &Router, msgs: &[Value]) -> Value {
    let (st, body) = call(app, "POST", "/v1/chat/completions", Some(json!({"model": "tiny", "messages": msgs, "max_tokens": 8, "temperature": 0}))).await;
    assert_eq!(st, StatusCode::OK, "{body}");
    serde_json::from_str(&body).unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn prewarm_then_request_reuses_and_metrics_show_it() {
    let app = intarsia_server::router(engine(true), None);
    // the user is typing the next message: prewarm what is there
    let mut typing = history();
    typing.push(json!({"role": "user", "content": "second question, still being typ"}));
    let (st, body) = call(&app, "POST", "/v1/intarsia/prefill", Some(json!({"messages": typing, "partial": true}))).await;
    assert_eq!(st, StatusCode::OK, "{body}");
    let pre: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(pre["cached_tokens"], 0);
    assert!(pre["computed_tokens"].as_u64().unwrap() > 100, "{pre}");

    // an unrelated request in between replaces the engine's held sequence
    let other = chat(&app, &[json!({"role": "user", "content": "something else entirely, from another tab"})]).await;
    assert_eq!(other["usage"]["prompt_tokens_details"]["cached_tokens"], 0);

    // the message is sent: everything up to the start of the typed message comes from the store
    let mut full = history();
    full.push(json!({"role": "user", "content": "second question, still being typed and now finished"}));
    let r = chat(&app, &full).await;
    let prompt = r["usage"]["prompt_tokens"].as_u64().unwrap();
    let cached = r["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap();
    let hist_len = ChatMl.encode(&ChatMl.apply_chat_template(&history(), &ChatOpts::default()).unwrap(), false).unwrap().len() as u64;
    println!("PREWARM-HTTP prompt {prompt} cached {cached} history {hist_len} prewarmed {}", pre["prompt_tokens"]);
    assert!(cached >= hist_len, "reused {cached} of the {hist_len}-token history");

    // same answer as a cold engine with the store on
    let cold = chat(&intarsia_server::router(engine(true), None), &full).await;
    assert_eq!(cold["choices"][0]["message"]["content"], r["choices"][0]["message"]["content"]);

    let (st, body) = call(&app, "GET", "/v1/intarsia/prefix-store", None).await;
    assert_eq!(st, StatusCode::OK);
    let info: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(info["enabled"], true);
    assert!(info["stats"]["hits"].as_u64().unwrap() >= 1, "{info}");
    assert!(!info["entries"].as_array().unwrap().is_empty());

    let (_, metrics) = call(&app, "GET", "/metrics", None).await;
    for fam in ["intarsia_prefix_store_hits_total", "intarsia_prefix_store_tokens_saved_total", "intarsia_prefix_store_tier_bytes{tier=\"host\"}"] {
        assert!(metrics.contains(fam), "{fam} missing from /metrics");
    }
    let hits = metrics.lines().find(|l| l.starts_with("intarsia_prefix_store_hits_total ")).unwrap();
    assert_ne!(hits, "intarsia_prefix_store_hits_total 0");
}

#[tokio::test(flavor = "multi_thread")]
async fn pinned_prefill_and_store_off() {
    let app = intarsia_server::router(engine(true), None);
    let (st, body) = call(&app, "POST", "/v1/intarsia/prefill", Some(json!({"prompt": "<|im_start|>system\ntools: list_servers read_log disk_usage<|im_end|>\n", "pin": "tools"}))).await;
    assert_eq!(st, StatusCode::OK, "{body}");
    let (_, body) = call(&app, "GET", "/v1/intarsia/prefix-store", None).await;
    let info: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(info["stats"]["pinned_entries"], 1, "{info}");

    let off = intarsia_server::router(engine(false), None);
    let (st, _) = call(&off, "POST", "/v1/intarsia/prefill", Some(json!({"prompt": "abc"}))).await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
    let (_, body) = call(&off, "GET", "/v1/intarsia/prefix-store", None).await;
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["enabled"], false);
    let (_, metrics) = call(&off, "GET", "/metrics", None).await;
    assert!(!metrics.contains("intarsia_prefix_store"));
}
