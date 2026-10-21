//! Background data sources, one thread each, all feeding one channel of [`Event`]s:
//! `/events` (SSE telemetry, via the bench crate's follower that `intarsia top` also uses), `/v1/intarsia/plan` (once),
//! `/metrics` (about once a second) and `/v1/models` (for the chat's model name).

use crate::chat::{self, ChatEvent, ChatMsg};
use crate::layout::Plan;
use intarsia_bench::http;
use intarsia_bench::snapshot::Snapshot;
use intarsia_bench::top::{spawn_sse, Msg};
use serde_json::Value;
use std::sync::mpsc::{self, Sender};
use std::time::Duration;

pub enum Event {
    Snap(Snapshot),
    Status(String),
    /// `None`: the engine has no plan (`null`).
    Plan(Option<Plan>),
    Metrics(String),
    /// The id the server lists first on `/v1/models`.
    Model(String),
    Chat(ChatEvent),
}

pub const METRICS_EVERY: Duration = Duration::from_secs(1);

pub fn spawn_all(origin: &str, tx: Sender<Event>) {
    // telemetry: the follower speaks the bench crate's Msg; translate
    let (mtx, mrx) = mpsc::channel::<Msg>();
    spawn_sse(origin, mtx);
    let t = tx.clone();
    std::thread::spawn(move || {
        for m in mrx {
            let ev = match m {
                Msg::Snap(s) => Event::Snap(s),
                Msg::Status(s) => Event::Status(s),
            };
            if t.send(ev).is_err() {
                return;
            }
        }
    });
    let origin = origin.to_string();
    std::thread::spawn(move || {
        let agent = http::agent(Some(Duration::from_secs(5)));
        let get = |path: &str| -> Option<String> {
            let mut r = agent.get(format!("{origin}{path}")).call().ok()?;
            if !r.status().is_success() {
                return None;
            }
            r.body_mut().read_to_string().ok()
        };
        let (mut have_plan, mut have_model) = (false, false);
        loop {
            if !have_plan {
                if let Some(v) = get("/v1/intarsia/plan").and_then(|t| serde_json::from_str::<Value>(&t).ok()) {
                    have_plan = true;
                    if tx.send(Event::Plan(Plan::from_json(&v))).is_err() {
                        return;
                    }
                }
            }
            if !have_model {
                let id = get("/v1/models")
                    .and_then(|t| serde_json::from_str::<Value>(&t).ok())
                    .and_then(|v| v["data"][0]["id"].as_str().map(str::to_string));
                if let Some(id) = id {
                    have_model = true;
                    let _ = tx.send(Event::Model(id));
                }
            }
            if let Some(text) = get("/metrics") {
                if tx.send(Event::Metrics(text)).is_err() {
                    return;
                }
            }
            std::thread::sleep(METRICS_EVERY);
        }
    });
}

/// Stream one chat turn on its own thread; events arrive as [`Event::Chat`].
pub fn spawn_chat(origin: &str, model: &str, history: &[ChatMsg], max_tokens: usize, thinking: bool, tx: Sender<Event>) {
    let body = chat::request_body(model, history, max_tokens, thinking);
    let origin = origin.to_string();
    std::thread::spawn(move || {
        let t = tx.clone();
        if let Err(e) = chat::stream(&origin, &body, move |ev| t.send(Event::Chat(ev)).is_ok()) {
            let _ = tx.send(Event::Chat(ChatEvent::Error(e.to_string())));
        }
    });
}
