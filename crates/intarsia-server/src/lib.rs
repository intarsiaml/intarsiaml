//! HTTP server over the core [`Engine`]: OpenAI (`/v1/chat/completions`,
//! `/v1/completions`, `/v1/models`), Anthropic (`/v1/messages`,
//! `/v1/messages/count_tokens`), `/health`, `/metrics` (Prometheus text) and
//! `/events` (SSE stream of the telemetry snapshot).
//!
//! The engine is synchronous (one worker thread, FIFO queue); every request is
//! submitted to it and a small bridge thread forwards its events to the async
//! side. Dropping the HTTP response drops the bridge, which drops the engine's
//! receiver, which cancels the generation.

mod anthropic;
mod api;
mod metrics;
mod openai;

use api::{blocking, AppState};
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::Router;
use intarsia_core::Engine;
use serde_json::json;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

pub use metrics::Metrics;

#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    /// Model name reported in responses and `/v1/models` (default: the engine's model id).
    pub served_model_name: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig { host: "127.0.0.1".into(), port: 8090, served_model_name: None }
    }
}

/// Build the router (exposed for tests and for embedding).
pub fn router(engine: Arc<dyn Engine>, served_model_name: Option<String>) -> Router {
    let name = served_model_name.unwrap_or_else(|| engine.info().id);
    let state = AppState::new(engine, name);
    Router::new()
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/completions", post(openai::completions))
        .route("/v1/models", get(openai::models))
        .route("/v1/models/{id}", get(openai::model))
        .route("/v1/messages", post(anthropic::messages))
        .route("/v1/messages/count_tokens", post(anthropic::count_tokens))
        .route("/health", get(health))
        .route("/metrics", get(prometheus))
        .route("/events", get(events))
        .with_state(state)
}

/// Run the server until Ctrl-C / SIGTERM. Blocks the calling thread.
pub fn serve(engine: Arc<dyn Engine>, cfg: ServerConfig) -> intarsia_core::Result<()> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .thread_name("intarsia-http")
        .build()
        .map_err(|e| intarsia_core::Error::Other(format!("tokio runtime: {e}")))?;
    rt.block_on(async move {
        let addr = format!("{}:{}", cfg.host, cfg.port);
        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .map_err(|e| intarsia_core::Error::Other(format!("bind {addr}: {e}")))?;
        let app = router(engine, cfg.served_model_name.clone());
        tracing::info!("intarsia server listening on http://{addr}");
        eprintln!("intarsia server listening on http://{addr}");
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|e| intarsia_core::Error::Other(format!("server: {e}")))
    })
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        if let Ok(mut s) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            s.recv().await;
        } else {
            std::future::pending::<()>().await;
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! { _ = ctrl_c => {}, _ = term => {} }
}

async fn health(State(st): State<AppState>) -> Response {
    let snap = match blocking(&st.engine, |e| Ok(e.telemetry())).await {
        Ok(s) => s,
        Err(e) => return e.into_response(),
    };
    let now =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0);
    let busy = snap.running + snap.waiting > 0;
    // Watchdog hint: work is queued/running but no engine step for a long time.
    let stalled = busy && snap.last_step_ms > 0 && now.saturating_sub(snap.last_step_ms) > 60_000;
    Json(json!({
        "status": if stalled { "stalled" } else { "ok" },
        "model": st.model_name,
        "steps_total": snap.steps_total,
        "last_step_ms": snap.last_step_ms,
        "running": snap.running,
        "waiting": snap.waiting,
        "requests_total": snap.requests_total,
        "uptime_s": snap.uptime_s,
    }))
    .into_response()
}

async fn prometheus(State(st): State<AppState>) -> Response {
    let r = blocking(&st.engine, |e| Ok((e.telemetry(), e.info()))).await;
    match r {
        Ok((snap, info)) => {
            ([("content-type", "text/plain; version=0.0.4; charset=utf-8")], metrics::render(&snap, &info, &st.metrics))
                .into_response()
        }
        Err(e) => e.into_response(),
    }
}

async fn events(State(st): State<AppState>) -> Sse<impl futures_util::Stream<Item = Result<Event, Infallible>>> {
    let tick = tokio::time::interval(Duration::from_millis(200));
    let stream = futures_util::stream::unfold((st, tick), |(st, mut tick)| async move {
        tick.tick().await;
        let snap = blocking(&st.engine, |e| Ok(e.telemetry())).await.ok()?;
        let ev = Event::default().json_data(&snap).unwrap_or_else(|_| Event::default().data("{}"));
        Some((Ok(ev), (st, tick)))
    });
    Sse::new(stream).keep_alive(KeepAlive::default())
}
