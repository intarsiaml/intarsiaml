//! HTTP server over the core [`Engine`]: OpenAI (`/v1/chat/completions`,
//! `/v1/completions`, `/v1/models`), Anthropic (`/v1/messages`,
//! `/v1/messages/count_tokens`), `/health`, `/metrics` (Prometheus text) and
//! `/events` (SSE stream of the telemetry snapshot).
//!
//! The engine is synchronous (one worker thread, FIFO queue); every request is
//! submitted to it and a small bridge thread forwards its events to the async
//! side. Dropping the HTTP response sets the request's cancel flag, which the
//! engine checks between prefill chunks and decode steps. The queue is bounded
//! (429 when full), `/health` turns 503 when the engine failed or stopped
//! stepping with work queued, and [`supervisor`] restarts a server process
//! that exits or trips its watchdog.

mod anthropic;
mod api;
pub mod health;
mod metrics;
mod openai;
mod prefix;
mod structured;
pub mod supervisor;

use api::{blocking, AppState};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::Router;
use intarsia_core::Engine;
use serde_json::json;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

pub use health::{EXIT_FATAL, EXIT_FAILED, EXIT_ORPHANED, EXIT_STALLED};
pub use metrics::Metrics;

#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    /// Model name reported in responses and `/v1/models` (default: the engine's model id).
    pub served_model_name: Option<String>,
    /// Requests that may wait behind the running ones before new ones get 429.
    pub max_queue: usize,
    /// Requests the engine runs at once (`--max-batch`; 1 for the single-sequence engine).
    pub max_running: usize,
    /// `max_tokens` when a request names none, bounded by the context (0 = only the context).
    pub default_max_tokens: usize,
    /// Seconds without an engine step, with work queued, before the server reports itself
    /// stalled (503 on `/health`); 0 disables.
    pub watchdog_s: f64,
    /// Exit the process (code [`EXIT_STALLED`] / [`EXIT_FAILED`]) when the watchdog trips,
    /// for a supervisor to restart.
    pub exit_on_stall: bool,
    /// Exit when this parent process is gone (the supervisor).
    pub parent_pid: Option<u32>,
    /// Serve the built-in web UI at `/` and `/ui/*` (`--no-ui` turns it off).
    pub ui: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            host: "127.0.0.1".into(),
            port: 8090,
            served_model_name: None,
            max_queue: 64,
            max_running: 1,
            default_max_tokens: 4096,
            watchdog_s: 60.0,
            exit_on_stall: false,
            parent_pid: None,
            ui: true,
        }
    }
}

/// Build the router with default limits (exposed for tests and for embedding).
pub fn router(engine: Arc<dyn Engine>, served_model_name: Option<String>) -> Router {
    router_with(engine, &ServerConfig { served_model_name, ..Default::default() })
}

/// Build the router from a full config.
pub fn router_with(engine: Arc<dyn Engine>, cfg: &ServerConfig) -> Router {
    let name = cfg.served_model_name.clone().unwrap_or_else(|| engine.info().id);
    let state = AppState::new(engine, name, cfg);
    app(state, cfg.ui)
}

fn app(state: AppState, ui: bool) -> Router {
    let router = Router::new()
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/completions", post(openai::completions))
        .route("/v1/models", get(openai::models))
        .route("/v1/models/{id}", get(openai::model))
        .route("/v1/messages", post(anthropic::messages))
        .route("/v1/messages/count_tokens", post(anthropic::count_tokens))
        .route("/health", get(health))
        .route("/metrics", get(prometheus))
        .route("/events", get(events))
        .route("/v1/intarsia/plan", get(plan))
        .route("/v1/intarsia/prefix-store", get(prefix::store_info))
        .route("/v1/intarsia/prefill", post(prefix::prefill))
        .with_state(state);
    if ui {
        router.merge(intarsia_ui::router())
    } else {
        router
    }
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
        let name = cfg.served_model_name.clone().unwrap_or_else(|| engine.info().id);
        let state = AppState::new(engine.clone(), name, &cfg);
        if cfg.watchdog_s > 0.0 || cfg.parent_pid.is_some() {
            health::spawn_watchdog(
                engine,
                state.progress.clone(),
                state.metrics.clone(),
                cfg.exit_on_stall,
                cfg.parent_pid,
            );
        }
        let app = app(state, cfg.ui);
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
    let (status, no_progress) = st.progress.status(&*st.engine, &snap);
    st.metrics.set_health(matches!(status, health::Status::Failed(_)), no_progress);
    let (label, code, detail) = match &status {
        health::Status::Ok => ("ok", StatusCode::OK, None),
        health::Status::Stalled(d) => (
            "stalled",
            StatusCode::SERVICE_UNAVAILABLE,
            Some(format!("no engine step for {:.0}s with work queued", d.as_secs_f64())),
        ),
        health::Status::Failed(why) => ("failed", StatusCode::SERVICE_UNAVAILABLE, Some(why.clone())),
    };
    (
        code,
        Json(json!({
            "status": label,
            "detail": detail,
            "model": st.model_name,
            "steps_total": snap.steps_total,
            "last_step_ms": snap.last_step_ms,
            "no_progress_s": no_progress.as_secs_f64(),
            "running": snap.running,
            "waiting": snap.waiting,
            "max_queue": st.max_queue,
            "requests_total": snap.requests_total,
            "uptime_s": snap.uptime_s,
        })),
    )
        .into_response()
}

/// The execution plan (`Engine::plan`): layers, blocks and the kernel chosen per op.
async fn plan(State(st): State<AppState>) -> Response {
    match blocking(&st.engine, |e| Ok(e.plan())).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => e.into_response(),
    }
}

async fn prometheus(State(st): State<AppState>) -> Response {
    let r = blocking(&st.engine, |e| Ok((e.telemetry(), e.info(), e.batch_metrics()))).await;
    match r {
        Ok((snap, info, batch)) => {
            let mut body = metrics::render(&snap, &info, &st.metrics);
            if let Some(b) = batch {
                body.push_str(&metrics::render_batch(&b));
            }
            if let Some(p) = &snap.prefix_store {
                body.push_str(&metrics::render_prefix_store(p));
            }
            ([("content-type", "text/plain; version=0.0.4; charset=utf-8")], body).into_response()
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
