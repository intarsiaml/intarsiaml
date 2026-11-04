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

/// How [`serve`]'s error starts when the listening socket could not be had.
const BIND_FAILED: &str = "bind";

/// Whether [`serve`] failed because it could not bind, after [`ServerConfig::bind_retry_s`]: a supervised child exits [`EXIT_FATAL`] then,
/// since a restart would reload the whole model only to try the same port again.
pub fn is_bind_failure(e: &intarsia_core::Error) -> bool {
    matches!(e, intarsia_core::Error::Other(m) if m.starts_with(BIND_FAILED))
}

pub use health::{EXIT_FATAL, EXIT_FAILED, EXIT_ORPHANED, EXIT_STALLED};
pub use metrics::Metrics;

#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// Seconds to keep trying a port that is in use or an address not there yet (0 = fail at once). The model is already loaded when the
    /// server binds, so this costs no reload.
    pub bind_retry_s: f64,
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
    /// Exit the process (code [`EXIT_FAILED`]) once the engine reports itself failed, even without a supervisor (a tensor-parallel
    /// group that lost a rank cannot recover in this process; whatever launched it restarts both ranks).
    pub exit_on_failed: bool,
    /// Exit when this parent process is gone (the supervisor).
    pub parent_pid: Option<u32>,
    /// Serve the built-in web UI at `/` and `/ui/*` (`--no-ui` turns it off).
    pub ui: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            bind_retry_s: 0.0,
            host: "127.0.0.1".into(),
            port: 8090,
            served_model_name: None,
            max_queue: 64,
            max_running: 1,
            default_max_tokens: 4096,
            watchdog_s: 60.0,
            exit_on_stall: false,
            exit_on_failed: false,
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
    router_with_shutdown(engine, cfg).0
}

/// [`router_with`] and the handle that ends its endless streams at shutdown (see [`Shutdown`]).
pub fn router_with_shutdown(engine: Arc<dyn Engine>, cfg: &ServerConfig) -> (Router, Shutdown) {
    let name = cfg.served_model_name.clone().unwrap_or_else(|| engine.info().id);
    let state = AppState::new(engine, name, cfg);
    let stop = Shutdown(state.shutdown.clone());
    (app(state, cfg.ui), stop)
}

/// Ends the streams that never finish by themselves when the server is told to stop. axum's graceful shutdown waits for
/// every open response, and `/events` (held open for good by `intarsia top` and the TUI, which reconnect forever) kept the
/// process, and the GPU memory it holds, alive until the supervisor's SIGKILL 30 s later. Generations in flight still
/// finish: their length is bounded.
pub struct Shutdown(Arc<tokio::sync::watch::Sender<bool>>);

impl Shutdown {
    /// Resolves with `signal`, after ending the `/events` streams: pass it to `with_graceful_shutdown`.
    pub async fn after(self, signal: impl std::future::Future<Output = ()>) {
        signal.await;
        self.0.send_replace(true);
    }
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
        let name = cfg.served_model_name.clone().unwrap_or_else(|| engine.info().id);
        let state = AppState::new(engine.clone(), name, &cfg);
        let stop = Shutdown(state.shutdown.clone());
        let probe = engine.clone();
        if wants_watchdog(&cfg) {
            health::spawn_watchdog(
                engine,
                state.progress.clone(),
                state.metrics.clone(),
                cfg.exit_on_stall,
                cfg.exit_on_failed,
                cfg.parent_pid,
            );
        }
        // After the watchdog, which also covers the retry: a failed group still exits 87 and an orphan still leaves.
        let Some(listener) = bind_retrying(&addr, cfg.bind_retry_s, &*probe).await? else {
            return Ok(()); // stopped while waiting for the port
        };
        let app = app(state, cfg.ui);
        tracing::info!("intarsia server listening on http://{addr}");
        eprintln!("intarsia server listening on http://{addr}");
        axum::serve(listener, app)
            .with_graceful_shutdown(stop.after(shutdown_signal()))
            .await
            .map_err(|e| intarsia_core::Error::Other(format!("server: {e}")))
    })
}

/// Bind `addr`, retrying what can clear by itself for up to `retry_s`: a port a stopping server still holds (a restart after exit 86 or 87
/// can get there first), an address the network has not brought up yet. Anything else fails at once. `None`: asked to stop meanwhile.
async fn bind_retrying(addr: &str, retry_s: f64, engine: &dyn Engine) -> intarsia_core::Result<Option<tokio::net::TcpListener>> {
    use std::io::ErrorKind::{AddrInUse, AddrNotAvailable};
    let t0 = std::time::Instant::now();
    let mut told = false;
    let stop = shutdown_signal();
    tokio::pin!(stop);
    loop {
        let e = match tokio::net::TcpListener::bind(addr).await {
            Ok(l) => return Ok(Some(l)),
            Err(e) => e,
        };
        // the watchdog waits 2 s before it exits 87: a retry that ends meanwhile must not turn that into a fatal 78
        if retry_s > 0.0 && engine.failed().is_some() {
            eprintln!("cannot bind {addr} ({e}) and the engine failed; exiting for restart");
            std::process::exit(EXIT_FAILED);
        }
        if !matches!(e.kind(), AddrInUse | AddrNotAvailable) || t0.elapsed().as_secs_f64() >= retry_s {
            return Err(intarsia_core::Error::Other(format!("{BIND_FAILED} {addr}: {e}")));
        }
        if !told {
            eprintln!("cannot bind {addr} ({e}); retrying for up to {retry_s} s");
            told = true;
        }
        tokio::select! {
            _ = &mut stop => return Ok(None),
            _ = tokio::time::sleep(std::time::Duration::from_millis(500)) => {}
        }
    }
}

/// Whether `serve` runs the watchdog: a stall limit, a supervisor to watch for, or an exit on a failed engine (which only the
/// watchdog performs: with `--watchdog-s 0` and no supervisor it was never started, so a failed engine never exited).
fn wants_watchdog(cfg: &ServerConfig) -> bool {
    cfg.watchdog_s > 0.0 || cfg.parent_pid.is_some() || cfg.exit_on_failed
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
    let mut stop = st.shutdown.subscribe();
    let stream = futures_util::stream::unfold((st, tick), |(st, mut tick)| async move {
        tick.tick().await;
        let snap = blocking(&st.engine, |e| Ok(e.telemetry())).await.ok()?;
        let ev = Event::default().json_data(&snap).unwrap_or_else(|_| Event::default().data("{}"));
        Some((Ok(ev), (st, tick)))
    });
    // ends with the server's shutdown, or the graceful shutdown would wait for this stream for good
    let stream = futures_util::StreamExt::take_until(stream, async move {
        let _ = stop.wait_for(|stopping| *stopping).await;
    });
    Sse::new(stream).keep_alive(KeepAlive::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_failed_engine_exit_needs_the_watchdog_even_without_a_stall_limit() {
        let off = ServerConfig { watchdog_s: 0.0, ..Default::default() };
        assert!(!wants_watchdog(&off));
        assert!(wants_watchdog(&ServerConfig { exit_on_failed: true, ..off.clone() }));
        assert!(wants_watchdog(&ServerConfig { parent_pid: Some(1), ..off.clone() }));
        assert!(wants_watchdog(&ServerConfig { watchdog_s: 60.0, ..off }));
    }
}
