//! `intarsia serve`: OpenAI / Anthropic API server.

use crate::common::ModelArgs;
use anyhow::Result;
use intarsia_core::Engine;
use std::sync::Arc;

#[derive(clap::Args, Debug)]
pub struct ServeArgs {
    /// Address to bind.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    /// Port to listen on (spark1 uses 8090 for Intarsia; 8000 belongs to vLLM).
    #[arg(long, default_value_t = 8090)]
    pub port: u16,
    /// Model name reported by the API (default: the model id).
    #[arg(long)]
    pub served_model_name: Option<String>,
    /// Requests allowed to wait behind the running one; further requests get HTTP 429.
    #[arg(long, default_value_t = 64)]
    pub max_queue: usize,
    /// Opt-in continuous batching: decode up to N sequences together, N in 2..=8. Metal only; refused on other
    /// backends and on models with recurrent layers. Default 1: the single-sequence engine. Only `serve` takes this flag.
    #[arg(long)]
    pub max_batch: Option<usize>,
    /// With `--max-batch` and `--spec`: speculative drafting runs only while fewer than N sequences are decoding
    /// (0..=9; 0 and 1 turn it off for every batch, 9 keeps it on at every size). No effect without `--spec` or without
    /// `--max-batch`.
    #[arg(long, default_value_t = intarsia_core::scheduler::SPEC_OFF_AT_DEFAULT as u8, value_parser = clap::value_parser!(u8).range(0..=9))]
    pub spec_off_at: u8,
    /// With `--max-batch`: prefill chunk (tokens) of a prompt that starts prefilling while other sequences decode
    /// (16..=2048, capped at the model's `--prefill-chunk`). A prompt that starts while nothing decodes uses the model
    /// chunk. Smaller chunks stall the decoding streams less per iteration; larger ones finish the prompt in fewer
    /// iterations. No effect without `--max-batch`. Default 256; 1024 for the hybrid mixture-of-experts models on CUDA, whose
    /// prefill chunk costs almost the same at any size (docs/results/2026-10-23-rb2-cuda-batching.md).
    #[arg(long, value_parser = clap::value_parser!(u16).range(16..=2048))]
    pub busy_chunk: Option<u16>,
    /// `max_tokens` for requests that name none, also bounded by the context (0 = only the context).
    #[arg(long, default_value_t = 4096)]
    pub default_max_tokens: usize,
    /// Watchdog: seconds without an engine step while requests are queued before /health turns
    /// 503 and the supervised server restarts (0 = off).
    #[arg(long, default_value_t = 60.0)]
    pub watchdog_s: f64,
    /// Run the server in this process without a supervisor (debugging; a stalled engine then only
    /// turns /health to 503).
    #[arg(long)]
    pub no_supervisor: bool,
    /// Seconds before the first restart of a dead server; doubles up to 30 s.
    #[arg(long, default_value_t = 1.0)]
    pub restart_backoff_s: f64,
    /// Do not serve the built-in web UI at `/` (the API, /metrics and /events stay).
    #[arg(long)]
    pub no_ui: bool,
    /// Internal: run as the supervised server process.
    #[arg(long, hide = true)]
    pub child: bool,
    #[command(flatten)]
    pub model: ModelArgs,
}

/// Supervisor: re-run this binary as `serve --child ...` and keep it alive.
fn supervise(args: &ServeArgs) -> Result<()> {
    let exe = std::env::current_exe()?;
    let mut child_args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    child_args.push("--child".into());
    let mut sc = intarsia_server::supervisor::SupervisorConfig::new(exe, child_args);
    sc.backoff_min = std::time::Duration::from_secs_f64(args.restart_backoff_s.max(0.01));
    let code = intarsia_server::supervisor::supervise(sc);
    std::process::exit(code);
}

pub fn run(args: ServeArgs) -> Result<()> {
    if !args.child && !args.no_supervisor {
        return supervise(&args);
    }
    let t0 = std::time::Instant::now();
    let mut model = args.model.clone();
    model.max_batch = args.max_batch;
    model.spec_off_at = Some(usize::from(args.spec_off_at));
    model.busy_chunk = args.busy_chunk.map(usize::from);
    model.max_queue = Some(args.max_queue);
    let loaded = match model.for_serving().load() {
        Ok(l) => l,
        // Restarting cannot fix a bad model path or recipe: tell the supervisor to stop.
        Err(e) if args.child => {
            eprintln!("Error: {e:#}");
            std::process::exit(intarsia_server::EXIT_FATAL);
        }
        Err(e) => return Err(e),
    };
    let loaded_max_batch = loaded.max_batch;
    let engine: Arc<dyn Engine> = loaded.engine;
    let info = engine.info();
    eprintln!(
        "loaded {} on {} ({}) in {:.1}s, context {} tokens",
        info.id,
        info.backend,
        info.device,
        t0.elapsed().as_secs_f64(),
        info.context_len
    );
    let parent_pid = args.child.then(|| std::env::var("INTARSIA_SUPERVISOR_PID").ok()?.parse().ok()).flatten();
    let cfg = intarsia_server::ServerConfig {
        host: args.host,
        port: args.port,
        served_model_name: args.served_model_name,
        max_queue: args.max_queue,
        max_running: loaded_max_batch,
        default_max_tokens: args.default_max_tokens,
        watchdog_s: args.watchdog_s,
        exit_on_stall: args.child,
        parent_pid,
        ui: !args.no_ui,
    };
    intarsia_server::serve(engine, cfg).map_err(|e| anyhow::anyhow!("{e}"))
}
