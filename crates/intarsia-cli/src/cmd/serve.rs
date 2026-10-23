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
    /// iterations. No effect without `--max-batch`.
    #[arg(long, default_value_t = intarsia_core::engine::BUSY_PREFILL_CHUNK as u16, value_parser = clap::value_parser!(u16).range(16..=2048))]
    pub busy_chunk: u16,
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
    /// Content-addressed prefix store: restore prefill already done (another conversation, the same system prompt,
    /// tool schema or documents) instead of recomputing it. `on` for the defaults (8 GiB of host memory), or
    /// `key=value` pairs separated by commas: `host_mb`, `device_mb` (default 0), `disk_mb` and `dir` (a disk tier that
    /// survives restarts of the same build), `page` (tokens per KV page, 16..=8192, default 256), `min`, `grid`,
    /// `boundary` (`|`-separated one-token texts that start a message; default `<|im_start|>`), `header` (texts that end
    /// a message header; default a newline), `any_cut=1` (restore at any length: only for chunk-invariant prefill), `roles` (cut only after these
    /// roles' headers, e.g. `assistant`: fewer passes, less reuse inside other messages).
    /// Single-sequence engine only.
    /// Overrides `INTARSIA_PREFIX_STORE`. See docs/results/2026-10-22-pc-prefix-store.md.
    #[arg(long)]
    pub prefix_store: Option<String>,
    /// With `--prefix-store`: a JSON file of prefixes to prefill and pin at startup, `[{"name": ..., "messages": [...],
    /// "tools": [...]}]` or `[{"name": ..., "prompt": "..."}]` (system prompts, tool schemas).
    #[arg(long)]
    pub prefix_store_pin: Option<std::path::PathBuf>,
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
    model.busy_chunk = Some(usize::from(args.busy_chunk));
    model.max_queue = Some(args.max_queue);
    if let Some(spec) = &args.prefix_store {
        model.prefix_store =
            intarsia_core::prefix_store::PrefixStoreConfig::parse(spec).map_err(|e| anyhow::anyhow!("--prefix-store: {e}"))?;
    }
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
    if let Some(path) = &args.prefix_store_pin {
        pin_prefixes(&*engine, path)?;
    }
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

/// `--prefix-store-pin`: prefill every listed prefix into the store and pin it (before the server takes requests).
fn pin_prefixes(engine: &dyn Engine, path: &std::path::Path) -> Result<()> {
    let text = std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("--prefix-store-pin {}: {e}", path.display()))?;
    let items: Vec<serde_json::Value> = serde_json::from_str(&text).map_err(|e| anyhow::anyhow!("--prefix-store-pin: {e}"))?;
    for it in items {
        let name = it["name"].as_str().ok_or_else(|| anyhow::anyhow!("--prefix-store-pin: every item needs a name"))?.to_string();
        let prompt = if let Some(p) = it["prompt"].as_str() {
            p.to_string()
        } else {
            let msgs = it["messages"].as_array().cloned().ok_or_else(|| anyhow::anyhow!("pin '{name}': messages or prompt"))?;
            let opts = intarsia_core::ChatOpts { add_generation_prompt: false, tools: it.get("tools").cloned(), ..Default::default() };
            engine.apply_chat_template(&msgs, &opts).map_err(|e| anyhow::anyhow!("pin '{name}': {e}"))?
        };
        let ids = engine.tokenize(&prompt, false).map_err(|e| anyhow::anyhow!("pin '{name}': {e}"))?;
        let req = intarsia_core::GenRequest {
            prompt: ids,
            max_tokens: 1,
            prewarm: Some(intarsia_core::Prewarm { pin: Some(name.clone()) }),
            ..Default::default()
        };
        let t0 = std::time::Instant::now();
        let mut done = None;
        engine
            .generate(req, &mut |ev| {
                if let intarsia_core::GenEvent::Done { prompt_tokens, cached_tokens, .. } = ev {
                    done = Some((prompt_tokens, cached_tokens));
                }
            })
            .map_err(|e| anyhow::anyhow!("pin '{name}': {e}"))?;
        if let Some((p, c)) = done {
            eprintln!("prefix store: pinned '{name}' ({p} tokens, {c} already stored) in {:.2}s", t0.elapsed().as_secs_f64());
        }
    }
    Ok(())
}
