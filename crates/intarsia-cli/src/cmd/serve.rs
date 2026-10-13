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
    #[command(flatten)]
    pub model: ModelArgs,
}

pub fn run(args: ServeArgs) -> Result<()> {
    let t0 = std::time::Instant::now();
    let loaded = args.model.load()?;
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
    let cfg =
        intarsia_server::ServerConfig { host: args.host, port: args.port, served_model_name: args.served_model_name };
    intarsia_server::serve(engine, cfg).map_err(|e| anyhow::anyhow!("{e}"))
}
