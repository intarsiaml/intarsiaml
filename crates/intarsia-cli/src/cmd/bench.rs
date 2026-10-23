//! `intarsia bench`: built-in speed benchmark.
//!
//! `intarsia bench speed --url http://HOST:PORT --model NAME` sweeps prompt sizes and
//! concurrency against any OpenAI-compatible server (ours, vLLM, ...). See docs/benchmarking.md.

use anyhow::{bail, Context, Result};
use intarsia_bench::speed::{render_table, run_speed, sanitize_label, SpeedConfig};
use serde_json::{Map, Value};
use std::path::PathBuf;

#[derive(clap::Args, Debug)]
pub struct BenchArgs {
    #[command(subcommand)]
    pub cmd: BenchCmd,
}

#[derive(clap::Subcommand, Debug)]
pub enum BenchCmd {
    /// End-to-end speed sweep (TTFT, prefill tok/s, decode tok/s, % of bandwidth floor).
    Speed(SpeedArgs),
    /// Prefix-reuse workloads (multi-turn chat with thinking, agent loop with a fixed system prompt and tool schema,
    /// RAG over the same documents): TTFT and prompt tokens computed vs served from cache, per request.
    Reuse(ReuseArgs),
}

#[derive(clap::Args, Debug)]
pub struct SpeedArgs {
    /// Server base URL (`http://host:port` or `http://host:port/v1`).
    #[arg(long)]
    pub url: String,
    /// Model name to send in requests.
    #[arg(long)]
    pub model: String,
    /// Environment variable holding the API key, if the server needs one.
    #[arg(long)]
    pub key_env: Option<String>,
    /// Prompt sizes in tokens.
    #[arg(long, value_delimiter = ',', default_value = "128,1024,4096,16384")]
    pub sizes: Vec<usize>,
    /// Forced output length per call.
    #[arg(long = "gen", default_value_t = 128)]
    pub gen: usize,
    /// Measured repetitions per cell (median is reported).
    #[arg(long, default_value_t = 3)]
    pub reps: usize,
    /// Discarded warmup calls per size.
    #[arg(long, default_value_t = 1)]
    pub warmup: usize,
    /// Concurrent stream counts; values above 1 add aggregate-throughput rows.
    #[arg(long, value_delimiter = ',', default_value = "1")]
    pub concurrency: Vec<usize>,
    /// Prompt size (tokens) for the concurrency rows.
    #[arg(long, default_value_t = 1024)]
    pub conc_size: usize,
    /// Prompts never exceed their size (aim ~2 % under, redo a rep the server counted longer): for A/B runs at a size
    /// equal to the prefill chunk.
    #[arg(long)]
    pub at_most: bool,
    /// Weight GB read per decoded token (e.g. 16.4 for Qwen3-8B bf16); enables % of floor.
    #[arg(long)]
    pub gb_per_token: Option<f64>,
    /// Memory bandwidth in GB/s (GB10: 221 measured); enables % of floor.
    #[arg(long)]
    pub bandwidth_gbs: Option<f64>,
    /// Result label; the JSON is written to <out-dir>/<label>.json.
    #[arg(long)]
    pub label: Option<String>,
    /// Thinking mode: `off` sends chat_template_kwargs.enable_thinking=false, `default` sends nothing.
    #[arg(long, default_value = "default", value_parser = ["on", "off", "default"])]
    pub thinking: String,
    /// Per-request timeout in seconds.
    #[arg(long, default_value_t = 900)]
    pub timeout_s: u64,
    /// Exit non-zero (after saving and printing the results) unless the server exports the graph metrics and
    /// every measured size replayed at least one CUDA graph; for A/B runs that must not silently decode eagerly.
    #[arg(long)]
    pub require_graphs: bool,
    /// Directory for the JSON result.
    #[arg(long, default_value = "results")]
    pub out_dir: PathBuf,
    /// Do not write the JSON file (still printed with --json).
    #[arg(long)]
    pub no_save: bool,
    /// Print the JSON result instead of the table.
    #[arg(long)]
    pub json: bool,
    /// Tensor-parallel degree of the server under test (`intarsia serve --tp N`). The server is what shards; this only
    /// records the degree in the result (`fingerprint.tp`). For the floor, pass `--gb-per-token` as the bytes the
    /// busiest rank reads per token (rank 0: its half of the layers plus the whole lm_head), not the model size.
    #[arg(long, default_value_t = 1)]
    pub tp: usize,
    /// Prefill chunk of the server under test (`serve` prints the resolved value and its source at load, e.g.
    /// "prefill chunk 2048 (recipe qwen3_5_moe, cuda)"). The server is what chunks; this only records the value
    /// in the result (`fingerprint.prefill_chunk`) so runs at different chunks are never compared by accident.
    #[arg(long)]
    pub prefill_chunk: Option<usize>,
    /// Extra host fingerprint field, KEY=VALUE (repeatable), stored in the JSON.
    #[arg(long = "fingerprint", value_name = "KEY=VALUE")]
    pub fingerprint: Vec<String>,
    /// JSON file with a host fingerprint (output of `intarsia doctor --json`), stored under `fingerprint.host`.
    #[arg(long)]
    pub fingerprint_file: Option<PathBuf>,
}

pub fn run(args: BenchArgs) -> Result<()> {
    match args.cmd {
        BenchCmd::Speed(a) => speed(a),
        BenchCmd::Reuse(a) => reuse(a),
    }
}

fn speed(a: SpeedArgs) -> Result<()> {
    if a.sizes.is_empty() || a.reps == 0 || a.gen < 2 {
        bail!("need at least one size, reps >= 1 and --gen >= 2");
    }
    let api_key = match &a.key_env {
        Some(v) => Some(std::env::var(v).with_context(|| format!("environment variable {v} is not set"))?),
        None => None,
    };
    let mut fp = Map::new();
    if let Some(p) = &a.fingerprint_file {
        let txt = std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?;
        fp.insert("host".into(), serde_json::from_str(&txt).with_context(|| format!("{} is not JSON", p.display()))?);
    }
    for kv in &a.fingerprint {
        let (k, v) = kv.split_once('=').with_context(|| format!("--fingerprint expects KEY=VALUE, got '{kv}'"))?;
        fp.insert(k.to_string(), Value::String(v.to_string()));
    }
    if a.tp == 0 {
        bail!("--tp must be at least 1");
    }
    if a.tp > 1 {
        fp.insert("tp".into(), Value::from(a.tp as u64));
    }
    if let Some(c) = a.prefill_chunk {
        fp.insert("prefill_chunk".into(), Value::from(c as u64));
    }
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let label = sanitize_label(&a.label.clone().unwrap_or_else(|| format!("{}-{now}", a.model)));
    let cfg = SpeedConfig {
        url: a.url.clone(),
        model: a.model.clone(),
        api_key,
        sizes: a.sizes.clone(),
        gen: a.gen,
        reps: a.reps,
        warmup: a.warmup,
        concurrency: a.concurrency.clone(),
        conc_size: a.conc_size,
        gb_per_token: a.gb_per_token,
        bandwidth_gbs: a.bandwidth_gbs,
        label: label.clone(),
        thinking: match a.thinking.as_str() {
            "off" => Some(false),
            "on" => Some(true),
            _ => None,
        },
        timeout_s: a.timeout_s,
        out_dir: Some(a.out_dir.display().to_string()),
        fingerprint: fp,
        at_most: a.at_most,
    };
    let res = run_speed(&cfg)?;
    if !a.no_save {
        std::fs::create_dir_all(&a.out_dir).with_context(|| format!("creating {}", a.out_dir.display()))?;
        let path = a.out_dir.join(format!("{label}.json"));
        std::fs::write(&path, serde_json::to_string_pretty(&res)? + "\n")?;
        eprintln!("wrote {}", path.display());
    }
    if a.json {
        println!("{}", serde_json::to_string_pretty(&res)?);
    } else {
        print!("{}", render_table(&res));
    }
    let rows = res["rows"].as_array().cloned().unwrap_or_default();
    if a.require_graphs {
        if res["graph"]["available"] != true {
            bail!("--require-graphs: the server does not export intarsia_graph_replays_total on /metrics");
        }
        let eager: Vec<String> = rows
            .iter()
            .filter(|x| x["graph"]["available"] == true && x["graph"]["replays"].as_u64() == Some(0))
            .map(|x| x["target_prompt_tokens"].to_string())
            .collect();
        if !eager.is_empty() {
            bail!("--require-graphs: no graph replays during the measured reps at size(s) {} (decode ran eagerly)", eager.join(", "));
        }
    }
    if !rows.is_empty() && rows.iter().all(|x| x["skipped"].is_string()) {
        bail!("every size was skipped: none fits the server's context window (see the warnings above); lower --sizes or raise the server's --max-seq");
    }
    let all_failed = !rows.is_empty() && rows.iter().all(|x| x["reps_ok"].as_u64() == Some(0));
    if all_failed {
        bail!("every call failed; see warnings above");
    }
    Ok(())
}

#[derive(clap::Args, Debug)]
pub struct ReuseArgs {
    /// Server base URL (`http://host:port` or `http://host:port/v1`).
    #[arg(long)]
    pub url: String,
    /// Model name sent in requests.
    #[arg(long)]
    pub model: String,
    /// Environment variable holding an API key (sent as a Bearer token).
    #[arg(long)]
    pub key_env: Option<String>,
    /// Workloads, comma-separated: chat, agent, rag.
    #[arg(long, default_value = "chat,agent,rag", value_delimiter = ',')]
    pub workloads: Vec<String>,
    /// Text the workloads are built from.
    #[arg(long, default_value = "tools/prompts/corpus_docs.txt")]
    pub corpus: PathBuf,
    /// Tokens generated per request.
    #[arg(long, default_value_t = 64)]
    pub max_tokens: usize,
    /// Turns per chat conversation (two conversations, interleaved).
    #[arg(long, default_value_t = 4)]
    pub turns: usize,
    /// Agent tasks (two requests each).
    #[arg(long, default_value_t = 4)]
    pub tasks: usize,
    /// RAG questions over the same documents.
    #[arg(long, default_value_t = 4)]
    pub questions: usize,
    /// Characters per RAG document (three documents; a chat turn quotes a quarter of this).
    #[arg(long, default_value_t = 6000)]
    pub doc_chars: usize,
    /// Write the requests sent (with the live answers as history) to DIR/<workload>.jsonl.
    #[arg(long)]
    pub record: Option<PathBuf>,
    /// Send the requests recorded in DIR instead of building them (byte-identical requests across runs).
    #[arg(long)]
    pub replay: Option<PathBuf>,
    /// Per-request timeout in seconds.
    #[arg(long, default_value_t = 600)]
    pub timeout_s: u64,
    /// Write the JSON result here.
    #[arg(long)]
    pub out: Option<PathBuf>,
}

fn reuse(a: ReuseArgs) -> Result<()> {
    let api_key = match &a.key_env {
        Some(v) => Some(std::env::var(v).with_context(|| format!("environment variable {v} not set"))?),
        None => None,
    };
    let cfg = intarsia_bench::reuse::ReuseConfig {
        url: a.url,
        model: a.model,
        api_key,
        workloads: a.workloads,
        corpus: a.corpus,
        max_tokens: a.max_tokens,
        turns: a.turns,
        tasks: a.tasks,
        questions: a.questions,
        doc_chars: a.doc_chars,
        record: a.record,
        replay: a.replay,
        timeout_s: a.timeout_s,
    };
    let res = intarsia_bench::reuse::run_reuse(&cfg)?;
    print!("{}", intarsia_bench::reuse::render_reuse(&res));
    if let Some(p) = &a.out {
        std::fs::write(p, serde_json::to_string_pretty(&res)?).with_context(|| format!("writing {}", p.display()))?;
        eprintln!("wrote {}", p.display());
    }
    Ok(())
}
