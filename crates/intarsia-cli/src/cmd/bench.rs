//! `intarsia bench`: built-in speed benchmark.
//!
//! `intarsia bench speed --url http://HOST:PORT --model NAME` sweeps prompt sizes and
//! concurrency against any OpenAI-compatible server (ours, vLLM, ...). See docs/benchmarking.md.

use anyhow::{bail, Context, Result};
use intarsia_bench::record::{self, RecordMeta};
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
    /// Append benchmark-record v1 JSON lines (model-benchmarks `schema/benchmark-record.v1.json`) after the run: one per
    /// measured prompt length and one per concurrency row. Without a value the file is
    /// /opt/code/model-benchmarks/data/records/intarsia-bench.jsonl. Only for an IntarsiaML server (checked before the
    /// sweep). Trust is `measured-here` only when `FLEETQ_JOB` is set (the run holds a fleetq slot), else `external`.
    /// docs/benchmarking.md, "Benchmark records".
    #[arg(long, value_name = "PATH", num_args = 0..=1, default_missing_value = record::DEFAULT_RECORDS_PATH, help_heading = "Benchmark record")]
    pub emit_record: Option<PathBuf>,
    /// Reuse this run id (a localeval run) for the record; the whole run is then ONE record, because schema/validate.py
    /// rejects a run_id repeated within a file. Default: `intarsia-bench:<label>-<unix time>:p<size>` / `:c<streams>`.
    #[arg(long, requires = "emit_record", help_heading = "Benchmark record")]
    pub record_run_id: Option<String>,
    /// Site hardware name, e.g. "DGX Spark GB10" (default: from the server host, as site/build_data.py names it).
    #[arg(long, requires = "emit_record", help_heading = "Benchmark record")]
    pub hardware: Option<String>,
    /// Site model family, e.g. "Qwen3.6-35B-A3B" (default: derived from the served model path and name).
    #[arg(long, requires = "emit_record", help_heading = "Benchmark record")]
    pub model_family: Option<String>,
    /// HF repo actually served (default: from a `models--Org--Name/snapshots/<sha>` served path).
    #[arg(long, requires = "emit_record", help_heading = "Benchmark record")]
    pub model_repo: Option<String>,
    /// HF revision sha of the served weights (default: from the served path).
    #[arg(long, requires = "emit_record", help_heading = "Benchmark record")]
    pub model_revision: Option<String>,
    /// Site quant format, e.g. NVFP4, MXFP4, BF16 (default: derived from the served path and name).
    #[arg(long, requires = "emit_record", help_heading = "Benchmark record")]
    pub quant: Option<String>,
    /// KV-cache dtype of the server (default: read from the server's `intarsia_info` metric).
    #[arg(long, requires = "emit_record", help_heading = "Benchmark record")]
    pub kv_dtype: Option<String>,
    /// The server's `--spec` value; anything but `off` makes the lane `spec:<value>` (default `plain`, or the `--spec`
    /// inside `--serve-cmd`). Spec decode is never compared with plain decode.
    #[arg(long, requires = "emit_record", help_heading = "Benchmark record")]
    pub spec: Option<String>,
    /// The full command that started the server (stored in engine.launch; api keys are redacted).
    #[arg(long, requires = "emit_record", help_heading = "Benchmark record")]
    pub serve_cmd: Option<String>,
    /// Environment variable of the server, KEY=VALUE (repeatable; keys with KEY/TOKEN/SECRET/PASSWORD are redacted).
    #[arg(long, requires = "emit_record", value_name = "KEY=VALUE", help_heading = "Benchmark record")]
    pub serve_env: Vec<String>,
    /// Path of the captured launch recipe (engine.launch.recipe_ref).
    #[arg(long, requires = "emit_record", help_heading = "Benchmark record")]
    pub recipe_ref: Option<String>,
    /// Engine version of the server (default: this binary's version; pass it when the server is another build).
    #[arg(long, requires = "emit_record", help_heading = "Benchmark record")]
    pub engine_version: Option<String>,
    /// Engine git commit of the server (default: `git rev-parse HEAD` of the working directory, `+dirty` when tracked
    /// files differ; pass it when the server was built elsewhere).
    #[arg(long, requires = "emit_record", help_heading = "Benchmark record")]
    pub engine_commit: Option<String>,
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
    let facts = a.emit_record.as_ref().map(|_| record::probe(&a.url, api_key.as_deref(), &a.model));
    if let Some(f) = &facts {
        if !f.is_intarsia() {
            bail!("--emit-record writes IntarsiaML records, but {} does not report itself as intarsia (/v1/models owned_by, /metrics intarsia_info); run it against `intarsia serve`", a.url);
        }
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
    if let (Some(path), Some(facts)) = (&a.emit_record, &facts) {
        emit_records(&a, path, &res, facts, a.out_dir.join(format!("{label}.json")))?;
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

/// `git rev-parse HEAD` of the working directory, `+dirty` when tracked files differ.
fn git_commit() -> Option<String> {
    let run = |args: &[&str]| std::process::Command::new("git").args(args).output().ok().filter(|o| o.status.success());
    let head = String::from_utf8(run(&["rev-parse", "HEAD"])?.stdout).ok()?.trim().to_string();
    let dirty = run(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|o| !o.stdout.is_empty());
    Some(if dirty { format!("{head}+dirty") } else { head })
}

fn local_hostname() -> Option<String> {
    let o = std::process::Command::new("hostname").output().ok().filter(|o| o.status.success())?;
    Some(String::from_utf8(o.stdout).ok()?.trim().to_string()).filter(|h| !h.is_empty())
}

fn emit_records(a: &SpeedArgs, path: &std::path::Path, res: &Value, facts: &record::ServerFacts, result_path: PathBuf) -> Result<()> {
    let meta = RecordMeta {
        record_run_id: a.record_run_id.clone(),
        hardware: a.hardware.clone(),
        model_family: a.model_family.clone(),
        model_repo: a.model_repo.clone(),
        model_revision: a.model_revision.clone(),
        quant: a.quant.clone(),
        kv_dtype: a.kv_dtype.clone(),
        spec: a.spec.clone(),
        serve_cmd: a.serve_cmd.clone(),
        serve_env: a.serve_env.clone(),
        recipe_ref: a.recipe_ref.clone(),
        engine_version: Some(a.engine_version.clone().unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string())),
        engine_commit: a.engine_commit.clone().or_else(git_commit),
        slot: std::env::var("FLEETQ_JOB").ok().filter(|v| !v.trim().is_empty()),
        command: Some(std::env::args().collect::<Vec<_>>().join(" ")),
        author: Some("intarsia bench".to_string()),
        result_path: (!a.no_save).then(|| result_path.display().to_string()),
        thinking: a.thinking.clone(),
        tp: a.tp,
        harness_version: env!("CARGO_PKG_VERSION").to_string(),
    };
    let recs = record::build_records(res, &meta, facts, local_hostname().as_deref());
    if recs.is_empty() {
        eprintln!("no measured cell: no record written");
        return Ok(());
    }
    record::append_jsonl(path, &recs)?;
    let incomplete = recs.iter().filter(|r| r["completeness"]["status"] == "incomplete").count();
    eprintln!(
        "appended {} benchmark record(s) to {} (trust {}, lane {}{})",
        recs.len(),
        path.display(),
        recs[0]["provenance"]["trust"].as_str().unwrap_or("?"),
        recs[0]["workload"]["lane"].as_str().unwrap_or("?"),
        if incomplete > 0 { format!("; {incomplete} incomplete, missing {}", recs[0]["completeness"]["missing"]) } else { String::new() }
    );
    Ok(())
}
