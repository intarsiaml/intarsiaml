//! `intarsia generate`: load a model and complete a prompt.

use crate::common::ModelArgs;
use anyhow::{bail, Result};
use clap::Args;
use intarsia_core::engine::{ChatOpts, Engine, GenEvent, GenRequest};
use serde_json::json;
use std::io::Write;

#[derive(Args, Debug)]
pub struct GenerateArgs {
    #[command(flatten)]
    pub model: ModelArgs,
    /// Prompt text. Raw completion unless --chat is given (no chat template, no special tokens).
    #[arg(long, short = 'p')]
    pub prompt: String,
    #[arg(long, default_value_t = 64)]
    pub max_tokens: usize,
    #[arg(long, default_value_t = 0)]
    pub min_tokens: usize,
    /// Keep generating past EOS (forced-length runs).
    #[arg(long)]
    pub ignore_eos: bool,
    /// 0 = greedy.
    #[arg(long, default_value_t = 0.0)]
    pub temperature: f32,
    #[arg(long, default_value_t = 1.0)]
    pub top_p: f32,
    #[arg(long, default_value_t = 0)]
    pub top_k: usize,
    #[arg(long)]
    pub seed: Option<u64>,
    /// Stop string (repeatable).
    #[arg(long)]
    pub stop: Vec<String>,
    /// Wrap --prompt in the model's chat template as a user message.
    #[arg(long)]
    pub chat: bool,
    /// System message for --chat.
    #[arg(long)]
    pub system: Option<String>,
    /// With --chat: render the template with enable_thinking=false.
    #[arg(long)]
    pub no_think: bool,
    /// Print the prompt before the completion.
    #[arg(long)]
    pub echo: bool,
    /// Print the generated token ids to stderr.
    #[arg(long)]
    pub show_ids: bool,
    /// Debug: print the N highest logits of the last prompt position (runs one extra prefill).
    #[arg(long)]
    pub show_logits: Option<usize>,
    /// Print every op's time, calls and share of the timed total to stderr at the end (use with `--profile-every 1`
    /// and `--max-tokens 1` for a pure prefill profile).
    #[arg(long)]
    pub op_table: bool,
}

pub fn run(args: GenerateArgs) -> Result<()> {
    let t_load = std::time::Instant::now();
    let loaded = args.model.load()?;
    let engine = loaded.engine;
    let info = engine.info();
    eprintln!(
        "loaded {} on {} in {:.1}s: {} layers, hidden {}, act {}, kv {}, weights {:.2} GB, context {} tokens (recipe {}{})",
        info.id,
        info.device,
        t_load.elapsed().as_secs_f64(),
        info.n_layers,
        info.hidden,
        info.act_dtype,
        info.kv_dtype,
        info.weights_bytes as f64 / 1e9,
        info.context_len,
        loaded.recipe_source,
        loaded
            .chat_template_source
            .as_deref()
            .map(|s| format!(", chat template {s}"))
            .unwrap_or_default(),
    );

    let text = if args.chat {
        let mut msgs = vec![];
        if let Some(s) = &args.system {
            msgs.push(json!({"role": "system", "content": s}));
        }
        msgs.push(json!({"role": "user", "content": args.prompt}));
        let opts = ChatOpts {
            add_generation_prompt: true,
            enable_thinking: if args.no_think { Some(false) } else { None },
            ..Default::default()
        };
        engine.apply_chat_template(&msgs, &opts).map_err(|e| anyhow::anyhow!("{e}"))?
    } else {
        args.prompt.clone()
    };
    let ids = engine.tokenize(&text, false).map_err(|e| anyhow::anyhow!("{e}"))?;
    if ids.is_empty() {
        bail!("the prompt tokenizes to nothing");
    }
    eprintln!("prompt: {} tokens", ids.len());

    if let Some(n) = args.show_logits {
        let mut last: Vec<f32> = vec![];
        engine
            .forward_logits_each(&ids, &mut |p, row| {
                if p + 1 == ids.len() {
                    last = row.to_vec();
                }
                Ok(())
            })
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut idx: Vec<usize> = (0..last.len()).collect();
        idx.sort_by(|a, b| last[*b].partial_cmp(&last[*a]).unwrap());
        let top: Vec<String> = idx.iter().take(n).map(|i| format!("{i}:{:.4}", last[*i])).collect();
        eprintln!("top logits (id:value): {}", top.join(" "));
    }
    let req = GenRequest {
        prompt: ids,
        max_tokens: args.max_tokens,
        min_tokens: args.min_tokens,
        ignore_eos: args.ignore_eos,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        seed: args.seed,
        stop: args.stop.clone(),
        stop_ids: vec![],
        cancel: None,
        trace_logits: None,
    };
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    if args.echo {
        write!(out, "{text}")?;
    }
    let mut generated: Vec<u32> = vec![];
    let mut done = None;
    let mut err = None;
    engine
        .generate(req, &mut |ev| match ev {
            GenEvent::Token { id, text } => {
                generated.push(id);
                let _ = write!(out, "{text}");
                let _ = out.flush();
            }
            GenEvent::Done { .. } => done = Some(ev),
            GenEvent::Error { message } => err = Some(message),
        })
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    writeln!(out)?;
    if let Some(m) = err {
        bail!("generation failed: {m}");
    }
    if args.show_ids {
        eprintln!("ids: {generated:?}");
    }
    if let Some(GenEvent::Done {
        finish_reason,
        prompt_tokens,
        completion_tokens,
        cached_tokens,
        prefill_ms,
        decode_ms,
        ..
    }) = done
    {
        let computed = prompt_tokens - cached_tokens;
        eprintln!(
            "finish={} prompt={} (cached {}) completion={} | prefill {:.0} ms ({:.1} tok/s) | decode {:.0} ms ({:.2} tok/s)",
            finish_reason.as_str(),
            prompt_tokens,
            cached_tokens,
            completion_tokens,
            prefill_ms,
            computed as f64 / (prefill_ms / 1e3).max(1e-9),
            decode_ms,
            (completion_tokens.saturating_sub(1)) as f64 / (decode_ms / 1e3).max(1e-9),
        );
    }
    let t = engine.telemetry();
    if args.op_table {
        // time_ms covers the timed launches only; scale to all launches of the op
        let est = |v: &intarsia_core::telemetry::OpStat| v.time_ms * v.calls as f64 / v.timed_calls.max(1) as f64;
        let mut rows: Vec<_> = t.ops.iter().map(|(k, v)| (k.clone(), est(v), v)).collect();
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let total: f64 = rows.iter().map(|r| r.1).sum();
        eprintln!("op table (estimated ms over all launches; total {total:.1} ms):");
        eprintln!("{:<24} {:>10} {:>7} {:>8} {:>7}", "op", "ms", "share%", "calls", "timed");
        for (k, ms, v) in &rows {
            eprintln!("{:<24} {:>10.2} {:>7.2} {:>8} {:>7}", k, ms, 100.0 * ms / total.max(1e-9), v.calls, v.timed_calls);
        }
    }
    let mut top: Vec<_> = t.ops.iter().collect();
    top.sort_by(|a, b| b.1.time_ms.partial_cmp(&a.1.time_ms).unwrap());
    let line: Vec<String> = top.iter().take(5).map(|(k, v)| format!("{k}={:.0}ms/{}", v.time_ms, v.calls)).collect();
    if let Some(c) = t.ops.get("all_reduce") {
        eprintln!(
            "tensor parallel: {} all-reduces ({:.1} per step), {} timed, avg {:.1} us each when timed (a timed call synchronises both sides, so it includes the ranks' skew)",
            c.calls,
            c.calls as f64 / t.steps_total.max(1) as f64,
            c.timed_calls,
            c.time_ms * 1e3 / c.timed_calls.max(1) as f64
        );
    }
    if t.spec_steps > 0 || args.model.spec != "off" {
        eprintln!(
            "spec: verify passes={} proposed={} accepted={} (acceptance {:.1}%, {:.2} tokens per pass)",
            t.spec_steps,
            t.spec_proposed,
            t.spec_accepted,
            100.0 * t.spec_accepted as f64 / t.spec_proposed.max(1) as f64,
            (t.spec_steps + t.spec_accepted) as f64 / t.spec_steps.max(1) as f64,
        );
    }
    eprintln!(
        "steps={} graph replays={} (captures {}) eager decode steps={} (profiled {}) top ops: {}",
        t.steps_total,
        t.graph_replays,
        t.graph_captures,
        t.decode_eager_steps,
        t.decode_profiled_steps,
        line.join(" ")
    );
    Ok(())
}
