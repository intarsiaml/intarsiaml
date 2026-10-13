//! `intarsia plan show`: load the model and print the resolved plan: which kernel runs every op, how
//! often per decode step, whether it passed admission, and the scratch workspace (ARCHITECTURE 3.6).

use crate::common::ModelArgs;
use anyhow::Result;
use clap::{Args, Subcommand};
use intarsia_core::engine::Engine;
use serde_json::Value;

#[derive(Args, Debug)]
pub struct PlanArgs {
    #[command(subcommand)]
    pub cmd: PlanCmd,
}

#[derive(Subcommand, Debug)]
pub enum PlanCmd {
    /// Resolve and print the plan for a model on a backend (loads the weights).
    Show(ShowArgs),
}

#[derive(Args, Debug)]
pub struct ShowArgs {
    #[command(flatten)]
    pub model: ModelArgs,
    /// Print the plan as JSON (the `GET /v1/intarsia/plan` document).
    #[arg(long)]
    pub json: bool,
}

pub fn run(args: PlanArgs) -> Result<()> {
    match args.cmd {
        PlanCmd::Show(a) => show(a),
    }
}

fn s<'a>(v: &'a Value, k: &str) -> &'a str {
    v[k].as_str().unwrap_or("?")
}

/// Plain-text rendering of the plan JSON.
pub fn render(p: &Value) -> String {
    use std::fmt::Write;
    let mut o = String::new();
    let _ = writeln!(
        o,
        "model {}  backend {}  device {} ({} MiB)  exactness {}",
        s(p, "model"),
        s(p, "backend"),
        s(&p["device"], "name"),
        p["device"]["mem_total"].as_u64().unwrap_or(0) >> 20,
        s(p, "exactness")
    );
    let _ = writeln!(
        o,
        "act {}  kv {}  prefill chunk {} ({})  scratch {} B  require_admitted {}  reference_only {}",
        s(p, "act_dtype"),
        s(p, "kv_dtype"),
        p["prefill_chunk"],
        s(p, "prefill_chunk_source"),
        p["scratch_bytes"],
        p["require_admitted"],
        p["reference_only"]
    );
    if p["tp"].is_object() {
        let t = &p["tp"];
        let _ = writeln!(
            o,
            "tensor parallel x{} ({} collective)  per rank: {} heads, {} kv heads, ffn {} (model: {}/{}/{})  all-reduce {} calls/decode step, {} B/rank",
            t["world"],
            s(t, "comm"),
            t["per_rank"]["n_heads"],
            t["per_rank"]["n_kv_heads"],
            t["per_rank"]["intermediate"],
            t["global"]["n_heads"],
            t["global"]["n_kv_heads"],
            t["global"]["intermediate"],
            t["all_reduce_calls_per_decode_step"],
            t["all_reduce_bytes_per_decode_step"]
        );
        for r in t["ranks"].as_array().cloned().unwrap_or_default() {
            let _ = writeln!(
                o,
                "  rank {}: {} ({} MiB)  weights {} MiB  kv cap {} tokens{}",
                r["rank"],
                s(&r, "device"),
                r["mem_total"].as_u64().unwrap_or(0) >> 20,
                r["weights_bytes"].as_u64().unwrap_or(0) >> 20,
                r["kv_cap_tokens"],
                if r["holds_lm_head"].as_bool().unwrap_or(false) { "  + lm_head" } else { "" }
            );
        }
    }
    // layers: collapse runs of identical layers
    let layers = p["layers"].as_array().cloned().unwrap_or_default();
    let key = |l: &Value| {
        l["blocks"]
            .as_array()
            .map(|b| b.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join("+"))
            .unwrap_or_default()
    };
    let mut i = 0;
    let _ = writeln!(o, "\nlayers ({}):", layers.len());
    while i < layers.len() {
        let mut j = i;
        while j + 1 < layers.len() && key(&layers[j + 1]) == key(&layers[i]) {
            j += 1;
        }
        let _ = writeln!(o, "  {:>3}..{:<3} {}", i, j, key(&layers[i]));
        i = j + 1;
    }
    let _ = writeln!(
        o,
        "\n{:<14} {:<26} {:<13} {:<9} {:>8} {:>9}",
        "op", "kernel", "exactness", "admitted", "calls/step", "scratch"
    );
    for op in p["ops"].as_array().cloned().unwrap_or_default() {
        let _ = writeln!(
            o,
            "{:<14} {:<26} {:<13} {:<9} {:>8} {:>9}",
            s(&op, "op"),
            s(&op, "kernel"),
            s(&op, "exactness"),
            if op["admitted"].as_bool().unwrap_or(false) { "yes" } else { "no" },
            op["calls_per_decode_step"],
            op["scratch_bytes"]
        );
    }
    o
}

fn show(a: ShowArgs) -> Result<()> {
    let loaded = a.model.load()?;
    let mut plan = loaded.engine.plan();
    if let Some(o) = plan.as_object_mut() {
        o.insert("prefill_chunk_source".into(), loaded.prefill_chunk.source.clone().into());
    }
    if a.json {
        println!("{}", serde_json::to_string_pretty(&plan)?);
    } else {
        print!("{}", render(&plan));
    }
    Ok(())
}
