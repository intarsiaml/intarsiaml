//! Reports of a qualify run: `quick-report.md` / `full-report.md` (headline first, then a per-stage table), the matching
//! `quick.json` / `full.json`, and the Prometheus text pushed by `--push-metrics`. The layout is the one documented in
//! `docs/qualify-report-template.md`; a test keeps the two in step.

use super::state::{StageRecord, State, Verdict};
use serde_json::{json, Value};
use std::fmt::Write;

pub struct Meta {
    pub model: String,
    pub model_dir: String,
    pub backend: String,
    pub tp: usize,
    pub reference: Option<String>,
    pub build_stamp: Option<String>,
    pub version: String,
    pub generated_unix: u64,
    pub quick_only: bool,
}

pub fn fmt_dur(s: f64) -> String {
    let s = s.max(0.0);
    if s < 90.0 {
        format!("{s:.0}s")
    } else if s < 5400.0 {
        format!("{}m{:02}s", (s / 60.0) as u64, (s % 60.0) as u64)
    } else {
        format!("{}h{:02}m", (s / 3600.0) as u64, ((s % 3600.0) / 60.0) as u64)
    }
}

fn included<'a>(state: &'a State, full: bool) -> Vec<&'a StageRecord> {
    state.stages.iter().filter(|s| full || s.section == "quick").collect()
}

fn section_wall(state: &State, section: &str) -> f64 {
    state.stages.iter().filter(|s| s.section == section).map(|s| s.wall_s).sum()
}

fn overall(stages: &[&StageRecord], expected_full: bool, have_full: bool) -> &'static str {
    if stages.iter().any(|s| s.verdict == Verdict::Error) {
        "ERROR"
    } else if stages.iter().any(|s| s.verdict == Verdict::Fail) {
        "FAIL"
    } else if stages.iter().any(|s| s.verdict == Verdict::MissingBaseline) {
        "MISSING-BASELINE"
    } else if expected_full && !have_full {
        "INCOMPLETE"
    } else {
        "PASS"
    }
}

fn num(v: &Value, digits: usize) -> String {
    v.as_f64().map(|x| format!("{x:.digits$}")).unwrap_or_else(|| "-".into())
}
fn sci(v: &Value) -> String {
    v.as_f64().map(|x| format!("{x:.2e}")).unwrap_or_else(|| "-".into())
}

fn stage<'a>(state: &'a State, id: &str) -> Option<&'a StageRecord> {
    state.get(id)
}

/// Speed rows of the stage `id`: (size, prefill, decode, pct of floor).
fn speed_rows(state: &State, id: &str) -> Vec<Value> {
    stage(state, id)
        .map(|s| s.metrics["rows"].as_array().cloned().unwrap_or_default())
        .unwrap_or_default()
        .into_iter()
        .filter(|r| r["skipped"].is_null())
        .collect()
}

fn accuracy_modes(s: &StageRecord) -> Vec<Value> {
    if let Some(a) = s.metrics.get("accuracy").filter(|a| a.is_object()) {
        return a["modes"].as_array().cloned().unwrap_or_default();
    }
    s.metrics["modes"].as_array().cloned().unwrap_or_default()
}

pub fn summary_json(state: &State, meta: &Meta, full: bool) -> Value {
    let stages = included(state, full);
    let have_full = state.stages.iter().any(|s| s.section == "full");
    let verdict = overall(&stages, full && !meta.quick_only, have_full);
    let failures: Vec<Value> = stages
        .iter()
        .filter(|s| matches!(s.verdict, Verdict::Fail | Verdict::MissingBaseline | Verdict::Error))
        .map(|s| json!({"stage": s.id, "verdict": s.verdict.label(), "reason": s.reason}))
        .collect();
    let warnings: Vec<Value> = stages
        .iter()
        .filter(|s| s.verdict == Verdict::Warn)
        .map(|s| json!({"stage": s.id, "verdict": s.verdict.label(), "reason": s.reason}))
        .collect();
    let skips: Vec<Value> = stages
        .iter()
        .filter(|s| s.verdict == Verdict::Skip)
        .map(|s| json!({"stage": s.id, "reason": s.reason}))
        .collect();
    let decode: Vec<Value> = speed_rows(state, "bench").iter().map(|r| json!({"size": r["size"], "decode_tok_s": r["decode_tok_s"], "prefill_tok_s": r["prefill_tok_s"], "pct_of_floor": r["pct_of_floor"], "ttft_s": r["ttft_s"]})).collect();
    let mut accuracy = vec![];
    for id in ["acc-short", "acc-format", "acc-long", "acc-4k", "tp"] {
        if let Some(s) = stage(state, id) {
            if s.verdict == Verdict::Skip {
                continue;
            }
            let modes = accuracy_modes(s);
            if modes.is_empty() {
                continue;
            }
            accuracy.push(json!({"stage": id, "verdict": if id == "tp" { s.metrics["accuracy"]["verdict"].clone() } else { json!(s.verdict.label()) }, "modes": modes.iter().map(|m| json!({"mode": m["mode"], "positions": m["positions"], "top1_pct": m["top1_pct"], "mean_kl": m["mean_kl"], "mean_kl21": m["mean_kl21"], "floor_pass": m["floor_pass"]})).collect::<Vec<_>>()}));
        }
    }
    let gate = stage(state, "gate").map(|s| s.metrics.clone()).unwrap_or(Value::Null);
    let wall = json!({"quick": section_wall(state, "quick"), "full": section_wall(state, "full"), "total": section_wall(state, "quick") + section_wall(state, "full")});
    json!({
        "kind": "intarsia-qualify", "schema_version": 1, "section": if full { "full" } else { "quick" },
        "model": meta.model, "model_dir": meta.model_dir, "backend": meta.backend, "tp": meta.tp, "reference": meta.reference,
        "build_stamp": meta.build_stamp.clone().unwrap_or_else(|| "unstamped".into()), "version": meta.version, "generated_unix": meta.generated_unix,
        "verdict": verdict,
        "exit_code": match verdict { "PASS" => 0, "FAIL" => 1, _ => 2 },
        "headline": {"speed": decode, "accuracy": accuracy, "quick_gate": gate},
        "failures": failures, "warnings": warnings, "skipped": skips, "wall_s": wall,
        "stages": stages.iter().map(|s| json!({"id": s.id, "title": s.title, "section": s.section, "verdict": s.verdict.label(), "reason": s.reason, "wall_s": s.wall_s, "started_unix": s.started_unix, "artifacts": s.artifacts, "metrics": s.metrics, "provenance": s.provenance, "notes": s.notes})).collect::<Vec<_>>(),
        "hooks": state.hooks,
    })
}

pub fn render(state: &State, meta: &Meta, full: bool) -> String {
    let stages = included(state, full);
    let have_full = state.stages.iter().any(|s| s.section == "full");
    let verdict = overall(&stages, full && !meta.quick_only, have_full);
    let mut o = String::new();
    let title = if full { "Qualification report" } else { "Qualification quick report" };
    let _ = writeln!(
        o,
        "# {title}: {} on {}{}\n",
        meta.model,
        meta.backend,
        if meta.tp > 1 && full { format!(" (tp variant: {})", meta.tp) } else { String::new() }
    );
    let doctor = stage(state, "doctor");
    let dev = doctor
        .map(|d| {
            format!(
                "{} ({})",
                d.metrics["device"]["name"].as_str().unwrap_or("?"),
                d.metrics["device"]["arch"].as_str().unwrap_or("?")
            )
        })
        .unwrap_or_else(|| "not recorded".into());
    let _ = writeln!(
        o,
        "_{} section, generated unix {}, intarsia {} build `{}`._\n",
        if full { "Full (quick + in-depth)" } else { "Quick" },
        meta.generated_unix,
        meta.version,
        meta.build_stamp.as_deref().unwrap_or("unstamped")
    );

    // ---- headline ----
    let _ = writeln!(o, "## Headline\n");
    let count = |v: Verdict| stages.iter().filter(|s| s.verdict == v).count();
    let _ = writeln!(
        o,
        "**Verdict: {verdict}** ({} stages: {} pass, {}{} fail, {}{} error, {} skipped)\n",
        stages.len(),
        count(Verdict::Pass),
        if count(Verdict::Warn) > 0 { format!("{} report-only (warn), ", count(Verdict::Warn)) } else { String::new() },
        count(Verdict::Fail),
        if count(Verdict::MissingBaseline) > 0 { format!("{} missing baseline, ", count(Verdict::MissingBaseline)) } else { String::new() },
        count(Verdict::Error),
        count(Verdict::Skip)
    );
    let _ = writeln!(o, "| | |\n|---|---|");
    let _ = writeln!(o, "| Model | {} (`{}`) |", meta.model, meta.model_dir);
    let _ = writeln!(o, "| Backend / device | {} / {} |", meta.backend, dev);
    if let Some(r) = stage(state, "recipe") {
        let _ = writeln!(o, "| Recipe | {} |", r.reason);
    }
    let rows = speed_rows(state, "bench");
    if rows.is_empty() {
        let _ = writeln!(o, "| Decode | not measured |\n| Prefill | not measured |");
    } else {
        let dec: Vec<String> = rows
            .iter()
            .map(|r| {
                format!(
                    "{} tok: **{}** tok/s ({}% of floor)",
                    r["size"],
                    num(&r["decode_tok_s"], 1),
                    r["pct_of_floor"].as_f64().map(|p| format!("{p:.0}")).unwrap_or_else(|| "n/a".into())
                )
            })
            .collect();
        let pre: Vec<String> =
            rows.iter().map(|r| format!("{} tok: **{}** tok/s", r["size"], num(&r["prefill_tok_s"], 0))).collect();
        let _ = writeln!(o, "| Decode | {} |", dec.join("; "));
        let _ = writeln!(o, "| Prefill | {} |", pre.join("; "));
    }
    for id in ["acc-short", "acc-format", "acc-long", "acc-4k"] {
        if let Some(s) = stage(state, id) {
            let txt = match s.verdict {
                Verdict::Skip => format!("skipped: {}", s.reason),
                _ => {
                    let m: Vec<String> = accuracy_modes(s)
                        .iter()
                        .map(|m| {
                            format!(
                                "{} top-1 {}% KL {}",
                                m["mode"].as_str().unwrap_or("?"),
                                num(&m["top1_pct"], 2),
                                sci(&m["mean_kl"])
                            )
                        })
                        .collect();
                    format!(
                        "**{}**{}",
                        s.verdict.label(),
                        if m.is_empty() { String::new() } else { format!(" ({})", m.join("; ")) }
                    )
                }
            };
            let label = match id {
                "acc-short" => "Accuracy (short golden)",
                "acc-format" => "Accuracy (format reference, same weights)",
                "acc-4k" => "Accuracy (4k golden)",
                _ => "Accuracy (long golden)",
            };
            let _ = writeln!(o, "| {label} | {txt} |");
        }
    }
    if let Some(g) = stage(state, "gate") {
        let _ = writeln!(o, "| Quick gate | **{}**: {} |", g.verdict.label(), md_cell(&g.reason));
    }
    let _ = writeln!(
        o,
        "| Wall time | quick {}{} |",
        fmt_dur(section_wall(state, "quick")),
        if full {
            format!(
                ", in-depth {}, total {}",
                fmt_dur(section_wall(state, "full")),
                fmt_dur(section_wall(state, "quick") + section_wall(state, "full"))
            )
        } else {
            String::new()
        }
    );
    if let Some(r) = &meta.reference {
        let _ = writeln!(o, "| Reference | {r} |");
    }
    let _ = writeln!(o);
    let _ = writeln!(o, "### Failures\n");
    let fails: Vec<&&StageRecord> =
        stages.iter().filter(|s| matches!(s.verdict, Verdict::Fail | Verdict::MissingBaseline | Verdict::Error)).collect();
    if fails.is_empty() {
        let _ = writeln!(o, "None.\n");
    } else {
        for s in fails {
            let _ = writeln!(o, "- `{}` **{}**: {}", s.id, s.verdict.label(), md_cell(&s.reason));
        }
        let _ = writeln!(o);
    }
    let warns: Vec<&&StageRecord> = stages.iter().filter(|s| s.verdict == Verdict::Warn).collect();
    if !warns.is_empty() {
        let _ = writeln!(o, "### Warnings (report-only, the run does not fail)\n");
        for s in warns {
            let _ = writeln!(o, "- `{}` **WARN**: {}", s.id, md_cell(&s.reason));
        }
        let _ = writeln!(o);
    }
    let _ = writeln!(o, "### Skipped\n");
    let skips: Vec<&&StageRecord> = stages.iter().filter(|s| s.verdict == Verdict::Skip).collect();
    if skips.is_empty() {
        let _ = writeln!(o, "None.\n");
    } else {
        for s in skips {
            let _ = writeln!(o, "- `{}`: {}", s.id, md_cell(&s.reason));
        }
        let _ = writeln!(o);
    }

    // ---- stages ----
    let _ = writeln!(o, "## Stages\n");
    let _ = writeln!(o, "| # | Stage | Section | Verdict | Wall | Result |\n|---|---|---|---|---|---|");
    for (i, s) in stages.iter().enumerate() {
        let _ = writeln!(
            o,
            "| {} | `{}` {} | {} | {} | {} | {} |",
            i + 1,
            s.id,
            s.title,
            s.section,
            s.verdict.label(),
            fmt_dur(s.wall_s),
            md_cell(&s.reason)
        );
    }
    let _ = writeln!(o);
    let _ = writeln!(o, "### Notes\n");
    let mut any_note = false;
    for s in &stages {
        for n in &s.notes {
            any_note = true;
            let _ = writeln!(o, "- `{}`: {}", s.id, md_cell(n));
        }
    }
    if !any_note {
        let _ = writeln!(o, "None.");
    }
    let _ = writeln!(o);

    // ---- speed ----
    let _ = writeln!(o, "## Speed\n");
    let mut any = false;
    for (id, label) in [("bench", "quick bench"), ("bench-long", "long-context bench"), ("tp", "tensor-parallel bench")]
    {
        if !stages.iter().any(|s| s.id == id) {
            continue;
        }
        let rows = speed_rows(state, id);
        if rows.is_empty() {
            continue;
        }
        any = true;
        let m = &stage(state, id).unwrap().metrics;
        let _ = writeln!(
            o,
            "**{label}** (floor: {} GB/token at {} GB/s; {})\n",
            num(&m["gb_per_token"], 2),
            num(&m["bandwidth_gbs"], 0),
            m["floor_inputs"].as_str().unwrap_or("no floor inputs")
        );
        let _ = writeln!(o, "| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |\n|---|---|---|---|---|---|---|");
        for r in rows {
            let _ = writeln!(
                o,
                "| {} | {} | {} | {} | {} | {} | {} |",
                r["prompt_tokens"].as_f64().map(|p| format!("{p:.0}")).unwrap_or_else(|| r["size"].to_string()),
                num(&r["ttft_s"], 3),
                num(&r["prefill_tok_s"], 0),
                num(&r["decode_tok_s"], 2),
                num(&r["decode_stdev"], 2),
                num(&r["floor_tok_s"], 1),
                num(&r["pct_of_floor"], 1)
            );
        }
        let _ = writeln!(o);
    }
    if !any {
        let _ = writeln!(o, "No speed numbers.\n");
    }
    if let Some(ab) = stage(state, "ab").filter(|s| s.verdict == Verdict::Pass) {
        let _ = writeln!(o, "**A/B against the reference (localeval)**: {}\n", ab.reason);
        let _ = writeln!(o, "| Prompt tokens | decode ours / reference | prefill ours / reference |\n|---|---|---|");
        for r in ab.metrics["ratios"].as_array().cloned().unwrap_or_default() {
            let _ =
                writeln!(o, "| {} | {}x | {}x |", r["size"], num(&r["decode_ratio"], 2), num(&r["prefill_ratio"], 2));
        }
        let _ = writeln!(o, "\n{}\n", ab.metrics["note"].as_str().unwrap_or(""));
    }

    // ---- accuracy ----
    let _ = writeln!(o, "## Accuracy\n");
    let mut any = false;
    for id in ["acc-short", "acc-format", "acc-long", "acc-4k", "tp"] {
        let Some(s) = stage(state, id).filter(|s| stages.iter().any(|x| x.id == s.id)) else { continue };
        let modes = accuracy_modes(s);
        if modes.is_empty() {
            continue;
        }
        any = true;
        let m = if id == "tp" { &s.metrics["accuracy"] } else { &s.metrics };
        let _ = writeln!(
            o,
            "**{id}** against `{}` ({}; gate class {})\n",
            m["golden"].as_str().unwrap_or("?"),
            m["golden_how"].as_str().unwrap_or("?"),
            m["gate_class"].as_str().unwrap_or("?")
        );
        let _ = writeln!(o, "| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |\n|---|---|---|---|---|---|---|---|---|---|---|");
        for m in modes {
            let _ = writeln!(
                o,
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                m["mode"].as_str().unwrap_or("?"),
                m["positions"],
                num(&m["top1_pct"], 2),
                num(&m["top5_pct"], 2),
                sci(&m["mean_kl"]),
                sci(&m["mean_kl21"]),
                sci(&m["max_kl21"]),
                sci(&m["tail_abs"]),
                m["nonfinite"],
                num(&m["greedy_match_pct"], 1),
                if m["floor_pass"] == true { "pass" } else { "FAIL" }
            );
        }
        let _ = writeln!(o);
    }
    if let Some(e) = stage(state, "envelope").filter(|s| s.verdict != Verdict::Skip) {
        let _ = writeln!(
            o,
            "**Regression gating artifacts**: {}\n\n`{}`\n",
            e.reason,
            e.metrics["use"].as_str().unwrap_or("")
        );
    }
    if let Some(g) = stage(state, "gate") {
        let _ = writeln!(o, "**Quick gate**: {}\n", md_cell(&g.reason));
    }
    if !any {
        let _ = writeln!(o, "No golden comparison ran (see Skipped).\n");
    }

    // ---- plan and recipe ----
    let _ = writeln!(o, "## Plan and recipe\n");
    if let Some(r) = stage(state, "recipe") {
        let _ = writeln!(o, "- recipe: {} ({})", md_cell(&r.reason), r.verdict.label());
        if let Some(a) = r.metrics["todos"].as_array().filter(|a| !a.is_empty()) {
            for t in a {
                let _ = writeln!(o, "  - TODO {}", t.as_str().unwrap_or(""));
            }
        }
    }
    if let Some(p) = stage(state, "plan") {
        let _ = writeln!(o, "- plan: {} ({})", md_cell(&p.reason), p.verdict.label());
        if let Some(ops) = p.metrics["ops"].as_array().filter(|a| !a.is_empty()) {
            let _ =
                writeln!(o, "\n| Op | Kernel | Exactness | Admitted | Calls / decode step |\n|---|---|---|---|---|");
            for op in ops {
                let _ = writeln!(
                    o,
                    "| {} | {} | {} | {} | {} |",
                    op["op"].as_str().unwrap_or("?"),
                    op["kernel"].as_str().unwrap_or("?"),
                    op["exactness"].as_str().unwrap_or("?"),
                    if op["admitted"] == true { "yes" } else { "no" },
                    op["calls_per_decode_step"]
                );
            }
        }
        if let Some(m) = p.metrics["missing_kernels"].as_array().filter(|a| !a.is_empty()) {
            let _ = writeln!(o, "\nMissing kernels:");
            for k in m {
                let _ = writeln!(o, "- {}", k.as_str().unwrap_or(""));
            }
        }
    }
    if let Some(k) = stage(state, "kernels") {
        let _ = writeln!(o, "\n- kernels test: {} ({})", md_cell(&k.reason), k.verdict.label());
    }
    let _ = writeln!(o);

    // ---- provenance ----
    let _ = writeln!(o, "## Provenance\n");
    let _ = writeln!(o, "| Stage | Build stamp | Host | Started (unix) | Command |\n|---|---|---|---|---|");
    for s in &stages {
        let cmd = s.provenance["commands"].as_array().and_then(|c| c.last()).and_then(|c| c.as_str()).unwrap_or("-");
        let _ = writeln!(
            o,
            "| `{}` | {} | {} | {} | `{}` |",
            s.id,
            s.provenance["build_stamp"].as_str().unwrap_or("-"),
            s.provenance["host"].as_str().unwrap_or("-"),
            s.started_unix,
            md_cell(cmd)
        );
    }
    let unstamped = stages.iter().any(|s| s.provenance["build_stamp"].as_str() == Some("unstamped"));
    let _ = writeln!(
        o,
        "\nModel fingerprint: `{}`.",
        stage(state, "doctor")
            .or(stages.first().copied())
            .map(|s| s.provenance["model_fingerprint"].as_str().unwrap_or("-"))
            .unwrap_or("-")
    );
    if unstamped {
        let _ = writeln!(o, "\nBuild stamp `unstamped` means a plain `cargo build`: baselines written here gate this binary, but `--pre-change` provenance checks refuse them as a \"before\".");
    }
    let _ = writeln!(o);
    for h in &state.hooks {
        let _ = writeln!(o, "- hook `{}`: {}", h["hook"].as_str().unwrap_or("?"), h);
    }

    // ---- artifacts ----
    let _ = writeln!(o, "\n## Artifacts\n");
    for s in &stages {
        for a in &s.artifacts {
            let _ = writeln!(o, "- `{a}` ({})", s.id);
        }
    }
    let _ = writeln!(o, "- `state.json` (stage records; `--resume` reads it)");
    o
}

fn md_cell(s: &str) -> String {
    let t: String = s.lines().next().unwrap_or("").chars().take(300).collect();
    t.replace('|', "\\|")
}

pub(crate) fn esc(v: &str) -> String {
    v.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', " ")
}

/// Prometheus exposition text for a Pushgateway group (`model` and `backend` are in the push URL).
pub fn prometheus_text(summary: &Value) -> String {
    let mut o = String::new();
    let section = summary["section"].as_str().unwrap_or("quick");
    let g = |o: &mut String, name: &str, help: &str| {
        let _ = writeln!(o, "# HELP {name} {help}\n# TYPE {name} gauge");
    };
    g(&mut o, "intarsia_qualify_decode_tok_s", "Single-stream decode tokens/s of the quick bench by prompt size.");
    for r in summary["headline"]["speed"].as_array().cloned().unwrap_or_default() {
        if let Some(v) = r["decode_tok_s"].as_f64() {
            let _ = writeln!(o, "intarsia_qualify_decode_tok_s{{size=\"{}\"}} {v}", r["size"]);
        }
    }
    g(&mut o, "intarsia_qualify_prefill_tok_s", "Prefill tokens/s of the quick bench by prompt size.");
    for r in summary["headline"]["speed"].as_array().cloned().unwrap_or_default() {
        if let Some(v) = r["prefill_tok_s"].as_f64() {
            let _ = writeln!(o, "intarsia_qualify_prefill_tok_s{{size=\"{}\"}} {v}", r["size"]);
        }
    }
    g(&mut o, "intarsia_qualify_pct_of_floor", "Decode speed as a percentage of the memory-bandwidth byte floor.");
    for r in summary["headline"]["speed"].as_array().cloned().unwrap_or_default() {
        if let Some(v) = r["pct_of_floor"].as_f64() {
            let _ = writeln!(o, "intarsia_qualify_pct_of_floor{{size=\"{}\"}} {v}", r["size"]);
        }
    }
    g(&mut o, "intarsia_qualify_accuracy_top1_pct", "Top-1 agreement with the golden, percent.");
    let mut kl = String::new();
    for a in summary["headline"]["accuracy"].as_array().cloned().unwrap_or_default() {
        for m in a["modes"].as_array().cloned().unwrap_or_default() {
            let l = format!(
                "stage=\"{}\",mode=\"{}\"",
                esc(a["stage"].as_str().unwrap_or("")),
                esc(m["mode"].as_str().unwrap_or(""))
            );
            if let Some(v) = m["top1_pct"].as_f64() {
                let _ = writeln!(o, "intarsia_qualify_accuracy_top1_pct{{{l}}} {v}");
            }
            if let Some(v) = m["mean_kl"].as_f64() {
                let _ = writeln!(kl, "intarsia_qualify_accuracy_mean_kl{{{l}}} {v}");
            }
        }
    }
    g(&mut o, "intarsia_qualify_accuracy_mean_kl", "Mean KL(reference || ours), nats.");
    o.push_str(&kl);
    g(
        &mut o,
        "intarsia_qualify_stage_pass",
        "1 when the stage passed, 0 when it failed or errored (skipped and report-only WARN stages are absent).",
    );
    let mut secs = String::new();
    for s in summary["stages"].as_array().cloned().unwrap_or_default() {
        let id = esc(s["id"].as_str().unwrap_or(""));
        match s["verdict"].as_str() {
            Some("PASS") => {
                let _ = writeln!(o, "intarsia_qualify_stage_pass{{stage=\"{id}\"}} 1");
            }
            Some("FAIL") | Some("ERROR") => {
                let _ = writeln!(o, "intarsia_qualify_stage_pass{{stage=\"{id}\"}} 0");
            }
            _ => {}
        }
        if let Some(w) = s["wall_s"].as_f64() {
            let _ = writeln!(secs, "intarsia_qualify_stage_seconds{{stage=\"{id}\"}} {w}");
        }
    }
    g(&mut o, "intarsia_qualify_stage_seconds", "Wall time of a stage.");
    o.push_str(&secs);
    g(
        &mut o,
        "intarsia_qualify_exit_code",
        "Exit code qualify would return for this section (0 pass, 1 gate failed, 2 pipeline failed or incomplete).",
    );
    let _ = writeln!(o, "intarsia_qualify_exit_code{{section=\"{section}\"}} {}", summary["exit_code"]);
    g(&mut o, "intarsia_qualify_section_seconds", "Wall time of the quick and the in-depth section.");
    let _ = writeln!(o, "intarsia_qualify_section_seconds{{section=\"quick\"}} {}", summary["wall_s"]["quick"]);
    let _ = writeln!(o, "intarsia_qualify_section_seconds{{section=\"full\"}} {}", summary["wall_s"]["full"]);
    g(&mut o, "intarsia_qualify_last_run_timestamp_seconds", "Unix time of the report.");
    let _ = writeln!(o, "intarsia_qualify_last_run_timestamp_seconds {}", summary["generated_unix"]);
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Map;

    fn rec(id: &str, section: &str, v: Verdict, reason: &str, metrics: Value) -> StageRecord {
        StageRecord {
            id: id.into(),
            title: format!("{id} title"),
            section: section.into(),
            verdict: v,
            reason: reason.into(),
            started_unix: 1,
            wall_s: 61.0,
            provenance: json!({"build_stamp": "unstamped", "host": "h", "commands": ["intarsia x"]}),
            artifacts: vec![format!("logs/{id}.out")],
            metrics,
            facts: Map::new(),
            notes: vec![],
        }
    }

    fn sample() -> (State, Meta) {
        let mut s = State::new(json!({}));
        s.put(rec("doctor", "quick", Verdict::Pass, "dev", json!({"device": {"name": "GPU", "arch": "gfx1201"}})));
        s.put(rec("recipe", "quick", Verdict::Pass, "builtin:qwen3 covers this model", json!({"todos": []})));
        s.put(rec("plan", "quick", Verdict::Pass, "ops", json!({"ops": [{"op": "matmul", "kernel": "k", "exactness": "fast", "admitted": true, "calls_per_decode_step": 3}]})));
        s.put(rec("kernels", "quick", Verdict::Skip, "cpu", Value::Null));
        s.put(rec("gate", "quick", Verdict::Pass, "perplexity 7", Value::Null));
        s.put(rec("acc-short", "quick", Verdict::Fail, "top-1 98%", json!({"golden": "g", "golden_how": "found", "gate_class": "bf16", "modes": [{"mode": "prefill", "positions": 10, "top1_pct": 98.0, "mean_kl": 1e-3, "mean_kl21": 1e-3, "floor_pass": false}]})));
        s.put(rec("bench", "quick", Verdict::Pass, "decode", json!({"gb_per_token": 16.4, "bandwidth_gbs": 627.0, "floor_inputs": "x", "rows": [{"size": 128, "prompt_tokens": 126.0, "decode_tok_s": 36.9, "prefill_tok_s": 2000.0, "pct_of_floor": 89.0}]})));
        s.put(rec("envelope", "full", Verdict::Pass, "written", json!({"use": "cmd"})));
        (
            s,
            Meta {
                model: "M".into(),
                model_dir: "/m".into(),
                backend: "hip".into(),
                tp: 1,
                reference: None,
                build_stamp: None,
                version: "0".into(),
                generated_unix: 5,
                quick_only: false,
            },
        )
    }

    /// Headings of `docs/qualify-report-template.md` and of a rendered report are the same, in the same order.
    #[test]
    fn rendered_report_follows_the_documented_template() {
        let (s, m) = sample();
        let template = include_str!("../../../../../docs/qualify-report-template.md");
        let heads = |t: &str| {
            t.lines()
                .filter(|l| l.starts_with("## ") || l.starts_with("### "))
                .map(|l| l.to_string())
                .collect::<Vec<_>>()
        };
        let full = render(&s, &m, true);
        assert_eq!(heads(&full), heads(template));
        assert!(full.contains("**Verdict: FAIL**"));
        let quick = render(&s, &m, false);
        assert_eq!(heads(&quick), heads(template));
        assert!(!quick.contains("`envelope`"));
        // headline first: the verdict comes before the stage table
        assert!(full.find("**Verdict").unwrap() < full.find("## Stages").unwrap());
    }

    #[test]
    fn summary_and_metrics() {
        let (s, m) = sample();
        let j = summary_json(&s, &m, false);
        assert_eq!(j["verdict"], "FAIL");
        assert_eq!(j["exit_code"], 1);
        assert_eq!(j["failures"][0]["stage"], "acc-short");
        assert_eq!(j["headline"]["speed"][0]["size"], 128);
        let t = prometheus_text(&j);
        assert!(t.contains("intarsia_qualify_decode_tok_s{size=\"128\"} 36.9"));
        assert!(t.contains("intarsia_qualify_stage_pass{stage=\"acc-short\"} 0"));
        assert!(
            !t.contains("intarsia_qualify_stage_pass{stage=\"kernels\"}"),
            "skipped stages are not reported as pass or fail: {t}"
        );
        assert!(t.contains("intarsia_qualify_exit_code{section=\"quick\"} 1"));
    }

    #[test]
    fn a_report_only_warn_does_not_fail_the_run() {
        let (mut s, m) = sample();
        s.put(rec("acc-format", "quick", Verdict::Pass, "prefill top-1 98.85% KL 4.40e-4", Value::Null));
        s.put(rec("acc-short", "quick", Verdict::Warn, "report-only: decode: KL 3.26e-2", Value::Null));
        let j = summary_json(&s, &m, false);
        assert_eq!(j["verdict"], "PASS");
        assert_eq!(j["exit_code"], 0);
        assert_eq!(j["failures"].as_array().unwrap().len(), 0);
        assert_eq!(j["warnings"][0]["stage"], "acc-short");
        let t = prometheus_text(&j);
        assert!(!t.contains("intarsia_qualify_stage_pass{stage=\"acc-short\"}"), "{t}");
        let r = render(&s, &m, false);
        assert!(r.contains("1 report-only (warn)"), "{r}");
        assert!(r.contains("- `acc-short` **WARN**: report-only"), "{r}");
        assert!(Verdict::Warn.is_done());
    }

    #[test]
    fn durations() {
        assert_eq!(fmt_dur(12.0), "12s");
        assert_eq!(fmt_dur(125.0), "2m05s");
        assert_eq!(fmt_dur(7300.0), "2h01m");
    }
}
