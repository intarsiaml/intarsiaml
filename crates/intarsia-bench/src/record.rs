//! Benchmark record v1 (model-benchmarks `schema/benchmark-record.v1.json`) from a `bench speed` result.
//!
//! `intarsia bench speed --emit-record PATH` appends one JSON line per measured cell: one per prompt length of the
//! speed sweep (a speed point carries prefill and decode tok/s together, so it covers both phases of that length) and
//! one per concurrency row. The sibling records of one bench run share `label` and the result file in
//! `provenance.url`. With `--record-run-id` (reusing a localeval run id) the whole run is ONE record instead, because
//! `schema/validate.py` rejects a run_id repeated within a file.
//!
//! Port notes (the Python is the spec; the site and the composer group on these strings):
//! * completeness: `schema/records.py:13-33` (`COMPARE_FIELDS`, `WANTED_FIELDS`, `finish`), see [`finish`].
//! * hardware names: `site/build_data.py:33-62` (`HOSTS`, `hardware()`), see [`site_hardware`].
//! * model family names: `site/build_data.py:67-101`, quant names: `site/build_data.py:106-121`. The site uses regexes
//!   over many models; this is the literal-substring subset for the models IntarsiaML serves, in the same order.
//!   Anything else needs `--model-family` / `--quant`, otherwise the field stays `"Unknown"` and the record is
//!   `incomplete` (kept, never dropped).
//! * secrets in the launch command: `site/build_data.py:29` (`SECRET_RE`), see [`scrub_secrets`].

use crate::http;
use anyhow::{Context, Result};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::time::Duration;

/// Default file the records are appended to when `--emit-record` has no value.
pub const DEFAULT_RECORDS_PATH: &str = "/opt/code/model-benchmarks/data/records/intarsia-bench.jsonl";

/// `schema/records.py:13-14`: a record missing one of these cannot be placed in a regime cell.
pub const COMPARE_FIELDS: &[&str] =
    &["hardware.name", "engine.name", "engine.version", "model.family", "model.repo", "quant.format", "workload.lane"];
/// `schema/records.py:16`: wanted for reproduction, listed in `missing` but not making a record incomplete.
pub const WANTED_FIELDS: &[&str] = &["model.revision", "quant.kv_dtype", "hardware.driver", "engine.launch.command"];

/// What the user told us about the server under test (flags), plus what the bench itself knows.
#[derive(Clone, Debug, Default)]
pub struct RecordMeta {
    /// Reuse this run id (a localeval run) and write one record for the whole run.
    pub record_run_id: Option<String>,
    pub hardware: Option<String>,
    pub model_family: Option<String>,
    pub model_repo: Option<String>,
    pub model_revision: Option<String>,
    pub quant: Option<String>,
    pub kv_dtype: Option<String>,
    /// The server's `--spec` value: `None`, `off` or a drafter name.
    pub spec: Option<String>,
    pub serve_cmd: Option<String>,
    pub serve_env: Vec<String>,
    pub recipe_ref: Option<String>,
    pub engine_version: Option<String>,
    pub engine_commit: Option<String>,
    /// `FLEETQ_JOB` when the bench ran inside a slot.
    pub slot: Option<String>,
    /// The bench command line as run.
    pub command: Option<String>,
    pub author: Option<String>,
    pub result_path: Option<String>,
    pub thinking: String,
    pub tp: usize,
    pub harness_version: String,
}

/// Facts read from the running server before the sweep.
#[derive(Clone, Debug, Default)]
pub struct ServerFacts {
    pub owned_by: Option<String>,
    pub root: Option<String>,
    pub max_model_len: Option<u64>,
    pub kv_dtype: Option<String>,
    pub backend: Option<String>,
    pub device: Option<String>,
    pub has_intarsia_info: bool,
}

impl ServerFacts {
    /// True when the server says it is IntarsiaML (`owned_by: intarsia` in /v1/models, or an `intarsia_info` metric).
    pub fn is_intarsia(&self) -> bool {
        self.owned_by.as_deref() == Some("intarsia") || self.has_intarsia_info
    }
}

/// Ask the server what it is: `/v1/models` (owner, root, context) and `/metrics` (`intarsia_info`: kv dtype, backend).
pub fn probe(url: &str, api_key: Option<&str>, model: &str) -> ServerFacts {
    let agent = http::agent(Some(Duration::from_secs(30)));
    let get = |u: String| -> Option<String> {
        let mut req = agent.get(&u);
        if let Some(k) = api_key {
            req = req.header("Authorization", format!("Bearer {k}"));
        }
        let mut resp = req.call().ok()?;
        resp.status().is_success().then(|| resp.body_mut().read_to_string().ok()).flatten()
    };
    let mut f = ServerFacts::default();
    if let Some(v) = get(format!("{}/models", http::api_base(url))).and_then(|t| serde_json::from_str::<Value>(&t).ok()) {
        if let Some(models) = v["data"].as_array() {
            if let Some(m) = models.iter().find(|m| m["id"].as_str() == Some(model)).or_else(|| models.first()) {
                f.owned_by = m["owned_by"].as_str().map(str::to_string);
                f.root = m["root"].as_str().map(str::to_string);
                f.max_model_len = m["max_model_len"].as_u64();
            }
        }
    }
    if let Some(t) = get(format!("{}/metrics", http::origin(url))) {
        if let Some(l) = parse_info_labels(&t) {
            f.has_intarsia_info = true;
            f.kv_dtype = l.get("kv_dtype").cloned();
            f.backend = l.get("backend").cloned();
            f.device = l.get("device").cloned();
        }
    }
    f
}

/// Labels of the `intarsia_info{...} 1` line (server `metrics.rs`: model, family, backend, device, act_dtype, kv_dtype).
pub fn parse_info_labels(metrics: &str) -> Option<BTreeMap<String, String>> {
    let line = metrics.lines().find(|l| l.starts_with("intarsia_info{"))?;
    let body = line.strip_prefix("intarsia_info{")?.rsplit_once('}')?.0;
    let mut out = BTreeMap::new();
    let mut rest = body;
    while let Some((k, after)) = rest.split_once("=\"") {
        let mut val = String::new();
        let mut chars = after.char_indices();
        let mut end = None;
        while let Some((i, c)) = chars.next() {
            match c {
                '\\' => {
                    if let Some((_, n)) = chars.next() {
                        val.push(n);
                    }
                }
                '"' => {
                    end = Some(i + 1);
                    break;
                }
                c => val.push(c),
            }
        }
        out.insert(k.trim_start_matches(',').trim().to_string(), val);
        rest = &after[end?..];
    }
    Some(out)
}

// -------------------------------------------------------------------------------------------- site names

/// `site/build_data.py:33-44`: server host -> (fleet box, site hardware name).
const HOSTS: &[(&str, &str, &str)] = &[
    ("tatooine", "tatooine", "Radeon AI PRO R9700 (tatooine)"),
    ("spark-4e77", "spark1", "DGX Spark GB10"),
    ("192.168.1.95", "spark1", "DGX Spark GB10"),
    ("spark-fa00", "spark2", "DGX Spark GB10"),
    ("100.96.95.89", "spark2", "DGX Spark GB10"),
    ("spark1", "spark1", "DGX Spark GB10"),
    ("spark2", "spark2", "DGX Spark GB10"),
    ("endor", "endor", "M5 Max 128 GB (endor)"),
];

/// Host part of a server URL.
pub fn url_host(url: &str) -> String {
    http::origin(url).split("://").last().unwrap_or("").split('/').next().unwrap_or("").split(':').next().unwrap_or("").to_string()
}

/// (box, site hardware name) for the machine the server runs on, after `site/build_data.py:48-62`. `host` is the URL
/// host, or the local hostname when the URL is loopback. Two sparks with `--tp 2` are "2x DGX Spark GB10 (TP2)"
/// (`build_data.py:60-61`; one spark has one GPU, so tp 2 on a spark means both). An unknown machine is `"Unknown"`.
pub fn site_hardware(host: &str, local_hostname: Option<&str>, tp: usize) -> (String, String) {
    let h = host.to_ascii_lowercase();
    let loopback = matches!(h.as_str(), "127.0.0.1" | "localhost" | "::1" | "[::1]" | "0.0.0.0");
    // a DNS name loses its domain (`endor.kuepker.haus` -> `endor`); an IP address keeps its dots
    let short = |n: &str| if n.chars().all(|c| c.is_ascii_digit() || c == '.') { n.to_string() } else { n.split('.').next().unwrap_or("").to_string() };
    let h = if loopback { short(&local_hostname.unwrap_or("").to_ascii_lowercase()) } else { short(&h) };
    let (bx, mut name) = HOSTS
        .iter()
        .find(|(k, _, _)| *k == h)
        .map(|(_, b, n)| (b.to_string(), n.to_string()))
        .unwrap_or_else(|| (h.clone(), "Unknown".to_string()));
    if name == "DGX Spark GB10" && tp == 2 {
        name = "2x DGX Spark GB10 (TP2)".into();
    }
    (bx, name)
}

/// First match wins, in the order of `site/build_data.py:67-90` (literal substrings of the lower-cased text).
const FAMILIES: &[(&[&str], &str)] = &[
    (&["reranker"], "Qwen3-Reranker-0.6B"),
    (&["flash-next", "flashnext"], "Qwen3.8-Flash-Next"),
    (&["qwen3.6-35b", "qwen36-35b", "qwen36", "fx3-35b", "intarsia-35b", "q36"], "Qwen3.6-35B-A3B"),
    (&["qwen3.5-35b", "qwen35-35b"], "Qwen3.5-35B-A3B"),
    (&["qwen3.5-9b", "qwen35-9b"], "Qwen3.5-9B"),
    (&["qwen3.8-27b", "qwen38-27b", "qwen27", "q38"], "Qwen3.8-27B"),
    (&["qwen3-8b", "qwen3_8b", "fx3-8b"], "Qwen3-8B"),
];

/// `site/build_data.py:93-101`: the strongest signal first (the served root, then the model name).
pub fn site_family(parts: &[&str]) -> Option<String> {
    for p in parts.iter().filter(|p| !p.is_empty()) {
        let low = p.to_ascii_lowercase();
        for (pats, fam) in FAMILIES {
            if pats.iter().any(|k| low.contains(k)) {
                return Some((*fam).to_string());
            }
        }
    }
    None
}

fn has_word(text: &str, w: &str) -> bool {
    let b = text.as_bytes();
    text.match_indices(w).any(|(i, _)| {
        let before = i == 0 || !b[i - 1].is_ascii_alphanumeric();
        let after = i + w.len() >= b.len() || !b[i + w.len()].is_ascii_alphanumeric();
        before && after
    })
}

/// `site/build_data.py:106-121` restricted to the formats IntarsiaML loads, same relative order.
pub fn site_quant(parts: &[&str]) -> Option<String> {
    let text = parts.join(" ").to_ascii_lowercase();
    let q = if text.contains("nvfp4") {
        "NVFP4"
    } else if text.contains("mxfp4") {
        "MXFP4"
    } else if text.contains("int4") {
        "INT4"
    } else if text.contains("int8") {
        "INT8"
    } else if has_word(&text, "fp8") || text.contains("-fp8") {
        "FP8"
    } else if text.contains("bf16") {
        "BF16"
    } else {
        return None;
    };
    Some(q.to_string())
}

/// `site/build_data.py:29`: `(--api-key[= ]|Bearer |sk-)` followed by 8+ token characters is replaced.
pub fn scrub_secrets(s: &str) -> String {
    let is_tok = |c: char| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-');
    let mut out = String::new();
    let mut i = 0;
    while i < s.len() {
        let rest = &s[i..];
        let pre = ["--api-key=", "--api-key ", "Bearer ", "sk-"].iter().find(|p| rest.starts_with(**p));
        if let Some(p) = pre {
            let after = &rest[p.len()..];
            let n = after.chars().take_while(|c| is_tok(*c)).count();
            if n >= 8 {
                out.push_str(p);
                out.push_str("<redacted>");
                i += p.len() + n;
                continue;
            }
        }
        let c = rest.chars().next().unwrap();
        out.push(c);
        i += c.len_utf8();
    }
    out
}

/// `models--Org--Name/snapshots/<sha>` in a path -> ("Org/Name", Some(sha)).
pub fn hf_from_path(p: &str) -> Option<(String, Option<String>)> {
    let seg = p.split('/').find(|s| s.starts_with("models--"))?;
    let repo = seg.trim_start_matches("models--").replacen("--", "/", 1);
    let mut it = p.split('/').skip_while(|s| *s != "snapshots");
    it.next();
    let rev = it.next().filter(|s| s.len() >= 7 && s.bytes().all(|b| b.is_ascii_hexdigit())).map(str::to_string);
    Some((repo, rev))
}

/// The lane of the run: an explicit `--spec`, else a `--spec X` inside the launch command, else `plain`.
/// `spec:<drafter>` for any drafter but `off`/`none` (`schema/README.md` Lane rule).
pub fn lane_of(spec: Option<&str>, serve_cmd: Option<&str>) -> String {
    let from_cmd = serve_cmd.and_then(|c| {
        let t: Vec<&str> = c.split_whitespace().collect();
        t.iter().position(|x| *x == "--spec").and_then(|i| t.get(i + 1).copied()).or_else(|| t.iter().find_map(|x| x.strip_prefix("--spec=")))
    });
    match spec.or(from_cmd).map(str::trim) {
        None | Some("") | Some("off") | Some("none") => "plain".into(),
        Some(d) => format!("spec:{d}"),
    }
}

/// Unix seconds -> `YYYY-MM-DDTHH:MM:SSZ` (civil-from-days, proleptic Gregorian).
pub fn iso_utc(secs: u64) -> String {
    let (days, rem) = ((secs / 86400) as i64, secs % 86400);
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem % 3600 / 60, rem % 60)
}

// ------------------------------------------------------------------------------------------ completeness

fn get<'a>(rec: &'a Value, path: &str) -> Option<&'a Value> {
    path.split('.').try_fold(rec, |v, k| v.as_object().and_then(|o| o.get(k)))
}

fn blank(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => true,
        Some(Value::String(s)) => s.is_empty() || s == "Unknown",
        _ => false,
    }
}

/// Port of `schema/records.py:27-33` `finish(rec)`: every blank compare or wanted field goes to
/// `completeness.missing`; the record is `incomplete` iff a blank field is a compare field. Blank is null, `""` or
/// `"Unknown"` (`records.py:29`).
pub fn finish(rec: &mut Value) {
    let missing: Vec<&str> = COMPARE_FIELDS.iter().chain(WANTED_FIELDS).copied().filter(|p| blank(get(rec, p))).collect();
    let status = if missing.iter().any(|p| COMPARE_FIELDS.contains(p)) { "incomplete" } else { "complete" };
    rec["completeness"] = json!({"status": status, "missing": missing});
}

// ---------------------------------------------------------------------------------------------- records

fn opt_str(v: &Option<String>) -> Value {
    v.as_ref().map_or(Value::Null, |s| Value::String(s.clone()))
}

fn s_or_unknown(v: Option<String>) -> String {
    v.filter(|s| !s.is_empty()).unwrap_or_else(|| "Unknown".into())
}

/// The setup shared by every record of one run (everything but the measured points).
fn base_record(res: &Value, m: &RecordMeta, facts: &ServerFacts, local_hostname: Option<&str>, run_id: &str) -> Value {
    let fp = &res["fingerprint"];
    let host_fp = &fp["host"];
    let server_host = url_host(res["url"].as_str().unwrap_or(""));
    // a fingerprint taken on the server machine names it best; otherwise the URL host (loopback: this machine)
    let fp_host = host_fp["host"]["hostname"].as_str().or(host_fp["hostname"].as_str());
    let host_for_hw = if matches!(server_host.as_str(), "127.0.0.1" | "localhost" | "::1" | "0.0.0.0") {
        fp_host.map(str::to_string).or_else(|| local_hostname.map(str::to_string)).unwrap_or(server_host.clone())
    } else {
        server_host.clone()
    };
    let tp = m.tp.max(1);
    let (bx, hw_name) = site_hardware(&host_for_hw, local_hostname, tp);
    let hw_name = m.hardware.clone().unwrap_or(hw_name);

    let model_name = res["model"].as_str().unwrap_or("");
    let root = facts.root.as_deref().unwrap_or("");
    let (hf_repo, hf_rev) = hf_from_path(root).unzip();
    let repo = m.model_repo.clone().or(hf_repo).or_else(|| facts.root.clone());
    let revision = m.model_revision.clone().or(hf_rev.flatten());
    let family = m.model_family.clone().or_else(|| site_family(&[root, repo.as_deref().unwrap_or(""), model_name, res["label"].as_str().unwrap_or("")]));
    let quant = m.quant.clone().or_else(|| site_quant(&[root, repo.as_deref().unwrap_or(""), model_name]));
    let kv = m.kv_dtype.clone().or_else(|| facts.kv_dtype.clone());

    let mut env = Map::new();
    for kv in &m.serve_env {
        if let Some((k, v)) = kv.split_once('=') {
            let secret = ["KEY", "TOKEN", "SECRET", "PASSWORD"].iter().any(|w| k.to_ascii_uppercase().contains(w));
            env.insert(k.to_string(), Value::String(if secret { "<redacted>".into() } else { scrub_secrets(v) }));
        }
    }
    let cmd = m.serve_cmd.as_deref().map(scrub_secrets);
    let args: Value = cmd
        .as_deref()
        .map(|c| Value::Array(c.split_whitespace().skip(1).map(|t| Value::String(t.into())).collect()))
        .unwrap_or(Value::Null);

    let trusted = m.slot.is_some();
    let mut notes = vec![format!(
        "intarsia bench speed {}; version and commit are the bench binary's own unless --engine-version/--engine-commit were given",
        if trusted { "inside a fleetq slot" } else { "outside a fleetq slot (trust external: self-reported, not counted in regime wins)" }
    )];
    if let Some(c) = fp["prefill_chunk"].as_u64() {
        notes.push(format!("server prefill chunk {c}"));
    }
    if let Some(b) = &facts.backend {
        notes.push(format!("server backend {b}{}", facts.device.as_ref().map(|d| format!(" ({d})")).unwrap_or_default()));
    }

    let (driver, sku, kernel, mem) = doctor_facts(host_fp);
    let mut rec = json!({
        "record_version": "1",
        "run_id": run_id,
        "date": iso_utc(res["timestamp_unix"].as_u64().unwrap_or(0)),
        "label": res["label"],
        "notes": notes.join("; "),
        "provenance": {
            "source": "intarsia-bench",
            "url": opt_str(&m.result_path),
            "author": opt_str(&m.author),
            "where": bx,
            "import_tool": "intarsia bench speed --emit-record",
            "import_tool_version": m.harness_version,
            "imported_at": null,
            "trust": if trusted { "measured-here" } else { "external" },
            "reproduced_by": null,
            "docs": [],
        },
        "hardware": {
            "name": hw_name, "sku": sku, "box": bx, "devices": if tp > 1 { json!(tp) } else { Value::Null },
            "tp": tp, "pp": null, "driver": driver, "kernel": kernel, "power_mode": null, "memory_gb": mem,
        },
        "engine": {
            "name": "IntarsiaML",
            "name_source": if facts.is_intarsia() { "server reports intarsia" } else { "bench flag" },
            "version": opt_str(&m.engine_version),
            "commit": opt_str(&m.engine_commit),
            "fork_of": null,
            "launch": {
                "command": opt_str(&cmd), "args": args,
                "env": if env.is_empty() { Value::Null } else { Value::Object(env) },
                "image": null, "recipe_ref": opt_str(&m.recipe_ref),
                "max_model_len": facts.max_model_len,
            },
            "fingerprint": null,
        },
        "model": {
            "family": s_or_unknown(family), "repo": opt_str(&repo), "revision": opt_str(&revision),
            "served_name": model_name, "arch": null,
        },
        "quant": {"format": opt_str(&quant), "bits_per_weight": null, "tensors": null, "activations": null, "kv_dtype": opt_str(&kv)},
        "workload": {
            "lane": lane_of(m.spec.as_deref(), m.serve_cmd.as_deref()),
            "thinking": m.thinking,
            "sampling": {"temperature": 0.0, "top_p": null, "top_k": null, "seed": null},
            "extra_body": null,
        },
        "measurement": {
            "harness": "intarsia-bench", "harness_version": m.harness_version, "command": opt_str(&m.command.as_deref().map(scrub_secrets)),
            "slot": opt_str(&m.slot), "exclusive": null,
            "reps": res["config"]["reps"], "warmup_reps": res["config"]["warmup"], "gen_tokens": res["config"]["gen"],
            "warnings": warnings_text(res),
        },
    });
    // required-string fields a server cannot always fill
    if rec["hardware"]["name"].as_str().is_none_or(str::is_empty) {
        rec["hardware"]["name"] = "Unknown".into();
    }
    rec
}

fn warnings_text(res: &Value) -> Value {
    let w: Vec<&str> = res["warnings"].as_array().map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
    if w.is_empty() {
        Value::Null
    } else {
        Value::String(w.join("; "))
    }
}

/// (driver, sku, kernel, memory_gb) from an `intarsia doctor --json` fingerprint stored under `fingerprint.host`.
fn doctor_facts(d: &Value) -> (Value, Value, Value, Value) {
    if !d.is_object() {
        return (Value::Null, Value::Null, Value::Null, Value::Null);
    }
    let smi = &d["gpu"]["nvidia_smi"];
    let driver = if let Some(v) = smi["driver_version"].as_str() {
        let cuda = d["gpu"]["cuda_toolkit_version"].as_str().map(|c| format!(", CUDA {c}")).unwrap_or_default();
        json!(format!("NVIDIA {v}{cuda}"))
    } else if let Some(r) = d["gpu"]["rocm_version"].as_str() {
        json!(format!("ROCm {}", r.trim()))
    } else if let Some(o) = d["host"]["os"].as_str() {
        json!(o)
    } else {
        Value::Null
    };
    let sku = match (smi["name"].as_str(), smi["compute_cap"].as_str()) {
        (Some(n), Some(cc)) => json!(format!("{n} (sm_{})", cc.replace('.', ""))),
        _ => Value::Null,
    };
    let mem = d["host"]["ram_total_bytes"].as_f64().map(|b| json!((b / 1e9 * 10.0).round() / 10.0)).unwrap_or(Value::Null);
    (driver, sku, d["host"]["kernel"].clone(), mem)
}

fn spread_pct(row: &Value) -> Value {
    match (row["decode_tok_s_min"].as_f64(), row["decode_tok_s_max"].as_f64(), row["decode_tok_s"].as_f64()) {
        (Some(lo), Some(hi), Some(med)) if med > 0.0 && row["reps_ok"].as_u64().unwrap_or(0) > 1 => json!(((hi - lo) / med * 1000.0).round() / 10.0),
        _ => Value::Null,
    }
}

fn speed_point(row: &Value) -> Value {
    json!({
        "prompt_tokens": row["prompt_tokens"].as_f64().map(|x| x.round() as u64).or(row["target_prompt_tokens"].as_u64()),
        "target_tokens": row["target_prompt_tokens"],
        "gen_tokens": row["gen_tokens"].as_f64().map(|x| x.round() as u64),
        "prefill_tps": row["prefill_tok_s"],
        "prefill_spread_pct": null,
        "decode_tps": row["decode_tok_s"],
        "decode_spread_pct": spread_pct(row),
        "ttft_s": row["ttft_s"],
        "itl_ms_p50": null, "itl_ms_p99": null,
        "cached_tokens": null,
        "forced_prefill": true,
    })
}

fn conc_point(row: &Value, gen: &Value) -> Value {
    json!({
        "streams": row["streams"], "prompt_tokens": row["target_prompt_tokens"], "gen_tokens": gen,
        "aggregate_tps": row["aggregate_decode_tok_s"], "per_stream_tps": row["per_stream_decode_tok_s"],
        "ttft_s": row["ttft_s"], "itl_ms_p50": null, "itl_ms_p99": null, "errors": row["errors"],
    })
}

/// A speed row counts as measured when it was not skipped and at least one rep succeeded.
fn measured(row: &Value) -> bool {
    !row["skipped"].is_string() && row["reps_ok"].as_u64().unwrap_or(0) > 0
}

/// Records for one `bench speed` result (`run_speed` JSON): one per measured prompt length and one per concurrency
/// row, or one record holding all of them when `meta.record_run_id` reuses a localeval run id. Each record is passed
/// through [`finish`].
pub fn build_records(res: &Value, meta: &RecordMeta, facts: &ServerFacts, local_hostname: Option<&str>) -> Vec<Value> {
    let rows: Vec<&Value> = res["rows"].as_array().map(|a| a.iter().filter(|r| measured(r)).collect()).unwrap_or_default();
    let conc: Vec<&Value> =
        res["concurrency"].as_array().map(|a| a.iter().filter(|r| r["aggregate_decode_tok_s"].is_number()).collect()).unwrap_or_default();
    let gen = &res["config"]["gen"];
    let base_id = format!("intarsia-bench:{}-{}", res["label"].as_str().unwrap_or("run"), res["timestamp_unix"].as_u64().unwrap_or(0));
    let mut out = vec![];
    if let Some(id) = &meta.record_run_id {
        let mut r = base_record(res, meta, facts, local_hostname, id);
        r["speed"] = Value::Array(rows.iter().map(|x| speed_point(x)).collect());
        r["concurrency"] = Value::Array(conc.iter().map(|x| conc_point(x, gen)).collect());
        out.push(r);
    } else {
        for row in rows {
            let mut r = base_record(res, meta, facts, local_hostname, &format!("{base_id}:p{}", row["target_prompt_tokens"]));
            r["speed"] = json!([speed_point(row)]);
            out.push(r);
        }
        for row in conc {
            let mut r = base_record(res, meta, facts, local_hostname, &format!("{base_id}:c{}", row["streams"]));
            r["concurrency"] = json!([conc_point(row, gen)]);
            out.push(r);
        }
    }
    for r in &mut out {
        finish(r);
    }
    out
}

/// Append one JSON line per record (one `write_all` each, so concurrent writers do not interleave inside a line).
pub fn append_jsonl(path: &Path, records: &[Value]) -> Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path).with_context(|| format!("opening {}", path.display()))?;
    for r in records {
        let mut line = serde_json::to_string(r)?;
        line.push('\n');
        f.write_all(line.as_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result() -> Value {
        json!({
            "kind": "intarsia-bench-speed", "label": "t1", "timestamp_unix": 1_761_177_600u64,
            "url": "http://spark-4e77:8090", "model": "qwen36-35b-a3b-nvfp4",
            "config": {"sizes": [1024, 4096, 16384], "gen": 128, "reps": 3, "warmup": 1, "concurrency": [1, 4], "conc_size": 1024},
            "fingerprint": {"tp": 1, "prefill_chunk": 2048},
            "warnings": ["size 16384: skipped"],
            "rows": [
                {"target_prompt_tokens": 1024, "prompt_tokens": 1010.0, "gen_tokens": 128.0, "ttft_s": 0.25, "prefill_tok_s": 4040.0,
                 "decode_tok_s": 70.0, "decode_tok_s_min": 69.0, "decode_tok_s_max": 71.4, "reps_ok": 3, "errors": 0},
                {"target_prompt_tokens": 4096, "prompt_tokens": 4088.0, "gen_tokens": 128.0, "ttft_s": 0.6, "prefill_tok_s": 6845.0,
                 "decode_tok_s": 68.0, "decode_tok_s_min": 68.0, "decode_tok_s_max": 68.0, "reps_ok": 1, "errors": 0},
                {"target_prompt_tokens": 16384, "prompt_tokens": null, "skipped": "too long", "reps_ok": 0}
            ],
            "concurrency": [
                {"streams": 4, "target_prompt_tokens": 1024, "ttft_s": 1.5, "per_stream_decode_tok_s": 20.0, "aggregate_decode_tok_s": 75.2, "errors": 0}
            ],
        })
    }

    fn facts() -> ServerFacts {
        ServerFacts {
            owned_by: Some("intarsia".into()),
            root: Some("/home/x/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/0123456789abcdef0123456789abcdef01234567".into()),
            max_model_len: Some(32768),
            kv_dtype: Some("bf16".into()),
            backend: Some("cuda".into()),
            device: Some("NVIDIA GB10".into()),
            has_intarsia_info: true,
        }
    }

    fn meta(slot: Option<&str>) -> RecordMeta {
        RecordMeta {
            thinking: "off".into(),
            tp: 1,
            engine_version: Some("0.1.0".into()),
            engine_commit: Some("abc1234".into()),
            serve_cmd: Some("intarsia serve --backend cuda --model /m --api-key sk-abcdef123456".into()),
            slot: slot.map(str::to_string),
            harness_version: "0.1.0".into(),
            ..Default::default()
        }
    }

    #[test]
    fn one_record_per_measured_cell() {
        let recs = build_records(&result(), &meta(Some("4401")), &facts(), None);
        let ids: Vec<&str> = recs.iter().map(|r| r["run_id"].as_str().unwrap()).collect();
        // the skipped 16k row is dropped; 2 speed cells + 1 concurrency cell
        assert_eq!(ids, ["intarsia-bench:t1-1761177600:p1024", "intarsia-bench:t1-1761177600:p4096", "intarsia-bench:t1-1761177600:c4"]);
        let p = &recs[0];
        assert_eq!(p["speed"].as_array().unwrap().len(), 1);
        assert_eq!(p["speed"][0]["prefill_tps"], 4040.0);
        assert_eq!(p["speed"][0]["decode_tps"], 70.0);
        assert_eq!(p["speed"][0]["prompt_tokens"], 1010);
        assert_eq!(p["speed"][0]["target_tokens"], 1024);
        // the bench stores medians as floats; the record's gen_tokens is an integer
        assert!(p["speed"][0]["gen_tokens"].is_u64() && p["speed"][0]["gen_tokens"] == 128);
        // (71.4-69.0)/70 = 3.43 %
        assert_eq!(p["speed"][0]["decode_spread_pct"], 3.4);
        // a single successful rep has no spread
        assert!(recs[1]["speed"][0]["decode_spread_pct"].is_null());
        assert!(p.get("concurrency").is_none());
        let c = &recs[2];
        assert_eq!(c["concurrency"][0]["streams"], 4);
        assert_eq!(c["concurrency"][0]["aggregate_tps"], 75.2);
        assert_eq!(c["concurrency"][0]["gen_tokens"], 128);
        assert!(c.get("speed").is_none());
    }

    #[test]
    fn setup_fields() {
        let r = &build_records(&result(), &meta(Some("4401")), &facts(), None)[0];
        assert_eq!(r["record_version"], "1");
        assert_eq!(r["date"], "2025-10-23T00:00:00Z");
        assert_eq!(r["engine"]["name"], "IntarsiaML");
        assert_eq!(r["engine"]["version"], "0.1.0");
        assert_eq!(r["engine"]["commit"], "abc1234");
        assert_eq!(r["hardware"]["name"], "DGX Spark GB10");
        assert_eq!(r["hardware"]["box"], "spark1");
        assert_eq!(r["model"]["family"], "Qwen3.6-35B-A3B");
        assert_eq!(r["model"]["repo"], "nvidia/Qwen3.6-35B-A3B-NVFP4");
        assert_eq!(r["model"]["revision"], "0123456789abcdef0123456789abcdef01234567");
        assert_eq!(r["quant"]["format"], "NVFP4");
        assert_eq!(r["quant"]["kv_dtype"], "bf16");
        assert_eq!(r["workload"]["lane"], "plain");
        assert_eq!(r["workload"]["thinking"], "off");
        assert_eq!(r["engine"]["launch"]["max_model_len"], 32768);
        // the secret in the launch command never reaches the record
        let cmd = r["engine"]["launch"]["command"].as_str().unwrap();
        assert!(!cmd.contains("abcdef123456") && cmd.contains("--api-key <redacted>"), "{cmd}");
        assert_eq!(r["engine"]["launch"]["args"][0], "serve");
        // no doctor fingerprint was given, so only the driver is unknown: still complete (a wanted field, not a compare field)
        assert_eq!(r["completeness"], json!({"status": "complete", "missing": ["hardware.driver"]}));
    }

    #[test]
    fn trust_follows_the_fleetq_slot() {
        let in_slot = &build_records(&result(), &meta(Some("4401")), &facts(), None)[0];
        assert_eq!(in_slot["provenance"]["trust"], "measured-here");
        assert_eq!(in_slot["measurement"]["slot"], "4401");
        let no_slot = &build_records(&result(), &meta(None), &facts(), None)[0];
        assert_eq!(no_slot["provenance"]["trust"], "external");
        assert!(no_slot["measurement"]["slot"].is_null());
        assert!(no_slot["notes"].as_str().unwrap().contains("outside a fleetq slot"));
    }

    #[test]
    fn finish_matches_records_py() {
        // records.py:27-33: blank = null, "" or "Unknown"; only a COMPARE field makes the record incomplete
        let mut r = json!({"hardware": {"name": "Unknown"}, "engine": {"name": "IntarsiaML", "version": "1", "launch": {"command": null}},
                           "model": {"family": "F", "repo": "r", "revision": null}, "quant": {"format": "NVFP4", "kv_dtype": null},
                           "workload": {"lane": "plain"}});
        finish(&mut r);
        assert_eq!(r["completeness"]["status"], "incomplete");
        assert_eq!(r["completeness"]["missing"], json!(["hardware.name", "model.revision", "quant.kv_dtype", "hardware.driver", "engine.launch.command"]));
        r["hardware"] = json!({"name": "DGX Spark GB10"});
        finish(&mut r);
        // only wanted fields missing: still complete
        assert_eq!(r["completeness"]["status"], "complete");
        assert_eq!(r["completeness"]["missing"], json!(["model.revision", "quant.kv_dtype", "hardware.driver", "engine.launch.command"]));
    }

    #[test]
    fn unknown_setup_is_kept_and_marked_incomplete() {
        let mut f = facts();
        f.root = Some("mystery".into());
        f.kv_dtype = None;
        let mut res = result();
        res["url"] = json!("http://10.9.9.9:8090");
        res["model"] = json!("m1");
        let mut m = meta(None);
        m.serve_cmd = None;
        let r = &build_records(&res, &m, &f, None)[0];
        assert_eq!(r["hardware"]["name"], "Unknown");
        assert_eq!(r["completeness"]["status"], "incomplete");
        let miss = r["completeness"]["missing"].as_array().unwrap();
        for p in ["hardware.name", "model.family", "quant.format", "engine.launch.command", "quant.kv_dtype"] {
            assert!(miss.iter().any(|x| x == p), "{p} in {miss:?}");
        }
    }

    #[test]
    fn hardware_names_follow_the_site() {
        assert_eq!(site_hardware("spark-4e77", None, 1), ("spark1".into(), "DGX Spark GB10".into()));
        assert_eq!(site_hardware("spark-fa00", None, 2).1, "2x DGX Spark GB10 (TP2)");
        assert_eq!(site_hardware("tatooine", None, 2).1, "Radeon AI PRO R9700 (tatooine)");
        assert_eq!(site_hardware("127.0.0.1", Some("endor.local"), 1), ("endor".into(), "M5 Max 128 GB (endor)".into()));
        assert_eq!(site_hardware("endor.kuepker.haus", None, 1), ("endor".into(), "M5 Max 128 GB (endor)".into()));
        assert_eq!(site_hardware("192.168.1.95", None, 1), ("spark1".into(), "DGX Spark GB10".into()));
        assert_eq!(site_hardware("10.1.2.3", None, 1).1, "Unknown");
        assert_eq!(url_host("http://spark-4e77:8090/v1"), "spark-4e77");
        assert_eq!(url_host("localhost:8090"), "localhost");
    }

    #[test]
    fn family_and_quant_follow_the_site() {
        assert_eq!(site_family(&["", "Qwen/Qwen3-8B"]).as_deref(), Some("Qwen3-8B"));
        assert_eq!(site_family(&["models--Qwen--Qwen3-Reranker-0.6B"]).as_deref(), Some("Qwen3-Reranker-0.6B"));
        assert_eq!(site_family(&["fx3-35b-intarsia"]).as_deref(), Some("Qwen3.6-35B-A3B"));
        assert_eq!(site_family(&["nothing"]), None);
        assert_eq!(site_quant(&["nvidia/Qwen3.6-35B-A3B-NVFP4"]).as_deref(), Some("NVFP4"));
        assert_eq!(site_quant(&["amd/Qwen3.8-27B-MXFP4"]).as_deref(), Some("MXFP4"));
        assert_eq!(site_quant(&["Qwen3-8B-FP8"]).as_deref(), Some("FP8"));
        assert_eq!(site_quant(&["Qwen3-8B"]), None);
    }

    #[test]
    fn lane_is_plain_unless_a_drafter_runs() {
        assert_eq!(lane_of(None, None), "plain");
        assert_eq!(lane_of(Some("off"), None), "plain");
        assert_eq!(lane_of(Some("mtp"), None), "spec:mtp");
        assert_eq!(lane_of(None, Some("intarsia serve --spec ngram --model m")), "spec:ngram");
        assert_eq!(lane_of(None, Some("intarsia serve --spec=off")), "plain");
        // an explicit flag wins over the command
        assert_eq!(lane_of(Some("mtp"), Some("intarsia serve --spec off")), "spec:mtp");
        let mut m = meta(None);
        m.spec = Some("ngram".into());
        assert_eq!(build_records(&result(), &m, &facts(), None)[0]["workload"]["lane"], "spec:ngram");
    }

    #[test]
    fn localeval_run_id_is_reused_in_one_record() {
        let mut m = meta(Some("12"));
        m.record_run_id = Some("20261022-120000-adhoc-x".into());
        let recs = build_records(&result(), &m, &facts(), None);
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0]["run_id"], "20261022-120000-adhoc-x");
        assert_eq!(recs[0]["speed"].as_array().unwrap().len(), 2);
        assert_eq!(recs[0]["concurrency"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn helpers() {
        assert_eq!(iso_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso_utc(1_700_000_000), "2023-11-14T22:13:20Z");
        assert_eq!(iso_utc(1_782_864_000), "2026-07-01T00:00:00Z");
        assert_eq!(scrub_secrets("serve --api-key=abcdefgh12 x Bearer tok_12345678 y sk-1234567890 z sk-short"), "serve --api-key=<redacted> x Bearer <redacted> y sk-<redacted> z sk-short");
        assert_eq!(hf_from_path("/a/models--Org--Name-1/snapshots/abcdef0123/x"), Some(("Org/Name-1".into(), Some("abcdef0123".into()))));
        assert_eq!(hf_from_path("/plain/dir"), None);
        let l = parse_info_labels("x 1\nintarsia_info{model=\"m\",family=\"f\",backend=\"cuda\",device=\"NVIDIA GB10\",act_dtype=\"bf16\",kv_dtype=\"fp8_e4m3\"} 1\n").unwrap();
        assert_eq!(l["kv_dtype"], "fp8_e4m3");
        assert_eq!(l["device"], "NVIDIA GB10");
        assert!(parse_info_labels("nothing 1").is_none());
    }

    #[test]
    fn doctor_fingerprint_fills_driver_and_sku() {
        let mut res = result();
        res["fingerprint"]["host"] = json!({
            "host": {"hostname": "spark-4e77", "kernel": "Linux 6.17.0", "os": "Ubuntu", "ram_total_bytes": 121_700_000_000u64},
            "gpu": {"nvidia_smi": {"name": "NVIDIA GB10", "driver_version": "580.1", "compute_cap": "12.1"}, "cuda_toolkit_version": "13.0"},
        });
        let r = &build_records(&res, &meta(None), &facts(), None)[0];
        assert_eq!(r["hardware"]["driver"], "NVIDIA 580.1, CUDA 13.0");
        assert_eq!(r["hardware"]["sku"], "NVIDIA GB10 (sm_121)");
        assert_eq!(r["hardware"]["kernel"], "Linux 6.17.0");
        assert_eq!(r["hardware"]["memory_gb"], 121.7);
    }

    #[test]
    fn records_append_as_lines() {
        let dir = std::env::temp_dir().join(format!("intarsia-rec-{}", std::process::id()));
        let path = dir.join("sub").join("r.jsonl");
        let recs = build_records(&result(), &meta(None), &facts(), None);
        append_jsonl(&path, &recs).unwrap();
        append_jsonl(&path, &recs[..1]).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text.lines().count(), recs.len() + 1);
        for l in text.lines() {
            let v: Value = serde_json::from_str(l).unwrap();
            assert_eq!(v["record_version"], "1");
        }
        std::fs::remove_dir_all(dir).ok();
    }
}
