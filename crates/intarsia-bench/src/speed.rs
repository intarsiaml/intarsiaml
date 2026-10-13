//! End-to-end speed sweep against any OpenAI-compatible server.
//!
//! Methodology (same as the localeval harness so numbers are comparable): every prompt starts
//! with a fresh random nonce so prefix caches cannot help, the output length is forced with
//! `ignore_eos` + `min_tokens` + `max_tokens`, a warmup call per size is discarded, and each
//! cell is the median of `reps` measured calls. Streams are read as SSE; TTFT is the time to
//! the first content delta and decode speed is (completion_tokens - 1) / (end - first token).

use crate::http;
use anyhow::{bail, Context, Result};
use serde_json::{json, Map, Value};
use std::collections::BTreeSet;
use std::io::{BufRead, BufReader};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug)]
pub struct SpeedConfig {
    pub url: String,
    pub model: String,
    pub api_key: Option<String>,
    pub sizes: Vec<usize>,
    pub gen: usize,
    pub reps: usize,
    pub warmup: usize,
    pub concurrency: Vec<usize>,
    /// Prompt size (tokens) used for the concurrency rows.
    pub conc_size: usize,
    /// Weight bytes read per decoded token, GB (for the bandwidth floor).
    pub gb_per_token: Option<f64>,
    pub bandwidth_gbs: Option<f64>,
    pub label: String,
    /// `Some(false)` sends `chat_template_kwargs.enable_thinking=false`.
    pub thinking: Option<bool>,
    pub timeout_s: u64,
    pub out_dir: Option<String>,
    /// Free-form host fingerprint merged into the JSON (`doctor` output, `key=value` pairs).
    pub fingerprint: Map<String, Value>,
}

#[derive(Clone, Debug)]
pub struct CallResult {
    pub ok: bool,
    pub error: Option<String>,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub start: Instant,
    pub first: Option<Instant>,
    pub end: Instant,
}

impl CallResult {
    fn failed(start: Instant, msg: String) -> Self {
        CallResult { ok: false, error: Some(msg), prompt_tokens: 0, completion_tokens: 0, start, first: None, end: Instant::now() }
    }
    pub fn ttft_s(&self) -> Option<f64> {
        self.first.map(|f| f.duration_since(self.start).as_secs_f64())
    }
    pub fn prefill_tok_s(&self) -> Option<f64> {
        let t = self.ttft_s()?;
        (t > 0.0 && self.prompt_tokens > 0).then(|| self.prompt_tokens as f64 / t)
    }
    pub fn decode_tok_s(&self) -> Option<f64> {
        let f = self.first?;
        let dt = self.end.duration_since(f).as_secs_f64();
        (dt > 0.0 && self.completion_tokens > 1).then(|| (self.completion_tokens - 1) as f64 / dt)
    }
}

// ---------------------------------------------------------------------------------------------
// Prompt construction

const WORDS: &[&str] = &[
    "amber", "harbor", "quiet", "lantern", "copper", "meadow", "signal", "orchid", "granite", "velvet", "river", "monsoon",
    "pebble", "saffron", "tundra", "ember", "falcon", "gable", "hollow", "ivory", "juniper", "kettle", "lagoon", "marble",
    "nectar", "obsidian", "paddle", "quartz", "ripple", "summit", "thistle", "umber", "violet", "willow", "yonder", "zephyr",
    "anchor", "beacon", "cobalt", "driftwood", "estuary", "fjord", "glacier", "heron", "inlet", "jetty", "kelp", "lichen",
    "mangrove", "nimbus", "oasis", "prairie", "quill", "reef", "sparrow", "timber", "uplands", "vineyard", "wharf", "yarrow",
];

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    pub fn from_clock() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static CTR: AtomicU64 = AtomicU64::new(0);
        let n = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1);
        let c = CTR.fetch_add(1, Ordering::Relaxed);
        let mut r = Rng(n ^ c.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03);
        for _ in 0..4 {
            r.next();
        }
        r
    }
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

const INSTRUCTION: &str = "Using the notes above, write a long, detailed essay. Do not stop early.";

/// A prompt of `sentences` random filler sentences behind a fresh nonce.
pub fn make_prompt(sentences: usize, rng: &mut Rng) -> String {
    let nonce = format!("{:016x}", rng.next());
    let mut s = format!("nonce {nonce} notes:\n");
    for i in 0..sentences {
        let w: Vec<&str> = (0..9).map(|_| WORDS[rng.below(WORDS.len())]).collect();
        s.push_str(&format!(
            "Note {}: {} {} {} {} {} {} {} {} {}.\n",
            i + 1, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8]
        ));
    }
    s.push('\n');
    s.push_str(INSTRUCTION);
    s
}

// ---------------------------------------------------------------------------------------------
// One streamed chat call

type Dropped = Arc<Mutex<BTreeSet<&'static str>>>;
const OPTIONAL_PARAMS: [&str; 3] = ["ignore_eos", "min_tokens", "stream_options"];

struct Client {
    agent: ureq::Agent,
    url: String,
    cfg: SpeedConfig,
    dropped: Dropped,
    /// The server's context window in tokens (`max_model_len` / `context_length` of `GET /v1/models`), when it says.
    ctx: Option<usize>,
}

/// Does an HTTP error body say the request was too long for the model (a property of the request, not of
/// optional parameters)?
fn is_context_error(body: &str) -> bool {
    let b = body.to_ascii_lowercase();
    ["context", "max_model_len", "maximum", "too long", "too many tokens", "exceed", "does not fit"]
        .iter()
        .any(|k| b.contains(k))
}

/// Does an HTTP error body complain about an unknown / unsupported parameter without being about length?
fn is_param_error(body: &str) -> bool {
    let b = body.to_ascii_lowercase();
    !is_context_error(&b) && ["unknown", "unrecognized", "unsupported", "extra inputs", "not permitted", "unexpected", "invalid"]
        .iter()
        .any(|k| b.contains(k))
}

impl Client {
    fn new(cfg: &SpeedConfig) -> Result<Self> {
        http::check_scheme(&cfg.url)?;
        Ok(Client {
            agent: http::agent(Some(Duration::from_secs(cfg.timeout_s))),
            url: format!("{}/chat/completions", http::api_base(&cfg.url)),
            cfg: cfg.clone(),
            dropped: Arc::new(Mutex::new(BTreeSet::new())),
            ctx: None,
        })
    }

    /// Ask the server for its context window (best effort; `None` when it does not say).
    fn fetch_context(&self) -> Option<usize> {
        let url = format!("{}/models", http::api_base(&self.cfg.url));
        let mut req = self.agent.get(&url);
        if let Some(k) = &self.cfg.api_key {
            req = req.header("Authorization", format!("Bearer {k}"));
        }
        let mut resp = req.call().ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let v: Value = serde_json::from_str(&resp.body_mut().read_to_string().ok()?).ok()?;
        let models = v["data"].as_array()?;
        let m = models.iter().find(|m| m["id"].as_str() == Some(self.cfg.model.as_str())).or_else(|| models.first())?;
        ["max_model_len", "context_length", "max_context_length", "context_len"]
            .iter()
            .find_map(|k| m[*k].as_u64())
            .map(|n| n as usize)
    }

    fn body(&self, prompt: &str, max_tokens: usize, force: bool) -> Value {
        let mut b = json!({
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": true,
            "stream_options": {"include_usage": true},
        });
        if force && max_tokens > 1 {
            b["ignore_eos"] = json!(true);
            b["min_tokens"] = json!(max_tokens);
        }
        if self.cfg.thinking == Some(false) {
            b["chat_template_kwargs"] = json!({"enable_thinking": false});
        }
        if let Some(o) = b.as_object_mut() {
            for p in self.dropped.lock().unwrap().iter() {
                o.remove(*p);
            }
        }
        b
    }

    fn chat(&self, prompt: &str, max_tokens: usize, force: bool) -> CallResult {
        let r = self.chat_once(prompt, max_tokens, force);
        if let Some(e) = &r.error {
            if e.starts_with("HTTP 400") || e.starts_with("HTTP 422") {
                // A rejected optional parameter: drop what the error names (or all optional ones when it only
                // says "unknown parameter") and retry once. A length error (prompt + max_tokens over the
                // context) is not a parameter problem: report it as it is and drop nothing.
                let named: Vec<&'static str> = OPTIONAL_PARAMS.iter().copied().filter(|p| e.contains(p)).collect();
                let drop = if !named.is_empty() {
                    named
                } else if is_param_error(e) {
                    OPTIONAL_PARAMS.to_vec()
                } else {
                    Vec::new()
                };
                let added: Vec<&'static str> = {
                    let mut d = self.dropped.lock().unwrap();
                    drop.into_iter().filter(|p| d.insert(p)).collect()
                };
                if !added.is_empty() {
                    let retry = self.chat_once(prompt, max_tokens, force);
                    if !retry.ok {
                        // the retry failed as well: the parameters were not the problem
                        let mut d = self.dropped.lock().unwrap();
                        added.iter().for_each(|p| {
                            d.remove(p);
                        });
                    }
                    return retry;
                }
            }
        }
        r
    }

    fn chat_once(&self, prompt: &str, max_tokens: usize, force: bool) -> CallResult {
        let start = Instant::now();
        let body = self.body(prompt, max_tokens, force).to_string();
        let mut req = self.agent.post(&self.url).header("Content-Type", "application/json");
        if let Some(k) = &self.cfg.api_key {
            req = req.header("Authorization", format!("Bearer {k}"));
        }
        let resp = match req.send(body.as_str()) {
            Ok(r) => r,
            Err(e) => return CallResult::failed(start, format!("request: {e}")),
        };
        let status = resp.status().as_u16();
        let mut reader = resp.into_body().into_reader();
        if !(200..300).contains(&status) {
            let mut s = String::new();
            let _ = std::io::Read::read_to_string(&mut std::io::Read::take(&mut reader, 800), &mut s);
            return CallResult::failed(start, format!("HTTP {status}: {s}"));
        }
        let mut first = None;
        let mut chunks = 0u64;
        let (mut pt, mut ct) = (0u64, None);
        for line in BufReader::new(reader).lines() {
            let line = match line {
                Ok(l) => l,
                Err(e) => return CallResult::failed(start, format!("stream: {e}")),
            };
            let Some(payload) = line.strip_prefix("data:") else { continue };
            let payload = payload.trim();
            if payload == "[DONE]" {
                break;
            }
            let Ok(ev) = serde_json::from_str::<Value>(payload) else { continue };
            if let Some(err) = ev.get("error") {
                return CallResult::failed(start, format!("stream error: {err}"));
            }
            if let Some(u) = ev.get("usage").filter(|u| u.is_object()) {
                pt = u["prompt_tokens"].as_u64().unwrap_or(pt);
                ct = u["completion_tokens"].as_u64().or(ct);
            }
            if let Some(ch) = ev["choices"].as_array() {
                for c in ch {
                    let d = &c["delta"];
                    let has = ["content", "reasoning_content", "reasoning"]
                        .iter()
                        .any(|k| d[*k].as_str().is_some_and(|s| !s.is_empty()));
                    if has {
                        chunks += 1;
                        first.get_or_insert_with(Instant::now);
                    }
                }
            }
        }
        let end = Instant::now();
        if first.is_none() {
            return CallResult::failed(start, "no tokens received".into());
        }
        CallResult { ok: true, error: None, prompt_tokens: pt, completion_tokens: ct.unwrap_or(chunks), start, first, end }
    }
}

// ---------------------------------------------------------------------------------------------
// Statistics

pub fn median(v: &[f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    Some(if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 })
}

fn med<I: Iterator<Item = f64>>(it: I) -> Option<f64> {
    median(&it.collect::<Vec<_>>())
}

fn r(x: Option<f64>, d: i32) -> Value {
    match x {
        Some(v) if v.is_finite() => {
            let p = 10f64.powi(d);
            json!((v * p).round() / p)
        }
        _ => Value::Null,
    }
}

fn floor_pct(decode: Option<f64>, cfg: &SpeedConfig) -> (Option<f64>, Option<f64>) {
    match (cfg.gb_per_token, cfg.bandwidth_gbs) {
        (Some(g), Some(b)) if g > 0.0 => {
            let floor = b / g;
            (Some(floor), decode.map(|d| d / floor * 100.0))
        }
        _ => (None, None),
    }
}

// ---------------------------------------------------------------------------------------------
// The sweep

fn calibrate(c: &Client) -> Result<(f64, f64)> {
    let mut rng = Rng::from_clock();
    let a = c.chat(&make_prompt(0, &mut rng), 1, false);
    if !a.ok {
        bail!("calibration call failed: {}", a.error.unwrap_or_else(|| "unknown error".into()));
    }
    // A few hundred filler sentences measure the per-sentence cost best; a small context window gets a shorter
    // probe (the probe must fit, so each step down is tried only when it can: at most ~20 tokens a sentence).
    let mut last_err = None;
    for n in [200usize, 100, 50, 20, 10] {
        if c.ctx.is_some_and(|ctx| n * 20 > ctx / 2) {
            continue;
        }
        let b = c.chat(&make_prompt(n, &mut rng), 1, false);
        if !b.ok {
            last_err = b.error;
            continue;
        }
        if a.prompt_tokens == 0 || b.prompt_tokens <= a.prompt_tokens {
            bail!("server did not report usable prompt_tokens (got {} and {}); cannot size prompts", a.prompt_tokens, b.prompt_tokens);
        }
        let per = (b.prompt_tokens - a.prompt_tokens) as f64 / n as f64;
        return Ok((a.prompt_tokens as f64, per));
    }
    bail!("calibration call failed: {}", last_err.unwrap_or_else(|| "no probe fits the server context".into()))
}

pub fn run_speed(cfg: &SpeedConfig) -> Result<Value> {
    let mut c = Client::new(cfg)?;
    eprintln!("target {}  model {}", c.url, cfg.model);
    c.ctx = c.fetch_context();
    match c.ctx {
        Some(n) => eprintln!("server context window: {n} tokens"),
        None => eprintln!("server did not report a context window; requests are not pre-checked"),
    }
    let (base, per) = calibrate(&c).context("calibrating prompt size")?;
    eprintln!("calibration: {base:.0} base tokens, {per:.2} tokens per filler sentence");
    let sentences_for = |size: usize| ((size as f64 - base).max(0.0) / per).round() as usize;
    // Estimated prompt tokens of a request sized `size`, plus `gen` generated, against the context window.
    let too_long = |size: usize, gen: usize| -> Option<String> {
        let ctx = c.ctx?;
        let est = base + sentences_for(size) as f64 * per;
        let need = (est * 1.02).ceil() as usize + gen + 16;
        (need > ctx).then(|| format!("~{est:.0} prompt + {gen} generated tokens (+2% and 16 slack) need {need}, the server context is {ctx}"))
    };

    let mut rows = vec![];
    let mut warnings: Vec<String> = vec![];
    for &size in &cfg.sizes {
        if let Some(why) = too_long(size, cfg.gen) {
            eprintln!("  size {size:>6} SKIPPED before sending: {why}");
            warnings.push(format!("size {size}: skipped, {why}"));
            rows.push(json!({
                "target_prompt_tokens": size, "prompt_tokens": null, "gen_tokens": null, "ttft_s": null,
                "prefill_tok_s": null, "decode_tok_s": null, "bandwidth_floor_tok_s": null, "pct_of_floor": null,
                "reps_ok": 0, "errors": 0, "skipped": why,
            }));
            continue;
        }
        let n = sentences_for(size);
        for _ in 0..cfg.warmup {
            let mut rng = Rng::from_clock();
            let w = c.chat(&make_prompt(n, &mut rng), cfg.gen.min(8), true);
            if !w.ok {
                warnings.push(format!("size {size}: warmup failed: {}", w.error.clone().unwrap_or_default()));
            }
        }
        let mut res = vec![];
        for _ in 0..cfg.reps {
            let mut rng = Rng::from_clock();
            let p = make_prompt(n, &mut rng);
            let x = c.chat(&p, cfg.gen, true);
            eprintln!(
                "  size {size:>6} rep: {}",
                match &x {
                    x if x.ok => format!(
                        "prompt={} gen={} ttft={:.2}s prefill={:.0} tok/s decode={:.1} tok/s",
                        x.prompt_tokens, x.completion_tokens, x.ttft_s().unwrap_or(0.0),
                        x.prefill_tok_s().unwrap_or(0.0), x.decode_tok_s().unwrap_or(0.0)
                    ),
                    x => format!("FAILED {}", x.error.clone().unwrap_or_default()),
                }
            );
            res.push(x);
        }
        let ok: Vec<&CallResult> = res.iter().filter(|x| x.ok).collect();
        let errors = res.len() - ok.len();
        if let Some(e) = res.iter().find_map(|x| x.error.clone()) {
            warnings.push(format!("size {size}: {errors}/{} calls failed, e.g. {e}", res.len()));
        }
        let dec = med(ok.iter().filter_map(|x| x.decode_tok_s()));
        let (floor, pct) = floor_pct(dec, cfg);
        let short = ok.iter().filter(|x| (x.completion_tokens as usize) < cfg.gen).count();
        if short > 0 {
            warnings.push(format!("size {size}: {short} calls returned fewer than {} tokens (server ignored min_tokens/ignore_eos?)", cfg.gen));
        }
        rows.push(json!({
            "target_prompt_tokens": size,
            "prompt_tokens": r(med(ok.iter().map(|x| x.prompt_tokens as f64)), 0),
            "gen_tokens": r(med(ok.iter().map(|x| x.completion_tokens as f64)), 0),
            "ttft_s": r(med(ok.iter().filter_map(|x| x.ttft_s())), 4),
            "prefill_tok_s": r(med(ok.iter().filter_map(|x| x.prefill_tok_s())), 1),
            "decode_tok_s": r(dec, 2),
            "bandwidth_floor_tok_s": r(floor, 2),
            "pct_of_floor": r(pct, 1),
            "reps_ok": ok.len(),
            "errors": errors,
        }));
    }

    // Concurrency rows.
    let mut conc_rows = vec![];
    for &n_streams in cfg.concurrency.iter().filter(|&&n| n > 1) {
        if let Some(why) = too_long(cfg.conc_size, cfg.gen) {
            eprintln!("concurrency {n_streams} SKIPPED before sending: {why}");
            warnings.push(format!("concurrency {n_streams}: skipped, {why}"));
            continue;
        }
        let n = sentences_for(cfg.conc_size);
        eprintln!("concurrency {n_streams} at ~{} prompt tokens", cfg.conc_size);
        let mut aggs = vec![];
        let mut per_stream = vec![];
        let mut ttfts = vec![];
        let mut fails = 0usize;
        for rep in 0..cfg.reps + 1 {
            let prompts: Vec<String> = (0..n_streams).map(|_| make_prompt(n, &mut Rng::from_clock())).collect();
            let results: Vec<CallResult> = std::thread::scope(|s| {
                let hs: Vec<_> = prompts.iter().map(|p| s.spawn(|| c.chat(p, cfg.gen, true))).collect();
                hs.into_iter().map(|h| h.join().expect("bench thread panicked")).collect()
            });
            if rep == 0 {
                continue; // warmup round
            }
            let ok: Vec<&CallResult> = results.iter().filter(|x| x.ok).collect();
            fails += results.len() - ok.len();
            if ok.is_empty() {
                continue;
            }
            let t0 = ok.iter().filter_map(|x| x.first).min();
            let t1 = ok.iter().map(|x| x.end).max();
            if let (Some(t0), Some(t1)) = (t0, t1) {
                let dt = t1.duration_since(t0).as_secs_f64();
                let toks: u64 = ok.iter().map(|x| x.completion_tokens.saturating_sub(1)).sum();
                if dt > 0.0 {
                    aggs.push(toks as f64 / dt);
                }
            }
            per_stream.extend(ok.iter().filter_map(|x| x.decode_tok_s()));
            ttfts.extend(ok.iter().filter_map(|x| x.ttft_s()));
        }
        let agg = median(&aggs);
        let (floor, pct) = floor_pct(agg, cfg);
        if fails > 0 {
            warnings.push(format!("concurrency {n_streams}: {fails} calls failed"));
        }
        conc_rows.push(json!({
            "streams": n_streams,
            "target_prompt_tokens": cfg.conc_size,
            "ttft_s": r(median(&ttfts), 4),
            "per_stream_decode_tok_s": r(median(&per_stream), 2),
            "aggregate_decode_tok_s": r(agg, 2),
            "pct_of_floor": r(pct, 1),
            "bandwidth_floor_tok_s": r(floor, 2),
            "errors": fails,
        }));
    }

    let dropped: Vec<String> = c.dropped.lock().unwrap().iter().map(|s| s.to_string()).collect();
    if !dropped.is_empty() {
        warnings.push(format!("server rejected optional parameters (the retry without them worked), dropped: {dropped:?}"));
    }
    let ts = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    Ok(json!({
        "kind": "intarsia-bench-speed",
        "schema_version": 1,
        "label": cfg.label,
        "timestamp_unix": ts,
        "url": cfg.url,
        "model": cfg.model,
        "config": {
            "sizes": cfg.sizes, "gen": cfg.gen, "reps": cfg.reps, "warmup": cfg.warmup,
            "concurrency": cfg.concurrency, "conc_size": cfg.conc_size,
            "gb_per_token": cfg.gb_per_token, "bandwidth_gbs": cfg.bandwidth_gbs,
        },
        "calibration": {"base_prompt_tokens": base, "tokens_per_filler_sentence": (per * 100.0).round() / 100.0},
        "fingerprint": Value::Object(cfg.fingerprint.clone()),
        "rows": rows,
        "concurrency": conc_rows,
        "warnings": warnings,
    }))
}

// ---------------------------------------------------------------------------------------------
// Rendering

fn cell(v: &Value, w: usize, dec: usize) -> String {
    match v.as_f64() {
        Some(x) => format!("{x:>w$.dec$}"),
        None => format!("{:>w$}", "-"),
    }
}

pub fn render_table(res: &Value) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "{}  model={}  gen={}  reps={}\n",
        res["label"].as_str().unwrap_or(""), res["model"].as_str().unwrap_or(""),
        res["config"]["gen"], res["config"]["reps"]
    ));
    let floor = res["rows"].as_array().and_then(|r| r.iter().find_map(|x| x["bandwidth_floor_tok_s"].as_f64()));
    s.push_str(&format!(
        "{:>8} {:>8} {:>9} {:>12} {:>11} {:>8}{}\n",
        "size", "prompt", "TTFT s", "prefill t/s", "decode t/s", "gen",
        if floor.is_some() { "  % of floor" } else { "" }
    ));
    for row in res["rows"].as_array().into_iter().flatten() {
        if row["skipped"].is_string() {
            s.push_str(&format!("{:>8} skipped: server context too small for this size\n", row["target_prompt_tokens"].to_string()));
            continue;
        }
        s.push_str(&format!(
            "{:>8} {} {} {} {} {}{}\n",
            row["target_prompt_tokens"].to_string(), cell(&row["prompt_tokens"], 8, 0), cell(&row["ttft_s"], 9, 3),
            cell(&row["prefill_tok_s"], 12, 0), cell(&row["decode_tok_s"], 11, 1), cell(&row["gen_tokens"], 8, 0),
            if floor.is_some() { format!("  {}", cell(&row["pct_of_floor"], 9, 1)) } else { String::new() }
        ));
    }
    if let Some(f) = floor {
        s.push_str(&format!("bandwidth floor: {f:.1} tok/s\n"));
    }
    if res["concurrency"].as_array().is_some_and(|a| !a.is_empty()) {
        s.push_str(&format!(
            "\n{:>8} {:>9} {:>14} {:>14}{}\n",
            "streams", "TTFT s", "per-stream t/s", "aggregate t/s", if floor.is_some() { "  % of floor" } else { "" }
        ));
        for row in res["concurrency"].as_array().into_iter().flatten() {
            s.push_str(&format!(
                "{:>8} {} {} {}{}\n",
                row["streams"].to_string(), cell(&row["ttft_s"], 9, 3), cell(&row["per_stream_decode_tok_s"], 14, 1),
                cell(&row["aggregate_decode_tok_s"], 14, 1),
                if floor.is_some() { format!("  {}", cell(&row["pct_of_floor"], 9, 1)) } else { String::new() }
            ));
        }
    }
    for w in res["warnings"].as_array().into_iter().flatten() {
        s.push_str(&format!("warning: {}\n", w.as_str().unwrap_or("")));
    }
    s
}

pub fn sanitize_label(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || "-_.".contains(c) { c } else { '-' }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    #[test]
    fn median_works() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), Some(2.5));
        assert_eq!(median(&[]), None);
    }

    #[test]
    fn prompts_differ_by_nonce() {
        let a = make_prompt(3, &mut Rng::from_clock());
        let b = make_prompt(3, &mut Rng::from_clock());
        assert_ne!(a.lines().next(), b.lines().next());
        assert!(a.contains("Note 3:"));
    }

    /// Minimal fake OpenAI server: prompt_tokens grows with the prompt length, 10 content chunks.
    fn fake_server() -> String {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        std::thread::spawn(move || {
            for stream in l.incoming() {
                let Ok(mut s) = stream else { break };
                std::thread::spawn(move || {
                    let mut buf = vec![0u8; 1 << 20];
                    let mut got = 0;
                    loop {
                        let n = s.read(&mut buf[got..]).unwrap_or(0);
                        if n == 0 {
                            return;
                        }
                        got += n;
                        let txt = String::from_utf8_lossy(&buf[..got]).to_string();
                        if let Some(i) = txt.find("\r\n\r\n") {
                            let cl = txt[..i].to_lowercase().split("content-length:").nth(1)
                                .and_then(|x| x.lines().next()).and_then(|x| x.trim().parse::<usize>().ok()).unwrap_or(0);
                            if got >= i + 4 + cl {
                                break;
                            }
                        }
                    }
                    let txt = String::from_utf8_lossy(&buf[..got]).to_string();
                    let pt = txt.matches("Note ").count() * 12 + 20;
                    let mut out = String::from("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n");
                    for i in 0..10 {
                        out.push_str(&format!("data: {{\"choices\":[{{\"delta\":{{\"content\":\"t{i}\"}}}}]}}\n\n"));
                    }
                    out.push_str(&format!("data: {{\"choices\":[],\"usage\":{{\"prompt_tokens\":{pt},\"completion_tokens\":10}}}}\n\ndata: [DONE]\n\n"));
                    let _ = s.write_all(out.as_bytes());
                });
            }
        });
        format!("http://{addr}")
    }

    #[test]
    fn sweep_against_fake_server() {
        let cfg = SpeedConfig {
            url: fake_server(), model: "fake".into(), api_key: None, sizes: vec![128, 512], gen: 10, reps: 2, warmup: 1,
            concurrency: vec![1, 2], conc_size: 128, gb_per_token: Some(1.0), bandwidth_gbs: Some(100.0),
            label: "t".into(), thinking: None, timeout_s: 20, out_dir: None, fingerprint: Map::new(),
        };
        let res = run_speed(&cfg).unwrap();
        let rows = res["rows"].as_array().unwrap();
        assert_eq!(rows.len(), 2);
        assert!(rows[1]["prompt_tokens"].as_f64().unwrap() > rows[0]["prompt_tokens"].as_f64().unwrap());
        assert_eq!(rows[0]["gen_tokens"].as_f64(), Some(10.0));
        assert_eq!(rows[0]["bandwidth_floor_tok_s"].as_f64(), Some(100.0));
        assert_eq!(res["concurrency"].as_array().unwrap().len(), 1);
        let t = render_table(&res);
        assert!(t.contains("prefill"));
    }

    /// Fake server with a context limit: `GET /v1/models` reports `max_model_len` (when `advertise`), a chat call
    /// whose prompt is over `limit` tokens gets HTTP 400 with a length message. Returns the URL and the prompt
    /// sizes of the chat calls it received.
    fn ctx_server(limit: usize, advertise: bool) -> (String, Arc<Mutex<Vec<usize>>>) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen2 = seen.clone();
        std::thread::spawn(move || {
            for stream in l.incoming() {
                let Ok(mut s) = stream else { break };
                let seen = seen2.clone();
                std::thread::spawn(move || {
                    let mut buf = vec![0u8; 1 << 20];
                    let mut got = 0;
                    loop {
                        let n = s.read(&mut buf[got..]).unwrap_or(0);
                        if n == 0 {
                            return;
                        }
                        got += n;
                        let txt = String::from_utf8_lossy(&buf[..got]).to_string();
                        if let Some(i) = txt.find("\r\n\r\n") {
                            let cl = txt[..i].to_lowercase().split("content-length:").nth(1)
                                .and_then(|x| x.lines().next()).and_then(|x| x.trim().parse::<usize>().ok()).unwrap_or(0);
                            if got >= i + 4 + cl {
                                break;
                            }
                        }
                    }
                    let txt = String::from_utf8_lossy(&buf[..got]).to_string();
                    if txt.starts_with("GET /v1/models") {
                        let body = if advertise {
                            format!("{{\"data\":[{{\"id\":\"fake\",\"max_model_len\":{limit}}}]}}")
                        } else {
                            "{\"data\":[{\"id\":\"fake\"}]}".to_string()
                        };
                        let _ = s.write_all(
                            format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len())
                                .as_bytes(),
                        );
                        return;
                    }
                    let pt = txt.matches("Note ").count() * 12 + 20;
                    seen.lock().unwrap().push(pt);
                    if pt > limit {
                        let body = format!("{{\"error\":{{\"message\":\"This model's maximum context length is {limit} tokens. However, you requested {pt} tokens.\"}}}}");
                        let _ = s.write_all(
                            format!("HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len())
                                .as_bytes(),
                        );
                        return;
                    }
                    let mut out = String::from("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n");
                    for i in 0..10 {
                        out.push_str(&format!("data: {{\"choices\":[{{\"delta\":{{\"content\":\"t{i}\"}}}}]}}\n\n"));
                    }
                    out.push_str(&format!("data: {{\"choices\":[],\"usage\":{{\"prompt_tokens\":{pt},\"completion_tokens\":10}}}}\n\ndata: [DONE]\n\n"));
                    let _ = s.write_all(out.as_bytes());
                });
            }
        });
        (format!("http://{addr}"), seen)
    }

    fn cfg_for(url: String, sizes: Vec<usize>) -> SpeedConfig {
        SpeedConfig {
            url, model: "fake".into(), api_key: None, sizes, gen: 10, reps: 1, warmup: 0, concurrency: vec![1],
            conc_size: 128, gb_per_token: None, bandwidth_gbs: None, label: "t".into(), thinking: None, timeout_s: 20,
            out_dir: None, fingerprint: Map::new(),
        }
    }

    #[test]
    fn oversize_requests_are_skipped_before_sending() {
        let (url, seen) = ctx_server(600, true);
        let res = run_speed(&cfg_for(url, vec![128, 5000])).unwrap();
        let rows = res["rows"].as_array().unwrap();
        assert!(rows[0]["skipped"].is_null() && rows[0]["reps_ok"] == 1, "{:?}", rows[0]);
        assert!(rows[1]["skipped"].is_string(), "{:?}", rows[1]);
        // calibration sends two calls; the 5000-token size was never sent
        assert!(seen.lock().unwrap().iter().all(|&pt| pt <= 600), "{:?}", seen.lock().unwrap());
        let w = res["warnings"].to_string();
        assert!(w.contains("skipped") && !w.contains("rejected optional parameters"), "{w}");
        assert!(render_table(&res).contains("skipped"));
    }

    #[test]
    fn a_context_400_is_not_reported_as_a_rejected_parameter() {
        // the server does not advertise its window, so the oversize request is sent and answered with HTTP 400
        let (url, seen) = ctx_server(600, false);
        let res = run_speed(&cfg_for(url, vec![128, 5000])).unwrap();
        let rows = res["rows"].as_array().unwrap();
        assert_eq!(rows[1]["errors"], 1);
        let w = res["warnings"].to_string();
        assert!(w.contains("maximum context length"), "{w}");
        assert!(!w.contains("rejected optional parameters"), "{w}");
        // one send for the 5000-token size, no retry without parameters
        assert_eq!(seen.lock().unwrap().iter().filter(|&&pt| (4900..5100).contains(&pt)).count(), 1);
    }
}
