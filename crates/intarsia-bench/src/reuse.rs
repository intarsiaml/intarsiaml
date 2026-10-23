//! Prefix-reuse workloads against an OpenAI-compatible server (`intarsia bench reuse`).
//!
//! `bench speed` defeats every prefix cache on purpose (a fresh nonce per prompt); this does the opposite. It replays
//! the request sequences that real clients send, where most of each prompt was already sent before, and measures per
//! request the time to first token and how many prompt tokens the server says it served from cache
//! (`usage.prompt_tokens_details.cached_tokens`), so prefill tokens computed = prompt - cached.
//!
//! Workloads (all deterministic, built from a text corpus):
//! - `chat`: two conversations interleaved turn by turn (two users, or two tabs), thinking on. Each turn sends the
//!   whole history; the template drops earlier turns' thinking. A server holding one sequence loses the other
//!   conversation's state on every switch.
//! - `agent`: a fixed system prompt and tool schema, then tasks of two requests each (task, then a follow-up with a
//!   tool result), every task a fresh conversation with the same prefix.
//! - `rag`: the same documents in the user message, followed by a different question each time.
//!
//! Assistant turns are recorded on the first run (`--record DIR`, the live answers) and replayed verbatim afterwards
//! (`--replay DIR`), so a run with the prefix store and one without it send byte-identical requests.

use crate::http;
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct ReuseConfig {
    pub url: String,
    pub model: String,
    pub api_key: Option<String>,
    pub workloads: Vec<String>,
    pub corpus: PathBuf,
    pub max_tokens: usize,
    pub turns: usize,
    pub tasks: usize,
    pub questions: usize,
    pub doc_chars: usize,
    pub record: Option<PathBuf>,
    pub replay: Option<PathBuf>,
    pub timeout_s: u64,
}

/// One request of a workload: the messages (and tools) sent, and the conversation it belongs to.
#[derive(Clone, Debug)]
struct Step {
    conv: String,
    messages: Vec<Value>,
    tools: Option<Value>,
    thinking: bool,
}

#[derive(Clone, Debug, Default)]
struct Res {
    ttft_ms: f64,
    e2e_ms: f64,
    prompt_tokens: u64,
    cached_tokens: u64,
    text: String,
}

fn corpus_slice(corpus: &str, start: usize, chars: usize) -> String {
    let start = start % corpus.len().max(1);
    let mut s: String = corpus.chars().skip(start).take(chars).collect();
    if s.chars().count() < chars {
        s.extend(corpus.chars().take(chars - s.chars().count()));
    }
    s
}

const SYSTEM_AGENT: &str = "You are a careful operations agent for a small hosting company. You can call tools to inspect \
servers, read logs, restart services and open tickets. Always explain your plan in one sentence before acting, prefer \
read-only tools first, never restart a production service without a ticket, and summarise what you found at the end. \
Answer in plain English, no markdown tables.";

fn agent_tools() -> Value {
    let tool = |name: &str, desc: &str, props: Value| {
        json!({"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": props.as_object().map(|o| o.keys().cloned().collect::<Vec<_>>()).unwrap_or_default()}}})
    };
    json!([
        tool("list_servers", "List the servers of a region with their role, load and status.", json!({"region": {"type": "string", "description": "Region code such as eu-1 or us-2."}})),
        tool("read_log", "Read the last lines of a service log on a server.", json!({"server": {"type": "string"}, "service": {"type": "string"}, "lines": {"type": "integer", "description": "How many lines, at most 500."}})),
        tool("disk_usage", "Disk usage per mount point of a server.", json!({"server": {"type": "string"}})),
        tool("restart_service", "Restart a service. Needs an open ticket id.", json!({"server": {"type": "string"}, "service": {"type": "string"}, "ticket": {"type": "string"}})),
        tool("open_ticket", "Open a ticket for the on-call team.", json!({"title": {"type": "string"}, "severity": {"type": "string", "enum": ["low", "medium", "high"]}, "body": {"type": "string"}})),
        tool("dns_lookup", "Resolve a host name and show its records.", json!({"name": {"type": "string"}, "type": {"type": "string", "enum": ["A", "AAAA", "CNAME", "MX", "TXT"]}})),
        tool("http_check", "Fetch a URL from the monitoring host and report status, latency and certificate expiry.", json!({"url": {"type": "string"}})),
        tool("query_metrics", "Query a Prometheus metric over a time range.", json!({"query": {"type": "string"}, "minutes": {"type": "integer"}})),
    ])
}

const AGENT_TASKS: &[&str] = &[
    "The status page says eu-1 web is slow. Find out why.",
    "A customer reports 502 errors on shop.example.net since this morning.",
    "Disk alerts fired on db-3. Check what is filling it.",
    "Certificates: which of our public endpoints expire within 30 days?",
    "The mail queue on mx-2 is growing. Investigate.",
    "Latency on the us-2 API doubled after the last deploy. What changed?",
];

const RAG_QUESTIONS: &[&str] = &[
    "Summarise the main argument of the second document in three sentences.",
    "Which document mentions a date, and what happened then?",
    "List every proper noun that appears in more than one document.",
    "What would a sceptical reader object to in the first document?",
    "Write one question each document leaves unanswered.",
    "Which two documents disagree most, and on what?",
];

fn user_text(corpus: &str, seed: usize, chars: usize, question: &str) -> String {
    format!("{}\n\n{question}", corpus_slice(corpus, seed * 7919, chars))
}

struct Client {
    agent: ureq::Agent,
    url: String,
    cfg: ReuseConfig,
}

impl Client {
    fn send(&self, s: &Step) -> Result<Res> {
        let mut b = json!({
            "model": self.cfg.model,
            "messages": s.messages,
            "max_tokens": self.cfg.max_tokens,
            "temperature": 0.0,
            "stream": true,
            "stream_options": {"include_usage": true},
            "chat_template_kwargs": {"enable_thinking": s.thinking},
        });
        if let Some(t) = &s.tools {
            b["tools"] = t.clone();
        }
        let start = Instant::now();
        let mut req = self.agent.post(&self.url).header("Content-Type", "application/json");
        if let Some(k) = &self.cfg.api_key {
            req = req.header("Authorization", format!("Bearer {k}"));
        }
        let resp = req.send(b.to_string().as_str()).map_err(|e| anyhow::anyhow!("request: {e}"))?;
        let status = resp.status().as_u16();
        let mut reader = resp.into_body().into_reader();
        if !(200..300).contains(&status) {
            let mut e = String::new();
            let _ = std::io::Read::read_to_string(&mut std::io::Read::take(&mut reader, 800), &mut e);
            bail!("HTTP {status}: {e}");
        }
        let mut r = Res::default();
        let mut first = None;
        let mut reasoning = String::new();
        for line in BufReader::new(reader).lines() {
            let line = line?;
            let Some(p) = line.strip_prefix("data:") else { continue };
            let p = p.trim();
            if p == "[DONE]" {
                break;
            }
            let Ok(ev) = serde_json::from_str::<Value>(p) else { continue };
            if let Some(err) = ev.get("error") {
                bail!("stream error: {err}");
            }
            if let Some(u) = ev.get("usage").filter(|u| u.is_object()) {
                r.prompt_tokens = u["prompt_tokens"].as_u64().unwrap_or(0);
                r.cached_tokens = u["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0);
            }
            for c in ev["choices"].as_array().into_iter().flatten() {
                let d = &c["delta"];
                for (k, out) in [("content", &mut r.text), ("reasoning_content", &mut reasoning)] {
                    if let Some(t) = d[k].as_str().filter(|t| !t.is_empty()) {
                        first.get_or_insert_with(Instant::now);
                        out.push_str(t);
                    }
                }
            }
        }
        let end = Instant::now();
        r.ttft_ms = first.map(|f| (f - start).as_secs_f64() * 1e3).unwrap_or(f64::NAN);
        r.e2e_ms = (end - start).as_secs_f64() * 1e3;
        if r.text.is_empty() {
            // a reply cut inside its thinking: keep the thinking as the answer (the template strips nothing then)
            r.text = reasoning;
        }
        Ok(r)
    }
}

/// Build (and run) one workload. With `replay`, the steps are read and sent as recorded; otherwise they are built
/// from the corpus with the live answers as history.
fn run_workload(c: &Client, name: &str, corpus: &str) -> Result<(Vec<Step>, Vec<Res>)> {
    let cfg = &c.cfg;
    if let Some(dir) = &cfg.replay {
        let steps = read_steps(&dir.join(format!("{name}.jsonl")))?;
        let mut res = Vec::new();
        for s in &steps {
            res.push(c.send(s)?);
        }
        return Ok((steps, res));
    }
    let mut steps = Vec::new();
    let mut res = Vec::new();
    match name {
        "chat" => {
            let mut hist: Vec<Vec<Value>> = vec![
                vec![json!({"role": "system", "content": "You are a thoughtful reading companion. Keep answers short."})],
                vec![json!({"role": "system", "content": "You are a terse literary critic. Answer in at most five sentences."})],
            ];
            for t in 0..cfg.turns {
                for (ci, h) in hist.iter_mut().enumerate() {
                    let q = ["What is this passage about?", "What does it leave out?", "Rewrite its first sentence plainly.", "Which claim is weakest?", "Name one follow-up question.", "Summarise our discussion so far."][t % 6];
                    h.push(json!({"role": "user", "content": user_text(corpus, 1 + ci * 31 + t, cfg.doc_chars / 4, q)}));
                    let s = Step { conv: format!("chat-{ci}"), messages: h.clone(), tools: None, thinking: true };
                    let r = c.send(&s)?;
                    h.push(json!({"role": "assistant", "content": r.text.clone()}));
                    steps.push(s);
                    res.push(r);
                }
            }
        }
        "agent" => {
            let tools = agent_tools();
            for (ti, task) in AGENT_TASKS.iter().take(cfg.tasks).enumerate() {
                let mut h = vec![json!({"role": "system", "content": SYSTEM_AGENT}), json!({"role": "user", "content": task})];
                let s = Step { conv: format!("agent-{ti}"), messages: h.clone(), tools: Some(tools.clone()), thinking: false };
                let r = c.send(&s)?;
                h.push(json!({"role": "assistant", "content": r.text.clone()}));
                steps.push(s);
                res.push(r);
                h.push(json!({"role": "user", "content": format!("Tool output:\n{}\nWhat next?", corpus_slice(corpus, 50_000 + ti * 1777, 600))}));
                let s = Step { conv: format!("agent-{ti}"), messages: h.clone(), tools: Some(tools.clone()), thinking: false };
                res.push(c.send(&s)?);
                steps.push(s);
            }
        }
        "rag" => {
            let docs: Vec<String> = (0..3).map(|i| corpus_slice(corpus, 20_000 + i * 15_000, cfg.doc_chars)).collect();
            let ctx = docs.iter().enumerate().map(|(i, d)| format!("<document {}>\n{d}\n</document>", i + 1)).collect::<Vec<_>>().join("\n");
            for (qi, q) in RAG_QUESTIONS.iter().take(cfg.questions).enumerate() {
                let s = Step {
                    conv: format!("rag-{qi}"),
                    messages: vec![
                        json!({"role": "system", "content": "Answer from the documents only. Say so when they do not say."}),
                        json!({"role": "user", "content": format!("{ctx}\n\nQuestion: {q}")}),
                    ],
                    tools: None,
                    thinking: false,
                };
                res.push(c.send(&s)?);
                steps.push(s);
            }
        }
        other => bail!("unknown workload '{other}' (chat, agent, rag)"),
    }
    if let Some(dir) = &cfg.record {
        std::fs::create_dir_all(dir)?;
        write_steps(&dir.join(format!("{name}.jsonl")), &steps)?;
    }
    Ok((steps, res))
}

fn write_steps(p: &Path, steps: &[Step]) -> Result<()> {
    let mut s = String::new();
    for st in steps {
        s.push_str(&json!({"conv": st.conv, "messages": st.messages, "tools": st.tools, "thinking": st.thinking}).to_string());
        s.push('\n');
    }
    std::fs::write(p, s).with_context(|| format!("writing {}", p.display()))
}

fn read_steps(p: &Path) -> Result<Vec<Step>> {
    let text = std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: Value = serde_json::from_str(l)?;
            Ok(Step {
                conv: v["conv"].as_str().unwrap_or("").to_string(),
                messages: v["messages"].as_array().cloned().unwrap_or_default(),
                tools: v.get("tools").filter(|t| !t.is_null()).cloned(),
                thinking: v["thinking"].as_bool().unwrap_or(false),
            })
        })
        .collect()
}

fn stats(v: &[f64]) -> Value {
    let mut s: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if s.is_empty() {
        return Value::Null;
    }
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| s[((s.len() - 1) as f64 * p).round() as usize];
    let mean = s.iter().sum::<f64>() / s.len() as f64;
    json!({"mean": (mean * 10.0).round() / 10.0, "median": (q(0.5) * 10.0).round() / 10.0, "p90": (q(0.9) * 10.0).round() / 10.0, "n": s.len()})
}

pub fn run_reuse(cfg: &ReuseConfig) -> Result<Value> {
    http::check_scheme(&cfg.url)?;
    let corpus = std::fs::read_to_string(&cfg.corpus).with_context(|| format!("reading corpus {}", cfg.corpus.display()))?;
    let c = Client {
        agent: http::agent(Some(Duration::from_secs(cfg.timeout_s))),
        url: format!("{}/chat/completions", http::api_base(&cfg.url)),
        cfg: cfg.clone(),
    };
    let mut out = serde_json::Map::new();
    for w in &cfg.workloads {
        let t0 = Instant::now();
        let (steps, res) = run_workload(&c, w, &corpus)?;
        let prompt: u64 = res.iter().map(|r| r.prompt_tokens).sum();
        let cached: u64 = res.iter().map(|r| r.cached_tokens).sum();
        let rows: Vec<Value> = steps
            .iter()
            .zip(&res)
            .map(|(s, r)| {
                json!({"conv": s.conv, "prompt_tokens": r.prompt_tokens, "cached_tokens": r.cached_tokens,
                       "computed_tokens": r.prompt_tokens - r.cached_tokens.min(r.prompt_tokens),
                       "ttft_ms": (r.ttft_ms * 10.0).round() / 10.0, "e2e_ms": (r.e2e_ms * 10.0).round() / 10.0})
            })
            .collect();
        out.insert(
            w.clone(),
            json!({
                "requests": res.len(),
                "prompt_tokens": prompt,
                "cached_tokens": cached,
                "computed_tokens": prompt - cached.min(prompt),
                "ttft_ms": stats(&res.iter().map(|r| r.ttft_ms).collect::<Vec<_>>()),
                "ttft_ms_total": (res.iter().map(|r| r.ttft_ms).filter(|x| x.is_finite()).sum::<f64>() * 10.0).round() / 10.0,
                "wall_s": (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0,
                "rows": rows,
            }),
        );
    }
    // the server's own view, when it has a prefix store
    let store = c
        .agent
        .get(&format!("{}/v1/intarsia/prefix-store", http::origin(&cfg.url)))
        .call()
        .ok()
        .and_then(|mut r| r.body_mut().read_to_string().ok())
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .map(|mut v| {
            if let Some(o) = v.as_object_mut() {
                o.remove("entries");
            }
            v
        })
        .unwrap_or(Value::Null);
    Ok(json!({"kind": "reuse", "model": cfg.model, "url": cfg.url, "max_tokens": cfg.max_tokens,
              "replayed": cfg.replay.is_some(), "workloads": out, "prefix_store": store}))
}

pub fn render_reuse(v: &Value) -> String {
    let mut s = String::from("workload  requests  prompt tok  cached tok  computed tok  TTFT mean ms  TTFT median ms  TTFT total s\n");
    if let Some(ws) = v["workloads"].as_object() {
        for (name, w) in ws {
            s.push_str(&format!(
                "{name:<9} {:>8}  {:>10}  {:>10}  {:>12}  {:>12}  {:>14}  {:>12.2}\n",
                w["requests"],
                w["prompt_tokens"],
                w["cached_tokens"],
                w["computed_tokens"],
                w["ttft_ms"]["mean"],
                w["ttft_ms"]["median"],
                w["ttft_ms_total"].as_f64().unwrap_or(0.0) / 1e3,
            ));
        }
    }
    s
}
