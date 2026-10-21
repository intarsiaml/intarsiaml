//! TUI state: everything the renderer shows, fed by telemetry snapshots, `/metrics` scrapes, the plan and chat events.
//!
//! Time is injected (`t` in seconds on any monotonic clock the caller keeps), so rates and windows are deterministic in
//! tests and the render path never reads a clock.

use crate::chat::{ChatEvent, ChatMsg, Role};
use crate::layout::{self, ArchMap, Plan};
use crate::prom::{Hist, Prom};
use intarsia_bench::snapshot::{OpStat, Snapshot};
use std::collections::{BTreeMap, VecDeque};

/// Sparkline history length (points) and spacing (seconds): 240 points x 0.5 s = 2 minutes.
pub const HIST: usize = 240;
pub const HIST_EVERY_S: f64 = 0.5;
/// Rates are taken over this many seconds of counter deltas.
const RATE_WINDOW_S: f64 = 1.2;
/// Latency percentiles are taken over the observations of the last this many seconds, when there are any.
const LAT_WINDOW_S: f64 = 30.0;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Quantiles {
    pub p50: Option<f64>,
    pub p95: Option<f64>,
    pub p99: Option<f64>,
    pub count: f64,
    /// True when computed over the recent window, false when over everything since server start.
    pub windowed: bool,
}

/// Scheduler metrics that exist only on a batching engine (`--max-batch N`); `None` fields render as `n/a`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BatchView {
    pub avg_rows: Option<f64>,
    pub queue_depth: Option<f64>,
    pub queue_depth_max: Option<f64>,
    pub step_ewma_ms: Option<f64>,
}

struct Scrape {
    t: f64,
    ttft: Option<Hist>,
    itl: Option<Hist>,
    /// (rows, decode passes) from `intarsia_batch_decode_rows_total`.
    rows: Vec<(f64, f64)>,
}

#[derive(Default)]
pub struct App {
    pub url: String,
    pub status: String,
    pub plan: Option<Plan>,
    pub plan_known: bool,
    pub map: ArchMap,
    map_ops: Vec<String>,
    pub snap: Option<Snapshot>,
    win: VecDeque<(f64, u64, u64, u64)>,
    pub rate_gen: f64,
    /// True when `rate_gen` is the step rate standing in for decode tok/s (the server books token counters when a request ends).
    pub rate_gen_est: bool,
    pub rate_prefill: f64,
    pub rate_steps: f64,
    last_push: f64,
    pub hist_gen: VecDeque<f64>,
    pub hist_prefill: VecDeque<f64>,
    pub hist_waiting: VecDeque<f64>,
    pub hist_kv: VecDeque<f64>,
    pub metrics_ok: bool,
    /// True when decode/prefill rates come from the batch scheduler's live counters in `/metrics`.
    pub live_rates: bool,
    prev_tokens: Option<(f64, f64, f64)>,
    scrapes: VecDeque<Scrape>,
    pub ttft: Quantiles,
    pub itl: Quantiles,
    pub batch: BatchView,
    /// Share of estimated op time per op (0..=1); empty when the server has not timed any op (profiling off).
    pub op_share: BTreeMap<String, f64>,
    pub ops_timed: bool,
    pub model: String,
    /// Model id for chat requests: the first id on `/v1/models` (the served name), else the telemetry model.
    pub chat_model: String,
    // chat
    pub chat: Vec<ChatMsg>,
    pub input: String,
    pub streaming: bool,
    pub stream_chunks: u64,
    pub last_chat: Option<(u64, f64)>,
    /// Lines scrolled up from the bottom of the chat history.
    pub chat_scroll: usize,
    pub quit: bool,
}

impl App {
    pub fn new(url: &str) -> App {
        let mut a = App { url: url.to_string(), status: "connecting...".into(), ..Default::default() };
        a.rebuild_map();
        a
    }

    fn rebuild_map(&mut self) {
        self.map = layout::build(self.plan.as_ref(), &self.map_ops);
    }

    pub fn ingest_plan(&mut self, plan: Option<Plan>) {
        self.plan_known = true;
        if plan != self.plan {
            self.plan = plan;
            self.rebuild_map();
        }
    }

    pub fn ingest_snapshot(&mut self, s: Snapshot, t: f64) {
        // map structure follows the op names the engine reports
        let names: Vec<String> = s.ops.keys().cloned().collect();
        if names != self.map_ops {
            self.map_ops = names;
            self.rebuild_map();
        }
        if !s.model.is_empty() {
            self.model = s.model.clone();
        }
        // rates from counter deltas over the last RATE_WINDOW_S
        self.win.push_back((t, s.generation_tokens_total, s.prompt_tokens_total.saturating_sub(s.cached_tokens_total), s.steps_total));
        while self.win.len() > 2 && self.win.front().is_some_and(|f| t - f.0 > RATE_WINDOW_S) {
            self.win.pop_front();
        }
        if let (Some(a), Some(b)) = (self.win.front(), self.win.back()) {
            let dt = b.0 - a.0;
            if dt >= 0.25 {
                let rate = |x: u64, y: u64| y.saturating_sub(x) as f64 / dt;
                self.rate_steps = rate(a.3, b.3);
                if !self.live_rates {
                    // The server books token counters when a request finishes, so the counter delta is a burst after
                    // the fact, not a rate: only trust it while something runs, else show idle.
                    self.rate_gen = if s.running > 0 { rate(a.1, b.1) } else { 0.0 };
                    self.rate_prefill = if s.running > 0 { rate(a.2, b.2) } else { 0.0 };
                    // while one runs the counters stand still: steps/s approximates decode tok/s (the web UI's map does the
                    // same and marks it with "~")
                    self.rate_gen_est = s.running > 0 && self.rate_gen == 0.0 && self.rate_prefill == 0.0 && self.rate_steps > 0.0;
                    if self.rate_gen_est {
                        self.rate_gen = self.rate_steps;
                    }
                }
            }
        }
        self.update_op_share(&s);
        if t - self.last_push >= HIST_EVERY_S || self.hist_gen.is_empty() {
            self.last_push = t;
            push(&mut self.hist_gen, self.rate_gen);
            push(&mut self.hist_prefill, self.rate_prefill);
            push(&mut self.hist_waiting, s.waiting as f64);
            let kv = if s.kv_capacity_tokens > 0 {
                s.kv_used_tokens as f64 / s.kv_capacity_tokens as f64
            } else if s.kv_total_bytes > 0 {
                s.kv_used_bytes as f64 / s.kv_total_bytes as f64
            } else {
                0.0
            };
            push(&mut self.hist_kv, kv * 100.0);
        }
        self.snap = Some(s);
    }

    fn update_op_share(&mut self, s: &Snapshot) {
        let est = |o: &OpStat| {
            if o.timed_calls == 0 {
                0.0
            } else {
                o.time_ms * (o.calls.max(o.timed_calls) as f64 / o.timed_calls as f64)
            }
        };
        let total: f64 = s.ops.values().map(est).sum();
        self.ops_timed = total > 0.0;
        self.op_share = if self.ops_timed { s.ops.iter().map(|(k, v)| (k.clone(), est(v) / total)).collect() } else { BTreeMap::new() };
    }

    pub fn ingest_metrics(&mut self, text: &str, t: f64) {
        let p = Prom::parse(text);
        self.metrics_ok = !p.samples.is_empty();
        let pick = |a: &str, b: &str| {
            let first = p.hist(a).filter(|h| h.count > 0.0);
            first.or_else(|| p.hist(b))
        };
        let rows: Vec<(f64, f64)> = p
            .all("intarsia_batch_decode_rows_total")
            .filter_map(|s| Some((s.label("rows")?.parse().ok()?, s.value)))
            .collect();
        self.scrapes.push_back(Scrape {
            t,
            ttft: pick("intarsia_ttft_seconds", "intarsia_batch_ttft_seconds"),
            itl: pick("intarsia_itl_seconds", "intarsia_batch_itl_seconds"),
            rows,
        });
        while self.scrapes.len() > 2 && self.scrapes.front().is_some_and(|f| t - f.t > LAT_WINDOW_S * 2.0) {
            self.scrapes.pop_front();
        }
        // oldest scrape inside the window (or the newest-but-one if every earlier one is older)
        let base = self.scrapes.iter().find(|s| t - s.t <= LAT_WINDOW_S).filter(|b| b.t < t);
        let cur = self.scrapes.back().unwrap();
        self.ttft = quantiles(cur.ttft.as_ref(), base.and_then(|b| b.ttft.as_ref()));
        self.itl = quantiles(cur.itl.as_ref(), base.and_then(|b| b.itl.as_ref()));
        let batching = p.get("intarsia_batch_iterations_total").is_some();
        // Live tok/s from the scheduler's counters (decode rows x passes, prefilled prompt tokens): they move per iteration,
        // unlike the telemetry token counters.
        if batching {
            let decoded: f64 = cur.rows.iter().map(|(r, n)| r * n).sum();
            let prefilled = p.get("intarsia_batch_prefill_tokens_total").unwrap_or(0.0);
            if let Some((pt, pd, pp)) = self.prev_tokens {
                let dt = t - pt;
                if dt >= 0.5 && decoded >= pd && prefilled >= pp {
                    self.rate_gen = (decoded - pd) / dt;
                    self.rate_prefill = (prefilled - pp) / dt;
                    self.rate_gen_est = false;
                    self.live_rates = true;
                    self.prev_tokens = Some((t, decoded, prefilled));
                } else if decoded < pd || prefilled < pp {
                    self.prev_tokens = Some((t, decoded, prefilled)); // counter reset
                }
            } else {
                self.prev_tokens = Some((t, decoded, prefilled));
            }
        }
        self.batch = BatchView {
            avg_rows: batching.then(|| avg_rows(&cur.rows, base.map(|b| b.rows.as_slice()))).flatten(),
            queue_depth: p.get("intarsia_batch_queue_depth"),
            queue_depth_max: p.get("intarsia_batch_queue_depth_max"),
            step_ewma_ms: p.get("intarsia_batch_step_ewma_ms"),
        };
    }

    // ---- chat -----------------------------------------------------------------------------------------------------

    /// Start a turn from the input line; returns the history to send. Ignores empty input and a turn in flight.
    pub fn submit(&mut self) -> Option<Vec<ChatMsg>> {
        let text = self.input.trim().to_string();
        if text.is_empty() || self.streaming {
            return None;
        }
        self.input.clear();
        if text == "/clear" {
            self.chat.clear();
            return None;
        }
        self.chat.push(ChatMsg { role: Role::User, text });
        self.streaming = true;
        self.stream_chunks = 0;
        self.chat_scroll = 0;
        Some(self.chat.clone())
    }

    pub fn chat_event(&mut self, ev: ChatEvent) {
        match ev {
            ChatEvent::Token(t) => {
                self.stream_chunks += 1;
                match self.chat.last_mut() {
                    Some(m) if m.role == Role::Assistant && self.streaming => m.text.push_str(&t),
                    _ => self.chat.push(ChatMsg { role: Role::Assistant, text: t }),
                }
            }
            ChatEvent::Done { tokens, secs } => {
                self.streaming = false;
                self.last_chat = Some((tokens, secs));
            }
            ChatEvent::Error(e) => {
                self.streaming = false;
                self.chat.push(ChatMsg { role: Role::Note, text: format!("error: {e}") });
            }
        }
    }
}

fn push(h: &mut VecDeque<f64>, v: f64) {
    if h.len() >= HIST {
        h.pop_front();
    }
    h.push_back(v);
}

fn quantiles(cur: Option<&Hist>, base: Option<&Hist>) -> Quantiles {
    let Some(cur) = cur else { return Quantiles::default() };
    let windowed = base.map(|b| cur.since(b)).filter(|d| d.count > 0.0 && d.buckets.last().is_some_and(|b| b.1 > 0.0));
    let (h, w) = match &windowed {
        Some(d) => (d, true),
        None => (cur, false),
    };
    Quantiles { p50: h.quantile(0.5), p95: h.quantile(0.95), p99: h.quantile(0.99), count: h.count, windowed: w }
}

fn avg_rows(cur: &[(f64, f64)], base: Option<&[(f64, f64)]>) -> Option<f64> {
    let delta: Vec<(f64, f64)> = cur
        .iter()
        .map(|&(r, n)| {
            let b = base.and_then(|b| b.iter().find(|x| x.0 == r)).map_or(0.0, |x| x.1);
            (r, if n >= b { n - b } else { n })
        })
        .collect();
    let mean = |v: &[(f64, f64)]| {
        let n: f64 = v.iter().map(|x| x.1).sum();
        (n > 0.0).then(|| v.iter().map(|x| x.0 * x.1).sum::<f64>() / n)
    };
    mean(&delta).or_else(|| mean(cur))
}

#[cfg(test)]
mod tests {
    use super::*;
    use intarsia_bench::snapshot::OpStat;

    fn snap(gen: u64, prompt: u64, steps: u64) -> Snapshot {
        let mut s = Snapshot { model: "m".into(), running: 1, generation_tokens_total: gen, prompt_tokens_total: prompt, steps_total: steps, ..Default::default() };
        s.ops.insert("matmul".into(), OpStat { calls: 100, timed_calls: 10, time_ms: 30.0 });
        s.ops.insert("rope".into(), OpStat { calls: 100, timed_calls: 10, time_ms: 10.0 });
        s
    }

    #[test]
    fn rates_follow_counter_deltas() {
        let mut a = App::new("u");
        a.ingest_snapshot(snap(0, 0, 0), 0.0);
        a.ingest_snapshot(snap(20, 100, 25), 1.0);
        assert!((a.rate_gen - 20.0).abs() < 1e-9);
        assert!((a.rate_prefill - 100.0).abs() < 1e-9);
        assert!((a.rate_steps - 25.0).abs() < 1e-9);
        // idle afterwards: the window slides past the burst and the rate returns to 0
        for i in 2..8 {
            a.ingest_snapshot(snap(20, 100, 25), i as f64);
        }
        assert_eq!(a.rate_gen, 0.0);
    }

    #[test]
    fn op_share_uses_estimated_time_and_is_empty_without_timing() {
        let mut a = App::new("u");
        a.ingest_snapshot(snap(0, 0, 0), 0.0);
        assert!(a.ops_timed);
        assert!((a.op_share["matmul"] - 0.75).abs() < 1e-9);
        let mut s = snap(0, 0, 0);
        for o in s.ops.values_mut() {
            o.timed_calls = 0;
            o.time_ms = 0.0;
        }
        a.ingest_snapshot(s, 1.0);
        assert!(!a.ops_timed);
        assert!(a.op_share.is_empty());
    }

    #[test]
    fn map_picks_up_new_telemetry_ops_and_plan() {
        let mut a = App::new("u");
        assert!(!a.map.from_plan);
        a.ingest_snapshot(snap(0, 0, 0), 0.0);
        assert!(a.map.rows.iter().any(|r| r.nodes.iter().any(|n| n.id == "op:matmul")));
        a.ingest_plan(Plan::from_json(&serde_json::json!({"layers": [{"index": 0, "blocks": ["attention", "dense_ffn"]}]})));
        assert!(a.map.from_plan);
        assert_eq!(a.map.layer_count, 1);
    }

    fn metrics(ttft_counts: [u32; 3], rows: [u32; 2]) -> String {
        let c = ttft_counts;
        format!(
            "intarsia_batch_iterations_total 5\nintarsia_batch_queue_depth 2\n\
             intarsia_batch_decode_rows_total{{rows=\"1\"}} {}\nintarsia_batch_decode_rows_total{{rows=\"2\"}} {}\n\
             intarsia_ttft_seconds_bucket{{le=\"0.1\"}} {}\nintarsia_ttft_seconds_bucket{{le=\"1\"}} {}\n\
             intarsia_ttft_seconds_bucket{{le=\"+Inf\"}} {}\nintarsia_ttft_seconds_sum 1\nintarsia_ttft_seconds_count {}\n",
            rows[0], rows[1], c[0], c[1], c[2], c[2]
        )
    }

    #[test]
    fn latency_is_windowed_when_new_observations_exist_else_cumulative() {
        let mut a = App::new("u");
        a.ingest_metrics(&metrics([10, 10, 10], [4, 0]), 0.0);
        // first scrape only: cumulative
        assert!(!a.ttft.windowed);
        assert_eq!(a.ttft.count, 10.0);
        // 4 new observations, all in (0.1, 1]
        a.ingest_metrics(&metrics([10, 14, 14], [4, 6]), 5.0);
        assert!(a.ttft.windowed);
        assert_eq!(a.ttft.count, 4.0);
        assert!(a.ttft.p50.unwrap() > 0.1);
        // batch size over the window: only the 6 two-row passes are new
        assert_eq!(a.batch.avg_rows, Some(2.0));
        assert_eq!(a.batch.queue_depth, Some(2.0));
        // nothing new for a while: back to cumulative
        a.ingest_metrics(&metrics([10, 14, 14], [4, 6]), 5.5);
        a.ingest_metrics(&metrics([10, 14, 14], [4, 6]), 60.0);
        assert!(!a.ttft.windowed);
        assert_eq!(a.ttft.count, 14.0);
    }

    #[test]
    fn batching_engine_gives_live_rates_from_scheduler_counters() {
        let m = |rows2: u32, prefill: u32| {
            format!(
                "intarsia_batch_iterations_total 5\nintarsia_batch_prefill_tokens_total {prefill}\n\
                 intarsia_batch_decode_rows_total{{rows=\"1\"}} 10\nintarsia_batch_decode_rows_total{{rows=\"2\"}} {rows2}\n"
            )
        };
        let mut a = App::new("u");
        a.ingest_metrics(&m(0, 0), 0.0);
        assert!(!a.live_rates, "needs two scrapes");
        // 1 s later: 20 two-row passes = 40 tokens, 300 prompt tokens
        a.ingest_metrics(&m(20, 300), 1.0);
        assert!(a.live_rates);
        assert!((a.rate_gen - 40.0).abs() < 1e-9);
        assert!((a.rate_prefill - 300.0).abs() < 1e-9);
        assert!(!a.rate_gen_est);
        // snapshots no longer overwrite them with the counter-burst estimate
        a.ingest_snapshot(snap(0, 0, 0), 1.1);
        a.ingest_snapshot(snap(0, 0, 50), 1.6);
        assert!((a.rate_gen - 40.0).abs() < 1e-9);
        // idle scrape: back to zero
        a.ingest_metrics(&m(20, 300), 2.0);
        assert_eq!(a.rate_gen, 0.0);
    }

    #[test]
    fn counter_burst_after_a_request_is_not_a_rate() {
        let mut a = App::new("u");
        a.ingest_snapshot(snap(0, 0, 0), 0.0);
        let mut done = snap(80, 100, 90);
        done.running = 0;
        a.ingest_snapshot(done, 0.5);
        assert_eq!(a.rate_gen, 0.0);
        assert_eq!(a.rate_prefill, 0.0);
    }

    #[test]
    fn non_batching_server_has_no_batch_view() {
        let mut a = App::new("u");
        a.ingest_metrics("intarsia_requests_running 0\n", 0.0);
        assert_eq!(a.batch, BatchView::default());
        assert_eq!(a.ttft, Quantiles::default());
    }

    #[test]
    fn chat_turn_lifecycle() {
        let mut a = App::new("u");
        assert!(a.submit().is_none());
        a.input = "hello".into();
        let h = a.submit().unwrap();
        assert_eq!(h.len(), 1);
        assert!(a.streaming);
        assert!(a.submit().is_none(), "a turn is in flight");
        a.chat_event(ChatEvent::Token("Hi".into()));
        a.chat_event(ChatEvent::Token(" there".into()));
        assert_eq!(a.chat.last().unwrap().text, "Hi there");
        a.chat_event(ChatEvent::Done { tokens: 2, secs: 0.5 });
        assert!(!a.streaming);
        // the next turn starts a new assistant message
        a.input = "again".into();
        a.submit().unwrap();
        a.chat_event(ChatEvent::Token("ok".into()));
        assert_eq!(a.chat.len(), 4);
        a.chat_event(ChatEvent::Error("boom".into()));
        assert_eq!(a.chat.last().unwrap().role, Role::Note);
        a.input = "/clear".into();
        a.submit();
        assert!(a.chat.is_empty());
    }
}
