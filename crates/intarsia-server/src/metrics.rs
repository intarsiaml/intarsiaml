//! Prometheus text exposition (hand-written, no dependency) plus the two
//! server-side histograms (TTFT, inter-token latency). Everything else comes
//! from the engine's [`TelemetrySnapshot`], see `docs/telemetry.md`.

use intarsia_core::engine::ModelInfo;
use intarsia_core::telemetry::TelemetrySnapshot;
use std::fmt::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

const TTFT_BUCKETS: &[f64] = &[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0];
const ITL_BUCKETS: &[f64] = &[0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0];

struct HistData {
    counts: Vec<u64>,
    sum: f64,
    count: u64,
}

pub struct Histogram {
    bounds: &'static [f64],
    data: Mutex<HistData>,
}

impl Histogram {
    fn new(bounds: &'static [f64]) -> Histogram {
        Histogram { bounds, data: Mutex::new(HistData { counts: vec![0; bounds.len()], sum: 0.0, count: 0 }) }
    }

    /// Record `n` observations of value `v` (seconds).
    pub fn observe_n(&self, v: f64, n: u64) {
        if n == 0 || !v.is_finite() {
            return;
        }
        let mut d = self.data.lock().unwrap();
        if let Some(i) = self.bounds.iter().position(|b| v <= *b) {
            d.counts[i] += n;
        }
        d.sum += v * n as f64;
        d.count += n;
    }

    pub fn count(&self) -> u64 {
        self.data.lock().unwrap().count
    }

    fn render(&self, out: &mut String, name: &str, help: &str) {
        let d = self.data.lock().unwrap();
        let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} histogram");
        let mut cum = 0u64;
        for (b, c) in self.bounds.iter().zip(&d.counts) {
            cum += c;
            let _ = writeln!(out, "{name}_bucket{{le=\"{b}\"}} {cum}");
        }
        let _ = writeln!(out, "{name}_bucket{{le=\"+Inf\"}} {}", d.count);
        let _ = writeln!(out, "{name}_sum {}\n{name}_count {}", d.sum, d.count);
    }
}

pub struct Metrics {
    pub ttft: Histogram,
    pub itl: Histogram,
    /// Requests refused with 429 because the queue was full.
    pub rejected: AtomicU64,
    failed: AtomicBool,
    stall_ms: AtomicU64,
}

impl Metrics {
    pub fn new() -> Metrics {
        Metrics {
            ttft: Histogram::new(TTFT_BUCKETS),
            itl: Histogram::new(ITL_BUCKETS),
            rejected: AtomicU64::new(0),
            failed: AtomicBool::new(false),
            stall_ms: AtomicU64::new(0),
        }
    }

    /// Liveness as last judged by `/health` or the watchdog.
    pub fn set_health(&self, failed: bool, no_progress: Duration) {
        self.failed.store(failed, Ordering::Relaxed);
        self.stall_ms.store(no_progress.as_millis() as u64, Ordering::Relaxed);
    }

    /// Called once per finished request with the engine's timing report.
    pub fn observe_done(&self, completion_tokens: usize, prefill_ms: f64, decode_ms: f64) {
        if completion_tokens == 0 {
            return;
        }
        self.ttft.observe_n(prefill_ms / 1e3, 1);
        if completion_tokens > 1 {
            let n = (completion_tokens - 1) as u64;
            self.itl.observe_n(decode_ms / 1e3 / n as f64, n);
        }
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

fn metric(out: &mut String, name: &str, kind: &str, help: &str, value: impl std::fmt::Display) {
    let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} {kind}\n{name} {value}");
}

pub fn render(snap: &TelemetrySnapshot, info: &ModelInfo, m: &Metrics) -> String {
    let mut o = String::with_capacity(4096);
    metric(
        &mut o,
        "intarsia_engine_steps_total",
        "counter",
        "Engine steps (prefill chunks and decode steps).",
        snap.steps_total,
    );
    metric(&mut o, "intarsia_requests_running", "gauge", "Requests currently executing.", snap.running);
    metric(&mut o, "intarsia_requests_waiting", "gauge", "Requests queued behind the running one.", snap.waiting);
    metric(&mut o, "intarsia_requests_total", "counter", "Requests accepted by the engine.", snap.requests_total);
    // The names vLLM and llama.cpp publish; localeval's idle check and fleet tooling read these.
    metric(&mut o, "vllm:num_requests_running", "gauge", "Requests currently executing (vLLM name).", snap.running);
    metric(&mut o, "vllm:num_requests_waiting", "gauge", "Requests queued (vLLM name).", snap.waiting);
    metric(
        &mut o,
        "intarsia_requests_rejected_total",
        "counter",
        "Requests refused with 429 because the queue was full.",
        m.rejected.load(Ordering::Relaxed),
    );
    metric(
        &mut o,
        "intarsia_engine_failed",
        "gauge",
        "1 when the engine worker has failed and the server answers 503.",
        m.failed.load(Ordering::Relaxed) as u8,
    );
    metric(
        &mut o,
        "intarsia_engine_no_progress_seconds",
        "gauge",
        "Seconds the engine has had work without completing a step (watchdog input).",
        m.stall_ms.load(Ordering::Relaxed) as f64 / 1e3,
    );
    let _ = writeln!(o, "# HELP intarsia_requests_finished_total Finished requests by reason.\n# TYPE intarsia_requests_finished_total counter");
    for (reason, n) in &snap.finished {
        let _ = writeln!(o, "intarsia_requests_finished_total{{reason=\"{}\"}} {n}", esc(reason));
    }
    metric(&mut o, "intarsia_tokens_prompt_total", "counter", "Prompt tokens processed.", snap.prompt_tokens_total);
    metric(&mut o, "intarsia_tokens_generated_total", "counter", "Tokens generated.", snap.generation_tokens_total);
    metric(
        &mut o,
        "intarsia_tokens_cached_total",
        "counter",
        "Prompt tokens served from the reused KV prefix.",
        snap.cached_tokens_total,
    );
    m.ttft.render(&mut o, "intarsia_ttft_seconds", "Time to first token.");
    m.itl.render(&mut o, "intarsia_itl_seconds", "Inter-token latency (per-request mean, weighted by token count).");
    metric(&mut o, "intarsia_kv_bytes_used", "gauge", "KV cache bytes in use.", snap.kv_used_bytes);
    metric(&mut o, "intarsia_kv_bytes_total", "gauge", "KV cache bytes allocated.", snap.kv_total_bytes);
    metric(&mut o, "intarsia_kv_tokens_used", "gauge", "KV cache tokens in use.", snap.kv_used_tokens);
    metric(&mut o, "intarsia_kv_tokens_capacity", "gauge", "KV cache capacity in tokens.", snap.kv_capacity_tokens);
    metric(&mut o, "intarsia_weights_bytes", "gauge", "Model weight bytes resident.", snap.weights_bytes);
    metric(&mut o, "intarsia_device_memory_total_bytes", "gauge", "Device memory total.", snap.device.total_mem_bytes);
    metric(
        &mut o,
        "intarsia_device_memory_free_bytes",
        "gauge",
        "Device memory free at scrape time.",
        snap.device.free_mem_bytes,
    );
    metric(
        &mut o,
        "intarsia_last_prefill_tokens_per_second",
        "gauge",
        "Prefill speed of the last finished request.",
        snap.last_prefill_tok_s,
    );
    metric(
        &mut o,
        "intarsia_last_decode_tokens_per_second",
        "gauge",
        "Decode speed of the last finished request.",
        snap.last_decode_tok_s,
    );
    metric(
        &mut o,
        "intarsia_graph_replays_total",
        "counter",
        "Decode steps replayed from a captured CUDA graph.",
        snap.graph_replays,
    );
    metric(&mut o, "intarsia_graph_captures_total", "counter", "Decode graphs captured.", snap.graph_captures);
    metric(
        &mut o,
        "intarsia_decode_eager_steps_total",
        "counter",
        "Single-token decode steps that ran through the eager (non-graph) path.",
        snap.decode_eager_steps,
    );
    metric(
        &mut o,
        "intarsia_decode_profiled_steps_total",
        "counter",
        "Eager decode steps that ran with per-op profiling (synchronises around every op).",
        snap.decode_profiled_steps,
    );
    metric(
        &mut o,
        "intarsia_spec_steps_total",
        "counter",
        "Speculative decoding verify passes (--spec).",
        snap.spec_steps,
    );
    metric(
        &mut o,
        "intarsia_spec_proposed_tokens_total",
        "counter",
        "Draft tokens proposed to speculative verify passes.",
        snap.spec_proposed,
    );
    metric(
        &mut o,
        "intarsia_spec_accepted_tokens_total",
        "counter",
        "Draft tokens accepted by speculative verify passes.",
        snap.spec_accepted,
    );
    metric(&mut o, "intarsia_uptime_seconds", "gauge", "Seconds since engine start.", snap.uptime_s);
    metric(
        &mut o,
        "intarsia_last_step_timestamp_seconds",
        "gauge",
        "Unix time of the last engine step (0 = none yet).",
        snap.last_step_ms as f64 / 1e3,
    );
    let _ = writeln!(o, "# HELP intarsia_op_seconds_total Cumulative sampled time per op (profiled launches only).\n# TYPE intarsia_op_seconds_total counter");
    for (op, s) in &snap.ops {
        let _ = writeln!(o, "intarsia_op_seconds_total{{op=\"{}\"}} {}", esc(op), s.time_ms / 1e3);
    }
    let _ = writeln!(o, "# HELP intarsia_op_calls_total Launches per op.\n# TYPE intarsia_op_calls_total counter");
    for (op, s) in &snap.ops {
        let _ = writeln!(o, "intarsia_op_calls_total{{op=\"{}\"}} {}", esc(op), s.calls);
    }
    let _ = writeln!(o, "# HELP intarsia_info Static build/model information.\n# TYPE intarsia_info gauge");
    let _ = writeln!(
        o,
        "intarsia_info{{model=\"{}\",family=\"{}\",backend=\"{}\",device=\"{}\",act_dtype=\"{}\",kv_dtype=\"{}\"}} 1",
        esc(&info.id),
        esc(&info.family),
        esc(&info.backend),
        esc(&info.device),
        esc(&info.act_dtype),
        esc(&info.kv_dtype)
    );
    o
}
