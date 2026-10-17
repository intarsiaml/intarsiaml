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

/// A scheduler [`LatencyHist`](intarsia_core::scheduler::LatencyHist) (milliseconds, per-bucket counts) as a Prometheus
/// histogram in seconds.
fn render_latency(o: &mut String, name: &str, help: &str, h: &intarsia_core::scheduler::LatencyHist) {
    use intarsia_core::scheduler::LatencyHist;
    let _ = writeln!(o, "# HELP {name} {help}\n# TYPE {name} histogram");
    let mut cum = 0u64;
    for (i, ms) in LatencyHist::BOUNDS_MS.iter().enumerate() {
        cum += h.counts.get(i).copied().unwrap_or(0);
        let _ = writeln!(o, "{name}_bucket{{le=\"{}\"}} {cum}", ms / 1e3);
    }
    let _ = writeln!(o, "{name}_bucket{{le=\"+Inf\"}} {}\n{name}_sum {}\n{name}_count {}", h.count, h.sum_ms / 1e3, h.count);
}

/// Scheduler metrics of a batching engine (`--max-batch N`): batch-size histogram of the decode passes, queue depth,
/// prefill work, request outcomes, latencies. The two request gauges differ from the `intarsia_requests_*` ones only in
/// where they are counted, see the help texts.
pub fn render_batch(b: &intarsia_core::scheduler::BatchMetrics) -> String {
    let mut o = String::with_capacity(4096);
    metric(&mut o, "intarsia_batch_iterations_total", "counter", "Scheduler iterations.", b.iterations);
    metric(&mut o, "intarsia_batch_decode_steps_total", "counter", "Batched decode passes.", b.decode_steps);
    let _ = writeln!(o, "# HELP intarsia_batch_decode_rows_total Decode passes by number of rows.");
    let _ = writeln!(o, "# TYPE intarsia_batch_decode_rows_total counter");
    for (rows, n) in b.batch_hist.iter().enumerate().skip(1) {
        let _ = writeln!(o, "intarsia_batch_decode_rows_total{{rows=\"{rows}\"}} {n}");
    }
    metric(&mut o, "intarsia_batch_prefill_chunks_total", "counter", "Prefill chunks run.", b.prefill_chunks);
    metric(&mut o, "intarsia_batch_prefill_tokens_total", "counter", "Prompt tokens prefilled.", b.prefill_tokens);
    metric(
        &mut o,
        "intarsia_batch_mixed_iterations_total",
        "counter",
        "Iterations that ran a decode phase and then a prefill chunk (the decoding requests waited behind the chunk).",
        b.mixed_iterations,
    );
    metric(
        &mut o,
        "intarsia_batch_running",
        "gauge",
        "Requests holding a KV slot, prefilling or decoding, as of the end of the scheduler's last iteration.",
        b.running,
    );
    metric(
        &mut o,
        "intarsia_batch_queue_depth",
        "gauge",
        "Requests accepted by the batch engine and not yet holding a slot, now: in the hand-off channel to the scheduler thread plus its FIFO queue. Live; counts a request until it is given a slot.",
        b.queue_depth,
    );
    metric(
        &mut o,
        "intarsia_batch_queue_depth_max",
        "gauge",
        "Highest intarsia_batch_queue_depth since start, taken at every submit.",
        b.queue_depth_max,
    );
    metric(&mut o, "intarsia_batch_cached_tokens_total", "counter", "Prompt tokens served from reused KV.", b.cached_tokens);
    metric(
        &mut o,
        "intarsia_batch_step_ewma_ms",
        "gauge",
        "EWMA (alpha 0.1) of the wall time of a whole scheduler iteration that ran a decode step: the decode pass (which first waits for the previous iteration's prefill chunk to finish on the GPU) plus enqueueing this iteration's chunk.",
        b.step_ewma_ms,
    );
    metric(&mut o, "intarsia_batch_requests_admitted_total", "counter", "Requests given a slot.", b.admitted);
    metric(&mut o, "intarsia_batch_requests_finished_total", "counter", "Requests that finished normally.", b.finished);
    metric(&mut o, "intarsia_batch_requests_cancelled_total", "counter", "Requests cancelled (queued, prefilling or decoding).", b.cancelled);
    metric(&mut o, "intarsia_batch_requests_failed_total", "counter", "Requests that ended with an error after being accepted.", b.failed);
    metric(
        &mut o,
        "intarsia_batch_requests_rejected_total",
        "counter",
        "Requests refused by the batch engine: queue full, or invalid at admission.",
        b.rejected,
    );
    render_latency(&mut o, "intarsia_batch_step_seconds", "Wall time of a scheduler iteration that ran a decode step.", &b.step);
    render_latency(
        &mut o,
        "intarsia_batch_chunk_call_seconds",
        "Wall time of the prefill_chunk call. A non-final chunk is enqueued and the call returns at once, so this is the host cost only; its GPU time shows up in the next decode step. The final chunk waits for the first token and includes its GPU time.",
        &b.chunk,
    );
    render_latency(&mut o, "intarsia_batch_ttft_seconds", "Submit to first token, per request.", &b.ttft);
    render_latency(&mut o, "intarsia_batch_itl_seconds", "Between consecutive tokens of one request.", &b.itl);
    o
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
    metric(&mut o, "intarsia_requests_running", "gauge", "Requests holding the engine now: the one running request (max-batch 1) or, with --max-batch N, those that started (got a slot), prefilling or decoding.", snap.running);
    metric(&mut o, "intarsia_requests_waiting", "gauge", "Requests accepted by the engine and not yet started, counted from submit: with --max-batch N, those in the hand-off channel and the scheduler queue (a request leaves when it gets a KV slot), plus a request being refused for a full queue until it is refused.", snap.waiting);
    metric(&mut o, "intarsia_requests_total", "counter", "Requests accepted into the engine queue (refused submits excluded); once idle, equals the sum of intarsia_requests_finished_total over reasons.", snap.requests_total);
    // The names vLLM and llama.cpp publish; localeval's idle check and fleet tooling read these.
    metric(&mut o, "vllm:num_requests_running", "gauge", "Same value as intarsia_requests_running (vLLM name).", snap.running);
    metric(&mut o, "vllm:num_requests_waiting", "gauge", "Same value as intarsia_requests_waiting (vLLM name).", snap.waiting);
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

#[cfg(test)]
mod tests {
    use super::*;
    use intarsia_core::scheduler::BatchMetrics;
    use std::time::Duration;

    #[test]
    fn batch_metrics_render_histograms_and_counters() {
        let mut b = BatchMetrics { queue_depth: 3, queue_depth_max: 7, mixed_iterations: 5, cancelled: 2, ..Default::default() };
        b.chunk.record(Duration::from_millis(70));
        b.chunk.record(Duration::from_millis(2000));
        let o = render_batch(&b);
        assert!(o.contains("\nintarsia_batch_queue_depth 3\n"), "{o}");
        assert!(o.contains("\nintarsia_batch_queue_depth_max 7\n"));
        assert!(o.contains("\nintarsia_batch_mixed_iterations_total 5\n"));
        assert!(o.contains("\nintarsia_batch_requests_cancelled_total 2\n"));
        // 70 ms is in the le=0.075 bucket and every later one; 2 s only in +Inf
        assert!(o.contains("intarsia_batch_chunk_call_seconds_bucket{le=\"0.05\"} 0\n"), "{o}");
        assert!(o.contains("intarsia_batch_chunk_call_seconds_bucket{le=\"0.075\"} 1\n"));
        assert!(o.contains("intarsia_batch_chunk_call_seconds_bucket{le=\"1\"} 1\n"));
        assert!(o.contains("intarsia_batch_chunk_call_seconds_bucket{le=\"+Inf\"} 2\n"));
        assert!(o.contains("intarsia_batch_chunk_call_seconds_count 2\n"));
        // an empty histogram renders zeros, not a panic
        assert!(o.contains("intarsia_batch_step_seconds_bucket{le=\"0.001\"} 0\n"));
        // every metric has HELP and TYPE
        assert_eq!(o.matches("# HELP ").count(), o.matches("# TYPE ").count());
    }
}
