//! Engine telemetry. One struct, [`TelemetrySnapshot`], feeds `/metrics`,
//! `/events` (SSE) and the TUI. Field meanings are documented in
//! `docs/telemetry.md`; keep that file in sync when changing this one.

use crate::backend::DeviceInfo;
use crate::ops::Op;
use serde::Serialize;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const TELEMETRY_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Default, Serialize)]
pub struct OpStat {
    /// Total launches since start.
    pub calls: u64,
    /// Launches whose duration was measured (profiling samples; all calls on cpu-ref).
    pub timed_calls: u64,
    /// Cumulative measured time in milliseconds over `timed_calls`.
    pub time_ms: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct DeviceTelemetry {
    pub name: String,
    pub backend: String,
    pub arch: String,
    pub total_mem_bytes: u64,
    /// Free memory measured at snapshot time.
    pub free_mem_bytes: u64,
    pub unified: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct TelemetrySnapshot {
    pub schema_version: u32,
    /// Milliseconds since the Unix epoch when the snapshot was taken.
    pub timestamp_ms: u64,
    pub uptime_s: f64,
    pub model: String,
    pub device: DeviceTelemetry,

    /// Monotonic count of engine steps (one forward pass: a prefill chunk or a decode step).
    /// A watchdog treats "work queued but this not advancing" as a wedge.
    pub steps_total: u64,
    /// Unix ms of the most recent step (0 if none yet).
    pub last_step_ms: u64,
    pub requests_total: u64,
    /// Requests currently executing (0 or 1 in Phase 0).
    pub running: u32,
    /// Requests queued behind the running one.
    pub waiting: u32,
    /// Finished requests by reason: "stop", "length", "cancelled", "error".
    pub finished: BTreeMap<String, u64>,

    pub prompt_tokens_total: u64,
    pub generation_tokens_total: u64,
    /// Prompt tokens served from the reused KV prefix.
    pub cached_tokens_total: u64,

    /// Throughput of the most recent finished request.
    pub last_prefill_tok_s: f64,
    pub last_decode_tok_s: f64,
    pub last_ttft_ms: f64,

    pub kv_used_bytes: u64,
    pub kv_total_bytes: u64,
    pub kv_used_tokens: u64,
    pub kv_capacity_tokens: u64,

    pub weights_bytes: u64,

    /// Decode steps replayed from a captured graph, and graphs captured (0 with `--no-graphs` or on backends without).
    pub graph_replays: u64,
    pub graph_captures: u64,

    /// Per-op cumulative time and call counts, keyed by op id (`rmsnorm`, `matmul`, ...).
    pub ops: BTreeMap<String, OpStat>,
}

struct Inner {
    model: String,
    device: Option<DeviceInfo>,
    finished: BTreeMap<String, u64>,
    ops: BTreeMap<Op, OpStat>,
    last_prefill_tok_s: f64,
    last_decode_tok_s: f64,
    last_ttft_ms: f64,
}

/// Shared, thread-safe collector. Cheap to update from the hot path.
pub struct Telemetry {
    start: Instant,
    steps: AtomicU64,
    last_step_ms: AtomicU64,
    requests: AtomicU64,
    running: AtomicU64,
    waiting: AtomicU64,
    prompt_tokens: AtomicU64,
    gen_tokens: AtomicU64,
    cached_tokens: AtomicU64,
    kv_used_bytes: AtomicU64,
    kv_total_bytes: AtomicU64,
    kv_used_tokens: AtomicU64,
    kv_cap_tokens: AtomicU64,
    weights_bytes: AtomicU64,
    graph_replays: AtomicU64,
    graph_captures: AtomicU64,
    inner: Mutex<Inner>,
    /// Free-memory probe for live device numbers (set by the engine).
    device_probe: Mutex<Option<Box<dyn Fn() -> DeviceInfo + Send + Sync>>>,
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

impl Default for Telemetry {
    fn default() -> Self {
        Self::new()
    }
}

impl Telemetry {
    pub fn new() -> Telemetry {
        Telemetry {
            start: Instant::now(),
            steps: AtomicU64::new(0),
            last_step_ms: AtomicU64::new(0),
            requests: AtomicU64::new(0),
            running: AtomicU64::new(0),
            waiting: AtomicU64::new(0),
            prompt_tokens: AtomicU64::new(0),
            gen_tokens: AtomicU64::new(0),
            cached_tokens: AtomicU64::new(0),
            kv_used_bytes: AtomicU64::new(0),
            kv_total_bytes: AtomicU64::new(0),
            kv_used_tokens: AtomicU64::new(0),
            kv_cap_tokens: AtomicU64::new(0),
            weights_bytes: AtomicU64::new(0),
            graph_replays: AtomicU64::new(0),
            graph_captures: AtomicU64::new(0),
            inner: Mutex::new(Inner {
                model: String::new(),
                device: None,
                finished: BTreeMap::new(),
                ops: BTreeMap::new(),
                last_prefill_tok_s: 0.0,
                last_decode_tok_s: 0.0,
                last_ttft_ms: 0.0,
            }),
            device_probe: Mutex::new(None),
        }
    }

    pub fn set_model(&self, name: &str, device: DeviceInfo, weights_bytes: u64) {
        let mut g = self.inner.lock().unwrap();
        g.model = name.to_string();
        g.device = Some(device);
        self.weights_bytes.store(weights_bytes, Ordering::Relaxed);
    }

    pub fn set_device_probe(&self, f: Box<dyn Fn() -> DeviceInfo + Send + Sync>) {
        *self.device_probe.lock().unwrap() = Some(f);
    }

    pub fn step(&self) {
        self.steps.fetch_add(1, Ordering::Relaxed);
        self.last_step_ms.store(now_ms(), Ordering::Relaxed);
    }

    /// Count a captured decode graph (`replay = false`) or one replay of it.
    pub fn graph_event(&self, replay: bool) {
        let c = if replay { &self.graph_replays } else { &self.graph_captures };
        c.fetch_add(1, Ordering::Relaxed);
    }

    pub fn steps_total(&self) -> u64 {
        self.steps.load(Ordering::Relaxed)
    }

    pub fn request_enqueued(&self) {
        self.requests.fetch_add(1, Ordering::Relaxed);
        self.waiting.fetch_add(1, Ordering::Relaxed);
    }

    /// Undo `request_enqueued` when the request could not be queued.
    pub fn request_rejected(&self) {
        self.requests.fetch_sub(1, Ordering::Relaxed);
        self.waiting.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn request_started(&self) {
        self.waiting.fetch_sub(1, Ordering::Relaxed);
        self.running.fetch_add(1, Ordering::Relaxed);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn request_finished(
        &self,
        reason: &str,
        prompt_tokens: usize,
        completion_tokens: usize,
        cached_tokens: usize,
        prefill: Duration,
        decode: Duration,
        ttft: Duration,
    ) {
        self.running.fetch_sub(1, Ordering::Relaxed);
        self.prompt_tokens.fetch_add(prompt_tokens as u64, Ordering::Relaxed);
        self.gen_tokens.fetch_add(completion_tokens as u64, Ordering::Relaxed);
        self.cached_tokens.fetch_add(cached_tokens as u64, Ordering::Relaxed);
        let mut g = self.inner.lock().unwrap();
        *g.finished.entry(reason.to_string()).or_insert(0) += 1;
        let computed = prompt_tokens.saturating_sub(cached_tokens) as f64;
        if prefill.as_secs_f64() > 0.0 && computed > 0.0 {
            g.last_prefill_tok_s = computed / prefill.as_secs_f64();
        }
        // the first completion token comes from prefill; decode steps = completion - 1
        let steps = completion_tokens.saturating_sub(1) as f64;
        if decode.as_secs_f64() > 0.0 && steps > 0.0 {
            g.last_decode_tok_s = steps / decode.as_secs_f64();
        }
        g.last_ttft_ms = ttft.as_secs_f64() * 1e3;
    }

    pub fn set_kv(&self, used_bytes: u64, total_bytes: u64, used_tokens: u64, cap_tokens: u64) {
        self.kv_used_bytes.store(used_bytes, Ordering::Relaxed);
        self.kv_total_bytes.store(total_bytes, Ordering::Relaxed);
        self.kv_used_tokens.store(used_tokens, Ordering::Relaxed);
        self.kv_cap_tokens.store(cap_tokens, Ordering::Relaxed);
    }

    /// Count an op launch; pass `Some(duration)` when it was timed.
    pub fn record_op(&self, op: Op, dur: Option<Duration>) {
        let mut g = self.inner.lock().unwrap();
        let s = g.ops.entry(op).or_default();
        s.calls += 1;
        if let Some(d) = dur {
            s.timed_calls += 1;
            s.time_ms += d.as_secs_f64() * 1e3;
        }
    }

    pub fn snapshot(&self) -> TelemetrySnapshot {
        let probe = self.device_probe.lock().unwrap();
        let live = probe.as_ref().map(|f| f());
        drop(probe);
        let g = self.inner.lock().unwrap();
        let dev = live.or_else(|| g.device.clone());
        TelemetrySnapshot {
            schema_version: TELEMETRY_SCHEMA_VERSION,
            timestamp_ms: now_ms(),
            uptime_s: self.start.elapsed().as_secs_f64(),
            model: g.model.clone(),
            device: dev
                .map(|d| DeviceTelemetry {
                    name: d.name,
                    backend: d.backend,
                    arch: d.arch,
                    total_mem_bytes: d.total_mem,
                    free_mem_bytes: d.free_mem,
                    unified: d.unified,
                })
                .unwrap_or_default(),
            steps_total: self.steps.load(Ordering::Relaxed),
            last_step_ms: self.last_step_ms.load(Ordering::Relaxed),
            requests_total: self.requests.load(Ordering::Relaxed),
            running: self.running.load(Ordering::Relaxed) as u32,
            waiting: self.waiting.load(Ordering::Relaxed) as u32,
            finished: g.finished.clone(),
            prompt_tokens_total: self.prompt_tokens.load(Ordering::Relaxed),
            generation_tokens_total: self.gen_tokens.load(Ordering::Relaxed),
            cached_tokens_total: self.cached_tokens.load(Ordering::Relaxed),
            last_prefill_tok_s: g.last_prefill_tok_s,
            last_decode_tok_s: g.last_decode_tok_s,
            last_ttft_ms: g.last_ttft_ms,
            kv_used_bytes: self.kv_used_bytes.load(Ordering::Relaxed),
            kv_total_bytes: self.kv_total_bytes.load(Ordering::Relaxed),
            kv_used_tokens: self.kv_used_tokens.load(Ordering::Relaxed),
            kv_capacity_tokens: self.kv_cap_tokens.load(Ordering::Relaxed),
            weights_bytes: self.weights_bytes.load(Ordering::Relaxed),
            graph_replays: self.graph_replays.load(Ordering::Relaxed),
            graph_captures: self.graph_captures.load(Ordering::Relaxed),
            ops: g.ops.iter().map(|(k, v)| (k.as_str().to_string(), v.clone())).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters() {
        let t = Telemetry::new();
        t.request_enqueued();
        let s = t.snapshot();
        assert_eq!((s.requests_total, s.waiting, s.running), (1, 1, 0));
        t.request_started();
        t.step();
        t.step();
        t.record_op(Op::Matmul, Some(Duration::from_millis(3)));
        t.record_op(Op::Matmul, None);
        t.request_finished(
            "stop",
            10,
            5,
            2,
            Duration::from_millis(100),
            Duration::from_millis(400),
            Duration::from_millis(100),
        );
        let s = t.snapshot();
        assert_eq!((s.steps_total, s.running, s.waiting), (2, 0, 0));
        assert_eq!(s.prompt_tokens_total, 10);
        assert_eq!(s.cached_tokens_total, 2);
        assert_eq!(s.ops["matmul"].calls, 2);
        assert_eq!(s.ops["matmul"].timed_calls, 1);
        assert!((s.last_prefill_tok_s - 80.0).abs() < 1e-6);
        assert!((s.last_decode_tok_s - 10.0).abs() < 1e-6);
        assert_eq!(s.finished["stop"], 1);
        assert!(serde_json::to_string(&s).unwrap().contains("steps_total"));
    }
}
