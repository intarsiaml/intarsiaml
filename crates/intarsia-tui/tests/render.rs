//! Headless render tests: build an `App` from fixtures (a synthetic hybrid-MoE plan; real server output in `live_*`),
//! draw it into a `TestBackend` and compare the text with a golden file. Regenerate with `UPDATE_SNAPSHOT=1 cargo test -p
//! intarsia-tui --test render`.

use intarsia_bench::snapshot::{Device, OpStat, Snapshot};
use intarsia_tui::app::App;
use intarsia_tui::chat::ChatEvent;
use intarsia_tui::layout::Plan;
use intarsia_tui::ui::{render_text, Theme};
use std::path::PathBuf;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests")
}

fn fixture(name: &str) -> String {
    std::fs::read_to_string(dir().join("fixtures").join(name)).unwrap_or_else(|e| panic!("fixture {name}: {e}"))
}

fn check_golden(name: &str, text: &str) {
    let path = dir().join("golden").join(name);
    if std::env::var_os("UPDATE_SNAPSHOT").is_some() {
        std::fs::write(&path, text).unwrap();
        return;
    }
    let want = std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("missing golden {name}; run with UPDATE_SNAPSHOT=1"));
    assert_eq!(text, want, "snapshot {name} differs (UPDATE_SNAPSHOT=1 to accept)");
}

fn snap(t: f64) -> Snapshot {
    // a decoding server: counters grow with t, KV filling, op timing on
    let n = (t * 20.0) as u64;
    let mut s = Snapshot {
        schema_version: 1,
        model: "Qwen/Qwen3.5-35B-A3B".into(),
        uptime_s: 100.0 + t,
        device: Device { name: "Apple M5 Max".into(), backend: "metal".into(), arch: "g17".into(), total_mem_bytes: 137_438_953_472, free_mem_bytes: 60_000_000_000, unified: true },
        steps_total: 1000 + n,
        requests_total: 7,
        running: 1,
        waiting: 2,
        prompt_tokens_total: 912,
        cached_tokens_total: 128,
        generation_tokens_total: 640 + n,
        last_prefill_tok_s: 850.2,
        last_decode_tok_s: 24.1,
        last_ttft_ms: 140.5,
        kv_used_tokens: 284,
        kv_capacity_tokens: 4096,
        kv_used_bytes: 41_943_040,
        kv_total_bytes: 603_979_776,
        weights_bytes: 20_000_000_000,
        ..Default::default()
    };
    s.finished.insert("stop".into(), 5);
    s.finished.insert("length".into(), 1);
    for (name, calls, ms) in [
        ("matmul", 40320u64, 3120.5),
        ("moe_experts", 1600, 900.0),
        ("gdn_scan", 1200, 500.0),
        ("attn_decode", 400, 300.0),
        ("rmsnorm", 3240, 80.0),
        ("logits", 40, 120.0),
        ("argmax", 40, 5.0),
        ("kv_append", 400, 20.0),
    ] {
        s.ops.insert(name.into(), OpStat { calls, timed_calls: calls / 16, time_ms: ms });
    }
    s
}

const METRICS: &str = "intarsia_batch_iterations_total 90\nintarsia_batch_queue_depth 2\nintarsia_batch_queue_depth_max 3\n\
intarsia_batch_step_ewma_ms 31.5\nintarsia_batch_decode_rows_total{rows=\"1\"} 40\nintarsia_batch_decode_rows_total{rows=\"2\"} 50\n\
intarsia_ttft_seconds_bucket{le=\"0.1\"} 0\nintarsia_ttft_seconds_bucket{le=\"0.25\"} 5\nintarsia_ttft_seconds_bucket{le=\"1\"} 7\n\
intarsia_ttft_seconds_bucket{le=\"+Inf\"} 7\nintarsia_ttft_seconds_sum 1.2\nintarsia_ttft_seconds_count 7\n\
intarsia_itl_seconds_bucket{le=\"0.025\"} 100\nintarsia_itl_seconds_bucket{le=\"0.05\"} 600\nintarsia_itl_seconds_bucket{le=\"0.1\"} 640\n\
intarsia_itl_seconds_bucket{le=\"+Inf\"} 640\nintarsia_itl_seconds_sum 25\nintarsia_itl_seconds_count 640\n";

fn app() -> App {
    let mut a = App::new("http://127.0.0.1:8090");
    a.ingest_plan(Plan::from_json(&serde_json::from_str(&fixture("plan_hybrid.json")).unwrap()));
    for i in 0..12 {
        a.ingest_snapshot(snap(i as f64 * 0.2), i as f64 * 0.2);
    }
    a.ingest_metrics(METRICS, 2.4);
    a.chat_model = "Qwen/Qwen3.5-35B-A3B".into();
    a.input = "why is the sky blue".into();
    a.submit().unwrap();
    for t in ["Because ", "of Rayleigh ", "scattering."] {
        a.chat_event(ChatEvent::Token(t.into()));
    }
    a.input = "and sunsets?".into();
    a
}

#[test]
fn hybrid_plan_golden_140x45() {
    let text = render_text(&app(), &Theme::dark(), 140, 45);
    for needle in ["request queue / scheduler", "prefill", "decode", "[A] x30", "[B] x10", "linear_attention > moe", "KV cache", "sampler", "output", "chat", "TTFT", "ITL", "matmul"] {
        assert!(text.contains(needle), "missing {needle:?} in:\n{text}");
    }
    // live values, none of them zero/placeholder
    assert!(text.contains("running 1 · waiting 2"), "{text}");
    assert!(text.contains("284 / 4096 tok (6.9%)"), "{text}");
    assert!(text.contains("batch size 1.6") || text.contains("batch size"), "{text}");
    check_golden("hybrid_140x45.txt", &text);
}

#[test]
fn profiling_off_reads_na_not_zero() {
    let mut a = app();
    let mut s = snap(3.0);
    for o in s.ops.values_mut() {
        o.timed_calls = 0;
        o.time_ms = 0.0;
    }
    a.ingest_snapshot(s, 3.0);
    let text = render_text(&a, &Theme::dark(), 140, 45);
    assert!(text.contains("time share n/a: server op profiling is off"), "{text}");
    assert!(text.contains("matmul") && text.contains("n/a"), "{text}");
}

#[test]
fn no_snapshot_yet_shows_waiting() {
    let a = App::new("http://127.0.0.1:1");
    let text = render_text(&a, &Theme::dark(), 120, 40);
    assert!(text.contains("waiting for plan"), "{text}");
    assert!(text.contains("waiting for http://127.0.0.1:1/events"), "{text}");
}

#[test]
fn narrow_terminal_stacks_panes() {
    let text = render_text(&app(), &Theme::dark(), 80, 60);
    assert!(text.contains("metrics") && text.contains("chat") && text.contains("architecture"), "{text}");
}

#[test]
fn light_theme_renders_the_same_text() {
    let a = app();
    assert_eq!(render_text(&a, &Theme::dark(), 140, 45), render_text(&a, &Theme::light(), 140, 45));
}
