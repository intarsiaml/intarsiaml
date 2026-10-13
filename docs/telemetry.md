# Telemetry

One struct feeds everything that watches the engine: `/metrics` (Prometheus), `/events` (SSE),
the TUI (`intarsia top`) and `intarsia doctor`/`bench`. It is
`intarsia_core::telemetry::TelemetrySnapshot` (serde `Serialize`), obtained from
`Engine::telemetry()`. A snapshot is cheap (atomics plus one short mutex); poll it at 4-10 Hz.

JSON shape (`schema_version` 1; additive changes keep the version, breaking ones bump it):

```json
{
  "schema_version": 1,
  "timestamp_ms": 1760000000000,
  "uptime_s": 12.3,
  "model": "Qwen/Qwen3-8B",
  "device": { "name": "NVIDIA GB10", "backend": "cuda", "arch": "sm_121",
              "total_mem_bytes": 130000000000, "free_mem_bytes": 52000000000, "unified": true },
  "steps_total": 1042,
  "last_step_ms": 1760000000000,
  "requests_total": 7,
  "running": 1,
  "waiting": 0,
  "finished": { "stop": 5, "length": 1, "cancelled": 0, "error": 0 },
  "prompt_tokens_total": 912,
  "generation_tokens_total": 640,
  "cached_tokens_total": 128,
  "last_prefill_tok_s": 850.2,
  "last_decode_tok_s": 24.1,
  "last_ttft_ms": 140.5,
  "kv_used_bytes": 41943040,
  "kv_total_bytes": 603979776,
  "kv_used_tokens": 284,
  "kv_capacity_tokens": 4096,
  "weights_bytes": 16400000000,
  "ops": { "matmul": { "calls": 40320, "timed_calls": 2520, "time_ms": 3120.5 }, "...": {} }
}
```

## Field meanings

| Field | Meaning | Suggested Prometheus name |
|---|---|---|
| `steps_total` | Monotonic count of engine steps. One step is one forward pass: a prefill chunk or a decode step. A watchdog treats "`running + waiting > 0` and `steps_total` not advancing for N seconds" as a wedge (vLLM's cuDNN FP8 hang kept `/metrics` alive while dead). | `intarsia_engine_steps_total` (counter) |
| `last_step_ms` | Unix ms of the latest step, `0` before the first. | gauge |
| `requests_total` | Requests accepted by `Engine::submit` (counted at submit). | `intarsia_requests_total` |
| `running` | Requests executing now (0 or 1 in Phase 0). | `intarsia_requests_running` (gauge) |
| `waiting` | Requests queued behind the running one. | `intarsia_requests_waiting` (gauge) |
| `finished` | Finished requests by reason: `stop`, `length`, `cancelled`, `error`. | `intarsia_requests_finished_total{reason}` |
| `prompt_tokens_total` | Prompt tokens of finished requests (including cached ones). | `intarsia_prompt_tokens_total` |
| `generation_tokens_total` | Completion tokens of finished requests (EOS counted, not emitted). | `intarsia_generation_tokens_total` |
| `cached_tokens_total` | Prompt tokens served from the reused KV prefix. | `intarsia_cached_tokens_total` |
| `last_prefill_tok_s` | Computed prompt tokens / prefill time of the last finished request. | gauge |
| `last_decode_tok_s` | (completion tokens - 1) / decode time of the last finished request. | gauge |
| `last_ttft_ms` | Time to first token of the last finished request. | gauge (the server derives histograms from `GenEvent::Done` per request) |
| `kv_used_bytes` / `kv_total_bytes` | Filled / allocated bytes of the KV cache (attention slots; recurrent state counts fully). | gauges |
| `kv_used_tokens` / `kv_capacity_tokens` | Tokens held / planned context length. | gauges |
| `weights_bytes` | Bytes of loaded weights. | gauge |
| `device.*` | Name, backend, arch, total memory, **free memory measured at snapshot time**, unified-memory flag. | `intarsia_device_free_bytes` |
| `ops.<op>.calls` | Launch count per op id (`embedding`, `rmsnorm`, `rmsnorm_add`, `qk_norm`, `rope`, `matmul`, `logits`, `kv_append`, `attn_prefill`, `attn_decode`, `silu_glu`, `argmax`, ...). | `intarsia_op_calls_total{op}` |
| `ops.<op>.timed_calls`, `ops.<op>.time_ms` | Launches whose duration was measured, and their cumulative milliseconds. Timing needs a device sync, so GPU backends sample (`LoadOptions::profile_every`, default 16 on GPUs; every step on cpu-ref). Estimated per-launch cost is `time_ms / timed_calls`; the share of step time per op is `time_ms` normalised across ops. | `intarsia_op_seconds_total{op}` |

Notes for consumers:

* All counters are monotonic since process start; compute rates from deltas of two snapshots.
* `running`, `waiting` and `steps_total` come from separate atomics, so a snapshot is not a transaction;
  do not assert invariants across them.
* The architecture map (`/events`) uses `ops` for per-kernel time share, `kv_*` for pool fill,
  `device.free_mem_bytes` for memory, and `running/waiting` for the scheduler node.
* Per-request latencies (TTFT, ITL, e2e) are not in the snapshot: take them from
  `GenEvent::Done { prefill_ms, decode_ms, .. }` in the server and feed histograms there.

## Where it is produced

`Telemetry` (`crates/intarsia-core/src/telemetry.rs`) is shared by the model (per-step and per-op
counters) and the engine worker (request lifecycle, token totals, KV fill). Create one with
`Telemetry::new()` and pass it to `Model::load` and `LocalEngine::new` (done by
`intarsia_models::load_engine`). Read it with `engine.telemetry()`.
