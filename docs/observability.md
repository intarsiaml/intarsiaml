# Observability

Everything the engine reports comes from one struct, `TelemetrySnapshot` (`docs/telemetry.md`).
`intarsia serve` exposes it three ways:

| Surface | Use |
|---|---|
| `GET /metrics` | Prometheus text. Scrape it; the Grafana dashboard reads it. |
| `GET /events` | SSE stream of the snapshot as JSON, 4-10 Hz. `intarsia top` reads it. |
| `intarsia doctor` | One-shot host fingerprint and pitfall checks (no server needed). |

## `intarsia top`

```sh
intarsia top --url http://spark-4e77:8090      # live
intarsia top --demo                              # synthetic data, no server
intarsia top --demo --exit-after 3               # smoke test, exits by itself
```

Quit with `q`, `Esc` or `Ctrl-C`. It reconnects automatically if the server restarts (status in the
header). Panels, top to bottom:

* header: model, device `[backend arch]`, uptime, connection state, steps/s, request totals by finish
  reason, token totals, weights, device memory;
* sparklines: decode tok/s and prefill tok/s of the last finished request (these update when a
  request finishes), and live engine steps/s;
* gauges: KV bytes, KV tokens, running vs waiting;
* kernel time share: per-op share of the recent time, with ms per launch. Computed from deltas
  between snapshots, with each op's sampled time scaled up by `calls / timed_calls`, then smoothed;
* architecture strip `API -> scheduler -> engine -> kernels -> device`. Each segment is shaded from
  dark slate through amber to red by live load:

  | Segment | Load |
  |---|---|
  | API | `(running + waiting) / 4` |
  | scheduler | 0.35 when a request runs, 0.6-1.0 as the queue grows |
  | engine | steps/s relative to the recent peak |
  | kernels | estimated kernel busy fraction (falls back to the engine load when ops are not timed) |
  | device | KV cache fill |

Operations are only as live as the data: GPU backends time one launch in `profile_every` (default
16), so the kernel shares are statistical and need a few seconds of load to settle.

## Prometheus

Metric names (see `crates/intarsia-server/src/metrics.rs`): `intarsia_engine_steps_total`,
`intarsia_requests_{running,waiting}`, `intarsia_requests_total`,
`intarsia_requests_finished_total{reason}`, `intarsia_tokens_{prompt,generated,cached}_total`,
`intarsia_ttft_seconds` and `intarsia_itl_seconds` (histograms), `intarsia_kv_bytes_{used,total}`,
`intarsia_kv_tokens_{used,capacity}`, `intarsia_device_memory_{total,free}_bytes`,
`intarsia_weights_bytes`, `intarsia_last_{prefill,decode}_tokens_per_second`,
`intarsia_op_seconds_total{op}`, `intarsia_op_calls_total{op}`, `intarsia_uptime_seconds`,
`intarsia_last_step_timestamp_seconds`, `intarsia_info{model,family,backend,device,...}`.

Scrape setup. The existing Prometheus in `/opt/code/missioncntrl_/deploy/monitoring/prometheus.yml`
has global `scrape_interval: 15s` and one job, `missioncntrl`, that uses a bearer token and
`metrics_path: /api/prometheus`. `intarsia serve` has no auth in Phase 0 and serves the default
`/metrics` path, so add a separate job (also in `deploy/grafana/prometheus-scrape.yml`):

```yaml
scrape_configs:
  # ... existing missioncntrl job unchanged ...
  - job_name: intarsia
    metrics_path: /metrics
    scrape_interval: 5s          # per-job override; the 15s global default also works
    static_configs:
      - targets: ["spark-4e77:8090"]
        labels: { host: spark1 }
```

The Prometheus container must be able to reach spark1:8090 and the server must listen on a
reachable interface (`intarsia serve --host 0.0.0.0`; the default is loopback). It is unauthenticated,
so keep it on the LAN or the tailnet. Check with `curl -s spark-4e77:8090/metrics | head` and, in
Prometheus, `up{job="intarsia"}`.

## Grafana

`deploy/grafana/intarsia.json` is a dark dashboard (uid `intarsia-engine`) with a Prometheus
datasource variable, plus `job` and `instance` filters, so it works with any Prometheus.

* Import: Grafana, Dashboards, New, Import, upload the JSON, pick the datasource; or
* Provision (as the missioncntrl dashboard does): mount `intarsia.json` into
  `/var/lib/grafana/dashboards` and `deploy/grafana/dashboards.yml` into
  `/etc/grafana/provisioning/dashboards/`.

Sections: status (scrape up, running, waiting, last decode/prefill tok/s, KV used, uptime and a
**Wedge** indicator), throughput (tokens/s, engine steps/s), TTFT and ITL percentiles, requests,
memory (KV bytes/tokens, device memory), kernels (time share, op time rate, launch rate).

Wedge is `1` when requests are running or waiting but `rate(intarsia_engine_steps_total[1m])` is
zero. A hung kernel keeps `/metrics` answering, so `up == 1` alone does not mean healthy. Alert on it:

```yaml
- alert: IntarsiaWedged
  expr: (sum(intarsia_requests_running + intarsia_requests_waiting) > 0) and (sum(rate(intarsia_engine_steps_total[1m])) == 0)
  for: 1m
```

TTFT and ITL percentiles use `histogram_quantile` over the server's buckets; ITL is the per-request
mean weighted by token count, not a per-token distribution. Counters advance when a request
finishes, so `tokens/s` panels are smoothed over request length; the sparklines in `intarsia top`
and the "last" gauges show the most recent request.

## `intarsia doctor`

```sh
intarsia doctor                    # checklist (backend auto: cuda if compiled in, else cpu)
intarsia doctor --json --out doctor.json
```

Reports OS, kernel, CPU, RAM, GPU (nvidia-smi name, driver, compute capability; `/proc/driver/nvidia/version`;
nvcc / `/usr/local/cuda/version.json`), the backend's `DeviceInfo`, kernel counts per arch, and (with a
real device) measured host-to-device, device-to-host and device-copy bandwidth over a 256 MiB buffer
(`--bw-mib`, 0 skips; device copy counts read plus write bytes). Checks, each with a status
(`ok`/`warn`/`fail`/`info`) and a fix hint where one exists:

* `sm_121_target`: a compute 12.1 GPU (GB10) must have kernels registered for `sm_121`;
* `cuda_toolkit`: CUDA >= 12.9 is needed to compile for sm_121;
* `transparent_hugepages`: `never` slows large weight loads, `madvise` is recommended;
* `disk_free`: warns under 10% or 20 GB, fails under 5 GB (spark1 runs near 95% full);
* `hip_gpu_max_hw_queues`: informational, for the future HIP backend;
* `serve_port`: port 8090 free (8000 is vLLM on spark1).

The CUDA-specific fields only appear on a binary built with `--features cuda`.
