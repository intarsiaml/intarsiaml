# Dashboard and TUI on CUDA (HIP not run), 2026-10-22

## Headline

The Grafana dashboard and `intarsia tui` work against a CUDA server (spark1, GB10, Qwen3-8B, single sequence, `--profile-every 16`).
The M4 query check (67 dashboard queries) gives 48 with data, 19 empty, 0 errors. All 19 empty queries are the "Batch scheduler" row,
which only the Metal batch engine (`--max-batch N`) feeds; CUDA rejects `--max-batch` ("needs the Metal backend"). They are empty by
design, so no metric was added. The row is now labelled "(Metal, --max-batch N)" and its panels show "Batch engine only" instead of a blank.
**HIP was not run**: both R9700s hold radiance (31 of 32 GB each), `fleetq when --host tatooine --mem 20` says 1 GB free, and the brief says not to
stop a fleet role. The 7 non-batch rows are backend-independent server code, so the CUDA result is the best available evidence for HIP, but it is not a HIP run.

## Query pass counts

| Backend | Queries | With data | Empty | Errors | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Metal (M4, 2026-10-21, `--max-batch 4`) | 67 | 67 | 0 | 0 | |
| CUDA (spark1, this run) | 67 | 48 | 19 | 0 | empty = Batch scheduler row (7 panels), by design |
| HIP (tatooine) | 67 | not run | | | needs a radiance window |

Everything else has data on CUDA, including device memory (total 130.7 GB, free, weights 16.4 GB), KV bytes and tokens, and the kernel
share panels (13 ops with `--profile-every 16`; CUDA serving defaults to 0, which would leave those three panels empty, as on Metal).
All 10 alert rules loaded and evaluate healthy (none firing). Raw: `raw/2026-10-22-ob/cuda/{qcheck.txt,metrics.txt,rules.json,targets.json}`.

## Method

fleetq #3579 (spark1, non-exclusive, 24 GB, vLLM left up). `scripts/deploy.sh spark1 --backend cuda` built the branch, then
`serve --backend cuda --max-seq 8192 --profile-every 16 --mem-budget 24 --no-ui` on port 18612. Load: `raw/2026-10-22-ob/load.py` (mixed
prompts 58 to 3766 tokens, four client names), then `localeval speed` (sizes 128/1024/4096, gen 64, 1 rep, concurrency 2); plumbing test,
no performance claim (the first request TTFT was 12.6 s cold). An ephemeral Prometheus v3.5.0 on endor (docker, port 19099, removed
afterwards) scraped the server; `qcheck.py` is the M4 script unchanged. No scrape job was added on `obs` (O8 stays with the user).
A throwaway compose stack (Prometheus 19090, Grafana 13000) confirmed the new `noValue` text renders: `docs/assets/observability-dashboard-cuda.png`.

## TUI on CUDA

`intarsia tui --snapshot --chat ...` ran on spark1 against the live server (default dark theme; `--snapshot` frames are text, so no colours are
stored). It shows queue, layer stack, GB10 device, kernel shares (matmul 74 to 76 %), KV 65 / 8192 tokens, chat reply, and the single-sequence
`~` step-rate fallback. Frames: `raw/2026-10-22-ob/cuda/tui-snapshot.txt`; the reply-finished frame is
`tui-golden-reply-finished.txt`. No TUI change was needed.

## architecture.svg

It had no dark/light handling: a fixed dark background and light text. It now carries a `prefers-color-scheme: light` block (light
background, darker text, node and accent colours); dark stays the default and is unchanged. Checked in a browser in both schemes.

## Files and checks

Changed: `docs/assets/architecture.svg`, `deploy/observability/grafana/dashboards/intarsia.json` (row title, `noValue`, descriptions; query
expressions untouched, the check above uses this file's targets), `docs/observability.md`, `docs/decisions/LOG.md`, new docs/assets png and raw
files. No Rust, kernel or default changed, so no cargo test run was needed. Grafana loaded the edited JSON without errors.

## Downtime and cleanup

None. vLLM and radiance untouched. Server, Prometheus container and compose stack removed; slot #3579 released. One near-miss: my
`pkill -f "intarsia serve"` on spark1 could have matched another lane's server; none was running (the other job was `intarsia accuracy`).

## Not done

HIP leg (see headline). A runnable script for it would reuse `run-cuda.sh` with `scripts/deploy.sh tatooine --backend hip` inside a
radiance window under the downtime caps.
