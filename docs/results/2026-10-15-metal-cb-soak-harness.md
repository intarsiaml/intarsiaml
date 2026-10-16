# Metal continuous batching: soak / load harness (`tools/cbsoak.py`)

Date 2026-10-15. Branch `metal-cb-soak` off `spike/metal` 28e5c5a. Tools only (`tools/cbsoak.py`, `tools/cbsoak_serve.sh`), no change under `crates/`. Purpose: the load and soak gates of M2 (`docs/research/continuous-batching-design.md` section 7: p95 inter-token latency under a prefill arrival, steady-state ITL, TTFT against solo, no starvation) and the leak check for any `--max-batch 2..8` build. It is a sibling of `tools/cbidentity.py` (exactness gate); this one measures latency, shedding and memory.

## Result

* `uv run tools/cbsoak.py selftest` passes (68 checks, about 50 s, no GPU, no network beyond localhost).
* End to end against today's `intarsia serve` (max_batch 1, Qwen3-8B BF16, Metal, M5 Max): two runs inside one exclusive fleetq slot (#2107), 105 + 47 requests, 0 errors, 0 shed, 5 client cancellations handled, memory flat. Numbers below.
* One unexplained observation: the second run decoded 1.8x slower than the first (section "E2E"). Not a harness effect (the server's own ITL histogram agrees). The proposed SLOs use run 1, which matches the known 36 tok/s single-stream rate.

## Usage

```
uv run tools/cbsoak.py run --url http://127.0.0.1:8099 --rate 0.5 --duration 180 --pid auto --out DIR
uv run tools/cbsoak.py run --url U --rate 4 --duration 1800 --cancel-frac 0.1 --pid 12345 \
    --max-itl-p95-ms 150 --max-error-rate 0.001 --max-mem-growth-mb 100 --out DIR
uv run tools/cbsoak.py report --raw DIR/raw.json [SLO flags]     # re-render or re-judge a finished run
uv run tools/cbsoak.py selftest
```

Server start plus runs in one fleetq slot (the wrapper takes no slot itself; run it as the slot's command, it only ever signals the server PID it started; several runs separated by `--then` share one server start):

```
fleetq run --host endor --exclusive --mem 24 --label "cbsoak ..." -- \
    tools/cbsoak_serve.sh OUTDIR --rate 0.5 --duration 180 [--then --rate 0.2 ...]
# env: INTARSIA_BIN, MODEL (~/models/Qwen3-8B), PORT (8107), SERVE_EXTRA="--max-batch 4"
```

Exit codes: 0 pass, 1 an SLO threshold violated (or nothing measurable for a threshold that was given), 2 harness failure (server unreachable, bad arguments).

Output in `--out`: `raw.json` (everything needed to re-render), `requests.jsonl` (one record per request without the ITL arrays), `report.json`, `report.md`.

### Load model

* **Open loop.** Arrival times are a Poisson process (`--rate` req/s, exponential gaps, `--seed`), precomputed, deterministic in the seed. A new thread per arrival; the schedule never waits for responses, so an overloaded server shows up as queueing and growing TTFT (and shedding once its queue is full), not as a slower client. `--max-inflight` (default 512) is only a client safety cap; arrivals above it are counted as `client_dropped` and mean the run is no longer open loop.
* **Prompt mix** `--prompt-mix tokens:weight,...` (default `128:2,512:3,1024:3,2048:2,4096:1`, +-10% jitter) and **max_tokens mix** `--max-tokens-mix` (default `32:2,128:4,256:2`). Prompts are raw slices of `tools/prompts/corpus_docs.txt` / `corpus_code.txt` (`/v1/completions`), and for targets >= 3000 tokens a share `--golden-frac` (0.25) comes from `golden_4k.jsonl` (the two raw cases and the chat case, `/v1/chat/completions`). The harness has no tokenizer: it sizes prompts in characters through a chars-per-token ratio measured per corpus with two 1-token calibration requests (here 3.37 docs, 4.15 code). The achieved `prompt_tokens` (from `usage`) is reported per bucket. Every prompt gets a unique nonce prefix (`--no-nonce` to allow prefix-cache reuse), `ignore_eos` forces the generated length (`--no-ignore-eos` to let it stop), temperature 0.
* **Warmup**: two unmeasured requests (smallest and largest bucket) before the first arrival, so first-use pipeline compilation is not in the numbers (`--no-warmup`).
* **Cancellations** `--cancel-frac F`: that share of requests is abandoned by the client at a uniform random time in `[--cancel-min-ms, --cancel-max-ms]` after sending (socket shut down mid-stream), whether or not the first token has arrived.
* All requests stream (`stream_options.include_usage`).

### Sampling during the run

* `--pid N` (or `auto`: `lsof` on the URL port): `footprint -p N -f bytes` (macOS physical footprint) and `ps -o rss=` every `--sample-s` (5 s). The footprint is the leak series when available, else RSS.
* `/metrics` is scraped each sample and the `intarsia_*` / `vllm:*` unlabeled series plus the labeled `intarsia_batch_decode_rows_total{rows}` are kept; `/v1/intarsia/plan` is read at start and end. A server without `/metrics` (404, refused) just yields an "unavailable" section.

## Metric definitions

| metric | definition |
|---|---|
| sent | requests dispatched (planned minus `client_dropped`) |
| ok / cancelled / shed / error | final state. **shed** = HTTP 429 or 503 on the initial response (counted separately as `shed_429`, `shed_503`). **error** = any other non-200, a read timeout (`--timeout`, default 300 s per read; `--request-timeout` 900 s total), a connection error, an error event, a stream that ends without `[DONE]`, or a request still running after `--drain-s`. **cancelled** = abandoned by this client on purpose. sent = ok + cancelled + shed + error |
| shed rate, error rate | shed / sent, error / sent (cancelled requests are neither) |
| TTFT | ms from just before connect to the first stream chunk with non-empty content. Includes queueing in the server. Cancelled requests that got a first chunk contribute |
| ITL | ms between consecutive content chunks of one request (pooled over all requests). One chunk is one token only if the server sends one token per chunk; `chunks per completion token` in the report checks that (1.000 here). A held-back token (stop-string or UTF-8 buffering) shows up as one long gap |
| solo TTFT | TTFT of requests that found no other request open on the client when they were sent (queue-free reference, per prompt bucket, with its n) |
| per-stream decode tok/s | per request `(chunks - 1) / sum(ITL)` |
| output tok/s | completion tokens (usage, else chunks) of ok and cancelled requests / seconds from the first send to the last completion; completed req/s likewise for ok requests |
| window table | `--window-s` (30 s) windows. Requests by send time (sent/ok/shed/err/canc, TTFT); ITL and tok/s by chunk arrival time; queue max from `/metrics` samples; process memory at the window end. Use it to see degradation over time |
| memory start/end/max | first / last / maximum sample |
| growth slope | Theil-Sen slope (median of pairwise slopes, robust to allocator spikes; OLS and r2 reported next to it) of the samples with `t >= --mem-warmup-s` (default 20% of the duration), in MB/min; `growth_mb` = slope x fitted span. **leak suspect** = growth >= 100 MB and r2 >= 0.6. The `--max-mem-growth-mb` SLO uses `growth_mb`. A plateau reached during warmup is ignored by construction |
| device memory | `intarsia_device_memory_total_bytes - ..._free_bytes` from `/metrics`, same analysis, SLO `--max-device-mem-growth-mb` |
| batch-size histogram | delta over the run of `intarsia_batch_decode_rows_total{rows=N}` (decode passes by number of rows; present once the scheduler is wired), mean rows per pass |

SLO flags (all optional): `--max-ttft-p50/p95/p99-ms`, `--max-itl-p50/p95/p99-ms`, `--max-error-rate`, `--max-shed-rate`, `--max-mem-growth-mb`, `--max-device-mem-growth-mb`, `--min-out-tok-s`, `--min-req-s`.

## What the selftest covers

In-process mock streaming server (completions and chat shapes, configurable first-token and inter-token delay, concurrency cap with 429 or 503, `/metrics`, optional slowdown after N seconds, optional device-memory leak): percentile math (`1..100` gives 50.5 / 95.05 / 99.01), Poisson schedule statistics (count, mean gap, gap CV, P(gap > mean), per-10 s dispersion, determinism), mix weights and plan clipping, 429 and 503 shedding accounting (client count equals the server's, never an error, cap respected, `--max-shed-rate` fails), cancellations (all-cancelled run, server sees the disconnects, 50% mix), leak-slope detection on synthetic series (+10 MB/min with spikes recovered within 10%; flat noisy series and a warmup plateau not flagged), SLO evaluation including fail-closed on unmeasurable metrics, window degradation detection (4x slowdown), `/metrics` absent, refused connection, read timeout, warmup accounting, `footprint`/`ps` parsing and a live `ps`/`footprint` read of the test process, CLI exit codes 0/1/2 and output files.

## E2E on `intarsia serve` (max_batch 1)

Binary: build of branch `metal-forward-no1` d3c4034 (28e5c5a plus one change to `Model::forward`, not on the serve path), copied to a pinned path. `serve --backend metal --max-seq 8192 --profile-every 0 --no-supervisor --no-ui`, Qwen3-8B BF16, endor (M5 Max). fleetq job #2107 (`fleetq run --host endor --exclusive --mem 24`), waited in line behind #2103-#2106, ran 8 min, exit 0, server stopped by the script (PID it started), no slot left. Raw: `docs/results/raw/2026-10-15-metal-cb-soak/` (`run1-0.5rps/`, `run2-baseline/`, `cbsoak_serve.log`, `server.log`, `metrics-end.txt`).

**Run 1**: 180 s, 0.5 req/s offered (Poisson, 105 sent = 0.58/s realised), `--max-tokens-mix 16:2,48:3,96:1`, `--cancel-frac 0.1`, `--sample-s 3`. The server was about 90% busy (mean running 0.90), queue depth up to 5.

| | |
|---|---|
| sent / ok / cancelled / shed / error | 105 / 100 / 5 / 0 / 0 |
| completed | 0.557 req/s, 25.6 output tok/s |
| TTFT ms (includes queueing) | p50 3013, p95 8445, p99 9678 |
| ITL ms | p50 27.8, p95 29.3, p99 30.6, max 67 |
| per-stream decode | 35.6 tok/s mean (36.5 p95) |
| process footprint (MB) | start 1691, end 1694, max 1694; slope 1.14 MB/min, fitted growth 3 MB; not a leak |
| device memory per `/metrics` | constant 17043 MB (weights 15.3 GiB, preallocated KV 1.1 GiB, other buffers); KV used max 0.67 GB |
| server-side TTFT / ITL mean | 370 ms / 28.0 ms (the server's TTFT excludes time spent waiting in its queue; the client's does) |

Per-bucket queue-free (solo) TTFT p50: 128 tokens 240 ms (n 3), 512: 297 (2), 1024: 334 (3), 2048: 592 (3), 4096: 1876 (1). Windows 0-5 show ITL p95 28.4-30.5 ms with no drift over 3 minutes; the TTFT swings with the queue (846 to 5476 ms window p50), which is what a single-slot server does at 90% load. The 5 cancellations did not disturb the following requests.

**Run 2** (same slot, same server, started right after): 200 s, 0.2 req/s, equal-weight buckets 128..4096, `--max-tokens-mix 16:1,48:1`, 47 requests, all ok. ITL p50 50.2, p95 91.8 ms (16.8 tok/s per stream, server-side ITL mean 57.5 ms agrees), solo TTFT p50 higher too (1024: 590 ms, 2048: 1037 ms) while 4096 solo was 1910 ms (n 6). So the whole GPU ran about 1.8x slower in run 2, decode more than prefill, from the first request after the first one (the t=3 s request ran at 27.9 ms per token). Not explained: no macmon trace was taken, and the slot was exclusive in fleetq terms but fleetq does not stop untracked GPU users or thermal/clock states (the overnight report documents clock drops). It does not look like a harness artifact: client and server ITL agree, the same client code produced run 1. Not rerun. Treat run 2's latencies as not a baseline; its memory (1694 to 1695 MB, 0.35 MB/min) and accounting (47/47 ok) are valid.

## Proposed M2 SLO thresholds

Derived from run 1 (single stream, no batching: ITL p50 27.8 / p95 29.3 / p99 30.6 ms; 36 tok/s) and from the M2 gate text in the design (p95 ITL <= 150 ms of running streams while a prompt prefills at chunk 256; steady-state p95 ITL <= 1.15 x the step time; arriving request TTFT <= 2.5 x solo; no starvation; section 8.1 FIFO and chunk 256 under load).

1. **Regression guard at `--max-batch 1`** (the new scheduler path must not slow the single stream; offered load below capacity, e.g. 0.2 req/s with the default mix, 10 min): `--max-itl-p50-ms 30 --max-itl-p95-ms 32 --max-itl-p99-ms 40` (about 1.1x the measured values; the design's 1.15 x step), `--max-error-rate 0 --max-shed-rate 0`.
2. **Mixed-load soak at `--max-batch 4` and `8`, offered rate below the saturation rate**, `--cancel-frac 0.1`, 30 min: `--max-itl-p95-ms 150` (the design's M2 ceiling, which already includes the 256-token prefill chunk stall), and a steady-state ITL check from a run without arriving long prompts at 1.15 x the step time that M1 measured at that B (design estimate: about 33 ms at B 7 and 1k context, so about 40 ms at B 7-8). `--max-ttft-p50-ms 1500 --max-ttft-p95-ms 5000 --max-ttft-p99-ms 8000` (the mix-weighted solo TTFT p50 is about 0.5 s, 4k prompts alone take 1.9 s, so 2.5 x gives 1.2 s for the average request and 4.7 s for a 4k one), `--max-error-rate 0.001`, `--max-shed-rate 0.01`. Check TTFT per bucket from the report against 2.5 x the solo column rather than only the pooled limits.
3. **Overload / shedding test** (offered rate above capacity, `--max-queue` small): not an SLO pass/fail; expect shed rate > 0 as 429 with `error rate 0`, a bounded ITL p95 for the admitted streams, and a recovery to the baseline in the last windows. Note the scheduler doc's open item: a full queue returns `BadRequest`, which would show up here as errors (HTTP 400) instead of shed until the status code is decided.
4. **Leak**: 30 min or longer after warmup, `--max-mem-growth-mb 100` for the process footprint (measured 3 MB over 2.4 min at max_batch 1) and `--max-device-mem-growth-mb 100`; a `leak suspect: YES` line in the report is a fail even without thresholds.
5. Throughput is judged relative to the M1 gates (aggregate tok/s against the serial baseline), not with an absolute `--min-out-tok-s`. At max_batch 1 the saturation capacity is one request per (prefill + n/36 s): about 1.1 req/s for 16-token answers, 0.25 req/s for the default tokens mix.

## Limits and gaps

* **Process footprint excludes the GPU.** `footprint` shows 1.7 GB for a process whose weights (16.4 GB) and KV are Metal buffers, so it catches host-side leaks only. The device-memory series comes from `/metrics` and is the engine's own accounting (constant 17043 MB here, preallocated), so a leak in unaccounted Metal allocations would be missed by both. System-wide GPU memory (e.g. `vm_stat`/`ioreg` or macmon) is not sampled.
* No GPU clock/thermal trace (the run-2 slowdown is therefore unexplained); no per-step timing (`intarsia_batch_step_ewma_ms` and the prefill/queue series are recorded only if the scheduler exposes them; at max_batch 1 they do not exist).
* ITL is per chunk. If the scheduler or speculation (M3) emits several tokens per chunk, ITL becomes per burst; the chunks-per-token ratio flags that.
* Prompt length is approximate (chars-per-token calibration; achieved `prompt_tokens` is reported). Corpus slices are 68-102 KB, so at most one 4k-token prompt is a large share of a file; slices overlap across requests (the nonce defeats prefix reuse, not repeated content).
* Open loop is not exact at very high rates: one thread per request, Python timing jitter of a few ms, `--max-inflight` cap. For rates above about 50 req/s run several instances.
* The leak slope on a 3-minute run is weak evidence (r2 0.02); the thresholds above assume 10-30 minute runs.
* The TTFT of a golden chat request includes the chat template and the unique system-prefix nonce; with `--no-nonce` the prefix cache applies (design 3.5), which the harness does not model separately.
* Not run: a batched server (the scheduler is on `metal-cb-wire`, not merged into this branch's base), so batch-size histogram, queue-depth and `rows` parsing were exercised on the mock only; `/metrics` series names were taken from `crates/intarsia-server/src/metrics.rs` on `metal-cb-wire` and the real max_batch 1 output. The 429/503 path was exercised on the mock only (the real server's `--max-queue` shedding was not driven).
* `tools/cbidentity.py` `one_request` calls `conn.sock.settimeout` after `getresponse()`; `http.client` detaches the socket once a `Connection: close` response starts, so its millisecond cancels (`cancel_after_ms`) may not fire mid-stream. `cbsoak` keeps its own socket reference. Not touched here (out of scope), worth a look.
