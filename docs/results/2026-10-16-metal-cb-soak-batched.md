# Metal continuous batching: M2 latency / SLO soak on a real batched server

Date 2026-10-16. Branch `metal-cb-soak2` off `spike/metal` 6d99a6f. Tools, docs and raw results only; nothing under `crates/` changed. Machine endor (Apple M5 Max), Qwen3-8B BF16, `serve --max-batch N`, `--max-seq 8192`, prefix reuse on (default), `--spec off`.

## Status

| item | status |
|---|---|
| A1 scripted scenario (`cbsoak.py scenario`, selftest) | **done**; run at B=8 (7 streams + 4k, 5 reps) and B=4 (3 streams + 4k, 5 reps) |
| A2 Poisson soak, 30 min, 10% cancels, `/metrics` histogram + queue depth | **B=8 done** (0.5 req/s, 1800 s). **B=4 soak not run** (queue, see below) |
| A3 `--max-batch 1` regression guard, 28e5c5a vs 6d99a6f | **done, partial**: one baseline run and one candidate run of 6 min (A/B, not A/B/A) |
| A4 real shedding (429 + Retry-After, rejected counter) | **done** (two bursts) |
| A5 scorecard | below, for what was measured |
| Part B (`--spec ngram` + prefix reuse identity) | **not run** (queue). Script and tool change are ready, untested on a server (section 8) |

Why partial: endor's queue was about 3 h deep behind other agents' slots; the B=4 soak, an A/B/A guard and Part B did not fit before the 11:20 stop, and the coordinator killed the queued runners of #2184, #2185 and #2186 (never started). fleetq jobs of this task: #2172 (B=8 ramp, 5 segments, exit 0), #2183 (B=8 scenario x5 + 30 min soak + shedding bursts, exit 0), #2189 (B=1 A/B + B=4 scenario x5, exit 0). Killed before start: #2184 B=4 scenario + soak, #2185 B=1 A/B/A/B, #2186 Part B. No server or runner of mine is left.

## Results at a glance (SLO scorecard)

Scored against the proposed M2 SLOs of `2026-10-15-metal-cb-soak-harness.md`. Load for rows marked soak: B=8, 0.5 req/s Poisson (897 requests, 1800 s), default prompt mix 128..4096 tokens, default max_tokens mix, 10% cancel plan (6.9% realised). The scenario rows are the exact gate scenario at chunk 256 under load (server default busy chunk).

| SLO | limit | measured | verdict |
|---|---|---|---|
| TTFT p50 (soak B=8) | 1500 ms | 446 ms | pass |
| TTFT p95 | 5000 ms | 1958 ms | pass |
| TTFT p99 | 8000 ms | 2730 ms | pass |
| ITL p95, pooled (soak B=8) | 150 ms | 100.2 ms | pass |
| ITL p95 of gaps overlapping a 3000+ token prefill (soak B=8, 83 prompts, 4019 gaps) | 150 ms | 121.4 ms (p50 104, max 193) | pass |
| same, exact scenario B=8 (7 streams at 1k + 4k at t=5 s, 5 reps, 700 gaps) | 150 ms | p50 125.4, **p95 140.8**, max 148.5; per-rep p95 122..145 | pass (6% margin) |
| same, scenario B=4 (3 streams + 4k, 5 reps, 294 gaps) | 150 ms | p50 111.8, p95 130.8, max 145.1 | pass |
| 4k TTFT under that load / warm solo 4k TTFT (design 2.5x) | 2.5x | B=8 2.17x (2.02x per prompt token), B=4 2.04x | pass |
| error rate | 0.001 | 0 / 897 | pass |
| shed rate (soak, default `--max-queue 64`) | 0.01 | 0 / 897 | pass |
| memory growth (process footprint, Theil-Sen fit after 6 min warmup, 24 min) | 100 MB | 27.5 MB (end minus start -111 MB; slope 1.15 MB/min, r2 0.06) | pass |
| device memory per `/metrics` (total minus free) | 100 MB | 0 MB | pass (engine accounting only, preallocated) |
| `--max-batch 1` same-slot A/B, ITL p50 / p95 candidate / baseline | <= 1.1x | 0.990 / 0.995 (28.0 vs 28.3 ms, 28.8 vs 28.9 ms; 3214 gaps each) | pass (one pair only) |
| overload: burst above `--max-queue` returns 429 + `Retry-After`, counted rejected, no 503 | n/a | 8 of 16 and 16 of 24 got 429 (Retry-After 1), rejected counter delta equal, 0 x 503, 0 errors, follow-up ok, health 0/0 | pass |
| B=4 soak (TTFT, ITL, memory over 30 min) | | not run | **not measured** |
| Part B identity with spec + reuse | | not run | **not measured** |

## Proposed SLO revisions (with evidence)

**Adopted (operator, 2026-10-16):** the user accepted all the revisions below as the M2 SLOs.

1. **Pooled ITL p95 <= 150 ms is too loose to guard anything; add a steady-state limit.** Measured pooled p95 is 100 ms, but the steady scenario windows are far lower: B=8 7 streams 35.7 / 37.4 ms (p50 / p95), 8 rows with the 4k stream decoding 43.9 / 45.4, B=4 29.7 / 30.6 and 32.3 / 33.5. The pooled p95 is made of prefill-overlap gaps (every arriving prompt above one chunk stalls the others, not only 3000+ ones), so it hides a steady-state regression of 2x. Propose: steady ITL p95 <= 60 ms at B<=8 and 1k to 4k context (about 1.3x the measured 45 ms at 8 rows), taken from the scenario before/after windows, and keep the in-prefill gate separate. The design estimate of about 33 ms (1.15x = 40 ms) at 7 rows and 1k context is met (35.7 / 37.4 ms measured), but at 8 rows with a 4k stream decoding the step is 44 ms, so the design number should be stated per row count and context.
2. **In-prefill ITL p95 <= 150 ms (chunk 256) is tight but correct for the exact scenario**: pooled 140.8, max 148.5, per-rep p95 122..145. Any busy chunk above 256 will exceed it (the step time scales with the chunk), so it is a real gate, not noise. The soak-pooled version (121 ms) has more room because the average batch is 2.6 rows, so it must not be used as the gate; use the scenario. Judge it on at least 5 reps pooled, as here.
3. **TTFT limits hold only below about 60% of saturation, say so in the SLO.** Capacity of B=8 on the default mix is about 0.86 req/s (116 tok/s output) from the ramp. At 0.42 req/s TTFT p50/p95/p99 = 535/1763/2482 ms, at 0.5 (soak) 446/1958/2730, at 0.7 req/s planned (0.83 realised) 4164/8683/9887 and the queue reaches 8. So 1500/5000/8000 is consistent with the "<= 0.6 x saturation" operating point, and fails by 3x one step above it. With one 30 min run behind it the p99 limit could be tightened to 4000 ms (1.5x margin) but that is one sample, not proposed as final.
4. **Memory SLO must exclude the first minutes.** The process footprint goes from 1.7 GB (server idle) to 9.6 GB within the first scenario/ramp minutes because the preallocated KV pool (9216 MB) is touched on first use (ramp run 1: 4.0 to 9.8 GB in 100 s, fit "growth" 2821 MB, flagged leak suspect). After that it is flat (2 to 5 MB per 100 s segment in the ramp; 27 MB over 24 min in the soak). The harness' 20% warm-up exclusion handles it for a 30 min run; a short run reports a false leak. The footprint also excludes GPU buffers.
5. **Error rate 0.001 over about 900 requests is zero tolerance** (one error is 0.0011). Fine for a gate; report the count. At saturation the harness itself produces "still running after --drain-s" errors (22 in the 1.8 req/s ramp segment), which are not server errors.

## Details

### Scenario (A1)

`uv run tools/cbsoak.py scenario --url U --streams N --reps 5`: N streams (prompts about 1000 tokens, max_tokens 800, ignore_eos) start together; at t = 5.0 s one about 4000-token prompt (max_tokens 400) is sent. A tiny warm-up request precedes every repetition (the weights are re-wired after about 3 s idle, about 170 ms); every prompt has a fresh nonce at its very start (prefix reuse is on, so a repeated prompt would be served from an idle slot). Windows: before = gaps of the streams after the last stream got its first token and before the 4k request was sent; during = gaps overlapping [4k send, 4k first token]; after = [4k first token + 1 s, + 8 s] with the 4k request still decoding (rows = N + 1). A repetition is invalid (reported, not averaged) if a stream was not decoding before the arrival, did not outlive the after window, or errored; all 10 repetitions were valid. Warm solo reference: 3 requests of the same size alone after a warm-up.

B=8, 7 streams (`soak-b8/soak/scenario.md`): before p50 35.7, p95 37.4, max 74.0 ms (1893 gaps); during p50 125.4, p95 140.8, p99 145.6, max 148.5 ms (700 gaps); after p50 43.9, p95 45.4, max 87.7 ms (6353 gaps). 4k TTFT 1904..2604 ms (p50 2110) against a warm solo 966..1003 ms (p50 970): 2.17x (the loaded 4k prompts averaged 4.4k tokens against 3.8k solo, so per token 2.02x). Server decode passes by rows during the scenario: 7 rows 1864, 8 rows 1995, little else.

B=4, 3 streams (`extra/b4-scenario/soak/scenario.md`): before 29.7 / 30.6 ms, during 111.8 / 130.8 / max 145.1 (294 gaps), after 32.3 / 33.5; 4k TTFT p50 1978 ms, 2.04x solo (969 ms). The literal 7-streams-at-B=4 version was not run: 7 streams cannot decode at once, so B=4 uses B-1 = 3 streams plus the 4k prompt.

### Ramp at B=8 (rate choice; `ramp8/`, job #2172)

100 s segments, 10% cancel plan, default mix, seeds 11..15, one server:

| planned rps | sent | completed rps | out tok/s | TTFT p50 / p95 / p99 ms | ITL p50 / p95 ms | queue max | mean rows | shed / err |
|---|---|---|---|---|---|---|---|---|
| 0.4 | 42 | 0.376 | 59 | 535 / 1763 / 2482 | 31.4 / 110 | 1 | 2.57 | 0 / 0 |
| 0.7 | 83 | 0.665 | 95 | 4164 / 8683 / 9887 | 45.3 / 133 | 8 | 5.71 | 0 / 0 |
| 1.0 | 74 | 0.633 | 91 | 2824 / 9716 / 10923 | 44.2 / 130 | 8 | 4.95 | 0 / 0 |
| 1.4 | 128 | 0.861 | 116 | 13983 / 23748 / 24008 | 42.6 / 124 | 23 | 6.93 | 0 / 0 |
| 1.8 | 189 | 0.827 | 111 | 37018 / 67464 / 70748 | 43.5 / 127 | 64 | 7.56 | 17 (429) / 22 (drain artefacts) |

Saturation about 0.85 req/s (about 115 tok/s); the soak rate 0.5 req/s is 58% of it. The 1.8 req/s segment filled the default queue of 64 and shed 17 with 429 (no 503). The first segment's footprint growth (2821 MB) is the KV first-touch described above.

### Soak at B=8, 0.5 req/s, 1800 s (`soak-b8/soak2/`, job #2183)

897 sent, 835 ok, 62 cancelled (6.9%), 0 shed, 0 error, 0.463 req/s completed, 63.8 tok/s. TTFT p50/p95/p99/max 446 / 1958 / 2730 / 4337 ms; ITL p50/p95/p99/max 30.1 / 100.2 / 113.9 / 243 ms (114178 gaps, 1.000 chunks per token); per-stream decode 27.8 tok/s mean. Per-window (60 s) ITL p95 28..114 ms and TTFT p95 up to 3291 ms, no drift over 30 min. Per bucket TTFT p50 (prompt tokens): 128: 110 ms, 512: 288, 1024: 485, 2048: 891, 4096: 1748 ms (warm solo from the scenario: 4k 970 ms; the soak's own "solo" column is mostly cold-start arrivals, 4096: 1175 ms n 11). Cold-start TTFT (arrival after more than 3 s with nothing open, 22 requests, includes the weight re-wire): p50 379, p95 1064 ms against 460 / 1960 ms for all others: the re-wire is not visible in the tails at this load (cold arrivals are low-load arrivals; the prompt mix is not matched, so this is not an isolated cost).

Server side (`/metrics`, real batched server, first time): batch-size histogram, decode passes by rows (delta): 1: 12609 (28.6%), 2: 12561 (28.5%), 3: 8370 (19.0%), 4: 5166 (11.7%), 5: 2866 (6.5%), 6: 1606 (3.6%), 7: 629 (1.4%), 8: 294 (0.7%), mean 2.59 rows. At this load the batch is rarely wide: the soak says little about 7-8 row behaviour, the scenario does. `intarsia_batch_queue_depth` sampled every 5 s: max 1 (mean 0.05), server-reported `queue_depth_max` 6; running max 8, mean 2.67; step EWMA mean 40 ms, max 113; KV used max 2.74 GB of 9.66 GB; 4135 prefill chunks; rejected 0. The parser (`batch_rows:N`, queue gauges) worked on the real endpoint (it had only been tested on a mock).

GPU clock (macmon, 1 s): mean 1.26 / 1.34 / 1.47 / 1.34 GHz over the four 10-minute bins of the B=8 slot, min bin mean no lower than 1.25 GHz, GPU power mean 34.6 W, utilisation mean 81%, max GPU temperature 99 C in the first bin and 86 C in the last. No clock collapse like the 1.8x slowdown of the 2026-10-15 run 2.

### `--max-batch 1` regression guard (job #2189, `extra/b1-*`, `extra/b1-ab.md`)

Same slot, same seed 31, 0.3 req/s for 360 s, `--max-tokens-mix 16:2,48:3,96:1`, default prompt mix, no `--max-batch` flag (1), 80 requests each: baseline `intarsia-base28e5c5a` (28e5c5a, the M1-era binary, rebuilt from that commit) then candidate `intarsia-head` (6d99a6f).

| run | requests | ITL n | ITL p50 | ITL p95 | ITL p99 | TTFT p50 | TTFT p95 | decode tok/s |
|---|---|---|---|---|---|---|---|---|
| 28e5c5a | 80 | 3214 | 28.3 | 28.9 | 29.3 | 433 | 2500 | 35.4 |
| 6d99a6f | 80 | 3214 | 28.0 | 28.8 | 29.4 | 417 | 2470 | 35.7 |

Ratio 0.990 (p50), 0.995 (p95), limit 1.1: pass. The GPU state was the same in both runs (macmon mean util 23% / 23%, mean power 7.7 / 7.8 W). Limits: it is one A and one B (not A/B/A), so slot drift is not bounded by an A' run; ITL of this server is so flat (p95 within 2% of p50) that the 1.1x guard has a lot of headroom. The M1-era healthy single-stream numbers (27.8 / 29.3 ms, 36 tok/s) are reproduced.

### Shedding (job #2183, `soak-b8-shed/`)

Server `--max-batch 4 --max-queue 4` (admit limit = max_queue + max_running = 8). `cbsoak.py burst`: all requests released together by a barrier.

| burst | status | Retry-After on 429 | rejected counter delta | peaks (`/metrics`) | admitted TTFT p50 / max | admitted ITL p95 / max | after |
|---|---|---|---|---|---|---|---|
| 16 x 512-token prompts, max_tokens 300 | 8 ok, 8 x 429 | 1 on all 8 | 8 | queue 4, running 4 | 5197 / 10264 ms | 30.2 / 101.9 ms | health ok, running 0 waiting 0; follow-up ok (68 ms TTFT) |
| 24 x 1024-token prompts, max_tokens 200 | 8 ok, 16 x 429 | 1 on all 16 | 16 | queue 4, running 4 | 4572 / 9010 ms | 31.4 / 112.9 ms | same; follow-up ok (85 ms) |

429 body: `the request queue is full (4 waiting); retry shortly` (`rate_limit_error`). No 503, no other error, client 429 count equals the server's counter. The 503 path is gone from the overload case (503 now only appears when the engine has failed). The decode of the 8 admitted requests was not disturbed (ITL p95 30 to 31 ms). One oddity, not investigated: in the second burst `intarsia_requests_waiting` peaked at 7 while `intarsia_batch_queue_depth` peaked at 4 (running 4, limit 8), so the two gauges count differently (the first is probably read between admission and hand-off to the scheduler).

## Hygiene and setup

* Machine: endor (Apple M5 Max, 128 GB), Qwen3-8B BF16 (`~/models/Qwen3-8B`), `serve --backend metal --max-seq 8192 --profile-every 0 --no-supervisor --no-ui`, prefix reuse on (default), `--spec off`.
* Binaries (built with `cargo build --release -p intarsia-cli --features metal`, copied to the scratch dir): `intarsia-head` = `spike/metal` 6d99a6f (`serve --max-batch 2..8`, review fixes, `--spec-off-at`, forward-hash fix), sha256 prefix `87516c7fa5785cb4`; `intarsia-base28e5c5a` = the M1-era baseline 28e5c5a, sha256 prefix `10abb4203759a958`.
* Harness: `tools/cbsoak.py` (new in this branch: `scenario` and `burst` subcommands, `Retry-After` capture, cold-start TTFT split), `tools/cbsoak_serve.sh` (now logs macmon, `ps` and swap at start and end), `tools/cbsoak_ab.py` (same-slot A/B ratio guard), `tools/cbsoak_table.py`. `uv run tools/cbsoak.py selftest`: 82 checks, all pass (includes a mock that stalls the other streams while a long prompt prefills, to prove the scenario statistic sees a stall and the before/after windows do not).
* Nothing under `crates/` was changed.



Each slot's start: `ps` top 12 by CPU plus swap, `macmon` idle sample 6 s before the server starts, `ps` again at the end (`hygiene.txt` and `macmon.jsonl` in each run directory; `tools/macmon_summary.py` summarises a macmon log).

* GPU idle before the server in every slot: utilisation 0 to 2% (mean 1 to 2%), GPU power 0.1 W. So `endor-kev` / `endor-tensorfold` and everything else were not using the GPU at the start of any slot. No separate GPU-use sample was taken at the end (the last macmon samples overlap the server shutdown), so use of the GPU by something else during a run cannot be excluded, only that nothing else was running at the start; the B=8 soak's utilisation (81% mean) is the server plus anything else.
* A constant CPU consumer sat in every `ps` snapshot: `mediaanalysisd` at about 220% CPU (up 14 h). It is not a GPU user at idle (idle GPU util 1%), but it is CPU load on the same machine as the client and server; flag, not quantified. Swap was 10.4 GB used of 11.3 GB in every snapshot (encrypted swap, the machine is memory-tight; the server's footprint of 9.7 GB did not change that).
* The weight re-wire after about 3 s idle: every measured window was preceded by a warm-up request (scenario: before each repetition; soak and ramp: the harness' two warm-up requests; B=1 guard: same). The cold-start split in the report (`ttft_cold_start_ms`) shows it is not visible in the TTFT tails of the soak.

## Engine bugs found

None. Observations that are not bugs: the `waiting` vs `queue_depth` gauge difference in the burst; process footprint ramps 1.7 to 9.6 GB on first KV use (expected from the preallocated pool, but it looks like a leak to a naive monitor).

## Tools added or changed

* `tools/cbsoak.py`: `scenario` (the scripted gate scenario, `--max-itl-prefill-p95-ms`, `--max-steady-itl-p95-ms`, `--max-long-ttft-x-solo`, exit 1 on violation), `burst` (shedding test: 429 + Retry-After + rejected counter + 0 x 503 + follow-up + drain), `Retry-After` captured per shed request, cold-start TTFT split in the report, mock server extended (stall while a long prompt prefills, listen backlog). Selftest 82 checks.
* `tools/cbsoak_serve.sh`: macmon log, `ps` and swap snapshots, `scenario` / `burst` / `run` as per-segment subcommand.
* `tools/cbsoak_ab.py` (same-slot A/B ratio guard), `tools/cbsoak_table.py` (one row per report), `tools/macmon_summary.py`.
* `tools/cbreuse.py gen --copy-frac F` (input-grounded copy-task cold prompts so that `--spec ngram` drafts), for Part B; checked offline against the tokenizer with a stubbed server, not against a real one.

## Not run, and how to resume

* **B=4 soak** (30 min, 0.35 req/s was the planned rate; the B=4 capacity was not measured, a short B=4 ramp should come first): `docs/results/raw/2026-10-16-metal-cb-soak-batched/slot_soak.sh OUT 4 0.35 3 1800` inside `fleetq run --host endor --exclusive --mem 24`. About 40 min.
* **A/B/A for the `--max-batch 1` guard** (a third run of the baseline after the candidate): `slot_extra.sh` first loop with `base head base`. About 20 min.
* **Part B**: `slot_partB.sh OUT` (about 35 min; CHAINS=100, COPY=0.5, B in 2 3 8). Expected checks: `compare-b*.log` 0 mismatches, `metrics-b*-{before,after}.txt` for `intarsia_spec_proposed_tokens_total` / `accepted` deltas (drafting happened) and `report-b*.json` decode-written KV reuse counts. Untested end to end.
* Literal 7 streams + 4k at B=4 (queueing case), the B=8 soak at a second rate, the B=8 soak with `--max-queue` default and a rate above saturation with cancellations are also not covered.
* `slot_b1_shed.sh` is superseded (shedding moved into the B=8 slot, the guard into `slot_extra.sh`) and was never run.

## Reproduce

```
cargo build --release -p intarsia-cli --features metal   # copy target/release/intarsia to a scratch dir
fleetq run --host endor --exclusive --mem 24 --label "cbsoak2 B8" -- docs/results/raw/2026-10-16-metal-cb-soak-batched/slot_soak.sh OUT 8 0.5 7 1800
fleetq run --host endor --exclusive --mem 24 --label "cbsoak2 extra" -- docs/results/raw/2026-10-16-metal-cb-soak-batched/slot_extra.sh OUT
uv run tools/cbsoak_ab.py --base OUT/b1-base --cand OUT/b1-head
```

Raw: `docs/results/raw/2026-10-16-metal-cb-soak-batched/{ramp8,soak-b8,soak-b8-shed,extra}/` (each: `raw.json`, `requests.jsonl`, `report.{json,md}`, `scenario.{json,md}`, `burst.{json,md}`, `server.log`, `macmon.jsonl`, `hygiene.txt`, `cbsoak_serve.log`, `metrics-end.txt`) and the slot scripts.
