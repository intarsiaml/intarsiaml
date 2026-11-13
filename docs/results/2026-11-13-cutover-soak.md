# Cutover soak: load, soak and quality (blockers 6 and 7)

Lane `soak`, branch `cutover-soak` from origin/main `45e6ce56b`, measurement only (no engine changes).

## Brief (verbatim from the launching agent, kept so it survives compaction)

You are lane `soak` in intarsiaml, a Rust LLM inference engine. Working folder `/opt/code/intarsiaml/.worktrees/soak` (git worktree, branch `cutover-soak` from origin/main `45e6ce56b`). Work only there; no other work folders. Results go to `docs/results/2026-11-13-cutover-soak.md`.

### Goal
Collect measurement-only evidence for cutover blockers 6 and 7 in `docs/results/2026-11-12-ab35b-cutover-followup.md` (section "Cutover blockers, ranked"). The question is whether intarsiaml can replace production vLLM for Qwen3.6-35B-A3B-NVFP4 on spark2. Do not change engine code. If a bug is found, write it down with evidence and stop that arm.

### Already established (do not redo)
- Blockers 1 to 3 closed by PRs #351 to #354 and #357.
- 2-stream replay with `--max-batch 2 --prefix-store on` matches vLLM: TTFT sum 73 s vs 77.8 s, decode 39.9 vs 39.8 tok/s per stream.
- localeval's ITL p99 is computed per character and inflates single-token stalls; use the engine's own `/metrics` gap histogram for stall claims (`docs/results/2026-11-13-interleave.md`, `2026-11-13-stream-burst.md`).
- GSM8K n=200 baseline for vLLM: 0.610 and 0.585. intarsiaml batched bf16 store-off: 0.710 (Run 2 of the ab35b doc).

### What to measure (intarsiaml on main HEAD; production-shaped serve: `--max-batch 2 --prefix-store on`, bf16 KV, defaults otherwise)

Window A, at most 55 min with vLLM down: load and soak
1. More than 2 streams: replay2 interleaved with K=3 and K=4, one seed each. Compare against vLLM at K=3 and K=4 (vLLM runs `--max-num-seqs 2` in production, so both engines queue). Run the vLLM arms before taking vLLM down, while it is up and quiet. Production vLLM is a live service: check `fleetq status`, use `--quiet gpu,cpu` on spark2. If vLLM can't be measured cleanly, say so.
2. Soak: about 35 min of continuous concurrent multi-turn load at K=2. Loop replay2 over fresh seeds. Shrink the store's host budget so eviction runs: pick `host_mb` from the store's bytes-per-token (about 66 KB per token) so evictions start within about 5 min. Sample every minute: `free -g` and MemAvailable, server RSS, prefix-store tier bytes, hits/evictions/errors from `/metrics`, p50 TTFT and decode per replay. Pass criteria: no errors; no RSS or MemAvailable drift beyond noise; store bytes within budget; TTFT and decode flat over time. Confirm earlyoom is active on spark2.

Window B, at most 55 min with vLLM down: quality (blocker 7). `local-model-bench` skill (localeval at `/opt/code/deepeval`; `uv run localeval ...`), same pinned GSM8K setup as ab35b Run 2: `--limit 200 --max-tokens 1024 --pin-gsm8k`, thinking off, temperature 0. Arms:
- (a) store-on bf16 at `--max-batch 2` (new; the prefix store changes greedy bits on CUDA, check accuracy against store-off).
- (b) `--kv-dtype fp8` with the store on.
- (c) one thinking-on quality run: GSM8K thinking on, n=100, `--max-tokens` sized for thinking, against vLLM thinking-on. Run the vLLM arm while vLLM is up and quiet.
Use localeval `compare` for paired verdicts against the existing vLLM and B runs.

### How to run
- Build: `REPO=/opt/code/intarsiaml/.worktrees/soak ABL='~/intarsia-soak' BHOST=spark2 FLEETQ_OWNER=claude-soak fleetq run --host spark2 --load cpu --mem 24 --expect 20 --label "soak: build" -- bash /opt/code/intarsiaml/.worktrees/ab35c/build.sh`
- Windows: template `scripts/interleave/window.sh` (stops `missioncntrl-guidance`, trap restores, waits for `/v1/models` 200); copy to `scripts/soak/` and adapt. Each window: `FLEETQ_OWNER=claude-soak fleetq run --host spark2 --takes-down retro --quiet gpu,cpu --mem 58 --expect 60 ...`, hard cap 55 min of vLLM downtime per window. Standing OK covers spark2 vLLM-down windows up to 60 min. Leave at least 15 min with vLLM up between the windows.
- Replay tooling: `scripts/interleave/replay2.py` on main; GSM8K arms via `scripts/ab35b/` and `scripts/ab35c/lib.sh`.

### Rules
- Run a real check for anything runnable you change; say which check could not run.
- No tool call or wait over 4 min (Bash timeout <= 240000); poll at most every 4 min; run long windows in the background.
- `fleetq status` first; always `--expect`; never `--force` or `--no-queue`.
- GB10: 121 GB unified, swaps and hangs instead of OOM-kill; give `--mem` the peak; kill a run if MemAvailable < 10 GB (15 GB if earlyoom is not active).
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; no secrets.
- Finish: DRAFT docs-only PR against main with the results doc and raw files; PR body ends with the Claude Code line. Report a per-blocker verdict (closed, open, new issue) with key numbers.

## Log

### Window A (done 2026-11-13, spark2, vLLM down 09:18:00 to 10:05:32 = 2852 s = 47.5 min; fleetq #9245, owner claude-soak)

Build: main `45e6ce56b` (binary stamp `ab35c-45e6ce56b0236`), fleetq #9242, 5 m 47 s. Serve shape for all intarsiaml arms: `--max-batch 2 --prefix-store ...`, bf16 KV, `--max-seq 131072 --mem-budget 40 --profile-every 0`, everything else default (no MTP: the store refuses drafting). earlyoom was `active` on spark2 at both window starts. Plan and prompts are the interleave lane's (`replay-plan.json`, 8 conversations x 6 turns, 8k to 33k tokens, thinking off, temperature 0, 160 max tokens); replay2 interleaved.

#### 1. More than 2 streams (K=3 seed 1, K=4 seed 2)

vLLM arms (fleetq #9244, `--quiet gpu,cpu`, production vLLM up and idle: `vllm:num_requests_running` 0, counters 0 before the first arm) ran first and recorded the traces (`raw/replay-trace-seed{1,2}.json`); intarsiaml replayed the same files on one server (K=3 then K=4, store state carried over, as vLLM's prefix cache was). vLLM's own request counters before and after each arm show exactly our 49 requests (48 + warmup) and nothing else (`raw/vllm-foreign-traffic.txt`): no production traffic inside the vLLM measurement. fleetq printed "4 in flight" for spark2-vllm right after the K=4 arm; vLLM's counters and `num_requests_running` showed 0, so I read that as our own tail.

| arm | requests | errors | TTFT sum s | TTFT p50 s | TTFT p90 s | decode tok/s median | cached frac | wall s | agg output tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V-k3 | 48 | 0 | 162.1 | 2.61 | 5.22 | 40.9 | 0.696 | 127 | 58.0 |
| L-k3 | 48 | 0 | 164.2 | 3.08 | 5.40 | 38.2 | 0.732 | 129 | 56.6 |
| V-k4 | 48 | 0 | 302.6 | 6.28 | 7.18 | 39.7 | 0.695 | 126 | 60.2 |
| L-k4 | 48 | 0 | 302.8 | 6.47 | 7.16 | 39.3 | 0.732 | 127 | 59.6 |

- intarsiaml against vLLM, same prompts, same K: TTFT sum +1.3 % at K=3 (164.2 vs 162.1 s) and +0.1 % at K=4 (302.8 vs 302.6 s); aggregate output 56.6 vs 58.0 tok/s (-2.4 %) at K=3 and 59.6 vs 60.2 (-1.0 %) at K=4. Both engines queue (2 run, K-2 wait) and TTFT grows with K the same way (K=2 was 73 vs 77.8 s, K=3 162, K=4 302). intarsiaml's cached fraction is higher (0.732 vs 0.696).
- Per-request decode median is a poor statistic here (it is bimodal: ~17 tok/s while the other stream prefills, 40 to 60 otherwise); 38.2 vs 40.9 at K=3 and 39.3 vs 39.7 at K=4 are inside that noise. One run per cell, one seed.
- Errors: 0 of 96 intarsiaml requests, `intarsia_requests_rejected_total` 0 (default `--max-queue` 64, so K=4 never came near a 429), `engine_failed` 0. Engine-side gaps (`intarsia_batch_itl_seconds`, no character weighting): K=3 7277 gaps, 107 above 0.25 s, 0 above 0.5 s; K=4 7505 gaps, 108 above 0.25 s, 0 above 0.5 s (K=2 on main per the interleave doc: 103 above 0.25 s of 7548).
- Lowest MemAvailable on that serve 76.0 GB.

#### 2. Soak (37 min, K=2, 10 replays of 48 requests = 480 requests, fresh seeds 11 to 20)

Store host budget: `--prefix-store host_mb=4096` (4.29 GB, half the 8 GiB default). The ~66 KB per token figure from the brief gives about 65k tokens of KV, two conversations at 28k; one replay inserts about 12.5 GB (E-main metrics: 12.49 GB of `insert_bytes_total` per replay), so eviction starts inside the first minute (first host eviction seen at the first 60 s sample, 5507 in total). The default 8 GiB budget also evicts inside one replay (E-main: 8.55 of 8.59 GB, 110 host evictions after 131 s), so the shrink is a stress case, not the only way to hit the eviction path: it drops the cached fraction from 0.73 to about 0.19 and so roughly triples the uncached prefill that runs beside decode. Each replay records its own trace from the server's replies (multi-turn with the model's own answers). No `INTARSIA_PREFIX_STORE_DEBUG`. A sampler on the box wrote one row per minute (`raw/sample-S.tsv`, `scripts/soak/sampler.sh`); `memwatch.sh` (2 s MemAvailable, kill under 10 GB) logged 1149 samples, lowest 79.8 GB, 0 kills.

| replay | start min | requests | errors | TTFT sum s | TTFT p50 s | TTFT p90 s | decode tok/s median (bimodal: see doc) | decode tok/s mean | agg output tok/s | cached frac | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S11 | 0.0 | 48 | 0 | 176.8 | 2.67 | 7.09 | 36.7 | 31.8 | 32.2 | 0.190 | 230 |
| S12 | 3.9 | 48 | 0 | 187.6 | 2.12 | 7.17 | 41.9 | 36.7 | 33.8 | 0.227 | 219 |
| S13 | 7.5 | 48 | 0 | 199.1 | 1.98 | 10.24 | 40.5 | 35.4 | 32.5 | 0.190 | 224 |
| S14 | 11.3 | 48 | 0 | 172.0 | 2.54 | 7.08 | 40.7 | 33.8 | 33.3 | 0.190 | 227 |
| S15 | 15.1 | 48 | 0 | 169.4 | 2.58 | 7.08 | 39.8 | 32.6 | 33.1 | 0.190 | 227 |
| S16 | 18.9 | 48 | 0 | 169.8 | 2.56 | 7.08 | 40.4 | 32.8 | 33.1 | 0.190 | 227 |
| S17 | 22.7 | 48 | 0 | 176.0 | 2.56 | 7.08 | 41.5 | 35.6 | 34.2 | 0.227 | 219 |
| S18 | 26.4 | 48 | 0 | 191.1 | 2.42 | 8.29 | 41.7 | 37.9 | 34.8 | 0.227 | 220 |
| S19 | 30.1 | 48 | 0 | 169.9 | 2.55 | 7.10 | 33.9 | 33.1 | 32.8 | 0.191 | 226 |
| S20 | 33.8 | 48 | 0 | 185.9 | 2.55 | 8.14 | 40.9 | 34.2 | 33.4 | 0.190 | 226 |
- ttft_sum: first half mean 180.970, second half mean 178.538, min 169.413, max 199.059
- ttft_p50: first half mean 2.376, second half mean 2.530, min 1.976, max 2.672
- dec_p50: first half mean 39.924, second half mean 39.670, min 33.856, max 41.938
- dec_mean: first half mean 34.077, second half mean 34.705, min 31.820, max 37.887
- cached: first half mean 0.198, second half mean 0.205, min 0.190, max 0.227
- wall: first half mean 225.224, second half mean 223.594, min 218.786, max 229.944

sampler: 38 samples, 09:24:36 to 10:01:40
| signal | first 5 mean | last 5 mean | min | max | slope/min, all | slope/min, from sample 6 (store full) |
|---|---:|---:|---:|---:|---:|---:|
| mem_avail_mb | 81330.8 | 80301.6 | 79813.0 | 83713.0 | -19.96 | -3.34 |
| mem_used_mb | 43278.0 | 44307.4 | 40896.0 | 44796.0 | 19.97 | 3.35 |
| swap_used_mb | 490.0 | 490.0 | 490.0 | 490.0 | 0.00 | 0.00 |
| rss_mb | 162.0 | 162.0 | 162.0 | 162.0 | 0.00 | 0.00 |
| intarsia_kv_bytes_used | 827.6 | 1021.9 | 398.4 | 1429.6 | 4.11 | 0.59 |
| intarsia_device_memory_free_bytes | 85282.0 | 84202.4 | 83690.0 | 87780.1 | -20.95 | -3.53 |
| intarsia_prefix_store_tier_bytes{tier="host"} | 3898.8 | 4052.8 | 2396.4 | 4293.3 | 2.71 | -3.18 |
| intarsia_prefix_store_entries | 63.2 | 70.4 | 35.0 | 79.0 | 0.12 | 0.07 |
| intarsia_prefix_store_tier_objects{tier="host"} | 495.4 | 488.0 | 309.0 | 634.0 | -0.31 | -0.98 |

- store host bytes peak 4293 MB against budget 4295 MB (within budget)
- lookups 489, hits 467, misses 22, host evictions 5507, rejected 0, inserts 5581
- requests 489, rejected 0, engine_failed 0, no_progress_s 0, finished length/stop 249/238
- first host eviction seen in sample 1 (of 38; 60 s apart)
- server gaps (batch_itl) total 74026: above 0.25 s 3002, above 0.5 s 0

Raw per-replay metrics scrapes: `raw/serve-S.metrics.after-seed*`.
