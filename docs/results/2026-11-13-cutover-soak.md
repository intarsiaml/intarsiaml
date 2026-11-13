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

Build: main `45e6ce56b` (binary stamp `ab35c-45e6ce56b023`), fleetq #9242, 5 m 47 s. Serve shape for all intarsiaml arms: `--max-batch 2 --prefix-store ...`, bf16 KV, `--max-seq 131072 --mem-budget 40 --profile-every 0`, everything else default (no MTP: the store refuses drafting). earlyoom was `active` on spark2 at both window starts. Plan and prompts are the interleave lane's (`replay-plan.json`, 8 conversations x 6 turns, 8k to 33k tokens, thinking off, temperature 0, 160 max tokens); replay2 interleaved.

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

Store host budget: `--prefix-store host_mb=4096` (4.29 GB, half the 8 GiB default). The ~66 KB per token figure from the brief gives about 65k tokens of KV, two conversations at 28k; one replay inserts about 12.5 GB (E-main metrics: 12.49 GB of `insert_bytes_total` per replay), so eviction starts inside the first replay (the second 60 s sample, about 80 s after the first request, already showed evictions; 5507 in total). The default 8 GiB budget also evicts inside one replay (E-main: 8.55 of 8.59 GB, 110 host evictions after 131 s), so the shrink is a stress case, not the only way to hit the eviction path: it drops the cached fraction from 0.73 to about 0.19 and so roughly triples the uncached prefill that runs beside decode. Each replay records its own trace from the server's replies (multi-turn with the model's own answers). No `INTARSIA_PREFIX_STORE_DEBUG`. A sampler on the box wrote one row per minute (`raw/sample-S.tsv`, `scripts/soak/sampler.sh`); `memwatch.sh` (2 s MemAvailable, kill under 10 GB) logged 1149 samples, lowest 79.8 GB, 0 kills.

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

### Window B (2026-11-13, spark2, vLLM down 10:46:33 to 11:35:32 = 2939 s = 49.0 min; fleetq #9247) and the vLLM thinking-on arm (#9246)

All three arms: `--max-batch 2 --prefix-store on` (default host budget), bf16 KV unless stated, no MTP (the store refuses drafting), `localeval quality --thinking off|on --structured prompt --bench gsm8k --max-tokens ... --pin-gsm8k` (the ab35b Run 2 setup: pinned template, temperature 0, no nonce). Raw: `raw/<run id>/{summary.md,result.json}`, `raw/{Ba,Bb,Bc,V-think}-gsm8k.out`, `raw/compare-*.txt` (localeval `compare`), `raw/gsm8k-style.json`. Paired verdicts are `scripts/soak/paired_gsm8k.py` (same question ids; exact McNemar on the discordant pairs), since localeval `compare` is an unpaired 2-standard-error test.

#### (a) store on, bf16 KV, GSM8K n=200, thinking off

| run | gsm8k_flexible | n | truncated | mean output tokens | worked-answer style | bare-number style (correct) |
|---|---:|---:|---:|---:|---:|---:|
| vLLM A1 / A2 (ab35b) | 0.610 / 0.585 | 200 | 1 / 1 | 43.0 / 44.6 | 50 / 53 | 150 (75) / 147 (66) |
| intarsiaml store off, no MTP (B-specoff, ab35b) | 0.710 | 200 | 1 | 72.5 | 86 | 114 (58) |
| **Ba: store on, bf16** | **0.715** | 200 | 2 | 72.2 | 77 | 123 (68) |

- Paired against store-off (B-specoff): 10 questions right only in store-off, 11 right only in store-on, McNemar p = 1.0; 127 of 200 reply texts identical (73 differ). localeval `compare`: +0.005, noise. The store does change greedy bits (73 differing replies), and accuracy does not move.
- Against vLLM: Ba is +0.105 over A1 (paired 9 vs 30, p = 0.001) and +0.130 over A2 (3 vs 29, p < 0.001), the same direction and size as store-off in ab35b (0.710): not below vLLM.
- **Caveat that limits what (a) tested:** the store's own counters for this arm read 200 lookups, 0 hits, 200 misses, 0 tokens saved, 200 inserts (`raw/serve-Ba.metrics`). The pinned GSM8K prompt is one user message, so the store has no boundary inside it to restore from, and every request prefills in full. What changed against store-off is therefore the prefill chunking and the capture work (cuts at message boundaries), not a restored prefix. The accuracy of an answer decoded after a restore is not measured here; the replay check against store-off (interleave lane) is the evidence for that path. A GSM8K variant with a shared system or multi-turn prefix would be needed to test it.

#### (b) store on, `--kv-dtype fp8`, GSM8K n=200, thinking off

| run | gsm8k_flexible | truncated | mean output tokens | worked-answer style (correct) | bare-number style (correct) |
|---|---:|---:|---:|---:|---:|
| Ba (bf16) | 0.715 | 2 | 72.2 | 77 (75) | 123 (68) |
| **Bb (fp8 KV)** | **0.655** | 0 | 56.8 | 73 (72) | 127 (59) |
| vLLM A1 / A2 (fp8 KV in production) | 0.610 / 0.585 | 1 / 1 | 43.0 / 44.6 | 50 (47) / 53 (51) | 150 (75) / 147 (66) |

- Paired Ba vs Bb: 20 right only in bf16, 8 right only in fp8, McNemar p = 0.036; against store-off B-specoff: 22 vs 11, p = 0.080; localeval `compare`: -0.060 and -0.055, both "noise" (2 standard errors ±0.093). 114 of 200 reply texts identical to Ba. Against vLLM (which also runs fp8 KV): Bb is +0.045 over A1 (paired p = 0.15) and +0.070 over A2 (p = 0.013).
- Reading: fp8 KV costs about 6 points on this n=200 set at the edge of what the set can resolve (p 0.04 against the same-store bf16 arm, 0.08 against store-off, inside the 2-SE line), and the loss sits in the bare-number replies (59 of 127 right against 68 of 123), while the worked answers stay at 72 of 73. It is not below vLLM, which runs the same KV dtype. `--kv-dtype fp8` is opt-in and the store accepted it (serve up, 200 requests, 0 errors, 0 hits as in (a)); a repeat at n=400 or more would be needed before calling the drop real or switching it on by default. Same limit as (a): 0 store hits.

#### vLLM thinking-on (V-think; fleetq #9246, `--quiet gpu,cpu`, vLLM up the whole time, 40.8 min)

GSM8K n=100, thinking on, `--max-tokens 8192`: **0.970** (97 of 100), 0 truncated, mean 1875.6 output tokens, decode 77.3 tok/s, `reasoning_content` present on 100 of 100 (qwen3 reasoning parser), `thinking_seen` 100. vLLM's own counters before and after: exactly 100 requests (no production traffic inside the run). The 5-problem pilot before it (1475 tokens, 19.4 s per problem) was a low sample: the real mean is 24.4 s per problem.

#### (c) thinking on, GSM8K, `--max-tokens 8192`: intarsiaml store-on bf16 (Bc) against vLLM (V-think)

Bc ran n=85, not 100: at ~24 s per problem (measured on vLLM, which finished first) n=100 needs 41 min, and window B had 39 min left after (a) and (b), so `window_b.sh` took the n that fits (85); Bc finished at 11:32, about 4 min before the 50-minute cap. The vLLM arm ran n=100 (it needed no window); the paired comparison uses the same first 85 question ids. Same prompts (median 483 prompt tokens, min and max identical on both). Both engines were asked for temperature 0.0: localeval puts `temperature` in every request body (`Client.chat` default 0.0, passed through `model.py`), thinking on or off, so both arms are greedy (the server's sampling default of 1.0 only applies to requests without a temperature). Paired localeval `compare` V-think (n=100) against Bc (n=85) is unpaired and the n differs: +0.018, noise (`raw/compare-Vthink-vs-Bc.txt`); the paired figures below use the same 85 ids.

| run | gsm8k_flexible | n | truncated | mean output tokens (median, max) | decode tok/s | `reasoning_content` present | wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| V-think (vLLM), all 100 | 0.970 | 100 | 0 | 1875.6 | 77.3 | 100 / 100 | 2445 s |
| V-think, first 85 ids | 0.965 (82) | 85 | 0 | 1857.9 (1574, 4638) | | 85 / 85 | |
| **Bc (intarsiaml, store on, bf16)** | **0.988 (84)** | 85 | 0 | 2133.1 (1771, 6318) | 85.3 | 85 / 85 | 2155 s |

- Paired (first 85 ids): intarsiaml right where vLLM is wrong on 2 questions, the reverse on 0, McNemar p = 0.5. The one question Bc got wrong (gsm8k-00012) vLLM also got wrong (and vLLM missed two more). No accuracy gap in either direction.
- Final-answer text differs on 32 of 85 questions. I printed all 32 pairs: the differences are **substantive style differences, not whitespace or parser artefacts**. After thinking, vLLM's `content` is a bare number on 74 of 85, intarsiaml's on 61 of 85; the rest are worked solutions ("**Solution**: ... <<3*20=60>>60 ... **Answer**: 20", the exemplar format, or "Let's think step-by-step"), 24 on intarsiaml against 11 on vLLM. This is the same direction as thinking off (worked answers: Ba 77 of 200, Bb 73, store-off B-specoff 86, against vLLM 50 and 53), so intarsiaml writes a worked final answer more often than vLLM in every arm, with and without thinking and with bf16 or fp8 KV. The cause is not found; accuracy is not affected.
- Reasoning parser: both engines return the chain of thought in `reasoning_content` (85 of 85) and the final answer in `content`; no `think` tag in either field, no empty reply, 0 truncated at 8192 on either engine, median `content` 2 characters on both. localeval streams, so the streaming path of both parsers was exercised, for plain no-tool prompts only. Tool calls inside thinking, and a `reasoning_content` field compared delta by delta, were not tested.
- **Difference worth knowing (greedy on both):** intarsiaml's thinking is longer on the same questions: mean 2133 against 1858 output tokens (+15 %), median 1771 against 1574 (+12.5 %), longer on 53 of 85, longest 6318 against 4638. Decode is faster (85.3 against 77.3 tok/s, +10 %), so wall per problem is nearly the same (25.3 against 24.5 s). A sign test on 53 longer of 85 gives p about 0.03, so it looks systematic rather than noise. I did not find the cause (not investigated; both runs are greedy with identical prompts, same token counts). It is not a quality loss here, but the 8192 cap sits closer to intarsiaml's tail.
- As in (a) and (b), the store saw no hits (single user message; `raw/serve-Bc.metrics`).

## Verdicts

### Blocker 6, unmeasured load and soak: CLOSED for more than 2 streams, the soak and thinking-on quality; four items stay open

- **More than 2 streams: closed.** K=3 and K=4 behave like vLLM: TTFT sums 164.2 vs 162.1 s and 302.8 vs 302.6 s, aggregate 56.6 vs 58.0 and 59.6 vs 60.2 tok/s, 0 errors, 0 rejections, no engine gap above 0.5 s. One seed per cell.
- **Soak: pass, with a stress budget.** 37 min, 480 requests, 0 errors, 0 rejections, `engine_failed` 0, no log warnings. Store host bytes: the peak over 38 one-minute samples and 10 per-replay scrapes was 4293 MB against a 4295 MB budget (a sampled peak, not a continuous bound), with 5507 evictions and 0 rejected inserts. Memory: MemAvailable 79.8 to 81.2 GB after the first 5 minutes with a fitted drift of -3.3 MB/min (about -0.1 GB over the run against 1.4 GB of peak-to-peak noise); mem_used 43.4 to 44.8 GB (+3.4 MB/min); `intarsia_device_memory_free_bytes` the same shape; swap flat at 490 MB (there before the run); server RSS flat at 162 MB (CUDA allocations live outside RSS, so MemAvailable and the device-free gauge are the signals). TTFT sum per replay 169 to 199 s with the first-half mean 181 and second-half mean 179 s; aggregate output 32.2 to 34.8 tok/s; wall 219 to 230 s; no trend. earlyoom `active`; memwatch never fired (lowest 79.8 GB).
- **The soak's budget is a stress case.** At `host_mb=4096` the cached fraction is 0.19 (0.73 at the 8 GiB default), so about 3 times more prompt tokens prefill beside decode (81 % uncached against 27 %): 3021 of 74355 engine gaps are above 0.25 s (4.1 %, against 1.4 % in the K=2 default runs), 0 above 0.5 s. That is the expected cost of a store smaller than the working set (8 conversations at 28k), not drift: it is the same in the first and the last replay. A soak at the 8 GiB default (the production shape) was not run; only the two K=3/K=4 replays used it.
- **Thinking-on quality: closed** (Bc 0.988 against vLLM 0.965 on the same 85 questions; see (c)).
- **Still open (4):** (1) real missioncntrl traffic (agent tool calls, long sessions, mixed sampled and greedy requests) was not available to this lane and was not replayed; (2) reasoning-parser equivalence is shown for GSM8K only (no tools, no deltas compared one by one); (3) no soak at the default 8 GiB budget; (4) the 15 % longer thinking is unexplained.

### Blocker 7, unexplained output differences: PARTLY CLOSED. Closed for the store-on chunking and capture path (bf16) and for thinking-on accuracy; OPEN for accuracy after a store restore and for the unexplained style difference; fp8 KV is a new small item

- Store on against store off (bf16, greedy): 0.715 against 0.710, paired 10 vs 11, p = 1.0, 73 of 200 replies differ in text. The store's bit changes do not move accuracy. **But** the store registered 0 hits in all three quality arms (the pinned GSM8K prompt is one message with nothing to restore): no answer in any quality arm was decoded after a restore. This tests the store-on chunking and capture path, not accuracy after a restore, which stays open and needs a GSM8K variant with a shared system prompt or multi-turn prefix in one more window.
- `--kv-dtype fp8` with the store: 0.655, 6 points below bf16 (paired 20 vs 8, p = 0.036; vs store-off p = 0.080; localeval compare calls it noise). Not below vLLM (0.610 and 0.585; vLLM also runs fp8 KV). Opt-in, so not a cutover blocker, but it should not become a default on this evidence; n=400 or more would settle it.
- Thinking on: no accuracy difference from vLLM (above). The style difference (intarsiaml writes a worked final answer more often than vLLM in every arm) is what the ab35b blocker called unexplained, and it is still unexplained; accuracy is not affected.

### New issues, none blocking
1. fp8 KV accuracy drop above (borderline).
2. Thinking traces +15 % longer than vLLM on identical prompts, cause not found.
3. GSM8K cannot exercise the store's restore path; a variant with a shared system prompt or multi-turn prefix is needed to measure accuracy after a restore.
4. Store sizing: at a budget below the working set the cached fraction collapses (0.19), so the production `host_mb` has to be chosen for the conversation count (the default 8 GiB gave 0.73 at 8 conversations of up to 33k).

## Fairness and limits

- One seed per K=3 / K=4 cell, one run per quality arm (greedy, so repeats would be identical on the same server state; no repeat of fp8 or store-on was run).
- The vLLM and intarsiaml K=3/K=4 arms ran back to back (vLLM 09:13 to 09:17, intarsiaml 09:18 to 09:23, same day, same box); vLLM's own counters showed no foreign requests in any vLLM arm (`raw/vllm-foreign-traffic.txt`). The production vLLM kept its prefix cache between K=3 and K=4 as intarsiaml's store did.
- vLLM's baseline GSM8K thinking-off numbers (A1 0.610, A2 0.585) are from the ab35b run before vLLM was restarted for other lanes; the thinking-on vLLM arm is new (this run).
- Replays use synthetic filler prompts with the interleave lane's plan; decode medians per request are bimodal under concurrency, so throughput claims use TTFT sums and aggregate output tokens per wall second.
- localeval's ITL p99 was not used for any stall claim: gap counts come from the engine's `intarsia_batch_itl_seconds`.
- Bc n=85 against the asked 100 (time); the paired comparison is on the same 85 ids.
- The `--quiet gpu,cpu` slot holds other fleetq jobs off; it does not stop missioncntrl requests. The vLLM counters are the check that none arrived.

## Raw results and scripts

- `docs/results/2026-11-13-cutover-soak/raw/`: window logs (`window-a.log`, `window-b.log`, `vllm-arms.log`, `*.out`), per-replay jsonl and summaries (`V-k3`, `V-k4`, `L-k3`, `L-k4`, `S11` to `S20`), traces (`replay-trace-seed{1,2}.json`, `soak-trace-seed*.json`), sampler and memwatch files (`sample-{L1,S}.tsv`, `mem-*.txt`), server logs and metrics (`serve-*`, including per-replay scrapes `serve-S.metrics.after-seed*`), localeval runs (`<run id>/{summary.md,result.json}`: V-pilot, V-think, Ba, Bb, Bc), `compare-*.txt` (localeval compare), `gsm8k-style.json`, `gsm8k-think-paired.json`, `recipes/`, `vllm-foreign-traffic.txt`, `index.txt`, `build.log`.
- localeval originals `/opt/code/deepeval/results/<run id>/`; unanimis ids: Ba 27e569c1, Bb 11d7269a, Bc ca8e7bbf, V-think 3b998021, V-pilot 56f0af2f.
- fleetq jobs (owner `claude-soak`): #9242 build, #9243 vLLM thinking pilot (n=5, discarded), #9244 vLLM K=3/K=4, #9245 window A, #9246 vLLM thinking n=100, #9247 window B.
- Scripts `scripts/soak/`: `lib.sh`, `vllm_arms.sh`, `window_a.sh`, `window_b.sh`, `memwatch.sh`, `sampler.sh`, `soak_table.py` (`load` / `soak`), `paired_gsm8k.py`. Tables: `python3 scripts/soak/soak_table.py docs/results/2026-11-13-cutover-soak/raw load|soak`; pairs: `python3 scripts/soak/paired_gsm8k.py [--first N] label=<result.json> ... --pairs a:b`.
