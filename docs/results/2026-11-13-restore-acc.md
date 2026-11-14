# restore-acc: accuracy after a prefix-store restore, and a soak at the default host budget

Lane `restore-acc`, branch `restore-acc` from origin/main `8b9535552` (binary stamp `restore-acc-8b95355529a8`, built on spark2 as fleetq #9248, 5 m 06 s), measurement only: no engine code changed, `prefix_store.rs` not touched. It closes two open items of `docs/results/2026-11-13-cutover-soak.md`: (1) the accuracy of answers decoded after a store restore (that doc's GSM8K arms had 0 store hits), (2) a soak at the store's default host budget (that doc used `host_mb=4096` as a stress case). Qwen3.6-35B-A3B NVFP4 on spark2 (DGX Spark GB10).

Serve shape of every arm: `intarsia serve --backend cuda --model <HF snapshot 1355db6a> --max-seq 131072 --mem-budget 40 --profile-every 0 --max-batch 2 --prefix-store ...`, bf16 KV, thinking off, no MTP (the store refuses drafting), no env variables set. The prefill pause is therefore at its default (`INTARSIA_PREFILL_PAUSE_ROWS` 768, PR #360), so item 2 is also the first soak of that default. earlyoom was `active` on spark2 at the start of both windows (checked by the window scripts).

## Summary

| gate | result |
|---|---|
| Item 1, restores happen (store hit counts and cached tokens > 0 on almost every request) | **met**: W 199 of 200 requests restored (store `hits` 199, `local_wins` 0, 1,373,896 tokens saved; median 6904 of 6976 prompt tokens cached); the one miss is the first, cold request |
| Item 1, W and C replies identical on every question | **met**: 200 of 200 reply texts and 200 of 200 generated token-id sequences identical; no diverging token to report. C was a cold prefill with the store on (0 hits, 0 cached tokens on all 200 rows) |
| Item 1, accuracy W against C, paired (McNemar) | **met**: 196 of 200 right in both (0.980), the same questions, discordant pairs 0 / 0, p = 1.0 |
| Item 1, extra: warm store at concurrency 2 (SW2, window D, after the soak) | replies again identical to W on 200 of 200 (text and token ids), 198 of 200 restored (the two misses are the two requests that started cold side by side) |
| Item 2, soak at the default host budget, K=2, about 30 min | **pass**: 32 min, 15 replays (735 requests), 0 errors, 0 rejected requests, `engine_failed` 0; store peak 8588 MB of an 8590 MB budget, 2866 host evictions (over-budget captures show up as evictions: the store has no insert-rejection counter; `rejected`, restores that failed their checks, is 0); TTFT and decode flat over time; MemAvailable fell about 8 GB (83.9 to 75.9 GB) while the store filled in the first 4 minutes and then flattened (about -4 MB/min over the last 17 minutes, 0.1 GB of noise); engine gaps above 0.25 s: 5 of 110,650 (all in the first two replays), above 0.5 s: 0; `intarsia_batch_paused_steps_total` 3179, so the pause default ran |

Open after this lane: arm O (store off) of item 1 was not run; restores were checked for bit-identity at one shape (a 6.9k-token shared prefix cut at a message boundary; the soak's multi-turn restores up to 28k tokens have no reference bits); no real missioncntrl traffic; one seed per soak. See Fairness and limits.

## Item 1: accuracy after a store restore (window C, 2026-11-13, fleetq #9250)

### Method

- **Inputs.** `scripts/soak/gsm8k_dump.py` froze the 200 test questions of the pinned localeval GSM8K arm (same ids `gsm8k-00000` to `gsm8k-00199` and golds: checked 200 of 200 equal against the previous lane's Ba `result.json`) and the first 32 train-split worked solutions (`raw/gsm8k-inputs.json`).
- **Prompt.** `scripts/soak/gsm8k_prefix.py` sends chat messages: one fixed system prompt, then 32 few-shot turns (user `**Problem**: <train question>\n\n**Answer**:`, assistant the worked train solution ending `#### n`), then the test question as the last user message. Every request therefore starts with the same prefix, 6944 to 7054 prompt tokens in all (median 6976), of which the shared part is 6904 tokens (the store's message-boundary cut before the last user message). Temperature 0, `chat_template_kwargs {"enable_thinking": false}` (reply is a plain worked solution, no reasoning content), `max_tokens 512`, non-streaming, `return_token_ids` (the intarsia extension: the generated ids, used for the first diverging token). Scoring is localeval's flexible-v1 (final number or `#### n` marker against the gold; a reply cut by `max_tokens` scores 0). This prompt differs from the pinned arm of the previous lane (one plain-text message, three exemplars), so the accuracies here are **not comparable with 0.715 / 0.610**; only W against C is a comparison.
- **Arm W** (store on, warm): `--prefix-store on` (default 8 GiB host budget). The first request prefills cold and captures; every later request should restore the shared prefix.
- **Arm C** (store on, cold for every request): `--prefix-store host_mb=1`. The store stays on, so the prefill chunk schedule (cuts at message boundaries, no tail merge) and the capture work are exactly W's; only the host tier is 1 MB, so every captured entry is evicted at once (13266 inserts, 13266 host evictions; `rejected` stays 0, that counter is for restores that fail their checks, not for over-budget inserts) and no lookup can find a prefix. This is an honest cold prefill with the store on, which is what a restore is designed to be bit-identical to. (A fresh store per request would need a server restart per request, about 45 s each; there is no clear endpoint.)
- **Both arms at concurrency 1** with `--max-batch 2` serving. Reason: at concurrency 2 which request shares a decode step with which depends on timing, so a difference between a concurrency-2 arm and C would mix batching effects with restore effects. A concurrency-2 arm on a warm store (SW2) was run as a separate observation in window D (below).
- **Proof that W's hits are store restores and C is cold**, from the store's own counters (diff of `/metrics` scrapes before and after each arm, `raw/{W,C}.storediff.json`) and not only from per-request `cached_tokens`: W `hits` 199, `local_wins` 0 (the batched engine's own-slot reuse did not provide any of them), `tokens_saved` 1,373,896 (= the sum of `cached_tokens`), `bytes_restored` 34.7 GB; C `hits` 0, `local_wins` 0, `tokens_saved` 0, `cached_tokens` 0 on all 200 rows.

### Results

| arm | n | errors | correct | acc | truncated (512) | mean output tokens | prompt tokens median (min-max) | requests with cached tokens | cached frac | median cached tokens (hits) | store hits / misses / local_wins | tokens_saved | rejected (failed restore checks) | paused steps | median latency s | wall s | concurrency |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| W | 200 | 0 | 196 | 0.980 | 2 | 145.3 | 6976 (6944-7054) | 199 | 0.984 | 6904 | 199 / 1 / 0 | 1373896 | 0 | 0 | 1.744 | 388 | 1 |
| C | 200 | 0 | 196 | 0.980 | 2 | 145.3 | 6976 (6944-7054) | 0 | 0.0 | 0 | 0 / 201 / 0 | 0 | 0 | 0 | 6.972 | 1220 | 1 |

- **Restores happened on nearly every request.** W: 199 of 200 requests restored (the one that did not is the first, cold: 6984 prompt tokens, 7.2 s); the median hit restored 6904 of about 6976 prompt tokens (cached fraction 0.984). Median request latency 1.74 s warm against 6.97 s cold (a cold request prefills about 7k tokens and, with the store on, captures at every message boundary; the decode is the same 145 tokens on average).
- **Gate 1, W and C replies identical on every question: met.** 200 of 200 reply texts identical and 200 of 200 generated token-id sequences identical (`raw/W-vs-C.json`, `scripts/soak/gsm8k_prefix_cmp.py`). No non-identical reply, so no first-diverging token to report. A restore on this model (hybrid GatedDeltaNet and attention layers, CUDA) gave the same bits as a cold prefill with the store on for all 199 restored requests.
- **Gate 2, accuracy W against C, paired: met, trivially.** 196 of 200 right in both arms (0.980), the same 196 questions (right only in W: 0, right only in C: 0, exact McNemar p = 1.0). The four wrong ones (`gsm8k-00012`, `00087`, `00089`, `00119`) are wrong in both; `00087` and `00119` are replies cut by `max_tokens 512` (finish_reason `length`, counted wrong), the other two are wrong numbers. Because the replies are identical token for token, the paired result carries no information beyond the identity; the 0.980 itself is a property of the 32-shot prompt, not of the store.
- **Arm O (store off) was not run** (it was optional): window C had 7 minutes left when C finished (see Deviations). Not needed for the gates.

### Deviations (item 1)

- **C's client stalled once and was resumed.** At about 14:17, 30 requests into C, a network blip between this machine and spark2 (fleetq's own server was unreachable at the same time) left the client's TCP socket hung with no response, although the server had finished that request (server `requests_total` 30, nothing running, GPU idle). I paused the window script (SIGSTOP of the window shell), killed the hung client, and re-ran the client with a new `--resume` flag (added to the driver and committed) that keeps the 29 rows already written and sends only the 171 missing questions, then continued the window script. Consequences: C's `lookups` / `requests_total` are 201 (the 30th question went to the server twice; the first answer was lost with the socket), C's `wall_s` of 1220 s covers the resumed part only (the stall cost about 7 minutes), and the request timeout of the driver was lowered from 900 s to 180 s for the resumed run (the window-D run uses it too). All 200 C rows come from one server process and one cold store configuration, and all 200 compare against W.
- **Window C ran W and C only.** The planned optional arms were skipped by the window script's time guard: O (store off, estimate 14.7 min) and W2 (concurrency 2, estimate 6.4 min) did not fit the 428 s left. A concurrency-2 warm arm was run in window D instead (SW2).
- The build reused `~/intarsia-soak` on spark2 (incremental `cargo build`, same target dir) and so overwrote the previous lane's binary there (`restore-acc-8b95355529a8` replaces `ab35c-45e6ce56b023`); every arm of this lane ran the new binary.
- vLLM downtime of window C: 14:02:51 to 14:48:50 = 2759 s = 46.0 min (`/v1/models` 200 verified after `docker start`), under the 55 min cap. fleetq jobs: #9248 build, #9249 smoke test of the driver against production vLLM (3 requests, `--quiet gpu,cpu`), #9250 window C.

## Item 2: soak at the default host budget (window D, 2026-11-13, fleetq #9251)

`--max-batch 2 --prefix-store on` (no `host_mb`: the 8 GiB default, 8590 MB), K=2, replay2 interleaved over fresh seeds 31 to 45 (the previous soak used 11 to 20, K=3 and K=4 used 1 and 2), each replay recording its own trace from the server's replies; the plan is the interleave lane's (8 conversations x 6 turns, 8k to 33k tokens, thinking off, temperature 0, 160 max tokens). 15 replays in 32 min (15:05:47 to 15:37:51), 735 requests (15 x 48 plus one warm-up each). A sampler on the box wrote one row per minute (`raw/sample-S.tsv`, `scripts/soak/sampler.sh`, now also `intarsia_batch_paused_steps_total`); `memwatch.sh` (2 s MemAvailable, kill under 10 GB) never fired. The window ran an extra arm after the soak (SW2, below), so the sampler table is cut at the soak's end (15:37:51).

### TTFT, decode and cached fraction per replay

| replay | start min | requests | errors | TTFT sum s | TTFT p50 s | TTFT p90 s | decode tok/s median (bimodal: see doc) | decode tok/s mean | agg output tok/s | cached frac | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S31 | 0.0 | 48 | 0 | 70.9 | 1.42 | 1.67 | 38.2 | 39.6 | 56.2 | 0.732 | 131 |
| S32 | 2.2 | 48 | 0 | 76.9 | 1.44 | 2.42 | 42.0 | 43.6 | 59.1 | 0.732 | 127 |
| S33 | 4.4 | 48 | 0 | 68.8 | 1.38 | 1.57 | 40.6 | 41.5 | 57.7 | 0.732 | 126 |
| S34 | 6.5 | 48 | 0 | 72.7 | 1.38 | 2.22 | 41.9 | 42.2 | 59.0 | 0.732 | 126 |
| S35 | 8.6 | 48 | 0 | 71.6 | 1.41 | 2.11 | 41.8 | 41.8 | 59.7 | 0.732 | 128 |
| S36 | 10.8 | 48 | 0 | 66.6 | 1.39 | 1.52 | 39.8 | 40.2 | 58.4 | 0.732 | 127 |
| S37 | 12.9 | 48 | 0 | 70.4 | 1.37 | 1.77 | 41.6 | 42.3 | 58.6 | 0.732 | 127 |
| S38 | 15.0 | 48 | 0 | 67.4 | 1.40 | 1.58 | 40.2 | 40.7 | 58.7 | 0.732 | 128 |
| S39 | 17.2 | 48 | 0 | 70.5 | 1.41 | 1.88 | 40.6 | 41.8 | 58.7 | 0.732 | 126 |
| S40 | 19.3 | 48 | 0 | 68.3 | 1.40 | 1.58 | 39.9 | 40.9 | 58.2 | 0.732 | 126 |
| S41 | 21.4 | 48 | 0 | 70.9 | 1.41 | 2.04 | 39.8 | 42.2 | 58.1 | 0.732 | 125 |
| S42 | 23.5 | 48 | 0 | 70.3 | 1.42 | 1.64 | 40.4 | 42.1 | 59.0 | 0.732 | 127 |
| S43 | 25.7 | 48 | 0 | 67.5 | 1.39 | 1.54 | 40.2 | 40.7 | 59.1 | 0.732 | 128 |
| S44 | 27.8 | 48 | 0 | 71.1 | 1.41 | 1.89 | 40.5 | 42.1 | 58.5 | 0.732 | 125 |
| S45 | 29.9 | 48 | 0 | 73.4 | 1.35 | 2.43 | 41.6 | 42.7 | 58.7 | 0.732 | 126 |
- ttft_sum: first half mean 71.128, second half mean 69.907, min 66.616, max 76.940
- ttft_p50: first half mean 1.399, second half mean 1.398, min 1.351, max 1.438
- dec_p50: first half mean 40.842, second half mean 40.393, min 38.227, max 41.974
- dec_mean: first half mean 41.590, second half mean 41.652, min 39.562, max 43.630
- cached: first half mean 0.732, second half mean 0.732, min 0.732, max 0.732
- wall: first half mean 127.445, second half mean 126.197, min 124.919, max 131.032


(Decode tok/s per request is bimodal under concurrency, about 17 while the other stream prefills and 40 to 60 otherwise, so use TTFT sums and aggregate output per wall second for trends; same caveat as the previous soak.) There is no trend in any column: TTFT sum 71.1 s (first half) against 69.9 s (second half), TTFT p50 1.40 s in both, aggregate output 56 to 60 tok/s, wall 125 to 131 s per replay, cached fraction 0.732 in every replay. For comparison the stress soak at `host_mb=4096` (previous lane) had a cached fraction of 0.19, TTFT sums of 169 to 199 s and about 33 tok/s aggregate: the default budget holds the working set (8 conversations of up to 33k tokens), the stress budget did not.

### Memory, store and engine counters

sampler: 32 samples, 15:06:06 to 15:37:10
| signal | first 5 mean | last 5 mean | min | max | slope/min, all | slope/min, from sample 6 (store full) |
|---|---:|---:|---:|---:|---:|---:|
| mem_avail_mb | 78286.8 | 75183.0 | 74983.0 | 83893.0 | -91.75 | -19.07 |
| mem_used_mb | 46322.2 | 49426.0 | 40716.0 | 49626.0 | 91.75 | 19.08 |
| swap_used_mb | 490.0 | 490.0 | 490.0 | 490.0 | 0.00 | 0.00 |
| rss_mb | 162.0 | 162.0 | 162.0 | 162.0 | 0.00 | 0.00 |
| intarsia_kv_bytes_used | 896.5 | 765.3 | 336.7 | 1428.9 | -4.19 | -7.79 |
| intarsia_device_memory_free_bytes | 82090.9 | 78835.7 | 78625.8 | 87976.6 | -96.24 | -20.00 |
| intarsia_prefix_store_tier_bytes{tier="host"} | 7019.8 | 8351.4 | 1797.4 | 8587.9 | 40.99 | 3.01 |
| intarsia_prefix_store_entries | 93.4 | 109.6 | 29.0 | 139.0 | 0.44 | 0.20 |
| intarsia_prefix_store_tier_objects{tier="host"} | 928.2 | 1098.0 | 231.0 | 1290.0 | 5.72 | 0.04 |

- store host bytes peak 8588 MB against budget 8590 MB (within budget)
- lookups 721, hits 676, misses 45, host evictions 2760, rejected 0, inserts 2899
- requests 721, rejected 0, engine_failed 0, no_progress_s 0, finished length/stop 279/440
- first host eviction seen in sample 2 (of 32; 60 s apart)
- intarsia_batch_paused_steps_total (prefill pause default ran when > 0): 3130
- server gaps (batch_itl) total 108340: above 0.25 s 5, above 0.5 s 0

Reading, with the minute-by-minute series in `raw/sample-S.tsv`:

- **MemAvailable.** 83.9 GB at the first sample (minute 1; 87.1 GB right after the load), 78.6 GB at minute 2, 77.0 GB at minute 3, 75.9 GB at minute 4 (the store fills: host tier bytes 1.8, 7.8, 8.6 GB at minutes 1 to 3), then 75.9 to 75.2 GB for the rest of the run. The fitted slope over all samples (-92 MB/min) is the fill phase, not a drift. Fitted from minute 4 (15:10) it is -22 MB/min, from 15:15 -8 MB/min, and over the last 17 minutes (from 15:20) **-4.2 MB/min** (about -0.07 GB over the period, against 0.1 GB of peak-to-peak noise); the plateau is 75.2 GB, and the lowest 2 s sample inside the soak is 74.4 GB at 15:25:11. The previous lane's stress soak fitted -3.3 MB/min from its sample 6, the same size. Server RSS flat at 162 MB (CUDA allocations live outside RSS), swap flat at 490 MB (there before the run), `intarsia_device_memory_free_bytes` the same shape as MemAvailable, `intarsia_kv_bytes_used` 0.3 to 1.4 GB with no trend. The plateau is about 5 GB lower than the stress soak's (79.8 to 81 GB), consistent with a host tier 4.3 GB larger (8.6 against 4.3 GB). earlyoom was `active`; the lowest MemAvailable of the whole serve, 72.7 GB, was during SW2 (15:41), far above the 10 GB kill line.
- **Store.** Host tier bytes peaked at 8588 MB in the sampler rows and 8589 MB over the 15 per-replay scrapes, against a budget of 8590 MB (sampled peaks, not a continuous bound), 2866 host evictions (the first within 3 minutes), 2955 inserts, 735 lookups, 690 hits, 45 misses, `intarsia_prefix_store_rejected_total` 0 (it counts restores or loads that fail their checks, corrupt or foreign objects; the store has no counter for refused inserts, an over-budget capture just shows up as an eviction); entries 29 to 139, host objects 231 to 1290.
- **Engine gaps** (`intarsia_batch_itl_seconds`, no character weighting): 110,650 gaps over the soak, 1505 above 0.15 s (1.4 %), **5 above 0.25 s (0.005 %), 0 above 0.5 s**. All 5 are in the first two replays (3 by the end of S31, 2 more in S32; none afterwards). The stress soak of the previous lane (main before the pause default) had 3002 above 0.25 s of 74,026 (4.1 %); the 2-stream default-budget run of the interleave lane before the pause had 103 above 0.25 s of 7548 (one replay). So the pause default removes the long stalls at a store that holds the working set too.
- **`intarsia_batch_paused_steps_total`**: 223 after the first replay, 3179 at the end of the soak (about 212 per replay): the pause default ran. (It is 0 in the GSM8K arms by construction: a request that restores 6.9k tokens prefills a tail of about 70 rows, under the 768-row threshold, and at concurrency 1 nothing decodes beside a cold prefill.)
- Requests 735, `intarsia_requests_rejected_total` 0, `engine_failed` 0, `no_progress_s` 0, finished length / stop 282 / 453 (160-token cap). Counters in this list and in the summary come from the seed-45 scrape (`raw/serve-S.metrics.after-seed45`, taken when the last replay ended); the sampler table above stops 41 s earlier (15:37:10) and so shows 721 requests, 2760 evictions, 2899 inserts, 3130 paused steps and 279 / 440 finished.
- Server logs (`raw/serve-W.log`, `serve-C.log`, `serve-S.log`): no line containing warn, error, panic or fail.

### Verdict (item 2)

Pass on every criterion of the brief: no errors; MemAvailable settled after the store filled (the residual slope is within noise); store bytes within budget with evictions and no failed restore checks; TTFT, decode and cached fraction flat for 32 minutes; no engine gap above 0.5 s and five above 0.25 s, all in the first two replays; the pause default ran. This closes open item (3) of the previous lane (no soak at the default 8 GiB budget).

### SW2: the GSM8K shared-prefix arm on the soak's server (concurrency 2)

After the soak, on the same server (store full, 2866 evictions behind it, 32 minutes of other conversations), the GSM8K arm was run once more at concurrency 2 (`SW2`, the production stream count), as a separate observation next to the gate pair:

| arm | n | errors | correct | acc | truncated (512) | mean output tokens | prompt tokens median (min-max) | requests with cached tokens | cached frac | median cached tokens (hits) | store hits / misses / local_wins | tokens_saved | rejected (failed restore checks) | paused steps | median latency s | wall s | concurrency |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| W | 200 | 0 | 196 | 0.980 | 2 | 145.3 | 6976 (6944-7054) | 199 | 0.984 | 6904 | 199 / 1 / 0 | 1373896 | 0 | 0 | 1.744 | 388 | 1 |
| C | 200 | 0 | 196 | 0.980 | 2 | 145.3 | 6976 (6944-7054) | 0 | 0.0 | 0 | 0 / 201 / 0 | 0 | 0 | 0 | 6.972 | 1220 | 1 |
| SW2 | 200 | 0 | 196 | 0.980 | 2 | 145.3 | 6976 (6944-7054) | 198 | 0.9791 | 6904 | 198 / 2 / 0 | 1366992 | 0 | 0 | 2.365 | 266 | 2 |

SW2 against W (`raw/W-vs-SW2.json`): 200 of 200 reply texts and token-id sequences identical, 196 of 200 right in both (same questions). 198 of 200 requests restored; the 2 misses are the two requests that started side by side on a store that had never seen this prefix (the first of them captured it). Median latency 2.37 s at concurrency 2 against 1.74 s at 1 (two streams share the GPU), wall 266 s for 200 requests. The restores still give the cold-with-store-on bits after the store served 735 other requests and evicted 2866 objects; `local_wins` 0. SW2's `paused steps` column is 0 for the reason given above.

### Deviations (item 2)

- The window ran SW2 after the soak (not in the brief), inside the 55 min cap; the soak table is cut at the soak's end for that reason.
- Window D was started with `fleetq run --at 15:05`, 16.2 min after vLLM came back from window C (14:48:50), so the 15 min gap held. vLLM downtime of window D: 15:05:04 to 15:45:48 = 2444 s = 40.7 min (`/v1/models` 200 verified after `docker start`); window C was 46.0 min; two windows, each under 55 min. fleetq job #9251. No job of this lane is held (`fleetq status` after the window: #9250 and #9251 done, spark2 vLLM up).

## Fairness and limits

- One server run per arm; greedy decoding, so a repeat on the same server state would give the same bits.
- The W-against-C identity is shown for this prompt family (a 6.9k-token shared prefix cut at message boundaries, 199 restores at concurrency 1 and an unchanged tail of about 70 tokens per question). It says nothing about restores of other lengths, multi-turn restores deep into a conversation (the soak replays below restore up to 28k tokens but record no reference bits), or restores from the disk tier (not configured).
- `intarsia_batch_paused_steps_total` is 0 in W, C and SW2 (tails of about 70 rows are under the 768-row pause threshold, and at concurrency 1 nothing decodes beside a cold prefill), so bit-identity after a restore is **not** shown with prefill parks active. The soak ran restores and pauses together (690 hits, 3179 paused steps, 0 errors) but has no reference bits to compare with.
- Accuracy after a restore against a store-off run (O) was not measured; the design claim, and the tests in `crates/intarsia-models/tests/prefix_store.rs`, are about identity with a cold prefill with the store on.
- The previous lane's observation that the store changes greedy bits against store off on CUDA (73 of 200 replies different, accuracy unchanged) is not re-measured here.

## Raw results and scripts

- `docs/results/2026-11-13-restore-acc/raw/`: `gsm8k-inputs.json` (frozen questions, golds and shots), `W.jsonl` / `C.jsonl` (one row per question: reply, token ids, finish reason, prompt, completion and cached tokens, latency, score), `*.summary.json`, `*.storediff.json`, `W-vs-C.json`, `window-c.log` / `window-d.log` and `.out`, server logs and metrics scrapes (`serve-*`), memory samples (`mem-*.txt`, 2 s MemAvailable) and the 60 s sampler (`sample-*.tsv`), the soak replays (`S31` to `S45`: `*-interleaved-seed*-k2.jsonl` / `.summary.json`, `soak-trace-seed*.json`, `replay-plan.json` copied from the previous lane).
- Scripts `scripts/soak/`: `build.sh`, `gsm8k_dump.py`, `gsm8k_prefix.py`, `gsm8k_prefix_cmp.py`, `restore_table.py`, `metric_diff.py`, `window_c.sh`, `window_d.sh`; extended from the previous lane: `lib.sh` (`SOAK_OUT`), `sampler.sh` (+ `intarsia_batch_paused_steps_total`), `soak_table.py` (+ paused steps, optional sampler cut-off time).
- Tables: `python3 scripts/soak/restore_table.py docs/results/2026-11-13-restore-acc/raw W C SW2`; `python3 scripts/soak/soak_table.py docs/results/2026-11-13-restore-acc/raw soak <HH:MM:SS cut-off>`; `python3 scripts/soak/gsm8k_prefix_cmp.py raw/W.jsonl raw/C.jsonl --label-a W --label-b C`.
