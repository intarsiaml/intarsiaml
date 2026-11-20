# ab35d: cutover scoreboard, Qwen3.6-35B-A3B NVFP4, intarsiaml at origin/main `25489d6aa` vs the production vLLM (spark2)

Lane `ab35d`, branch `ab35d` from origin/main `25489d6aa` (merge of PR #464), 2026-11-19 21:37 to 23:30 EST, fleetq owner `claude-overnight-ab35d`. Measurement only: no engine code changed, no default changed. It refreshes the cutover scoreboard of `docs/results/2026-11-12-ab35b-cutover-followup.md` (section "Cutover blockers, ranked", 7 blockers) after the fixes that landed since: tool-result 400 fix (#349, #351), store settle and pre-faulted host buffers (#352), 2-stream prefix store (#353), seeded sampled drafting in the batched engine, MTP with the prefix store, shortest-prefill-first (#457), prefix herd (#458), side-stream eager (#453) and CUDA verify-attention rows (#464). A is the production vLLM on spark2 (never touched, measured idle), B is intarsiaml in its shipping configuration, run beside it on the same box.

## Summary

**Of the seven ab35b blockers, four are closed (1 tool-result turns, 2 prefix reuse with two streams, 3 the store's decode stall, 4 MTP together with reuse), two are partly closed (5 sampled decode, 6 load and soak) and one is unchanged and explained, not closed (7 output differences).** On every cell measured, intarsiaml at `25489d6aa` in its shipping configuration is at parity with or ahead of the production vLLM, with two exceptions found today: uncached prompt prefill with the prefix store on is 11 to 21 % slower than vLLM (TTFT +12 to +27 %; the same binary with the store off is 4 to 8 % faster), and sampled decode at 64k is 17 % slower (bf16 KV; with `--kv-dtype fp8` it is 8 % faster). Nothing measured here points at a correctness problem. What a cutover still needs is evidence of operation (hours, real traffic, thinking-on quality), not more speed work. No default changed and no cutover is recommended beyond what the numbers show.

Headline numbers, B / A on the same box, same client, same prompts (ratios of means; ABBA unless stated):

| cell | A vLLM | B intarsiaml | B / A |
|---|---|---|---|
| tool calls and receipt turns (3 tools, stream and non-stream, thinking off and default) | all pass except `get_weather` calls the model skipped (3/5) | all pass (receipts 0 of 28 before) | closed |
| 2-stream multi-turn replay, 4 seeds: summed TTFT | 75.7 s (69.5 to 89.6) | 75.6 s (69.8 to 79.4) | 1.00 (parity) |
| same: wall time / combined output / cached fraction | 122.9 s / 60.9 tok/s / 0.695 | 117.1 s / 63.9 tok/s / 0.732 | 0.95 / 1.05 / - |
| 1-stream interleaved replay (2 seeds): decode per request / wall | 75.9 tok/s / 158 s | 117.4 tok/s / 125 s | 1.55 / 0.79 |
| greedy decode 4k / 16k / 32k | 78.2 / 75.6 / 72.4 tok/s | 119.9 / 99.7 / 81.8 | 1.53 / 1.32 / 1.13 |
| greedy uncached TTFT 4k / 16k / 32k | 0.635 / 2.61 / 6.10 s | 0.775 / 3.31 / 6.85 | 1.22 / 1.27 / 1.12 (store off: 0.93 / 0.96 / 0.94) |
| sampled T=0.7 decode 4k / 32k / 64k | 77.7 / 71.7 / 65.3 tok/s | 98.7 / 70.6 / 54.2 | 1.27 / 0.985 / 0.83 (fp8 KV at 32k / 64k: 1.15 / 1.08) |
| GSM8K n=200, thinking off | 0.640 and 0.575 (same requests twice) | 0.715 | +0.075 (noise) and +0.140 (better); style effect |
| 25 min soak, 624 requests, mixed greedy and T=1.0, 2 streams | not run (idle) | 0 errors, store full and stable, 96 % of tokens from MTP steps | - |

## Setup

### Arms and builds

- **B (intarsiaml, the shipping config)**: origin/main `25489d6aa`, binary stamp `ab35d-25489d6aa2f6`, built on spark2 in `~/intarsiaml-ab35d` (fleetq #10320, `--load cpu`, `CARGO_BUILD_JOBS=3 MAX_JOBS=3 FLASHINFER_NVCC_THREADS=1`, `cargo build --release --locked --features cuda -p intarsia-cli`, 6 min 15 s, vLLM up). Command line, identical in every window (`scripts/ab35d/window.sh`):

  ```
  ~/intarsiaml-ab35d/bin/intarsia serve --backend cuda --model <HF snapshot 1355db6a of nvidia/Qwen3.6-35B-A3B-NVFP4> \
      --port 8191 --host 0.0.0.0 --served-model-name guidance --max-seq 131072 --mem-budget 40 --profile-every 0 \
      --max-batch 2 --prefix-store on
  ```

  No environment variables. `--max-seq 131072 --mem-budget 40 --profile-every 0` are what every earlier ab35 / soak lane used (131072 is vLLM's context); `--max-batch 2 --prefix-store on` is the shipping shape of the soak2 doc. Everything else is the default. What the startup banner (`raw/serve-w1.log`) confirms is active: `--spec mtp --spec-k 2` (recipe default), **batched MTP N=2** (a sequence drafts while at most 2 sequences decode; soak2 ran N=1), MTP together with the store (`--spec mtp drafts on store requests`), **prefix herd on** (default since herdflip), batched MTP on sampled requests (seed-exact), `verify attention rows: captured: engaged` (#464), bf16 KV (the checkpoint declares fp8 KV, the engine runs bf16: banner note), prefill chunk 2048, sampling defaults from the checkpoint (`temperature=1 top_p=0.95 top_k=20`). Shortest-prefill-first (`INTARSIA_PREFILL_ORDER`, default `shortest`) and side-stream eager (`INTARSIA_SIDE_EAGER`, default on) have no banner line; their defaults are read from `docs/tuning.md`. The banner line `sampled requests (temperature > 0): speculation Match (default; single sequence, one rank)` is stale for the batched engine (the next banner lines say sampled requests draft there).
- **A (production vLLM)**: container `missioncntrl-guidance` (fleetq role `retro`), image `missioncntrl/vllm-openai:65b7662d`, vLLM `0.26.1rc1.dev602+g65b7662d3`, started 20:11 EST tonight (after the AC power-cycle), served on the Spark's tailscale address `spark-fa00:8000`, model names `guidance` and `capcom`. From `docker inspect`: `vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --host 100.96.95.89 --port 8000 --served-model-name guidance capcom --quantization modelopt_fp4 --max-model-len 131072 --max-num-seqs 2 --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --kv-cache-memory-bytes 8589934592 --gpu-memory-utilization 0.32 --enable-prefix-caching --enable-prompt-tokens-details --enable-chunked-prefill --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3`, env `VLLM_USE_FLASHINFER_MOE_FP4=0`, `VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel`, no speculative config. Same arguments as ab35 / ab35b. It was not stopped, restarted or reconfigured; no `--takes-down` anywhere.
- **Box**: spark2 (GB10, driver 580.178.04), vLLM resident (about 38 GB), B's pool `--mem-budget 40` plus its 8 GiB host store tier beside it. Lowest MemAvailable during any B window 35.7 GB (kill line 15 GB, earlyoom active), 0 watchdog kills, 0 warn / error lines in any serve log.
- **Client**: this Mac (endor) over the network to `spark-fa00:8000` (A) and `spark-fa00:8191` (B), the same path for both arms. TTFT therefore includes the network round trip on both.

### Windows (fleetq jobs, all `--quiet gpu,cpu`, one B server per window, vLLM up; B and A alternate request streams, never concurrent)

| window | job | cells | note |
|---|---|---|---|
| w1 | #10321 | tools, 2-stream replay | 21:43 to 21:57 |
| w2 | #10322 | greedy speed, sampled decode | 21:57 to 22:10 |
| w3 | #10323 | GSM8K n=200 (A, B, A) | 22:10 to 22:19 |
| w4 | #10324 | soak, 25 min, B only | 22:19 to 22:46 |
| w5 | #10325 | control: greedy speed with the store off | 22:47 to 22:53 |
| w6 | #10326 | control: sampled decode with fp8 KV | **aborted, no data**: I edited the window script while the job was starting and bash hit a syntax error; its files (`window-w6.log`, `serve-w6.*`) are kept as is, w8 is the rerun |
| w7 | #10327 | control: 1-stream interleaved replay (blocker 3's original shape) | 22:55 to 23:06 |
| w8 | #10328 | control: sampled decode with fp8 KV, A vs B | 23:06 to 23:14 |
| w9 | #10329 | 2 more replay seed pairs (seeds 3 and 2), A B B A | 23:15 to 23:25 |

Tools: localeval 0.4.0 (`/opt/code/deepeval`, git `2c5524c`) for the speed sweep and GSM8K (run ids in `raw/index.txt`); localeval has no multi-turn, sampled or tool mode, so the replay (`scripts/interleave/replay2.py`), sampled (`scripts/ab35b/sampled.py`) and tool (`scripts/ab35d/toolsmoke.py`, a copy of ab35c's with `--thinking default`) cells use the same scripts as the earlier lanes, which import localeval's streaming `Client` (same TTFT, decode tok/s and `cached_tokens` definitions as `localeval speed`). Speed uses `--thinking off --structured prompt --sizes 4096,16384,32768 --gen 256 --reps 3` (1 warmup per run, nonce first in every prompt so no arm gets a cache hit); each run is one speed sweep, order A1 B1 B2 A2 (ABBA).

### Replay inputs

`scripts/interleave/replay2.py`, `--streams 2 --order interleaved`, 8 conversations x 6 turns, prompts 8k to 33k tokens, 160 max tokens, temperature 0, thinking off, about 989k prompt tokens per run. Plan `replay-plan.json` and the seed 0 trace are the files of ab35c window 3 and the interleave / mtp-store lanes (md5 `4e2e721a...` and `ca913d2a...`, identical in all of them); the seed 1 trace is the one the cutover-soak lane recorded with the same plan. Both traces hold vLLM's recorded answers, so A and B saw byte-identical requests. (My first copy of the plan came from ab35b, which has a different plan; I replaced it before any run after checking prompt totals against window 3: 988,922 tokens there, 988,463 to 988,821 here.) Four seeds in two windows, each ABBA: w1 A(s0) B(s0) B(s1) A(s1), w9 A(s3) B(s3) B(s2) A(s2). Seed 2 uses the cutover-soak lane's trace; seed 3 had no trace, so its first step (A) recorded vLLM's live answers into `replay-trace-seed3.json` and B replayed them. No arm replays prompts its own cache or store has seen of the same seed; a later seed follows an earlier one on the same server (a store that is already full, as in production).

### Clock guard and shared-box check (lane rules of 2026-11-19)

Every cell is bracketed by marks; the box's `nvidia-smi` was sampled every 2 s for the whole window and the "SW Power Capping" counter read before and after each cell (`scripts/ab35d/window.sh`, `guard_end`, which aborts the window on SM < 2000 MHz at > 90 % util for 3 or more samples, a counter growth over 1 s, or any vLLM request completing during a B cell). **No cell tripped it.** Across all cells: minimum SM clock 2333 MHz (B, under load; A 2385 MHz), mean 2372 to 2402 MHz, SW power cap delta 0 us in every cell, maximum power 93 W, temperature 40 to 65 C. The vLLM request counters moved by exactly our own request count during every A cell (the replay and speed counts are the cell's requests plus one probe) and by 0 during every B cell, so no production request ran on the GPU during our timing. The table is under "Clock guard per cell".

## Cell 1: tool calls and receipt turns (blocker 1)

`scripts/ab35d/toolsmoke.py`, three tools (nested object and array arguments in `create_event`), `tool_choice` auto, temperature 0; each call must return exactly one `tool_calls` entry with the right name, `finish_reason` `tool_calls`, strict-JSON `arguments` with the right values and no raw markup in `content`; then the tool result goes back as a `role: tool` message (the assistant turn carrying the OpenAI-standard string `arguments`) and the final reply must contain the result value. 5 non-stream and 2 stream calls per tool with thinking off; 3 and 2 with the stock client default (thinking on, reasoning split); `response_format` json_schema 5 (3) times. Each engine with its own parser (vLLM `qwen3_xml`, intarsiaml default).

| check (calls / receipts, non-stream; stream calls / receipts) | A vLLM, thinking off | B, thinking off | A, thinking default | B, thinking default |
|---|---|---|---|---|
| `add_numbers` | 5/5, 5/5; 2/2, 2/2 | 5/5, 5/5; 2/2, 2/2 | 3/3, 3/3; 2/2, 2/2 | 3/3, 3/3; 2/2, 2/2 |
| `create_event` | 5/5, 5/5; 2/2, 2/2 | 5/5, 5/5; 2/2, 2/2 | 3/3, 3/3; 2/2, 2/2 | 3/3, 3/3; 2/2, 2/2 |
| `get_weather` | 3/5, 3/5; 0/2, 0/2 (the model answered in plain text "15 C" without a call) | 5/5, 5/5; 2/2, 2/2 | 3/3, 3/3; 2/2, 2/2 | 3/3, 3/3; 2/2, 2/2 |
| `response_format` json_schema | 5/5 | 5/5 | 3/3 | 3/3 |

B passes every call and every receipt turn, stream and non-stream (ab35b's build before #349: every receipt turn HTTP 400, 28 of 28). The `get_weather` misses on A are the model answering without a tool on this prompt (ab35b saw the same on both engines, 0/5 to 1/5, not deterministic); that B got 5/5 this time is the same non-determinism, not a parser difference. Raw: `raw/{A,B}-tools*.{out,json}`. Small n (5 + 2 per tool).

Also measured while B was up (window w7, `raw/window-w7.log`): B answers a request for model `capcom` (the second name vLLM serves) and for an unknown name alike, replying with its served name `guidance`; its `/v1/models` lists only `guidance`. So the `capcom` alias used by mission control needs nothing on B's side.

## Cell 2: two concurrent multi-turn conversations with prefix reuse (blockers 2, 3, 4)

| arm | seed | errors | TTFT sum s | TTFT p50 / p90 s | cached frac | decode tok/s per stream (median) | combined out tok/s | wall s | client ITL p99 ms median / max | server gaps >0.15 / >0.3 (vLLM) or >0.25 (B) / >0.5 of N |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 0 | 0 | 89.6 | 2.11 / 2.50 | 0.695 | 54.3 | 61.9 | 122.4 | 212 / 1890 | 60 / 34 / 20 of 7537 |
| A | 1 | 0 | 73.7 | 1.37 / 2.14 | 0.696 | 40.9 | 60.4 | 121.4 | 356 / 1914 | 96 / 64 / 35 of 7297 |
| A | 2 | 0 | 69.5 | 1.34 / 2.20 | 0.695 | 41.4 | 61.0 | 123.8 | 373 / 901 | 108 / 72 / 40 of 7507 |
| A | 3 | 0 | 70.0 | 1.35 / 2.14 | 0.693 | 41.0 | 60.2 | 123.8 | 381 / 1445 | 108 / 72 / 40 of 7406 |
| B | 0 | 0 | 79.4 | 1.57 / 2.03 | 0.732 | 46.0 | 62.8 | 120.9 | 212 / 1354 | 144 / 17 / 2 of 7548 |
| B | 1 | 0 | 75.1 | 1.48 / 2.32 | 0.732 | 47.8 | 65.1 | 112.4 | 292 / 1142 | 102 / 1 / 0 of 7277 |
| B | 2 | 0 | 69.8 | 1.41 / 1.61 | 0.732 | 46.8 | 65.8 | 114.6 | 225 / 814 | 124 / 1 / 0 of 7505 |
| B | 3 | 0 | 78.0 | 1.53 / 1.95 | 0.732 | 45.1 | 61.9 | 120.6 | 256 / 933 | 152 / 19 / 2 of 7423 |

Per-turn TTFT median s (cached tokens median), the same four runs:

| turn | A s0 | A s1 | A s2 | A s3 | B s0 | B s1 | B s2 | B s3 |
|---|---|---|---|---|---|---|---|---|
| 1 | 2.15 (0) | 2.22 (0) | 2.26 (0) | 2.22 (0) | 1.72 (44) | 1.56 (44) | 1.55 (44) | 1.92 (44) |
| 2 | 1.20 (6288) | 1.21 (6288) | 1.19 (6288) | 1.20 (6288) | 1.36 (8041) | 1.20 (8038) | 1.21 (8046) | 1.39 (8043) |
| 3 | 1.15 (12576) | 1.12 (12576) | 1.12 (12576) | 1.12 (12576) | 1.49 (13063) | 1.30 (13071) | 1.30 (13074) | 1.47 (13066) |
| 4 | 2.46 (16768) | 1.36 (16768) | 1.35 (16768) | 1.35 (16768) | 1.75 (18088) | 1.75 (18085) | 1.39 (18093) | 1.53 (18091) |
| 5 | 2.09 (23056) | 1.23 (23056) | 1.25 (23056) | 1.24 (23056) | 1.57 (23111) | 1.50 (23104) | 1.48 (23113) | 1.51 (23116) |
| 6 | 2.48 (27248) | 1.50 (27248) | 1.50 (27248) | 1.55 (27248) | 1.60 (28135) | 1.56 (28127) | 1.57 (28131) | 1.55 (28142) |

B's MTP counters over each B replay (`/metrics` diff of `intarsia_spec_*`):

| cell | spec steps | proposed | accepted | acceptance | completion tokens | share of tokens made in spec steps (steps+accepted)/tokens |
|---|---|---|---|---|---|---|
| replay-B-s0 | 2714 | 5420 | 4609 | 85.0 % | 7589 | 0.965 |
| replay-B-s1 | 2646 | 5285 | 4420 | 83.6 % | 7318 | 0.966 |
| replay-B-s2 | 2713 | 5413 | 4558 | 84.2 % | 7546 | 0.964 |
| replay-B-s3 | 2686 | 5359 | 4508 | 84.1 % | 7464 | 0.964 |

How to read it:

- **Reuse (blocker 2).** B's cached fraction is 0.732 against A's 0.695 (about 724k of 988k prompt tokens served from the store against 687k from vLLM's cache). Summed TTFT over a replay: B 79.4, 75.1, 69.8 and 78.0 s, A 89.6, 73.7, 69.5 and 70.0 s; means 75.6 and 75.7 s, **parity**. A's spread (69.5 to 89.6; the 89.6 is the first replay after the tool cell, turns 4 to 6 at 2.1 to 2.5 s against 1.2 to 1.6 s later) is as wide as B's (69.8 to 79.4). Per turn, B is faster on the first (uncached, 8k) turn (1.55 to 1.92 s against 2.15 to 2.26 s) and A is faster on the mid turns that restore a prefix (turns 2 and 3: 1.12 to 1.21 s against 1.20 to 1.49 s; turns 4 to 6 1.2 to 1.6 s against 1.4 to 1.75 s), by 0.1 to 0.3 s. Wall time B 120.9, 112.4, 114.6, 120.6 s against A 122.4, 121.4, 123.8, 123.8 s (B 4.7 % shorter on the mean). ab35b's shipped path had 0 cached tokens and 2.7x the TTFT sum; ab35c window 3's batched store arm read 73.4 s against vLLM's 77.8 s, the same picture.
- **Decode under interleaving (blocker 3).** Per-stream decode medians: B 46.0, 47.8, 46.8, 45.1 tok/s (mean 46.4), A 54.3, 40.9, 41.4, 41.0 (per-request decode at two streams is bimodal, about 17 tok/s while the other stream prefills and 40 to 60 otherwise; the 54.3 is the same first replay). Combined output B 62.8, 65.1, 65.8, 61.9 against A 61.9, 60.4, 61.0, 60.2 tok/s (B +4.9 % on the mean). The old signature (decode 41 to 51 against 68 to 79 plain) is gone; see the 1-stream replay below for that exact shape.
- **Tail.** Client ITL p99 (localeval's character-weighted figure; it overstates a ~330 ms gap as ~1.4 s, see ab35c window 3) median 212 to 292 ms on B against 212 to 381 ms on A. Server-side gap counts use different definitions (vLLM: its per-token inter-token-latency histogram; B: `intarsia_batch_itl_seconds`, the gap between decode steps), so only the order of magnitude compares: gaps above 0.5 s are 2, 0, 0 and 2 of about 7.4k on B and 20, 35, 40 and 40 on A.
- **MTP with the store (blocker 4).** B drafts on store requests: acceptance 83.6 to 85.0 %, and 96.4 to 96.6 % of all generated tokens come out of speculative steps at 2 streams (batched MTP N=2). Soak2, at N=1, drafted for about 10 % of tokens. The speed effect is in the next cells.

## Cell 2b: the same replay with one stream, store on (blocker 3's original shape, window w7)

| arm | seed | errors | TTFT sum s | TTFT p50 / p90 s | cached frac | decode tok/s (median, mean) | combined out tok/s | wall s | server gaps >0.15 / >0.3 (vLLM) or >0.25 (B) / >0.5 of N |
|---|---|---|---|---|---|---|---|---|---|
| A | 0 | 0 | 58.6 | 1.18 / 1.43 | 0.695 | 75.8, 75.3 | 47.5 | 159.8 | 0 / 0 / 0 of 7545 |
| A | 1 | 0 | 58.7 | 1.18 / 1.42 | 0.696 | 75.9, 75.7 | 47.0 | 156.4 | 0 / 0 / 0 of 7312 |
| B (raw name N) | 0 | 0 | 63.8 | 1.32 / 1.52 | 0.732 | 118.2, 117.2 | 58.8 | 129.1 | 0 / 0 / 0 of 7548 |
| B (raw name N) | 1 | 0 | 58.1 | 1.26 / 1.35 | 0.732 | 116.6, 117.2 | 60.4 | 121.1 | 0 / 0 / 0 of 7277 |

Same plan and seed traces, one request in flight at a time (`--streams 1`), A B B A on seeds 0 1 1 0, B with the store on (the raw files call B's arm `N` in this control script). This is the shape in which ab35b measured the store stall (decode 41 to 51 tok/s interleaved against 68 to 79 plain; vLLM 70 to 76):

- Decode per request: **B 118.2 / 116.6 tok/s against A 75.8 / 75.9** (x1.55; MTP on, store on, interleaved conversations): the stall is gone and MTP is back. Summed TTFT 63.8 and 58.1 s against 58.6 and 58.7 s (+9 % and -1 %: parity inside A's run to run spread of the 2 stream replays), cached fraction 0.732 against 0.695, wall 129 and 121 s against 160 and 156 s (B 19 to 23 % shorter, from the decode rate), combined output 58.8 and 60.4 tok/s against 47.5 and 47.0.


## Cell 3: single-stream greedy decode and prefill, 4k / 16k / 32k

decode tok/s: median of the 3 reps per run (min-max), runs in ABBA order

| size (real prompt tokens) | A1-speed | B1-speed | B2-speed | A2-speed | A mean | arm mean | arm / A |
|---|---|---|---|---|---|---|---|
| 4096 (4091) | 77.8 (77.8-78.6) | 119.3 (117.3-121.4) | 120.4 (119.5-123.9) | 78.6 (77.4-78.7) | 78.2 | 119.9 | 1.533 |
| 16384 (16680) | 75.8 (75.5-75.9) | 98.9 (98.3-105.7) | 100.5 (98.3-100.6) | 75.4 (75.2-75.7) | 75.6 | 99.7 | 1.319 |
| 32768 (33925) | 72.4 (72.4-73.8) | 81.0 (79.6-81.5) | 82.6 (79.5-87.4) | 72.4 (71.8-72.8) | 72.4 | 81.8 | 1.130 |

TTFT s: median of the 3 reps per run (min-max), runs in ABBA order

| size (real prompt tokens) | A1-speed | B1-speed | B2-speed | A2-speed | A mean | arm mean | arm / A |
|---|---|---|---|---|---|---|---|
| 4096 (4091) | 0.636 (0.634-0.637) | 0.808 (0.761-0.824) | 0.742 (0.722-0.772) | 0.635 (0.633-0.642) | 0.635 | 0.775 | 1.220 |
| 16384 (16680) | 2.611 (2.607-2.616) | 3.353 (2.948-3.394) | 3.267 (2.899-3.286) | 2.608 (2.605-2.616) | 2.610 | 3.310 | 1.268 |
| 32768 (33925) | 6.110 (6.108-6.201) | 7.290 (7.225-7.297) | 6.418 (6.350-6.683) | 6.094 (6.092-6.112) | 6.102 | 6.854 | 1.123 |

prefill tok/s (uncached prompt tokens / TTFT): median of the 3 reps per run (min-max), runs in ABBA order

| size (real prompt tokens) | A1-speed | B1-speed | B2-speed | A2-speed | A mean | arm mean | arm / A |
|---|---|---|---|---|---|---|---|
| 4096 (4091) | 6430 (6428-6453) | 5058 (4963-5376) | 5512 (5298-5661) | 6441 (6376-6456) | 6435 | 5285 | 0.821 |
| 16384 (16680) | 6388 (6377-6397) | 4967 (4907-5650) | 5098 (5068-5745) | 6386 (6367-6391) | 6387 | 5033 | 0.788 |
| 32768 (33925) | 5552 (5471-5554) | 4650 (4646-4693) | 5279 (5070-5335) | 5564 (5547-5565) | 5558 | 4965 | 0.893 |

- **Decode: B ahead at every size** (MTP): x1.53 at 4k, x1.32 at 16k, x1.13 at 32k, outside the 10 % noise line at 4k and 16k and just over at 32k. A's two runs agree within 1 % (77.8 / 78.6, 75.8 / 75.4, 72.4 / 72.4).
- **Prefill and TTFT: B behind, 11 to 21 % on prefill throughput** (TTFT +22 / +27 / +12 %). This is new against ab35 (B-default, no store: 4k TTFT 0.59 vs 0.63 s, 16k 2.51 vs 2.61 s, ahead). B's own two runs also differ at 32k (4650 / 5279 tok/s), more than A's.
- **Cause narrowed with a control (window w5, same binary, `--max-batch 2` without `--prefix-store`, arm N):**

decode tok/s: median of the 3 reps per run (min-max), runs in ABBA order

| size (real prompt tokens) | ctl-A1-speed | ctl-N1-speed | ctl-N2-speed | ctl-A2-speed | A mean | arm mean | arm / A |
|---|---|---|---|---|---|---|---|
| 4096 (4089) | 78.3 (76.4-80.7) | 120.1 (118.1-121.0) | 119.8 (119.5-120.9) | 78.9 (78.7-79.1) | 78.6 | 119.9 | 1.526 |
| 16384 (16677) | 76.0 (76.0-76.5) | 105.2 (99.1-106.0) | 100.6 (98.6-102.5) | 75.5 (75.2-76.0) | 75.8 | 102.9 | 1.358 |
| 32768 (33928) | 72.0 (71.9-72.3) | 81.7 (80.4-82.3) | 80.8 (76.8-81.5) | 72.4 (71.3-72.7) | 72.2 | 81.3 | 1.125 |

TTFT s: median of the 3 reps per run (min-max), runs in ABBA order

| size (real prompt tokens) | ctl-A1-speed | ctl-N1-speed | ctl-N2-speed | ctl-A2-speed | A mean | arm mean | arm / A |
|---|---|---|---|---|---|---|---|
| 4096 (4089) | 0.639 (0.639-0.705) | 0.592 (0.587-0.651) | 0.589 (0.587-0.590) | 0.635 (0.635-0.696) | 0.637 | 0.591 | 0.927 |
| 16384 (16677) | 2.623 (2.616-2.655) | 2.522 (2.522-2.593) | 2.510 (2.508-2.519) | 2.620 (2.612-2.623) | 2.622 | 2.516 | 0.960 |
| 32768 (33928) | 6.102 (6.101-6.110) | 5.767 (5.702-5.771) | 5.704 (5.686-5.767) | 6.094 (6.089-6.096) | 6.098 | 5.736 | 0.941 |

prefill tok/s (uncached prompt tokens / TTFT): median of the 3 reps per run (min-max), runs in ABBA order

| size (real prompt tokens) | ctl-A1-speed | ctl-N1-speed | ctl-N2-speed | ctl-A2-speed | A mean | arm mean | arm / A |
|---|---|---|---|---|---|---|---|
| 4096 (4089) | 6394 (5802-6403) | 6909 (6284-6966) | 6944 (6938-6979) | 6438 (5882-6444) | 6416 | 6926 | 1.080 |
| 16384 (16677) | 6359 (6282-6374) | 6603 (6425-6605) | 6644 (6622-6650) | 6358 (6351-6377) | 6358 | 6623 | 1.042 |
| 32768 (33928) | 5560 (5553-5561) | 5874 (5871-5942) | 5944 (5879-5963) | 5559 (5557-5565) | 5560 | 5909 | 1.063 |

  With the store off the same binary is ahead of vLLM again (prefill x1.08 / x1.04 / x1.06, TTFT 8, 4 and 6 % lower), and decode is unchanged (x1.53 / x1.36 / x1.13). So the 11 to 21 % prefill loss belongs to running with the prefix store on, in this back-to-back cold-prompt pattern (every request is a new prompt, so each one pays store capture and write: over the 12 requests of window w2's B1 speed run the store inserted 6.7 GB, 20.5 s of `insert_seconds_total` and 4.2 s of settle, `snap/speed-B1.b.*.gz`). Where in the capture path the time goes is **not isolated** here (no engine change allowed); the replay cell (turn 1 TTFT: B 1.56 to 1.72 s against A's 2.15 to 2.22 s at 2 streams) shows no such loss when the other stream's work overlaps it.

## Cell 4: sampled decode, T=0.7, 4k / 32k / 64k (blocker 5)

`scripts/ab35b/sampled.py` (localeval `speed` is greedy only): temperature 0.7, top_p 0.95, nonce prompt per rep, exactly 256 tokens, 3 reps, single stream. B's `--max-batch 2` engine now drafts on sampled requests.

decode tok/s: median of the 3 reps per run (reps), runs in ABBA order

| size (real prompt tokens) | A1 | B1 | B2 | A2 | A mean | arm mean | arm / A |
|---|---|---|---|---|---|---|---|
| 4096 (4069) | 77.7 (76.0, 77.7, 80.3) | 99.0 (101.3, 98.4, 99.0) | 98.4 (111.1, 92.4, 98.4) | 77.6 (76.9, 79.3, 77.6) | 77.7 | 98.7 | 1.270 |
| 32768 (33921) | 71.7 (71.6, 71.7, 72.2) | 72.3 (79.5, 72.3, 63.1) | 69.0 (69.3, 69.0, 67.0) | 71.8 (71.8, 72.0, 71.6) | 71.7 | 70.6 | 0.985 |
| 65536 (68961) | 65.2 (65.2, 65.6, 65.2) | 53.8 (51.7, 53.8, 56.2) | 54.7 (55.9, 54.7, 53.9) | 65.3 (67.2, 65.3, 65.0) | 65.3 | 54.2 | 0.831 |

TTFT s: median of the 3 reps per run (reps), runs in ABBA order

| size (real prompt tokens) | A1 | B1 | B2 | A2 | A mean | arm mean | arm / A |
|---|---|---|---|---|---|---|---|
| 4096 (4069) | 0.64 (0.63-0.74) | 0.72 (0.68-0.74) | 0.67 (0.67-0.69) | 0.64 (0.63-0.69) | 0.64 | 0.70 | 1.091 |
| 32768 (33921) | 6.11 (6.09-6.12) | 6.45 (6.41-6.45) | 6.28 (6.26-6.36) | 6.11 (6.08-6.12) | 6.11 | 6.36 | 1.042 |
| 65536 (68961) | 15.88 (15.85-15.90) | 15.91 (15.77-16.02) | 15.70 (15.68-15.84) | 15.87 (15.86-15.97) | 15.87 | 15.81 | 0.996 |

B MTP acceptance on the sampled requests (accepted / proposed from the /metrics diff over the run's 3 reps):

- B1: 4096: 59.5 % (2.189 tokens per verify), 32768: 60.5 % (2.209 tokens per verify), 65536: 57.8 % (2.155 tokens per verify)
- B2: 4096: 61.5 % (2.224 tokens per verify), 32768: 54.0 % (2.076 tokens per verify), 65536: 60.2 % (2.202 tokens per verify)

- **4k: B x1.27** (sampled MTP acceptance 59 to 62 %, about 2.2 tokens per verify). **32k: parity** (x0.985, B's reps spread 63 to 80 tok/s). **64k: B x0.83** (54.2 against 65.3 tok/s), a little worse than ab35b's -12.6 % (54.8 against 62.7, plain batched decode without drafting). vLLM's 64k run today is 65.3 against 62.7 then.
- So the second half of the old blocker 5 (the batched engine drafted nothing on sampled requests) is closed and gives +27 % at short context; the long-context deficit is still there. At 64k MTP gains nothing: B decodes sampled requests at 64k at the rate ab35b measured without drafting.

### Control: `--kv-dtype fp8` on B (window w8)

decode tok/s: median of the 3 reps per run (reps), runs in ABBA order

| size (real prompt tokens) | ctl-A1 | ctl-N1 | ctl-N2 | ctl-A2 | A mean | arm mean | arm / A |
|---|---|---|---|---|---|---|---|
| 32768 (33871) | 71.7 (71.7, 71.6, 71.7) | 83.2 (81.3, 83.2, 86.6) | 82.0 (82.0, 80.7, 86.4) | 71.6 (71.9, 71.0, 71.6) | 71.6 | 82.6 | 1.153 |
| 65536 (68843) | 65.1 (65.6, 65.0, 65.1) | 69.4 (68.7, 69.6, 69.4) | 71.8 (71.8, 91.4, 68.5) | 65.6 (66.3, 65.6, 65.2) | 65.4 | 70.6 | 1.081 |

TTFT s: median of the 3 reps per run (reps), runs in ABBA order

| size (real prompt tokens) | ctl-A1 | ctl-N1 | ctl-N2 | ctl-A2 | A mean | arm mean | arm / A |
|---|---|---|---|---|---|---|---|
| 32768 (33871) | 6.09 (6.08-6.09) | 7.43 (6.65-7.52) | 6.59 (6.55-6.65) | 6.08 (6.08-6.08) | 6.08 | 7.01 | 1.152 |
| 65536 (68843) | 15.86 (15.85-15.89) | 18.74 (18.69-18.80) | 17.35 (17.11-17.96) | 15.82 (15.81-15.90) | 15.84 | 18.04 | 1.139 |

B MTP acceptance on the sampled requests (accepted / proposed from the /metrics diff over the run's 3 reps):

- ctl-N1: 32768: 58.3 % (2.164 tokens per verify), 65536: 56.0 % (2.116 tokens per verify)
- ctl-N2: 32768: 56.6 % (2.131 tokens per verify), 65536: 67.6 % (2.351 tokens per verify)

- With `--kv-dtype fp8` (B otherwise as shipped, store on) sampled decode is **x1.15 at 32k (82.6 against 71.6 tok/s) and x1.08 at 64k (70.6 against 65.4)**, against x0.985 and x0.83 with the shipped bf16 KV: the long-context sampled deficit is the KV cache dtype (decode at 64k reads twice the bytes with bf16). Its price in this window: TTFT 15 % higher at 32k and 64k (7.0 and 18.0 s against 6.1 and 15.8 s). B's 64k reps are not tight (N2: 71.8, 91.4, 68.5). **Not measured: the accuracy of fp8 KV** (the checkpoint declares fp8 KV, the engine ignores its scales), so this shows the lever, not that flipping it is safe. No default is changed here.


## Cell 5: GSM8K n=200, paired, thinking off (blocker 7)

`localeval quality --bench gsm8k --limit 200 --max-tokens 1024 --pin-gsm8k`, no nonce, `--thinking off --structured prompt`, same 200 questions and prompt on every run, temperature 0 (localeval's own), order A1 B1 A2. `raw/gsm8k-compare.json` (exact reply text per question id, `scripts/ab35b/compare_replies.py`), `raw/compare-*.txt` (localeval `compare`).

| run | accuracy (flexible) | number-only replies: right of count | worked replies: right of count | mean output tokens |
|---|---|---|---|---|
| A1 vLLM | 0.640 | 74 of 146 | 54 of 54 | 42.7 |
| A2 vLLM (same requests again) | 0.575 | 75 of 160 | 40 of 40 | 32.3 |
| B | 0.715 | 68 of 123 | 75 of 77 | 72.2 |

| pair | accuracy | localeval verdict (2 SE) | identical reply text | only the first right / only the second right |
|---|---|---|---|---|
| A1 vs A2 | 0.640 vs 0.575 | noise (0.097) | 131 of 200 | 22 / 9 |
| A1 vs B | 0.640 vs 0.715 | noise (0.093) | 118 of 200 | 6 / 21 |
| A2 vs B | 0.575 vs 0.715 | better, +0.140 (0.095) | 116 of 200 | 6 / 34 |

The picture is ab35b's, unchanged: vLLM does not reproduce its own replies at temperature 0 (131 of 200 identical, 69 differ; ab35b: 72; cause still not established); B scores equal or higher on identical prompts, entirely through the reply style (B writes a worked solution on 77 of 200, A on 54 and 40; where both work the problem out they are right 54 of 54 and 75 of 77, number-only replies are right about half the time on both). That is not a capability claim and not a defect; it is an unexplained systematic difference in style between the engines on the same weights (bf16 vs fp8 KV is one untested candidate). Thinking-on quality was not measured.

## Soak at current main (blocker 6)

Window w4: one B server for 25 min (`--max-batch 2 --prefix-store on`, default host store), `scripts/soak2/replay2.py` `--temp-mode mixed` (conversations 2, 3, 6, 7 at temperature 1.0, the rest greedy), 2 streams, fresh seeds 81 to 93, `scripts/ab35d/window.sh w4 soak`, vLLM idle beside it (its counters unchanged, 619 requests before and after). Shipping defaults of today: herd on, shortest-prefill-first, N=2 batched MTP, captured verify-attention rows. `scripts/soak/soak_table.py` for the per-replay and drift tables (`raw/sample-w4.tsv` is the 60 s sampler).

| replay | start min | requests | errors | TTFT sum s | TTFT p50 s | TTFT p90 s | decode tok/s median (bimodal: see doc) | decode tok/s mean | agg output tok/s | cached frac | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S81 | 0.0 | 48 | 0 | 76.2 | 1.54 | 1.79 | 43.8 | 44.4 | 60.8 | 0.732 | 124 |
| S82 | 2.1 | 48 | 0 | 73.2 | 1.40 | 1.85 | 46.9 | 48.8 | 64.4 | 0.732 | 116 |
| S83 | 4.1 | 48 | 0 | 71.9 | 1.45 | 1.60 | 45.0 | 46.8 | 63.1 | 0.732 | 117 |
| S84 | 6.0 | 48 | 0 | 73.1 | 1.46 | 2.09 | 46.3 | 48.4 | 64.5 | 0.732 | 116 |
| S85 | 8.0 | 48 | 0 | 71.5 | 1.48 | 1.67 | 46.5 | 47.7 | 64.9 | 0.732 | 117 |
| S86 | 10.0 | 48 | 0 | 74.0 | 1.49 | 2.20 | 47.3 | 49.6 | 65.2 | 0.732 | 116 |
| S87 | 11.9 | 48 | 0 | 76.3 | 1.41 | 2.61 | 46.4 | 48.6 | 63.4 | 0.732 | 115 |
| S88 | 13.9 | 48 | 0 | 71.1 | 1.47 | 1.60 | 45.6 | 46.8 | 64.1 | 0.732 | 116 |
| S89 | 15.9 | 48 | 0 | 74.8 | 1.39 | 2.51 | 46.4 | 50.0 | 62.9 | 0.732 | 115 |
| S90 | 17.8 | 48 | 0 | 74.7 | 1.48 | 2.34 | 46.3 | 50.0 | 65.0 | 0.732 | 114 |
| S91 | 19.7 | 48 | 0 | 76.5 | 1.48 | 2.16 | 46.0 | 49.7 | 63.6 | 0.732 | 115 |
| S92 | 21.7 | 48 | 0 | 72.8 | 1.46 | 2.09 | 45.7 | 47.4 | 63.4 | 0.732 | 116 |
| S93 | 23.6 | 48 | 0 | 78.6 | 1.42 | 2.58 | 47.5 | 50.3 | 64.3 | 0.732 | 115 |
- ttft_sum: first half mean 73.293, second half mean 74.971, min 71.120, max 78.621
- ttft_p50: first half mean 1.470, second half mean 1.444, min 1.386, max 1.536
- dec_p50: first half mean 45.967, second half mean 46.278, min 43.762, max 47.522
- dec_mean: first half mean 47.622, second half mean 48.993, min 44.408, max 50.306
- cached: first half mean 0.732, second half mean 0.732, min 0.732, max 0.732
- wall: first half mean 117.569, second half mean 115.287, min 114.042, max 124.130

sampler: 26 samples, 22:20:45 to 22:45:48
| signal | first 5 mean | last 5 mean | min | max | slope/min, all | slope/min, from sample 6 (store full) |
|---|---:|---:|---:|---:|---:|---:|
| mem_avail_mb | 40945.4 | 36490.2 | 35791.0 | 48641.0 | -184.19 | -18.62 |
| mem_used_mb | 83663.6 | 88118.8 | 75968.0 | 88818.0 | 184.19 | 18.61 |
| swap_used_mb | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 | 0.00 |
| rss_mb | 162.0 | 162.0 | 162.0 | 162.0 | 0.00 | 0.00 |
| intarsia_kv_bytes_used | 895.7 | 732.8 | 0.0 | 1430.7 | -14.19 | -24.80 |
| intarsia_device_memory_free_bytes | 42934.6 | 38262.2 | 37530.3 | 51004.2 | -193.17 | -19.59 |
| intarsia_prefix_store_tier_bytes{tier="host"} | 6367.7 | 8475.6 | 0.0 | 8589.6 | 86.69 | 3.34 |
| intarsia_prefix_store_entries | 76.2 | 104.4 | 0.0 | 134.0 | 1.03 | 0.37 |
| intarsia_prefix_store_tier_objects{tier="host"} | 1608.0 | 2121.2 | 0.0 | 2522.0 | 22.47 | -0.61 |

- store host bytes peak 8590 MB against budget 8590 MB (within budget)
- lookups 638, hits 599, misses 39, host evictions 2392, rejected 0, inserts 2511
- requests 625, rejected 0, engine_failed 0, no_progress_s 0, finished length/stop 279/344
- first host eviction seen in sample 2 (of 26; 60 s apart)
- intarsia_batch_paused_steps_total (prefill pause default ran when > 0): 2637
- server gaps (batch_itl) total 93857: above 0.25 s 22, above 0.5 s 1

Summary: 13 replays, 624 requests (312 greedy, 312 at T=1.0), **0 errors**, 0 rejected, `engine_failed` 0, `no_progress` 0, 0 warn / error lines in the serve log; the store reached its 8 GiB host budget in minute 3 and held it (peak 8590 MB of 8590 MB, 2392 evictions, 599 hits of 638 lookups, 0 failed restores, cached fraction 0.732 in every replay); MTP 36,524 verify steps, acceptance 77.6 %, 96 % of generated tokens from speculative steps; server gaps 93,857 of which 22 above 0.25 s and 1 above 0.5 s (soak2: 1 and 0 of 137k, at N=1 and before herd and shortest-prefill-first; vLLM's histogram in the replay cell had 20 and 35 above 0.5 s per run). TTFT sum per replay 71.1 to 78.6 s, slope +0.22 s per replay (standard error 0.16, 1.3 SE, no drift; soak2 flagged +0.39 at 2.6 SE). Decode per stream 46 tok/s median, 64 tok/s aggregate (soak2: 42, 59.6). **Memory**: MemAvailable lowest 35.7 GB beside vLLM (kill line 15 GB), swap 0, slope from the store-full point -18.6 MB per minute (about 0.5 GB over the 25 min; soak2 was flat at -0.7): too short to say whether it is a leak or the host store and page cache still settling, flagged for a longer soak. Not paired with vLLM (A was not under load in this window), so it shows B's stability, not a comparison.

## Per-blocker verdicts

| # | blocker (ab35b) | verdict | today, B against vLLM (A) | ab35b |
|---|---|---|---|---|
| 1 | Tool-result turns must work | **closed** | Every call and receipt turn passes on B, stream and non-stream, thinking off and default (`add_numbers`, `create_event`, `get_weather` 5/5 and 5/5, 2/2 and 2/2; json_schema 5/5); A the same apart from `get_weather` calls the model skipped (3/5, 0/2). Small n. | 28 of 28 receipts HTTP 400 (build before #349) |
| 2 | Prefix reuse with 2 streams | **closed** | Cached fraction 0.732 vs 0.695; summed TTFT 75.6 vs 75.7 s (4 seeds, parity); wall 117.1 vs 122.9 s; first-turn TTFT 1.55 to 1.92 vs 2.15 to 2.26 s, mid turns 0.1 to 0.3 s slower. | 0 cached tokens, summed TTFT 157.7 vs 58.6 s |
| 3 | Store decode stall when conversations interleave | **closed** | 1 stream: decode 118.2 / 116.6 vs 75.8 / 75.9 tok/s; 2 streams: per stream 46.4 vs 41.0 tok/s (steady seeds), combined 63.9 vs 60.9 tok/s. | 41 to 51 vs 68 to 79 tok/s |
| 4 | MTP together with prefix reuse | **closed** (and now a speed lead) | Acceptance 84 to 85 % on store requests at 2 streams, 96.5 % of tokens from speculative steps; single-stream greedy decode x1.53 / 1.32 / 1.13 at 4k / 16k / 32k. | a request drafted or used the store, not both |
| 5 | Sampled decode at long context; drafting on sampled requests in the batched engine | **partly**: drafting is closed, long context is not | Drafts on sampled requests (acceptance 54 to 62 %): 4k x1.27, 32k x0.985 (parity), **64k x0.83** (54.2 vs 65.3 tok/s; before x0.87). With `--kv-dtype fp8` (control): 32k x1.15, 64k x1.08, at +15 % TTFT; fp8 accuracy not measured. | 4k +7 %, 32k -5 %, 64k -12.6 %; no drafting on sampled |
| 6 | Unmeasured load and soak | **partly** | New: 25 min, 13 replays, 624 requests, 0 errors, store at budget, MemAvailable min 35.7 GB beside vLLM, no TTFT drift (+0.22 s per replay, 1.3 SE); with the earlier soak2 (38 min, 864 requests at the default budget, 0 errors; GSM8K after a store restore 196 of 200 against 195 of 200 with the store off) and restore-acc (restored replies equal cold ones, 200 of 200 token-id sequences). Not done: more than 2 streams, hours, real missioncntrl traffic, thinking-on quality, a restart drill; MemAvailable slope -18.6 MB per minute unexplained. | never run |
| 7 | Unexplained output differences against vLLM | **open, unchanged** (low) | GSM8K 0.640 / 0.575 (A, same requests twice) vs 0.715 (B): +0.075 noise, +0.140 better; all of it reply style (B works the problem out on 77 of 200, A on 54 and 40; right 75 of 77 and 54 of 54 / 40 of 40 when they do); vLLM repeats itself on 131 of 200. Cause not found, thinking-on not run. | same picture |
| new | Uncached prefill with the store on | **new finding** | Prefill 5285 / 5033 / 4965 vs 6435 / 6387 / 5558 tok/s at 4k / 16k / 32k (x0.82 / 0.79 / 0.89; TTFT +22 / +27 / +12 %). Store off, same binary: 6927 / 6624 / 5909 (x1.08 / 1.04 / 1.06). Not seen in the 2-stream replay. | store off in ab35 (ahead) |

## What still blocks a cutover, most blocking first

1. **No evidence of operation yet (blocker 6).** The longest run of today's defaults is 25 minutes of synthetic replays at 2 streams (624 requests, 0 errors); the earlier soaks predate herd, shortest-prefill-first, N=2 batched MTP, side-stream eager and the captured verify-attention rows. Missing: hours of real missioncntrl traffic (a shadow or canary run on a Spark, thinking on, with a memory trace), a restart and crash-recovery drill, more than 2 streams if production ever uses them, and an explanation for the small steady MemAvailable decline (-18.6 MB per minute after the store filled). This is the item that decides the cutover; the speed numbers do not.
2. **Prefix store tax on cold prompts.** With the store on, a back-to-back stream of new prompts prefills 11 to 21 % slower than vLLM (TTFT +12 to +27 % at 4k / 16k / 32k); with it off the same build is 4 to 8 % faster. The store is what gives reuse, so it is not optional. The 2-stream replay does not show the loss (first-turn TTFT is better than vLLM's), so it matters for single cold long prompts and 1-stream use. Cause not isolated (store insert 20.5 s and settle 4.2 s over the 12 requests of one speed run); engine work, small to medium.
3. **Sampled decode at 64k.** 17 % behind vLLM with bf16 KV (production requests without a temperature are sampled at the checkpoint default 1.0). `--kv-dtype fp8` turns it into +8 % at 64k and +15 % at 32k at 15 % more TTFT; the accuracy of fp8 KV on this checkpoint (it ignores the declared scales) is unmeasured, so this is a measurement plus a flag, not yet a decision.
4. **Output-style difference and thinking-on quality (blocker 7).** B answers GSM8K with a worked solution more often than vLLM on the same weights, scoring the same within each style; cause unknown and thinking-on quality not run. Low, but it is the one place where the two engines behave differently on identical prompts.
5. Minor, not blocking: the startup banner still says sampled requests use speculation `Match` (single sequence) although the batched engine drafts them; B accepts any request model name (so the `capcom` alias works) but lists only `guidance` in `/v1/models`.

## Fairness and limits

- **Replay is four seeds, one run per arm and seed.** A's TTFT sum ranges 69.5 to 89.6 s (the high one is the first replay after the tool cell) and B's 69.8 to 79.4 s, so the A to B difference of the means (75.7 vs 75.6 s) is far inside the spread. The replay says parity on TTFT sum and a few percent for B on wall time and combined output; it cannot rank the two further.
- **Replayed history**: the traces hold vLLM's recorded answers, so B sees vLLM's continuations, not its own. With the store, a continuation that shares B's own tokens could hit differently on the last token before the generation prompt (ab35b's caveat); not tested.
- **Server state differs by cell**: B's server was fresh at the start of each window, so the speed and sampled arms ran with a store that held whatever the earlier cells put in (B1 before B2); vLLM's prefix cache likewise. The nonce-first prompts defeat both caches; the store's capture cost is paid regardless, which is the point of cell 3's control.
- **Shipping config on both sides, not equal settings**: B bf16 KV against A fp8 KV, B MTP against A no speculation, B store 8 GiB host (+ device tier) against A's 8 GiB fp8 KV cache. The comparison is "what you would get", not a controlled single factor; the controls (store off, fp8 KV) isolate one factor each on B only.
- **Client over the network** (Mac to spark2): TTFT and the client-side ITL include it and are the same for both arms; localeval's ITL p99 is character-weighted (ab35c).
- **Synthetic prompts** (filler notes), thinking off for the speed, sampled and GSM8K cells (the tool cell also runs the stock default). Real missioncntrl traffic was not replayed.
- **No cell exercised more than 2 concurrent streams** (vLLM runs `--max-num-seqs 2`, so B was matched to it), hours of operation, restarts under load, or thinking-on quality.
- **Build watchdog**: the build script's own free-memory watchdog never ran (its ssh call waited on a backgrounded subshell, so the watchdog loop started only after cargo had finished; the script is fixed in this branch). I started a manual watchdog from 21:41, four minutes after the build started; the first four minutes (rustc only, no nvcc yet) were unwatched except by earlyoom. Free memory read 73 to 76 GB throughout the watched part; fleetq showed the box at 76 GB free after the build. Server windows had the 2 s watchdog from the first byte.
- **w6** aborted by my own edit of a running script (no data); the rerun is w8.

## Raw results and how to reproduce

- `docs/results/2026-11-19-ab35d-cutover-scoreboard/raw/`: per-window logs (`window-w*.log`, `w*.out`), clock samples (`clk-w*.tsv`), marks (`marks-w*.tsv`), MemAvailable (`mem-w*.txt`), serve logs and `/metrics` before / after (`serve-w*.{log,metrics*}`), per-cell `/metrics` of both engines (`snap/<cell>.{vllm,b}.{before,after}.gz`), replay (`{A,B}-interleaved-seed{0,1}-k2.{jsonl,summary.json}`, `ctl1-*-k1.*`, soak `S81..S93-*`, `sample-w4.tsv`), sampled (`*-sampled.{json,out}`), tools (`*-tools*.{json,out}`), localeval result dirs and `index.txt`, `gsm8k-compare.json`, `compare-*.txt`, `build.log`, `recipes/`, the plan and traces. localeval originals under `/opt/code/deepeval/results/<run id>/`.
- Scripts: `scripts/ab35d/{build.sh,window.sh,window-ctl.sh,memwatch.sh,clk.sh,toolsmoke.py,tables.py}`; the tables above are `python3 scripts/ab35d/tables.py <raw dir>`. Windows: `FLEETQ_OWNER=claude-overnight-ab35d fleetq run --host spark2 --quiet gpu,cpu --mem 55 --expect N --label ... -- bash scripts/ab35d/window.sh w<k> <cells>` (control windows: `window-ctl.sh`, `CTL_SERVE` for B's flags). Build dir `~/intarsiaml-ab35d` on spark2 removed at the end.
- fleetq jobs (owner `claude-overnight-ab35d`): #10320 build, #10321 w1, #10322 w2, #10323 w3, #10324 w4, #10325 w5, #10326 w6 (aborted), #10327 w7, #10328 w8, #10329 w9, #10330 cleanup of `~/intarsiaml-ab35d` (removed; no fleetq slot held, no server of ours running at the end).

## Clock guard per cell

| window | cell | samples | at util>90 | of those SM<2000 | min SM MHz | mean SM MHz | max W | SW power cap delta us | vLLM delta (requests / gen tokens) |
|---|---|---|---|---|---|---|---|---|---|
| w1 | tools | 119 | 108 | 0 | 2392 | 2401 | 43 | 0 | 76 / 9228 |
| w1 | replay-A-s0 | 61 | 59 | 0 | 2385 | 2395 | 76 | 0 | 49 / 7586 |
| w1 | replay-B-s0 | 61 | 56 | 0 | 2366 | 2389 | 86 | 0 | 0 / 0 |
| w1 | replay-B-s1 | 56 | 49 | 0 | 2379 | 2390 | 85 | 0 | 0 / 0 |
| w1 | replay-A-s1 | 61 | 59 | 0 | 2392 | 2396 | 73 | 0 | 49 / 7346 |
| w2 | speed-A1 | 31 | 28 | 0 | 2392 | 2398 | 70 | 0 | 11 / 2306 |
| w2 | speed-B1 | 32 | 22 | 0 | 2372 | 2386 | 82 | 0 | 0 / 0 |
| w2 | speed-B2 | 29 | 23 | 0 | 2372 | 2386 | 87 | 0 | 0 / 0 |
| w2 | speed-A2 | 31 | 29 | 0 | 2385 | 2394 | 74 | 0 | 11 / 2306 |
| w2 | sampled-A1 | 51 | 49 | 0 | 2385 | 2396 | 81 | 0 | 11 / 2306 |
| w2 | sampled-B1 | 52 | 36 | 0 | 2340 | 2378 | 93 | 0 | 0 / 0 |
| w2 | sampled-B2 | 52 | 41 | 0 | 2333 | 2372 | 93 | 0 | 0 / 0 |
| w2 | sampled-A2 | 51 | 48 | 0 | 2385 | 2396 | 82 | 0 | 11 / 2306 |
| w3 | gsm8k-A1 | 68 | 51 | 0 | 2392 | 2402 | 45 | 0 | 200 / 8545 |
| w3 | gsm8k-B1 | 71 | 51 | 0 | 2392 | 2393 | 54 | 0 | 0 / 0 |
| w3 | gsm8k-A2 | 55 | 45 | 0 | 2392 | 2402 | 46 | 0 | 200 / 6453 |
| w5 | ctlspeed-A1 | 32 | 30 | 0 | 2392 | 2398 | 72 | 0 | 11 / 2306 |
| w5 | ctlspeed-N1 | 27 | 24 | 0 | 2372 | 2385 | 91 | 0 | 0 / 0 |
| w5 | ctlspeed-N2 | 26 | 25 | 0 | 2346 | 2382 | 91 | 0 | 0 / 0 |
| w5 | ctlspeed-A2 | 32 | 29 | 0 | 2385 | 2395 | 74 | 0 | 11 / 2306 |
| w7 | ctlreplay1-A-s0 | 79 | 71 | 0 | 2385 | 2399 | 72 | 0 | 49 / 7594 |
| w7 | ctlreplay1-N-s0 | 65 | 56 | 0 | 2372 | 2390 | 89 | 0 | 0 / 0 |
| w7 | ctlreplay1-N-s1 | 61 | 57 | 0 | 2359 | 2393 | 91 | 0 | 0 / 0 |
| w7 | ctlreplay1-A-s1 | 78 | 69 | 0 | 2385 | 2395 | 72 | 0 | 49 / 7361 |
| w8 | ctlsampled-A1 | 45 | 44 | 0 | 2385 | 2395 | 78 | 0 | 8 / 1538 |
| w8 | ctlsampled-N1 | 50 | 33 | 0 | 2372 | 2387 | 87 | 0 | 0 / 0 |
| w8 | ctlsampled-N2 | 47 | 32 | 0 | 2379 | 2386 | 89 | 0 | 0 / 0 |
| w8 | ctlsampled-A2 | 45 | 43 | 0 | 2392 | 2398 | 84 | 0 | 8 / 1538 |
| w9 | replay-A-s3 | 62 | 61 | 0 | 2385 | 2399 | 70 | 0 | 49 / 7455 |
| w9 | replay-B-s3 | 61 | 53 | 0 | 2372 | 2387 | 85 | 0 | 0 / 0 |
| w9 | replay-B-s2 | 58 | 52 | 0 | 2372 | 2389 | 85 | 0 | 0 / 0 |
| w9 | replay-A-s2 | 61 | 59 | 0 | 2385 | 2395 | 73 | 0 | 49 / 7556 |
