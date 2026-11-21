# ab35e: cutover scoreboard, intarsiaml `eeca95f8e` vs the production vLLM (spark2)

## Brief

Lane `ab35e`: refresh the cutover scoreboard. Compare current intarsiaml main (eeca95f8e) against production vLLM, both serving the 35B NVFP4, on the same box with the same client. Docs-only PR, which merges on green.

Repo intarsiaml. Worktree `/opt/code/intarsiaml/.worktrees/ab35e` (branch `ab35e`, from origin/main eeca95f8e). All scratch and build dirs inside it.

### Why
The last scoreboard was `docs/results/2026-11-19-ab35d-cutover-scoreboard.md` (lane ab35d, main 25489d6a). Since then these became default on CUDA:
- `INTARSIA_PREFIX_STORE_PREFAULT` (#470): store-on cold TTFT.
- `INTARSIA_CUDA_ATTN_KV_ONCE` (#473): sampled 64k was x0.81 of vLLM and should now be ~x1.16; greedy long context is faster too.
- CUDA copy drafts (#474): mb1 only, so not reached at `--max-batch 2`.
Blocker 7 was attributed to a reply-style lean, mostly fp8 KV; see `docs/results/2026-11-20-style7.md`. Blocker 6 (memory) is run separately on spark1 by lane memlong. The soak cell is skipped.

### Do
Re-run ab35d's cells with its own scripts (`scripts/ab35d/`, plus `scripts/ab35b/` and `scripts/soak2/` as it used them). Same server command line for B (`serve --max-batch 2 --prefix-store on` plus whatever ab35d used), same vLLM A, same order (ABBA), same clock checks.
1. Cell 1: tool-result turns.
2. Cell 2 / 2b: prefix-cached replay, 2 streams and 1 stream.
3. Cell 3: single-stream greedy decode and prefill at 4k / 16k / 32k (add 64k if ab35d's script allows).
4. Cell 4: sampled decode T=0.7 at 4k / 32k / 64k.
5. Cell 5: GSM8K n=200 pinned, thinking off. One extra B arm with `--kv-dtype fp8_e4m3`, for the style comparison only.
6. Cold-prompt TTFT store-on vs vLLM at 4k / 16k / 32k (storettft's cell).
Then a blocker table #1-#7 with status and numbers against ab35d's, plus a one-paragraph cutover verdict. Note any regression vs ab35d.

### Fleet rules
spark2 only (spark1 runs a soak). Production vLLM there (role retro) is never stopped or restarted; A only means sending it requests. Every spark2 command that builds, starts a server or benchmarks goes through fleetq (`--quiet gpu,cpu` for timing, `--load cpu` for builds), never `--force` / `--no-queue`. GB10 memory budget, `MAX_JOBS=4`, kill below 15 GB available. Windows below 2300 MHz SM clock are discarded. No tool call over 4 min. At the end: no fleetq slot held, no server running, remote box dir removed.

## Summary

**Result: nothing regressed against ab35d, and the two gaps ab35d found are closed or nearly so. Of the seven blockers, four stay closed (1 to 4), sampled long-context decode (5) is newly closed (x1.165 at 64k, was x0.83), blocker 7 is re-attributed (style, not a defect) but not eliminated, and blocker 6 (operation evidence) is not measured here (memlong runs it on spark1).** On every cell intarsiaml at `eeca95f8e` in its shipping configuration (`serve --max-batch 2 --prefix-store on`) is ahead of the production vLLM except cold-prompt TTFT with the store on, which is +4 to +9 % slower than vLLM at 4k / 16k / 32k (ab35d: +12 to +27 %) and level at 64k. Same box (spark2), same client (this Mac), A never touched, no vLLM request ran during any B cell, SM clock never below 2320 MHz (the 2300 MHz rule discarded no window), SW power cap delta 0 in every cell.

Headline numbers, B / A, ratios of means, ABBA unless stated (ab35d in the last column):

| cell | A vLLM | B intarsiaml | B / A | ab35d B / A |
|---|---|---|---|---|
| tool calls and receipt turns (3 tools, stream and non-stream, thinking off and default) | A misses 1 of 5 `get_weather` calls and 1 of 2 `add_numbers` stream calls (the model answered in plain text) | all pass | closed | closed |
| 2-stream multi-turn replay, 4 seeds: summed TTFT | 89.9 s (81.1 to 98.6) | 71.5 s (69.1 to 75.1) | 0.80 | 1.00 |
| same: wall / combined output / cached fraction | 121.9 s / 61.4 tok/s / 0.695 | 105.6 s / 70.8 tok/s / 0.732 | 0.87 / 1.15 | 0.95 / 1.05 |
| 1-stream interleaved replay (2 seeds): decode per request / wall | 75.4 tok/s / 158.6 s | 134.0 tok/s / 115.7 s | 1.78 / 0.73 | 1.55 / 0.79 |
| greedy decode 4k / 16k / 32k / 64k | 77.8 / 74.9 / 72.5 / 65.9 tok/s | 123.5 / 114.3 / 105.9 / 89.0 | 1.59 / 1.53 / 1.46 / 1.35 | 1.53 / 1.32 / 1.13 / - |
| greedy uncached TTFT 4k / 16k / 32k / 64k (store on) | 0.637 / 2.612 / 6.092 / 15.861 s | 0.679 / 2.837 / 6.323 / 15.825 | 1.07 / 1.09 / 1.04 / 1.00 | 1.22 / 1.27 / 1.12 / - |
| sampled T=0.7 decode 4k / 32k / 64k | 77.3 / 71.6 / 65.5 tok/s | 99.6 / 85.9 / 76.2 | 1.29 / 1.20 / 1.165 | 1.27 / 0.985 / 0.83 |
| GSM8K n=200 pinned, thinking off, flexible score | 0.605 and 0.625 (same requests twice) | 0.715 (fp8 KV arm: 0.655) | +0.110 (beyond 2 SE) and +0.090 (noise) | +0.075 and +0.140 |
| 25 min soak | not run | not run (memlong, spark1) | - | 0 errors |

## Setup

- **B (intarsiaml, the shipping config)**: origin/main `eeca95f8e`, binary stamp `ab35e-eeca95f8efec`, built on spark2 in `~/intarsiaml-ab35e` (fleetq #10453, `--load cpu --mem 55`, `CARGO_BUILD_JOBS=3 MAX_JOBS=3 FLASHINFER_NVCC_THREADS=1`, `cargo build --release --locked --features cuda -p intarsia-cli`, about 4 min, vLLM up, lowest MemAvailable during the build 71 GB). Command line identical to ab35d's in every window (`scripts/ab35e/window.sh`): `intarsia serve --backend cuda --model <HF snapshot 1355db6a of nvidia/Qwen3.6-35B-A3B-NVFP4> --port 8191 --host 0.0.0.0 --served-model-name guidance --max-seq 131072 --mem-budget 40 --profile-every 0 --max-batch 2 --prefix-store on`. No environment variables. Defaults new since ab35d that this run exercises: `INTARSIA_PREFIX_STORE_PREFAULT` (#470) and `INTARSIA_CUDA_ATTN_KV_ONCE` (#473; the banner says "KV cache once: on"); the CUDA copy drafts of #474 are mb1-only and are not reached at `--max-batch 2`. bf16 KV (banner: the checkpoint declares fp8 KV, the engine runs bf16). Extra arm for cell 5 only: the same line plus `--kv-dtype fp8_e4m3` (window w4; its banner has no "bf16 here" line).
- **A (production vLLM)**: container `missioncntrl-guidance` (fleetq role `retro`), the same vLLM as ab35d (`0.26.1rc1.dev602`, same arguments, recipe in `raw/recipes/A.txt`), sent requests only: never stopped, restarted or reconfigured, no `--takes-down`.
- **Box / client**: spark2, vLLM resident (about 38 GB) plus B's `--mem-budget 40` and its 8 GiB host store tier. Lowest MemAvailable in any B window 28.8 GB (w3) and 32.7 GB (w4), both with another tenant on the box (see Fairness: lane inkling's `hf download` job #10469 started at 18:50:20); 37.7 GB or more in every other window, kill line 15 GB, 0 watchdog kills, 0 warn / error lines in any serve log. Client: this Mac over the network to `spark-fa00:8000` (A) and `:8191` (B), same path for both.
- **Windows** (fleetq owner `claude-overnight-ab35e`, all `--quiet gpu,cpu`, one fresh B server per window, vLLM up, A and B never concurrent):

| window | job | cells |
|---|---|---|
| w1 | #10455 | tools, 2-stream replay seeds 0 and 1 (A B B A) |
| w2 | #10459 | greedy speed (4k / 16k / 32k / 64k, which is also the cold-prompt TTFT cell), sampled decode 4k / 32k / 64k (each A B B A), fresh server, so the speed cell is the fresh-server, back-to-back regime storettft measured |
| w3 to w6 | #10466 (one slot for the four windows; the queue held us behind lane samphost's 2 h gate) | w3 GSM8K A B A; w4 GSM8K B with fp8 KV; w5 1-stream replay (A N N A, seeds 0 1 1 0; the raw name `N` is B with the store on); w6 2-stream replay seeds 3 and 2 (A B B A) |

  A first w3 job (#10465) was cancelled while still queued and merged into #10466; no data lost. Build #10453, cleanup #10471.
- **Scripts and inputs**: `scripts/ab35e/` is `scripts/ab35d/` with the paths renamed, the build pinned to `eeca95f8e`, `65536` added to the greedy sizes, a `BARGS` hook for B's extra flags and a B-only GSM8K cell (`gsm8kb`); ab35b's `sampled.py`, `scripts/interleave/replay2.py` and localeval 0.4.0 (git `2c5524c`) are used as ab35d did. The replay plan and the four seed traces are copies of ab35d's (vLLM's recorded answers), so A and B again saw byte-identical requests, and the same requests as in ab35d. Nonce first in every speed and sampled prompt (no cache hit for either arm); order, flags and clock guard as in ab35d (`guard_end` aborts on SM < 2000 MHz at > 90 % util for 3 samples, SW cap growth > 1 s, or vLLM traffic during a B cell; it never tripped, and the table below shows min SM 2320 MHz).

## Cell 1: tool calls and receipt turns (blocker 1)

Same script and counts as ab35d (`scripts/ab35e/toolsmoke.py`; 5 non-stream and 2 stream calls per tool with thinking off, 3 and 2 with the default; json_schema 5 and 3). B: every call and every receipt turn passes, stream and non-stream, thinking off and default (`get_weather`, `add_numbers`, `create_event` all 5/5, 5/5, 2/2, 2/2 and 3/3, 3/3, 2/2, 2/2; json_schema 5/5 and 3/3). A, thinking off: `create_event` passes; `get_weather` 1/5 calls and receipts and 2/2 stream; `add_numbers` stream 1/2 (the model answered "417 + 285 = 702" in plain text); thinking default all pass. The misses on A are the model not calling a tool (not deterministic; ab35b and ab35d saw it on `get_weather`), not a parser defect. Small n. Raw: `raw/{A,B}-tools*.{out,json}`.

## Cell 2: two concurrent multi-turn conversations with prefix reuse (blockers 2, 3, 4)



| arm | seed | errors | TTFT sum s | TTFT p50 / p90 s | cached frac | decode tok/s per stream (median) | combined out tok/s | wall s | client ITL p99 ms median / max | server gaps >0.15 / >0.3 (vLLM) or >0.25 (B) / >0.5 of N |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 0 | 0 | 96.8 | 2.14 / 2.45 | 0.695 | 55.7 | 62.0 | 122.0 | 110 / 641 | 44 / 19 / 12 of 7521 |
| A | 1 | 0 | 81.1 | 1.50 / 2.32 | 0.696 | 45.0 | 60.8 | 120.7 | 341 / 897 | 76 / 50 / 27 of 7297 |
| A | 2 | 0 | 98.6 | 2.16 / 2.54 | 0.695 | 55.0 | 62.2 | 121.9 | 139 / 521 | 40 / 18 / 10 of 7536 |
| A | 3 | 0 | 83.1 | 1.56 / 2.32 | 0.693 | 44.9 | 60.6 | 122.9 | 330 / 892 | 75 / 50 / 26 of 7412 |
| B | 0 | 0 | 72.2 | 1.47 / 1.94 | 0.732 | 52.8 | 72.1 | 105.3 | 180 / 974 | 106 / 0 / 0 of 7548 |
| B | 1 | 0 | 75.1 | 1.39 / 2.55 | 0.732 | 53.3 | 70.6 | 103.6 | 249 / 1118 | 87 / 0 / 0 of 7277 |
| B | 2 | 0 | 69.1 | 1.41 / 1.55 | 0.732 | 51.9 | 70.5 | 107.1 | 213 / 696 | 121 / 0 / 0 of 7505 |
| B | 3 | 0 | 69.4 | 1.44 / 1.56 | 0.732 | 52.0 | 70.1 | 106.5 | 202 / 787 | 112 / 0 / 0 of 7423 |

Per-turn TTFT median s (cached tokens median):

| turn | A s0 | A s1 | A s2 | A s3 | B s0 | B s1 | B s2 | B s3 |
|---|---|---|---|---|---|---|---|---|
| 1 | 2.14 (0) | 2.22 (0) | 1.28 (0) | 2.25 (0) | 1.48 (44) | 1.48 (44) | 1.48 (44) | 1.49 (44) |
| 2 | 2.14 (6288) | 1.95 (6288) | 2.14 (6288) | 1.98 (6288) | 1.18 (8041) | 1.19 (8038) | 1.21 (8046) | 1.20 (8043) |
| 3 | 1.25 (12576) | 1.51 (12576) | 1.86 (12576) | 1.89 (12576) | 1.27 (13063) | 1.28 (13071) | 1.29 (13074) | 1.30 (13066) |
| 4 | 2.43 (16768) | 1.36 (16768) | 2.45 (16768) | 1.38 (16768) | 1.62 (18088) | 1.37 (18085) | 1.39 (18093) | 1.38 (18091) |
| 5 | 2.08 (23056) | 1.24 (23056) | 2.12 (23056) | 1.26 (23056) | 1.47 (23111) | 1.50 (23104) | 1.46 (23113) | 1.46 (23116) |
| 6 | 1.97 (27248) | 1.49 (27248) | 2.56 (27248) | 1.52 (27248) | 1.53 (28135) | 1.55 (28127) | 1.54 (28131) | 1.55 (28142) |

B's MTP counters over each replay (`/metrics` diff of `intarsia_spec_*`):

| cell | spec steps | proposed | accepted | acceptance | completion tokens | share of tokens made in spec steps (steps+accepted)/tokens |
|---|---|---|---|---|---|---|
| replay-B-s0 | 2723 | 5432 | 4600 | 84.7 % | 7589 | 0.965 |
| replay-B-s1 | 2648 | 5287 | 4422 | 83.6 % | 7318 | 0.966 |
| replay-B-s2 | 2708 | 5405 | 4564 | 84.4 % | 7546 | 0.964 |
| replay-B-s3 | 2680 | 5346 | 4503 | 84.2 % | 7464 | 0.962 |

How to read it:

- **Reuse (blocker 2).** Cached fraction B 0.732, A 0.695. Summed TTFT: B 72.2, 75.1, 69.1, 69.4 s (mean 71.5; ab35d 75.6), A 96.8, 81.1, 98.6, 83.1 s (mean 89.9; ab35d 75.7). **B is ahead by 20 % this time, but most of that is A**: its mid turns (2 and 3, 1.25 to 2.14 s against 1.12 to 1.21 s in ab35d, and turn 4 to 6 at 2.0 to 2.6 s on seeds 0 and 2) were slower than in ab35d, and the seeds alternate high and low (96.8 / 81.1 / 98.6 / 83.1), which is the order-dependent pattern ab35d saw once (89.6 s on its first replay). The cause is on vLLM's side (its prefix-cache state or production traffic between our cells; we did not look into it). Treat the TTFT sum as "B at least at parity, probably ahead", and B's own number as the stable one: 69.1 to 75.1 s in both lanes. B's first-turn TTFT (uncached 8k) is 1.48 s against 1.28 to 2.25 s and its restore turns are now 1.18 to 1.62 s.
- **Wall and decode under interleaving (blocker 3).** Wall time B 103.6 to 107.1 s against A 120.7 to 122.9 s (B 13 % shorter; ab35d 5 %). Combined output B 70.1 to 72.1 tok/s against A 60.6 to 62.2 (+15 %; ab35d +5 %): B's decode is faster (KV once, below) and the store no longer stalls it. Per-stream decode medians B 51.9 to 53.3 (ab35d 45.1 to 47.8), A 44.9 to 55.7.
- **Tail.** Server-side gaps above 0.3 s: B 0 of about 7.4k in every run, A 18 to 50. (Definitions differ between the engines, as in ab35d: vLLM's inter-token-latency histogram against B's `intarsia_batch_itl_seconds`.) Client ITL p99 median 180 to 249 ms on B against 110 to 341 ms on A.
- **MTP with the store (blocker 4).** Acceptance 83.6 to 84.7 %, 96.2 to 96.6 % of tokens come out of speculative steps, as in ab35d.

## Cell 2b: the same replay with one stream, store on (blocker 3's original shape, window w5)

| arm | seed | errors | TTFT sum s | TTFT p50 / p90 s | cached frac | decode tok/s (median, mean) | combined out tok/s | wall s | server gaps >0.15 / >0.3 (vLLM) or >0.25 (B) / >0.5 of N |
|---|---|---|---|---|---|---|---|---|---|
| A | 0 | 0 | 59.7 | 1.20 / 1.44 | 0.695 | 75.5, 75.4 | 47.2 | 160.5 | 0 / 0 / 0 of 7528 |
| A | 1 | 0 | 59.1 | 1.19 / 1.43 | 0.696 | 75.3, 75.7 | 46.9 | 156.7 | 0 / 0 / 0 of 7307 |
| B (raw name N) | 0 | 0 | 59.5 | 1.26 / 1.34 | 0.732 | 135.1, 133.5 | 64.8 | 117.1 | 0 / 0 / 0 of 7548 |
| B (raw name N) | 1 | 0 | 58.1 | 1.26 / 1.35 | 0.732 | 132.7, 131.8 | 64.1 | 114.2 | 0 / 0 / 0 of 7277 |

Decode per request B 135.1 / 132.7 tok/s against A 75.5 / 75.3 (x1.78; ab35d x1.55 at 118.2 / 116.6), summed TTFT 59.5 and 58.1 s against 59.7 and 59.1 s (parity, ab35d +9 % and -1 %), cached fraction 0.732 against 0.695, wall 117 and 114 s against 161 and 157 s (B 27 % shorter), combined output 64.8 and 64.1 against 47.2 and 46.9 tok/s. No server gaps on either side.

## Cell 3: single-stream greedy decode and prefill, 4k / 16k / 32k / 64k (also the cold-prompt TTFT cell, item 6)

localeval `speed --structured prompt --thinking off --sizes 4096,16384,32768,65536 --gen 256 --reps 3`, nonce prompts, first cell of a fresh B server (store on, so this is storettft's regime: back-to-back cold prompts on a fresh server; the store captures and settles every request).

### Greedy single-stream speed, shipping config B (store on) vs A (localeval speed, nonce prompts, 256 forced tokens, 3 reps + 1 warmup per run)


decode tok/s: median of the 3 reps per run (min-max), runs in ABBA order

| size (real prompt tokens) | A1-speed | B1-speed | B2-speed | A2-speed | A mean | arm mean | arm / A |
|---|---|---|---|---|---|---|---|
| 4096 (4090) | 76.9 (76.8-78.7) | 121.3 (119.9-124.7) | 125.7 (125.2-129.9) | 78.8 (77.1-78.9) | 77.8 | 123.5 | 1.587 |
| 16384 (16654) | 75.0 (74.8-76.0) | 116.7 (111.7-116.8) | 111.9 (110.3-113.2) | 74.8 (74.8-76.0) | 74.9 | 114.3 | 1.526 |
| 32768 (33903) | 72.6 (72.2-76.0) | 107.7 (99.9-112.0) | 104.1 (100.3-116.6) | 72.5 (71.0-73.7) | 72.5 | 105.9 | 1.460 |
| 65536 (68930) | 65.8 (64.6-66.2) | 93.0 (88.1-94.5) | 85.1 (84.8-87.7) | 66.1 (65.5-66.1) | 65.9 | 89.0 | 1.350 |

TTFT s: median of the 3 reps per run (min-max), runs in ABBA order

| size (real prompt tokens) | A1-speed | B1-speed | B2-speed | A2-speed | A mean | arm mean | arm / A |
|---|---|---|---|---|---|---|---|
| 4096 (4090) | 0.636 (0.633-0.641) | 0.688 (0.670-0.716) | 0.671 (0.669-0.707) | 0.637 (0.636-0.638) | 0.637 | 0.679 | 1.068 |
| 16384 (16654) | 2.610 (2.605-2.624) | 2.843 (2.821-2.885) | 2.832 (2.816-2.878) | 2.615 (2.615-2.617) | 2.612 | 2.837 | 1.086 |
| 32768 (33903) | 6.086 (6.072-6.245) | 6.321 (6.318-6.388) | 6.326 (6.325-6.349) | 6.098 (6.089-6.149) | 6.092 | 6.323 | 1.038 |
| 65536 (68930) | 15.864 (15.857-15.904) | 15.806 (15.794-15.831) | 15.843 (15.839-15.850) | 15.857 (15.855-15.873) | 15.861 | 15.825 | 0.998 |

prefill tok/s (uncached prompt tokens / TTFT): median of the 3 reps per run (min-max), runs in ABBA order

| size (real prompt tokens) | A1-speed | B1-speed | B2-speed | A2-speed | A mean | arm mean | arm / A |
|---|---|---|---|---|---|---|---|
| 4096 (4090) | 6433 (6377-6464) | 5940 (5716-6099) | 6099 (5778-6113) | 6415 (6411-6431) | 6424 | 6020 | 0.937 |
| 16384 (16654) | 6382 (6347-6392) | 5858 (5772-5903) | 5882 (5789-5912) | 6369 (6365-6371) | 6375 | 5870 | 0.921 |
| 32768 (33903) | 5571 (5429-5583) | 5360 (5304-5363) | 5359 (5341-5360) | 5560 (5514-5568) | 5565 | 5360 | 0.963 |
| 65536 (68930) | 4345 (4334-4347) | 4360 (4353-4363) | 4352 (4350-4353) | 4349 (4344-4349) | 4347 | 4356 | 1.002 |

- **Decode**: x1.59 / 1.53 / 1.46 / 1.35 at 4k / 16k / 32k / 64k (ab35d x1.53 / 1.32 / 1.13): the 16k to 64k gains are consistent with the once-per-step KV read (#473; not isolated here, no control arm); 4k moves little.
- **Cold-prompt TTFT with the store on (the storettft cell)**: +6.8 / +8.6 / +3.8 % at 4k / 16k / 32k (ab35d +22 / +27 / +12 %), level at 64k (-0.2 %); prefill 6020 / 5870 / 5360 / 4356 tok/s against 6424 / 6375 / 5565 / 4347 (x0.94 / 0.92 / 0.96 / 1.00; ab35d x0.82 / 0.79 / 0.89). The prefault default (#470) took away about two thirds of the gap; the rest is storettft's cause B (the store's schedule: a few extra passes per request) plus the settle, as that doc says. With the store off the same build was 4 to 8 % faster than vLLM in ab35d; not re-measured here.

## Cell 4: sampled decode, T=0.7, 4k / 32k / 64k (blocker 5)

`scripts/ab35b/sampled.py`, 3 reps per size, A B B A.

### Sampled decode, shipping config (T=0.7, top_p 0.95, 256 forced tokens, 3 reps, nonce prompts)


decode tok/s: median of the 3 reps per run (reps), runs in ABBA order

| size (real prompt tokens) | A1 | B1 | B2 | A2 | A mean | arm mean | arm / A |
|---|---|---|---|---|---|---|---|
| 4096 (4072) | 77.7 (75.9, 77.7, 78.1) | 99.1 (99.1, 117.1, 96.7) | 100.1 (103.3, 100.1, 99.5) | 76.9 (76.2, 76.9, 78.1) | 77.3 | 99.6 | 1.289 |
| 32768 (33899) | 71.7 (71.7, 71.8, 71.6) | 85.9 (86.0, 85.9, 83.8) | 85.9 (92.2, 85.9, 85.2) | 71.5 (71.5, 71.9, 71.5) | 71.6 | 85.9 | 1.200 |
| 65536 (68910) | 65.6 (66.5, 65.6, 64.5) | 75.1 (78.4, 75.1, 74.3) | 77.4 (74.0, 80.7, 77.4) | 65.3 (65.7, 65.2, 65.3) | 65.5 | 76.2 | 1.165 |

TTFT s: median of the 3 reps per run (reps), runs in ABBA order

| size (real prompt tokens) | A1 | B1 | B2 | A2 | A mean | arm mean | arm / A |
|---|---|---|---|---|---|---|---|
| 4096 (4072) | 0.63 (0.63-0.64) | 0.67 (0.67-0.74) | 0.67 (0.67-0.67) | 0.64 (0.64-0.64) | 0.63 | 0.67 | 1.057 |
| 32768 (33899) | 6.08 (6.07-6.08) | 6.34 (6.30-6.41) | 6.33 (6.29-6.38) | 6.08 (6.07-6.10) | 6.08 | 6.34 | 1.042 |
| 65536 (68910) | 15.86 (15.83-15.93) | 15.83 (15.79-15.87) | 15.74 (15.69-15.77) | 15.85 (15.83-15.88) | 15.85 | 15.78 | 0.996 |

B MTP acceptance on the sampled requests (accepted / proposed from the /metrics diff over the run's 3 reps):

- B1: 4096: 63.3 % (2.264 tokens per verify), 32768: 56.0 % (2.116 tokens per verify), 65536: 58.0 % (2.158 tokens per verify)
- B2: 4096: 60.3 % (2.205 tokens per verify), 32768: 60.7 % (2.212 tokens per verify), 65536: 61.3 % (2.224 tokens per verify)

**64k is x1.165 (76.2 against 65.5 tok/s; ab35d x0.83), 32k x1.20 (was x0.985), 4k x1.29 (was x1.27).** TTFT is level with vLLM at 64k and +4 to +6 % at 4k / 32k. MTP acceptance on sampled requests 56 to 63 % (ab35d 54 to 62 %). bf16 KV throughout; the fp8 KV control of ab35d was not repeated for speed.

## Cell 5: GSM8K n=200, paired, thinking off (blocker 7)

`localeval quality --bench gsm8k --limit 200 --max-tokens 1024 --pin-gsm8k`, order A1 B1 A2, then B with `--kv-dtype fp8_e4m3` (a separate B server, same flags otherwise). Primary score is localeval's `gsm8k_flexible`; `raw/gsm8k-compare.json` holds exact replies per question (`scripts/ab35b/compare_replies.py`).

| run | accuracy (flexible) | number-only replies: right of count | worked replies: right of count | mean output tokens |
|---|---|---|---|---|
| A1 vLLM | 0.605 | 76 of 154 | 45 of 46 | 40.2 |
| A2 vLLM (same requests again) | 0.625 | 70 of 145 | 55 of 55 | 44.3 |
| B bf16 KV | 0.715 | 68 of 123 | 75 of 77 | 72.2 |
| B `--kv-dtype fp8_e4m3` (extra arm) | 0.655 | 59 of 127 | 72 of 73 | 56.8 |

| pair | identical reply text | only the first right / only the second right |
|---|---|---|
| A1 vs A2 | 131 of 200 | 11 / 15 |
| A1 vs B | 111 of 200 | 7 / 29 |
| A2 vs B | 121 of 200 | 7 / 25 |
| A1 vs B fp8 | 119 of 200 | 15 / 25 |
| A2 vs B fp8 | 112 of 200 | 14 / 20 |
| B vs B fp8 | 114 of 200 | 20 / 8 |

The picture is ab35d's, and style7's explanation holds: vLLM replies number-only on 145 to 154 of 200 questions, B on 123, and a worked answer is right 45 of 46 to 75 of 77 times, a number-only answer about half the time; whichever engine writes the worked solution more often scores higher. localeval compare (`raw/compare-*.txt`, 2 SE about 0.094): B vs A1 +0.110 **better**, B vs A2 +0.090 noise, A1 vs A2 +0.020 noise, B vs B fp8 -0.060 noise, B fp8 vs A2 +0.030 noise. **With fp8 KV B moves toward vLLM** on every column (worked replies 73 against 77, mean output 56.8 against 72.2 tokens, score 0.655 against 0.715), by roughly half of the distance to A on mean output length, consistent with style7's "about 40 % fp8 KV" and an unexplained remainder. vLLM repeats itself on 131 of 200 (ab35d: 131). B's 2 truncated replies are the same two ids as in ab35d. Thinking-on quality was not measured. **Greedy output unchanged by the new defaults**: B's 200 GSM8K replies are byte-identical (200 of 200) to ab35d's B1 replies (`compare_replies.py`, store, prefault and KV once all on versus ab35d's `25489d6aa`).

## Per-blocker verdicts, ab35d against ab35e

| # | blocker (ab35b) | ab35d | ab35e (this run, `eeca95f8e`) |
|---|---|---|---|
| 1 | Tool-result turns must work | closed | **closed.** All calls and receipts pass on B (stream and non-stream, thinking off and default, json_schema 5/5); A skips a tool on `get_weather` (1/5) and one `add_numbers` stream call. |
| 2 | Prefix reuse with 2 streams | closed | **closed.** Cached fraction 0.732 vs 0.695; summed TTFT B 71.5 s (ab35d 75.6) vs A 89.9 s (ab35d 75.7; A noisier and slower this time); wall 105.6 vs 121.9 s. |
| 3 | Store decode stall when conversations interleave | closed | **closed.** 1 stream: decode 135.1 / 132.7 vs 75.5 / 75.3 tok/s; 2 streams: combined 70.8 vs 61.4 tok/s, 0 server gaps above 0.3 s. |
| 4 | MTP together with prefix reuse | closed, a speed lead | **closed, lead larger.** Acceptance 83.6 to 84.7 %, 96 % of tokens from spec steps; greedy decode x1.59 / 1.53 / 1.46 / 1.35 at 4k / 16k / 32k / 64k. |
| 5 | Sampled decode at long context | partly (64k x0.83) | **closed.** 4k / 32k / 64k x1.29 / 1.20 / 1.165 with bf16 KV (the fp8 KV workaround is no longer needed for speed). |
| 6 | Unmeasured load and soak | partly | **unchanged here, not measured.** Soak skipped by brief (lane memlong, spark1). Lowest MemAvailable 37.7 GB in the speed, sampled and replay windows beside vLLM (28.8 GB in the GSM8K window with another tenant, not B), 0 kills. ab35d's 25 min / 624 requests / 0 errors predates #470, #473 and #475 at main; hours, real missioncntrl traffic, a restart drill, thinking-on quality remain open. |
| 7 | Unexplained output differences vs vLLM | open (low) | **explained (style7), not eliminated.** First-token reply style, about 40 % from fp8 KV (fp8 arm: 0.655, worked replies 73 vs 77, mean 56.8 vs 72.2 tokens), rest unexplained; GSM8K B 0.715 vs A 0.605 / 0.625; no correctness problem found. |
| new | Uncached prefill with the store on | new finding (TTFT +12 to +27 %) | **mostly closed.** TTFT +6.8 / +8.6 / +3.8 % at 4k / 16k / 32k, 0 % at 64k (prefill x0.94 / 0.92 / 0.96 / 1.00). Remainder is the store's own schedule and settle (storettft); B ahead of A on first-turn TTFT in the replay. |

**Regressions against ab35d: none.** Every B number is equal or better (tools, replay, decode, sampled, GSM8K, memory). A's replay TTFT is worse than in ab35d (see cell 2), which flatters the TTFT comparison, not B.

## Cutover verdict

On the same box and client, intarsiaml at `eeca95f8e` in the shipping configuration matches or beats the production vLLM on everything measured today: 1.35 to 1.78 x decode for a single stream (greedy, sampled and replay), 1.15 x combined throughput and 13 % shorter wall time with two streams, 0 server-side stalls above 0.3 s, summed TTFT at parity or better in reuse workloads, all tool and receipt turns pass, and the two gaps of ab35d (sampled 64k, cold prompts with the store) are closed or down to +4 to +9 % TTFT on cold 4k to 32k prompts. The speed argument against cutover is gone. What still decides it is not speed: there is no evidence of operation under today's defaults beyond ab35d's 25 minutes (blocker 6; memlong is producing the memory and hours evidence on spark1), no real missioncntrl traffic or thinking-on quality run, no restart drill, and one behavioural difference (blocker 7: reply style, mostly explained by fp8 KV and not a defect) that a consumer comparing outputs will notice. A cutover is reasonable to attempt as a shadow or canary on a Spark once memlong's result is in; this document does not change any default and recommends nothing beyond what the numbers show.

## Fairness and limits

- **Same limits as ab35d**: shipping config on both sides, not equal settings (B bf16 KV, MTP, 8 GiB host store; A fp8 KV, no speculation); Mac client over the network (TTFT includes it for both); synthetic filler prompts, thinking off for speed, sampled and GSM8K; at most 2 concurrent streams; replay uses vLLM's recorded answers as history; one run per arm and seed.
- **A drifted**: its replay TTFT sums (81 to 99 s) are above ab35d's (70 to 90 s) with a high / low alternation by seed; all other A numbers (decode, prefill, sampled, tools) match ab35d to about 1 to 3 %. The replay TTFT ratio B / A (0.80) therefore overstates B's lead; B alone moved from 75.6 to 71.5 s.
- **Cell 6 is cell 3**: the cold-prompt TTFT cell storettft uses is the same localeval speed run as ab35d's, run first on a fresh B server. Storettft's steady-state regime (a warm host tier, which it found gives +13 / +12 / +11 % before its fix and smaller after) was not separately measured here.
- **w3 / w4 MemAvailable minima** (28.8 and 32.7 GB), not B's memory: B filled its store during B1 and MemAvailable settled at 39.0 GB at 18:49:36, the level ab35d's same window held flat to its end (39.5 GB minimum). It then fell 39 to 29 GB from 18:50:36, during A2 while B was idle, which began 16 s after fleetq job #10469 (lane inkling-2-mini, `hf download` and `docker pull`, a net / disk job that shares the box) started at 18:50:20; w4 ran in the same period (47 GB free at its start with nothing of ours running, 56 GB used). Host store bytes at the end of w3 8.57 GB and of w4 7.69 GB, the 8 GiB cap, as in ab35d. All other windows, with no other tenant, stayed above 37.7 GB. Kill line 15 GB was never near.
- **Statistics**: GSM8K 2 SE about 0.093 at n=200; localeval's verdicts: B vs A1 better (+0.110), B vs A2 noise (+0.090). No soak. No fp8-KV speed or accuracy beyond GSM8K.
- **Build watchdog** (`scripts/ab35e/build.sh`, the ab35d version with the SHA pinned): it logged MemAvailable every 20 s this time (71 to 73 GB during the build).

## Raw results and how to reproduce

- `docs/results/2026-11-20-ab35e-cutover-scoreboard/raw/`: per-window logs (`window-w*.log`), `chain.log` / `chain2.log` (fleetq output), clock samples (`clk-w*.tsv`), marks, MemAvailable (`mem-w*.txt`), serve logs and `/metrics` before / after (`serve-w*.{log,metrics*}`), per-cell `/metrics` of both engines (`snap/`), replay (`{A,B}-interleaved-seed{0..3}-k2.*`, `ctl1-*-k1.*`), sampled (`*-sampled.{json,out}`), tools, localeval result dirs and `index.txt`, `gsm8k-compare.json`, `tables.md`, `build.log`, `recipes/` (`B.txt` is the last window's recipe; w4's fp8 line is in `window-w4.log`), plan and traces. localeval originals under `/opt/code/deepeval/results/<run id>/`.
- Scripts: `scripts/ab35e/{build.sh,window.sh,window-ctl.sh,chain.sh,chain2.sh,memwatch.sh,clk.sh,toolsmoke.py,tables.py}`; the tables are `python3 scripts/ab35e/tables.py <raw dir>`. Windows: `FLEETQ_OWNER=claude-overnight-ab35e fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect N --label ... -- bash scripts/ab35e/window.sh w<k> <cells>`.
- fleetq jobs (owner `claude-overnight-ab35e`): #10453 build, #10455 w1, #10459 w2, #10465 (cancelled while queued), #10466 w3 to w6, #10471 cleanup of `~/intarsiaml-ab35e` (removed). No slot held and no server of ours running at the end.

## Clock guard per cell

| window | cell | samples | at util>90 | of those SM<2000 | min SM MHz | mean SM MHz | max W | SW power cap delta us | vLLM delta (requests / gen tokens) |
|---|---|---|---|---|---|---|---|---|---|
| w1 | tools | 119 | 110 | 0 | 2392 | 2400 | 43 | 0 | 75 / 9351 |
| w1 | replay-A-s0 | 61 | 59 | 0 | 2385 | 2395 | 75 | 0 | 49 / 7570 |
| w1 | replay-B-s0 | 53 | 50 | 0 | 2372 | 2385 | 87 | 0 | 0 / 0 |
| w1 | replay-B-s1 | 53 | 49 | 0 | 2372 | 2386 | 88 | 0 | 0 / 0 |
| w1 | replay-A-s1 | 61 | 59 | 0 | 2392 | 2396 | 74 | 0 | 49 / 7346 |
| w2 | speed-A1 | 61 | 58 | 0 | 2392 | 2396 | 81 | 0 | 14 / 3074 |
| w2 | speed-B1 | 55 | 47 | 0 | 2327 | 2376 | 92 | 0 | 0 / 0 |
| w2 | speed-B2 | 55 | 49 | 0 | 2320 | 2372 | 92 | 0 | 0 / 0 |
| w2 | speed-A2 | 60 | 57 | 0 | 2385 | 2396 | 82 | 0 | 14 / 3074 |
| w2 | sampled-A1 | 50 | 49 | 0 | 2385 | 2396 | 82 | 0 | 11 / 2306 |
| w2 | sampled-B1 | 49 | 32 | 0 | 2320 | 2365 | 92 | 0 | 0 / 0 |
| w2 | sampled-B2 | 48 | 29 | 0 | 2340 | 2369 | 92 | 0 | 0 / 0 |
| w2 | sampled-A2 | 50 | 48 | 0 | 2385 | 2397 | 82 | 0 | 11 / 2306 |
| w3 | gsm8k-A1 | 65 | 50 | 0 | 2392 | 2402 | 47 | 0 | 200 / 8039 |
| w3 | gsm8k-B1 | 70 | 59 | 0 | 2392 | 2392 | 59 | 0 | 0 / 0 |
| w3 | gsm8k-A2 | 85 | 52 | 0 | 2392 | 2402 | 43 | 0 | 200 / 8868 |
| w4 | gsm8k-B-fp8 | 61 | 44 | 0 | 2392 | 2393 | 56 | 0 | 0 / 0 |
| w5 | ctlreplay1-A-s0 | 80 | 71 | 0 | 2392 | 2400 | 70 | 0 | 49 / 7577 |
| w5 | ctlreplay1-N-s0 | 59 | 51 | 0 | 2372 | 2386 | 89 | 0 | 0 / 0 |
| w5 | ctlreplay1-N-s1 | 58 | 53 | 0 | 2359 | 2388 | 89 | 0 | 0 / 0 |
| w5 | ctlreplay1-A-s1 | 78 | 75 | 0 | 2392 | 2396 | 70 | 0 | 49 / 7356 |
| w6 | replay-A-s3 | 61 | 60 | 0 | 2385 | 2398 | 70 | 0 | 49 / 7461 |
| w6 | replay-B-s3 | 55 | 49 | 0 | 2372 | 2382 | 83 | 0 | 0 / 0 |
| w6 | replay-B-s2 | 54 | 49 | 0 | 2372 | 2385 | 87 | 0 | 0 / 0 |
| w6 | replay-A-s2 | 60 | 59 | 0 | 2392 | 2396 | 75 | 0 | 49 / 7585 |

## Progress notes

- 2026-11-20 17:08: build done (#10453, stamp ab35e-eeca95f8efec); w1 and w2 run in the 17:08 to 17:57 gap.
- 18:12: lane samphost's 2 h ABBA gate took the box; cancelled my queued w3 (#10465) and re-queued w3 to w6 as one job (#10466), which started 18:43 after the gate ended early.
- 19:17: all windows done (w3 to w6 in 34 min); remote dir removed (#10471); doc written.
