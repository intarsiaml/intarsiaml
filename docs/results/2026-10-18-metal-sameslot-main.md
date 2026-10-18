# Metal same-slot re-measurement of origin/main: intarsia vs mlx-lm vs llama.cpp

Qwen3-8B BF16, endor (Apple M5 Max, macOS 27.0 build 26A428). 2026-10-18, 17:35-18:00 EDT. Branch `metal-sameslot-main`. Raw data: `docs/results/raw/2026-10-18-metal-sameslot-main/` (`sameslot.sh`, `sameslot-retry.sh`, `analyze.py`, `analysis-main.txt`, `analysis-retry.txt`, `analysis-pooled.txt`, `*-gpu-idle.txt`, `*-macmon.tsv`, `localeval-results/`).

Repeat of `docs/results/2026-10-15-metal-final-sameslot.md` (same method, same server flags, same models, same localeval arguments; scripts reused with paths and the run order adapted, plus a macmon clock sampler added). Compared with that doc's **SAFE** (`spike/metal` a2102bf, chunk 256) and **CAND** (`metal-candidate2`, chunk 2048) figures.

- **main** = `origin/main` 5eabb1a7 (Merge PR #11), `cargo build --release -p intarsia-cli --features metal` from a detached worktree, binary md5 9810733ce9a084dcf83542a9065abe22. `serve` defaults: `plan show` and the server log both report `prefill chunk 2048 (recipe qwen3, metal)`.
- Order ABBA: intarsia, mlx-lm, llama.cpp, intarsia, one exclusive fleetq slot, `localeval speed --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off`. Servers started and stopped one at a time; every port confirmed closed and the pid gone before the next.
- **The machine was contended** (section 2), so the whole ABBA sequence was run twice in two slots: batch 1 (`main-*`, fleetq #2466) and, as the one permitted retry of each arm, batch 2 (`retry-*`, fleetq #2470). Both are reported; neither is dropped. The two batches disagree more than the 2026-10-15 pair did, so read the pooled figures with that spread in mind.

## 1. Same-slot speed

Decode tok/s median (min-max) at 128 / 1k / 4k prompts. Marginal prefill = delta prompt tokens / delta TTFT median versus the previous row (no figure for the 128 row). The 1k row has about 990 uncached tokens, under localeval's 1024 threshold, so its prefill tok/s columns are overhead-influenced; the marginal column cancels the fixed overhead. localeval stores only min/median/max per row, so ranges for the marginal figure are bounds (`analysis-*.txt`).

Batch 1 (fleetq #2466, 17:35-17:44):

| label | localeval run id | prompt tok | decode 128 | decode 1k | decode 4k | marginal prefill 1k | marginal prefill 4k | prefill tok/s 1k (range) | prefill tok/s 4k (range) |
|---|---|---|---|---|---|---|---|---|---|
| main-1 | 20261018-173600-adhoc-main-1 | 113/990/4086 | 33.4 (32.8-35.0) | 34.9 (34.4-35.0) | 34.3 (33.8-34.4) | 3716 | 3406 | 3352 (3239-3433) | 3393 (3319-3464) |
| main-mlx-lm | 20261018-173726-adhoc-main-mlx-lm | 134/991/4088 | 33.8 (33.8-33.9) | 33.4 (33.2-33.4) | 31.9 (31.7-32.1) | 4160 | 3426 | 2324 (2251-2331) | 3078 (2930-3087) |
| main-llamacpp | 20261018-173857-adhoc-main-llamacpp | 116/990/4087 | 34.9 (34.8-35.0) | 34.6 (34.3-34.7) | 33.5 (33.5-33.6) | 2933 | 2456 | 2545 (2470-2563) | 2474 (2472-2488) |
| main-2 | 20261018-174234-adhoc-main-2 | 115/990/4088 | 36.1 (35.9-36.4) | 34.9 (34.9-35.9) | 34.5 (33.6-34.5) | 4397 | 2494 | 3946 (3816-3974) | 2737 (2334-3415) |

Batch 2, retry (fleetq #2470, 17:44-18:00):

| label | localeval run id | prompt tok | decode 128 | decode 1k | decode 4k | marginal prefill 1k | marginal prefill 4k | prefill tok/s 1k (range) | prefill tok/s 4k (range) |
|---|---|---|---|---|---|---|---|---|---|
| retry-1 | 20261018-174636-adhoc-retry-1 | 111/992/4088 | 36.3 (35.6-36.4) | 36.0 (35.8-36.2) | 34.7 (34.7-35.0) | 4472 | 3283 | 3974 (3891-3981) | 3425 (2641-3541) |
| retry-mlx-lm | 20261018-174918-adhoc-retry-mlx-lm | 113/990/4088 | 33.7 (33.6-33.9) | 33.2 (33.1-33.4) | 31.4 (31.2-31.7) | 4237 | 2781 | 2293 (2282-2339) | 2647 (2431-2866) |
| retry-llamacpp | 20261018-175315-adhoc-retry-llamacpp | 135/993/4090 | 34.5 (34.4-34.7) | 33.6 (33.4-34.4) | 32.0 (31.7-32.4) | 3178 | 1869 | 2542 (2448-2544) | 1999 (1641-2309) |
| retry-2 | 20261018-175749-adhoc-retry-2 | 113/991/4090 | 35.6 (35.4-36.0) | 35.4 (34.9-35.4) | 33.8 (33.3-33.9) | 4390 | 2344 | 3923 (3767-3923) | 2596 (2310-3375) |

Marginal prefill bounds (from TTFT min/max of the two rows involved), tok/s:

| run | marginal 1k | marginal 4k |
|---|---|---|
| main-1 | 3716 (3480-3846) | 3406 (3283-3534) |
| main-mlx-lm | 4160 (3826-4534) | 3426 (3189-3503) |
| main-llamacpp | 2933 (2828-3155) | 2456 (2439-2492) |
| main-2 | 4397 (4227-4464) | 2494 (2064-3303) |
| retry-1 | 4472 (4340-4495) | 3283 (2385-3444) |
| retry-mlx-lm | 4237 (4079-4474) | 2781 (2463-3120) |
| retry-llamacpp | 3178 (3011-3190) | 1869 (1473-2264) |
| retry-2 | 4390 (4181-4434) | 2344 (2044-3266) |

### Percentages, next to 2026-10-15

main = mean of its two intarsia runs; the references are the single mlx-lm and llama.cpp run of the same batch. Pooled = mean of all 4 intarsia runs against the mean of the 2 mlx-lm and the 2 llama.cpp runs. Prefill basis is marginal; the prefill tok/s basis is the cross-check. The 2026-10-15 columns are copied from that doc.

| metric | 10-15 SAFE | 10-15 CAND | main batch 1 | main batch 2 | main pooled |
|---|---|---|---|---|---|
| marginal prefill 1k, tok/s | 4054 | 4083 | 4057 | 4431 | 4244 |
| 1k % of mlx-lm / llama.cpp | 94.1 % / 140.5 % | 94.8 % / 141.5 % | 97.5 % / 138.3 % | 104.6 % / 139.4 % | 101.1 % / 138.9 % |
| marginal prefill 4k, tok/s | 3126 | 3169 | 2950 | 2814 | 2882 |
| 4k % of mlx-lm / llama.cpp | 96.3 % / 164.8 % | 97.7 % / 167.1 % | 86.1 % / 120.1 % | 101.2 % / 150.5 % | 92.9 % / 133.3 % |
| prefill tok/s 1k, % mlx-lm / % llama.cpp | 159 % / 147 % | 160 % / 148 % | 157 % / 143 % | 172 % / 155 % | 165 % / 149 % |
| prefill tok/s 4k, % mlx-lm / % llama.cpp | 109 % / 161 % | 111 % / 162 % | 100 % / 124 % | 114 % / 151 % | 106 % / 136 % |
| decode 128 / 1k / 4k tok/s | 36.68 / 36.61 / 35.63 | 36.79 / 36.54 / 35.55 | 34.73 / 34.90 / 34.38 | 35.92 / 35.70 / 34.26 | 35.32 / 35.30 / 34.32 |
| decode % of 40.6 tok/s byte floor | 90.3 / 90.2 / 87.7 % | 90.6 / 90.0 / 87.6 % | 85.5 / 86.0 / 84.7 % | 88.5 / 87.9 / 84.4 % | 87.0 / 86.9 / 84.5 % |
| decode vs mlx-lm | 106.3 / 107.8 / 109.0 % | 106.6 / 107.6 / 108.8 % | 102.6 / 104.5 / 107.9 % | 106.7 / 107.5 / 109.2 % | 104.6 / 106.0 / 108.5 % |
| decode vs llama.cpp | 103.6 / 104.3 / 104.9 % | 103.9 / 104.1 / 104.6 % | 99.5 / 100.8 / 102.6 % | 104.1 / 106.1 / 107.0 % | 101.8 / 103.5 / 104.8 % |

References (decode 128 / 1k / 4k tok/s; marginal prefill 1k / 4k): mlx-lm batch 1 33.8 / 33.4 / 31.9; 4160 / 3426. mlx-lm batch 2 33.7 / 33.2 / 31.4; 4237 / 2781. llama.cpp batch 1 34.9 / 34.6 / 33.5; 2933 / 2456. llama.cpp batch 2 34.5 / 33.6 / 32.0; 3178 / 1869. (2026-10-15: mlx-lm 34.5 / 34.0 / 32.7; 4309 / 3244. llama.cpp 35.4 / 35.1 / 34.0; 2885 / 1897.) The mlx-lm "peak" decode figures sit above the byte floor again (burst delivery, not sustained decode); do not read them as beating the floor.

### Drift, run 1 to run 2 (same binary, same slot, 7-9 minutes apart)

| | decode 128 / 1k / 4k | marginal prefill 1k | marginal prefill 4k | prefill tok/s 4k |
|---|---|---|---|---|
| batch 1 | 33.4->36.1 (+8.2 %) / 34.9->34.9 (+0.2 %) / 34.3->34.5 (+0.6 %) | 3716->4397 (+18.3 %) | 3406->2494 (-26.8 %) | 3393->2737 (-19.3 %) |
| batch 2 | 36.3->35.6 (-2.0 %) / 36.0->35.4 (-1.9 %) / 34.7->33.8 (-2.6 %) | 4472->4390 (-1.8 %) | 3283->2344 (-28.6 %) | 3425->2596 (-24.2 %) |

Decode is stable to a few percent except the 128 row of batch 1 run 1 (range 32.8-35.0, the widest decode range of the day). Serve-path 4k prefill moved 19-29 % between runs in both batches, in the opposite direction to 2026-10-15 (run 2 slower here, faster there). The 4k prefill percentages of any single run are therefore resolved only to about +-25 %.

## 2. Machine state, clock and contention caveats

- **GPU idle before each run** (`ioreg` "Device Utilization %", median of 3 one-second samples, start limit 10 %, the script waits 10 s up to 12 times then proceeds). 2026-10-15: 0, 0, 0, 0, 1, 1 %. This time the box was never idle:
  - batch 1: 10, 8, 9, 9 % (main-2 waited 7 times first, samples up to 13 %).
  - batch 2: 10 (5 waits), 9 (6 waits), 10 (11 waits, samples up to 31 %), 10 (11 waits, samples up to 35 %). The last two arms were started at the first sample set whose median met the limit.
  - Source: the interactive desktop (WindowServer ~27 % CPU, the Syrtis app ~50 % CPU, a Chrome GPU helper). Not touched, since they are the user's applications. `pgrep -fl zoom.us` was empty in all 48 sample sets. endor-kev and endor-tensorfold were left running as standing services (fleetq lists them as "assumed up").
  - The 2026-10-15 run was at 02:35 at night, so the contention is likely a time-of-day difference, not an endor change. It has not been proven that it is the cause of the slowdown below.
- **GPU clock** (macmon, `macmon pipe -i 1000` sampled through each timed run; new in this run, 2026-10-15 recorded no clock; `*-macmon.tsv`). Median GPU MHz over samples with GPU utilisation above 50 %: main-1 1322 (p10 1050, min 901), main-2 1295 (min 856), retry-1 1321 (min 1166), retry-2 1274 (min 997); mlx-lm 1276 and 1273; llama.cpp 1147 and 1116. GPU temperature peaked at 87-93 C in every arm; main-1 started already at 82 C. The clock dips (856-1050 MHz minima in main-1 and main-2) coincide with the arms whose 128-row decode and 4k prefill were lowest. The 2026-10-15 same-slot method recorded no clock (this sampler is a method addition), so there is no clock to compare against that doc's runs. The same-day follow-up `docs/results/2026-10-15-metal-p3-noise.md` measured serve-path 4k reps at 1120 to 1590 MHz and found TTFT scaling with clock, so the 1274-1322 MHz medians here (1116-1147 for llama.cpp) are on the low side of that range; that supports clock as a factor but does not quantify it for this run.
- **All engines were slower than on 2026-10-15, intarsia a little more.** Decode versus 10-15 (pooled, tok/s): intarsia 35.3 / 35.3 / 34.3 against CAND 36.8 / 36.5 / 35.6 (-4.0 / -3.4 / -3.5 %); mlx-lm 33.8 / 33.3 / 31.6 against 34.5 / 34.0 / 32.7 (-2.0 / -2.1 / -3.4 %); llama.cpp 34.7 / 34.1 / 32.8 against 35.4 / 35.1 / 34.0 (-2.0 / -2.8 / -3.5 %). A 2-4 % loss across all three engines points at the machine state, not at a code change on main, but this data cannot separate the two: the 10-15 CAND binary was not run in these slots, so there is no same-slot control for code drift. Standing relative to mlx-lm and llama.cpp is unchanged (104.6-108.5 % of mlx-lm pooled vs 106.3-109.0 % before).
- The mlx-lm and llama.cpp 4k references moved 20-30 % between the two batches (mlx-lm 4k marginal 3426 vs 2781; llama.cpp 2456 vs 1869), so a prefill percentage that is "above mlx-lm" or "below mlx-lm" at 4k depends on which batch you read. Pooled is the least noisy single figure here, and it is still a mean of two draws per reference.

## 3. Verdicts

Thresholds as in 2026-10-15: prefill committed >= 50 % of mlx-lm, target >= 65 %, stretch >= 85 %, at 1k and 4k; decode >= 85 % of the 40.6 tok/s byte floor, >= 90 % of mlx-lm, not below llama.cpp.

| criterion | batch 1 | batch 2 | pooled |
|---|---|---|---|
| prefill committed / target / stretch (marginal, 1k / 4k) | PASS (97.5 % / 86.1 %) | PASS (104.6 % / 101.2 %) | PASS (101.1 % / 92.9 %) |
| decode >= 85 % of byte floor (128 / 1k / 4k) | PASS / PASS / **FAIL** (85.5 / 86.0 / 84.7 %) | PASS / PASS / **FAIL** (88.5 / 87.9 / 84.4 %) | PASS / PASS / **FAIL** (87.0 / 86.9 / 84.5 %) |
| decode >= 90 % of mlx-lm | PASS (103-108 %) | PASS (107-109 %) | PASS (105-109 %) |
| decode not below llama.cpp | **128: 99.5 %, 0.5 % below**; 1k, 4k PASS | PASS (+4 to +7 %) | PASS (+2 to +5 %) |

- Prefill: stretch is met on the pooled figures and in both batch means, with the same thin 4k margin as before (batch 1 at 86.1 %; single runs vary +-25 %). 1k prefill is at or above the 10-15 CAND level (4244 vs 4083 marginal, +4 %, inside the noise); 4k is 9 % lower than CAND (2882 vs 3169), inside the +-25 % single-run spread, not a resolved change.
- Decode: **the 85 % byte-floor gate fails at 4k in every view (84.4-84.7 %)**, by 0.4-0.6 points, where it passed on 2026-10-15 (87.6-87.7 %). 128 and 1k still pass (85.5-88.5 %), with less margin. The 4k decode of mlx-lm (31.4-31.9) and llama.cpp (32.0-33.5) fell by a similar amount, so I attribute it to the machine state (section 2), but this is an inference, not a measured control. A rerun at night or with the 10-15 CAND binary as a same-slot control would settle it.
- The one decode-vs-llama.cpp miss is batch 1 at 128 tokens (34.73 vs 34.9), driven by the wide main-1 range (32.8-35.0); it is within noise and passes pooled.

## 4. Versions and IDs

- intarsia: origin/main 5eabb1a7490243d1da888e785269be40d4d2eba6, binary md5 9810733ce9a084dcf83542a9065abe22, built in 35 s from a detached worktree. Serve flags: `--backend metal --device 0 --max-seq 8192 --profile-every 0 --no-supervisor --no-ui`, env `INTARSIA_CPU_FREE_MEM_GB=60`, chunk 2048 from `recipes/qwen3.toml` (not overridden).
- Tools unchanged from 2026-10-15: mlx 0.32.1, mlx-lm 0.31.3, mlx-metal 0.32.1 (`/Users/josef/.local/bin/mlx_lm.server`, `/Users/josef/models/Qwen3-8B`, `--prompt-cache-size 1`, `enable_thinking=false`); llama.cpp `llama-server` 0.6.0 build 11429 commit d81235049, BF16 GGUF `-ngl 999 -fa on -c 8192 --cache-ram 0`; macOS 27.0 (26A428); localeval 0.2.1, deepeval head 69c7a848bba836b70782aced73d278ee4b4d2fa6.
- fleetq: #2466 (batch 1) and #2470 (batch 2), both `--host endor --exclusive --mem 40 --expect 60`, released by the script trap, status done.
- localeval run ids: tables in section 1; result dirs copied to `docs/results/raw/2026-10-18-metal-sameslot-main/localeval-results/`.
- unanimis records (logged by localeval): main-1 96a90dea-4a8b-4dca-9bc9-793a2c7c14c0, main-mlx-lm 2eebf97d-4462-43bb-859c-88fc0b942a27, main-llamacpp 1aca5558-d894-4209-8a67-1d2f69d49ed3, main-2 5ed15dee-c7e1-4c43-9c81-d466b24a4ac4, retry-1 920ecf3b-1d20-43d2-9611-e265d725aa4c, retry-mlx-lm 86b40ea9-13fb-4cc4-96b5-610737c83c34, retry-llamacpp a87c4a01-c9fb-4951-ab14-69b9406c2476, retry-2 b41ff9fb-0ced-4c91-ae16-e66cf19320ff.
- Method differences from 2026-10-15: order ABBA with one intarsia build instead of SAFE/CAND/mlx-lm/llama.cpp/CAND/SAFE; run labels `main-*` / `retry-*`; macmon sampler added (raw JSON reduced to `*-macmon.tsv`); two batches instead of one. Everything else (ports 8093 / 8092 / 8091, server commands, idle gate, 5 s gap between arms, `analyze.py` formulas, the 40.6 floor) is as before. `sameslot-retry.sh` is `sameslot.sh` with the label prefix and output file names changed.

## 5. Not done

No gates (cargo test, kernels test, accuracy, greedy-id identity, plain-prefill ABBA, `qualify`): this was a speed re-measurement only. No 16k serve-path speed. No same-slot run of the 10-15 CAND binary as a control (it still exists under the earlier session scratchpad `final/intarsia-cand`). mlx-lm and llama.cpp ran once per batch. The slowdown cause (machine state vs main) is not isolated.
