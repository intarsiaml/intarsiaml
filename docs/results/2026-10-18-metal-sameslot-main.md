# Metal same-slot re-measurement of origin/main: intarsia vs mlx-lm vs llama.cpp

Qwen3-8B BF16, endor (Apple M5 Max, macOS 27.0 build 26A428). 2026-10-18, 17:35-18:00 EDT. Branch `metal-sameslot-main`. Raw data: `docs/results/raw/2026-10-18-metal-sameslot-main/` (`sameslot.sh`, `sameslot-retry.sh`, `sameslot-control.sh`, `plain-abba.sh`, `analyze_plain.py`, `plain/`, `compare/`, `analysis-control.txt`, `analyze.py`, `analysis-main.txt`, `analysis-retry.txt`, `analysis-pooled.txt`, `*-gpu-idle.txt`, `*-macmon.tsv`, `localeval-results/`).

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
- **All engines were slower than on 2026-10-15, intarsia a little more.** Decode versus 10-15 (pooled, tok/s): intarsia 35.3 / 35.3 / 34.3 against CAND 36.8 / 36.5 / 35.6 (-4.0 / -3.4 / -3.5 %); mlx-lm 33.8 / 33.3 / 31.6 against 34.5 / 34.0 / 32.7 (-2.0 / -2.1 / -3.4 %); llama.cpp 34.7 / 34.1 / 32.8 against 35.4 / 35.1 / 34.0 (-2.0 / -2.8 / -3.5 %). A 2-4 % loss across all three engines points at the machine state, not at a code change on main, the control run in section 5 (CAND and main in one slot) confirms this for decode. Standing relative to mlx-lm and llama.cpp is unchanged (104.6-108.5 % of mlx-lm pooled vs 106.3-109.0 % before).
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
- Decode: **the 85 % byte-floor gate fails at 4k in every view (84.4-84.7 %)**, by 0.4-0.6 points, where it passed on 2026-10-15 (87.6-87.7 %). 128 and 1k still pass (85.5-88.5 %), with less margin. The 4k decode of mlx-lm (31.4-31.9) and llama.cpp (32.0-33.5) fell by a similar amount, so I attribute it to the machine state (section 2), and the same-slot control in section 5 confirms it: the 10-15 CAND binary decodes 34.16 tok/s at 4k today (84.1 % of the floor), identical to main within 0.4 %.
- The one decode-vs-llama.cpp miss is batch 1 at 128 tokens (34.73 vs 34.9), driven by the wide main-1 range (32.8-35.0); it is within noise and passes pooled.

## 4. Versions and IDs

- intarsia: origin/main 5eabb1a7490243d1da888e785269be40d4d2eba6, binary md5 9810733ce9a084dcf83542a9065abe22, built in 35 s from a detached worktree. Serve flags: `--backend metal --device 0 --max-seq 8192 --profile-every 0 --no-supervisor --no-ui`, env `INTARSIA_CPU_FREE_MEM_GB=60`, chunk 2048 from `recipes/qwen3.toml` (not overridden).
- Tools unchanged from 2026-10-15: mlx 0.32.1, mlx-lm 0.31.3, mlx-metal 0.32.1 (`/Users/josef/.local/bin/mlx_lm.server`, `/Users/josef/models/Qwen3-8B`, `--prompt-cache-size 1`, `enable_thinking=false`); llama.cpp `llama-server` 0.6.0 build 11429 commit d81235049, BF16 GGUF `-ngl 999 -fa on -c 8192 --cache-ram 0`; macOS 27.0 (26A428); localeval 0.2.1, deepeval head 69c7a848bba836b70782aced73d278ee4b4d2fa6.
- fleetq: #2466 (batch 1), #2470 (batch 2), #2473 (control run, section 5) and #2474 (plain-prefill ABBA, section 6), all `--host endor --exclusive --mem 40 --expect 60`, released by the script trap, status done.
- localeval run ids: tables in section 1; result dirs copied to `docs/results/raw/2026-10-18-metal-sameslot-main/localeval-results/`.
- unanimis records (logged by localeval): main-1 96a90dea-4a8b-4dca-9bc9-793a2c7c14c0, main-mlx-lm 2eebf97d-4462-43bb-859c-88fc0b942a27, main-llamacpp 1aca5558-d894-4209-8a67-1d2f69d49ed3, main-2 5ed15dee-c7e1-4c43-9c81-d466b24a4ac4, retry-1 920ecf3b-1d20-43d2-9611-e265d725aa4c, retry-mlx-lm 86b40ea9-13fb-4cc4-96b5-610737c83c34, retry-llamacpp a87c4a01-c9fb-4951-ab14-69b9406c2476, retry-2 b41ff9fb-0ced-4c91-ae16-e66cf19320ff.
- Method differences from 2026-10-15: order ABBA with one intarsia build instead of SAFE/CAND/mlx-lm/llama.cpp/CAND/SAFE; run labels `main-*` / `retry-*`; macmon sampler added (raw JSON reduced to `*-macmon.tsv`); two batches instead of one. Everything else (ports 8093 / 8092 / 8091, server commands, idle gate, 5 s gap between arms, `analyze.py` formulas, the 40.6 floor) is as before. `sameslot-retry.sh` is `sameslot.sh` with the label prefix and output file names changed.

## 5. Control run: main versus the 2026-10-15 CAND binary, same slot

Follow-up to the open question in section 2 (machine state or code). One exclusive fleetq slot (#2473, 18:03-18:14 EDT), order ABBA: CAND, main, main, CAND, same `serve` flags and `localeval speed --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off` as before (`sameslot-control.sh`, labels `ctl-*`). CAND = `intarsia-cand` from the 2026-10-15 session (md5 0db48d81362233c9c8c44d69d9d092bf, built 2026-10-15 02:36, a65664d: SAFE + metal-p3-chunk, chunk 2048). main = 5eabb1a7 (md5 9810733ce9a084dcf83542a9065abe22, verified before the run). Both servers report `prefill chunk 2048 (recipe qwen3, metal)`, so the only variable is the code. Raw: `ctl-*` files, `compare/*.txt`, `analysis-control.txt`.

GPU idle before each run (median of 3 samples after the wait loop): 9, 10 (3 waits, samples up to 21), 0 (4 waits, samples up to 24), 10 (3 waits, samples up to 22). The box was contended again, a little less than in section 2; `pgrep zoom.us` empty. macmon median GPU clock over samples with utilisation above 50 %: cand-1 1283 MHz, main-1 1293, main-2 1284, cand-2 1339 (p10 1190-1254, minima 912-1168); GPU temperature 59-92 C in all four. The clocks are within 4 % of each other, so clock does not explain a difference between arms.

| run | localeval run id | decode 128 | decode 1k | decode 4k | marginal prefill 1k | marginal prefill 4k | prefill tok/s 4k (range) |
|---|---|---|---|---|---|---|---|
| ctl-cand-1 | 20261018-180407-adhoc-ctl-cand-1 | 36.0 (35.9-36.3) | 35.3 (35.2-35.9) | 33.7 (33.3-33.7) | 4320 | 3198 | 3338 (2558-3456) |
| ctl-main-1 | 20261018-180648-adhoc-ctl-main-1 | 36.0 (36.0-36.1) | 35.7 (35.0-35.8) | 34.3 (33.3-34.4) | 4402 | 3250 | 3392 (2464-3511) |
| ctl-main-2 | 20261018-180943-adhoc-ctl-main-2 | 36.0 (35.9-36.2) | 35.1 (35.1-35.8) | 33.8 (33.6-33.9) | 4454 | 2700 | 2925 (2090-3387) |
| ctl-cand-2 | 20261018-181218-adhoc-ctl-cand-2 | 35.8 (35.8-36.2) | 35.6 (35.4-35.9) | 34.7 (34.4-35.0) | 4342 | 3613 | 3673 (3578-3717) |

Mean of the two runs per binary: decode 128 / 1k / 4k CAND 35.90 / 35.47 / 34.16, main 36.00 / 35.42 / 34.03 (main/CAND +0.3 / -0.2 / -0.4 %); marginal prefill 1k CAND 4331, main 4428 (+2.2 %); marginal prefill 4k CAND 3406, main 2975 (-12.6 %); prefill tok/s 4k CAND 3506, main 3159 (-9.9 %).

`localeval compare` (A to B; speed differences under 10 % are marked noise, fleet finding 6a7efe40), unanimis records: cand-1 1ceb231b-3ff8-4cd4-b644-2c86de6ce984, main-1 5fd8702d-f82b-46ac-bf7d-e6553c0705b8, main-2 9b0e912c-70fe-4888-a8d1-ea7fb746feb0, cand-2 ad0fee0e-f2dc-4a42-b157-46b50323184e:

| comparison (A to B) | decode 128 / 1k / 4k | marginal prefill 1k | marginal prefill 4k | prefill tok/s 4k |
|---|---|---|---|---|
| cand-1 to main-1 (adjacent pair) | -0.0 / +1.1 / +1.8 %, noise | +1.9 %, noise | +1.6 %, noise | +1.6 %, noise |
| cand-2 to main-2 (adjacent pair) | +0.6 / -1.4 / -2.5 %, noise | +2.6 %, noise | **-25.3 %, slower** | **-20.4 %, slower** |
| cand-1 to cand-2 (same binary) | -0.5 / +0.8 / +3.0 %, noise | +0.5 %, noise | +13.0 %, faster | +10.0 %, faster |
| main-1 to main-2 (same binary) | +0.2 / -1.7 / -1.3 %, noise | +1.2 %, noise | **-16.9 %, slower** | **-13.8 %, slower** |

Verdict:

- **Decode, including 4k: machine state, not a regression.** Every decode comparison is noise (under 3 %), main/CAND is +0.3 / -0.2 / -0.4 % across 128 / 1k / 4k, and the old CAND binary itself now decodes 34.2 tok/s at 4k against 35.55 on 2026-10-15 (-3.9 %), the same loss main shows (34.0 vs CAND's 35.55 then, -4.3 %). The 4k decode floor FAIL in sections 1-3 (84.4-84.7 % of 40.6) is therefore the machine today: CAND at 34.16 is 84.1 % of the floor in this same slot. Under today's conditions the 85 % gate cannot be met by either build.
- **1k prefill: no difference** (+2.2 %, both pairs noise).
- **4k prefill: not resolved, leaning no regression.** The means put main 12.6 % (marginal) / 9.9 % (tok/s) below CAND, but that is carried entirely by the second pair: the adjacent first pair is +1.6 %, and `localeval compare` flags the second pair as slower (-20 to -25 %) at the same time as it flags main-1 to main-2 (the same binary) as slower by -14 to -17 %, while CAND against itself moved +10 to +13 %. The run-to-run spread of one binary at 4k (13-17 %) is as large as the main-versus-CAND gap, so it cannot be attributed to the code from four runs. One pattern to keep in view: in all three ABBA sequences of this report the second main run was the slow one at 4k (3406 to 2494, 3283 to 2344, 3250 to 2700 marginal), so a small real 4k prefill loss in main, or a position effect in the sequence, is not excluded. Plain-prefill ABBA (`generate --max-tokens 1`, the lower-noise measure in the 2026-10-15 gates) with more alternations would settle it; that was run afterwards and is section 6 (no difference).

## 6. Plain-prefill ABBA at 4k: CAND versus main (fleetq #2474)

Settles the 4k serve-path prefill question left open in section 5. One exclusive fleetq slot (#2474, 18:15-18:23 EDT), `generate --max-tokens 1 --profile-every 0 --max-seq 20000 --temperature 0 --backend metal` (the flags of `gates.sh` `prun()`), no chunk override (both builds log `prefill chunk 2048 (recipe qwen3, metal)`), the P1 4k prompt `docs/results/raw/2026-10-14-metal-phase-c/prompt-4k.txt` (4062 tokens, cached 0). Binaries as in section 5 (CAND md5 0db48d81362233c9c8c44d69d9d092bf, main md5 9810733ce9a084dcf83542a9065abe22). One discarded warm-up per binary (CAND 1226 ms, main 1181 ms), then 8 rounds, order flipped each round (C M, M C, C M, ..., i.e. C M M C C M M C C M M C C M M C), 20 s between runs, 8 measurements per binary. Script `plain-abba.sh`, analysis `analyze_plain.py`, raw `plain/` (stderr, idle, 100 ms macmon trace per run, `run-times.txt`), `analysis-plain.txt`.

Per-binary prefill, 8 runs each:

| binary | prefill ms median | IQR | min-max | tok/s median | IQR |
|---|---|---|---|---|---|
| CAND (2026-10-15) | 1209.5 | 1206.0-1217.0 | 1194-1232 | 3357.4 | 3337.5-3367.5 |
| main 5eabb1a7 | 1204.5 | 1197.2-1209.2 | 1179-1258 | 3371.7 | 3358.4-3391.7 |

main/CAND = 0.996 on median prefill ms (main 0.4 % faster), tok/s +0.4 %. The two IQRs overlap.

Paired per-round differences (main minus CAND within each round, prefill ms):

| round | order | CAND ms | main ms | main - CAND ms | main vs CAND % (ms) | CAND MHz | main MHz | MHz diff (main-CAND) | CAND temp | main temp |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | C M | 1208 | 1201 | -7 | -0.58 % | 1617 | 1617 | +0 | 58 | 60 |
| 2 | M C | 1206 | 1186 | -20 | -1.66 % | 1617 | 1617 | +0 | 51 | 62 |
| 3 | C M | 1211 | 1206 | -5 | -0.41 % | 1617 | 1617 | +0 | 54 | 60 |
| 4 | M C | 1194 | 1258 | +64 | +5.36 % | 1617 | 1617 | +0 | 65 | 63 |
| 5 | C M | 1232 | 1179 | -53 | -4.30 % | 1617 | 1617 | +0 | 53 | 59 |
| 6 | M C | 1206 | 1203 | -3 | -0.25 % | 1617 | 1617 | +0 | 51 | 62 |
| 7 | C M | 1215 | 1210 | -5 | -0.41 % | 1617 | 1617 | +0 | 55 | 58 |
| 8 | M C | 1223 | 1209 | -14 | -1.14 % | 1617 | 1617 | +0 | 65 | 63 |

Median paired gap -0.50 % (main faster), mean -0.42 %, range -4.30 % to +5.36 %. Main slower in **1 of 8** rounds (round 4, +64 ms, main-07 at 1258 ms, the slowest run of the 16), faster in 7 of 8, none equal.

Clock, temperature, idle: macmon at `-i 100` returned a single sample per run (it covers the run, with GPU utilisation 99.8 %), so the clock is one figure per run, not a trace. It read 1617 MHz in all 16 runs, the ceiling seen in the earlier traces, so there is no clock variation for the gap to follow (correlation undefined). GPU temperature read 51-65 C, CAND mean 56.7 C, main 60.8 C; prefill ms against temperature r = -0.01 over the 16 runs, paired gap against paired temperature difference r = -0.43 over 8 rounds (not significant at n = 8, and the sign is the opposite of heat slowing main). GPU power 96-100 W except main-03 (122 W, 1186 ms). GPU idle before each run (median of 3 ioreg samples; the first of the three reads 70-97 % in every run, an ioreg artefact also present in the earlier gates, which the median discards): 0-24 %, median 14 % for both binaries. So the box was as contended as in sections 1-5, but the contention was spread evenly over both binaries by the alternation. `pgrep zoom.us` empty.

Verdict by the stated rule (regression only if main is slower in at least 7 of 8 rounds and the median paired gap is above 3 %): main is slower in 1 of 8 rounds and the median gap is -0.5 %, so **noise, no regression**. Plain prefill at 4k is the same for both builds to within 1 %, with a one-run spread of up to 5 % either way. The 4k serve-path differences in sections 1 and 5 (main 12.6 % below CAND on the mean, second main run slow in every sequence) were therefore not a prefill compute difference; they came from the serve path's run-to-run variation (the same binary moved 13-29 % between runs there) and the order of runs, not from the engine's prefill kernels. Clock and temperature do not explain the residual spread here (clock constant, temperature uncorrelated).

## 7. Not done

No gates (cargo test, kernels test, accuracy, greedy-id identity, `qualify`): this was a speed re-measurement only. No 16k serve-path speed, and plain prefill was measured at 4k only (1k and 16k not repeated). mlx-lm and llama.cpp ran once per batch and were not part of the control or plain-prefill runs. The cause of the serve-path 4k run-to-run spread (section 1 drift) is not isolated.
