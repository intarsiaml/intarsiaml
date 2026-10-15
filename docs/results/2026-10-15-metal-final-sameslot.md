# Metal final same-slot measurement: SAFE vs CAND vs mlx-lm vs llama.cpp

Qwen3-8B BF16, endor (Apple M5 Max, macOS 27.0 build 26A428). 2026-10-15, 02:35-02:55 EDT. Branch `metal-candidate2`. Raw data: `docs/results/raw/2026-10-15-metal-final/` (`gates.sh`, `sameslot.sh`, `analyze.py`, `plain_summary.py`, `gates/`, `sameslot/`, `analysis.txt`).

- **SAFE** = `spike/metal` a2102bf (B5, B6, P1 NAX GEMM, P2, P4/P5 tile pick + prefill qk_pf, F1b NAX flash attention; Metal prefill chunk 256).
- **CAND** = `metal-candidate2` a65664d = SAFE + `git merge --no-ff metal-p3-chunk` (Metal prefill chunk 2048 via `[tuning.metal] prefill_chunk = 2048` in `recipes/qwen3.toml`, plus P3's four Metal baseline/envelope JSONs). The merge was conflict-free; outside `docs/` it adds only `recipes/qwen3.toml` (+10 lines) and four `golden/baselines/*.metal.*.json`. P3 stays held for the `qualify` acc-long prefix-reuse FAIL at chunk 2048 (golden coverage gap: longest long-golden prompt 700 tokens), not for a numeric regression.

Both binaries: `cargo build --release -p intarsia-cli --features metal`, each from its own worktree (`/opt/code/intarsiaml-final-cand`, `/opt/code/intarsiaml-final-safe`; the main checkout was not touched), copied to `.../scratchpad/final/intarsia-{cand,safe}`. `plan show` confirms SAFE `prefill chunk 256 (default)`, CAND `prefill chunk 2048 (recipe qwen3, metal)`.

## 1. Same-slot speed (one exclusive fleetq slot, job #1965)

Order SAFE, CAND, mlx-lm, llama.cpp, CAND, SAFE. `localeval speed --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off`. Servers started and stopped one at a time; every port confirmed closed and pid gone before the next. GPU idle (3 samples) before each timed run: medians 0, 0, 0, 0, 1, 1 % (start limit 10 %); `pgrep -fl zoom.us` empty every time.

Decode tok/s median (min-max) at 128 / 1k / 4k prompts. Marginal prefill is one figure per row (delta prompt tokens / delta TTFT median versus the previous row). localeval stores only min/median/max per row (no per-rep values), so the marginal range is the extreme bound below the table, and the per-row prefill tok/s range is in the last two columns.

| label | localeval run id | prompt tok | decode 128 | decode 1k | decode 4k | marginal prefill 1k | marginal prefill 4k | prefill tok/s 1k (range) | prefill tok/s 4k (range) |
|---|---|---|---|---|---|---|---|---|---|
| final-safe-1 | 20261015-024254-adhoc-final-safe-1 | 113/991/4089 | 36.7 (36.7-36.9) | 36.6 (36.4-36.6) | 35.6 (35.6-35.7) | 4084 | 2761 | 3697 (3684-3700) | 2939 (2686-3455) |
| final-cand-1 | 20261015-024446-adhoc-final-cand-1 | 132/990/4088 | 36.8 (36.7-37.0) | 36.4 (36.3-36.6) | 35.5 (35.5-35.6) | 3813 | 2436 | 3517 (3468-3567) | 2630 (2629-2655) |
| final-mlx-lm | 20261015-024608-adhoc-final-mlx-lm | 112/991/4089 | 34.5 (34.4-34.5) | 34.0 (33.9-34.0) | 32.7 (32.7-32.7) | 4309 | 3244 | 2319 (2238-2319) | 2958 (2886-3062) |
| final-llamacpp | 20261015-024739-adhoc-final-llamacpp | 110/990/4091 | 35.4 (35.3-35.4) | 35.1 (35.1-35.1) | 34.0 (33.9-34.0) | 2885 | 1897 | 2503 (2382-2530) | 2013 (1930-2115) |
| final-cand-2 | 20261015-024937-adhoc-final-cand-2 | 115/990/4088 | 36.8 (36.7-36.9) | 36.6 (36.6-36.6) | 35.6 (35.4-35.7) | 4353 | 3902 | 3903 (3896-3909) | 3904 (3451-3927) |
| final-safe-2 | 20261015-025128-adhoc-final-safe-2 | 135/988/4087 | 36.7 (36.6-37.0) | 36.6 (36.6-36.6) | 35.6 (35.3-35.7) | 4024 | 3490 | 3680 (3674-3690) | 3535 (3271-3546) |

There is no marginal prefill for the 128 row (first row). The 1k row has about 990 uncached tokens, under localeval's 1024 threshold, so its prefill tok/s figures are flagged `*` as overhead-influenced mid-length numbers (documented noise there is up to +-39 %); the marginal column cancels the fixed overhead. mlx-lm's decode "peak" figure (best 32-token stretch) is 41.1 / 40.9 / 38.0 tok/s, well above its median decode of 34.5 / 34.0 / 32.7; intarsia's peak equals its median within 0.3.

Marginal prefill median (extreme bound from TTFT min/max of the two rows involved), tok/s:

| run | marginal 1k | marginal 4k |
|---|---|---|
| final-safe-1 | 4084 (4084-4262) | 2761 (2472-3386) |
| final-cand-1 | 3813 (3698-3900) | 2436 (2424-2470) |
| final-mlx-lm | 4309 (3977-5052) | 3244 (3132-3465) |
| final-llamacpp | 2885 (2708-3034) | 1897 (1795-2040) |
| final-cand-2 | 4353 (4332-4442) | 3902 (3328-3941) |
| final-safe-2 | 4024 (4005-4121) | 3490 (3162-3506) |

Two notes on the references. (1) mlx-lm decode in this slot (34.5 / 34.0 / 32.7) is lower than the P3 doc's earlier same-slot mlx-lm decode (36.0 / 36.0 / 34.7); against those earlier figures intarsia would be about 102 % rather than 106-109 %, so part of the margin over mlx-lm is an mlx-lm dip. The >= 90 % verdict holds either way. (2) mlx-lm's 41.1 "peak" decode is above the 40.6 tok/s byte floor; peak is localeval's best 32-token stretch (burst delivery in the stream), not sustained decode, and must not be read as mlx-lm beating the floor.

### Percentages (SAFE and CAND = mean of their two runs)

Prefill basis is the marginal figure (the cleanest one localeval gives); the prefill tok/s basis is shown as a cross-check.

| metric | SAFE | CAND |
|---|---|---|
| marginal prefill 1k, tok/s | 4054 | 4083 |
| % of mlx-lm (4309) / llama.cpp (2885) at 1k | 94.1 % / 140.5 % | 94.8 % / 141.5 % |
| marginal prefill 4k, tok/s | 3126 | 3169 |
| % of mlx-lm (3244) / llama.cpp (1897) at 4k | 96.3 % / 164.8 % | 97.7 % / 167.1 % |
| prefill tok/s 1k (mean of medians) / % mlx-lm (2319) / % llama.cpp (2503) | 3688 / 159 % / 147 % | 3710 / 160 % / 148 % |
| prefill tok/s 4k / % mlx-lm (2958) / % llama.cpp (2013) | 3237 / 109 % / 161 % | 3267 / 111 % / 162 % |
| decode 128 / 1k / 4k tok/s | 36.68 / 36.61 / 35.63 | 36.79 / 36.54 / 35.55 |
| decode % of 40.6 tok/s byte floor (128 / 1k / 4k) | 90.3 / 90.2 / 87.7 % | 90.6 / 90.0 / 87.6 % |
| decode vs mlx-lm (128 / 1k / 4k) | 106.3 / 107.8 / 109.0 % | 106.6 / 107.6 / 108.8 % |
| decode vs llama.cpp (128 / 1k / 4k) | 103.6 / 104.3 / 104.9 % | 103.9 / 104.1 / 104.6 % |

### Drift, run 1 to run 2 (same binary, same slot, 7-9 minutes apart)

| | decode 128 / 1k / 4k | marginal prefill 1k | marginal prefill 4k | prefill tok/s 4k |
|---|---|---|---|---|
| SAFE | 36.7->36.7 / 36.6->36.6 / 35.6->35.6 (0.0 %) | 4084->4024 (-1.5 %) | 2761->3490 (+26.4 %) | 2939->3535 (+20.3 %) |
| CAND | 36.8->36.8 / 36.4->36.6 / 35.5->35.6 (<= +0.5 %) | 3813->4353 (+14.2 %) | 2436->3902 (+60.2 %) | 2630->3904 (+48.4 %) |

Decode is stable to well under 1 %. Serve-path prefill is not: the 4k figure moved +20 % (SAFE) and +48 % (CAND) between runs, with run 1 the slower one for both binaries (SAFE prefill tok/s ranges 2686-3455 then 3271-3546; CAND 2629-2655 then 3451-3927). So SAFE-versus-CAND serve-path prefill differences below roughly 25 % at 4k are not resolvable from this slot; the cleaner A/B is the plain-prefill ABBA below. The only robust serve-path statement is that both builds sit in the same band as mlx-lm.

## 2. Verdicts

Thresholds: prefill committed >= 50 % of mlx-lm, target >= 65 %, stretch >= 85 %, at 1k and 4k; decode >= 85 % of the 40.6 tok/s byte floor, >= 90 % of mlx-lm, not below llama.cpp.

| criterion | SAFE | CAND |
|---|---|---|
| prefill committed (>= 50 % mlx-lm), 1k / 4k | PASS (94 % / 96 %) | PASS (95 % / 98 %) |
| prefill target (>= 65 %) | PASS | PASS |
| prefill stretch (>= 85 %) | PASS at both (marginal basis) | PASS at both (marginal basis) |
| decode >= 85 % of byte floor | PASS (90.3 / 90.2 / 87.7 %) | PASS (90.6 / 90.0 / 87.6 %) |
| decode >= 90 % of mlx-lm | PASS (106-109 %) | PASS (107-109 %) |
| decode not below llama.cpp | PASS (+3.6 to +4.9 %) | PASS (+3.9 to +4.6 %) |

Caveats on the prefill verdict: (1) it rests on marginal prefill, which is a difference of two TTFT medians and carries the large run-to-run spread above; on the prefill tok/s basis both builds are 109-111 % of mlx-lm at 4k (above it), and 159-160 % at 1k, where mlx-lm's own 1k TTFT (0.43 s) is mid-length-noisy (its 1k marginal, 4309, is higher than its 4k, 3244). (2) the single mlx-lm and llama.cpp runs have no repeat, so their reference values carry the same uncertainty. (3) The stretch margin at 4k is thin per run: CAND run 1 (marginal 2436 = 75 % of mlx-lm, 2630 tok/s = 89 % of mlx-lm's 2958) is under 85 % on the marginal basis, SAFE run 1 (2761 = 85.1 %) is at the edge; the means clear it only because run 2 of each was fast. Treat the stretch verdict as "met on the mean of two runs, not on every run".

SAFE versus CAND: no decode difference (within 0.2 tok/s). Serve-path prefill is statistically indistinguishable in this slot (means within 1.5 %, drift larger than the difference). Plain prefill ABBA (below) shows CAND ahead at 1k (+17 %) and 16k (+10 %) and tied at 4k.

## 3. Gate results (fleetq #1964, CAND unless noted)

- `cargo test -p intarsia-backend-metal --release -- --test-threads=1`: exit 0, 76 passed, 0 failed (`gates/cargo-test-metal.txt`).
- `intarsia kernels test --backend metal -v`: exit 0, 23 kernels, 272 cases, all ADMITTED (`gates/kernels-test.txt`).
- Accuracy, `--gate-class bf16`, both fp32 goldens, both builds: PASS everywhere, numbers identical to the expected SAFE values for CAND as well (no golden prompt crosses the 2048 chunk boundary, so CAND runs the same single-chunk path for 10/10 long prompts; SAFE splits 2/10 long prompts across 256-chunks).

| golden | mode | SAFE top-1 / KL | CAND top-1 / KL | verdict |
|---|---|---|---|---|
| short (434 pos) | prefill | 100.00 % (434/434) / 2.832e-4 | 100.00 % / 2.832e-4 | PASS both |
| short | decode (teacher-forced) | 100.00 % / 3.496e-4 | 100.00 % / 3.496e-4 | PASS both |
| long (1434 pos) | prefill | 99.72 % (1430/1434) / 3.699e-4 | 99.72 % (1430/1434) / 3.699e-4 | PASS both |
| long | decode (teacher-forced) | 100.00 % / 3.517e-4 | 100.00 % / 3.517e-4 | PASS both |

  Accuracy warnings (informational, both builds): "no prompt crosses a prefill chunk boundary" for the short golden (SAFE and CAND) and for the long golden at CAND chunk 2048; this is the known coverage gap.
- Greedy ids SAFE vs CAND (`generate --temperature 0 --ignore-eos --show-ids`, prompts `docs/results/raw/2026-10-14-metal-phase-b/prompt-short.txt` and `prompt-4k.txt`): identical for 48 and 256 tokens on both prompts (4 of 4 identical, no divergence; `gates/identity/summary.txt`).
- Plain prefill ABBA (`generate --max-tokens 1`, prompt-1k from `docs/results/raw/2026-10-14-metal-phase-c/`, prompt-4k same dir, prompt-16k from `.../metal-phase-b/b2a/`; A=SAFE B=CAND, each at its own default chunk, 3 rounds = 6 runs per binary per prompt, one discarded warm run; GPU idle 0-5 % at start). Medians with min-max:

| prompt | SAFE (chunk 256) ms | CAND (chunk 2048) ms | SAFE tok/s | CAND tok/s | CAND / SAFE |
|---|---|---|---|---|---|
| 1k (about 973 tok) | 444 (430-448) | 378 (372-388) | 2194 | 2570 | 1.17x |
| 4k (about 4062 tok) | 1490 (1329-1716) | 1506 (1209-1685) | 2727 | 2701 | 0.99x (noise: ranges overlap fully) |
| 16k (about 16.2k tok) | 8600 (7519-8800) | 7794 (7727-8041) | 1928 | 2127 | 1.10x |

  The 4k and 16k ranges are wide (the first round was fast, later rounds slowed for both builds; 16k SAFE 7519 ms in round 1, 8.6-8.8 s later), so only the 1k gain is clear-cut here; P3's own sweep reported +19 % at 1k, +19 % at 4k, +4 % at 16k on a slightly earlier base.

## 4. Versions and IDs

- intarsia: SAFE head a2102bf3ce2ac7f6ff7357e229ea0a0670087c87; CAND head a65664d286ae73b151dc4dcbfc82229a24014c71 (branch `metal-candidate2`, parent a2102bf + metal-p3-chunk 177da74).
- mlx 0.32.1, mlx-lm 0.31.3, mlx-metal 0.32.1 (`/Users/josef/.local/bin/mlx_lm.server`, model `/Users/josef/models/Qwen3-8B`, `--prompt-cache-size 1`, `enable_thinking=false`).
- llama.cpp: `llama-server` version 0.6.0 build 11429 commit d81235049 (Homebrew, AppleClang 21.0.0), BF16 GGUF `/Users/josef/models/Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf`, `-ngl 999 -fa on -c 8192 --cache-ram 0`. (Phase B used build 11146 / 0.5.0; the llama.cpp build differs from that earlier report.)
- macOS 27.0 (26A428). localeval 0.2.1, deepeval 4.2.6, harness git 69c7a84 (deepeval head 69c7a848bba836b70782aced73d278ee4b4d2fa6).
- intarsia serve flags: `--backend metal --device 0 --max-seq 8192 --profile-every 0 --no-supervisor --no-ui`, env `INTARSIA_CPU_FREE_MEM_GB=60`.
- localeval run ids: see the table in section 1. unanimis records: safe-1 b416f223-385d-4bc2-8a97-6c82e060c84d, cand-1 8781d599-6415-496a-b186-1f10e2b3861b, mlx-lm 0d0e8d94-a6e7-4303-9466-1f15a127b1ee, llamacpp 440fcc65-1e5f-4e9e-ac87-cff8e2c69fcd, cand-2 fb6cc4ef-0467-433b-8912-36432a076229, safe-2 23a279e5-1430-4692-8ef0-c2c4e2b289bc. Result dirs copied to `docs/results/raw/2026-10-15-metal-final/sameslot/localeval-results/`.
- fleetq: #1964 gates (exclusive, 24 GB), #1965 same-slot (exclusive, 40 GB); both released by trap, status done.

## 5. Not done

No `qualify` run (CAND is known to FAIL acc-long prefix-reuse at 2048; unchanged from P3). No 16k serve-path speed. mlx-lm and llama.cpp were run once each (no repeat, so no drift estimate for them).

Binary md5: intarsia-safe 9ead26226e096c542a4ee7ba63e9dbef, intarsia-cand 0db48d81362233c9c8c44d69d9d092bf (paths under the session scratchpad `final/`).
