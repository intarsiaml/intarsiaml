# CUDA 35B prefill chunk 8192: the package, and why it is not merged

Lane df, 2026-10-25. User decision 2026-10-25 00:06 EDT: "turn the MoE prefill speed-up on by default and use 8192-token prefill chunks on CUDA".
The first half (PF default-on) is merged (#131, `docs/results/2026-10-25-df-moe-pf-default.md`). This is the second half, built on the evidence of
#130 (`docs/results/2026-10-24-c8k-prefill-chunk-8192.md`, which this branch carries). **Status: implemented and gated, one gate fails, not merged**
(the failing gate is the busy-chunk KL gate; loosening it, or choosing another option below, is the user's call).

## What the branch does

1. `recipes/qwen3_5_moe.toml` `[tuning.cuda] prefill_chunk = 8192`.
2. Busy chunk (prefill chunk of a prompt that starts while other streams decode, `serve --max-batch`): default `min(solo chunk, 2048)` on CUDA
   (`default_busy_chunk`, `crates/intarsia-models/src/load.rs`; unit-tested). Other backends keep busy = solo (HIP 4096, Metal 2048). At a solo chunk of
   2048 or less nothing changes.
3. `qualify` acc-4k: when the reuse gate of `accuracy --mode all` cannot work at the plan chunk (no 4k prompt can cross an 8192 chunk, or the plan
   stage was skipped so the chunk is unknown), the stage runs `--mode both` and the reuse is judged by its capped run at `min(plan chunk, 1024)` (which it
   already did). Every other plan chunk (2048, HIP 4096, Metal 2048) takes the old path (`acc_4k_mode`, unit-tested).
4. acc-short baseline + noise envelope, and the CUDA self-golden dump of the acc-oracle stage, re-recorded at 8192 from the stamped clean build
   `debb7cf` (code-tree 119a1e58ae49). acc-short numbers are identical to the 2048 baseline (every acc-short prompt is under 2048 tokens: one chunk at
   either); the envelope keeps its perturbation (`chunk=512`, same measured noise); only the identity fields change. The self-golden changes in content
   too (the committed dump was written by an old unstamped build), the oracle floor passes at 8192 (prefill top-1 95.99 %, KL 1.06e-2).
5. The busy-chunk KL gate (`real_35b_busy_chunk_prefill_kl_gate`) and the long-prompt identity test follow the chunks instead of hard-coding solo 2048.
6. New `golden/oracle/` 12k and `-g4k` oracles from #130 come along. The `long_docs_12k` / `long_code_12k` split-6000 pair (`780bd40a`) is **dropped**
   (reverted): `long_docs_12k` at its default split 6000 failed a hand-run reuse at chunk 2048 (warm 7.9e-2 vs cold 3.7e-3, a near-tie: cold 2048 vs cold
   1024 differ by KL 4.2e-2), and it matters only for plan chunks of at most 2048 on that file, which is Metal's and the old CUDA default. Without the
   file the gate falls back to the warm-vs-cold comparison there (passes: 5 of 5 judged at chunk 2048 and 1024, below). Nothing is relaxed; a gate that is
   red on a file nobody needs for the 8192 plan is not shipped.

## The comparison (spark2, one window per measurement, 3 rounds interleaved)

| solo / busy chunk | prefill tok/s 1k | 4k | 16k | worst decode stall 4 streams, new 4k / 16k prompt (max ITL) | busy-chunk KL gate | bitwise batched = solo |
|---|---|---|---|---|---|---|
| **2048 / 2048** (main, PF on) | 5311 | 6614 | 6040 | 338 / 385 ms | n/a (busy = solo) | yes |
| **4096 / 4096** (d) | 5308 (-0.1 %) | 7008 (+6.0 %) | 6437 (+6.6 %) | 594 / 681 ms (1.76x / 1.77x) | n/a (busy = solo) | yes |
| **8192 / 8192** (b) | 5342 (+0.6 %) | 6988 (+5.7 %) | 6671 (+10.4 %) | 592 / 1255 ms (1.75x / 3.26x) | n/a (busy = solo) | yes |
| **8192 / 2048** (a, this branch) | 5342 (+0.6 %) | 6988 (+5.7 %) | 6671 (+10.4 %) | 339 / 384 ms (1.00x / 1.00x) | **FAIL** (below) | **no** |

Speed: `localeval speed` sizes 1024/4096/16384, gen 128, 3 reps, thinking off, one fresh `intarsia serve` per arm per round, round 2 reversed
(`raw/speed/`, `raw/speed-table.md`); percentages against the first row. Rows (b) and (a) have the same solo prefill (the busy chunk only matters with
`--max-batch`), measured once as the 8192 arm. Stall: `itl4.py`, 4 streams, `serve --max-batch 4`, a new 4k or 16k prompt arriving (burst scenario max
ITL, median of 3 rounds of 2 reps; the stagger scenario's max is within 3 ms of it; p99 and aggregate decode in `raw/itl-table.md`); (b) is main with
`--prefill-chunk 8192` (busy = solo), (d) is main with `--prefill-chunk 4096`, (a) is this branch with its defaults. Aggregate 4-stream decode is 97-100
tok/s (4k prompt) and 35-39 tok/s (16k prompt) in all four, p50 ITL 22 / 26 ms. Per-round values and the arms' serve logs are in `raw/itl/`.

Reading:

- The stall limit (about 1.1x of 2048: 338 / 385 ms) is met only by (a). 4096 and 8192-with-busy-8192 stall every decoding stream for a whole prompt
  chunk: 4096 holds a 4k prompt (594 ms) or a 4096-token chunk of a 16k prompt (681 ms); 8192 holds 1255 ms for a full chunk of a 16k prompt. Per the
  brief no gates were run at 4096 (stall above 1.1x).
- **8192 without PF beats 2048 with PF at 4k**: PF engages only for chunks of at most 2048 tokens (64 pairs per expert), so on 4k and 16k solo
  prefill at 8192 it does not run. Measured in the same window: 4k 6988 vs 6614 tok/s with PF at 2048 (+5.7 %), 16k 6671 vs 6040 (+10.4 %); at 1k both
  chunks are one chunk and PF engages (5342 vs 5311, +0.6 %). Before PF existed c8k measured +6.4 % at 4k; PF is worth +1.4 % at 4k and +5.9 % at 1k at 2048
  (2048 PF=0: 6525 / 5012), so the 8192 gain over the PF=0 baseline is +7.1 % at 4k. 4096 gets +6.0 % at 4k, the same as 8192, and +6.6 % at 16k.
- The decode speed is unchanged (85 / 82.5 / 75 tok/s at 1k / 4k / 16k, all arms).

## The failing gate: busy-chunk KL vs the solo run

`real_35b_busy_chunk_prefill_kl_gate` (35B on CUDA, 5 prompts of 1.5k to 16k tokens of repository docs text, 64 greedy decode rows each; prefill the prompt with
the solo chunk, and in the busy chunk, teacher-force the same tokens; KL(solo || busy) per row; floor = the acc-short ref-noise 3.161e-3 mean 21-way KL,
factor 1, unchanged). Here solo = 8192 (control: hand-chunked solo equals `forward`, KL exactly 0), busy = the new default 2048:

| prompt tokens | mean 21-way KL | floor 3.161e-3 | top-1 flips /64 |
|---:|---:|---|---:|
| 1536 | 0 (one chunk either way) | pass | 0 |
| 2560 | 2.633e-3 | pass | 1 |
| 4096 | **1.095e-2** | **fail (3.5x)** | 2 |
| 8192 | **7.453e-3** | **fail (2.4x)** | 2 |
| 16384 | 3.142e-3 | pass (barely) | 0 |

All 320 rows: mean 21-way KL 4.84e-3, full-vocab 6.39e-3, 5 top-1 flips (1.6 %). `raw/dfB-gb/klgate.log`. The gate also reports the opt-in busy 1024 and 256
(1024: 6.2e-3 at 1536, 7.7e-3 at 2560, 1.2e-2 at 4096, 8.5e-3 at 8192, 2.2e-3 at 16384; the same class of failure as rb2's finding for busy 1024 against solo 2048,
`docs/results/2026-10-23-rb2-cuda-batching.md` section 11, which is why the default busy chunk was made equal to the solo chunk). So the package as
specified (busy 2048 beside solo 8192) is not bitwise and exceeds the floor on 2 of 5 prompts.

What is not affected: both the 8192 and the 2048 schedule are individually inside every golden gate (3 KL goldens, below); a batched request that prefills
beside decoding streams equals a `LocalEngine` loaded at chunk 2048 token for token (all requests, 3 scenarios, `raw/dfB-gb/batch-long.log`), and with
busy chunk set to 8192 equals the `LocalEngine` at 8192. What differs is that in a busy server the same prompt gets the 2048 numerics and in an idle one
the 8192 numerics, by more than vLLM's own run-to-run noise on some prompts.

Options (user's call): (a) accept it and turn this gate into a report (or raise its floor) for the busy-vs-solo difference: loosens a gate; (b) busy = solo
8192, bitwise, but the stall is 592 / 1255 ms (1.75x / 3.3x); (c) keep 2048 (no change from main: only PF default-on, #131); (d) 4096 is bitwise but
stalls 594 / 681 ms and gains about the same as 8192 at 4k, 3.5 % less at 16k.

## Gates (spark2 `--quiet gpu,cpu`; base tree = main after #131 (`df-dev`); PR B tree `df-pfb` = this branch at `debb7cf` code-tree 119a1e58ae49, final tree `df-pfc` for qualify; lane-private golden dir `df-golden`)

| gate | result | fleetq / raw |
|---|---|---|
| 3 KL goldens at the recipe chunk 8192 (short, long, 4k) | golden gate PASS x3; 21-way KL (scored rows) 5.408e-3 / 1.419e-2 / 1.247e-2 (at 2048: 5.408e-3 / 1.417e-2 / 1.214e-2); floor 3.0e-2 unchanged | #4984, `raw/dfB-kl/` |
| acc-short against the re-recorded baseline + envelope (`--pre-change debb7cf`) | PASS: prefill 6.82514e-3 (baseline 6.82514e-3, limit 7.84891e-3), decode 6.98641e-3 (8.03437e-3), per-prompt limits from the envelope (k = 2) | #4991, `raw/dfB-rebase2/` |
| `qualify` acc-short / acc-long / acc-4k (c8k's stage selection, plan skipped) at 8192 | PASS / PASS / PASS: acc-short prefill 6.83e-3 decode 6.99e-3; acc-long prefill top-1 94.35 % KL 4.28e-2, decode 97.50 % 1.04e-2; acc-4k prefill top-1 92.88 % KL 8.76e-2, decode 97.92 % 1.01e-2 (at 2048 on main: 6.83e-3 / 6.99e-3; 94.20 % 4.35e-2, 98.12 % 1.02e-2; 92.82 % 8.60e-2, 97.92 % 9.78e-3). acc-4k reuse runs at min(8192, 1024) = 1024 and judges the 4k prompts | #5087 (final tree), #5079, #5075 (main 2048), `raw/qualify/` |
| `qualify` acc-short / acc-long / acc-4k, default stages | recipe, plan, gate, acc-short, acc-oracle (against the re-recorded dump), acc-long: PASS; acc-4k: MISSING-BASELINE, which is not the chunk: golden/baselines has no CUDA baseline for the 4k golden (only Metal's), so the stage reads MISSING-BASELINE on main at 2048 too unless the `--model-name` selection of c8k's invocation is used (there it is PASS, row above); acc-format SKIP | #5087, `raw/qualify/` |
| prefix-reuse gate, long golden (committed CPU continue oracles incl. the new 12k ones) | recipe chunk 8192: PASS, 2 judged (the 12k prompts), 3 skipped (the 4k prompts cannot cross an 8192 chunk, by design); chunk 1024: PASS 5 of 5 judged; chunk 2048: PASS 5 of 5; reuse-split golden s9500 at 8192: PASS 2 judged | #5034, `raw/dfB-reuse2/` |
| `batch_cuda_real` (35B and 8B) | 3 passed, 0 failed (6 ignored) | #4985, `raw/dfB-gb/batch.log` |
| `batch_cuda_real --ignored real_35b_batched_engine_long_prompts_equal_the_local_engine` | 1 passed: 4 requests x 3 scenarios IDENTICAL to the LocalEngine at the busy chunk (2048); with `--busy-chunk 8192` IDENTICAL to the LocalEngine at 8192 | #4985, `raw/dfB-gb/batch-long.log` |
| `batch_cuda_real --ignored real_35b_busy_chunk_prefill_kl_gate` | **FAIL** (above) | #4985, `raw/dfB-gb/klgate.log` |
| prefix reuse / unit: backend-cuda moe tests, `kernels test --quick` (77 kernels, 531 cases, 0 FAIL, PF on and off), `docs_env_vars` | pass | #4986, `raw/job-bgt.log` |
| decode stall, 4 streams, new 4k / 16k prompt (max ITL) | 339 / 384 ms against main's 338 / 385 ms (<= 1.1x) | #5030, `raw/itl/` |
| prefill speed 1k / 4k / 16k vs main, interleaved x3, same window | +0.6 % / +5.7 % / +10.4 % | #5031, `raw/speed/` |
| no-op, 8B CUDA: plan show (json) and greedy ids (256 after ~1.2k tokens, 64 after ~4k), main vs this branch | byte-identical plan, 2 of 2 ids identical | #4979, `raw/dfB-noop/` |
| no-op, 35B CUDA at an explicit `--prefill-chunk 2048` (busy-chunk change leaves chunk <= 2048 alone) | plan byte-identical, 4 of 4 ids identical | #4980, `raw/dfB-noop2/` |
| 35B CUDA at the new default (8192): plan show differs only in the chunk and the scratch sizes (84 -> 336 MB); ids | 256 after 1.2k identical (one chunk either way); 64 after 3.4k differ (one 8192 chunk against 2048 + 1.3k: the chunk numerics above, inside the goldens) | #4979 |
| no-op, Metal batching off (endor): plan show 8B / 35B and ids (8B and 35B, short and ~4k), main vs this branch | plans byte-identical, 4 of 4 ids identical | #4989, `raw/metal-noop/` |
| no-op, HIP batching off (tatooine, `--takes-down radiance`, 47 s of downtime, GPU 0 main vs GPU 1 this branch in parallel): plan show 8B / 35B and ids | plans byte-identical, 4 of 4 ids identical | #5029, `raw/hip-noop/` |

The first c8k-style qualify run on `debb7cf` (`raw/qualify/dfB-q3-new`, #5079) failed acc-4k with "nothing ran" because that invocation skips the plan stage, so the stage did not know the plan chunk and kept `--mode all`; `acc_4k_mode` now treats an unknown chunk like an uncrossable one (final tree `825143d`, rows above). The acc-4k MISSING-BASELINE of the default-stage run is a missing committed CUDA baseline for the 4k golden, not a result of this change; `raw/qualify/dfB-q` is the first default-stage run (acc-oracle MISSING-BASELINE until the dump was re-recorded at 8192).

Why the Metal and HIP runs: the busy-chunk default lives in code HIP and Metal share (`load.rs`, `matches!(backend_name, "cuda" | "hip" | "metal")`). The
change is gated on the backend name (`default_busy_chunk`; unit test: HIP and Metal keep busy = solo for every chunk), the runs show the engine path is
untouched with batching off.
