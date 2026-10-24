# c8k: should the 35B CUDA recipe raise `prefill_chunk` from 2048 to 8192?

Lane c8k, 2026-10-24. Evidence for one user decision; the recipe default is not changed in this PR (`recipes/qwen3_5_moe.toml`
untouched). Follows #125 (fp8-lt default-on, merged `6189466f`); every number below is main `6189466f` (fp8-lt default), CUDA,
Qwen3.6-35B-A3B-NVFP4, GB10 (spark1 and spark2).

## Result

**Recommendation: adopt 8192 for the solo chunk, as a package, not as the one-line recipe edit.** The prefill gain is real and no
accuracy gate moves. But the one-line edit (a) turns `qualify` acc-4k red on CUDA for a structural reason, and (b) makes the batching
engine's busy chunk 8192 too, which lengthens the worst decode pause 1.75x (4k prompts) to 3.3x (16k). Both have a small, measured fix
that this PR does not make (the brief says to change nothing):

1. recipe `[tuning.cuda] prefill_chunk = 8192` (and the explanatory comment above it);
2. `qualify` acc-4k: run its in-run reuse at `min(plan chunk, 1024)` like the separate capped run already does, or drop reuse from that
   `--mode all` run (today: "nothing ran: no prompt can cross a chunk boundary at prefill chunk 8192 ... not a pass", exit 1);
3. busy chunk default `min(solo, 2048)` for the 35B on CUDA (today `--busy-chunk 2048` does this per deployment; measured below);
4. re-record the acc-short baseline at 8192 in the same change (identical numbers, see "Baseline").

If the user prefers not to touch qualify and batching now, the evidence also supports leaving 2048 as it is: the gain is 6-11 % on
prompts of 4k and longer and nothing below 2k changes.

| | chunk 2048 (main) | chunk 8192 |
|---|---|---|
| Solo prefill, 4k / 16k tok/s | 6598 / 6059 | 7023 / 6733 (**+6.4 % / +11.1 %**) |
| Solo prefill vs vLLM, 1k / 4k / 16k | -9.3 % / +2.2 % / -4.8 % | -9.0 % / +8.9 % / +5.8 % |
| Accuracy and reuse gates run by hand (thresholds unchanged) | 1 FAIL in the new 12k s6000 control (below) | **0 FAIL** (table below) |
| `qualify` acc-short / acc-long / acc-4k, real stages | PASS / PASS / PASS (exit 0) | PASS / PASS / **FAIL** (exit 1, structural) |
| Decode stall when a prompt arrives beside 3 decoding streams, 4k prompt (max ITL) | 336 ms | **590 ms** |
| same, 16k prompt | 383 ms | **1264 ms** |
| 4-stream aggregate decode | 96 / 112 tok/s (4k burst / stagger) | 99 / 112 tok/s (no change) |
| Stall with `--busy-chunk 2048` on top of 8192 | | 337 ms (4k), 382 ms (16k): the 2048 values |

What the decision turns on:

1. **Prefill gain is real and reproduces the lt numbers** (+8.9 % / +5.8 % vs vLLM at 4k / 16k here, +8.6 % / +6.4 % in the brief; 3
   interleaved rounds, per-round spread about 1 point). 1k is unchanged (a 1k prompt is one chunk either way).
2. **No accuracy gate moves.** KL goldens, acc-short against main's baseline, acc-4k accuracy and the prefix-reuse gate all pass at 8192
   with no threshold touched. Thinner than the 2048 evidence, stated plainly: the reuse gate at 8192 can only be judged on the two 12k
   prompts (a 4k prompt cannot cross an 8192 boundary), and `long_code_12k` continue has warm/cold 1.71x (limit 1.5x + 3e-3, passes at 71 %
   of the limit).
3. **The qualify acc-4k problem is structural, not numeric.** Its `--mode all` run on the 4k golden (prompts of 4064, 4064 and 4631 tokens)
   needs at least one prompt that crosses a chunk boundary; at 8192 none can, so the gate reports FAIL with "nothing ran". The accuracy
   part of that stage passes (floor PASS). The HIP recipe has the same shape at 4096 and survives only because `chat_reuse_4k` (4631)
   still crosses it.
4. **The cost is the decode stall under batching** (`--max-batch`, opt-in), not decode throughput. Steady-state ITL (p50 21-27 ms) and
   aggregate decode are the same at both chunks. What changes is the worst single gap each time a new prompt is prefilled while others
   decode: one chunk's worth of GEMMs, 336 ms -> 590 ms for a 4k prompt and 383 ms -> 1264 ms for 16k (vLLM, same window, 4k: max ITL
   313-654 ms). The stalls are fewer and longer: total stalled time is about equal (see the ITL section). `--busy-chunk 2048` removes the
   growth in full and keeps the solo prefill gain.
5. **Not covered:** `--busy-chunk 2048` under solo 8192 is not KL-gated (the existing `real_35b_busy_chunk_prefill_kl_gate` hard-codes
   solo 2048 and asserts it); it is by design not bitwise to the solo run, like the shipped 1024 busy chunk. Reuse at 8192 is judged on
   two prompts. The 12k s6000 oracles (below) make a main-side gate look bad at chunk 2048 in a hand-run arm; qualify does not exercise
   it (`q2048` passes).

## Gates at chunk 8192 (fp8-lt default, thresholds unchanged)

Binary: main `6189466f` clean (`head=6189466 code-tree=9cac1d016898 dirty=no`), `--prefill-chunk 8192` on the command line for the
accuracy runs. batch_cuda_real and `qualify` at 8192 run on a Spark copy whose recipe has `prefill_chunk = 8192` (the recipe override;
repo untouched, stamp `dirty=yes`; the override script edits both the `[tuning.cuda]` and `[tuning.metal]` lines, only the CUDA one
matters on these boxes and only that one would change). Controls at 2048 are the same build, same session.

| Gate | chunk 8192 | chunk 2048 (control) | Raw |
|---|---|---|---|
| KL golden short (10 prompts) | PASS | PASS | `raw/gates/ga2/kl-*-vllm-top20-c*.log` |
| KL golden long (5 prompts) | PASS | PASS | `raw/gates/ga2/kl-*-vllm-long-top20-c*.log` |
| KL golden 4k (3 prompts) | PASS | PASS | `raw/gates/ga2/kl-*-vllm-4k-top20-c*.log` |
| acc-short vs main's baseline (`--pre-change c37a8d3`) | PASS (floor PASS, baseline PASS; `--allow-config-mismatch` for the chunk) | PASS | `raw/gates/ga2/accshort-c*.log` |
| acc-4k accuracy (`--mode both`, `--min-top1 0` as qualify) | PASS | PASS | `raw/gates/ga2/acc4k-c*.log` |
| Prefix reuse, long golden (5 prompts) | PASS: 2 judged (12k), 3 skipped (4k, cannot cross) | PASS without the s6000 oracles (`ru`); **FAIL with them** (`ru2`: `long_docs_12k` continue at split 6000, warm 7.9e-2 vs cold 3.7e-3, 21x), see Finding | `raw/gates/ru/reuse-L*.log`, `raw/gates/ru2/reuse-L2048.log` |
| Prefix reuse, 12k at split 9500 (new oracles) | PASS: 2 of 2 judged | PASS 2 of 2 | `raw/gates/ru/reuse-S*.log` |
| Prefix reuse, 4k golden | 8192: structurally cannot run (all 3 prompts skipped, FAIL by design); 2048: PASS 3 of 3; 1024: PASS 3 of 3 | | `raw/gates/ru/reuse-F*.log` |
| batch_cuda_real (default tests, recipe override 8192) | 3 passed, 0 failed, 6 ignored | not rerun (lt flip run at 2048: `2026-10-23-lt-fp8-lt-reuse/raw/flip/batch.log`) | `raw/gates/gb/batch.log` |
| batch_cuda_real `--ignored` long prompts (2.6k, 4k, 16k) | 1 passed: R0-R3 IDENTICAL to the LocalEngine, busy prefill chunk 8192 | at 2048: lt flip run | `raw/gates/gb/batch-long.log` |
| `qualify` acc-short | PASS (floor only: "baseline not requested" in this run) | PASS | `raw/gates/q8192`, `raw/gates/q2048` |
| `qualify` acc-long | PASS | PASS | same |
| `qualify` acc-4k | **FAIL**: accuracy floor PASS, prefix reuse "nothing ran" | PASS (reuse at the 1024 cap: 3 of 3 judged with the new `-g4k` oracles) | same |

Failed gates, counted: **chunk 8192: 1** (qualify acc-4k, structural: no prompt of the 4k golden can cross an 8192 chunk; the hand-run
reuse arm F8192 fails the same way; every numeric gate passes). **Chunk 2048 (main, control): 1** (`long_docs_12k` continue at split 6000
against the new s6000 oracle, hand-run only; `qualify` on the same tree passes). The 2048 FAIL is not an 8192 result: 8192 never runs
that split (6000 does not cross an 8192 boundary).

### KL goldens, per prompt (2048 / 8192)

KL goldens use `accuracy --mode prefill`; short prompts are under 2048 tokens, so chunk 8192 is the same schedule (numbers identical to 6
digits, a useful harness control). The long and 4k goldens are where the schedule differs.

vllm-top20: gate 2048 PASS (multi-chunk prompts 0/10), 8192 PASS (multi-chunk 0/10)

| prompt | top-1 % 2048 / 8192 | KL21 2048 / 8192 | KL21 scored 2048 / 8192 | greedy-track 2048 / 8192 (tokens before first divergence) |
|---|---|---|---|---|
| count | 100.00 / 100.00 | 3.718e-03 / 3.718e-03 | 3.718e-03 / 3.718e-03 | 38/38 / 38/38 |
| story | 97.44 / 97.44 | 5.432e-03 / 5.432e-03 | 5.432e-03 / 5.432e-03 | 4/39 / 4/39 |
| code | 98.85 / 98.85 | 1.018e-03 / 1.018e-03 | 1.018e-03 / 1.018e-03 | 11/87 / 11/87 |
| science | 96.20 / 96.20 | 1.180e-02 / 1.180e-02 | 1.180e-02 / 1.180e-02 | 0/79 / 0/79 |
| chat_short | 94.44 / 94.44 | 8.360e-03 / 8.360e-03 | 2.048e-02 / 2.048e-02 | 2/18 / 2/18 |
| chat_system | 96.88 / 96.88 | 1.896e-02 / 1.896e-02 | 1.329e-02 / 1.329e-02 | 10/32 / 10/32 |
| chat_multiturn | 96.55 / 96.55 | 8.936e-03 / 8.936e-03 | 2.456e-02 / 2.456e-02 | 2/58 / 2/58 |
| chat_reasoning | 94.44 / 94.44 | 2.372e-02 / 2.372e-02 | 1.680e-02 / 1.680e-02 | 2/90 / 2/90 |
| long_prose_300 | 99.00 / 99.00 | 2.881e-03 / 2.881e-03 | 2.881e-03 / 2.881e-03 | 0/300 / 0/300 |
| long_prose_700 | 98.86 / 98.86 | 5.617e-03 / 5.617e-03 | 5.617e-03 / 5.617e-03 | 0/700 / 0/700 |

vllm-long-top20: gate 2048 PASS (multi-chunk prompts 5/5), 8192 PASS (multi-chunk 2/5)

| prompt | top-1 % 2048 / 8192 | KL21 2048 / 8192 | KL21 scored 2048 / 8192 | greedy-track 2048 / 8192 (tokens before first divergence) |
|---|---|---|---|---|
| long_docs_4k | 93.80 / 93.80 | 1.754e-02 / 1.802e-02 | 1.754e-02 / 1.802e-02 | 0/4064 / 0/4064 |
| long_code_4k | 97.02 / 97.12 | 7.306e-03 / 7.289e-03 | 7.306e-03 / 7.289e-03 | 18/4064 / 21/4064 |
| chat_reuse_4k | 87.39 / 87.76 | 2.775e-01 / 2.664e-01 | 8.541e-03 / 1.197e-02 | 8/4631 / 64/4631 |
| long_docs_12k | 93.12 / 93.24 | 2.099e-02 / 2.080e-02 | 2.099e-02 / 2.080e-02 | 16/12064 / 5/12064 |
| long_code_12k | 97.07 / 97.25 | 8.577e-03 / 8.642e-03 | 8.577e-03 / 8.642e-03 | 18/12064 / 21/12064 |

vllm-4k-top20: gate 2048 PASS (multi-chunk prompts 3/3), 8192 PASS (multi-chunk 0/3)

| prompt | top-1 % 2048 / 8192 | KL21 2048 / 8192 | KL21 scored 2048 / 8192 | greedy-track 2048 / 8192 (tokens before first divergence) |
|---|---|---|---|---|
| long_docs_4k | 93.75 / 93.63 | 1.659e-02 / 1.758e-02 | 1.659e-02 / 1.758e-02 | 0/4064 / 0/4064 |
| long_code_4k | 96.73 / 96.97 | 7.168e-03 / 7.038e-03 | 7.168e-03 / 7.038e-03 | 18/4064 / 21/4064 |
| chat_reuse_4k | 88.58 / 88.62 | 2.559e-01 / 2.587e-01 | 3.535e-02 / 2.652e-02 | 8/4631 / 64/4631 |

Differences between the chunks are in both directions. No prompt other than `chat_reuse_4k` moves by more than 1.0e-3 in KL21 or
0.4 points of top-1. `chat_reuse_4k` is the chunk-sensitive prompt the recipe comment names: its scored-rows KL21 goes 8.5e-3 -> 1.2e-2
on the long golden and 3.5e-2 -> 2.7e-2 on the 4k golden (up on one, down on the other), its greedy-track 8 -> 64 tokens; the all-rows
KL21 (2.7e-1) is dominated by the user turn, which the floor does not score. The gate passes at both chunks on all three goldens.

### Prefix reuse (accuracy --mode reuse), per case

Oracle-relative rule, unchanged: warm KL(oracle||warm) at most 1.5 x KL(oracle||cold) + 3e-3. All `restore` rows 32/32 and every `aligned`
row bit-identical to the cold run (the reuse path is exact at chunk boundaries).

| Arm (golden, chunk, split) | Prompt | cold KL(oracle) | warm KL(oracle) | warm/cold | limit (1.5x cold + 3e-3) | Verdict |
|---|---|---:|---:|---:|---:|---|
| long, **8192**, 12062 (default n-2) | long_docs_12k | 7.00e-3 | 5.22e-3 | 0.75x | 1.35e-2 | PASS |
| long, **8192**, 12062 | long_code_12k | 3.25e-3 | 5.57e-3 | **1.71x** | 7.87e-3 | PASS (71 % of limit) |
| long, **8192**, 9500 | long_docs_12k | 1.40e-2 | 4.97e-3 | 0.35x | 2.40e-2 | PASS |
| long, **8192**, 9500 | long_code_12k | 2.58e-3 | 1.58e-3 | 0.61x | 6.88e-3 | PASS |
| long, 8192 | 3 x 4k prompts | | | | | skipped: cannot cross a boundary |
| long, 2048, 9500 (control) | long_docs_12k | 2.28e-2 | 1.54e-2 | 0.67x | 3.72e-2 | PASS |
| long, 2048, 9500 | long_code_12k | 2.20e-3 | 3.22e-3 | 1.46x | 6.30e-3 | PASS |
| long, 2048, 6000 (default split) | long_docs_12k | 3.74e-3 | 7.93e-2 | **21.2x** | 8.60e-3 | **FAIL** |
| long, 2048, 6000 | long_code_12k | 2.46e-3 | 1.97e-3 | 0.80x | 6.69e-3 | PASS |
| long, 2048 | long_docs_4k (s3072) / long_code_4k (s3072) / chat_reuse_4k (s4521) | 1.72e-2 / 1.03e-2 / 5.07e-3 | 1.53e-2 / 1.25e-2 / 4.38e-3 | 0.89x / 1.22x / 0.86x | 2.87e-2 / 1.85e-2 / 1.06e-2 | PASS x3 |
| 4k golden, 2048 (new `-g4k` oracles) | long_docs_4k / long_code_4k / chat_reuse_4k | 8.50e-3 / 1.02e-2 / 5.07e-3 | 1.24e-2 / 1.53e-2 / 4.38e-3 | 1.46x / **1.50x** / 0.86x | 1.57e-2 / 1.83e-2 / 1.06e-2 | PASS x3 |
| 4k golden, 1024 (qualify's cap) | long_docs_4k / long_code_4k / chat_reuse_4k | 1.10e-2 / 6.78e-3 / 4.38e-3 | 1.18e-2 / 8.88e-3 / 4.86e-3 | 1.07x / 1.31x / 1.11x | 1.96e-2 / 1.32e-2 / 9.58e-3 | PASS x3 |
| 4k golden, 8192 | 3 prompts | | | | | all skipped: gate FAIL by design (nothing to judge) |

### Finding: the 2048 s6000 control (a FAIL that is not an 8192 result)

`long_docs_12k` continue at its default split 6000 against the new s6000 oracle fails at chunk 2048 (`ru2/reuse-L2048.log`): the cold run
is 3.7e-3 from the oracle, the warm (prefix-restored) run 7.9e-2, 21x, limit 8.6e-3. Without an oracle for that split the gate judged the
same case by greedy agreement only (`32/32 ok`), so this was not visible before the new oracles. `reuse-diag` on the same prompt
(`raw/gates/rd`) says it is not a restore defect:

| Run (long_docs_12k, split 6000) | mean KL(oracle \|\| cold) | mean KL(oracle \|\| warm) | Rule |
|---|---:|---:|---|
| chunk 2048 | 3.74e-3 | **7.93e-2** | FAIL (limit 8.60e-3) |
| chunk 1024 (what `qualify` uses) | **8.48e-2** | 6.20e-3 | PASS (limit 1.30e-1) |
| chunk 2048, `long_code_12k` | 2.46e-3 | 1.97e-3 | PASS |

At chunk 2048 the engine's own diagnostics for these 32 rows: cold chunk 2048 vs cold chunk 1024 mean KL 4.2e-2 (max 1.22, one argmax
flip); warm vs cold at the same chunk 4.5e-2; cold 2048 vs cold 2047 4.4e-3; cold 2048 vs cold 4096 bitwise identical; each run
repeated is bitwise identical (deterministic). So for this prompt the logits of the 32 rows move by about the oracle distance when only
the chunk schedule changes: whichever run lands on the far side of an unstable position is the one "far" from the oracle (cold at 1024,
warm at 2048). The oracle-relative rule (warm <= 1.5 x cold + 3e-3) passes when cold is the far one and fails when warm is. That is my
reading of the diagnostics, not proven by a bisect of the unstable position. It is the same class as the long_docs_4k near-tie of gc2,
at 12k. Not touched: the rule, its factors, the oracle.

Consequence for this PR: `golden/oracle` now holds a reference that a hand-run reuse at chunk 2048 and default split fails. `qualify`
on main's tree passes with it present (`q2048`, exit 0: its reuse runs are capped at 1024), so nothing on main's gates turns red, but
anyone running `accuracy --mode reuse --prefill-chunk 2048` on the long golden with this directory will see that FAIL. The s6000 files are
in their own commit (`c8k: continue oracles s6000 ...`) so the user can drop them with one revert; I did not.

## The references that were missing

The prefix-reuse gate needs a prompt that crosses a chunk boundary and, for the oracle-relative rule, an f64 CPU `continue` oracle whose
frozen split equals the split the gate runs. At 8192 only the two 12k prompts of the long golden can cross a boundary (the 4k golden has
none; its prompts are 4064, 4064 and 4631 tokens). New files, all `golden/oracle/`, each built as in
`2026-10-23-lt-fp8-lt-reuse.md` handoff and `2026-10-22-gc2` (frozen Metal cold stream on endor, f64 CPU oracle on endor, 38.7-38.9 GB
RSS, 14 min for 4k, 42 min for 12k; `scripts/c8k/endor-oracles.sh`, `endor-oracles-v2.sh`):

| File | Prompt | Frozen split | Frozen at chunk | For |
|---|---|---|---|---|
| `...continue-long_code_4k-s2000-g4k...` | 4k golden `long_code_4k` | 2000 | 1024 | acc-4k reuse at chunk <= 1024 (qualify's cap) |
| `...continue-long_code_4k-s3072-g4k...` | same | 3072 | 2048 | acc-4k reuse at 2048 |
| `...continue-long_docs_12k-s12062...`, `...long_code_12k-s12062...` | long golden 12k | 12062 | 8192 | reuse at 8192, default split (n - 2) |
| `...long_docs_12k-s9500...`, `...long_code_12k-s9500...` | 12k with `reuse_split` 9500 | 9500 | 1024 | reuse at 8192 and 2048 with 2.5k tokens of new text after the restored prefix |
| `...long_docs_12k-s6000...`, `...long_code_12k-s6000...` | 12k, default split at chunk <= 2048 | 6000 | 1024 | the chunk-2048 control |

`long_docs_4k` (s2000, s3072) and `chat_reuse_4k` (s4521) `-g4k` files already existed (gc2, #123); with `long_code_4k` the 4k golden
now has an oracle for every prompt at both qualify splits. `golden/reuse-splits/qwen36-35b-a3b-nvfp4-vllm-long-reuse-s9500.safetensors`
is the long golden with `reuse_split = 9500` on the two 12k prompts (metadata only; arrays and ids byte for byte;
`scripts/c8k/make_split_golden.py`). The oracle-build gate reports `FAIL` and exit 1 for every oracle (the "golden" is the teacher-forced
input, KL 2.3 against meaningless reference rows); the dump is written regardless, as in gc2. Each file's sha256 matches its sidecar.

## Speed, same window (spark2, retro down, vLLM up)

Method: localeval (`/opt/code/deepeval`, local-model-bench skill), `speed --sizes 1024,4096,16384 --gen 128 --reps 3`, one warmup, 3
interleaved rounds (vLLM, 2048, 8192; the middle round reversed), fresh server per visit, `--thinking off`. Prefill tok/s =
uncached prompt tokens / TTFT, median of 3 reps per visit; in brackets the per-round values. vLLM: `vllm-openai:65b7662d`, retro's
launcher flags with `--max-num-seqs 16` (lane vr), `--max-num-batched-tokens 8192`, fp8 KV, prefix caching on, on the same GPU. The
`spark2-vllm`/retro container was stopped with `fleetq --takes-down retro` and restarted afterwards (health and a chat completion
checked): down 1682 s in the measurement run. An earlier attempt (`raw/w0-aborted-script-edit`) stopped and restored retro after 484 s
because I edited the script while bash was running it; no numbers from it are used.

| prefill tok/s | 1024 | 4096 | 16384 |
|---|---:|---:|---:|
| vllm | 5494 (5480, 5494, 5520) | 6453 (6442, 6453, 6458) | 6362 (6309, 6362, 6363) |
| i2048 | 5009 (4924, 5012, 5009) | 6598 (6527, 6598, 6598) | 6059 (5993, 6059, 6065) |
| i8192 | 5005 (4986, 5035, 5005) | 7023 (7016, 7034, 7023) | 6733 (6740, 6733, 6724) |

| vs vLLM (same round, median of rounds) | 1024 | 4096 | 16384 |
|---|---:|---:|---:|
| i2048 | -9.3 % (-10.1, -8.8, -9.3) | +2.2 % (+1.3, +2.2, +2.2) | -4.8 % (-5.0, -4.8, -4.7) |
| i8192 | -9.0 % (-9.0, -8.4, -9.3) | +8.9 % (+8.9, +9.0, +8.8) | +5.8 % (+6.8, +5.8, +5.7) |

| 4 streams (aggregate decode tok/s, per-stream, TTFT s) | 1k | 4k |
|---|---|---|
| vllm | 156.2 (159.7, 155.7, 156.2); 47.7; 0.60 | 104.4 (105.1, 104.4, 103.7); 45.9; 2.12 |
| i2048 | 145.5 (145.5, 142.8, 147.2); 42.6; 0.51 | 96.2 (98.6, 96.2, 96.0); 34.5; 1.56 |
| i8192 | 144.1 (144.1, 145.3, 142.7); 42.2; 0.50 | 99.6 (101.8, 99.2, 99.6); 34.9; 1.45 |

Batched aggregate decode at both chunks (the `4 streams` table above, localeval `--concurrency 4`, `--max-batch 4`): at 1k prompts
145.5 vs 144.1 tok/s, at 4k 96.2 vs 99.6 tok/s: equal within the 3 % round spread, slightly better at 8192 because the 4 prompts
prefill in fewer, larger chunks (TTFT 1.56 s -> 1.45 s). vLLM aggregate is 156 / 104 tok/s.

## Decode stall under batching (ITL, 4 streams)

`scripts/c8k/itl-window.sh`, spark2, `--max-batch 4`, no vLLM, retro up and idle (0 running requests recorded before each visit),
3 interleaved rounds. localeval's `--concurrency` reports aggregate and per-stream decode only (no ITL), so `scripts/c8k/itl4.py` reuses
its client and prompt method and keeps every inter-token interval. Two scenarios, 4 streams of ~4k (and ~16k) tokens:
**burst** (all four start together: three prompts prefill beside the first stream's decode) and **stagger** (A decodes; B, C, D arrive 1.5 s
(4k) / 4 s (16k) apart, each prefilling beside the decoders). ITL = gap between streamed tokens, pooled over all streams.

4k prompts, 4 streams (ms; median of rounds, per-round values in brackets)

| arm | scenario | pooled ITL p50 | p99 | max | agg decode tok/s |
|---|---|---:|---:|---:|---:|
| chunk 2048 | burst | 22 (22, 23, 22) | 329 (327, 331, 329) | 338 (338, 337, 340) | 96 (96, 95, 96) |
| chunk 2048 | stagger | 21 (21, 21, 21) | 27 (27, 26, 28) | 336 (336, 336, 336) | 112 (111, 112, 112) |
| chunk 8192 | burst | 22 (22, 21, 22) | 584 (584, 583, 587) | 591 (594, 588, 591) | 99 (99, 102, 99) |
| chunk 8192 | stagger | 21 (21, 20, 21) | 26 (28, 26, 26) | 590 (589, 590, 590) | 112 (112, 113, 112) |
| chunk 8192 + --busy-chunk 2048 | burst | 22 (23, 22, 22) | 333 (333, 332, 334) | 338 (339, 338, 338) | 96 (96, 96, 96) |
| chunk 8192 + --busy-chunk 2048 | stagger | 21 (21, 21, 20) | 32 (32, 26, 46) | 337 (337, 338, 335) | 112 (112, 111, 113) |

16k prompts, 4 streams (ms; median of rounds, per-round values in brackets)

| arm | scenario | pooled ITL p50 | p99 | max | agg decode tok/s |
|---|---|---:|---:|---:|---:|
| chunk 2048 | burst | 27 (27, 27, 27) | 373 (371, 374, 373) | 383 (383, 382, 385) | 35 (35, 35, 35) |
| chunk 2048 | stagger | 26 (26, 26, 25) | 352 (352, 352, 351) | 383 (380, 383, 385) | 72 (72, 72, 72) |
| chunk 8192 | burst | 27 (27, 27, 27) | 1249 (1249, 1252, 1249) | 1262 (1258, 1262, 1262) | 38 (38, 38, 38) |
| chunk 8192 | stagger | 25 (24, 25, 25) | 36 (48, 32, 36) | 1264 (1265, 1264, 1262) | 75 (75, 74, 75) |
| chunk 8192 + --busy-chunk 2048 | burst | 27 (27, 27, 27) | 373 (373, 371, 373) | 382 (381, 382, 384) | 36 (36, 36, 36) |
| chunk 8192 + --busy-chunk 2048 | stagger | 26 (26, 26, 26) | 352 (348, 353, 352) | 382 (383, 380, 382) | 73 (73, 72, 73) |

vLLM in the speed window (4 streams of ~4k, same two scenarios, 2 reps x 3 rounds; the prefill-A/B windows, `raw/w1`):

| ITL, 4 streams of ~4k prompts (ms) | scenario | pooled p50 | pooled p99 | pooled max | agg tok/s |
|---|---|---:|---:|---:|---:|
| vllm | burst | 20 (20, 20, 20) | 33 (41, 28, 33) | 654 (654, 646, 655) | 105 |
| vllm | stagger | 20 (20, 20, 20) | 147 (147, 149, 147) | 313 (392, 312, 313) | 112 |
| i2048 | burst | 22 (22, 22, 22) | 329 (328, 331, 329) | 336 (336, 336, 336) | 97 |
| i2048 | stagger | 20 (21, 20, 20) | 38 (41, 36, 38) | 336 (337, 336, 335) | 111 |
| i8192 | burst | 22 (22, 22, 22) | 580 (580, 583, 578) | 588 (590, 588, 584) | 100 |
| i8192 | stagger | 20 (20, 20, 20) | 33 (44, 33, 33) | 593 (595, 593, 589) | 113 |


Reading: p50 and aggregate decode are unchanged. Intervals above 250 ms, per run (`pooled_itl_ms.gt_250ms` in `raw/i1/*.json`): 4k
burst 12 of 508 at 2048 (2.4 %) vs 6 of 508 at 8192 (1.2 %); 16k burst 48 of 508 (9.4 %) vs 12 of 508 (2.4 %); stagger 12 of 1276 vs 6 of
1276 (4k) and 48 of 1916 vs 12 of 1916 (16k). So the burst p99 *is* the stall, in the stagger scenario p99 stays at the decode floor and
the **max** is the stall. Fewer, longer pauses: at 8192 the stalled intervals are half (4k) or a quarter (16k) as many, each 1.75x (4k) or
3.3x (16k) as long; the total time the decoders spend stalled is about the same (4k: 12 x 336 ms = 4.0 s vs 6 x 590 ms = 3.5 s; 16k: 48 x
383 ms = 18 s vs 12 x 1264 ms = 15 s), because the prefill work is the same and 8192 does it a little faster. What gets worse is the
worst single pause, not the average. A stall is one
chunk of prefill: 336 ms for a 2048-token chunk, 590 ms for a whole 4064-token prompt in one 8192 chunk, 1264 ms for a full 8192 chunk
of a 16k prompt. `--busy-chunk 2048` returns all of it to the 2048 numbers (4k: 338 / 337 ms; 16k: 382 / 382 ms) while solo prefill stays
at 8192.

## Baseline

`golden/baselines/qwen36-35b-a3b-nvfp4-vllm-top20.cuda.baseline.json` records `prefill chunk 2048`; the accuracy tool's strict config
check refuses a baseline compare at another chunk ("WARNING: this run differs from the baseline: prefill chunk ..."), so the 8192 run
used `--allow-config-mismatch` (identity of the golden is still checked; no limit changed). Every acc-short prompt is under 2048 tokens,
so the run is the same single chunk and the mean KL is identical to the baseline's at both chunks (6.82514e-3 / 6.98641e-3). If the
default is changed, the same change should re-record that baseline at 8192 (numbers will be identical) or the next qualify run reads a
config mismatch.

## Incidents and limits

- **Shared Spark directories, my fault.** My build/rjob scripts rsynced into the shared `~/intarsia-decfuse/pf-golden` (and, from build.sh,
  `pf-scripts/lt`). (1) `pf-golden` short and long goldens became symlinks to Mac paths: spark2 from about 23:10 to 00:24, spark1 from
  about 00:16 to 00:24 (restored from the repo files, fleetq #4942/#4943); any other lane's accuracy run that read them in those windows
  failed to open them. (2) New oracles appeared in `pf-golden/oracle`: `-g4k` and s12062 (spark1 from about 00:16, spark2 from about 23:10, first sync of each), s9500
  (spark1 about 03:31, spark2 about 02:08), s6000 (spark1 about 03:31, spark2 05:12). The reuse gate finds oracles in the golden's directory, so a lane
  running reuse on the long golden at chunk <= 2048 and split 6000 after those times would have been judged against s6000 and seen the
  FAIL above. I checked the one reuse run I know of in the window (mp, spark1 `out/mpgB2/reuse.log`, 03:16): before s6000 arrived, and
  the other files were "not used". I did not check other lanes' runs. (3) `pf-scripts/lt` was overwritten with main's copies (no
  `--delete`). All lane files were moved out of the shared directory at about 05:25 and the lane now uses
  `~/intarsia-decfuse/pf-golden-c8k` on both Sparks (`scripts/c8k/private-golden.sh`). A lane that wants these references copies them from
  `golden/oracle` after this PR merges.
- Spark gate attempt 1 (`raw/gates/ga`): the 8192 short, long and acc-short arms failed to open the symlinked goldens (counted, not hidden);
  the whole set was rerun as `raw/gates/ga2`, which the tables use.
- The first reuse job was released before it started (`#4926`); its remote script kept running after the local wrapper died and was
  stopped by PID through fleetq. It produced no used output. The first speed-window attempt (`raw/w0-aborted-script-edit`) stopped and
  restored retro after 484 s because I edited the script while bash was running it; no numbers from it are used.
- `reuse-diag` runs print "Error: unsupported: cannot roll recurrent state back ..." for the restore case (diag mode keeps no snapshot
  there); the non-diag runs judge restore fine. The continue lines are what the table uses.
- Reuse at 8192 is judged on two prompts (12k docs, 12k code) at two splits each. Other lengths between 4k and 12k, and anything above
  12k, are not covered (the repo's goldens have no such prompts).
- The oracles are f64 CPU runs of one frozen Metal stream per prompt (teacher-forced). Their cold/warm distances change with the engine's
  chunk (`KL(oracle||cold)` 1.40e-2 at 8192 vs 2.28e-2 at 2048 for `long_docs_12k` split 9500); the rule compares warm to cold of the
  same run.
- Speed numbers: one machine pair and one prompt generator (localeval filler text); ITL pooled across the streams of one server.
- `qualify` here runs acc-short, acc-long and acc-4k only (the other stages skipped), `--no-unim`, and acc-short was floor-only ("baseline
  not requested"); the baseline compare is the hand-run row.

## Reproduce

Spark side (fleetq slots, `--quiet gpu,cpu`): `scripts/c8k/build.sh` (tree `c8k-main`, plus `RECIPE_CHUNK=8192 TESTS=1` for `c8k-r8192`),
`gatesA.sh` (KL goldens, acc-4k, acc-short), `gatesB.sh` (batch_cuda_real + long), `reuse.sh` (reuse arms), `qualify.sh`,
`speed-window.sh` (localeval prefill + 4-stream aggregate, vLLM in the window, `--takes-down retro`), `itl-window.sh` (ITL).
Table scripts: `gate-table.py`, `tables.py`, `itl-table.py`. Endor side: `metal-dump.sh`, `endor-oracles.sh`, `endor-oracles-v2.sh`.
`trace.json` records the run.
