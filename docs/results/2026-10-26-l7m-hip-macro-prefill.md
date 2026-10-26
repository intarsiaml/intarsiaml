# Lane l7m: macro prefill pass on HIP for the 35B recipe (2026-10-26)

Branch `lane-l7/hip-macro`, base origin/main `b8fb4e6dd`. Host tatooine, one R9700 (card 1), Qwen3.6-35B-A3B-NVFP4, `serve` / `generate` on the HIP backend.

## What changed

- `recipes/qwen3_5_moe.toml`: `[tuning.hip] prefill_macro = 8192`. The macro pass (`INTARSIA_PREFILL_MACRO`, single-stream since lane sk, batched since PR #165) was CUDA-only by recipe; HIP had no `prefill_macro`, so it was off. No kernel and no model-loop code changed. The chunk stays 4096.
- `crates/intarsia-models/tests/recipe_macro.rs`: new test of the shipped sizes.
- `crates/intarsia-models/tests/batch_cuda_real.rs`: the macro-bitwise test asserted that a macro pass ran for every macro size, including 4096. That is right on CUDA (chunk 2048, so 4096 = two chunks) and wrong on HIP (chunk 4096, one chunk is the plain schedule by `effective_macro_rows`). New assert, see below.
- `docs/tuning.md`: `INTARSIA_PREFILL_MACRO` row (default is the recipe's value). `scripts/l7m/*`: lane scripts. `scripts/l7m/neg.patch`: the negative-control patch, applied to a throwaway copy only.
- Off switch: `INTARSIA_PREFILL_MACRO=0` (bitwise the old schedule, and `plan show` byte-identical to main).

## The assert change at `batch_cuda_real.rs:876` (reviewed by the coordinator before this window)

Old: `assert!(len < 2 * chunk || passes > 0, "... the macro schedule did not run")`. The first HIP run failed here at len 8292, macro 4096, "0 macro passes". It was not a bitwise mismatch: the compare on the next line never ran for that case, and all 12 earlier cases (lens 1536 and 4096) passed it. Macro 4096 with chunk 4096 is clamped to 0 by `effective_macro_rows`, so no pass is the specified behaviour.

New (commit 707f662e9): `macro_expected = len >= 2*chunk && rows >= 2*chunk`; assert a pass ran iff `macro_expected`. Both numbers are the test's own, not what the model reports. The bitwise compare and `assert_eq!(m.macro_rows(), 16384)` at the start are unchanged.

Shown in one window (fleetq #5676, `raw/final-gates/`):

| Run | Expected | Result |
|---|---|---|
| unchanged test, origin/main tree | fails at the same line (chunk size, not this branch) | FAILED at :876, len 8292 macro 4096, 0 passes (`old-main.log`) |
| new test, branch | passes, bitwise at macro 4096/8192/16384 | 2 passed in 52.9 s; passes: len 8292 1 (8192 and 16384), len 16655 2 (`bit.log`) |
| new test, control tree, no env | passes (hooks inert) | 1 passed (`neg-none.log`) |
| control `INTARSIA_L7M_NEG=nomacro` (macro_plan forced to `(0, 1)`, `macro_rows()` still 8192) | fails "did not run" | FAILED at :880, len 8292 macro 8192, 0 passes (`neg-nomacro.log`) |
| control `INTARSIA_L7M_NEG=ulp` (lowest bit of K[0] of the first attention layer flipped after a macro pass) | fails the bitwise compare | FAILED at :882 (`neg-ulp.log`) |

## Gates

| Gate | Result | Log |
|---|---|---|
| macro bitwise vs plain schedule (35B logits of last prompt row + 16 decode steps, 1.5k / 4k / 8.3k / 16.7k tokens, whole and recurrent schedule) | pass | `final-gates/bit.log` |
| batched engine macro == plain local engine | pass (6 prefill calls vs 17 chunks) | `final-gates/bit.log` |
| 35B batched == solo, `INTARSIA_PREFILL_MACRO=8192` (the shipped size) | 17,550 (row, step) pairs bitwise identical, 164.9 s; batched engine == local engine, long prompts, 3 tests pass | `gates/gc-m8192.log` |
| the same at 16384 | 17,550 pairs bitwise identical | `gates/gc.log`, `gc-long.log` |
| 256 greedy ids after 1.5k / 4.1k / 8.3k / 16.6k-token prompts: main vs branch default vs branch `INTARSIA_PREFILL_MACRO=0` | identical at all four lengths, 256 ids each | `final-gates/ids.log` (appended across runs; the last 12 lines are this window) |
| `plan show` text and json: main vs branch with macro off | byte-identical | `gates/plan.log` |
| `plan show`, main vs branch default | differs by design: scratch 167,905,792 B to 671,615,488 B (the macro workspace of the experts and the fp8 GEMM) | `gates/plan.log` |

## Speed (localeval speed, interleaved main/branch x3, 3 reps each, card 1, quiet window)

Prefill tok/s, median of 9; `raw/final-ab/`. Single stream = `serve` default; `--max-batch 16` = a lone prompt prefilling at the 16-slot server (no 16k prompt fits at max-seq 17408 there, so 12k).

| Mode | Prompt | main | branch (8192) | change |
|---|---|---|---|---|
| single stream | 1k | 7193 | 7214 | +0.3 % |
| single stream | 4k | 10312 | 10348 | +0.3 % |
| single stream | 16k | 9159 | 9441 | +3.1 % |
| `--max-batch 16` | 1k | 7145 | 7256 | +1.6 % |
| `--max-batch 16` | 4k | 10373 | 10420 | +0.5 % |
| `--max-batch 16` | 12k | 9403 | 9961 | +5.9 % |

Decode tok/s is unchanged (medians within -0.6 %..+0.7 %, same as the noise between rounds): the macro pass is a prefill schedule and a decoding step never takes it.

Earlier runs (raw in `raw/screen1`, `screen2`, `ab`): macro 16384 gave 16k +4.9 % single stream and 12k +5.7 % at `--max-batch 16`; macro 8192 gave +3.4 % at 16k. Chunk 2048 was rejected (4k -8.1 % with the macro off, -2.0 % with it).

## Cost

- Context under `serve --max-batch 16 --max-seq 17408`: 16223 tokens (macro off) -> 15410 with 8192 (-5.0 %) -> 13788 with 16384 (-15.0 %). Single stream: 17408 at every size. This is why 8192 ships and 16384 does not (`raw/gates/ctx.log`, `ctxplan-*`).
- Latency: a request that arrives while a macro pass runs waits for it (about 0.6 s per 4096 rows; at most about 1.2 s at 8192). The batched scheduler only takes a macro step for a prompt that prefills alone with nothing decoding (PR #165). A mixed check (12k long prompt + 1k short, `--max-batch 16`, `raw/mix`) showed short TTFT p50 0.154 s vs 0.156 s and p99 1.49 s vs 1.36 s (16384 branch, run before the size change); weak evidence, because the macro probably did not engage while the short requests decoded.

## Decision

L7 coordinator, 2026-10-26: keep `prefill_macro = 8192` default-on. The `--max-batch 16 --max-seq 17408` context goes from 16223 to 15410 tokens (-5.0 %, within the 5 % limit). `plan show` at the default moves by design (scratch 167,905,792 -> 671,615,488 B); the byte-identical `plan show` check is for changes that must not move the plan, and with `INTARSIA_PREFILL_MACRO=0` it is byte-identical to main, so it passed. 16384 (+4.5 % at 16k) is not shipped: -15 % context at 16 slots.

## Raw logs

`docs/results/2026-10-26-l7m-hip-macro-prefill/raw/`: `final-gates` (this window: old-main, bit, neg-*, ids), `final-ab` (12 localeval outputs, `window-driver.log`), `gates` (earlier window: G-C, plan, context), `ab`, `mix`, `screen1`, `screen2`. fleetq jobs: #5676 (window), #5671 (build), #5650 (G-C), #5649 (earlier gates).
