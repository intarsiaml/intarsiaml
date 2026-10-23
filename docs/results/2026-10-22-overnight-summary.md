# Overnight summary, 2026-10-22

## Decisions waiting for you

1. **Prefill chunk default (O12):** raise CUDA 8B from 256 to 4096 after a bitwise check. Recommend: queue the 256 vs 4096 bitwise check, then raise the default if it passes.
2. **PR #104 (35B 4k golden):** merging makes Metal 35B qualify FAIL on main. Recommend: set Metal 35B chunk to 2048 in the recipe, then merge #104.
3. **PR #102 (structured output):** the O11 gap is output length, not speed. Recommend: drop the neutral persistent-worker/fill code, merge the docs and opt-in `INTARSIA_CONSTRAIN_PROFILE` timers. Fixing the length is option (c), your call.
4. **PR #77 (Llama-3.1 / Qwen2.5 rope+bias):** needs about 1 h of exclusive endor. Recommend: approve the window (onboard quick about 11 min, Qwen3-8B Metal no-op about 40 min).
5. **New: sampling speed:** temperature > 0 decode is 26.3 vs 37.1 tok/s greedy (-29%) on Metal 8B. Recommend: work the follow-up task chip.
6. **Fable finding 9 (quant-format enum and accuracy-flag refactor):** still deferred for a design decision. Recommend: decide the design before any refactor.

## Merged overnight

| PR | what | key number |
|---|---|---|
| #90 | Metal MoE prefill, default on ([docs](2026-10-22-m35-metal-moe.md)) | 3x prefill; greedy ids matched main |
| #92 | Metal 35B op tests | 9/9 mutants killed |
| #93 | CI type-checks CUDA/HIP Rust via check-only switches | n/a |
| #94 | Metal 35B decode profile ([docs](2026-10-22-md-metal-35b-decode-profile.md)) | 124 tok/s, 55% of roofline, 764 dispatches/token |
| #95 | opt-in `INTARSIA_PREFILL_TAIL_MERGE`, HIP only ([docs](2026-10-22-tc-tail-chunk.md)) | HIP R9700 35B: 16k +3.2%, 4408-token +10%; not bitwise on CUDA |
| #97 | CUDA 35B prefill profile ([docs](2026-10-22-sp2-cuda-prefill.md)) | routed experts 35%, fp8 GEMMs 19-22%, GDN 13-14% |
| #98 | Metal quality, Qwen3-8B MMLU ([docs](2026-10-22-mq-metal-quality.md)) | intarsia 0.673 / mlx-lm 0.679 / llama.cpp 0.676 (tie) |
| #99 | Spark standings, 4 engines ([docs](2026-10-22-fx3-spark-standings.md)) | 35B decode #1 (+7-12% over vLLM at 128/1k/4k) |
| #87 | oracle-relative reuse gate ([docs](2026-10-22-fx5c-oracle-gate.md)) | warm passes if mean KL(oracle,warm) <= 1.5x cold + 3e-3 |
| #96 | warm-vs-cold oracle measurement ([docs](2026-10-22-fx5c-reuse-vs-oracle.md)) | noise, not a bug: CUDA 0.83x, Metal 1.09x |
| #103 | prefill-chunk bitwise study ([docs](2026-10-22-ck-prefill-chunk.md)) | CUDA 35B not bitwise above chunk 2048 |

#89 was closed as superseded.

## Open PRs

| PR | state | what's blocking |
|---|---|---|
| #104 | open | Decision 2. CUDA acc-4k gave no verdict: no 35B long golden, and spark1 memory guard (22 of 34 GB). CUDA tp=2 dump never recorded. ([docs](2026-10-22-gc-gate-coverage.md) on PR #104) |
| #102 | open | Decision 3. Structured output, no speed lever. ([docs](2026-10-22-so-structured-output.md) on PR #102) |
| #101 | open | Gemma-4 drill. Full 26B never ran onboard: no 20 GB+ lease admitted. 2-layer real-weight slice on CUDA: top-1 95.2%, KL 3.5e-3 vs HF. No time-to-quick number yet against the 2 h target. ([docs](2026-10-22-g4-gemma4-onboard.md) on PR #101) |
| #100 | open, draft | Metal fused decode folds draft. Lane result pending; a bisect was run. |
| #77 | open, draft | Decision 4. Needs about 1 h of exclusive endor. ([docs](2026-10-22-fx4-rope-bias.md) on PR #77) |
| #91 | open, draft | Won't merge. |
| #66 | open, draft | Won't merge. |

## Key findings

- CUDA 8B `--prefill-chunk 4096` gives 1.7-1.9x prefill (2.4k to about 4.1k tok/s). Chunk 256 vs 4096 is still unverified for bitwise output. ([docs](2026-10-22-ck-prefill-chunk.md))
- CUDA 35B is not bitwise above chunk 2048; greedy ids diverge at token 5. Cause not traced. All other backend/model pairs are bitwise. ([docs](2026-10-22-ck-prefill-chunk.md))
- Spark concurrency 1/4/16: intarsia CUDA is flat (no batching) while vLLM and SGLang scale. This is the biggest remaining Spark gap. ([docs](2026-10-22-fx3-spark-standings.md))
- The O11 1.7x structured-output gap comes from output length (an extra `reasoning` key when `additionalProperties` is unspecified). Constrained decode is only 1.8% slower per token. ([docs](2026-10-22-so-structured-output.md) on PR #102)
- Metal 8B 16k prefill gets 10.5% slower at chunk 4096. The earlier "byte-identical" claim used prompts too short to test this. ([docs](2026-10-22-ck-prefill-chunk.md))

## Process notes

- Endor was the bottleneck all morning: a stale job from another session (#4093) held it about 80 min, and an mdl reservation caused head-of-line blocking until re-queued.
- FX4's first reruns were invalid: built without `--features metal`. Goldens moved to `/Users/josef/Models/golden/`.
- The tc lane got only 2.4 min between two radiance downtime windows (rule is at least 5 min). Total radiance downtime was 38 min in 7 windows, all 20 min or less.
- vLLM on spark1 was down twice for SGLang (21 and 13 min). Both restored.
- The Gemma-4 lane was blocked by fleet capacity: lease #4206 (26 GB) was granted at 10:34 and sat idle until 11:40.
