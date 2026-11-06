# lm_head fp8 becomes the default for the Qwen3.8-27B on Metal and HIP

> User 2026-11-06: 27B fp8 default.

> User 2026-11-06: CUDA too.

Lane `lane-lh/lmhead-fp8-default`, worktree `.worktrees/lmhead-fp8-default`, base `origin/main` `af6aa5226`. This is the follow-up to `docs/results/2026-11-05-lmhead-fp8-27b.md`, which shipped the quantised head opt-in. The user's decision overrides the 1.0 % top-1 flip bound that doc set (fp8 measured 2.0 to 2.5 % flips in prefill on both backends and 1.0 to 1.6 % in the Metal decode window; MMLU 0.8368 against 0.8316 and GSM8K 0.52 against 0.48 for bf16).

**Status: Metal and CUDA done; HIP window pending (section "HIP" below).**

## What changed

- `recipes/qwen3_5_vl.toml` sets `lm_head_quant = "fp8"` under `[tuning.metal]`, `[tuning.hip]` and `[tuning.cuda]`. Only the Qwen3.8-27B recipe has the key, so every other model keeps its head.
- `INTARSIA_LMHEAD_QUANT=off` (or `none`) turns it back off; `=mxfp4` and `=fp8` still work as before. The env var wins over the recipe.
- `crates/intarsia-core/src/model.rs` no longer skips CUDA for the quantised head (the first lane had left CUDA out).
- Test `shipped_27b_recipe_lm_head_quant` in `crates/intarsia-models/tests/recipe_27b_chunk.rs` pins the three keys.
- Rows in `docs/tuning.md`, `docs/recipes.md`, `docs/decisions/LOG.md`.

The fp8 head is one f32 scale per tensor (not per row), as in the opt-in lane. The MTP draft head is the first 65536 rows of the same head tensor (first lane doc), so the one change covers both.

## Quality: not re-measured on Metal and HIP, measured on CUDA

The quantiser and the kernels are the ones the first lane measured, so Metal and HIP quality are the first lane's numbers (`docs/results/2026-11-05-lmhead-fp8-27b.md`): mean KL 6.2e-4 to 7.6e-4, prefill top-1 flips 1.95 to 2.45 %, MMLU 0.8368 vs 0.8316, GSM8K 0.52 vs 0.48 (HIP, 10 per task and 100 problems, so noisy). The pass rule written there (mean KL <= 1e-3 and flips <= 1.0 %) fails on flips. This doc does not claim it passes. fp8 is the default because the user decided so.

CUDA (spark1, GB10), 4k ids-prompts 4k-0..2, 12175 prefill positions and 3 x 128 teacher-forced decode positions, `intarsia accuracy` self-golden (raw: `raw/cuda-quality/`):

| | mean KL (21-way) | top-1 flips |
|---|---|---|
| prefill | 7.57e-4 | 308 of 12175 = 2.53 % |
| decode (teacher-forced) | 7.08e-4 | 7 of 384 = 1.82 % |

Same size as HIP and Metal. Caveat: `cquality.sh` wrote the reference ("self") golden without `INTARSIA_LMHEAD_QUANT=off`, so the reference is the fp8 head and the `off` arm is the one that carries the comparison (roles reversed against the first lane; the KL is not symmetric but the magnitude matches). The fp8 arm compares fp8 to itself: KL 1.6e-12, 0 flips. No CUDA 16k run and no CUDA MMLU/GSM8K run. Pass rule: same as the first lane; fails on flips (2.53 %).

## Gates

Metal (endor, job #8733; raw `raw/metal-gates/`):

- `plan show` byte-identical base vs head for the 8B, a Llama and the 35B. 27B head with `INTARSIA_LMHEAD_QUANT=off` is byte-identical to base. 27B base vs head differs only in the `logits` and `matmul_fp8` kernel lines.
- Greedy ids (256, short and ~4k prompt): 35B head == base (`38af2663`, `70ff3ce4`). 27B head with off == base (`382cc926`, `32154d87`); 27B head == base with `=fp8` (`84d3b76f`, `e0d97739`), i.e. the default is exactly the old opt-in. 27B head vs base differs, as intended.
- `kernels test`: base exit 0, 49 kernels, 980 cases, 0 FAIL lines (head the same; the diff is in `raw/metal-gates/kernels.diff`).

CUDA (spark1, jobs #8745, #8752; raw `raw/cuda-gates/`):

- `plan show` byte-identical base vs head for the 8B and the 35B, text and JSON. 27B head+off byte-identical to base. 27B base vs head: 6 changed lines in the text plan (`logits_bf16*` out, `matmul_fp8_gemv`, `_rows`, `_mma` in), 54 in the JSON.
- 27B ids, MTP k2 default, 256 ids: head run 1 == run 2 (`e94b8a9f` short, `06f2648f` 4k); head+off == base (`6d1d4ebc`, `d2e0487f`); head vs base differ, as intended.
- 35B ids head vs base: identical, short (`4e8fc691`) and 4k (`905cb3f0`), 256 ids each.
- `kernels test` was not run on CUDA (no change to a CUDA kernel in this lane).

HIP (tatooine): {HIP_GATES}

Workspace: `cargo test --workspace --no-fail-fast` on endor, jobs #8729 and #8742: exit 0, 1363 passed, 0 failed (log `.scratch/ws-final.log`, not committed).

## Speed

Metal (endor, 27B, 2 ABAB pairs, 1024 and 4096 prompts, decode tok/s, MTP k2 default; `raw/metal-speed-table.md`): the fp8 default is 1.52x (1k) and 1.31x (4k) of `off` by the medians, but the visits are very noisy (default 43.6 and 30.5 at 1k, off 23.7 and 25.0), so read this as "faster", not as a ratio. The first lane's Metal numbers (plain +9 to 11 %, MTP +21 to 23 %, job #8640) are the better figures.

CUDA (spark1, 27B, 1024 prompt, 128 tokens, 3 reps per visit, ABAB, `raw/cuda-speed-table.md`, job #8747 after #8745 and a script fix):

| Mode | default (fp8) | off (bf16) | ratio |
|---|---|---|---|
| MTP k2 | 27.0 tok/s | 24.7 | 1.094 |
| plain | 15.06 | 13.76 | 1.095 |

Visit spread was under 1.5 % within each arm. 4k and 16k were not measured on CUDA.

HIP: {HIP_SPEED}

## Not done

{NOT_DONE}

