# RIP: row-invariant prefill on CUDA (dense Qwen3-8B): any split, busy chunk or prefill-written prefix is bitwise the cold prefill

Lane `rip`, branch `lane-rip/row-invariant-prefill`, base `origin/main` `f2bb3410f` (#231, BI8). spark2 (spark-fa00, GB10 sm_121),
`spark2-vllm` (retro, 38 GB) up and idle throughout. Follows BI8 (`2026-10-29-bi8-serve-batch-invariance.md`), which made prefix
reuse exact by reusing only where the cold schedule cuts. Raw: `2026-10-29-rip-row-invariant-prefill/raw/`.

## Result

**On CUDA, a dense prefill pass's bits depended on its row count only through small-M kernel dispatch. Pinning one GEMM algorithm
for every M and running the flash attention at every row count makes every row's bits independent of the pass.** On Qwen3-8B the
one-pass prefill of a 2164- and a 7172-token prompt is bitwise (last-row logits, every KV row, 8 greedy decode steps and their KV)
the same prompt split at k = 1, 7, 255, 256, 1000, plen - 1 (and 4095 / 4096 / 4097 for the long one), prefilled in 256-row busy
chunks, and resumed from a held prefix at 1, 1000 and plen - 1. With the ordinary prefill 10 of those 24 cases differ.

**What "reuse any" means now** (the CUDA default, `Model::reuse_len`, `crates/intarsia-core/src/model.rs:695`): a dense request
resumes at **any row a row-invariant prefill pass wrote** (`SequenceState::inv_len`), at any length, not only at its cold chunk
cuts. It does **not** reuse rows a decode step wrote: decode runs the M <= 4 GEMV and the split-KV decode attention, whose bits differ
from the prefill kernels' (findings 1 and 3 below), so a cold prefill of a previous reply gives other bits. The literal old rule
(`INTARSIA_PREFIX_REUSE=any`, reuse up to the common prefix) is the mutant the multi-turn gate catches.

- **Multi-turn chat:** turn 2 resumes from turn 1's whole prompt and re-prefills only turn 1's reply plus the new message (BI8: on
  CUDA a < 4096-token history was never reused, so the whole history re-prefilled every turn).
- **Repeated prompts** (the BI8 served harness): resume at plen - 1, as before BI8, now exact.
- **Busy chunks:** a long prompt admitted while others decode prefills in 256-row chunks, now bitwise its cold prefill (the second
  gap BI8 listed).

SPEED_SUMMARY

## 1. The row-count-dependent ops (Qwen3-8B, CUDA, `origin/main` `f2bb3410f`)

Method: `crates/intarsia-backend-cuda/tests/row_invariance.rs` runs each prefill kernel once over a 4096-row pass (attention: 4352
rows over the same KV cache) and again over windows of it: (first row, rows) for 25-30 windows (1 row anywhere, 2..17 rows, 255 /
256 / 1000 rows, the rest of the pass), each window's rows sliced from the same inputs and, for attention, `pos0` = its first row.
A kernel is row-invariant when every window's rows equal the one-pass rows byte for byte. The same comparison at model level is
`real_qwen3_8b_prefill_is_row_invariant` with `INTARSIA_RIP_PLAIN=1` (the ordinary prefill). Raw: `raw/kernel.log`, `raw/plain.log`.

| # | Op / kernel | Chosen by | Row-invariant? | Evidence |
|---|---|---|---|---|
| 1 | `matmul_bf16_gemv` (warp-per-row GEMV), M <= 4 | `crates/intarsia-backend-cuda/src/kernels.rs:381` (`when m <= 4`, priority 10); `kernels/cuda/matmul.cu:229` | per row yes, but **other bits than the cuBLAS rows** of a longer pass | 10-11 of 11 M <= 4 windows differ from the cuBLAS one-pass rows, all four 8B projections |
| 2 | `matmul_bf16` (cuBLAS `GemmEx`, default heuristic), M > 4 | `kernels.rs:378`; `kernels/cuda/matmul.cu:289` (algorithm, tile and split-K picked per (M, N, K)) | **no** | N=4096/12288 projections: M = 1 and M = 17 windows differ; k/v projection N=1024: 8 of the 11 windows of 2..17 rows differ (M = 2, 3, 4, 5, 7, 17; M = 9 and 16 matched), plus M = 1. M = 255 / 256 / 1000 / 4095 matched here, by the heuristic's choice, not by contract |
| 3 | `attn_decode_bf16_splitkv` for a 1-row pass (a resumed prompt's last token), and the decode CUDA graph | `crates/intarsia-core/src/model.rs:3596` (`t == 1` -> `Op::AttnDecode`); `crates/intarsia-core/src/model/graph_step.rs:192`; split count from KV length, `kernels/cuda/attention_splitkv.cu:39` (`splitkv_count`), `:548` | **no** (other bits than the flash rows) | 8 of 9 one-row windows differ from the flash one-pass row (the one that matches is row 0) |
| 4 | `attn_prefill_bf16_splitkv`, 2..16 rows | `kernels.rs:431` (`m <= SPLITKV_MAX_ROWS`); split count `attention_splitkv.cu:39` from `pos0 + T` and the row count, `:252`, `:480` | **no** | all 10 windows of 2..16 rows differ from the flash one-pass rows |
| 5 | `attn_prefill_bf16_flash`, > 16 rows | `kernels.rs:423`; key tiles from absolute key 0, `kernels/cuda/attention_flash.cu:124`, `:140` | **yes** | every window identical at 32/8 x 128 and 8/8 x 128, any `pos0` |
| 6 | norms, qk-norm, rope, silu-glu, residual add, embedding, kv-append, the 1-row head GEMV, argmax | one block per row (size from the row length) or per element | **yes** | `row_wise_ops_do_not_depend_on_m`, `head_gemv_rows_do_not_depend_on_m`: every window identical |

So BI8's "every two-pass split differs" was, more exactly, every split that leaves a pass of 1..17 rows (and the k/v GEMM's
small-M heuristic choices). At model level with the ordinary prefill (`raw/plain.log`): splits at k = 255, 256, 1000, 4095, 4096
and reuse from 1 and 1000 were already equal; k = 1, 7, plen - 1, 4097, reuse from plen - 1, and the 7172-token busy-chunk
schedule (its last chunk is 4 rows: GEMV + split-KV) differ, at the first logits row and every KV row.

Not involved: norms do not reduce over a block whose size depends on M, and the flash prefill already processes KV in absolute
64-key tiles with a fixed online-softmax merge order per query row.

## 2. What changed

Selection-level, no new attention kernel; one new GEMM launcher.

- **`matmul_bf16_inv`** (`kernels/cuda/matmul_lt.cu`): cuBLASLt with **one algorithm per (N, K)**, chosen once from the heuristic at
  M = 4096 (the dense CUDA prefill chunk), no workspace, split-K forced to 1 / no reduction scheme, and checked to support M = 1;
  then used for every M. One algorithm with no split-K means one CTA's K loop in one fixed order per output element whatever M is.
  On sm_121 the heuristic picks algo 67 (tile 416 / 409) for all four projections (`INTARSIA_LT_DEBUG=1`). Registered at priority -5,
  so ordinary calls keep `matmul_bf16` / `matmul_bf16_gemv`.
- **`KernelImpl::row_invariant`** (`crates/intarsia-core/src/registry.rs`) marks kernels that promise row invariance at every row
  count their predicate accepts; **`OpCall::prefill_inv`** (`crates/intarsia-core/src/ops.rs:394`) makes `Registry::select` take
  only those (`registry.rs:368`) and fail otherwise. CUDA marks: `matmul_bf16_inv`, the two flash prefills (which now also accept
  <= 16 rows when `ShapeInfo::prefill_inv`), the row-wise ops, kv-append, the M <= 4 head GEMV (a prefill head is its last row
  alone), argmax (`crates/intarsia-backend-cuda/src/kernels.rs`).
- **`Model::forward_prefill`** (`crates/intarsia-core/src/model.rs:2178`): `forward` for prompt passes. With row-invariant prefill
  on, every call of the pass carries `prefill_inv`; a 1-row pass runs `AttnPrefill` (flash), not `AttnDecode`
  (`model.rs:3685`), and skips the decode graph (`graph_step.rs:193`). Every prompt pass of both engines goes through it
  (`crates/intarsia-core/src/engine.rs:1560`, `:1580`, `:1589`, `:859`; `crates/intarsia-core/src/engine/batch.rs:134-155`, `:297`).
  `HeadMode::AllLogits` (accuracy) keeps the ordinary path. Decode, batched decode and spec verify are unchanged.
- **`SequenceState::inv_len`** (`crates/intarsia-core/src/state.rs:186`): rows `0..inv_len` were all written by row-invariant
  prefill passes. Advanced by `forward_prefill` when its pass extends it; lowered by `truncate` and spec rollback; zeroed by
  `load_prefix`; never advanced by decode. `Model::reuse_len` (`model.rs:695`) resumes at `want.min(inv_len)`.
- **When it is on** (`model.rs:1458`, `crates/intarsia-core/src/plan.rs:266`): the backend asks for it
  (`Backend::row_invariant_prefill`, default false; CUDA true, `crates/intarsia-backend-cuda/src/device.rs:207`;
  `INTARSIA_ROW_INVARIANT_PREFILL=0/1` overrides), the model is attention-only on one rank, and a plan probe (1-row and
  chunk-row prefill dry runs) finds a `row_invariant` kernel for every op. Otherwise it switches off with a one-line notice naming
  the ops, and reuse keeps the backend's rule (BI8's exact ladder on CUDA).

PLAN_SECTION

## 3. Gates

GATES_SECTION

## 4. Speed

SPEED_SECTION

## 5. The 35B hybrid and Gemma-4

HYBRID_SECTION

## 6. Not done / follow-ups

- Reusing decode-written rows exactly would need decode to produce prefill bits (the pinned Lt GEMM and flash attention at M = 1
  in decode): a decode-speed cost on a memory-bound path, not measured here.
- Row-invariant kernels for the quantised GEMMs (fp8, NVFP4, MXFP4) and the MoE path would extend this to the 35B's attention
  layers and Gemma-4; the 35B's recurrent state still reuses only snapshots.
