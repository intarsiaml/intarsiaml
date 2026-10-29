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
(`INTARSIA_PREFIX_REUSE=any`, reuse up to the common prefix) stays available as an override. It is not bitwise:
`real_qwen3_8b_decode_written_rows_are_not_prefill_rows`, section 3.

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
- **The plan covers both prefills** (`crates/intarsia-core/src/plan.rs`): with row-invariant prefill on, each multi-row prefill
  probe is recorded twice, once row-invariant and once ordinary. Kernels and scratch for prompt passes that skip `forward_prefill`
  stay reserved. A 1-row prefill probe is added, since a resumed prompt's last token now runs `AttnPrefill`.
- **When it is on** (`model.rs:1458`, `crates/intarsia-core/src/plan.rs:266`): the backend asks for it
  (`Backend::row_invariant_prefill`, default false; CUDA true, `crates/intarsia-backend-cuda/src/device.rs:207`;
  `INTARSIA_ROW_INVARIANT_PREFILL=0/1` overrides), the model is attention-only on one rank, and a plan probe (1-row and
  chunk-row prefill dry runs) finds a `row_invariant` kernel for every op. Otherwise it switches off with a one-line notice naming
  the ops, and reuse keeps the backend's rule (BI8's exact ladder on CUDA).

## 3. Gates

All on spark2, the branch at `0f94ac8` (before the `origin/main` merge and the plan fix in section 2; see the re-check at the
end of this section). Raw logs in `raw/`.

| Gate | Result |
|---|---|
| Kernel level (`tests/row_invariance.rs`): `matmul_bf16_inv` on the four 8B projections, the flash prefill at 32/8 and 8/8 x 128, every row-wise op, the 1-row head GEMV, 25-30 windows each | **pass**, every window byte-identical (4 / 4 tests) |
| Model level, new (`real_qwen3_8b_prefill_is_row_invariant`): 2164- and 7172-token prompts; one pass vs split at k = 1, 7, 255, 256, 1000, plen - 1 (+ 4095, 4096, 4097), vs 256-row busy chunks, vs resume from a held prefix at 1, 1000, plen - 1; last-row logits, every KV row, 8 greedy steps and their KV | **pass**, 24 / 24 cases bitwise equal. Ordinary prefill (`INTARSIA_RIP_PLAIN=1`): 10 / 24 differ |
| BI8's probe (`real_qwen3_8b_prefix_reuse_probe`, the 16 equal.py prompts): every split, self reuse at plen - 1, cross-prompt reuse | **pass**: self reuse 16 / 16 EQUAL, 0 splits differ, 0 cross reuses differ (BI8's base: every split `x0`) |
| Multi-turn chat (`real_qwen3_8b_reuse_and_busy_chunks_equal_the_cold_reference`): three turns of ~2.8k tokens, 48 greedy tokens each, single-sequence and `--max-batch 16` engines; then a ~1.7k-token prompt admitted while 4 streams decode | **pass**: turns 2 and 3 resume from the whole previous prompt (cached 2802, 2872) and equal turn 2 / 3 cold on both engines; the busy-chunked prompt equals its cold run |
| BI8's 8B gates: batched decode == solo step (N 2..16), batched engine == cold reference | **pass** |
| Served greedy vs the cold reference (BI8 harness, the probe's cold ids), reuse at its new default, 64 tokens | **pass**: solo 1 / 4 / 16 streams x 3 reps **9 / 9**, `--max-batch 16` 1 / 4 / 8 / 16 x 3 reps **12 / 12** identical (reps 2+ resume every prompt at plen - 1) |
| equal.py 21 cells (7 sampling settings x 1 / 4 / 16 streams), solo vs `--max-batch 16` | **pass**, 21 / 21, 9408 ids |
| `rows_exact`, `row_invariance`, CUDA unit tests | **pass** (9 + 2 ignored, as on main; 4) |
| `batch_cuda_real` 35B (`real_35b_batched*`) | **pass**: 4290 (row, step) pairs bitwise, batched engine == local engine (2 opt-in tests ignored, as on main) |
| 35B 21-cell served identity, base (`origin/main` `f2bb3410f`) vs fix, `--max-batch 16` | **pass**, 21 / 21, 9408 ids |
| `plan show` (txt and json) base vs fix | 35B and Gemma-4 **byte-identical**; 8B changes, see below |
| Mac `cargo test --workspace --release` (after the merge) | **pass**, 1094 passed, 0 failed |

**`plan show` on the 8B** changes by design. With `0f94ac8` it lost `matmul_bf16` and `attn_prefill_bf16_splitkv` and its scratch
fell from 748800 to 532480 B: the plan's prefill probes recorded only the row-invariant pass. That diff exposed a gap. A prompt pass
that does not go through `forward_prefill` (accuracy's all-logits passes, direct `Model::forward` callers) still takes split-KV for
<= 16 rows, and deep in the context it would have needed more scratch than the plan reserved: an error, not wrong bits. Fix
(`crates/intarsia-core/src/plan.rs`): with row-invariant prefill on, every prefill probe of more than one row also runs as the
ordinary prefill. The 8B plan then differs from base only by the added `matmul_bf16_inv` row. Pinned by
`real_qwen3_8b_ordinary_short_pass_fits_the_plan` (16 / 9 / 5-row ordinary passes at row 8176). PLAN_RECHECK

**The token-level mutant was not a valid test, so it was replaced.** `INTARSIA_PREFIX_REUSE=any` (resume inside the previous reply's
decode-written rows; turn 2 cached 2849 instead of 2802) still gave the same 48 greedy tokens per turn. Token equality cannot tell
whether the bits match. The bitwise version is `real_qwen3_8b_decode_written_rows_are_not_prefill_rows`: hold a ~600-token prompt
plus 16 decode steps, cold-prefill the same tokens, then compare. The prompt rows must be equal, and the decode-written rows or the
next logits must differ, else the `inv_len` limit is needless. Resuming at `inv_len` must equal the cold prefill. DECODE_ROWS


## 4. Speed

SPEED_SECTION

## 5. The 35B hybrid and Gemma-4

Row-invariant prefill is **off** for both, by the load-time rule (`model.rs:1458`, `plan.rs:266`). Their plans are
byte-identical to base, and the 35B's served identity and batched gates pass unchanged.

- **Qwen3.6-35B-A3B (GDN + attention, NVFP4 MoE):** a recurrent state disqualifies it before any probe. Its reuse stays
  snapshot-based (`restorable_len`), and its cold schedule runs the last prompt token alone, so a repeated prompt resumes at a point of
  its own schedule (BI8). **Its busy-chunk prefill is still not bitwise its solo prefill.** The existing check
  `real_35b_busy_chunk_prefill_kl_gate` is a KL gate, not a bitwise one. Making it bitwise needs row-invariant NVFP4 / MoE GEMMs and a
  row-invariant GDN chunked scan, which is out of this lane's scope.
- **Gemma-4 (the only checkpoint on spark2 is `Gemma-4-26B-A4B-NVFP4`, MoE):** the plan probe lists the ops without a row-invariant
  kernel. All are ops the 8B does not use:
  - `add_scaled_bf16` and `embedding_scaled_bf16` (`kernels.rs:490`)
  - `gelu_glu_bf16` (`:481`)
  - `rope_prop_bf16`, partial rotary (`:483`)
  - the head_dim 256 / 512 attention `attn_prefill_bf16_ext` / `attn_decode_bf16_ext` (`:494`)
  - the NVFP4 MoE experts and top-k

  The row-wise ones are very likely invariant (one row per block) and only need the tag plus a kernel test. The `_ext` attention
  needs the same window test as the flash kernel. Gemma's attention also has its own 1-row path in `model/gemma.rs`, which
  `forward_prefill`'s `AttnPrefill`-for-one-row rule does not reach yet. No dense Gemma-4 checkpoint is on spark2.


## 6. Not done / follow-ups

- Reusing decode-written rows exactly would need decode to produce prefill bits (the pinned Lt GEMM and flash attention at M = 1
  in decode): a decode-speed cost on a memory-bound path, not measured here.
- Row-invariant kernels for the quantised GEMMs (fp8, NVFP4, MXFP4) and the MoE path would extend this to the 35B's attention
  layers and Gemma-4; the 35B's recurrent state still reuses only snapshots.
