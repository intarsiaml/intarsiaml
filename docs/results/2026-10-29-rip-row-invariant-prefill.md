# RIP: row-invariant prefill on CUDA (dense Qwen3-8B), opt-in: any split, busy chunk or prefill-written prefix is bitwise the cold prefill

Lane `rip`, branch `lane-rip/row-invariant-prefill`, base `origin/main` `f2bb3410f` (#231, BI8), merged with `origin/main` `f9fabeb13`
(#232). spark2 (spark-fa00, GB10 sm_121), `spark2-vllm` (retro, 38 GB) up and idle throughout. Follows BI8
(`2026-10-29-bi8-serve-batch-invariance.md`), which made prefix reuse exact by reusing only where the cold schedule cuts. Raw:
`2026-10-29-rip-row-invariant-prefill/raw/`.

## Result

**On CUDA, a dense prefill pass's bits depended on its row count only through small-M kernel dispatch. With one pinned GEMM
algorithm for every M and the flash attention at every row count, each row's bits no longer depend on the pass.** On Qwen3-8B the
one-pass prefill of a 2164- and a 7172-token prompt is bitwise equal (last-row logits, every KV row, 8 greedy decode steps and their
KV) to the same prompt:
- split at k = 1, 7, 255, 256, 1000 and plen - 1 (and 4095 / 4096 / 4097 for the long one)
- prefilled in 256-row busy chunks
- resumed from a held prefix at 1, 1000 and plen - 1

With the ordinary prefill, 10 of those 23 cases differ. Served greedy equals the cold reference at 1 / 4 / 8 / 16 streams x 3 reps,
and multi-turn turn 2 / 3 resume from the whole previous prompt and equal their cold runs.

**It costs more than 2% prefill, so it ships as an opt-in knob, default off: `INTARSIA_ROW_INVARIANT_PREFILL=1`** (or
`LoadOptions::row_invariant_prefill`). The pinned cuBLASLt GEMM is 17-24% slower than cuBLAS's per-M choice at M ~ 1000. 8B prefill
is **-10.4% at 1k tokens**, +1.4% at 4k, +0.2% at 16k (3 rounds). The 16-stream cell's aggregate is -1.8%, from its 16 1k prefills;
the decode kernels are unchanged (1-stream decode -0.1%). The table is in section 4. No single heuristic-chosen algorithm is at or under cuBLAS
at both M = 991 and M = 4096 (ref-M sweep, section 4). **The default is the user's decision.** With the knob off, nothing changes:
the 8B, 35B and Gemma-4 `plan show` are byte-identical to base, and CUDA keeps BI8's exact reuse ladder.

**What reuse means with the knob on** (`Model::reuse_len`, `crates/intarsia-core/src/model.rs:699`): a dense request resumes
at **any row a row-invariant prefill pass wrote** (`SequenceState::inv_len`), at any length, not only at its cold chunk cuts. That is
"any" for prompt rows. It is **not** the literal "any" of before BI8: rows a decode step wrote are not reused. Decode runs the M <= 4
GEMV and the split-KV decode attention (findings 1 and 3 below), and measured on the 8B, 16 decode-written rows differ bitwise from
the cold prefill of the same tokens (`real_qwen3_8b_decode_written_rows_are_not_prefill_rows`). `INTARSIA_PREFIX_REUSE=any` still
forces the literal rule and is not bitwise.

- **Multi-turn chat:** turn 2 resumes from turn 1's whole prompt and re-prefills only turn 1's reply plus the new message (with the
  knob off, as under BI8, a < 4096-token history is never reused on CUDA).
- **Repeated prompts** (the BI8 served harness): resume at plen - 1, exact.
- **Busy chunks:** a long prompt admitted while others decode prefills in 256-row chunks, bitwise its cold prefill (the second gap
  BI8 listed).

## 1. The row-count-dependent ops (Qwen3-8B, CUDA, `origin/main` `f2bb3410f`)

Method: `crates/intarsia-backend-cuda/tests/row_invariance.rs` runs each prefill kernel once over a 4096-row pass (attention: 4352
rows over the same KV cache) and again over windows of it: (first row, rows) for 25-30 windows (1 row anywhere, 2..17 rows, 255 /
256 / 1000 rows, the rest of the pass), each window's rows sliced from the same inputs and, for attention, `pos0` = its first row.
A kernel is row-invariant when every window's rows equal the one-pass rows byte for byte. The same comparison at model level is
`real_qwen3_8b_prefill_is_row_invariant` with `INTARSIA_RIP_PLAIN=1` (the ordinary prefill). Raw: `raw/first/kernel.log`, `raw/first/plain.log` (re-run on the final tree: `raw/final/kernel.log`).

| # | Op / kernel | Chosen by | Row-invariant? | Evidence |
|---|---|---|---|---|
| 1 | `matmul_bf16_gemv` (warp-per-row GEMV), M <= 4 | `crates/intarsia-backend-cuda/src/kernels.rs:381` (`when m <= 4`, priority 10); `kernels/cuda/matmul.cu:229` | per row yes, but **other bits than the cuBLAS rows** of a longer pass | 10-11 of 11 M <= 4 windows differ from the cuBLAS one-pass rows, all four 8B projections |
| 2 | `matmul_bf16` (cuBLAS `GemmEx`, default heuristic), M > 4 | `kernels.rs:378`; `kernels/cuda/matmul.cu:289` (algorithm, tile and split-K picked per (M, N, K)) | **no** | N=4096/12288 projections: M = 1 and M = 17 windows differ; k/v projection N=1024: 8 of the 11 windows of 2..17 rows differ (M = 2, 3, 4, 5, 7, 17; M = 9 and 16 matched), plus M = 1. M = 255 / 256 / 1000 / 4095 matched here, by the heuristic's choice, not by contract |
| 3 | `attn_decode_bf16_splitkv` for a 1-row pass (a resumed prompt's last token), and the decode CUDA graph | `crates/intarsia-core/src/model.rs:3596` (`t == 1` -> `Op::AttnDecode`); `crates/intarsia-core/src/model/graph_step.rs:192`; split count from KV length, `kernels/cuda/attention_splitkv.cu:39` (`splitkv_count`), `:548` | **no** (other bits than the flash rows) | 8 of 9 one-row windows differ from the flash one-pass row (the one that matches is row 0) |
| 4 | `attn_prefill_bf16_splitkv`, 2..16 rows | `kernels.rs:431` (`m <= SPLITKV_MAX_ROWS`); split count `attention_splitkv.cu:39` from `pos0 + T` and the row count, `:252`, `:480` | **no** | all 10 windows of 2..16 rows differ from the flash one-pass rows |
| 5 | `attn_prefill_bf16_flash`, > 16 rows | `kernels.rs:423`; key tiles from absolute key 0, `kernels/cuda/attention_flash.cu:124`, `:140` | **yes** | every window identical at 32/8 x 128 and 8/8 x 128, any `pos0` |
| 6 | norms, qk-norm, rope, silu-glu, residual add, embedding, kv-append, the 1-row head GEMV, argmax | one block per row (size from the row length) or per element | **yes** | `row_wise_ops_do_not_depend_on_m`, `head_gemv_rows_do_not_depend_on_m`: every window identical |

So BI8's "every two-pass split differs" was, more exactly, every split that leaves a pass of 1..17 rows (and the k/v GEMM's
small-M heuristic choices). At model level with the ordinary prefill (`raw/first/plain.log`): splits at k = 255, 256, 1000, 4095, 4096
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
- **`Model::forward_prefill`** (`crates/intarsia-core/src/model.rs:2187`): `forward` for prompt passes. With row-invariant prefill
  on, every call of the pass carries `prefill_inv`; a 1-row pass runs `AttnPrefill` (flash), not `AttnDecode`
  (`model.rs:3694`), and skips the decode graph (`graph_step.rs:193`). Every prompt pass of both engines goes through it
  (`crates/intarsia-core/src/engine.rs:1560`, `:1580`, `:1589`, `:859`; `crates/intarsia-core/src/engine/batch.rs:134-155`, `:297`).
  `HeadMode::AllLogits` (accuracy) keeps the ordinary path. Decode, batched decode and spec verify are unchanged.
- **`SequenceState::inv_len`** (`crates/intarsia-core/src/state.rs:186`): rows `0..inv_len` were all written by row-invariant
  prefill passes. Advanced by `forward_prefill` when its pass extends it; lowered by `truncate` and spec rollback; zeroed by
  `load_prefix`; never advanced by decode. `Model::reuse_len` (`model.rs:699`) resumes at `want.min(inv_len)`.
- **The plan covers both prefills** (`crates/intarsia-core/src/plan.rs`): with row-invariant prefill on, each multi-row prefill
  probe is recorded twice, once row-invariant and once ordinary. Kernels and scratch for prompt passes that skip `forward_prefill`
  stay reserved. A 1-row prefill probe is added, since a resumed prompt's last token now runs `AttnPrefill`.
- **When it is on** (`model.rs:1467`, `crates/intarsia-core/src/plan.rs:266`): the backend asks for it
  (`Backend::row_invariant_prefill`, false everywhere, CUDA included, `crates/intarsia-backend-cuda/src/device.rs:209`;
  `INTARSIA_ROW_INVARIANT_PREFILL=1` or `LoadOptions::row_invariant_prefill` turns it on), the model is attention-only on one rank, and a plan probe (1-row and
  chunk-row prefill dry runs) finds a `row_invariant` kernel for every op. Otherwise it switches off with a one-line notice naming
  the ops, and reuse keeps the backend's rule (BI8's exact ladder on CUDA).

## 3. Gates

All on spark2. The rows marked * ran on `0f94ac8` (row-invariant prefill on by default, before the `origin/main` merge and the plan
fix); the rest ran on the final tree `f23b334`, where the knob is off by default and the GPU tests opt in through
`LoadOptions::row_invariant_prefill`. The prefill code path with the knob on is the same in both. Raw logs: `raw/run/` (the combined gates + speed job, `scripts/rip/all.sh`), `raw/final/` (final-tree re-checks, `scripts/rip/tests.sh`).

| Gate | Result |
|---|---|
| Kernel level (`tests/row_invariance.rs`): `matmul_bf16_inv` on the four 8B projections, the flash prefill at 32/8 and 8/8 x 128, every row-wise op, the 1-row head GEMV, 25-30 windows each | **pass**, every window byte-identical (4 / 4 tests), on `0f94ac8`* and again on `f23b334` |
| Model level, new (`real_qwen3_8b_prefill_is_row_invariant`): 2164- and 7172-token prompts; one pass vs split at k = 1, 7, 255, 256, 1000, plen - 1 (+ 4095, 4096, 4097), vs 256-row busy chunks, vs resume from a held prefix at 1, 1000, plen - 1; last-row logits, every KV row, 8 greedy steps and their KV | **pass**, 23 / 23 cases bitwise equal, on `0f94ac8`* and again on `f23b334`. Ordinary prefill (`INTARSIA_RIP_PLAIN=1`): 10 / 23 differ |
| BI8's probe (`real_qwen3_8b_prefix_reuse_probe`, the 16 equal.py prompts): every split, self reuse at plen - 1, cross-prompt reuse * | **pass**: self reuse 16 / 16 EQUAL, 0 splits differ, 0 cross reuses differ (BI8's base: every split `x0`) |
| Multi-turn chat (`real_qwen3_8b_reuse_and_busy_chunks_equal_the_cold_reference`): three turns of ~2.8k tokens, 48 greedy tokens each, single-sequence and `--max-batch 16` engines; then a ~1.7k-token prompt admitted while 4 streams decode * | **pass**: turns 2 and 3 resume from the whole previous prompt (cached 2802, 2872) and equal turn 2 / 3 cold on both engines; the busy-chunked prompt equals its cold run |
| BI8's 8B gates: batched decode == solo step (N 2..16), batched engine == cold reference * | **pass** |
| Served greedy vs the cold reference (BI8 harness, the probe's cold ids), reuse at its new default, 64 tokens * | **pass**: solo 1 / 4 / 16 streams x 3 reps **9 / 9**, `--max-batch 16` 1 / 4 / 8 / 16 x 3 reps **12 / 12** identical (reps 2+ resume every prompt at plen - 1) |
| equal.py 21 cells (7 sampling settings x 1 / 4 / 16 streams), solo vs `--max-batch 16` * | **pass**, 21 / 21, 9408 ids |
| `rows_exact`, `row_invariance`, CUDA unit tests * | **pass** (9 + 2 ignored, as on main; 4) |
| `batch_cuda_real` 35B (`real_35b_batched*`) * | **pass**: 4290 (row, step) pairs bitwise, batched engine == local engine (2 opt-in tests ignored, as on main) |
| 35B 21-cell served identity, base (`origin/main` `f2bb3410f`) vs fix, `--max-batch 16` * | **pass**, 21 / 21, 9408 ids |
| `plan show` (txt and json) base vs fix | 35B and Gemma-4 **byte-identical** *; 8B **byte-identical** at the default (`f23b334`), and with the knob on its only diff is the added `matmul_bf16_inv` row (see below) |
| Decode-written rows vs prefill rows (`real_qwen3_8b_decode_written_rows_are_not_prefill_rows`), ordinary short passes at row 8176 (`real_qwen3_8b_ordinary_short_pass_fits_the_plan`) | **pass** on `f23b334` (below) |
| Mac `cargo test --workspace --release` (final tree) | **pass**, 1096 passed, 0 failed |

**`plan show` on the 8B** changes by design. With `0f94ac8` it lost `matmul_bf16` and `attn_prefill_bf16_splitkv` and its scratch
fell from 748800 to 532480 B: the plan's prefill probes recorded only the row-invariant pass. That diff exposed a gap. A prompt pass
that does not go through `forward_prefill` (accuracy's all-logits passes, direct `Model::forward` callers) still takes split-KV for
<= 16 rows, and deep in the context it would have needed more scratch than the plan reserved: an error, not wrong bits. Fix
(`crates/intarsia-core/src/plan.rs`): with row-invariant prefill on, every prefill probe of more than one row also runs as the
ordinary prefill. The 8B plan then differs from base only by the added `matmul_bf16_inv` row. Pinned by
`real_qwen3_8b_ordinary_short_pass_fits_the_plan` (16 / 9 / 5-row ordinary passes at row 8176). Re-checked on the final tree (`f23b334`, default off): the 8B `plan show` is **byte-identical to base**, and with
`INTARSIA_ROW_INVARIANT_PREFILL=1` its only diff is the added `matmul_bf16_inv` row; the ordinary-pass test passes.

**The token-level mutant was not a valid test, so it was replaced.** `INTARSIA_PREFIX_REUSE=any` (resume inside the previous reply's
decode-written rows; turn 2 cached 2849 instead of 2802) still gave the same 48 greedy tokens per turn. Token equality cannot tell
whether the bits match. The bitwise version is `real_qwen3_8b_decode_written_rows_are_not_prefill_rows`: hold a ~600-token prompt
plus 16 decode steps, cold-prefill the same tokens, then compare. The prompt rows must be equal, and the decode-written rows or the
next logits must differ, else the `inv_len` limit is needless. Resuming at `inv_len` must equal the cold prefill.

Measured (final tree): 555 prompt rows + 16 decode rows. The **prompt rows are EQUAL**, and **the decode-written KV rows and the
next-token logits DIFFER** from the cold prefill of the same 571 tokens. Resuming at `inv_len` (555) and prefilling the 16 reply
tokens gives **EQUAL** logits and KV. So reusing decode-written rows would not be exact, and the `inv_len` limit is required.


## 4. Speed

Speed was measured on `0f94ac8` (default on, ref M 4096, before the `origin/main` merge). The prefill kernels are the same in the final
tree with the knob on. spark2, `--quiet gpu,cpu`, `spark2-vllm` idle, localeval, 3 reps + 1 warm-up per cell, a fresh server per
cell, every prompt led by a nonce.
- **8B arms:** A = `INTARSIA_ROW_INVARIANT_PREFILL=0` (the ordinary prefill), B = row-invariant, same binary, interleaved A B per
  round, 3 rounds.
- **Cells:** prefill on a solo server (`--sizes 1024,4096,16384 --gen 16`); the 16-stream decode cell on `--max-batch 16`
  (`--sizes 1024 --gen 256 --concurrency 16`, as SB8 / BI8).
- **35B:** row-invariant prefill is off for it, so the env changes nothing. Its arms are A = `origin/main` (`f2bb3410f`), B = the fix,
  1 round.

`cached max` is localeval's largest cached prompt count. B's 4-6 are the chat template's first tokens before the nonce, against the
991-16652-token prompts. Raw: `raw/run/speed/` (localeval output), `raw/speed-results/` (its result.json files), `raw/speed-table.md` (`scripts/rip/agg.py`).

| model | metric | size / streams | A per round | B per round | A median | B median | B vs A | cached max |
|---|---|---:|---|---|---:|---:|---:|---:|
| 8b | decode agg tok/s | 16 | 160 / 160 / 160 | 158 / 157 / 157 | 160.3 | 157.4 | -1.8% |  |
| 8b | decode tok/s 1 stream | 1024 | 15.3 / 15.2 / 15.2 | 15.3 / 15.2 / 15.2 | 15.2 | 15.2 | -0.1% |  |
| 8b | prefill tok/s | 1024 | 4881 / 4209 / 4699 | 4213 / 4280 / 4211 | 4699.2 | 4212.6 | -10.4% | 6 |
| 8b | prefill tok/s | 16384 | 3753 / 3752 / 3738 | 3802 / 3750 / 3761 | 3752.1 | 3760.7 | +0.2% | 5 |
| 8b | prefill tok/s | 4096 | 4681 / 4636 / 4685 | 4824 / 4748 / 4424 | 4681.2 | 4748.2 | +1.4% | 6 |
| 35b | decode agg tok/s | 16 | 274 | 272 | 274.0 | 271.9 | -0.8% |  |
| 35b | decode tok/s 1 stream | 1024 | 85.4 | 85.0 | 85.4 | 85.0 | -0.5% |  |
| 35b | prefill tok/s | 1024 | 5091 | 5278 | 5091.0 | 5277.9 | +3.7% | 0 |
| 35b | prefill tok/s | 16384 | 6637 | 6628 | 6636.9 | 6628.0 | -0.1% | 0 |
| 35b | prefill tok/s | 4096 | 6822 | 6933 | 6822.4 | 6932.9 | +1.6% | 0 |

- **1k prefill is a real cost**, and the only one over 2%. B's TTFT at 991 tokens is a tight 0.230 s in every round (spread 0.7-21%).
  A's is 0.203-0.234 s: A is bimodal across server starts, B is not.
- **16-stream decode cell:** per-stream decode is -1.0% (11.17 -> 11.08 tok/s) and TTFT +11% (2.22 -> 2.46 s). The cell's 16 1k
  prefills interleave with decode, so the cost is the 1k prefill again. 1-stream decode (`forward` with decode kernels, unchanged) is
  -0.1%.
- **35B:** within noise, as expected for unchanged kernels. It ran 1 round, not 3: the knob does nothing on the 35B (auto-off), so
  only a base-vs-fix regression check was needed there.

**GEMM ref-M sweep** (`gemm_inv_timing_at_8b_shapes`, `--ignored`, the 8B's 7 projection GEMMs of one layer on random operands, us per
layer, the final tree). `INTARSIA_INV_REF_M` is the row count cuBLASLt's heuristic picks the one algorithm for (`raw/final/timing.log`):

| ref M | algorithm (q/o, k/v, gate/up, down) | M = 991 inv / cuBLAS | 1024 | 2048 | 4096 |
|---:|---|---:|---:|---:|---:|
| 512 | 67 / 67 / 21 / 67 | 1.202 | 1.112 | 1.102 | 1.075 |
| 1024 | 67 / 21 / 21 / 67 | 1.167 | 1.068 | 1.083 | 1.035 |
| 2048 | 67 (tile 416 / 412) | 1.188 | 1.156 | 0.967 | 0.963 |
| 4096 (default) | 67 (tile 416 / 409) | 1.239 | 1.136 | 0.943 | 0.966 |

cuBLAS takes 4.13 ms per layer at M = 991 and 4.50 ms at 1024. Its own choice at 991 is one that no fixed-M heuristic query returns.
No single algorithm is at or under 1.0 at both ends, so the ref M stays 4096: the best at the 4096-row chunks, which dominate long
prompts. Closing the 1k gap needs one of these (not done):
- a fixed split-K for every M: split-K is row-invariant when the split count and reduction order do not vary with M
- a per-architecture table of fixed algorithm configs, which stays deterministic across processes, unlike runtime autotuning
- two algorithms dispatched by M, allowed only if the window test shows them bitwise-equal on every shape


## 5. The 35B hybrid and Gemma-4

Row-invariant prefill is **off** for both, by the load-time rule (`model.rs:1467`, `plan.rs:266`). Their plans are
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
