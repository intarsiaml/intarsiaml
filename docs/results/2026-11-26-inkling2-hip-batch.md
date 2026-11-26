# Inkling-2-Mini batched decode on HIP, one R9700 (lane ink-hip-batch, 2026-11-26)

Branch `ink/hip-batch` from main `87c9745c0`. Goal: `serve --max-batch N` (several streams decoding in one pass) works and is fast
for Inkling-2-Mini on the HIP backend. Today HIP refuses `--max-batch > 1` with `NoKernel`: no HIP kernel registers `.row_table()`
(`model/batch.rs` `row_tables_wanted`: Inkling-2 needs row-table kernels of `ConvMix` and `InkAttnDecode`, or of the pair).
Raw: `docs/results/2026-11-26-inkling2-hip-batch/raw/`.

## Briefs and messages received

- 2026-11-26 ~11:40 (coordinator): go ahead with the RT template parameter on `ink_row_kernel` / `ink_attn_dec_k`, not a copy.
  Two asks: (1) a test that pins the `RT = false` instantiations to main's bits: solo greedy md5 at 64 / 256 tokens, short / 4k,
  equal to main; (2) the `RowArgs` additions go at the end of the struct. Merge origin/main before the PR (pfflip will likely
  land first).
- 2026-11-26 ~11:45 (coordinator): `--mem` on tatooine-r9700 counts against the card's 32 GB, which radiance holds. Build #12035
  could never start and blocked every lane's queue. It was cancelled (`fleetq release 12035`) and resubmitted without `--mem`.
  No `--mem` on any tatooine job.

- 2026-11-26 ~12:00 (coordinator): two overlaps before the gate.
  1. ink-hip-spec also adds "hip" to `inkling2::ink_rows_exact`. Whichever lands second takes the other's line as is, without
     reordering. (This lane does not need that line: section 1.1.)
  2. PR #526 (HIP prefill flip) is merging. MoE WMMA prefill, flash prefill attention and chunk 2048 become the HIP Inkling
     defaults, and `tests/inkling2.rs` `backend()` "hip" pins `inkattn::set_flash(0)` and `inkmoe::set_wmma(0)`. Asks:
     - rebase onto main once it lands;
     - check that the batched rows (N rows at decode) never take the WMMA expert or flash attention path under the new defaults;
     - run the solo greedy md5 check (64 / 256, short / 4k) on the rebased binary with no env vars set;
     - say here which paths the batched forward took.
  Reminder: no `--mem` on tatooine jobs.

## 1. Design (written before any kernel)

### 1.1 What a batched Inkling-2 step needs on HIP

- `forward_batch` sets `exact_rows` for the whole pass (`Model::verifying()` is that flag, and `exec_call_one` copies it into every
  call). So every row-local op of the B-row pass takes the backend's `rows_exact` kernel where the op has one: `InkInProj`
  (`ink_in_proj_bf16_gemv_rows`), `Matmul` wo (`matmul_bf16_gemv_rows`), the router and unembed `Logits` (`logits_bf16_gemv_rows`),
  `MoeExperts` / `MoeExpertsMxfp8` (`_rows`, union-grouped). HIP already has all of them (used by the Qwen batching and verify).
  The ops without one (`AttnResMix`, `LoopBlend`, `MoeRoute`, norms, embedding, argmax) keep the solo kernel, one block per row.
  So `ink_rows_exact` (the core line lane ink-hip-spec owns) is **not needed** here, and this lane leaves it alone.
- The stateful ops run in their row-table form (`inkling2::ops`, "Row tables"): row r is the T = 1 call at device position
  `pos[r]` on sequence r's own rings and caches. In the HIP batched step these are `ConvMix` (twice per visit) and
  `InkAttnDecode` (op 75; on HIP the model takes op 75 at every decode visit, and a batch takes it whenever no row-table pair exists).
  `SconvRes` and the pair (`InkKvPrologue` + `RelSinkAttn`) are not called by the batched step, so they get no row-table kernel.
- New kernels: `conv_mix_bf16_rows` and `ink_attn_decode_bf16_rows`, registered `.row_table()`. With the `_rows` suffix they take
  the admission of the kernel they twin (`Registry::effective_admission`). Their gate is a bitwise test against the solo T = 1
  launches (`crates/intarsia-backend-hip/tests/ink_rows.rs`).

### 1.2 Kernel form: an `RT` template parameter on the solo kernels (Metal's pattern, HIP's `splitkv_kernel<.., ROWS>`)

- `ink_row_kernel<NE, FAST, DPP, RT = false>` (ConvMix). With RT, row t:
  - reads its ring from table entry `tab[t][index] + carrier offset`;
  - takes position `pos[t]`;
  - its conv history is itself (`qp == p` reads x row t) plus older positions from its own ring.
  The ring row is always written in-kernel (T = 1 rule), and `ring_after` never runs. The `[N, T, D]` planes, the mix and the
  blend (ctl GEMV, gate, norm: row-parallel) keep the call's global row index and T = B.
- `ink_attn_dec_k<GP, DPP, RT = false>` (op 75). With RT, block row r:
  - remaps its `InkP` to the T = 1 call of sequence r: row-indexed pointers shifted by r, its own scratch block, caches / rings
    from table entries `index .. index + 3` plus the carriers' offsets, `pos0 = max(pos[r], 0)`;
  - takes its ticket at global row r.
  The rows launcher always takes the fused kernel. The solo launcher's split path (full layers above `INTARSIA_INK_DEC_FUSED_KEYS`)
  gives the same bits as the fused kernel (both are bitwise the pair).
- The `RT = false` instantiations are the existing code. New fields go at the end of `RowArgs` / `InkP` only.
- Carrier offsets are passed from Rust (`TensorRef.offset` of each state tensor). An Inkling-2 ring slot puts both of its planes
  behind one table entry (`upload_row_table`: `(buf, buf)`), so plane 1 is reached only through its offset.

### 1.3 What changed in files other lanes touch

- `kernels/hip/ink_mix.hip`:
  - `RowArgs` gains 4 fields at its end;
  - `ink_row_kernel` gains a 4th template parameter `RT = false`: a 1-line ring rebind at the top, `pos0` / `own0` per row, two conv
    conditions (`qp >= own0`) and the ring-write condition (`RT ||`);
  - `launch_rows` gains an `if (a.rtab)` arm before the existing switch;
  - `ik_conv_mix_bf16` becomes a wrapper of `conv_mix_run`, the same body plus two lines (`rt_args`);
  - new: `rt_args`, `ik_conv_mix_bf16_rows`.
- `kernels/hip/ink_attn.hip`:
  - `InkP` gains 4 fields at its end;
  - `ink_attn_dec_k` gains a 3rd template parameter `RT = false`: the row remap (`ink_rt_row`) and the ticket index;
  - new launcher `ik_ink_attn_decode_bf16_rows` at the end of the file.
- The `RT = false` instantiations compile to the code main had: the remap sits behind `if constexpr` / constant `RT ?` selects, and
  `own0 == pos0` when `RT` is false. Proof by bits: section 3 (solo md5 at 64 / 256, short / 4k, equal to main).
- Rust: two `TABLE` entries and registrations (`.row_table()`), two dispatch arms. HIP tests:
  - new `tests/ink_rows.rs`;
  - `tests/ink_mix.rs` skips row-table names in its every-name admission loop;
  - a registry lib test in `kernels.rs`.
- `crates/intarsia-models`:
  - `tests/inkling2_batch.rs` gains a `hip` arm;
  - `examples/batch_step.rs` runs on HIP too (`--backend hip`, plus `--model`, `--mem-budget`, `--prefill-chunk`, `--batched-only`
    and a p99 column; the Metal output keeps its columns).

## 2. Window 1: bitwise tests (fleetq #12049; radiance down 12:18:04 - 12:20:00, 1:56)

Log: `raw/window1-12049-unit.log`. Stage: `raw/stage_unit.sh`. Binary b2 = `d7c6a90f7` + the ring-write fix, on main `87c9745c0`.

| Check | Result |
|---|---|
| `tests/ink_rows.rs` `conv_mix_rows_bitwise_solo` | **645 row-table calls bitwise the solo T = 1 calls**: every output row, every ring byte. Covers widths 64 / 2304, every mix / blend form and epilogue, B 1/2/3/5/16, positions 0 .. 9001 (before the conv history fills, at the ring wrap), mix masks 3 / 0, DEC3 default / 0 |
| `tests/ink_rows.rs` `attn_decode_rows_bitwise_solo` | **40 row-table calls bitwise the solo device-position calls**: every output row, every K / V cache and conv-ring byte, pads included. Covers SWA ring layers and full layers with a 2600-row cache, positions across the window, the ring and 2048 keys (above 2048 the solo launch takes op 75's split path), B 1/2/3/5/16, DEC3 default / 0 |
| `tests/ink_attn.rs` `rows_exact_real_shapes` (the solo kernel's own gate) | 63 cases bitwise, PASS |
| `tests/ink_mix.rs` (no bench) | 7 PASS. `ink_mix_ops_match_the_reference` FAILED only on the new name: its loop over `inkmix::TABLE` wants an admission summary for `conv_mix_bf16_rows`, which admission never grades (a row-table twin takes its base's admission). Every graded kernel ADMITTED. The test now skips row-table names (commit after `67e040598`); rerun in window 2 |
| `intarsia kernels test --op conv_mix` | `conv_mix_bf16` 532 cases, 0 fail, ADMITTED |
| `intarsia kernels test --op ink_attn_decode` | `ink_attn_decode_bf16` 70 cases, 0 fail, ADMITTED |
| `crates/intarsia-models/tests/inkling2_batch.rs` on HIP (tiny fixture, `CI=1`) | **8 / 8 PASS**: batched == solo at 2 / 5 / 16, argmax, sampler rows softcapped, membership changes, rollback, failed pass resets, P12 plan, batched engine == solo engine |
| host-only (Mac): `-p intarsia-backend-hip --lib` (35, includes the new registry test), `--test rows_registry` (ordinary pins unchanged) | PASS |

## 3. Window 2: batch_step on the real model did not fit (fleetq #12057; radiance down 12:33:34 - 12:35:08, 1:34)

Log: `raw/window2-12057-bstep-oom.log`. `examples/batch_step.rs` holds two state sets (solo and batched) of `max(batches)`
sequences, but `enable_batch(16)` budgets the KV of 16. Results:

- The short run died in `hipMalloc` of 4 259 840 bytes, which is one SWA ring of a visit (1040 x 8 x 128 x 2 x 2).
- At 4k, the KV fraction 0.5 (the `LoadOptions` default) gave a 3285-token context per sequence at 16 rows.
- `plan show` has no `--max-batch`.

The ink_mix every-name admission test passed in this window with the row-table names skipped (8 ADMITTED).

Fix (`528773455`):
- `batch_step --solo-replay`: the batched runs first, then the batched states are dropped and every sequence is replayed solo on a
  fresh state, token by token, with each solo token asserted equal to its batched token. One state set of 16 then fits.
- `--kv-fraction`.
- a `# memory` line: weights, workspace, KV budget, context per sequence, rings per sequence.

## 4. Window 3: solo gate of the first form (fleetq #12064; radiance down 12:44:34 - 12:47:27, 2:53)

Log: `raw/window3-12064-gate-b3.log`. Stage: `raw/stage_gate.sh`. Main = `40b85a391` (origin/main after #526, binary md5
`77088458…`). Branch b3 = `7b3a2b8eb` (the first kernel form merged onto it, md5 `8dd22392…`). Default env, `generate --spec off`,
graphs: 1 capture + 255 replays, 0 eager steps.

| | short 64 | 4k 64 | short 256 | 4k 256 |
|---|---|---|---|---|
| greedy md5, main = branch (every rep) | `04cdb28a…` | `2252566a…` | `9a27c21b…` | `9675bbb9…` |

| decode tok/s (256 tok, ABBA 4 + 4) | main | branch b3 | ratio (95 % CI) |
|---|---|---|---|
| short | 76.92 / 76.92 / 76.88 / 76.86 | 76.66 / 76.62 / 76.59 / 76.61 | **0.9964 ± 0.0005** |
| 4k | 74.09 / 74.04 / 74.03 / 74.02 | 73.81 / 73.79 / 73.75 / 73.75 | **0.9964 ± 0.0003** |

- The bits are main's.
- The first form was **0.36 % slower** in all 8 of 8 pairs. That is small, but it is not noise.
- Its `RT = false` code is main's, except that every launch of the row, blend and attention kernels carried a 24-byte
  (`RowArgs`) or 48-byte (`InkP`) larger argument block. Under graph replay the HIP runtime copies each node's kernel arguments at
  every launch, and dec3 found the decode step bound by dispatch.
- Fix (`68d1321af`): the row-table data lives in derived structs, `RowArgsRT : RowArgs` and `InkPR : InkP`. Only the RT
  instantiations take them (`std::conditional_t<RT, …>` as the kernel parameter), so the solo kernels' arguments are main's, byte
  for byte.
- The coordinator's ask to put "the RowArgs additions at the end of the struct" is moot: `RowArgs` and `InkP` are unchanged
  from main. Re-gated in section 6.

## 5. Window 4: unit gate of the final form, `forward_batch` on the real model (fleetq #12076; radiance down 13:00:30 - 13:03:37, 3:07)

Log: `raw/window4-12076-unit-bstep.log`. Kernel trace: `raw/window4-trace-b4.log`. Stage: `raw/stage_bstep.sh`. Binary b5 = `68d1321af`
(md5 `8ebceca5…`). Default env: main after #526 (prefill chunk 2048, MoE WMMA and flash prefill on). KV fraction 0.9.

### 5.1 Unit gate (b5)

| Check | Result |
|---|---|
| `tests/ink_rows.rs` | conv_mix_bf16_rows **645**, ink_attn_decode_bf16_rows **40** row-table calls bitwise the solo calls: PASS |
| `tests/ink_attn.rs` `rows_exact_real_shapes` | 63 cases bitwise: PASS |
| `kernels test --op conv_mix` / `--op ink_attn_decode` | 532 / 70 cases, 0 fail, ADMITTED |
| `tests/inkling2_batch.rs` on HIP, tiny fixture, `CI=1` | 8 / 8 PASS |

### 5.2 Batched == solo on Inkling-2-Mini (`batch_step --solo-replay`, real-text prompts)

Every batched token was compared with the token that sequence's own solo decode produces from the same inputs:

| Run | Rows | Tokens compared | Result |
|---|---|---|---|
| short: row i at 256 + 8 i tokens, B 1 / 4 / 8 / 16 in turn, 5 rounds x 16 steps | 16 | 2320 | **all equal** |
| 2k: row i at 2040 + 3 i, B 4 / 16, 3 rounds x 8 steps (the solo steps cross 2048 keys: op 75's fused kernel -> its split path) | 16 | 480 | **all equal** |

### 5.3 Pure decode speed (`batch_step`, per-step time = every stream's inter-token latency)

| Context | B | step ms p50 | step ms p99 | aggregate tok/s | x solo |
|---|---|---|---|---|---|
| short (256 - 376) | 1 | 12.99 | 13.08 | 77.0 | 1.00 |
| | 4 | 23.59 | 24.41 | 169.6 | 2.21 |
| | 8 | 33.25 | 34.31 | 240.6 | 3.14 |
| | 16 | 55.35 | 56.64 | **289.1** | **3.77** |
| 2k (2040 - 2101) | 4 | 25.05 | 25.86 | 159.7 | 2.16 |
| | 16 | 61.61 | 62.47 | 259.7 | 3.51 |
| 4k (4096) | 1 | 13.63 | 13.66 | 73.4 | 1.00 |
| | 4 | 25.61 | 26.27 | 156.2 | 2.13 |
| | 8 | 36.77 | 37.75 | 217.6 | 2.96 |
| | 16 | 63.25 | 64.14 | **253.0** | **3.45** |

- B 1 in a batched model is the solo step: 77.0 / 73.4 tok/s. The solo median in the replay was 13.04 ms (76.7 tok/s).
- The step-time p99 is within 2.5 % of the p50 at every B.
- The `x solo` column divides by the same run's B 1.

### 5.4 Which kernels the batched rows take (coordinator ask, PR #526 defaults)

Trace: `INTARSIA_HIP_TRACE=1`, 4 sequences of 255 prompt tokens (one 2048-chunk prefill each), then 32 batched steps at B 4.

| Pass | Kernels (launches) |
|---|---|
| prefill: 4 sequences x 48 visits | `moe_experts_situ_bf16` 176, `moe_experts_mxfp8_bf16` 16, `ink_kv_prologue_bf16` 192, `rel_sink_attn_bf16` 192, `ink_in_proj_bf16_matmul` 192, `conv_mix_bf16` 384, `matmul_bf16` 192, `logits_bf16` 192 |
| batched steps: 32 x 48 visits | `moe_experts_situ_bf16_rows` 1408, `moe_experts_mxfp8_bf16_rows` 128, `ink_in_proj_bf16_gemv_rows` 1536, `conv_mix_bf16_rows` 3072, `ink_attn_decode_bf16_rows` 1536, `matmul_bf16_gemv_rows` 1536, `logits_bf16_gemv_rows` 1568 |

- The batched rows never reach the tensor-core expert path. `kernels/hip/ink_moe.hip` `run()` takes WMMA only when `!rows`, that is
  for ordinary names and never the `_rows` twins, and `forward_batch`'s `exact_rows` picks the `_rows` twins.
- The batched rows never reach the flash attention either. The batched step calls no `RelSinkAttn` (0 launches beyond the 192 of the
  prefill), and the flash form only runs inside `rel_sink_attn_bf16` for calls of more than 16 rows.
- The batched paths are exactly the `_rows` kernels listed above.

### 5.5 Memory (the plan's `# memory` line, R9700 32 624 MiB)

| Item | MiB |
|---|---|
| weights | 12 694 |
| workspace, one sequence (with its rings) | ~973 |
| rings per extra sequence (SWA KV of 40 visits + conv rings, x 2: live + ring snapshot) | 348 |
| full-attention KV (8 full visits x 8 heads x 128 x 2 x bf16) | 32 KiB / token = 132 per sequence at 4224 tokens |
| free after weights + workspace (B 16) | 13 730 |

Per sequence at a 4k context: 348 + 132 = **480 MiB**. `serve`'s KV budget is `--kv-fraction` x free, default 0.5:
- at 0.5 the budget is 6 865 MiB, and B 16 clamps to **3285 tokens** per sequence (measured in window 2);
- at 0.9 the budget is 12 356 MiB, and B 16 at 4224 tokens uses 5 222 + 2 112 = 7 334 MiB of it.

Largest B at 4k on 32 GB:
- memory alone would allow about 26 at 0.9, and about 29 with all the free memory: (13 730 + 348) / 480;
- the engine's row cap (`kernels::ROWS_MAX` = `batch_row_cap` = 16) binds first, so the answer is **B 16 at 4k with `--kv-fraction 0.9`**;
- at the default 0.5 it is **B 15**: (6 865 + 348) / 480.

Peak VRAM per B is measured in section 7.

## 6. Window 5: solo gate of the final form b5 (fleetq #12080; radiance down 13:07:35 - 13:10:29, 2:54)

Log: `raw/window5-12080-gate-b5.log`. Main = `40b85a391` (`77088458…`). Branch b5 = `68d1321af` (`8ebceca5…`). The method is the
same as section 4.

- **Greedy md5 equal to main** at 64 and 256 tokens, short and 4k, on every rep (the same four md5 as section 4).

| decode tok/s (256 tok, ABBA 4 + 4) | main | branch b5 | ratio (95 % CI) |
|---|---|---|---|
| short | 76.94 / 76.88 / 76.85 / 76.86 | 76.59 / 76.56 / 76.54 / 76.51 | **0.9957 ± 0.0004** |
| 4k | 74.06 / 74.06 / 74.02 / 74.05 | 73.73 / 73.75 / 73.70 / 73.67 | **0.9955 ± 0.0007** |
| 4k prefill (ms) | 1563 / 1570 / 1575 / 1576 | 1565 / 1569 / 1577 / 1575 | 0.9997 ± 0.0018 (speed) |

- Moving the row-table data out of the solo kernels' arguments did not remove the gap: b3 was 0.9964 and b5 is 0.9957.
- The gap is in graph-replayed decode only. Prefill, which runs the same row kernel eagerly, is equal.

## 7. Where the 0.43 % went: the RT instantiations changed the solo kernels' code (fleetq #12083, #12086, #12087, cpu)

Logs: `raw/isa-12083-12086-12087.log`. Scripts: `raw/isa.sh`, `raw/isa2.sh`, `raw/isa3.sh`, `raw/isa_cmp.py`. Method: build.rs's flags, device
assembly only (`--cuda-device-only -S`), then per kernel the instruction stream (labels normalised) and the descriptor (VGPR / SGPR
count, LDS, scratch, kernarg size), main against the branch; a branch kernel's trailing `RT = false` template argument is dropped to
pair it with main's.

| Module | Form | main's kernels identical in the branch | Kernels that differ |
|---|---|---|---|
| ink_mix.hip | b5 (RT instantiations in the module) | 31 / 31 | none (24 RT kernels added) |
| ink_attn.hip | b5 (RT instantiations in the module) | 24 / 30 | all six `ink_attn_dec_k<GP, DPP>` (the fused decode kernel) |
| ink_attn.hip | b5 without the row-table launcher (no RT instantiation) | 30 / 30 | none |
| ink_mix.hip / ink_attn.hip | **b6**: row-table launchers in their own TUs | **31 / 31, 30 / 30, no kernel added** | none |

- The six differing kernels had the same VGPR / SGPR / LDS counts and 2 to 7 instructions more or fewer, in a different register
  assignment and order around the LDS loads of the attention body. Their source is unchanged for `RT = false`: the difference comes
  from the RT instantiations sharing the module. Without them the code is main's.
- The solo decode visits that kernel 40-48 times a step, so that difference is the likely cause of the 0.9957 ratio of sections 4 and 6.
- **b6** (`ff352fc37`): the row-table launchers move to two new translation units, `kernels/hip/ink_mix_rows.hip` and
  `kernels/hip/ink_attn_rows.hip`. Each one includes its base file with `IK_INK_MIX_ROWS_TU` / `IK_INK_ATTN_ROWS_TU` defined. The guard
  leaves out the base file's launchers, and in ink_mix.hip also the RT launch branch of `launch_rows`, which there returns
  `IK_ERR_UNSUPPORTED`. The kernels, the RT template parameter and the helpers stay in the base files, so nothing is copied. The
  solo modules now compile to main's device code, kernel for kernel. The RT kernels live in modules that the solo path never loads
  a kernel from.
