# Inkling-2-Mini batched decode on HIP, one R9700 (lane ink-hip-batch, 2026-11-26)

Branch `ink/hip-batch` from main `87c9745c0`. Goal: `serve --max-batch N` (several streams decoding in one pass) works and is fast
for Inkling-2-Mini on the HIP backend. Today HIP refuses `--max-batch > 1` with `NoKernel`: no HIP kernel registers `.row_table()`
(`model/batch.rs` `row_tables_wanted`: Inkling-2 needs row-table kernels of `ConvMix` and `InkAttnDecode`, or of the pair).
Raw: `docs/results/2026-11-26-inkling2-hip-batch/raw/`.

## 0. Summary (final binary b8 = this branch merged onto main `3226af2b9`, #528)

| tok/s, one R9700 | main `3226af2b9` | branch b8 | change |
|---|---|---|---|
| Prefill 4k (defaults since #528: chunk 2048, MoE WMMA, flash, PF2 forms; the 256- / 64-token runs) | 3094 / 3080 | 3084 / 3072 | -0.3 % (single runs, noise; ABBA on b7: 1.0014 ± 0.0030) |
| Prefill 16k | not run (COMMON-1126 baseline 2793 predates #528) | not run; solo device code identical to main (section 12) | 0 expected |
| Decode short (`--spec off`, 256 tok) | 76.90 | 76.83 | -0.1 % (single run; ABBA on b7 vs main3: 1.0003 ± 0.0003) |
| Decode 4k (`--spec off`, 256 tok) | 74.09 | 74.03 | -0.1 % (single run; ABBA on b7 vs main3: 1.0001 ± 0.0006) |
| Serve, 16 real prompts one at a time (solo server `--spec off`, 128 tok each, prefill included; b7, solo device code = main's) | 54.4 | 54.4 (same engine) | baseline is plain decode; HIP's default solo since #527 is DSpark, faster one at a time |
| Serve, real prompts at once: 4 / 8 / 16 (128 tok each, prefill included, 2 reps each; **b7**) | refused (NoKernel) | **94.2 / 106.0 / 116.2** | 16 at once: **2.14x** the same 16 one at a time (`--spec off`) |
| Serve aggregate, localeval 1k filler: 1 / 4 / 8 / 16 streams (256 tok, 3 reps; **b7**) | refused at > 1 | 66.6 / **130.0 / 168.4 / 189.9** | new |
| Pure batched decode B 16 (`batch_step`, real text), short / 4k (**b5**: same kernel source, built before the TU split, so the RT kernels' code may differ slightly) | refused | **289.1 / 253.0** (3.77x / 3.45x the solo step; section 5.3) | new |

- **Verdict: ships, opt-in by `--max-batch`.** Two bar items are short (below), both for lack of radiance time, not for a failure.
  - Batched == solo holds bitwise at every level:
    - kernels: 645 + 40 row-table calls (b5, b6, b7, **b8**);
    - model: 2800 tokens at B 1 / 4 / 8 / 16 (section 5.2, **b5**); the tiny-fixture batch suite on **b8**;
    - server: the real prompts sent at once against the solo server's replies: 16 / 16 (window 8), and 4 / 4, 8 / 8, 16 / 16
      (window 10), all on **b7**.
  - **The shipped head (b8, on #528) has no real-model batched == solo run.** Its evidence: the row-table unit test, the tiny
    batch suite, solo md5 = main and solo device ISA = main. #528's PF2 prefill forms also run in the busy chunks of a batching
    server; the real-model and serve checks ran on b7, before #528 (no window was left).
  - The solo path is main's:
    - solo device code identical to main 3226af2b9, kernel for kernel (31 / 31, 32 / 32);
    - greedy md5 at 64 / 256, short and 4k, equal to main on b8;
    - solo decode ratio 1.000 (ABBA, b7).
- **Real-prompt sweep (window 10)**: the 16 real prompts are 49 to 4253 tokens (25 752 in all) against 2048 generated tokens, so the
  aggregate is prefill-bound. Sending them all at once takes 17.6-17.7 s. Sending them one at a time to the solo server takes 37.6 s.
- **ITL p50 / p99** (server histogram buckets):
  - localeval 1k, at 1 / 4 / 8 / 16 streams: p50 10-20 / 20-35 / 20-35 / 50-75 ms; p99 10-20 / 20-35 to 250-500 / 250-500 /
    250-500 ms. The tail is other streams' 1k prefill chunks.
  - real prompts at once, at 4 / 8 / 16: p50 20-35 / 20-35 / 50-75 ms; p99 250-1000 ms. Here the tail is the 2048-token chunks of
    the 3-4k prompts.
  - Pure decode step time p50 / p99 at B 16: 55.4 / 56.6 ms.
- **VRAM**:
  - measured: 19.7 GB for the solo server; 28.5-28.6 GB for `--max-batch 16 --max-seq 4608 --kv-fraction 0.9` (4 visits);
  - **B 4 / B 8 not measured**: interpolated between the measured B1 19.7 and B16 28.6 GB, about 21.5 / 23.9 GB.
  - The largest B at 4k on 32 GB is 16, the engine's row cap, with about 4 GB to spare. It needs `--kv-fraction` of 0.6 or more.
- **Recipe**:
  - `max_batch = 16` (the `[defaults]` value) is the right cap on HIP.
  - No `busy_chunk` line is needed: HIP's default busy chunk is the solo chunk 2048. Busy chunk 256 breaks batched == solo and
    costs 25 % at 8 streams.
- **Since #527**: a batching server loads no DSpark draft and takes plain batched steps (section 9.1). One stream is faster on the
  solo server with DSpark.
- **Bar items not met (radiance budget, coordinator's call)**:
  1. The real-prompt sweep has 2 reps per level, not 3. N 16's two reps are window 8's and window 10's 16-at-once runs.
  2. VRAM per B is measured at B 1 and B 16 only; B 4 / B 8 are interpolated, as above.
- **Radiance down, whole lane: 31:30** over 10 windows (cap 30:00; the coordinator allowed up to 32:00 for window 10).

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
- 2026-11-26 ~14:05 (coordinator, after #528 merged as `3226af2b9`): one more window, about 3:30, hard limit 32:00 lane total,
  under 5:00. Order: (1) the full #528 re-gate, ink_rows bitwise first, then the 4k-256 md5 for each arm, then the 64-token md5 and
  the rest; (2) the real-prompt sweep on one B16 server, serve_bitwise conc mode on the ids-prompts at 4, 8 and 16 at once,
  aggregate plus server ITL. Skip the VRAM fix: write B4 / B8 as interpolated between the measured B1 and B16, labelled not
  measured. If step 1 fails, stop and report, no retry window. Final radiance total in the doc and the PR body.
- 2026-11-26 ~13:35 (coordinator): PR #527 (HIP DSpark default-on) is merged; main is `3c6ff64f0`. Rebase onto it before the gate.
  1. `inkling2::ink_rows_exact` on main already lists "hip". Take main's line and drop ours. (This branch never changed that line,
     so the merge took main's as is.)
  2. On HIP, spec decode (DSpark k3, cut 0.5) and the verify graph are default-on for Inkling-2, one sequence only. Check that
     `--max-batch N > 1` takes plain batched steps when the draft would load, not per-row spec, and say which path ran.
  3. `docs/tuning.md`: #527 edited the `INTARSIA_INK_ROWS_EXACT` row and five others in place. Merge into their text; add no
     duplicate row. (This branch adds no tuning row.)
  4. Run the bitwise checks (RT = false pins, solo greedy md5 at 64 / 256, short and 4k) on the rebased binary with no env vars set.

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

### 1.4 Final form (b7; it differs from 1.3 in two steps)

- b5 (section 6): `RowArgs` and `InkP` are main's. The row-table data lives in derived structs, `RowArgsRT` (`RowRt`) and `InkPR`
  (`InkRt`), and only the RT instantiations take them. The solo kernels' arguments are main's, byte for byte.
- b6 / b7 (sections 7-8): the row-table launchers live in two new translation units, `kernels/hip/ink_mix_rows.hip`
  (`ik_conv_mix_bf16_rows`) and `kernels/hip/ink_attn_rows.hip` (`ik_ink_attn_decode_bf16_rows`). Each includes its base file
  with a guard macro. In the base files the guards wrap the extern "C" launchers and, in ink_mix.hip, the RT launch branch of
  `launch_rows`. The solo modules compile to main's device code, kernel for kernel.
- `batch_step` adds `--solo-replay` and `--kv-fraction` (section 3).
- No change to `docs/tuning.md`, `recipes/inkling2.toml`, `inkling2::ink_rows_exact` or any core / server code.

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
- The solo decode visits that kernel 40-48 times a step. Window 6 (section 8) confirms that this difference caused the 0.9957 ratio
  of sections 4 and 6: with the split, the ratio is 1.000.
- **b6** (`ff352fc37`): the row-table launchers move to two new translation units, `kernels/hip/ink_mix_rows.hip` and
  `kernels/hip/ink_attn_rows.hip`. Each one includes its base file with `IK_INK_MIX_ROWS_TU` / `IK_INK_ATTN_ROWS_TU` defined. The guard
  leaves out the base file's launchers, and in ink_mix.hip also the RT launch branch of `launch_rows`, which there returns
  `IK_ERR_UNSUPPORTED`. The kernels, the RT template parameter and the helpers stay in the base files, so nothing is copied. The
  solo modules now compile to main's device code, kernel for kernel. The RT kernels live in modules that the solo path never loads
  a kernel from.

## 8. Window 6: b6 unit gate and solo gate (fleetq #12088; radiance down 13:27:26 - 13:31:09, 3:43)

Log: `raw/window6-12088-unit-gate-b6.log`. Stage: `raw/stage_w6.sh`. Main = `40b85a391` (`main2`), branch b6 = `ff352fc37`. No env
vars set. Summary by `raw/gate_ratio.py`.

| Check | Result |
|---|---|
| `tests/ink_rows.rs` | conv_mix_bf16_rows **645**, ink_attn_decode_bf16_rows **40** row-table calls bitwise the solo calls: PASS |
| `tests/ink_attn.rs` `rows_exact_real_shapes` | PASS |
| `kernels test --op conv_mix` / `--op ink_attn_decode` | 532 / 70 cases, 0 fail, ADMITTED |
| `tests/inkling2_batch.rs` on HIP, tiny fixture, `CI=1` | 8 / 8 PASS |
| greedy md5, 64 and 256 tokens, short and 4k | **equal to main** on every rep (the same four md5 as sections 4 and 6) |

| decode tok/s (256 tok, ABBA 4 + 4) | main | branch b6 | ratio (95 % CI) |
|---|---|---|---|
| short | 76.91 / 76.95 / 76.91 / 76.87 | 76.83 / 76.93 / 76.92 / 76.94 | **0.9999 ± 0.0013** |
| 4k | 74.08 / 74.05 / 74.02 / 74.02 | 74.09 / 74.08 / 74.03 / 74.06 | **1.0003 ± 0.0003** |
| 4k prefill (ms) | 1553 / 1562 / 1568 / 1573 | 1555 / 1560 / 1567 / 1573 | 1.0002 ± 0.0017 (speed) |

- The gap of b3 / b5 (0.9957) is gone: the solo decode of b6 equals main's within noise, at both contexts.

## 9. Window 7: b7 rebased onto main `3c6ff64f0` (#527), unit gate and solo gate (fleetq #12094; radiance down 13:40:19 - 13:44:02, 3:43)

Log: `raw/window7-12094-unit-gate-b7.log`. Main = `3c6ff64f0` (`main3`, md5 `69758c1f…`). Branch b7 = `e61050d5a` (`d7ac419c…`), the
merge of origin/main into b6. #527 did not touch `ink_mix.hip`, `ink_attn.hip` or `ik_hip.h`, so the solo modules are still main's,
kernel for kernel (section 7). No env vars set. The gate runs `generate --spec off`: since #527, HIP's default solo decode is DSpark,
and these are the plain-decode bits that the RT = false instantiations must pin.

| Check | Result |
|---|---|
| `tests/ink_rows.rs` | conv_mix_bf16_rows **645**, ink_attn_decode_bf16_rows **40** row-table calls bitwise the solo calls: PASS |
| `tests/ink_attn.rs` `rows_exact_real_shapes` | PASS |
| `kernels test --op conv_mix` / `--op ink_attn_decode` | 532 / 70 cases, 0 fail, ADMITTED |
| `tests/inkling2_batch.rs` on HIP, tiny fixture, `CI=1` | 8 / 8 PASS |
| greedy md5, 64 and 256 tokens, short and 4k | **equal to main** on every rep (`04cdb28a…`, `9a27c21b…`, `2252566a…`, `9675bbb9…`) |

| decode tok/s (256 tok, ABBA 4 + 4) | main `3c6ff64f0` | branch b7 | ratio (95 % CI) |
|---|---|---|---|
| short | 76.87 / 76.91 / 76.87 / 76.86 | 76.89 / 76.95 / 76.88 / 76.89 | **1.0003 ± 0.0003** |
| 4k | 74.04 / 74.09 / 74.06 / 74.04 | 74.09 / 74.08 / 74.05 / 74.04 | **1.0001 ± 0.0006** |
| 4k prefill (ms) | 1554 / 1564 / 1568 / 1574 | 1553 / 1558 / 1569 / 1571 | 1.0014 ± 0.0030 (speed) |

### 9.1 Which path a batching server takes since #527 (coordinator ask 2)

`crates/intarsia-models/src/load.rs`, as merged:
- **No `--spec`** (the default): `dspark_default` returns `None` when `--max-batch` > 1 (its unit test `dspark_default_tests`
  asserts `None` at 2 / 4 / 16 for cuda, metal and hip). The draft is not loaded, and the batching engine takes plain batched
  steps: `forward_batch` with the row-table kernels above.
- **`--spec dflash --max-batch N > 1`**: refused before the weights are read, with "--spec dflash with the batching engine (use
  --max-batch 1)". This is the same on every backend.
- **`--spec mtp` / `auto` with `--max-batch N > 1`**: refused unless batched MTP is on (`[tuning.<backend>] batched_mtp` or
  `INTARSIA_BATCHED_MTP`). The Inkling-2 recipe sets neither.
- So no batched path ever runs per-row spec on HIP. The servers of windows 8 and 9 run without `--spec`, and their logs show the
  path (section 10).

## 10. Serve level: windows 8 and 9 (fleetq #12095, #12096; radiance down 13:44:34 - 13:49:06, 4:32, and 13:49:34 - 13:53:37, 4:03)

Raw: `raw/serve-w8/`, `raw/serve-w9/` (driver logs, server logs, localeval result.json and timestamped output, `/metrics` polls every
0.2 s gzipped, summaries). Scripts: `raw/serve_window.sh`, `raw/serve_ctl.sh`, `raw/hold.sh`, `raw/serve_bitwise.py`, `raw/itl_seg.py`,
`raw/serve_sum.py`. Plans: `raw/visits-w8.txt`, `raw/visits-w9.txt`. Binary b7 throughout. `intarsia serve -m Inkling-2-Mini
--backend hip --max-seq 4608`, plus `--max-batch 16 --kv-fraction 0.9` for the batching servers. No env vars set.

### 10.1 The path the servers took

- Batching servers: `continuous batching: up to 16 sequences (--max-batch), busy prefill chunk 2048`. No DSpark line, so no draft
  was loaded. This is the default with no `--spec` (section 9.1): plain batched steps.
- The default busy chunk on HIP is the solo chunk, 2048, even without `--busy-chunk`. HIP batches Inkling-2 as a hybrid model
  (`allow_recurrent_batch`), and `load.rs` then sets `busy_chunk = model.prefill_chunk()`. Window 9's `mb16-def` server ran
  without the flag and logged `busy prefill chunk 2048`. So `[tuning.hip]` needs no `busy_chunk` line.
- Admission on plan shapes: the first start ran 134 cases in 101 s, with 23 kernels checked. Later starts hit the cache
  (`134 cached`) and started in 17-18 s.

### 10.2 Batched == solo through the server (16 real prompts)

The 16 prompts of `serve_bitwise.py` come from `scripts/cmx/ids-prompts`: the 7 files plus 9 prefixes of the two 4k documents, 49 to
4253 prompt tokens each, greedy, 128 forced tokens.
- **Batched**: all 16 sent at once to a `--max-batch 16` server (window 8, busy chunk 2048).
- **Solo**: the same 16, one at a time, to a server with no `--max-batch` and `--spec off` (window 9).

| Batched server | Replies equal to solo (text md5 and token count) |
|---|---|
| busy chunk 2048 (the default) | **16 / 16**. No cached tokens. The 4253-token prompts prefill in three 2048 chunks beside decoding rows. |
| busy chunk 256 (`--busy-chunk 256`) | 8 / 16. Equal: the three prompts under 60 tokens, prompts of 226 and 234 tokens, one of 823 tokens and the two 4k prompts. These fit one 256 chunk, or prefilled before the server got busy (a lone prompt takes the solo chunk). The 8 others diverge from character 19-321. |

The prefill (chunk 2048, MoE WMMA and flash attention) is not chunk-invariant. Batched == solo therefore needs the busy chunk equal
to the solo chunk, and that is HIP's default.

### 10.3 Throughput and latency: localeval speed, 1k prompt, 256 forced tokens, 1 / 4 / 8 / 16 streams

- Each stream count ran three times: window 8 reps 1-2 (`mb16-bc2048`) and window 9 rep 3 (`mb16-def`, the same configuration with
  the default busy chunk).
- Every stream sends localeval's filler prompt with its own nonce. These are synthetic prompts. Real text is in 10.2 and in
  section 5.3 (`batch_step --prompts text`).
- The aggregate is completed tokens over the wall time of the whole level, so it includes every stream's prefill and TTFT.
- ITL and step percentiles come from the server's own histograms (`intarsia_batch_itl_seconds`, `intarsia_batch_step_seconds`),
  split by level. They are given as the bucket the percentile falls in; the bounds are 10, 20, 35, 50, 75, 150, 250, 500 ms.

| Streams | aggregate tok/s, reps 1 / 2 / 3 | mean | x 1 stream | per-stream decode tok/s (median) | TTFT median / max s | ITL p50 bucket | ITL p99 bucket |
|---|---|---|---|---|---|---|---|
| 1 | 65.7 / 67.0 / 67.0 | **66.6** | 1.00 | 74.1 / 75.6 / 75.6 | 0.45 / 0.45 | 10-20 ms | 10-20 ms |
| 4 | 128.1 / 134.1 / 127.8 | **130.0** | 1.95 | 39.5 / 41.9 / 39.4 | 1.51 / 1.75 | 20-35 ms | 20-35 to 250-500 ms |
| 8 | 168.7 / 167.9 / 168.5 | **168.4** | 2.53 | 26.4 / 26.2 / 26.3 | 2.37 / 3.51 | 20-35 ms | 250-500 ms |
| 16 | 189.2 / 190.2 / 190.2 | **189.9** | 2.85 | 14.8 / 14.9 / 14.8 | 4.17 / 7.20 | 50-75 ms | 250-500 ms |

- The ITL p50 bucket holds the pure decode step of section 5.3: 13.0 / 23.6 / 33.3 / 55.4 ms at B 1 / 4 / 8 / 16.
- The p99 bucket of 250-500 ms is a decoding row's wait while another stream's 1k prompt prefills in one 2048-row chunk, about
  0.45 s. At 4 streams there are about 6 such waits in about 1020 gaps (0.6 %), close to the 99th percentile, so the two ways of
  splitting the levels read it on either side (reps 1-2 split by decode rows, because localeval's stdout was block-buffered then;
  rep 3 by localeval's own timestamped lines).
- Per-stream decode tok/s is localeval's rate after the first token. At 16 streams it is lower than 1 / 55 ms = 18 tok/s, because
  of those prefill waits.
- Over the whole of window 8, 97.3 % of the 19 882 gaps between tokens were at most 75 ms, and 136 were over 500 ms. Those 136 were
  the 4k prompts' 2048 chunks in the real-prompt phase.
- Pure decode, no prefill in the window (section 5.3, `batch_step`, real text, b5; the row-table kernels' source has not changed
  since, though they now compile in their own TUs):
  aggregate 77.0 / 169.6 / 240.6 / **289.1** tok/s at B 1 / 4 / 8 / 16 short (3.77x the solo step), and 73.4 / 156.2 / 217.6 /
  **253.0** at 4k (3.45x). Metal's B 16 was 173.7 tok/s, 3.18x (`2026-11-21-inkling2-batch.md` section 4).

### 10.4 Busy-chunk check (window 9, `--busy-chunk 256` against the default 2048, 8 streams once)

| Busy chunk | 8 streams aggregate tok/s | ITL p99 bucket | step p99 bucket | 16 real prompts at once: wall | ... equal to solo |
|---|---|---|---|---|---|
| 2048 (default) | 168.4 (3 reps) | 250-500 ms | 250-500 ms | 17.6 s | 16 / 16 |
| 256 | 126.4 (-25 %) | 150-250 ms | 250-500 ms | 35.1 s | 8 / 16 |

- A 256 chunk halves the worst ITL bucket, but costs a quarter of the aggregate at 8 streams and doubles the wall time of the
  real-prompt mix. It also breaks batched == solo.
- **Keep the default** (the solo chunk). The 16-stream level of the 256 arm ran out of window time and was cut (localeval exit 142).

### 10.5 Memory

| Server | VRAM peak, card 0 (sysfs, sampled every 0.5 s over the visit) |
|---|---|
| solo (no `--max-batch`, `--max-seq 4608`, default `--kv-fraction 0.5`) | **19 662 MiB** |
| `--max-batch 16 --max-seq 4608 --kv-fraction 0.9` (window 8, and window 9 twice) | **28 611 / 28 518 / 28 518 MiB** |

- The batching server reserves its 16 sequences at load (section 5.5: 348 MiB of rings and snapshots plus 32 KiB per token of
  full-attention KV per sequence, about 492 MiB at 4608 tokens). The peak is therefore that of 16 sequences at a 4.6k context,
  whatever the prompts.
- **B 4 and B 8 were not measured** (coordinator's call: the radiance budget went to the re-gate and the real-prompt sweep).
  Interpolated linearly between the measured B1 19.7 and B16 28.6 GB: B 4 about 21.5 GB, B 8 about 23.9 GB. The B1 end is the
  solo engine at `--kv-fraction 0.5`. Both engines reserve KV for their context, not the whole budget (`memplan.rs`), so the
  fraction does not move the peak; the batch workspace at B 4 / 8 is the unknown.
- **Largest B at 4k on 32 GB: 16**, the engine's row cap (`kernels::ROWS_MAX` = `Model::BATCH_MAX_ROWS` = 16). It fits with about
  4 GB to spare at a 4608-token context. Memory alone would allow about 8 more sequences at 480-492 MiB each, if the row cap were
  raised.
- B 16 at 4k needs a KV budget of at least about 7.9 GB, which means `--kv-fraction` of 0.6 or more (0.9 tested). At the default
  0.5, the plan clamps B 16 to 3285 tokens per sequence (window 2).

### 10.6 Recipe cap on HIP

- `max_batch = 16`, the `[defaults]` value, is the right cap on HIP. It equals the engine's row cap and fits at a 4k context, so
  no HIP override is needed.
- Batching stays opt-in (`--max-batch`). A one-stream user keeps the solo server, which since #527 speculates with DSpark (1.19x at
  1k, 1.34x short). A batching server loads no draft, so it starts to win at 2 or more streams: 130 tok/s at 4 against 66.6 plain
  and about 80-90 with DSpark.

## 11. Windows, PR, leftovers

| Window | fleetq | What | Radiance down |
|---|---|---|---|
| 1 | #12049 | unit tests (b2) | 12:18:04 - 12:20:00, 1:56 |
| 2 | #12057 | batch_step on the real model (did not fit) | 12:33:34 - 12:35:08, 1:34 |
| 3 | #12064 | solo gate b3 | 12:44:34 - 12:47:27, 2:53 |
| 4 | #12076 | unit gate b5, batch_step bitwise / speed / trace | 13:00:30 - 13:03:37, 3:07 |
| 5 | #12080 | solo gate b5 (0.9957) | 13:07:35 - 13:10:29, 2:54 |
| 6 | #12088 | unit + solo gate b6 (TU split: 1.000) | 13:27:26 - 13:31:09, 3:43 |
| 7 | #12094 | unit + solo gate b7 (on main `3c6ff64f0`) | 13:40:19 - 13:44:02, 3:43 |
| 8 | #12095 | serve: batched replies, localeval reps 1-2 | 13:44:34 - 13:49:06, 4:32 |
| 9 | #12096 | serve: solo replies, rep 3, busy chunk 256 | 13:49:34 - 13:53:37, 4:03 |
| 10 | #12108 | #528 re-gate (b8 vs main4) + real-prompt sweep 4 / 8 / 16 (coordinator-approved past 30:00, limit 32:00) | 14:08:55 - 14:12:00, 3:05 |
| | | **total** | **31:30** (cap 30:00; 32:00 allowed for window 10) |

- CPU jobs: builds #12035 (cancelled by the coordinator: `--mem`), #12087, #12091, #12102 and others; device-ISA jobs #12083,
  #12086, #12102; pushes #12098, #12100, #12106 (endor).
- PR: https://github.com/intarsiaml/intarsiaml/pull/529.
- Left on tatooine:
  - `~/intarsiaml-ink-hip-batch`: binaries `intarsia-b8` and `intarsia-main4` (b7 / main3 deleted after window 10), logs, and the
    serve admission cache in `.scratch/home`;
  - the volume `intarsia-cargo-target-ink-hip-batch` (2.3 GB).
  No container or process of the lane is running.
- Not measured, because the window time went to the gates and the serve checks:
  - bar items: a third rep of the real-prompt sweep, and VRAM of B 4 / B 8 servers (interpolated in 10.5) — see section 0;
  - not in the bar: serve at a 4k filler prompt (the pure-decode 4k numbers are section 5.3), and prefill 16k (same device code as
    main).

## 12. Window 10: #528 re-gate and the real-prompt sweep (fleetq #12108; radiance down 14:08:55 - 14:12:00, 3:05)

Raw: `raw/serve-w10/` (pre-stage output `pre.out`, its verdict `gate10-check.txt`, driver log, server log, per-level replies and
meta, `/metrics` poll gzipped, `summary.txt`). Scripts: `raw/stage_w10.sh`, `raw/gate10_check.py`, `raw/serve_window.sh` (the `PRE`
stage and the `real:` mode), `raw/serve_bitwise.py` (`conc:N:OFF`), `raw/real_sum.py`, `raw/itl_seg.py`. Plan: `raw/visits-w10.txt`.

### 12.1 The merge of #528

- #528 (`3226af2b9`, HIP prefill PF2 forms default-on) conflicted with this branch in `ink_attn.hip`. The conflict was two
  independent additions next to each other: `InkRt` / `InkPR` here, the PF2 helpers there. Both were kept. Merge `8dc0db447`.
- `InkPR` inherits `InkP`, so it gets #528's new field `qhi`. Only `ink_flash_k` reads it, `ink_attn_dec_k` does not, and the rows
  launcher value-initialises `InkP` (`InkP p = {};`).
- Device ISA against main `3226af2b9` (#12102, cpu): `ink_mix.hip` 31 / 31 and `ink_attn.hip` 32 / 32 kernels identical, none added.
- Binaries: main4 = `3226af2b9` (md5 `fdd5c44e…`), b8 = `8dc0db447` (`2f41908f…`). No env vars set.

### 12.2 Re-gate (pre-stage, 100 s, verdict GATE PASS)

| Check, in run order | Result |
|---|---|
| `tests/ink_rows.rs` | 2 / 2 PASS: row-table calls bitwise the solo calls |
| greedy md5 4k 256, main4 / b8 | `9675bbb9…` / `9675bbb9…` **equal** (the same md5 as main before #528) |
| greedy md5 short 256 | `9a27c21b…` **equal** |
| greedy md5 short 64 / 4k 64 | `04cdb28a…` / `2252566a…` **equal** |
| `tests/ink_attn.rs` `rows_exact_real_shapes` | PASS |
| `kernels test --op conv_mix` / `--op ink_attn_decode` | 532 / 70 cases, 0 fail, ADMITTED |
| `tests/inkling2_batch.rs` on HIP, tiny, `CI=1` | 8 / 8 PASS |

Single runs, main4 / b8: decode 4k 74.09 / 74.03 tok/s, short 76.90 / 76.83 tok/s, prefill 4k 3094 / 3084 tok/s. These are noise
around equal; the ABBA ratio is section 9's. The device code is main's (12.1).

### 12.3 Real-prompt sweep: 4, 8 and 16 of the real prompts at once

- One `--max-batch 16 --max-seq 4608 --kv-fraction 0.9` server, default busy chunk (2048), no `--spec`. Binary **b7**: its
  plan-shape admission was cached, so it started in 18 s; b8's would have rerun 134 cases (about 85 s) and not fit the window. b7 and
  b8 differ only by #528's prefill forms.
- Prompts: the 16 of `serve_bitwise.py` (49 to 4253 tokens), sorted by length. A level of N takes every (16 / N)-th from an offset,
  so each set spans short to 4k. Rep r uses offset r - 1. Rep 1 sends the plain prompts, so its replies can be compared with the
  solo server's (window 9). Rep 2 puts a tag in front, so no prefix is shared (the server reported 0 cached tokens in every rep).
- 128 forced greedy tokens per request. Aggregate = all completion tokens / the wall time from the first send to the last reply,
  prefill included. ITL and step buckets come from the server histograms over the level's span.

| N at once | rep | prompt tokens | aggregate tok/s | request latency median / max s | ITL p50 / p99 bucket | replies equal to solo |
|---|---|---|---|---|---|---|
| 4 | 1 | 5 241 | 91.9 | 5.54 / 5.57 | 20-35 / 500-1000 ms | **4 / 4** |
| 4 | 2 | 5 434 | 96.6 | 5.27 / 5.30 | 20-35 / 250-500 ms | (tagged) |
| 8 | 1 | 12 019 | 110.8 | 9.21 / 9.24 | 20-35 / 500-1000 ms | **8 / 8** |
| 8 | 2 | 13 789 | 101.3 | 9.98 / 10.11 | 20-35 / 500-1000 ms | (tagged) |
| 16 | 1 | 25 752 | 115.7 | 17.31 / 17.70 | 50-75 / 500-1000 ms | **16 / 16** |
| 16 | window 8 | 25 752 | 116.7 (wall 17.55 s) | | | 16 / 16 |
| 1 (solo server, one at a time) | window 9 | 25 752 | 54.4 (wall 37.63 s) | | | (reference) |

- Means: **94.2 / 106.0 / 116.2 tok/s** at 4 / 8 / 16 at once, with 2 reps each. The 16-at-once set is the same workload as the solo
  server's one-at-a-time run, and it finishes in 17.6 s instead of 37.6 s: **2.14x**.
- This workload is prefill-bound: about 12.6 prompt tokens per generated token. So the gain over the 1k filler sweep (189.9 at 16)
  is smaller. Every decoding row waits while the 3-4k prompts prefill in 2048-token chunks of about 0.7 s each, and those waits are
  the 500-1000 ms p99 bucket. The p50 bucket is the batched decode step (20-35 ms at 4 / 8 rows, 50-75 ms at 16).
- Rep 2 of N 16 was skipped by the time guard (24 s left, 46 needed), so the bar's 3 reps were not reached (section 0).
- VRAM peak of the visit: 28 518 MiB (card 0), as in windows 8 and 9.

