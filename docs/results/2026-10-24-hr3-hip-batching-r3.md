# hr3: row tables, graphs and the fused MoE block for batched decode on HIP (R3 port)

Lane hr3, 2026-10-23 evening to 2026-10-24 early morning, tatooine card 0 (R9700, gfx1201), Qwen3.6-35B-A3B NVFP4, `serve --max-batch 16`.
Ports the CUDA R3 work of [rb3](2026-10-23-rb3-batching-r3.md) (PR #121) to the HIP batching of
[hb](2026-10-23-hb-hip-hybrid-batching.md) (PR #120). Branch `lane-hr3/hip-r3-batching`, base `origin/main` 6189466f (PR #125). Draft PR #127.
Machine-readable trace: [trace.json](2026-10-24-hr3-hip-batching-r3/trace.json). Raw logs: `2026-10-24-hr3-hip-batching-r3/raw/`.

## 1. Result

Every decode row of a batched step is still bitwise the solo step (section 5). 35B aggregate decode, localeval, 1024-token prompt, 128
generated, 3 reps per burst, `serve --max-batch 16` on both sides, main and branch served one after the other inside the same fleetq slot
(`--quiet gpu,cpu`, radiance down), three bursts per cell (window 7, fleetq #4929; main = `origin/main` 6189466f, built the same way):

| streams | main | hr3 | change | bursts main | bursts hr3 |
|---|---|---|---|---|---|
| 1 | 157.5 | **158.0** | +0.3% (noise, limit was -1%) | 157.4, 157.6, 157.4 | 158.1, 157.9, 157.9 |
| 4 | 194.4 | **212.6** | +9.4% | 194.2, 194.1, 194.8 | 216.0, 211.3, 210.5 |
| 16 | 231.8 | **257.1** | +10.9% | 230.8, 232.5, 232.1 | 255.8, 260.6, 254.9 |
| 4, 4096-token prompts | 141.2 | **145.3** | +2.9% | one run of 3 reps | one run of 3 reps |

An earlier window (6, fleetq #4905, tree before the last two small fixes, one burst per arm) gave the same picture: 4 streams 193.3 (main) vs 204.6, 16 streams 231.0 vs 256.6. The lane brief's main figures (157.8 / 195.2 / 234.7) agree with the main column within the burst-to-burst spread (about 1%, up to 3% at 4 streams). Prefill was not touched by this lane (no prefill code path changed).
Single-stream decode does not move: a solo request runs the solo step, none of the new code is on its path.

What each lever is worth (hr3 tree with one lever switched off, same window, three bursts, aggregate decode tok/s):

| arm | 1 | 4 | 16 | 4 at 4k |
|---|---|---|---|---|
| main | 157.5 | 194.4 | 231.8 | 141.2 |
| **hr3** | **158.0** | **212.6** | **257.1** | **145.3** |
| no fused MoE block rows (`INTARSIA_HIP_DISABLE=moe_block_bf16_rows`) | 157.6 | 203.5 | 256.6 | 142.1 |
| no graphs (`INTARSIA_BATCH_GRAPHS=0`, eager batched step) | 157.6 | 205.8 | 254.8 | 143.6 |
| no row tables (`INTARSIA_ROW_TABLES=0`, one launch per row per op, which also rules out graphs) | 157.6 | 198.9 | 231.9 | 140.2 |

* (a) row tables: +6.9% at 4 streams (198.9 to 212.6) and +10.9% at 16 (231.9 to 257.1), the largest part of the gain.
* (b) graphs per row count: +3.3% at 4 streams, +0.9% at 16 (CUDA got +7% at 4 streams; the HIP step is less launch-bound).
* (c) fused MoE block at 3 and 4 rows: +4.5% at 4 streams, nothing at 16 (the cap is 4 rows, see section 4).
* (d) untouched; device sampling stays opt-in and is not available on HIP (section 2).

Model-loop step time (`real_35b_batch_step_timing`, eager, greedy, no engine, window 7, rows 2..16), the final tree with and without the fused
MoE block rows kernel (`INTARSIA_HIP_DISABLE=moe_block_bf16_rows`), and main's step from hb's last run (different session, shown for scale only;
its "solo step" line is the test's cold first step, not a figure):

| rows | main (hb #4620) ms | hr3 without moe rows ms | hr3 ms | hr3 tok/s aggregate |
|---|---|---|---|---|
| 2 | 9.6 | 9.43 | 9.44 | 212 |
| 3 | | 12.11 | 11.63 | 258 |
| 4 | 14.4 | 13.83 | 13.61 | 294 |
| 5 | | 16.27 | 16.30 | 307 |
| 6 | | 17.81 | 17.83 | 337 |
| 8 | 24.2 | 22.23 | 22.14 | 361 |
| 12 | | 30.86 | 30.77 | 390 |
| 16 | 46.0 | 41.75 | 41.60 | 385 |

The served numbers (213 and 257 tok/s at 4 and 16 streams) are far below the model-loop ones (294 and 385): see section 4 and the open points.

## 2. What changed

**(a) Row tables (HIP kernels in `kernels/hip`).** One launch per layer and op for all rows of a step, instead of one per row. The kernels take
the table of device addresses of `OpCall::rows` (`u8 [rows, stride, 8]`, the same contract as CUDA's) and, where the row's position matters,
the positions of all rows on the device (`pos`, `i32 [rows]`):

| kernel (`row_table` in the registry) | twin of | grid |
|---|---|---|
| `gdn_conv_bf16_rows` | `gdn_conv_bf16` (T = 1: conv and state update in one kernel) | `.y` = row |
| `gdn_delta_rule_bf16_v2_rows` | `gdn_delta_rule_bf16_v2` (T = 1), state at table entry plus byte offset | `.z` = row |
| `kv_append_bf16_rows` | `kv_append_bf16` | one thread per (row, head, vector) |
| `attn_decode_bf16_splitkv_rows` | `attn_decode_bf16_splitkv` with a device position (the one-row decode grid) | `.z` = row, scratch for `rows` rows |
| `rope_bf16`, position per token | `rope_bf16` | `pos` of numel T > 1: token t at `pos[t]` |

Each is the one-row kernel's code with the row's pointers taken from the table, so row r is the one-row launch on sequence r alone
(`tests/row_tables.rs`, section 5). The solo step runs the fused `gdn_conv_delta_bf16` and `gate_split_qk_norm_rope_kv_bf16`; the batched step runs
the chains of those folds with the row-table kernels, which the fusion table proves equal (`crate::fusion` gets HIP entries for
`GateSplitQkNormRopeKv`, `QkNormRopeKv` and `GdnConvDelta`, each naming its existing bitwise test; `QkNorm` and `Rope` join the row-parallel ops of
the batch-consistency check; `GateSplit` may run once over all rows or once per row). `model/batch.rs` does not use the per-sequence fold when
row tables run (a launch bound to one sequence's cache cannot be in a batched graph).

**(b) HIP graphs per batch size.** No new code: `Backend::supports_graphs` is already true on HIP, and the generic batched-graph path of rb3
(`graph_step.rs`, one graph per row count, row tables rewritten before each replay) engages as soon as the step has no per-row launches left.
The A/B above is the measurement. The `moe_block` kernel's sync counters are reset by the kernel itself, so replaying it in a graph is safe
(`moe_block_graph_replay_is_bitwise_eager`, an existing test, passes).

**(c) `moe_block_bf16_rows`.** `moe_block` fuses top-k, routed experts, shared expert and combine in one launch, but only at `T * K <= 16`, i.e.
T <= 2 at top-8: steps of 3 or more rows ran the unfused chain (about 7 launches per layer). Every role of the kernel already works on one
token, so lifting the bound to T <= 16 (`ik_moe_block_bf16_rows`, registered `rows_exact`, same kernel code, only the shape check differs) keeps
each token bitwise the T = 1 launch on it alone: `moe_block_rows_are_the_solo_block_per_token` compares 1,252 tokens over 168 launches (T = 1..16,
six shapes including the 35B's top-8 of 256, norm on and off, with routing ties) with T = 1 blocks, plus a negative control per launch.
The batched step could not take it before: the block is only tried when the shared-gate logit was already computed by the router pair, and the pair
is a one-row kernel. In `quant/moe.rs` a batched pass now computes the gate logit with its own GEMV (the second half of the pair, a registered
equivalent) first, when a row-exact `MoeBlock` is selectable (`OpRunner::fused_available`, asked only with batching on and a row-exact pass, so
nothing changes with batching off, section 5). **The model takes it for 3 and 4 rows only** (`MOE_BLOCK_ROWS_MAX` in `quantmoe.rs`): past
that it is slower than the unfused chain (section 4). The kernel itself stays row-exact for 1..16.

**(d) Multi-row fp8/nvfp4 GEMVs: not touched.** hb measured them ALU-bound past 4 rows; the profile below says why they are the next lever and what
they cost (section 4), and no change was made.

**Device sampling** stays opt-in (`INTARSIA_DEVICE_SAMPLING=1`, user decision 2026-10-23 17:57) and is not ported: HIP has no `topk_rows`
(`supports_topk` is false), and the speed matrix is greedy, where it changes nothing.

**Shared crates.** `plan.rs`: `admit` leaves row-table kernels to their base's admission (they cannot be graded with table-less cases) and queues
the base kernel of a twin that no planned call uses, to be tested by its sweep (section 6); `GateSplit` joins `WIDE_OR_PER_ROW_OPS`, `QkNorm` and
`Rope` join `ROW_PARALLEL_OPS`; unit test `batch_check_accepts_hip_folds_per_row_or_as_row_table_chains`. `fusion.rs`: the three HIP entries.
`model.rs` and `quant/moe.rs`: `OpRunner::fused_available` and the gate-logit-first path above. `model/batch.rs`: eight lines. All inert on CUDA
and Metal (no `MoeBlock` kernel there; section 5).

## 3. Method

* Speed: `intarsia serve --backend hip --model Qwen3.6-35B-A3B-NVFP4 --max-seq 8192 --max-batch 16` in the dev container on card 0, then
  `localeval speed --url http://tatooine:8294/v1 --model intarsia --thinking off --sizes 1024 --gen 128 --reps 3 --concurrency 4,16` (and
  `--sizes 4096 --concurrency 4`), inside the fleetq job (`--takes-down radiance --quiet gpu,cpu`). Main and branch are built by the same
  command into separate cargo target volumes and served one after the other in the same job, so there is no drift between them; every arm is a
  fresh server. Burst-to-burst spread is about 1% (up to 3% at 4 streams). Run ids are in trace.json.
* Radiance was taken down for each window and restored at its end (both ranks `active` and `up` in `fleetq status`).
* One lesson of the tooling: `rsync -a` keeps the edit time of a file, and cargo skips the rebuild when the build is newer than that time.
  Twice a "final" serve binary was stale (one test binary rebuild proved it); the windows after that sync with `--no-times --checksum` and touch the
  sources, and the serve binary's op list was checked from the model-loop test (`OPLIST`).

## 4. Why the numbers are what they are

Step profile of the final tree (`real_35b_batch_step_profile`, each op synchronised, so the wall is longer than the real step; ms per step):

| B = 16 (op time sum 54.95 ms) | calls | ms | share |
|---|---|---|---|
| matmul_fp8 (attention and GDN projections) | 130 | 14.17 | 25.8% |
| moe_experts (unfused chain, rows kernel) | 40 | 13.76 | 25.0% |
| matmul_nvfp4 (shared expert and lm head) | 121 | 7.61 | 13.9% |
| matmul (bf16: router, gate logit) | 140 | 6.03 | 11.0% |
| gdn_delta_rule | 30 | 3.90 | 7.1% |
| rmsnorm_add | 80 | 1.79 | 3.3% |
| attn_decode, gdn_conv (now one launch per layer) | 10, 30 | 1.41, 1.05 | 4.5% |
| moe_topk, silu_glu, moe_combine, gated_rmsnorm, rope, qk_norm, the rest | | 5.2 | 9.5% |

At B = 4 (wall 23.9 ms profiled): moe_block 6.8, matmul_fp8 5.6, matmul 3.2, rmsnorm_add 1.8, gdn_delta_rule 1.7, nvfp4 0.7, the per-row ops 2.2.

* **Row tables** remove the per-row launches of five ops: before, B = 16 meant 16 launches of each of conv, delta rule, append, rope, qk_norm and
  attention per layer; now one. The remaining cost of these ops is 8.7% of the B = 16 step.
* **The fused MoE block is slower past 4 rows.** Uncapped (all rows 3 to 16 through `moe_block_bf16_rows`) the model-loop step was 23.73 ms at 8 rows
  (22.21 ms without) and 47.57 ms at 16 rows (41.75 ms without, window 7), +7% and +14%: the fused block reads each token's experts separately, while the
  unfused `moe_experts_bf16_rows` shares an expert's weights between the rows routed to it (about half the expert reads at 16 rows). At 3 and 4 rows
  the fused block still wins (12.11 to 11.63 ms, 13.83 to 13.61 ms; 4 streams +4.5% served), so the cap is 4. Raw logs for both runs are in `raw/`.
* **(d) is the next lever.** At B = 16 the fp8, nvfp4 and bf16 GEMVs plus the expert chain are 76% of the profiled step; the fp8 GEMV alone is 25.8%
  (109 us per call). hb found these kernels ALU-bound past 4 rows (they decode the weights once per row-tile), which the profile does not
  contradict (the fp8 time grows 2.5x from B = 4 to B = 16 while the bytes read stay the same). A tensor-core (WMMA) multi-row fp8 kernel that is also
  bitwise the one-row GEMV is a different kind of change (the accumulation order is part of the contract) and was not attempted here.
* **Served vs model loop.** At 4 streams the model loop does 13.6 ms per step (294 tok/s aggregate) but serve delivers 213 tok/s, i.e. 18.8 ms per
  step; at 16 streams 41.6 ms against 62 ms. So 5 to 20 ms per step is not in the model-loop step. Where it goes (scheduler and engine thread, per-row events and SSE, detokenising,
  the host side of the batched step, prompt prefills of staggered streams inside the burst) was not profiled here, so it is an open point, not a finding.

## 5. Gates

All on the final tree (head `ef27cfed` for the GPU gates, see the PR for the exact head) unless marked.

| gate | result | fleetq |
|---|---|---|
| G-B row-table kernels (`tests/row_tables.rs`, M = 1..16, 2 to 3 shapes each, positions 0..1199): conv 27.5 MB, delta rule 323 MB, KV append 33 MB, attention 2.2 MB, rope per-token positions 2.4 MB, and the fused solo `gdn_conv_delta_bf16` against conv + delta rule 300 MB, all byte-identical to the solo launches | 8 of 8 tests pass | #4929 (and #4905, #4903) |
| negative controls (swapped table entries for conv, delta rule, KV append and attention, swapped positions for attention and rope each change the output) | pass, inside the 8 | same |
| G-B `moe_block_bf16_rows` (`tests/moe_block.rs`): 168 launches, 1,252 tokens bitwise the T = 1 block, 144 negative controls differ; existing `moe_block` tests (chain equality, counters, graph replay, dispatch order, 6,000-launch sentinel stress) | 7 of 7 | #4929 |
| G-B R0 regression (`rows_exact`, `rows_registry`) | 5 of 5 and 2 of 2 | #4929 |
| G-C 35B (`real_35b_batched_decode_is_bitwise_the_solo_step`): batched decode == solo step | 17,550 (row, step) pairs bitwise identical, 3 tests pass 0 fail (35B bitwise, engine == LocalEngine, busy-chunk KL gate) | #4929 |
| batched engine == LocalEngine, including the `--ignored` long-prompt test (`real_35b_batched_engine_long_prompts_equal_the_local_engine`, read unfiltered) | `1 passed; 0 failed`, 8 filtered out | #4929 |
| tiny hybrid (`batch_recurrent`, HIP, N = 2..16 x 130 steps, decode beside prefill chunks, prefix reuse, memory plan) | 10 of 10 (after the pick check learned that the MoE block of a batched step may be the fused rows kernel) | #4929 (second part) |
| HIP backend suite (`cargo test --no-fail-fast -p intarsia-backend-hip --tests`) | all 34 test binaries ok, 0 failed (`raw/w7-hip-suite.log`) | #4929 |
| No-op, batching off, HIP: `plan show` text and `--json` byte-identical to main; 256 greedy ids after a 1.2k-token prompt identical; `--spec mtp --spec-k 2` ids identical to main and to spec off | identical (3,184 B and 12,961 B plans, ids sha equal) | #4929 |
| No-op, CUDA (spark1): 8B and 35B plans byte-identical, 256 ids identical, `intarsia-core` lib 115 of 115, CUDA batching tests 10 of 10 | @@CUDA@@ | @@CUDAJOB@@ |
| No-op, Metal (endor): plans byte-identical, ids identical, `batch_recurrent` on Metal 10 of 10 | @@METAL@@ | @@METALJOB@@ |
| Mac `cargo test --workspace --release --no-fail-fast` 871 passed, 2 failed, 17 ignored; the 2 failures are the `rows_registry` pin tests that also fail on main (section 6) (`raw/workspace-tests-mac.txt`) | local (Mac) |
| `intarsia-core` unit tests (incl. `batch_check_accepts_hip_folds_per_row_or_as_row_table_chains`), `docs_env_vars` | 115 of 115, 2 of 2 | local |

The first G-C run of the final tree failed once: `batched_hybrid_decode_is_bitwise_the_solo_step` asserted that the batched plan's `MoeExperts`
picks contain a rows kernel, and with the fused block (3 to 4 rows) there are no `MoeExperts` calls. The numerical comparison in that test had not
failed; the pick check was updated to accept `moe_block_bf16_rows` and the test re-run (10 of 10).

## 6. Things found on the way

* **The tests run without admission; `serve` does not.** The first `serve` of the branch failed at load with `no kernel for kv_append ... [admitted
  kernels only]` while every G-B and G-C test was green: a row-table twin takes the admission of the kernel it twins
  (`Registry::effective_admission`), and on HIP the plan never selects the base (the solo step runs the fused op), so the base stayed untested.
  Same trap as rb3's first lesson. Fixed in `plan::admit` (queue the base, test it by its sweep). The load line then reported the row-table kernels
  as `REJECTED` (the plan-shape admission graded them with calls that carry no table); it now skips them like the sweep does, and the load line has
  no `REJECTED` (39 kernels checked, 636 cases).
* **`INTARSIA_ROW_TABLES=0` failed to load on HIP** once the fusion table had the `GateSplitQkNormRopeKv` entry: the per-sequence fold, expanded to
  its chain, counted `gate_split` B times where the rule wanted once. Fixed (`GateSplit` in `WIDE_OR_PER_ROW_OPS`) with a unit test for both forms.
* **The HIP attention tests iterate every kernel of an op** and so called the row-table attention kernel without a table (`IK_ERR_INVALID_ARG`);
  they skip `row_table` kernels now, as CUDA's do. Found by the full-suite run, not by the targeted ones.
* **A batched step with the fused MoE block changed `plan show` with batching off** (a `moe_block` scratch of 18,432 B instead of 9,216 B): the new
  gate-logit-first path was also taken by solo steps of two rows (prefill tails, `fused_ok` accepts `T * K <= 16`). Restricted to batched passes
  (`batch_rows >= 2` and row-exact); the no-op gate is what caught it.
* **Pre-existing failure on `origin/main`:** `cargo test --release -p intarsia-backend-cuda --test rows_registry` fails two tests on the Mac
  (`ordinary_calls_pick_what_they_picked_before_the_rows_kernels` at `rows_registry.rs:178`, `exact_rows_calls_pick_the_rows_kernels_only` at `:186`).
  This branch has no diff in `crates/intarsia-backend-cuda` or `kernels/cuda`, and `origin/main` has not moved since the branch point, so the
  test binary is main's. The pin tables show why: they still expect `matmul_fp8_gemm_reg` / `matmul_fp8_pair_gemm_reg` where main now selects
  `matmul_fp8_gemm_lt` and no pair kernel (the fp8 cuBLASLt default of #125). Not touched here.

## 7. Decisions and open points

* Row tables, graphs and the fused MoE block at 3 and 4 rows are on by default on HIP with `--max-batch`; nothing changes without `--max-batch`
  (section 5, no-op). Switches: `INTARSIA_ROW_TABLES=0`, `INTARSIA_BATCH_GRAPHS=0`, `INTARSIA_HIP_DISABLE=moe_block_bf16_rows`.
* Open (1): where the 5 to 20 ms per step between the model loop and `serve` go (section 4). Not specific to HIP, and larger than any kernel lever left.
* Open (2): a bitwise multi-row fp8 GEMV (lever (d)); the fp8 GEMV is 25.8% of the B = 16 step.
* Open (3): device sampling on HIP (`topk_rows`), opt-in per the user decision; the greedy matrix does not use it.
* Open (4): the `rows_registry` pin tests fail on main (see section 6); a one-line fix of the expected tables, owner lane-lt.

## 8. Reproduce

```
# G-B (tatooine, fleetq job with --takes-down radiance, dev container, HIP card 0):
INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test row_tables --test moe_block --test rows_exact --test rows_registry -- --test-threads=1 --nocapture
# G-C:
INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=/models/Qwen3.6-35B-A3B-NVFP4 \
  cargo test --release -p intarsia-models --test batch_recurrent --test batch_cuda_real -- --test-threads=1 --nocapture
#   long prompts: ... --test batch_cuda_real -- --ignored real_35b_batched_engine_long_prompts_equal_the_local_engine
#   model-loop timing: INTARSIA_TIMING_ROWS=2,3,4,5,6,8,12,16 ... -- --ignored real_35b_batch_step_timing   (profile: real_35b_batch_step_profile)
# speed: intarsia serve --backend hip --model /models/Qwen3.6-35B-A3B-NVFP4 --max-seq 8192 --served-model-name intarsia --max-batch 16
#        uv run --project /opt/code/deepeval localeval speed --url http://tatooine:8294/v1 --model intarsia --thinking off --sizes 1024 --gen 128 --reps 3 --concurrency 4,16
# A/B switches: INTARSIA_ROW_TABLES=0  INTARSIA_BATCH_GRAPHS=0  INTARSIA_HIP_DISABLE=moe_block_bf16_rows
# no-op: scripts/hr3/noop-gen.sh in the dev container of each tree (plan show --json, plan show, 256 greedy ids, --spec mtp --spec-k 2)
```

fleetq jobs: tatooine final windows #4929 (window 7: speed, no-op, G-B, G-C, long prompts, HIP suite) and #4939 (window 8: tiny hybrid after the pick-check fix); spark1 @@CUDAJOB@@; endor @@METALJOB@@. Earlier build and gate windows are in `raw/` and the PR history.
