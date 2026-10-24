# hr3: row tables, graphs and the fused MoE block for batched decode on HIP (R3 port)

Lane hr3, 2026-10-23 evening, tatooine card 0 (R9700, gfx1201), Qwen3.6-35B-A3B NVFP4, `serve --max-batch 16`. Ports the CUDA R3 work of
[rb3](2026-10-23-rb3-batching-r3.md) (PR #121) to the HIP batching of [hb](2026-10-23-hb-hip-hybrid-batching.md) (PR #120). Branch
`lane-hr3/hip-r3-batching`, base `origin/main` 6189466f. Machine-readable trace: [trace.json](2026-10-24-hr3-hip-batching-r3/trace.json).

## 1. Result

Every decode row of a batched step is still bitwise the solo step (section 5). 35B aggregate decode, localeval, 1024-token prompt, 128
generated, 3 reps per cell, `serve --max-batch 16` on both sides, same session and same quiet slots (main = `origin/main` 6189466f, built
the same way; three bursts per cell for the 1024-token cells, two for the 4096-token cell):

| streams | main | hr3 | change | bursts (main / hr3) |
|---|---|---|---|---|
| 1 | 158.0 | **158.2** | +0.1% (noise) | 158.2, 158.0, 157.8 / 158.5, 158.2, 157.8 |
| 4 | 195.5 | **205.0** | +4.9% | 196.2, 197.2, 193.0 / 206.5, 204.9, 203.5 |
| 16 | 233.5 | **256.1** | +9.7% | 233.5, 233.6, 233.3 / 255.9, 256.2, 256.3 |
| 4, 4096-token prompts | 138.7 | **144.8** | +4.4% | 138.6, 138.7 / 145.9, 143.7 |

(The lane brief's main figures, 157.8 / 195.2 / 234.7, agree with these within the run-to-run spread.) Single-stream decode does not move
(a solo request runs the solo step; none of the new code is on its path). CUDA got +8 to +14% from row tables alone and +7% from graphs at
4 streams; here the sum is +4.9% at 4 and +9.7% at 16 streams, with graphs worth +2.7% at 4 streams and +3.8% at 4 streams with 4k
prompts (A/B below).

| config (hr3 tree) | 1 | 4 | 16 | 4 at 4k |
|---|---|---|---|---|
| main | 158.0 | 195.5 | 233.5 | 138.7 |
| hr3 | 158.2 | 205.0 | 256.1 | 144.8 |
| hr3, `INTARSIA_BATCH_GRAPHS=0` (eager batched step) | 158.0 | 199.7 | 255.4 | 139.5 |
| hr3, `INTARSIA_ROW_TABLES=0` (one launch per row per op) | @@NOROWS1@@ | @@NOROWS4@@ | @@NOROWS16@@ | @@NOROWS4K@@ |

Prefill is unchanged (6.1 to 6.3k tok/s at 1k, 9.7 to 10.0k at 4k on both sides).

@@STEP@@

## 2. What changed

**(a) Row tables (HIP kernels, `kernels/hip`).** One launch per layer and op for all rows of a step, instead of one per row. The kernels
take a table of device addresses (`u8 [rows, stride, 8]`, the same `OpCall::rows` contract as CUDA's) and the position of each row on the
device (`pos`, `i32 [rows]`):

| kernel (`row_table` in the registry) | twin of | grid |
|---|---|---|
| `gdn_conv_bf16_rows` | `gdn_conv_bf16` (T = 1: conv and state update in one kernel) | `.y` = row |
| `gdn_delta_rule_bf16_v2_rows` | `gdn_delta_rule_bf16_v2` (T = 1), state at table entry + byte offset | `.z` = row |
| `kv_append_bf16_rows` | `kv_append_bf16` | one thread per (row, head, vector) |
| `attn_decode_bf16_splitkv_rows` | `attn_decode_bf16_splitkv` with a device position (the one-row decode grid) | `.z` = row, scratch for `rows` rows |
| `rope_bf16`, position per token | `rope_bf16` | `pos` of numel T > 1: token t at `pos[t]` |

Each is the one-row kernel's code with the row's pointers taken from the table, so row r is the one-row launch on sequence r alone
(`tests/row_tables.rs`, section 5). The solo step runs the fused `gdn_conv_delta_bf16` and `gate_split_qk_norm_rope_kv_bf16`; the batched
step runs the chains of those folds with the row-table kernels, which the fusion table proves equal (`crate::fusion` gets HIP entries for
`GateSplitQkNormRopeKv`, `QkNormRopeKv` and `GdnConvDelta`, each naming its existing bitwise test; `QkNorm`/`Rope` join the row-parallel
ops of the batch-consistency check, `GateSplit` may run once over all rows or once per row). `model/batch.rs` does not use the
per-sequence fold when row tables run (a launch bound to one sequence's cache cannot be in a batched graph).

**(b) HIP graphs per batch size.** No new code: `Backend::supports_graphs` is already true on HIP, and the generic batched-graph path of rb3
(`graph_step.rs`, one graph per row count, row tables rewritten before each replay) engages as soon as the step has no per-row launches
left. The A/B above (`INTARSIA_BATCH_GRAPHS=0`) is the measurement: +2.7% at 4 streams, +0.3% at 16, +3.8% at 4 streams with 4k prompts.
The `moe_block` kernel's sync counters are per (device, stream) and are reset by the kernel itself, so replaying it in a graph is safe
(`moe_block_graph_replay_is_bitwise_eager` is an existing test, passing).

**(c) `moe_block_bf16_rows`.** `moe_block` fuses top-k, routed experts, shared expert and combine in one launch, but only at `T * K <= 16`,
i.e. T <= 2 at top-8: steps of 3 or more rows ran the unfused chain (about 7 launches per layer). Every role of the kernel already works
on one token, so lifting the bound to T <= 16 (`ik_moe_block_bf16_rows`, registered `rows_exact`, same kernel code, only the shape check
differs) keeps each token bitwise the T = 1 launch on it alone: `moe_block_rows_are_the_solo_block_per_token` compares 1,252 tokens over
168 launches (T = 1..16, six shapes including the 35B's top-8 of 256, norm on and off, with routing ties) with T = 1 blocks, plus a negative
control per launch.

**(d) Multi-row fp8/nvfp4 GEMVs: not touched.** hb measured them ALU-bound past 4 rows. @@PROFILE_D@@

**Device sampling** stays opt-in (`INTARSIA_DEVICE_SAMPLING=1`, user decision 2026-10-23 17:57) and is not ported: HIP has no `topk_rows`
(`supports_topk` is false), and the speed matrix is greedy, where it changes nothing. Open follow-up.

**Core changes (shared crates).** `plan.rs`: `admit` also queues the *base* kernel of a row-table twin that no planned call uses and tests it
by its op's sweep (see section 6, the twin inherits its base's admission, and on HIP the base of `kv_append`/`gdn_conv`/`gdn_delta_rule` is
never selected by the solo plan); `WIDE_OR_PER_ROW_OPS` gains `GateSplit`; `ROW_PARALLEL_OPS` gains `QkNorm` and `Rope`; unit test
`batch_check_accepts_hip_folds_per_row_or_as_row_table_chains`. `fusion.rs`: the three HIP entries. `model/batch.rs`: eight lines. All
three are inert on CUDA and Metal (section 5).

## 3. Method

* Speed: `intarsia serve --backend hip --model Qwen3.6-35B-A3B-NVFP4 --max-seq 8192 --max-batch 16` in the dev container on card 0, then
  `localeval speed --url http://tatooine:8294/v1 --model intarsia --thinking off --sizes 1024 --gen 128 --reps 3 --concurrency 4,16` (and
  `--sizes 4096 --concurrency 4`) inside the fleetq job (`--takes-down radiance --quiet gpu,cpu`). Main and branch are built by the same
  command into separate cargo target volumes, served one after the other in the same job, so there is no drift between them. Decode noise
  is about +-2% between bursts. Raw logs: `2026-10-24-hr3-hip-batching-r3/raw/`.
* Radiance was taken down for each window and restored at its end (both ranks `active`, checked with `fleetq status`).

## 4. Why the other numbers are what they are

@@WHY@@

## 5. Gates

@@GATES@@

## 6. Things found on the way

* **The tests run without admission; `serve` does not.** The first `serve` of the branch failed at load with `no kernel for kv_append ...
  [admitted kernels only]` while every G-B and G-C test was green: a row-table twin takes the admission of the kernel it twins
  (`Registry::effective_admission`), and on HIP the plan never selects the base (the solo step runs the fused op), so the base stayed
  untested. Same trap as rb3's first lesson. Fixed in `plan::admit` (queue the base, test it by its sweep); the load line now reports
  `REJECTED: <the _rows kernels>` because the admission sweep skips row-table kernels, as on CUDA; they are admitted through their bases.
* **`INTARSIA_ROW_TABLES=0` failed to load on HIP** once the fusion table had the `GateSplitQkNormRopeKv` entry: the per-sequence fold,
  expanded to its chain, counted `gate_split` B times where the rule wanted once. Fixed (`GateSplit` in `WIDE_OR_PER_ROW_OPS`) with a unit
  test for both forms; the A/B row above is the check that it loads and runs.
* **Pre-existing failure on `origin/main`:** `cargo test -p intarsia-backend-cuda --test rows_registry` fails two tests
  (`ordinary_calls_pick_what_they_picked_before_the_rows_kernels`, `exact_rows_calls_pick_the_rows_kernels_only`) on a Mac with no GPU, at
  6189466f with none of this branch. Not touched here; a task chip was raised.
* The first CUDA no-op job (#4841) compared two copies of the same stale binary (cargo was not on the non-login ssh PATH): it proved nothing and
  is not counted; #4853 is the real one (it checks the binary is newer than the sources).

## 7. Decisions and open points

* Row tables, the fused MoE block at every row count and the batched graphs are on by default on HIP with `--max-batch`; nothing changes
  without `--max-batch` (section 5, no-op).
* Open: device sampling on HIP (needs `topk_rows`); the multi-row GEMVs past 4 rows (@@OPEN_D@@); the `rows_registry` failures on main.

## 8. Reproduce

```
# G-B (tatooine, radiance down, fleetq job; all in the dev container, HIP card 0):
INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test row_tables --test moe_block --test rows_exact --test rows_registry -- --test-threads=1 --nocapture
# G-C:
INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=/models/Qwen3.6-35B-A3B-NVFP4 \
  cargo test --release -p intarsia-models --test batch_recurrent --test batch_cuda_real -- --test-threads=1 --nocapture
#   long prompts: ... --test batch_cuda_real -- --ignored real_35b_batched_engine_long_prompts_equal_the_local_engine
# speed: intarsia serve --backend hip --model /models/Qwen3.6-35B-A3B-NVFP4 --max-seq 8192 --served-model-name intarsia --max-batch 16
#        uv run --project /opt/code/deepeval localeval speed --url http://tatooine:8294/v1 --model intarsia --thinking off --sizes 1024 --gen 128 --reps 3 --concurrency 4,16
# A/B switches: INTARSIA_ROW_TABLES=0  INTARSIA_BATCH_GRAPHS=0  INTARSIA_HIP_MOE_BLOCK=0
```

Raw logs: `2026-10-24-hr3-hip-batching-r3/raw/`. fleetq jobs: @@JOBS@@.
