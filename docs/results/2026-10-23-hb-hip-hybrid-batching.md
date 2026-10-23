# Hybrid continuous batching on HIP (Qwen3.6-35B-A3B NVFP4, AMD Radeon AI PRO R9700), lane hb, 2026-10-23

**Status: R0, R1 and R2 gated on HIP; speed measured.** Draft PR, not merge. Branch `lane-hb/hip-hybrid-batching`, built on #117
(`lane-rb2/recurrent-batching-cuda`, now merged) with `origin/main` 240e9d2a merged in. Mirrors
[`2026-10-23-rb2-cuda-batching.md`](2026-10-23-rb2-cuda-batching.md); read that for the design (G-B, the kernel-pick comparison in
`check_batch_ops`, the fusion-equivalence table, recurrent state in the memory plan). Every number is **measured** with a
fleetq job id unless marked otherwise. Box: tatooine, one R9700 (gfx1201, 32 GB) per run, radiance down for the windows
(taken down with `--takes-down radiance`, restored, jobs released).

## 0. Summary

| | before (main 240e9d2a) | now, `serve --backend hip --max-batch 16` | note |
|---|---|---|---|
| tok/s, 1 stream (localeval, 1024 prompt, 128 gen) | 158.2 | **157.8** | not slower (section 6) |
| aggregate tok/s, 4 streams | 134.7 | **195.2** (x1.45) | `--max-batch 4` gives 197.4 |
| aggregate tok/s, 16 streams | 135.0 | **234.7** (x1.74) | `--max-batch 8` gives 225.6, `--max-batch 4` 199.9 |
| G-B (row r of M rows == row 0 of the 1-row kernel, bitwise) | n/a | **pass, M = 1..16**, 5 kernels | section 1 |
| G-C, 35B on HIP, N = 2..16 (every N), 130 steps | n/a | **17,550 (row, step) pairs bitwise identical** | section 3 |
| batched engine vs `LocalEngine` | n/a | **token-identical**, 4 and 16 slots, 20 requests; 2.6k / 4k / 16k-token prompts beside a decoder | section 3 |
| batching off: `plan show` (JSON, text) and 256 greedy ids vs main | | **byte-identical**, also with `--spec mtp --spec-k 2` | section 5 |

The cap is 16 (`Backend::batch_row_cap` of HIP): G-B is proven to 16, VRAM is not binding (section 7), and the model-loop step
saturates only beyond 8 (+5% aggregate from 8 to 16 rows). The 4-stream speed is the same for every cap (195 to 197): the
8-row and 4-row caps lose only when more streams than slots are queued.

## 1. R0: G-B on HIP

`crates/intarsia-backend-hip/tests/rows_exact.rs` (needs a GPU; `cfg(ik_hip)`), run with `INTARSIA_REQUIRE_GPU=1`. Row r of an
M-row call of the `rows_exact` kernel is compared as bytes with row 0 of the **ordinary 1-row kernel the solo decode step
runs**, on row r's input alone, for every M = 1..16 and every row r (inputs: normal, all-zero, duplicate of row 0, a 40-binade
spread, saturating and exactly-on-the-clip e4m3 inputs, one huge element, tiny magnitudes; weights also with a wide scale
range; the rows of one call also in reversed order). No tolerance. Run twice in #4592 (once before the nvfp4 regrouping of this section, once after; the kernels are unchanged since): all pass.

| op | solo (1-row) kernel | `rows_exact` kernel | cases per M (rows checked at M = 16) | result |
|---|---|---|---|---|
| fp8 GEMV, W8A8 and W8A16, bf16 and f32 out | `matmul_fp8_gemv_v2` | `matmul_fp8_gemv_rows` | 132 (2,112) | **bitwise, M = 1..16** |
| nvfp4 GEMV, bf16 and f32 out | `matmul_nvfp4_gemv_v2` | `matmul_nvfp4_gemv_rows` | 32 (512) | **bitwise, M = 1..16** |
| bf16 matmul | `matmul_bf16_gemv` | `matmul_bf16_gemv_rows` | 22 (352) | **bitwise, M = 1..16** |
| logits GEMV (f32 out, 248,320 x 2,048 and 64) | `logits_bf16_gemv` | `logits_bf16_gemv_rows` | 21 (336) | **bitwise, M = 1..16** |
| MoE experts, T tokens, top-8 and top-4, distinct / shared / invalid ids / zero weights | `moe_experts_bf16_v2` at T = 1 | `moe_experts_bf16_rows` | 16 (256) | **bitwise, T = 1..16** |
| row-parallel: `rmsnorm`, `rmsnorm_add`, `silu_glu`, `sigmoid_mul`, `gated_rmsnorm`, `add`, `embedding`, `argmax_f32`, `moe_topk` (v1 and v2), `moe_combine` | the ordinary kernel | the ordinary kernel (no `_rows`) | 1 each at M = 16 (4 for top-k) | **each of 16 rows == that row alone** |

Shapes: the 35B's (GDN in_qkv 8192 x 2048, q / z 4096 x 2048, k / v 512 x 2048, o_proj 2048 x 4096, router 256 x 2048, GDN
a / b 32 x 2048, shared gate 1 x 2048, the head 248,320 x 2048, the MoE layer with 64 of its 256 experts, hidden 2048,
intermediate 512, top-8), Qwen3-8B's (4096 / 1024 / 12288), and edges (K = 16, K with a ragged tail, K % 8 != 0 scalar path,
K past one staging chunk at M = 16, N not a multiple of the 8 warps).

**Negative controls** (the gate can fail): `INTARSIA_HIP_GB_CONTROL=<rows kernel>=<ordinary kernel>` runs the ordinary kernel
where the driver would run the `_rows` one (#4592, `raw/hb-neg3.log`). Each fails at once:

| control | first failure |
|---|---|
| `matmul_fp8_mma_f8` (WMMA) for fp8 | M = 1, row 0 differs from `matmul_fp8_gemv_v2` in 1 of 8,192 elements |
| `matmul_nvfp4_mma` | M = 1, 375 of 512 elements |
| `matmul_bf16` / `logits_bf16` (WMMA) | M = 1, 227 of 256 elements (logits) |
| `moe_experts_bf16_v2` at T = 16 (its tensor-core path starts at 16 tokens) | T = 16, token 3 differs from the T = 1 call in 57 of 2,048 elements |

This is the Fable review's warning (HIP WMMA paths are M-dependent from 16 tokens, so not in batched decode), measured: the
`_rows` kernels never take a WMMA path; the ordinary registrations keep them for prefill and the verify pass of other callers.

### Which kernels needed a `_rows` variant

All five GEMV / MoE kernels got one; none needed new device arithmetic (each is under an hour of work). The 1-row kernels were
already row-independent in their arithmetic (a lane's walk over K and one warp sum per output do not depend on M); what was
M-dependent was the **dispatch**:

* `matmul_fp8_gemv_rows`: the default v2 kernel (2 rows per warp, staging Q = 2) instantiated for M up to 12; M = 13..16
  (and larger K, where 12 x K x 2 bytes exceeds the 48 KiB of staged activations) go in near-equal groups, one launch per
  group. The ordinary `matmul_fp8_gemv_v2` stops at M = 4 and `matmul_fp8_mma_f8` (WMMA) is chosen above. Registered only at
  the default staging (`INTARSIA_HIP_FP8_Q` moved: the plan refuses to load batched rather than run another arithmetic).
* `matmul_nvfp4_gemv_rows`: v2 takes 1..8 rows; `matmul_nvfp4_mma` is chosen above 8. Rows go in near-equal groups of at
  most 4. Timing is why: the M = 8 instantiation is 4x slower than M = 4 (head: 410 us at M = 1, 769 us at M = 4, 3,272 us at
  M = 8), so two groups of 4 (1,543 us) beat one of 8, and 16 rows are 4 launches (3,093 us, against 6,457 us as 2 x 8).
* `matmul_bf16_gemv_rows`, `logits_bf16_gemv_rows`: the existing templated GEMV instantiated for M = 1..16 (the ordinary
  launcher refuses M > 4 and the WMMA GEMM is chosen above).
* `moe_experts_bf16_rows`: `moe_experts_bf16_v2` without its tensor-core path (guarded by a parameter, T <= 16 enforced).

Micro-timings (`rows_exact -- --ignored timing`, 2048 x 4096 fp8, us): M = 1 18.9, 4 22.5, 8 44.8, 16 90.5; MoE (64 experts,
top-8, `_rows`) T = 2 60, 4 144, 8 220, 16 330, within 10% of the ordinary v2 kernel where it exists (`raw/hb-time2.log`, `raw/hb-gb1-first-run.log`; the fp8 time grows linearly past M = 4, the GEMV is ALU-bound
there, which is the R3 work, not this lane).

Row-exact selection is pinned by `tests/rows_registry.rs` (registry only, no GPU): ordinary calls pick exactly what they
picked before (`ORDINARY_PINS` generated on #117's head) and `exact_rows` calls pick the `_rows` kernel for M = 1..16, none
above 16.

## 2. R1: plumbing for HIP

* `Backend::batch_row_cap` = 16 (`crate::kernels::ROWS_MAX`); `Model::allow_recurrent_batch` and the loader accept `hip`;
  the loader's `hybrid_batch` accepts `hip` as it does `cuda`, so `--max-batch N` batches the hybrid model on HIP with no extra flag.
* The busy prefill chunk defaults to the solo prefill chunk for hybrid MoE on HIP as on CUDA (4096 for the 35B on HIP,
  decision row 2026-10-23 12:55). The long-prompt engine test passes with it (section 3). Control: busy chunk 1024 is **also**
  identical on those four prompts on HIP, so the HIP prefill GEMMs are not chunk-sensitive on them; the default stays the solo
  chunk because that is bitwise by construction, not by measurement. A 4096-token chunk stalls the decoders for one chunk
  (not measured here: the benchmark's prompts are 1024 tokens).
* Fusion-equivalence table (`crates/intarsia-core/src/fusion.rs`): the HIP entries are `MatmulPair`, `MatmulFp8Pair`,
  `MatmulNvfp4Glu`, `GatedRmsnorm`, `GatedRmsnormMatmulFp8`, `RmsNormAddMatmulPair` (tests: `fused_decode.rs`),
  `RmsNormAddMatmulFp8Pair`, `SigmoidMulMatmulFp8`, `SharedDownCombine` (`fusion_p1.rs`) and `MoeBlock` (`moe_block.rs`,
  bitwise the chain topk, experts, glu, shared-down-combine). `Model::fused_ok` skips its "kernel selectable at every row
  count" probe for an op that has an entry (the batched step runs the proven unfused chain). Without the entries the solo
  step lost `moe_block` (it serves T x top_k <= 16, i.e. T <= 2), the two fp8 folds and `SharedDownCombine` (<= 8 rows) as soon
  as batching was on, and a single stream paid 10% (section 6, "folds dropped").
* `INTARSIA_HIP_GB_CONTROL` (test-only) and `INTARSIA_BATCH_TEST_BACKEND=hip` are documented in `docs/tuning.md`.

## 3. R2: bring-up gates (G-C)

All on HIP, bf16 activations and KV, `INTARSIA_REQUIRE_GPU=1`.

| gate | result | fleetq |
|---|---|---|
| tiny hybrid (`batch_recurrent.rs`, `INTARSIA_BATCH_TEST_BACKEND=hip`): `forward_batch` N = 2, 3, 4, 8, 16 x 130 steps on three variants; batched engine vs single-sequence engine; decode beside prefill chunks; prefix reuse; memory plan | 7 of 7 pass; the batched plan really runs the `_rows` kernels (`check_cuda_batch_picks` asserts the picks on any GPU backend) | #4592, #4620 |
| **35B NVFP4**, `forward_batch`, mixed context lengths 3 to 400, greedy and seeded-sampled rows, **every N = 2..16**, 130 steps each | **17,550 (row, step) pairs bitwise identical** (logits bit for bit, argmax token for token, state identical), 205 s; decode beside 256-token prefill chunks bitwise, prefilled state equals the prompt prefilled alone | #4592 (before the fold fix), #4620 (final tree) |
| **35B engine**: 20 requests (6 seeded-sampled), 130 tokens, `LocalEngine` alone vs `BatchedEngine` with 4 and 16 slots, one stream at a time and all at once (queueing, slot reuse) | token- and text-identical | #4592, #4620 |
| **35B engine, long prompts** (`real_35b_batched_engine_long_prompts_equal_the_local_engine`, `--ignored`): 4,096, 2,600 and 16,384 tokens (sampled) prefilled beside a decoding neighbour, 2 and 4 slots, three submission orders | **IDENTICAL to the `LocalEngine` run** for every request; control busy chunk 1024: identical too | #4592, #4620 |
| Qwen3-8B dense on HIP (same gate) | **not run to completion**: the test did not finish in 8 minutes (188% CPU, no output after the memory plan) and was killed by PID. Not the lane's target; open | #4592 |

Model-loop step times (eager, no graphs, `real_35b_batch_step_timing`, #4620): B = 2 9.6 ms (209 tok/s aggregate), 4 14.4 ms
(278), 8 24.2 ms (331), 16 46.0 ms (348). (The "solo step 57 ms" line of that test is its cold first step; not a figure.)

## 4. Other tests

* `cargo test --release -p intarsia-backend-hip --tests` on the final tree (#4592): every test file passes (admission with
  the new kernels, `quick_coverage`, `moe_block`, `fused_decode`, `fusion_p1`, `fusion_p2`, `decode_v2`, `rows_exact`,
  `rows_registry`, ...; `raw/hb-final.log`).
* Mac: `cargo test --no-fail-fast -p intarsia-core -p intarsia-models -p intarsia-cli -p intarsia-backend-hip -p intarsia-server
  -p intarsia-backend-cpu`: 634 passed, 0 failed (the cpu batching tests, `fusions_name_existing_tests`, the docs env-var table
  test). CUDA and Metal no-ops were not rerun: the shared code touched (the fusion table gets HIP entries only, `fused_ok`
  skips a probe only for an op with a table entry on the running backend, the loader matches `hip` next to `cuda`) is inert
  on both.

## 5. No-op gates and MTP

| check | result | fleetq |
|---|---|---|
| HIP, batching off, 35B: `plan show --json`, `plan show`, and 256 greedy ids (`generate --ignore-eos --show-ids`), branch vs `origin/main` 240e9d2a | **byte-identical** (12,961 / 3,184 / 1,036 B) on the first build and on the final tree | #4592, #4620 |
| `--spec mtp --spec-k 2` on HIP, 256 ids, branch vs main, and vs `--spec off` | **identical** to both; acceptance 66.5% (145 of 218) on both; decode 167.6 (branch) vs 163.0 tok/s (main), inside the noise | #4620 |

Registering `rows_exact` kernels on HIP moves the MTP verify pass (M = k + 1) onto them, as on CUDA (rb2 section 5): the ids did
not change here. The load line changes accordingly (main: "no row-exact verify kernels here"; branch: "verify rows run the row-exact kernels").

## 6. Speed (localeval, `intarsia serve` on tatooine card 0, 35B NVFP4, 1024-token prompt, 128 generated, 3 reps)

`uv run localeval speed --url http://tatooine:PORT/v1 --model intarsia --sizes 1024 --gen 128 --reps 3 --concurrency 4,16`
inside the fleetq job. Aggregate decode tok/s (localeval's definition); 1 stream is the sweep row. One run per cell, decode noise
about +-5%.

| server | 1 stream | 4 streams | 16 streams | run ids |
|---|---|---|---|---|
| main 240e9d2a, `serve` (one sequence at a time) | 158.2 | 134.7 | 135.0 | `20261023-135539-adhoc-hb-main-off`, #4592 |
| branch, `--max-batch 4` | 158.0 | 197.4 | 199.9 | `20261023-141405-adhoc-hb2-on4`, #4620 |
| branch, `--max-batch 8` | 157.9 | 195.0 | 225.6 | `20261023-141334-adhoc-hb2-on8`, #4620 |
| branch, `--max-batch 16` | **157.8** | **195.2** | **234.7** | `20261023-141304-adhoc-hb2-on16`, #4620 |
| *(intermediate, folds dropped)* `--max-batch 16` / 8 / 4 | 141.2 / 143.7 / 143.4 | 195.3 / 195.0 / 197.0 | 232.9 / 227.0 / 200.0 | `20261023-140030-adhoc-hb-on16`, `-140205-adhoc-hb-on8`, `-140134-adhoc-hb-on4` (#4592) |

Per-stream decode at 4 streams is 57.6 tok/s (main: one stream at 158 while three wait); TTFT at 16 streams 1.38 s (main 7.25 s).
The baseline column is the engine as it serves today (one sequence at a time: aggregate stays at the single-stream
speed, a little below because prompt prefills interleave).

Where the rest is: on the R9700 the fp8 and nvfp4 multi-row GEMVs are ALU-bound
past 4 rows (section 1) and the step is eager (the batched step runs without graphs; the solo step replays one). At 16 streams
the step is 46 ms (348 tok/s model loop); the served 234.7 includes the prefills of 16 prompts and per-token host work. R3 (K1 / K2 kernels to M = 16)
and R5 (graph buckets) are the levers, as in the rb plan.

## 7. Cap and memory

Cap 16. Evidence: G-B holds to 16; the 35B plan with 16 slots on a 32 GB R9700: weights 20.8 GB, workspace 0.96 GB, recurrent
state 62 MiB per sequence (942 MiB for 16), KV 20,480 B per token (167 MB per 8k-token sequence); the planner's KV budget was
11.9 GB, so VRAM is not what binds (`raw/hb-gc2a-final.log`, "batch plan notes"). Speed: 16 over 8 slots is +4% at 16
streams (234.7 vs 225.6, inside the noise) and the model-loop step goes 331 to 348 tok/s; 8 over 4 slots is +13% at 16
streams. Nothing is lost at 16 for 4 streams.

## 8. Open

* The batched step is eager. HIP supports graphs; per-row launches are a larger share of the step than on CUDA (R5).
* fp8 and nvfp4 multi-row GEMVs are not bandwidth-shaped past 4 rows (R3).
* `moe_block` (T x top_k <= 16) serves only T <= 2 at top-8, so the batched step with 3 or more rows runs the unfused chain
  (proven equivalent, but more launches). A `moe_block` for T <= 16 is the next MoE lever.
* Qwen3-8B dense on HIP: the batching gate did not complete (section 3); needs a look before `--max-batch` is offered for dense
  models on HIP.
* The busy-chunk stall of a 4096-token chunk on HIP is not measured.
* `plan show` has no `--max-batch`; the batched plan is visible only through `GET /v1/intarsia/plan` of a running server.

## 9. Reproduce

```
# G-B (tatooine, radiance down, fleetq job):
INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test rows_exact -- --test-threads=1 --nocapture
INTARSIA_HIP_GB_CONTROL=matmul_fp8_gemv_rows=matmul_fp8_mma_f8 cargo test --release -p intarsia-backend-hip --test rows_exact fp8   # must FAIL
# G-C:
INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=/models/Qwen3.6-35B-A3B-NVFP4 \
  cargo test --release -p intarsia-models --test batch_recurrent --test batch_cuda_real -- --test-threads=1 --nocapture
# (the long-prompt engine test is #[ignore]: add --ignored real_35b_batched_engine_long_prompts_equal_the_local_engine)
# speed: intarsia serve --backend hip --model ... --max-seq 8192 --served-model-name intarsia --max-batch 16
#        uv run --project /opt/code/deepeval localeval speed --url http://tatooine:8294/v1 --model intarsia --sizes 1024 --gen 128 --reps 3 --concurrency 4,16
```

Raw logs: `2026-10-23-hb-hip-hybrid-batching/raw/`. fleetq jobs: #4567 (build), #4592 (G-B, G-C, no-op, first speed), #4619
(rebuild), #4620 (final G-C, no-op, MTP, speed).
