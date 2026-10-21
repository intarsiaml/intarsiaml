# Core-op fusion, phase 1: P5 + P2a + P4b as three sync-free fused ops (HIP kernels), and the `v3-nofence` stress test, 2026-10-19

> **Update (2026-10-21, docs-accuracy pass, see `docs/decisions/LOG.md`):** Merged to main as PR #22 (`f3cbba21`); "not pushed" / "nothing merged" wording in this page describes the state when written. C2/C1: the gate described here (kill `v3-nofence` first) was not met as written; step 0 killed `v3-noorder` instead and P2' was built on the user's "start now" / "P2 now", then made default-on in PR #28 (`0cea68fc`, kill switch `INTARSIA_HIP_MOE_BLOCK=0`).

Branch `perf/core-op-fusion`, final rebase on origin/main `4962d8fa` (CUDA prefill-4 merged). All GPU numbers were measured
on the branch rebased on `10eef81c` (HIP round 6 merged), against a `10eef81c` base (section 12 has what the last rebase changes). Plan: the proposal
`docs/proposals/2026-10-19-core-op-fusion.md` (user decision "fuse yes", 2026-10-19 01:12 EDT), section 11 steps 0 and 1. Raw
material: `docs/results/2026-10-19-core-op-fusion-p1/raw/`. Every number below is measured unless marked "derived" or "not
measured".

## Headline

* **Qwen3.6-35B-A3B NVFP4 decode on one R9700: 145.6 -> 149.0 tok/s at 128 tokens (+2.3 %), +2.2 to +2.4 % at every prompt
  size up to 16384.** This is an interleaved A/B over 3 rounds against `10eef81c`, in the same window. The plan has **60
  fewer launches per decode step: 475 -> 415.**
  - **Accuracy:** each fold is bitwise the two launches it replaces (device tests). End to end, the 256 greedy ids on a
    4408-token prompt match base, and every printed KL21 digit equals base (35b-short, 35b-long, 8B). HIP logits were not
    compared directly.
  - **No regressions:** prefill is unchanged, and the 8B is unchanged (its weights are bf16, so no fold applies).
* **New ops:** three `Op`s, each tried first, with the unfused chain as fallback.
  - `RmsNormAddMatmulFp8Pair` (P5): the attention input norm folded into the k / v projections.
  - `SigmoidMulMatmulFp8` (P4b): the output gate folded into o_proj.
  - `SharedDownCombine` (P2a): the shared-expert down projection plus `MoeCombine`.
  - Each has a CPU oracle, admission cases and a HIP kernel. The HIP kernel is bitwise the two launches it replaces, on device:
    86 shapes, and 12 of 12 mutants killed.
  - **CUDA and Metal register none of the three ops,** so they resolve to their existing path. Tests pin that, and the CUDA
    no-op run shows it (section 9).
* **`v3-nofence` is NOT killed.**
  - The new sentinel stress test kills the other v3 mutants, including `v3-early` (80 of 960 launches differ) and
    `v3-noreset` / `v3-nowait` (640 of 960). On the real tree it passes 960 + 4800 launches.
  - `v3-nofence` passes it too, on both bases.
  - The gfx1201 ISA explains why: the mutant drops only the per-wave `global_inv scope:SCOPE_DEV`. The `__syncthreads()` right
    before it still emits `global_inv scope:SCOPE_SE` in every wave. So the mutant is very likely **equivalent on this ISA**,
    and no test can kill it here.
  - **Verdict: the bar of proposal step 0 ("mutant killed on hardware") is not met. v3 stays opt-in
    (`INTARSIA_HIP_MOE_V3=1`).** Section 6 covers this.

## 1. Results vs base (HIP, tatooine R9700 gfx1201, W3, interleaved A1 B1 A2 B2 A3 B3)

35B (`raw/bench/base-pf-35b-A*.json` vs `raw/bench/pf-35b-B*.json`; per round, then the mean; serve bench with graphs, 5 decode reps,
errors 0, eager steps 0):

| prompt | base prefill tok/s | P1 prefill tok/s | delta | base decode tok/s | P1 decode tok/s | delta |
|---|---|---|---|---|---|---|
| 128 | 1941 / 1958 / 1934 (1944) | 1944 / 1947 / 1946 (1946) | +0.07 % | 145.5 / 145.7 / 145.6 (145.6) | 149.0 / 149.0 / 149.0 (149.0) | **+2.32 %** |
| 1024 | 7094 / 7102 / 7089 (7095) | 6392 / 7118 / 7104 (6871) | -3.16 % (one outlier, B1) | 144.8 / 145.0 / 144.8 (144.9) | 148.1 / 148.3 / 148.2 (148.2) | **+2.29 %** |
| 4096 | 7976 / 8644 / 8023 (8214) | 8644 / 7960 / 8660 (8421) | +2.52 % (bimodal on both sides) | 140.4 / 140.9 / 140.5 (140.6) | 144.0 / 144.1 / 143.8 (144.0) | **+2.39 %** |
| 16384 | 7229 / 7216 / 7238 (7227) | 7199 / 7236 / 7201 (7212) | -0.21 % | 128.1 / 128.2 / 128.0 (128.1) | 130.9 / 130.8 / 131.0 (130.9) | **+2.20 %** |

Prefill chunks above 8 rows run no fold. The fp8 folds take one row and the nvfp4 fold 1 to 8 rows, so only a short prefill
tail chunk can select one, and it is bitwise anyway. The prefill rows are noise.
- **1024:** the delta is one B1 outlier (6392); B2 and B3 sit with base.
- **4096:** both sides alternate between ~7970 and ~8650.

8B (`raw/bench/base-pf8x-A*` vs `pf8x-B*`): prefill +0.40 / -0.22 / +0.13 % at 1024 / 4160 / 16384, decode +0.03 / +0.00 / +0.06 %.
This is a no-op, as expected: the 8B has bf16 projections, and its plan is identical (section 7).

**Launches per decode step** (`plan show --json`, sum of `calls_per_decode_step`, `raw/plans/diff.txt`): **35B 475 -> 415 (-60)**,
exactly the predicted rows:

| op / kernel | base | P1 |
|---|---|---|
| `rmsnorm_add` / `rmsnorm_add_bf16` | 11 | 1 |
| `matmul_fp8_pair` / `matmul_fp8_pair_gemv` | 40 | 30 |
| `rmsnorm_add_matmul_fp8_pair` / `rmsnorm_add_matmul_fp8_pair_gemv` (P5) | - | 10 |
| `sigmoid_mul` / `sigmoid_mul_bf16` | 10 | 0 |
| `matmul_fp8` / `matmul_fp8_gemv_v2` | 20 | 10 |
| `sigmoid_mul_matmul_fp8` / `sigmoid_mul_matmul_fp8_gemv` (P4b) | - | 10 |
| `matmul_nvfp4` / `matmul_nvfp4_gemv_v2` | 41 | 1 |
| `moe_combine` / `moe_combine_bf16` | 40 | 0 |
| `shared_down_combine` / `shared_down_combine_nvfp4_gemv` (P2a) | - | 40 |

The other 22 (op, kernel) rows are identical. **8B: 364 -> 364**, all 14 rows identical.

The proposal derived the gain as about 84 us and +1.2 % (section 11). The measured gain is about 0.16 ms per step and +2.3 %:
6.868 -> 6.711 ms at 128 tokens (derived from the tok/s). So the 60 launches cost about 2.6 us each here, not the calibrated
1.4 us (derived). Not measured: a profile that splits that between launch gaps and the removed intermediate round trips (normed
row, gated row, shared-expert output).

## 2. What was built

**Core (`crates/intarsia-core`)**

* `ops.rs` adds three `Op` variants. They are appended at the end of the enum, so every existing op's admission case seeds are
  unchanged: the sweeps seed from the discriminant. The change also adds names, arity and `OpCall::with_rows`, plus unit tests.
  - `RmsNormAddMatmulFp8Pair`: inputs x, res, norm w, w0, scales0, w1, scales1; outputs normed, sum, out0, out1; attrs eps,
    w_offset.
  - `SigmoidMulMatmulFp8`: inputs a, gate, w, scales; output out.
  - `SharedDownCombine`: inputs x, w (u8), block scales, global scale, routed, logit; output out.
* `quant/ops.rs`: checks and shape facts (n and weight dtype from the first GEMV).
* `quant/cases.rs`: sweep cases with small, odd and 35B shapes, K tails, M up to 40, both act modes and both w_offsets; plan-time
  cases and labels.
* `quant/moe.rs`: P2a call site. `SharedDownCombine` is tried only when the shared-gate logit was already computed (folded into
  the router pair) and the shared down weight is NVFP4. On `false` the existing chain runs in the existing order.
* `model.rs`:
  - P5 (`attn_norm_kv`): tried in `run_layers` for an attention block with a pending residual and fp8 k / v weights. On success
    the residual buffers flip exactly as `norm_in` would, and `attn_project` skips the k / v pair (`kv_done`). The same decision
    applies to the solo and the batched step.
  - P4b: in `attn_out_proj`, tried for a gated block with fp8 o_proj and no TP partition.
  - **The `check_batch_ops` trap:** with batching on, `fused_ok` takes one of the three ops only if a kernel is selectable at
    every batch row count 2..=`batch_rows` (`with_rows` probes, `exact_rows` set like the batched pass). Otherwise the solo step
    falls back too, instead of the plan failing the batch-multiset check. Older fused ops are untouched by this.
* `model/batch.rs`: `kv_done` passthrough.

**CPU oracle (`crates/intarsia-backend-cpu`)**

* `fused_ref.rs`: each op is the composition of the ops it replaces, with every intermediate stored in its tensor's dtype, as
  in the unfused path.
* `fast.rs` and `reference.rs` route the new ops to it.

**HIP**

* `kernels/hip/fp8.hip`: two new staging prologues of the shared fp8 GEMV body.
  - `PRO == 3`: `norm_add_stage`, which is `rmsnorm_kernel<true,true>` statement for statement over the block's 256 threads.
    Block 0 alone stores normed and sum.
  - `PRO == 4`: `sigmoid_mul_chunk`, which is `sigmoid_mul_kernel`'s expression.
  - The launchers `ik_rmsnorm_add_matmul_fp8_pair_gemv` and `ik_sigmoid_mul_matmul_fp8_gemv` copy the pair kernel's default
    shape (R = 1, 8 warps) and `gemv_rows(N, true)`.
  - The existing PRO 0 / 2 instantiations only gained default parameters.
* `kernels/hip/moe_shared.hip` (new file; nothing in `moe.hip` or `nvfp4.hip` changed for it): `shared_down_combine_kernel<M>`,
  M = 1..8.
  - It is `nvfp4_gemv_v2_kernel`'s lane / iteration / fma order with its bf16 store, then `moe_combine_kernel`'s epilogue on
    lane 0.
  - A row is complete inside one warp, so there is no wait.
* `kernels/hip/ik_hip_abi.h`, `src/ffi.rs`, `src/device.rs`: launcher declarations and dispatch.
* `src/quantmoe.rs`: registration.
  - The fp8 folds take `m == 1`, `K * 2 <= 49152` (the LDS row), bf16 out, and only while `INTARSIA_HIP_FP8_Q` and
    `INTARSIA_HIP_FP8_PAIR` are at their defaults (`fp8_staging_default`). With a moved knob, the chain they mirror stages
    differently, so they step aside.
  - The nvfp4 fold takes `1 <= m <= 8`.
* `kernels/hip/moe.hip` (test seams only; default-off, so behaviour is unchanged): `INTARSIA_HIP_MOE_V3_SKEW_ALL` and
  `INTARSIA_HIP_MOE_V3_PREREAD` are packed into the existing `skew` argument (section 6).

**Tests**

* HIP `tests/fusion_p1.rs`: each fold against the two launches it replaces on the same device, bitwise, plus the CPU reference.
  It covers 24 + 14 + 48 shapes (35B shapes, K tails, N not a block multiple, distinct segment scales, M = 1..8) and checks that
  out-of-range shapes are refused.
* HIP `tests/moe_v3.rs`: `v3_sentinel_stress`.
* HIP `src/kernels.rs` pins the fold selection (one row; nine rows refused; f32 out refused). It also pins that the unfused ops
  resolve to exactly the kernels the folds mirror at the 35B decode shapes.
* `intarsia-models/tests/fusion_p1.rs`: the tiny hybrid fixture with fp8 attention projections, on a CPU backend with and without
  the new ops.
  - Logits are bitwise equal over 12 prefill + 6 decode tokens (f32 W8A16, bf16 W8A16, bf16 W8A8).
  - The fused ops replace exactly 4 launches per step, and the backend without them runs none.
  - A mutation that drops the residual-buffer flip fails it.
* CPU `tests/quant_hf.rs`: the MoE block's fused path is bitwise the unfused one on both CPU executors, and runs
  `SharedDownCombine` in place of the matmul and the combine.
* CUDA and Metal `src/kernels.rs`, two tests each:
  - `phase1_folds_are_absent_and_the_unfused_selection_is_pinned`: the three ops are absent, and the selection at 12
    35B / 8B decode shapes is pinned.
  - `kernel_list_is_unchanged_by_core_op_fusion_p1`: the full sorted (name, op) list, 47 CUDA kernels and 27 Metal, is
    pinned. The branch's diff in both crates' `src/` is test code only (`git diff 4962d8fa -- crates/intarsia-backend-{cuda,metal}/src`),
    so this list is main's.

**Scripts**

* `scripts/hip-decode/fusion-p1.sh`: stages `mutbuild`, `mutrun`, `tests`, `stress`, `nofencestress`, `adm`, `idsoff`, `plans`,
  and `r:<remote.sh stage>`.
* `fusion-window.sh`: one radiance-down window with two GPU lanes, restart trap and downtime lines.
* `plan_diff.py` and `ab_table.py`.
* `mutate.py`: 14 new mutants.
* `scripts/cuda-decode/fusion-noop.sh` (vLLM down) and `fusion-noop-up.sh` (vLLM up).

## 3. Tried and dropped

* **v3 stress, first version:** the round-5 seam (only the last gate_up role late) plus a sentinel was not enough. A stale read
  needs the line to be cached before the store lands, so `PREREAD` (the down waves read their h lines before the wait) and
  `SKEW_ALL` (every gate_up role late, 1 to 4 times the skew) were added. Neither kills `v3-nofence` (section 6).
  - **Stop rule:** one stress run plus one 600-round run, then no more variants.
* **Batch-row probe from row 1:** the probe started at row 1. It was dropped for rows 2..=`batch_rows` with `exact_rows`, the
  way the batched pass selects.
* **remote.sh `kernels` stage for admission:** it runs the 35B `--quick`. It was replaced by `fusion-p1.sh adm` (full sweeps, 35B
  and 8B, both trees).
* **First CUDA window** (job 2647): `window.sh` already prefixes `bash`, and the script argument carried a second `bash`. So it
  exited 126 at once, with 3.0 min of vLLM downtime and nothing measured. It was rerun with the path only (section 9).
* **W1 (job 2636)** ran the whole accuracy and mutant set on the pre-round-6 base (`b0109efe`), before round 6 merged
  (`raw/logs/w1-oldbase*.log`). The results were the same as W2's. W2 repeated everything on the rebased tree against `10eef81c`,
  and only W2 / W3 numbers are reported as the result.

## 4. Accuracy (W2, `10eef81c` base vs rebased branch, same window, `raw/logs/w2.log`)

| check | base | P1 |
|---|---|---|
| 35b-short golden (`qwen36-35b-a3b-nvfp4-vllm-top20-meta`), KL21 prefill / decode | 7.417e-3 / 1.172e-2, PASS | 7.417e-3 / 1.172e-2, PASS |
| 35b-long golden, chunk 4096, KL21 prefill / decode | 4.356e-2 (gate FAIL) / 9.273e-3 | 4.356e-2 (gate FAIL) / 9.273e-3 |
| 8B short, KL21 prefill / decode | 1.700e-4 / 3.165e-4 PASS | 1.700e-4 / 3.165e-4 PASS |
| 8B long, KL21 prefill / decode | 3.629e-4 / 3.128e-4 PASS | 3.629e-4 / 3.128e-4 PASS |
| 256 greedy ids after the 4408-token prompt | - | IDS4K IDENTICAL to base |
| same, with the three folds disabled (`INTARSIA_HIP_DISABLE=...`, the kill switch) | - | IDS4K IDENTICAL to base |

Every printed digit is equal, and the ids are identical. Together with the per-kernel bitwise tests (section 5), the bar
"bitwise, or accuracy within base's range" is met. HIP logits were not compared directly.

The 35b-long prefill gate FAIL (KL21 4.356e-2 vs the 3.0e-2 gate) is the same number as base and as round 5. It is a pre-existing
prefill property, untouched here: the folds run in decode only.

## 5. Bitwise tests and mutants (HIP, W2 GPU 1, `raw/logs/w2-gpu1.log`)

* **Tests:** `fusion_p1` (P5 24 cases, P4b 14, P2a 48, all bitwise), `moe_v3` (5 tests, stress included), `fused_decode` (8),
  `decode_v2` (5) and `device_pos` (4) all pass.
* **Phase-1 mutants: 12 / 12 killed by `fusion_p1`.**
  - `p1-n5-sc0`, `-nowrite`, `-round`, `-woff`, `-stage`
  - `p1-g4-sig`, `-gate`
  - `p1-dc-logit0`, `-sground`, `-soround`, `-tail`, `-row`
* **Selection mutants: 2 / 2 killed** by the lib tests on the Mac, rerun on the final tree (`raw/logs/mut-lib.log`).
  - `p1-sel-m2`: the one-row P5 fold also taken for two rows.
  - `p1-sel-dc9`: the P2a fold also taken for nine rows.
  - The same final-tree run kills `v3-default` and `v3-prio`.
* **v3 mutants: 6 / 7 killed.**
  - `v3-noreset`, `v3-early`, `v3-nowait`, `v3-hrow` and `v3-gurow` fail, and the stress test is among the failing tests for
    noreset / early / nowait.
  - `v3-ticket` hangs and is killed by its 45 s timeout (exit 137).
  - **`v3-nofence` passes** (section 6).

## 6. `v3-nofence` (proposal step 0)

**What the test does** (`tests/moe_v3.rs::v3_sentinel_stress`, about 8 s)

* For each of 4 shapes (35B T = 1 and T = 2, two small ones), it runs 120 launches. Before every launch, h is filled with a bf16
  NaN sentinel (0x7fc0).
* Two activations alternate, and the skew cycles 0 / 2 / 6.
* `PREREAD` is on: every down wave loads exactly the h lines it later reads, with the same plain `global_load_b128` as the real
  reads, before the wait. `SKEW_ALL` is on: every gate_up role is late.
* A second v3 instance runs on stream 1 at the same time.
* Every output of both streams is compared bitwise with v2, and NaN outputs are counted.

**Result**

* On the real tree: 960 launches, 0 differ. With 600 rounds (`nofencestress`, 4800 launches): 0 differ.
* **The `v3-nofence` binary also gives 0 / 960 and 0 / 4800, in both W1 and W2.**
* The test is not toothless: it catches `v3-early` (80 / 960) and `v3-noreset` / `v3-nowait` (640 / 960).

**Why** (`raw/isa/`: `hipcc -O3 --offload-arch=gfx1201 -S` of `moe_experts_p1_v3_kernel<4>` with and without the fence)

* The mutant's only ISA difference is 2 removed instructions after the barrier: `s_wait_loadcnt 0x0; global_inv scope:SCOPE_DEV`
  (`v3k4.diff`).
* The remaining sequence in the mutant is `s_barrier_signal -1; s_barrier_wait -1; global_inv scope:SCOPE_SE`. The
  `__syncthreads()` emits a workgroup-scope acquire, an SE-scope invalidate, in every wave. Thread 0's spin loop also does
  `global_inv scope:SCOPE_DEV` after each acquire load.
* So every wave still drops its stale near-cache lines before the h reads. On a single-die gfx1201, with a device-coherent L2,
  the extra DEV-scope invalidate is very likely redundant for this access pattern. This is inferred from the ISA and the memory
  model, not from a hardware document.

**Verdict:** the bar is not met; v3 stays opt-in. The mutant is very likely equivalent on this target, so killing it would need
a different mutation (one that also removes the barrier's acquire), or a part where SE scope is not enough. The fence stays in
the kernel: it is what the HIP memory model requires, independent of what gfx1201 happens to emit. The stress test and the two
seams remain as a regression test (default off; `INTARSIA_HIP_V3_STRESS_ROUNDS`, default 120 rounds per shape).

**Effect on proposal step 3 (P2'):** P2' is gated on step 0 ("mutant killed on hardware"). That gate cannot be passed as written,
so the decision belongs to the user. Options:
- (a) Accept "equivalent on this ISA, with the ISA evidence" as passing.
- (b) Define a stronger mutant (fence plus barrier acquire removed).
- (c) Keep P2' blocked.

## 7. Admission per backend

| backend | model | base kernels / cases | P1 kernels / cases | new rows |
|---|---|---|---|---|
| HIP (W2, full sweep, no `--quick`) | 35B | 46 / 808, 0 FAIL | 49 / 853, 0 FAIL | +3 kernels, +45 cases (15 + 10 + 20), all ADMITTED. P5 and P2a max error 0; P4b 1.9e-9 vs cpu-ref, inside tolerance |
| HIP (W2, full sweep) | 8B | 46 / 821 | 49 / 866 | the same +3 / +45 (the sweep is per kernel) |
| CUDA (`--quick`, vLLM-up slot) | 35B, 8B | 46 / 143, 0 FAIL | 46 / 143, 0 FAIL | none; logs identical apart from timing; 39 ADMITTED / 7 SKIPPED / 0 FAIL per model on both trees (the 7 SKIPPED are pre-existing) |
| Metal | - | not run here (endor) | | registers none of the three ops (pinned by `phase1_folds_are_absent_...`) |

The plans on HIP select the new kernels at the decode shapes only (section 1). The 8B plan is identical to base.

## 8. Memory

Plan `scratch_bytes`: 35B 167905792 -> 167905792, 8B 748800 -> 748800, so unchanged. The folds write into the buffers the unfused
ops used (k / v, o, the MoE output) and need no workspace. The unused intermediates (`MoeScratch::sh_out`, the gated-q buffer)
are still allocated, so nothing was freed either. The same weight tensors are loaded; `plan show` does not report weight bytes, so that is not measured, but no weight is added or repacked.

## 9. CUDA no-op (spark1, `fusion-base` = `10eef81c` export vs `fusion-dev` = this branch)

Two parts, both measured on the same pair of trees, built on spark1 (job 2628, vLLM up).

**vLLM-down window** (job 2652, `scripts/cuda-decode/fusion-noop.sh` = `w-noop.sh fusion-base fusion-dev nospeed`;
log in `raw/logs/cuda-window2.log`):

* **Greedy ids, 256 tokens, 35B and 8B:** B-graph, A-eager and B-eager all match A-graph. That is "IDS ... == A-graph" for all 6
  comparisons.
* **35b-short golden (prefill chunk 256), A and B:** the same KL21, 6.271e-3 prefill and 1.556e-2 decode. PASS on both.
  - These are CUDA's own digits; the HIP digits in section 4 come from a different backend.
* **Self-golden dump compare:** 60 tensors per tree, per-position TSV identical.
* **8B fp32 goldens on B:** short 2.897e-4 / 3.238e-4, long 4.004e-4 / 3.332e-4 (full-vocab KL prefill / decode). PASS.

**vLLM-up slot** (job 2660, `scripts/cuda-decode/fusion-noop-up.sh`, `raw/cuda/`):

* **`plan show --json`, 35B and 8B:** byte-identical between A and B (md5 prefixes `58a6100d2900` and `d1847a362f17`).
  - 35B: 525 launches per decode step on both; 8B: 544 on both.
  - Scratch is unchanged: 83963392 and 748800.
* **`kernels test --backend cuda --quick`:** 46 kernels, 143 cases, 0 FAIL, for the 35B and the 8B, on both trees. The logs are
  identical apart from timing lines.
  - Both trees exit 1 ("kernel admission failed") with the same seven SKIPPED kernels, which ran 0 quick cases on `cuda`:
    `attn_prefill_bf16_flash`, `gdn_delta_rule_bf16_lanes`, `matmul_fp8_gemm`, `matmul_fp8_gemm_f8`, `matmul_fp8_gemm_reg`,
    `moe_experts_nvfp4_mma_bf16` and `moe_experts_w4a16_bf16`. Per model and tree: 39 ADMITTED, 7 SKIPPED, 0 FAIL
    (`raw/cuda/adm/{A,B}-{35b,8b}.log`). That is pre-existing and identical on A and B. (An earlier version of this doc said
    three SKIPPED kernels; the logs show seven.)
  - The full (non-quick) CUDA sweep was not run.

## 10. Every shared file touched, and what changes per backend

| file | CPU | HIP | CUDA | Metal |
|---|---|---|---|---|
| `crates/intarsia-core/src/ops.rs` (3 appended `Op`s, `with_rows`, tests) | new ops known | new ops known | known, unregistered | known, unregistered |
| `crates/intarsia-core/src/quant/ops.rs` (checks, shape facts) | used by the oracle | used by selection | unused (no kernel) | unused |
| `crates/intarsia-core/src/quant/cases.rs` (sweep / plan-time cases) | sweep runs for its fused kernels | +45 admission cases | none: sweeps run per registered kernel | none |
| `crates/intarsia-core/src/quant/moe.rs` (P2a call site) | fused path taken (its oracle) | fused path at decode | `try_fused` -> false -> the existing chain, same order | MoE not on Metal; unchanged |
| `crates/intarsia-core/src/model.rs` (P5 / P4b call sites, `fused_ok` batch probe) | fused path taken | fused at M = 1 with fp8 k / v / o | false -> unchanged (pinned) | false; the batch probe only runs for the 3 new ops, after `select` already failed |
| `crates/intarsia-core/src/model/batch.rs` (`kv_done` passthrough) | as solo | HIP has no batched step | unchanged | `kv_done` always false (bf16 / no fp8 k, v); unchanged |
| `crates/intarsia-backend-cpu/src/{fused_ref,fast,reference}.rs` | the oracle | - | - | - |
| `crates/intarsia-backend-cuda/src/kernels.rs` (test only) | - | - | coverage test exempts the 3 ops; pin test | - |
| `crates/intarsia-backend-metal/src/kernels.rs` (test only) | - | - | - | pin test |
| `crates/intarsia-backend-cpu/tests/quant_hf.rs` (fused MoE block test) | runs on every host | - | - | - |
| `crates/intarsia-models/tests/fusion_p1.rs` (new; CPU backends only) | runs in every `cargo test --workspace`, Metal's and CUDA's hosts included | - | - | - |
| HIP-only: `kernels/hip/{fp8.hip,moe_shared.hip,moe.hip,ik_hip_abi.h}`, `crates/intarsia-backend-hip/src/{ffi,device,quantmoe,kernels}.rs`, `tests/{fusion_p1,moe_v3}.rs` | - | kernels, registration, seams, tests | - | - |
| scripts: `scripts/hip-decode/{fusion-p1.sh,fusion-window.sh,plan_diff.py,ab_table.py,mutate.py}`, `scripts/cuda-decode/{fusion-noop.sh,fusion-noop-up.sh}` | - | measurement only | measurement only | - |

## 11. Metal no-op check to run (Metal spike session, endor)

Base origin/main `4962d8fa` (the branch's merge-base) vs this branch's HEAD (section 15).

**Getting the branch:** `perf/core-op-fusion` is local and unpushed. Someone pushes it after review, or the Metal session takes a
bundle: `git bundle create core-op-fusion-p1.bundle 4962d8fa..perf/core-op-fusion` on this Mac, then
`git fetch core-op-fusion-p1.bundle perf/core-op-fusion` on endor.


1. `cargo test --workspace` on endor. It includes `intarsia-models` `batch_metal`, `fusion_p1`, and the Metal
   `phase1_folds_are_absent_and_the_unfused_selection_is_pinned` and `kernel_list_is_unchanged_by_core_op_fusion_p1`.
2. `intarsia plan show --backend metal --model <Qwen3-8B> --json` on both trees. The JSON must be byte-identical. The 35B NVFP4
   checkpoint does not load on Metal before or after, since there are no fp8 / nvfp4 / MoE kernels.
3. `intarsia kernels test --backend metal --model <Qwen3-8B>`: kernel and case counts equal on both trees.
4. 8B greedy ids (256 tokens) and the two 8B fp32 goldens (`acc8`) on both trees: identical ids, identical KL digits.
5. With `--max-batch 8` (batched decode): the batch identity test. `cargo test -p intarsia-models --test batch_metal` covers it.

## 12. Caveats and env vars

* **Kill switch:** `INTARSIA_HIP_DISABLE=rmsnorm_add_matmul_fp8_pair_gemv,sigmoid_mul_matmul_fp8_gemv,shared_down_combine_nvfp4_gemv`
  (measured: ids identical with it).
* `INTARSIA_NO_FUSION=1` disables them along with every other fused op.
* **Staging knobs:** moving `INTARSIA_HIP_FP8_Q` or `INTARSIA_HIP_FP8_PAIR` off their defaults silently disables the two fp8
  folds (by design, see section 2).
* **TP:** the folds are off under TP > 1 (`fused_ok`), and P4b is not tried with a TP partition.
* **Batched decode on HIP:** not exercised end to end. HIP has no row-exact kernels, so no batched step. The CPU model test covers
  the solo path; the `with_rows` probe is unit-tested only.
* **`moe.hip`:** the seam edit changed v3's codegen (the PREREAD branch). Round-5 v3 perf numbers no longer apply to this build's
  v3. v3 is opt-in and was not re-benched.
* **Gain vs proposal:** the +2.3 % exceeds the proposal's derived +1.2 %. The per-launch price the proposal calibrated does not
  carry over to this set, and no profile was taken (not measured).
* **Last rebase, not re-measured on a GPU:** `10eef81c` -> `4962d8fa` brings only CUDA code (`kernels/cuda/gdn.cu`, the CUDA
  crate's GDN split for prefill) and docs / scripts, with no HIP or core file. So every HIP number carries over to the final tree
  unchanged. The CUDA no-op ran against `10eef81c`; on the final tree only the Mac checks ran. They pass: `cargo test --workspace`
  is 100 binaries, 484 passed, 0 failed, with the CUDA and Metal pin tests included (`raw/logs/mac-checks-final.log`).
* **Speculative verify on HIP with the 35B:** not exercised. P2a admits 1 to 8 rows, so a verify step selects it (the verify
  path sets `exact_rows`, and the kernel is deterministic and row-independent). The fp8 folds step aside above one row.
* **Post-measurement edit:** in `fp8.hip`'s `norm_add_stage`, the parameter `woff` was renamed `w_off`. Its multiply line was
  textually identical to `gated_norm_chunk`'s, which broke the older `gn-woff` mutant's anchor (2 matches). After the rename,
  the gfx1201 ISA of `fp8.hip` is identical to the measured one apart from the per-file `__hip_cuid_` symbol (736197 lines,
  CPU-slot job 2665/2666). `gn-woff` and `p1-n5-woff` apply and build again.
* **Mutant anchors:** on the final tree all 105 `mutate.py` mutations apply except 4 pre-existing round-6 ones (`moe-z-step`,
  `pairs-sred`, `pairs-xrow`, `pairs-out` in `moe_wmma.h`). Those match twice on origin/main already and are not touched here.
  - **Rerun on GPU in this task:** the 12 p1 mutants and the 7 v3 mutants.
  - **Rerun on the Mac:** the 4 lib mutants.
  - **Not rerun:** the older sets (fp8, gdn, flash, qkr, and so on), whose kernels this task did not change. The exception is
    `fp8.hip`'s shared GEMV body, which gained the two prologues; its existing users are covered by `fused_decode` and `decode_v2` in
    the W2 test run, and by the identical ids and KL digits. The fp8 suites (`fp8_f8`, `fp8_pair_shapes`) were not rerun.
* **Round-6 defaults** are in both A and B: `INTARSIA_HIP_FLASH_HPB=4`, `INTARSIA_HIP_MOE_MW=5`.

## 13. Next phases (not done here, proposal sections 7, 5, 6, 3)

1. **P4a `GateSplitQkNormRopeKv`** (-10 launches): needs the batch-accounting decision for the per-row op.
2. **P3 `GdnConvDelta`** (-30): needs a design note on the conv-state write hazard before code.
3. **P2' `MoeBlock`** (-160, subsumes P1 / P2b): gated on step 0. See section 6 for why that gate is not passed and the three
   options.
4. **P1 alone:** do not build (proposal section 11).

## 14. Downtime and slots

| service | window | job | stop -> serving | minutes |
|---|---|---|---|---|
| radiance (tatooine) | W1: admission, KL, ids, tests, mutants on the old base | 2636 | 02:45:28 -> 02:53:51 | 8.4 |
| radiance | W2: the same on the rebased tree vs `10eef81c`, plans | 2639 | 03:05:58 -> 03:14:35 | 8.6 |
| radiance | W3: A/B x3, 35B and 8B | 2646 | 03:15:58 -> 03:30:35 | 14.6 |
| **radiance total** | | | | **31.6 of 45** |
| vLLM guidance (spark1) | CUDA window 1 (failed at once, exit 126) | 2647 | 03:29:05 -> 03:32:08 | 3.0 |
| vLLM guidance | CUDA window 2: ids, goldens, dump compare | 2652 | 03:32:22 -> 03:43:44 | 11.3 |
| **vLLM total** | | | | **14.3** (one 11.3-min window that measured, plus the 3.0-min failed start) |

Builds and mutant builds ran in CPU-only tatooine slots (jobs 2627, 2635, 2637), and CUDA builds in a vLLM-up spark1 slot (2628).
Every window was released only after `/v1/models` answered.

## 15. Final HEAD

The final commit of `perf/core-op-fusion` (not pushed) is the commit that adds this line. Code tree `scripts/tree-stamp.sh`
`code-tree=20fb0fe6ea1b` (docs excluded); the code is the same as in `d6d20d3`.

* **Measured tree:** the same commits on `10eef81c` (`60d19bfe`). Synced to tatooine `~/intarsiaml-fusion` and spark1
  `~/intarsia-decfuse/fusion-dev`.
* **Mac checks on both trees:**
  - `cargo test --workspace`
  - `cargo check --workspace`
  - `cargo check -p intarsia-cli --features hip`
  - `cargo check -p intarsia-cli --features metal`
  - `cargo build -p intarsia-backend-metal` (it builds on this Mac)
  - Logs: `raw/logs/mac-checks.log` and `mac-checks-final.log`.
