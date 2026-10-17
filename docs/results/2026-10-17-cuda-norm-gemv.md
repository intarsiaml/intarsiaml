# CUDA norm folded into the GEMV, Qwen3.6-35B-A3B NVFP4 on spark1 (GB10, sm_121), branch `perf/cuda-norm-gemv`

Base: `check/cuda-noop-norm-gemv` = `perf/hip-norm-gemv` `3ad530e` plus scripts/doc ([no-op check](2026-10-17-cuda-noop-norm-gemv.md), which
shows that base is bit-identical to `origin/main` `c00af21` on CUDA). The branch is rebased onto `origin/main` `df18c6a` (PR #5 merged);
the GPU runs below were taken before the rebase, on `perf/hip-norm-gemv` + this branch (trees `~/intarsia-decfuse/{hipnorm,cudanorm}`,
built from `ae79062` before the rebase; the CUDA sources are those of the HEAD of this branch, `dirty=no`). Scripts: `scripts/cuda-decode/`; logs and JSON: `2026-10-17-cuda-norm-gemv/raw/`.

## 1. Result

The two HIP ops now exist as CUDA kernels, registered with admission, and the model uses them without any change (it already asks the
registry, and falls back to the old chain when no kernel serves the call).

| 35B, graphs on, `bench speed` gen 128, 3 reps per round, mean of 3 interleaved rounds | before | after | change |
|---|---|---|---|
| decode 128 tokens (tok/s) | 80.37 | **81.07** | +0.87% |
| decode 1024 | 79.77 | **80.43** | +0.84% |
| decode 4096 | 77.20 | **77.73** | +0.69% |
| prefill 1024 (tok/s) | 3946 | 3935 | -0.3% (noise) |
| prefill 4096 | 4707 | 4631 | -1.6% (one low rep set, below) |
| launches per decode step (nsys, graph nodes) | 724 | **625** (-99) | |
| decode step span (nsys, 156 steady steps) | 12388 us | 12261 us (-127 us, -1.0%) | |
| kernel time per step | 12313 us | 12197 us (-117 us) | |
| idle inside the step | 74.8 us (0.6%) | 64.6 us (0.5%) | -10 us |

The gain is about 1%, as the brief expected: the CUDA graph had almost no gaps, so removing launches gains little, and what is
gained is kernel time (-117 us per step, 0.9%). Per-round numbers (base then dev; rounds base-dev, dev-base, base-dev). Decode 128:
79.9 / 80.4 / 80.8 vs 81.2 / 81.3 / 80.7; 1024: 79.3 / 79.7 / 80.3 vs 80.5 / 80.7 / 80.1; 4096: 76.8 / 77.1 / 77.7 vs 77.8 / 78.0 / 77.4.
Rep-to-rep spread inside a run is about 0.1 tok/s, but the round-to-round spread of one arm is 0.5 to 0.9 tok/s (the base arm drifts up
in round 3: 80.8), so the +0.7 to +0.9% is larger than the within-run noise but of the size of the drift between rounds. Dev is above base in rounds 1 and 2 at
all three sizes and a hair below it in round 3 (80.7 vs 80.8, 80.1 vs 80.3, 77.4 vs 77.7): 6 of 9 pairs. The nsys span change (-1.0%) agrees with the end-to-end figure. I would call it a
real gain of about 0.8%, measured with 3 reps per round (REPS 3 instead of the usual 5, to fit the downtime budget: 9 reps per arm and size).
Prefill is not touched by design (the fused ops are M = 1 only); the 4096 prefill mean is lowered by one dev rep set (4501 in round 1;
4695 and 4698 in rounds 2 and 3, base 4683 / 4749 / 4690), the same single-low-round pattern as in `2026-10-14-cuda-decode-fusion.md`.
`qualify --quick-only` prints 81.1 tok/s at 128 tokens (83% of the byte floor; 80.7 / 82% on 2026-10-14).

## 2. What was fused

| op | kernel (file) | replaces (launches/step on the 35B) | launches saved |
|---|---|---|---|
| `RmsNormAddMatmulPair` | `rmsnorm_add_matmul_pair_bf16_gemv` (`kernels/cuda/matmul.cu`, `rmsnorm_add_pair_kernel`) | `rmsnorm_add` + `matmul_pair_bf16_gemv` before the MoE router + shared-expert gate (40) and before the GDN b + a gates (29) | 69 |
| `GatedRmsnormMatmulFp8` | `gated_rmsnorm_matmul_fp8_gemv` (`kernels/cuda/fp8.cu`, `fp8_gemv_gated_kernel`) | `gated_rmsnorm` + `matmul_fp8_gemv` of the GDN out_proj (30) | 30 |

724 - 99 = 625 launches per step, as nsys shows (`raw/prof-{base,dev}/steps.txt`; 40 + 29 + 30 expected). Layer 0 keeps the unfused norm (no
residual to fold), so 69 not 70 launches; `qualify` reports `calls_per_decode_step` 69 and 30 for the two kernels, both ADMITTED.

Design (the HIP design, adapted to the CUDA kernels):

* **Bitwise by construction.** Each block redoes the add and the norm in the standalone kernel's order (the 8-element chunks per
  thread, the bf16 rounding of the sum before it is squared, `ik_block_sum` over 256 threads: `rmsnorm`'s launch picks 256 threads for any
  K >= 2048, and for a narrower row its smaller block holds the same chunks in the same threads while the extra warps add exact zeros),
  keeps the normalised row in shared memory (K <= 8192, 16 KB) and runs `gemv_body` unchanged on it: the body gained one template flag
  that reads `x` from shared instead of `__ldg`, the arithmetic is the same source. Only block 0 stores `normed` and `sum`; the launcher
  refuses a `normed` / `sum` that overlaps `x` / `res`.
* **Gated norm into the fp8 staging.** `fp8_gemv_body` gained a `PRO` flag: the staging loop produces the gated-norm chunk of 8 values
  (`gated_norm_chunk`, `gated_rmsnorm_kernel`'s expression) and then, for W8A8, the same e4m3 quantisation the elementwise staging does. A
  head is 16 chunks, so the 16 lanes of a half warp hold one head and the xor butterfly from offset 8 sums the same values in the same order as the
  standalone 32-thread block (its first step adds exact zeros, and a + b == b + a bitwise). All 32 lanes call the shuffle (full mask);
  lanes past the last chunk (odd row counts) run on a valid index and drop the result.
* **Registration**: `quantmoe.rs` TABLE + `all()` (priority 10, `Exactness::Deterministic`, `TOL` 1.5e-2 / 1e-3 as HIP), dispatch in
  `device_quantmoe.rs`, declarations in the CUDA `extern` block; `kernels.rs`'s table test no longer lists the ops as HIP-only.
* No prefetch of the first weight loads before the norm prologue (HIP does that). Not done: there was no downtime budget for a measure,
  change, re-measure cycle, and the kernels already shed their launches; a prefetch would only touch loads (bitwise safe) and might add a
  little.

Per-kernel times (nsys, us per step): norm + pair: removed `rmsnorm_add` (69 launches, 146.6 us) and `matmul_pair` (69, about 382 us),
added `rmsnorm_add_pair_kernel` (69, 444.0 us, 6.4 us each): -84 us. Gated: removed `gated_rmsnorm` (30, 56.7 us) and the out_proj GEMV
(30, 1236 us), added `fp8_gemv_gated_kernel` (30, 1241.1 us, 41.4 us each): -52 us. Sum -136 us against -117 us measured on the kernel total
(the other kernels move by a few us between profiles).

## 3. Accuracy and identity

| check | result |
|---|---|
| fused kernels vs the unfused CUDA chain, bitwise (`tests/fused_decode.rs`, GB10) | pass (8 tests in window B1 plus the heavy-tail test in B2): 11 shapes x w_offset for the pair (K 8 to 8192, K 40 / 96 below the 256-thread block, 4104 past it, 520), 10 shapes x W8A8 / W8A16 x w_offset for the gated fp8 (R 1 to 32, N 1 to 8192, odd R), `near_reference` against the cpu kernel |
| many draws | norm + pair: 3000 well-scaled rows + **8000 heavy-tailed rows** (2^U(-s,s) per element, s cycling 2, 4, 6, 8, 10); gated + fp8: **2500 W8A8** + 1000 W8A16 draws of the model's shape; all bitwise |
| `backend-cuda` tests: admission, fused_decode, quantmoe | pass: admission 4, fused_decode 8, quantmoe 10 (3 ignored); `every_cuda_kernel_matches_the_reference` runs the registered kernels |
| `qualify --quick-only` 35B | **PASS 7/7**: 43 kernels, 595 cases, gate ppl 1.22, acc-short prefill 97.92% KL 5.87e-3 / decode 98.12% KL 1.54e-2, bench 81.1 tok/s |
| `accuracy` 35b-short vs vLLM golden, dev | prefill 97.99% KL 6.271e-3, decode 98.12% KL 1.556e-2, PASS: the numbers of base |
| self-golden dumps, dev vs base | **0 of 60 tensors differ**, per-position TSV identical |
| 8B fp32 goldens, dev | 8b-short PASS (100% KL 2.897e-4; 100% KL 3.238e-4), 8b-long PASS (99.65% KL 4.004e-4; 100% KL 3.332e-4): the numbers of base |
| 256 greedy ids, 35B and 8B, dev graph / `--no-graphs` / `INTARSIA_NO_FUSION=1` | all six id lists are byte-identical to the base build's (`raw/b1/idproof.txt`) |

**Graph proof.** nsys of dev: 160 steps of one graph, 625 kernel nodes each. The 256-token graph run reports `steps=257 graph replays=256 (captures 1)`
and the `--no-graphs` run `graph replays=0`, 256 eager steps, same ids; `INTARSIA_NO_FUSION=1` runs the old chain (also the same ids).
Every A/B arm of `bench speed --require-graphs`: `eager_steps 0`, 1152 replays, both arms. The id runs shared the GPU with two other lanes and are not speed measurements.

## 4. Mutation checks

`scripts/cuda-decode/mutate.py` applies one deliberate bug to a copy of the kernel sources; `mutbuild.sh` builds one test binary per mutation (CPU, vLLM
up) and `wb1-correct.sh` runs them in the first vLLM-down window against the tests of their kernel. 18 mutations of the new kernels, all must fail:

| mutation | caught by |
|---|---|
| norm: `sum` not stored (`norm-skip-sum`); `normed` not stored (`norm-skip-normed`); sum squared unrounded (`norm-sum-round`); `+w_offset` dropped (`norm-woff`) | `..._is_bitwise_norm_then_pair` and `..._many_rows_bitwise` |
| norm: second matrix' blocks start one late (`norm-pair-block`); first matrix' block count drops its partial block (`norm-nb0`); the GEMV reads the global `x` not the staged row (`norm-read-x`); barrier between the smem row and its readers removed (`norm-sync`) | `..._is_bitwise_norm_then_pair` |
| **norm: the 8 squares of a thread added in reverse order (`norm-order`)** | **survived 3000 well-scaled draws; killed by the heavy-tailed test (draw 281)** |
| gated: normalised value not rounded to bf16 (`gn-round`); `+w_offset` dropped (`gn-woff`); silu on the wrong operand (`gn-silu`); weight chunk index wrong (`gn-weight`); 32-lane reduction, two heads share a sum (`gn-group`); W8A8 quantisation skipped (`gn-actq`); last chunk of the row not staged (`gn-tail`) | `gated_rmsnorm_matmul_fp8_is_bitwise_norm_then_gemv` and/or `..._many_draws_bitwise` |
| gated: xor butterfly in the opposite order (`gn-order`); 8 squares of a lane in reverse order (`gn-chunk-order`) | `gated_rmsnorm_matmul_fp8_many_draws_bitwise` only (needs the draws) |

The unmutated binary passes all 8 tests; `mutcheck.sh` checks off-GPU that every mutation pattern matches exactly once (it found `gn-actq` matching the
GEMM kernel too before the run). 18 of 18 killed in the end, but **`norm-order` was a survivor of the first run**, the same finding as the HIP page.
Wrong assumption recorded: "3000 draws of N(0, scale) rows expose a different summation order of the squares". bf16 values have 8 significant bits, so their
squares and the partial sums are exact in f32 while the terms sit within a few binades; a numpy emulation of the kernel's reduction (`norm-order` ordering against the
correct one) shows the f32 sum of squares differing in 31 of 3000 well-scaled draws and changing no bf16 output; with 2^U(-s,s) heavy tails it differs in
327 of 2500 draws and flips 9 outputs. The heavy-tailed test runs 8000 draws (about 20 flips expected); it was run once more in window B2: unmutated passes 8000, `norm-order` fails at draw 281.
`norm-sync`, a missing barrier, is caught on a run that happened to race; one run is evidence, not proof.

## 5. Files changed against `perf/hip-norm-gemv` (and so against `origin/main`)

```
crates/intarsia-backend-cuda/src/device_quantmoe.rs   +33   dispatch of the two kernels (extern decls, match arms)
crates/intarsia-backend-cuda/src/kernels.rs           +3 -5  table test: the two ops are no longer HIP-only
crates/intarsia-backend-cuda/src/quantmoe.rs          +20    TABLE entries and registration with admission
crates/intarsia-backend-cuda/tests/fused_decode.rs    +189   5 new tests and a helper
kernels/cuda/matmul.cu                                +118   XS flag of gemv_body, norm_add_stage, rmsnorm_add_pair_kernel, launcher
kernels/cuda/fp8.cu                                   +100   gated_norm_chunk, PRO flag of fp8_gemv_body, fp8_gemv_gated_kernel, launcher
kernels/abi/ik_abi.h                                  +2 -2 comment only ("HIP only so far" -> "HIP and CUDA")
scripts/cuda-decode/*                                        window / build / mutation scripts
docs/results/2026-10-17-cuda-*                               pages and raw logs
```

**Shared code touched: none.** No change in `intarsia-core`, `intarsia-models`, the fused-op definitions (`ops.rs`, `fused_ref.rs`, `quant/`), the registry, or any
other backend crate. The only shared file is `kernels/abi/ik_abi.h`, and only in a comment. So HIP and Metal behaviour cannot have changed from this branch
(their kernels, ABI declarations and registrations are untouched); `crates/intarsia-backend-cuda` and `kernels/cuda` hold every functional change.
Mac, after the rebase onto `df18c6a`: `INTARSIA_CPU_FREE_MEM_GB=8 cargo test --workspace` 456 passed, 0 failed; `cargo check --workspace` and
`cargo check -p intarsia-cli --features hip` green. (The CUDA crate's `.cu` and `tests/` are compiled only on spark1; the table test
`names_unique_and_ops_covered` runs on the Mac and covers the new registrations.)

## 6. Not done, uncertain

* The GPU numbers were taken before the rebase onto `df18c6a`. `main` moved in `intarsia-core` `engine.rs`, `engine/batch.rs`, `scheduler.rs`, `fault.rs`, tests between
  `3ad530e` and `df18c6a`. I read the engine / scheduler diff: it counts the tokens of failed requests in telemetry and adds a `fault-inject` feature-gated hook;
  nothing on the decode path of a normal request. The rebased HEAD was built on spark1 (`build.sh cudanorm --tests`, CLI and all CUDA test binaries, exit 0) and
  Mac-checked, but its GPU tests and benchmarks were not re-run.
* The 15 mutations of the earlier fused kernels (pair, fp8 pair, glu, gated, MoE v2) were not re-run although `gemv_body` and `fp8_gemv_body` were edited
  (a defaulted template flag each); their unmutated tests pass (fused_decode 8 of 8, quantmoe 10 of 10), and the new bitwise tests compare against the unfused kernels that use the same bodies.
* Speed: 3 reps per round (not 5), 3 rounds, one prompt-length draw per cell; +0.7 to +0.9% decode is at the edge of the round-to-round drift.
* No first-weight-load prefetch before the norm (HIP has it), not tried for lack of downtime.
* The fused ops are M = 1 only; the 8B (dense, no GDN / MoE) is untouched by them and was run only for ids and goldens.
* Single GPU; TP = 2 on CUDA not run.

## 7. Process: spark1, downtime

Builds and mutation binaries under `fleetq run --host spark1 --mem 16` (vLLM up, CPU only: `build.sh cudanorm --tests`, `mutbuild.sh`). All GPU work in
exclusive leases that take `guidance` down (`window.sh`, trap restore, `FLEETQ_JOB` on every ssh, `/v1/models` wait at 100.120.88.104:8000, release), no
`--force`, no `--no-queue`, no lease held while coding. The unit tests and the mutation binaries ran only inside windows (no co-tenancy with vLLM).

| window | content | vLLM down |
|---|---|---|
| A #2335 | no-op check of `perf/hip-norm-gemv` (see its page) | 14.8 min |
| B1 #2346 | nsys base + dev, backend tests, 18 mutation binaries, ids, 35b-short dumps, 8B goldens | 11.2 min |
| B2 #2352 | heavy-tail norm test (unmutated + `norm-order`), `qualify --quick-only`, 3 A/B rounds | 13.3 min |
| total | | **39.3 min** (Part B 24.5), each window under 20 |

vLLM answered `/v1/models` after each window (last: 14:30:00); `downtime.log` has the lines.
