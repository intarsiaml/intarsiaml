# Metal prefill plan (2026-10-14): from 5 TFLOPS to the neural accelerators

Status: proposed, for operator decision. Nothing in this document is implemented. Branch `spike/metal`, host endor (Apple M5 Max, 40-core GPU, `applegpu_g17s`, macOS 27.0, Xcode SDK `MacOSX27.0.sdk`, Metal compiler 32023.921). Model Qwen3-8B BF16.

Naming. ADR-0009 section 5 already uses "phase C" for Qwen3.6-35B-A3B NVFP4 (about 70 h) and "phase D" for replay and fusion. This plan is therefore a **prefill track, items P0 to P5, inserted before ADR phase C**; it is not a rename of phase C. ADR-0009 states no prefill target. It lists "does `mpp::tensor_ops` pay off for prefill on M5" as an open question, and `docs/research/macos-metal-backend-scope.md` says "prefill: not a gate until phase D; on M5 the neural accelerators decide it". This plan proposes the first prefill targets (section 4). A NAX GEMM is also groundwork for NVFP4 prefill (MLX ships `fp_quantized_nax.h` and `quantized_nax.h`), which is a reason to do it before phase C and not after.

## 1. Outcome

- **Today** (same-slot localeval, marginal prefill): intarsia 350 tok/s at 1k and 205 at 4k, against mlx-lm 4423 / 3134 and llama.cpp 3068 / 2082. Plain `generate` prefill of the 973-token and 4062-token prompts: 335 and 226 tok/s.
- **Where the time goes** (GPU timestamp trace, section 2). At 4k: 57% matmul at 5.5 TFLOPS, **43% attention at 0.63 TFLOPS**, 0.4% everything else. At 1k: 86% matmul, 13.5% attention. The GPU is 99.9% busy in the normal stream, so launch overhead, encode and host work are not the problem; both heavy kernels are simply slow.
- **Ceiling.** MLX 0.32.1 on the same machine reaches 45 to 63 TFLOPS bf16 on the large Qwen3-8B GEMM shapes at M >= 1024 (51 TFLOPS on 8192^3), and 8 to 45 TFLOPS in SDPA (26 for a 256-row chunk against a 4k cache). Intarsia's GEMM is at about 9% of that, its attention at about 2%. MLX's NAX GEMM source shows the mechanism: `mpp::tensor_ops::matmul2d` (Metal Performance Primitives, Metal 4) on the M5 neural accelerators, with operands loaded straight from device memory into registers.
- **Neither big item alone gets past about 2x at 4k.** GEMM alone: 18.1 s -> about 10 s (403 tok/s). Attention alone: about 10.7 s (380 tok/s). Both are needed; both together with a larger chunk project to 1.9k to 2.2k tok/s at 4k.
- **Proposal.** Exit criterion: **>= 50% of mlx-lm marginal prefill at 1k and 4k** (>= 2210 and >= 1570 tok/s), target 65%, stretch 85%. Committed cost about 36 h (P0 to P3 plus measurement); P4 and P5 are optional (about 10 h) and worth under 10% each once P1 to P3 land.
- **First step, "do item P0/P1":** a 4 h go/no-go spike of one `matmul2d` bf16 GEMM at Qwen3-8B shapes. If it does not reach 20 TFLOPS at M = 1024 on 12288x4096, the plan falls back to a `simdgroup_matrix` kernel and the targets drop (section 6, fallback).

## 2. Measured breakdown

Measured in fleetq #1869 (endor, exclusive, 24 GB lease; `endor-kev` and `endor-tensorfold` resident and untouched). Binary `target/release/intarsia` at HEAD `62d602e`, `--features metal`. Raw: `docs/results/raw/2026-10-14-metal-phase-c/` (commands in `profile.sh`, tables from `analyze_prefill.py`, traces gzip'd).

### 2.1 Method

- Prompts: `prompt-4k.txt` (4062 tokens, from phase B) and `prompt-1k.txt`. **`prompt-1k.txt` did not exist in phase B's raw directory**; I derived it from the first 3950 bytes of `prompt-4k.txt`, which tokenizes to **973 tokens** (so "1k" below is 973 tokens: 3 chunks of 256 plus 205).
- Engine chunk: **256 tokens** (`DEFAULT_PREFILL_CHUNK`; `recipes/qwen3.toml` sets none, so Metal uses 256; only `recipes/qwen3_5_moe.toml` sets 2048). Each chunk runs the whole model once, so the weights are streamed 4 times at 1k and 16 times at 4k.
- Per-op GPU times come from the backend's own GPU-timestamp trace (`INTARSIA_METAL_PROFILE=kernel:<file>`, phase B section 5): every launch is its own command buffer and is waited on, so records never overlap. TFLOPS = 2*M*N*K over the kernel time. Attention FLOPs = 4*D*H*(causal visible q.k pairs), D = 128, H = 32. The `cb` mode (stream as normal) gives GPU busy time.
- The engine's `--profile-every 1 --op-table` was also run (`optable/`). It is host-synchronised, so it overstates small ops (its embedding row, 51 ms and 286 ms, is an artefact; the trace says 0.1 and 0.3 ms). Use the trace; the op table is kept only because phase A used it. Its totals (3.30 s at 1k, 20.9 s at 4k) are 11 to 16% above the trace because they include the host sync per op, and its matmul / attention split (79 / 12% at 1k, 54 / 39% at 4k) is correspondingly distorted.
- Cross-check: kernel sum 18.10 s against plain prefill 17.95 s at 4k, 2.97 s against 2.91 s at 1k.
- Reference ceiling: `mlx_gemm_bench.py` in the same slot, MLX 0.32.1 (`/Users/josef/.local/share/uv/tools/mlx-lm`), bf16 `x @ w.T` and `mx.fast.scaled_dot_product_attention` at the same shapes.
- GPU idle (ioreg Device Utilization %) read **16%** immediately before the run, above phase B's 10% bar. Plain prefill of 226 tok/s at 4k and 335 tok/s at 1k agree with phase A (213 to 233) and phase B (205 to 226 marginal), so I did not rerun. Thermal state was not measured.

### 2.2 Time per op class (kernel trace, chunk 256)

| Class | 1k (973 tok) ms | share | 4k (4062 tok) ms | share |
|---|---:|---:|---:|---:|
| matmul (7 per layer: q, k, v, o, gate, up, down) | 2548 | 85.9% | 10280 | 56.8% |
| attention (`attn_prefill_bf16`) | 400 | 13.5% | 7744 | 42.8% |
| elementwise / row (rmsnorm, qk-norm+rope+kv fold, silu_glu, embedding, argmax) | 19 | 0.6% | 74 | 0.4% |
| lm_head (M = 1 GEMV, last row only) | 2.3 | 0.1% | 2.2 | 0.0% |
| **kernel sum** | **2966** | | **18098** | |
| plain `generate` prefill | 2906 ms (335 tok/s) | | 17947 ms (226 tok/s) | |
| GPU busy in the normal stream (`cb`) | 100.0% | | 99.9% | |

### 2.3 Matmul by shape (4k prompt, chunk 256; the 1k prompt is the same within 6%)

FLOPs per prompt token are 13.89 GFLOP (6.94 B non-embedding parameters x 2); the 4062-token prompt is 56.43 TFLOP of matmul.

| Shape (n x k) | Layer role | calls | us/call | total ms | share of prefill | TFLOPS |
|---|---|---:|---:|---:|---:|---:|
| 12288 x 4096 | gate, up | 1080 (+72 tail) | 4376 | 4727 | 26.1% | 5.89 |
| 4096 x 12288 | down | 540 (+36) | 4906 | 2650 | 14.6% | 5.25 |
| 4096 x 4096 | q, o | 1080 (+72) | 1575 | 1701 | 9.4% | 5.45 |
| 1024 x 4096 | k, v | 1080 (+72) | 560 | 605 | 3.3% | 3.83 |
| M = 222 tail chunk, all shapes | | 252 | | 597 | 3.3% | 3.4 to 5.8 |

Aggregate **5.49 TFLOPS** at 4k, 5.31 at 1k.

### 2.4 Attention (4k, chunk 256)

`attn_prefill_bf16` ms for all 36 layers, per 256-token chunk: 25, 77, 128, 185, 226, 307, 335, 418, 522, 616, 680, 753, 809, 875, 920, 865. Time grows linearly with the cached context, so the whole prefill is quadratic; 7.74 s for 4.87 TFLOP = **0.63 TFLOPS**. At chunk 1024 (4k) it is 6.70 s, 0.73 TFLOPS. Why it is slow (`kernels/metal/attention.metal`, `attn_prefill_bf16`): 8 query rows per threadgroup (one simdgroup each), K tile staged as bf16 pairs but QK is a scalar per-lane dot over D with `qs` read from threadgroup memory, P.V is a serial loop over keys with `simd_broadcast` and V rows read from device memory by every row, and the 4 q heads of a GQA group each re-read the same K and V. No matrix instructions at all.

### 2.5 Chunk size (plain prefill, 4062 tokens; 1k prompt in last row)

| Chunk | prefill ms | tok/s |
|---|---:|---:|
| 128 | 22365 | 181.6 |
| 256 (default) | 17947 / 17962 / 18004 | 226 |
| 512 | 16868 | 240.8 |
| 1024 | 16318 | 248.9 |
| 2048 | 16450 | 246.9 |
| 1k prompt, chunk 1024 | 2536 | 383.7 (against 335 at 256) |

Chunk 1024 gains about 10 to 14% today. It gains little because the current GEMM runs at a flat 5.3 to 6.3 TFLOPS whatever M is (kernel trace at chunk 1024: 5.9 TFLOPS aggregate). **That changes once the GEMM is fast.** MLX's GEMM on the same shapes (bf16, this slot):

| M | q/o 4096x4096 | k/v 1024x4096 | gate/up 12288x4096 | down 4096x12288 | whole-layer mix |
|---:|---:|---:|---:|---:|---:|
| 256 | 16.8 | 5.1 | 37.4 | 33.8 | 24.7 |
| 1024 | 45.0 | 28.5 | 56.1 | 50.3 | 50.0 |
| 2048 | 51.9 | 36.8 | 60.2 | 54.8 | |
| 4096 | 52.7 | 44.3 | 57.0 | 52.4 | |

(TFLOPS. "Whole-layer mix" = Qwen3-8B per-layer FLOPs over the summed MLX times.) So with a good GEMM the chunk is worth 2x, not 10%. mlx-lm itself prefills in steps of 2048 (`prefill_step_size` default in `mlx_lm/generate.py`).

### 2.6 Hardware ceiling, and where the number comes from

Apple publishes no absolute GPU TFLOPS for the M5 Max. Apple's newsroom claims "over 4x peak GPU compute for AI" against M4 from a neural accelerator in each GPU core ([Apple M5 press release](https://secure.businesswire.com/news/home/20251014240640/en/Apple-unleashes-M5-the-next-big-leap-in-AI-performance-for-Apple-silicon)). Third-party estimates disagree and are unverified: 33.2 TFLOPS FP16 ([flopper.io spec sheet](https://flopper.io/gpu/apple-m5-max/spec-sheet.pdf)) and about 70 TFLOPS FP16 ([Skorppio](https://skorppio.com/blog/apple-m5-max-vs-nvidia-ai-deep-dive)). **I use the measured MLX ceiling on this machine as the reference**: bf16 51.4 TFLOPS (8192^3), 52.1 fp16, 56 to 63.5 at the large Qwen shapes (M = 1024 to 2048), 35.8 fp32. The 52 TFLOPS fp16 figure above the 33.2 estimate, and the 8192^3 bf16 / fp16 equality, mean MLX is on the neural accelerators here; I have not proven it by disabling them (no switch found). Intarsia's 5.5 TFLOPS is 9 to 11% of the MLX ceiling.

The mlx-lm localeval number is consistent with that ceiling: 4423 tok/s x 13.89 GFLOP = 61 TFLOPS of matmul at 1k, 3134 x 13.89 = 44 TFLOPS at 4k including attention.

### 2.7 How MLX does it (read from the installed package, headers only; the dispatch C++ and tile choices are not shipped)

`mlx/backend/metal/kernels/steel/gemm/nax.h`, `gemm_nax.h`, `kernels/steel_gemm_fused_nax.h`, `steel/attn/kernels/steel_attention_nax.h`:

- GEMM: `mpp::tensor_ops::matmul2d` with descriptor `(16, 32, 16, transpose_a, transpose_b, true, multiply_accumulate)` and `execution_simdgroup`; 16x16 fragments; per-simdgroup sub-tile `SM x SN` held in cooperative tensors (registers); K advanced `SK = 32` at a time; **A and B loaded straight from device memory, no threadgroup staging**; accumulator float, bf16 in; edge handling by `load_safe` / `store_safe`; function-constant `align_M/N/K`. B is read as `W[N,K]` with `transpose_b = true`, so **no weight repacking is needed**.
- The sixth descriptor argument, which MLX sets to `true`, is `relaxed_precision` ("set it to true to allow implementation to sacrifice accuracy for performance", `MPPTensorOpsMatMul2d.h` in the macOS 27.0 SDK). **This is the main accuracy question for P1** (section 5, risks).
- Attention: a separate NAX flash kernel (`steel_attention_nax.h`), same primitive.

## 3. Reuse of the existing contracts

No core change is needed. All items are new `KernelImpl` registrations (`crates/intarsia-backend-metal/src/kernels.rs`, `priority` and `when` predicates, as `matmul_bf16_gemv` and `attn_decode_bf16_splitkv` already do) plus launchers, and one recipe key.

- `Op::Matmul` with M > 1 already reaches the backend as `matmul_bf16` (priority 0, any M, tiled `simdgroup_float8x8` GEMM, f32 staging in threadgroup memory, `kernels/metal/gemm_bf16.metal`). It stays as the fallback and the admission oracle partner.
- `Op::AttnPrefill` is registered once (`attn_prefill_bf16`); CUDA and HIP register a flash variant beside it (`attn_prefill_bf16_flash`, tensor-core mma, 64 query rows per block, longest-first block order, online softmax in the log2 domain). The Metal flash kernel takes the same op contract.
- The fused ops `MatmulPair` (gate+up, k+v) only serve M = 1 on Metal (`kernels.rs`: `s.m == 1`); prefill runs gate and up as two GEMMs (`model.rs` `ffn_block`).
- CUDA's prefill track set the gate this plan reuses: the new GEMM must keep `matmul` rows independent of the call (the test `matmul_fp8_gemm_f8_rows_do_not_depend_on_the_call`, `docs/results/2026-10-13-prefill-cuda.md` section 4). Prefix reuse and the chunk envelope depend on it.
- `prefill_chunk` per backend is the recipe's `[tuning.<backend>] prefill_chunk` (ADR-0008 decision 2; `crates/intarsia-core/src/recipe.rs`).

## 4. Targets

Metric: localeval `speed`, `--sizes 128,1024,4096 --gen 256 --reps 3 --thinking off`, **marginal prefill** (prompt-token delta over TTFT delta), one fleetq slot with intarsia, mlx-lm and llama.cpp run back to back and intarsia again, exactly the phase B "Same-slot baselines" protocol (driver in `docs/results/raw/2026-10-14-metal-phase-b/sameslot/`). Reference engine: **mlx-lm**, as in the decode criteria ("at least 90% of mlx-lm"). I do not set a separate llama.cpp bar: llama.cpp sits at 69% of mlx-lm at 1k and 66% at 4k, so "match llama.cpp" and "about two thirds of mlx-lm" are the same bar.

| Level | 1k marginal tok/s | 4k marginal tok/s | Derived kernel gates |
|---|---:|---:|---|
| Today | 350 (7.9% of mlx-lm) | 205 (6.5%) | GEMM 5.5 TFLOPS, attention 0.63 |
| **Exit (committed): >= 50% of mlx-lm** | >= 2210 | >= 1570 | GEMM >= 35 TFLOPS aggregate at chunk 1024; attention >= 10 TFLOPS at 4k |
| Target: >= 65% (llama.cpp-class) | >= 2870 | >= 2040 | GEMM >= 40; attention >= 15 |
| Stretch: >= 85% | >= 3760 | >= 2660 | GEMM >= 50 (MLX parity); attention >= 25 |

Also required at every level: decode tok/s unchanged (phase B same-slot figures, within the ~5% noise), `qualify --backend metal` PASS, accuracy KL gates unchanged (section 5).

Why 50% committed: first implementations of a hand-written kernel on a new primitive typically land at 60 to 70% of the vendor's tuned one (the CUDA fp8 GEMM reached 53 TFLOPS against 75 to 87 for the best on the same box, `2026-10-13-prefill-cuda.md` section 3). 50% of mlx-lm is 70 to 80% of MLX's kernel rate once attention and elementwise are included. The marginal measure also undercounts a quadratic-attention engine, which is why the gates are stated in kernel TFLOPS too.

### Projection from the measured breakdown

Kernel-sum projection of plain prefill, using the measured elementwise time (19 ms at 1k, 74 to 79 ms at 4k) and the measured FLOPs (matmul 13.52 / 56.43 TFLOP, attention 0.28 / 4.87 TFLOP at 1k / 4k). Plain tok/s is not the marginal figure; marginal is lower for intarsia at 4k (205 against 226 plain today).

| Scenario | 1k s | 1k tok/s | 4k s | 4k tok/s |
|---|---:|---:|---:|---:|
| Today (chunk 256) | 2.97 | 328 | 18.10 | 224 |
| P1 only, GEMM 25 TFLOPS (MLX's rate at M = 256) | 0.96 | 1014 | 10.07 | 403 |
| P2 only, attention 15 TFLOPS | 2.59 | 376 | 10.68 | 380 |
| P1 + P2 at chunk 256 (25 / 15) | 0.58 | 1682 | 2.66 | 1530 |
| P1 + P2 + P3, conservative (GEMM 30, attention 8, chunk 1024) | 0.51 | 1924 | 2.57 | 1581 |
| **Committed** (GEMM 35, attention 10) | 0.43 | 2240 | 2.18 | 1865 |
| Target (40 / 15) | 0.38 | 2583 | 1.81 | 2239 |
| MLX parity (50 / 25) | 0.30 | 3226 | 1.40 | 2896 |

Neither P1 nor P2 alone reaches 2x at 4k (1.8x and 1.7x). After both, elementwise is 4 to 6% of 4k prefill, so P4 and P5 are second-order.

## 5. Work items, in order

Hours include the correctness gates and a same-slot measurement. All estimates are for one engineer with the repo context; ranges are 80% intervals.

### P0. Probe: does the toolchain give us `matmul2d`? (done, no GPU; the device test is the P1 spike)

Done already, no GPU: a 20-line `matmul2d` kernel (bf16 tensors from device pointers, float destination, `execution_simdgroups<4>`) compiles with `xcrun metal -std=metal4.0` and links to a metallib. It does not compile with `-std=metal3.1` (no `dextents`). All current `kernels/metal/*.metal` also compile unchanged with `-std=metal4.0` and link (610 KB metallib). **Not verified**: that such a metallib loads and creates a pipeline on `applegpu_g17s`, that it runs, and its numerics. `build.rs` pins `-std=metal3.1`; the cheapest safe change is a per-file std list so only the new file moves to metal4.0 and every existing kernel's bytes stay as admitted.

### P1. NAX bf16 GEMM for Matmul, M >= 16 (about 14 h, range 10 to 20)

- **What changes.** New `kernels/metal/gemm_nax_bf16.metal`: threadgroup of 4 simdgroups, `matmul2d` over `X[M,K]` (A) and `W[N,K]` (transpose_right, so weights are read as stored), float accumulate, bf16 round-to-nearest-even once at the store (same as `gemm_bf16_o_bf16`). Fixed tile configuration for **all** M (no M-aware selection), so a row's result cannot depend on the call; start with a 64x64 or 128x64 threadgroup tile, K dynamic. Edge tiles through sliced tensors (Qwen3-8B dims are all multiples of 128; the tail is M only: 205, 222, ...). New `KernelImpl` `matmul_bf16_nax`, `when`: bf16 acts and weights, bf16 out, `m >= 16`, `k % 32 == 0`, priority 5 (GEMV keeps `m == 1` at priority 10; tiled GEMM stays as the fallback). Launcher in a **new** file `crates/intarsia-backend-metal/src/gemm_nax.rs`, added to `LAUNCHER_SOURCES` in `build.rs` (otherwise the fingerprint will not invalidate admission, the phase B step 3 lesson). Registry must also drop the kernel if the device or OS cannot load the pipeline (macOS < 26 cannot load a metal4.0 library), so the engine falls back instead of failing.
- **Spike first (go/no-go, about 4 h).** `examples/gemm_nax.rs` in the style of `gemv_b3.rs`: the shapes in 2.3 at M = 256, 1024, 2048; report TFLOPS and max error against an f64 reference. Gate: >= 20 TFLOPS at M = 1024 on 12288x4096, else switch to the fallback below.
- **Expected gain.** Matmul 10.3 s -> 2.3 s at chunk 256 with 25 TFLOPS, 1.1 to 1.6 s at chunk 1024 with 35 to 50 TFLOPS. Alone: 4k 18.1 s -> about 10 s; 1k 2.97 s -> about 1.0 s (the matmul part is 4.7x faster; attention then holds 0.4 s of it).
- **Files.** New: `kernels/metal/gemm_nax_bf16.metal`, `src/gemm_nax.rs`, `examples/gemm_nax.rs`, `tests/gemm_nax.rs`. Edited: `build.rs` (per-file std, launcher list), `src/kernels.rs` (table, registration), `src/lib.rs` or `src/dense.rs` (dispatch), `tests/common`.
- **Correctness gates.**
  1. Kernel admission against the cpu reference: `intarsia qualify --backend metal` `kernels` stage, tolerance `BF16_TOL` unchanged (phase A: `matmul_bf16` max rel 3.2e-3 mean 1.2e-4 across 21 admitted cases).
  2. Unit tests against an f64 reference at the Qwen3-8B shapes and M in {16, 17, 205, 222, 256, 973, 1024, 2048}; worst error reported in bf16 ulps (the spike measured 0.97 half-ulp for the GEMV).
  3. **Row independence** (CUDA precedent): rows of a 1024-row call are bitwise equal to the same rows computed in calls of 5, 17, 64, 205 rows at other offsets. This guards prefix reuse and the chunk-perturbation stage.
  4. **`relaxed_precision` false against true** against the f64 reference, and the KL result for each. Start with `false`; accept `true` only if both the f64 error and the KL gate are unchanged. This is the main accuracy risk.
  5. `intarsia accuracy --gate-class bf16` on both goldens (short, long): KL gate 1e-3 (current prefill KL 2.608e-4 and 3.738e-4, top-1 100% / 99.86%), no non-finite rows; greedy ids on `prompt-short` and `prompt-4k` against today's output.
  6. `qualify --backend metal` full: envelope stage with `chunk` perturbation (the metal envelope today is chunk-only).
- **Measurement.** `docs/results/raw/2026-10-14-metal-phase-c/profile.sh` + `analyze_prefill.py` for per-shape TFLOPS (one slot, about 25 min), then localeval same-slot at the end of P3, not after each item (a 30 min slot each; the effect sizes here are 2x or more).
- **Fallback if the spike fails (< 20 TFLOPS) or NAX numerics fail the KL gate.** `simdgroup_matrix<bfloat,8,8>` GEMM: bf16 tiles in threadgroup memory (not f32, half the traffic), 64x64x32 tiles, double-buffered, 8 simdgroups. Ceiling on ALU matrix instructions is about 10 to 15 TFLOPS (my estimate from the 16 to 33 TFLOPS third-party figures; not measured on this chip), so +2x to +2.7x on the matmul part. Targets then become: exit >= 25% of mlx-lm. 10 h.
- **Weight layout.** No repack: `matmul2d` reads `W[N,K]` as stored. Revisit only if the GEMM lands below 70% of MLX's rate at M = 1024.

### P2. Flash prefill attention (about 14 h, range 10 to 20)

- **What changes.** New `kernels/metal/attention_flash.metal`: query block of 32 or 64 rows, K/V tiles of 32 to 64 keys, S = Q.K^T and O += P.V on matrix instructions (`matmul2d` if P1 passes, else `simdgroup_matrix`), f32 online softmax, **absolute** key-tile boundaries from key 0 (so the result for a row does not depend on the chunk it arrives in, as the current kernel and CUDA's), per-row causal masks, blocks issued longest-first, bf16 out. Step 2 inside the item: share each K/V tile across the 4 q heads of a GQA group (4x less K/V traffic; Qwen3-8B has 32 q / 8 kv heads). Register as `attn_prefill_bf16_flash` (CUDA/HIP name) with a higher priority than `attn_prefill_bf16`.
- **Expected gain.** 7.7 s -> 0.3 to 1.0 s at 4k (attention at 5 to 15 TFLOPS equivalent, against MLX SDPA at 8 to 45; chunk-256-against-4k SDPA in MLX is 25.8 TFLOPS); 0.40 s -> 0.03 s at 1k. Alone: 4k 18.1 s -> about 10.7 s.
- **Files.** New: `kernels/metal/attention_flash.metal`, `crates/intarsia-backend-metal/src/flash.rs` (new file; **do not touch `src/attn.rs` or `attention_splitkv.metal`**, the decode-attention work owns them), `tests/flash_prefill.rs`. Edited: `build.rs` (launcher list), `src/kernels.rs`.
- **Known precision difference.** The current kernel keeps P in f32 and multiplies bf16 V (converted to f32); a matrix-instruction P.V casts P to bf16, as CUDA's and HIP's flash kernels do (they pass the same goldens at prefill KL of about 2 to 4e-4, phase A comparison table, within the same 1e-3 gate). Test it against the f64 reference with the existing `prefill_matches_f64_reference` harness (`tests/attention.rs`) and report the KL.
- **Correctness gates.** Admission vs cpu (`BF16_TOL`, as `attn_prefill_bf16` 2.0e-3 max, 1.1e-4 mean today); f64 test over T in {1, 5, 64, 205, 256, 1024}, pos0 in {0, 256, 3840, 8191}, D in {64, 96, 128, 256}, head counts 32/8 and 1; **chunk invariance** (bitwise: the same prompt prefilled with chunks 64, 256, 1024 gives identical attention output rows); the goldens as in P1 gate 5; `qualify` envelope.
- **Measurement.** The trace table (attention TFLOPS and ms per chunk) and the localeval slot.
- **Risk.** Threadgroup memory budget at D = 128 with bf16 K/V tiles and a 4-head group is tight (32 KB per threadgroup); if matrix-instruction S/P layouts differ from the CUDA ones, the online-softmax code needs the Metal fragment layout (`BaseNAXFrag::get_coord` in the MLX header documents it for NAX).

### P3. Chunk size for Metal, re-measured after P1 and P2 (about 4 h)

- **What changes.** Sweep chunk 512, 1024, 2048, 4096 with the kernel trace and plain prefill after P1 and P2, then set `[tuning.metal] prefill_chunk = N` in `recipes/qwen3.toml` (a per-backend key already supported). Expected optimum 1024 to 2048 (MLX's GEMM goes from 25 to 50 TFLOPS between M = 256 and 1024 and flattens after; mlx-lm uses 2048). Workspace cost at 2048 is about 150 MB for gate/up/act and small for the rest, against 128 GB.
- **Expected gain.** Worth about 1.5 to 2x on the matmul part with a fast GEMM (section 2.5), about 10% today.
- **Cost that is easy to miss.** `prefill_chunk` is part of the accuracy baseline identity: `crates/intarsia-cli/src/cmd/accuracy/baseline.rs` refuses a mismatch, and the envelope's `chunk` perturbation is derived from the baseline's chunk (chunk / 4). So P3 re-records the Metal baselines and the noise envelope (`qualify` does this; about 17 min of GPU for the full run). Do it once, after P1 and P2, not per item. Also affects the prefix-reuse alignment stage ("aligned" restores at a multiple of the chunk).
- **Gates.** Bitwise identical output across chunks 64, 256, 1024 for the new kernels (the same chunk invariance test as P2), then the full `qualify --backend metal` run.

### P4. Fused gate+up and qkv GEMMs for M > 1 (optional, about 5 h)

k/v are 1024x4096, the worst-utilised shape (3.8 TFLOPS today, 5.1 even in MLX at M = 256). Concatenating k and v (or q, k, v) and gate and up into one weight read raises N and reduces launches (252 matmul launches per chunk, 4032 per 4k prefill). Needs a prefill `MatmulPair` registration or load-time weight concatenation (the latter changes the weight pipeline, bigger). Expected gain after P1: 3 to 6% at 1k, 2 to 4% at 4k. Do only if the P1 trace shows k/v below 60% of the other shapes' TFLOPS at the chosen chunk.

### P5. Epilogue and elementwise fusion for prefill (optional, about 5 h)

After P1 to P3, `rmsnorm_add` (14 us x 288 per 1k), `qk_norm_rope_kv`, `silu_glu` are 19 ms at 1k and 74 to 79 ms at 4k, which is 4 to 6% of the projected 1.4 to 2.6 s prefill. `silu_glu` fused into the gate/up GEMM epilogue is the largest piece. Skip unless the post-P3 trace shows more than 8% in this class.

### Sequencing

P1 spike (4 h, go/no-go) -> P1 (10 h) -> P2 (14 h; reuses the `matmul2d` binding experience from P1) -> P3 (4 h; one baseline re-record) -> same-slot localeval and write-up (4 h) = **about 36 h committed**. P4 and P5 only against the post-P3 trace. P1 first because it is the largest single gain at 1k (86% of time), carries the NAX go/no-go that also decides P2's approach, and benefits NVFP4 prefill later. P2 cannot be skipped: at 4k it is 43% of the time and the only item whose cost grows quadratically (16k prefill today is 95 tok/s, 177 s; phase A).

## 6. Risks

1. **NAX numerics.** `relaxed_precision` may lower the internal accumulation precision; MLX sets it to true. Gate P1 on `false` first and measure `true` separately. If `false` is much slower than `true`, the f64 error and KL decide, not the TFLOPS. Accumulation order inside the hardware tile is not under our control, so "fixed K order" must be shown by the row-independence test, not assumed.
2. **Row independence and prefix reuse.** M-dependent tile selection would make a row depend on the call (205- and 222-row tails occur on every prompt). Use one tile config; keep the CUDA test as a hard gate.
3. **Metal 4 availability.** A metal4.0 library needs macOS 26 or later; endor is macOS 27.0 and is the only Metal host in the fleet. The registry must degrade to the existing kernels if the pipeline cannot be created. Not verified on the device (P0).
4. **MLX is proof of the hardware, not of our kernel.** Intarsia may reach 60 to 70% of MLX's kernel; the 50% committed exit allows for it. The spike gate (20 TFLOPS at M = 1024) is the early warning.
5. **Accuracy of bf16 P in attention** (P2). Precedent: CUDA and HIP flash passes the same goldens. Measure; the current f32-P kernel remains registered as the fallback.
6. **Measurement noise.** Phase B recorded up to +-39% run-to-run on mid-length marginal prefill in fleet history. Today's gaps are 7 to 15x so it does not matter now; for the final 50% / 65% judgement use two bracketing intarsia runs as in the phase B slot and report both.
7. **Thermal.** Not measured. Sustained 50 TFLOPS on a laptop-class chassis (Mac17,6) may throttle on the longer runs; GPU idle was 16% before this session's run. The 4k run is only 2 to 18 s.
8. **Cold start.** The first request after a cold page cache runs at 7.4 tok/s (phase A, 16.4 GB read from disk through mmap'd weights). Every number in this plan is warm (a discarded 4k warm-up preceded all measurements, `plain/warm.err`). Nothing here fixes cold start.
9. **Coordination.** Another agent works on decode attention (`src/attn.rs`, `attention_splitkv.metal`, the phase B doc). This plan adds only new files plus one-line additions to `build.rs` (`LAUNCHER_SOURCES`, per-file std) and `kernels.rs`; expect merge conflicts there.
10. **Launcher fingerprint.** New launcher files must be added to `LAUNCHER_SOURCES` or admission results go stale silently.

## 7. Out of scope

- Decode (GEMV, split-KV, dispatch count, ICB), covered by phase B and the other agent.
- Quantised prefill (fp8, NVFP4 dequant+mma) and MoE prefill: phase C of the ADR. P1 is groundwork only.
- Cold-start page-in (`madvise(WILLNEED)`, `MTLResidencySet`, warm-up prefill at load): separate, about 3 h, independent of this plan.
- Tensor parallelism, batching of several sequences in one prefill, prefix-cache redesign, graph replay of prefill.
- Linking MLX or MPS (ADR-0009 decision 2). `mpp::tensor_ops` is used from our own kernels.
- Gated delta-net and hybrid-model prefill.
- Changing any core op contract or accuracy tolerance. No gate is loosened by this plan.

## 8. Not measured, and limits

- NAX pipeline load and run on this device, and `relaxed_precision` numerics (only compile and link were tested, no GPU).
- Non-NAX `simdgroup_matrix` ceiling on this chip (no switch found to disable NAX in MLX; the 10 to 15 TFLOPS fallback figure is an estimate).
- The M5 Max absolute peak (no Apple figure; see 2.6).
- Thermal behaviour, and any effect of the 16% idle GPU reading and the resident `endor-kev` / `endor-tensorfold` services.
- mlx-lm's and llama.cpp's own prefill kernel times. Their marginal tok/s come from the phase B same-slot run (`2026-10-14-metal-phase-b.md`, "Same-slot baselines"); I did not rerun them. MLX GEMM and SDPA numbers above are from a micro benchmark (one `mx.eval` loop of 20 iterations after 5 warm-ups for GEMM, 10 for SDPA), not from inside mlx-lm; the k/v 1024x4096 M = 256 point (421 us) looks like a small-kernel anomaly and I did not repeat it.
- One run per configuration for the traces; the two plain runs per size differ by 0.1%.
- The 1k prompt is 973 tokens, a prefix of the 4k prompt (section 2.1).

## 9. Reproduce

```sh
cd /opt/code/intarsiaml
cargo build --release -p intarsia-cli --features metal
fleetq acquire --host endor --exclusive --mem 24 --label "metal prefill profile" --expect 40 --wait   # id N
FLEETQ_JOB=N docs/results/raw/2026-10-14-metal-phase-c/profile.sh
fleetq release N
python3 docs/results/raw/2026-10-14-metal-phase-c/analyze_prefill.py docs/results/raw/2026-10-14-metal-phase-c/trace 4k 256
python3 docs/results/raw/2026-10-14-metal-phase-c/analyze_prefill.py docs/results/raw/2026-10-14-metal-phase-c/trace 4k 1024
```

Raw files in `docs/results/raw/2026-10-14-metal-phase-c/`: `plain/` (prefill speed, chunk 256, warm-up), `chunk/` (sweep), `optable/` (engine op table), `trace/` (GPU timestamp traces, `kernel-*` and `cb-*`, gzip), `mlx_gemm_bench.jsonl` and `.py`, `gpu-idle-before.txt`, `profile-console.log`, `release.log`. fleetq #1869.
