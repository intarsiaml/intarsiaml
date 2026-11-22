# Lane ink-m-mix: Inkling-2-Mini depth-mix, conv and in-projection kernels on Metal (2026-11-21)

Branch `ink/m-mix` (worktree `.worktrees/ink-m-mix`, base `3feffee6e`). Target endor, M5 Max (40-core GPU, 614 GB/s). Not merged, not pushed.

## 0. Briefs received

- None after the initial brief (no SendMessage course corrections arrived during the lane).

## 1. What was built

| Kernel (registry name) | Op | Rows | Shader(s) | Form |
|---|---|---|---|---|
| `ink_in_proj_bf16_gemv` | InkInProj (73) | 1 | `ink_in_proj_gemv_r1p1s8u2b1` | one GEMV dispatch over the stacked `[Nq+Nk+Nv, D]` weight; the threadgroup's segment picks its x block (`x[min(j, S-1)]`), so S = 1 and S = 3 are the same launch |
| `ink_in_proj_bf16_gemv_rows` | InkInProj | 1..16, `rows_exact` | `ink_in_proj_gemv_r<R>p1s8u2b<R>` | the same family with R rows sharing one weight pass; row r bitwise the 1-row launch |
| `ink_in_proj_bf16_nax` | InkInProj | >= 2 (prefill) | `ink_in_proj_nax_t*` (metal4.0 library) | NAX GEMM over the three segments in one grid; the tile is `gemm_nax::pick_variant(T, N)`, each segment bitwise `matmul_bf16_nax` on it |
| `conv_mix_bf16` | ConvMix (74) | any | `ink_mix_row_e<NE>n<NS>q<QB>` (+ blend: `ink_in_proj_gemv` over `down`, `ink_blend_gate_g*`, `ink_blend_norm`) | one threadgroup per row: conv + residual / sum + depth mix + norm (or the blend's pair) |
| `attn_res_mix_bf16` | AttnResMix (67) | any | `ink_mix_row_*` | the same row kernel with the conv off |
| `sconv_res_bf16` | SconvRes (69) | any | `ink_mix_row_*n1q1` | the same row kernel with the mix off |
| `loop_blend_bf16` | LoopBlend (68) | any | `ink_blend_pair`, h GEMV, `ink_blend_gate_g*`, `ink_blend_norm` | the blend alone |

Files: `kernels/metal/inkling2_mix.metal`, `kernels/metal/inkling2_mix_nax.metal`, `crates/intarsia-backend-metal/src/inkmix.rs`,
`crates/intarsia-backend-metal/tests/inkmix.rs`, `crates/intarsia-backend-metal/examples/ink_mix_bench.rs`,
`crates/intarsia-core/src/inkling2/cases/real.rs`. Shared files: `lib.rs` (+4 lines: module, dispatch), `kernels.rs` (+14:
table rows, shader lists, the pinned-list test ignores the new-op kernels), `build.rs` (+2: source and metal4 list),
`inkling2/cases.rs` (+5: the real-shape cases outside `--quick`).

### 1.1 InkInProj GEMV

- Lane mapping and reduction of `gemv_multirow.metal`'s `rt` kernels, with the lane count fitted to D = 2304: 8 simdgroups x 32
  lanes x 2 chunks of 16 bytes = 4096 elements per step, so K = 2304 is one tail pass of 288 chunks; per (weight row, x row) one
  f32 accumulator in chunk order, `simd_sum`, then the 8 simdgroup partials summed in order.
- R rows only add accumulators (x rows loaded per chunk, the weight chunk shared), so row r of an R-row launch is bitwise the
  1-row launch. Families compiled: (simdgroups, chunks per lane) = (8,2) (9,1) (3,3) (1,9), each with P = 1/2/4 weight rows per
  threadgroup and RB = R or 8 row passes (`INTARSIA_INK_PROJ_CFG`, `_P`, `_RB`, `_GRID` are measurement knobs; the family is
  fixed for all R because it sets the bits).
- A grid-stride form over virtual threadgroups and a chunk axis on the grid's y (used by the blend's h projection over up to 256
  rows: 16-row chunks plus one tail launch) keep the per-row arithmetic unchanged.
- Fallback for W or x not 16-byte aligned or D % 8 != 0: three `matmul_bf16_gemv_rows` launches on views (bitwise per row too).
  The real checkpoint's `wq_du`, `gate_attn`, `wr_du`, `attn_sink_proj`, `wk_dv`, `wv_dv`, `wo_ud` and controller tensors are all
  16-byte aligned in the safetensors file (checked from the header); `mtp.safetensors` tensors are all 8 mod 16. The stacked
  weight is built by the loader anyway (its own allocation).

### 1.2 The row kernel (ConvMix / AttnResMix / SconvRes)

- One threadgroup per row, ceil(D / NE) threads (NE = 3 at D = 2304: 768 threads), thread `tid` owns elements `tid + j * tpg`.
- Conv: K taps read position-indexed (`p - l >= pos0` from the call's x, else ring row `(p - l) % Rr`, negative positions zero),
  `y = rA((sum + x) * (K+1)^-0.5)`, then `last` = `rA((res + addend) + y)` (sum), `res + y` (f32) or `y`. The ring rows of the
  call's last `Rr` positions are written in-kernel when no row of the launch reads a ring row another row writes
  (`T + K - 1 <= Rr`, always true at decode and verify: Rr = K-1+16); otherwise a second launch `ink_ring_write` writes them after
  the reads (prefill chunks longer than 16 rows).
- Depth mix: NS = N + 1 sources and QB = 1 or 4 queries per batch are template parameters, so the per-thread arrays stay in
  registers and all loads are issued unconditionally (clamped, masked) before the first use. One reduction pass per query batch
  carries the NS source sums of squares and the QB x NS dots (simd_sum, simdgroup partials in threadgroup memory); simdgroup 0
  finishes it with lane = (query, source): the partials summed in simdgroup order, `w_n`, the logit, and the softmax over the 8
  lanes of a query with xor shuffles. Then the weighted sum, and the norm (one more reduction) or the f32 mix.
- With a blend (ConvMix with msg), the same launch also writes the controller pair `rA([d w(d), m w(m)])` of every stream.
- Every per-row reduction order is fixed by (D, N, Q), never by T or by the row's position in the launch, so the kernels are
  rows-exact and chunk-invariant (prefill rows are bitwise the decode rows).

### 1.3 The blend (LoopBlend, ConvMix with msg)

`pair` (in the row kernel for ConvMix, `ink_blend_pair` for LoopBlend) -> `h = rA(pair . down^T)` (the InkInProj GEMV with the
streams as its segments: S x 64 rows of 2D) -> `ink_blend_gate` (one thread per channel, its 64 gain and 64 forget weights in
registers, 16 rows per threadgroup, per-simdgroup partial sums of y^2) -> `ink_blend_norm` (one threadgroup per (stream, row):
the partials in slice order). Chunks of 256 rows (scratch 21 MB at D 2304, S 3; sized by `KernelImpl::scratch` from the call's
rows). Needs G % 8 == 0 and 16-byte aligned gain / forget (the checkpoint's are).

## 2. Admission and exactness

`intarsia kernels test --backend metal --op <op>` (full sweep, fleetq jobs on endor), against the f64 cpu reference:

| op | kernel | cases | fail | max_abs | rel_l2 | result |
|---|---|---:|---:|---:|---:|---|
| ink_in_proj | ink_in_proj_bf16_gemv | 6 | 0 | 2.68e-4 | 1.19e-5 | ADMITTED |
| ink_in_proj | ink_in_proj_bf16_gemv_rows | 24 | 0 | 8.28e-4 | 1.41e-5 | ADMITTED |
| ink_in_proj | ink_in_proj_bf16_nax | 23 | 0 | 3.15e-3 | 7.46e-5 | ADMITTED |
| conv_mix | conv_mix_bf16 | 532 | 0 | 4.53e-3 | 1.18e-3 | ADMITTED |
| attn_res_mix | attn_res_mix_bf16 | 97 | 0 | 2.02e-3 | 3.66e-5 | ADMITTED |
| sconv_res | sconv_res_bf16 | 64 | 0 | 1.88e-3 | 1.23e-5 | ADMITTED |
| loop_blend | loop_blend_bf16 | 27 | 0 | 3.23e-3 | 2.64e-4 | ADMITTED |

- Tolerances: bf16 outputs `Tol(1.5e-2, 6e-3)` (one bf16 ulp, `BF16_TOL` of the other bf16 Metal kernels), f32 mix outputs
  `Tol(1e-3, 2e-4)` (`F32_TOL`).
- Real-shape cases added (`cases/real.rs`, full sweep only): D 2304; InkInProj [6552 + 1056 + 1056, 2304] with S = 1 and 3 at
  rows 3 / 7 / 16 / 2048; ConvMix and SconvRes with Rr = 19 at pos0 0 and 29 (reads across the ring wrap), rows 1 / 3 / 7 / 16 /
  2048 (the 2048-row chunk takes the deferred ring write), 2..7 stored sources, Q 1 and 3, norm and the blend with S = 1 and 3
  (G 64); AttnResMix N up to 7, Q up to 4, with and without norm / addend; LoopBlend S = 1 / 3, Sd = 1 / 3.
- Rows-exact tests, `cargo test -p intarsia-backend-metal --test inkmix` (3 tests, pass): an R-row call is compared bitwise with R
  one-row calls made in order (the ring carried between them), every output and the final ring: InkInProj R = 1..16 (real
  shape, S 1 and 3); ConvMix / SconvRes at D 2304 (Rr 19) and D 64 (Rr 7, so rows past 4 read rows the call writes) for T = 1, 2,
  3, 7, 16, 40, 300, pos0 0 / 2 / 29, host and device position, every epilogue and mix variant, blend S 1 and 3; AttnResMix and
  LoopBlend R up to 300 (two blend chunks, the h GEMV's chunks and tail).
- Plan-level classification: the row kernels are each op's only Metal kernel, so a verify or batched step picks the same kernel
  as the solo step (`plan::check_batch_ops` accepts that); InkInProj's verify rows pick the `rows_exact` twin.
- Not bitwise across forms: the NAX prefill kernel differs from the GEMV (and its tile, so its bits, depends on T), the same
  convention as `matmul_bf16_nax` vs `matmul_bf16_gemv`. Prefill needs the metal4.0 library (`intarsia_metal4.metallib`); without
  it InkInProj has no kernel above one row for ordinary calls.

## 3. Timing

Method: `ink_mix_bench` (this lane's example). GPU time per call from the backend's command-buffer trace
(`INTARSIA_METAL_PROFILE=cb:`), launches dependent (the serial encoder's barriers, as in a decode stream), each case on its own
operand set rotated over more than the system cache (12 x 40 MB weights for InkInProj), about 30 ms of back-to-back GPU work per
sample so the GPU runs at its working clock. Median of 10-15 samples; p10 / p90 in the raw logs. Every run in a
`fleetq run --quiet gpu,cpu` slot; machine reference `scripts/metal-speed-gate/refbw.py` (mlx, 1 GiB GEMV stream) read
FINAL_REFBW GB/s around the runs (baseline 554).

Important: an earlier version of the bench (one short command buffer per sample, the GPU idle in between) under-reported every
kernel by 25-40 % (the existing `matmul_bf16_gemv` read 326 GB/s at [8664, 2304]); the numbers below are from the fixed method.

FINAL_TABLES

## 4. Launches per visit (decode, my ops)

Visit composition from spec a2 / a12 / a15 and the contract's ConvMix: each visit has two sublayer boundaries (after the attention
output, after the MLP output), each one ConvMix; looped visits (passes 1 and 2 of layers 6-17, 24 of the 48) blend both sublayer
inputs (attention side S = 3 streams q/k/v, MLP side S = 1) and need the message mix.

FINAL_LAUNCHES

## 5. Limits

- InkInProj at 16 rows is ALU-bound (~246 GB/s; the 8-row form 476). The x unpack costs one ALU op per fma; the existing
  three-Matmul form is no faster (242 GB/s). Fixing it needs a matrix-unit form (below).
- The blend is 4 launches (row + pair, h GEMV, gate, norm): the one-launch ConvMix target holds only without it.
- The row kernels are latency-bound at decode: one threadgroup per row on one GPU core.
- LoopBlend / ConvMix-with-blend require G % 8 == 0 and 16-byte aligned gain / forget (true for the checkpoint; the launcher
  returns an error otherwise instead of a slow path).
- Prefill InkInProj needs the metal4 library; no metal3.1 multi-row fallback for T > 16 in ordinary calls.

## 6. Next ideas (ranked)

FINAL_IDEAS
