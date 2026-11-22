# Lane ink-m-mix: Inkling-2-Mini depth-mix, conv and in-projection kernels on Metal (2026-11-21)

Branch `ink/m-mix` (worktree `.worktrees/ink-m-mix`, base `3feffee6e`). Target endor, M5 Max (40-core GPU, 614 GB/s). Not merged, not pushed.

## 0. Briefs received

- None after the initial brief (no SendMessage course corrections arrived during the lane).

## 1. What was built

| Kernel (registry name) | Op | Rows | Shader(s) | Form |
|---|---|---|---|---|
| `ink_in_proj_bf16_gemv` | InkInProj (73) | 1..8 (ordinary calls; priority above NAX) | `ink_in_proj_gemv_r1p1s8u2b1` | one GEMV dispatch over the stacked `[Nq+Nk+Nv, D]` weight; the threadgroup's segment picks its x block (`x[min(j, S-1)]`), so S = 1 and S = 3 are the same launch |
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

- One threadgroup per row, ceil(D / NE) threads (NE = 3 at D 2304: 768 threads; NE 4 measured better for prefill, section 3.5), thread `tid` owns elements `tid + j * tpg`.
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
  rows-exact and chunk-invariant (prefill rows are bitwise the decode rows; tested bitwise to 300 rows at D 64 and to 40 rows at
  D 2304, which covers the blend's 16-row h-GEMV chunks and tail at K 4608 and the deferred ring write).

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
| ink_in_proj | ink_in_proj_bf16_gemv | 18 | 0 | 4.63e-4 | 1.19e-5 | ADMITTED |
| ink_in_proj | ink_in_proj_bf16_gemv_rows | 24 | 0 | 8.28e-4 | 1.41e-5 | ADMITTED |
| ink_in_proj | ink_in_proj_bf16_nax | 23 | 0 | 3.15e-3 | 7.46e-5 | ADMITTED |
| conv_mix | conv_mix_bf16 | 532 | 0 | 4.53e-3 | 1.18e-3 | ADMITTED |
| attn_res_mix | attn_res_mix_bf16 | 97 | 0 | 2.02e-3 | 3.66e-5 | ADMITTED |
| sconv_res | sconv_res_bf16 | 64 | 0 | 1.88e-3 | 1.23e-5 | ADMITTED |
| loop_blend | loop_blend_bf16 | 27 | 0 | 3.23e-3 | 2.64e-4 | ADMITTED |

- The same sweep with the blend's gate geometry changed (`INTARSIA_INK_BLEND_CS=256 _RB=4` and `=32 / 64`) gives the identical
  max_abs / rel_l2 for every kernel, as the per-channel arithmetic and the 32-channel partial tree do not depend on it.
- Also run: `cargo test -p intarsia-backend-metal --lib` (18 pass), `--test quick_coverage` (pass), `kernels test --quick` for
  the five ops (all ADMITTED).
- Tolerances: bf16 outputs `Tol(1.5e-2, 6e-3)` (one bf16 ulp, `BF16_TOL` of the other bf16 Metal kernels), f32 mix outputs
  `Tol(1e-3, 2e-4)` (`F32_TOL`).
- Real-shape cases added (`cases/real.rs`, full sweep only): D 2304; InkInProj [6552 + 1056 + 1056, 2304] with S = 1 and 3 at
  rows 3 / 7 / 16 / 2048; ConvMix and SconvRes with Rr = 19 at pos0 0 and 29 (reads across the ring wrap), rows 1 / 3 / 7 / 16 /
  2048 (the 2048-row chunk takes the deferred ring write), 2..7 stored sources, Q 1 and 3, norm and the blend with S = 1 and 3
  (G 64); AttnResMix N up to 7, Q up to 4, with and without norm / addend; LoopBlend S = 1 / 3, Sd = 1 / 3.
- Rows-exact tests, `cargo test -p intarsia-backend-metal --test inkmix` (3 tests, pass): an R-row call is compared bitwise with R
  one-row calls made in order (the ring carried between them), every output and the final ring: InkInProj R = 1..16 (real
  shape, S 1 and 3); ConvMix / SconvRes at D 2304 (Rr 19) and D 64 (Rr 7, so rows past 4 read rows the call writes) for T = 1, 2,
  3, 7, 16, 40, 300 (D 2304: up to 40), pos0 0 / 2 / 29, host and device position, every epilogue and mix variant, blend S 1 and 3; AttnResMix and
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
561.8 / 559.2 GB/s around the runs (baseline 554).

Important: an earlier version of the bench (one short command buffer per sample, the GPU idle in between) under-reported every
kernel by 25-40 % (the existing `matmul_bf16_gemv` read 326 GB/s at [8664, 2304]); the numbers below are from the fixed method.

Raw logs and the scripts that produced them: `docs/results/2026-11-21-inkling2-metal-mix/raw/` (`final_proj.log`, `mix3.log`,
`sweep3.log`, `blendsweep.log`; `sweep1_short_burst_timing_invalid.log` is the first sweep under the broken timing method). Binary
for 3.1-3.4 built from `d580d471b`; the later gate refactor (`9e818db11`: rows per threadgroup and the partial index from the
launch parameters) is checked against it in `blendsweep.log` (section 3.5).

### 3.1 InkInProj decode / verify (real shape: W [6552 + 1056 + 1056, 2304] bf16 = 39.9 MB, byte floor 65.0 us at 614 GB/s)

| rows | InkInProj S=1 us | S=3 us | GB/s (S=3) | % of 614 | 3 x `matmul_bf16_gemv(_rows)` us (3 launches) | GB/s | `matmul_bf16_nax` on the stacked shape us |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 69.6 | 69.8 | **572** | **93 %** | 73.1 | 546 | - |
| 2 | 70.4 | 71.9 | 557 | 91 % | 76.9 | 520 | 106.3 |
| 4 | 72.6 | 72.5 | **552** | 90 % | 77.9 | 514 | 107.3 |
| 8 | 83.3 | 83.3 | 482 | 79 % | 82.8 | 485 | 106.4 |
| 16 | 161.8 | 167.9 | **241** | 39 % | 155.6 | 260 | 106.9 |

- Decode (1 row) streams at 572 GB/s, 93 % of the 614 GB/s byte floor and above the machine reference (mlx 1 GiB GEMV, 560 GB/s in
  the same slot); the three-launch Matmul form reaches 546. Rows 1-4 cost almost nothing over one row.
- 16 rows are ALU-bound (the bf16 unpack of x costs one op per fma): 241 GB/s, no better than the existing multi-row Matmul GEMV
  (260) and slower than NAX (107 us, not rows-exact). The exact_rows path must stay on the GEMV family; ordinary 2-8-row calls now
  take the GEMV (`ink_in_proj_bf16_gemv` registered for 1..=8 rows at priority 6, above NAX), 9 and more take NAX.
- Exploration (same harness, `raw/sweep3.log`): families (8,2) / (9,1) / (3,3) / (1,9) were within 3 % of each other at 1-4 rows;
  (1,9) is best at 16 rows (147 us with 8-row passes) but 3 % slower at 1 row, so the decode-optimal (8,2) is the default (the family
  must be the same for every row count). P = 2 weight rows per threadgroup and grid-stride caps were slower or equal.

### 3.2 `wo` (the existing Matmul, [2304, 3072] bf16 = 14.2 MB, floor 23.1 us)

| rows | kernel | us | GB/s | % of 614 |
|---:|---|---:|---:|---:|
| 1 | `matmul_bf16_gemv` | 25.7 | **551** | 90 % |
| 2 | `matmul_bf16_gemv_rows` | 26.5 | 535 | 87 % |
| 4 | `matmul_bf16_gemv_rows` | 27.8 | 511 | 83 % |
| 8 | `matmul_bf16_gemv_rows` | 29.3 | 486 | 79 % |
| 16 | `matmul_bf16_gemv_rows` | 53.2 | 269 | 44 % (ALU-bound, as above) |

`wo_ud` is 16-byte aligned in the checkpoint, so these are the serving kernels' numbers. No new kernel needed at decode.

### 3.3 Prefill, one 2048-row chunk

| op | kernel | us | |
|---|---|---:|---|
| InkInProj S=1 / S=3 | `ink_in_proj_bf16_nax` | 1297 / 1295 | 63 TFLOP/s (one grid, the three segments) |
| wo | `matmul_bf16_nax` | 467 | 62 TFLOP/s |
| ConvMix, N=4 Q=1 norm | `conv_mix_bf16` | 274 | + 1 deferred ring-write launch |
| ConvMix, N=4 Q=3 norm | `conv_mix_bf16` | 395 | |
| ConvMix + blend S=3 / S=1 | `conv_mix_bf16` | 4238 / 1791 | 8 chunks x 4-5 launches |
| AttnResMix N=4 Q=1 norm / N=2 Q=4 f32 | `attn_res_mix_bf16` | 149 / 216 | |
| LoopBlend S=3 | `loop_blend_bf16` | 3928 | |
| SconvRes | `sconv_res_bf16` | 118 | |

### 3.4 Row ops at decode / verify (D 2304, us per call, one launch unless noted)

| case | 1 row | 4 rows | 16 rows | launches |
|---|---:|---:|---:|---:|
| ConvMix N=4 Q=1 norm (plain sublayer boundary) | **8.5** | 8.7 | 9.0 | 1 |
| ConvMix N=4 Q=3 norm (into a full-attention layer) | 11.4 | 11.5 | 12.3 | 1 |
| ConvMix N=3 Q=1 + blend S=3 (looped attention input) | 35.4 | 39.0 | 71.7 | 4 |
| ConvMix N=3 Q=1 + blend S=1 (looped MLP input) | 27.7 | 31.3 | 53.9 | 4 |
| AttnResMix N=4 Q=1 norm | 5.6 | 5.6 | 6.0 | 1 |
| AttnResMix N=2 Q=4 f32 (loop messages) | 5.2 | 5.3 | 5.6 | 1 |
| LoopBlend S=3 | 25.5 | 30.4 | 62.7 | 4 |
| SconvRes res + addend + sum | 4.4 | 4.4 | 4.6 | 1 |

The row kernel went from 22.3 us (ConvMix plain, first version) to 8.5 us: the softmax moved onto the lanes of simdgroup 0 (22.3 ->
16.8), then the template over sources / query batch with all loads issued up front (16.8 -> 8.5; the loads had serialized behind
per-source branches). One dependent launch of a trivial kernel costs about 4 us here (SconvRes 4.4 us), so the plain ConvMix is about
two launch costs.


### 3.5 Gate geometry and row-kernel width (`raw/blendsweep.log`, binary of `4531bfefc`)

- The gate refactor did not move the timings: LoopBlend S=3 at the old geometry (64 channels, 16 rows per threadgroup) 25.1 / 62.8
  us at 1 / 16 rows (3.4: 25.5 / 62.7), ConvMix + blend S=3 34.2 / 37.9 / 70.6 us at 1 / 4 / 16 rows (3.4: 35.4 / 39.0 / 71.7).
- Rows per gate threadgroup: 4 rows is 12 % faster at 16 rows (55.3 vs 62.8 us), equal at 1 row (24.9) and 1 % faster at 2048
  rows (3878 vs 3924 us); 64 or 128 channels per threadgroup are equal, 256 is slower. **Recommended default: 4 rows, 64
  channels** (`BLEND_RB = 4`; bits identical, the admission sweep with `INTARSIA_INK_BLEND_RB=4` gives the same max_abs /
  rel_l2, `raw/adm-knobs.log`). The
  geometry barely moves the 2048-row blend, so its cost is in the h projection and the per-chunk launches, not the gate shape.
- Row kernel NE 4 (576 threads per row at D 2304) against NE 3 (768): ConvMix plain 8.45 vs 8.46 us at 1 row, 233 vs 274 us at
  2048 rows (-15 %); ConvMix Q=3 11.9 vs 11.4 us (+4 %, 8 visits per token); AttnResMix N=4 Q=1 5.41 vs 5.60 us, 137 vs 149 us at
  2048. **Recommended default: NE 4 from D 2048** (`mix_geom`). NE changes the per-row reduction tree (fixed per D, so still
  rows-exact) and needs one admission + rows-exact run at NE 4 before it is switched.
- **Not switched in this branch:** both are two-line changes in `inkmix.rs` (`BLEND_RB`, `mix_geom`); the verification build sat
  behind 11 queued endor jobs at the end of the lane, so HEAD keeps the verified NE 3 / 16-row defaults that every table here
  measures. Expected effect: 2048-row ConvMix plain 274 -> 233 us, AttnResMix 149 -> 137 us, 16-row LoopBlend 63 -> 55 us,
  decode unchanged within 4 %.

## 4. Launches per visit (decode, my ops)

Visit composition from spec a2 / a12 / a15 and the contract's ConvMix: each visit has two sublayer boundaries (after the attention
output, after the MLP output), each one ConvMix; looped visits (passes 1 and 2 of layers 6-17, 24 of the 48) blend both sublayer
inputs (attention side S = 3 streams q/k/v, MLP side S = 1) and need the message mix.

| visit kind | my launches (+ `wo`) | us at 1 row | us at 4 rows | us at 16 rows |
|---|---|---:|---:|---:|
| plain (24 visits: prelude, pass 0, coda) | InkInProj 1 + ConvMix 2 = **3** (+1) | 69.6 + 8.5 + 8.5 = **86.6** (+25.7) | 90.0 (+27.8) | 179.8 (+53.2) |
| looped (24 visits: passes 1-2 of layers 6-17) | InkInProj 1 + message AttnResMix 1 + ConvMix-blend S=1 4 + ConvMix-blend S=3 4 = **10** (+1) | 69.8 + 5.2 + 27.7 + 35.4 = **138.1** (+25.7) | 148.1 (+27.8) | 299.1 (+53.2) |
| per token (48 visits) | 24 x 3 + 24 x 10 = **312** (+48) | 5.39 ms (+1.23) | 5.71 ms (+1.33) | 11.5 ms (+2.55) |

Full-attention visits (8) use the Q=3 ConvMix into them (+2.9 us each). The output mix (a16) is one more AttnResMix per token
(~5.6 us). Prefill per 2048-row chunk, same composition: plain visit 1297 + 2 x 274 = 1.85 ms, looped visit 1295 + 216 + 1791 +
4238 = 7.54 ms, so 225 ms per chunk for my ops (+ 22 ms wo), of which the 48 blends are 145 ms.


## 5. Limits

- InkInProj at 16 rows is ALU-bound (241-249 GB/s; the 8-row form 482). The x unpack costs one ALU op per fma; the existing
  three-Matmul form is no faster (242 GB/s). Fixing it needs a matrix-unit form (below).
- The blend is 4 launches (row + pair, h GEMV, gate, norm): the one-launch ConvMix target holds only without it.
- The row kernels are latency-bound at decode: one threadgroup per row on one GPU core.
- LoopBlend / ConvMix-with-blend require G % 8 == 0 and 16-byte aligned gain / forget (true for the checkpoint; the launcher
  returns an error otherwise instead of a slow path).
- Prefill InkInProj needs the metal4 library; no metal3.1 multi-row fallback for T > 16 in ordinary calls.

## 6. Next ideas (ranked)

1. **Blend launches (decode).** 192 of the 312 launches per token and 1.5 ms of the 5.4 ms are the 48 blends (4 launches each).
   A gate + norm fusion with a last-threadgroup reduction (device atomics and `atomic_thread_fence`, which needs the metal3.2 /
   metal4.0 library; this library is metal3.1) cuts one launch each with bitwise the same reduction tree (the norm already sums
   the 32-channel partials in channel order). Folding the h projection in too needs a grid-wide dependency, so 3 launches is the
   floor without a persistent kernel. Also for the skeleton: the loop messages of one pass share their sources (the bank), so
   one AttnResMix with Q = 12 x 4 per pass replaces 12 calls (the kernel takes any Q).
2. **Blend prefill.** 145 ms per 2048-row chunk. The h projection at 16 rows per chunk is ALU-bound; above 16 rows a NAX GEMM
   for h (as InkInProj prefill does, not bitwise decode) is the next step; the gate geometry is already swept (3.5: at most 1 %
   at 2048 rows). Larger blend chunks (fewer launches per 2048 rows) cost scratch (21 MB per 256 rows).
3. **A matrix-unit rows-exact GEMV family for 9-16 rows.** InkInProj and `wo` at 16 rows run at 241-269 GB/s against 107 us for NAX.
   A simdgroup-matrix form whose 1-row launch uses the same 8x8 accumulation (x padded to 8 rows) would keep G0 and should take
   16-row verify to about the 8-row cost; the 1-row decode must then be re-measured (it must stay at ~570 GB/s).
4. **Row-kernel latency.** 8.5 us per ConvMix is about two launch costs; the remaining serial chain is the conv loads, three
   threadgroup reductions and the stores. Prefill at 2048 rows (274 us) is far from the ~100 us byte floor: more rows per
   threadgroup for long chunks is possible only if the per-row reduction tree stays the same (it is per row already, so a
   multi-row threadgroup that keeps the per-row thread count is a free choice).
5. **InkInProj fp8 / NVFP4 variants** for the opt-in quantized-projection path (perf-strategy lever 4): the same segment-select
   GEMV over fp8 weights would halve the 39.9 MB per visit.

