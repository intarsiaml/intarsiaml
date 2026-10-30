# HLOAD: load-then-wait serialisation in the HIP one-row fp8 and MoE decode kernels (R9700, Qwen3.6-35B-A3B NVFP4)

Lane hload, 2026-10-30. Branch `lane-hload/main` on main `e72ce1964` (K4FP8 #276 merged). Hardware: 2x AMD R9700 (gfx1201, tatooine),
ROCm 7.2.4 (image `intarsia-hip-dev:rocm7.2.4-rust1.90.0`), model `Qwen3.6-35B-A3B-NVFP4`. Main arm = tree `~/intarsiaml-tp2-k4fp8`
(crates, kernels, Cargo files byte-identical to main `e72ce1964`, checked with `rsync -c`); branch arm = `~/intarsiaml-tp2-hload`.

## Summary

K4FP8 found that weight loads written as `cond ? load : 0` (and a conditional prefetch) made the compiler put `s_wait_loadcnt 0x0`
before compute in `fp8_gemv_rows_kernel`, so the weight stream and the math ran in series. This lane audited the rest of the decode
path for the same pattern: the one-row fp8 GEMV family (`fp8_gemv_body`: M = 1..4 and the decode folds) and every MoE decode kernel.

- **The pattern is real in four places**, in three shapes: (a) the one-row fp8 body: branchy weight loads, and the staging's activation
  loads queued *behind* the whole weight window (loads return in order, so staging waited for every weight); (b) the routed G2 down
  kernel: each pair's `h` chunk loaded inside the pair loop right before its use, one serial L2 round trip per pair after the weight
  window; (c) the fused MoE block (1-2 rows) and the grouped shared-expert gate/up: the fp4 lookup table written *after* the weight
  window behind a `__syncthreads`, whose workgroup release fence is `s_wait_loadcnt 0x0` (the whole window), plus per-expert / per-token
  serial activation loads.
- **Fixes (all bit-identical to main, per-row accumulation order unchanged):** branch-free loads (clamped addresses) in the one-row
  body, the staging's first activation chunks requested before the weights in `fp8_gemv_kernel` (M = 1..4); a new
  `moe_down_g2f_kernel` for the one-lane-iteration down (I <= 512: the 35B and its TP=2 shard) that requests up to 4 pairs' `h` chunks
  together with the weights, with 4 rows per warp by default; the fp4 table before the existing ticket barrier in `moe_block_kernel`
  plus a ring of 4 in-flight expert `h` chunks in its consumer; table-first and next-iteration `x` prefetch in `shared_glu_rows_kernel`.
- **Kernel effect (same-window, interleaved, us):** routed down -6 % (1 row) to -13 % (16 rows) at TP=1, -11 % to -15 % at the TP=2 shard;
  one-row fp8: q_proj -4.2 %, GDN in_proj pair -2.9 %, M = 2 rows -3 to -13 %; fused MoE block (1 row) -2 %; shared gate/up -4 to -26 %.
- **Step (batched == solo gate, in process):** TP=1 16 rows 31.4 -> 30.5 ms (-2.9 %), TP=2 16 rows 23.1 -> 22.1 ms (-4.3 %); 2..8 rows
  -1.6 to -3.6 %; 1 row -0.4 to -0.6 %.
- **Served A/B (3 interleaved rounds, `--max-batch 16`, 1k prompts, gen 512, spec off, medians):** TP=1 1 / 4 / 8 / 16 streams
  0.991 / 1.004 / 1.020 / **1.030**; TP=2 1.006 / 1.025 / 1.038 / **1.034**. Single stream with `--max-batch 1`, TP=1: 1k 1.000,
  16k 1.003. **The single-stream goal is not met:** the one-row gains (fp8 -0.4..-4 %, fused MoE block -2 %) add up to ~0.5 % of the
  1-row step (gate: -0.6 % TP=1, -0.4 % TP=2), inside the served noise. The gain is at 2+ rows.
- **Correctness:** every touched kernel bitwise equal to main in standalone harnesses (incl. ragged shapes), all HIP fp8 / MoE repo
  tests pass, batched == solo bitwise on the real 35B at TP=1 and TP=2, 16-stream greedy outputs byte-identical main vs branch at TP=1
  and TP=2.
- **Negatives (kept out):** a ping-pong next-tile prefetch in the one-row body (+1..+3 us: at decode K a lane has 1-2 tiles and the
  doubled code costs more), an LDS-only barrier instead of `__syncthreads` (no gain alone), the early activation request in the folds and
  the pair kernel (+0.5..+2.4 us), any `x` prefetch in the routed gate/up (no gain at 4 pairs, register spill at 8: it is weight-bound),
  and a ring of token `x` chunks in the grouped shared down (no gain). The routed gate/up, the largest MoE kernel, keeps its serial
  per-pair `x` waits: they hit L2 and do not limit it.

## 1. Where the step goes (which kernels matter)

From HPROF (`2026-10-30-hprof-r9700-16stream-profile.md`, TP=1 profiled step, before HFP8/K4FP8, so the dense fp8 share at 16 rows is
smaller now):

| group (kernels) | B=1 | B=16 |
|---|---:|---:|
| dense fp8 projections (B=1: `fp8_gemv_kernel<1>`, pair, gated, sigmoid, norm-pair folds; B=16: `fp8_gemv_rows_kernel`) | 2.47 ms (38 %) | 11.18 ms (28 %) |
| MoE fused block (`moe_block_kernel`: router top-k, routed, shared, combine) | 1.92 ms (30 %) | - |
| MoE routed experts (`moe_gate_up_g2_kernel` + `moe_down_g2_kernel`) | - | 11.57 ms (29 %) |
| shared expert (`shared_glu_rows_kernel`, `shared_down_combine_rows_kernel`) | - | 2.05 ms (5 %) |
| lm_head (`nvfp4_gemv_v2_kernel<1>`) | 0.46 ms (7 %) | 2.53 ms (6 %) |
| MoE routing glue (top-k, group, reduce) | - | 0.41 ms (1 %) |

At 16 rows the routed down is about 42 % of the routed time (bench: 127 of 305 us per layer), so about 12 % of the step; at 1 row the
one-row fp8 family and the fused MoE block are two thirds of the step. M = 2..4 rows (2-4 streams) run `fp8_gemv_kernel<M>` for K = 4096
(out_proj / o_proj) and every M = 2 projection, the one-pass rows kernel otherwise (`launch_rows`, `kOnePassMinShort/Long`).

## 2. ISA audit (gfx1201, main `e72ce1964` vs branch)

Method: `scripts/hload/isa.sh` compiles each `kernels/hip/*.hip` with the crate's flags to assembly (`--cuda-device-only -S`);
`scripts/hload/waits.py` walks every kernel in program order and counts, per `s_wait_loadcnt`, the loads it retires (a straight-line
model): a **drain** retires >= 4 loads at once before math, a **serial** wait retires the last 1-3 loads in flight before math (the
load / wait / use / next-load pattern). Static counts are a guide; each verdict below was read off the assembly
(`raw/isa/{main,fin}-*.s.gz`, the full per-kernel counts in `raw/isa/waits-summary-{main,fin}.txt.gz`).

| kernel (decode use) | full wait before compute in the main loop (main) | what waits | step share | branch |
|---|---|---|---|---|
| `fp8_gemv_kernel<M=1..4>` (one-row fp8: q_proj M=1, every projection M=2, K=4096 at M=3-4) | **yes** | 3 drains: staging `x` loads issued after the 8-16 weight loads (wait 0 before the first LDS store), the `__syncthreads` release fence, the branchy loads | part of 38 % (B=1) | staging `x` first, branch-free; 1 drain left (K = 4096: the second tile) |
| `fp8_gemv_pair_kernel` (GDN in_proj_qkv + z, M=1) | **yes** (6 drains) | same body twice | largest M=1 fp8 kernel | branch-free only (early `x` measured slower) |
| `fp8_gemv_gated_kernel`, `_sigmoid_kernel`, `_norm_pair_kernel` (out_proj, o_proj, attn k/v + norm, M=1) | **yes** (3 / 3 / 6 drains) | same body | rest of the 38 % | branch-free only |
| `fp8_gemv_rows_kernel` (M >= 3/5, K4FP8) | no (fixed in #276) | - | 28 % (B=16) | unchanged |
| `moe_down_g2_kernel<2>` (routed down, 1-16 rows) | **yes, serial** (46 serial sites) | each pair's `h` chunk loaded in the pair loop under `if (q < pc)`, waited before its fmas, after the weight window | ~12 % (B=16) | `moe_down_g2f_kernel<4>`: `h` of 4 pairs with the weights, one wait |
| `moe_gate_up_g2_kernel<2>` (routed gate/up) | serial (69 sites) | `x` per (iteration, pair) in L2 | ~17 % (B=16) | unchanged (prefetch: no gain / spill, section 5) |
| `moe_block_kernel<PRE,8>` (fused MoE, 1-2 rows) | **yes** | producers: fp4 table + `__syncthreads` after the weight window (drain); consumer: `h` per expert serial (K + 1 round trips) | 30 % (B=1) | table before the ticket barrier, branch-free producers, ring of 4 `h` |
| `shared_glu_rows_kernel` (grouped shared gate/up, 3-16 rows) | **yes** | table barrier after the window; `x` per (iteration, token) serial | ~2 % (B=16) | table first, branch-free, next iteration's `x` prefetched |
| `shared_down_combine_rows_kernel` | serial (113 sites) | `x` per token | ~3 % (B=16) | unchanged (ring: no gain) |
| `moe_topk_v2`, `moe_group`, `moe_reduce_v8` | no compute behind loads worth overlapping (16 / 2 / 8 loads) | - | 1 % | unchanged |
| `nvfp4_gemv_v2_kernel<1>` (lm_head), `nvfp4_glu_kernel`, `shared_down_combine_kernel<1>` | no (1 drain at the window end, by design) | - | 7 % / - | unchanged |

## 3. Changes

All in `kernels/hip`; every output is the old kernel's bit for bit (same fma chains in the same order, same roundings; only which loads
are in flight when, and which address a lane that never uses a value reads).

1. `fp8.hip` `fp8_gemv_body`: weight loads from clamped addresses (a row past N reads row 0, a chunk past K reads chunk 0; neither is
   used) instead of `cond ? ik_ldcs(..) : 0`; the staging split into loads (`sload`) and arithmetic (`gated_norm_math`,
   `sigmoid_mul_math`, which the old `*_chunk` helpers now wrap); with `PRE` (default for `PRO == 0`, off in `fp8_gemv_pair_kernel`)
   the first two staging chunks of each thread are requested before the weight window (`sched_barrier` keeps the order).
2. `moe.hip`: `moe_down_g2f_kernel<RW>` (I / 16 <= 32, one lane iteration): all loads unconditional from valid addresses, `h` of up to
   4 pairs requested with the weights, pairs 5-8 of a chunk (rare) loaded as used; `g2_gate_up_launch` / `g2_down_launch` helpers;
   `INTARSIA_HIP_MOE_G2` down default `auto` (4 rows per warp where the twin applies, else 2; `docs/tuning.md`).
3. `moe_block.hip`: the fp4 table written before the ticket `__syncthreads` every role already passes (the producers' own barrier after
   the window is gone), branch-free producer windows, consumer `h` chunks in a ring of 4 (`IK_MB_HRING`) plus the shared activation chunk
   requested right after the acquire.
4. `moe_block_grouped.hip` `shared_glu_rows_kernel`: table first, branch-free window, `x` of iteration j + 1 requested while j computes.

## 4. Per-kernel microbench (before / after)

Standalone harnesses that `#include` the kernel file (`scripts/hload/bench_*.hip`, built with `scripts/hload/bench.sh`, run in GPU windows
with `scripts/hload/mbwin.sh`), real 35B shapes, weights rotated beyond the 64 MB infinity cache. Final window `raw/mb6` (job #7447): main
and branch interleaved twice, the minimum shown (the two reps agree within 0.5 %).

### 4.1 One-row fp8 family (us per call, actq on)

| kernel, shape (N x K) | use | TP | main | branch | change |
|---|---|---|---:|---:|---:|
| pair 8192+4096 x 2048 | GDN in_proj_qkv + z, M=1 | 1 | 44.87 | 43.55 | -2.9 % |
| gated 2048 x 4096 | GDN out_proj + gated norm, M=1 | 1 | 19.16 | 19.09 | -0.4 % |
| norm-pair 512+512 x 2048 | attn input norm + k / v, M=1 | 1 | 8.19 | 8.09 | -1.2 % |
| sigmoid 2048 x 4096 | attn o_proj + output gate, M=1 | 1 | 18.26 | 18.32 | +0.3 % |
| v2 8192 x 2048 | attn q_proj, M=1 | 1 | 29.52 | 28.28 | -4.2 % |
| rows M=2 8192 x 2048 | GDN qkv / attn q, 2 rows | 1 | 32.18 | 29.36 | -8.8 % |
| rows M=2 4096 x 2048 | GDN z, 2 rows | 1 | 19.04 | 16.88 | -11.3 % |
| rows M=2 2048 x 4096 | out_proj / o_proj, 2 rows | 1 | 17.47 | 16.92 | -3.1 % |
| rows M=2 512 x 2048 | attn k, v, 2 rows | 1 | 5.60 | 5.33 | -4.8 % |
| rows M=3 / M=4 2048 x 4096 | out_proj / o_proj, 3-4 rows | 1 | 19.28 / 19.85 | 18.88 / 19.44 | -2.1 % / -2.1 % |
| pair 4096+2048 x 2048 | GDN in_proj, M=1 | 2 | 23.05 | 22.11 | -4.1 % |
| gated 2048 x 2048 | out_proj, M=1 | 2 | 12.29 | 12.16 | -1.1 % |
| norm-pair 256+256 x 2048 | k / v, M=1 | 2 | 6.09 | 5.99 | -1.6 % |
| sigmoid 2048 x 2048 | o_proj, M=1 | 2 | 11.89 | 11.77 | -1.0 % |
| v2 4096 x 2048 | q_proj, M=1 | 2 | 17.00 | 15.74 | -7.4 % |
| rows M=2 2048 x 2048 | z / out, 2 rows | 2 | 11.88 | 10.29 | -13.4 % |
| rows M=2 256 x 2048 | k, v, 2 rows | 2 | 5.45 | 4.82 | -11.6 % |

Ablation (`raw/mb5`, job #7445, 2x2 on the final loop, 2 interleaved reps each; `pl` = early staging + LDS-only barrier, `nl` = no early
staging + LDS barrier, `ps` = early staging + `__syncthreads`, `ns` = neither):

| shape | main | pl | nl | ps | ns |
|---|---:|---:|---:|---:|---:|
| pair 8192+4096 x 2048 | 44.89 | 45.30 | 43.70 | 45.30 | 43.70 |
| gated 2048 x 4096 | 19.18 | 21.62 | 19.13 | 20.57 | 19.09 |
| v2 8192 x 2048 | 29.51 | 28.06 | 28.44 | 28.24 | 28.42 |
| v2 4096 x 2048 | 16.98 | 19.01 | 16.06 | 15.82 | 16.06 |
| rows M=2 8192 x 2048 | 32.12 | 28.85 | 32.00 | 29.39 | 32.10 |
| rows M=2 2048 x 2048 | 11.86 | 10.01 | 11.86 | 10.29 | 11.87 |
| pair 4096+2048 x 2048 | 23.07 | 24.99 | 22.08 | 24.89 | 22.12 |

Branch-free loads alone (`ns`) give the pair kernel and q_proj their gain; the early staging request pays only in `fp8_gemv_kernel`
(`ps`); the LDS-only barrier adds nothing on its own and misbehaves at v2 4096 x 2048 with early staging (19.0 us), so the branch keeps
`__syncthreads` and turns early staging on for `fp8_gemv_kernel` only (= `ps` there, `ns` in the folds and the pair kernel). The ping-pong
prefetch of the next weight tile, tried first (`raw/mb3`, `raw/mb4`), made every shape slower by 1-3 us (v2 512 x 2048: 5.50 -> 8.61).

### 4.2 MoE (us per call)

Routed G2 projections, Zipf routing (exponent 0.9: 8 / 15 / 26 / 45 / 75 distinct experts at 1 / 2 / 4 / 8 / 16 rows, HPROF measured 42 at
8 and 69 at 16), one binary, `BENCH_OLD=1` (main's `moe_down_g2_kernel<2>`) vs default (branch), interleaved:

| rows | down I=512 main | branch | change | down I=256 (TP=2) main | branch | change | gate_up I=512 (unchanged) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14.98 | 14.05 | -6.2 % | 14.07 | 12.56 | -10.7 % | 23.2 |
| 2 | 23.84 | 21.86 | -8.3 % | 22.50 | 19.93 | -11.4 % | 35.9 |
| 4 | 39.57 | 35.58 | -10.1 % | 38.15 | 33.29 | -12.7 % | 56.0 |
| 8 | 67.77 | 59.89 | -11.6 % | 68.17 | 58.83 | -13.7 % | 95.8 |
| 16 | 129.01 | 112.28 | -13.0 % | 116.00 | 99.08 | -14.6 % | 178.2 |

Attribution (`raw/mb1`, job #7425, I = 512, 16 rows): main kernel at 2 rows per warp 129.4, main kernel at 4 rows per warp 143.3 (worse),
new kernel at 2 rows per warp 111.8, at 4 rows per warp 106.7. The load fix is the gain (-13.5 %); 4 rows per warp adds ~4 % on top only
with the new kernel (fewer VGPRs: 135 vs 207 for the old kernel at 4, ROCm 7.2.4).

Fused MoE block (`ik_moe_block_bf16` 1 row, `_rows` 2 rows; 2 reps, main / branch): I = 512 1 row 56.33, 56.19 / 54.76, 55.36 (-2.1 %),
2 rows 88.82, 89.39 / 88.65, 88.63 (-0.5 %); I = 256 (TP=2) 1 row 43.17, 42.81 / 43.14, 42.83 (0 %), 2 rows 67.00, 65.96 / 66.76, 66.63 (0 %).
The same -1.5..-2.8 % at I = 512, 1 row, held in three windows (mb2, mb6 twice).

Grouped shared expert (3-16 rows; IS 512 / 256), gate/up main -> branch:

| rows | IS=512 | IS=256 |
|---:|---:|---:|
| 3 | 8.26 -> 7.30 (-11.6 %) | 8.41 -> 6.58 (-21.8 %) |
| 4 | 9.44 -> 8.03 (-14.9 %) | 9.63 -> 7.10 (-26.3 %) |
| 8 | 10.53 -> 10.07 (-4.4 %) | 8.78 -> 7.19 (-18.1 %) |
| 16 | 15.99 -> 14.71 (-8.0 %) | 10.10 -> 9.52 (-5.7 %) |

`shared_down_combine_rows_kernel` unchanged (12.9-23.6 us; a ring of token `x` chunks measured 12.9-23.4: no gain, reverted).

## 5. What did not help

- **Routed gate/up `x` prefetch.** (a) a twin kernel with the `x` of the next lane iteration in flight for 4 pairs: 256 VGPRs, 380-1520
  B/lane of scratch (spill) at either rows-per-warp; (b) the minimal form, `x` of the first 2 / 4 / 8 pairs requested at the top of each
  iteration of the existing kernel (`raw/mb1`): 4 pairs 24.2 / 36.9 / 56.9 / 96.7 / 179.8 us vs 24.2 / 36.9 / 57.1 / 96.3 / 180.8 (same
  binary, no change), 8 pairs +20..+34 % (spill). The kernel streams 2 x 512 x 1024 bytes of weights per expert; its `x` chunks are 4 KB
  per token, L2-resident and shared by all 64 blocks of an expert, so the serial `x` waits are short and hidden by the other waves.
- Ping-pong next-tile prefetch, LDS-only barrier, early staging in the folds and the pair kernel: section 4.1.
- Ring of 2 / 3 instead of 4 expert `h` chunks in the fused block (`raw/mb2`): within 1 % of 4.
- Shared down token ring: section 4.2.
- Note on harness noise: two builds of the same `moe_gate_up_g2_kernel` source measured 26.2 vs 23.1 us at 1 row in different binaries
  (code placement); every before/after above is therefore same-binary (MoE) or same-window interleaved (the rest).

## 6. End to end

Served A/B, `scripts/hload/window.sh` (from K4FP8's), one server per arm and round, arms interleaved, order reversed on even rounds;
`localeval speed` aggregate decode tok/s (1 stream = that stream's decode tok/s); `--max-batch 16`, 1k prompts, gen 512, spec off,
thinking off; tables by `scripts/hload/table.py` (result files copied to `raw/localeval`). base = main (tree `k4fp8`), new = branch.

TP=1 (job #7453, `raw/w-tp1`):

| streams | base r1 | base r2 | base r3 | new r1 | new r2 | new r3 | base median | new median | new / base |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 164.7 | 167.1 | 166.7 | 165.3 | 165.0 | 165.2 | 166.7 | 165.2 | 0.991 |
| 4 | 304.2 | 299.7 | 304.7 | 304.3 | 305.4 | 308.1 | 304.2 | 305.4 | 1.004 |
| 8 | 401.3 | 406.6 | 407.2 | 412.7 | 414.9 | 418.1 | 406.6 | 414.9 | 1.020 |
| 16 | 475.3 | 473.4 | 474.6 | 489.6 | 488.9 | 488.1 | 474.6 | 488.9 | 1.030 |

TP=2 (job #7456, `raw/w-tp2`):

| streams | base r1 | base r2 | base r3 | new r1 | new r2 | new r3 | base median | new median | new / base |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 192.9 | 188.1 | 192.7 | 193.8 | 193.9 | 189.2 | 192.7 | 193.8 | 1.006 |
| 4 | 378.2 | 378.2 | 374.9 | 386.3 | 389.6 | 387.8 | 378.2 | 387.8 | 1.025 |
| 8 | 536.6 | 535.4 | 538.8 | 556.3 | 558.0 | 556.9 | 536.6 | 556.9 | 1.038 |
| 16 | 668.1 | 662.5 | 660.8 | 684.8 | 686.6 | 684.0 | 662.5 | 684.8 | 1.034 |

Single stream, `--max-batch 1`, TP=1, prompts of 1k and 16k tokens (job #7457, `raw/w-b1`):

| prompt | base r1 | base r2 | base r3 | new r1 | new r2 | new r3 | base median | new median | new / base |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1k | 167.4 | 164.3 | 164.4 | 165.4 | 162.7 | 164.3 | 164.4 | 164.3 | 1.000 |
| 16k | 146.2 | 145.7 | 145.8 | 146.3 | 146.4 | 146.3 | 145.8 | 146.3 | 1.003 |

At 1 stream the branch is flat (0.991-1.006 across the three windows, the run-to-run spread is ~2 %): the one-row kernels it speeds up
save ~30-40 us of a 6.3 ms step (gate below: -0.04 ms). At 4-16 streams the routed down (-10..-15 %), the M = 2..4 fp8 rows and the
shared gate/up give +2..+4 %, matching the in-process step times.

Greedy identity (job #7452, `raw/w-greedy`, `scripts/k4fp8/greedy.py`: 16 concurrent different ~1k-token prompts, temperature 0, gen
192): the JSONL outputs of main and branch are byte-identical at TP=1 (16 / 16 streams, 13013 chars) and at TP=2 (16 / 16, 12702 chars).

Step time from the batched == solo gate (the real 35B in process, graphs on; 130 steps per batch size; main = tree `k4fp8`, job #7451,
`raw/steps-main`; branch = job #7450, `raw/gates-b`; `scripts/hload/steps.py`):

| TP | rows | main ms/step | branch ms/step | branch - main |
|---|---:|---:|---:|---:|
| TP=1 | 1 (solo) | 6.32 | 6.28 | -0.04 (-0.6 %) |
| TP=1 | 2 | 9.20 | 8.90 | -0.30 (-3.3 %) |
| TP=1 | 4 | 12.40 | 12.20 | -0.20 (-1.6 %) |
| TP=1 | 8 | 18.20 | 17.70 | -0.50 (-2.7 %) |
| TP=1 | 16 | 31.40 | 30.50 | -0.90 (-2.9 %) |
| TP=2 | 1 (solo) | 5.41 | 5.38 | -0.02 (-0.4 %) |
| TP=2 | 2 | 7.60 | 7.40 | -0.20 (-2.6 %) |
| TP=2 | 4 | 10.20 | 9.90 | -0.30 (-2.9 %) |
| TP=2 | 8 | 14.00 | 13.50 | -0.50 (-3.6 %) |
| TP=2 | 16 | 23.10 | 22.10 | -1.00 (-4.3 %) |

The batched step prints with 0.1 ms resolution; the solo step is the mean of 15 lines (batch 2..16: N solo steps each).

## 7. Correctness

- Standalone, bitwise vs main's kernels (`cmp` of every output, `raw/mb6` + `raw/mb3`): one-row fp8 family 18 decode shapes + 27 ragged
  / odd shapes (N 1..8192, K 16..16400, M 1..4, every fold, actq on and off): 72 files identical; MoE G2 gate_up + down at 1-16 rows,
  I 512 and 256 (up to 13 pairs on one expert, so the rare >4-pair path runs): 10 identical; fused block 1-2 rows, I 512 / 256: 4
  identical; grouped shared gate/up + down, 3-16 rows: 8 identical. `scripts/k4fp8/bench_fp8.hip` check mode on the branch (rows kernel
  vs one-row kernel, M 1..16, 13 shapes incl. ragged): 260 checks, 0 failed.
- Repo tests on the branch (job #7449, `raw/gates-a`): HIP crate `rows_exact`, `fp8_f8`, `rows_registry`, `frag_layout`,
  `quick_coverage`, `fp8_pair_shapes`, `fusion_p1`, `fused_decode`, `decode_v2`, `wmma_folds`, `ar_norm`, `packed_rows`, `moe_block`
  (11 tests, incl. the G2 configs), `moe_pairs`, `moe_v3`, `quantmoe`: all pass. `docs_env_vars` (intarsia-cli): pass.
- `real_35b_batched_decode_is_bitwise_the_solo_step` on the branch (job #7450, `raw/gates-b`): TP=1 and TP=2, 17550 (row, step) pairs
  bitwise each.
- 16-stream greedy byte-identity main vs branch (job #7452): byte-identical JSONL at TP=1 and TP=2 (section 6).

## 8. Reproduce

```
scripts/hload/isa.sh <tree> <tag>                       # on tatooine, cpu job: gfx1201 assembly per kernel file
python3 scripts/hload/waits.py <file.s> --summary       # the wait audit (--match REGEX, --trace)
KTREE=<main tree> scripts/hload/bench.sh <tree> bench_moe main ; scripts/hload/bench.sh <tree> bench_moe fin   # cpu job
RUNS="a@bench_moe-main b@bench_moe-fin:BENCH_OLD=1" OUT=<dir> fleetq run ... -- scripts/hload/mbwin.sh       # GPU window
STEPS="hip moe tp1 tp2" OUT=<dir> fleetq run ... -- scripts/hload/tests.sh                                     # gates + step times
ARMS="tp1:base tp1:new" ROUNDS=3 OUT=<dir> fleetq run ... -- scripts/hload/window.sh                          # A/B (MODE=greedy, MB=1)
python3 scripts/hload/table.py <window dir> ; python3 scripts/hload/steps.py <main gates dir> <branch gates dir>
```

fleetq jobs (all `claude-908e11e8`, label prefix HLOAD): CPU (ISA dumps, harness builds, header copy) #7414, #7417-#7422, #7424,
#7427-#7430, #7432, #7435, #7438, #7439, #7441, #7442, #7444, #7446; microbench GPU windows mb1 #7425, mb2 #7431, mb3 #7440, mb4 #7443,
mb5 #7445, mb6 #7447; branch build #7448; gates #7449, #7450; main step times #7451; greedy #7452; A/B TP=1 #7453, TP=2 #7456; single stream #7457.
