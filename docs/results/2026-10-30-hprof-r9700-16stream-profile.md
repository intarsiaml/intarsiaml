# HPROF: where the time goes in one batched decode step, Qwen3.6-35B-A3B NVFP4 on R9700 (HIP), 16 streams

2026-10-30. Branch `lane-hprof/profile16`, built from origin/main `d4936a3d4` plus the env-gated instrumentation of this lane
(`head=de2cc12 code-tree=19d108537c2a`, `~/intarsiaml-tp2-hprof` on tatooine). Hardware: AMD R9700 (gfx1201), HIP backend, ROCm 7.2.4
dev image, tatooine. Model `/data/models/intarsia/Qwen3.6-35B-A3B-NVFP4` (40 layers: 30 GatedDeltaNet + 10 full attention, every layer a
256-expert top-8 MoE with a shared expert; hidden 2048, expert intermediate 512, vocab 248320). Server flags as the r9s / r9t lanes
(`serve --max-batch 16 --spec off --profile-every 0 --max-seq 18432`, graphs on). No optimisation was made in this lane.

## Summary

1. **The 8 -> 16 flattening is the decode step itself, not the host, the scheduler or prefill.** An unprofiled 8-row step takes 20.3 ms and
   a 16-row step 38.8 ms (TP=1; 14.7 -> 27.6 ms at TP=2). That is 1.91x the time for 2x the rows, so decode-only throughput goes
   394 -> 412 tok/s (+4.5 %) at TP=1 and 550 -> 580 (+5.5 %) at TP=2. Past 4 rows the step is a line of about 2.3 ms per extra row at TP=1
   (a line through the 8- and 16-row steps leaves about 3 ms fixed, 8 % of the 16-row step).
2. **The host is not in the picture.** Per decode step the scheduler spends about 0.04 ms (upload 0.009, graph launch 0.024, token
   tail), the GPU idles 0.028 ms between steps (0.010 ms in the trace), and the batched step is a replayed HIP graph.
3. **Where the 39.9 ms (profiled) goes at 16 rows, TP=1:** dense fp8 projections 11.2 ms (28 %), MoE routed experts 11.6 ms (29 %),
   GatedDeltaNet 3.6 ms (9 %), small bf16 GEMVs 3.3 ms (8 %), lm_head 2.5 ms (6 %), shared expert 2.1 ms (5 %), attention 1.8 ms (4 %),
   idle gaps between the 1001 launches 3.2 ms (8 %), everything else 3 %.
4. **What explains 8 -> 16** (the +18.4 ms): dense fp8 +5.7 ms (31 %; the weights are read twice or three times instead of once, see
   below), MoE routed +5.0 ms (27 %; 42 -> 70 distinct experts, bandwidth-bound), small bf16 GEMVs +2.5 ms (14 %; the M=16 instantiation
   spills registers to scratch, 4.3x for 2x rows), GatedDeltaNet +1.7 ms, lm_head +1.3 ms (4 weight passes instead of 2), shared expert
   +1.0 ms, attention +0.8 ms, idle gaps +0.5 ms.
5. **Ranked bottlenecks (ceiling = ms per step saved at 16 rows, TP=1, if the kernel hit its roof):** (1) dense fp8 projections: up to
   9.2 ms; (2) the latency-bound small ops (bf16 GEMVs + shared expert): about 5.0 ms; (3) MoE routed experts below bandwidth roof: 3.8 ms;
   (4) lm_head in four M=4 passes: 2.1 ms; (5) launch gaps: about 1.5 ms. Together 21.6 ms, i.e. a 16-row step of about 18 ms
   (about 880 tok/s decode-only) as an upper bound. Section 8 has file:line and the evidence for each.
6. **Distinct experts are well below the uniform estimate** that earlier lanes used: 42 per layer at 8 rows and about 70 at 16 rows
   (uniform: 57 and 102), so the routed-expert floor is 7.7 ms rather than 11.3-12.1 ms (l7).

## 1. Method

* **Per-kernel GPU time: `rocprofv3 --kernel-trace` around `intarsia serve`**, in the `intarsia-hip-prof` image, with the HIP graphs on
  (the replayed batched step shows up kernel by kernel, with start/end timestamps, grid sizes, VGPR, LDS and scratch). The server is ended
  through the new `INTARSIA_HIP_EXIT_FILE` hook so the profiler flushes. `rocprofv3` costs 7 / 5 / 3 % of step time at TP=1 (1 / 8 / 16
  rows) and 17 / 13 / 7 % at TP=2 (table 1): the shares below are shares of a slightly longer step.
* **A step** is the kernels of one GPU between two `argmax` launches (greedy sampling; exactly one per step). Its row count B is the Z grid of
  `gdn_delta_rows_v2_kernel` (1 for the solo step). Per B only the steps with the modal kernel count are used (a prefill chunk mixed into a
  step changes the count), 175-371 steps per B. An idle lead-in (> 1 ms hole) is dropped. Analysis: `scripts/hprof/analyze.py`.
* **Workload** (`scripts/hprof/drive.py`, run by `scripts/hprof/window.sh`): phases b1, same8, same16, div8, div16 with a pause in between;
  ~1k prompt tokens per stream, 192 forced output tokens (384 for the unprofiled runs), temperature 0, thinking off. `same` = what
  `localeval speed --concurrency N` sends (identical filler text with a nonce in front); `div` = N different ~1k-token cuts of repo docs.
  Steady steps have 1.0-1.45k tokens of context. The B sweep (2, 4, 6, 10, 12, 13, 14) is a second window with `same` phases only.
* **Unprofiled step time and host cost: `INTARSIA_HOST_PROFILE=1`** (existing) on a plain server; `localeval speed` there reproduces the
  r9s / r9t aggregates.
* **Distinct experts: new `INTARSIA_HIP_ROUTE_STATS=1`** (`crates/intarsia-backend-hip/src/route_stats.rs`, hook at `device.rs:1442`):
  eager mode only (`INTARSIA_BATCH_GRAPHS=0`), stream sync plus a read-back of the top-k ids after each grouped MoE block. Rows 1-2 use
  the fused per-token block and are not recorded (1 row = 8 distinct by definition).
* **Windows** (all `fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --quiet gpu,cpu`, restore trap and guard from
  r9s / r9t): #7274 TP=1 prof + route + plain (radiance down 00:35:17-00:40:26, 5 min), #7277 TP=2 prof + plain (00:41:13-00:44:13, 3 min),
  #7279 TP=1 row-count sweep (00:46:53-00:48:40), #7281 TP=2 `INTARSIA_HIP_AR_LL_MAX` test (00:49:43-00:50:38). Build #7272 (`--load cpu`).
  #7273 was started by mistake (script not executable), never ran anything and was released at once; radiance was not touched.
  Radiance answered `/v1/models` again within 15 s of each restore and is up on both ranks now.

## 2. Step time, unprofiled and profiled; how the flattening arises

| ms per step | B=1 | B=8 | B=16 |
|---|---:|---:|---:|
| TP=1 unprofiled (host-profile, decode-only iterations, full-row ones) | 6.09 | 20.2-20.5 | 38.7-39.3 |
| TP=1 profiled (trace wall) | 6.52 | 21.49 | 39.91 |
| TP=2 unprofiled | 5.17-5.24 | 14.67-14.74 | 27.60-27.62 |
| TP=2 profiled | 6.10 | 16.59 | 29.70 |
| decode-only tok/s, TP=1 (rows / unprofiled step) | 164 | 394 | 412 |
| decode-only tok/s, TP=2 | 191 | 550 | 580 |

**The measured aggregates are fully explained by prefill plus these step times.** `localeval` divides the tokens by the wall time that
includes the serial prefill of all streams (one 1k prompt takes about 130 ms, and decode of the others stalls while it runs). With
`aggregate = N*G / (TTFT_last + G*step_N)` (G = 512 tokens, TTFT_last from the plain-run driver: 1.11 / 2.36 s at TP=1, 0.90 / 1.89 s at TP=2):
TP=1 356.1 and 368.5 tok/s against 356.0 and 371.2 measured by r9s; TP=2 487 and 511 against 482.3 and 510.5 measured by r9t. Prefill is
9.7 % of the 8-stream run and 10.6 % of the 16-stream run, so it is not what flattens the curve; the step time is. (Within the 16-stream
run the first 2.4 s are prefill and every stream then decodes at 16 rows until the end.)

### Step time against rows (TP=1, profiled)

| B | 1 | 2 | 4 | 6 | 8 | 10 | 12 | 13 | 14 | 16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| step wall ms | 6.52 | 10.04 | 13.97 | 17.46 | 21.49 | 27.28 | 29.76 | 32.85 | 34.75 | 39.91 |
| launches / step | 268 | 588 | 828 | 829 | 869 | 870 | 870 | 1001 | 1001 | 1001 |
| ms per extra row (from previous column) | - | 3.52 | 1.96 | 1.75 | 2.02 | 2.89 | 1.24 | 3.09 | 1.90 | 2.58 |
| dense fp8 projections | 2.47 | 2.71 | 3.23 | 4.02 | 5.48 | 6.97 | 7.87 | 8.93 | 9.66 | 11.18 |
| MoE routed experts (B=1, 2: fused block) | 1.92 | 3.38 | 3.80 | 5.42 | 6.57 | 7.83 | 9.01 | 9.69 | 10.11 | 11.57 |
| small bf16 GEMVs | - | 0.39 | 0.51 | 0.64 | 0.76 | 2.80 | 1.91 | 2.24 | 2.53 | 3.30 |
| lm_head | 0.46 | 0.47 | 0.64 | 1.08 | 1.27 | 1.70 | 1.91 | 2.24 | 2.35 | 2.53 |
| GatedDeltaNet | 0.24 | 0.59 | 1.07 | 1.30 | 1.97 | 2.19 | 2.60 | 2.78 | 2.98 | 3.63 |
| shared expert | - | - | 0.83 | 0.96 | 1.09 | 1.29 | 1.74 | 1.83 | 1.90 | 2.05 |

The curve is a line of about 2.3 ms per row from 4 rows up, with two steps: at 10 rows (the bf16 GEMV and fp8 M=10+ instantiations spill
registers, section 5) and at 13 rows (the fp8 projections need a second pass above `kRowsGroup = 12`, 870 -> 1001 launches). There is no
row count at which a fixed cost is amortised any further.

## 3. Per-op breakdown (ms per step, share of the profiled step wall)

TP=1 (card 0):

| group | B=1 | B=8 | B=16 |
|---|---:|---:|---:|
| MoE routed experts (gate_up + down) | - | 6.57 (31 %) | 11.57 (29 %) |
| MoE fused block (B=1: router + top-k + routed + shared) | 1.92 (30 %) | - | - |
| dense fp8 projections (GDN qkv/z/out, attention q/k/v/o) | 2.47 (38 %) | 5.48 (25 %) | 11.18 (28 %) |
| GatedDeltaNet (conv, delta rule, gated norm) | 0.24 (4 %) | 1.97 (9 %) | 3.63 (9 %) |
| shared expert (gate+up, down+combine) | - | 1.09 (5 %) | 2.05 (5 %) |
| lm_head (NVFP4 GEMV) | 0.46 (7 %) | 1.27 (6 %) | 2.53 (6 %) |
| attention without projections (split-KV, rope, KV append, gate) | 0.24 (4 %) | 0.95 (4 %) | 1.79 (4 %) |
| small bf16 GEMVs (router, GDN b/a, shared-gate logit) | - | 0.76 (4 %) | 3.30 (8 %) |
| MoE routing glue (top-k, group, reduce) | - | 0.38 (2 %) | 0.41 (1 %) |
| norms / residual add | 0.29 (4 %) | 0.22 (1 %) | 0.22 (1 %) |
| sampling (argmax) | 0.05 (1 %) | 0.04 | 0.04 |
| embedding, copies | 0.01 | 0.01 | 0.01 |
| idle between launches inside the step (host-side and graph gaps) | 0.85 (13 %) | 2.73 (13 %) | 3.18 (8 %) |
| **step wall (profiled)** | **6.52** | **21.49** | **39.91** |
| kernel launches per step | 268 | 869 | 1001 |

TP=2 (rank 0 of 2; rank 1 is the same within 1-2 %, `raw/analysis/tp2-rank1-report.json`):

| group | B=1 | B=8 | B=16 |
|---|---:|---:|---:|
| MoE routed experts | - | 4.59 (28 %) | 7.75 (26 %) |
| MoE fused block | 1.50 (25 %) | - | - |
| dense fp8 projections | 1.34 (22 %) | 3.12 (19 %) | 6.13 (21 %) |
| GatedDeltaNet | 0.21 (4 %) | 1.09 (7 %) | 2.02 (7 %) |
| shared expert | - | 1.00 (6 %) | 1.85 (6 %) |
| lm_head | 0.23 (4 %) | 0.64 (4 %) | 1.29 (4 %) |
| attention without projections | 0.22 (4 %) | 0.61 (4 %) | 0.97 (3 %) |
| small bf16 GEMVs | - | 0.75 (5 %) | 3.17 (11 %) |
| all-reduce (custom P2P kernels, `ar2ll_kernel` / `ar2_kernel`) | 0.84 (14 %) | 1.07 (6 %) | 2.17 (7 %) |
| MoE routing glue | - | 0.37 (2 %) | 0.38 (1 %) |
| norms / residual add | 0.28 (5 %) | 0.21 (1 %) | 0.21 (1 %) |
| sampling, embedding, copies | 0.09 (2 %) | 0.26 (2 %) | 0.44 (1 %) |
| idle between launches inside the step | 1.36 (22 %) | 2.90 (17 %) | 3.35 (11 %) |
| **step wall (profiled)** | **6.10** | **16.59** | **29.70** |
| kernel launches per step | 389 | 926 | 1075 |

TP=2 shrinks what is bandwidth-bound (MoE routed 11.57 -> 7.75 ms, fp8 11.18 -> 6.13, lm_head 2.53 -> 1.29, GatedDeltaNet 3.63 -> 2.02)
and does **not** shrink the latency-bound ops: the small bf16 GEMVs stay at 3.17 ms and the shared expert at 1.85 ms. On top come 2.17 ms of
all-reduce. That is why TP=2 gives 1.34x at 16 rows and not 2x.

## 4. Distinct routed experts per layer per step

`INTARSIA_HIP_ROUTE_STATS` on a TP=1 eager server, same workload (192 steps of `same`/`div` phases, 384 tokens per stream). Uniform
independent routing would give `256 * (1 - (1 - 8/256)^B)`.

| rows | uniform formula | measured mean (min-max over layer calls) | same-prompt phase | different-prompt phase | weight passes by the grouped kernel | most pairs on one expert (mean / max) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 8 (by construction; fused block, not recorded) | | | 8 | 1 |
| 8 | 57.4 | 42.0 (15-64), 30 201 layer calls | 39.4-41.1 | 45.9-46.2 | 42.0 | 4.9 / 8 |
| 16 | 101.9 | 68.9 (30-112), 27 547 layer calls | 62.9 -> 70.1 -> 73.0 | 67.2 -> 72.1 -> 73.0 | 69.5 | 8.0 / 16 |

(Phase columns are the means of consecutive windows of 4000 layer calls, about 100 steps; the 16-row value rises during decode as the
streams drift apart. Rows 11-15, seen only in the three-step tail of a phase: 47.4, 50.6, 52.7, 55.1, 57.7.) The routing is concentrated:
at 16 rows the most popular expert of a layer is chosen by 8 of the 16 rows on average, and the early layers spread more than the late ones
(mean distinct at 16 rows: layers 1-5 80-92, layers 9-40 58-76; per layer in `raw/window-tp1-0035/serve-tp1-route.log`; index 0 in that
log is a warm-up call). The grouped kernel reads an expert once per chunk of 8 pairs (`moe.hip:37`), so passes exceed distinct experts by
only 0.8 %. The `same` phases (what `localeval` sends) and the `div` phases route alike, and the MoE kernels take the same time in both
(gate_up 6.88 vs 7.01 ms, down 4.59 vs 4.66 ms per step at 16 rows, `raw/analysis/tp1-same16-vs-div16-moe-times.txt`), so the benchmark
workload is not flattering the MoE here.

## 5. Per-kernel roofline at 16 rows (TP=1)

Bytes are the weight (or state, or KV) bytes the kernel has to stream per call, from the model dimensions; MoE uses 70 distinct experts per
call (section 4), attention a context of 1200 tokens. Roof = 640 GB/s (the lane brief; the best kernels in the trace reach 624-656 GB/s:
GatedDeltaNet state 656, lm_head at M=1 624). TFLOPS are effective FLOPs of the row-wise dot products. Rows are ordered by time per step.

| kernel (calls per step) | us/call | ms/step | bytes/call | GB/s | % of 640 | TFLOPS | bound |
|---|---:|---:|---:|---:|---:|---:|---|
| MoE routed gate_up, `moe_gate_up_g2_kernel` (40) | 173.6 | 6.94 | 82.6 MB | 476 | 74 % | 3.1 | near bandwidth |
| fp8 GEMV N=8192 K=2048, M=8, two passes per layer (80) | 63.5 | 5.08 | 16.8 MB | 264 | 41 % | 4.2 | neither (occupancy / per-row work) |
| MoE routed down, `moe_down_g2_kernel` (40) | 115.6 | 4.63 | 41.3 MB | 357 | 56 % | 2.3 | below bandwidth |
| GatedDeltaNet delta rule, fp32 state read + write (30) | 102.3 | 3.07 | 67.1 MB | 656 | 102 % | 0.3 | at bandwidth roof (state traffic) |
| lm_head NVFP4, 4 launches of M=4 (4) | 632.2 | 2.53 | 286.1 MB | 452 | 71 % | 6.4 | below roof, 4 passes |
| fp8 GEMV N=2048 K=4096, M=5, M=6, three passes (80 + 40) | 28.5 / 32.5 | 2.28 + 1.30 | 8.4 MB | 294 / 258 | 46 / 40 % | 2.9-3.1 | neither |
| fp8 GEMV N=4096 K=2048, M=8, two passes (60) | 36.3 | 2.18 | 8.4 MB | 231 | 36 % | 3.7 | neither |
| bf16 GEMV GDN b/a, `gemv_kernel<16>`, 2 blocks (60) | 23.0 | 1.38 | 0.26 MB | 11 | 2 % | 0.2 | latency (scratch spill, 2 blocks) |
| attention split-KV (10) | 126.1 | 1.26 | 39.3 MB | 312 | 49 % | 2.5 | below bandwidth |
| shared expert gate+up, `shared_glu_rows_kernel` (40) | 28.9 | 1.16 | 1.18 MB | 41 | 6 % | 2.3 | latency (64 blocks) |
| bf16 GEMV router N=256, 16 blocks (40) | 24.2 | 0.97 | 1.05 MB | 43 | 7 % | 0.7 | latency (scratch spill) |
| bf16 GEMV shared-gate logit N=1, 1 block (40) | 23.7 | 0.95 | 0.004 MB | 0 | 0 % | 0.0 | latency (scratch spill, 1 block) |
| shared expert down + combine (40) | 22.3 | 0.89 | 0.59 MB | 26 | 4 % | 1.5 | latency |
| fp8 GEMV N=512 K=2048, M=8 (40) | 8.6 | 0.34 | 1.05 MB | 122 | 19 % | 2.0 | latency |

No kernel exceeds 6.4 TFLOPS effective, and the matrix instructions are not used in the decode step at all (section 6): there is no
ALU or WMMA roof that any of these is near. At 8 rows the same kernels read: gate_up 488 GB/s (76 %, 42 experts), down 395 GB/s (62 %),
fp8 N=8192 M=8 264 GB/s, GatedDeltaNet 647 GB/s (101 %), lm_head 449 GB/s. At 1 row: lm_head 624 GB/s (98 %), fp8 N=8192 M=1 530 GB/s
(83 %), fused MoE block about 340 GB/s.

**Why the fp8 projections sit at 36-46 % of the roof at M=4-8 (code reading plus resource usage, no hardware counters were taken):**
`fp8_gemv_kernel<M>` keeps M x 16 accumulators per weight row per lane; the VGPR count rises with M (M=1: 128, M=4: 184, M=8: 248,
M=9: 256, M=10-12: 256 plus scratch spills of 72 / 160 / 284 bytes per thread) so occupancy falls from about 12 to about 6 waves per SIMD,
and each block first stages and re-quantises all M x K activations into LDS (`fp8.hip:196-225`, M x K x 2 bytes per block, the reason for
the 48 KiB cap behind `kRowsGroup = 12`, `fp8.hip:409` / `:605`) before it reads a weight. The time of a pass grows with M: N=8192, 1 row
31.7 us, 8 rows 63.5 us, for the same 16.8 MB. This is a hypothesis consistent with the data; it was not isolated.

**Weight passes at 16 rows** (`fp8.hip:597-625`, `nvfp4.hip:369-394`): a K=2048 matrix is read in 2 groups of 8 rows, a K=4096 matrix in 3
groups (6 + 5 + 5), the lm_head in 4 groups of 4. The fp8 projections stream 2.9 GB per step at 16 rows (1.6 GB at 8 rows, 1.28 GB once);
the lm_head 1.14 GB (0.29 GB once). That is exactly the doubling from 8 to 16 rows: each pass costs the same at M=8 as before.

## 6. Matrix instructions, NVFP4 dequantisation, kernel structure

* **No kernel of the decode step uses WMMA / MFMA.** The 33 distinct kernel/grid instances of a 16-row step (`raw/analysis/tp1-prof-report-B1-8-16.json`)
  are row-loop GEMVs with scalar FMA or `v_dot2` (`__builtin_amdgcn_fdot2`, `fp8.hip:268`) and one accumulation chain per row. The WMMA
  GEMM kernels (`f8_gemm_kernel`, `gemm_wmma_kernel`, `moe_wmma*.h`) appear in the trace only in prefill chunks. A WMMA rows family exists
  (`INTARSIA_HIP_WMMA_ROWS=1`, `docs/tuning.md:263`, `docs/results/2026-10-25-wg-hip-wmma-gemv.md`) but is off by default. The batched
  step runs with `exact_rows = true` (`crates/intarsia-core/src/model/batch.rs:375`): every row must be bitwise what the solo step gives,
  which is why the kernels are per-row loops. Any replacement has to keep that, or change the contract.
* **NVFP4 (e2m1 values, e4m3 scale per 16 elements, f32 global scale) is dequantised inside the GEMV loops**, from a 16-entry shared-memory
  LUT (`ik_fp4_value` / `ik_fp4x8_to_f32`, `kernels/hip/ik_quant.h:95-112`), once per weight load and reused over the rows held in
  registers: routed experts `moe.hip:564` (LUT), `:614-619` (g2 gate/up scales), `:688`, `:732-734` (g2 down); shared expert
  `moe_block_grouped.hip:51-63` and `:123-133`; lm_head and other dense NVFP4 `nvfp4.hip:23-38`, `:101-115`. The fp8 weights of the
  attention and GatedDeltaNet projections go through `e4m3x2_to_f16x2` (`fp8.hip:39`, used `:258-259`) into `v_dot2`; the activations
  are quantised to fp8 in LDS per block (`fp8.hip:196-225`).
* **MoE routed experts:** a counting sort groups the 8 x B (token, expert) pairs (`moe_group_kernel`, `moe.hip:356`, 2.2 us); the g2
  kernels (`moe.hip:561` gate_up, `:685` down; launch geometry `:1617`, `:1632`) give one block column to each of up to 8 x B active
  slots and read an expert's weights once per chunk of 8 pairs. It is a grouped GEMV, not a GEMM; there is no per-token re-read.
* **Per-row loops that do not amortise:** the shared expert (`moe_block_grouped.hip:27` and `:102`, 64 blocks, 15.3 us at 8 rows ->
  28.9 us at 16) and the bf16 GEMVs (`matmul.hip:82`, 1 / 2 / 16 blocks; `gemv_kernel<8>` 240 VGPR, 5.5 us -> `gemv_kernel<16>` 256 VGPR plus
  440 bytes of scratch per thread, 23.0 us).

## 7. Attribution of 8 -> 16 rows, and which kernels scale with rows (TP=1)

| group | B=8 ms | B=16 ms | delta ms | share of the +18.4 ms | ratio | what scales |
|---|---:|---:|---:|---:|---:|---|
| dense fp8 projections | 5.48 | 11.18 | +5.70 | 31 % | 2.04x | weight passes (1 -> 2, 2 -> 3), constant cost per pass |
| MoE routed experts | 6.57 | 11.57 | +5.00 | 27 % | 1.76x | distinct experts 42 -> 70 (1.67x), bytes |
| small bf16 GEMVs | 0.76 | 3.30 | +2.53 | 14 % | 4.31x | M=16 spills registers (superlinear) |
| GatedDeltaNet | 1.97 | 3.63 | +1.66 | 9 % | 1.84x | per-sequence fp32 state, bytes (at roof) |
| lm_head | 1.27 | 2.53 | +1.26 | 7 % | 1.99x | weight passes 2 -> 4 |
| shared expert | 1.09 | 2.05 | +0.96 | 5 % | 1.89x | per-row serial loops |
| attention without projections | 0.95 | 1.79 | +0.84 | 5 % | 1.89x | KV bytes per sequence |
| idle between launches | 2.73 | 3.18 | +0.45 | 2 % | 1.17x | launches 869 -> 1001 (about 3.2 us each) |
| routing glue, norms, sampling, other | 0.65 | 0.68 | +0.03 | 0 % | 1.05x | flat |
| **step** | 21.49 | 39.91 | +18.41 | 100 % | 1.86x | |

Everything that is not flat grows with the rows; the only flat items (about 0.7 ms) are tiny. Two items grow faster than the rows: the
small bf16 GEMVs (spills) and, at B > 12, the fp8 pass count. Four items are structurally per-row in this design: the GatedDeltaNet
state (2.1 MB per sequence per layer read and written, fp32), the KV bytes, the shared-expert loops, and the MoE experts (sub-linear only
because the rows overlap in routing). Nothing in the step gets cheaper per row as rows are added except the first pass over the dense
weights, and the kernels do not use it (they re-read at every group).

## 8. Ranked bottlenecks at 16 rows (TP=1), with ceilings

Ceiling = ms per step saved at 16 rows if the kernel ran at the roof on its real bytes (640 GB/s) or at the latency floor of its few bytes.

| # | bottleneck | now | ceiling | file:line | evidence |
|---|---|---:|---:|---|---|
| 1 | Dense fp8 projections: 2.9 GB streamed per step instead of 1.28 GB, at 40 % of the roof | 11.18 ms | **-9.2 ms** (single pass at roof: 2.0 ms); -6.2 ms if only the pass count is fixed at today's per-pass rate | `kernels/hip/fp8.hip:409` (`kRowsGroup = 12`), `:597-625` (launcher, groups), `:232-271` (accumulation, VGPR 248), `:196-225` (activation staging); call sites `gdn_batch.rs:59`, `batch.rs:842` | section 5; per-pass 63.5 us at M=8 vs 31.7 us at M=1; step doubles 8 -> 16 with unchanged per-pass cost |
| 2 | Latency-bound small ops: bf16 GEMVs (3.30 ms) and shared expert (2.05 ms) move 75 MB per step | 5.35 ms | **-5.0 ms** (about 0.3 ms left) | `kernels/hip/matmul.hip:82` (`gemv_kernel`, `:330-339` rows launcher), `moe_block_grouped.hip:27`, `:102`; `quantmoe.rs:372` | 2-16 blocks of 256 threads; M=16 instantiation 440 B/thread scratch; per-call 22-29 us for 0.004-1.2 MB; unchanged at TP=2 |
| 3 | MoE routed experts below the bandwidth roof (74 % gate_up, 56 % down) on 70 distinct experts | 11.57 ms | **-3.8 ms** (-3.3 ms at 600 GB/s) | `kernels/hip/moe.hip:561` (gate_up g2), `:685` (down g2), `:1617`, `:1632` (launch) | 123.9 MB per layer; the floor is 7.7 ms, not the 11.3-12.1 ms of l7 |
| 4 | lm_head in four M=4 passes (the M=8 instantiation spills: 3.2 ms for 8 rows) | 2.53 ms | **-2.1 ms** (one pass: 0.45 ms) | `kernels/hip/nvfp4.hip:369-394` (`kRowsGroup = 4`, groups) | 4 x 632 us; single row reaches 98 % of the roof |
| 5 | 1001 launches per step, about 3.2 us idle at every dependent boundary | 3.18 ms | about **-1.5 ms** (halve the launches) | the per-layer launch chains, e.g. `fp8_gemv_kernel` 300 launches, MoE glue 120, norms 101 | idle inside the graph: 0.85 / 2.73 / 3.18 ms at 268 / 869 / 1001 launches; unprofiled bound: step 38.8 - kernel busy 36.7 = 2.1 ms |

If all five landed the 16-row step would be about 18 ms (about 880 tok/s decode-only against 412 now); the first two are 65 % of that.
Not a kernel fix but the largest fixed lever left after these: the GatedDeltaNet state traffic (3.07 ms, at the roof) scales with
rows and sits on the fp32 state (`mamba_ssm_dtype float32`); a bf16 state would halve it (-1.5 ms) at a numerics cost.

TP=2 only: the second all-reduce of every layer (kind 2, 40 per step) is a 19.4 us tagged-line (`ar2ll_kernel<2>`) call at 8 rows and
falls onto the 43.1 us wide path (`ar2_kernel<2>`) at 16 rows, where the message reaches the 64 KiB limit (`comm.rs:29`
`LL_MAX_DEFAULT`, `kernels/hip/allreduce.hip:333` `ll_fits`, `:355`). With `INTARSIA_HIP_AR_LL_MAX=262144` it stays on the tagged-line
path at 26.5 us: all-reduce 2.17 -> 1.57 ms and the profiled step 29.70 -> 29.01 ms (-0.7 ms, -2.3 %; one server, one run).

## 9. Host side

* **Launches per step** (decode, graph replay): TP=1 268 (1 row) / 869 (8) / 1001 (16); TP=2 389 / 926 / 1075 per rank.
* **Batched decode is captured in a HIP graph**, one per row count, replayed with `hipGraphLaunch`
  (`crates/intarsia-core/src/model/graph_step.rs:296-381`, `INTARSIA_BATCH_GRAPHS` default on). The trace shows the replayed kernels; the
  step wall equals the sum of kernel times plus the in-graph gaps of section 3.
* **Host cost per decode step** (host-profile, 16 rows, TP=1): upload 0.009 ms, graph launch 0.024 ms, detok + send 0.024 ms, everything
  else under 0.003 ms; the rest of the 38.7 ms is the wait for the GPU (sync 38.65 ms). `gpu-idle-gap` 0.028 ms per step at TP=1, 0.026 ms
  at TP=2 (the trace measures 10 us from one step's last kernel to the next step's first). Scheduler, sampling and SSE are not a factor.
* **Sampling** is on the GPU: `argmax_kernel<float>` 42 us per step at 16 rows (0.1 %), a 64-byte download.
* **Prefill interleaving:** a 1k-token prefill is one iteration of about 130 ms (`decode+prefill` iterations in the host-profile); it
  stalls the other streams' decode and is what turns the 412 tok/s decode rate into the measured 371 tok/s aggregate (section 2).

## 10. What this changes in earlier results

* hr3 / l7 called the fp8 projections ALU-bound from re-decoding weights per row tile. The trace says the time is the number of weight
  passes (two or three at 16 rows) times a per-pass cost that already sits at 36-46 % of the roof, with VGPR-limited occupancy; FLOPs are
  4-7 TFLOPS effective.
* l7's routed-expert floor of 11.3-12.1 ms assumed about 100 distinct experts (uniform routing). Real routing gives about 70 and a
  floor of 7.7 ms; the kernels are at 74 % / 56 % of it, not at the floor.
* hg (2026-10-25) measured 3.5 ms of host launch per 42.2 ms decode iteration; on this tree the graph launch is 0.024 ms and the iteration 38.7 ms. What changed was not investigated; the host is not on the critical path now.

## 11. Limits

* One run per configuration. The step-to-step spread inside a run is small (same16 vs div16 bursts: 39.74 vs 40.17 ms; B=8 bursts
  21.27 vs 21.68 ms); run-to-run variation was not measured.
* `rocprofv3` inflates the step (section 1), most at TP=2 and small row counts; per-kernel times inside it may be inflated too. The
  idle-gap figure (3.2 us per launch boundary) is the least certain: part of it may be the profiler's.
* The roof of 640 GB/s is the brief's; observed peaks are 624-656 GB/s. Bytes of the routed experts use the mean of 70 distinct experts from
  a separate eager run of the same workload (63-73 over the decode); attention bytes assume 1200 tokens of context (the steady steps have
  1.0-1.45k). Percentages of the roof move by about +-5 points with those inputs.
* The explanation of the low fp8 efficiency (occupancy, staging) is from code and resource usage, not from counters.
* Only the TP=1 row sweep, MoE statistics and the LL_MAX test go beyond the 1 / 8 / 16 rows asked for. The distinct-expert numbers are TP=1
  only (routing is the same at TP=2). The B=1 row of the TP tables is the solo-step path of a `--max-batch 16` server.

## Reproduce

`scripts/hprof/build.sh` (fleet job, `--load cpu`), then `ARM=tp1 STEPS="prof route plain" scripts/hprof/window.sh` and
`ARM=tp2 STEPS="prof plain" scripts/hprof/window.sh` under `fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --quiet gpu,cpu`;
the sweep was `ARM=tp1 STEPS=prof TAGSUF=-sweep PHASES="same2,same4,same6,same10,same12,same13,same14"`, the all-reduce test
`ARM=tp2 STEPS=prof TAGSUF=-llmax XENV="-e INTARSIA_HIP_AR_LL_MAX=262144" PHASES="same8,same16"`. Analysis:
`python3 scripts/hprof/analyze.py {explore,steps,dump,report} <k_kernel_trace.csv.gz> ...`.

Raw data in `docs/results/2026-10-30-hprof-r9700-16stream-profile/raw/`: `window-tp1-0035/` (prof, route, plain: kernel traces `prof-tp1-prof/`,
`serve-*.log` with host-profile and route-stats, `drive-*.jsonl`, `localeval-tp1-plain.log`), `window-tp2-0041/`, `window-tp1-0046/` (sweep),
`window-tp2-0049/` (LL_MAX), `analysis/` (reports as JSON, step tables, VGPR / LDS / scratch of every kernel), `build.log`, `fleetq-run-*.log`.
