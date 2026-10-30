# MPROF: Metal 35B batched decode, where one 16-row step goes (2026-10-30)

Lane `mprof`, branch `lane-mprof/profile16`, base `main` `d4936a3d4`. endor (Apple M5 Max 40-core, 128 GB, macOS 27.0), Metal,
Qwen3.6-35B-A3B NVFP4 (`~/models/Qwen3.6-35B-A3B-NVFP4`), MTP off. **Measurement only: no engine code changed** (the binary and the
test are `d4936a3d4` as built; only scripts and data are added). Raw traces, logs and scripts: [raw/](2026-10-30-mprof-metal-16stream-profile/raw/)
(`raw/scripts/` has every driver and analyser, `raw/analysis/` the generated tables quoted here). Every GPU run took a fleetq slot on
endor (jobs in section 12).

## 0. Result in one page

**Where a 16-row decode step goes.** Harness step (continuous batching engine bypassed, `Model::forward_batch`, 16 rows at ~1.05k
context, 2.3 s per cell): **35.9 ms plain** (446 tok/s); 37.0 ms summed over in-stream per-dispatch records (`ce1`, which costs 3 %
over plain). Served, in the sustained 16-stream burst: **42.5 ms** per step (377 tok/s), see section 6. Share of the 37.0 ms:

| op group | ms / step | share |
|---|---:|---:|
| MoE routed experts (gate-up 9.07, down 5.23, gather 0.22, combine 0.13) | 14.65 | 40 % |
| dense fp8 projections (GDN qkv / z / out, attention q+gate / k / v / o) | 10.15 | 27 % |
| lm_head (nvfp4 248320x2048) + argmax | 4.17 | 11 % |
| GDN layers (delta rule 2.27, in_a/in_b pair 0.34, conv 0.19, gated norm 0.19) | 2.98 | 8 % |
| full-attention core (split-KV decode 1.76, qk_norm_rope_kv 0.08, gates 0.03) | 1.88 | 5 % |
| shared expert (nvfp4 gate / up / down + silu) | 1.77 | 5 % |
| router + shared-gate GEMV + top-k | 0.94 | 3 % |
| norms / embedding | 0.48 | 1 % |

GPU idle / host gaps are not a factor: 0.08 ms per step at 16 rows with profiling off (host profile), 0.7 ms (1.7 %) in `cb` profile
mode, which inflates it. Sampling is a device argmax of 0.03 ms. Section 8.

**Why 8 -> 16 streams adds only about +4 %.** Two separate effects, neither of them the host:

1. *The step does not amortize.* Doubling the rows takes the step from 20.7 to 37.0 ms (**x1.79**; plain 20.0 -> 35.9, x1.79; served
   23.2 -> 42.5 ms, x1.83), so decode-only throughput rises only +11 % (harness) / +9 % (served). Of the +16.3 ms: **8.2 ms (50 %)**
   is the dense multi-row GEMV family (fp8 +4.68, lm_head +2.08, shared expert +0.91, router/pair +0.46), which has no 9..16-row
   kernel: `row_chunks` (`crates/intarsia-backend-metal/src/lib.rs:237-273`) runs 9..16 rows as two launches of at most 8 and
   the second launch costs the same as the first (fp8: 5.5 ms at 8 rows, 10.0-10.2 ms at 9, 12 and 16 rows, so it is *per
   launch*, not per row); **6.1 ms (38 %)** is the routed experts, whose distinct-expert weight bytes grow x1.5 and whose pair
   count grows x2; **2.0 ms (12 %)** is per-row state and KV traffic (GDN delta rule +1.15, attention +0.83), which is inherent.
2. *The served number includes prefill.* localeval's "aggregate decode tok/s" is completion tokens over the **wall** of the whole
   burst. The 16 prompts are prefilled one at a time (16 x 245 ms), with only two short decode steps between prefills, so at
   16 streams 19 % of the wall is prefill, ramp and drain (16 % at 8 streams). Steady decode is 79 % of the wall.

   | | 8 streams | 16 streams | 16 / 8 |
   |---|---:|---:|---:|
   | steady full-batch step, wall | 23.15 ms | 42.47 ms | x1.83 |
   | decode-only tok/s (rows / step) | 345.6 | 376.7 | **+9.0 %** |
   | prefill + ramp + drain share of the burst wall | 16.1 % | 19.0 % | |
   | aggregate tok/s, same trace (`ce1` mode) | 294.1 | 314.1 | +6.8 % |
   | aggregate tok/s, mean of 5 plain/profile server runs (section 6) | 302 | 315 | **+4.2 %** |

**Top bottlenecks at 16 rows, ranked by what a roof-speed kernel would save** (floor = unique bytes / 580 GB/s, the read bandwidth
measured here, section 4; all ms per 37.0 ms step, harness regime; the served sustained step is 10-15 % larger everywhere):

| # | bottleneck | now | floor | saves | where |
|---|---|---:|---:|---:|---|
| 1 | fp8 projections: two 8-row launches, each at 235-290 GB/s (about 45 % of attainable) and each re-reading 1.28 GB of weights | 10.15 ms | 2.2 ms | **7.9 ms** | `kernels/metal/fp8.metal:132-231` (`_xd`, M 2..4 per launch), `:304-382` (`fp8_rows_mmt`, M 5..8), `crates/intarsia-backend-metal/src/fp8.rs:356,502`, `lib.rs:237-273` |
| 2 | routed experts: gate-up at 412 GB/s of distinct bytes (71 %), down tracks *pairs* not distinct experts | 14.30 ms | 9.7 ms | **4.6 ms** | `kernels/metal/moe.metal:556-640` (per-pair kernels `moe_dec_gu_p` / `moe_dec_dn_p`), `src/moe.rs:232-243`, `lib.rs:252` |
| 3 | lm_head: scalar FMA, time proportional to rows (0.26 ms per row), 12 % of the nvfp4 byte roof | 4.14 ms | 0.5 ms | **3.6 ms** | `kernels/metal/nvfp4.metal:81-147` (`nv_gemv_impl`, 8-row form), `src/nvfp4.rs:28-45,170-185` |
| 4 | small-N nvfp4 / bf16 GEMVs (shared expert 512x2048 + 2048x512, router 257x2048, GDN in_a+in_b 64x2048): one-row shader looped over rows, chunked | 2.6 ms | 0.2 ms | **2.4 ms** | `src/nvfp4.rs:28-45,176-182` (`nv_rows_mb`, `NARROW_N` 4096), `kernels/metal/gemv_multirow.metal:79-99,218` |
| 5 | attention decode: 36 % of the KV read roof (4 head-split threadgroups re-read each KV head, short rows idle) | 1.76 ms | 0.63 ms | **1.1 ms** | `kernels/metal/attention_splitkv.metal:532-570`, `src/attn.rs:20-78`, `src/rowtab.rs:419-433` |

The five add up to 19.7 ms, which would put the 16-row step at about 17 ms (930 tok/s decode-only); the unique-byte floor of the
whole step is 9.7 GB (7.3 GB weights, 2.0 GB GDN state read+write, 0.36 GB KV) = 16.7 ms. **Not a bottleneck:** GDN delta rule
(2.27 ms, already at the bandwidth roof), host work, sampling. **Constraint every kernel lane inherits:** all multi-row kernels are
`rows_exact` (each row bitwise the solo step, `crates/intarsia-core/src/registry.rs:356-364`, lane mb), and the roofs above assume that is
relaxed or reproduced; the NAX / `mpp::tensor_ops` kernels are registered but unreachable in a decode batch. **And one trap for
A/B work:** every kernel runs 10-15 % slower after 20-30 s of sustained load than in a 2-3 s harness cell (section 7).

## 1. Method

* **Harness.** `crates/intarsia-models/tests/batch_metal_real.rs::real_35b_batch_step_real_traffic` (lane mm): B sequences each
  prefilled with a ~1000-token slice of `docs/*.md` (`1000 + 7 i` tokens, different offsets), then greedy `forward_batch` steps
  (each row's argmax fed back), 4 warm-up + 32 measured steps per B, 1.5 s apart. Row i has context about `1000 + 7 i + 18 .. 200`
  tokens. B = 1, 4, 8, 9, 12, 16 (9 shows the chunking cliff, 12 the middle), run twice in opposite B order (`raw/w1`
  ascending, `raw/w1b` descending), the tables below are the mean of the two. B = 1 through `forward_batch` is not the served
  solo path (that one uses token feedback); its 8.10 ms matches the served one-row step (8.11 ms, section 8) and MDEC's 8.1 ms.
* **Trace modes** (`INTARSIA_METAL_PROFILE`, `crates/intarsia-backend-metal/src/profile.rs`; none of this is new code):
  * `cb:` + `INTARSIA_METAL_COMMIT_EVERY=1` (**`ce1`**): normal stream, one command buffer per dispatch, GPU timestamps. Consecutive
    buffers can overlap on the GPU, so a record is charged only the time after the previous one ended ("exclusive"); the sum
    over a step equals the step within 3 % (below). **Primary per-op source.** Caveat: two independent launches that run
    concurrently (GDN qkv and z; shared-expert gate and up) are charged jointly to the first, so single rows of such pairs
    (and tiny kernels such as the shared `silu_glu`, 0.00 at B 4/8) are approximate; group totals are exact.
  * `cb:` (**`cb`**): normal stream, committed buffers as they are; host open / commit times. For command-buffer counts and gaps.
  * `kernel:` (**`kernel`**): each launch its own buffer, waited on. **Used only for the routing labels** (`n` = distinct experts, `k`
    = pairs on each MoE launch, `lib.rs:295-309`). Its durations are not usable at B <= 8: the GPU idles between waited launches and
    drops its clock (tiny `rmsnorm_add` 19.6 us at B=1 vs 6.0 us at B=16 in the same run; sums of 31 / 28 / 21 / 36 / 32 / 43 ms at
    B 1 / 4 / 8 / 9 / 12 / 16 against 8 .. 36 ms plain; `raw/analysis/groups_kernel_mode_desc.md`).
* **Served.** `intarsia serve --max-batch 16 --max-seq 8192` (recipe `prefill_chunk` 4096), `localeval speed --sizes 1024 --gen 512
  --reps 1 --concurrency 1,8,16` (CHUNK4K's burst method), one fresh server per arm: plain x2, `INTARSIA_HOST_PROFILE=1`, `cb`, `ce1`,
  and `kernel` (16 streams, gen 128, routing labels only). The 16-stream localeval prompts are one filler text with only a nonce prefix.
* **Roofs measured here** (`raw/scripts/roof/`, `raw/roof.txt`, a standalone Metal micro-benchmark): DRAM read **580 GB/s** max
  (566 median; 1-3 GiB buffers; nominal 614), fp32 FMA **15.5 TFLOPS**, `simdgroup_float8x8` MMA **16.4 TFLOPS** (f32) and 15.8
  (f16 inputs): the classic simdgroup matrix op is *not* faster than scalar FMA on this GPU; the matrix speedup is only the NAX /
  `mpp::tensor_ops` path (MLX bf16 GEMM about 51 TFLOPS, `docs/research/metal-prefill-plan.md` section 2).
* **Profiling overhead** (harness, mean ms / step): plain 8.10 / 14.05 / 20.02 / 26.00 / 30.48 / 35.91 at B 1 / 4 / 8 / 9 / 12 / 16
  (5 runs, spread <= 2 %, independent of B order); `cb` mode 9.06 / 15.65 / 21.40 / 27.40 / 32.79 / 40.24 (**+12 % at 16**, so `cb`
  GPU times are not step times); `ce1` 8.53 / 14.88 / 21.26 / 28.36 / 32.52 / 37.37 in the descending run (+4 % at 16; the ascending
  run's B=1 cell was 15.05 ms, host-bound in that run's 770 single commits, unexplained, and not visible in the GPU record sums used below; the B=1 sum 8.33 ms is +3 % over plain like the other cells).

## 2. Step time and per-op breakdown, batch 1 / 4 / 8 / 16

Plain step (`raw/analysis/plain_step_times.md`): 

| B | ms / step | tok/s | ms per row-token | step vs B=1 | dispatches / step | command buffers / step |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8.10 | 124 | 8.10 | x1.00 | 764 | 7 |
| 4 | 14.05 | 285 | 3.51 | x1.73 | 804 | 8 |
| 8 | 20.02 | 400 | 2.50 | x2.47 | 804 | 8 |
| 9 | 26.00 | 346 | 2.89 | x3.21 | 1125 | 10 |
| 12 | 30.48 | 394 | 2.54 | x3.76 | 1125 | 10 |
| 16 | 35.91 | 446 | 2.24 | x4.43 | 1125 | 10 |

Eight rows cost 3.2x less per token than one; 9 rows are *slower in total throughput* than 8 (346 vs 400 tok/s) because the chunked
launches start at 9; 16 rows are only 11.5 % above 8.

Per-op breakdown (`ce1`, in-stream exclusive ms / step, mean of two runs and 32 steps each; share of the step in parentheses;
dispatch counts: B 1 = 764, B 4-8 = 804, B 9-16 = 1125). `raw/analysis/breakdown_table.md`, generator `raw/scripts/mk_breakdown.py`:

| op group | B=1 | B=4 | B=8 | B=9 | B=12 | B=16 | 8 -> 16 (ms, x) |
|---|---|---|---|---|---|---|---|
| **MoE routed experts** | 1.67 (20%) | 4.53 (32%) | 8.51 (41%) | 9.61 (35%) | 11.75 (37%) | 14.65 (40%) | +6.14, x1.72 |
| &emsp;experts gate-up (fused silu*up) | 0.84 | 2.62 | 5.34 | 6.14 | 7.41 | 9.07 | +3.74, x1.70 |
| &emsp;experts down | 0.71 | 1.58 | 2.84 | 3.13 | 4.00 | 5.23 | +2.39, x1.84 |
| &emsp;expert gather / grouping | 0.00 | 0.21 | 0.21 | 0.22 | 0.22 | 0.22 | +0.01, x1.05 |
| &emsp;combine (weighted sum + shared) | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.13 | +0.00, x1.01 |
| **dense fp8 projections** | 3.59 (43%) | 5.07 (35%) | 5.47 (26%) | 10.02 (36%) | 10.04 (32%) | 10.15 (27%) | +4.68, x1.86 |
| &emsp;fp8 8192x2048 (GDN qkv x30, attn q+gate x10) | 1.99 | 2.10 | 2.32 | 4.28 | 4.47 | 4.49 | +2.17, x1.93 |
| &emsp;fp8 2048x4096 (GDN out x30, attn o x10) | 1.19 | 1.64 | 1.87 | 3.25 | 3.14 | 3.22 | +1.35, x1.72 |
| &emsp;fp8 4096x2048 (GDN z x30) | 0.36 | 1.02 | 1.03 | 1.93 | 1.94 | 1.95 | +0.92, x1.89 |
| &emsp;fp8 512x2048 (attn k, v x10 each) | 0.05 | 0.31 | 0.25 | 0.55 | 0.49 | 0.48 | +0.24, x1.97 |
| **lm_head + argmax** | 0.53 (6%) | 1.15 (8%) | 2.09 (10%) | 2.52 (9%) | 3.22 (10%) | 4.17 (11%) | +2.08, x2.00 |
| &emsp;lm_head nvfp4 248320x2048 -> f32 | 0.51 | 1.13 | 2.06 | 2.50 | 3.19 | 4.14 | +2.08, x2.01 |
| **GDN layers** | 0.76 (9%) | 1.06 (7%) | 1.60 (8%) | 1.86 (7%) | 2.28 (7%) | 2.98 (8%) | +1.38, x1.86 |
| &emsp;delta rule (recurrent state) | 0.32 | 0.63 | 1.12 | 1.26 | 1.65 | 2.27 | +1.15, x2.03 |
| &emsp;in_a + in_b bf16 pair GEMV (64x2048) | 0.12 | 0.14 | 0.18 | 0.28 | 0.30 | 0.34 | +0.16, x1.90 |
| &emsp;conv1d + state | 0.14 | 0.11 | 0.13 | 0.14 | 0.16 | 0.19 | +0.05, x1.38 |
| &emsp;gated norm: rmsnorm | 0.10 | 0.09 | 0.09 | 0.09 | 0.09 | 0.10 | +0.01, x1.12 |
| &emsp;gated norm: silu_glu | 0.08 | 0.09 | 0.08 | 0.08 | 0.08 | 0.09 | +0.01, x1.14 |
| **full attention (core)** | 0.36 (4%) | 0.72 (5%) | 1.05 (5%) | 1.15 (4%) | 1.43 (5%) | 1.88 (5%) | +0.83, x1.80 |
| &emsp;attention decode split-KV | 0.26 | 0.60 | 0.93 | 1.03 | 1.32 | 1.76 | +0.83, x1.89 |
| &emsp;qk_norm_rope_kv | 0.07 | 0.08 | 0.08 | 0.07 | 0.08 | 0.08 | +0.00, x1.05 |
| **shared expert** | 0.46 (6%) | 0.77 (5%) | 0.86 (4%) | 1.34 (5%) | 1.51 (5%) | 1.77 (5%) | +0.91, x2.06 |
| &emsp;nvfp4 gate / up 512x2048 | 0.24 | 0.46 | 0.45 | 0.67 | 0.79 | 0.93 | +0.48, x2.05 |
| &emsp;nvfp4 down 2048x512 | 0.20 | 0.31 | 0.40 | 0.57 | 0.62 | 0.73 | +0.33, x1.81 |
| &emsp;silu_glu | 0.03 | 0.00 | 0.00 | 0.10 | 0.10 | 0.10 | +0.10, x0.00 |
| **router + top-k** | 0.53 (6%) | 0.56 (4%) | 0.64 (3%) | 0.78 (3%) | 0.83 (3%) | 0.94 (3%) | +0.31, x1.48 |
| &emsp;router + shared-gate GEMV (257x2048) | 0.20 | 0.23 | 0.30 | 0.44 | 0.50 | 0.60 | +0.30, x2.02 |
| &emsp;top-k | 0.33 | 0.33 | 0.34 | 0.33 | 0.34 | 0.34 | +0.00, x1.01 |
| **norms / embedding** | 0.43 (5%) | 0.45 (3%) | 0.48 (2%) | 0.48 (2%) | 0.47 (2%) | 0.48 (1%) | +0.01, x1.01 |
| &emsp;rmsnorm_add (2 per layer) | 0.42 | 0.45 | 0.47 | 0.47 | 0.47 | 0.48 | +0.01, x1.01 |
| **total (sum of in-stream record times)** | 8.33 | 14.30 | 20.69 | 27.75 | 31.55 | 37.03 | +16.34, x1.79 |

Reading it:

* **Flat in B** (cost is launch / latency): top-k (0.33 ms for 40 launches), rmsnorm_add (0.42-0.48 for 80), combine, expert gather, gated
  norm, qk_norm_rope_kv, argmax. About 330 byte-free dispatches, 1.8 ms at 16 rows (4.9 %).
* **Step function at 9** (second launch): fp8 projections 5.47 -> 10.02 ms from 8 to 9 rows, then flat to 16 (10.04, 10.15); shared expert,
  router, in_a/in_b and lm_head step up the same way but keep growing with rows afterwards.
* **Linear in rows:** lm_head (0.51, 1.13, 2.06, 2.50, 3.19, 4.14: 0.26 ms per additional row), GDN delta rule (0.32 .. 2.27, 0.14 per row, state
  bytes), attention decode (0.26 .. 1.76, KV bytes).
* **Sub-linear but not amortized:** routed experts 1.67 -> 14.65 ms (x8.8 for 16x the rows, x1.72 from 8 to 16).
* The one-row kernels are better than their multi-row siblings on the same bytes: fp8 at B=1 is 3.59 ms (356 GB/s of 1.28 GB) against
  5.07 ms for the 4-row `_xd` form on the same weights, so the multi-row forms spend about a third more time than pure weight streaming
  needs even before row 5.

## 3. Distinct routed experts per layer per step

Read from the `kernel`-mode MoE launch labels (`n` = distinct expert ids with at least one pair, `k` = valid pairs; 40 MoE launches
per step, 32 steps per cell). Uniform-routing reference: `256 (1 - (1 - 8/256)^B)`. Per-expert weight bytes: 1.77 MB (gate-up
1024x2048 fp4 + scales 1.18 MB, down 2048x512 0.59 MB), so 566 MB per step at B=1 (matches MDEC).

| B | pairs / layer | harness distinct / layer (asc run; desc run) | min .. max over layers and steps | formula | measured / formula | expert MB / step (mean of the two runs) | formula MB / step |
|---:|---:|---|---|---:|---:|---:|---:|
| 1 | 8 | 8.0 ; 8.0 | 8 .. 8 | 8.0 | 1.00 | 566 | 566 |
| 4 | 32 | 26.4 ; 27.7 | 15 .. 32 | 30.5 | 0.89 | 1918 | 2162 |
| 8 | 64 | 49.7 ; 50.5 | 35 .. 63 | 57.4 | 0.87 | 3548 | 4065 |
| 9 | 72 | 54.3 ; 54.2 | 35 .. 70 | 63.6 | 0.85 | 3843 | 4505 |
| 12 | 96 | 66.8 ; 65.3 | 43 .. 91 | 81.1 | 0.81 | 4677 | 5742 |
| 16 | 128 | 81.5 ; 76.9 | 50 .. 114 | 102.0 | 0.78 | 5607 | 7219 |
| **served, 16 streams** (filler prompts, gen 128, `raw/w2/s-kernel`) | 128 | **63.2** | 34 .. 105 | 102.0 | **0.62** | 4475 | 7219 |

* Distinct experts grow **x1.5-1.6 from 8 to 16 rows** in the harness (x1.64 / x1.52 in the two runs), not x2: the expert bytes per step
  go from 3.5 to 5.6 GB. The served burst reads 63 distinct at 16 rows against 77-82 in the harness (no 8-row served routing run was taken). 1.57-1.66 pairs share each expert at 16 rows.
* Routing is correlated and workload dependent: the docs corpus (different offsets of one markdown set) reads 78 % of the uniform
  formula at B=16; the near-identical served filler prompts read 62 %. Per layer, the first layers are the least concentrated
  (95-96 distinct at B=16 in layers 0-2, 65-78 in layers 35-39). **Expert bytes per step, and hence the MoE share, depend on
  the text**, which is why the served step can have a lower MoE traffic than the harness and still be slower (section 7).
* MoE time against these counts (12 cells, `raw/analysis/moe_fit.txt`): gate-up time is fitted by the distinct bytes (2.48 us per MB
  = 403 GB/s, max residual 0.37 ms) better than by pairs (0.69); **down is fitted by the pair count** (0.94 us per pair per layer
  set, max residual 0.08 ms) and not by distinct bytes (0.42): the per-pair kernels let each pair stream its expert's block, and
  only part of the second read of a shared expert hits the cache (`moe.metal:556-569`).

## 4. Per-kernel roofline at batch 16 (and 8)

Time = `ce1` exclusive, mean of the two runs. "Stream MB" = bytes the launches ask for (weights counted once per launch, so a chunked op counts
them twice); "unique MB" = once per step. nvfp4 weights include their 1 byte per 16 scale (x1.125); MoE bytes are the measured distinct experts;
attention bytes are the KV read (`ctx x 2048 B` per row and layer, 10 layers); delta-rule bytes are the f32 state read + write
(2 MiB per row and layer, 30 layers). Activations are < 1 % and omitted. Roof = 580 GB/s measured read bandwidth. FLOPs are 2 M N K
(experts: 6.29 MFLOP per pair). `raw/analysis/roofline_ce1.md` has B = 1 / 4 / 8 / 16, generator `raw/scripts/an_roof.py`.

| kernel (B=16) | ms | launches | us / launch | stream MB | unique MB | GB/s unique | % of 580 | TFLOPS | bound by |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| experts gate-up | 9.07 | 40 | 227 | 3738 | 3738 | 412 | 71 % | 2.4 | bytes of distinct experts; short of roof, per-pair re-reads |
| experts down | 5.23 | 40 | 131 | 1869 | 1869 | 358 | 62 % | 2.1 | pair count (each pair streams its block) |
| fp8 8192x2048 (GDN qkv, attn q+gate) | 4.49 | 80 | 56 | 1342 | 671 | 150 (299 streamed) | 26 % | 4.8 | neither: launch-bound at 235-300 GB/s, 2 launches |
| lm_head nvfp4 248320x2048 f32 | 4.14 | 2 | 2069 | 572 | 286 | 69 (138 streamed) | 12 % | 3.9 | ALU: scalar FMA 25 % of the 15.5 TFLOPS peak, 0.26 ms per row |
| fp8 2048x4096 (GDN out, attn o) | 3.22 | 80 | 40 | 671 | 336 | 104 (208) | 18 % | 3.3 | as above |
| delta rule (state f32 read+write) | 2.27 | 30 | 76 | 2013 | 2013 | 889 | 153 % | 0.9 | bandwidth: at roof (see note) |
| fp8 4096x2048 (GDN z) | 1.95 | 60 | 33 | 503 | 252 | 129 (258) | 22 % | 4.1 | as fp8 |
| attention decode split-KV | 1.76 | 10 | 176 | 364 | 364 | 207 | 36 % | 1.7 | latency / redundant reads, not bytes |
| nvfp4 gate/up 512x2048 (shared) | 0.93 | 160 | 5.8 | 94 | 47 | 51 (101) | 9 % | 2.9 | launch / per-row loop |
| nvfp4 down 2048x512 (shared) | 0.73 | 80 | 9.1 | 47 | 24 | 32 (65) | 6 % | 1.8 | as above |
| router + shared-gate bf16 pair 257x2048 | 0.60 | 80 | 7.5 | 84 | 42 | 70 (140) | 12 % | 1.1 | launch-bound |
| fp8 512x2048 (attn k, v) | 0.49 | 40 | 12 | 42 | 21 | 43 (86) | 7 % | 1.4 | launch-bound |
| in_a + in_b bf16 pair 64x2048 | 0.34 | 60 | 5.6 | 16 | 8 | 23 (47) | 4 % | 0.4 | launch-bound |
| rmsnorm_add, top-k, gather, conv, combine, gated norm, silu, qk_norm_rope_kv, argmax | 1.81 | about 330 | 2.6-8.6 | 0 | 0 | | | | launch / latency (byte-free) |

* **Nothing at 16 rows is compute-bound against a matrix roof**: the FLOPs of the whole step are about 89 GFLOP (fp8 41, lm_head
  16, experts 32), 1.7 ms at the MLX bf16 GEMM rate (51 TFLOPS) and 5.7 ms at the 15.5 TFLOPS FMA peak. The compute roof only matters
  for the *scalar* kernels, which today run at 2-5 TFLOPS (13-31 % of FMA peak, which also has to pay for the fp8 / fp4 decode).
* **fp8:** per launch 5.0-5.5 ms (1.28 GB, 235-255 GB/s) for 4, 5, 6 or 8 rows, i.e. independent of rows; B=1's one-row kernel does the same bytes in
  3.6 ms. So the 8 -> 16 fp8 growth is the second launch, not row arithmetic.
* **MoE** is the only large group close to a byte roof (62-71 % of 580 GB/s on distinct bytes), and a hard floor exists: 5.6 GB / 580 = 9.7 ms against 14.3 ms.
* **Delta rule**: 2.0 GB of state traffic in 2.27 ms is 889 GB/s, above any DRAM roof measured here (read-only 580). The read half (1.0 GB, 444 GB/s,
  77 % of the roof) is real; the write-back of the in-place update must drain behind later kernels, so it is charged elsewhere. Treat the kernel as at its roof.
  (`gdn.metal:499-564`: whole 128x128 f32 state per head read and written once, 4 simdgroups per threadgroup.)
* **ce1 caveat on rows of overlapping pairs:** B=1 `fp8 4096x2048` shows 702 GB/s because it overlaps the qkv launch next to it; read pair totals.

## 5. How each kernel scales from 8 to 16 rows

`ce1` ms / step, 8 -> 16 (sum +16.34 ms, x1.79); mechanism established from the B sweep (section 2), the counts (section 3) and the code (section 9):

| kernel group | 8 | 16 | delta | x | mechanism | share of delta |
|---|---:|---:|---:|---:|---|---:|
| routed experts | 8.51 | 14.65 | +6.14 | 1.72 | distinct expert bytes x1.55, pairs x2 (down) | 38 % |
| fp8 projections | 5.47 | 10.15 | +4.68 | 1.86 | **second launch** (flat per launch) | 29 % |
| lm_head | 2.09 | 4.17 | +2.08 | 2.00 | per-row scalar FMA (and two launches) | 13 % |
| GDN delta rule + conv | 1.25 | 2.45 | +1.20 | 1.96 | per-row state bytes (inherent) | 7 % |
| attention decode | 0.93 | 1.76 | +0.83 | 1.89 | per-row KV bytes, 4 TGs per KV head | 5 % |
| shared expert | 0.86 | 1.77 | +0.91 | 2.06 | per-row loop / second launch | 6 % |
| router, in_a+in_b | 0.48 | 0.94 | +0.46 | 1.9 | second launch | 3 % |
| everything else (norms, top-k, gated norm, qk_norm_rope_kv, gates) | 1.10 | 1.13 | +0.03 | 1.0 | flat | 0 % |

The reason 8 -> 16 adds only 4 % in the served number is therefore: 50 % of the added step time is the chunk-and-reread of the dense
GEMV family (no 16-row kernel), 38 % is MoE bytes that rise x1.5, and 12 % is per-row state; none of them amortizes, and what is
amortized (norms, top-k, the first weight read) is already under 10 % of the step. The dense weights (1.69 GB) are streamed twice at 16 rows; the extra
stream would cost 2.9 ms at the measured roof, but the family grows by 8.2 ms, because each launch runs at 25-45 % of the roof.

## 6. Served 16-stream burst

localeval aggregate = completion tokens / burst wall (`speed.py:133`). Server arms (this run, `raw/w2/*-localeval.out`):

| arm | 1 stream | 8 streams | 16 streams | 16 / 8 |
|---|---:|---:|---:|---:|
| plain 1 | 115.9 | 332.6 | 325.2 | -2.2 % |
| plain 2 | 112.1 | 276.6 | 299.2 | +8.2 % |
| host profile | 117.5 | 301.8 | 309.7 | +2.6 % |
| `cb` profile | 117.8 | 306.5 | 326.4 | +6.5 % |
| `ce1` profile | 107.4 | 294.1 | 314.1 | +6.8 % |
| mean | 114.1 | 302.3 | 314.9 | **+4.2 %** |

The 8-stream cell moves 277-333 tok/s between server starts; CHUNK4K (2026-10-29) measured 307-336 at 8 and 332-352 at 16 streams. A single pair cannot resolve a +4 % effect. The robust quantity is the steady step (below).

Decomposition of one burst from the `ce1` trace (steps cut at each embedding launch; `raw/scripts/an_served.py`, `raw/analysis/served_ce1.md`):

| | 8 streams | 16 streams |
|---|---:|---:|
| burst wall | 13.93 s | 26.08 s |
| prefill steps (one prompt at a time, 990 tokens, no batching) | 8 x 238 ms = 1.90 s (13.7 %) | 16 x 245 ms = 3.92 s (15.0 %) |
| decode steps between prefills (two per prefill: a 1-row step of about 14 ms, then one step of the k active rows) | 15 steps, 0.23 s | 31 steps, 0.65 s |
| steady full-batch decode | 504 steps, 11.30 s (81 %), 23.15 ms wall / step | 496 steps, 20.58 s (79 %), 42.47 ms / step |
| drain (finishing streams, rows N-1 .. 1) | 7 steps, 0.11 s | 15 steps, 0.39 s |
| decode-only tok/s (rows / step wall) | 345.6 | 376.7 (+9.0 %) |
| aggregate tok/s | 294.1 | 314.1 (+6.8 %) |

The per-op groups of the served steady step (ce1 ms): 8 rows: MoE 8.45, fp8 6.05, lm_head 2.47, GDN 1.79, attention 1.25, shared 0.94, router 0.71, norms 0.52
(22.18); 16 rows: MoE 15.68, fp8 11.47, lm_head 4.91, GDN 3.21, attention 2.28, shared 2.00, router 1.05, norms 0.53 (41.14). Same ranking and same 8 -> 16
growth as the harness; each group is 7-18 % larger (next section). The 16-stream burst never has fewer than 16 rows except in the ramp and drain,
so the batch is full for 79 % of the wall; the only way to a bigger aggregate is a cheaper step.

## 7. Sustained load and the harness-served gap

The served steady step (42.5 ms at 16 rows, 23.15 at 8) is 10-15 % above the harness step (35.9, 20.0), at the same row counts and with
*fewer* distinct experts (section 3). In the `ce1` served trace *every* group is slower, including those whose work does not depend on
the data: lm_head 4.91 vs 4.14 ms (+18 %), fp8 11.47 vs 10.15 (+13 %), MoE 15.68 vs 14.65 (+7 %), GDN 3.21 vs 2.98 (+8 %). Test (`raw/scripts/w3.sh`, `raw/w3/`), the
same harness, one B, 64 steps (2.3 s) against 800 or 1200 steps (30 s):

| B | 64 steps (ms / step) | 800 / 1200 steps (ms / step) | change |
|---:|---|---|---:|
| 16 | 35.50, 35.66 | 39.60, 42.12 (800 steps) | +11 % .. +18 % |
| 8 | 19.78, 19.65 | 22.27, 22.05 (1200 steps) | +12 % |

The context grows by 800-1200 tokens over the long runs (attention is 1.8 ms at 1.1k context at B=16 and scales with it), which accounts for about 0.5-0.7 ms of the
4-6 ms at B=16 and of the 2.3 ms at B=8; the rest is a slowdown of the GPU under sustained load (clock / power state; a clock reading was not
available without root, so the cause is inferred from the uniform slowdown of data-independent kernels, not measured). The single-stream step (low power) is unaffected: 8.11 ms served vs 8.10 harness.
**Consequence for later lanes:** a 2-3 s harness cell flatters every multi-row kernel by 10-15 %; the served-regime figure is the one to quote, and an A/B must use the same duration on both arms.

## 8. Host side (question 6)

* **Command buffers and dispatches per step** (`cb` trace, `raw/analysis/harness_cb.md`, `served_cb.md`): 7 buffers / 764 dispatches at 1 row, 8 / 804 at 2..8,
  **10 / 1125 at 9..16** (the chunking adds 321 dispatches, 40 % more). A buffer is committed after 4 dispatches (`FIRST_COMMIT`, `lib.rs:165`) and every 128 (`MAX_DISPATCHES`, `lib.rs:158`), so
  the GPU starts about 0.3-0.6 ms after the host begins encoding (cb-mode first-buffer lag 0.29 / 0.60 / 0.50 / 0.31 / 0.40 / 0.46 ms at B 1 / 4 / 8 / 9 / 12 / 16).
* **Encode time** (host open -> commit, summed over the step's buffers): 1.39 / 1.93 / 2.19 / 3.01 / 3.20 / 3.50 ms at B 1 / 4 / 8 / 9 / 12 / 16 in the harness; 0.74 / 1.13 / 1.70 ms
  at 1 / 8 / 16 rows served. It runs concurrently with the GPU, which is busy for 8 / 21 / 40 ms; commit calls take 0.01-0.07 ms.
* **GPU idle at 16 rows, served steady state**: 0.078 ms per step with profiling off (`INTARSIA_HOST_PROFILE=1`: 255 gaps, `raw/w2/s-host-server.err`), i.e. 0.2 % of
  the 42.1-42.9 ms step. The `cb` profile, which perturbs the stream, shows 0.31 ms inside the step plus 0.39 ms before the next one (1.7 %; harness 0.64 + 1.75 ms, also profile-inflated).
* **Host phases at 16 rows** (host profile, decode-only, per step, ms): setup 0.006, `launch` (encode) 1.83 / 1.75, `sync` (GPU wait) 40.96 / 40.28, accept 0.002, on_token 0.002, detok 0.026, send 0.035; wall 42.9 / 42.1.
  The scheduler and sampling cost 0.07 ms per step; greedy rows sample by a device argmax (0.03 ms), so no logits leave the GPU. At 8 rows: launch 1.31 / 1.19, sync 21.3 / 22.1, gap 0.045 / 0.039 ms; 1 row: 8.1 ms, all `step-setup`.
* **What the host does stall on**: prefill. A new stream's prefill (245 ms, whole prompt in one chunk) runs between two decode steps, so during the 3.9 s ramp the earlier streams advance
  one token per 245 ms. That is the 16-19 % of the burst wall that is not decode (section 6); it is a scheduling property, not a kernel one.
* **Verdict**: at 16 rows the engine is GPU-bound to within 2 %; the host is not a bottleneck.

## 9. Matrix units, dequantization, row handling (question 5), with file:line

Line numbers are at `d4936a3d4`; `~` marks a range not re-checked by hand after a read-only survey. 9..16 rows: `lib.rs:237-273` (`row_chunks`; `CHUNK_TILE` 8 at `:229`, `CHUNK_ROWS_MAX` 16 at `:231`) splits fp8, nvfp4,
bf16 matmul / pair / logits GEMVs into two launches of `ceil(m/2)` and `floor(m/2)` rows, each re-reading all weights (and repeating the fp8 q(x) prep dispatch); the MoE rows kernel is exempt (`:252`) and runs one launch for up to 16 tokens. Decode selection is `rows_exact` only (`crates/intarsia-core/src/registry.rs:356-364`), so the NAX GEMM kernels (`gemm_nax_*.metal`, `moe_nax.metal`) are never selected.

| op | Rust | Metal kernel | matrix units? | NVFP4 / fp8 dequant |
|---|---|---|---|---|
| fp8 projections, M 2..4 | `launch_fp8_rows_xd` `fp8.rs:356`, `XD_MB` 4 `:544` | `fp8_xprep_bf16` (`fp8.metal:111`), then `fp8_gemv_bf16_m4_xd` (`:262`, impl `:132-231`) | no: scalar fma | e4m3 decoded once per 16-weight chunk per output row, **hoisted out of the row loop** (`fp8_dec16` `:175`; row loop `:176-203`, `RS` 2, `UN` 2); q(x) is prepared once per call |
| fp8 projections, M 5..8 | same, `mmt_min()` = 5 (`fp8.rs:502`) | `fp8_xprep_t` (`fp8.metal:284`), `fp8_rows_mmt_bf16` (`:394`, impl `:304-382`) | **yes, but plain `simdgroup_float8x8` fp32** (`:353-354`, two MMAs per K=16; measured peak 16.4 TFLOPS, equal to scalar FMA); rows padded to 8 | decoded once per threadgroup with `fp8_x2` (`:324`), 8 outputs x 8 rows per simdgroup; weights loaded as 16-bit pieces |
| fp8 one row | `fp8.rs` | `fp8_gemv_bf16_m1` (`:251`, `RS` 4, `UN` 4) | no | per 16-weight chunk |
| lm_head (n 248320) | `nvfp4.rs:170-185`, grid over 8-row shader for n > `NARROW_N` 4096 | `nv_gemv_f32_m8` (`nvfp4.metal:161`; impl `nv_gemv_impl` `:81-147`) | **no: scalar fma** (`nv_dot16d` `:69`, multi-row loop `:105-122`) | `nv_dec16` once per weight block (`:109`), **hoisted out of the M loop** (`:110`); `RS` 2, `NV_SG` 8 |
| shared expert 512x2048 / 2048x512 | `nvfp4.rs:28-45,176-182`, `nv_rows_mb` | `nv_gemv_*_m1` over a grid of M threadgroup rows | no | **per row**: each row's threadgroup re-decodes inline (`:123-136`, `nv_dot16`) and re-reads the weights from cache |
| router + shared gate, in_a/in_b | `multirow.rs:28` (kernel name) | `gemv_pair_bf16_rt_u2s8_mr` (`gemv_multirow.metal:218`, body `rt_rows16` `:79`) | no: `dot8` fma chain (`:54-64`) | bf16, none |
| routed experts | `moe.rs` rows branch `:232-243` | `moe_dec_gu_p` (`moe.metal:584`), `moe_dec_dn_p` (`:630`), `moe_gather_*` (`:546-554`) | no: LUT + fma `moe_blk_dot` (`:35-42`) | **per pair**: a threadgroup row per (token, k) pair (`MOE_PR` 2 rows per simdgroup, `:580`); an expert's blocks are decoded again by every pair that routes to it; no grouping, shared experts rely on cache (`:556-569`). The grouped `moe_lead` form (`:289-316`) is reachable with `INTARSIA_METAL_MOE_ROWS_FAT=0` |
| GDN delta | `rowtab.rs` (delta launch) | `gdn_delta_rule_rows_t` (`gdn.metal:499-564`) | no | f32 state, whole 128x128 state per head read once and written once per row (`:521-524`, `:563`) |
| attention decode | `launch_attn_rowtab` `rowtab.rs:355` | `attn_decode_rowtab_t` (`attention_splitkv.metal:532`) + combine (`:659`) | no (`ask_step` fma `:91-117`) | KV is bf16; `head_split` 4 for the 35B (`attn.rs:72-78`), so each KV head is read by 4 threadgroups (`:546-558`); stage-1 grid `(nkv*hs, max_split, R)` (`rowtab.rs:419-433`) |
| top-k, combine, norms, gated norm, conv, gates | | one launch for all rows | no | none |

The Metal backend registers no fusions for this model (`MatmulFp8Pair`, `RmsNormAddMatmulPair`, `GatedRmsnorm*`, `GdnConvDelta`, `GateSplitQkNormRopeKv`, `MatmulNvfp4Glu`, `SharedDownCombine`, `MoeBlock`: `kernels.rs:821-847,1037`), which is where the 1125 dispatches come from.

## 10. Bottlenecks, ceilings, and what they need

Floors use the measured 580 GB/s on unique bytes; "saves" is ms of the 37.0 ms harness step at 16 rows (multiply by 1.1-1.15 for the served regime). Items 1-4 are one family: the row-exact multi-row GEMV code that has no 16-row kernel.

1. **fp8 projections (10.15 ms -> 2.2; saves 7.9 ms).** Two 8-row launches at 235-290 GB/s each. Even a *single* 16-row launch at today's 5.5 ms per launch saves 4.7 ms (the conservative reading: do the second chunk for free); to reach the roof the kernel needs a faster per-launch design for 9..16 rows (matrix units on the fp8 -> bf16 dequantized tile, 41 GFLOP per step = 0.8 ms at the NAX rate). Code: `fp8.metal:132-231`, `:304-382`; `fp8.rs:356,502`; `lib.rs:237-273`. Bitwise-equal-to-solo rows (`registry.rs:356-364`) are the design constraint: a different accumulation order would need the KL / top-1 gates, not the bitwise ones.
2. **Routed experts (14.30 ms -> 9.7; saves 4.6 ms).** gate-up 9.07 ms for 3.7 GB (412 GB/s), down 5.23 ms for 1.9 GB (358 GB/s but fitted by pair count). The floor is real (5.6 GB of distinct expert bytes at 16 rows, workload dependent: 4.5 GB on the served filler prompts); closing the gap means every distinct expert streamed once at roof (grouped form: `moe.metal:289-316`, lane mm measured the per-pair form faster at <= 8 rows, `moe.metal:556-569`). Code: `moe.metal:546-640`, `moe.rs:232-243`.
3. **lm_head (4.14 ms -> 0.5; saves 3.6 ms).** The one large kernel that is compute / issue-bound rather than byte-bound: 8 rows per launch at 2.06 ms (3.9 TFLOPS, 25 % of the FMA peak, 12 % of the byte roof), time proportional to rows (0.26 ms per row); two launches at 16 rows. With one 16-row matrix-unit launch it would sit at its byte floor (286 MB, 0.49 ms). Code: `nvfp4.metal:81-147`, `nvfp4.rs:170-185`.
4. **Small-N GEMVs (shared expert, router, in_a/in_b: 2.6 ms -> 0.2; saves 2.4 ms).** 380 launches of 4-12 us at 4-12 % of the roof (`nvfp4.rs:28-45,176-182` runs the one-row shader once per row; `gemv_multirow.metal:79-99`).
5. **Attention decode (1.76 ms -> 0.63; saves 1.1 ms).** 207 GB/s of KV, 36 % of roof, with a 4x redundant KV read per head (`attn.rs:72-78`) and idle split threadgroups on short rows (`attention_splitkv.metal:554`); grows x1.9 from 8 to 16 rows. Ceiling rises with context.
6. Small: about 330 byte-free launches cost 1.8 ms (5 %); fusion is blocked by the row-exact registrations (no Metal fusions registered).

Not worth a lane: GDN delta rule (2.27 ms, at the bandwidth roof; only a narrower state would help and that moves numerics), host / scheduler (0.2 % idle), sampling (0.03 ms).

## 11. Caveats

* Harness contexts are about 1.0-1.2k tokens and its text is one markdown corpus; routing, hence the MoE bytes, move with the text (section 3). Longer contexts raise attention (x2 time per doubling of KV bytes) and nothing else.
* `ce1` charges overlapped launches to the earlier one; group totals are reliable, single rows of overlapping pairs are not (B=1 fp8 4096x2048). `cb` mode is +12 % at 16 rows; `kernel` durations are unusable below 9 rows (clock state), used for routing labels only.
* Server-start to server-start spread is 10-20 % at 8 streams (277-333 tok/s); only the steady-step and decomposition figures should be compared across lanes.
* The sustained-load explanation (section 7) is inferred, no clock / power reading.
* Roofs: 580 GB/s is a read-only stream; mixed read / write traffic (delta rule) cannot be judged against it. FMA / MMA peaks are from `raw/scripts/roof/` on this box, not published figures.
* Numbers are for MTP off and `--max-batch 16`; the workspace of 16 sequences is allocated (942 MiB), which does not change the kernels at <= 8 rows.

## 12. fleetq jobs and artifacts

All on endor, owner `claude-908e11e8-mprof`: **#7270** build (`--load cpu`), **#7271** window 1 (harness, plain x3 / cb / ce1 / kernel; `--quiet gpu,cpu`, 26 GB), **#7275** window 1b (descending B, ce1, kernel x2, plain),
**#7276** window 2 (served: plain x2, host, cb, ce1, kernel; 28 GB), **#7278** roofs micro-benchmark, **#7280** window 3 (sustained load). Peak process footprint of the 35B at `--max-batch 16`: 20-21 GB (CHUNK4K), well inside `endor-kev` 18 GB + `endor-tensorfold` 32 GB residents.

* The large traces (`ce1`, `kernel`, 7 files) are stored trimmed: host-time fields dropped, times rebased to the first record, 7 decimals; the tables here were regenerated from the trimmed files (differences from the untrimmed ones <= 0.01 ms).
* Scripts: `raw/scripts/w1.sh w1b.sh w2.sh w3.sh`, analysers `mp.py an_groups.py mk_breakdown.py an_roof.py an_experts.py an_cb.py an_served.py an_served_cb.py an_served_experts.py`, roofs `roof/roof.metal roof.swift`.
* Data: `raw/w1/` (ascending B), `raw/w1b/` (descending), `raw/w2/` (served; `s-ce1.jsonl.gz` 19 MB, `s-kernel`, `s-cb`), `raw/w3/` (sustained), `raw/roof.txt`, `raw/analysis/*.md` (every table above).
* Reproduce: `raw/scripts/build.sh` (`cargo build --release --locked -p intarsia-cli --features metal` and `cargo test --release --locked -p intarsia-models --test batch_metal_real --no-run`, target dir `.lane/target`), then `fleetq run --host endor --quiet gpu,cpu --mem 26 --expect 15 -- raw/scripts/w1.sh OUT` (and `w1b.sh`, `w2.sh`, `w3.sh`). Roofs: `xcrun -sdk macosx metal -std=metal3.1 -O2 -c roof.metal` + `metallib` + `swiftc -O roof.swift`, run as `roof roof.metallib`.
