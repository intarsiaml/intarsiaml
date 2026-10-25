# mr2: where the batched Qwen3.8-27B MXFP4 decode step spends its time on HIP (lane mr, 2026-10-25)

Branch `lane-mr/hip-27b-batch-profile` (off main 5e9cc6da, which contains #145). Measurement only: the one code change is in the test harness
(`batch_cuda_real real_35b_batch_step_profile` takes `INTARSIA_TIMING_ROWS`, has warm-up steps and prints 40 rows; `rows_exact
timing_mxfp4_rows` times the lm_head rows kernel up to M=16). Nothing in the engine or the kernels changed. Qwen3.8-27B-Quark-AWQ-MXFP4, one
R9700 (card 0), HIP, tatooine with radiance down in fleetq slots, ~1000 tokens of context per sequence, decode-only steps (argmax head).
Scripts: scripts/mr/profile.sh (+ build.sh, win.sh). Raw: docs/results/2026-10-25-mr-hip-27b-batch-profile/raw/.

## Summary

* The unprofiled batched step (`forward_batch`, graph-replayed): 30.3 ms at 1 row, 36.8 at 2, 42.1 at 4, 59.5 at 8, 82.7 at 12, 110.9 at 16
  (33 / 54 / 95 / 134.5 / 145 / 144 tok/s aggregate). Aggregate saturates at ~145 tok/s from 12 rows on: past 8 rows each extra row costs
  6.4 ms (8 -> 16) against 3.9 ms (4 -> 8).
* **The MXFP4 GEMVs are still 66-69 % of the step** (`matmul_mxfp4_pair` 44 % + `matmul_mxfp4` 25 % at B=16) and rise 2.4x from B=2 to B=16 where
  a weight-bound ideal is 1.0x. The GEMV time per extra row is constant (~12.5 us on the 17408x5120 shape) while the weight bytes are read once, which points at ALU rather than bytes (estimate below, not
  confirmed with counters). This corrects the #145 doc, which said the remaining gap was not the MXFP4 GEMV: the step profile shows the GEMVs are the largest remaining item.
* **lm_head rows has a cliff above 8 rows**: 4.4 ms flat up to M=8, then 7.8 ms at 12 and 15-17 ms at 16 (4x the 8-row cost for 2x the rows;
  ideal: flat, the 2.5 GB of weights are streamed once). 12 % of the B=16 step, 22 % of the step growth from 8 to 16 rows.
* GatedDeltaNet (7 %) and attention (3 %) are not the problem: both scale with the bytes per sequence and sit near their floors.
* The pair launcher's two launches are not measurable at this granularity and bounded by <0.5 ms per step.
* Served 27B aggregate at 8 streams was 61 tok/s against 134 tok/s of the bare 8-row step: more than half of the gap to the step is outside the
  model step (prefills of the other streams, scheduling), see "Served vs bare step".

## Unprofiled step time (`real_35b_batch_step_timing`, raw/timing.log)

| rows B | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| ms per step | 30.3 | 36.8 | 39.2 | 42.1 | 46.2 | 50.4 | 59.5 | 82.7 | 110.9 |
| aggregate tok/s | 33.0 | 54.4 | 76.6 | 95.0 | 108.1 | 119.2 | 134.5 | 145.1 | 144.3 |
| ms per extra row | | 6.5 | 2.4 | 3.0 | 4.1 | 4.1 | 4.6 | 5.8 | 7.0 |

(The "solo step 101 ms" line of that test is the eager one-row `forward`, not the captured graph the server replays; the server's one-row
step is the 30.3 ms of B=1.)

## Per-op profile (`real_35b_batch_step_profile`, profile_every=1: a sync after every op, 8 steps after 2 warm-up steps, ms per step)

Profiling adds a sync per op (~1,400 ops per step), so the sum is above the unprofiled step (B=2: 50.5 against 36.8, B=16: 120.2 against 110.9); read the shares and the
growth, not the absolute level of the small ops. The flat 9 ms of "everything else" is mostly that per-op overhead. **B=1 cannot be profiled this
way**: the one-row step runs as a captured graph (or, solo, its ops are not timed one by one: op sum 5.3 ms against 31.9 wall), so B=2 is the
low end here and the one-row reference is the unprofiled 30.3 ms and the kernel micro-timings below.

| op (calls/step) | B=2 | B=4 | B=8 | B=12 | B=16 | B=16 / B=2 | ideal | share of step B=8 | B=16 |
|---|---|---|---|---|---|---|---|---|---|
| matmul_mxfp4_pair (176) | 21.36 | 23.17 | 31.72 | 42.55 | 52.62 | 2.46 | 1.0 (weight bytes) | 44% | 44% |
| matmul_mxfp4 (144) | 12.62 | 14.43 | 18.84 | 24.43 | 30.29 | 2.40 | 1.0 (weight bytes) | 26% | 25% |
| logits (1) | 4.26 | 4.29 | 4.42 | 7.78 | 15.01 | 3.52 | 1.0 (weight bytes) | 6% | 12% |
| gdn_delta_rule (48) | 2.23 | 2.86 | 4.84 | 6.82 | 8.80 | 3.95 | 8.0 (state bytes per sequence) | 7% | 7% |
| attn_decode (16) | 0.93 | 1.26 | 2.01 | 2.80 | 3.61 | 3.88 | 8.0 (KV bytes per sequence) | 3% | 3% |
| everything else (norms, silu, gdn_conv, rope, qk_norm, kv_append, gate ops, ...) | 9.12 | 9.13 | 9.49 | 9.72 | 9.92 | 1.09 | 1.0 | 13% | 8% |
| **op time sum** | **50.5** | **55.1** | **71.3** | **94.1** | **120.2** | 2.38 | | | |

Ranked by share of the B=16 step: (1) `matmul_mxfp4_pair` 44 %, (2) `matmul_mxfp4` 25 %, (3) `logits` 12 %, (4) `gdn_delta_rule` 7 %, (5) `attn_decode` 3 %. The
other ~50 op kinds are 8 % together and flat in B (norms, silu, rope, qk-norm, kv-append, gate ops, gdn_conv: row-wise kernels that do the same
work per row but fill the GPU better with more rows).

Growth of the step from 8 to 16 rows (48.9 ms profiled): pair +20.9, mxfp4 +11.4, logits +10.6, gdn_delta_rule +4.0, attn_decode +1.6, all others +0.4.
From 2 to 8 rows (20.8 ms): pair +10.4, mxfp4 +6.2, gdn_delta_rule +2.6, attn_decode +1.1, logits +0.2.

## Hypotheses

* **GatedDeltaNet recurrent per-seq loop: killed.** `gdn_delta_rule` is one row-table launch per layer for all sequences (48 calls/step), 8.8 ms at 16
  rows = 0.55 ms per sequence. The recurrent state is 149 MiB per sequence (plan note), read and written once per step: 0.31 GB per sequence
  at the ~600 GB/s the card streams = 0.52 ms. So it sits at the byte floor; the only lever is the state size/precision, not the loop. `gdn_conv`
  is 1.4 ms and flat.
* **Attention rows: killed (small).** `attn_decode` 3.6 ms at 16 rows = 0.23 ms per sequence for ~64 MB of KV each (65,536 B/token, 1k tokens): 2x the
  byte floor of 0.11 ms, 3 % of the step. Would matter at long context, not here.
* **lm_head rows: confirmed, and it is a cliff, not a slope.** micro-timing of `logits_bf16_gemv_rows` (248320x5120 bf16, raw/micro.log): M=1 4.24 ms
  (599 GB/s), 2 4.26, 4 4.34, 5 4.43, 6 4.45, 8 4.50, **12 7.85, 16 16.70 ms (152 GB/s)**. The same shape as the MXFP4 GEMV before #145: a kernel
  that is fine to 8 rows and loses its register budget above. Ideal at 16 rows is ~4.5-5 ms.
* **Pair launcher's two launches: killed.** At M >= 2 `matmul_mxfp4_pair_gemv_rows` is two launches of the rows kernel (the M=1 LDS fusion does not apply). The
  step has 176 pair calls; two launches instead of one at graph-replay gaps of at most 2-3 us bounds the effect at 0.5 ms of a 59-111 ms step
  (<1 %, an estimate). The pair op's time is the GEMV time of its two members (gate/up 17408x5120 twice: 2 x 272 us at M=16 in the micro-timing).

## MXFP4 GEMV at 8-16 rows: ALU estimate (not confirmed)

Rows kernel micro-timing (17408x5120, raw/micro.log): 80.4 us at M=1, 156.4 at M=8, 274.6 at M=16, i.e. 12.3-14.7 us per extra row, while the weights
(45 MB) are read once. 89.1 M weight elements per row in 12.5 us is 7.1 T fma/s; add the bf16 unpack shared by two weight rows (about half an op per fma)
and it is ~10.7 T VALU lane-ops/s. A single-issue R9700 lane rate is about 4096 lanes x ~2.9 GHz = 11.9 T/s (clock is an assumption; the advertised TFLOPS
counts dual issue). So the kernel would be running at roughly 90 % of single-issue VALU, and more rows then cost proportionally. This arithmetic is an
estimate; the check that would confirm it (instruction mix in the ISA, VOPD dual-issue use, or counters) was queued (fleetq #5260, cpu-only) but not run
because tatooine was held exclusively by another job; `scripts/mr/isa.sh` is in the branch for whoever runs it.

Row-exactness limits the options: the per-row summation order (one sequential fma chain per 32-element group, then a fixed warp-sum) must stay, which rules
out WMMA/dot-instruction accumulation unless the M=1 path changes with it (that changes bits against main: a user decision, not made here).

## Ranked levers (nothing implemented, per the brief)

1. **`logits_bf16_gemv_rows` above 8 rows** (bf16 lm_head, kernels/hip/matmul.hip `gemv_dispatch<M>`): the same cliff #145 fixed for MXFP4. Micro: 4.5 ms at M=8, 7.8 at 12, 16.7 at 16 (ideal ~4.6). Profiled
   step: 15.0 ms at B=16 against ~4.5 achievable, so about 10 ms of the 111 ms step (~9 %: 144 -> ~158 tok/s aggregate at 16 rows) and ~3 ms of the 83 ms step at 12 rows.
   Cheapest fix on the list: the register-shape pattern from #145, row-exactness is the same argument (per-row lane order unchanged), one launcher, one test case in `rows_exact`. Whether the cause is a spill
   should be checked first with `scripts/mr/isa.sh` (its second half prints VGPR/spill per `gemv_dispatch` M; not run). This is lane wg's file family (bf16 small-row), so coordinate before touching it.
2. **MXFP4 GEMV ALU cost** (69 % of the step): needs fewer VALU ops per fma without changing the summation order. Candidates, each to be checked in the ISA first: packed/dual-issue use on gfx12,
   cheaper fp4 -> f32 decode (the LUT/convert shared by 2 rows already), sharing the bf16 unpack across 4 weight rows for M <= 4 only (4 rows spilled above 6 in #145, so only a per-M choice helps).
   Upside is bounded by the ALU floor; a 15 % GEMV gain is ~10 % of the step. Larger gains need the numerics decision above.

Everything else: GatedDeltaNet and attention sit at their byte floors (killed above), the flat tail is 8 % of the step.

## Served vs bare step

localeval at `--max-batch 8` (PR #145 doc): 61.5 tok/s aggregate and 11.6 tok/s per stream at 8 streams. The bare 8-row step is 59.5 ms = 16.8 tok/s per
stream, 134.5 aggregate. localeval's figure includes the streams' 1k-token prefills (~1 s each at ~980 tok/s, serialised) and scheduling between
steps, so about half the served gap to the step is outside the model step. That is a serve-loop question (lane hg's serve-gap work), not
an MXFP4 kernel one, and sets the ceiling for what step-time fixes can show in the served numbers: a 15 % faster step moves a prefill-diluted
benchmark far less than 15 %.

## Not done

* B=1 per-op reference (graph-replayed; needs a profile mode that times ops inside a captured graph, or an eager one-row batched step).
* ISA / instruction-mix check of the rows kernel and the VGPR/spill check of `logits_bf16_gemv_rows` (script ready, job not run: tatooine was held exclusively).
* No per-launch timing inside the pair op (the <0.5 ms bound assumes 2-3 us per extra graph node, not measured), no GPU hardware counters (rocprof not used).
* Nothing implemented, per the brief.
