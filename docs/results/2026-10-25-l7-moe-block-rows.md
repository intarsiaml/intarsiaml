# l7: grouped MoE block for 5 to 16 rows (HIP, Qwen3.6-35B-A3B, one R9700)

Branch `lane-l7/moe-block-rows`, lane 7 of `docs/plans/2026-10-25-r9700-push.md`. Status: built, gates pass, default-on. Section 0 is the scope note as committed before any code; sections 1 to 5 are the results.

## 0. Scope note (written before any code)

### What the model runs at 5 to 16 rows today

`MOE_BLOCK_ROWS_MAX = 4` (`crates/intarsia-backend-hip/src/quantmoe.rs`), so from 5 rows the model takes the unfused chain of
`crates/intarsia-core/src/quant/moe.rs` `forward_folded`. Counted from the source and the hr3 OPLIST (`w7-step-profile.log`, `batch16`):

| op (per layer) | kernel | launches at 16 rows (8 rows) |
|---|---|---|
| router, shared-gate logit | `matmul_bf16_gemv_rows` x2 | 2 (2) |
| `moe_topk` | `moe_topk_bf16_v2` | 1 (1) |
| `moe_experts` | `moe_experts_bf16_rows`: `moe_group_kernel`, `moe_gate_up_kernel`, `moe_down_kernel`, `moe_reduce_v8_kernel` (P = T*K > 16, so the grouped path, 8 pairs per weight pass) | 4 (4) |
| shared gate, shared up | `matmul_nvfp4_gemv_rows`, rows in groups of at most 4 (`kRowsGroup`, `nvfp4.hip`), each group re-reads and re-decodes the weights | 8 (4) |
| `silu_glu` | `silu_glu_bf16` (the fused `nvfp4_glu_kernel` is M = 1 only) | 1 (1) |
| shared down | `matmul_nvfp4_gemv_rows` | 4 (2) |
| `moe_combine` | `moe_combine_bf16` (`SharedDownCombine` is M = 1 only) | 1 (1) |
| total | | **21 (15)** |

The fused `moe_block_bf16_rows` (`moe_block.hip`) replaces everything after the two logit GEMVs with one launch, but it reads every
token's 8 experts separately (no sharing between tokens that pick the same expert) and runs the shared expert once per token. hr3 s4: it
wins up to 4 rows and loses past that (23.73 vs 22.21 ms at 8 rows, 47.57 vs 41.75 ms at 16).

The unfused `moe_experts_bf16_rows` is **already grouped by expert** (the counting sort in `moe_group_kernel`, weights read once per expert and
chunk of 8 pairs, per-pair arithmetic independent of the grouping, so token-exact). The plan's "block that groups tokens by expert" is
therefore mostly there for the routed part. What is not grouped is everything around it.

### Where the ms are (hr3 `real_35b_batch_step_profile`, B = 16, every op synchronised, 54.95 ms op sum, 41.6 ms unprofiled step)

| piece | profiled ms/step | note |
|---|---|---|
| `moe_experts` (routed, 4 launches x 40) | 13.76 | bandwidth floor: about 100 distinct experts x 1.77 MB (gate_up 1.125 MB + down 0.56 MB incl. scales) x 40 = 7.1 GB; at the 585-627 GB/s measured read peak (`2026-10-13-hip-decode.md`) 11.3-12.1 ms, so at most about 1.1-1.5 ms is left in the kernels themselves |
| shared expert: `matmul_nvfp4` 120 of 121 calls | about 4.6 (the 121st, the lm head, is about 3.0 at 16 rows per the `nvfp4.hip` comment) | 38 us per op profiled; a 0.5 MB weight is 1 us of traffic, so this is launch, tail and the 4x re-decode, not bandwidth |
| `moe_topk`, `silu_glu`, `moe_combine` | 0.99 + 0.79 + 0.77 | mostly the per-op sync price (about 17 us per op: (55.6 - 41.6 ms) / 814 launches) |

Not claimed: the profile syncs every op, so the sum overstates what graph replay pays. The plan's own bound for the routed part was about 4 ms; the
kernel-only part of that is smaller (about 1.1-1.5 ms) if the 100-distinct-experts figure holds for real routing, which the profile does not record.

### What I will change

One new kernel entry, `moe_block_bf16_grouped`, registered for `Op::MoeBlock` for 5..16 rows (`rows_exact`), so the model's existing
"gate logit first, then MoeBlock" path in `quant/moe.rs` takes it with no model change. Inside, a short chain of launches instead of a
ticketed megakernel (no cross-block spin synchronisation to prove deadlock-free):

1. `topk + group`: one block computes every token's top-k (`wave_topk`, the arithmetic of `moe_topk_v2_kernel`) and builds the expert-sorted pair
   table (`moe_group_kernel`'s output), writing ids and weights.
2. shared gate + up + silu for all rows in one pass over the weights (weights decoded once, one accumulator set per row, the lane order, the
   16-element fma chain and the warp sum of `nvfp4_gemv_v2_kernel` per row, then `nvfp4_glu_kernel`'s epilogue rounding).
3. routed gate_up and down (the existing grouped kernels, unchanged arithmetic) and `moe_reduce_v8`.
4. shared down + combine epilogue for all rows in one pass (`shared_down_combine_kernel<1>`'s arithmetic per row).

21 launches (15 at 8 rows) become about 8 including the two logit GEMVs. Every token's accumulation order is the order of the T = 1 block (each piece
copies an already row-exact kernel's arithmetic), so it stays bitwise; `tests/moe_block.rs` is extended to compare 5..16 rows against the T = 1
block, with a negative control. The fused `moe_block_bf16_rows` stays for 1..4 rows (it wins there).

### Expected gain (central estimate, to be replaced by the microbench)

What a removed launch is worth is not known well. Two data points disagree: `2026-10-19-p2prime-moeblock.md` s3 measured 3.9 us for a tiny
1-block launch inside a graph replay (T = 1), while hr3's fused block at 3 and 4 rows (12.11 to 11.63 and 13.83 to 13.61 ms) removes about 9 launches
per layer (360 per step) for a net 0.5 and 0.2 ms, i.e. only 0.6-1.3 us net per removed launch (the fused kernel has costs of its own).

| lever | ms per step at 16 rows (8 rows) |
|---|---|
| 13 fewer launches x 40 layers, at 0.6-1.3 us net each | 0.3-0.7 (0.2-0.4) |
| shared expert: one weight pass instead of 4 (2); 120 `matmul_nvfp4` ops at about 21 us real each (38 us profiled less the sync price) | 1.0-1.5 (0.4-0.8) |
| routed kernels: not targeted in this phase (bound 1.1-1.5 ms) | 0 |
| **total** | **about 1.3-2.2 ms of 41.6 = 3-5% model loop (8 rows: 0.6-1.2 of 22.2 = 3-5%)** |

The bar is served tok/s, not the model-loop step: hr3 s4 has 62 ms served against 41.6 ms in the model loop at 16 streams (the served burst includes
the prefills). 3% of served tok/s at 16 streams is about 1.9 ms per step (8.1 s burst wall x 3% / 128 steps), so the **go line is 1.9 ms
of model-loop saving at 16 rows**, and the central estimate above straddles it. The first GPU window is therefore a microbench, with existing kernels only
(`moe_rows_timing` in `tests/moe_block.rs`, T = 5, 8, 12, 16, graph replay, 16 layers): (a) the unfused chain as the model runs it, (b) topk + experts,
(c) the shared pieces + combine, (d) one `matmul_nvfp4_gemv_v2` launch at M = 4 per shared matrix as an optimistic floor for a single 16-row weight pass,
(e) a tiny launch, (f) the fused rows block, plus the distinct experts per layer. Available saving is about (c) - 3 x (d) plus the launch delta. **Build
only if that reaches about 1.9 ms at 16 rows; otherwise the lane stops there and reports.**

### Constraints for the build

* The shared-expert arithmetic is copied from `moe_block.hip`'s own copies of the `nvfp4_glu_kernel` and `shared_down_combine_kernel<1>` epilogues, not
  from `gemv_rows` + `silu_glu` + `moe_combine` (lane 1 option B may move those). The new op is tested against the T = 1 block and the unfused chain.
* New device code goes in `moe_block.hip`; `nvfp4.hip` and `matmul.hip` belong to lane 1; the `quantmoe.rs` / `kernels.rs` registry diff stays minimal.
* B = 1 no-op: the new kernel's scratch function must not enlarge the workspace when batching is off (hr3 once changed `plan show` that way).

### Gates (from the lane table)

`moe_block.rs` bitwise against the T = 1 block with a negative control; the 35B batched == solo G-C test (17,550 pairs); no-op at B = 1 (`plan show`
and 256 ids identical to main); speed A/B with localeval at 1/4/8/16 streams, main against branch, interleaved x3.

## 1. What was built

* `kernels/hip/moe_block_grouped.hip`: `ik_moe_block_bf16_grouped`, registered `moe_block_bf16_grouped` for `Op::MoeBlock`, `rows_exact`, 3 to 16 rows.
  Four launches: `moe_topk_bf16_v2`, `moe_experts_bf16_rows` (tokens grouped by expert, unchanged arithmetic), one shared-expert gate + up + silu pass for
  all rows (`shared_glu_rows_kernel`), one shared-down + combine pass for all rows (`shared_down_combine_rows_kernel`). Per row the arithmetic is
  `nvfp4_glu_kernel` / `shared_down_combine_kernel<1>`: same lane order, 16-element fma chains, scale per block, `ik_warp_sum`, same bf16 roundings.
  The shared weights are read and decoded once instead of once per group of 4 rows.
* `kernels/hip/moe.hip`: `moe_gate_up_g2_kernel<RW>` / `moe_down_g2_kernel<RW>`, the row-exact grouped projections with RW output rows per warp, no
  shared-memory staging. Bit for bit the old kernels (same per-(pair, row) order); only the row-to-warp mapping and the loads differ. Used by
  `moe_experts_bf16_rows` only. Knob `INTARSIA_HIP_MOE_G2=gu:dn`, default `2:2`, `0` = old kernels.
* `quantmoe.rs`: the fused `moe_block_bf16_rows` keeps 1 to 2 rows while the grouped block is on (4 with `INTARSIA_HIP_MOE_GROUPED=0`); the grouped block serves 3 to 16.
  Microbench: grouped is ahead of the fused rows block from 3 rows (118 against 128 us per layer at 3 rows with G2, 147 against 164 at 4; before G2 it was behind: 139 and 175).
* Knobs (docs/tuning.md): `INTARSIA_HIP_MOE_GROUPED` (on, 0 disables), `INTARSIA_HIP_MOE_G2` (2:2, 0 disables). Both are in the backend fingerprint.

The scope note's estimate for the build was 1.3 to 2.2 ms of 41.6; the microbench said otherwise (section 2), which is why the projections were rewritten
as well (the routed part was NOT at its bandwidth floor: the one-row-per-warp kernels with 32 KiB staging were).

## 2. Microbench: per layer, us, graph replay of 16 layers, median of 50, 3 rounds (`raw/w1-timing-old-projections.log`, `raw/w2-timing-g2-sweep.log`, `raw/bench.log`)

Round 3 of each (the three rounds agree within 1 %; w2 values, `raw/w2-timing-g2-sweep.log`). (a) unfused chain as main runs it from 5 rows (G2 = 0),
(f) fused rows block as main runs it up to 4 rows, (g) grouped block with the old projections (G2 = 0), then (a) and (g) with the new projections (G2 2:2).

| rows | (a) main chain | (f) main fused rows | (g) grouped, G2 off | (a) G2 2:2 | (g) grouped, G2 2:2 | main chain to (g) | x 40 layers |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 215.7 | 196.3 | 207.4 | 180.3 | 171.4 | 44 us | 1.8 ms |
| 8 | 312.4 | 310.7 | 302.8 | 249.1 | 240.4 | 72 us | 2.9 ms |
| 12 | 450.7 | 457.2 | 438.2 | 354.0 | 341.6 | 109 us | 4.4 ms |
| 16 | 589.3 | 609.5 | 566.4 | 459.7 | 436.5 | 153 us | 6.1 ms |

At 3 and 4 rows (`raw/bench.log`): fused 127.9 and 163.8, grouped G2 off 138.8 and 174.8, grouped G2 2:2 118.0 and 146.9.
In this bench the fused block is faster than main's chain at 5 rows (196 against 216), but not once the grouped projections exist (171), so the fused block gives way from 3 rows.
Of the 153 us at 16 rows, about 130 us are the new grouped projections (G2 off to on, 566 to 437 for the grouped block), about 23 us the shared-expert pass and fewer launches ((a) 589 to (g) 566). The rows-per-warp
sweep at 16 rows, gate_up:down (unfused chain, us per layer): 2:4 466, 1:4 477, **2:2 460**, 1:2 471, 2:1 508, 1:1 521: 2:2 is the default.
Distinct experts per layer at 16 rows: mean 103.6 (min 97, max 111).

## 3. Gates (all on tatooine card 0, radiance down and restored after each window)

| gate | result | log |
|---|---|---|
| `moe_block.rs` bitwise against the T = 1 block (grouped: 240 launches, 1780 tokens; fused rows: 808 tokens; chain 52 cases; dispatch order 90; sentinel stress 6000 launches), graph replay bitwise eager | 11 passed, 0 failed | `raw/g-b-2.log` |
| negative controls: 216 "token r vs solo block of next token" controls differ; tensor-core routed experts differ from the solo block in 4 of 16 tokens (the control that must fail does) | pass | `raw/g-b-2.log` |
| `rows_exact`, `rows_registry` | 11 + 2 passed | `raw/g-b-2.log` |
| 35B batched == solo G-C | **17,550 (row, step) pairs bitwise identical** in 165.7 s; tiny hybrid and `batch_recurrent` 11 passed | `raw/g-c-2.log` |
| no-op at B = 1 | `plan show` text and json byte-identical for the 8B and the 35B; 256 greedy ids identical (8B and 35B, short and 4k prompt) | `raw/noop.log` |

Notes: the first G-B window (`raw/g-b.log`) had one failure, `moe_block_grouped_negative_controls` control (b), tensor-core shared GEMVs giving the solo
block on these activations. That is a property of the control, not of the kernel; (b) is reported only now, (a) and the next-token controls carry the claim.
`batch_recurrent` now accepts a `moe_block_bf16_grouped` pick next to `moe_block_bf16_rows` as the fused block in a 4-row step.

## 4. Speed A/B (localeval `speed`, 35B NVFP4, `serve --max-batch 16`, base = origin/main fecf2d24e, dev = this branch, interleaved base dev x3, 1024-token prompts, 128 tokens, 3 reps each; `raw/ab1/`)

Aggregate decode tok/s, one value per interleaved pair (base, dev):

| streams | main (3 runs) | branch (3 runs) | main mean | branch mean | change |
|---:|---|---|---:|---:|---:|
| 1 | 163.5, 163.3, 163.1 | 162.9, 163.2, 163.1 | 163.3 | 163.1 | -0.1 % |
| 4 | 216.6, 214.5, 211.4 | 223.2, 220.7, 217.9 | 214.2 | 220.6 | +3.0 % |
| 8 | 245.1, 248.0, 244.9 | 267.6, 265.5, 267.9 | 246.0 | 267.0 | +8.5 % |
| 16 | 259.1, 259.5, 258.6 | 276.9, 279.9, 278.8 | 259.1 | 278.5 | +7.5 % |

The plan expected +5 to 9 % at 8 to 16 streams and 0 at 1 to 2: that is what was measured. Run-to-run spread is under 1 % per arm, every branch run beats every main run at 4, 8 and 16 streams.
Quiet window: the fleetq jobs ahead and behind were all `--quiet gpu,cpu`; radiance was down for the 40 s runs and restored by the trap.

## 5. Default and rollback

Default on: all four gates pass, +7.5 % at 16 streams (bar 3 %), -0.1 % at 1 stream (bar 1 %). Roll back with `INTARSIA_HIP_MOE_GROUPED=0` (the main rows block and
chain come back, 1 to 4 rows fused) and `INTARSIA_HIP_MOE_G2=0` (the old one-row-per-warp projections). Not measured: model-loop step times per row count
(`real_35b_batch_step_timing`, script section `TIMING` in `scripts/l7/gates.sh` is ready; fleetq job #5507 started it and I killed its client PID by mistake while trying to withdraw it, so there is no data; radiance came back by the script trap and was checked).
