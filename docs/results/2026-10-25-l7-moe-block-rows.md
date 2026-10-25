# l7: grouped MoE block for 5 to 16 rows (HIP, Qwen3.6-35B-A3B, one R9700)

Branch `lane-l7/moe-block-rows`, lane 7 of `docs/plans/2026-10-25-r9700-push.md`. Status: scope note (nothing built, nothing run on a GPU yet).

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

| lever | ms per step at 16 rows (8 rows) |
|---|---|
| 13 fewer launches x 40 layers (about 2-3 us each in a graph replay: the plan puts the dispatch gap at 1.5 us, plus tail) | 1.0-1.6 (0.6-1.0) |
| shared expert: one weight pass instead of 4 (2) | 1.0-1.5 (0.4-0.8) |
| routed kernels: not targeted in this phase (bound 1.1-1.5 ms) | 0 |
| **total** | **2.0-3.1 ms of 41.6 = 4.8-7.5% (8 rows: 1.0-1.8 of 22.2 = 4.5-8%)** |

Served tok/s moves less than the model-loop step: hr3 s4 has 62 ms served against 41.6 ms in the model loop at 16 streams, so the same saving is
about 3.2-5% of a served step at 16 streams. That is above the 3% stop line on the central estimate and close to it on the low end, so the first GPU
window is a microbench (`moe_rows_timing` in `tests/moe_block.rs`: per-piece cost and the price of a launch inside a graph replay at 5, 8, 12, 16 rows,
unfused chain against the fused block) before the kernels are written. If it shows less than 1.25 ms (3% of 41.6) available at 16 rows, the lane stops there
and reports.

### Gates (from the lane table)

`moe_block.rs` bitwise against the T = 1 block with a negative control; the 35B batched == solo G-C test (17,550 pairs); no-op at B = 1 (`plan show`
and 256 ids identical to main); speed A/B with localeval at 1/4/8/16 streams, main against branch, interleaved x3.
