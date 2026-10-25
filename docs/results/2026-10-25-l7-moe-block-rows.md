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
