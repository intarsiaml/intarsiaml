# DSv4.1-Flash decode at TP=2: where intarsiaml loses to TensorFold, and what to change

Fable review, 2026-11-01. Read-only. Code: `.worktrees/dsv41-graphs` (dsv41/graphs), TensorFold Zig tree `.worktrees/blackfold/src/tensorfold`
(cubin-only EXL3 kernels; their open source is `/opt/code/TensorFold/src/tensorfold/cuda/exl3/`). Traces re-analysed with
`.worktrees/fable-decode/grid_breakdown.py` (per-kernel, per-grid; outputs `grid-r0.txt` = spec on, window 2 ab2; `grid-ab1-r0.txt` = spec off).
"TF" numbers marked inferred come from its code and its 36.7 ms/round measurement, not a profile.

## 1. Per-category gap, ms per pass (rank 0; rank 1 within 0.1 ms)

| category | ours 1 row (measured, ab1) | ours 4 rows (measured, ab2) | TF (inferred) | notes |
|---|---|---|---|---|
| EXL3 dense linears (`linear_kernel`, `split_kernel`+finish) | 11.9 | 17.2 | ~8 at any rows <= 16 | TF's doc: 221 GB/s at 1 and 16 rows, "rows cost within 5 %" (`docs/recipes/exl3.md` in the TF tree, MiMo). Our per-shape rates below are 23-125 GB/s |
| MoE `grouped_cp_kernel` (gate/up + down) | 8.85 | 19.8 | same bytes, so ~equal per row | both passes run at ~230 GB/s if experts are ~3.3 bpw; the 4-row pass reads ~2.2x the 1-row bytes (~15 distinct expert slots a layer, not 25). Not a lever except via fewer rows |
| NCCL (80 f32 all-reduces + 2 bf16 Engram + 1 gather) | 3.19 (36 us each) | 4.27 (50 us each) | unknown; RDMA ring gathers, 83 a pass | TF: all-gather of f32 partials, summed in rank order in the consumer (`comm.zig:1-3`, `prompt.zig:131-134`), over its own RDMA ring (`rdma_gather.cu`), no NCCL in the hot path |
| hyper-connections (`hc_sub_partial` + `hc_sub_finish`) | 2.01 | 2.44 | ~1-1.5 | TF puts `_hc_mix_part` + `_hc_sinkhorn` on a side stream (`tri_hc.zig:122-140`, `round.zig:429-434`) |
| router / indexer f16 GEMV (`dense_f16w_vec_kernel` g96) | 0.81 | 1.00 | - | fine |
| attention, indexer, norms, rope, rot_in, transposes, MoE small kernels | 3.3 | 4.0 | similar | ~1500 of the ~2100 nodes a pass |
| idle inside the pass (graph node gaps, NCCL skew) | 2.0 | 3.4 (incl. ~1.3 ms Markov host loop) | ~1.5 | ~1 us a node is the graph floor |
| **pass span** | **32.05** | **53.4** (52.1 without the Markov prefix) | TF table 30.0 / 44.9 is stale (`draft.zig:657-659`, "carried over") | |
| draft round overhead (block span 6.6 = 4.2 kernels + 2.4 idle; Markov 1.3 kernels + host) | - | ~8 ms a round | 3.5 (constant) | round period measured 60.03 ms = 53.4 + 6.6 |

Missing numbers and the cheapest way to get them:
- TF per-kernel pass profile: `nsys profile --cuda-graph-trace=node` on the BLACKFOLD head container for one 48-token request, then
  `grid_breakdown.py` on its CSV (pass split at the head gather `dsv41_rdma_collect`). One fleetq window; settles TF's 1-row pass and its NCCL-vs-RDMA cost.
- Bits per weight per tensor (EAQ varies them): read the EXL3 K2 per tensor from the rank-cache meta; all GB/s figures below assume ~3 bpw unless marked.
- Distinct experts a verify layer: dump routes (`INTARSIA_DSV41_ROUTES=1` exists) for a few 4-row passes.
- NCCL all-gather vs all-reduce latency at 20-80 KB on the two Sparks: `nccl-tests all_gather_perf / all_reduce_perf -b 16K -e 128K`, 10 min slot.

## 2. Ranked changes (cheap, high confidence first). ms/token at 2.3 tokens a round.

**1. Fewer verify rows (lever 2): `INTARSIA_DSV41_SPEC_K=2` now; then TF's `chooseK`.** `crates/intarsia-models/src/dsv41_engine.rs:708-709` fixes k = 3;
TF picks k per round from the confidence head: survival = product of sigmoid(conf), value = expected tokens / (round_ms[rows] + draft_ms) (`draft.zig:666-685`).
Evidence: our 4th row costs 7.1 ms and buys 2.29 - 2.22 ~ 0.07 tokens (TF gets 2.22 tokens from 3.2 rows with the same drafter and checkpoint); a row pays
only above 0.28 tokens. Expect round 60 -> ~53 ms at ~2.2 tokens: **-2 to -2.5 ms/token** (confidence high for k=2 beating k=3; the lane's -0.5..-2 is an
under-estimate). Numerics: none (same kernels, fewer rows; verify-row bits already differ from 1-row). Gate: tokens/round and tok/s over the localeval sweep + spec golden.

**2. EXL3 dense linears at bandwidth (lever 1, but bigger and differently than planned).** Measured rates (grid x blk, us/call -> GB/s at ~3 bpw):
wo_a slice 1024x4096 g8 (4 a layer): split-K 16.8 us = 90 GB/s at 1 row, GEMV 35 us = 43 GB/s at 4 rows; wkv 512x5120 g4: 12.7 / 41.8 us = 75 / 23 GB/s;
wq_a 1280x5120 g10: 27.6 / 46.3 us; wq_b 1280x16384 g128: 65 us = 117 GB/s both; wo_b 4096x5120 g40: 61.6 / 71.4 us = 123 / 106 GB/s; head g505: 1.06 ms = 234 GB/s (6-bit).
Only the head saturates. Three causes, all visible in `kernels/cuda/exl3.cu`:
 (a) `splitk_count` (exl3.cu:891) stops at the first S with `nb*S >= 48` (1 block per SM); TF's plan doubles `sk` until `nb*sk >= 192` and keeps `kt/(sk*wk) >= 8`
     (`zig/src/families/dsv41/exl3_linear.zig:23-41`; grids 64/320/32/80/64 for 4096x1024/4096x5120/5120x512/5120x1280/1280x4096). The null microbench result
     for wo_b at S=2 (62 vs 59 us) is consistent with bytes-in-flight being the limit, not block count alone: 640 warps x one k-step ahead ~0.5 MB in flight vs the
     ~2-3 MB the head and the MoE kernel keep in flight (Little's law at 230 GB/s).
 (b) `linear_kernel`/`split_kernel` prefetch one k-step in registers (`cur/nxt`, exl3.cu ~420-470); our own `grouped_cp_kernel` (exl3_moe_cp.cu:1-80) uses a
     3-stage cp.async ring per warp and reaches ~230 GB/s on the same trellis format. TF's dense kernel: "coalesced register loads of the next k-step" plus 4-warp blocks.
 (c) one launch per matrix: TF groups up to 8 same-width matrices into one `glinear` launch (`exl3_linear.zig:10, 78, 175-215`: wq_a + wkv + compressor as one
     group, wo_a's slices as one, `RotOut`/`RopeOut` epilogues fold the next rotation and RoPE in, `:140, :144`). Ours: 4 wo_a launches of 1.5 MB each at 17-35 us.
 Change: route dense decode linears (rows <= 16) through a grouped-linear launch built on `grouped_cp_kernel`'s loader (one "expert place" per matrix, SK so that
 blocks >= 192, 4 warps, split sums in the epilogue), batching same-K matrices; at minimum fix (a) and apply split-K to verify rows (`INTARSIA_DSV41_EXL3_SPLITK_VERIFY=1`
 and raise `INTARSIA_DSV41_EXL3_SPLITK` above 128 so g40/g128 split too; runtime.rs:930-934). Expect 17.2 -> ~9 ms at 4 rows, 11.9 -> ~8.5 at 1 row:
 **-3 to -4 ms/token** (confidence medium-high on direction; the floor needs the bpw check). Numerics: near-tie (new fixed sum order, deterministic per shape),
 same class as the accepted split-K. Effort: 2-4 days. Gate: `bench_splitk`-style microbench per shape >= 200 GB/s, 3-layer eager-vs-graph bitwise tests, near-tie probe, MMLU 0.879 +- 0.01.
 Cheap first step (hours, env only): A/B `SPLITK_VERIFY=1` + threshold 256 in one window; expect -3 to -5 ms a verify pass from the g4/g8/g10 calls alone.

**3. Draft round on the device: argmax, Markov loop, confidence.** `runtime.rs:596-640` runs `block_size` = 5 Markov steps (not k) each as a separate graph
with a D2H sync, downloads all verify logits (4 x 129280 f32) for a host argmax, and the Markov head GEMV (`dense_f16w_vec_kernel` g16160, 66 MB f16) costs 261 us x 5 =
1.3 ms a round. Measured host-side idle a round: 2.4 ms before the block + ~1.3 ms around the Markov steps. TF: `roundArgmax` on device + one sync (`model.zig:742-746`),
the whole draft pass incl. Markov steps and confidence is one graph keyed by (N, steps), one sync, two small DtoH copies (`draft.zig:115-132, 468-480`).
Steps: (i) loop only k steps (one line): -0.5 ms kernels + 2 syncs; (ii) device argmax of verify rows (`kernels/cuda/argmax.cu`, same first-strict-max semantics) and
download 4 ints; (iii) fold Markov embed/head/argmax into the draft graph. Expect **-1 to -1.3 ms/token**. Bitwise for (i) and (ii); (iii) bitwise if the add order
logits + bias is kept. Effort: (i) 1 h, (ii) 0.5 day, (iii) 1-2 days. Gate: `speculative_greedy_equals_plain_greedy`, spec golden 640/640, round period under nsys.

**4. Collectives: all-gather + rank-order sum instead of all-reduce, then NCCL env A/B.** 80 f32 all-reduces at 36-50 us = 3.2-4.3 ms a pass. At world 2, NCCL ring
all-reduce is two network hops (reduce-scatter + all-gather); an all-gather of the 20-80 KB partial is one. TF does exactly this and sums in the consumer
(`comm.zig:1-3`; its `_hc_post_only` adds the gathered partials). Ours: `Extra::AllReduceInto` (graph.rs:247, 1126; nccl.rs:201) then `ik_f32_to_bf16` (80 calls a pass):
replace with `ncclAllGather` into `[2, t, 5120]` and fuse "sum ranks in order + round" into that convert kernel. Bitwise at world 2 (f32 add is commutative, so
p0 + p1 equals NCCL's sum; order only matters at world >= 3). Expect 36 -> ~22 us: **-0.5 to -0.7 ms/token**, confidence medium (needs the nccl-tests number).
Also A/B `NCCL_MAX_NCHANNELS=1` in the same window (LL protocol is already in use, per the kernel name). Effort: 1 day. An RDMA ring like TF's (`rdma.zig`, host proxy
thread, `dsv41_rdma_stage`/`collect` kernels with PDL) is 1-2 weeks and only worth it if the profile in section 1 shows TF's gathers under ~15 us.

**5. Fewer graph nodes: fold rotations and finishes.** Per pass: `rot_in_kernel` 376 calls (~0.8 ms), `split_finish_kernel` 247 calls, `transpose01_kernel` 160 calls
(wo_a group/token-major), `ik_f32_to_bf16` 80. TF folds the input rotation into the previous linear's epilogue (`RotOut`), reduces split-K in the last-arriving block
(`linear.cu:~200-240`, `counters`), and feeds the attention merge's rotated output straight into wo_a (`_sparse_attn_merge_rot2`, `tri_attn.zig:48-56`). ~700 fewer nodes
at ~1 us each plus their own time: **-0.6 to -0.9 ms/token**. Bitwise where it is pure fusion (rot_in, finish). Effort: 2-3 days, falls out of item 2 if the grouped
linear takes the TF epilogues. Gate: node count per pass under nsys, 3-layer bitwise tests.

**6. L2 prefetch of the next weights (TF `TF_DS_L2_PREFETCH`).** `tf_ds_l2_prefetch_kernel` (`zig/kernels/cuda/dsv41_ops.cu:209-258`): one 128-thread block on a
prefetch stream issues `cp.async.bulk.prefetch.L2` in 1 MiB waves paced at ~230 GB/s (`ops.zig:176-189`), three launches a layer (after q linears: wo_a + wo_b;
after wo_b: router + shared expert; after experts: next layer's attn linears), 16 MiB budget (`round.zig:355-396, 768`). It fills the DRAM idle time during collectives,
hc and attention (~150-250 us a layer here). Expect -1 to -2 ms a pass: **-0.5 to -0.9 ms/token**, confidence low-medium (GB10 L2 size and the sm_121 `cp.async.bulk.prefetch`
path must be checked). No numerics change. Effort: 2 days (second captured stream with fork/join events). Do after items 2 and 4: it hides latency those also remove.

**7. HC side stream (TF `TF_DS_HC_SIDE`).** Our two hc kernels are 2.0-2.4 ms a pass on the critical path, latency-bound (160 blocks, 16-21 us). TF keeps only
`_hc_post_only` + `_hc_finish_s` on the main stream and runs the dots + Sinkhorn on a side stream joined at the next mix (`tri_hc.zig:57-66, 122-140`; `round.zig:1039-1042`).
Expect -0.5 to -1 ms a pass (**-0.2 to -0.4 ms/token**). Bitwise (same kernels). Effort: 1-2 days. Low priority.

Sum of 1-5: roughly 60 -> 40-43 ms a round at ~2.2 tokens = 18-20 ms/token (50-55 tok/s). TF's 16.5 needs 6-7 as well, or its RDMA gathers.

## 3. What in the current levers and brief does not hold

- **Lever 3 (MoE at 4 rows, -0.9 to -1.7) has no support.** Both MoE passes run at ~230 GB/s for ~3.3 bpw experts; the 4-row pass reads ~2.2x the bytes (~15 distinct
  slots a layer from routing overlap of adjacent tokens). The kernel is the one part of the pass already at TF's level; only fewer rows (item 1) cut it. Minor: at 4 rows
  `MT = ceil(maxm/16)` uses `maxm = P = 28`, so grid z doubles (28x9x16) and half the programs exit at `rows_sh[0] < 0`; a few us a launch, not worth a lane.
- **Lever 1's estimate (-1.4 to -2.3) is low by about 2x** because it assumes the 1-row split-K rate (12 ms) is the target; the target is bandwidth (~8 ms), and the
  g40/g128 linears also run at half bandwidth at 1 row (see item 2).
- **Lever 2's estimate (-0.5 to -2) is low**: the measured marginal value of row 4 is ~0.07 tokens for 7.1 ms (section 2, item 1).
- **Lever 4 (-0.5 to -0.9) is plausible, but the brief's "TF gathers over an RDMA ring" hides the cheaper half**: TF's gain is partly the all-gather shape (one hop) and
  the fused rank-order sum, which NCCL can do today (item 4).
- **Lever 5 (gaps -0.5 to -1)**: the idle inside the verify pass includes ~1.3 ms of the Markov host loop that the pass split attributes to the verify prefix, and
  the 2.4 ms "block" idle is host work between verify and draft (argmax over 4 x 129280 on host, truncate, tp_remote). It is item 3, not a graph-gap problem.
- **"A 4-row verify pass is 52.9 ms and GPU bound"**: 53.4 ms span includes 1.3 ms Markov kernels + their syncs; the verify pass itself is ~52.1 ms with 50.0 ms of kernels.
- **"TF's 1-row pass is presumably much faster than our 31 ms"**: supported only by inference. TF's `round_ms` table (30.0/35.2/39.9/44.9) is a Python-build constant
  (`draft.zig:657`), and its Zig build measures 36.7 ms a round at 3.2 rows incl. ~3.5 ms draft, so its 1-row pass is <= ~25 ms if its per-row cost is >= the MoE
  bytes (~3.8 ms). A TF nsys profile (section 1) is the one measurement that would settle it.
- **"Each extra verify row costs 7.1 ms, about half MoE"**: holds (MoE +3.7, EXL3 +1.8, NCCL +0.4, other +1.2 per row). After item 2, a row should cost ~4.5 ms, which
  moves `chooseK`'s break-even to ~0.17 tokens a row and makes k=3 pay more often; retune after item 2 lands.
- TF's EXL3 kernels in the blackfold tree are cubins; `lane_gemv.cu`, `qmm_*.cu`, `experts*.cu` there are Nemotron's 4-bit affine kernels, not trellis. The porting
  map already says this; the open TF source for the trellis kernels is `/opt/code/TensorFold/src/tensorfold/cuda/exl3/` (`linear.cu`: last-block split reduction via
  `counters`; `experts_grouped.cuh`: the grouped kernel ours was written from).
