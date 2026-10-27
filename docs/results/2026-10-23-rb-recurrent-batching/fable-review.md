> Archived 2026-10-26: Fable review of PR #114 (lane rb) at fa596759, salvaged from the lane worktree. Findings describe that tree and may since be fixed on main.

# Review of PR #114 (lane rb, hybrid continuous batching)

## Headline

The roofline arithmetic is right (I reproduce 161/164 tok/s at N=4 and 230/266 at N=16 from the per-group bandwidths). The design's three big premises are wrong in the other direction: it overestimates the cost of exactness (CUDA's fp8/nvfp4 GEMVs are already row-exact *by construction*, see `kernels/cuda/fp8.cu:60-110`), overstates the B=1 fusion regression (4.6%, not 12%), and underestimates host-side costs at N=16 (sampling at a 248,320-token vocab, prefill stalls). The result: the 18.5-day plan with no speed-up until day 10.5 can be a ~5-day path to ~140 tok/s at 4 streams, and the "bitwise or KL?" question (O11) mostly dissolves.

## Ranked findings

### Blockers (for the plan, not the prototype PR)

**B1. `check_batch_ops` compares op multisets, not kernel picks; the MoE arm will silently go non-bitwise on CUDA.** `crates/intarsia-core/src/model.rs:1586-1597` dispatches `forward_folded` with T = b rows under `Seqs::Batch`. On CUDA, `quantmoe.rs:92,248,264` select `moe_experts_decode_v2_bf16` only for `m <= DECODE_MAX_TOKENS = 1`; T in 2..31 goes to `moe_experts_bf16` (grouped warp-per-row, `Exactness::Deterministic`, tolerance-admitted). Same `Op`, different kernel, so `plan.rs:315-325` passes. This is exactly the MTP verify-row failure mode, reproduced in the batch path. Today only the CPU-only guard in `allow_recurrent_batch` hides it. Fix: the batch plan check must compare kernel *picks* per op against the decode plan (the plan already records picks, `load.rs:359`), exempting only ops registered `rows_exact`. This one check would also have caught the MTP finding.

**B2. The roofline is a decode-only, greedy-only ceiling; the composer metric is neither.** Not modelled: (a) host sampling. `sampler.rs:37-51` does `select_nth_unstable` + sort over the vocab on the host per sampled row; at V = 248,320 that is ~0.5-1 ms per row (estimate), so N=16 sampled costs 8-16 ms per 62 ms step (-12..-22%); `model/batch.rs:193-199` also does one synchronous ~1 MB `download` per logits row. (b) Prefill stalls: a 1k prompt is 4 busy chunks of ~40 ms (measured 6.3k tok/s marginal prefill); with 256-token outputs that is 9% of wall at N=4 and 14% at N=16 (my `ceiling.py`). The lane's 0.05 ms/row host term is off by 10-20x for sampled requests. Realistic targets after the lane's S5: 4 streams 135-150, 16 streams 185-210 greedy, 160-185 sampled, unless top-k/top-p moves on device (1-2 days, worth +10-20% at N=16, more than K7's expert grouping).

### Major

**M1. The "17 days of row-exact kernels" premise is wrong: the CUDA decode GEMVs are already row-independent.** `fp8.cu:60-110`: warp per output row, no split-K, per-row `acc[m]` with the same lane-strided K partition and the same `fmaf` chain for every m, static W8A8 `input_scale` (no batch amax). Row r of the M=4 kernel is by construction bitwise row 0 of the M=1 kernel; same structure in `nvfp4.cu:5,20-68` (M ≤ 8). The fp8 pair kernel (`fp8.cu:124-130`) calls `fp8_gemv_body<1>` twice and `tests/fused_decode.rs` already proves it bitwise two GEMVs. "CUDA registers no `rows_exact` kernels" (`kernels.rs`) is a registry omission, not a numerics fact. K1 is a template-M extension to 16 (x staging in smem grows to 16 x K x 2 B = 64 KB at K = 2048, so stage fp8 x at 32 KB) plus a G-B unit test: ~2 days, not 4; K2 ~0.5 day. K3: `moe_decode.cu:225` already has T as a grid dimension in the down kernel; lifting `DECODE_MAX_TOKENS` is ~1 day. Total bitwise kernel work ~8-9 days, most of it needed for speed anyway.

**M2. S2 (4 days, "exact by construction, x1.0") is mostly wasted.** Its per-row dense projections (N weight reads: at N=4 the step is ~50 ms, slower per token than solo) are thrown away in S3, and the per-row `GatedRmsnormMatmulFp8` fallback (`gdn_batch.rs:453-474`) alone adds N x 252 MB per step (+17% at N=4). Bring up directly on the existing multi-row GEMVs at N ≤ 4 (see plan).

**M3. The B=1 regression is 4.6%, not 12%.** Of the 13,954 to 12,406 us gain in the fusion doc, 950 us is the MoE down v2 *kernel* (selected by `m <= 1`, unaffected by `batch_rows`); the four fusions sum to 202+71+207+83+11 = 574 us. Section 9, O-question 3 and LOG O13 inherit the wrong number. Avoidable without multi-row fused forms: the fusions are engineered bitwise their unfused sequences (`matmul_pair_is_bitwise_two_gemvs`, `matmul_fp8_pair_is_bitwise_two_gemvs`, the PRO norm staging comment `fp8.cu:20-33`), so relax `fused_ok`'s batch rule (`model.rs:1635-1680`) for fusions with a registered bitwise-equivalence test, and keep solo fused at B=1 while B ≥ 2 runs unfused multi-row. Dispatching the fused solo path only when one sequence happens to be active is the wrong fix: it makes a request's bits depend on load.

**M4. Expert grouping (K7, 4 days, S6) is nearly free with a pair sort.** The lane's grouped/per-pair gap is +2% at 4, +16% at 16. Sort the N x 8 (row, expert) pairs by expert before launching the decode kernel: CTAs for the same expert run adjacently, the duplicate 1.77 MB reads hit L2, and per-pair arithmetic is unchanged (the down kernel reduces each token's 8 experts in top-k order, not launch order), so it stays bitwise. Do it inside K3. Skewed real routing (fewer than 102 distinct experts at N=16) only helps. So: 0.57 GB/seq is correct for the per-pair kernel as written, an overestimate of ~5% at N=4 and ~20% at N=16 once pairs are sorted.

**M5. Prefill chunk size depends on load, so "bitwise to solo" is only "to solo at the same chunk schedule".** `scheduler.rs:960` picks busy chunk 256 vs solo 2048; on GPU the prefill GEMM tiles (and the recurrent snapshot positions, `engine/batch.rs:154-180`) then differ from a `LocalEngine` run. Pre-existing from Metal, but the doc should state it in section 10: decode rows can be batch-invariant; prefill is chunk-dependent and needs a KL gate (busy vs solo chunk), which is the vLLM/SGLang reality. Also for the MoE, a 256-token chunk reads almost all 256 experts per layer (2,048 picks), so cost per chunk is nearly flat in chunk size: busy chunk 1024 for the 35B cuts stall count 4x for ~1.3-1.5x per-chunk time. Measure and change the default for MoE recipes.

### Minor

**m1. `gdn_batch.rs:393-428, 468-474`: per-row GDN and out-projection calls run with the `exact_rows` mark on**, unlike `attn_block_batch` (`model/batch.rs:277,289`), which turns it off around per-row calls. Harmless on CPU (no `rows_exact` kernels); once CUDA registers them (K1), a 1-row call may select a different kernel than the solo step. Mirror the attention code.

**m2. `model/batch.rs:248-270`: the gated-attention prologue fold is all-or-nothing across rows**; the solo decision is per row. If any row's `gate_qkv_call` fails `fused_ok`, every row falls back to the unfused split, differing from the solo step of the rows that would have fused. Shape-based today, so no observable failure; assert agreement instead of degrading silently.

**m3. `model/batch.rs:167`: b = 1 in a batched server goes through `forward` and `graph_step.rs:178-180`, which recaptures whenever the state id changes.** At low load with slots alternating (LRU tie-break in `pick_slot`), every new request's first decode step recaptures (~a full eager step + instantiate). Perf only; K6 fixes it with one captured graph per slot or per bucket.

**m4. Memory plan (`model/batch.rs:91-109`) counts KV only**; live recurrent (63 MB) plus 4 snapshots per slot is 5 GB at N=16, allocated lazily by `state.rs:173-177` inside the scheduler thread (cudaMalloc during serving). Not binding on 128 GB; preallocate in `enable_batch` and count it.

**m5. Snapshot eviction defeats long-prefix reuse**: `prefill_chunk` snapshots after every busy chunk with keep = 4, so a 2k prompt holds only its last four boundaries; a 1k system prompt followed by different turns gets `cached = 0`. Same as solo; the pc lane is the fix, but say so.

**m6. `engine/batch.rs:154-180`: the `Some(m)` branch discards the `StepOut` of the first pass** instead of checking `StepOut::None` as `Seq::prefill_step` (`engine.rs:1144`) does.

**m7. Test coverage gaps**: the fixture is f32 (`act_dtype`, `kv_dtype` F32, `fp8_act: None`), and `to_fp8` converts only qkv/z/out, so the `(BF16, BF16)` `RmsNormAddMatmulPair`/`MatmulPair` arms and the MXFP4 arms of `gdn_batch.rs:343-389` never run (the 35B's `in_a`/`in_b` are BF16). Those arms cannot run on CPU at all; they need a CUDA tiny-hybrid test.

**m8. Spec guard and `Seq::build(.., recurrent)`** are correct. `pick_slot`/`restorable_len`/`truncate` ordering is correct. Solo and batched chunk schedules agree (`next_prefill_chunk_tm` with `avoid_one_row = false` in both, `scheduler.rs` vs `engine.rs:1120-1130`). I found no correctness bug in the CPU prototype's engine logic.

## What the 7 CPU tests cannot catch

Kernel selection changes under one `Op` (B1); M-dependent arithmetic (mma, grouped MoE, HIP WMMA); fused-vs-unfused bit differences; bf16 rounding between ops (f32 fixture); the "last prompt token alone" rule; graph/state binding and recapture; stream ordering of snapshot copies and restores; the memory plan; the BF16/MXFP4 fold arms (m7); any host-time or bandwidth number; expert routing skew; cancel/fail mid-pass leaving per-row GDN state half-advanced (correctly fatal, but untested on a device).

## Recommended staged plan (one lane, working days, aggregate tok/s at 4 / 16 streams)

| stage | content | days | expected 4 / 16 |
|---|---|---|---|
| R0 | G-B unit tests on spark1: fp8 GEMV M=1..4 row r == M=1 row 0; nvfp4 M=1..8; MoE decode v2 with T > 1 after lifting `DECODE_MAX_TOKENS`; register `rows_exact` on CUDA for what passes | 0.5 | n/a |
| R1 | S1 plumbing as proposed + kernel-pick comparison in `check_batch_ops` (B1) + `exact_rows` off around GDN per-row loops (m1) + recurrent/snapshot term in the memory plan (m4) + cap per backend | 1.5 | CPU only |
| R2 | CUDA bring-up on the *existing* multi-row GEMVs at N ≤ 4; GDN per row (option B, as prototyped); attention per row (exists); MoE via lifted `DECODE_MAX_TOKENS` with expert-sorted pairs (M4); fused ops unfused at B ≥ 2 under the equivalence rule (M3), solo keeps its fusions. G-C bitwise on tiny hybrid + 35B, G-F | 3 | ~140 / n/a (eager; GPU 23.6 ms + ~4 ms launch/host) |
| R3 | K1/K2 to M = 16 (fp8-staged x), multi-row pair/PRO forms, `--max-batch 16` | 3 | ~140 / 175-190 |
| R4 | Device-side top-k/top-p sampling and one `[b, V]` logits download (B2) | 2 | ~145 / 190-205 |
| R5 | K4/K5 state and position tables, K6 graph buckets (per-row launches are ~8-12% of an eager N=16 step; tables are required by graphs, not by bandwidth) | 4 | 150-160 / 205-225 greedy |

Critical path to the first shipped speed-up: 5 days (R0-R2, closes the vLLM gap at 4 streams). To the full result: 14 days versus 18.5, with a ceiling of 160-165 / 230-245 (per-pair, with a 1.6x GEMV compute penalty at 16 rows) and 265 at 16 with sorted pairs. Discount the 16-stream numbers 10-15% for prefill stalls in a concurrent benchmark; raise the busy chunk for MoE (M5).

## Exactness policy (O11)

Keep bitwise, but define it correctly: the gate is row-exactness *within the batch path* (G-B: row r of an M-row kernel equals row 0 of the same kernel at M = 1, for every M up to the cap), which gives batch-invariance by construction, and it is cheap here because every CUDA decode kernel that matters is already warp-per-row with a fixed K partition and no split-K; the only things to forbid are M-dependent kernels (mma paths, the grouped MoE kernel for T in 2..31, HIP WMMA from 16 tokens) and dynamic per-tensor activation scaling (the recipe uses static W8A8 scales, so none). The reference for G-C/G-D should be the batched engine at N = 1, which runs the same kernels; equality of that step to `LocalEngine` is then a separate, one-time admission like every other kernel (bitwise where the kernels are literally the same, which is all of them except the four fusions, whose fused-vs-unfused bitwise tests already exist). Prefill cannot be made chunk-invariant on GPU, so state that honestly and KL-gate busy-chunk prefill against solo-chunk prefill (that is where vLLM/SGLang-style gating belongs). The "KL fallback for batching" option is not needed: the expensive part of "bitwise to solo" was never the kernel numerics but the op-multiset rule that forced solo to drop its fusions, and that rule should be replaced by a kernel-pick plus proven-equivalence check (B1, M3).

Everything above is review only; nothing was edited, committed or pushed. The one scratch file is `/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/ceiling.py`.