# sk: single-stream CUDA prefill of the 35B on a Spark, against vLLM (2026-10-25)

Lane `sk`, branch `lane-sk/spark-prefill`, base origin/main `cf84efa4` (PR #149 merged). DGX Spark spark2 (GB10, sm_121), `Qwen3.6-35B-A3B-NVFP4`,
default recipe (chunk 2048, PF on, fp8 lt on), default env. Raw logs: `2026-10-25-sk-spark-prefill/raw/`, scripts: `scripts/sk/`.
Every GPU use was a fleetq job on spark2 (ids in section 7). **Status: L1 built and gated; default-on for the 35B on CUDA (recipe), opt-out `INTARSIA_PREFILL_MACRO=0`. Draft PR, not merged.**

## 0. Premise check: the gap the brief describes is gone at 4k

The brief starts from 10-22 (-9 % at 4k, -15 % at 16k vs vLLM). Since then PF (#131), the cuBLASLt fp8 projections (#117 lt, default for W8A8
since 10-23) and the GEMM-pair kernels landed. Fresh numbers, spark2, `localeval speed` sizes 1024/4096/16384, gen 128, 3 reps, thinking off,
one fresh `intarsia serve` per round (default flags, `--max-seq 20480 --mem-budget 45`), vLLM = spark2's own `retro`-role server (serving `guidance`
and `capcom`; `--max-model-len 131072 --max-num-seqs 2 --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --gpu-memory-utilization 0.32
--enable-prefix-caching --enable-chunked-prefill`), 3 interleaved rounds (round 2 reversed), one fleetq slot (#5438, `--quiet gpu,cpu`):

| prefill tok/s, median of 3 round medians (min-max) | 1k (993 tokens) | 4k (4089) | 16k (16655) |
|---|---|---|---|
| intarsia main `cf84efa4` | 5255 (5172-5308) | 6547 (6474-6573) | 6015 (6000-6025) |
| vLLM (spark2) | 5470 (5372-5475) | 6382 (6362-6409) | 6299 (6288-6306) |
| intarsia vs vLLM, per request | -3.9 % (localeval: overhead-dominated at 1k) | **+2.6 %** | **-4.5 %** |
| marginal prefill (delta tokens / delta TTFT) | | 1k-4k: 7117 vs 6758 (+5.3 %) | 4k-16k: 5866 vs 6271 (-6.5 %) |
| TTFT s | 0.189 / 0.182 | 0.624 / 0.640 | 2.769 / 2.643 |

(`raw/base3/`, aggregated by `scripts/sk/agg.py`; decode: main 85.3 / 82.8 / 75.2 tok/s at 1k / 4k / 16k, vLLM 80.0 / 78.8 / 76.2.) At 4k main is ahead;
the remaining gap is at 16k (-4.5 % per request, -6.5 % marginal), where main's throughput falls from 6547 to 6015 tok/s while vLLM's stays flat.
The vLLM reference is spark2's server, not spark1's as the brief names: same checkpoint and flags, same box as the intarsia run, so no cross-box
difference; it reproduces the 10-22 figure of the same server (6389 / 6140 then, 6382 / 6299 now).

The benchmark prompts matter for the levers: the 4k prompt is 4089 tokens = 2048 + 2040 + 1 (the hybrid schedule runs the last token alone), so there
is no short tail; the 16k prompt is 16655 = 8 x 2048 + 271 (+1): a 271-row tail pass that streams all expert weights again.

## 1. Where a prefill goes (nsys `-t cuda`, `generate --max-tokens 1`, spark2, main; `raw/prof-main/`)

Prompts as lane sp2: 14000 / 55000 chars of `tools/prompts/corpus_docs.txt` = 4189 / 16355 tokens (4189 = 2048 + 2048 + 93: the 4k profile carries a
93-row tail pass that the localeval 4k prompt does not have). Kernel time summed by family, % of the kernel total (674.7 ms / 2643.0 ms):

| op family | 4k ms | 4k % | 16k ms | 16k % | note |
|---|---|---|---|---|---|
| routed experts, `moe_w4a16_kernel` gate/up + down (PF on at <= 2048 rows) | 262.2 | 38.9 | 1047.0 | 39.6 | 3.28 ms per layer and 2048-row chunk |
| routed experts, 93-row tail (`moe_mma_kernel`) | 51.7 | 7.7 | 0 | 0 | one full weight pass for 93 rows |
| GDN delta rule (`gdn_delta_split_kernel`) | 111.5 | 16.5 | 439.8 | 16.6 | strict left-to-right chain per head: latency bound, bitwise-pinned |
| dense matmuls (fp8 lt `nvjet`/`xmma`, `lt_actq8`, bf16 router / in_a / in_b via cuBLAS) | 97.3 | 14.4 | 342.3 | 13.0 | vendor GEMMs |
| norms (`rmsnorm`, `gated_rmsnorm`) | 36.3 | 5.4 | 142.2 | 5.4 | at the byte floor |
| `moe_reduce` + `moe_combine` | 32.4 | 4.8 | 128.6 | 4.9 | 234 us per call, ~320 GB/s: at the floor |
| attention `flash_kernel` (10 layers) | 24.1 | 3.6 | 309.6 | 11.7 | grows with context (16k: 66 TFLOP/s) |
| GDN conv | 21.7 | 3.2 | 84.9 | 3.2 | |
| shared expert (`nvfp4_reg`) | 16.5 | 2.4 | 68.0 | 2.6 | |
| rest | 21.1 | 3.1 | 80.6 | 3.1 | |

Idle gaps under 20 us total 6.9 / 17.1 ms (1 % / 0.6 %): not a lever. The engine's own `--op-table` agrees on the order (experts 41 %, matmul 22 %,
GDN delta 13 %, fp8 matmul 10 % of a synchronised, timed pass; `raw/prof-main/*/optable.err`).

The routed experts are 51 % of a 4189-token prefill and 40 % of a 16k one. Their time is a serial sum of weight DRAM streaming, A-tile L2 traffic and a
fixed skeleton (lane sp2 ablations, `docs/results/2026-10-22-sp2-cuda-prefill.md` section 2), and every 2048-row chunk streams all ~18 GB of expert
weights again. A call with more rows reads each weight once for all of them (lane mp microbench, `docs/results/2026-10-23-mp-cuda-moe-prefill.md`,
launcher incl. grouping and reduce, uniform routing: 2048 rows 3506 us with PF, 4096 rows 5504 us, i.e. 1508 us less than two 2048-row calls per layer,
x 40 layers = 60 ms = 9.7 % of the 4089-token TTFT).

## 2. Scope: levers, bitwise status, expected gain

All gains are against main's TTFT at the benchmark prompts (0.624 s at 4k, 2.769 s at 16k).

| # | lever | bitwise? | expected | notes |
|---|---|---|---|---|
| L1 | **macro prefill pass**: run several consecutive 2048-row chunks of one prompt layer by layer in one pass (the existing packed-prefill machinery with one state, each chunk a segment); the row-wise kernels (experts, shared expert, norms, quantisers) see all rows and read their weights once, attention, KV append and the GDN recurrence run per chunk; the vendor GEMMs (cuBLAS / cuBLASLt pick their tiling by row count) run per chunk | yes by construction if every batched kernel is row-wise bitwise, the vendor GEMMs stay chunk-sized | 4k: -60 ms (+10 %); 16k: the 271-row tail pass (~55 ms) goes away and the experts run at 8192 rows per pass (microbench: 8192 rows 9840 us launcher vs 4 x 3506 us): about +8 to +12 % | opt-in (`INTARSIA_PREFILL_MACRO=<rows>`), because a long pass delays a request that arrives meanwhile; the chunk stays 2048, so PR #138 is untouched |
| L2 | extend the PF (L2 prefetch) schedule of the gate/up kernel past 64 pairs per expert (> 2048 rows per call) | yes (same sums) | +1 to +2 % on top of L1 at 4k (PF is worth ~1.4 % at 4k at 2048 rows) | only useful once L1 makes calls larger |
| L3 | `moe_reduce` fused into the down GEMM (last arriver per token and column block) | yes | <= 3.6 % at 4k, needs a counter array in shared scratch | lane sp2 section 4; not attempted |
| L4 | `moe_reduce` + `moe_combine` + next `RmsNormAdd` fused | yes | <= 1.5 % | |
| L5 | flash attention schedule at long context | no (summation order of the softmax) | up to 3 to 5 % at 16k | opt-in only; 16k only |
| L6 | GDN delta (16.5 %) | bitwise-pinned strict chain | 0 | latency bound; nothing bitwise |

Sum of the bitwise levers at 4k: L1 alone is estimated at ~10 % (measured 5.6 %, section 4), well above the 3 % bar, so the lane went on to step 3 with L1.
L2 to L4 were not built.

## 3. What was built (L1, macro prefill pass)

`Model::forward` of a hybrid (recurrent-state) sequence cuts the prompt into the usual chunks (2048) and runs up to `macro_rows` rows of consecutive chunks as
ONE layer-by-layer pass. It reuses the packed-prefill machinery (`Seqs::Packed`, `attn_block_packed`, `gdn_block_segs`): each chunk is a segment of the
same `SequenceState` with its own first position (`Seg::pos`, new), so attention, KV append and the GatedDeltaNet conv + delta rule run chunk by chunk, in
order, with the shapes of the plain schedule. The row-wise kernels (routed experts, shared expert, norms, quantisers, top-k, combine) run once over all rows,
so each expert weight is read once per pass instead of once per chunk.

* **First attempt, not bitwise.** Everything batched over all rows. Greedy ids (256, `raw/ids1/`) equal main at 1k (one chunk) and **differ at 4k and 16k**
  for macro 4096, 8192 and 16384 (the three agree with each other). The vendor GEMMs (cuBLASLt fp8 projections, cuBLAS bf16 router / in_a / in_b / shared
  gate, `Exactness::Fast`) choose their tiling and split-K by the row count, so a call of 4096+ rows is not the bits of two 2048-row calls. This is the same
  reason the opt-in packed prefill is not bitwise on CUDA (it fails `real_35b_packed_prefill_equals_the_local_engine` on main too, `raw/g-packed-main.log`)
  and why chunk size changes bits on CUDA (tail merge, PR #138's busy-chunk KL).
* **Second attempt, bitwise.** The ops `matmul`, `matmul_fp8`, `matmul_fp8_pair`, `matmul_pair` run once per chunk inside a macro pass (`Model::exec_call` slices
  the activation and the outputs by the pass's segment table), everything else over all rows. The pinned kernels (`moe_w4a16`, `nvfp4_reg`, norms, quantisers)
  are row-wise bitwise, so only the vendor GEMMs needed the split. Diagnostic knob `INTARSIA_PREFILL_MACRO_SPLIT=<ops>|none` for bisecting.
* Schedule: `scheduler::macro_group_len` spreads the chunks evenly over as few passes as the cap allows (a 16655-token prompt at 8192 is 9 chunks in 3 passes
  of 3, not 8192 + 8192 + a 271-row pass). `engine.rs` `prefill_step` hands `forward` the headless body in macro steps (single-sequence engine, no prefix
  store, no prewarm); the last token still runs alone with the head, as before.
* Knobs: `INTARSIA_PREFILL_MACRO=<rows>` (docs/tuning.md), `LoadOptions::prefill_macro`, recipe `[tuning.<backend>] prefill_macro`; `recipes/qwen3_5_moe.toml`
  sets 8192 for CUDA. Rows are clamped to the context (`--max-seq`) and to whole chunks, off below two chunks and under tensor parallelism. Plan: the dry-run
  probes the pass's row count too, so the kernel scratch and admission cover it.
* Not covered: `serve --max-batch` (the batched scheduler feeds one chunk per iteration; unchanged), the prefix store, `HeadMode::AllLogits` (accuracy tooling),
  MTP states. Prefix-reuse recurrent snapshots are taken after a pass, not after every chunk (reuse granularity only, no numerics).

## 4. Speed (same-window localeval, spark2, 3 interleaved rounds, round 2 reversed, `--quiet gpu,cpu` slots #5454 and #5475)

Final window (#5475, `raw/ab2/`): main `cf84efa4` vs the branch with its default (recipe 8192) vs spark2's vLLM:

| prefill tok/s, median of 3 rounds (min-max) | 1k | 4k (4090 tok) | 16k (16657 tok) |
|---|---|---|---|
| main | 5196 (5188-5238) | 6559 (6496-6567) | 6026 (6024-6038) |
| branch, default (macro 8192) | 5284 (5167-5291) | **6924 (6885-6932)** | **6592 (6591-6614)** |
| vLLM | 5378 (5368-5502) | 6418 (6342-6421) | 6296 (6292-6297) |
| branch vs main | +1.7 % (bands overlap: noise) | **+5.6 %** (every branch round above every main round) | **+9.4 %** (same) |
| branch vs vLLM | -1.7 % | **+7.9 %** | **+4.7 %** |
| TTFT s main / branch | 0.191 / 0.188 | 0.623 / 0.591 | 2.764 / 2.527 |

Earlier window (#5454, `raw/ab1/`, dev2 build with the env set): macro 8192 +5.4 % / +10.0 % and macro 16384 +5.5 % / +11.4 % at 4k / 16k, 1k unchanged. 8192 is the
default: 16384 buys 1.4 points at 16k for 336 MB more kernel scratch (see below). Decode is unchanged (branch 83.9 / 81.5 / 74.2 vs main 84.8 / 82.3 / 74.5 tok/s, inside the
2 % decode noise). Footprint: `plan show` scratch 84 MB -> 336 MB at 8192 (672 MB at 16384; `raw/plan1/`), workspace rows grow to the pass size (inside the
memory plan, so the KV budget shrinks, not the box); MemAvailable with the server up is 1.0 GB lower (`raw/ab2/mem.txt`).

## 5. Gates (spark2 unless noted; dev3 = this branch's build, base = origin/main `cf84efa4`)

| gate | result | raw |
|---|---|---|
| 256 greedy ids vs main at 1k / 4k / 16k (1006 / 4074 / 16242 tokens): branch default (macro on) and `INTARSIA_PREFILL_MACRO=0` | **identical** (all six) | `raw/ids2/`, `ids2.log` (#5468) |
| logits of the last prompt row and of 16 greedy decode steps, macro 4096 / 8192 / 16384 vs off, prompts 1.5k / 4k / 8.3k / 16.6k, headless + last token and one whole `forward` | **bit for bit identical** (24 combinations, 9 macro-engaging lengths; passes counted) | `real_35b_macro_prefill_is_bitwise_the_plain_schedule`, `g-macro-bitwise.log` (#5470); also on the dev2 build `macro-bitwise1.log` |
| batched == solo, 35B: `real_35b_batched_decode_is_bitwise_the_solo_step`, `real_35b_batched_engine_equals_the_local_engine` | 2 passed | `g-batch-default.log` (#5471) |
| `real_35b_batched_engine_long_prompts_equal_the_local_engine` (LocalEngine reference with macro on = BatchedEngine plain 2048 chunks, prompts 2.6k / 4k / 16k) | 1 passed | `g-long-prompts.log` (#5472) |
| `intarsia kernels test --backend cuda --quick` | 77 kernels, 531 cases, 0 FAIL (no kernel changed) | `g-ktest.log` (#5476) |
| `plan show` knob off (`MACRO=0`) vs main | json byte-identical (md5 dd5a67be...); with macro on only the scratch bytes differ | `raw/plan1/` |
| CPU tiny hybrids (dense FFN, MoE W8A16, MoE W8A8): macro vs plain, `LocalEngine` end to end and `Model::forward` logits + 4 decode steps; scheduler helpers | pass | `batch_recurrent.rs` `macro_*`, `tests/scheduler.rs` `macro_*` |
| `cargo test --workspace` (Mac, CPU) | 930 passed, 0 failed (`raw/cargo-test-workspace-mac.log`) | |
| KL goldens | not applicable: the change is bitwise (logits identical), and the golden gates score rows through `AllLogits`, which the macro schedule does not take | |

Pre-existing failures reproduced on main (same test, same binary family, `g-klgate-main.log`, `g-packed-main.log`), not caused by this change: `real_35b_busy_chunk_prefill_kl_gate`
(chunk 1024 vs 2048, above the ref-noise floor; its control, hand-chunked 2048 vs `forward`, is KL exactly 0 on this branch at 1.5k to 16k tokens, i.e. `forward` with
macro on is the hand-chunked bits) and `real_35b_packed_prefill_equals_the_local_engine` (opt-in packed multi-prompt prefill; vendor GEMMs again).

## 6. Decisions and risks for the coordinator

1. **Default-on is in the PR** (recipe `prefill_macro = 8192` for CUDA 35B): bitwise, +5.6 % at 4k, +9.4 % at 16k, neutral at 1k, per the brief's default rule. Revert in one line
   (delete the recipe key) or per run with `INTARSIA_PREFILL_MACRO=0`.
2. **Overlap with PR #138** (chunk 8192, waits on the user): this delivers its single-stream speed-up (+5.6 / +9.4 against #138's +5.7 / +10.4 at 8192 without PF)
   with the 2048 numerics intact, so the busy-chunk KL gate that blocks #138 does not arise. Only single-sequence serving gets it; `--max-batch` serving does not, and #138's
   question stays open for that path.
3. A request that arrives during a pass waits for it (about 0.6 s per 4096 rows): in the single-sequence engine it waits for the whole prefill anyway; cancellation is checked per pass.
4. The gap premise in the brief was stale (section 0): at 4k the engine was already ahead of vLLM, and the remaining 16k gap is closed by this change on the same box.
   The 1k gap (about 2 to 4 %, one chunk, localeval calls 1k overhead-dominated) is untouched: L1 does not apply to a single chunk.
5. Not done: L2 (PF past 2048 rows), L3/L4 fusions, a macro schedule for the batched engine's idle-solo case; unanimis capture was refused (this session is bound to project unanimis, not intarsiaml).

## 7. Fleet log (all spark2; vLLM `retro` left up, nothing taken down)

| id | what |
|---|---|
| #5432, #5445, #5450, #5451, #5467, #5469, #5496 | builds (main, dev, dev2, dev3 binaries and CUDA test binaries), `--load cpu` |
| #5438 | localeval main vs vLLM x3 (baseline, `raw/base3/`), `--quiet gpu,cpu` |
| #5439 | nsys + op-table profile of main (`raw/prof-main/`), `--quiet gpu,cpu` |
| #5448, #5468 | ids: first attempt (`ids1`), final (`ids2`) |
| #5452, #5470 | macro bitwise test on dev2 / dev3 |
| #5454, #5475 | localeval A/B (`ab1`, `ab2`), `--quiet gpu,cpu` |
| #5466 | `plan show` (`plan1`) |
| #5471, #5472, #5473, #5474, #5476 | batch_cuda_real gates, KL gate, packed test, kernels test on dev3 |
| #5504, #5505 | the KL gate and packed test on main (controls) |
