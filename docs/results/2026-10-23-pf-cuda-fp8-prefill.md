# Lane pf: CUDA fp8 W8A8 prefill projection GEMM on GB10 (2026-10-23)

Target: DGX Spark GB10 (sm_121, CUDA 13.0), Qwen3.6-35B-A3B-NVFP4, tp=1, spark2 (profile, localeval, gates) and spark1 (accuracy
runs). Branch `lane-pf/cuda-fp8-prefill` from main `055775da`. Raw logs and scripts: `2026-10-23-pf-cuda-fp8-prefill/raw/`,
`scripts/pf/`. Companion data: `trace.json` and `techniques.jsonl` in `2026-10-23-pf-cuda-fp8-prefill/`.

## Headline

* **The kernel the brief asks for already exists on main, opt-in.** `matmul_fp8_gemm_lt` (cuBLASLt, native e4m3 tensor cores,
  static `input_scale` activation quantiser, one pinned algorithm per (N, K) so rows do not depend on the call;
  `kernels/cuda/fp8_gemm_lt.cu`, `INTARSIA_CUDA_FP8_GEMM_LT=1`, round 7). So this lane did not write a new GEMM: a CUTLASS vendor
  or a hand-written mma.sync kernel cannot beat 130 TFLOPS by much on a 243 TFLOPS e4m3 roof, and the cuBLASLt path already
  runs at vLLM's CUTLASS speed (round 7: 41.2 ms against 43.5 ms per 2048-token chunk). The lane measured it again against
  vLLM in one window, profiled it per kernel class, and ran every gate.
* **Speed (localeval, spark2, interleaved x3, vLLM in the same window).** The default path trails vLLM by -9.6 % at 4k and
  -14.0 % at 16k (vLLM +10.7 % / +16.3 %; `gaps.md` says +16.2 % / +15.9 %). With `INTARSIA_CUDA_FP8_GEMM_LT=1` and
  `--prefill-chunk 8192` intarsia is ahead of vLLM: **+8.6 % at 4k, +6.4 % at 16k**. At chunk 2048 (the 35B recipe default)
  lt alone is +1.8 % / -3.8 %: the lt GEMM closes most of the projection gap and the chunk size closes the rest. 1k stays
  -9 % on every arm (not a projection problem, see below). Decode is unchanged (83.8-84.0 / 81.3-81.7 / 73.8-74.2 tok/s on
  every arm, vLLM 77.8 / 77.3 / 74.7).
* **lt does not pass the gates cleanly, so no default-on proposal.** KL against all three goldens is inside the floors and
  equal to the default's (table below). But (1) the prefix-reuse `continue` rows fail at chunk 2048 on 2 of 5 prompts
  (`chat_reuse_4k`, `long_code_4k`; default passes 15 of 15 rows) and at chunk 1024 / 4096 on `chat_reuse_4k`, and (2)
  `acc-short` fails its baseline-relative gate (mean KL 5.87e-3 -> 6.83e-3, limit 6.75e-3). The cause of (1) is not
  found; row invariance of the lt GEMM itself holds (`matmul_fp8_gemm_lt_rows_do_not_depend_on_the_call` passes, M down to 5).
  Verdict for the coordinator: **lt stays opt-in.** `docs/decisions/LOG.md` has the row.
* **Where the time is** (roofline below): with lt on, the fp8 projections run at 131-143 TFLOPS (54-59 % of the e4m3 roof)
  and are 11-12 % of kernel time. The routed-expert kernel is 41-47 % of kernel time at 22-33 % of the bf16 roof, the GDN
  recurrence 16-18 %. The brief says MoE is not the lever; after lt it is the largest single class. One MoE experiment is
  recorded as a separate, unmerged branch (see "Out of scope but measured").

## Premise check (what the repo and the checkpoint already have)

| Item | Finding |
|---|---|
| CUDA prefill dense projections at T >= 256, default | `matmul_fp8_gemm_reg` (bf16-MMA register GEMM over e4m3-valued bf16, separate `fp8_actq` pass). 36-70 TFLOPS. |
| Opt-in alternatives on main | `matmul_fp8_gemm_f8` (own e4m3 mma.sync, about 65 TFLOPS), `matmul_fp8_gemm_cublas` (bf16 cuBLAS over an expansion), `matmul_fp8_gemm_lt` (cuBLASLt e4m3, 99-143 TFLOPS). |
| Checkpoint weights (nvidia/Qwen3.6-35B-A3B-NVFP4, modelopt MIXED_PRECISION) | fp8 e4m3 per-tensor: GDN `in_proj_qkv`, `in_proj_z`, `out_proj` (30 layers), attention `q/k/v/o_proj` (10 layers): 1279.4 M weights, 2.56 GFLOP per token. Each has a scalar `weight_scale` and a static `input_scale` (F32, one per tensor). MoE experts and the shared expert are NVFP4 (U8 + E4M3 block scales + `weight_scale_2` + `input_scale`); `lm_head` NVFP4; `in_proj_a/b`, router gate, embeddings, vision tower, mtp are bf16. |
| Activation scales | Static `input_scale` exists for every fp8 projection, so the lt path quantises activations with the checkpoint's own scale (vLLM's scheme). No dynamic scale is needed or implemented; a layer without one (input_scale <= 0) falls back to the bf16 weight-only kernel inside the same launch. |
| HIP | `kernels/hip/ik_f8_gemm.h` (lane hp, PR #115) is the HIP analogue; the CUDA analogue is `fp8_gemm_lt.cu`. |
| #40 (reuse judge) | Merged 2026-10-21 15:26 UTC. The 2026-10-21 LOG row's condition for re-evaluating lt is therefore met; the re-evaluation is this document. |

## Roofline

nsys, `intarsia generate --max-tokens 1`, chunk as named, single sequence, spark2 (job #4574, raw/prof1). Prompt 4189 and
16014 tokens. Kernel-time classes from `scripts/pf/classes.py` over `scripts/s1/rank.py` output (top-24 kernels; the rest is
in "other"). Roofs measured by `kernels/cuda_checks/mma_peak.cu` in round 6: bf16 m16n8k16 122 TFLOPS, e4m3 m16n8k32 243
TFLOPS (mma.sync, sustained). Work: fp8 projections 2.559 GFLOP/token; routed experts (8 x 3 x 2048 x 512 x 2 x 40 layers)
2.013 GFLOP/token.

**4189 tokens (about the 4k bucket)**

| class | default ms (%) | lt ms (%) | lt, chunk 4096 ms (%) |
|---|---|---|---|
| routed experts (moe_w4a16 + moe_mma) | 318.5 (41.8) | 316.1 (46.9) | 259.0 (41.0) |
| moe topk/group/reduce/combine | 36.0 (4.7) | 33.5 (5.0) | 37.2 (5.9) |
| fp8 projection GEMMs | 152.0 (20.0) | 82.0 (12.2) | 82.6 (13.1) |
| activation quantiser | 14.7 (1.9) | 11.5 (1.7) | 17.6 (2.8) |
| shared expert nvfp4 | 29.9 (3.9) | 21.4 (3.2) | 20.6 (3.3) |
| GDN recurrence | 111.3 (14.6) | 109.0 (16.2) | 112.6 (17.8) |
| GDN conv/gate | 40.0 (5.3) | 40.3 (6.0) | 39.2 (6.2) |
| flash attention | 23.5 (3.1) | 23.1 (3.4) | 22.7 (3.6) |
| rmsnorm | 22.5 (3.0) | 23.3 (3.5) | 26.1 (4.1) |
| other | 13.1 (1.7) | 13.7 (2.0) | 14.4 (2.3) |
| **kernel total** | **761.5** | **673.9** | **632.0** |

**16014 tokens (about the 16k bucket)**

| class | default ms (%) | lt ms (%) | lt, chunk 4096 ms (%) |
|---|---|---|---|
| routed experts | 1063.1 (36.7) | 1064.4 (41.0) | 808.7 (34.2) |
| moe topk/group/reduce/combine | 135.0 (4.7) | 131.8 (5.1) | 141.1 (6.0) |
| fp8 projection GEMMs | 601.4 (20.8) | 286.4 (11.0) | 292.2 (12.4) |
| activation quantiser | 52.7 (1.8) | 42.7 (1.6) | 66.4 (2.8) |
| shared expert nvfp4 | 65.8 (2.3) | 73.3 (2.8) | 69.2 (2.9) |
| GDN recurrence | 420.3 (14.5) | 421.9 (16.3) | 421.4 (17.8) |
| GDN conv/gate | 149.2 (5.2) | 152.1 (5.9) | 147.6 (6.2) |
| flash attention | 290.6 (10.0) | 289.6 (11.2) | 267.9 (11.3) |
| rmsnorm | 73.8 (2.5) | 88.3 (3.4) | 99.4 (4.2) |
| other | 43.3 (1.5) | 44.3 (1.7) | 48.9 (2.1) |
| **kernel total** | **2895.2** | **2594.8** | **2362.8** |

**Achieved against roof**

| class | T | default | lt | roof | lt as % of roof |
|---|---|---|---|---|---|
| fp8 projections (GEMM only) | 4189 | 70.5 TFLOPS | 130.7 | e4m3 243 | 54 % |
| fp8 projections (GEMM only) | 16014 | 68.1 | 143.1 | e4m3 243 | 59 % |
| fp8 projections + activation quant | 16014 | 62.6 | 124.5 | e4m3 243 | 51 % |
| routed experts (w4a16, bf16 mma) | 4189 | 26.5 | 26.7 | bf16 122 | 22 % |
| routed experts, lt + chunk 4096 | 4189 | | 32.6 | bf16 122 | 27 % |
| routed experts | 16014 | 30.3 | 30.3 | bf16 122 | 25 % |
| routed experts, lt + chunk 4096 | 16014 | | 39.9 | bf16 122 | 33 % |

Reading: lt takes the projections from 62-70 to 125-143 TFLOPS and removes 70 ms (4k) / 315 ms (16k) of kernel time. What
remains is not a GEMM-roof problem: 41-47 % is the grouped fp4 expert kernel at about a quarter of the bf16 roof (weight
stream and dequant bound, see `2026-10-18-cuda-moe-w4a16.md`), and 16-18 % is the GDN recurrence, which is
latency-bound (work per token is small). Chunk 4096 helps the expert kernel (every expert's weights are read once per chunk, so
fewer chunks means fewer weight streams): 316 -> 259 ms at 4k.
Wall time in these profiles includes one-time costs (the lt heuristic call shows as a 326 ms gap before the first
`nvjet` kernel in the 16k lt, chunk-4096 profile); the localeval numbers below use warmed servers.

ncu was not available (`ERR_NVGPUCTRPERM` on spark2, raw/moebench), so the roofline is nsys kernel time over counted work, not
hardware counters.

## Speed: localeval, interleaved x3, same-window vLLM

`scripts/pf/le-window.sh` (job #4575, window w1), one fresh `intarsia serve --max-seq 20480` per arm per round (rounds 1 and 3
in order, round 2 reversed), localeval `speed --sizes 1024,4096,16384 --gen 128 --reps 3`, thinking off, vLLM = the box's
`guidance` role (max-num-seqs 2, max-num-batched-tokens 8192, fp8 KV, prefix caching), idle. Prompts are 989-993, 4091 and 16656
tokens. Median of the three rounds; per-round values in brackets. spark2, 2026-10-23 13:11-13:26. Results: `raw/w1/*.result.json`.

| arm (all `main` 055775da) | prefill tok/s 1k | 4k | 16k | vs vLLM 1k | 4k | 16k | decode tok/s 1k / 4k / 16k |
|---|---|---|---|---|---|---|---|
| vLLM | 5456 (5523/5419/5456) | 6465 (6472/6465/6446) | 6306 (6306/6313/6295) | ref | ref | ref | 77.8 / 77.3 / 74.7 |
| default (chunk 2048) | 4555 (4495/4555/4594) | 5842 (5772/5849/5842) | 5422 (5409/5463/5422) | -16.5 % | -9.6 % | -14.0 % | 83.9 / 81.3 / 74.1 |
| lt (chunk 2048) | 4940 (4927/4940/4952) | 6580 (6560/6580/6583) | 6066 (6058/6086/6066) | -9.5 % | +1.8 % | -3.8 % | 84.0 / 81.7 / 74.2 |
| lt, chunk 4096 | 4933 (4949/4933/4565) | 6985 (6985/7070/6361) | 6470 (6422/6470/6503) | -9.6 % | +8.0 % | +2.6 % | 83.8 / 81.5 / 74.0 |
| lt, chunk 8192 | 4952 (4952/4998/4941) | 7020 (6979/7020/7062) | 6712 (6689/6712/6720) | -9.2 % | +8.6 % | +6.4 % | 83.8 / 81.4 / 73.8 |

Before -> after, lt at chunk 8192 against the default: 1k +8.7 %, 4k +20.2 %, 16k +23.8 %; lt alone at chunk 2048: +8.5 % /
+12.6 % / +11.9 %. Round 3 of the chunk-4096 arm is low at 1k and 4k (4565 and 6361, cause not investigated); the median is
unaffected.

The 1k bucket stays about 9 % behind vLLM on every arm. It was not profiled here (the nsys runs are 4k and 16k); a 1k prompt
is a single chunk, so per-layer fixed costs such as the expert weight streams weigh more. Not addressed.

A second window (w2, job #4607, dev build, same vLLM comparison) is in `raw/w2/` and is discussed under "Out of scope but
measured".

## Accuracy

All numbers are 21-way KL on scored rows (outside system and user turns), `intarsia accuracy` / `intarsia qualify` with the
checked-in goldens; static floors 3.0e-2 (4-bit class golden) and 2.0e-2 (CPU-oracle self golden). Build is unstamped, so
the qualify `envelope` stage ERRORs on every arm including main (no code stamp); this is not caused by the change.

**Prefill KL / top-1 against the three vLLM goldens** (chunk shown; rows from raw/a1, a2, a3 and raw/q1)

| arm | short KL | short top-1 | long KL | long top-1 | 4k KL | 4k top-1 |
|---|---|---|---|---|---|---|
| default, chunk 2048 | 4.666e-3 | 97.92 | 1.486e-2 | 94.17 | 1.224e-2 | 92.81 |
| default, chunk 4096 | 4.666e-3 | 97.92 | 1.442e-2 | 94.36 | 1.270e-2 | 92.95 |
| lt, chunk 2048 | 5.408e-3 | 98.27 | 1.417e-2 | 94.20 | 1.214e-2 | 92.82 |
| lt, chunk 4096 | 5.408e-3 | 98.27 | 1.419e-2 | 94.34 | 1.232e-2 | 92.86 |
| lt, chunk 8192 | 5.408e-3 | 98.27 | 1.419e-2 | 94.35 | 1.247e-2 | 92.88 |

KL is inside the floors and within run-to-run reference noise of the default on every golden (reference noise on `long` is
8.4e-3, on `short` 3.0e-3). Decode (teacher-forced, `acc-short`): KL 1.675e-2 default, 5.833e-3 lt; `acc-long` decode 9.58e-3
default, 9.54e-3 lt.

**`intarsia qualify` accuracy stages, 35B CUDA, spark2** (jobs #4602, #4623; `--skip bench bench-long ab tp doctor kernels`)

| stage | default | lt | rule / note |
|---|---|---|---|
| acc-short | PASS | **FAIL** (baseline gate) | floor passes; baseline-relative: mean KL 5.87e-3 -> 6.83e-3 (limit 6.75e-3), tail-bucket KL, and per-prompt |tail diff| for `science` and `chat_system`; top-1 improves 97.92 -> 98.27 |
| acc-oracle | PASS | **FAIL** | absolute floor PASS (KL 9.24e-3 against 2.0e-2); fails the "logits moved against the recorded main dump" check (mean KL 8.1e-3 against 1e-4): expected for any numerics change, the dump is main's own output |
| acc-long | PASS | PASS | KL 1.49e-2 / 1.42e-2 |
| acc-4k | FAIL | FAIL | fails on main too in this setup ("prefix reuse at chunk 1024: 3 judged", cause not investigated); `accuracy --mode reuse` on the same golden passes for the default (below) |
| envelope | ERROR | ERROR | unstamped build, both arms |

**Prefix-reuse gate** (`accuracy --mode reuse` on the long golden, 5 prompts x continue / restore / aligned, fixed 0.25-nat
stream tolerance; raw/a1, a3)

| arm | chunk 1024 | chunk 2048 | chunk 4096 |
|---|---|---|---|
| default | PASS (15/15 rows) | PASS (15/15) | PASS (9/9) |
| lt | FAIL: `chat_reuse_4k` continue (3/32 stream match; diverges at token 3) | FAIL: `chat_reuse_4k` continue (3/32) and `long_code_4k` continue (8/32) | FAIL: `chat_reuse_4k` continue (5/32) |

`restore` and `aligned` rows (bitwise by construction) pass for lt in every case. The failing rows are `continue`, where warm
and cold runs split the prompt differently; the stream diverges at a position with a small margin (warm margin 0.110 nats,
cold 0.317) and shared rows differ by 0.5-0.9 nats. This is a chaotic threshold, not a systematic shift: lt together with the
MoE in-place knob (below) passes the same rows at chunk 1024 / 4096, and chunk 8192 judges 6 rows and none fail. We did not
find why lt amplifies chunk-split differences more than the default does, although both quantise activations to e4m3 with the
same rounding. Open hypotheses, untested: accumulation-order sensitivity of the e4m3 MMA at the quantisation boundary; the
pinned algorithm chosen at M = 2048 behaving differently for the partial last tile.

## No-op (path off)

This PR carries scripts, docs and this study and no engine code (the lt kernel is already on main), so its no-op is trivially
true. The only engine-code change of the lane is the separate MoE branch; its no-op was run as pf-dev (that branch) against a
main build with every new knob unset, spark2 job #4654:

| check | result |
|---|---|
| greedy ids, 35B, 256 after a ~1.5k-token chat prompt | identical (sha a7c3e613) |
| greedy ids, 35B, 64 after a ~4.2k-token prompt | identical (sha 5efcc225) |
| greedy ids, Qwen3-8B, 256 / 64 | identical (sha e2c61289 / fcafb118) |
| `plan show` Qwen3-8B (json) / 35B | byte-identical (8758 B / 3729 B) |

No shared core or models code changed, so the Metal no-op was not run.

## Out of scope but measured: in-place accumulation in the w4a16 MoE kernel (branch `lane-pf/moe-w4a16-ip`, not in this PR)

The brief says MoE is not the lever, and this PR does not touch it. Because the profile shows the expert kernel is the largest
class after lt, one small experiment was done and pushed as its own branch for the coordinator to take or drop:
`INTARSIA_CUDA_MOE_W4A16_IP=1` lets each k16 mma accumulate into the running f32 sum instead of forming a zero-based product
sum and adding it with a separate fp32 add (kernels/cuda/moe_w4a16.cu; the default instantiations are unchanged).

* Microbench (T sweep, 256 experts, top-8, H 2048, I 512, raw/moebench): 4096 tokens 5310 -> 4949 us (+7 %), 8192 tokens 9655 ->
  8511 us (+13 %), 2048 tokens 3571 -> 3639 us (no gain). BM = 128 was slower at every T (removed again).
* localeval (spark2, dev build, w2): lt chunk 4096 4k/16k 6993 / 6459 -> 7291 / 6692 (+4.3 % / +3.6 %), lt chunk 8192
  7028 / 6724 -> 7284 / 7001 (+3.6 % / +4.1 %), vLLM in the same window 6445 / 6339. So lt8k + IP is +13.0 % / +10.4 % over vLLM.
* It is not bitwise the default kernel and it is less accurate at kernel level: rel_l2 against the cpu reference 2.9e-4 against
  8.5e-5 (3.4x, the existing w4a16 bar is 2.5x) and 0.61 % of bf16 outputs off the reference against 0.04 %. End to end KL is
  unchanged within noise (short 4.666e-3, long 1.451e-2, 4k 1.234e-2; with lt: 5.408e-3 / 1.420e-2 / 1.207e-2). With IP alone
  the reuse gate fails `long_docs_4k` continue at chunk 1024 (oracle-relative warm/cold 1.80x, limit 1.5x) and passes at 2048.
* So IP is a measured open question, not a proposal: it needs a decision on whether MoE is in scope and whether 3.4x kernel
  error is acceptable for +4 %.

`INTARSIA_PREFILL_TAIL_MERGE=1024` (lane tc) was measured in the same window w2 on top of lt: 4k +0.5 % / 16k +2.9 % at
chunk 4096, 4k -0.5 % / 16k +3.2 % at chunk 8192. With lt and IP it failed the reuse gate in a2 (`ltip8k-tm`: nothing reused in
several rows), so it is not combined with anything here.

## Decisions and open questions

1. lt stays opt-in (LOG row). To make it default-on someone must make the `continue` reuse rows pass at the recipe chunk (2048)
   or the user must accept the gate failing; the acc-short baseline gate additionally needs re-recording or a margin decision.
2. 35B recipe `prefill_chunk` 2048 -> 8192 is worth +6.7 % (4k) / +10.6 % (16k) prefill with lt on (w1 rows lt vs lt8k); it
   changes numerics on tail rows, so it is a recipe decision for the user, not made here. Its effect with lt off was not measured.
3. Is MoE in scope for this lane? (IP branch above.)
4. The `gaps.md` row `vllm-gb10-w8a8-fp8-projection-gemm` can be marked "implemented, opt-in (lt)": the GEMM is on main; what
   is missing is the default-on decision.

## Reproduce

```
fleetq run --host spark2 --load cpu --expect 25 --label "build" -- env HOST=spark2 scripts/pf/build.sh pf-base <git archive dir> <stamp>
fleetq run --host spark2 --quiet gpu,cpu --mem 35 --expect 30 -- scripts/pf/rjob.sh spark2 pf/prof.sh prof1 14000,54000 def:pf-base:2048 lt:pf-base:2048:INTARSIA_CUDA_FP8_GEMM_LT=1 lt4k:pf-base:4096:INTARSIA_CUDA_FP8_GEMM_LT=1
fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 70 -- env HOST=spark2 HN=spark-fa00 scripts/pf/le-window.sh w1 3 vllm def:pf-base:2048 lt:pf-base:2048:INTARSIA_CUDA_FP8_GEMM_LT=1 ...
python3 scripts/pf/le-table.py raw/w1/index.txt
fleetq run ... -- scripts/pf/rjob.sh spark2 pf/qualify.sh q1 def:pf-base lt:pf-base:INTARSIA_CUDA_FP8_GEMM_LT=1
fleetq run ... -- scripts/pf/rjob.sh spark1 pf/acc.sh a3 def2k:pf-base:2048 lt2k:pf-base:2048:INTARSIA_CUDA_FP8_GEMM_LT=1
```

## fleetq jobs

Builds #4555 (spark1), #4573 (spark2), #4584, #4624, #4630, #4640, #4658; profile #4574; localeval windows #4575 (w1), #4607 (w2);
MoE microbench #4599, ncu attempt #4601; qualify #4602, #4623; accuracy #4604 (a1), #4627 (a2), #4641 (a3); MoE-branch gates
and no-op #4654; MoE IP test #4656. All owner `claude-f31ccf91-pf`, all released.
