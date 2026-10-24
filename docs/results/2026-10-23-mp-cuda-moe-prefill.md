# Lane mp: CUDA routed-expert MoE prefill on GB10, L2-prefetch schedule (2026-10-23)

Target: DGX Spark GB10 (sm_121, CUDA 13.0), Qwen3.6-35B-A3B-NVFP4, tp=1. Spark2 (microbench, nsys, localeval, accuracy), spark1 (unit tests, batching gates).
Branch `lane-mp/moe-prefill` off main `6189466f` (includes #125, lt default-on). Raw logs: `2026-10-23-mp-cuda-moe-prefill/raw/`; scripts: `scripts/mp/`;
trace data: `trace.json`, `techniques.jsonl` in `2026-10-23-mp-cuda-moe-prefill/`. Predecessors: [pf](2026-10-23-pf-cuda-fp8-prefill.md), [lt](2026-10-23-lt-fp8-lt-reuse.md).

## Headline

* **Opt-in, bitwise.** `INTARSIA_CUDA_MOE_W4A16_PF=1` adds an L2-prefetch schedule to the gate/up half of the w4a16 routed-expert kernel
  (`kernels/cuda/moe_w4a16.cu`) and a 64-row tile, for prefill chunks of up to 2048 tokens (64 pairs per expert at 256 experts, top-8).
  The arithmetic and accumulation order are untouched: the output bytes equal the default kernel's (and the old grouped kernel's) on every
  test shape, so the kernel error bar is unchanged and nothing was loosened. Default off; the default path is the kernel of main.
* **Kernel (real layer shape, 256 experts, top-8, H 2048, I 512, interleaved A/B):** 1k tokens -7.1 % (uniform routing) / -10.5 % (skewed),
  2k tokens -3.8 % / -5.3 %. Above 2048 tokens the knob does nothing (4k and 8k unchanged by construction).
* **Prefill, localeval, same-window vLLM, interleaved x3 (chunk 2048):** 1k +7 % over main (-0.6 % / +1.2 % against vLLM in the same window, was -7.4 % / -5.2 %), 4k +1 % (+2.5 % / +4.5 % against vLLM), 16k +0.6 % / +4.4 % (-3.9 % / -3.2 % against vLLM; noisy, expected about +1.5 %). Two windows (w1, w2). Decode unchanged.
* **Why not more:** the kernel is not compute-bound where it matters. At 1k tokens it runs at the speed of its own weight stream
  (the DRAM read of 436 MB of expert weights per layer at about 200 GB/s is 2.2 ms of the 2.9 ms), and at 4k the fixed traffic of the
  `y` intermediate (written by the down GEMM, read by the combine kernel) and the skeleton cost more than the tensor-core work. About ten
  further schedule changes were tried and none moved the kernel by more than noise (section Tried and dropped).

## Profile (nsys, microbench of the production launcher, real layer shape, uniform routing)

| tokens | knob | gate/up GEMM | down GEMM | combine | grouping | total (us) |
|---|---|---|---|---|---|---|
| 1024 | off | 1784 | 969 | 122 | 10 | 2886 |
| 2048 | off | 2066 | 1263 | 334 | 20 | 3683 |
| 4096 | off | 2786 | 1900 | 708 | 37 | 5430 |
| 1024 | PF=1 | 1561 | 966 | 134 | 10 | 2672 |
| 2048 | PF=1 | 1878 | 1278 | 330 | 20 | 3506 |
| 4096 | PF=1 | 2837 | 1930 | 700 | 37 | 5504 |

Mean per launch over 9 launches (the checked run, 2 untimed and 6 timed reps). Tile height: 32 rows at 1k with the knob off, 64 elsewhere; PF=1 uses 64 rows at 1k and 2k, and at 4k the knob is over its cap, so those rows are the knob-off kernel again (the differences are run-to-run).

Per layer the four kernels are the whole MoE. At 4k the combine kernel (`moe_reduce_kernel`) is 13 % and is DRAM-bound (it reads the
134 MB `y` once: 190 GB/s). Kernel time of the production launcher alone (`raw/microbench-skew0.txt`, column `P`): 2.91 ms at 1k (17.8 TFLOPS),
3.52 ms at 2k, 5.42 ms at 4k, 9.84 ms at 8k (about 44 TFLOPS), i.e. 17 to 33 % of the bf16 roof, matching pf's 22 to 33 %.

Ablation of the production kernel (compile-time switches in a scratch copy, spark1, microseconds for 1k / 2k / 4k tokens; the scratch copy was
the production source, the numbers are for finding the bound, not for shipping):

| variant | 1k | 2k | 4k |
|---|---|---|---|
| production | 2890 | 3490 | 5260 |
| weight stream and epilogue only (no dequant, no mma, no activation refill) | 2693 | 3191 | 4215 |
| same, weights of only 4 experts (L2-resident) | 897 | 1320 | 2410 |
| full compute, weights of only 4 experts | 1704 | 2694 | 4856 |
| nothing but pipeline skeleton, epilogue stores, grouping and combine | 608 | 1136 | 2228 |
| grouping and combine kernels alone | 200 | 379 | 743 |

Reading: at 1k the DRAM weight stream is the kernel (2.7 of 2.9 ms; making the weights L2-resident takes it to 0.9 ms), and the stream
runs at 200 to 212 GB/s with the kernel's 64-byte-per-row cp.async units (standalone probe, same in every row-skew layout), below the
273 GB/s peak. At 4k, tensor-core and dequant work is only about 1 ms of 5.3 ms; the rest is the fixed `y`/`h` traffic and the combine.

## Approach

`cp.async.bulk.prefetch.L2` (sm_90+) asks L2 to pull contiguous spans without occupying shared memory or registers:

1. at block start: the block's block-scales for the gate rows and the up rows, one contiguous span each (two requests per span);
2. while streaming the packed weights: two 64-byte units per row ahead of the first `cp.async` that consumes them, two groups ahead.
   One thread per row issues the request.
3. tile height 64 rows when the knob is on (a taller tile halves weight re-reads from the tile loop); `INTARSIA_MOE_BM` still wins.

Gate/up only (the down half is not helped: bisects below). Template flag `PF` on `moe_w4a16_kernel`; with the knob off the instantiations are
the old ones. The prefetch moves no data the kernel does not read and does not change any arithmetic.

## Kernel microbench (final code, `scripts/mp/dev.sh`, 20 interleaved reps, median us; P = knob off, PF = knob on)

| tokens | uniform routing: P / PF / change | skewed routing (zipf 0.7): P / PF / change |
|---|---|---|
| 512 | 2762 / 2464 / -10.8 % | 2719 / 2426 / -10.7 % |
| 1024 | 2912 / 2704 / -7.1 % | 3122 / 2795 / -10.5 % |
| 1536 | 3167 / 2987 / -5.7 % | 3560 / 3220 / -9.5 % |
| 2048 | 3525 / 3392 / -3.8 % | 3859 / 3653 / -5.3 % |
| 2560 | 3908 / 4030 / +3.1 % | 4274 / 4276 / +0.0 % |
| 3072 | 4324 / 4325 / +0.0 % | 4707 / 4709 / +0.0 % |
| 4096 | 5417 / 5405 / -0.2 % | 5606 / 5617 / +0.2 % |
| 8192 | 9839 / 10029 / +1.9 % | 9884 / 9882 / -0.0 % |

Bit-identical to the default kernel in every row (the bench compares the two outputs byte for byte; no row is marked as differing). Rows from 3072 up run the same kernel on both sides, so their spread (up to 2 %) is run-to-run noise. The 2560 row of the uniform column was run before the cap was lowered from 80 to 64 pairs per expert.

2560 tokens is the reason for the cap: with a cap of 80 pairs per expert the knob made 2560 tokens 3 % slower (uniform routing: 3908 to 4030 us;
kept in `raw/microbench-skew0.txt`), so it now stops at 64.

## Tried and dropped (all bitwise-identical rewrites; kernel microseconds at 1k / 2k / 4k tokens, production about 2900 / 3500 / 5300)

* Loop specialised on the number of active m16 tiles, three forms (`MASPEC`): equal at 1k; 2k and 4k up to 16 % worse (ptxas spills 16 to 224 bytes in the largest instantiations; the production kernel has 162 registers and no spill).
* Packed `add.rn.f32x2` accumulate (same bits as two fadd; verified on device): no change.
* Weight words read per k16 from shared memory (8-byte loads, unroll 1, 2, 4): within 3 %.
* Vectorised combine kernel (8 columns per thread): 0 to 1 %.
* Activation `cp.async` with L2 evict-last: no change. Weight `cp.async` with an evict-first cache hint: illegal instruction (not resolved; the hint form was never made to run).
* Prefetching the whole tile's weights at block start (4 KB and 8 KB spans): helps the weights-only variant (-12 % at 1k) but not the full kernel (+2.5 % at 1k, +9 % at 2k).
* Prefetch group and distance sweep (1, 2, 4, 8 units per request; 1 to 6 groups ahead): all within 3 % of each other; 2 and 2 chosen.
* Prefetch for both GEMMs instead of gate/up only: 2 to 4 % worse (2k 3463 against 3370 us, 4k 5784 against 5565 us).
* 256-row blocks: up to +10 %; 64-row blocks and lookahead 4: +34 % at 2k and +55 % at 4k; 8 n8-tiles per warp: no change at 1k, worse above.
* Streaming store (`st.cs`) for the down output: no change.
* The in-place accumulation of `lane-pf/moe-w4a16-ip` (+4 %, 3.4x kernel error against the 2.5x bar) was not revisited; this lane keeps the accumulation order.

## Accuracy

The knob changes no arithmetic, so every gate is expected to read exactly as main's; they were run with `INTARSIA_CUDA_MOE_W4A16_PF=1` on the final code
(`head=bd4eeab code-tree=19ccb898bb10`, stamped build `mp-dev`, tree identical to the PR head except docs). Logs: `raw/gates/`.

| gate | result | job |
|---|---|---|
| MoE kernel error test (`moe_w4a16`: bitwise vs the old grouped kernel at tile heights auto/32/64 with the knob off and on, 4 tests; `moe_experts_w4a16_is_as_accurate_as_old_kernel` also asserts knob-on output equals knob-off) | 4 passed, 0 failed; the existing 2.5x bar untouched | #4910 |
| `quantmoe` 13 passed / 0 failed (5 ignored benches), `moe_nvfp4` 5 passed / 0 failed, `docs_env_vars` 2 passed (the new knob has a `docs/tuning.md` row) | pass | #4910 |
| `kernels test --backend cuda --quick`, knob unset and knob on | 77 kernels, 531 cases, 0 FAIL lines (exit 0) both | #4910 |
| acc-short against main's committed CUDA baseline (`golden/baselines/qwen36-35b-a3b-nvfp4-vllm-top20.cuda.baseline.json`, recorded from the lt build `c37a8d3`; `--pre-change c37a8d3`) | PASS: prefill mean KL 6.82514e-3 (baseline 6.82514e-3, limit 7.84891e-3), decode 6.98641e-3 (6.98641e-3, 8.03437e-3), both per-prompt gates ok, floor verdict PASS, baseline verdict PASS | #4907 |
| 3 KL goldens at prefill chunk 2048 (short, long, 4k) | golden gate PASS x3; 21-way KL (scored rows) 5.408e-3 / 1.417e-2 / 1.214e-2 (floor 3.0e-2) | #4907 |
| control: the same three goldens on main's build, knob absent | the printed `prefill floor` lines and the per-prompt tables are identical to the knob-on run | #4872 |
| prefix-reuse gate with the CPU continue oracles (`golden/oracle/`, #123) at chunk 2048 | GATEB_REUSE | #4941 |
| `batch_cuda_real` (35B and 8B, batched decode bitwise the solo step, engine equals local engine) | GATEB_BATCH | #4941 |
| `batch_cuda_real --ignored real_35b_batched_engine_long_prompts_equal_the_local_engine` | GATEB_LONG | #4941 |

An earlier pass of the same gates on the first code (cap 80 pairs per expert, unstamped build) passed too (accuracy #4849 with the acc-short step refused for
the missing build stamp, KL x3 PASS; stamped acc-short #4864; reuse + batching + long-prompt #4848, all 0 failed). Only the cap changed after
that, and it matters only for chunks of 2049 to 2560 tokens, which chunk 2048 never produces; the final pass above repeats everything on the final code.

## Speed

Method: `scripts/mp/le-window.sh` (pf's harness): `localeval speed` sizes 1024/4096/16384, gen 128, 3 reps + 1 warmup, thinking off; one fresh
`intarsia serve` per arm per round, vLLM (`spark2-vllm`, the box's running `guidance` launcher, idle) in the same window, 3 rounds, round 2 in
reverse arm order. Slot: `--quiet gpu,cpu --mem 50` on spark2, no role taken down (vLLM stays up and idle, as in pf's w1). `def` = main
`6189466f` (lt default-on, chunk 2048); `pf` = this branch with `INTARSIA_CUDA_MOE_W4A16_PF=1`, chunk 2048. Prefill tok/s, median of the 3
reps of each round, then median over rounds (rounds in brackets in the table). Window w1 ran with the first cap (80 pairs
per expert), w2 with the final code; both use chunk 2048, where the cap never binds (a chunk has at most 2048 tokens = 64 pairs per expert),
so w1 and w2 exercise the same kernels.

| window | size | vLLM | main (def) | PF=1 | def vs vLLM | PF=1 vs vLLM | PF=1 vs def |
|---|---|---|---|---|---|---|---|
| w1 | 1k | 5371 (5464/5371/5219) | 4976 (4887/4999/4976) | 5341 (5287/5341/5384) | -7.4 % | -0.6 % | +7.3 % |
| w1 | 4k | 6461 (6490/6461/6428) | 6549 (6505/6554/6549) | 6624 (6573/6654/6624) | +1.4 % | +2.5 % | +1.1 % |
| w1 | 16k | 6312 (6353/6312/6312) | 6031 (6019/6041/6031) | 6067 (6035/6067/6082) | -4.5 % | -3.9 % | +0.6 % |
| w2 | 1k | 5299 (5427/5253/5299) | 5022 (5075/4913/5022) | 5364 (5327/5379/5364) | -5.2 % | +1.2 % | +6.8 % |
| w2 | 4k | 6391 (6454/6358/6391) | 6615 (6660/6278/6615) | 6678 (6678/6592/6679) | +3.5 % | +4.5 % | +1.0 % |
| w2 | 16k | 6319 (6350/5687/6319) | 5859 (6110/5859/5684) | 6114 (6105/6114/6119) | -7.3 % | -3.2 % | +4.4 % |

Prefill tok/s, median over 3 rounds (rounds 1/2/3 in brackets). `raw/le-table.py raw/w1` regenerates the numbers; localeval run ids are in `raw/*/index.txt` and each `*.out`.

Decode tok/s is unchanged (def 82.8 / 80.4 / 73.1, pf 83.0 / 80.4 / 72.9 in w1; def 83.7 / 81.1 / 73.9, pf 83.5 / 81.0 / 73.6 in w2).
The 16k row of w2 is the noisy one (def rounds 6110 / 5859 / 5684, vLLM rounds 6350 / 5687 / 6319): read the 16k gain as +0.6 to +4 %, not as a
number. The expected gain at 16k is about 1.5 %: the MoE kernel is 40 to 47 % of prefill kernel time and the knob shortens it by 3 to 4 % at
2048-token chunks.

## Decisions and open questions

1. **Default.** Left off as asked. The knob is bitwise and has no accuracy cost, so the only argument against default-on is the size of the win: 7 % of
   1k-prompt prefill, about 1 % at 4k and 16k. Recommendation: default on for chunk <= 2048 (the knob already returns to the old kernel above that).
2. The PF-off instantiations of the SiLU kernel (`GELU=false`, the 35B path) differ from main's in the order of about 27 of 900 to 2400 SASS
   instructions in the prologue (scheduling only; the GELU variants are byte-identical SASS, `raw/gates/sass.log`). Results are bitwise equal
   (unit tests, and KL on the three goldens printed identically to main's build), but "identical to main" is shown by outputs, not by machine code.
3. Remaining MoE lever at 4k and above is the `y` round trip (134 MB written and read per layer at 4k) and the combine kernel (13 %); removing it needs a
   different reduction structure (sorted by token) that changes accumulation order and so the kernel error bar. Not attempted.
4. pf's open item on the 35B `prefill_chunk` recipe (2048 vs 8192) is untouched; PF only helps chunks <= 2048.

## Reproduce

```
fleetq run --host spark2 --quiet gpu,cpu --expect 12 -- scripts/mp/dev.sh spark2 "1024,2048,4096" 20 0 base       # microbench, prod vs PF, bitwise check
fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 75 -- env HOST=spark2 HN=spark-fa00 scripts/mp/le-window.sh w2 3 vllm def:mp-base:2048 pf:mp-dev:2048:INTARSIA_CUDA_MOE_W4A16_PF=1
scripts/pf/rjob.sh spark2 mp/gatesA.sh <out> mp-dev <baseline.json> 1     # acc-short + 3 KL
scripts/pf/rjob.sh spark1 mp/gatesB.sh <out> mp-dev 1                      # reuse gate, batch_cuda_real, long-prompt test
scripts/pf/rjob.sh spark1 mp/gputests.sh mp-dev                            # moe_w4a16, quantmoe, moe_nvfp4, docs_env_vars, kernels test --quick
```

## fleetq jobs

JOBS
