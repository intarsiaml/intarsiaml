# sp2: CUDA prefill profile of the 35B on GB10, and why the routed-expert kernel has no cheap bitwise lever left (2026-10-22)

Lane `sp2`, branch `lane-sp2/cuda-prefill`, origin/main `db66c31a`. DGX Spark (GB10, sm_121, CUDA 13.0), `Qwen3.6-35B-A3B-NVFP4`, tp=1,
default chunk 2048, default env (no cuBLAS opt-in). **No kernel change ships.** The deliverable is the profile, the answer to "is
there a gate/up vs down imbalance on CUDA like Metal's" (no), a sweep of 9 schedules of the routed-expert kernel (no schedule beats
the shipped one by a reproducible margin), the ablation that says where the time goes, and the chunk-size evidence for the vLLM gap.
Every speed number below names its slot type; all in-engine profiles are EXCLUSIVE slots on spark1 with vLLM up and idle.

## 1. Per-op profile of main (exclusive spark1 slot #4041, `raw/prof-main-4k-16k.out`)

`intarsia generate --max-tokens 1`, nsys `-t cuda`, kernel time summed per kernel family. Percentages are of the kernel total
(776.6 ms at 4189 tokens = chunks 2048 / 2048 / 93; 3390.9 ms at 16355 tokens = 7 x 2048 + 2019). The wall-clock tok/s of these
runs are not used (they include a 534 ms one-time cuBLAS / cuBLASLt lazy init).

| op | 4k ms | 4k % | 16k ms | 16k % | achieved | vs floor |
|---|---|---|---|---|---|---|
| routed experts, `moe_w4a16_kernel` gate/up (2 modes, 80 / 320 calls) | 165.6 + 102.4 = 268.0 | 34.5 | 732.1 + 464.7 = 1196.8 | 35.3 | gate/up 2070 us/call: 33 TFLOP/s, 146 GB/s weights (158 GB/s with x, h); down 1280 us: 27 TFLOP/s, 118 GB/s (184 GB/s with h, y) | 1.8x / 1.6x over the 290 GB/s byte floor |
| GDN delta rule, `gdn_delta_split_kernel` (90 / 240) | 110.8 | 14.3 | 454.5 | 13.4 | sequential in tokens: 1.23 ms per 2048-token call | latency bound, strict FADD chain (bitwise) |
| fp8 projections, `gemm_reg_pair_kernel` + `gemm_reg_kernel` | 85.4 + 58.3 | 11.0 + 7.5 | 457.2 + 280.7 | 13.5 + 8.3 | 63 TFLOP/s per chunk (S1: 5.24 TFLOP in 83.6 ms) | cuBLAS opt-in is 19 ms/chunk faster (S1) |
| routed experts, 93-token tail `moe_mma_kernel` (40 + 40) | 58.9 | 7.6 | 0 | 0 | 1473 us per layer, ~430 MB of weights: ~290 GB/s | at the byte floor |
| `moe_reduce_kernel` + `moe_combine_kernel` | 28.1 + 5.8 | 3.6 + 0.7 | 141.3 + 27.0 | 4.2 + 0.8 | 234 us/call, 75 MB: ~320 GB/s | at the floor |
| attention `flash_kernel` (10 layers) | 24.6 | 3.2 | 332.2 | 9.8 | 16k: 21.9 TFLOP in 332 ms = 66 TFLOP/s | grows with context |
| `rmsnorm` / `gated_rmsnorm` / `gdn_conv` | 22.3 / 13.4 / 21.1 | 2.9 / 1.7 / 2.7 | 91.7 / 58.9 / 85.9 | 2.7 / 1.7 / 2.5 | 74 us, 234 us per call | at the floor |
| `wq_gemm`, `nvfp4_reg`, `fp8_actq`, rest | 17.2 + 16.4 + 14.7 + ~20 | ~9 | | ~8 | small | |
| idle: 2877 gaps under 20 us | 6.3 | 0.8 | 16.2 | 0.5 | | not a lever |

Reading: routed experts are 42 % of 4k prefill (incl. the tail) and 35 % of 16k. Everything else that is large is either
bound by a strict summation order that bitwise forbids changing (GDN), already at the byte floor (norms, conv, reduce, tail), or
S1's (fp8 GEMMs).

### Is there a gate/up vs down imbalance like Metal's (M35, 10x per weight byte)? No.

Per 2048-token call, per weight byte (gate/up 302 MB with scales, down 151 MB): gate/up 146 GB/s, down 118 GB/s. Per FLOP: 33 vs
27 TFLOP/s. The ratio of times is 1.62 for a 2.0 ratio of bytes and FLOPs, so gate/up is 1.24x more efficient per byte than down,
the opposite sign and two orders of magnitude smaller than on Metal. Down is relatively weaker only because K = 512 gives each tile
8 k64 steps (pipeline fill and a scattered 67 MB y store are poorly amortised). Both kernels sit at 55 to 60 % of the 290 GB/s
that the tail kernel shows the LPDDR5x can deliver.

## 2. Where the time of the routed-expert kernel goes (T = 2048 microbench, non-exclusive: relative numbers only)

`moe_w4a16.cu` was given temporary ablation switches (`raw/moe_w4a16-sweep-knobs.patch`, never merged). Call time of the full
op (group + gate/up + down + reduce), T = 2048, real layer shape, uniform routing:

| variant | call us | what it removes |
|---|---|---|
| baseline | 3600 | |
| no dequant (raw words as B) | 3607 | nothing visible: hidden behind DRAM |
| no mma (fadd of the B words) | 3510 | nothing visible |
| no weight cp.async after the prologue | 2806 | -22 %: weight DRAM |
| all blocks read expert 0 (L2-resident weights) | 2695 | -25 % |
| no A loads / no scale loads | 3223 / 3451 | -10 % / -4 % |
| L2 weights + no dequant / + no mma / + neither | 2470 / 2390 / 2230 | compute adds 0.2 to 0.3 ms each once DRAM is out |
| no loads, no dequant, no mma | 2050 | the skeleton (barriers, ldmatrix, 4 fadd per mma, epilogue, stores) |
| no main loop at all | 1130 | prologue, epilogue, y/h stores (0.45 ms, 84 MB), reduce (0.25 ms) |

The kernel is a serial composition of weight DRAM time, A-tile L2 time, skeleton and stores, not bound by any one unit, so no single
knob moves it. Removing the dequant or the mma alone changes nothing; the 4 fadd per mma that bitwise requires (every k16 sum is formed
from a zero accumulator and added with an fp32 add, see `moe_grouped.cu`; accumulating in the mma was tried by the original author and is
less accurate) cost issue slots that no schedule gets back. `add.rn.f32x2` is accepted by ptxas for sm_121 but lowered to two FADD
(`raw/` note in section 4), so there is no packed-add shortcut on GB10.

## 3. Schedules tried (bitwise by construction; `moe_w4a16` tests 4/4 pass for each at the test shapes; R = 256 fails the small test shapes)

In-engine, 4k prompt, nsys, per-call average of the 80 calls, exclusive slots #4056 / #4079 (`raw/sweep1-*.out`, `raw/sweep2-*.out`).
Per-mode configs (gate/up and down independently; the shipped launcher uses one joint config):

| config (BM 64) | gate/up us | down us |
|---|---|---|
| R128 JW4 MINB2 LA2 LR2 (shipped), 5 runs | 2102 / 2050 / 2057 / 2151 / 2024 | 1321 / 1254 / 1283 / 1320 / 1235 |
| R128 JW2 MINB2 (8 warps/block), 4 runs | 1982 / 2081 / 1996 / 1969 | 1198 / 1242 / 1181 / 1197 |
| R128 JW2 MINB3 LA1, 4 runs | 1987 / 2011 / 1978 / 2012 | 1247 / 1250 / 1227 / 1227 |
| R128 JW4 MINB3 LA1 | 2042 | 1286 |
| R256 JW4 MINB1 | 2152 | 1296 |
| R256 JW2 MINB2 | 2399 | 1279 |
| R64 JW4 MINB2 / MINB4 LA1 | 2871 / 2344 | 1823 / 1427 |
| R128 JW4 LA3 LR4 (deeper ring) | 2749 | 1854 |
| BM 128 (T = 2048, microbench) JW2 / JW4 | 4037 to 4316 vs 3587 | |
| BM 32 (chunks of 768 to 1792 tokens, microbench) 5 variants | all within the 3 % noise | |

Run-to-run spread of the shipped config is 6 % (2024 to 2151), larger than the best candidate's mean gain (JW2: about -3 % gate/up,
-5 % down, about 1.3 % of 4k prefill). In the same T = 2048 microbench, three interleaved rounds gave JW2 +1.5 % slower at 2048 and
-4 % at 4096 (the box was shared with a CPU job). Not reproducible, so not shipped. The register file (162 to 166 regs, no spill) would
allow three 128-thread blocks per SM and that was tried (MINB3): no change, which agrees with the ablation (not latency bound).

## 4. Dead ends worth recording

* ncu is unavailable (`ERR_NVGPUCTRPERM`, needs a driver setting that is not ours to change); the ablations above stand in for it.
* `add.rn.f32x2` compiles for sm_121 and becomes two FADD (cuobjdump): no FADD2 on this part.
* GDN delta is locked to the strict left-to-right chain by the bitwise contract (kernel header); the remaining slack is latency.
* Fusing `moe_reduce` into the down GEMM (last arriver per token and column block, deterministic order) could hide part of the 0.25 ms
  but needs a counter array in the scratch, i.e. a change of `moe_experts_mma_scratch` in shared core, and gains at most 3.6 % of 4k
  prefill. Not attempted in this lane.

## 5. The gap to vLLM is weight re-streaming, not a slow kernel

S1's same-box numbers (spark1 vLLM, round 7): 6353 tok/s at 4k, 6275 at 16k; intarsia main 5739 / 5432 (-10 % / -11.5 %). The
profile shows where that goes. At chunk 2048 every full pass over the model re-reads 21 GB of weights (about 75 ms at 290 GB/s): 3 passes
at 4k (2048 / 2048 / 93; the 93-token tail alone costs 59 ms in the experts), 8 at 16k. A larger chunk reads them fewer times.

Same exclusive slot (#4105, spark1, vLLM up, main binary, default env, 2 runs per cell, `raw/chunk-ids-speed.out`):

| `--prefill-chunk` | 4k prefill ms (2 runs) | tok/s | vs 2048 | 16k prefill ms (2 runs) | tok/s | vs 2048 |
|---|---|---|---|---|---|---|
| 2048 (default) | 1018 / 988 | 4177 | | 3175 / 3187 | 5142 | |
| 4096 | 936 / 949 | 4445 | +6.4 % | 2988 / 2951 | 5508 | +7.1 % |
| 8192 | 934 / 864 | 4660 | +11.6 % | 3046 / 2859 | 5539 | +7.7 % |

(`generate` wall figures: they include first-run warmup the A/B harness excludes, so only the ratios are meaningful.)

**Greedy ids are byte-identical across chunk 2048 / 4096 / 8192**: 256 ids on the 1485-token prompt and the first id on the 4189- and
16355-token prompts have the same md5 in all six runs. That is evidence (not proof: three prompts) that the chunk size is
numerics-neutral on CUDA for this model, so a larger default would be a memory and performance decision, not a numerics one. It is the
largest remaining lever for the vLLM gap and is not in this lane's scope (lanes fx3 and tc work on chunk handling and the tail); it is
recorded here because the kernel work above shows that no routed-expert schedule gets near it.

## 6. What was and was not run

No code ships, so the gates that apply to a kernel change (kernels test, 256 greedy ids, qualify drift, 8B no-op, main-vs-head A/B) were
not run: there is no head to compare. The schedule candidates passed `moe_w4a16` (bitwise vs `moe_experts_mma_bf16`) except where noted.
Reproduce: `scripts/sp2/` (build.sh, prof-job.sh, sweep-job.sh, mbench.sh, bitwise-job.sh, chunk-job.sh; fleetq slots as in each header).

## 7. Fleet log

| job | host | slot | what |
|---|---|---|---|
| #4033, #4034 | spark1 | non-excl, 2 GB | read-only ls of trees / goldens |
| #4035, #4038, #4053, #4074, #4076, #4078, #4086, #4101 | spark1 | non-excl, 8 to 16 GB | builds (base tree, dev tree with sweep knobs) |
| #4041 | spark1 | exclusive, vLLM up | per-op profile 4k + 16k |
| #4047 | spark1 | exclusive | ncu attempt (ERR_NVGPUCTRPERM) |
| #4056, #4079 | spark1 | exclusive | schedule sweep + ablations, 4k nsys |
| #4055 | spark1 | non-excl | bitwise tests per schedule |
| #4073, #4075 and the later microbench jobs (ablations, BM 128 / BM 32, JW2 x3; ids not individually recorded) | spark1 | non-excl, relative only | T = 2048 / 4096 microbench |
| #4105 | spark1 | exclusive | prefill-chunk ids and speed |

No vLLM downtime was used; no role was taken down.
