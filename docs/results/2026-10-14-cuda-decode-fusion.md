# CUDA decode fusion, Qwen3.6-35B-A3B NVFP4 on spark1 (GB10, sm_121), branch `perf/cuda-decode-fusion`

Base `6418ef1` (arch/plan). Built and measured on spark1 (CUDA 13.0, nsys 2025.3), trees `~/intarsia-decfuse/{base,dev}`
(base = the arch/plan tip, dev = this branch; the one commit between them in git history is a build script, no code), scripts
`scripts/cuda-decode/`, raw logs and JSON in `docs/results/2026-10-14-cuda-decode-fusion/`. Continues
`2026-10-14-hip-decode-2.md` (the same four fused ops on HIP) and `2026-10-13-prefill-cuda.md` (A/B method).

## 1. Result

| 35B decode, graphs on, `bench speed` gen 128, 5 reps, mean of 3 interleaved rounds | before | after |
|---|---|---|
| 128 prompt tokens | 72.3 tok/s | **80.3 tok/s** (+11.1%) |
| 1024 | 71.8 | **79.7** (+11.1%) |
| 4096 | 69.5 | **77.0** (+10.8%) |
| launches per decode step (nsys, graph nodes) | 944 | **724** (-220) |
| decode step span (nsys, 156 steady steps) | 13.95 ms | **12.42 ms** (-11.0%) |
| idle gaps inside the step | 98.6 us (0.7%) | 75.8 us (0.6%) |

vLLM on the same box (figure from the task brief, 76-78 tok/s at 128 tokens, not re-measured here): 128 and 1024 are above it
(80.3 / 79.7); at 4096 we are at 77.0 (vLLM at 4096 was not given). `qualify --quick-only` prints 80.7 / 80.0 / 77.5
and calls 80.7 "82% of the byte floor".

| | before | after |
|---|---|---|
| 35B prefill 128 / 1024 / 4096 (tok/s) | 1262 / 3873 / 4660 | 1292 / 3907 / 4606 |
| 8B decode 128 / 1024 / 4096 (one round) | 15.01 / 14.86 / 14.42 | 15.10 / 14.95 / 14.52 |
| 8B prefill 128 / 1024 / 4096 (one round) | 1467 / 2791 / 2672 | 1429 / 2756 / 2748 |

Prefill is untouched by design (the pair and glu ops are M = 1; only `gated_rmsnorm` also runs on prefill rows). The 4096
prefill mean is 1.2% lower, which is one outlier: dev round 1 measured 4473, rounds 2 and 3 4655 and 4691 (base 4659, 4668,
4653). The 8B is dense bf16: no fused op applies to it, so its row is a confirmation that nothing moved, from one round, not
three (downtime budget). 8B decode is 15 tok/s on this box because 16 GB of bf16 weights stream at about 270 GB/s.

Per-round numbers (decode tok/s, base then dev): 128: 72.20 / 72.85 / 71.93 and 80.75 / 80.06 / 80.22; 1024: 71.62 / 72.25 /
71.43 and 80.19 / 79.44 / 79.56; 4096: 69.38 / 70.05 / 69.20 and 77.56 / 76.76 / 76.77. Rounds were base-dev, dev-base,
base-dev. Rep-to-rep stdev inside a run is 0.1 tok/s; the gap between the arms is 80 times that.

## 2. What was built

1. **The four fused decode ops as CUDA kernels**, registered with admission in `crates/intarsia-backend-cuda/src/quantmoe.rs`
   (priority 10, `Exactness::Deterministic`), same ops, names and ABI as on HIP (`kernels/abi/ik_abi.h`). The model needed no
   change: it asks the registry and falls back to the unfused chain when a call has no kernel; `INTARSIA_NO_FUSION=1` still turns
   them off. The CUDA table test no longer lists them as optional.

   | op (kernel) | replaces | launches/step |
   |---|---|---|
   | `matmul_pair_bf16_gemv` (matmul.cu `gemv_pair_kernel`) | 2 x bf16 GEMV of one x: GDN b+a (30), router + shared gate (40) | -70 |
   | `matmul_fp8_pair_gemv` (fp8.cu `fp8_gemv_pair_kernel`) | 2 x fp8 GEMV: GDN qkv+z (30), attention k+v (10) | -40 |
   | `matmul_nvfp4_glu_gemv` (nvfp4.cu `nvfp4_glu_kernel`) | shared expert gate + up GEMV + silu_glu | -80 |
   | `gated_rmsnorm_bf16` (norm.cu `gated_rmsnorm_kernel`) | rmsnorm + silu_glu (GDN output norm) | -30 |

   Bitwise by construction: the pair kernels run the unfused kernel's own body (`gemv_body`, `fp8_gemv_body`, now `__device__`
   functions shared with it) with the block index offset into the second matrix, each segment staging its own scales; the glu
   kernel keeps one warp per row for gate and for up (same lane and iteration order, same fma chain), rounds both to bf16 as
   the GEMV store would, then applies `silu_glu`'s expression (`ik_silu`, now in `ik_launch.h`, used by elementwise.cu too); the
   gated norm is `rmsnorm_kernel`'s reduction and bf16 rounding followed by `silu_glu`.
2. **`moe_experts_decode_v2_bf16`** (`kernels/cuda/moe_decode.cu`, priority 20, the old decode kernel stays registered as the
   fallback and the bitwise oracle). This is the step-4 item, chosen from the baseline profile (section 3): the old
   `moe_dec_down_kernel` (14.3% of kernel time, 50 us per layer) keeps all 8 experts of a row in registers (160 registers per
   thread, 8 KB dynamic smem for the staged h: one 256-thread block per SM, 256 blocks = 5.3 waves on 48 SMs), 174 GB/s on cold weights. v2 gives a warp
   an expert slot instead of an output row (`R` = 4 rows per block, R independent weight loads per lane, no staging) and sums
   the 8 per-slot terms in slot order 0..K-1 from shared memory, the order the old kernel uses. Microbenchmark
   (`quantmoe.rs bench_moe_decode_cold`, 96 routings, cold weights, CUDA graph): 81.4 us -> 60.5 us per call, 174 -> 234 GB/s;
   R = 1/2/4/8 gave 60.5 / 63.1 / 61.0 / 64.0 and gate_up unroll 1/2/4 made no difference (60.4 / 61.0 / 60.4), so R = 4 and
   unroll 2 stay (`INTARSIA_MOE_DOWN_R`, `INTARSIA_MOE_GU_UNROLL` for A/B). In the model: 50.0 -> 26.3 us per layer.
3. Tests, scripts and the window tooling (section 6).

## 3. Profile, before and after (nsys, graph nodes traced, 160 greedy tokens)

Method: `nsys profile -t cuda --cuda-graph-trace=node`, steps cut at the embedding kernel, first 5 and the prefill dropped,
`scripts/cuda-decode/nsys_steps.py`. Files: `profile/before-steps.txt`, `raw/prof-dev/steps.txt`.

| kernel group | before: n/step, us/step | after: n/step, us/step | change |
|---|---|---|---|
| fp8 GEMV (single + pair) | 130, 5995.6 | 50 + 40 pair, 2376.9 + 3416.9 = 5793.8 | -201.8 |
| MoE down (old / v2) | 40, 2000.7 | 40, 1050.3 | **-950.4** |
| MoE gate_up + silu | 40, 1818.8 | 40, 1808.6 | -10.2 |
| nvfp4 GEMV incl. lm_head (1314 us) | 121, 1872.2 | 41, 1479.7 + 40 glu, 321.9 = 1801.6 | -70.6 |
| bf16 GEMV (single / pair) | 140, 590.7 | 70 pair, 383.8 | -206.9 |
| GDN delta rule | 30, 687.6 | 30, 672.5 | -15.1 |
| rmsnorm + gated rmsnorm + silu_glu | 131 + 70, 256.5 + 84.1 | 101 + 30 gated, 200.9 + 57.2 | -82.5 |
| everything else (13 small kernels) | 648 | 637 | -11 |
| **total kernel time (sum of the rows) / launches** | **13954 us / 944** | **12406 us / 724** | **-1548 us / -220** |

What the data says. (a) The graph had almost no gaps to begin with: 98.6 us idle in 13.95 ms, so removing 220 launches
removed only 23 us of idle (the rows use per-step means; the median kernel sums are 13852 and 12345 us). Unlike HIP (3 us of gap per launch, section 3 of the HIP doc), the CUDA gain from the four fusions
is kernel time: the pair kernels read x once and fill the GPU better (-480 us across the pair/glu/norm rows). (b) The MoE
down restructure is the larger part, 950 of 1548 us (61%). So the +11% is about 60% the v2 kernel and 40% the
four fusions; I have no isolated measurement of the split on the end-to-end rate (no `INTARSIA_NO_FUSION` arm was run, no
budget), only the per-kernel sums above. (c) The step is now 12.4 ms: fp8 GEMVs 47% of kernel time, MoE gate_up 14.6%, lm_head
10.6%, MoE down v2 8.5%, GDN 5.4%.

## 4. Correctness

**Fused vs unfused, bitwise** (`crates/intarsia-backend-cuda/tests/fused_decode.rs`, 4 tests, GB10, passing): each fused
kernel's output bits equal the chain it replaces, run on the same device through the unfused kernels, over the model's shapes
and edges (segments shorter than a block, odd splits, K = 8/16, K past the unrolled window, W8A8 + W8A16 segments in one call,
different weight and input scales per segment, different global scales for gate and up, scalar path of the norm d % 8 != 0,
w_offset 0 and 1); each case also checks the cpu reference. `tests/quantmoe.rs moe_experts_decode_matches_old_kernel` now
runs both decode kernels against `moe_experts_bf16` bitwise on 9 shapes including K below and above the 8 warps, duplicate and
invalid ids.

**Admission** (`every_cuda_kernel_matches_the_reference`, `admission-fused.log`): matmul_pair 6 cases max_abs 0, fp8_pair 15
cases rel_l2 2.7e-5, nvfp4_glu 6 cases 0, gated_rmsnorm 16 cases 3.6e-7: ADMITTED. (That run predates the v2 kernel; the
`qualify` kernels stage, 41 kernels / 569 cases on the model's shapes, passed on the final build.)

**Mutation checks** (`scripts/cuda-decode/mutate.{py,sh}`, on a copy of the tree on spark1; logs `mutation-fused.log`,
`mutation-run1.log`, `mutation-run2-v2.log`). 14 deliberate bugs, each required to fail the tests; all 14 failed:

| mutation | caught by |
|---|---|
| pair: second matrix block index shifted by 1; first matrix block count drops its partial block | `matmul_pair_is_bitwise_two_gemvs` |
| fp8 pair: second segment uses the first segment's scales; second segment writes the first output | `matmul_fp8_pair_is_bitwise_two_gemvs` |
| glu: up scaled by the gate's global scale; gate not rounded to bf16 before silu; up accumulates gate block scale | `matmul_nvfp4_glu_is_bitwise_gate_up_silu` |
| gated norm: silu applied to the wrong operand; normalised value not rounded (vector path); (scalar path) | `gated_rmsnorm_is_bitwise_rmsnorm_then_silu_glu` |
| v2 down: last slot dropped; no rounding before the routing weight; upper half of h replaced; invalid id not zeroed; wrong row's scales | `moe_experts_decode_matches_old_kernel` |

One mutation survived on the first run: summing the 8 slot terms in reverse order (`mutation-run1.log`, `down-order`). The
terms are bf16 values within a few binades, so their fp32 sums are exact and any order gives the same bits. The tests
therefore do **not** guard the summation order of v2; v2 keeps the old order by construction. The mutation was replaced by
"last slot dropped", which is killed. Wrong assumption recorded: "a different order changes the bits".

**Whole model, 35B** (window 3, `raw/w3/`):

| check | result |
|---|---|
| `accuracy --backend cuda` 35b-short, vLLM golden sha `b6b51e1b1699c66c`, the invocation of `scripts/acc35b/w1.sh` | floor verdict **PASS**: prefill top-1 97.99%, 21-way KL 6.271e-3; decode top-1 98.12%, KL 1.556e-2, tail excess -0.0022, no non-finite rows |
| same on the base tree, dumps compared (`cmp_dump.py`) | **60 of 60 tensors bit-identical** (top-64 log-probs of every prefill and decode row), per-position TSV identical: the numbers did not move at all, as bitwise fusion predicts |
| 256 greedy tokens, graph vs `--no-graphs` (dev) | **ids identical**; also identical to the base build's 256 ids |
| 160 greedy tokens under nsys, base vs dev | generated text identical |
| `intarsia qualify --quick-only` (35B, spark1) | **PASS 7/7** (doctor, recipe, plan, kernels 41 / 569 cases, gate ppl 1.22, acc-short prefill 97.92% KL 5.87e-3 / decode 98.12% KL 1.54e-2, bench 80.7 tok/s) |
| 8B fp32 goldens, `--gate-class bf16` | 8b-short: PASS (prefill top-1 100%, KL 2.90e-4; decode 100%, 3.24e-4); 8b-long: PASS (99.65%, 4.00e-4; 100%, 3.33e-4) |

`35b-short` is advisory per ADR-0008; it is reported because the dumps being bit-identical is the stronger statement.

**Graph proof.** nsys of the dev build shows 160 steps of one graph, 724 kernel nodes each; every `bench speed --require-graphs`
arm reports `eager_steps 0` and 1920 replays (3 sizes x 5 reps x 128 steps); the 256-token
graph run and the eager run produce the same ids; the graph run is 80.96 tok/s, the eager 72.27 on the same build, the
base-build graph 72.45.

Mac: `INTARSIA_CPU_FREE_MEM_GB=8 cargo test --workspace` green; `cargo check --workspace` and
`cargo check -p intarsia-cli --features hip` green.

## 5. Not done, uncertain

* The fp8 GEMVs (47% of the kernel time) were not touched. I did not analyse their achieved bandwidth against shapes; qualify's
  byte floor puts the whole step at 82%, so at most about 18% is left for everything. The step-4 item was spent on MoE
  down (a defect visible in the baseline profile: 160 registers, one block per SM) rather than on the largest share; I chose
  it for its measured headroom, not because it was the largest.
* The split of the +11% between the four fusions and v2 is from per-kernel sums, not from an `INTARSIA_NO_FUSION=1` arm.
* The vLLM figure is the brief's. 8B has one round, not three. Prefill 4096 had one low dev rep set (see section 1).
* The summation-order of v2 is not covered by a test (section 4).
* Commit trailer: the brief says "Claude Opus 5.5"; a later system note asked for "Sonnet 5.5". I kept the brief's trailer.

## 6. Process: spark1, downtime

Builds ran under `fleetq run --host spark1 --mem 0` with vLLM up; correctness tests and mutation runs under `--mem 4..8`
with vLLM up (no timing claims from them, except the cold MoE microbenchmark which ran with vLLM idle, GPU 0%: 81.4 -> 60.5
us is a ratio on a quiet GPU and was not used for any end-to-end claim). The four vLLM-down windows used
`scripts/cuda-decode/window.sh` (a copy of the acc35b window script: exclusive fleetq lease taking `guidance` down,
`docker stop`, memory wait, `FLEETQ_JOB` on every ssh including cleanup/pkill, trap restore, `/v1/models` wait, release).
No `--force`, no `--no-queue`, no lease held while coding.

| window | content | down |
|---|---|---|
| W1 #1730 | baseline nsys profile | 5.5 min |
| W2 #1753 | dev nsys, A/B rounds 1-2 | 10.1 min |
| W3 #1757 | accuracy dev + base, 8B goldens, graph proof | 9.6 min |
| W4 #1759 | qualify, A/B round 3, 8B A/B | 19.2 min (vLLM took 3.8 of it to come back) |
| total | | **44.4 min**, each window under 20 |

After the last window the vLLM container answered `/v1/models` (model id `guidance`) at 20:53:32 per the window script, and
again when checked at 20:54. `downtime.log` has the lines.

Wrong assumptions recorded: the W2 early-abort guard compared nsys stdout and printed "DIFFERS" because the nsys banner
(temp file name, progress bar) differs; the text itself is identical (checked afterwards by stripping the banner). The
first `mutate.py` order mutation was an equivalent mutant on the test data (section 4).

Files: `kernels/cuda/{matmul,fp8,nvfp4,norm,moe_decode,elementwise}.cu`, `ik_launch.h`, `kernels/abi/ik_abi.h`,
`crates/intarsia-backend-cuda/src/{quantmoe,device_quantmoe,kernels}.rs`, `tests/{fused_decode,quantmoe}.rs`,
`scripts/cuda-decode/*`.
