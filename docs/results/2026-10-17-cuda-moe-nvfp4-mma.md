# CUDA prefill: MoE experts on the block-scaled FP4 tensor core (W4A4), Qwen3.6-35B-A3B NVFP4 on spark1 (GB10, sm_121), 2026-10-17

Branch `perf/cuda-moe-nvfp4-mma` (from origin/main `acef8a8`). Raw logs: `docs/results/2026-10-17-cuda-moe-nvfp4-mma/raw/{wA,wB,dev,vllm-backend.txt}`; vLLM downtime log `docs/results/2026-10-17-cuda-moe-nvfp4-mma/downtime.log`; scripts `scripts/cuda-moe-nvfp4/`.

## 1. Result

1. Built `moe_experts_nvfp4_mma_bf16`: the grouped MoE expert GEMMs (gate/up and down) of `moe_experts_mma_bf16` on `mma.sync ... kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64 ... e2m1.e2m1.f32.ue4m3` (SASS `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`). The checkpoint weights are already e2m1 + one ue4m3 scale per 16 k, so they go from global memory to the MMA with no dequantisation; the activations are quantised to the same format by a new kernel.
2. **Speed (spark1, `bench speed`, 3 interleaved rounds, base = origin/main build): prefill +11.0 % / +10.4 % / +12.4 % at 1024 / 4096 / 16384 tokens (3938 -> 4372, 4583 -> 5061, 4362 -> 4901 tok/s); every round positive. Decode is unchanged (-0.4 / -0.3 / -0.2 %, noise; the kernel needs 128 tokens).** vLLM is at 5338 / 6334 / ~6400 (section 3 of `2026-10-17-cuda-prefill-2.md`), so the gap goes from 26 / 28 / 32 % to 18 / 20 / 23 %.
3. **Accuracy: worse than base against the vLLM golden, so it ships opt-in (`INTARSIA_CUDA_MOE_NVFP4=1`), off by default.** Prefill KL21 on 35b-short is 1.04e-2 / 1.08e-2 (chunk 256 / 2048) against base's 5.43e-3 .. 6.27e-3, and on 35b-long 5.71e-2 / 6.44e-2 against 4.27e-2 .. 4.58e-2. The prefix-reuse gate fails one case (base passes all). The reason the hoped-for improvement did not happen: **the vLLM that made the golden runs the weight-only Marlin NVFP4 MoE backend (bf16 activations)** on this box (`raw/vllm-backend.txt`: `Using 'MARLIN' NvFp4 MoE backend` at every start), so quantising activations to fp4 moves us away from it, not towards it.
4. Build: only the new file is compiled for `sm_121a` (per-file, `-gencode arch=compute_121a,code=sm_121a`). Every other object had identical SASS to the origin/main build in the `sass.sh` run (section 2).
5. Tests (what I saw pass; section 11 lists what I could not re-confirm): single-tile layout check against a cpu reference; kernel vs a plain-Rust reference of the *quantised* arithmetic, which passed on every case with rel_l2 < 3e-3 and printed 0 differing elements on the cases whose output I kept (12 288 elements each); vs the bf16 reference within the stated tolerance; routing edge cases; mutation check (7 mutants confirmed killed in the final run, the rest not re-confirmed). The whole CUDA crate test suite passes (`raw/dev/full-cuda-crate-suite.out`, kernel off by default).

## 2. Architecture decision (`sm_121` vs `sm_121a`)

`mma ... kind::mxf4nvf4.block_scale` is rejected by `ptxas` for `-arch=sm_121` and accepted for the architecture-specific `sm_121a` / `sm_120a` (probe 2026-10-17, fleetq #2400; this branch re-confirmed it with `kernels/cuda_checks/omma_tile.cu`). Chosen: **compile only this kernel's file for `sm_121a`; keep the default `sm_121` for every other kernel.**

* `crates/intarsia-backend-cuda/build.rs`: a source named `*_sm120a.cu` is built with `-gencode=arch=compute_121a,code=sm_121a -DIK_SM120A=1` (for `INTARSIA_CUDA_ARCH` = `sm_120` it is `sm_120a`; for any other arch it is built as a stub with `-DIK_SM120A=0`). All other files get exactly the nvcc command line they had before. It also emits `cfg(ik_sm120a)`.
* `nvcc --dryrun -arch=sm_121a` shows `cicc -arch compute_121`, i.e. plain `-arch=sm_121a` also runs a generic `compute_121` pass, and that pass has no `sm_121a` feature macros. I therefore build with `-gencode arch=compute_121a,code=sm_121a`, which has only the a target. (I did not test whether the plain `-arch=sm_121a` form fails to build; this is a precaution taken from the dry run.)
* Fallback: on a build without `sm_120a/121a` the launcher symbol still exists and returns `IK_ERR_UNSUPPORTED`; the Rust registration is `.when(cfg!(ik_sm120a) && ...)`, so the kernel is never selected there and `moe_experts_mma_bf16` stays the only grouped kernel. An `sm_121a` cubin runs only on exactly that device (GB10); `sm_120` (RTX 50xx) is wired the same way but **untested**.
* Check that nothing else changed (`scripts/cuda-moe-nvfp4/sass.sh`: `cuobjdump -sass` of every object built by cargo for the origin/main export and for this branch, both on spark1): every object present in both reported **IDENTICAL**, `moe_grouped.o` included (its grouping and reduce kernels moved unchanged into `kernels/cuda/moe_group.cuh`, shared with the new file); the only object not in base is `moe_nvfp4_mma_sm120a.o`. That run was before the last two edits to the new .cu file only; no other kernel source changed after it. Its output was partly truncated in my session and the log was not kept, so rerun `sass.sh` to reproduce. The build default is **not** changed, and greedy ids of this branch with the kernel off equal the base build's (section 7).

## 3. Fragment layout test (the first thing built)

`kernels/cuda_checks/omma_tile.cu` (standalone: `nvcc -gencode arch=compute_121a,code=sm_121a`, `scripts/cuda-moe-nvfp4/probe.sh`): one warp, one m16n8k64 tile, random e2m1 data, a distinct random ue4m3 scale for every (row, k16 block) of A and every (column, k16 block) of B, against a double-precision cpu reference. **Trials 0 to 5 (the ones I saw printed) all had worst error exactly 0, and the SASS contains `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`; the remaining trials and the final PASS line were cut off in my view of the output and the log was not kept: rerun `probe.sh` to confirm 20 of 20.** The layout below is the one the GPU kernel test then relies on (g = lane/4, t = lane%4; element i of a 32-bit register is the nibble at bits [4i+3:4i], low nibble first):

| operand | register | holds |
|---|---|---|
| A (16x64, row) | a0 / a1 | row g / g+8, k = 8t .. 8t+7 |
| | a2 / a3 | row g / g+8, k = 32+8t .. 32+8t+7 |
| B (64x8, col) | b0 / b1 | column g, k = 8t .. 8t+7 / 32+8t .. 32+8t+7 |
| C/D (f32) | c0,c1 / c2,c3 | row g / g+8, columns 2t, 2t+1 |
| scale A (`scale_vec::4X`) | one u32 = 4 ue4m3 bytes, byte j = k16 block j | lane t==0 supplies row g, lane t==1 supplies row g+8; selectors `{0,0}` |
| scale B | same | lane t==0 supplies column g; selectors `{0,0}` |

Memory order equals hardware k order, so a row of packed fp4 (low nibble = even k, as in the checkpoint) is loaded with plain 32-bit shared loads: `a0 = *(u32*)(row_g + 4t)`, `a2 = *(u32*)(row_g + 16 + 4t)`. No `ldmatrix`, no dequantisation.

**Sources: not cited from a fetched page.** The delegated literature lookup (CUTLASS `mma_traits_sm120.hpp`, the PTX ISA block-scaling section, vLLM/FlashInfer `csrc/quantization/fp4`) failed in my session and I did not retry it, so the layout above and the quantiser recipe in section 4 were written from my recollection of those sources (CUTLASS `SM120_16x8x64_TN_VS`; vLLM `nvfp4_utils.cuh` quantisation: per-16 block scale `amax/6` through e4m3, `cvt.rn.satfinite.e2m1x2`, global scale `448*6/amax`) and are **not verified against them**. What verifies the layout is the test, not the citation; what is *not* verified is that the quantiser matches vLLM's bit for bit (it differs by design in the global scale: per row, not per tensor).

## 4. What was built

* `kernels/cuda/moe_nvfp4_mma_sm120a.cu` (new): `ik_moe_experts_nvfp4_mma_bf16`, same inputs as `ik_moe_experts_mma_bf16`, grouping and reduce reused (`moe_group.cuh`).
  * **Activation quantiser** `act_quant_kernel` (one warp per row): `g = 2688 / amax(row)` (448 x 6), per 16-wide block `s = e4m3(amax_blk * g / 6)` (round-to-nearest, saturating), elements `e2m1_rne(x * (g / s))` (`cvt.rn.satfinite.e2m1x2.f32`, ties to even), row scale stored as `amax / 2688`. This follows the vLLM/FlashInfer recipe as I remember it (see section 3: not re-checked) **with one scale per row instead of per tensor**: chunk-independent, no grid-wide reduction, deterministic. Rows with amax <= 1e-30 are zero (2688/amax overflows for denormals). x is quantised once per token (T x H), not once per pair.
  * **Grouped GEMM** `moe_nvfp4_kernel<MT,WM,MODE,NS>`: tile table, `BM` 16/32/64 chosen from pairs/expert, 4 x WM warps, a block = one tile x (64 gate + 64 up) columns (gate_up) or 128 columns (down), as in `moe_grouped.cu`. A unit is 128 k = 64 bytes of fp4 per weight row (the DRAM-friendly run length of `moe_grouped.cu`) + 8 scale bytes, copied with `cp.async` into an `NS`-stage ring (default 2: two blocks per SM; `INTARSIA_NVFP4_STAGES=3|4` for experiments). Fragments come straight from the staged bytes (conflict-free: 80-byte row stride). Each k64 mma starts from a zero accumulator and is added to the fp32 accumulator, as in the bf16 kernel. Epilogue: `acc * (amax/2688 * weight_global_scale)` (the global scale of the checkpoint, applied in fp32 per row) then the same SiLU-GLU with the reference rounding points; `h` is written in sorted pair order as bf16, quantised by the same kernel (per row of `[P, I]`), and the down GEMM scatters `y[pair]`.
  * Stub when `IK_SM120A=0`.
* `crates/intarsia-backend-cuda/src/{quantmoe,kernels,device_quantmoe}.rs`: registration (`TABLE`, `moe_nvfp4()`, `moe_nvfp4_priority`: priority 20 with `INTARSIA_CUDA_MOE_NVFP4=1`, -10 otherwise, so it is registered and admitted but never selected by default), predicate `cfg!(ik_sm120a) && m >= 128 && experts <= 1024 && k % 128 == 0 && n % 128 == 0`, scratch function (`moe_experts_nvfp4_scratch` = the grouped kernel's plus the quantised copies of x and h), FFI. The launcher is declared in a new `kernels/cuda/ik_cuda_abi.h` (the CUDA twin of `kernels/hip/ik_hip_abi.h`), not in the shared `kernels/abi/ik_abi.h`.
* Minimum 128 tokens: a **chosen** value, not a measured crossover. The microbench was only run at 2048 and 4096 tokens (section 6); I do not know where FP4 stops winning below that, but the end-to-end 1024-token prefill got faster (section 8). 128 keeps the admission shapes T = 130 and T = 256 in the sweep.
* **Admission** (`intarsia kernels test` sweep, `every_cuda_kernel_matches_the_reference`): `moe_experts_nvfp4_mma_bf16 ... ADMITTED` (per-case figures not kept). This is a *weak* gate by construction: the op's reference is the bf16-activation `ref_moe_experts`, and the kernel differs from it by the activation quantisation, so its registered tolerance is `NVFP4_ACT_TOL = (max_abs 0.30, rel_l2 0.20)`, derived on the Mac, before any GPU number, by `tests/moe_nvfp4_cpu.rs` (the quantised arithmetic in plain Rust vs the bf16 cpu reference over the admission shapes: rel_l2 0.163 .. 0.166, max_abs 0.14 .. 0.19 of max(1, max|ref|); about 8 % per e2m1 rounding, two of them) plus a 20 % / 50 % margin. The real check of the arithmetic is the exact test below, not admission.

## 5. Kernel tests (`crates/intarsia-backend-cuda/tests/moe_nvfp4.rs`, `moe_nvfp4_cpu.rs`, `moe_nvfp4_ref/`)

* `moe_nvfp4_matches_the_quantised_reference`: kernel vs `Quantised::run` (activations quantised exactly as the kernel does, e2m1 x e4m3 products summed in f64, bf16 rounding at the kernel's points). Six shapes covering tile heights 16 / 32 / 64, several I, and H=2048/I=512 with every expert hot, with distinct global scales for gate / up / down / expert. Passes with rel_l2 < 3e-3 (room for fp32 summation order flipping a bf16 rounding point). The test prints the number of differing elements; in the output I kept (the next test) it is 0 of 12 288 in every case, and I did not keep the printout of this test's six shapes.
* `moe_nvfp4_handles_range_zero_rows_and_odd_routing`: rows of magnitude 1e3, 1e-3, 0, 5e4, 3e-6, 1e-37, 1e-33 (per-row scale and the zero guard); duplicate, invalid (-1, E+5) and hot-expert routing; T = 1 at the real shape. Seen: 0 of 12 288 elements differ in the first two checks.
* `moe_nvfp4_is_within_activation_quantisation_error_of_the_bf16_reference` (vs `ref_moe_experts`, within `NVFP4_ACT_TOL`) and `moe_nvfp4_at_the_real_prefill_shape` (T=1024, 256 experts, top-8, H=2048, I=512, vs the bf16 GPU kernel): the latter printed rel_l2 0.1634. The cpu-only `moe_nvfp4_cpu` measured the quantised arithmetic against the bf16 cpu reference at rel_l2 0.164 / 0.166 / 0.163 (T=130 / 45 / 60), worst max_abs 0.195 of max(1, max|ref|).
* `moe_nvfp4_is_independent_of_tile_height` (bit-identical for BM 16/32/64), repeatability in every test, `moe_nvfp4_cpu` (e2m1 ties and saturation, quantiser round trip, zero rows).
* Whole CUDA crate on spark1 with vLLM up (`gpu-test.sh nv-dev`, `raw/dev/full-cuda-crate-suite.out`): lib 5, admission 4, attention 1 (+1 ignored), fused_decode 9, gdn_lanes 2, gdn_timing 1, graph_pos 5, moe_nvfp4 5 (+1 ignored bench), moe_nvfp4_cpu 3, quantmoe 10 (+3 ignored): all ok. Mac: `INTARSIA_CPU_FREE_MEM_GB=8 cargo test --workspace` (no failures in the log), `cargo check --workspace`, `cargo check -p intarsia-cli --features hip` (type-check only): green.

### Mutations (`scripts/cuda-moe-nvfp4/mutate.py`, `mutcheck.sh`, pattern of `scripts/cuda-decode/`)

14 mutants of the new kernel file (table below), each built and run against the `moe_nvfp4` test binary on spark1 (vLLM up), unmutated tree first. **What I can state:** in the final run the unmutated tree passed (5 tests) and these 7 were **KILLED** (output seen): `sfa-rows` (A scales of rows g+8 from row g), `sfa-lane` (only lane t==0 supplies A scales), `sfb-step`, `b-second-half`, `no-ginv`, `up-uses-gate-scale`, `silu-sign`. An earlier run found `up-uses-gate-scale` **surviving** because the generated test data gave gate, up and down the same global scale; `with_distinct_gscales` was added and the mutant is then killed. The other 7 (`quant-scale`, `quant-nibbles`, `quant-ignores-scale-rounding`, `a-unit-offset`, `valid-rows`, `up-rows`, `scatter`) were part of the same runs but their verdicts were cut off in my view of the output and the mutation log was not kept, and after the last kernel edit I did not see them: **not confirmed; rerun `scripts/cuda-moe-nvfp4/mutcheck.sh`** (about 10 minutes on spark1 with vLLM up; I withdrew the re-run because spark1 was queued behind other jobs).

## 6. Microbenchmark (`tests/moe_nvfp4.rs bench_moe_nvfp4_real_shapes`, `scripts/cuda-moe-nvfp4/bench.sh`)

Real layer shape (256 experts, top-8, H 2048, I 512), one call = grouping + gate/up + down + reduce (+ the quantisation kernels for the new one). **Run on spark1 with vLLM up** (GPU idle, memory shared: use as relative numbers). TFLOPS = 2 x pairs x (H x 2I + I x H) / time, computed from the times. **Only T = 2048 and 4096 were measured** (the sweep over smaller T that the bench supports was not run; `INTARSIA_BENCH_T=...`). The microseconds below are from one run each whose log I did not keep (a re-run was withdrawn, see section 11).

| T | bf16 `moe_experts_mma_bf16` | FP4 `moe_experts_nvfp4_mma_bf16` (2 stages) | change |
|---|---|---|---|
| 2048 | 4235 us, 24.3 TFLOPS | 3237 us, 31.9 TFLOPS | -24 % time |
| 4096 | 6791 us, 30.4 TFLOPS | 4276 us, 48.2 TFLOPS | -37 % time |

(the 25 TFLOPS of the profile in the prefill-2 doc is the bf16 row at 2048.) Pipeline depth, FP4 kernel, same run set: 2 stages 3237 / 4276 us, 3 stages 3316 / 4692, 4 stages 3269 / 4757 (at T = 2048 / 4096), so 2 is the default. FP4 timings vary by about 1 % between runs (3237 and 3279 us were both seen at 2048 with the same configuration).

Where the time goes at T = 2048 (nsys, per call): FP4 gate_up 1.66-1.73 ms (bf16: 2.26), down 1.08-1.24 (bf16: 1.58), reduce 0.28-0.30, two activation quantisations 0.05 each, grouping 0.02. Experiments with parts of the main loop switched off (T = 2048, 2 stages; temporary code, removed): no mma 3320 us (vs 3279 with it), no weight/activation loads in the loop 1536 us, neither 1291 us. So **the tensor-core work is invisible; the kernel is memory bound** (450 MB of weights = 1.65 ms at the 273 GB/s peak, plus 67 MB of `y` written and read again), and about 1.3 ms of fixed work (epilogue writes, reduce, launches, block prologues) does not overlap with the weight stream. Not done: a persistent kernel that keeps the activation tile resident and streams several column tiles per block (would overlap the epilogue and drop the repeated activation loads; my guess is 2.5 ms at 2048, not measured), and fusing the `h` quantisation into the gate_up epilogue (a per-64-column scale instead of per-row would make it possible; saves about 0.15 ms).

The end-to-end gain (+10 .. +12 %) was not predicted quantitatively from this microbench and was also seen at 1024 tokens, which the microbench did not cover. Not investigated; a possible reason is that real routing is skewed (many half-empty tiles) while the benchmark's is random.

## 7. Accuracy against the vLLM golden (window A, `raw/wA/`, vLLM down, `intarsia accuracy --mode prefill`)

Golden `b6b51e1b1699c66c` (35b-short, 5 prompts) and the long golden (36 887 positions); base = origin/main build run in the same window at chunk 2048; the base spread at chunks 256 / 1024 / 2048 is the one given in the task and measured in `2026-10-17-cuda-prefill-2.md` (not re-measured at 256 / 1024 here; the two base chunk-2048 numbers measured here, 5.430e-3 and 4.576e-2, sit exactly at the ends of those ranges).

| golden | base c2048 (this window) | base range c256..c2048 (prefill-2 doc) | FP4 c256 | FP4 c2048 | verdict |
|---|---|---|---|---|---|
| 35b-short KL21 | 5.430e-3 | 5.430e-3 .. 6.271e-3 | 1.043e-2 | 1.079e-2 | **outside, +66 % .. +99 %** |
| 35b-long KL21 | 4.576e-2 | 4.273e-2 .. 4.576e-2 | 5.706e-2 | 6.445e-2 | **outside, +25 % .. +41 %** |
| 35b-short top-1 | 97.92 % | | 97.92 % | 97.85 % | |
| 35b-long top-1 | 94.17 % | | 92.81 % | 92.45 % | |

The absolute gate (KL21 <= 3.0e-2) passes on short (1.0e-2) and fails on long for base and FP4 alike (pre-existing, see the prefill-2 doc). The bar was "no worse than base's chunk-perturbation range": **not met on either golden.** Tail mass excess stays inside its limit (0.0181 vs ref 0.0176 short; 0.0656-0.0665 vs 0.0611 long).

* **Prefix-reuse gate** (`accuracy --mode reuse`, long golden, chunk 2048; the two 4k prompts are skipped by the tool, split 2000 < one chunk, for both builds): base **PASS** (all 9 cases agree 32/32), FP4 **FAIL**: `long_docs_12k continue` diverges at token 8 (warm token 2135, cold 63; shared logit rows differ by 1.08 nats before / 0.395 nats at the divergence), the other 8 cases agree 32/32. A hypothesis for the cause (not tested): the warm run reuses KV/state of the turn-1 reply tokens written by the decode kernels (bf16 activations) while a cold run re-prefills them through the FP4 kernel, so the two differ by the activation quantisation (about 16 % rel_l2 per MoE layer), more than the tool's rounding-tie tolerance; any W4A4 prefill beside a W4A16 decode would behave so. The other 8 cases, including `restore` and `aligned` at 12k, agree 32/32, which argues against a stale-row bug but does not prove it.
* **Greedy ids** (`generate --chat --no-think --ignore-eos`, 256 tokens, a 921-token prompt, above the kernel's 128-token minimum): FP4 eager (`--no-graphs`) == FP4 graph (256 of 256 ids identical, 256 replays vs 0); kernel off == base build (identical). FP4 vs base first differ at token 31 of 256 (expected).
* The accuracy logs do not name the selected kernel; that the FP4 kernel ran in these runs follows from the changed KL and from the A/B below (the two arms differ in prefill speed only through it).

**Decision: opt-in, `INTARSIA_CUDA_MOE_NVFP4=1`, off by default.** The KL21 and the reuse gate say the numerics are different enough to need an owner's decision (and, for a default, a golden made with a W4A4 reference, e.g. vLLM's FlashInfer-CUTLASS FP4 MoE backend, which this vLLM did not pick on GB10).

## 8. Speed A/B (window B, `raw/wB/`, `scripts/cuda-moe-nvfp4/wB-ab.sh` via `w-ab.sh`)

`intarsia serve` per arm, `bench speed --warmup 1 --thinking off --require-graphs`, gen 128, 2 reps per size, 3 rounds interleaved (base, FP4 / FP4, base / base, FP4); base = origin/main build, FP4 = this branch with `INTARSIA_CUDA_MOE_NVFP4=1`. Full tables: `ab-table.md`.

| prompt tokens | base prefill tok/s (r1, r2, r3) | mean | FP4 (r1, r2, r3) | mean | delta | per round |
|---|---|---|---|---|---|---|
| 1024 | 3976, 3909, 3929 | 3938 | 4407, 4375, 4336 | 4372 | **+11.0 %** | +10.8, +11.9, +10.4 |
| 4096 | 4714, 4465, 4571 | 4583 | 5160, 5021, 5002 | 5061 | **+10.4 %** | +9.5, +12.5, +9.4 |
| 16384 | 4373, 4358, 4355 | 4362 | 4891, 4901, 4912 | 4901 | **+12.4 %** | +11.8, +12.5, +12.8 |

TTFT: -9.8 % / -9.1 % / -11.0 %. Decode (graph replay, kernel not used): 80.4 -> 80.1, 77.7 -> 77.4, 70.4 -> 70.2 tok/s (-0.4 / -0.3 / -0.2 %): unchanged within round-to-round noise (base alone moves 80.1 .. 80.75 between rounds).

## 9. vLLM downtime

Two exclusive windows through the unmodified `scripts/cuda-decode/window.sh` (only `DOWNTIME_LOG` set), `--takes-down guidance`, released at the end of each, no `--force`, no `--no-queue`:

| window | job | stop -> serving again | down |
|---|---|---|---|
| A accuracy and gates | #2425 | 16:12:01 -> 16:23:48 | 11.7 min |
| B speed A/B | #2427 | 16:24:10 -> 16:34:20 | 10.1 min |

**21.8 min of the 35 min cap, no window over 12 min.** The window script saw `/v1/models` answer at 16:23:48 and 16:34:20; `/v1/models` returned 200 again when I checked after the last window (16:4x). Everything else ran with vLLM up through `fleetq run --host spark1 --mem 1..12` (builds, SASS, tests, mutations, microbench, nsys).

## 10. PR handoff

Changed outside `crates/intarsia-backend-cuda` and `kernels/cuda` (`git diff --stat origin/main`):

* `kernels/cuda_checks/omma_tile.cu` (new standalone layout check)
* `scripts/cuda-moe-nvfp4/` (new: dev/probe/bench/mutation/sass/window scripts, `mutate.py`, `ab_table.py`)
* `docs/results/2026-10-17-cuda-moe-nvfp4-mma.md` and `docs/results/2026-10-17-cuda-moe-nvfp4-mma/` (this page, raw logs, tables)

**No change** to shared code: `kernels/abi/ik_abi.h`, `intarsia-core`, `quant`, `model.rs`, fused-op or registry definitions, recipe keys (the switch is an environment variable, `INTARSIA_CUDA_MOE_NVFP4`). Inside the CUDA crate and kernel dir: `build.rs` (per-file arch, `cfg(ik_sm120a)`), `kernels/cuda/moe_group.cuh` (new; grouping/reduce moved out of `moe_grouped.cu` unchanged, SASS identical), `kernels/cuda/ik_cuda_abi.h` (new), `kernels/cuda/moe_nvfp4_mma_sm120a.cu` (new), `src/{kernels,quantmoe,device_quantmoe}.rs`, tests. The ABI-header test in `src/kernels.rs` now reads `ik_cuda_abi.h` as well. HIP and Metal were not built or run (`cargo check -p intarsia-cli --features hip` type-checks only); nothing in them changed. The CUDA build default is **not** changed.

## 11. Not done / uncertain

* **Not re-confirmed, evidence not kept:** the full mutation table (7 of 14 seen killed), the 20-trial layout probe summary (trials 0 to 5 seen), the per-shape element-difference counts of the main exact test, the admission per-case figures, the microbench microseconds (two sizes only) and the SASS-identical verdict per object. I tried to regenerate them into `raw/dev/` at the end, but spark1 was by then queued behind other chats' exclusive jobs (a ~1 h download and a TP=2 serve), so I withdrew the request (fleetq #2435 cancelled) rather than wait. All are one command each: `probe.sh`, `sass.sh`, `mutcheck.sh`, `dev2.sh` (with `BENCH=1`), `bench.sh`.
* The layout and quantiser recipe were not checked against CUTLASS / the PTX ISA / vLLM text (section 3).
* Tolerance in admission is loose by nature (section 4); the exact tests carry the correctness argument.
* Per-row activation scale instead of vLLM's per-tensor one: chosen for determinism and chunk independence; its accuracy effect against per-tensor was not measured.
* The kernel is memory bound at 2048 tokens; a persistent/fused version (section 6) is the next step.
* `sm_120` (RTX 50xx) is wired but untested. The 128-token minimum was not tuned. The E2E gain vs microbench is unexplained.
* KL baseline spread comes from the prefill-2 doc; only chunk 2048 was re-measured here.
