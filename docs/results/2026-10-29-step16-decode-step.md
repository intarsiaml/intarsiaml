# step16: B = 16 decode step of the 35B on spark1: GDN at its bandwidth floor, vLLM's GDN state is f32 too, f16 staging not faster; FHFMA in the fp8 persistent rows kernel (below 12 rows), bitwise the same, no in-engine gain

Lane step16, 2026-10-29, branch `lane-step16/decode-step` from `origin/main` ebf8192a0 (#235). spark1 (spark-4e77, GB10 sm_121),
Qwen3.6-35B-A3B NVFP4, `serve --backend cuda --max-batch 16 --max-seq 8192 --mem-budget 40`.

**Status: draft PR. The speed gates are not met (B = 16 step unchanged, 16 streams +0.2 %), and the one code change is not
measurably faster in the engine. The coordinator decides whether the findings alone are worth the merge.**

What the lane found, in short:

- **GDN at B = 16 is at the bandwidth floor.** Per layer the delta rule streams 64 MB of f32 state at about 232 GB/s. The conv
  streams 3.6 MB at about 220 GB/s. No exact layout, fusion or occupancy change can make it shorter. Section 1.
- **vLLM keeps the GDN state in f32 too**, so the GDN state bytes are not where the ~4 ms per step to vLLM comes from (an
  earlier read of the vLLM source suggested bf16; the Qwen3.5 config override says f32). Section 1.
- **Staging the W8A8 activations as f16 in the e4m3 rows kernel** (the #230 follow-up) is exact but **not faster** at 16 rows.
  Measured, reverted, patch kept in raw. Section 2.
- **The change: FHFMA in `fp8_gemv_rows_full_kernel`**, the persistent kernel that runs every 35B fp8 projection below 12 rows
  (1 to 11 streams). Bitwise the same.
  - Evicted micro: 3 to 13 % faster per call at 4 and 8 rows.
  - In the engine's graph: **not faster** (B = 8 nsys: 66.1 -> 66.3 us per call; step micro B = 8: within noise).
  - Nothing at 16 rows (not the kernel used there). Sections 3 and 5.

| streams | serve A/B (median of 3 per-round ratios) | gate |
|---|---:|---|
| 16 | +0.2 % | +2 %: **fail** |
| 8 | +1.2 % | not below -1 %: pass |
| 4 | +0.5 % | not below -1 %: pass |
| 1 | -0.5 % | not below -1 %: pass |

B = 16 step micro-timing (`timing_35b_batched_step`): 44.47 -> 44.48 ms, +0.0 % (gate -3 %: **fail**). B = 8: 27.69 -> 27.62 ms,
-0.3 %, inside the base runs' own spread. Section 5.

Every exactness gate passes:

- `batch_cuda_real` on the 35B: 4290 (row, step) pairs bitwise, plus the engine test (`raw/batch-real.txt`).
- `rows_exact`: 9 tests, every form, M = 1..16 (`raw/rows-exact.txt`).
- Served greedy tokens identical to main: 21/21 cells, 9408 ids, single stream included (`raw/ident/compare.log`).
- `plan show` byte-identical for 8B, 35B and Gemma-4, json and txt (`raw/plan.log`).
- Mac `cargo test --workspace --release`: exit 0, 1090 passed, 0 failed over 204 test binaries (`raw/mac-cargo-test.log`).

Scripts: `scripts/step16/` (from moe16c / srv16: `build.sh`, `micro.sh`, `w1.sh`, `w2.sh`, `steptime.sh`, `prof.sh`, `ident.sh`,
`plan.sh`, `le-window.sh`, `batch-real.sh`). Raw: `2026-10-29-step16-decode-step/raw/`.

## 1. GDN at B = 16: at the floor

Per-call times from the moe16b nsys run at B = 16 (`2026-10-28-moe16b-step-overlap/raw/{base,dev}-b16/rank.txt`). `gdn.cu`
has not changed since. W2's own profile is in section 5.

| kernel (35B, 30 GDN layers) | per call | bytes per call (16 rows) | rate | per step |
|---|---:|---:|---:|---:|
| `gdn_delta_split_kernel<128, 2, true>` | 276 to 286 us | 64 MB: state [32, 128, 128] f32, 2 MB per sequence, read and written | ~232 GB/s | ~8.4 ms |
| `gdn_conv_rows_kernel` | 16.4 us | 3.6 MB: conv state [8192, 3] f32 read and written, x and y bf16 | ~220 GB/s | ~0.5 ms |

- 232 GB/s is the rate the best decode kernels on this box reach (the one-row fp8 GEMV: 200 to 219 GB/s; the routed experts:
  84 to 88 % of 230 GB/s, MOE16).
- The delta kernel reads and writes each state row once, coalesced. The q / k / v / a / b inputs are under 1 % of the bytes.
  4096 one-warp blocks are launched (16 rows x 32 heads x 8 row groups), each with 8 KB of state in flight.
- Fusing conv into delta saves one launch, not bytes. It would also add an op to the recorded plan, so `plan show` would change.
- So no exact layout, fusion or occupancy change shortens GDN. Only fewer state bytes would, and that is not exact.

**vLLM's state dtype.** The spark1 vLLM (`missioncntrl/vllm-openai:65b7662d`, 0.26.1rc1.dev602) runs without
`--mamba-ssm-cache-dtype`:

- The generic default (`MambaStateDtypeCalculator._mamba_state_dtype`, "auto") would give the model dtype, bf16.
- But `Qwen3_5ForConditionalGenerationConfig.verify_and_update_config` (`model_executor/models/config.py`) sets the cache
  dtype from the HF config's `mamba_ssm_dtype`, and this checkpoint's `config.json` says `"mamba_ssm_dtype": "float32"`.

The checkpoint's `architectures` is `Qwen3_5MoeForConditionalGeneration`. vLLM's `MODELS_CONFIG_MAP` maps it to that override,
and `text_config.mamba_ssm_dtype` is `float32`. vr16's 40.6 ms run (`2026-10-27-vr16-vllm-seqs16.md`) used the same image
(`65b7662d`), also without the flag. So vLLM streams the same f32 state bytes. Its shorter step (~40.6 ms vs ~44.5 ms, SRV16) comes from elsewhere. The likely place
is kernel efficiency in the routed MoE (21.7 ms per step at about 85 % of 230 GB/s), outside this lane's brief. The lane did not
profile vLLM: the server on the box is the production one.

## 2. f16-staged activations in the e4m3 rows kernel (`fp8_gemv_rows_q_kernel`): exact, not faster (reverted)

**What was tried.** #230's next step: stage the W8A8 rows as their f16 pairs instead of e4m3 codes, so the tile loop drops the
x-side F2FP (128 of 144 per 16 weight bytes per lane at 16 rows).

- Rows m < MH were staged as f16 (`ik_fp8x16_to_h2` of the codes, done once in the prologue), the rest as codes, in
  (M + MH) * K bytes.
- MH = M fits at K = 2048 (64 KB, 1 block per SM). At K = 4096 only MH = M / 2 fits (96 KB).
- Exact by construction: the FHFMA chain gets the same f16 bits it got before.
- `rows_exact` passed under both variants (`raw/micro-1.log`).

**Result** (evicted micro-timing, `timing_rows_evict`, median of 1 to 2 runs per variant, us; `raw/micro-1.log`). Run-to-run
noise is about ±1 to 3 us: base and dev with the knob unset run the same kernel and differ by up to 3 us.

| shape, M = 16 | base | f16 all rows (2x8, 1 block / SM) | f16 all rows, 2x16 | f16 half the rows | f16 half the rows, 2x16 |
|---|---:|---:|---:|---:|---:|
| 8192x2048 (GDN qkv, attn q) | 84.32 | 85.28 | 87.17 | 84.50 | 86.44 |
| 4096x2048 (GDN z) | 48.38 | 48.84 | 45.95 | 49.28 | 47.45 |
| 512x2048 (k, v) | 15.04 | 14.66 | 14.30 | 14.31 | 14.32 |
| 2048x4096 (GDN out, o_proj) | 54.05 | 52.51 | 52.02 | 50.89 | 52.04 |

No variant is clearly faster at 8192x2048 or 2048x4096, the projections with 80 of the 110 per-step calls. After #230 the
16-row kernel is no longer limited by the e4m3 -> f16 conversions. The halved blocks per SM and the larger prologue store eat
what the shorter tile loop saves. The patch is in `raw/fp8-rows-q-f16-staging.diff`.

**Headroom left in the fp8 projections at 16 rows, for scale.** Per step:

- 8192x2048 x 40 calls: ~83 us per call against ~76 us at 2 rows (the kernel's own plateau, ~220 GB/s): ~0.3 ms.
- 4096x2048 x 30: ~49 vs ~39 us: ~0.3 ms.
- 2048x4096 x 40: ~51 vs ~41 us: ~0.4 ms.
- k, v x 20: ~14 vs ~8 us: ~0.1 ms.

That is about 1.1 ms (2.5 % of the step) even if 16 rows cost what 2 rows cost. An exact -3 % step is not available in the fp8
projections alone.

## 3. The change: FHFMA in the fp8 persistent rows kernel (below 12 rows): faster alone, not in the engine

`fp8_gemv_rows_full_kernel` (bf16-staged, `[M][2][K / 16]`) runs every 35B fp8 projection at 1 to 11 rows (K = 2048: M <= 11;
K = 4096: M <= 11). #230 did not change it. Now:

- **W8A8** (`sc[1] > 0`, every 35B projection; decided on device as before): the prologue stores the f16 images of the
  re-quantised values (`ik_pack8_f16`, new in `ik_quant.cuh`) instead of bf16 images. The tile loop converts the weights to f16
  pairs (`ik_fp8x16_to_h2`) and runs `ik_fma2_f16`.
- **W8A16**: the old f32 `fmaf` loop, unchanged.
- The two loops are one templated `rows_full_tiles<M, NT, W, H16>`, picked once per block.

**Why exact.** The same argument as #230:

- An e4m3 value is exact in f16 and in bf16, so the staged f16 image holds the same value the bf16 image held.
- `fma.rn.f32.f16` is bitwise `fmaf` on the f32 images (checked exhaustively in #230: 0 of 18.8 M results differ).
- The chain order (16 elements in order, then `acc += s`, then the warp tree), the f32 products and the roundings are unchanged.

On arch below sm_100, `ik_fma2_f16` falls back to `fmaf`.

**Micro-timing** (evicted, `timing_rows_evict`, "unset" = the launcher's default policy, 3 runs each, base / fh / fh / base /
base / fh, medians, us; `raw/micro-2.log`):

| shape | M = 2 | M = 4 | M = 8 | M = 12, 16 |
|---|---:|---:|---:|---|
| 8192x2048 | 74.79 -> 75.92 (noise: runs 74.6-77.6 both) | 79.02 -> 78.82 | **81.48 -> 78.03 (-4.2 %)** | e4m3 kernel, unchanged |
| 4096x2048 | 39.03 -> 38.84 | **39.85 -> 38.65 (-3.0 %)** | **44.27 -> 41.04 (-7.3 %)** | unchanged |
| 512x2048 | 8.32 -> 8.14 | **10.19 -> 8.81 (-13.5 %)** (k / v at 4+ rows take the e4m3 kernel in the engine, `small_n`) | 10.26 -> 10.23 | unchanged |
| 2048x4096 | **41.12 -> 39.65 (-3.6 %)** | **41.71 -> 39.93 (-4.3 %)** | **46.28 -> 42.99 (-7.1 %)** | unchanged |

At M = 12 / 16 the launcher takes the e4m3 kernel (`rows_q`), which this change does not touch. The deltas there (-2.1 to +3.7 %)
are noise.

In the engine's graph the same kernel is not faster (B = 8 nsys, section 5). The change is kept because it is exact, a strict
instruction cut (no f32 unpack of x or weights), and matches the e4m3 kernel since #230. Cost: the second tile loop grows each binary that
holds the kernel by about 4.4 MB (CLI 100.6 -> 105.0 MB, `batch_cuda_real` 90.1 -> 94.6 MB, `rows_exact` 77.4 -> 81.9 MB). If the
coordinator prefers no change without a measured gain, the branch can merge the docs only.

## 4. Serve A/B (W1, `scripts/step16/w1.sh`, `raw/w1/`)

Setup:

- localeval `--sizes 1024 --gen 256 --reps 3 --concurrency 1,4,8,16 --thinking off`.
- 3 interleaved rounds (even rounds reversed), a fresh server per arm.
- base = `origin/main` ebf8192a0, dev = this branch.

Aggregate tok/s:

| streams | base (r1, r2, r3) | dev (r1, r2, r3) | per-round ratio | median |
|---|---|---|---|---:|
| 1 | 81.0, 79.9, 80.7 | 80.1, 81.0, 80.3 | -1.1 / +1.4 / -0.5 % | -0.5 % |
| 4 | 187.5, 174.5, 172.9 | 175.0, 175.3, 174.6 | -6.7 / +0.5 / +1.0 % | +0.5 % |
| 8 | 233.2, 232.9, 234.6 | 239.4, 235.8, 236.7 | +2.7 / +1.2 / +0.9 % | **+1.2 %** |
| 16 | 298.0, 296.3, 296.9 | 298.7, 298.8, 296.5 | +0.2 / +0.8 / -0.1 % | +0.2 % |

- Round 1's base at 4 streams (187.5) is the outlier. The other five 4-stream runs, base and dev, lie within 172.9 to 175.3.
- 16 streams is unchanged, as expected: every B = 16 kernel is the same.
- 8 streams is +1.2 % (+0.9 to +2.7 % per round). The kernel micro would predict about that: the fp8 calls are ~2.5 to 3.3 us
  shorter at 8 rows, ~0.35 ms of a ~28 ms step. But the B = 8 nsys and step micro show no shorter kernel or step in the engine
  (section 5), so the lane reads +1.2 % as noise, not as the change's effect.

## 5. Step micro-timing and per-kernel tables (W2)

**Step micro-timing** (`timing_35b_batched_step`, `scripts/step16/steptime.sh`, `raw/steptime.txt`):

- Setup: one process per run, order base / dev / dev / base. Each run: 20 warm-up steps, then three 60-step windows on a
  growing context (64 to 264 tokens); the middle window is compared.

| run | B = 8 (three windows, ms) | B = 16 (three windows, ms) |
|---|---|---|
| base | 26.51 / 27.57 / 28.89 | 43.00 / 44.40 / 45.89 |
| dev | 26.59 / 27.65 / 28.86 | 43.07 / 44.45 / 45.91 |
| dev | 26.59 / 27.60 / 28.87 | 43.04 / 44.51 / 45.82 |
| base | 26.64 / 27.82 / 29.02 | 43.18 / 44.53 / 45.89 |
| **middle window, mean of 2** | base 27.69, dev 27.62 (-0.3 %) | base 44.47, dev 44.48 (+0.0 %) |

**Per-kernel at B = 16** (nsys, `--cuda-graph-trace=node`, 16-stream burst of 1k-token prompts x 256 tokens, the decode steps
after the last prefill; `raw/{base,dev}-b16/`, `raw/buckets-b16.txt`). Steps 167 / 166, step span 42.01 / 42.35 ms.

| bucket (ms per step) | base | dev | dev - base |
|---|---:|---:|---:|
| routed MoE (gate_up + down_v2) | 18.94 | 19.12 | +0.18 |
| GDN (delta, conv) | 8.96 | 8.90 | -0.06 |
| fp8 K=2048 (GDN qkv / z, attn q) | 5.15 | 5.15 | +0.00 |
| fp8 K=4096 (GDN out, o_proj) | 2.35 | 2.34 | -0.01 |
| attention | 2.24 | 2.20 | -0.04 |
| shared expert nvfp4 (gate, up, down) | 1.59 | 1.56 | -0.03 |
| lm_head nvfp4 | 1.52 | 1.51 | -0.01 |
| norms, top-k, glue | 1.08 | 1.07 | -0.01 |
| bf16 small GEMVs (router, GDN b / a, shared gate) | 0.71 | 0.68 | -0.03 |
| fp8 k / v 512x2048 | 0.37 | 0.37 | +0.00 |
| **sum of kernel durations** | **42.91** | **42.90** | **-0.01** |

The B = 16 kernels are the same code in both arms, so every delta here is noise.

| kernel at B = 16 | grid | per call base / dev | per step base / dev |
|---|---|---:|---:|
| `gdn_delta_split_kernel` | 8x32x16 | 279.2 / 279.5 us | 8.41 / 8.36 ms |
| `gdn_conv_rows_kernel` | 64x16 | 18.3 / 17.9 us | 0.55 / 0.54 ms |
| `fp8_gemv_rows_q_kernel` K=2048 | 96 | 73.3 / 73.8 us | 5.15 / 5.15 ms |
| `fp8_gemv_rows_q_kernel` K=4096 | 48 | 58.6 / 58.8 us | 2.35 / 2.34 ms |
| `fp8_gemv_rows_q_kernel` k / v | 32 | 18.6 / 18.6 us | 0.37 / 0.37 ms |

**At B = 8, where the change runs** (`raw/{base,dev}-b8/`, `raw/buckets-b8.txt`; buckets.py files the rows_full kernel under
"glue"). Steps 138 / 137, step span 30.64 / 30.69 ms.

| kernel at B = 8 | calls per step | per call base / dev | per step base / dev |
|---|---:|---:|---:|
| `fp8_gemv_rows_full_kernel` (every fp8 projection but k / v) | 110 | 66.1 / 66.3 us | 7.31 / 7.30 ms |
| `fp8_gemv_rows_q_kernel` k / v (`small_n`) | 20 | 16.4 / 17.3 us | 0.33 / 0.35 ms |
| sum of kernel durations | | | 30.51 / 30.45 ms |

The kernel whose evicted micro is 3 to 7 % faster at 8 rows takes the same time in the graph. A plausible reason, not
verified: in the graph it runs beside the side-stream projections (z / b / a, k / v; `INTARSIA_SIDE_PROJ`, default on), so the
two share DRAM and the call is bandwidth-bound there. The micro launches one projection alone. Without ncu (no counter access)
the lane cannot tell which pipe limits it in the graph.

## 6. Gates

| gate | result |
|---|---|
| B = 16 step -3 % (`timing_35b_batched_step`) | **fail**: +0.0 % (44.47 -> 44.48 ms) |
| serve +2 % at 16 streams (3 rounds) | **fail**: +0.2 % |
| no stream count below -1 % | pass: -0.5 / +0.5 / +1.2 / +0.2 % |
| `batch_cuda_real` 35B bitwise | **pass**: 4290 pairs, 2 tests |
| `rows_exact` | **pass**: 9 tests |
| served greedy tokens identical (equal.py, 21 cells) | **pass**: 21/21, 9408 ids |
| `plan show` byte-identical 8B / 35B / Gemma-4 | **pass**: json and txt |
| single-stream decode ids unchanged | **pass**: the n = 1 cells of equal.py |
| Mac `cargo test --workspace --release` | **pass**: exit 0, 1090 passed, 0 failed |

No knob: the change is default-on and exact, like #230's FHFMA, so `docs/tuning.md` has no new row.

## 7. Process

GPU windows: **3 of 3**.

- **#6837, W1** (`--quiet gpu,cpu --mem 50`, ~17 min): rows_exact, `batch_cuda_real` 35B, served-token identity, 3-round
  serve A/B.
- **#6838, W2** (`--quiet gpu,cpu --mem 50`, ~8 min): the step micro-timing (valid). Its four nsys profiles held no decode
  steps: the copied `burst.py` read the deleted srv16 tree's prompt corpus. They are kept in `raw/w2-broken-prof/`; only the
  last one (base-b8) ran after the fix.
- **#6840, W3** (`--quiet gpu,cpu --mem 50`, ~6 min): the four profiles again (base / dev at 16 and 8 streams). fleetq held
  it until GPU work it did not track on spark1 had stopped.

Short micro jobs (< 5 min, `--quiet gpu,cpu`):

- #6814 (~3 min): rows_exact under the f16-staging variants plus the timing in section 2.
- #6821 (~2 min): rows_exact plus the timing in section 3.

CPU and no-load jobs:

- Builds: #6808, #6819, #6825 (`--load cpu`).
- `plan show`: #6831 (`--load cpu`).
- vLLM config reads: #6802 to #6806, #6833, #6834 (`--load none`).
- Mac test: #6832 (endor, `--load cpu,gpu`).

spark1-vllm stayed up throughout, no `--takes-down`. No lane process was left on spark1 (checked in #6847), and
`~/intarsiaml-step16` was removed at the end.

## 8. Next steps

1. **The step gap to vLLM is not GDN.** Both engines stream the same f32 state at the floor. What remains is kernel efficiency
   where we are below the floor:
   - routed MoE: ~85 % of 230 GB/s, 21.7 ms;
   - the fp8 projections at 16 rows: ~1.1 ms above the kernel's 2-row plateau, section 2;
   - lm_head: 1.86 vs 1.24 ms.
   An nsys of vLLM at 16 streams on a non-production server would show which kernels it shortens. That needs its own vLLM
   instance on the box.
2. **fp8 at 4 to 16 rows:** f16 staging does not pay, and the FHFMA cut does not show in the graph at 8 rows. An instruction
   cut that wins in an evicted micro need not win beside the side-stream projections; judge the next fp8 change by an in-graph
   nsys first. Left: the batch-invariant mma path with a KL check (opt-in, not exact), or a
   split that keeps the per-lane chain order and gives each SM more independent weight streams.
3. **Non-exact, not tried:** a bf16 GDN state would halve the 8.4 ms delta-rule traffic (~4 ms per step). Neither vLLM nor the
   checkpoint does that (`mamba_ssm_dtype: float32`), so it would be an accuracy trade beyond the reference. Opt-in with a KL
   check at most.
