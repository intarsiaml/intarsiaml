# mm: faster batched decode on Metal: row-exact MoE, fp8 and nvfp4 rows kernels (35B)

Lane mm, 2026-10-25, endor (Apple M5 Max, 128 GB), Qwen3.6-35B-A3B NVFP4, `serve --backend metal --max-batch 8`. Follows
[mr3](2026-10-23-mr3-metal-row-axis.md) (PR #126) and [mb](2026-10-23-mb-metal-hybrid-batching.md) (PR #124). Branch
`lane-mm/metal-moe-grouped`. Measured and gated against main 2c2d9941, then rebased on 971f1dcd (between the two, main
changed only CUDA test files and docs).

## Result

localeval (`local-model-bench`), 35B, ~1k-token prompt, 512 generated, median of 3 reps after a warm-up, `--max-batch 8`,
main 2c2d9941 and the branch back to back in one exclusive fleetq slot (#5062). Aggregate decode tok/s:

| streams | main | branch | change |
|---:|---:|---:|---:|
| 1 | 123.2 | 124.2 | +0.8% |
| 4 | 173.2 | 213.6 | +23% |
| 8 | 189.3 | 243.2 | +28% |
| 16 (cap 8, two waves) | 185.8 | 244.4 | +32% |

In process, same slot (`real_35b_batch_step_real_traffic`, ms per step, mean of 2 runs, `raw/speed/step-time.log`):

| B | main | branch | change |
|---:|---:|---:|---:|
| 1 | 8.00 | 8.00 | 0 |
| 2 | 12.28 | 9.95 | -19% |
| 4 | 18.64 | 13.74 | -26% |
| 8 | 31.44 | 21.30 | -32% (254 to 376 tok/s) |

The served gain (+28% at 8 streams) is smaller than the step gain (+48% tok/s at B = 8). The per-stream decode median at
8 streams is 37.3 tok/s on the branch (26.8 ms per token) against a 21.3 ms step; main shows the same ~5.5 ms gap (27.2
tok/s, 36.8 ms, against 31.4). That time is spent outside `forward_batch` (scheduler, sampling, streaming); not measured
further here.

Single stream is unchanged (+0.8% served, 8.00 ms in process in both arms). Every batched row stays bitwise the solo step: all gates below pass, and each new kernel has a negative control that makes
its gate fail.

Batching off is byte-identical to main (no-op below). The change only affects `--max-batch` decode, so per the lane rules it
can go default-on; **recommendation: merge default-on**, each part has its own opt-out env var.

## What changed

Three row-exact kernels for the batched step (`forward_batch`, 2..8 rows), each keeping every row's arithmetic, order and
roundings of the one-row kernel:

1. **fp8 `matmul_fp8_gemv_rows`, `_xd` form** (`kernels/metal/fp8.metal`, `fp8_gemv_{bf16,f32}_m4_xd`). The staged form
   had every threadgroup re-quantize all 8 activation rows into 32 KB of threadgroup memory (`q(x)` of 8 x K per threadgroup,
   and the 32 KB capped the threadgroups per core). Now `fp8_xprep_bf16` writes `q(x)` of all rows once into the call's
   scratch, and the GEMV reads it straight from device memory with a 4-row tile (4 measured faster than 8 or 2 at every 35B
   shape). Same expression, same fma chain per row. `INTARSIA_METAL_FP8_ROWS_XD=0` takes the staged form back.
2. **MoE `moe_experts_bf16_rows`, fat threadgroups** (`kernels/metal/moe.metal`, `moe_dec_gu_p` / `moe_dec_dn_p`). One
   threadgroup row per (token, slot) pair, a simdgroup covers 2 weight rows of the pair's expert (was 1), the activation
   block is loaded once for both. `INTARSIA_METAL_MOE_ROWS_FAT=0` takes back mb's kernels.
3. **nvfp4 `matmul_nvfp4_gemv_rows`, narrow matrices** (`crates/intarsia-backend-metal/src/nvfp4.rs`). Matrices of at
   most 4096 rows (the shared experts, 512 / 2048 rows) run the one-row shader over a grid of M threadgroup rows; the 8-row
   shader gave them only 32 / 128 threadgroups. `INTARSIA_METAL_NV_ROWS_SPLIT=0` keeps the 8-row shader.

Profiling: `INTARSIA_METAL_PROFILE=kernel` now labels every dispatch of a multi-dispatch launch (`<kernel>#n`) and gives
the fp8 / nvfp4 weight shape and the MoE routing (distinct experts, pairs). New test `real_35b_batch_step_real_traffic`
(`crates/intarsia-models/tests/batch_metal_real.rs`, ignored) and analyzer `scripts/mm/kprof.py`.

## Profile (milestone 1)

The per-kernel mode waits on every launch, which lowers the GPU clock: absolute sums are about 1.5x the real step. Read
the shares. Raw traces in `raw/k*.jsonl.gz`. The "main" column is the branch's test binary with
`INTARSIA_METAL_FP8_ROWS_XD=0` before the MoE and nvfp4 changes existed, so it ran main's kernels (`raw/k8base.jsonl.gz`);
the branch column is the final build (`raw/speed/k8-branch.jsonl.gz`).

| B = 8 share of GPU time | main | branch |
|---|---:|---:|
| MoE experts (gate-up, down, gather) | 42.1% | 36.9% |
| fp8 GEMV rows | 33.8% | 30.4% |
| nvfp4 shared-expert GEMV rows | 6.6% | 5.5% |
| nvfp4 lm_head (248320 x 2048, f32 out) | 5.9% | 9.3% |
| row-table kernels (delta rule, attention, conv, prologue) | 6.5% | 10.2% |
| rest | 5.1% | 7.7% |

- **MoE was 42%, not 24%.** The profiler labelled only the first of the MoE launch's three dispatches (gate-up), so mr3's
  24% missed the down projection and the gather ("other" in its trace).
- B = 16 (mr3's cap-16 chunking, two 8-row launches) has the same shares on main: 43.4% MoE, 34.3% fp8.
- Real traffic: 46.6 distinct experts among the 64 (token, slot) pairs of a B = 8 MoE call; 25.5 of 32 at B = 4.

## Why not "read each expert once"

The brief asked for an expert-grouped kernel, so each expert's weights are read once per step. mb's `_x` kernels already
do that: a leader pair serves up to 8 pairs of its expert (`moe_lead`, `MOE_XR = 8`). Ablation of the same fat kernel
(gate-up 2, down 4 rows per simdgroup) with 8 / 4 / 2 / 1 pairs per leader (`raw/logs/micro2.log`, `timing_rows`, T = 8,
us per call, mean of 2 reps):

| pairs per threadgroup row | random | identical routing | overlap |
|---|---:|---:|---:|
| mb `_x` (1 row / simdgroup, 8 pairs) | 357 | 235 | 308 |
| fat, 8 pairs | 243 | 216 | 234 |
| fat, 4 pairs | 215 | 194 | 200 |
| fat, 2 pairs | 207 | 181 | 190 |
| fat, 1 pair | 212 | 157 | 168 |
| fat, 1 pair, 2 / 2 rows (shipped, `micro3.log`) | 199 | 161 | 167 |

With random routing 1, 2 and 4 pairs are within the spread; with shared experts 1 pair wins clearly. Grouping saves little
DRAM traffic here (only 27% of pairs share an expert, and the second read of a block comes from the
cache when the pairs run at about the same time), and it serializes the pairs of an expert in one threadgroup. In the
real step, 1 pair per threadgroup row was 0.6 ms faster at B = 8 than 8 pairs (23.8 vs 24.4 ms, `raw/logs/step2.log`).
The win came from fatter threadgroups (2 rows per simdgroup: fewer threadgroups and leader scans, more loads in flight),
not from grouping. 2 rows beat 1 / 4 / 8 rows for both gate-up and down (`raw/logs/micro3.log`).

## Next levers (milestone 3)

- **16-row GEMV tile: not built, deprioritized.** With the `_xd` form the tile is no longer capped by threadgroup memory,
  but the sweep goes the other way: a 4-row tile beat the 8-row one at every shape (fp8 8192 x 2048 at M = 8: 64 vs 75 us,
  `raw/logs/micro3.log`). Cap 16 by chunking on the branch kernels: B = 16 39.5 ms/step, 405 tok/s vs 379 at B = 8 (+7%;
  main: +3% per mr3, and 61.9 ms vs 31.3 here, no gain). The cap stays 8; raising it is a separate decision.
- **Single-stream command-buffer overhead: not attempted** in this lane; B = 1 unchanged.
- The branch's B = 8 profile is the map for the next lane: MoE 37%, fp8 GEMV rows 30% (the 8192 x 2048 gate/up of the
  shared expert and the 2048 x 4096 output projections are the largest), row-table kernels 10%, lm_head 9% (nvfp4
  248320 x 2048 with f32 out, still the 8-row shader). Outside the GPU, the ~5.5 ms per step that `serve` spends
  outside `forward_batch` at 8 streams is now about a fifth of the served step.

## Gates

G-C, the long prompts, `batch_recurrent`, dense 8B and `kernels test` ran in job #5048 at commit 94e2ca06; G-B and the
negative controls ran again in #5068 at the head commit. Between the two only the negative-control (`NEG = true`) forms of
the MoE kernels changed; the default (`NEG = false`) code is the same, and so are the no-op and speed binaries (01f96dec).

| Gate | Result | Log |
|---|---|---|
| G-B `tests/rows_exact.rs` (Metal) | pass, 7 of 7: fp8 1,344 cases / 6,720 rows (staged and `_xd`, unaligned scratch), nvfp4 168 / 840, MoE 168 / 840 (random, identical, overlapping routing), bf16 pair, row-parallel | `raw/gates2/gb-rows_exact.log` |
| G-B `tests/row_tables.rs` (Metal) | pass, 8 of 8 | `raw/gates2/gb-row_tables.log` |
| New negative controls, data level (`negative_controls_differ_from_the_one_row_calls`) | each differs from the one-row calls: fp8 rows swapped 2,040 / 4,096 bytes, q(x) skipped 2,542 / 4,096; MoE routing of two tokens swapped 7,444 / 32,768, grouped prefill path at T = 8 4 / 32,768 | `raw/gates2/gb-rows_exact.log` |
| New negative controls, kernel level (`INTARSIA_METAL_NEGCTL`) | each makes its gate fail: `fp8_xd` (other association on odd blocks) at byte 29,368; `moe_gu` (split scale fma) at 16,594; `moe_dn` (reversed in-block chain) at 58,041 | `raw/gates2/neg-*.log` |
| `intarsia kernels test --backend metal` | exit 0, 44 kernels, 817 cases | `raw/gates/kernels-test.log` |
| G-C 35B: `forward_batch` N = 2..8 == solo step; BatchedEngine == LocalEngine | pass: 4,550 (row, step) pairs bitwise identical; engine test pass | `raw/gates/gc-35b.log` |
| G-C long prompts 2.6k / 4k / 16k | pass | `raw/gates/gc-long.log` |
| `batch_recurrent` (Metal) | pass, 10 of 10 | `raw/gates/batch_recurrent.log` |
| Dense Qwen3-8B batched bitwise (`batch_metal`) | pass: 138 (row, step) pairs, 48 decode rows beside 256-token prefill chunks | `raw/gates/dense-8b.log` |
| Metal batching-off no-op vs main (8B and 35B: `plan show` text and `--json`, 256 greedy ids and output text after an 11-token and a 4.6k-token prompt) | 12 of 12 byte-identical | `raw/noop/` |
| Workspace tests | 879 passed, 3 failed. 2 are `intarsia-backend-cuda/tests/rows_registry.rs`, which fail on main 2c2d9941 too (the fp8 picks after the cuBLASLt default; PR #128 re-pinned them and was merged during
the lane, so they pass on the rebased branch). 1 was `docs_env_vars` (a tuning row cited two files), fixed, passes on the branch | `raw/gates/workspace.log` |

## Not run, and why

- CUDA no-op (spark1 / spark2) and HIP no-op (tatooine): the branch touches only `intarsia-backend-metal`, Metal shaders,
  one Metal-only test file, `docs/tuning.md` and scripts. No shared runtime crate changed, so per the lane rules they are
  not required.

## Raw

`raw/gates/`, `raw/gates2/` (gate logs), `raw/noop/` (plan show and ids, both arms), `raw/speed/` (localeval and step time),
`raw/logs/` (A/B and micro benchmark logs), `raw/k*.jsonl.gz` (per-kernel traces). Scripts: `scripts/mm/`.
