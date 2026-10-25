# mm step 2: fp8 multi-row GEMV on simdgroup matrices, row-exact (Metal, 35B batched decode)

Lane mm, 2026-10-25, endor (Apple M5 Max, 128 GB), Qwen3.6-35B-A3B NVFP4, `serve --backend metal --max-batch 8`. Follows
[mm step 1](2026-10-25-mm-metal-moe-grouped.md) (PR #136, merged 5b7205ca). Branch `lane-mm/metal-fp8-mmt` on main
0b84f499, draft PR #141. Experiment history (losers included) on branch `lane-mm/metal-step2`.

## Result

localeval (`local-model-bench`), 35B, ~1k-token prompt, 512 generated, `--max-batch 8`, main 0b84f499 and the branch
interleaved (main, branch, x 3) in one exclusive fleetq slot (#5157), median of the 3 reps. Aggregate decode tok/s (1
stream: the sweep's decode median):

| streams | main | branch | change |
|---:|---:|---:|---:|
| 1 | 123.2 | 123.5 | +0.2% |
| 4 | 216.3 | 216.3 | 0 (4 rows: the new kernel is not used) |
| 8 | 259.7 | 267.7 | **+3.1%** |
| 16 (cap 8, two waves) | 255.0 | 266.4 | **+4.5%** |

Per-rep values (`raw/speed/localeval-*-r{2,3,4}.log`): main 8 streams 264.5 / 259.7 / 253.3, branch 263.5 / 279.2 /
267.7; main 16 streams 258.6 / 252.2 / 255.0, branch 260.4 / 266.4 / 268.2. A first pair in the earlier slot #5156
(`raw/speed/localeval-{main,branch}.log`) ran slower for both arms (main 231.0 / branch 235.1 at 8 streams), probably
other load on endor; it is kept but not in the medians (all 4 reps: +3.5% at 8, +3.9% at 16).

The served gain is smaller than the step gain (-7.6% step time at B = 8), as in step 1: the served 8-stream wave spends
part of its time below 8 rows (arrivals, prefill chunks of 2048 that stall decoding rows) and outside the GPU step.

In process (`real_35b_batch_step_real_traffic`, ms per step, main 0b84f499 and branch A B A B in slot #5156,
`raw/speed/step-time.log`):

| B | main | branch | change |
|---:|---:|---:|---:|
| 1 | 8.13 | 8.23 | +1.2% (kernel not used; run noise) |
| 2 | 10.34 | 10.05 | -2.8% (kernel not used; main rep 1 slow) |
| 4 | 14.01 | 13.96 | 0 (kernel not used) |
| 8 | 21.60 | 19.95 | **-7.6%** (370 to 401 tok/s) |

Per-kernel B = 8 profile of the branch (`raw/speed/k8-branch.txt`, GPU sum, 804 launches per step):
`matmul_fp8_gemv_rows` 6.82 ms (25.2%), MoE gate+up 6.53 ms (24.1%), nvfp4 rows 4.41 ms (16.3%), MoE down 3.82 ms
(14.1%), GDN delta rule 1.52 ms, attention 1.17 ms. (Step 1's profile had fp8 rows at about 30%.)

Every batched row stays bitwise the solo step (gates below); the new kernel has its own negative control that makes the
fp8 rows gate fail. Batching off is byte-identical to main (no-op below): `matmul_fp8_gemv_rows` is selected only for
`rows_exact` calls of the batched step.

## What changed

`matmul_fp8_gemv_rows` (the fp8 projections of the batched step: attention q/k/v/o, GDN in/out projections, about 30%
of the B = 8 step after step 1) at **5..=8 rows** now runs on 8x8 float simdgroup matrices (`fp8_rows_mmt_{bf16,f32}`,
`kernels/metal/fp8.metal`), after a pre-pass `fp8_xprep_t` that writes q(x) of all rows block-transposed (`xt[u][k][m]`,
bf16, zero-padded to 8 rows) into the call's scratch.

Why it is row-exact. Two facts measured on endor (`raw/probe/probe*.swift`, 262144 and 4 x 131072 elements):

1. `simdgroup_multiply_accumulate` on float 8x8 equals, bit for bit, the per-element chain `c = fma(a[i][k], b[k][j], c)`
   for k = 0..7 from c: all elements equal on wide-range, subnormal-producing, inf/NaN/-0 and cancellation data (NaN
   positions identical). Reversed k order or `c + chain(0)` do not match (141813 / 191569 of 262144).
2. `simd_sum` equals the xor butterfly 1, 2, 4, 8, 16 (4096 / 4096), i.e. the pairwise tree over lanes 0..31 in index order
   (the 16..1 butterfly matches 1581 / 4096, a sequential sum 1275).

The one-row kernel's value for (row m, output n) is: lane l sums the 16-term fma chains of blocks u = l, l + 32, ... in
that order (`fp8_dot16`, NaN when the block holds a NaN code), then `simd_sum`, then the scale. The new kernel computes
block u's chain for 8 outputs x 8 rows as two MMAs from 0 (k = 0..7, then 8..15; fma commutes in its multiplicands, so
`W x q(x)^T` gives the same chain per element), sets a block whose weight row holds a NaN code to NaN, and adds it to the
accumulator of class u mod 32. The 8 simdgroups of a threadgroup take classes 4s..4s+3 and fold them ((0+1)+(2+3)), the
threadgroup folds the 8 results ((0+1)+(2+3))+((4+5)+(6+7)): the same pairwise tree. Same scale expression and rounding.

Operand layout (measured): a lane's two elements of an 8x8 matrix are row `(lane & 16 ? 4 : 0) + ((lane >> 1) & 3)`,
columns `((lane >> 3) & 1) * 4 + (lane & 1) * 2` and `+ 1`. With weights as the left operand a lane's weight bytes are two
adjacent bytes of one row (one ushort load), its activations two adjacent rows of one k (one uint load).

Selection: the kernel is padded to 8 rows, so it costs the same at M = 2 as at 8, while `_xd` adds a 4-row tile at M = 5.
It takes M >= 5. `INTARSIA_METAL_FP8_ROWS_MMT=0` turns it off, `=N` (2..=8) moves the threshold (the gate runs it at 2).
`INTARSIA_METAL_NEGCTL=fp8_mmt` swaps the two MMAs of a block (k = 8..15 first): the gate must fail.

Scratch of `matmul_fp8_gemv_rows`: `max(M, 8) * K * 2 + 16` bytes (was `M * K * 2 + 16`).

Micro timings, `timing_rows` (`raw/micro-gate.log`, us per launch, 200 launches):

| N x K | `_xd` M = 5 | new M = 5 | `_xd` M = 8 | new M = 8 | change at 8 |
|---|---:|---:|---:|---:|---:|
| 8192 x 2048 | 50.8 | 52.6 | 62.3 | 52.5 | -16% |
| 4096 x 2048 | 34.1 | 28.8 | 34.5 | 28.9 | -16% |
| 512 x 2048 | 10.4 | 8.9 | 10.5 | 8.9 | -15% |
| 2048 x 4096 | 36.4 | 32.2 | 36.1 | 32.2 | -11% |

For scale, the one-row kernel at M = 1 reads 8192 x 2048 in 27 us (about 620 GB/s), so the new kernel is still about 2x
the weight-streaming floor at 8 rows.

## Tried and dropped (branch `lane-mm/metal-step2`)

All gate-clean where they ran; none beat what is kept.

| idea | result (8192 x 2048, M = 8, us) |
|---|---|
| `_xd` tile / unroll sweep (`MB, RS, UN`: 4,2,0,2 / 4,2,1,4 / 8,2,1,* / 8,1,1,*) | 56.7 to 67.9 vs 62.3, no stable gain |
| `_xd` with f32 q(x) rows, swapped grid | 57.8 to 60.8, within noise of `_xd` |
| word-interleaved f32 multi-row GEMV ("v2") | 91.9 |
| simdgroup MMA with x as the left operand (byte weight loads) | 60.2 (-3%) |
| same, 2 / 4 output tiles per simdgroup | 69.4 / 177.0 |
| transposed MMA (kept) with q(x) as e4m3 bytes (W8A8) | 58.1 vs 52.8 bf16: decode ALU costs more than the traffic saved |

Ablations of the x-as-left-operand MMA kernel (timing only): no MMA 56.0, no NaN shuffles 60.3, no activation loads 47.9:
activation traffic and single-byte weight loads were the cost, which the transposed layout addresses.

## Gates

All on endor, branch a5663990, one exclusive slot #5124 (`scripts/mm/gates.sh`, logs in `raw/gates/`):

| gate | result |
|---|---|
| `tests/rows_exact.rs` (default: new form at M >= 5) | pass |
| fp8 rows gate with the new form at every M (`INTARSIA_METAL_FP8_ROWS_MMT=2`) and off (`=0`) | pass / pass |
| `tests/row_tables.rs` | pass |
| negative control `NEGCTL=fp8_mmt` (new form, MMAs of a block swapped) | gate fails as required: M = 5, first differing byte 9028 of 81944 |
| negative controls `fp8_xd` / `moe_gu` / `moe_dn` | gate fails as required (bytes 29368 / 16594 / 58041) |
| `kernels test --backend metal` | pass, 44 kernels, 817 cases |
| G-C 35B batched == LocalEngine | pass, 4550 (row, step) pairs bitwise identical |
| G-C long prompts 2.6k / 4k / 16k | pass |
| `batch_recurrent` (metal) | pass, 11 tests |
| dense Qwen3-8B batched | pass, 138 pairs + 48 rows beside prefill chunks bitwise |
| workspace | 884 passed, 1 failed: `docs_env_vars` (the new env var had no `docs/tuning.md` row); row added in 111699cf, test passes |

Gate strength checked before trusting it: with the NaN-flag shuffles removed, or with constant activations, the fp8 rows
gate fails (NaN weights, M = 2..8 and odd N 130 / 33 / 65 are in its cases). The gate's NaN weight cases and the MMA
probe's specials cover the NaN path. CUDA / HIP no-ops not run: no shared crate changed (only
`crates/intarsia-backend-metal`, `kernels/metal`, tests, docs, scripts).

## No-op (batching off) vs main 0b84f499

Slot #5156, `scripts/mm/noop.sh` (`raw/noop/`): `plan show` text and `--json`, 256 greedy ids and the generated text
after a short and a ~4k-token prompt, Qwen3-8B and the 35B: **12 of 12 IDENTICAL**.

## Not done / next

- 16-row tile: the kernel is written for R = 2 (two activation tiles per weight tile, weights read once), but Metal's
  batched step caps at 8 rows (`BATCH_MAX_ROWS`); a 16-row cap needs the nvfp4, MoE and attention rows kernels too.
- Single-stream command-buffer overhead: the step-1 serve profile put host gaps at about 1% of decode; not a lever.
- The kernel is about 2x the weight-streaming floor at 8 rows; the remaining cost was not isolated further (no GPU
  limiter counters from the command line).
