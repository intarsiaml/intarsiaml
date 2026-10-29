# gdnx: GDN decode delta kernel and the row-exact fp8 pair, both bitwise; B = 16 step -1.3 %

Lane gdnx, 2026-10-29, branch `lane-gdnx/gdn-fuse` from main 5c39315f6. spark1 (spark-4e77, GB10 sm_121),
`nvidia/Qwen3.6-35B-A3B-NVFP4`. Targets from VPROF (docs/results/2026-10-29-vprof-vllm-step.md): the GDN delta kernel
(vLLM 250 vs our 279 us per layer in nsys) and merged fp8 projections. **Status: draft PR** (the B = 16 step gate of -3 %
is not met; see Gates).

## Result

| change | default | what | exact |
|---|---|---|---|
| `gdn_delta_dec_kernel` (`kernels/cuda/gdn.cu`) | on (`INTARSIA_GDN_DEC=0` off) | T = 1 delta rule, dk = 128, solo and row-table launch | bitwise the split / rows kernels |
| `matmul_fp8_pair_gemv_rows` (`kernels/cuda/fp8.cu`) | on (`INTARSIA_CUDA_FP8_PAIR_ROWS=0` off) | row-exact `Op::MatmulFp8Pair`, 1..16 rows: GDN qkv + z and attention k + v of the batched step in one launch each | bitwise the two single row-exact calls and the M = 1 pair |

`timing_35b_batched_step` (dev binary, env A/B in one window, #6886, two interleaved rounds, the second in reverse
order; the middle of the test's three 60-step runs, ms):

| arm | B = 16 r1 | r2 | B = 8 r1 | r2 |
|---|---:|---:|---:|---:|
| both off (= main) | 44.49 | 44.48 | 27.79 | 27.80 |
| **both on (default)** | **43.91** | **43.95** | **27.55** | **27.64** |
| delta kernel only | 44.16 | 44.16 | 27.62 | 27.58 |
| pair only | 44.12 | 44.12 | 27.66 | 27.86 |
| both on, `INTARSIA_SIDE_PROJ=0` | 44.06 | 44.10 | 27.84 | 27.96 |
| delta kernel only, `INTARSIA_SIDE_PROJ=0` | 44.38 | 44.54 | 28.04 | 27.99 |

B = 16: -0.56 ms (-1.3 %); each change alone gives about -0.35 ms. B = 8: -0.2 ms (-0.7 %). The side stream stays on:
with the pair, z no longer runs beside qkv, but b / a still do, and turning the side stream off is not faster
(raw: `raw/steptime-w1.txt`, `raw/w1.log`).

## Per kernel at B = 16 (nsys, VPROF's units)

Window #6909 (06:37-06:41): `scripts/gdnx/prof.sh` (step16's: `serve --max-batch 16`, a 16-stream burst of ~1k-token prompts,
256 tokens, CUDA graphs with `--cuda-graph-trace=node`), base and dev one burst each (`raw/base-b16`, `raw/dev-b16`).

| kernel(s) per layer | base | dev | per step |
|---|---:|---:|---:|
| GDN delta rule | `gdn_delta_split_kernel` 279.3 us | `gdn_delta_dec_kernel` 272.8 us (-2.3 %) | -0.20 ms |
| GDN qkv + z (8192 + 4096 x 2048) | two `rows_q` calls, ~142 us | one `rows_q_pair` 129.4 us (-9 %) | -0.38 ms |
| attention k + v (512 + 512 x 2048) | two calls, 2 x 18.4 us | one pair 27.6 us (-25 %) | -0.09 ms |
| GDN b / a pair (bf16, side stream) | 3.2 us | 8.9 us | +0.34 ms of side-stream time |
| GDN conv | 17.2 us | 17.7 us | +0.02 ms |

(qkv + z for base: the 96-block `rows_q` bucket is 30 qkv + 30 z + 10 attention q calls, 5.17 ms; attention q alone is
89.8 us in dev, so 30 x (qkv + z) = 4.27 ms.) The b / a GEMV runs on the side stream. It used to overlap qkv while z
ran beside it; now it overlaps the longer pair kernel and is slowed by sharing the SMs, mostly hidden. The kernel sum is
44.52 -> 44.29 ms. One burst per arm is too noisy for the wall (the passes differ: 252 vs 245); the step timing and
the serve A/B above are the speed measures. The delta kernel gains less in the graph (-2.3 %) than in the evicted micro
(-4 %).

## 1. GDN delta rule at decode

**Floor probe first.** `scripts/gdnx/bwprobe.cu`: an in-place read-scale-write of 30 x 32 MB f32 (the B = 16 state
footprint), one warp per block, event-timed:

| float4 per lane (bytes per warp) | 1 (512) | 2 (1 KB) | 4 (2 KB) | 8 (4 KB) | 16 (8 KB) | 32 (16 KB) |
|---|---:|---:|---:|---:|---:|---:|
| us per 32 MB | 269.3 | 274.4 | 283.7 | 290.2 | 291.9 | 298.2 |

Cache hints (`__ldcs/__stcs`, `__ldg`, `L2::256B`, an `evict_first` policy) change nothing (< 0.5 %). Block size
(32 / 128 / 256 threads) and in- vs out-of-place change nothing either. What matters is how much each warp reads before
it writes back: smaller is faster. The split kernel moves 8 KB per warp through a shared-memory tile, and it sits at
that 8 KB floor (micro 296 us vs probe 292). So "at the bandwidth floor" (step16) holds for its access pattern, but a
smaller per-warp footprint goes lower.

**The kernel.** The split kernel needs lane l of a row to own contiguous groups, because it hands the left-to-right
sum from lane to lane. The new kernel gathers the sums through shared memory instead, so the lane-to-group mapping is
free. A warp owns RPW rows (default 1). Lane L holds float4 group L of each row, so the state is read and written as
whole 512-byte rows straight into registers, with no tile. q and k are the same for every row of a head. Each lane
normalises its own group as `raw * norm`, after the split kernel's lane-strided sums of squares and warp tree, so the
values are the same. Each group's dot product uses the same FMUL / FFMA chain and goes to `gs[row][group]`. Lane r then
adds row r's 32 values left to right from 0, the FADD chain the split kernel spreads over lanes. Then come delta and
the update, and the output the same way. The arithmetic is written with the same `__f*_rn` intrinsics, so the bits
cannot depend on contraction.

Evicted micro (`tests/row_tables.rs` `timing_gdn_rows`, 35B shape, 30 layers' own buffers, best of 15, us per call):

| rows | split (DEC=0) | dec RPW 1, 1 warp/block | dec RPW 1, 2 warps/block | RPW 2 | RPW 4 | RPW 8 | RPW 16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23.3 | 20.8 | 21.0 | 21.0 | 21.4 | 22.0 | 21.9 |
| 4 | 79.5 | 71.6 | 73.4 | 73.3 | 75.3 | 75.3 | 76.3 |
| 8 | 152.5 | 142.5 | 143.6 | 144.5 | 148.0 | 149.1 | 149.4 |
| 16 | 296 | 280-294 | 283.1 / 283.3 | 285 | 292.7 | 296 | 296.4 |

The default is one row per warp, with 2 warps per block from 12 rows (WPB 1 swung 280 to 294 at 16 rows; WPB 2 held
283) and 1 below. That is -4 % at 16 rows and -6 to -10 % at 1..8 rows. The other variants tried were slower or equal:
LPR 4 / 8 for the split kernel (-1 to -3 %, bits unchanged); direct uncoalesced loads into the split kernel's layout
(+18 %); and four channels per thread for the conv rows kernel (+11 % at 16 rows, and not bitwise as first written).
All were dropped.

**Not closed.** 283 us is about 5 % above the 512-byte probe (269), which does no math. The rest of the per-block
latency (the row-table pointer, q / k loads, the two serial 32-add chains) keeps it from the floor. VPROF's 250 us for
vLLM is in nsys units: our split kernel measured 279 there vs 296 here, so vLLM is near the probe floor in those units.
**conv** (vLLM 7.6 vs our 18.3 us) is bytes, not kernel quality: vLLM's conv state is bf16, ours f32. Every value our
conv state holds is a past bf16 input (`ik_b2f(x)`) or zero, so a bf16 conv state would be bitwise. But it changes the
recurrent-buffer layout (`byte_off`, prefill conv, snapshots, the other backends), so it is out of this lane: a
follow-up of about -0.3 ms per step.

## 2. Row-exact fp8 pair

A literal concatenation of the weight rows is not exact. Every rows kernel applies one weight scale and one input scale
per call (`sc[0]`, `sc[1]`), and qkv and z have their own scales. A concatenated output buffer would also fail the dense
check of the consumers. So the pair is two matrices in one launch. The e4m3-staged rows kernel's body became
`rows_q_seg` (unchanged arithmetic; `rows_q_tiles` takes a `base` tile offset). `fp8_gemv_rows_q_pair_kernel` runs it
for matrix 0 from tile 0 and for matrix 1 from tile `tiles0`, so the warps of a persistent grid stride over both. x is
staged once when the input scales are equal (they are in the checkpoint) and again when they differ. W8A16 halves work
too, by the staged-row-group path. The launcher takes the pair kernel exactly where both single calls would take the
e4m3-staged kernel (M >= 12, or both matrices <= 1024 outputs at M >= 4, with the same shapes: 2x8x2, or 1x16 for
k + v). Elsewhere it makes the two single calls, so 2..11 rows of qkv + z keep today's kernels.

`matmul_fp8_pair_gemv_rows` is registered `rows_exact` for M 1..=16. `try_fused(Op::MatmulFp8Pair)` in
`gdn_batch.rs` and `kv_pair` in attention then fire in the batched step with no planner change. The ordinary
selection is unchanged (`rows_registry.rs` ORDINARY pins). The exact-rows pin for `matmul_fp8_pair` moves from
`1..=1 matmul_fp8_pair_gemv; 2..=4096 -` to `1..=16 matmul_fp8_pair_gemv_rows`, which is intentional.

A q + k + v triple would need a new op (attention q is 8192 rows with its own gate split). It was not attempted, since
the pair gave about -0.35 ms.

## 3. Routed down projection (target 3)

Not attempted: the windows went to measuring targets 1 and 2 and to the serve A/B.

## Exactness

- `tests/row_tables.rs` (`gdn_delta_rows_are_the_solo_delta_rule`, `gdn_conv_rows_are_the_solo_conv`) and
  `tests/gdn_split.rs` (split vs the one-thread rows kernel, T = 1 .. 2048) pass under DEC 1 / 2 / 4 / 16, WPB 1 / 2 / 4,
  and DEC = 0.
- `tests/rows_exact.rs`: 10 pass, including the new `fp8_pair_rows_are_bitwise_the_one_row_gemvs` (5 shape pairs x 5
  scale combinations x ROWS_Q default / all x M 1..16, against the one-row GEMV per matrix and the M = 1 pair) and the
  existing fp8 rows-q suite over the refactored kernel.
- `batch_cuda_real` 35B: 4290 (row, step) pairs bitwise, defaults on (#6886, `raw/batch-real.txt`).
- `plan show` (#6892, `raw/plan.txt`): 8B and Gemma-4 byte-identical. The 35B differs by one candidate row,
  `matmul_fp8_pair matmul_fp8_pair_gemv_rows deterministic no 0 0` (0 calls per decode step: the solo step's ops are
  unchanged; the new kernel only serves exact-rows calls).
- Served greedy / sampled identity and the serve A/B: section W2 below.

## W2: served identity and serve A/B

Window #6897 (06:18-06:31). base = main 5c39315f6, dev = this branch, both `serve --max-batch 16`, default env.

- **Served identity** (`scripts/gdnx/ident.sh`, `scripts/htk/equal.py`: 7 sampling settings x 1 / 4 / 16 streams, seeds
  100 + i, 64 tokens each): **21/21 cells identical, 9408 token ids**. The greedy 1-stream cell is the single-stream decode
  check (`raw/ident/compare.log`).
- **Serve A/B** (`scripts/gdnx/le-window.sh`, localeval speed, ~1k-token prompts, 256 tokens, 3 interleaved rounds,
  aggregate decode tok/s; raw `raw/w2/`, run ids `raw/w2/index.txt`):

| streams | base r1 / r2 / r3 | dev r1 / r2 / r3 | base mean | dev mean | delta |
|---:|---|---|---:|---:|---:|
| 1 | 79.1 / 81.1 / 81.1 | 81.0 / 81.0 / 81.5 | 80.4 | 81.2 | +0.9 % |
| 4 | 170.3 / 173.4 / 177.8 | 174.7 / 175.4 / 173.8 | 173.8 | 174.6 | +0.5 % |
| 8 | 226.9 / 227.5 / 228.6 | 229.9 / 230.7 / 236.8 | 227.7 | 232.5 | +2.1 % |
| 16 | 287.0 / 292.7 / 293.0 | 295.4 / 292.0 / 293.6 | 290.9 | 293.7 | +1.0 % |

## Gates

- B = 16 step -3 % or better: **not met** (-1.3 %).
- Serve +2 % at 16 streams: **not met** (+1.0 %).
- Nothing below -1 % at 1 / 4 / 8 streams: met (+0.9 / +0.5 / +2.1 %).
- Exactness for default-on: all met (batch_cuda_real, rows_exact, 21/21 served cells, single-stream greedy ids, plan show
  as explained above).

So the PR is a **draft**: exact and positive at every stream count, but under the speed gates.

## Process

GPU jobs on spark1 (all `--quiet gpu,cpu` except where noted): micro jobs under 5 min: #6867, #6869, #6870, #6873,
#6875, #6877, #6879, #6881 (rows_exact), plus #6872 (probe compile error, no GPU work). Windows: #6886 (W1:
batch_cuda_real + 12 step-timing runs, 06:00-06:11), #6897 (W2: identity + 3-round serve A/B, 06:18-06:31) and #6905 +
#6909 (W3: nsys base vs dev; #6905 profiled nothing in its 2 minutes because the copied burst.py read step16's prompt
path, retried at once as #6909, 4 minutes): three windows, about 32 minutes of GPU in all. Builds (#6866, #6868, #6874, #6876, #6878, #6880, #6882, #6883) and plan show (#6892) ran
`--load cpu`; #6871 (`--load none`) read vLLM's `fused_recurrent.py` from the running container. spark1-vllm was not
touched.

## Follow-ups

- bf16 conv state: exact (the state only holds bf16 inputs), about -0.3 ms per step at B = 16, but a layout change
  across the recurrent buffer, prefill and the other backends.
- The decode delta kernel is about 5 % above the 512-byte read-write probe: the per-block latency (row-table pointer,
  q / k loads, the serial row sums) is what is left.
- An attention q + k + v triple (a new op), and the routed down projection (VPROF target 3), were not attempted.
