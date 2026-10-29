# moedn: the 35B's routed-expert down projection at B = 16: one read per distinct expert, -36 us per layer, bitwise

Lane moedn, 2026-10-29, branch `lane-moedn/down-proj` (base main efb8b4b3b, #246 included; all step and serve numbers
below are on that base). spark1 (spark-4e77, GB10 sm_121), `nvidia/Qwen3.6-35B-A3B-NVFP4`. **Status: PR. Not merged.**

## Result

The multi-token (rows) MoE decode launchers, which serve every batched decode step from 1 to 16 rows, now run their down
projection as `moe_dec_down_dd_kernel` (`kernels/cuda/moe_decode.cu`). A block owns 16 output rows of all T tokens. It
groups the T x 8 (token, slot) pairs by distinct expert and reads each distinct expert's rows once. The per-(pair, row)
arithmetic and the slot-order sum are v2's statements, so outputs are bitwise main's. `INTARSIA_MOE_DOWN_DD=0` restores the
v2 kernel.

| measure | main (v2 down) | dev (dd down) | change |
|---|---:|---:|---:|
| down projection, nsys serve burst, B = 16 decode window (us / layer) | 202.4 | 166.1 | **-36.3** |
| down projection, nsys serve burst, whole burst average (us / call) | 189.0 | 161.7 | -27.3 |
| down projection, cold micro T = 16, Zipf routing D = 59.5 (us / call) | 187.5 | 157.4 (RT 8) / ~159 (RT 16) | -28 to -30 |
| `timing_35b_batched_step` B = 16 (ms, mean of 3 runs, 2 interleaved rounds) | 43.82 / 43.81 | 42.55 / 42.59 | **-1.24 ms (-2.8 %)** |
| `timing_35b_batched_step` B = 8 (ms, same) | 27.61 / 27.81 | 27.12 / 27.17 | -0.56 ms (-2.0 %) |
| 16-stream burst wall under nsys (s) | 14.41 | 13.83 | -4.0 % |

Serve A/B (localeval, 1 / 4 / 8 / 16 streams, 3 interleaved rounds): section 4.

Gates: `batch_cuda_real` 35B 4290 (row, step) pairs bitwise. rows_exact (incl. `moe_decode_rows_are_bitwise_the_one_token_kernel`:
the dd kernel at T = 1..16 against the v2 T = 1 kernel, with distinct, same-expert, shared-pair, invalid-id and wide-weight
routings, the 35B shape, a top-4 shape and the gelu twin with I / 16 = 8). row_tables, quantmoe and moe_nvfp4 pass.
`plan show` is byte-identical for the 8B, 35B and Gemma-4 (json and text). Served `equal.py`: section 4. Mac
`cargo test --workspace --release` is green.

## 1. Why the down projection was slow

The v2 down kernel on the rows path (`moe_dec_down_v2_kernel<4, true>`; 64 registers, 1.3 KB shared memory,
`raw/resource-usage.txt`) launches grid (T, H / 4) = 16 x 512 = 8192 blocks of 256 threads. Warp w of a block owns slot w of
one token and 4 output rows. With I / 16 = 32 equal to the warp width, each lane does one u iteration: a block is one
~8 KB burst with no loop, then a barrier and an 8-term sum. Every (token, slot) pair reads and decodes its expert's rows
itself: 128 pairs on ~61 distinct experts at T = 16 (moe16 measured 61.45 in serve), so about 2.1 reads per distinct
expert. The token-fastest grid lets the repeats hit L2. The weight loads are `__ldcs` (evict-first), though, and each repeat
costs a full decode and a dependent chain per pair.

The per-T micro (`bench_moe_rows_cold`, 256 experts, Zipf 1.15 routing that gives serve's D, `raw/micro3*.txt`) shows the cost
is per pair, not per byte. gate_up stays at ~230 GB/s for every T. v2 down falls from 212 GB/s at T = 4 (1.39 pairs per
distinct expert) to 209 at T = 8 (1.68) and 187 at T = 16 (2.15). gate_up gets its x from shared memory (staged once per block),
with 4 unrolled u iterations per lane (H / 16 = 128).

vLLM's `marlin_moe_wna16` (VPROF raw dump: 144 blocks x 128 threads, `moe_align_block_size` sorts pairs by expert) reads each
distinct expert once: 163 us, 215 GB/s.

## 2. Change

`moe_dec_down_dd_kernel<R, G>`, used by the rows launchers (`ik_moe_experts_decode_v2_rows_bf16` and its gelu twin) unless
`INTARSIA_MOE_DOWN_DD=0`. The solo (T <= 1) v2 launchers and the original kernel are unchanged. Kernel names, registry entries
and scratch sizes stay the same, so `plan show` does not change.

- Grid H / RT blocks (RT = R x G rows, default 16: 128 blocks; `__launch_bounds__(256, 3)` keeps all of them resident at
  80 registers). Shared memory is P x RT f32 terms plus 6P + 1 ints (P = T x K <= 256): at most 38 KB.
- Prologue: the block validates the ids (`0 <= v < E`, else the slot's terms are 0 as in v2) and counting-sorts the pairs by
  distinct expert (first pair, rank, a warp prefix sum of the per-expert counts).
- Work: (expert, R-row group) items, handed out by a shared counter so warps that draw a hot expert (10+ pairs under skewed
  routing) do not hold the others up. A warp loads the item's R rows and scales into registers once. The next item's rows and
  its first pair's h chunk load while the current one computes, and the next pair's h chunk loads while the current pair
  computes.
- Per (pair, row): v2's statements unchanged: the lane-strided u loop (kept general for I / 16 > 32), the 16-term `pdot`
  fmaf chain, `acc = fmaf(pdot, sv, acc)` from 0, `ik_warp_sum`, `round_bf16(round_bf16(s * g) * w)` into the slot's term.
  After a barrier, one thread per (token, row) sums slots 0..K-1 from +0. Neither the item order nor which warp runs an item
  can change a bit.

First version (items round-robin over warps, a linked list of pairs, one-item weight prefetch, no h prefetch): 177 us at
T = 16, only -10. Pair-level h prefetch, the counting sort, dynamic items and the register cap took it to 157-159.

| down kernel, cold micro (nsys, us / call) | T = 1 | T = 4 | T = 8 | T = 16 |
|---|---:|---:|---:|---:|
| v2 (main) | 23.4 | 64.5 | 107.7 | 187.5 |
| dd RT = 8, R = 4 | 25.3 | 61.2 | 99.2 | 157.4 |

| full op (gate_up + down), cold micro, us / call | T = 1 | T = 4 | T = 8 | T = 16 |
|---|---:|---:|---:|---:|
| v2 (main) | 67.1 | 184.1 | 303.0 | 491.7 |
| dd RT = 8, R = 4 | 68.1 | 180.6 | 294.6 | 462.7 |
| **dd RT = 16, R = 4 (default)** | **65.5** | **178.8** | **294.0** | **463.6** |
| dd RT = 16, R = 2 | 65.4 | 179.9 | 298.9 | 474.5 |
| dd RT = 32, R = 4 | 65.6 | 182.2 | 303.1 | 489.6 |
| dd R = 8 (RT 16 / 32) | 76.3 / 69.6 | 206.3 / 190.5 | 335.2 / 312.8 | 518.1 / 508.7 |

gate_up is the same kernel in every row, so the full-op differences are the down kernel's. RT = 16, R = 4 is the default: it
matches RT = 8 at T = 16 and is not slower at T = 1 (the 1-stream serve cell runs this kernel too). RT = 8 is +1 us at
T = 1. R = 8 spills (168 B of stack at the 80-register cap; R = 4 has 8 B). The micro's nsys table groups the RT = 16 variants under one grid. That is why
only RT = 8 has a clean per-kernel row.

## 3. Step timing (W1)

`timing_35b_batched_step` (graphs on, device argmax, 3 runs of 60 steps each), one dev binary, `INTARSIA_MOE_DOWN_DD=0` vs
default, interleaved off / on / on / off (`raw/steptime-w1.txt`):

| arm | B = 8 runs (ms) | B = 16 runs (ms) |
|---|---|---|
| off (v2), round 1 | 26.42 27.54 28.88 | 42.47 43.73 45.28 |
| on (dd), round 1 | 25.94 27.04 28.39 | 41.11 42.47 44.08 |
| on (dd), round 2 | 25.90 27.05 28.57 | 41.13 42.62 44.03 |
| off (v2), round 2 | 26.61 27.80 29.01 | 42.37 43.89 45.15 |

B = 16: -1.24 ms per step (-31 us per layer x 40). B = 8: -0.56 ms.

nsys serve burst, `serve --max-batch 16` under the 16-stream burst, the same dev binary with and without the knob
(`raw/dev-b16/`, `raw/dev-off-b16/`; `shapes.py` decode window 4..11 s): down 202.4 -> 166.1 us per layer at 40 calls per step.
gate_up 329.5 / 322.6 (the same kernel; run-to-run). These profiles' window gives higher absolute times than VPROF's 188 / 283
(a different burst window). The A/B is same-binary, same-window.

## 4. Served identity and serve A/B (W2)

PENDING

## 5. Not done / caveats

- The cold micro routes by a Zipf over the experts. It matches serve's measured distinct counts (59.5 vs 61.45 at T = 16,
  38.2 vs 37.91 at T = 8), not serve's actual expert popularity.
- Not tried: a transposed multi-row warp reduction (fewer shuffles per pair). It is bitwise only if it reproduces
  `ik_warp_sum`'s xor tree per row. Also not tried: staging h in shared memory per expert subset. Each block still reads all
  pairs' h from L2 (P x I x 2 = 128 KB per R-row group).
- ncu not used (VPROF: it hung on this kernel). Registers and shared memory are from `cuobjdump --dump-resource-usage`.

## GPU use

Two GPU windows: W1 (#6995, 12 min: gates, batch_cuda_real, step timing, two nsys bursts) and W2 (identity + serve A/B).
Short checks (< 3 min each): #6986, #6987 and #6989 (the micro, the last two under nsys). Builds and `plan show` ran as
`--load cpu` slots. spark1-vllm stayed up; no `--takes-down`.

Scripts: `scripts/moedn/` (from conv16 / step16: build, gates, batch-real, steptime, prof, ident, plan, le-window, w1, w2;
new: micro.sh, mk.py).
