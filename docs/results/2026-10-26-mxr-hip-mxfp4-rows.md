# mxr: multi-row MXFP4 GEMV on HIP through exact int8 dot products (lane mxr, 2026-10-26)

Branch `lane-mxr/hip-mxfp4-rows` (off main ec9e36b8), draft PR #176. Phase 0 (standalone harness, windows w1-w3) and Phase 1 (the
production change, gate window w4, localeval window w5). Hardware: tatooine, 2x R9700 (gfx1201), ROCm 7.2.4 dev image, radiance down
inside fleetq slots and restored after each.

## Phase 1: what changed and the gates

**Change.** `matmul_mxfp4_gemv_rows` and `matmul_mxfp4_pair_gemv_rows` at M = 2..16 run the exact int8 kernel by default:
- the pre-pass `mxfp4_xprep_i8_kernel` writes the bf16 QDQ copy, the int8 codes, the group factors and per-call flags;
- `mxfp4_i8_rows_kernel<M, XP>` does the work. The pair is one pre-pass plus one launch over the rows of both matrices; main ran two of each.

M = 1 is unchanged, both kernel and bits. Details:
- **Flags:** calls with an out-of-range activation group (x scale byte outside [3, 244]: subnormal, huge, inf, NaN) take the one-row fp
  chain inside the same kernel. All-zero groups stay on the int8 path.
- **W4A16** (`--mxfp4-act none`) keeps main's bf16 rows kernel. It carries the new `OpAttrs::mxfp4_w4a16`, set in `QTensor::call` and
  in `Model::try_mxfp4_pair` (one line in `model.rs`), and the HIP backend runs the `_bf16` launcher entries for it. Through the int8
  kernel's flagged chain, W4A16 measured 1.5-2.1x main's bf16 rows kernel (raw/w4/t-timing.log, act=0), hence the route.
- **Kill switch:** `INTARSIA_HIP_MXFP4_I8=0` brings back the bf16 rows kernel for everything (`docs/tuning.md`).
- **Scratch:** the registered size grows from 2MK + 16 to about 3.13MK + 32M + 64 bytes. `plan show` for the 27B is byte-identical
  anyway.
- **XP rule:** 4 when N <= 5120 or K > 8192, else 2. It changes speed only (w4 timing, both XP values per shape).

**Gates (w4, fleetq #5739, both cards, 10:23-10:39; raw/w4/):**

| gate | result |
|---|---|
| rows_exact `mxfp4_rows_are_bitwise_the_one_row_gemv`: 20 shapes (27B tp1 + tp2 per-rank, edges), W4A4 + W4A16, bf16 + f32 out, M = 1..16, rows reversed | pass |
| rows_exact `mxfp4_pair_rows_are_bitwise_the_one_row_gemv` (new; main had no pair rows test): gate/up, k/v at tp1 and tp2, odd sizes | pass |
| rows_exact `mxfp4_rows_edge_inputs_are_bitwise` (new): the zero-group and bound cases above, plus flagged rows from row 5 on (M = 2..5 int8, M >= 6 fp chain), and edge weight scale bytes 0..3, 250..255 | pass |
| the same three with `INTARSIA_HIP_MXFP4_I8=0` | pass |
| negative control (`INTARSIA_HIP_GB_CONTROL=matmul_mxfp4_gemv_rows=matmul_mxfp4_mma`) | fails, as it must |
| rows_registry | pass |
| mutants (scripts/mxr/mutants.sh: each mutation must apply exactly once; binary hashes against clean): m1 flag never read, m2 group factor one binade off, m3 code halves swapped, m4 fp chain order reversed | clean passes; all 4 caught (m1 on set B W4A4 flagged rows, m2 and m3 on set A, m4 on W4A16) |
| 27B batched decode == solo step, M = 2..16 (`real_35b_batched_decode_is_bitwise_the_solo_step`), dev and base | pass, both |
| G0 HIP: 8B and 35B `plan show` json + txt and 256 greedy ids at short / 4k, dev vs main | byte-identical |
| 27B 256 greedy ids at short / 4k / 16k, tp1; `plan show` json + txt | identical; byte-identical |
| 27B 256 greedy ids at short / 4k, tp2 | identical |
| admission (`every_hip_kernel_matches_the_reference`), w5, dev on card 0 and main on card 1 | fails on both with the same set: 10 WMMA kernels with NO CASES (main's known debt), 0 kernels failing; `matmul_mxfp4_gemv_rows` 42 cases and `matmul_mxfp4_pair_gemv_rows` 82 cases (M = 1..16) ADMITTED |
| `kernels test --quick` (dev), w5 | 77 kernels, 365 cases, 0 fail; the two rows kernels ADMITTED (8 and 12 cases). No kernel is newly registered: the int8 path sits behind the existing names, so quick and admission debt does not grow |
| mutants rebuilt from 28415520 (final routing), w5 | clean passes; m1 (set B W4A4, M=6 row 5), m2 and m3 (set A, M=2), m4 (set A W4A16, M=5) all FAIL (raw/w5/mutants.txt) |

What ran which build:
- The rows_exact and rows_registry binaries were rebuilt inside w4 (10:25) from the final source, routing included.
- The `intarsia` binary used for the no-op runs and the tp2 ids is the 09:48 build of c35783e9. It is the same except for the W4A16
  routing, which no single-stream W4A4 path takes.
- The mutants run in w4 were built from c35783e9 (hashes in raw/mutants-build-c35783e9.log). They were rebuilt from 28415520
  (raw/mutants-build-28415520.log) and run again in w5 with the same outcome.
- No MXFP4 op runs on the 8B (bf16) or the 35B (NVFP4): the op is chosen from the checkpoint's detected format
  (`quant/mod.rs::op_for`). The 8B and 35B no-op is therefore one by construction, and checked anyway.
- CUDA and Metal compile none of the kernel code. The only shared Rust change is the `OpAttrs::mxfp4_w4a16` field (default false)
  and its two setters (`QTensor::call`, `Model::try_mxfp4_pair`). `OpAttrs` derives only `Clone, Debug`, and no non-test code formats,
  hashes or serialises it (grep over `crates/`), so `plan show`, admission cache keys and fingerprints cannot depend on it. That is why
  the c35783e9 `intarsia` binary's HIP `plan show` and ids stand for 28415520. The single-stream 27B path never reaches the `_rows`
  launchers either: M = 1 runs `matmul_mxfp4_gemv`, prefill runs `matmul_mxfp4_mma_fold`. The CUDA and Metal G0 below ran on the
  final source.

**Kernel timing, production code (w4, raw/w4/t-timing.log; us per call including the pre-pass; ratio to the M = 1 launch):**

| shape | M=1 | M=2 main / new | M=4 main / new | M=8 main / new | M=16 main / new |
|---|---|---|---|---|---|
| 17408x5120 | 80.8 | 1.04 / 0.98 | 1.14 / 1.00 | 1.77 / **1.06** | 3.32 / 1.64 |
| 5120x17408 | 80.8 | 1.09 / 1.03 | 1.30 / 1.06 | 1.81 / **1.13** | 3.35 / 1.61 |
| 12288x5120 | 59.8 | 1.06 / 1.00 | 1.19 / 1.01 | 1.77 / **1.09** | 3.26 / 1.65 |
| 10240x5120 | 51.1 | 1.07 / 0.98 | 1.23 / 1.02 | 1.80 / **1.10** | 3.24 / 1.67 |
| 6144x5120 | 34.4 | 1.13 / 0.99 | 1.30 / 1.05 | 1.81 / **1.15** | 2.91 / 1.64 |
| 5120x6144 | 35.0 | 1.24 / 1.16 | 1.60 / 1.26 | 2.15 / **1.52** | 3.11 / 1.96 |
| 1024x5120 | 14.1 | 1.09 / 0.94 | 1.29 / 1.03 | 1.66 / **1.06** | 2.53 / 1.34 |
| tp2 8704x5120 | 44.7 | 1.11 / 1.00 | 1.26 / 1.03 | 1.82 / **1.13** | 2.99 / 1.66 |
| tp2 5120x8704 | 45.6 | 1.08 / 1.00 | 1.32 / 1.06 | 1.89 / **1.14** | 3.08 / 1.59 |
| tp2 5120x5120 | 30.8 | 1.10 / 1.00 | 1.31 / 1.07 | 1.79 / **1.13** | 2.86 / 1.58 |
| tp2 3072x5120 | 21.8 | 1.13 / 0.98 | 1.31 / 1.03 | 1.76 / **1.13** | 3.00 / 1.48 |
| tp2 5120x3072 | 21.3 | 1.14 / 1.00 | 1.34 / 1.11 | 1.82 / **1.22** | 2.86 / 1.63 |
| tp2 512x5120 | 10.8 | 1.31 / 1.06 | 1.52 / 1.15 | 2.01 / **1.25** | 3.02 / 1.53 |

"new" is the XP the shape rule picks. Every arm was checked bitwise against the M = 1 launch of each row, with 0 mismatches. The harness's
all-zero activation row 3 stays on the int8 path: from M = 4 the timings show none of the flagged chain's 1.5-2x cost. The GDN b / a projections (48x5120, 24x5120 per rank) are launch-bound: 1.28x at
M=2 against main's 1.70x.

**Served (w5, fleetq #5747, 11:00-11:35; raw/w5/).** localeval speed on the 27B (`intarsia serve --max-batch 16 --max-seq 8192`,
1k-token prompt, 512 generated tokens, thinking off, 1 stream plus 4 / 8 / 16 streams). Arms were base (main ec9e36b8) and dev
(28415520) at tp1 (card 0) and tp2 (both cards), a fresh server per arm, run x3 interleaved with round 2 reversed. Every arm was
checked (`scripts/mxr/le-summary.py`):
- all 12 arms complete;
- the 1-stream row has 512 completion tokens (min = max = 512);
- every concurrency row has n == streams and no errors;
- no warnings.

localeval records no per-request token counts for the concurrency rows, and the serve logs do not print them either. There the
512-token length rests on the harness forcing it (`ignore_eos` + `min_tokens`, "forced_length" true on every arm) plus the n and
error checks, not on a count.

Aggregate decode tok/s, mean of 3 (the three values in raw/w5/localeval-summary.txt are within 0.4 %):

| tp | arm | 1 stream | 4 streams | 8 streams | 16 streams |
|---|---|---|---|---|---|
| 1 | main | 32.9 | 84.8 | 114.1 | 133.0 |
| 1 | dev | 32.9 | 98.7 | 155.7 | 201.8 |
| | dev / main | -0.1 % | **+16.4 %** | **+36.4 %** | **+51.7 %** |
| 2 | main | 53.3 | 130.6 | 181.0 | 213.7 |
| 2 | dev | 53.3 | 154.4 | 241.8 | 310.2 |
| | dev / main | +0.1 % | **+18.2 %** | **+33.5 %** | **+45.2 %** |

Single-stream speed is unchanged, as it must be: M = 1 runs the same kernel. Expressed as an effective step against the 1-stream step
(streams / aggregate vs 1 / single), 8 streams at tp1 cost 2.31x a 1-stream step on main and 1.69x on dev. That includes localeval's
staggered prefills and is not a pure 8-row step. The captured batched-step profile (30.3 vs 59.5 ms on main) was not re-run.
The dev ratio is in line with the ~1.4-1.5x estimate above plus the prefill interleaving.


## Phase 0 result: go, with design (a). The M=1 kernel and its bits stay unchanged.

On the 27B's W4A4 path, the 32-element group sum of the production GEMV is exact in fp32. Each product is
e2m1 x e2m1 x 2^ex, which is (integer code product) / 4 x 2^ex, with the codes doubled to integers in {0, 1, 2, 3, 4, 6, 8, 12}. Every
partial sum is a multiple of 2^(ex-2) of magnitude at most 4608 x 2^(ex-2), so it fits the 24-bit significand. The same exactness makes
tp=2 bitwise tp=1 on this model (lane T2, unanimis 769d59ea).

So t = sum_e wf[e] * xq[e] equals (float)T * 2^(sbx - 129), where T is the integer dot of the doubled codes. `v_dot4_i32_iu8` computes T
in any order. After that the kernel keeps the production arithmetic:
- `acc = fma(t, s_w, acc)` per lane, in the same lane-to-group order;
- the same xor-butterfly warp sum.

Every row is therefore bitwise the production M=1 launch by construction. No numerics decision and no KL gate are needed, and M=1 keeps
its kernel and its bits.

The bound needs the x scale byte to lie in [3, 244]:
- below 3, partial sums could be subnormal and FTZ-dependent;
- above 244, they can overflow, and 253 to 255 means inf or NaN.

The pre-pass flags any call with such a group, or with W4A16 (`act == 0`). A flagged call runs the production fp chain for the whole
call. Real activations never reach those scales: |x| would have to be about 1e-36 or 1e35.

Harness results, us per call including the pre-pass. Ratios are to the production M=1 launch of the same shape. "now" is main's row-exact
launcher `matmul_mxfp4_gemv_rows`; "int8" is the int8 kernel with 2 weight rows per warp and a 2-step activation prefetch (r2 xp2):

| shape (27B role, calls/step) | M=1 now | int8 M=1 | M=2 now / int8 | M=4 now / int8 | M=8 now / int8 | M=16 now / int8 |
|---|---|---|---|---|---|---|
| 17408x5120 (MLP gate/up, 128) | 80.8 | 78.2 (0.97) | 1.03 / 0.97 | 1.16 / 1.00 | 1.90 / **1.09** | 3.31 / **1.66** |
| 5120x17408 (MLP down, 64) | 80.7 | 80.3 (1.00) | 1.08 / 1.01 | 1.30 / 1.06 | 1.94 / **1.18** (xp4 1.15) | 3.36 / 1.72 (xp4 1.63) |
| 12288x5120 (attn q+gate, 16) | 59.9 | 58.9 (0.98) | 1.05 / 1.02 | 1.17 / 1.00 | 1.89 / **1.12** | 3.25 / 1.66 |
| 10240x5120 (GDN qkv, 48) | 51.2 | 49.2 (0.96) | 1.07 / 0.97 | 1.21 / 1.01 | 1.75 / **1.13** | 3.24 / 1.67 |
| 6144x5120 (GDN z, 48) | 34.5 | 33.5 (0.97) | 1.14 / 0.98 | 1.30 / 1.03 | 1.77 / **1.17** | 2.89 / 1.67 |
| 5120x6144 (attn o / GDN out, 64) | 35.0 | 38.8 (1.11) | 1.25 / 1.13 | 1.60 / 1.22 | 2.12 / 1.68 (xp4 **1.41**) | 3.19 / 1.92 (xp4 1.86) |
| 1024x5120 (attn k, v, 32) | 14.1 | 12.9 (0.91) | 1.09 / 0.94 | 1.28 / 1.01 | 1.65 / **1.13** (xp4 1.07) | 2.52 / 1.47 (xp4 1.37) |

Raw: raw/w3-timing.log (window w3); the M=1 / prefetch-free arms are in raw/w2-timing.log.

**Bitwise.** Every exact arm of all three harness rounds matched the production M=1 launch on every row. The runs covered:
- M = 1..16;
- 14 shapes: all 27B GEMV shapes, plus N = 1, 7, 9, 33, 300, 301 and K = 32, 64, 2080, 4096, 6176, 17408 (partial warps and tails);
- three input modes:
  - 0: plain;
  - 1: x groups at scale bytes 0..4 and 246..254, subnormal x, inf and NaN elements, zero and -0 groups, weight scale bytes 0..3,
    250..255 including NaN rows;
  - 2: x scale bytes 3..6 and 238..244, the fast path's extremes, plus the same weight scales.

That is 0 mismatches in 42 of 42 runs per round (raw/w{1,2,3}-edge.log). The variant without the fallback mismatches on every mode-1 run,
14 of 14, so the edge inputs do reach the fallback. It matches on modes 0 and 2. This works as a built-in negative control.

Correction (review after Phase 0): about 1 in 24 mode-2 groups is an all-zero group. In these harness rounds an all-zero group had
scale byte 0 and flagged the call, so the per-call-flag arms (fb4, xp*) took the fp chain on every mode-2 run. At scale bytes 3..6 and
238..244, only the arms without the fallback show that the int8 path is bitwise. They match on every mode-2 run.
Phase 1 lets amax == 0 stay on the int8 path (T = 0, factor 1, so t = +0, the same as the fp chain, which starts at +0.f and never yields
-0). It tests this in `rows_exact.rs` (`mxfp4_rows_edge_inputs_are_bitwise`): zero rows of both signs mixed with live rows, -0
activations, a zero first group, and groups at exactly bytes 3 and 244.

**Step-level estimate (not measured yet).** Main's 27B one-row step is 30.3 ms and its batched 8-row step 59.5 ms (from the mr batch
profile). Weighting the table by calls per step:
- the MXFP4 GEMVs take 23.2 ms at M=1;
- at M=8 they take 44.0 ms now and about 26.6 ms with int8 (1.15x of M=1);
- at M=16 they take about 76.7 ms now and about 38 ms with int8.

| | step M=1 | step M=8 | ratio | step M=16 |
|---|---|---|---|---|
| main (measured, batched decode of 8 sequences) | 30.3 | 59.5 | 1.96 | 110.9 |
| int8 rows (estimate: main minus the GEMV saving) | ~30 | ~42-45 | ~1.4-1.5 | ~72-78 |
| floor if the MXFP4 GEMVs cost nothing extra | 30.3 | ~38.7 | ~1.28 | |

Cross-check against the mr batch profile (`matmul_mxfp4_pair` + `matmul_mxfp4` op times, profiled, B=2 to B=8: 33.98 to 50.56 ms,
+16.6 ms): the harness says +19.6 ms over the same range, about 15 % more growth than the measured step. Scaled by 0.85, the saving
at M=8 is about 14.8 ms instead of 17.4, which gives the range in the table. The measured step at M=8 comes in W5.

The non-GEMV part of the batched 8-sequence step grows from about 7.1 to 15.5 ms. The main items:
- GatedDeltaNet state: about 0.5 ms per sequence, a byte floor;
- attention;
- lm_head rows.

So 1.3x for **batched decode** of 8 sequences is not reachable through the GEMV alone. An 8-row speculative verify of **one** sequence
(dfh) reads the GDN state and KV once, so its non-GEMV growth is smaller, and about 1.3x looks reachable there (estimate). The coordinator's
dfh numbers (eager 8-row verify 85 ms) are eager and not comparable to the captured 30.3 ms step.

## What limits M=8 today, and in the int8 kernel

The production row-exact kernel is VALU-issue bound. Per activation row and 32-element group, a lane does:
- 32 fma;
- the bf16 unpack of x, shared by 2 weight rows;
- the LDS LUT read of each weight.

The ISA (raw/mxr0-isa.txt) has 958 VALU at M=8, and the slope is about 12.5 us per extra row on 17408x5120.

The int8 kernel's per-row work per group is:
- 8 `v_dot4_i32_iu8` per weight row;
- cvt, mul and fma;
- 2 x 128-bit loads of int8 codes (half the bytes of bf16) and one 4-byte scale load.

The weight decode (nibble to int8 through 2 `v_perm_b32` tables plus a sign-mask `v_perm` and `v_bfi`, 15 VALU per 8 weights) is done once
per weight, not once per row. Static loop body at M=8, R=2 (raw/isa-w1-i8-m8-r2-nofb-loop.s): 128 dot4, 48 perm, 16 cvt / mul / fma, 38
64-bit address ops.

Measured, kernel only (rocprofv3 kernel trace, raw/kernel-trace-summary.txt), 17408x5120:

| kernel | M=1 | M=8 | M=16 |
|---|---|---|---|
| main `mxfp4_gemv_rows_kernel` (M=1: `mxfp4_gemv1_lds_kernel`) | ~80 | 134 | 259 |
| int8 rows, R=2, no prefetch | 72.5 | 90.7 | 134.5 |

Round by round:
- **Round 1 (w1).** A per-group fallback branch inside the loop cost 40-100 % at M=8-16, through VGPRs (65 to 111-180) and occupancy (16
  to 8-12).
- **Round 2 (w2).** Moved the fallback to a per-call flag, read with scalar loads. That made it free (r2 fb4 equals the no-fallback arm
  within 1-2 %).
- **Round 3 (w3).** The rest was latency. The activation loads of each row sit behind a scheduler barrier, so with few waves per SIMD each
  row's L0/L2 latency is exposed. A 2-step register queue of activation loads (xp2) took 17408x5120 M=8 from 98 to 88 us and
  5120x17408 from 125 to 96 us.
- **Remaining.** The small-N shapes (N = 5120: down 5120x17408 and o/out 5120x6144) have only 2560 warps at R=2. They remain the worst at
  1.15-1.4x.

rocprofv3 `--pmc` on this stack (gfx1201, ROCm 7.2.4) returns 0 for SQ_INSTS_*, SQ_INST_CYCLES_VALU, SQ_WAVE_CYCLES, TCP_* and GL2C_*.
Only GRBM_GUI_ACTIVE, SQ_BUSY_CYCLES and SQ_WAVES count (raw/pmc-summary.txt). The limiter analysis above therefore rests on ISA
counts, kernel-trace times and ablations, not counters.

**M=1.** The int8 path runs M=1 bitwise and is 0-4 % faster than main's LDS kernel on the K=5120 shapes, even with its separate pre-pass
launch (about 3 us):
- 78.2 vs 80.8 on 17408x5120;
- 12.9 vs 14.1 on 1024x5120.

It is 11 % slower on 5120x6144, where the fused-pre-pass M=1 variant ("m1 lds r1": 33.7-33.8 vs 35.0, raw/w3-timing.log) wins instead.
This is a small free win, not part of this lane's default change unless the gates show it holds in the step.

## Estimates asked for in the brief

- **(a) Row-exact, M=1 unchanged:** measured above.
  - Kernel-level M=8 / M=1 is 1.09-1.18 on the five large shapes, 1.41 on 5120x6144 and 1.07-1.13 on 1024x5120.
  - The MXFP4 GEMV share of the step goes 1.89x to about 1.15x at M=8, and 3.3x to about 1.65x at M=16.
- **(b) Co-designed M=1 + multi-row pair:** not needed. (a) already shares one reduction order with M=1 by exactness, so (b) would only buy
  numerics freedom (WMMA-ordered sums), which is not needed for these ratios. The remaining lever for M > 8 is an iu8 WMMA kernel. It
  would cut the per-row cost, which is now about 2.9 us per row on 17408x5120 from M=8 to 16, mostly the fp32 chain plus x loads.
  That kernel can stay row-exact too: the fp32 chain needs one accumulator per lane-slot, kept in waves, plus an LDS reproduction of the
  butterfly. Not started; it is the Phase 2 candidate if M=16 matters.
- **Radiance reference (ISA only, raw/radiance-isa-study.txt, Apache-2.0).** Radiance's 1-8 row MXFP4 GEMV is also a weight-in-registers
  `v_dot4` / WMMA kernel. It differs in two ways:
  - its activations are fp8 (W4A8, not Quark's W4A4);
  - its reduction order is fixed by K alone, independent of M (same idea as here).

  Its per-kernel times were not measured; that would need a rocprof run of radiance itself in a window.

## Phase 1 plan (as written after Phase 0; the shipped change is above)

1. `kernels/hip/mxfp4.hip`: add `mxfp4_xprep_i8_kernel` and `mxfp4_i8_rows_kernel<M, R=2, XP>` (XP 2 at K <= 8192, 4 above; R and XP
   chosen per shape from this table). `matmul_mxfp4_gemv_rows` and `matmul_mxfp4_pair_gemv_rows` at M >= 2 use them:
   - the pre-pass writes the bf16 QDQ x as before, plus the int8 codes, scales and flags; scratch grows from 2MK to about 3.2MK bytes;
   - the pair launcher's M >= 2 path does one pre-pass for both halves instead of two.

   Kill switch `INTARSIA_HIP_MXFP4_I8=0` brings back today's rows kernel, with a `docs/tuning.md` row.
2. The slow path (flagged call) is the production fp chain with a small register footprint. W4A16 (`--mxfp4-act none`, opt-in) always
   takes it, so W4A16 multi-row may get slower than today. To be measured; if it does, the kill-switch path is used for W4A16.
3. Gates as in the brief:
   - `rows_exact` bitwise for M = 1..16 on all shapes, extended with the edge inputs above;
   - quick and admission cases for each new kernel;
   - two mutants;
   - 27B batched == solo and ids == main;
   - G0 no-op on 8B and 35B on HIP, CUDA and Metal;
   - localeval at 1/4/8/16 streams, tp1 and tp2.

## Method and reproduction

- Harness:
  - `scripts/mxr/mxr0.hip` (round 1);
  - `mxr1.hip` (round 2: fallback modes, pre-pass with 4 threads per group, fused M=1);
  - `mxr2.hip` (round 3: activation prefetch queue).

  Each includes `kernels/hip/mxfp4.hip` and uses the production `ik_matmul_mxfp4_gemv` per row (W4A4, `act = 1`) as the reference. It
  runs 3 rotating weight copies, so the 64 MB MALL cannot hold a matrix, with 200 calls x 3 rounds and the minimum taken.
- Scripts:
  - `scripts/mxr/build.sh <harness>`: hipcc in `intarsia-hip-dev:rocm7.2.4-rust1.90.0`, plus the ISA and resource usage, summarised by
    `scripts/mxr/isa.py`;
  - `scripts/mxr/win.sh`: radiance down, timing, edge runs and rocprofv3 on card 0, then radiance restored and checked on :8080;
  - `scripts/mxr/sync.sh`.
- fleetq: builds #5712, #5713, #5717, #5718, #5721, #5722 (`--load cpu`); windows #5714 (w1), #5719 (w2), #5723 (w3)
  (`--takes-down radiance --quiet gpu,cpu`). Radiance answered on :8080 after each window. That is 3 of the lane's 5 windows.
- Harness binary hashes (sha256, printed by build.sh):
  - mxr0 b0fb1ef3… (w1);
  - mxr1 87a862f2… (w2);
  - mxr2 6a0ceb7a… (w3).
