# mr: scope for a faster row-exact MXFP4 multi-row GEMV on HIP (no code, no GPU)

Lane mr, 2026-10-25, branch `lane-mr/rows2-scope`. Status: proposed, waiting for the coordinator's go or no-go.
Sources: `kernels/hip/mxfp4.hip` (`mxfp4_gemv_rows_kernel`), docs/results/2026-10-25-mr-hip-mxfp4-rows.md (VGPR, occupancy,
variant timings), docs/results/2026-10-25-mr-hip-logits-rows.md section 2 (ISA of `mxfp4_gemv_rows_kernel<16, float>`,
raw/isa.log), PR #147 profile (GEMVs 69 % of the 16-row step; 83 ms of 111 ms profiled; 100 ms unprofiled).

## Answer in three lines

1. The rows kernel already loads and dequantises each weight block once for all M rows. That lever is spent.
2. What binds it is VALU issue (fma slots, bf16 unpack, a short-K epilogue), not bytes, LDS or occupancy.
3. A bitwise-safe path exists for roughly 11 to 24 ms of the 100 ms 16-row step (85 to 89 ms with lever A, 76 to 81 ms with A+B). Not the 4x.

## 1. What binds the current rows kernel at M=8 and M=16

State: R=2 weight rows per warp, 8 warps per block, x is the QDQ'd bf16 pre-pass output read straight from global/L0.
- Registers: 104 VGPR at M=2 up to 134 at M=16, no spill, occupancy 10 to 12 waves per SIMD. Not register- or occupancy-bound.
- LDS: only the 16-entry fp4 LUT (64 B per block), plus ds_bpermute in the warp-sum epilogue. Not LDS-bound.
- Bytes: 17408x5120 at M=16 is 272 us for 47 MB of weights = 173 GB/s against 590 GB/s measured at M=1. Not byte-bound.
- Time grows linearly in M (M=8 156, M=16 272 us, about 14 us per extra row), the signature of per-row ALU work.
- Instruction mix, M=16 (static, one loop body plus epilogue): 1684 VALU. About 956 are fma slots (382 `v_dual_fmac_f32` = 764
  fmas, 574 single, so 57 % of fmas dual-issue), about 330 are and/lshl/lshr (the bf16 unpack of x, redone per activation row
  and shared only by the 2 weight rows), about 160 `v_add_f32` plus 160 `ds_bpermute_b32` in the epilogue, the rest dequant
  and addressing. Slope 91 VALU per extra activation row.
- Estimate (clock assumed 2.4 to 2.9 GHz, not measured under load): issue count alone gives 165 to 200 us of the 272 us.
  The rest is dependency latency in the epilogue (K=5120 gives only 5 loop trips per lane), x loads at 64 B lane stride,
  and a wave-quantisation tail (8704 warps over about 1400 resident = 6.2 rounds, so up to 10 to 14 % idle at the end).
- Is dequant repeated per row? No. `wf[R][32]` is built once per group and reused for all M. What repeats per (n, m) is
  32 fmas (irreducible), the x unpack (halved by R=2; measured 1 row to 2 rows: 324 to 260 us at M=16), and the epilogue.
- Why 4x off the coordinator's bounds: the 18 ms FMA bound needs 100 % dual issue and zero other VALU; at today's 57 % pairing
  it is about 26 ms, plus the unpack, dequant and epilogue slots. The 21 ms byte bound is not in play at 16 rows.

## 2. Same M=1 order and one load plus dequant per weight block for all rows: possible?

Yes, and it is what ships. M=1 order: lane l walks groups u = l, l+32, ...; per group a 32-long sequential fma chain
t = fma(w[e], x[e], t) from 0; acc = fma(t, s, acc); then the xor-butterfly warp sum. Every (n, m) pair owns its own chain and
accumulator, so M rows are M independent chains per lane. Weight load and dequant are shared (done). The fma itself cannot
be shared: each is a distinct product w[n,k] * x[m,k] with its own rounding. Two things cannot be used under this rule:
- WMMA/MMA: its internal accumulation order is not the 32-chain, so row r would differ from the M=1 GEMV. Only usable if M=1
  moves to the same kernel, which changes bits against main (a decision for the coordinator, not in this scope; unmeasured at M=2..16).
- Packed bf16 dot2: different order, same objection (recorded in the logits-rows doc).

Bitwise-safe levers left, ranked:
- A. x pre-converted to f32 by the pre-pass (`mxfp4_xprep_kernel` writes f32 instead of bf16): deletes the ~330 unpack VALU
  (about 20 % of slots, the same kind of work that moving from 1 to 2 rows cut by 20 % in time). Cost: scratch doubles (M*K*4),
  x loads double in bytes (8 dwordx4 instead of 4 per row per trip); the TA/L0 side is unmeasured and could take the gain.
- B. Raise VOPD pairing from 57 % toward 85 %: interleave two activation rows so four independent chains are live, or hand-emit
  `v_dual_fmac_f32` (bank rules make this fiddly; `__builtin_amdgcn_sched_barrier` currently blocks it and was added to avoid spills).
  Slots saved at best about 190 of 1684.
- C. Order-preserving reduce-scatter warp sum: measured 3 to 7 % at M=8..16 on branch `lane-mr/hip-mxfp4-alu-experiment`.
  Parked by the coordinator; stays parked.
- Dead: R=4 weight rows (M=8: 149 to 143 us, loses on large shapes), x in LDS (160 KB at M=16 exceeds the 128 KB per WGP).

## 3. Expected result and hours

Percentages applied to the unprofiled 16-row step (99.85 ms; GEMVs about 75 % of it). Estimates, not measurements.
| scope | GEMV time | 16-row step | tok/s at 16 rows | hands-on hours | wall hours (fleetq queue) |
|---|---|---|---|---|---|
| today | 100 % | 100 ms | 160 | | |
| P0 ablation only (decides go/no-go) | | | | 2 | 3 |
| P0 + A | -15 to -20 % | 85 to 89 ms | 180 to 188 | 6 | 10 |
| P0 + A + B | -25 to -32 % | 76 to 81 ms | 198 to 210 | 12 | 18 |
| theoretical floor, row-exact, all fmas dual-issued | about -45 % | about 65 ms | | | |
B=8 gains about half as much (the step is 60 ms and the GEMV share is lower).
- P0 (no production code): variants in `scripts/mr/mxbench.hip` on 17408x5120 and 5120x17408 at M=8 and 16: (a) f32 x, (b) x loads
  removed (pure ALU), (c) fmas only. One fleetq quiet window, about 10 min GPU, radiance down. Go criterion for A: (a) at least
  10 % faster than shipped at M=16 on both shapes and bitwise equal to M=1. If (b) is not much faster than shipped, x traffic
  binds and A is a no-go.
- P1 (A in production): pre-pass writes f32, scratch size in the registered kernel, rows kernel reads f32; `rows_exact` M=1..16
  at the 27B shapes already exists and has its negative control. Same footprint as #145 and #151 (HIP only).
- P2 (B): optional, only if P1 lands; uncertain, can slip to a day.
- Gates (about 4 h wall, two radiance-down windows): rows_exact, `kernels test --quick`, 8B/35B/27B no-op vs main, 27B batched equals
  solo N=2..16 (about 11 min), localeval A/B at `--max-batch` 8 and 16.

## Decision needed

Go on P0 only (cheapest, answers whether A is worth building), or P0 + A, or stop. PR #147 is not superseded by #151
(its doc and raw logs are the "before" cited by #151) and was left open.
