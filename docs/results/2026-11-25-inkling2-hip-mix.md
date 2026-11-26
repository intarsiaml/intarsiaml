# Inkling-2-Mini HIP, package B (mix + projection), lane ink-hip-mix, 2026-11-25

Branch `ink/hip-mix`, merged `ink/hip` (skeleton `6c5ee5d67`, on origin/main `d8c3c7148`).
Ops: SconvRes 69, AttnResMix 67, LoopBlend 68, ConvMix 74, InkInProj 73 (+ `_rows` twin).

## Briefs received (kept against context loss)

- Shared brief: `/opt/code/intarsiaml/.worktrees/hip-swarm/COMMON.md` (ownership, correctness bar, tatooine rules: windows <= 20 min,
  <= 3 per lane, none after 06:00 EST, radiance up by 07:00, report once at pass or 06:30 EST).
- Lead update 22:00 EST: `git merge ink/hip` now; module in the exact shape `TABLE` / `OPS` / `all()` /
  `#[cfg(ik_hip)] pub(crate) mod launch { extern "C" {..} pub(crate) unsafe fn launch(kernel, &InkArgs) -> Option<c_int> }`;
  per-file `extern "C"`; do not touch `UNIMPLEMENTED` or shared tests; **skip MoeRoute** (lead wrote it); correctness forms first
  (`ink_row_kernel` / blend, `in_proj_gemv_kernel` bf16 arm); no `.row_table()`, no `.act_quant()`.

## Status (2026-11-25 21:45 EST)

All five ops ported and ADMITTED on the R9700 in the first unit window; rows-exact (bitwise) tests pass; bench below. Branch `ink/hip-mix`.

Files: `kernels/hip/ink_mix.hip` (row kernel + blend: port of `ink_row_kernel`, `ink_ring_kernel`, `ink_blend_gate_kernel`,
`ink_blend_norm_kernel`; the controller h projection is a new plain wave GEMV `ink_ctl_gemv_kernel`, not the CUDA mma split-K),
`kernels/hip/ink_proj.hip` (`in_proj_gemv_kernel<M>` for T <= 16; T > 16 = three `ik_matmul_bf16` calls), `kernels/hip/ik_hip_abi_inkmix.h`,
`crates/intarsia-backend-hip/src/inkmix.rs`, `crates/intarsia-backend-hip/tests/ink_mix.rs`, `HANDOFF.md`.
Launchers: `ink_in_proj_bf16_gemv`, `_gemv_rows` (twin), `ink_in_proj_bf16_matmul` (T > 16), `sconv_res_bf16`, `attn_res_mix_bf16`,
`conv_mix_bf16`, `loop_blend_bf16`. Not built: e4m3 / P12 InkInProj, row-table forms (as briefed).

## Tests (window #11927, radiance down 21:42:36 - 21:43:36, up again after 15 s; log `raw/window1-unit-bench.log`)

`cargo test --release -p intarsia-backend-hip --test ink_mix -- --nocapture --test-threads=1` (dev image, card 0, `INTARSIA_REQUIRE_GPU=1`): 4 passed.
- `ink_mix_ops_match_the_reference` (admission vs `ink_ref`, the CUDA tolerances): ink_in_proj_bf16_gemv 24 cases ADMITTED (worst max_abs 3.8e-3,
  rel_l2 7.3e-5), `_gemv_rows` 24 ADMITTED, `ink_in_proj_bf16_matmul` 5 ADMITTED (3.1e-3 / 4.7e-5), conv_mix_bf16 532 ADMITTED (4.5e-3 / 1.2e-3),
  attn_res_mix_bf16 93 ADMITTED (2.0e-3 / 4.9e-5; the Q = 48 case is rejected by `.when`), sconv_res_bf16 64 ADMITTED (1.9e-3 / 1.2e-5),
  loop_blend_bf16 27 ADMITTED (3.2e-3 / 2.6e-4). 0 failing cases.
- `ink_in_proj_rows_exact`, `ink_conv_mix_rows_exact`, `ink_attn_res_mix_and_blend_rows_exact`: BITWISE rows-exact (T-row call = T one-row calls, ring
  carried, final ring equal) at D 2304 and 64, all epilogue / mix / blend variants, T up to 300 for the mixes.
- Host: `INTARSIA_NO_HIP=1 cargo test -p intarsia-backend-hip --lib` 31 passed (ABI header test, names unique, op coverage).

## Bench (same window; `cargo test ... ink_mix_bench -- --ignored --nocapture`; launches back to back, host launch overhead included)

| kernel / shape | us | vs roofline |
|---|---|---|
| in_proj gemv S1 T1 (39.9 MB weights) | 66.9 | 597 GB/s = 95% of 627 |
| in_proj gemv S3 T1 | 65.7 | 608 GB/s = 97% |
| in_proj gemv S1 T4 | 70.8 | 564 GB/s = 90% |
| in_proj gemv S1 T16 / S3 T16 | 131 / 139 | 304 / 286 GB/s (SIMT FMA bound, 4.9 TFLOP/s) |
| in_proj matmul (WMMA) T64 / T512 | 89.8 / 295.7 | 445 GB/s; 28.5 / 69.1 TFLOP/s |
| sconv_res T1 / T16 / T256 / T1024 | 6.3 / 7.0 / 21.5 / 66.0 | activation traffic 274-357 GB/s at T >= 256 |
| attn_res_mix N6 Q1 T1 / T16 / T256 / T1024 | 11.7 / 12.6 / 46.0 / 167 | 180-198 GB/s at T >= 256 |
| loop_blend S3 G64 T1 / T4 / T16 / T256 / T1024 | 29.2 / 41.8 / 65.9 / 217 / 868 | the slow one (below) |

Notes: decode InkInProj is on the byte floor. LoopBlend is the speed candidate: 4 launches per call; its prefill cost is 0.85 us per row
(868 us per 1024 rows; the h wave-GEMV re-reads the 3.5 MB controller weights per group of 8 rows, the WMMA form named in the port map would fix that);
T16 verify rows pay 66 us. Not tuned tonight beyond the correctness form.

## Not verified / open

- Not run through the model (G1 / G2: the lead's window). `-ffp-contract` is left at hipcc's default (fast); rows-exactness holds because every
  row runs the same compiled code, and the admission tolerances pass; no kernel is on `NO_FP_CONTRACT_SOURCES`.
- `rsqrtf` in the conv scale is the hardware approximation (CUDA's too); admitted within tolerance.
