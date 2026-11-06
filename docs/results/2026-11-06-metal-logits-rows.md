# Metal M-row scalar logits (lane dm, split out of lane-dm/metal-dflash2)

Branch `lane-dm/metal-logits-rows` off main `551c56181` (#322 merged, `INTARSIA_METAL_MXFP4_MR` default on).
Coordinator, 2026-11-06 (by message): split LOGITS_ROWS out into its own PR, default on (`INTARSIA_METAL_LOGITS_ROWS=0` disables), gates as below.

## What and why

The Qwen3.8-27B lm_head sits at byte 8 mod 16 of `model.safetensors` (absolute offset 219640; the embedding too). On Metal the 1-row
`Logits` launch therefore takes `gemv_f32o_scalar` (5.8 ms a call: 557 ms over 96 steps in lane df's op table, vs 4.3 ms for the 16-byte
`gemv_f32o_v` on an aligned copy), and `logits_bf16_gemv_rows` (the verify head, rows_exact) fell back to M one-row launches: the MTP k2
verify head was 3 x 5.8 ms. `logits_rows_scalar_m{2..16}` (kernels/metal/logits_rows.metal) is the M-row twin of `gemv_f32o_scalar`:
each simdgroup owns 4 output rows, the M activation rows are staged in threadgroup memory per 512-element K tile (a multiple of 32, so
element i stays with lane i % 32 in ascending order), every row's fma chain and `simd_sum` are the 1-row kernel's. It runs only where
the 1-row launch would take the scalar kernel; the 16-byte M-row shaders and every M = 1 call are untouched.

## Pre-registered speed rule (written before the ABAB)

Same lane binary, 27B, default spec (MTP k2, pipelined, MR default on), `generate` 256 forced tokens, prompts short-0 and 4k-1, visits
A B A B (A = `INTARSIA_METAL_LOGITS_ROWS=0`, B = unset), 3 runs a visit, decode tok/s. **Ship default-on if B / A of the two-visit
medians is >= 1.08 at both prompts**; `--spec off` B / A within 0.97..1.03 (M = 1 untouched).

## Result (fleetq #8811 gates + ABAB, #8822 diagnosis; `raw/steps.log`, `raw/diag/steps.log`)

* No-op gates: `plan show` 8B / 27B / 35B byte-identical lane vs base; `kernels list` identical; `kernels test` 49 kernels / 980 cases,
  identical after dropping timing lines; greedy ids 8B and 35B (short-0, 4k-1) lane == base; 27B MTP k2 (pipelined, eager, knob 0) and
  MTP k4 == `--spec off` at short-0 and 4k-1. Kernel tests: `tests/logits_rows.rs` (bitwise, M 2..16), `multirow`, `rows_exact`, lib.
* ABAB (generate, decode tok/s, 3 runs a visit): MTP k2 short-0 A 52.7 / 51.5 / 52.2, 54.1 / 53.5 / 53.8; B 53.1 / 52.1 / 53.3, 52.9 / 50.6 / 54.3;
  4k-1 A 46.5 / 45.9 / 48.5, 50.1 / 49.1 / 49.9; B 49.3 / 48.9 / 49.5, 49.6 / 49.4 / 47.9. B / A ~1.00: **the pre-registered rule (>= 1.08) is not met.**
* Why (diagnosis, eager trace, two reps): verify + rollback 36.1 / 35.8 ms with the knob at 0, 35.7 / 36.2 ms with it on: no effect.
  Main now runs the 27B with an **fp8 lm_head by default** (#320, `INTARSIA_LMHEAD_QUANT`, load log "lm_head quantised to fp8"): a fresh
  device buffer, verify head `matmul_fp8_gemv_rows`, so the bf16 `Logits` path this kernel speeds up is no longer taken. My +22 %
  measurement was on a pre-#320 base (bf16 head). The kernel helps only with the bf16 head (`INTARSIA_LMHEAD_QUANT=off`) or other models
  whose head sits off 16 bytes.
* The aligned-copy question is moot for the default 27B (the fp8 head is already a device copy). With `INTARSIA_LMHEAD_QUANT=off` an
  aligned copy would cost +2.54 GB resident (the whole-file buffer keeps the original pages) and change `--spec off` bits (the 16-byte M = 1
  kernel sums in another order).

**Verdict: not opened as a PR (rule not met on main); reported to the coordinator.**
