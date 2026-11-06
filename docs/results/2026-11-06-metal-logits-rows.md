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
