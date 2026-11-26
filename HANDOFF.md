# HANDOFF: lane ink-hip-mix (package B: mix + projection), branch `ink/hip-mix`

Status, numbers and rejected variants: `docs/results/2026-11-25-inkling2-hip-mix.md` (the authority). This file says how to wire the lane in.

## Wiring (nothing outside the lane's files is needed)
- Drop-in over the skeleton stub: `crates/intarsia-backend-hip/src/inkmix.rs` (TABLE / OPS / all() / `launch::launch`, exact shape),
  `kernels/hip/ik_hip_abi_inkmix.h`, `kernels/hip/ink_mix.hip`, `kernels/hip/ink_proj.hip`, `kernels/hip/ink_route_fused.hip` (build.rs globs `kernels/hip/*.hip`).
- No edit of `kernels.rs`, `device.rs`, `build.rs`, `UNIMPLEMENTED` or shared tests. Host tests pass (`INTARSIA_NO_HIP=1 cargo test -p intarsia-backend-hip --lib`, 32 tests).
- GPU tests: `crates/intarsia-backend-hip/tests/ink_mix.rs` (admission of all six ops, rows-exact bitwise, fast-form == first-form bitwise, route == Logits + MoeRoute bitwise; ignored bench `ink_mix_bench`),
  `tests/ink_mix_probe.rs` (ignored; needs an `INK_PROBE` build of ink_mix.hip).

## Launchers
| launcher | op | notes |
|---|---|---|
| `ink_in_proj_bf16_gemv` (+ `_rows` twin, same launcher, `.rows_exact()`) | InkInProj 73 | T <= 16, one wave per weight row, 95-97% of 627 GB/s at T=1 |
| `ink_in_proj_bf16_matmul` | InkInProj 73 | T > 16: three `ik_matmul_bf16` WMMA GEMMs, `Exactness::Fast` |
| `sconv_res_bf16`, `attn_res_mix_bf16`, `conv_mix_bf16`, `loop_blend_bf16` | 69, 67, 74, 68 | one block per row (rows-exact, chunk-invariant); blend in chunks of 256 rows with scratch (`blend_bytes`) |
| `ink_route_bf16` | InkRoute 76 | router GEMV + top-k, T <= 4; bitwise the pair; NOT faster (15.9 vs 15.7 us at T=1): leave `INTARSIA_INK_ROUTE_FUSED` off unless an end-to-end A/B says otherwise |

## Registration rules honoured
`.when(..)` as the CUDA twin (AttnResMix: `kv_len <= 8`, `n <= 4`: queries > 4 are refused; LoopBlend 1..=3 streams; ConvMix `n <= 4`; all `k <= 4608`); no `.row_table()`,
`.act_quant()`, `.situ()`; launchers never size grids from host `pos0` (the row kernel reads `*pos` when `a.pos` is non-null: graph-capturable); scratch via `.scratch(..)`
(ConvMix sized at 3 streams / rank 128, as CUDA). Not built: e4m3 / P12 InkInProj, row-table forms.

## Heads-up: AttnResMix with more than 4 queries
`model/inkling2.rs` issues `AttnResMix` over `msg_q_all` (4 * n_core queries) only when `INTARSIA_INK_MSG_PRE` is on or the backend is `metal`; the default HIP plan makes the 4-query
per-visit call. Forcing `INTARSIA_INK_MSG_PRE=1` on HIP would hit `NoKernel` (the `.when` refuses Q > 4, as CUDA's). The Q = 48 case is not in the quick admission sweep that ran.

## Speed round additions
- ink_mix.hip default fast forms (mask 3): row kernel with unconditional loads for <= 16 rows (attn_res_mix 13.1 -> 10.2 us, conv_mix 14.8 -> 10.3, sconv_res 6.7 -> 5.6), fused blend launches
  for <= 2 rows (loop_blend T=1 28.1 -> 20.7, conv_mix + blend 37.9 -> 25.4). Bits identical to the first forms. Revert path without a rebuild: env `INTARSIA_INK_MIX_FAST=0`
  (read once, first launch) or `inkmix::set_mix_mode(0)`.
- The fused blend and the route kernel use device-global tickets (`g_gate_ctr`, `g_route_ctr`, reset by the last block): launches of the same kernel must not overlap on different streams.
  Same-stream order (the engine's) and graph capture are fine.
