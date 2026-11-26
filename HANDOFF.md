# HANDOFF: lane ink-hip-mix (package B: mix + projection), branch `ink/hip-mix`

State: kernels, module and tests written, merged on top of `6c5ee5d67`; GPU status is in `docs/results/2026-11-25-inkling2-hip-mix.md`
(the authority; this file only says how to wire the lane in).

## Wiring (nothing outside the lane's files is needed)
- Drop-in over the skeleton stub: `crates/intarsia-backend-hip/src/inkmix.rs` (TABLE / OPS / all() / `launch::launch`, exact shape),
  `kernels/hip/ik_hip_abi_inkmix.h`, `kernels/hip/ink_mix.hip`, `kernels/hip/ink_proj.hip` (build.rs globs `kernels/hip/*.hip`).
- No edit of `kernels.rs`, `device.rs`, `build.rs`, `UNIMPLEMENTED` or shared tests. `intarsia-backend-hip` host tests pass
  (`INTARSIA_NO_HIP=1 cargo test -p intarsia-backend-hip --lib`: ABI header test, names unique, ops covered).

## Dispatch arms (all in `inkmix::launch::launch`, per-file `extern "C"`)
| launcher | op | ABI symbol | inputs / outputs |
|---|---|---|---|
| `ink_in_proj_bf16_gemv` (+ `_rows` twin, same launcher, `.rows_exact()`) | InkInProj 73 | `ik_ink_in_proj_bf16_gemv[_rows]` | x, w -> pq, pk, pv; T <= 16 |
| `ink_in_proj_bf16_matmul` | InkInProj 73 | `ik_ink_in_proj_bf16_matmul` | T > 16: three `ik_matmul_bf16` WMMA GEMMs |
| `sconv_res_bf16` | SconvRes 69 | `ik_sconv_res_bf16` | x taps res add (+ pos0, `a.pos`) -> y ring(in place) sum |
| `attn_res_mix_bf16` | AttnResMix 67 | `ik_attn_res_mix_bf16` | src last add queries norm (+ eps) -> out |
| `conv_mix_bf16` | ConvMix 74 | `ik_conv_mix_bf16` | 11 inputs, pos0 / pos, eps -> y ring sum mix; scratch |
| `loop_blend_bf16` | LoopBlend 68 | `ik_loop_blend_bf16` | dest msg down gain forget norm (+ eps) -> out; scratch |

## Registration rules honoured
`.when(..)` as the CUDA twin (AttnResMix: `kv_len <= 8`, `n <= 4`: queries > 4 are refused, same `.when` as CUDA; LoopBlend 1..=3 streams; ConvMix `n <= 4`;
all `k <= 4608`); no `.row_table()`, no `.act_quant()`, no `.situ()`; launchers never size grids from host `pos0` (the row kernel reads
`*pos` when `a.pos` is non-null, so they are graph-capturable); scratch via `.scratch(..)` (blend: `blend_bytes` in `inkmix.rs`, mirrors
`ink_mix.hip`). ConvMix scratch is sized at 3 streams / rank 128 (not in `ShapeInfo`), as CUDA.

## Plan-shape notes for the lead
- `InkInProj` T <= 16 selects `ink_in_proj_bf16_gemv` (priority 10; the `_rows` twin only for `exact_rows` calls); T > 16 the matmul form
  (`Exactness::Fast` as CUDA's cuBLAS form: WMMA GEMM, not bitwise the GEMV, chunk invariance not promised).
- Row kernels are chunk-invariant (one block per row, same code at every T): no twin kernels needed.
- Not built: e4m3 / P12 `InkInProj` (the model falls back to `MatmulFp8` per pack / the bf16 stack), row-table forms.

## Heads-up: AttnResMix with more than 4 queries
`model/inkling2.rs` issues `AttnResMix` over `msg_q_all` (4 * n_core queries, `INK_MSG_PRE_ROWS`) only when `INTARSIA_INK_MSG_PRE` is on or the
backend is `metal` (`ink_msg_pre_env().unwrap_or(name == "metal")`), so the default HIP plan only makes the 4-query per-visit call this kernel
takes. Forcing `INTARSIA_INK_MSG_PRE=1` on HIP would hit `NoKernel` (the `.when` refuses Q > 4, as CUDA's). The admission sweep I ran (quick, 93
AttnResMix cases, Q in {1, 3, 4}) does not contain the Q = 48 case; refusal there is by registration, not exercised on the GPU.
