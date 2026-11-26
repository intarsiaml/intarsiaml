# HANDOFF: lane ink-hip-moe (package C, experts) -> lead ink-hip

Branch `ink/hip-moe`. State and numbers: `docs/results/2026-11-25-inkling2-hip-moe.md`.

## Files (all mine, drop over the skeleton's stubs; nothing in a shared file changed)
- `kernels/hip/ink_moe.hip` (new; `build.rs` globs it)
- `kernels/hip/ik_hip_abi_inkmoe.h` (declarations added)
- `crates/intarsia-backend-hip/src/inkmoe.rs` (TABLE / OPS / all() / launch::launch in the skeleton's shape)
- `crates/intarsia-backend-hip/tests/ink_moe.rs`

## Kernels (TABLE)
| Launcher | Op | Registration |
|---|---|---|
| `moe_experts_situ_bf16` | `MoeExperts` | `.situ()`, weights u8, `when(s.situ && E <= 1024 && K <= 64 && H % 32 == 0 && I % 32 == 0)`, `SITU_TOL (1.5e-2, 2e-3)`, scratch `moe_experts_scratch` |
| `moe_experts_situ_bf16_rows` | `MoeExperts` | the same + rows 1..=16, priority -1, `.rows_exact()` |
| `moe_experts_mxfp8_bf16` | `MoeExpertsMxfp8` | `.ext().situ()`, weights e4m3, `MOE_TOL (1.5e-2, 1e-3)`, same shape rule and scratch |
| `moe_experts_mxfp8_bf16_rows` | `MoeExpertsMxfp8` | the same + rows 1..=16, priority -1, `.rows_exact()` |

No `.act_quant()`, no `.row_table()`. The launchers return `IK_ERR_UNSUPPORTED` for `ink.act_quant != 0` or `ink.moe_layout != 0`
(the dispatch packs them into `aq` as CUDA does), so the model must keep `moe_layout = 0` on HIP (it does: `moe_repack_wanted`
is cuda-only per the port map).

## ABI symbols (`ik_hip_abi_inkmoe.h`)
`ik_moe_experts_situ_bf16`, `ik_moe_experts_situ_bf16_rows`, `ik_moe_experts_mxfp8_bf16`, `ik_moe_experts_mxfp8_bf16_rows`
(signatures as CUDA's `ik_cuda_abi_inkmm.h`), plus two test / bench setters `ik_ink_moe_probe(int)`, `ik_ink_moe_set_pre(int)`
(Rust: `inkmoe::set_probe`, `inkmoe::set_pre`).

## Dispatch
`inkmoe::launch::launch(kernel, &InkArgs)`: inputs 0..=7 (situ: x, ids, weights, gate_up w / s, down w / s, global scales) or
0..=6 (MXFP8), output 0, `a.sc`, `a.s`; attrs `ink.{situ, gate_cap, up_cap, act_quant, moe_layout}`, `gelu`. No `pos`, no row
table. Returns `IK_ERR_INVALID_ARG` for a call of the wrong op or a non-situ `MoeExperts` call.

## Admission
- `intarsia kernels test --backend hip --op moe_experts` (the situ cases of `quant/cases.rs`) and `--op moe_experts_mxfp8`
  (`inkling2/cases.rs`). Quick cases cover all four kernels (shapes H 64 / 128, I 32 / 64, T 1 / 7 / 16 / 40).
- The non-quick sweep has T = 2048 at the real shape: on HIP that is the SIMT pair-table path (no tensor cores yet), slow but
  the same arithmetic; expect seconds per case on the GPU plus the cpu reference's time.

## Plan-shape notes
- Scratch is the core `moe_experts_scratch` layout for all four (pairs | offsets | active | h | y), must be 16-byte aligned.
- Rows-exact at every T (one code path; the pair-table path above 512 pairs gives the same bits), so `ink_rows_exact` can
  treat the ordinary names as it treats CUDA's `_rows` twins; only the twins are registered `.rows_exact()`.
- Knobs: `INTARSIA_HIP_INK_MOE_PRE` = 1 | 3 | 9 (load window; bitwise the same output; default 3). Not part of any fingerprint
  (no value changes). A `docs/tuning.md` row is the lead's file: please add it if the knob survives the bench.
- Prefill: every launch above ~1 ms hits the R9700 dispatch stall; the experts are not split into pieces yet (speed phase).
