# HANDOFF: lane ink-hip-attn (package A: InkKvPrologue 70, RelSinkAttn 71)

Branch `ink/hip-attn` = origin/main `d8c3c7148` + `ink/hip` skeleton `6c5ee5d67` + this lane. Results:
`docs/results/2026-11-25-inkling2-hip-attn.md`.

## What the lead drops in

Files touched (all lane-owned, nothing shared):
- `kernels/hip/ink_attn.hip` (new; `build.rs` globs it)
- `kernels/hip/ik_hip_abi_inkattn.h` (the two declarations + `ik_ink_attn_attrs`)
- `crates/intarsia-backend-hip/src/inkattn.rs` (the skeleton's shape: `TABLE`, `OPS`, `all()`, `launch::launch`)
- `crates/intarsia-backend-hip/tests/ink_attn.rs` (GPU tests; `#![cfg(ik_hip)]`)
- `docs/results/2026-11-25-inkling2-hip-attn*` (doc, window / run / stage scripts)

## Dispatch, ABI

| Launcher (TABLE) | Op | ABI symbol | Inputs -> outputs (InkArgs) | Scratch |
|---|---|---|---|---|
| `ink_kv_prologue_bf16` | 70 | `ik_ink_kv_prologue_bf16` | i(0..8) -> o(0..6), `pos` | none |
| `rel_sink_attn_bf16` | 71 | `ik_rel_sink_attn_bf16` | i(0..7) -> o(0..3), `sc`, `pos` | `attn_scratch`: min(T, 64) * H * 32 * (Dh + 2) * 4 B (real model: T >= 64 -> 25.6 MB) |

Registration (as the CUDA twin): `acts/weights bf16`, `Exactness::Deterministic`, `BF16_TOL (1.5e-2, 6e-3)`. No `.rows_exact()`,
`.row_table()`, `.priority()`, `.act_quant()`. Predicates: prologue `H % Hk == 0, H/Hk <= 4, 1 <= Dh <= 256` (dr <= 32, K <= 4 at
launch); attention `H/Hk in 1..=3, Dh in {8,16,32,64,128}`. Op 75 is not built (the model falls back to 70 + 71).

## Plan-shape notes

- Device position: grids never depend on the host `pos0` when `pos` is non-null (split count from the capacity bound
  `min(window, Cr)` or `Cr`, as CUDA's `attn_geometry`). The attention runs a call as 64-row slices in stream order; each
  slice is its own launch pair, so a 2048-row chunk is 32 x (s1 + comb [+ kv write]) launches.
- Rows-exact by construction (a row's split map is a function of its position and window only); `ink_rows_exact` is
  backend-name gated, so HIP stays on the plain path regardless.
- FP contraction: `ink_attn.hip` has `#pragma clang fp contract(off)` at file scope (hipcc default is fast-honor-pragmas), so
  no `build.rs` / `NO_FP_CONTRACT_SOURCES` entry is needed for 70/71. If op 75 is built later and must be bitwise the pair, it
  goes in the same file under the same pragma.
- Admission cases: `inkling2/cases.rs` `InkKvPrologue` / `RelSinkAttn` (quick: head_dim 16 shapes, T 1 / 7 / 40, plus the
  T=40 mid case at head_dim 128; full: the real model at T 1 / 4 / 16 / 2048). Every registered kernel has a quick case
  (`quick_coverage` passes on the stub build).

## State

See the results doc's Status section for test results and fleetq job ids.
