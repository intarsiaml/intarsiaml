# HIP admission tests on main: WMMA kernels and the 3-token MoeBlock quick case (lane wadm)

Lane `lane-wadm/wmma-admission`, 2026-10-26. Goal: make `every_hip_kernel_has_a_quick_case` and `every_hip_kernel_matches_the_reference` (`crates/intarsia-backend-hip/tests/`) pass on main without skip lists, loosened tolerances or removed kernels. No kernel, tolerance or registry entry changed.

## What was actually failing

The brief said 10 WMMA kernels lacked quick cases. That was only half right.

| Test | Cause on origin/main (`1f979fad9`) | Fix |
|---|---|---|
| `every_hip_kernel_has_a_quick_case` | `moe_block_bf16_grouped` (lane l7) takes T 3..=16 tokens; the quick `MoeBlock` sweep stopped at T=2. Not a WMMA kernel: the test already filters `*wmma*` out, and `quick_coverage_wmma.rs` (knob on) already passes. | The quick sweep gets the (T=3, E=16, K=8, H=208, I=64, IS=16) case the full sweep already had (`crates/intarsia-core/src/quant/cases.rs`). It now also runs in the CUDA and Metal quick sweeps; the full sweep had it, so nothing new is exercised there beyond the quick run. |
| `every_hip_kernel_matches_the_reference` | The 10 opt-in WMMA kernels (`INTARSIA_HIP_WMMA_ROWS=1`) reject every shape with the knob unset, so they run no case (`NO CASES`) and `all_pass()` is false. With the knob on their GEMV twins reject theirs; the knob is read once per process, so no single process grades both (`2026-10-25-wg-hip-wmma-gemv.md`, verification table). Their sweep cases already existed. | The test runs in two passes (below). |

## The admission test now

Pass 1 (knob unset) requires every kernel that ran a case to be admitted. The kernels with no case there are re-graded by a child run of the same test (`current_exe --exact`, `INTARSIA_TEST_CHILD=<kernel list>`, `INTARSIA_HIP_WMMA_ROWS=1`, `INTARSIA_HIP_WMMA_TABLE=fp8a8=all,bf16=all`, the table `quick_coverage_wmma.rs` uses). In the child each listed kernel is run alone (`only_kernel`) and must produce exactly one summary that is admitted with at least one case. The parent requires the child to exit 0 and print `child graded N/N` with N the size of the list, so a child that skips (no device) or crashes fails the test. There is no kernel-name filter: a kernel with no case in either pass fails.

## Kernels, tolerances

The 10 kernels and their registered numerics (unchanged): all `Exactness::Deterministic`, none `.row_table()`. Tolerance is `TOL` = (max_abs 1.5e-2, rel_l2 1e-3) for the fp8 and pair kernels (the same constant their GEMV twins use), `BF16_TOL` = (1.5e-2, 6e-3) for `matmul_bf16_wmma` and `_rows` (as `matmul_bf16_gemv`), `F32_OUT_TOL` (1e-3, 2e-4) for f32 outputs of the fp8 matmuls. The reference is the CPU backend's reference kernel of the op, through `run_admission`.

Result, window `fleetq #5789` (tatooine card 0, release build, clean tree `5190dbbda`+exec bit), child pass, full sweep:

| kernel | cases | failed | worst max_abs | worst rel_l2 |
|---|---|---|---|---|
| matmul_bf16_wmma | 12 | 0 | 4.371e-4 | 2.037e-5 |
| matmul_bf16_wmma_rows | 12 | 0 | 4.371e-4 | 2.037e-5 |
| matmul_fp8_wmma | 24 | 0 | 1.136e-3 | 4.216e-5 |
| matmul_fp8_wmma_rows | 48 | 0 | 2.183e-3 | 5.205e-5 |
| matmul_pair_bf16_wmma | 4 | 0 | 0 | 0 |
| rmsnorm_add_matmul_pair_bf16_wmma | 8 | 0 | 0 | 0 |
| matmul_fp8_pair_wmma | 15 | 0 | 6.010e-4 | 2.587e-5 |
| gated_rmsnorm_matmul_fp8_wmma | 12 | 0 | 5.061e-4 | 3.996e-5 |
| rmsnorm_add_matmul_fp8_pair_wmma | 12 | 0 | 0 | 0 |
| sigmoid_mul_matmul_fp8_wmma | 10 | 0 | 3.868e-5 | 2.784e-6 |

All ten sit far inside tolerance (worst max_abs 2.2e-3 against 1.5e-2). No kernel was wrong against the reference.

## Mutants (the cases bite)

Four mutant admission binaries were built before the window (`scripts/wadm/build.sh`) and run in it in the knob-on pass (the pass the parent spawns); each had to fail and name the mutated kernel.

| mutant | change | result |
|---|---|---|
| m1 | `device.rs`: `matmul_pair_bf16_wmma` writes its two outputs swapped | CAUGHT: 6 FAIL rows, 3 numeric (max_abs 1.6, rel_l2 1.4), 3 `IK_ERR_SHAPE` (N0 != N1 trips the launcher's shape check); also caught through the whole test (`m1-full`, parent path, 67 s) |
| m2 | `device.rs`: `matmul_fp8_pair_wmma` outputs swapped | CAUGHT: 17 FAIL rows (5 numeric, the rest `IK_ERR_SHAPE`) |
| m3 | `device.rs`: `rmsnorm_add_matmul_fp8_pair_wmma` writes the sum and normed outputs swapped | CAUGHT: 14 FAIL rows, all numeric (rel_l2 0.43-0.44) |
| k1 | `ik_f8_rows.h`: the W8A8 WMMA path drops the second half of each 32-wide K chunk | CAUGHT: 38 FAIL rows on `matmul_fp8_wmma` and `_rows`, rel_l2 0.5-0.75 |

Control: the clean binary in the same mode with all five mutated kernels listed: exit 0, `child graded 5/5`.

## Run

One window, `fleetq #5789` (`--takes-down radiance`, both R9700s), radiance stopped 14:55:03, back 14:57:32 (`/v1/models` answered 15 s after the start command; 2.5 min down). Clean binary, all of `tests/admission.rs`: 6 passed in 66 s, `child graded 10/10`. An earlier job (#5780) was released unused: `win.sh` was not executable in git, so the job died before touching radiance; fixed in the next commit.

Raw logs: `docs/results/2026-10-26-wadm-wmma-admission/raw/` (`clean.out`, per-mutant `.out`/`.err`, `window.log`). Local checks: `quick_coverage` and `quick_coverage_wmma` pass, `cargo test -p intarsia-core --lib` 132 pass, `scripts/prepush.sh` (docs-sync tests) pass.

## Not done

CUDA and Metal `kernels test --quick` were not re-run with the extra T=3 MoeBlock case (the full sweep already had it).
