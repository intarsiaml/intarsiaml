# NXP: Qwen3.6-35B-A3B prefill on the Metal NAX path (2026-10-27)

Lane NXP (`lane-nxp/metal-nax-prefill`), started 2026-10-26 23:55 EDT on endor (Apple M5 Max, 128 GB). Model `~/models/Qwen3.6-35B-A3B-NVFP4`. Base binary: `origin/main` e86d8c336 (M35P merge). Raw data and scripts: [raw](2026-10-27-nxp-metal-nax-prefill/raw/).

**Status: PARTIAL, not merged.** The code, the admission and mutant checks and the first speed and KL numbers are in. endor dropped out of fleetq at ~01:04 (the fleetq server could no longer resolve the name `endor`, a DNS problem from the gateway swap; the box itself was fine), so the gates that need a GPU slot did not run: G2 (MMLU), the G3 exactness runs on the candidate, the decode-64 KL re-run, the third clean speed round and the extra-model `plan show`. They are queued in fleetq (#5973 to #5979) and listed under "Pending" below. Nothing was timed outside a fleetq slot.

## What changed

| change | files | knob |
|---|---|---|
| `matmul_fp8_nax`: fp8 projections at M > 8 (prefill). `fp8_xprep_bf16` quantises x (as before), the new `fp8_w_to_bf16` decodes the e4m3 weights to bf16 into the plan scratch (exact), `gemm_nax_fp8_*` runs `matmul2d` with a float cooperative destination, scales by `ws * is` in registers and rounds once (bf16 RNE or f32). Tile picked by `gemm_nax::pick_variant`. Registered at priority 5, above `matmul_fp8_mma` (0) and below the GEMV class. | `kernels/metal/gemm_nax_fp8.metal`, `kernels/metal/fp8.metal` (`fp8_w_to_bf16`), `crates/intarsia-backend-metal/src/fp8.rs`, `kernels.rs`, `lib.rs`, `build.rs` | `INTARSIA_METAL_FP8_NAX=0` restores `matmul_fp8_mma` |
| `moe_mm_nax_{gu,dn}_m{8,16,32,64}`: the grouped (prefill) MoE expert kernels with the 8x8 `simdgroup_matrix` MMAs replaced by `matmul2d` on the staged threadgroup tiles. Staging, grid, epilogue unchanged (helpers moved unchanged into `moe_mm.h`). Decode paths (T*K <= 16) untouched. | `kernels/metal/moe_nax.metal`, `moe_mm.h`, `moe.metal`, `src/moe.rs` | `INTARSIA_METAL_MOE_NAX=0` restores `moe_mm_*` |

Both are default-on in this branch. Per the coordinator's decision the default flips to opt-in if G1 or G2 fails (not decided yet, see below). Note: commit ed7cb9871 holds both kernels (its message names only the MoE one; the fp8 commit was folded in when the worktrees were split).

Two things learned on the way (wrong assumptions, kept so nobody repeats them):
- `ct.store()` of a float cooperative tensor into a **threadgroup** tensor with a runtime-strided `tensor_inline` view produced zeros in the MoE kernel (the same call into device memory works for the fp8 GEMM). The MoE epilogue therefore writes each valid element with `get_multidimensional_index` instead. A standalone probe of the exact tile and stride shapes was correct, so the store path is the likely cause, not the MMA (not isolated further).
- A bf16 destination cooperative tensor cannot be stored from a float one; the fp8 GEMM scales the float tensor and converts element-wise into a second bf16 cooperative tensor.

## Speed (G4), localeval speed, 35B, `--max-batch 1`, one fleetq slot #5948, interleaved base/cand rounds

Method as M35P: server per arm, `localeval speed --sizes 1024,4096,16384 --gen 256 --reps 3`, CPU sampled every 4 s (`raw/g4/*-load.log`, checked by `raw/loadchk.py`). **Base round 1 is discarded**: another lane's debug test binaries (`spec-*`, `tp_batch-*`, 1100% CPU, not in a fleetq slot) overlapped it. Rounds 2 and 3 are clean for both arms; the cand arm was clean in all three. Medians of the 3 reps per arm and round.

Candidate = fp8 NAX + MoE NAX (both default-on).

| prefill tok/s | 1k | 4k | 16k |
|---|---:|---:|---:|
| base round 2 / 3 (clean) | 1513 / 1623 | 1578 / 1566 | 1373 / 1389 |
| cand rounds 1 / 2 / 3 | 3100 / 3184 / 3278 | 2755 / 2824 / 2986 | 2253 / 2238 / 2362 |
| cand / base (paired rounds 2, 3) | 2.1 / 2.0x | **1.79 / 1.91x** | **1.63 / 1.70x** |
| M35P base (reference) | 1631 | 1788 | 1507 |

| decode tok/s | 1k | 4k | 16k |
|---|---:|---:|---:|
| base round 2 / 3 | 120.5 / 122.4 | 118.7 / 119.6 | 108.0 / 108.6 |
| cand rounds 1 / 2 / 3 | 119.5 / 120.5 / 123.0 | 117.9 / 117.4 / 119.0 | 108.9 / 108.6 / 108.3 |
| cand / base (paired) | 1.00 / 1.00 | 0.989 / 0.995 | 1.01 / 1.00 |

G4 status: 4k +20% needs >= 2146 against M35P's 1788: cand median 2824 (+58%), pass; 16k improved: pass. Decode >= 0.99x: 4k paired ratios 0.989 and 0.995, borderline (decode noise is about 5%); the decode kernels are not touched, so this is noise or the larger plan scratch. Formally "median of 3 interleaved rounds" is **not yet met** for the base arm (2 clean rounds); G4b (#5978) adds two more rounds. Today's base is also 12% below M35P's 1788 at 4k (1570), which is why the paired ratios are the comparison.

Kernel-mode profile at 4k (`raw/kernel-4096-*.jsonl.gz`; kernel mode serialises, so use shares): with fp8 NAX only, `matmul_fp8_nax` is 235 ms (12.6%, 260 calls, 0.9 ms each); with both, 258 ms in the fp8 GEMM and the three MoE sub-kernels fall from 946 ms to 328 ms, and the GDN delta rule (449 ms, 33%) is now the biggest item.

## Accuracy (G1): `intarsia accuracy` against the vLLM goldens, same commands as M35P (`raw/g1/`)

21-way KL, prefill / decode. Decode uses 16 positions per prompt on short and long and 64 on 4k (same as M35P), so decode cells are noisy. M35P's reference noise floor: 0.00301 (35b-short), 0.008417 (35b-long).

| golden | base (M35P) | fp8 NAX only | candidate (fp8 + MoE) |
|---|---|---|---|
| 35b-short | 0.00578 / 0.01169 | 0.00467 / 0.00677 | 0.00559 / 0.00695 |
| 35b-long | 0.01430 / 0.01229 | 0.01395 / 0.01238 | 0.01461 / 0.01568 |
| 35b-4k | 0.01244 / 0.00690 | 0.01231 / 0.01032 | 0.01209 / 0.00896 |

Prefill is within 0.0004 of base on all three. Decode moves by -0.0047 (short), +0.0034 (long), +0.0021 (4k), all inside the noise floor; short decode improved. The class floor verdict (KL <= 0.03) passes on every cell. The decode cells depend on the KV cache the new prefill wrote, so they are not independent of the change even though the decode kernels are unchanged.

`baseline-gate` against the recorded Metal baselines: prefill passes on short and 4k. **The 4k decode baseline gate FAILS for the candidate** (192 positions: top-1 97.4% vs 98.96% recorded, mean KL 0.00953 vs 0.00722, per-prompt gate fails on long_docs_4k top-1 and chat_reuse_4k KL; log in `raw/g1/cand/4k-vs-metal-baseline.log`). This gate has fixed small margins and no noise envelope; the rows are 64 positions per prompt. It is a reason to run the larger decode sets before calling G1 passed: decode-64 on short and long and decode-256 on 4k, base vs fp8-only vs candidate in one slot (#5973, #5977). Until those run, **G1 is not decided**; by the coordinator's rule a fail makes both knobs opt-in.

## Exactness properties (G3), what ran

Run on the fp8-only tree (before the MoE kernel) in slot #5903, all pass: `rows_exact` (batched rows equal solo; 4550 (row, step) pairs bitwise in the 35B hybrid test), fp8 rows with `INTARSIA_METAL_FP8_ROWS_MMT` 2 and 0, `row_tables`, G-C, `long`, `rec` (batch_recurrent, 14 tests including packed prefill, prefix reuse, macro prefill equals plain solo), `dense`.

Unit-level, on the candidate (`cargo test --release -p intarsia-backend-metal`, all 27 test binaries pass, `raw/cratetests.log`):
- `tests/fp8_nax.rs` (new, 6 tests): output equals the bf16 NAX GEMM on the decoded weights, bitwise; scaled once and rounded once; a row's bits do not depend on the call's M, the row position, the tile or the output offset (10 offset/length cases x 4 tiles x bf16/f32); run-to-run deterministic; a NaN weight code poisons only its column (as `mma`); registered for M > 8 only, above `mma`.
- `tests/moe.rs::moe_experts_rows_do_not_depend_on_the_tile` (new): a token's bits are the same in 3 / 10 / 24 / 40 / 100-token calls (8/16/32/64-row tiles) for both the NAX and the old kernels.
- So prefill chunk-size invariance at the op level **holds** (bitwise) for the NAX fp8 GEMM and the NAX MoE. Whether the whole-model ids are chunk-invariant is the c512 arm of `ids2.sh` (pending).

Decode untouched: `plan show` for 35B (`raw/plans/`) differs from base in exactly one place, the new `matmul_fp8_nax` row (plus a smaller `matmul_fp8_mma` scratch, 16777232 to 65552 bytes, since mma now only serves M <= 8); every decode pick (`matmul_fp8_gemv`, `_gemv_rows`, the MoE decode kernels, attention, GDN) is identical. `plan show` cannot show the MoE split, which lives inside `moe_experts_bf16`: the decode branch (T*K <= 16, `moe_dec_*`) is not edited and the new code is reached only through the grouped branch's `moe_dispatch`.

## Admission and mutants (G6)

`intarsia kernels test --backend metal --model <35B> --op matmul_fp8`: `matmul_fp8_nax` 27 cases, 0 failed, max_abs 4.1e-3, rel_l2 7.8e-5, ADMITTED (`mma` 3.0e-3 / 1.2e-4). `--op moe_experts`: `moe_experts_bf16` 12 cases, 0 failed, ADMITTED. Six planted mutants, each its own binary, `raw/mutants/`:

| mutant | change | caught |
|---|---|---|
| m1 | fp8 GEMM drops the `ws*is` scale | FAIL, max_abs 7.7e3 |
| m2 | `fp8_w_to_bf16` zeroes the last 16-weight group of each row | FAIL, rel_l2 0.52 |
| m3 | `fp8_w_to_bf16` swaps two decoded pairs | FAIL, rel_l2 0.63 |
| m4 | bf16 output truncates instead of round-to-nearest-even | FAIL, max_abs 5.6e-3 (rel_l2 4e-3) |
| m5 | MoE gate-up drops the up scale | FAIL, rel_l2 31.8 |
| m6 | MoE down drops the expert scale | FAIL, rel_l2 19.1 |

The clean binary passes both ops. G6 is met for the kernel sweep; the full `kernels test` over all ops on the candidate is pending (#5977).

## G5 other models

Qwen3-8B BF16 `plan show` is byte-identical (`raw/plans/*-8b.txt`). No other fp8 model is registered on Metal here, the NAX fp8 kernel only enters for `Op::MatmulFp8` with M > 8, and the MoE one only for the NVFP4 experts. Llama-3.1-8B, Qwen2.5-7B and the two 27B checkpoints: pending (#5979).

## Pending (queued in fleetq, #5973 to #5979, not run)

| job | gate | what |
|---|---|---|
| #5973 | G1 | decode-64 on short and long, base vs fp8-only vs candidate, one slot |
| #5974 | G2 | `localeval quality --bench mmlu --limit 25` (n=1425), candidate, bar >= 0.830 (base 0.840, SE ~0.010) |
| #5975 | G3 | `gates.sh gb gc long rec dense` on the candidate (rows_exact, 35B batch vs solo, batch_recurrent, dense) |
| #5976 | G3 | greedy ids, 4 prompts, base and candidate, arms `--spec off` / `mtp` (k=2) / `ngram` (k=4) / `--prefill-chunk 512`: spec equals off, chunk-512 equals default, and how far candidate ids diverge from base |
| #5977 | G1/G6 | full `kernels test` on the candidate, 4k decode-256 base vs candidate |
| #5978 | G4 | two more interleaved speed rounds |
| #5979 | G5 | `plan show` base vs candidate on Llama-3.1-8B, Qwen2.5-7B and the 27B checkpoints |

## Decision

Not made. Ship-as-default needs G1 (decode-64/256), G2 (MMLU >= 0.830) and G3 on the candidate. If G1 or G2 fails, flip both defaults to opt-in (change `unwrap_or(true)` style default in `fp8::nax_enabled` and `moe::nax_enabled`, update `docs/tuning.md` rows) and say so here. The fp8-only binary is available as a split option: if fp8-only passes and the candidate fails, fp8 could be default-on and MoE opt-in.
