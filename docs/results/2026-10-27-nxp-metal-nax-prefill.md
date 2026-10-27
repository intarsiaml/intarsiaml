# NXP: Qwen3.6-35B-A3B prefill on the Metal NAX path (2026-10-27)

Lane NXP (`lane-nxp/metal-nax-prefill`), started 2026-10-26 23:55 EDT on endor (Apple M5 Max, 128 GB). Model `~/models/Qwen3.6-35B-A3B-NVFP4`. Base binary: `origin/main` e86d8c336 (M35P merge). Raw data and scripts: [raw](2026-10-27-nxp-metal-nax-prefill/raw/).

**Status: all gates G1 to G6 pass; both kernels ship default-on.** The first pass at 01:04 was interrupted when endor dropped out of fleetq (the fleetq server could not resolve the name `endor` after a gateway swap; the box was fine). The queued jobs #5973 to #5979 ran when it came back at 02:34 and finished 03:29. Nothing was timed outside a fleetq slot.

| gate | result | evidence |
|---|---|---|
| G1 KL vs vLLM goldens | pass | prefill within 0.0004 of base on short, long, 4k; decode inside M35P's noise floor at 16, 64 and 256 positions, and lower than base at 4k decode-256 |
| G2 MMLU n=1425 | **pass, 0.842** (1200/1425; base 0.840, bar 0.830) | `raw/mmlu/`, localeval run `20261027-024048-adhoc-nxp-intarsia35`, unanimis 963b2be7 |
| G3 exactness | pass; ids differ from base | batched rows equal solo; MTP k=2 equals `--spec off` bitwise on 4 prompts; chunk 512 equals the default chunk bitwise; decode picks unchanged |
| G4 speed | pass | 4k prefill 2832 vs 1544 tok/s same-slot base (1.83x; +58% over M35P's 1788), 16k 2283 vs 1366 (1.67x), decode 0.992 / 0.992 / 1.000x |
| G5 other models | pass | `plan show` byte-identical on Qwen3-8B, Llama-3.1-8B, Qwen2.5-7B, Qwen3.8-27B-Quark-AWQ-MXFP4 |
| G6 admission | pass | full `kernels test` on the candidate: 0 failed cases; 6 of 6 planted mutants caught |

## What changed

| change | files | knob |
|---|---|---|
| `matmul_fp8_nax`: fp8 projections at M > 8 (prefill). `fp8_xprep_bf16` quantises x (as before), the new `fp8_w_to_bf16` decodes the e4m3 weights to bf16 into the plan scratch (exact), `gemm_nax_fp8_*` runs `matmul2d` with a float cooperative destination, scales by `ws * is` in registers and rounds once (bf16 RNE or f32). Tile picked by `gemm_nax::pick_variant`. Registered at priority 5, above `matmul_fp8_mma` (0) and below the GEMV class. | `kernels/metal/gemm_nax_fp8.metal`, `kernels/metal/fp8.metal` (`fp8_w_to_bf16`), `crates/intarsia-backend-metal/src/fp8.rs`, `kernels.rs`, `lib.rs`, `build.rs` | `INTARSIA_METAL_FP8_NAX=0` restores `matmul_fp8_mma` |
| `moe_mm_nax_{gu,dn}_m{8,16,32,64}`: the grouped (prefill) MoE expert kernels with the 8x8 `simdgroup_matrix` MMAs replaced by `matmul2d` on the staged threadgroup tiles. Staging, grid, epilogue unchanged (helpers moved unchanged into `moe_mm.h`). Decode paths (T*K <= 16) untouched. | `kernels/metal/moe_nax.metal`, `moe_mm.h`, `moe.metal`, `src/moe.rs` | `INTARSIA_METAL_MOE_NAX=0` restores `moe_mm_*` |

Both are default-on in this branch. Per the coordinator's decision the default flips to opt-in if G1 or G2 fails (not decided yet, see below). Note: commit ed7cb9871 holds both kernels (its message names only the MoE one; the fp8 commit was folded in when the worktrees were split).

Two things learned on the way (wrong assumptions, kept so nobody repeats them):
- `ct.store()` of a float cooperative tensor into a **threadgroup** tensor with a runtime-strided `tensor_inline` view produced zeros in the MoE kernel (the same call into device memory works for the fp8 GEMM). The MoE epilogue therefore writes each valid element with `get_multidimensional_index` instead. A standalone probe of the exact tile and stride shapes was correct, so the store path is the likely cause, not the MMA (not isolated further).
- A bf16 destination cooperative tensor cannot be stored from a float one; the fp8 GEMM scales the float tensor and converts element-wise into a second bf16 cooperative tensor.

## Speed (G4), localeval speed, 35B, `--max-batch 1`, interleaved base/cand rounds in fleetq slots #5948 and #5978

Method as M35P: server per arm, `localeval speed --sizes 1024,4096,16384 --gen 256 --reps 3`, CPU sampled every 4 s (`raw/g4/*-load.log`, `raw/g4b/*-load.log`, checked by `raw/loadchk.py`; a sample with a foreign process above 300% CPU discards the arm). **Base round 1 of #5948 is discarded**: another lane's debug test binaries (`spec-*`, `tp_batch-*`, 1100% CPU, outside fleetq) overlapped it. That leaves 4 clean base rounds and 5 clean candidate rounds (3 in #5948, 2 in #5978); each value below is the median of the 3 reps of one round. Candidate = fp8 NAX + MoE NAX, both default-on.

| prefill tok/s | 1k | 4k | 16k |
|---|---:|---:|---:|
| base, 4 clean rounds | 1513 / 1623 / 1506 / 1502 | 1578 / 1566 / 1514 / 1522 | 1373 / 1389 / 1338 / 1358 |
| base median | 1510 | 1544 | 1366 |
| cand, 5 rounds | 3100 / 3184 / 3278 / 3243 / 3158 | 2755 / 2824 / 2986 / 2970 / 2832 | 2253 / 2238 / 2362 / 2283 / 2288 |
| cand median | 3184 | **2832** | **2283** |
| cand / base | 2.11x | **1.83x** | **1.67x** |
| M35P base (reference) | 1631 | 1788 | 1507 |

| decode tok/s | 1k | 4k | 16k |
|---|---:|---:|---:|
| base median | 122.4 | 119.5 | 108.65 |
| cand median | 121.4 | 118.5 | 108.6 |
| cand / base | 0.992 | 0.992 | 1.000 |

G4: 4k +20% needs >= 2146 against M35P's 1788 (cand 2832, +58%; +83% against today's slower same-slot base): pass. 16k improved: pass. Decode >= 0.99x: pass, narrowly at 1k and 4k (decode noise is about 5%, and the decode kernels are not touched; the larger plan scratch is the only difference). Today's base is about 14% below M35P's 1788 at 4k, which is why the ratios use the same-slot base.

Kernel-mode profile at 4k (`raw/kernel-4096-*.jsonl.gz`; kernel mode serialises, so use shares): with fp8 NAX only, `matmul_fp8_nax` is 235 ms (12.6%, 260 calls, 0.9 ms each); with both, 258 ms in the fp8 GEMM and the three MoE sub-kernels fall from 946 ms to 328 ms, and the GDN delta rule (449 ms, 33%) is now the biggest item.

## Accuracy (G1): `intarsia accuracy` against the vLLM goldens, same commands as M35P (`raw/g1/`)

21-way KL, prefill / decode. Decode uses 16 positions per prompt on short and long and 64 on 4k (same as M35P), so decode cells are noisy. M35P's reference noise floor: 0.00301 (35b-short), 0.008417 (35b-long).

| golden | base (M35P) | fp8 NAX only | candidate (fp8 + MoE) |
|---|---|---|---|
| 35b-short | 0.00578 / 0.01169 | 0.00467 / 0.00677 | 0.00559 / 0.00695 |
| 35b-long | 0.01430 / 0.01229 | 0.01395 / 0.01238 | 0.01461 / 0.01568 |
| 35b-4k | 0.01244 / 0.00690 | 0.01231 / 0.01032 | 0.01209 / 0.00896 |

Prefill is within 0.0004 of base on all three. Decode moves by -0.0047 (short), +0.0034 (long), +0.0021 (4k), all inside the noise floor; short decode improved. The class floor verdict (KL <= 0.03) passes on every cell. The decode cells depend on the KV cache the new prefill wrote, so they are not independent of the change even though the decode kernels are unchanged.

Decode with more positions (slot #5973 for short and long at 64 positions per prompt, #5977 for 4k at 256; `raw/g1-64/`, `raw/kt/`):

| golden, decode KL | base | fp8 NAX only | candidate |
|---|---|---|---|
| 35b-short (64 pos) | 0.00539 | 0.00516 | 0.00526 |
| 35b-long (64 pos) | 0.00883 | 0.01019 | 0.01042 |
| 35b-4k (256 pos) | 0.01360 | n/a | 0.01208 |

The long decode gap (+0.0016) is far inside the 0.0084 noise floor, and 4k decode is better than base. The earlier 16-position decode cells were therefore sampling noise.

`baseline-gate` against the recorded Metal baselines: prefill passes on short and 4k. The 4k decode baseline gate at 64 positions (192 rows) **fails** for the candidate (top-1 97.4% vs 98.96% recorded, mean KL 0.00953 vs 0.00722; log in `raw/g1/cand/4k-vs-metal-baseline.log`). That gate has fixed small margins and no noise envelope. The 256-position run above gives 0.01208 for the candidate against 0.01360 for base, so the fail is position-sampling noise, not a regression. The recorded Metal baselines are not updated here.

G1: pass.

## Exactness properties (G3)

Candidate, slot #5975 (`raw/g3c/`), all pass: `rows_exact`, fp8 rows with `INTARSIA_METAL_FP8_ROWS_MMT` 2 and 0, `row_tables`, G-C (4550 (row, step) pairs bitwise identical, batched 35B decode rows equal solo), `long`, `batch_recurrent` (14 tests: packed prefill, prefix reuse, macro prefill equals plain solo, hybrid decode beside prefill chunks), `dense` (138 pairs bitwise; 48 decode rows bitwise beside 256-token prefill chunks).

Greedy ids, slot #5976 (`raw/ids2/`), 256 ids, prompts short-0, 4k-0, 4k-1, 16k-0, `--chat --no-think --temperature 0 --ignore-eos`:
- `--spec mtp --spec-k 2` equals `--spec off`: 4 of 4 bitwise for base and for the candidate (MTP runs on Metal for this model).
- `--prefill-chunk 512` equals the default chunk: 4 of 4 bitwise for base and for the candidate. **Prefill chunk-size invariance was bitwise before and still is**, at the whole-model level.
- `--spec ngram` is refused by the engine for a model with recurrent state (`unsupported: speculative decoding on a model with recurrent state`), for base and candidate alike; not a regression.
- Candidate ids vs base (`--spec off`): first divergence at token 44, 56, 44 and 75 of 256 (short-0, 4k-0, 4k-1, 16k-0). This is the expected change; the KL and MMLU gates above are what bound it.

Unit level, on the candidate (`cargo test --release -p intarsia-backend-metal`, all 27 test binaries pass, `raw/cratetests.log`):
- `tests/fp8_nax.rs` (new, 6 tests): output equals the bf16 NAX GEMM on the decoded weights, bitwise; scaled once and rounded once; a row's bits do not depend on the call's M, the row position, the tile or the output offset (10 offset/length cases x 4 tiles x bf16/f32); run-to-run deterministic; a NaN weight code poisons only its column (as `mma`); registered for M > 8 only, above `mma`.
- `tests/moe.rs::moe_experts_rows_do_not_depend_on_the_tile` (new): a token's bits are the same in 3 / 10 / 24 / 40 / 100-token calls (8/16/32/64-row tiles) for both the NAX and the old kernels.

Decode untouched: `plan show` for 35B (`raw/plans/`) differs from base in exactly one place, the new `matmul_fp8_nax` row (plus a smaller `matmul_fp8_mma` scratch, 16777232 to 65552 bytes, since mma now only serves M <= 8); every decode pick (`matmul_fp8_gemv`, `_gemv_rows`, the MoE decode kernels, attention, GDN) is identical. `plan show` cannot show the MoE split, which lives inside `moe_experts_bf16`: the decode branch (T*K <= 16, `moe_dec_*`) is not edited and the new code is reached only through the grouped branch's `moe_dispatch`. Batched decode and MTP verify rows (M <= 8) never reach `matmul_fp8_nax` (registered for M > 8); the `rows_exact`/G-C results above are the proof for the MoE part.

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

The clean binary passes both ops. Full `kernels test` on the candidate (slot #5977, `raw/kt/`): 45 kernels, 820 cases on the 35B shapes and 844 on the default shapes, 0 failed. The one untested kernel, `attn_prefill_bf16_flash_nax`, ran 0 cases (pre-existing, unrelated; plan-time admission covers it). G6: pass.

## G5 other models

`plan show --backend metal`, base vs candidate, byte-identical: Qwen3-8B BF16, Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Qwen3.8-27B-Quark-AWQ-MXFP4 (`raw/plans/`, `raw/plans2/`). `~/models/qwen38-27b` does not load on either build (safetensors dtype U32 unsupported), the same error text for both. The NAX fp8 kernel only enters for `Op::MatmulFp8` with M > 8 and the MoE one only for the NVFP4 experts, so no other registered Metal model changes.

## Decision

Both kernels ship default-on. G1 to G6 pass; the 35B greedy ids are no longer identical to the old Metal build (divergence after 44 to 75 tokens in the 256-token probes), which the coordinator decision of 2026-10-26 23:55 allowed. Rollback is `INTARSIA_METAL_FP8_NAX=0` and/or `INTARSIA_METAL_MOE_NAX=0` (bitwise the previous build). Not done here: the GDN delta rule is now the largest prefill item (33% of GPU time at 4k), and flash attention at 16k; the recorded Metal baselines in `golden/baselines` were not re-recorded.
