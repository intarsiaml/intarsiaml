# vrc: cost of the speculative verify pass, Qwen3.8-27B MXFP4 tp=2 on the R9700s (HIP), 2026-10-28

Lane `lane-vrc/hip-verify-cost`, base `origin/main` `4f47b3495`. tatooine, 2x R9700 (gfx1201), `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`, `serve --tp 2 --devices 0,1 --max-batch 1`, W4A8 (recipe default),
localeval `--judge none`, 512 forced tokens, temperature 0, the seven category files of `scripts/dfh/cats/` (setup of `2026-10-27-dcp-dflash-cost.md`, driver `scripts/vrc/cat-window.sh`).
Raw logs: `2026-10-28-vrc-hip-verify-cost/raw/`. Scripts: `scripts/vrc/` (copy of `scripts/dcp/` plus the stages below). Tables: `tools/dfh/cat_table.py`, `tools/dfh/vrc_spread.py`.
Status: PR, see the end. The change is one kernel launch in place of `q` launches plus `q - 1` slot copies per GatedDeltaNet layer of a verify pass; it is bitwise, behind a knob.

## Headline

| | mtp k2 (3-row verify) | dflash k7 (8-row verify) |
|---|---|---|
| verify + rollback, short prompt (mean per pass, tp=2) | 21.68 -> 20.80 ms | 29.22 -> 25.89 ms |
| verify + rollback, 3k context | 22.56 -> 21.63 ms | 31.64 -> 28.28 ms |
| decode tok/s, new / old, median of 3 interleaved rounds, 7 categories | **x1.031 .. x1.039** | **x1.082 .. x1.089** |

`--spec off` is unchanged (56.7-56.8 tok/s at 1k, 56.0 at 4k, equal to the 56.9 / 56.1 of the last two lanes). Greedy ids equal `--spec off` 18 of 18 for both `--spec mtp --spec-k 2` and `--spec dflash --spec-k 7`.
The 27B tp=2 default stays `--spec mtp --spec-k 2`: DFlash2 is still below MTP k2 on chat, prose and ho-recall (section 5).

## 1. Microbench (step 1): the W4A8 GEMV is not the lever

`tests/verify_gemv_bench.rs` (`VRC_BENCH=1`), `matmul_mxfp4_a8_gemv_rows` / `_pair_` on the 27B tp=2 per-rank shapes, weights rotated over 400 MB of copies (no infinity-cache hits), best of 5 passes of back-to-back
launches; GB/s counts weight bytes (e2m1 + e8m0). M >= 2 includes the separate activation quantisation launch (`mx_a8_quant_kernel`, 2-3 us), M = 1 uses the fused kernel (#214).
`raw/bench2/bench.txt`, column `r2` of the first block is the shipped kernel (`raw/bench/` is the same measurement on the base build).

| shape (per rank) | M | us | GB/s | x M=1 | R=4 form us | fused-quant (`INTARSIA_HIP_MXFP4_A8_FUSE=1`) us |
|---|---:|---:|---:|---:|---:|---:|
| gate+up pair 8704+8704, K 5120 | 1 | 77.6 | 610 | 1.00 | 77.7 | 77.8 |
| | 3 | 82.6 | 573 | 1.06 | 82.8 | 83.9 |
| | 8 | 91.4 | 518 | 1.18 | 101.9 | 228.4 |
| down 5120, K 8704 | 1 | 42.4 | 558 | 1.00 | 42.7 | 42.7 |
| | 3 | 47.9 | 494 | 1.13 | 48.7 | 56.9 |
| | 8 | 55.6 | 426 | 1.31 | 63.8 | 63.3 |
| qkv+z 8192, K 5120 | 1 | 40.4 | 552 | 1.00 | 40.5 | 40.4 |
| | 3 | 44.7 | 498 | 1.11 | 47.0 | 45.5 |
| | 8 | 52.3 | 426 | 1.29 | 54.0 | 107.3 |
| q+gate 6144, K 5120 | 1 | 31.0 | 539 | 1.00 | 31.1 | 31.1 |
| | 3 | 35.1 | 476 | 1.13 | 34.5 | 35.4 |
| | 8 | 41.4 | 404 | 1.34 | 43.7 | 79.0 |
| out/o 5120, K 3072 | 1 | 17.6 | 475 | 1.00 | 17.6 | 17.5 |
| | 3 | 21.9 | 382 | 1.24 | 22.0 | 22.5 |
| | 8 | 27.5 | 304 | 1.56 | 29.7 | 40.5 |
| k+v pair 512+512, K 5120 | 1 | 8.9 | 313 | 1.00 | 8.9 | 8.9 |
| | 3 | 12.6 | 221 | 1.42 | 12.7 | 12.5 |
| | 8 | 15.3 | 182 | 1.72 | 15.3 | 21.1 |

(M = 2 and 4 are in the raw file.) The kernel already reads each weight byte once for all M rows (two weight rows per warp share the activation loads), so it is not at 1x only because of the extra pre-pass launch and
the per-row dot work: +6 to +24 % at M = 3 on the five large shapes (of which 2-3 us is the pre-pass), +18 to +56 % at M = 8. Summed over a 27B layer that is about 1 ms of the 3-row verify and about 3 ms of the 8-row one.

Tried and rejected, so the kernel is unchanged in this PR:
* A four-weight-rows-per-warp form for M 2..8 (activation loads shared by 4 rows instead of 2, template `R`, outputs bitwise, hash `fac47447935a8383` identical, 3 of 3 planted faults caught): **slower** than R = 2 at M >= 4
  (gate+up M=8 101.9 vs 91.4 us, down 63.8 vs 55.6) and equal at M = 2..3. The hypothesis (activation traffic) was wrong; the kernel is bound by per-row dot work, not by loads. The code is at commit `849b0bd` of this branch, raw in `raw/bench2/`.
* The fused quantisation (`INTARSIA_HIP_MXFP4_A8_FUSE=1`) for M >= 2: wins only at M = 2 on K <= 5120 (0.3-1.7 us), loses at M >= 3 on every K = 17408 / 8704 shape and 2.5x at M = 8, as #214 found.
The CUDA multi-row-per-warp idea of #215 has nothing to port: HIP's kernel is already that design.

`gdn_conv_delta_bf16` (`kernels/hip/gdn_fused.hip`) at the 27B tp=2 per-rank shape (8 key / 24 value heads of 128): T = 1 / 2 / 3 / 4 / 8 tokens in one launch take 7.8 / 8.5 / 10.3 / 11.3 / 16.4 us.
The kernel already walks T tokens in order inside the block. That is the lever (section 2).

## 2. Where the verify spent its time (step 2)

`Model::gdn_block` ran a speculative verify (`verifying()`, `t > 1`) **one row at a time**: per GDN layer `t` launches of `gdn_conv_delta_bf16` (T = 1 each) and, after every row but the last, a `backend.copy` of the layer's whole recurrent
slot (conv state + delta state, 1.6 MB at tp=2) into the rollback buffer `SpecRows`. At 48 GDN layers that is 8 + 7 copies per layer at k7 and 3 + 2 at k2: about 70 us per layer at q = 8 (3.3 ms per pass), about 20 us at q = 3.
The profiled `gdn_conv_delta` share of the DCP table (9.5 ms profiled, 15 %) is this loop. The T-token kernel takes 16.4 us at T = 8 (table above), so one launch is enough if it also writes the per-row states.

Change (`kernels/hip/gdn_fused.hip`, `ik_gdn_conv_delta_snap_bf16`; `crates/intarsia-core/src/model/gdn_block.rs`, `crates/intarsia-backend-hip`):
* `gdn_conv_delta_kernel` gets a `SNAP` variant (a template bool: the decode kernel's code is unchanged) that, after token `r < T - 1`, stores the lane's delta-state registers, the value-row conv history and (from one designated block per key head) the q/k conv history
  into snapshot `r`, laid out exactly as the live slot. Nothing the walk computes reads them, so outputs and final states are the plain launch's.
* `GdnConvDelta` accepts `T - 1` extra outputs (arity check relaxed for that op only); the HIP device passes them as the snapshots. The verify block builds them as views of the `SpecRows` buffers at the slot offset and runs the op once.
  Used only on HIP, 2..=16 rows, not while a graph is recorded (the captured tp=1 verify keeps its path), 16-byte-aligned slot offsets; otherwise the previous row-by-row loop runs.
* Knob `INTARSIA_HIP_GDN_VERIFY_FUSE=0` restores the row-by-row loop (docs/tuning.md). Seam `INTARSIA_HIP_GDN_SNAP_MUT=1..4` plants faults for the test.

## 3. Verify cost, before and after (`INTARSIA_SPEC_TRACE=1`, tp=2, window s1, `raw/s1/ids/`)

Mean of the traced passes after the first 4 (n = 99 for k2, 63-66 for k7); "old" is `INTARSIA_HIP_GDN_VERIFY_FUSE=0`; the draft part is untouched (2.3 ms mtp, 13.7 / 16.7 ms dflash).

| arm | prompt | verify + rollback old | new | saved | decode tok/s old -> new (one run) |
|---|---|---:|---:|---:|---|
| mtp k2 | short (41) | 21.68 ms | 20.80 ms | 0.88 | 92.5 -> 95.4 |
| mtp k2 | 3k | 22.56 | 21.63 | 0.93 | 89.8 -> 92.9 |
| dflash k7 | short | 29.22 | 25.89 | 3.33 | 87.1 -> 94.2 |
| dflash k7 | 3k | 31.64 | 28.28 | 3.36 | 74.7 -> 80.3 |

A plain step is 19 ms; the 8-row verify is now 25.9 ms, the 3-row one 20.8 ms. What remains above a plain step at q = 8 (about 7 ms): the M = 8 GEMVs (section 1, about 3 ms), 16 x 8 per-row `attn_decode` launches
(`verify_attn_per_row`, HIP's split-KV partition follows the row count), 129 all-reduces, rmsnorm. Not attempted here (section 7).

## 4. Decode, tp=2, 3 interleaved rounds (window f1, `raw/f1/`)

Arms per round `mtp2 mtp2-old dfl7 dfl7-old off` (order reversed on even rounds), a fresh `serve` per arm and round; "old" = `INTARSIA_HIP_GDN_VERIFY_FUSE=0`. Pooled decode tok/s, median of 3 rounds, x = vs `off`, t/p = tokens per verify pass (server counters).

| category | off | mtp k2 old | mtp k2 new | t/p | dflash k7 old | dflash k7 new | t/p |
|---|---:|---:|---:|---:|---:|---:|---:|
| ho-chat | 56.9 | 86.0 (1.51x) | **89.2 (1.57x)** | 2.29 | 71.0 (1.25x) | **77.0 (1.35x)** | 3.04 |
| ho-code | 56.9 | 103.5 (1.82x) | **106.3 (1.87x)** | 2.74 | 124.2 (2.18x) | **135.4 (2.38x)** | 5.32 |
| ho-prose | 56.8 | 76.7 (1.35x) | **79.1 (1.39x)** | 2.04 | 58.3 (1.03x) | **63.3 (1.11x)** | 2.50 |
| ho-recall | 56.9 | 91.2 (1.60x) | **94.4 (1.66x)** | 2.43 | 85.3 (1.50x) | **92.0 (1.62x)** | 3.66 |
| rcp-code | 56.9 | 103.1 (1.81x) | **106.6 (1.87x)** | 2.74 | 123.7 (2.17x) | **134.4 (2.36x)** | 5.30 |
| rcp-prose | 56.9 | 79.2 (1.39x) | **82.3 (1.45x)** | 2.12 | 61.0 (1.07x) | **66.1 (1.16x)** | 2.62 |
| rcp-recall | 57.0 | 94.0 (1.65x) | **97.0 (1.70x)** | 2.50 | 97.2 (1.71x) | **105.6 (1.85x)** | 4.17 |

New / old per round (`raw/f1/`, `tools/dfh/vrc_spread.py`): mtp k2 1.024-1.041 in every category and round (medians 1.031-1.039); dflash k7 1.077-1.090 (medians 1.082-1.089). The round-to-round spread of any arm is 0.1-1.3 %
(`off` 0.1-0.4 %). Acceptance (t/p) is identical between old and new in every category, as it must be for a bitwise change.

Plain decode (`localeval speed`, gen 128, 3 reps, 3 rounds, `off`; the changed code is not reached at one row): 1k 56.72 / 56.74 / 56.83 tok/s, 4k 56.02 / 56.02 / 56.03 (`raw/f1/speed-off-r*.json`);
the last lanes measured 56.9 / 56.1 (#214). No `off-old` arm was run: `INTARSIA_HIP_GDN_VERIFY_FUSE` only acts on `t > 1`.

## 5. Default

DFlash2 k7 vs MTP k2 with the new verify: ho-code 135.4 vs 106.3, rcp-code 134.4 vs 106.6, rcp-recall 105.6 vs 97.0 (dflash ahead); ho-recall 92.0 vs 94.4, ho-chat 77.0 vs 89.2, ho-prose 63.3 vs 79.1,
rcp-prose 66.1 vs 82.3 (dflash behind). Not >= in every category, so the 27B HIP tp=2 default stays `--spec mtp --spec-k 2` (the rule of this lane). dflash's gap to MTP on chat / prose is its 2.5-3.0 tokens per pass against a 20 ms+ pass.

## 6. Gates

* **ids**: `--spec off` vs `--spec dflash --spec-k 7` and `--spec mtp --spec-k 2`, tp=2, 18 prompts (6 short, 6 ~3k, 6 ~10k), 256 greedy ids each: **18 / 18 identical for both** (`raw/f1/ids/`; also the 4 earlier old/new arms of `raw/s1/ids/`).
* **text**: every category case of the new arms (7 categories x 3 rounds, 108 cases each for mtp k2 and dflash k7) has the same text as the `--spec off` case of the same round: 216 / 216 (`raw/f1/text-compare.txt`; the old mtp arm 108 / 108, the old dflash arm 107 / 108).
  The one old-dflash miss (`ho-recall` `recall-tls`, round 1) diverges from `off` at char 240 on current main; the new path matched in all three rounds. Not investigated here (it is in the code this PR replaces).
* **kernel bitwise** (`tests/gdn_verify_snap.rs`, GPU): the one-launch snapshotting kernel vs T single-row launches + slot copies, outputs, final conv and delta states and all 360 snapshots, T in 2..=16, three head layouts (8/24, 16/32, 4/8 heads), save buffers at a nonzero
  slot offset: byte-equal, hash `96775dc37fec95a1`. **4 of 4 planted faults caught** (`INTARSIA_HIP_GDN_SNAP_MUT` = 1 snapshot index + 1, 2 no v history, 3 no q/k history, 4 delta-state offset + 16 bytes; `raw/gdn/mut.txt`).
  The W4A8 GEMV is untouched (`mxfp4_a8_fuse` hash `fac47447935a8383`, as in #214). `fusion_p2` (5), `row_tables` (8), `rows_exact` (13), `mxfp4_a8` (6) GPU tests green.
* **batched == solo / spec paths** (GPU, `INTARSIA_BATCH_TEST_BACKEND=hip`, `raw/f1/ids/steps.log`): `batch`, `batch_recurrent` (14), `mtp` (21), `hybrid`, `fusion_p2`, `graphs`, `hybrid_mxfp4`, `spec` and the tp=2 `tp_hybrid`, `tp_mtp`, `tp_batch`, `tp`: all pass.
* **`plan show`**: 8B (tp1), 27B (tp=2), 35B (tp1) printed (`raw/f1/ids/plan-*.out`, `plantext-*.out`; no base-binary comparison was run); the 27B plan JSON is byte-identical with the knob on and off. No planner code changed (`ops.rs` only relaxes the output count of `GdnConvDelta`,
  the pinned plan tests of the Mac workspace pass); the new path is a verify-time launch the plan does not list.
* **Mac** `cargo test --workspace --release --no-fail-fast`: green, 1075 passed, 0 failed (final tree).

## 7. Not done / assumptions

* The 8-row verify still runs `attn_decode` once per row (16 layers x 8) and 129 all-reduces eagerly. A row-table attention call (the batched-decode `attn_decode_bf16_splitkv_rows`, bitwise to the solo launch) fed with one sequence's
  cache pointer in every row and positions `pos0 + r` should cut about 3 ms at q = 8 and about 1 ms at q = 3; it needs a per-pass table upload and scratch in the `--max-batch 1` configuration and was not attempted (budget).
  A captured tp=2 verify graph (single-rank today) and fewer all-reduces are likewise untouched.
* The GEMV at M 3..8 is 6-56 % above M = 1; two attempts (R = 4, fused quantisation at M >= 2) did not help, see section 1. A kernel that keeps the per-row dot work off the critical path is the open question.
* The old dflash text mismatch above.

## 8. Fleet

Radiance-down fleetq jobs: #6407 (bench), #6409 (gdn tests), #6410 (phases + one speed round), #6415 (R = 4 bench; #6414 was the same job started with a wrong stage name, 13 s, nothing run), #6417 (ids + gates + 3 speed rounds, 2 h 28 min).
Five GPU windows. Radiance restored and `/v1/models` answering after each (`raw/*/window.log`, `guard.log`; checked again after #6417). CPU-slot builds #6403-6406, #6408, #6413, #6416.
Cleanup: `~/intarsiaml-vrc`, the docker volumes `intarsia-cargo-vrc-dev` and `intarsia-cargo-vrc-dev-hiptest` (seeded from `intarsia-cargo-ldf`) and the worktree `.worktrees/vrc` are removed after the PR is final.
