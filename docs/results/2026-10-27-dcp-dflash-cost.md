# DFlash2 cost per spec pass, Qwen3.8-27B MXFP4 on HIP tp=2, lane dcp, 2026-10-27

Lane dcp (`lane-dcp/dflash-cost`, off origin/main `e1b077dc6`; user's "go 1" for R9700 27B speculation). Goal: cut the DFlash2 cost
per speculative pass so that `--spec dflash` beats MTP k2. Acceptance already matches radiance's (2.5 tokens per pass on prose, 5.3 on
code; `2026-10-26-dfh-hip-dflash.md` section 16), so the lever is the cost per pass.

Setup as in `2026-10-26-spt-spec-retune.md`: tatooine, 2x R9700, `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`, `serve --tp 2 --max-batch 1`, localeval
`--judge none`, 512 forced tokens, temperature 0, the 7 category files of lane dfh, tables by `tools/dfh/cat_table.py`. Scripts:
`scripts/dcp/` (copies of `scripts/spt/`; one tree `~/intarsiaml-dcp`, removed afterwards). Raw: `docs/results/2026-10-27-dcp-dflash-cost/raw/`.
Five fleetq windows (#6393 attr, #6394/6397 builds on cpu slots, rows, ids, speed, ids x18), radiance restored and `/v1/models`
answering after each.

## 1. Result in one paragraph

The DFlash2 draft ran its 56 bf16 matmuls and its `[7, V]` logits pass on the **generic** kernels (`matmul_bf16`, a WMMA GEMM, and
`logits_bf16`), which read the weights at about 320 GB/s, half the card's 640 GB/s. The row-exact twins (`matmul_bf16_gemv_rows`,
`logits_bf16_gemv_rows`) were already in the tree behind `INTARSIA_DFLASH_ROWS_KERNELS=1` (default off). With them on, the draft goes
from **21.1 to 13.8 ms** per pass on a prose prompt (24.2 to 16.9 ms at 3k context), with the same drafts and the same acceptance. The
knob is now **on by default for a DFlash2 draft** (`INTARSIA_DFLASH_ROWS_KERNELS=0` restores the old kernels; DFlash v1 keeps default
off). DFlash2 k7 is now 1.02-2.17x of off, beats MTP k2 on code (+19-20%) and on rcp-recall (+3%), and still loses to MTP k2 on
chat, prose and ho-recall, so **the 27B tp=2 default stays `--spec mtp --spec-k 2`**.

## 2. Attribution of one dflash pass at tp=2 (step 1)

Method: `INTARSIA_SPEC_TRACE=1` now also prints a per-phase line for each DFlash2 draft (`spec: dflash2 draft flush .. block .. head ..`;
the phases are separated by a stream synchronise, trace mode only, so each is device time plus its launch gap), and the existing
`spec: mtp draft X ms, verify + rollback Y ms` line. Prompt `short-0` (prose, 41 tokens), 256 greedy tokens, k7, mean over the
63 passes after the first four (`raw/attr/`, `raw/rows/`). An eager per-op table (`--profile-every 1 --op-table`, `raw/attr/optable-dfl7.err`;
profiled passes are synchronised per op, so those ms are inflated about 1.3x) gives the verify-side split.

| part of one pass (prose, tp=2) | before (generic bf16 kernels) | after (row kernels) | eager or captured |
|---|---|---|---|
| draft: context flush (fc over 5 taps, K/V of the draft layers) | 1.40 | 1.44 | eager |
| draft: block pass (5 layers: 56 bf16 matmuls M=8, convs, 8 attention calls per layer) | 10.37 | **6.47** | eager |
| draft: `[7, V]` logits + argmax, bf16 head, rank 0 whole copy | 7.84 | **4.44** | eager |
| draft: selector projection | 0.08 | 0.08 | eager |
| draft: logits download (7 MB) + host top-16 and selector walk | 0.69 + 0.74 | 0.69 + 0.69 | host |
| **draft total** | **21.12** | **13.82** | |
| verify (8 rows) + rollback | 29.24 | 29.05 | eager at tp=2 (the captured verify graph is single-rank) |
| **pass total** | **50.4** | **42.9** | plain step 19.0 |

Same on a 3k-token context (`4k-1`): draft 24.24 to 16.94 (block 13.49 to 9.61, head 7.84 to 4.44), verify + rollback 31.63, pass 55.9 to 48.6 ms.
The draft's block grows with context (the sliding-window attention is one `AttnDecode` per row).

Verify side, from the eager op table (profiled ms per pass, 26 passes; shares of the 1666 ms total): MXFP4 pair + single GEMVs
13.1 + 11.3 (41%), `gdn_conv_delta` 9.5 (15%, row-exact per row), `attn_decode` 5.4 (8%, includes the draft's), all-reduce 3.6 (6%,
129 per pass-equivalent step, 26 us each), `rmsnorm_add` 3.2, bf16 `matmul` (draft) 7.0, `logits` (verify + draft heads) 4.6. Rollback
(per-rank recurrent row copy) sits inside the verify + rollback figure and was not split out; there is no separate idle gap to speak of:
the sum of the synced draft phases equals the draft total to 0.01 ms, and the host select is 1.4 ms of it.

Reading: after the change a pass is 13.8 draft + 29 verify. The remaining draft is near the bandwidth line (block 3.45 GB in 6.5 ms = 530 GB/s,
head 2.54 GB in 4.4 ms = 570 GB/s); further draft cuts need fewer bytes (see 3.2) or running the draft off the critical path.

## 3. Draft changes (step 2)

### 3.1 Row kernels for the draft: kept, default on for DFlash2

`INTARSIA_DFLASH_ROWS_KERNELS` (existing, lane dfh) sets `exact_rows` around the draft's block pass and head, so `matmul_bf16_gemv_rows` and
`logits_bf16_gemv_rows` run instead of the generic kernels. Measured (`raw/rows/`, k7, tp=2, same binary, ids against `--spec off`):

| arm | prompt | draft ms (flush / block / head) | passes | accepted | ids vs off |
|---|---|---|---|---|---|
| generic (before) | short-0 | 21.12 (1.40 / 10.37 / 7.84) | 67 | 185 | identical |
| row kernels | short-0 | 13.82 (1.44 / 6.47 / 4.44) | 67 | 185 | identical |
| generic (before) | 4k-1 | 24.24 (1.45 / 13.49 / 7.84) | 70 | 184 | identical |
| row kernels | 4k-1 | 16.94 (1.42 / 9.61 / 4.44) | 70 | 184 | identical |

Acceptance and pass counts are equal to the unit (the two kernel families give the same drafts here), so no draft numerics moved.

### 3.2 fp8 draft weights: measured, rejected

The brief asked for the draft's linears as fp8 with the HIP fp8 GEMV/rows kernels. The FP8 checkpoint (`Qwen3.8-27B-DFlash2-FP8`) is
128x128 block-scaled (`weight_scale_inv`), which the loader does not read (per-tensor scale only), so I quantised the bf16 draft at load
instead (per-tensor e4m3 W8A16 through `Fp8E4m3::quantize`: layers, fc, and a separate fp8 copy of the draft's head), behind a temporary
env knob (not merged; the code is in git history of the branch only as the commit that was reverted):

| arm (short-0, tp=2) | flush | block | head | draft total | ids | passes / accepted |
|---|---|---|---|---|---|---|
| generic bf16 | 1.40 | 10.37 | 7.84 | 21.12 | identical | 67 / 185 |
| generic path, fp8 layers + head | 3.87 | 15.21 | 5.92 | 26.62 | identical | 68 / 185 |
| row kernels, bf16 | 1.44 | 6.47 | 4.44 | 13.82 | identical | 67 / 185 |
| row kernels, fp8 layers + head | 3.99 | 12.05 | 4.80 | 22.53 | identical | 69 / 183 |
| row kernels, fp8 head only | 1.39 | 6.47 | 4.77 | 14.09 | identical | 68 / 184 |

The fp8 rows GEMV at M=7..8 is slower than bf16 on these shapes (block 12.1 vs 6.5 ms for half the bytes, about 140 GB/s), and an fp8 head
(4.77) is not faster than the bf16 row-kernel head (4.44). Acceptance dropped slightly (per-tensor fp8 weights). Assumption that failed: that the
HIP fp8 rows kernel is bandwidth-bound at 8 rows like the MXFP4 int8 one; it is not, so fp8 draft weights need an fp8 kernel tuned for M=8
first. Dropped, not shipped.

### 3.3 Not done (budget)

- vocab-parallel draft head across the two ranks (head 4.4 to about 2.2 ms): needs rank 1 to join the draft and an id exchange; not attempted.
- device-side top-k selector: host select + download is 1.4 ms of 13.8; kernel not written.
- captured tp=2 verify graph and overlapping draft with verify: larger work, not started. The verify (29 ms, 8 rows, eager) is now the larger half
  of a pass and the main remaining lever; radiance's whole pass is 18.5 ms.

## 4. Gates

- **ids**: `--spec dflash --spec-k 7` vs `--spec off` at tp=2, lane dfh's 18 prompts (6 short, 6 ~3k, 6 ~11k), 256 greedy ids, final head: **18 / 18
  identical** (`raw/ids/`; an earlier 6-prompt run at the same code, 6 / 6, `raw/ids6/`). The row-kernel runs of window 2 were also identical on 2 prompts.
- **acceptance** (tokens per pass, from the speed window's counters): ho-chat 3.04, ho-code 5.32, ho-prose 2.50, ho-recall 3.66, rcp-code 5.30,
  rcp-prose 2.62, rcp-recall 4.17; equal to the pre-change figures (3.22 / 5.32 / 2.50 / 3.60 / 5.20 / 2.54 / 3.89 of the spt doc) within prompt-state noise; the
  paired A/B above shows identical accepted counts.
- **`plan show`**: the diff against origin/main touches only `crates/intarsia-core/src/model/dflash.rs` (the draft's kernel-selection default and the trace),
  `docs/tuning.md` and docs/scripts; nothing a config without `--spec dflash` reads. `INTARSIA_DFLASH_ROWS_KERNELS` is read only when a draft is loaded.
- **Mac** `cargo test --workspace --release`: see the PR (the first run failed only `docs_env_vars` on the since-removed fp8 knob; rerun green).

## 5. Decode, tp=2, interleaved x3 (window speed, #6398)

Arms `mtp2 dfl7; dfl7 off mtp2; mtp2 dfl7` (mtp2 and dfl7 three rounds, off one, as in the spt doc), 49 runs, 0 problems (`raw/speed/cat-table.txt`,
per-round values and spread in `raw/speed/spread.txt`; round-to-round spread of dfl7 0.2-1.9%, mtp2 0.2-0.5%). tok/s, pooled per category:

| category | off | MTP k2 | DFlash2 k7 after | DFlash2 k7 before (spt window B, other main) | dfl7 after vs mtp2 |
|---|---|---|---|---|---|
| ho-chat | 56.9 | 86.2 (1.52x) | 70.3 (1.24x, 3.04 t/p) | 63.7 (1.20x of 53.2) | -18% |
| ho-code | 56.9 | 103.2 (1.81x) | 123.3 (2.17x, 5.32 t/p) | 105.7 (1.99x) | **+19%** |
| ho-prose | 56.6 | 76.6 (1.35x) | 57.8 (1.02x, 2.50 t/p) | 49.5 (0.93x) | -25% |
| ho-recall | 56.8 | 91.0 (1.60x) | 84.7 (1.49x, 3.66 t/p) | 71.2 (1.34x) | -7% |
| rcp-code | 56.8 | 103.4 (1.82x) | 122.4 (2.16x, 5.30 t/p) | 103.3 (1.94x) | **+18%** |
| rcp-prose | 57.0 | 79.1 (1.39x) | 60.4 (1.06x, 2.62 t/p) | 50.4 (0.95x) | -24% |
| rcp-recall | 56.8 | 94.0 (1.65x) | 96.6 (1.70x, 4.17 t/p) | 76.7 (1.44x) | **+3%** |

Off is 56.9 here against 53.2 in the spt window: origin/main moved (#214 fused M=1 path) between the two windows, so the "before" column is
not same-tree; the same-binary before/after is section 3.1 (draft 21.1 to 13.8 ms). Prose no longer loses to off (0.93-0.95x to 1.02-1.06x).

## 6. Decision

No default change: DFlash2 is at least MTP k2 only on ho-code, rcp-code and rcp-recall; it loses on chat (-18%), prose (-24 to -25%) and ho-recall (-7%),
so the flip rule (dflash >= MTP k2 in every category) fails. The 27B tp=2 HIP default stays `--spec mtp --spec-k 2`. The change that merges is the
DFlash2 draft's row-kernel default, which only makes `--spec dflash` faster (output unchanged, ids 18 / 18). Since dflash beats mtp2 by 18-19% on code,
the adaptive choice by running acceptance (dflash on high-acceptance traffic, mtp2 otherwise) is the next product step; it is not built here.
