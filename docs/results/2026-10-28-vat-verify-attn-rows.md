# VAT: one attention launch per layer for the rows of a speculative verify pass (HIP)

Qwen3.8-27B MXFP4 (`amd/Qwen3.8-27B-Quark-AWQ-MXFP4`), `--tp 2 --devices 0,1 --max-batch 1`, tatooine 2x R9700 (HIP), 2026-10-28.
Branch `lane-vat/verify-attn-rows` (PR #218), base `origin/main` `bb48df640` (after #217, lane vrc).
Status: draft PR; every gate is met except one (section 4: the DFlash2 category text gate, a flake the old path shows too).

## Headline

* A verify pass of 2..16 rows runs each full-attention layer's `AttnDecode` as ONE row-table launch (`attn_decode_bf16_splitkv_rows`, the batched-decode kernel) instead of `q` single-row launches (16 layers x q on the 27B).
  Every table row names the sequence's own K / V cache and row `r` reads position `pos0 + r`, so row `r` sees the prefix plus the rows `<= r` through the same per-row inner loop as the solo kernel: bitwise, GPU-tested against the per-row launches, 4 of 4 planted faults caught.
  `INTARSIA_HIP_VERIFY_ATTN_ROWS=0` keeps the per-row launches (docs/tuning.md). Spec default untouched (27B tp=2 stays `--spec mtp --spec-k 2`).
* Verify + rollback per pass (mean, `INTARSIA_SPEC_TRACE=1`, one run each, `raw/w1/`):

  | arm | prompt | old (knob 0) | new | saved |
  |---|---|---:|---:|---:|
  | mtp k2 (3 rows) | short (41 tok) | 20.79 ms | 20.47 ms | 0.32 ms |
  | mtp k2 | 3k | 21.64 | 21.20 | 0.44 |
  | dflash k7 (8 rows) | short | 25.89 | 24.14 | 1.75 |
  | dflash k7 | 3k | 28.23 | 26.26 | 1.97 |

  This is about 60 % of the VRC estimate (3 ms at q = 8, 1 ms at q = 3). Why less was not measured; candidates: the row-table launch always uses the one-row grid at its maximum split count (64 splits x kv heads blocks per row, most exit at once) where the per-row host-position launch sizes its grid to the key count, and the combine is one launch of `q` x heads blocks.
* Decode tok/s, same binary, knob on vs off, pooled median of 3 interleaved rounds, 7 categories (`raw/w2/`, `tools/dfh/vrc_spread.py`):

| category | off | off-old | MTP k2 old | MTP k2 new | new/old | DFlash2 k7 old | DFlash2 k7 new | new/old | spread new/old (k2, k7) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ho-chat | 56.9 | 56.8 | 89.1 | 90.2 | x1.012 | 76.2 | 79.6 | x1.045 | 0.4%/0.1%, 0.5%/1.0% |
| ho-code | 57.0 | 56.9 | 106.7 | 107.8 | x1.010 | 133.7 | 138.8 | x1.038 | 0.7%/0.6%, 0.7%/0.6% |
| ho-prose | 56.9 | 56.9 | 79.3 | 80.4 | x1.014 | 62.8 | 65.5 | x1.043 | 0.4%/0.3%, 0.1%/0.3% |
| ho-recall | 56.9 | 56.9 | 94.2 | 95.6 | x1.015 | 91.8 | 95.6 | x1.041 | 0.2%/0.0%, 0.4%/0.5% |
| rcp-code | 56.9 | 57.0 | 106.7 | 108.6 | x1.018 | 133.1 | 138.9 | x1.044 | 0.3%/0.2%, 0.5%/0.2% |
| rcp-prose | 56.9 | 56.8 | 82.0 | 83.3 | x1.016 | 65.6 | 68.4 | x1.043 | 0.5%/0.1%, 0.4%/0.0% |
| rcp-recall | 56.9 | 56.8 | 97.1 | 98.7 | x1.016 | 104.6 | 109.1 | x1.043 | 0.5%/0.4%, 0.3%/0.1% |

  DFlash2 k7: x1.038 .. x1.045 in all 7 categories (round-to-round 0.0-1.0 %); MTP k2: x1.010 .. x1.018 (below the 2 % bar, none slower).
  Spec off is the same binary either way (`off` vs `off-old` 56.8-57.0 tok/s in all categories; `localeval speed` 1k / 4k: 56.8 / 56.1 vs 56.8 / 56.0). The 27B tp=2 default (MTP k2) gains 1-2 %.
  Ratios in the table are ratios of medians; `raw/w2/spread.md` has per-round ratios (k7 1.031-1.048, k2 1.008-1.022).

## 1. What changed

* `crates/intarsia-core/src/model/verify_attn.rs` (new): `Model::verify_attn_rows`, called from `attn_rows_core` in the verify branch (`verifying()` / head rows, backend `verify_attn_per_row()` = HIP, eager passes only: `dpos` is `None` and no graph recording). It builds an `AttnDecode` `OpCall` with a `RowTab` (`stride 2`, `index 0`) and `pos` = `[pos0, pos0 + 1, ..]`.
* Tables (`--max-batch 1` has none, so the model owns a small device buffer): one `i32` position vector shared by all layers (256 bytes at offset 0) plus one 16-row region of (K, V) cache addresses per state slot (256 bytes each). A region is written only when the layer's cache buffers change (first pass, new sequence state), the position vector when `(pos0, t)` changes, i.e. once per pass by its first attention layer (a staged pinned async copy, no stream sync). Nothing is uploaded while the plan records.
* Falls back to the per-row launches (and says so once on stderr, `verify attention rows: ...`) when the knob is `0`, the pass has fewer than 2 or more than 16 rows, no kernel fits, a buffer has no device address, or the plan's scratch is smaller than the call needs. `attrs.pos0` stays 0 as in the batched row-table calls (`OpCall::check` bounds `pos0 + T` by the cache capacity, which the rows near a full cache exceed in the call's own terms).
* Plan: the verify probes now record `attn_decode_bf16_splitkv_rows` (12.7 MB scratch at 27B tp=2, inside the existing 17.8 MB reservation, so `scratch_bytes` is unchanged). `plan show` for the 8B and 35B (spec off) is unchanged (no `splitkv_rows` row, `raw/w1/plan-8b.out`, `plan-35b.out`). For the 27B tp=2 (default MTP k2) the JSON differs from the knob-off JSON by exactly one added op row (`attn_decode_bf16_splitkv_rows`, `admitted: false` in the dry plan: a row-table kernel takes the admission of the kernel it twins at run time, `Registry::effective_admission`), `raw/w1/plan-27b-tp2*.out`. No base-binary plan was taken (one tree, `~/intarsiaml-vat`).
* Test seams: `INTARSIA_HIP_VAT_MUT=1..4` (docs/tuning.md), read only by the table builders.

## 2. Bitwise test (GPU, `crates/intarsia-backend-hip/tests/verify_attn_rows.rs`)

Tables are built with the model's own builders; one K / V cache named by every row of a layer's region (layer 3 of a 5-layer buffer, so the offsets are exercised), 14 passes per shape with t in 2..=16 and pos0 in 0..1483 in the order a spec run produces them (t shrinks and grows at the same pos0), 3 shapes (12/2/256, 24/4/256, 32/8/128 heads / kv heads / head_dim), keys with scores of order one so that every key of the prefix weighs in. Reference: per-row `attn_decode_bf16_splitkv` with the host position, as the old verify path launched. **2,928,640 bytes identical** (`raw/w2/ids/hash.txt`). Planted faults (each must FAIL, `raw/w2/ids/mut.txt`): 1 every row at `pos0`, 2 last row one position short, 3 rows past the first take the K address as V, 4 position vector not rewritten when only `t` changes: **4 of 4 caught at the first pass**.
(The first window ran an earlier version of the test with outlier-heavy keys whose softmax is dominated by one key: it caught mutants 3 and 4 but missed 1 and 2 on the first pass, which is why the keys changed; `raw/w1/`.)

## 3. Verify cost method

`INTARSIA_SPEC_TRACE=1`, `generate --spec mtp --spec-k 2` and `--spec dflash --spec-draft Qwen3.8-27B-DFlash2 --spec-k 7`, prompts `scripts/dfh/ids-prompts/short-0.txt` and `4k-1.txt`, 256 forced tokens, mean of `verify + rollback` over the passes after the first four (`scripts/vat/stages-phase.sh`, `raw/w1/steps.log`). The old arm is the same binary with the knob `0`.

## 4. Gates

| gate | result |
|---|---|
| greedy ids = `--spec off`, tp=2, 18 prompts (6 short, 6 ~3k, 6 ~10k), 256 ids, MTP k2 and DFlash2 k7 | **18 / 18 both** in the rerun (`raw/w3/ids/`). The first run (`raw/w2/ids/`) had 17 / 18 for both arms: prompt `short-5`, where the spec arms (new path) gave the id list of `raw/w3` and of the VRC window (md5 `5d21626f84ad`) and the **plain `--spec off` run differed** (md5 `890643df86a8`). Four more off runs, the new and old (knob 0) MTP and DFlash arms in `raw/w3` all gave `5d21626f84ad`: a one-off plain-decode difference, not the verify path. |
| category texts = off (7 categories, 512 tokens, 3 rounds, `tools/vat/text_compare.py`) | MTP k2: **108 / 108 new, 108 / 108 old**. DFlash2 k7: new **107 / 108**, old 108 / 108 (`raw/w2/text-compare.txt`); `ho-chat` `chat-sleep` round 3 diverges at char 145. **Not met as stated.** Repeat (`raw/w3`, ho-chat only, 5 rounds, new and old alternating): new 19 / 20 (`chat-sleep` round 3, char 74), **old 19 / 20** (`chat-sleep` round 5, char 107); off is identical between its two runs. VRC's window had the same on main (old dflash 107 / 108, `recall-tls`). The DFlash2 text flake is present at the same rate in the old path, on a different token each time; it is not investigated here (it is in neither this PR's nor VRC's code). |
| new launch byte-equal to per-row launches, 3+ planted faults caught | 2,928,640 bytes identical; 4 / 4 caught (section 2) |
| batched == solo | `intarsia-models` GPU tests `batch`, `batch_recurrent`, `tp_batch`, `tp_mtp`, `mtp`, `spec`, `hybrid`, `fusion_p2`, `graphs`, `hybrid_mxfp4`, `tp_hybrid`, `tp` all `ok` (`raw/w2/ids/steps.log`); HIP `row_tables`, `attention`, `rows_registry`, `rows_exact` ok (`raw/w1/`) |
| `plan show` 8B / 27B / 35B | section 1 |
| Mac `cargo test --workspace --release --no-fail-fast` | 1075 passed, 0 failed (final code tree `f9837cc`; later commits touch only scripts, tools, docs) |
| CI | see the PR |
| speed: same binary, knob on vs off, >= 3 interleaved rounds; spec off not slower | section Headline (3 rounds, order alternated; off 56.8-57.0 both) |

## 5. Not done / uncertain

* Saved 0.3-0.4 ms (k2) and 1.8-2.0 ms (k7) per pass against the estimated 1 and 3 ms; the verify trace is one run per cell.
* The DFlash2 category text flake (above) and the one `--spec off` `short-5` outlier are unexplained; both predate this change as far as the old-path arms show.
* Captured passes (tp=1 graph, device positions) and `noncausal_rows` (the DFlash draft block) keep the per-row launches; the all-reduces (129 per step) are untouched.
* The lane did not take a plan from a base binary; the 8B / 35B argument is that their plans have no verify probes (spec off).

## 6. Fleet

fleetq jobs on tatooine (Radiance taken down for GPU ones): builds `#6419`, `#6420`, `#6424` (CPU slot); `#6421` (a window that ran nothing: `STAGES` was not forwarded to the box; Radiance down 10 s); `#6422` (w1: test, mutants, plan, phase, 4 min); `#6426` (w2: mutants, ids, gates, 3 rounds of category speed, 80 min); `#6428` (w3: ids rerun, short-5 repeats, ho-chat repeats, 23 min). Radiance restored and `/v1/models` answering after each (`raw/*/guard.log`, `window.log`, `raw/downtime.log`).
Scripts: `scripts/vat/` (copy of `scripts/vrc/`: `window.sh`, `cat-window.sh`, `stages-*.sh`). Cleanup: `~/intarsiaml-vat`, docker volumes `intarsia-cargo-vat-dev` and `intarsia-cargo-vat-dev-hiptest`, and `.worktrees/vat` are removed after the PR is final.
