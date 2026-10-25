# sp: speculative decoding on HIP (lane sp, 2026-10-25)

Branch `lane-sp/hip-spec`. Draft status: baseline and GPU numbers pending (tatooine queue). Sections are filled as windows
complete; nothing below the "Method" heading is a measurement until it says so.

## Method gaps (stated up front)

**Concurrency on natural text is not measurable with localeval today.** `suite.py` `run_conc` calls `self.prompt_for(size)`
with the default prompt set (filler-v1) and ignores the cell's `prompt_set`; `localeval speed --concurrency` builds filler
prompts too (`speed.py`). The suite's only concurrency cells (`conc-plain-1k`, `conc-plain-4k`) are filler. So this lane
reports single-stream decode on natural-v1 text (`speed-spec`, 128/1k/4k/16k) and aggregate decode at 1/4/16 streams on
**filler** text. Filler text is repetitive in a way that is neither the natural-text acceptance of the DFlash2 comparison
nor a worst case for an MTP head, so the filler aggregate numbers say little about the speculation gain at concurrency.

**Request for the Model Benchmark Suite session (relay):** please add a `conc-spec` cell (lane `spec`, prompt_set
`natural-v1`, params as `conc-plain-1k`: streams [1, 4, 16], prompt_tokens 1024, gen_tokens 256, closed loop, 60 s window),
and make `run_conc` pass `cell.get("prompt_set") or "filler-v1"` to `prompt_for` (and to `warmup_run_start`). Each stream
should send the frozen natural-v1 text for that size, byte for byte; with N identical concurrent prompts the prefix cache
will hit, so the cell either needs distinct natural texts per stream (for example rotating the frozen 1k text's start by
stream index, hashed and frozen like the rest) or must run with the prefix cache disabled and say so in the record
(`--prefix-cache-disabled`). Without it, any claim about speculation gains at concurrency (Radiance `--draft-short-from 8`,
+12% at 8 streams, +18% at 16) cannot be checked against the same text.

## Method

- Baseline and equality: `scripts/sp/baseline.sh` (greedy 256 ids per natural-v1 prompt size, `--show-ids`, md5 compared
  across spec off / eager MTP / captured MTP; acceptance and tok/s from `generate`'s own lines).
- Speed: localeval `suite --tier adhoc` (`scripts/sp/arm.sh`), main vs branch in the same window.
- Gates: `scripts/sp/gates.sh`.

## 1. Baseline: 35B (Qwen3.6-35B-A3B-NVFP4), HIP, one R9700, window 1 (2026-10-25 02:17, fleetq #5039)

`intarsia generate`, natural-v1 prompts (128 / 1k / 4k / 16k tokens), 256 greedy ids with `--ignore-eos`, `--spec mtp --spec-k 2`.
Main (`2c2d9941`, eager draft + verify) on card 0, branch on card 1, same window. Decode tok/s as printed by `generate`
(single run per cell, so differences under about 1% are noise; the table is a baseline, not a speed claim).

| prompt | main spec off | main MTP k=2 | branch MTP k=2 captured | branch k=2 eager, old per-row build | acceptance | tokens/pass |
|---|---|---|---|---|---|---|
| 128 | 164.9 | 189.2 (1.15x) | 190.8 | 183.0 | 75.4% | 2.50 |
| 1k | 164.9 | 196.3 (1.19x) | 199.1 | 189.1 | 80.9% | 2.62 |
| 4k | 160.6 | 203.5 (1.27x) | 205.7 | 196.3 | 90.6% | 2.80 |
| 16k | 146.2 | 158.4 (1.08x) | 159.9 | 155.1 | 73.3% | 2.47 |

- **Equality:** the 256-id md5 is identical across main spec off, main MTP, branch captured, branch eager and branch spec off, at
  all four sizes (ce656b29, 419d4320, 4963646e, 62089f5c). Acceptance counts are identical between main and the branch's
  captured step (153/203, 157/194, 164/181, 151/206 accepted/proposed).
- **No-op with spec off:** branch spec off is md5-identical to main spec off and within 1% in speed (166.1 / 165.5 / 160.9 / 146.4).
- **Main already graph-captures the plain decode steps of an MTP run** (`graph replays=103 (captures 5)`); the draft and the verify
  pass were eager. The captured step moves both into one replay and gains about 1% at k=2 (0.8 to 1.0% per size, single
  runs). It is not where the time is.
- The MTP head's HIP path was already in good shape: 73 to 91% acceptance on natural text, 2.5 to 2.8 tokens per pass.
- **Per-pass cost:** at 128 tokens a pass takes 13.2 ms against 6.0 ms for a plain step (2.2x). The 35B is a mixture-of-experts
  model: a 3-row verify pass touches the union of three tokens' experts, so the pass reads about twice the weights of a one-token
  step. That caps the speculation gain on this model near the measured 1.15 to 1.27x and is why Radiance's 2.4x comes from the
  dense 27B, where a 3-row pass costs about one weight sweep. The 27B is the real target (window 2).
- The old eager per-row verify attention cost about 3% (183.0 vs 189.2 at 128); it is now used by the captured step only, eager
  verify keeps main's multi-row call.

## 2. Qwen3.8-27B (MXFP4, tp1, one R9700): MTP on HIP for the first time (window 2, fleetq #5066)

The 27B checkpoint ships an MTP head (`mtp.*`, bf16, dense SwiGLU MLP, 17408 wide). Main could not load it (the loader assumed
a MoE head); the branch loads it (`HeadFfn::Dense`, recipes `qwen3_5` and `qwen3_5_vl` gain `[mtp]`). The 27B's MXFP4 verify
matmuls need row-exact kernels: `matmul_mxfp4_gemv_rows` and `matmul_mxfp4_pair_gemv_rows` (same launchers, registered as
`rows_exact`). **`rows_exact` on the GPU, run before any 27B number below: `mxfp4_rows_are_bitwise_the_one_row_gemv` passes**
(12 shapes including the 27B's 12288x5120, 17408x5120, 5120x17408, W4A4 and W4A16, bf16 and f32 out, M = 1..8, every row
compared as bytes against the one-row GEMV; `raw/hip/w2-rows-mxfp4.log`).

`generate`, natural-v1 prompts, 256 greedy ids, branch, captured step:

| prompt | spec off | MTP k=2 | MTP k=3 | k=2 acceptance, tok/pass | k=3 acceptance, tok/pass | 256-id md5 off / k=2 / k=3 |
|---|---|---|---|---|---|---|
| 128 | 33.06 | 47.73 (1.44x) | 42.44 (1.28x) | 77.4%, 2.54 | 67.5%, 3.02 | equal |
| 1k | 32.92 | 45.60 (1.39x) | 40.75 (1.24x) | 72.1%, 2.44 | 64.0%, 2.92 | equal |
| 4k | 32.36 | 51.62 (1.60x) | 48.98 (1.51x) | 91.7%, 2.83 | 86.4%, 3.59 | equal |
| 16k | 30.81 | 46.12 (1.50x) | 42.02 (1.36x) | 87.1%, 2.74 | 78.1%, 3.34 | equal |

- Greedy output is bitwise the spec-off output at every size and both k (md5 abcdf0cf, 3592ef43, 7755ec5a, b1b0bea4).
- k=3 is slower than k=2 on the 27B at all sizes: each extra verify row plus an extra draft costs more than the extra accepted
  token (53 ms per pass at k=2, 71 ms at k=3, 30 ms for a plain step).
- **Eager MTP (graph capture off) on the 27B is NOT lossless at 4k and 16k with main's multi-row verify attention**: the
  256-id md5 differs from spec off (e747535b and 350da9a5 against 7755ec5a and b1b0bea4). HIP's multi-row split-KV
  attention partitions the keys by the call's row count, so its row t is not bitwise the decode step's. It matched on the 35B in
  the 256-token runs of section 1, so this was latent there. The captured step does not have the problem (it calls the decode
  attention once per row). Fix on the branch: verify passes on HIP, eager or captured, run the attention one decode call per
  row (`Backend::verify_attn_per_row`); re-checked in window 3 (section 4).

## 3. 35B draft length (window 2): k=1/2/3, captured, branch

| prompt | k=1 | k=2 | k=3 | k=2 tok/pass | k=3 tok/pass |
|---|---|---|---|---|---|
| 1k | 188.9 | 199.8 | 190.0 | 2.62 | 3.02 |
| 4k | 193.5 | 206.5 | 221.6 | 2.80 | 3.64 |

All md5 equal to spec off. A pass costs 9.9 / 13.2 / 16.0 ms at k=1 / 2 / 3 (plain step 6.0 ms): each extra row costs about 3 ms
because the MoE verify touches more experts. The best k depends on the text (1k: k=2; 4k: k=3, +7%), so an acceptance-driven
draft length would beat any fixed k by a few percent on the 35B; not implemented.

## 4. Eager verify fix, host round trip, per-op tables (window 3, fleetq #5095)

**Eager verify with per-row attention is lossless on both models.** Graph capture off (`INTARSIA_SPEC_GRAPH=0`), k=2, 256 ids:
27B md5 equals spec off at 128 / 1k / 4k / 16k (abcdf0cf, 3592ef43, 7755ec5a, b1b0bea4; the 4k and 16k cells differed before the
fix); 35B md5 equals window 1 at all four sizes (ce656b29, 419d4320, 4963646e, 62089f5c). Eager decode, tok/s: 27B 47.5 / 45.1 /
50.9 / 45.5 (captured 47.7 / 45.6 / 51.6 / 46.1), 35B 185.3 / 192.5 / 199.8 / 155.7 (captured 190.8 / 199.1 / 205.7 / 159.9).

**The host round trip is not worth removing.** `INTARSIA_SPEC_TRACE=1` prints, per captured step, the replay-and-download time and the
host time since the previous step ended (acceptance, rollback, emission, stop checks): 27B 53.5 ms per step, 0.004 ms gap
(104 steps); 35B 13.1 ms per step, 0.008 ms gap (97 steps). Whole-run decode time per pass is 53.8 ms (27B) and 13.2 ms (35B),
so everything outside the graph replay is 0.3 ms and 0.1 ms per pass: under 1%, against the 5% bar for device-side acceptance.
Not implemented. (The upload of the position table and the download of the outputs are inside the step time; they are a few
microseconds of copies next to a 13 to 53 ms graph.)

**Where the 27B pass goes** (eager k=2 run with the profiler on, `--op-table`, 1k prompt; the profiled run is slower than the
captured one, 35.5 tok/s, so read shares not milliseconds):

| op | share |
|---|---|
| matmul_mxfp4 + matmul_mxfp4_pair (all dense projections, main and head) | 59.3% |
| logits (lm_head: 104 verify calls of 3 rows, 208 draft calls of 1 row) | 16.2% |
| matmul (bf16: the head layer's projections) | 6.2% |
| gdn_conv_delta | 5.4% |
| rmsnorm_add | 4.2% |
| attn_decode (one call per row) | 2.5% |
| rest | 6.2% |

The lm_head is swept 3 times per pass (two drafts and the verify rows); the drafts alone are about 11% of the pass. 35B shares:
moe_experts 19.8%, matmul_fp8 19.1%, matmul (bf16) 12.3%, matmul_nvfp4 12.0%, gdn_conv_delta 8.6%.

## 5. Verify GEMV and lm_head cost against the row count (window 4, fleetq #5127, one R9700, idle)

`rows_exact` test `timing_mxfp4_rows` (`--ignored`), W4A4 dynamic, bf16 out, microseconds per call, ratio to M=1 in brackets. The
`_rows` twin is within 3% of the base kernel at every M (same launcher).

| shape (N x K) | M=1 | M=2 | M=3 | M=4 | M=5 | M=8 |
|---|---|---|---|---|---|---|
| 17408 x 5120 (MLP gate / up) | 80 | 82 (1.03) | 107 (1.34) | 148 (1.85) | 222 (2.8) | 1524 (19) |
| 5120 x 17408 (MLP down) | 80 | 86 (1.07) | 107 (1.33) | 142 (1.78) | 208 (2.6) | 1534 (19) |
| 12288 x 5120 (q, gated) | 59 | 61 (1.04) | 79 (1.35) | 108 (1.83) | 158 (2.7) | 904 (15) |
| 5120 x 6144 (o) | 35 | 40 (1.13) | 50 (1.42) | 63 (1.78) | 86 (2.4) | 457 (13) |
| 1024 x 5120 (k, v) | 13 | 15 (1.12) | 16 (1.22) | 22 (1.64) | 23 (1.77) | 85 (6.4) |

- One row streams the weights at about 590 GB/s (memory bound); two rows are free; **three rows cost 1.3x, four 1.8x, five 2.7x,
  and M = 8 falls off a cliff (13 to 19x)** because the per-lane accumulators of the `M`-row GEMV no longer fit. The verify pass
  of a k-draft step has M = k + 1 rows, so this is why k=3 loses to k=2 on the 27B and why k >= 4 would be hopeless without a
  better small-row MXFP4 GEMV. Handed to the wg lane (row-invariant small-row GEMV family) with these numbers; no MXFP4 kernel
  is written in this lane.
- lm_head, 248320 x 5120 bf16: 4.25 ms per sweep for M = 1, 4.27 (M=2), 4.28 (M=3), 4.35 (M=4): flat in M. A k=2 pass sweeps it
  3 times (two drafts and the verify rows): 12.7 ms of the 53.5 ms pass, 8.5 ms of it in the drafts.

## 6. `--mtp-draft-vocab N` (window 5, fleetq #5136; opt-in)

The MTP head's argmax runs over lm_head rows 0..N (token ids 0..N: the frequent end of the BPE vocabulary) instead of all 248320;
the verify pass keeps the full head. Captured k=2, natural-v1, 256 ids. **The 256-id md5 equals spec off in all 24 cells**
(same md5 as sections 1 and 2); only acceptance and speed move. Decode tok/s (acceptance):

27B (full head: 47.7 / 45.6 / 51.6 / 46.1; acceptance 77.4 / 72.1 / 91.7 / 87.1%):

| prompt | N=16384 | N=32768 | N=65536 |
|---|---|---|---|
| 128 | 49.1 (62.7%) | 52.3 (70.1%) | **53.2** (77.4%) |
| 1k | 47.6 (59.7%) | 49.8 (65.2%) | **50.3** (71.0%) |
| 4k | 52.8 (75.4%) | 55.2 (81.9%) | **55.8** (88.0%) |
| 16k | 48.3 (75.7%) | 50.3 (81.9%) | **50.5** (86.1%) |

35B (full head: 190.8 / 199.1 / 205.7 / 159.9; acceptance 75.4 / 80.9 / 90.6 / 73.3%):

| prompt | N=16384 | N=32768 | N=65536 |
|---|---|---|---|
| 128 | 179.9 (60.2%) | 186.5 (65.2%) | **199.1** (74.1%) |
| 1k | 185.8 (64.4%) | 192.4 (68.7%) | **209.5** (80.9%) |
| 4k | 196.1 (75.7%) | 205.7 (82.7%) | **216.2** (90.6%) |
| 16k | 146.0 (55.6%) | 157.8 (64.5%) | **163.1** (69.8%) |

- N=65536 (26% of the vocabulary) is the best of the three on both models: **27B +8 to +12%** over the full head (53.2 / 50.3 /
  55.8 / 50.5 tok/s against 47.7 / 45.6 / 51.6 / 46.1), **35B +2 to +5%**, with acceptance within about 4 points of the full
  head. Smaller N saves more head time but loses more tokens than it saves. Total against spec off on the 27B at N=65536:
  1.61x / 1.53x / 1.72x / 1.64x.
- Larger N (98304, 131072) was not run; N=65536 already sits near the full-head acceptance.
- Single runs, one prompt per size: differences of 1 to 2% are noise.

## 7. Gates

| gate | result |
|---|---|
| MTP equality, CPU backend (`tests/mtp.rs`: captured step, eager, scripted acceptance, dense head, draft equality, draft vocab) | 11 pass |
| `intarsia-core` lib, `intarsia-models`, `intarsia-backend-hip` lib on the rebased tree | pass |
| HIP `rows_exact` + `rows_registry` incl. MXFP4 rows | pass |
| HIP `kernels test --quick` | 64 kernels, 353 cases, all admitted |
| HIP 35B `batch_cuda_real` (`real_35b_*`) | pass |
| HIP 35B `batch_cuda_real --ignored` long prompts | pass |
| HIP `batch_recurrent --test-threads=1` | 11 pass |
| HIP dense-8B `real_qwen3_8b_batched_decode_is_bitwise_the_solo_step` | n/a on HIP (dense batching refused by design, same as #133); `gates.sh` runs the `real_35b` tests only |
| HIP no-op, spec off, branch vs main (35B ids 128/1k/4k/16k) | identical md5 (section 1); re-checked against main 0b84f499 in window 6 |
| CUDA (spark1): MTP equality + no-op vs main, core lib, `mtp`, `batch_recurrent` | pass (`raw/cuda`) |
| Metal (endor): MTP equality + no-op vs main | pass (`raw/metal`) |

## 8. Does the draft vocabulary cap hold on Chinese and code? (window 6, fleetq #5145; 27B, k=2, ~1k-token prompts)

Prompts (frozen in the lane's scratch, not suite prompts): a 1376-character Chinese passage on speculative decoding and a 3754-character
excerpt of `crates/intarsia-core/src/model/batch.rs`. Raw completion, 256 greedy ids, `raw/hip/extras/`.

| prompt | spec off | k=2 full head | k=2 `--mtp-draft-vocab 65536` | md5 off / full / 65536 |
|---|---|---|---|---|
| Chinese | 32.95 | 43.51 (acceptance 65.9%, 2.32 tok/pass) | **36.63 (acceptance 51.4%, 2.03 tok/pass)** | equal (2a3c0cbf) |
| Rust code | 32.90 | 51.97 (acceptance 88.6%) | 57.99 (acceptance 88.6%) | equal (f8e69e22) |

- **Code holds**: acceptance is identical to the full head and the cap gains +11.6%.
- **Chinese does not**: the reduced head cuts acceptance from 65.9% to 51.4% and speed falls 16% below the full-head MTP (36.6 against
  43.5 tok/s; still above spec off). Chinese tokens sit above id 65536 in Qwen's vocabulary, so the draft cannot propose them.
- Output is bitwise the spec-off output either way. Decision: `--mtp-draft-vocab` stays **opt-in**; use it on English and code
  workloads, leave it off for CJK or multilingual traffic. Not run: other languages, other N for CJK text.

No-op against main at 0b84f499 (35B, window 6, `raw/hip/extras/x-35.log`): spec off md5 ce656b29 / 419d4320 on both trees at 128 / 1k (166.2 /
166.1 tok/s main, 166.6 / 165.8 branch); MTP k=2 md5 equal, 191.1 / 196.5 tok/s main eager against 191.8 / 198.9 tok/s branch captured.

## 9. Plain serving, main vs branch (window 7, localeval, fleetq #5146; 35B, spec off, `--max-batch 16`)

`localeval suite --tier adhoc` (`speed-plain`, `conc-plain-1k`, filler text), one server per arm on one R9700, main at `0b84f499`
against the branch (rebased on it). Runs: `/opt/code/deepeval/results/20261025-035559-adhoc-sp-main-plain` and
`20261025-041016-adhoc-sp-br-plain` (raw: `raw/serve/`). **This is the spec-off no-op, not a speculation number**: MTP is single
sequence, so there is no speculation at 4 or 16 streams (batched speculation is not implemented).

| | main | branch |
|---|---|---|
| decode, 123 tok prompt | 164.2 | 164.4 |
| decode, 1k | 163.3 (one rep at 81.2: spread rejected) | 163.3 |
| decode, 4k | 159.3 | 159.2 |
| prefill marginal, 1k to 4k | 9655 | 9700 |
| aggregate decode, 1 stream | 145.1 | 149.3 |
| aggregate decode, 4 streams | 238.9 | 238.9 |
| aggregate decode, 16 streams | 312.2 | 311.2 |

The 16k cell is skipped on both arms (`--max-batch 16` clamps the context to 16223). The 1k main cell was rejected by localeval's spread
rule on one slow rep; the median equals the branch's. The concurrency rows are filler text and say nothing about speculation.

## 10. Final CUDA and Metal gates on the PR head (`scripts/sp/final-gates.sh`; spark1 fleetq #5176, endor #5175)

Dev = PR head (code of `98bf81b1`, plus the scripts-only commit `928ceab6`) against main `0b84f499`, both rebuilt. Logs: `raw/final/`.

| check | CUDA (spark1) | Metal (endor) |
|---|---|---|
| spec off: `plan show` json and text byte-identical, 8B and 35B | pass | pass |
| spec off: 256 greedy ids identical to main, 8B and 35B, short and 4k | pass | pass |
| 35B MTP k=2 ids == spec off, natural-v1 128 and 1k | pass | pass |
| 35B MTP k=2 ids == main MTP, 128 and 1k | pass | pass |
| MTP acceptance equal to main (128 / 1k, accepted of proposed) | 161/186, 151/207 | 156/195, 159/191 |
| `--mtp-draft-vocab 65536` + MTP k=2 ids == spec off, 128 | pass | pass |
| `intarsia-core` lib / `tests/mtp` / `batch_recurrent` on CUDA | 116 / 11 / 11 pass | n/a |

## 11. Served natural-text speed, MTP k=2, 35B (window 6b, localeval `speed-spec`, fleetq #5155)

`localeval suite --tier adhoc --cells speed-spec --lane spec:mtp`, one `intarsia serve` per arm on one R9700 (`--max-batch 1`), natural-v1
128 / 1k / 4k / 16k, 256 forced tokens, 3 reps (5 at 4k), median decode tok/s (min to max in `raw/serve/*.result.json`). Runs:
`/opt/code/deepeval/results/20261025-0500*-adhoc-sp-{main-mtp2,br-mtp2,br-mtp2-v65536}`. Arms: main `0b84f499` (eager MTP),
the PR #142 head (captured step), and the PR head with `--mtp-draft-vocab 65536`.

| arm | 128 | 1k | 4k | 16k | accepted of drafted (whole run) |
|---|---|---|---|---|---|
| main, eager MTP k=2 | 195.9 | 164.4 | 160.3 | 151.8 | 73.3% (3614/4928) |
| branch, captured | 203.4 | 168.0 | 163.1 | 150.9 | 73.7% (3638/4936) |
| branch + draft-vocab 65536 | 197.0 | 178.9 | 166.5 | 151.7 | 70.5% (3274/4642) |
| main spec off, window 7 (filler text, not the same prompts) | 164.2 | 163.3 | 159.3 | not run | |

- The captured step gains +3.8% / +2.2% / +1.7% / -0.6% over main's eager MTP: small, as in the in-process numbers (about 1%), and the
  16k cell is inside the rep spread (4 to 7%). Draft-vocab 65536 gains +8.8% at 1k and +2.1% at 4k over the captured arm and nothing
  at 128 or 16k; with 3 reps at 4 to 9% spread, only the 1k gain is clearly outside the noise.
- **Served MTP at 1k, 4k and 16k is at spec-off speed.** Explained in section 12: it is the 35B's pass cost against a fast captured
  plain step on natural text, not the serve path.

## 12. Why served MTP gains so little on the 35B at 1k+ (HIP window 9, fleetq #5189; CUDA #5208; Metal #5206)

Same request body as localeval (chat, temperature 0, `ignore_eos`, 256 forced tokens, `--max-batch 1`, natural-v1), `/metrics` spec
counters read around every request (`scripts/sp/servedclient.py`), medians of 3. Chat-template `generate` with the same prompts beside it
(`scripts/sp/served-split.sh`, `win9.sh`; raw: `raw/followup/served-split/{cuda,metal}.log`).

| backend, size | served off | served MTP | `generate` off | `generate` MTP | served tok/pass | served ms/pass | plain ms/token |
|---|---|---|---|---|---|---|---|
| HIP 128 | 164.7 | 204.0 | 165.5 | 200.9 | 2.79 | 13.7 | 6.1 |
| HIP 1k | 164.0 | 168.1 | 165.6 | 187.3 | 2.33 | 13.9 | 6.1 |
| HIP 4k | 159.3 | 158.5 | 161.3 | 163.9 | 2.29 | 14.4 | 6.3 |
| HIP 16k | 144.6 | 149.8 | 146.7 | 154.8 | 2.47 | 16.5 | 6.9 |
| CUDA 128 | 84.4 | 118.9 | 84.5 | 113.6 | 2.80 | 23.6 | 11.8 |
| CUDA 1k | 83.9 | 100.8 | 84.8 | 99.7 | 2.41 | 24.0 | 11.9 |
| CUDA 4k | 81.6 | 96.7 | 82.3 | 95.5 | 2.36 | 24.4 | 12.3 |
| Metal 128 | 125.2 | 187.5 | 125.1 | 179.7 | 2.77 | 14.5 | 8.0 |
| Metal 1k | 123.8 | 159.3 | 123.8 | 162.6 | 2.40 | 15.1 | 8.1 |
| Metal 4k | 120.8 | 142.9 | 120.8 | 143.1 | 2.33 | 16.4 | 8.3 |

- **The serve path is not the cause.** `INTARSIA_SPEC_GATE=0` and `INTARSIA_SPEC_TRACE=1` change nothing outside the rep spread on any
  backend; with the gate off HIP serves 201.7 / 175.9 / 164.3 / 145.8 at the same acceptance. Served and `generate` agree within the
  rep spread on CUDA and Metal. On HIP the served pass is 4 to 7% slower than `generate`'s (`generate` ran on the other card at the
  same time, so some of it is host contention), which is smaller than the gain being missed.
- **The cause is economics.** A HIP pass costs 13.7 to 16.5 ms against 6.1 to 6.9 ms for a graph-replayed plain step: 2.2x to 2.4x.
  MTP breaks even at 2.2 to 2.4 tokens per pass; natural chat text gives 2.3 to 2.5 at 1k and beyond, so there is no gain. At 128
  tokens it gives 2.8 and the gain is +24%. CUDA and Metal keep a gain because their plain step is slow (11.8 and 8.0 ms), so the
  same pass is only 1.8 to 2.0x a plain step. The 35B is a mixture of experts: a 3-row verify reads the union of three tokens' experts
  (section 1), which a dense model does not pay.
- Earlier in-process "1.15 to 1.27x" numbers were on text with higher acceptance; on these chat prompts `generate` MTP is 1.13x at 1k,
  1.02x at 4k and 1.05x at 16k over `generate` off.
- No engine change fixes this at k=2. What moves it is a cheaper pass (fewer expert reads per verify row, or the draft-vocab head, section
  5) or more tokens per pass. The 27B (dense, section 2) is where spec pays on HIP.

## 13. A measured-cost gate for MTP: built, measured, shipped off (fleetq #5246 HIP; #5234 CUDA; #5233 Metal)

Section 12 found that at 1k and beyond a HIP 35B pass costs about what its tokens would as plain steps (speedup 1.00 to 1.03), so
the coordinator asked for a gate on the measured cost: per request, running averages of tokens per pass and ms per pass, a probe of
the plain-step cost (one plain step every 128 passes, the first after the 2-pass warm-up), and a fall back to plain steps while
`tokens_per_pass / (ms_per_pass / ms_per_plain_token)` stays under 1.03 for 16 passes; after 64 plain tokens (doubling while retries
keep failing, 16x at most) it tries speculation again with fresh averages. It extends `SpecGate` (`spec.rs`, `CostParams`/`CostGate`),
applies to `--spec mtp` in the single-sequence engine only (beside other sequences the step time is shared), reports
`intarsia_spec_costgate_{off,on,plain_steps,probe_steps}_total` in /metrics, and has recipe keys (`[mtp] cost_*`) and
`INTARSIA_SPEC_COSTGATE` (0, 1 or `min_speedup,window,probe_every,reprobe_tokens`). Unit tests drive it with scripted timings, including
samples that arrive a step late (as the engine delivers them); engine tests force constant flipping between speculation and plain decode
and keep greedy ids equal to spec off for k = 1..3 with the model's own and scripted drafts.

**Decision: off by default on every backend, the code and the opt-in kept.** It has no measured benefit anywhere.

HIP, served 35B (`--max-batch 1`, localeval-shaped chat request, natural-v1, 256 forced tokens, 5 reps, median decode tok/s), the
first version of the gate, which left the head's catch-up for the end of a plain stretch:

| size | spec off | MTP, gate off | MTP, gate on | gate on vs off | requests that fell back (of 5) |
|---|---|---|---|---|---|
| 128 | 165.0 | 201.7 | 199.3 | -1.2% | 0 |
| 1k | 164.1 | 173.4 | 162.7 | -6% | 2 |
| 4k | 159.4 | 165.4 | 149.8 | -6% | 4 |
| 16k | 144.9 | 148.4 | 122.3 | -18% | 5 |

- **MTP with the gate off was at or above spec off at every size** (+22%, +6%, +4%, +2%; the 4k and 16k cells are inside the rep spread).
- **With the gate on, a request that fell back ran far under spec off.** A plain step inside a speculating request cost 9 to 12 ms (implied by the
  served throughput, not traced) against 6.3 to 6.9 ms for a spec-off step; the extra is most likely the MTP head catching up on each row.
  That version deferred the catch-up to one batched pass, which made the probe look like a 6 ms step and hid the real cost in a pass the
  gate does not time, so it fell back where it should not have. The fix (`33e13f6d`: no deferral, plain steps pay the catch-up and are timed
  honestly) makes the gate inert on these texts: it falls back only where plain-in-session beats speculation. Reaching spec-off speed would
  take dropping the head for the rest of the request, because the skipped positions have no saved hidden states to catch up from; not built.
- `generate` (no fallback happened): ids equal spec off at 128 / 1k / 4k / 16k for the 35B (md5 29eaec13, c942b461, 18193ddb, 02617b6e) and the
  27B, and tok/s equal with the gate on and off (35B 203.1 / 188.1 / 164.0 / 155.3 against 202.2 / 187.5 / 163.8 / 155.4; 27B 51.7 / 45.9 / 46.6 / 37.7
  against 51.4 / 45.6 / 46.8 / 37.6).

CUDA and Metal, gate forced on (`INTARSIA_SPEC_COSTGATE=1`), served 35B, 3 reps: it never switched off (`cg_off` 0, one plain probe per
request) and made no difference outside the rep spread. Metal 128 / 1k / 4k: 193.4 / 163.5 / 142.3 tok/s forced on against 187.1 / 154.9 / 144.4
default (the 4k cell is -1.5%, inside the 141 to 145 against 144 to 149 spread, and the earlier -3% at 4k with two probes per request did not
reproduce). CUDA: 118.2 / 95.8 / 96.0 against 118.5 / 98.0 / 93.9. One CUDA chat `generate` run at 1k with the gate forced on fell back after
21 passes of early low acceptance and lost the rest of the request (85 tok/s); that is what the shorter, doubling retry stretch fixes.

Gates with the default (off) on the final head: spec-off no-op against main, and MTP k=2 ids and acceptance equal to main at 128 and 1k, on
HIP (35B and 27B), CUDA and Metal; see section 14.

## 14. Gates on the final head `4c317919` (default path: cost gate off)

Every run stamps the source head and counts the `intarsia_spec_costgate` strings in the binary it runs (CUDA 2, Metal 1, HIP 2; main's binary 0).
Logs: `raw/final-head/{hip,cuda,metal}-run.log`; HIP window 11 is fleetq #5275 (35B and 27B, 128 and 1k, natural-v1, main against the head on the two cards in
parallel), CUDA #5266, Metal #5265. 0 failing checks on all three:

| check | HIP | CUDA | Metal |
|---|---|---|---|
| spec-off no-op against main: plan json and text byte-identical, 256 greedy ids identical (8B and 35B, short and 4k) | PASS | PASS | PASS |
| 35B MTP k=2 ids == spec off, 128 and 1k | PASS | PASS | PASS |
| 35B MTP k=2 ids == main's MTP, 128 and 1k | PASS | PASS | PASS |
| 35B acceptance counts == main's (passes / proposed / accepted) | 102/203/153, 97/194/157 | 93/186/161, 104/207/151 | 98/195/156, 96/191/159 |
| 27B MTP k=2 ids == spec off and == main's MTP, 128 and 1k | PASS | not run | not run |
| 27B acceptance == main's | 100/199/154, 104/208/150 | | |
| `--mtp-draft-vocab 65536` ids == spec off (35B, 128) | PASS (window 8) | PASS | PASS |
| cost gate forced on (`INTARSIA_SPEC_COSTGATE=1`) ids == spec off | PASS (chat `generate` 35B and 27B, window 10) | PASS | PASS |
| `cargo test` intarsia-core lib / MTP / `batch_recurrent --test-threads=1` (on the box) | not run (CPU run on the Mac: 123 / 17 / n/a) | 125 / 17 / 11 | not run |

So the default decode path is unchanged against main on all three backends: same ids, same acceptance counts, same plans. HIP follow-up gates
(window 8, head `1ee4d0ac`, section 11 follow-ups) also passed: eager head catch-up cost +0.7 to +0.9% faster than main (35B) and +0.5% (27B).
