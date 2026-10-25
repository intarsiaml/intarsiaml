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
