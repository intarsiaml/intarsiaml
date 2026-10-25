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
