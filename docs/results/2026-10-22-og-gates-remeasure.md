# Open rows O1, O4, O5: measured, nothing flipped (lane og)

Branch `lane-og/gate-remeasure`, off origin/main `f9087322`. Raw data and the scripts that produced it: `raw/2026-10-22-og-gates/`,
`scripts/og-gates/`. No flag, default or numerics-affecting code changed; the lane adds scripts, raw results and this page.
Each verdict below says **resolved-by-measurement** or **still open**; whether to act on it is the user's call.

| Row | Verdict | One line |
|---|---|---|
| O4 (35B 4096-token HIP prefill delta) | **resolved-by-measurement: noise** | n=5 interleaved pairs: -0.02 % (95 % CI -0.22 to +0.18 %) |
| O5 (chunk*k+1 prefill tail, CUDA / HIP) | HIP **resolved-by-measurement**; CUDA **still open** (no spark1 slot yet) | HIP: every tail of 1 to 16 rows differs bitwise from the no-tail run, so the 1-row flag would not make HIP invariant; CUDA: see section 3 |
| O1 (Phase 1 decode gate vs vLLM) | **still open** (no spark1 slot yet) | see section 1 |

CUDA_O1_STATUS_PLACEHOLDER

## 1. O1: Phase 1 decode gate on current main vs same-day vLLM (spark1)

PENDING

## 2. O4: 35B 4096-token HIP prefill, is the -0.30 % / -0.22 % noise?

Round 9 recorded 35B prefill at 4096 tokens as -0.30 % (W9, base `ab887732` vs final `5bf91f0f`) and -0.22 % (W7), ranges
overlapping, and noted that the B prompts were 10 to 50 tokens longer. This run asks the cheap question: do the two round-9
levers that could touch prefill (lever 1 page-locked staging, lever 2 split trigger) move 4096-token prefill?

**Method** (`scripts/og-gates/o4-ab.sh`, window on tatooine, one R9700, `intarsia-cargo-target-og` build of main `f9087322`):
- one binary, env-only arms: `dflt` (defaults) and `off` (`INTARSIA_HIP_STAGE=0` and `INTARSIA_HIP_SPLIT_MIN_US=0`, which restore
  the pre-round-9 copy path and round 8's split rule, per `docs/results/2026-10-19-hip-round9.md`);
- a fresh `serve` per sample (35B NVFP4, `--max-seq 8192`, graphs on, recipe prefill chunk 4096, confirmed in the serve log) and
  `intarsia bench speed --at-most --sizes 4096,3968 --gen 16 --reps 5` (median of 5 reps is one sample);
- ABBA order, 5 rounds, so 5 paired samples per size; `--at-most` keeps every prompt below its size, so 4096 stays inside ONE
  4096-token chunk (round 7 saw 4079 to 4120-token prompts and a second chunk past 4096);
- rule fixed before the run (`scripts/og-gates/o4-analyze.py`): noise if the 95 % CI (Student t, 4 df) of the paired relative
  difference includes 0.

**Result** (prefill tok/s, difference = (dflt - off) / off; prompt tokens per sample are in the raw JSON, 3963 to 4003 at target 4096):

| target | pairs | mean diff | sd | 95 % CI | positive pairs | verdict |
|---|---|---|---|---|---|---|
| 4096 | 5 | **-0.02 %** | 0.16 | -0.22 to +0.18 % | 2 of 5 | noise |
| 3968 (control, well below the chunk) | 5 | -0.07 % | 0.30 | -0.45 to +0.30 % | 3 of 5 | noise |

Decode, as a by-product: dflt is +0.54 % (4096) and +0.60 % (3968) over off, positive in 5 of 5 pairs, CI above 0. That matches
round 9's lever-1 decode gain (+0.6 %, W2).

**Verdict: resolved-by-measurement, noise.** With levers 1 and 2 switched off against on, the 4096-token prefill difference is
-0.02 % with an interval of about +-0.2 %, smaller than the -0.30 % that round 9 reported. Limits: this tests the two levers on
current main, not the exact `ab887732` vs `5bf91f0f` binaries (lever 3 is opt-in and off in both arms; the stamp/diagnostic
code of round 9 is in both arms); a real effect below about 0.2 % is not excluded. Round 9's own suspect, the longer prompts, is
a bench artefact: per-sample token counts differed by up to 36 between arms here too, with no visible effect on tok/s.
The per-lever A/B that the row offered ("add it to G2") is therefore not needed for this delta.

Raw: `raw/2026-10-22-og-gates/o4/` (10 bench JSONs, serve heads, `analysis.txt`), `window-o4.log`.

## 3. O5: is prefill invariant to a chunk*k+1 tail on CUDA and HIP?

`Backend::avoid_single_row_prefill` (`crates/intarsia-core/src/backend.rs:172`) turns a `(C, 1)` tail into `(C-1, 2)`. It is
ANDed with `!has_recurrent()` (`model.rs:1213`, `batch.rs:132`), so it can never apply to the hybrid 35B; the dense Qwen3-8B BF16
is the model that matters, and that is what was measured.

**Method** (`scripts/og-gates/o5-tails.sh`, `tailcmp.py`): `intarsia accuracy --write-self-golden` prefills the three prompts of
the existing 8B 4k golden (4064, 4064, 4557 tokens; all positions are dumped as top-20 rows) at many `--prefill-chunk` values;
the tail of a prompt is `L mod C` rows. `tailcmp.py` compares every dump with the single-chunk run (`C = 4557`) bit by bit
(top-20 ids and the float32 log-probabilities), and reports rows that differ, the first differing row, argmax flips and the
KL(ref || run) per row over the shared top-20 ids. Decode mode (`--mode decode`, 16 teacher-forced decode positions after a prefill
of 4000 / 4000 / 4493 tokens) shows whether the tail row's KV state differs. The reference is bitwise reproduced by the other
no-tail schedules.

### HIP (tatooine R9700, Qwen3-8B BF16, build of main `f9087322`, two radiance windows)

| schedule (chunk) | prompt 4064 tail rows | rows that differ from single chunk | worst-row KL (prompt 0 / 1) | argmax flips |
|---|---|---|---|---|
| 4096, 2032 x2, 1016 x4 (control: no tail) | 0 | **0** (bitwise equal) | 0 | 0 |
| 4063 and 239 x17 (chunk*k+1) | 1 | **1** (the tail row) | 4.6e-5 / 4.3e-7 | 0 |
| 4062 and 2031 x2 (chunk*k+2) | 2 | **2** (the tail rows) | 6.0e-5 / 4.0e-4 | 0 |
| 4061, 4060, 4059 | 3, 4, 5 | 3, 4, 5 (the tail rows) | up to 8.4e-4 | 0 |
| 4056, 4048 | 8, 16 | 8, 16 (the tail rows) | up to 8.5e-4 | 0 |
| 4032, 4000, 3936, 3808 | 32, 64, 128, 256 | **0** | 0 | 0 |

The 4557-token prompt shows the same pattern (tail 1, 2, 16 rows differ; tails of 461 or more equal); its worst-row KL is about
1e-14 because its last rows are the near-deterministic chat-template tokens, so the bits differ while the distribution does not.
Decode mode: **all 6 schedules, including tails of 1 and 2 rows, are bitwise equal** to the reference at every decode position,
so the tail rows' hidden state and KV cache are identical; what differs is the logit projection of the rows of a small tail
chunk.

**HIP verdict (resolved-by-measurement): the 1-row tail is not invariant, but neither is a 2-row tail, nor any tail up to 16 rows;
tails of 32 rows or more are bitwise invariant.** Turning `(C, 1)` into `(C-1, 2)` on HIP would therefore not remove the
non-invariance (it would trade a 1-row for a 2-row tail, which differs just the same). The effect is small (worst-row KL 8.5e-4,
0 argmax flips in 4064 + 4064 + 4557 positions) and is confined to the logits of the last few prompt rows. Making HIP invariant
would need a tail floor between 17 and 32 rows, a different rule from the Metal flag; this lane did not test it. Left to the user.

### CUDA (spark1)

CUDA_O5_PLACEHOLDER

Raw: `raw/2026-10-22-og-gates/o5-hip/` (logs, `tailcmp-prefill.txt/.json`, `tailcmp-decode.txt`; the 26 dump files, 52 MB, are not
committed).

## Downtime (radiance, tatooine)

Windows logged in `raw/2026-10-22-og-gates/downtime.log`; the `wall` column there includes the up-to-5-minute wait the script
adds before a window so radiance serves at least 5 minutes in between. Radiance was actually down for 522 s (O5 8B sweep),
204 s (O4 A/B), 13 s (a failed start of the second O5 window, `exit 127`, quoting bug in `hip-window.sh`, nothing ran) and 171 s
(O5 tails 32 to 256): 910 s in total, longest 8.7 min, every gap at least 5 min, radiance answering `/v1/models` after every window.

## fleetq jobs

tatooine: #3624 (CPU build, tree `~/og-hip`, volume `intarsia-cargo-target-og`), #3628 and #3599 (read-only looks), #3629 (O5),
#3635 (O4), #3645 (failed attempt), #3651 (O5 tails 32 to 256). spark1: #3583 (CUDA build of main, tree `og-main`), the rest below.
