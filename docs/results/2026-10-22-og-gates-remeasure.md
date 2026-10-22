# Open rows O1, O4, O5: measured, nothing flipped (lane og)

Branch `lane-og/gate-remeasure`, off origin/main `f9087322`. Raw data and the scripts that produced it: `raw/2026-10-22-og-gates/`,
`scripts/og-gates/`. No flag, default or numerics-affecting code changed; the lane adds scripts, raw results and this page.
Each verdict below says **resolved-by-measurement** or **still open**; whether to act on it is the user's call.

| Row | Verdict | One line |
|---|---|---|
| O4 (35B 4096-token HIP prefill delta) | **resolved-by-measurement: noise** | n=5 interleaved pairs: -0.02 % (95 % CI -0.22 to +0.18 %) |
| O5 (chunk*k+1 prefill tail, CUDA / HIP) | **resolved-by-measurement** (both backends) | a 1-row tail is not bitwise invariant on either backend, and neither is a 2-row tail: HIP differs up to 16 rows (32 and up is invariant), CUDA up to 64 rows (128 and up is invariant). The `(C,1)` to `(C-1,2)` flag would not make either invariant. |
| O1 (Phase 1 decode gate vs vLLM) | **resolved-by-measurement: PASS at 128 / 1k / 4k, FAIL at 16k** | decode vs same-day vLLM: +9.4 / +7.3 / +5.5 / -2.0 %; against the doc's absolute "86-90 tok/s" every size fails |

## 1. O1: Phase 1 decode gate on current main vs same-day vLLM (spark1)

The gate (`docs/ARCHITECTURE.md:322`): "Qwen3.6-35B-A3B NVFP4 on spark1 correct and >= vLLM single-stream (86-90 tok/s)". It was
recorded FAIL on 2026-10-13 (72.3 / 71.8 / 69.6 against 76.1 / 77.6 / 76.4 tok/s at 128 / 1k / 4k) and never re-run.

**Method** (`scripts/og-gates/o1-driver.sh`, one exclusive spark1 slot, job #3701; vLLM `guidance` stayed up and untouched):
- our server: `og-main` build of origin/main `f9087322`, **defaults**, started
  beside the idle vLLM with `--max-seq 20480 --mem-budget 30 --profile-every 0` (the serve flags of every earlier localeval arm);
- `localeval speed` (`/opt/code/deepeval`, `--thinking off --sizes 128,1024,4096,16384 --gen 128 --reps 3`) against vLLM
  (`spark-guidance`) and against our server, 3 rounds, vLLM first in rounds 1 and 3, ours first in round 2;
- a 2 s poller on vLLM's `/metrics` ran the whole slot; no round was dropped for foreign load on vLLM;
- rule fixed before the run (`scripts/og-gates/o1-analyze.py`): per size and round delta = ours - vLLM decode tok/s (median of 3
  reps); PASS if the median delta >= 0 and at least 2/3 of rounds are >= 0, FAIL if the median is < 0 and at least 2/3 are < 0,
  else MIXED; a median within +-1 tok/s is flagged noise.

**Result** (decode tok/s, per-round values in `raw/2026-10-22-og-gates/o1/analysis.md`, localeval run ids and `result.json` in
`o1/results/`):

| prompt (target) | vLLM median | ours median | delta | rounds ours >= vLLM | verdict |
|---|---|---|---|---|---|
| 128 | 76.3 | 83.5 | +7.2 (+9.4 %) | 3 / 3 | **PASS** |
| 1024 (990 tokens) | 77.4 | 83.0 | +5.6 (+7.3 %) | 3 / 3 | **PASS** |
| 4096 | 76.0 | 80.2 | +4.1 (+5.5 %) | 3 / 3 | **PASS** |
| 16384 (16.7k tokens) | 73.3 | 71.8 | -1.5 (-2.0 %) | 0 / 3 | **FAIL** (just outside the 1 tok/s noise band) |

Prefill, for information only (not part of this gate): ours / vLLM at 4k is 5691 / 5845 tok/s (-2.6 %, medians over the rounds) and at
16k 5300 / 6310 (-16 %), with the default 2048-token chunk. The 1k column is a 990-token prompt, so it is TTFT overhead.

**Verdict: resolved-by-measurement.** Against same-day vLLM the single-stream decode gate is met at 128, 1k and 4k and missed at
16k by 1.5 tok/s in all three rounds. The 2026-10-13 FAIL no longer holds at short and medium context (the LOG row already noted CUDA decode fusion moving 72 to 80 tok/s). **Two readings, for the user to choose:** (a) relative to the same-day vLLM, as the 2026-10-13 run and the
phrase ">= vLLM single-stream" do: PASS at 3 of 4 sizes; (b) the literal absolute figure in the sentence, "86-90 tok/s": ours is
83.5 / 83.0 / 80.2 / 71.8, so every size misses it; that figure does not match what vLLM does on this box today (73 to 77 tok/s), so
it looks stale. Nothing was flipped; the 16k decode shortfall is the only measured miss.
Limits: three rounds, one box, one day; vLLM and our server shared the box (vLLM idle during our arm, 64 GB resident);
localeval decode figures carry about +-5 % noise on other days (unanimis 6a7efe40), here the rounds agree within 2 tok/s.

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

### CUDA (spark1 GB10, Qwen3-8B BF16, build of main `f9087322`, job #3700, vLLM up)

Same prompts, same chunk list, same comparison (`raw/2026-10-22-og-gates/o5-cuda/`).

| schedule (chunk) | prompt 4064 tail rows | rows that differ from single chunk | worst-row KL (prompt 0 / 1) | argmax flips |
|---|---|---|---|---|
| 4096, 2032 x2, 1016 x4 (control: no tail) | 0 | **0** (bitwise equal) | 0 | 0 |
| 4063 and 239 x17 (chunk*k+1) | 1 | **1** (the tail row) | 5.3e-4 / 8.1e-5 | 0 |
| 4062 and 2031 x2 (chunk*k+2) | 2 | **2** | 9.3e-5 / 9.6e-5 | 0 |
| 4061, 4060, 4059 | 3, 4, 5 | 3, 4, 5 | up to 3.7e-4 | 0 |
| 4056, 4048 | 8, 16 | 8, 16 | up to 9.5e-4 | 0 |
| 4032, 4000 | 32, 64 | 32, 64 | up to 7.6e-4 | 0 |
| 3936, 3808 | 128, 256 | **0** | 0 | 0 |

The 4557-token prompt: tails of 1, 2 and 16 rows differ (worst-row KL 1e-14 to 1e-3, the near-1e-14 ones are the peaked
chat-template tail tokens), tails of 461 or more are equal.

**CUDA verdict (resolved-by-measurement): same picture as HIP, with a higher threshold.** The 1-row tail is not invariant, and
neither is a 2-row tail, nor any tail up to 64 rows; 128 rows and more are bitwise equal, and so are the no-tail multi-chunk
schedules. The `(C,1)` to `(C-1,2)` flag would not make CUDA invariant either. The effect is confined to the logits of the last
prompt rows, with 0 argmax flips and KL below 1e-3 in 4064 + 4064 + 4557 positions.

Aside (not caused by the tail, reported because it showed up): in decode mode the CUDA dump after a **multi-chunk** prefill
differs from the single-chunk one at the decode positions, also with no tail (chunk 2000 x2 of the 4000-token prompt: all 16
positions differ, worst KL 1.9e-3 and 3.1e-6 on the two prompts, 0 flips), while HIP is bitwise equal there. In prefill mode the
same CUDA schedules are bitwise equal at every position. Not investigated here; it may relate to the known CUDA prefix-reuse
gaps (`docs/results/2026-10-21-cuda-round7.md`) and is a lead for whoever owns that gate. Single-chunk runs were bitwise
reproducible across processes (chunk 4493 and 4492 of the 4000-token prompts).

Overall O5 answer to "which backends need the flag": the flag as written helps none of CUDA, HIP; it targets Metal's `m == 1`
decode-kernel routing. If chunk-invariant first tokens are wanted on CUDA/HIP, the rule would have to keep tails at or above
32 (HIP) or 128 (CUDA) rows, which is a different rule and a numerics-visible choice; not done.

Raw: `raw/2026-10-22-og-gates/o5-hip/` and `o5-cuda/` (logs, `tailcmp-prefill.txt/.json`, `tailcmp-decode.txt`; the 52 MB + 52 MB of
dump files are not committed, `tailcmp.py` re-reads them from the dump directories).

## Downtime (radiance, tatooine)

Windows logged in `raw/2026-10-22-og-gates/downtime.log`; the `wall` column there includes the up-to-5-minute wait the script
adds before a window so radiance serves at least 5 minutes in between. Radiance was actually down for 522 s (O5 8B sweep),
204 s (O4 A/B), 13 s (a failed start of the second O5 window, `exit 127`, quoting bug in `hip-window.sh`, nothing ran) and 171 s
(O5 tails 32 to 256): 910 s in total, longest 8.7 min, every gap at least 5 min, radiance answering `/v1/models` after every window.

## fleetq jobs

spark1: #3583 (CUDA build of main, tree `og-main`, non-exclusive), #3700 (O5 CUDA, non-exclusive, 28 GB, 25 min), #3701 (O1,
exclusive, 32 GB, 5 min; vLLM up the whole time, nothing taken down). tatooine: #3624 (CPU build, tree `~/og-hip`, volume `intarsia-cargo-target-og`), #3628 and #3599 (read-only looks), #3629 (O5),
#3635 (O4), #3645 (failed attempt), #3651 (O5 tails 32 to 256).
