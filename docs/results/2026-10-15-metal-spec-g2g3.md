# Metal spec G2/G3: drafter gate, speed sweep, realistic workload, merge bar

Date 2026-10-15. Branch `metal-spec-ngram` (worktree `/opt/code/intarsiaml-spec`), on top of the G1 tip `44c4a91`
(`docs/results/2026-10-15-metal-spec-g1.md`). Machine endor (Apple M5 Max), Qwen3-8B BF16, greedy `--spec ngram` k = 4.
Plan: `docs/research/metal-next-levers.md` (branch `metal-next-levers`) section 7, gates G2 and G3.
Not merged into `spike/metal`, not pushed.

**Verdict: G2 met, G3 met on the category means (grounded rows 2.46x mean / 2.21x geomean, open-ended rows 1.06x mean),
with three of ten grounded rows below 1.3x; identity holds everywhere; the merge bar passes.**

| item | result |
|---|---|
| Gate rule | drafts only from suffixes of 2 or 3 tokens (was 1..3), and 6 consecutive verify passes without an accepted draft back off 2, 4, .. 32 plain steps before the next probe (section 1) |
| Identity re-run | 24 cases x off / on4 / on6 vs `spike/metal` c556008: **72 of 72 byte-identical** (stdout, ids, finish, completion tokens); plus 13 cut-path cases with the gate off: 26 of 26 (section 2) |
| G2 speed (ABBA, fresh server per pass) | spec off 36.0 / 36.0 / 35.0 tok/s (128 / 1k / 4k), spec on **1.05x / 1.07x / 1.12x**, no row below spec off, all four passes above 1450 MHz (section 3) |
| G3 per row | grounded **2.46x** mean (2.21x geomean, 1.10x to 4.01x), open-ended **1.06x** mean (1.01x to 1.15x), ids identical on all 20 rows x 3 reps x 2 modes (section 4) |
| Merge bar | all items pass: `cargo test` 233 passed, metal 101 passed, kernels 27 / 321 admitted, accuracy unchanged vs c556008 (long 99.72% / 3.699e-4), qualify PASS, ids spec off = spike/metal, cuda + hip check (section 5) |

## 1. The gate

### Why

G1's open-ended story ran at 0.93-0.99x: the drafter proposed on 106 of 244 steps (a 4-draft verify pass costs about 24% more than the
plain step it replaces) and 3% of the drafts were accepted. The cost is the *rejected pass*; a pass that accepts one draft already pays
for itself (2 tokens for about 1.24 steps).

### Rule (`crates/intarsia-core/src/spec.rs`, `GateParams` / `SpecGate`, wired into `generate_one` in `engine.rs`)

1. **Minimum match length 2.** `propose_min(ctx, k, 2)`: the longest-suffix search tries 3, then 2, never 1. A one-token suffix match
   is the main source of rejected passes on prose (any repeated word matches).
2. **Back-off after consecutive rejected passes.** Per request: after **6** verify passes in a row in which no draft was accepted, stop drafting
   for **2** plain steps, then probe with one pass; every further failed probe doubles the skip (2, 4, 8, 16, 32 = `skip_max`);
   any accepted draft resets the streak and the skip length. A request where nothing is ever accepted costs 16 wasted passes in 256 tokens (6, then probes at steps 8, 13, 22, 39, 72, 105, ..)
   instead of one per matching step, and a request that never gets a match pays nothing.

Constants `GATE_MIN_N = 2`, `GATE_FAILS = 6`, `GATE_SKIP0 = 2`, `GATE_SKIP_MAX = 32`. `INTARSIA_SPEC_GATE=fails,skip0,skip_max,min_n`
overrides them for experiments (`0,4,64,1` = no gate, one-token suffixes allowed = the G1 drafter). The gate only decides *whether a step
drafts*; verification, acceptance and rollback are untouched, so identity is unchanged by construction (and re-measured in section 2).
No adaptive draft length: with 1-token matches gone, an accepted-length EWMA gained nothing in the replay below.

Unit tests (`spec.rs`, 14 tests in the module, 7 new): gate stays open after one miss, exponential back-off schedule (passes at steps
0, 1, 6, 15, 32, 49, 66 for `gate(2, 4, 16)`), an accepted pass resets the skip length, a hit on the probe reopens at once,
`fails = 0` disables the gate, the default gate costs at most 20 passes in 256 hopeless steps and never closes while passes are accepted,
`propose_min` clamps and skips short suffixes. The engine tests (`intarsia-models/tests/spec.rs`, cpu backend, run through the default gate) keep
passing, and still hit the cut paths: with `INTARSIA_SPEC_TRACE=1` the default gate gives 218 `min_tokens cut` and 230 `unemitted` trace lines
(gate off: 296 and 150).

### How it was chosen

Greedy streams are identical with spec on and off, so the accepted count of every verify pass is known from the plain stream: the engine loop
(drafter + gate) can be replayed offline over `prompt ids + generated ids`. `raw/.../gate/sim.py`, `grid.py` do that with a cost of 1 per plain
step and 1 + 0.06 x drafts per verify pass (fitted on the G1 open-ended trace: 106 passes of 4 drafts = 0.95x). The replay reproduces the
real runs exactly:

```
simulator vs real runs: 42 of 42 (config, case) pairs reproduce the measured verify-pass and accepted-draft counts exactly
```

(7 gate configurations x 6 prompts = 42 pairs.) Real sweep (slot #2001, `gate/sweep.log`), one
`generate` per cell, decode tok/s, verify passes (p), accepted drafts (a), 256 tokens. The tok/s of the plain rows moves 30.7 to 35.5 between
repeats of the same case (single runs, clock), so only p and a are used to rank the rules; the two `off` rows are the references.

| cfg | open-256 | grounded-256 | golden-docs-256 | golden-code-256 | 4k-256 | short-256 |
|---|---|---|---|---|---|---|
| 1-off | 35.4 p0 a0 | 35.5 p0 a0 | 33.0 p0 a0 | 32.6 p0 a0 | 31.1 p0 a0 | 31.1 p0 a0 |
| 2-ngram_0_4_64_1 | 31.3 p106 a12 | 105.7 p65 a190 | 30.4 p166 a41 | 54.0 p106 a138 | 98.5 p67 a188 | 70.6 p95 a158 |
| 3-ngram_2_4_64_1 | 30.2 p11 a2 | 106.9 p64 a187 | 31.4 p13 a4 | 45.3 p37 a87 | 94.4 p67 a184 | 53.9 p77 a108 |
| 4-ngram_3_4_64_1 | 33.2 p16 a3 | 115.6 p65 a190 | 31.5 p14 a4 | 46.8 p52 a99 | 100.2 p67 a188 | 75.0 p95 a158 |
| 5-ngram_2_8_64_1 | 33.5 p13 a3 | 111.9 p62 a185 | 32.6 p12 a4 | 47.8 p44 a94 | 94.8 p66 a181 | 53.4 p71 a103 |
| 6-ngram_0_4_64_2 | 31.4 p27 a7 | 112.6 p64 a189 | 32.4 p59 a31 | 53.2 p69 a121 | 94.5 p67 a188 | 70.6 p87 a144 |
| 7-ngram_2_4_64_2 | 33.4 p19 a6 | 106.8 p62 a183 | 32.9 p28 a19 | 42.3 p31 a66 | 98.4 p67 a184 | 64.1 p79 a133 |
| 8-ngram_1_4_64_1 | 32.5 p12 a2 | 89.7 p54 a165 | 32.2 p14 a3 | 40.8 p30 a66 | 87.8 p61 a172 | 46.9 p58 a89 |
| 9-off | 34.6 p0 a0 | 32.5 p0 a0 | 33.1 p0 a0 | 30.7 p0 a0 | 31.0 p0 a0 | 33.2 p0 a0 |

Configs: 2 = G1 drafter (no gate, 1-token suffixes), 3 = back off after 2 failures (skip 4, x2 up to 64), 4 = after 3, 5 = after 2 with skip 8, 6 = minimum
suffix 2 only, 7 = minimum suffix 2 + back off after 2, 8 = back off after 1. A back-off on 1-token matches alone (3, 4, 5, 8) hurts exactly the
text that hits in bursts (golden-code: 138 accepted drafts ungated, 66 to 99 with it; short: 158 to 89 to 108). Minimum suffix 2 removes the
rejected passes at their source (open 106 to 27 passes, 12 to 7 accepted) while keeping most of the accepted drafts.

Replay over 26 greedy streams (6 sweep prompts + the 20 G3 rows; speedup vs plain, same cost model):

Simulated speedup vs plain decode (cost model: plain step 1, verify pass 1 + 0.06 x drafts), replayed over the greedy streams; k = 4

| row | ungated (min_n 1) | min_n 2 only | min_n 2 + streak 6 (CHOSEN: 6,2,32,2) | min_n 1 + streak 2 (4,64) | min_n 1 + streak 3 (4,64) | min_n 3 only |
|---|---|---|---|---|---|---|
| sweep-open-256 | 0.95 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| sweep-grounded-256 | 3.16 | 3.13 | 3.13 | 3.06 | 3.16 | 3.12 |
| sweep-golden-docs-256 | 1.00 | 1.07 | 1.08 | 1.00 | 1.00 | 1.04 |
| sweep-golden-code-256 | 1.78 | 1.68 | 1.69 | 1.43 | 1.50 | 1.61 |
| sweep-4k-256 | 3.07 | 3.07 | 3.07 | 2.93 | 3.07 | 2.60 |
| sweep-short-256 | 2.13 | 1.93 | 1.93 | 1.54 | 2.13 | 1.72 |
| g3-grounded-edit-docs-comments | 2.06 | 2.05 | 2.05 | 1.86 | 1.89 | 1.98 |
| g3-grounded-edit-error-msgs | 3.25 | 3.25 | 3.25 | 3.23 | 2.90 | 3.14 |
| g3-grounded-edit-field | 2.98 | 2.95 | 2.95 | 2.83 | 2.84 | 2.89 |
| g3-grounded-edit-refactor-model | 2.94 | 2.95 | 2.95 | 2.60 | 2.94 | 2.79 |
| g3-grounded-edit-rename-fn | 3.56 | 3.41 | 3.41 | 3.56 | 3.56 | 3.34 |
| g3-grounded-extract-fields | 2.11 | 1.89 | 1.89 | 1.67 | 1.83 | 1.69 |
| g3-grounded-extract-ops | 1.77 | 1.73 | 1.73 | 1.46 | 1.44 | 1.65 |
| g3-grounded-summ-bench | 1.20 | 1.20 | 1.20 | 1.06 | 1.05 | 1.16 |
| g3-grounded-summ-health | 1.08 | 1.10 | 1.07 | 1.06 | 1.03 | 1.09 |
| g3-grounded-summ-overview | 1.17 | 1.14 | 1.14 | 0.99 | 0.99 | 1.10 |
| g3-open-brainstorm-appnames | 1.12 | 1.04 | 1.04 | 1.05 | 1.07 | 1.02 |
| g3-open-brainstorm-lessons | 1.07 | 1.07 | 1.07 | 1.01 | 1.02 | 1.05 |
| g3-open-brainstorm-weekend | 1.12 | 1.09 | 1.09 | 1.03 | 1.13 | 1.07 |
| g3-open-explain-hashmap | 0.98 | 1.01 | 1.00 | 1.00 | 0.99 | 1.01 |
| g3-open-explain-kvcache | 1.01 | 1.06 | 1.06 | 0.99 | 1.02 | 1.03 |
| g3-open-explain-tides | 1.11 | 1.10 | 1.10 | 1.05 | 1.03 | 1.05 |
| g3-open-story-cartographer | 0.95 | 0.99 | 0.99 | 0.99 | 1.01 | 1.00 |
| g3-open-story-chef | 0.96 | 1.02 | 1.02 | 0.99 | 0.99 | 1.01 |
| g3-open-story-lighthouse | 0.97 | 1.00 | 1.00 | 0.99 | 0.99 | 1.00 |
| g3-open-story-robot | 0.95 | 1.00 | 1.00 | 0.99 | 0.99 | 1.00 |
| **g3 grounded mean** | 2.21 | 2.17 | 2.16 | 2.03 | 2.05 | 2.08 |
| **g3 open mean** | 1.024 | 1.039 | 1.038 | 1.011 | 1.023 | 1.025 |
| **g3 open min** | 0.948 | 0.992 | 0.995 | 0.993 | 0.992 | 0.998 |
| **sweep open-256 (story)** | 0.950 | 1.002 | 1.002 | 0.997 | 0.997 | 1.002 |

Honest reading: **minimum suffix 2 does almost all of the work** (open-ended story rows 0.95 to 1.00 in the replay). The back-off costs
nothing on these 256-token streams (grounded mean 2.17 vs 2.16, open min 0.992 vs 0.995) and bounds the damage on text that is
all 2-gram matches and no hits, for which minimum suffix 2 alone pays one rejected pass per matching step for as long as the answer runs
(the replay streams are 256 tokens, so the longer-output case is argued, not measured). I kept it as a safety net; it is about 60 lines of code plus the tests, and a few counters of state per request.

## 2. Identity re-run (final code)

Binary built from `df6d543`, the code tip (stamp `head=df6d543 code-tree=ddce5e644b9e dirty=no`; `git diff df6d543 HEAD -- crates kernels recipes tools scripts` is
empty). 24 cases of the G1 matrix (`cases-24.txt` = G1's `cases-all` 19 + `cases-c` 5: max_tokens, stop strings, min_tokens, EOS inside a run,
4k filler, natural 4k docs and code, input-grounded and open-ended prompts), each as `--spec off`, `--spec ngram` (k = 4, `on4`) and `--spec ngram --spec-k 6` (`on6`),
`--show-ids`, compared with the `spike/metal` c556008 binary (`base`). **The base outputs are the G1 run's** (same binary sha256
`ffb54608..61bb888e6`, same prompts, deterministic greedy), copied from `raw/2026-10-15-metal-spec-g1/identity-final` and `identity-c`, not re-run.

**Result: all 72 variant runs (24 x off / on4 / on6) are byte-identical to base in stdout and ids line, with the same finish reason and completion tokens
(`identity-vs-base.md`); off vs base and on vs off are also ALL IDENTICAL (`identity-vs-off.md`).** Drafts: on4 1046 passes / 4183 proposed /
2584 accepted (61.8%), on6 906 / 5433 / 2735 (50.3%) (G1, ungated: 1001 / 4003 / 2446 over 19 cases). 28 of the 48 spec-on runs ended inside an
accepted run (KV rolled back), same as before.

The gate makes `min_tokens` cuts rare (the cut needs a verify row whose argmax is a stop id inside the window: 0 of 48 runs here), so the cut path would
have gone untested by this matrix. So the 13 stop / max / min_tokens / EOS cases were run once more with the gate **off**
(`INTARSIA_SPEC_GATE=0,4,64,1`, the G1 drafter, same binary), on4 + on6: **26 of 26 identical to base**, 4 `min_tokens cut` events and 20 mid-run endings
(`identity-nogate-vs-base.md`). The cpu engine tests cover the cut path under the default gate (218 cuts).

| case | finish / tokens | off, on4, on6 vs base: stdout and ids | on4: passes / accepted / proposed | on6: passes / accepted / proposed | ended inside an accepted run (on4 / on6) | decode tok/s base / off / on4 / on6 |
|---|---|---|---|---|---|---|
| 4k-256 | length / 256 | identical | 67 / 188 / 268 (70.1%) | 55 / 200 / 330 (60.6%) | no / no | 33.81 / 34.97 / 116.68 / 112.02 |
| 4k-48 | length / 48 | identical | 13 / 36 / 52 (69.2%) | 11 / 38 / 66 (57.6%) | yes / yes | 33.72 / 35.29 / 111.75 / 109.83 |
| 4k-max44 | length / 44 | identical | 12 / 33 / 48 (68.8%) | 11 / 38 / 66 (57.6%) | yes / yes | 34.24 / 35.47 / 113.45 / 100.87 |
| 4k-max9 | length / 9 | identical | 4 / 6 / 16 (37.5%) | 4 / 8 / 24 (33.3%) | yes / yes | 33.6 / 35.12 / 56.82 / 53.02 |
| 4k-stop250 | stop / 85 | identical | 24 / 62 / 96 (64.6%) | 20 / 66 / 120 (55.0%) | yes / yes | 34.13 / 35.56 / 111.01 / 104.81 |
| golden-code-256 | length / 256 | identical | 67 / 121 / 268 (45.1%) | 60 / 130 / 360 (36.1%) | yes / yes | 33.58 / 33.94 / 56.89 / 56.14 |
| golden-code-48 | length / 48 | identical | 14 / 6 / 56 (10.7%) | 14 / 6 / 84 (7.1%) | no / no | 33.35 / 33.7 / 38.18 / 36.12 |
| golden-docs-256 | length / 256 | identical | 55 / 31 / 220 (14.1%) | 55 / 31 / 330 (9.4%) | no / no | 34.0 / 34.28 / 38.55 / 37.02 |
| golden-docs-48 | length / 48 | identical | 12 / 6 / 48 (12.5%) | 12 / 6 / 72 (8.3%) | yes / yes | 34.23 / 35.38 / 38.15 / 36.8 |
| grounded-256 | length / 256 | identical | 64 / 189 / 256 (73.8%) | 52 / 201 / 312 (64.4%) | no / no | 31.51 / 35.76 / 132.71 / 138.39 |
| grounded-512 | stop / 329 | identical | 82 / 242 / 328 (73.8%) | 69 / 255 / 414 (61.6%) | no / no | 32.0 / 36.06 / 127.14 / 122.54 |
| grounded-max102 | length / 102 | identical | 24 / 77 / 96 (80.2%) | 20 / 81 / 120 (67.5%) | yes / yes | 35.32 / 36.22 / 132.48 / 143.36 |
| grounded-max188 | length / 188 | identical | 47 / 140 / 188 (74.5%) | 39 / 151 / 234 (64.5%) | yes / yes | 35.34 / 35.62 / 130.67 / 136.83 |
| grounded-min320-stopfence | stop / 328 | identical | 82 / 242 / 328 (73.8%) | 69 / 255 / 414 (61.6%) | no / no | 35.23 / 36.25 / 131.12 / 138.15 |
| grounded-min331 | length / 400 | identical | 97 / 297 / 388 (76.5%) | 83 / 312 / 498 (62.7%) | no / yes | 34.01 / 36.11 / 128.64 / 133.48 |
| grounded-min335 | length / 400 | identical | 97 / 297 / 388 (76.5%) | 83 / 312 / 498 (62.7%) | no / yes | 34.96 / 36.37 / 132.77 / 132.02 |
| grounded-stop-atomic | stop / 224 | identical | 55 / 168 / 220 (76.4%) | 44 / 181 / 264 (68.6%) | yes / yes | 34.81 / 35.96 / 133.13 / 139.66 |
| grounded-stop-topp | stop / 115 | identical | 30 / 86 / 120 (71.7%) | 25 / 93 / 150 (62.0%) | yes / yes | 35.94 / 36.7 / 124.47 / 131.25 |
| grounded-stop-vecstring | stop / 186 | identical | 47 / 140 / 188 (74.5%) | 38 / 145 / 228 (63.6%) | yes / no | 35.58 / 35.9 / 130.82 / 141.88 |
| open-256 | length / 256 | identical | 27 / 7 / 107 (6.5%) | 27 / 7 / 159 (4.4%) | no / no | 31.67 / 36.44 / 37.37 / 36.89 |
| repeat-eos | stop / 16 | identical | 3 / 11 / 12 (91.7%) | 2 / 12 / 12 (100.0%) | yes / yes | 34.58 / 36.7 / 108.83 / 131.75 |
| short-256 | length / 256 | identical | 87 / 144 / 348 (41.4%) | 82 / 150 / 492 (30.5%) | no / yes | 35.62 / 36.06 / 81.05 / 72.24 |
| short-48 | length / 48 | identical | 16 / 23 / 64 (35.9%) | 14 / 25 / 84 (29.8%) | no / no | 35.8 / 35.93 / 70.09 / 73.31 |
| short-stopkettles | stop / 57 | identical | 20 / 32 / 80 (40.0%) | 17 / 32 / 102 (31.4%) | yes / yes | 35.06 / 36.82 / 72.96 / 76.59 |

(Decode tok/s here are traced single runs from the identity matrix, not speed data; base is from the G1 run.)

## 3. G2: `localeval speed`

`localeval speed --url ... --model intarsia-metal --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off`, one exclusive fleetq slot (#2011), four passes in ABBA
order **spec off, spec on, spec on, spec off**, a fresh `intarsia serve` per pass (same binary, `--spec off|ngram`, `--no-supervisor --no-ui`, port 8093, killed and port closed
between passes), GPU idle check (3 samples, median <= 10%) before each, `macmon pipe -i 100` for the whole slot, `/metrics` spec counters polled every second.
Clock rule: a pass is dropped if the median GPU MHz over its three 4k-prefill request windows is below 1450. localeval 0.2.1; script `g2/g2_run.sh`, analysis `g2/g2_analyze.py`.

| pass | localeval run | idle util % | MHz 4k prefill | MHz decode 128/1k/4k | fast? | decode 128 | decode 1k | decode 4k (median of 3) | marginal prefill 1k / 4k tok/s |
|---|---|---|---|---|---|---|---|---|---|
| p1-off | 20261015-122505-adhoc-p1-off | 4 | 1617 | 1285/1362/1398 | FAST | 36.0 (36.0/36.0/35.5) | 36.1 (36.1/35.9/36.3) | 35.3 (35.0/35.3/35.3) | 4385 / 3875 |
| p2-on | 20261015-122729-adhoc-p2-on | 1 | 1453 | 1316/1409/1386 | FAST | 37.7 (36.4/38.1/37.7) | 38.5 (38.3/38.5/38.9) | 39.3 (38.6/39.3/39.5) | 4318 / 3445 |
| p3-on | 20261015-122933-adhoc-p3-on | 1 | 1494 | 1372/1408/1391 | FAST | 37.6 (37.8/36.7/37.6) | 38.6 (38.2/38.6/40.1) | 39.2 (40.3/39.2/39.1) | 4346 / 3596 |
| p4-off | 20261015-123127-adhoc-p4-off | 2 | 1576 | 1328/1372/1360 | FAST | 36.4 (35.9/36.4/36.4) | 36.0 (36.4/36.0/35.7) | 34.8 (34.8/35.1/34.7) | 4368 / 3839 |


All four passes are above 1450 MHz at 4k prefill (1453 to 1617), so none is dropped; the filtered and unfiltered tables are the same.

### All passes (ABBA, no clock filter)

| size | spec off tok/s | spec on tok/s | on / off | spec off vs 36.7/36.6/35.6 | passes (off, on) used |
|---|---|---|---|---|---|
| 128 | 36.0 (n=6, 35.5-36.4) | 37.7 (n=6, 36.4-38.1) | **1.05x** | -1.9% | p1-off,p2-on,p3-on,p4-off |
| 1024 | 36.0 (n=6, 35.7-36.4) | 38.5 (n=6, 38.2-40.1) | **1.07x** | -1.5% | p1-off,p2-on,p3-on,p4-off |
| 4096 | 35.0 (n=6, 34.7-35.3) | 39.2 (n=6, 38.6-40.3) | **1.12x** | -1.5% | p1-off,p2-on,p3-on,p4-off |


**Criteria.** Spec off stays within noise of 36.7 / 36.6 / 35.6: 36.0 / 36.0 / 35.0 (-1.9% / -1.5% / -1.5%; the two off passes agree within 0.4 tok/s, and
localeval's decode noise is about +/-5%). No spec-on row falls below spec off (1.05x, 1.07x, 1.12x; the lowest spec-on rep, 36.4, is at the level of the off medians). Per-rep decode of
the two spec-on passes is 36.4 to 38.1 at 128, 38.2 to 40.1 at 1k, 38.6 to 40.3 at 4k, the off passes 35.5 to 36.4, 35.7 to 36.4, 34.7 to 35.3. See the control pass in section 6 for off vs the
spike/metal binary.

### spec counters from /metrics (spec-on passes, per prompt size: 3 reps x 256 tokens)

| pass | size | verify passes | proposed | accepted | acceptance | tokens per pass (1 + accepted/passes) |
|---|---|---|---|---|---|---|
| p2-on | 128 | 83 | 332 | 32 | 9.6% | 1.39 |
| p2-on | 1024 | 171 | 684 | 64 | 9.4% | 1.37 |
| p2-on | 4096 | 211 | 844 | 110 | 13.0% | 1.52 |
| p3-on | 128 | 80 | 320 | 23 | 7.2% | 1.29 |
| p3-on | 1024 | 171 | 684 | 67 | 9.8% | 1.39 |
| p3-on | 4096 | 218 | 872 | 115 | 13.2% | 1.53 |

**How much the synthetic filler flatters n-gram.** The prompt is `Note 1: the .. will .. the ..` sentences repeated, then "Continue these notes with a long, detailed account." With the gate in place it
flatters very little: acceptance is 7 to 10% at 128 and 1k and 13% at 4k, which gives 1.05x / 1.07x / 1.12x. The G3 open-ended rows (short prompt, no grounding) give 1.01x to 1.15x, mean 1.06x
(stories 1.02x with 8% acceptance, explanations 1.08x with 19%, brainstorms 1.10x with 19%), and the natural 4k docs prompt of the identity matrix 1.12x at 14% acceptance. So the filler sweep sits
inside the range of open-ended text: +0.03x above the story rows at 128 (1.05x vs 1.02x), level with the open-ended mean (1.06x), and at 4k equal to natural text (1.12x). The bigger error is the other way round: the sweep says nothing about the
grounded work where n-gram matters (section 4: 1.1x to 4.0x), so a G2 table alone understates the feature by 2x on that workload and overstates it on stories by 3 to 10 points. The G1 numbers for the
ungated drafter were 0.93 to 0.99x on open-ended text; with the gate the sweep and the open-ended rows are both at or above 1.0.

## 4. G3: realistic workload

### Dataset

`g3/g3-realistic.jsonl` (`localeval quality --dataset` format: `name`, `category`, `input`, `messages`; loads with localeval's own `load_dataset`, 20 rows), built by
`g3/make_dataset.py` from `tools/prompts/corpus_code.txt` (Rust: `engine.rs` and `model.rs` excerpts of 70 to 130 lines) and `corpus_docs.txt` (architecture, bench and metrics docs):

* **10 grounded:** 5 *edit this code* (rename a function, add doc comments, add a field, change error messages, wrap fields in `Option`; "output the complete modified code"),
  3 *summarise this doc while quoting* (6 verbatim quotes with their meaning, about 200 words), 2 *extract fields* (JSON array copied from a table, verbatim descriptions). Prompts 590 to 2260 tokens.
* **10 open-ended, no grounding:** 4 stories, 3 explanations (KV cache, hash map, tides), 3 brainstorms. Prompts 27 to 35 tokens.

All rows: chat template with thinking off (`--chat --no-think`), greedy, 256 new tokens, EOS honoured (none stops early). Each row is the user message of one single-turn conversation (`g3/prompts/<row>.txt`).

**Why `generate` and not `localeval quality`.** localeval makes one call per row, records one speed figure per row and the text, no ids; it cannot give
3 runs per row or an ids comparison, and a spec-on and a spec-off pass would be two servers at different times. `generate` timings and `--show-ids` give all three.
Per row: `intarsia generate ... --max-tokens 256 --chat --no-think --show-ids [--spec off | --spec ngram]`, 3 reps per mode, mode order alternating by rep (off,on / on,off / off,on) in the same
exclusive slot (#2011), one fresh process per run, same binary for both modes. Decode tok/s is `generate`'s own `decode ... ms (X tok/s)` (prefill excluded). The dataset file is ready for
`localeval quality --dataset g3/g3-realistic.jsonl --judge none --max-tokens 256` against a spec-on and a spec-off server; I did not run it.

### Results (`g3/g3-analysis.md`; per-run files in `g3/runs/`)

| row | cat | prompt tok | gen tok | off tok/s (3 reps) | on tok/s (3 reps) | on/off (median) | on/off (pooled ms) | passes | proposed | accepted | acceptance | ids identical |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| grounded-edit-docs-comments | grounded | 855 | 256 | 35.9 (36.4/35.9/35.6) | 82.0 (82.0/77.8/82.7) | **2.29x** | 2.25x | 59 | 236 | 148 | 63% | yes |
| grounded-edit-error-msgs | grounded | 1161 | 256 | 35.2 (35.2/33.9/35.9) | 137.6 (137.9/135.7/137.6) | **3.91x** | 3.92x | 56 | 224 | 192 | 86% | yes |
| grounded-edit-field | grounded | 847 | 256 | 36.2 (35.9/36.2/36.3) | 124.0 (121.5/124.0/127.0) | **3.42x** | 3.43x | 68 | 272 | 188 | 69% | yes |
| grounded-edit-refactor-model | grounded | 1014 | 256 | 36.2 (36.3/36.1/36.2) | 126.0 (126.0/121.8/126.7) | **3.48x** | 3.45x | 69 | 276 | 188 | 68% | yes |
| grounded-edit-rename-fn | grounded | 1038 | 256 | 36.3 (36.3/35.7/36.3) | 145.5 (146.8/145.2/145.5) | **4.01x** | 4.04x | 57 | 228 | 198 | 87% | yes |
| grounded-extract-fields | grounded | 930 | 256 | 36.0 (36.3/36.0/35.1) | 74.7 (74.7/74.0/76.0) | **2.07x** | 2.09x | 62 | 248 | 137 | 55% | yes |
| grounded-extract-ops | grounded | 2263 | 256 | 35.1 (34.9/35.5/35.1) | 66.5 (66.5/66.0/68.0) | **1.89x** | 1.90x | 72 | 288 | 125 | 43% | yes |
| grounded-summ-bench | grounded | 1388 | 256 | 36.0 (36.3/35.6/36.0) | 44.5 (44.5/43.9/45.2) | **1.23x** | 1.24x | 48 | 192 | 54 | 28% | yes |
| grounded-summ-health | grounded | 1042 | 256 | 36.0 (35.7/36.0/36.5) | 39.7 (39.6/39.7/40.2) | **1.10x** | 1.11x | 46 | 184 | 30 | 16% | yes |
| grounded-summ-overview | grounded | 586 | 256 | 36.4 (36.4/36.4/36.0) | 43.4 (43.4/43.4/43.3) | **1.19x** | 1.20x | 41 | 163 | 41 | 25% | yes |
| open-brainstorm-appnames | open | 35 | 256 | 36.2 (36.2/36.2/36.7) | 39.3 (39.3/39.2/39.8) | **1.09x** | 1.08x | 41 | 164 | 22 | 13% | yes |
| open-brainstorm-lessons | open | 27 | 256 | 35.7 (35.7/36.2/35.4) | 39.2 (39.2/39.9/39.1) | **1.10x** | 1.10x | 24 | 96 | 22 | 23% | yes |
| open-brainstorm-weekend | open | 34 | 256 | 36.8 (36.2/36.8/36.8) | 41.2 (41.3/41.1/41.2) | **1.12x** | 1.13x | 31 | 124 | 29 | 23% | yes |
| open-explain-hashmap | open | 34 | 256 | 36.8 (36.5/36.8/36.8) | 37.1 (37.0/37.1/37.1) | **1.01x** | 1.01x | 13 | 52 | 2 | 4% | yes |
| open-explain-kvcache | open | 32 | 256 | 36.9 (36.9/36.8/36.9) | 39.6 (39.6/39.4/39.6) | **1.07x** | 1.07x | 18 | 72 | 18 | 25% | yes |
| open-explain-tides | open | 35 | 256 | 36.8 (36.8/36.8/36.6) | 42.2 (42.2/42.2/42.2) | **1.15x** | 1.15x | 39 | 156 | 33 | 21% | yes |
| open-story-cartographer | open | 32 | 256 | 36.9 (36.9/36.9/36.9) | 37.2 (37.3/37.2/37.2) | **1.01x** | 1.01x | 18 | 72 | 3 | 4% | yes |
| open-story-chef | open | 32 | 256 | 36.8 (36.8/36.8/36.8) | 38.3 (38.3/38.3/38.3) | **1.04x** | 1.04x | 16 | 64 | 10 | 16% | yes |
| open-story-lighthouse | open | 31 | 256 | 36.8 (36.8/36.6/36.9) | 37.7 (37.7/37.7/37.7) | **1.02x** | 1.02x | 21 | 84 | 6 | 7% | yes |
| open-story-robot | open | 34 | 256 | 36.9 (36.9/36.5/36.9) | 37.5 (37.5/37.5/37.3) | **1.02x** | 1.02x | 20 | 79 | 5 | 6% | yes |

| category | rows | mean of ratios | geomean | median | min | max | rows >= target |
|---|---|---|---|---|---|---|---|
| grounded | 10 | 2.460x | 2.205x | 2.179x | 1.10x | 4.01x | 7/10 (target 1.3x) |
| open | 10 | 1.062x | 1.061x | 1.058x | 1.01x | 1.15x | 10/10 (target 0.97x) |

Identical ids on every row (all reps, both modes): YES

By group: edit rows 3.42x (74% of drafts accepted), extract rows 1.98x (49%), summarise-while-quoting rows **1.18x** (23%; the three rows are 1.10x, 1.19x, 1.23x), stories 1.02x (8%), explanations 1.08x (19%), brainstorms 1.10x (19%).
**Targets:** grounded >= 1.3x: met on the mean (2.46x; geomean 2.21x; median 2.18x) but **not on every row**: the three summarise-while-quoting rows (1.10x to 1.23x) are below 1.3x because the
model paraphrases around the quotes, so there is little to copy. Open-ended >= 0.97x: **met on all 10 rows** (min 1.01x). Identical ids: **met on every row**, all 6 runs of a row (3 off, 3 on) emit the same ids.
The replay predicted these within a few percent in the open-ended rows and underestimated the grounded rows (3.25x predicted vs 3.91x measured on the error-message edit): a verify pass costs less than the 0.24
steps the replay charges.

## 5. Merge bar (tip `df6d543` code, slot #2011, binary stamp `head=df6d543 code-tree=ddce5e644b9e dirty=no`)

| check | result |
|---|---|
| `cargo test --release -p intarsia-core -p intarsia-models -p intarsia-server -p intarsia-cli` | PASS: 233 passed, 0 failed (`tests/cargo-test.txt`, CPU) |
| `cargo test -p intarsia-backend-metal --release -- --test-threads=1` | PASS: 101 passed, 0 failed (`merge/tests/cargo-test-metal.txt`) |
| `intarsia kernels test --backend metal -m Qwen3-8B -v` | PASS: 27 kernels, 321 cases, all ADMITTED, 0 FAIL / REJECT (`merge/kernels/kernels-test.out`) |
| accuracy, spec off (default), baseline-gated like p3-final | short prefill 100.00% / 2.832e-4, decode 100.00% / 3.313e-4; **long prefill 99.72% / 3.699e-4**, decode 99.80% / 3.538e-4; 4k prefill 99.28% / 2.274e-4, decode 98.96% / 2.548e-4 (21-way); floor, baseline and prefix-reuse verdicts all PASS (`merge/accuracy/`) |
| vs spike/metal c556008 | unchanged: the long figures equal the c556008 merge bar (99.72%, KL 3.699e-4) to the printed digits; the recorded c556008 baselines (`golden/baselines`) gate the run with `--pre-change 9eeb96c`, as in p3-final |
| greedy ids, spec off vs spike/metal c556008 | identical, 6 of 6: short, 4k, w4107 (4107 tokens, 2048+2048+11 chunks) at 48 and 256 tokens (`merge/identity/summary.txt`); also the 24-case matrix of section 2 |
| `qualify --backend metal --bandwidth-gbs 614` (fresh, full) | **PASS**: 11 PASS, `ab` and `tp` SKIP (`merge/qualify/full-report.md`); decode 36.6 / 36.5 / 35.5 tok/s at 128 / 1k / 4k (qualify output, not benchmark data) |
| `cargo check -p intarsia-cli --features cuda` / `--features hip` | PASS / PASS (`tests/cargo-check-cuda.txt`, `-hip.txt`; compile checks only, no CUDA/HIP device here) |

The cargo test and cuda / hip checks ran on the CPU outside the slot (nothing in them touches the GPU) on `df6d543`; every later commit changes `docs/results/` only
(`git diff df6d543 HEAD -- crates kernels recipes tools scripts deploy Cargo.toml Cargo.lock` is empty at the tip), so the slot binary, the tests and the checks are all on the same code.
`cargo check` for CUDA and HIP compile the cuda / hip crates and the CLI with those features; the gate touches no backend code and no kernel.

## 6. G2 control: spike/metal vs this branch with `--spec off`

Not run: the control slot (#2030) was still queued behind other sessions' jobs at the deadline, and was cancelled. Spec off vs spike/metal rests on the byte-identical ids (section 2, section 5) and on the code path: with `--spec off` the verify buffers and kernels are not allocated and `generate_one` takes the plain branch (`spec_on` false, no drafter, no gate call).

## 7. Commits, slots, files

Commits on `metal-spec-ngram` (not merged, not pushed), after G1's `44c4a91`:

* `df6d543` metal G2: drafter gate (`spec.rs` `GateParams` / `SpecGate` / `propose_min`, `engine.rs` wiring), unit tests, gate sweep + simulator data. **The only code commit.**
* `9e2fcab` metal G2/G3: scripts (slot, identity, G2 ABBA + macmon, G3 runner, merge bar), G3 dataset, analysis tools.
* the results commit(s) that add this document and the raw data of the slot.

fleetq (all released, `fleetq status` shows none of mine running or queued): #2000 (the first sweep binary was built without the Metal backend: every `generate` failed at once, no data, released), #2001 gate sweep (11:22 to 11:29),
#2004 spec-off ids of the 20 G3 rows for the replay (11:30 to 11:33), #2011 identity + G2 + G3 + merge bar in one exclusive slot (12:19:25 to 12:56:59), #2010, #2014, #2015 cancelled while queued
(superseded by #2011), #2030 cancelled while queued.

Raw data and scripts: `docs/results/raw/2026-10-15-metal-spec-g2g3/`: `gate/` (sweep logs and per-run `.err`/`.out`, `table.py`, simulator `sim.py` `grid.py` `grid2.py` `grid3.py`, `sim-results.md`,
`sim_vs_real.py`), `identity/` + `identity-nogate/` + `identity-vs-*.md`, `g2/` (`g2_run.sh`, `passes/`, `macmon.jsonl`, `localeval/` result.json, `g2-analysis.md`), `g3/` (`make_dataset.py`,
`g3-realistic.jsonl`, `prompts/`, `g3_run.sh`, `runs/`, `ids/` = the spec-off ids the gate replay used, `g3-analysis.md`, `g3-summary.json`), `merge/`, `tests/`, `slot.sh`, `slotA.sh`, `slots.log`.
