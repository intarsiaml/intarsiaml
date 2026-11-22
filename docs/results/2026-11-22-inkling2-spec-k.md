# Inkling-2-Mini DSpark on Metal: draft length and verify rule (lane ink-spec-land, Brief L3)

Measurement only. Branch `ink/spec-k` from main 22303d704 (worktree `.worktrees/ink-spec-k`). No default changes here, and no
merge to main while the #491 freeze is on.

## Brief L3 (coordinator, 2026-11-22 ~16:00), queued after the F4/F8 rerun (#11634) and the night speed bar #11621

- Reference: mlx-serve v26.11.4 runs the same DSpark head (`mtp.safetensors`, block 8) with another verify rule. Source read
  (not copied): `.worktrees/mlx-serve-src/src/plugins/inkling2/plugin.zig:290-435` (commit 2fe9be8). They draft up to 7 per
  round; verify a variable-length prefix: drafts while the running product of the drafter's confidences is >= `cut`, never past
  a stop id; `cut = clamp(1.34 * sqrt(r), 0.3, 0.8)`, r = the marginal cost of one more verified row over a plain step (running
  least-squares slope of verify ms against draft count, over the plain step's ms), 0.5 until measured. The verify forward
  appends nothing, then commits the accepted rows (no rollback). Headline 116 tok/s single stream on an M4 Max, greedy, best
  scenario. Ours on endor (M5 Max): 1.12-1.28x over spec off at k3, gate 0.8.
1. If k or the gate were swept before, show those numbers first.
2. Metal, greedy, quiet slot, A B B A: k 3 / 5 / 7 x three rules: today's gate 0.8; a cumulative-product cut at a fixed 0.5;
   cumulative with the cut from our measured r.
3. Contexts short, 1k, 4k, 16k. Prompts: the usual set plus one (predictable) code prompt.
4. Measure verify ms for 1..7 rows (gives r), acceptance, tokens per round, decode tok/s, and whether output stays identical to
   `--spec off`.
- A rule that needs a code change goes behind a measurement-only env on the new branch. Report before building any default.

## 1. What was swept before (all endor M5 Max, Metal, greedy unless noted)

Lane ink-mtp, `docs/results/2026-11-21-inkling2-mtp.md` section 3 and the offline model; numbers copied from there and from the
#11227 log (`.worktrees/ink-mtp/.scratch/gate_c.log`). Every DSpark run listed gave ids identical to `--spec off`.

**#10806** (binary c770aaef3, host ingest, first drafter; greedy golden 12 prompts x 128):

| policy | tok/s | vs off | tokens / verify pass | acceptance |
|---|---:|---:|---:|---:|
| spec off | 60.21 | 1.00 | - | - |
| k1, gate 0.6 | 63.71 | 1.06 | 1.80 | 79.6% |
| k2, gate 0.6 | 71.27 | 1.18 | 2.23 | 67.4% |
| k3, gate 0.6 | 68.50 | 1.14 | 2.40 | 56.9% |
| k3, gate 0.8 | 69.65 | 1.16 | 2.55 | 71.7% |
| k4, gate 0.6 | 69.24 | 1.15 | 2.53 | 52.0% |
| k7, gate 0.6 | 63.75 | 1.06 | 2.63 | 42.2% |

Prose (2 texts x 128): off 61.55, k3 gate 0.6 66.00 (2.33 / pass), k7 gate 0.6 63.30 (2.52 / pass).

**#10874** (17c67dacf, drafter 5.6 -> 4.3 ms per cycle): greedy off 62.58; k2 gate 0.6 79.84 (1.28x); k3 gate 0.6 80.46 (1.29x);
k3 gate 0.8 80.64 (1.29x, 2.54 / pass, 72.0%). Prose off 63.35, k2 gate 0.6 74.46.

**#11227** (5d73d0fa8, device ingest; `INTARSIA_DSPARK_TAU=0 INTARSIA_DSPARK_CUT=0.5` vs k3 gate 0.8, both `--spec-k 3`,
ABAB x 3; medians):

| set | k3 gate 0.8 tok/s | k3 cut 0.5 tok/s | cut / gate |
|---|---:|---:|---:|
| golden (12 x 128) | 85.95 | 80.38 | 0.935 |
| 1k | 73.81 | 68.49 | 0.928 |
| 4k | 98.48 | 101.03 | 1.026 |
| 16k | 74.44 | 74.48 | 1.001 |
| prose | 74.83 | 73.60 | 0.984 |

Golden counters: gate 0.8 drafted 2.17 per pass, 71.8% accepted, 3.17 rows / pass, 2.56 tokens / pass; cut 0.5 drafted 2.98
per pass (the cap), 41.6% accepted, 3.98 rows / pass, 2.24 tokens / pass. Verify at 4 rows: cycle 28.44 ms, of which the
drafter 3.08 ms. The offline model had predicted cut 0.5 at +3.4% (spark2 #11150, refit #11210); it measured -6.5%.

**Not swept before:** k5 at any rule; k7 with gate 0.8 or any cut; a cut from a measured r; the code prompt.

Verify cost known so far: about +2.9 ms per extra row (#10874 totals), 3.16 ms in the refit; plain step 16.0 ms then, about
13.2 ms on today's main (76 tok/s at 256). That puts r near 0.2-0.24 and mlx-serve's formula at a cut of about 0.60-0.66.

### Why cut 0.5 hardly cut: our confidence is not theirs

- Ours (`model/dspark.rs`, `dspark_draft_inner`): p1 = the softmax top-1 of the softcapped logits over the step's 64 candidates
  only (`INTARSIA_DSPARK_CAND`, default 64: the top-64 base logits, Markov bias added on those). The vocab tail outside the
  candidates is left out of the sum, so p1 is an upper bound of the full-vocab probability.
- Theirs (`draft.zig` `markov`): the softmax top-1 over the full vocab of the softcapped base logits + the Markov bias.
- #11227 shows ours is overconfident: with the product of three p1 nearly always >= 0.5, only 41.6% of drafts were accepted. So a
  cut of 0.5 on our p1 is a much looser rule than 0.5 on theirs, and the brief's rules (b) and (c) need the full-vocab confidence
  to be the rule mlx-serve runs.
- `INTARSIA_DSPARK_CAND=0` (the exact head) already computes p1 over the full vocab, but it runs the fold GEMV and downloads two
  vocab rows every step, so its tok/s carries a slower drafter.

## L3 ruling (coordinator, 2026-11-22 ~16:30)

1. Run the probe tonight (default load, about 10 min, main's binary); it need not wait for #11621.
2. From the probe traces, simulate offline the tokens per round and the cost-adjusted speedup of each rule under both p1
   definitions, with the measured r (+2.9-3.2 ms per row).
3. Build (ii), the device full-vocab softmax behind a measurement-only env, only if the simulation shows a full-vocab cut beating
   the best rule on our p1 by >= 5% at some length; otherwise skip it and write down why.
4. The quiet job runs CAND=0 arms only for the simulation's best k, not the full grid; report measured and cost-adjusted numbers.

## 2. Probe and offline simulation

**Probe #11640** (endor, default load, main 22303d704 binary): DSpark `--spec-k 7`, gate off (`INTARSIA_DSPARK_TAU=0`, no cut),
`INTARSIA_SPEC_TRACE=1`, so every pass drafts 7 and prints each step's p1 and the accepted count. Sets: greedy golden (12 x 128),
`code_tests` (a raw unit-test file continuation, 188 prompt tokens, x 256; `.scratch/code.json`), prose (2 x 128), longp
(1021 / 4108 / 16455-token prompts x 128). Traced passes with 7 drafts: greedy 607, code 81, prose 107, 1k 51, 4k 32, 16k 39 (the
long sets are one prompt each, so their numbers are noisy).

The `INTARSIA_DSPARK_CAND=0` half (option (i), the exact head as the full-vocab confidence) failed to load on Metal: its per-step
fold GEMV (`matmul` m=1 n=200058 k=256 bf16) has no row-exact kernel, and the draft's calls run under `[row-exact verify kernels
only]`. So option (i) does not exist on today's main. In its place, `INTARSIA_DSPARK_PFULL=1` (9866126fd, measurement only):
the drafts stay the candidate head's, and the gate and the cut read the full-vocab p1, computed on the host from the downloaded
base rows and the host fold copy (several ms a step). With the trace on, each step prints both probabilities. Unit test
`full_p1_matches_naive_and_bounds_the_candidates`; the CPU DSpark tests and `docs_env_vars` pass (bespin #11643). Probe 2 (#11646)
runs the same sets with it.

**Simulation** (`.scratch/sim.py`): each traced pass is replayed under a rule. A rule that verifies n of the 7 drafts accepts
min(a, n) and emits min(a, n) + 1 tokens over n + 1 rows. The cost model per round:
- drafter 1.88 + 0.40 k + 0.25 (steps - 1) ms (#11227's 3.08 ms at k3 plus the refit's per-row head and per-step Markov costs);
- verify P + r n, with r the extra ms per verified row and P the plain step: 13.9 ms short, then 15.1 / 15.3 / 16.6 ms at 1k / 4k /
  16k (#11564's off decode);
- for full-vocab p1, optionally +0.3 ms per Markov step (a device full-vocab softmax: the 102 MB fold GEMV plus a reduction).

The passes are the k7 run's anchors, not each rule's own walk. With our p1 (candidates) the best rule against gate 0.8 k3:

| set | gate 0.8 k3 (x vs plain) | best rule, r = 2.9 | of base | best rule, r = 3.2 | of base |
|---|---:|---|---:|---|---:|
| greedy | 1.275 | cut 0.61 (from r) k3 | 1.002 | gate 0.8 k3 | 1.000 |
| code | 1.647 | gate 0.8 k3 | 1.000 | gate 0.8 k3 | 1.000 |
| prose | 1.120 | cut 0.61 k5 | 1.035 | gate 0.8 k5 | 1.023 |
| 1k | 1.107 | cut 0.5 k3 | 1.077 | cut 0.5 k3 | 1.067 |
| 4k | 1.595 | cut 0.6 k7 | 1.152 | cut 0.6 k7 | 1.140 |
| 16k | 1.456 | cut 0.61 k5 | 1.089 | cut 0.5 k5 | 1.077 |

On greedy and code, k5 / k7 lose 2-17% under every rule: an extra verified row costs about 21% of a plain step (r / P = 0.21),
and drafts 4-7 are accepted too rarely to pay for it. The long-context rows are single prompts of 32-51 passes (the 4k prompt is a
counting continuation, 95.7% acceptance), so they are leads for the quiet job, not results. Full tables: `.scratch/sim_c64.txt`.
