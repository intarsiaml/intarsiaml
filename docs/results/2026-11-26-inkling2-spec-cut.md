# Inkling-2-Mini on Metal: DSpark draft rule, cut 0.5 against gate 0.8 (lane ink-spec-cut, 2026-11-26)

Branch `ink/spec-cut` (worktree `.worktrees/ink-spec-cut`), based on main `254e9985a`. Question: should the Metal DSpark default change from
the per-step gate (`INTARSIA_DSPARK_TAU` 0.8, k3, PR #496) to the cumulative cut (`INTARSIA_DSPARK_CUT` 0.5, no gate, k3)?

## Verdict

**Yes, flip.** Within one process, with the rule alternated request by request, the cut is faster than the gate at 1k, 4k and 16k prompt
tokens (paired CIs above 1.0 at all three, 2 / 3, 3 / 3 and 2 / 3 prompts at or above +2 %, no prompt below 0.99 at those lengths) and
indistinguishable from it on short prompts (+0.6 %, CI 1.000-1.012). Greedy ids equal `--spec off` in all 1200 requests of the A/B and
in the real-model default-binary check. The default is Metal only: CUDA and cpu keep the gate 0.8 (their cut was never measured).

| decode, cut 0.5 k3 / gate 0.8 k3 | short (15 prompts) | 1k | 4k | 16k |
|---|---:|---:|---:|---:|
| paired ratio, geometric mean [95 % CI] | **1.006** [1.000-1.012] | **1.042** [1.035-1.050] | **1.119** [1.110-1.128] | **1.032** [1.022-1.041] |
| paired ratio, median [95 % CI] | 1.007 [0.997-1.014] | 1.050 [1.036-1.058] | 1.116 [1.105-1.135] | 1.042 [1.028-1.053] |
| cluster bootstrap (per process x cycle) of the geomean | [1.003-1.009] | [1.038-1.047] | [1.115-1.123] | [1.029-1.035] |
| pairs, pairs with B > A | 300, 163 | 120, 95 | 84, 84 | 48, 40 |
| prompts with median ratio >= 1.02 / < 0.99 / total | 6 / 7 / 15 | 2 / 0 / 3 | 3 / 0 / 3 | 2 / 0 / 3 |
| lowest per-prompt median ratio | 0.928 | 0.993 | 1.074 | 0.999 |
| median tok/s, A (gate) / B (cut) | 84.8 / 83.9 | 82.1 / 86.1 | 74.6 / 83.2 | 89.7 / 94.2 |
| draft acceptance A / B (accepted / proposed, verify passes only) | 71 % / 58 % | 80 % / 72 % | 92 % / 81 % | 90 % / 82 % |

Per process (geomean B / A, independent processes p1 / p2): short 1.010 / 1.003, 1k 1.036 / 1.049, 4k 1.113 / 1.125, 16k 1.033 / 1.031.
The gate the lane set (>= +2 % at a length on more than one prompt, paired CI lower bound > 1.0 there, no length with the upper bound
below 0.99) holds: 1k, 4k and 16k pass; every length's upper bound is above 1.0.

Short prompts are a wash on average but not per prompt: 6 of 15 prompts gain >= 2 % and 7 of 15 lose more than 1 % (worst median 0.928).
At k3 the cut is strictly more permissive than the gate: a gate-kept chain has every top-1 probability >= 0.8, so its product over at most three
steps is >= 0.512, above the cut 0.5; the cut also drafts through steps with p1 below 0.8 while the running product stays >= 0.5. It therefore
proposes more tokens per cycle and gets a lower acceptance (table), and wins where a verify pass is dear relative to a draft step (long contexts).
(At k5 and k7 the cut can stop earlier than the gate; those arms are only in the cross-process sweep.) The engine's counters do not count plain
steps (an empty draft) as verify passes, so tokens per pass are not comparable between the arms and are not tabulated. Per short prompt (median
of 20 pairs, prompts in file order): 0.979 1.069 1.012 1.004 **0.928** 1.080 0.961 **0.933** 1.027 1.037 1.056 0.988 0.979 1.094 0.986; the two worst
are not an early-EOS artifact (119 and 128 tokens generated).

## What changed (code)

- `crates/intarsia-core/src/model/dspark.rs`: `draft_rule(metal, tau_env, cut_env)` replaces the two inline reads. Metal: TAU default 0
  (no per-step gate), CUT default 0.5, **but the CUT default applies only while `INTARSIA_DSPARK_TAU` is unset**, so `INTARSIA_DSPARK_TAU=0.8`
  alone is exactly today's rule (gate 0.8, no cut). `INTARSIA_DSPARK_CUT=0` turns the cut off. Other backends: unchanged (TAU 0.8, cut only
  when set). `--spec-k` default stays 3. Unit test `rule_tests::defaults_and_env_overrides`.
- `docs/tuning.md` rows for both variables, `recipes/inkling2.toml` comment, `docs/decisions/LOG.md` rows, `--spec-draft` help text.
- Ids are unaffected by construction: a draft only proposes, every emitted token is a verify row's argmax.

## Method

Cross-process timing on endor is unreliable (a process settles about 25 % slower after 0.4-2 s of GPU work, lane ink-mode), so the A/B runs
**inside one process**. A measurement-only patch (kept out of the tree, `.scratch/diag.patch` on the lane's worktree: a run-time
`(tau, cut)` override read per draft step, and `generate --prompt-ids` looping cycles x prompts x arms with
`INTARSIA_DIAG_ARMS="g3=0.8,0;c5=0,0.5" INTARSIA_DIAG_CYCLES=n`) alternates the two rules per request: cycle 0 is warm-up and dropped, even
cycles run A then B, odd cycles B then A, the sequence is reset before every request (no prefix reuse), so each B sample is compared with the
A sample of the same prompt and cycle a couple of seconds away. Decode tok/s is the engine's own clock (`decode[...]`, 128 new tokens,
`--max-seq 20480`, no `--ignore-eos`), `generate -m Inkling-2-Mini --backend metal --profile-every 0 --spec dflash --spec-k 3`, all
`INTARSIA_*` variables unset, diag binary md5 `564f3616` (main `254e9985a` plus the patch). One process per length class, run twice (p1 then p2):
short = the 15 prompts of the sweep's `greedy` (10), `code` (3) and `prose` (2) sets (10 cycles per process), 1k / 4k / 16k = the three prompts
each of `longp` and `longp2` at that length (20 / 14 / 8 cycles per process). The CI is a 4000-draw bootstrap of the paired ratios; the cluster
bootstrap resamples whole (process, cycle) blocks, which is the more conservative reading since the prompts of one cycle share the box state.

Runs: fleetq #11993 (`--load cpu`, diag build, exit 0), #11994 (exclusive, spec-off reference for every class, then the two A/B passes, 00:23-01:14,
box load average 1.6-3.6, exit 0).

### Ids

`--spec off` on every prompt of every class, then every request of the A/B (both arms, warm-up cycles included, 1200 requests, 128 tokens
each): **1200 / 1200 identical to the spec-off ids of the same prompt**, so the rule change never reaches the output, and the override path
(rule flipped between requests) is clean too.

## The timed sweep that prompted it (fleetq #11991, cross-process, A B B A, 2 reps; binary = main `22303d704` + a measurement-only variable)

Decode ratio to today's gate 0.8 k3, short / 1k / 4k / 16k: cut 0.5 k3 1.019 / 1.062 / 1.138 / 1.042; cut 0.5 k5 1.007 / 1.078 / 1.125 / 1.051;
cut 0.56 (fitted from the draft's cost slope) k3 1.015 / 1.054 / 1.104 / 1.029; gate 0.8 k5 0.992 / 1.027 / 0.993 / 1.023. Generated ids were
identical to spec-off in every arm. The within-process table above reproduces the sign and the order of the effect at every length, with a smaller
short-prompt gain (1.006 against 1.019, inside the cross-process noise) and a nearly identical 1k / 4k / 16k (1.042 / 1.119 / 1.032 against 1.062 /
1.138 / 1.042). Not re-measured within-process: cut 0.5 k5 (the sweep has it at most +1.5 % over cut k3 at 1k and 16k and below it at short and 4k,
under the 2 % bar the lane set), so k stays 3.

## Tests (flip tree `bb2eb28d7`, fleetq #11996, exclusive)

TESTS_PLACEHOLDER

## Revert

`INTARSIA_DSPARK_TAU=0.8` restores today's rule exactly (gate 0.8, no cut) with no other change; `--spec off` or `INTARSIA_SPEC_DEFAULT=0` turns
speculation off. `INTARSIA_DSPARK_CUT=0.3` (and so on) tunes the cut. One-constant revert in code: `CUT_DEFAULT_METAL` and the Metal branch of `draft_rule`.

## Not measured

CUDA and cpu (unchanged by this flip), cut 0.5 k5 within one process, the serve path (the rule lives in the draft, so `serve` and `generate`
read it the same way: one read per model load), sampled requests (they take plain steps).
