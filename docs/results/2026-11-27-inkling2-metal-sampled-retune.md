# Inkling-2 sampled DSpark on Metal: held-out retune of the adaptive draft length (lane ink-metal-sampled2)

Branch `ink/metal-sampled2`, from origin/main `aa7beaa73`. Opt-in on Metal; no default changes. Status: in progress (method pre-registered below
before the first Metal slot; results are filled in as they arrive).

## Result

(to be written when the measurements are in)

## 0. Brief (verbatim from the coordinator, 2026-11-27 02:07 EST, condensed only by dropping the tool-use boilerplate)

> You are lane **ink-metal-sampled2** in the IntarsiaML repo. The user is asleep; the coordinator (me) is your contact via SendMessage to "main". Work fully autonomously within this brief.
>
> **Goal.** Make sampled DSpark drafting with adaptive draft length pass its pre-registered gate on **Metal** (endor, Apple M5 Max) for Inkling-2-Mini, as an **opt-in** setting. Metal sampled drafting stays OFF by default; any Metal default flip is the user's call in the morning, not yours.
>
> **Background.** PR #543 (merged, `3fb1b5f8d`) added the adaptive policy (`crates/intarsia-core/src/spec/adapt.rs`). Knobs: env `INTARSIA_SPEC_SAMPLED_ADAPT` and recipe key `[tuning.<backend>] sampled_adapt`. Opt-in switch: `INTARSIA_SPEC_SAMPLED_DFLASH=1`. HIP passed and is now default-on (#545). Metal FAILED with `[tuning.metal] sampled_adapt = "d=0.237,r=0.236/0.418/0.583,g=0.85,margin=0.0,drop=0.0,probe=3,hold=8,hold_max=32"`: `short_b` pol/off 0.939 (T 1.0) and 0.944 (T 0.7); T 0.7 4k pol/fix 0.942 and 1k 0.976; the grid pick was fitted to seed 11's trace and seed 7 accepted much less; per-prompt 128-token single-seed cells are noisy. #543's suggested direction (untested): a stop rule that needs more evidence before stopping a well-accepting request, and a longer hold once stopped.
>
> **Gate (fixed in advance, from #543; do NOT relax).** 1. Every prompt pol/off >= 0.98 of plain sampling, `short_b` included, at T 1.0 and T 0.7. 2. Each class (short, 1k, 4k) keeps >= 98 % of the fixed-k3 gain in #543's pre-registered ratio form (pol >= 0.98 x fix); also report the gain form (pol - 1 >= 0.98 (fix - 1)). 3. Seeded ids identical to `--spec off`: 100 seeds x T 1.0 and T 0.6. 4. Greedy ids identical to main (7 prompts x 64 tokens, `--spec off` and default).
>
> **Method you must pre-register** (before the first Metal slot): held-out seeds (calibrate on one seed set, e.g. 11, 13, 17, 19; gate on a disjoint set including seed 7, e.g. 7, 23, 29; never tune on a gate seed; several seeds per prompt; per-prompt ratios with 95 % CI over seeds x reps); fix the pass/fail arithmetic and the CI method before measuring; prefer recipe-only changes (a code change to `adapt.rs` needs every new knob's default to reproduce today's behaviour exactly, proven by a cpu unit test under the HIP recipe string; HIP is the live default and there is no HIP GPU time; if a change would alter HIP behaviour, stop and report); one retry of the sim/grid at most if the first pick fails the replay, then measure what you have and report.
>
> **Resources and rules.** Worktree `.worktrees/ink-metal-sampled2` only. fleetq mandatory for endor's GPU (`--exclusive --expect <MIN>`; `--load cpu` for builds/tests). **Metal budget: <= 60 min of exclusive slots in total, each slot <= 20 min, none starting after 07:00 EST.** endor is a 140 W MacBook Pro: >= 45 s idle gaps between timed processes, ABBA order, group by macmon power state. No shell `timeout` on macOS. No tool call or wait longer than 4 min. Pushes via `fleetq run --host endor --load cpu -- git -C <worktree> push`. Metal-only code in shared test files must be `#[cfg(target_os = "macos")]`.
>
> **Deliverables.** This doc, opening with the result: speed table first (per prompt and per class, pol/off and fix/off with 95 % CI at T 1.0 and 0.7, calibration seeds and gate seeds separate), then the gate verdict per clause. If it passes: update `[tuning.metal] sampled_adapt` in `recipes/inkling2.toml` with a comment citing this doc (stays opt-in: no `sampled_dflash = true` under `[tuning.metal]`). A `docs/decisions/LOG.md` row, a unanimis capture (source names "intarsiaml"), a PR to `main` (do NOT merge). If the gate fails: still write the doc and PR it as docs only, naming clause, prompt, backend and the numbers. Write every coordinator message you receive into this doc.

## Later coordinator messages

(none yet)

## 1. Pre-registered method (written 2026-11-27 ~02:30 EST, before any Metal slot of this lane)

### 1.1 Seeds, prompts, arms

- **Calibration seeds: 11, 13, 17, 19.** Used for the cost-table check, the plain-step overhead measurement, the traces, the simulator and the choice of parameters.
  Seed 7 (the failing seed of #543) and every other seed of the gate set are never used for any of those.
- **Gate seeds: 7, 23, 29**, disjoint from the calibration set. One process per (arm, T, seed): 3 paired observations per ratio.
- **Prompts** (`raw/speed.json`, md5 `91808e2bb72d8ffcff79b5fe44f19669`): warm-up (excluded), `short_a` (golden `count`, 35 ids), `short_b` (golden `story`, 40 ids, the low-acceptance
  story prompt), `short_c` (golden `code`, 101 ids), `len1k` (1021 ids), `len4k` (4108 ids), all taken from lane ink-dflash2's `speed.json`. The file #543's slot used was lost with
  its worktree; this one is a reconstruction (`short_a` was reported as 36 ids there, `count` has 35), so absolute numbers are not comparable cell by cell with #543's, the classes and prompt names are.
  Classes: short = the three short prompts pooled (sum of tokens / sum of decode ms), 1k, 4k.
- **Run shape** (as #543): `generate --backend metal --max-seq 8192 --max-tokens 128 --ignore-eos --top-k 0 --top-p 1.0 --show-ids --profile-every 0`, recipe default DSpark k3 (cut 0.5,
  device sampler), `--seed S --temperature T`, T in {1.0, 0.7}. Arms, **all on one binary** (main `aa7beaa73`, `intarsia-b1`) unless a code change is made:
  `off` = `INTARSIA_SPEC_SAMPLED_DFLASH` unset; `fix` = `INTARSIA_SPEC_SAMPLED_DFLASH=1 INTARSIA_SPEC_SAMPLED_ADAPT=0` (fixed k3: #543's "fixed k" arm; main's PR #535 behaviour);
  `pol` = `INTARSIA_SPEC_SAMPLED_DFLASH=1 INTARSIA_SPEC_SAMPLED_ADAPT=<the chosen string>`.
- **Order and power**: 45 s GPU-idle gap before every timed process; `macmon pipe -i 250` for the whole slot; per seed a Latin-square order of the three arms (seed 7: off fix pol; seed 23: fix pol off;
  seed 29: pol off fix), both temperatures in separate slots with the same orders. A process's mean GPU power and seconds above 65 W are reported; a process that sat above 65 W for more than
  3 s, or whose arm-mean power differs by more than 25 % from its pair's, is flagged and shown, not dropped.

### 1.2 Estimator, CI, pass/fail arithmetic (fixed now)

- Per (prompt or class, T, seed): ratio of tok/s of two arms, from the `decode[name]: N tokens, X ms` lines of the two processes of the same seed (class: pooled tokens / pooled ms).
- Over the gate seeds: **geometric mean of the per-seed ratios** (the point estimate) and a **95 % t-interval on the log ratios** (n = 3, t(2) = 4.303). The interval is reported everywhere;
  with 3 pairs it is wide and a pass is decided on the point estimate, exactly as in #543. A cell whose lower bound is also >= 0.98 is marked "strict" (informational).
- **Clause 1** passes iff all ten (prompt, T) cells have pol / off >= 0.98 (point estimate).
- **Clause 2** passes iff all six (class, T) cells have pol / off >= 0.98 x (fix / off) (ratio form, pre-registered); the gain form (pol - 1 >= 0.98 (fix - 1)) is printed next to it for information only.
- **Clause 3**: `inkling2_sampled_real` (real model), 100 seeds (0..100) x T 1.0 and 0.6, 24 tokens, k3, drafting off vs the policy with the chosen string: every shared seed identical, all 100 per
  temperature shared (a time box that cuts the run short is a FAIL of the "100" part, reported with the count).
- **Clause 4**: greedy `generate --show-ids` over `raw/md5.json` (6 short golden prompts + the 4108-id prompt), 64 tokens, `--spec off` and the recipe default: the two `ids[...]` hashes
  must be equal, and equal to the same run on main (the same binary when the change is recipe-only).
- The in-sample (calibration) numbers are never part of the verdict; they only decide which string goes to the gate.

### 1.3 Calibration and the choice of the string

1. **Cost table**: kept as measured in #543's slot (`d = 0.237`, `r = 0.236 / 0.418 / 0.583`; D 3.22 ms, P 13.62 ms). One traced fixed-k3 run (seed 13, T 1.0) re-checks it with the 15 % rule of #543.
2. **Plain-step overhead of the policy mode** (`po`): arms `off` and `plainonly` (`INTARSIA_SPEC_SAMPLED_DFLASH=1 INTARSIA_SPEC_SAMPLED_ADAPT=drop=-100,hold=1000000,hold_max=1000000,probe=1`:
   one draft call and verify per prompt, then plain steps only, but with the draft stash running), ABBA, seed 13, T 1.0. Rationale: #543's `short_b` arm took 116 plain steps and 8 verifies in 127 tokens and still
   ran 6 % slower than off; if plain steps in the policy mode are themselves slower than off's, no stop rule can reach 1.0 on that prompt. `po` = off tok/s / plainonly tok/s - 1 over all prompts pooled.
   **If `po` >= 0.01 the simulator charges every policy plain step `1 + po` plain-step units** (the off arm stays 1); else 0. If plainonly / off < 0.98 the first clause cannot be met by a recipe on any prompt
   that mostly stops, and the report says so with the number.
3. **Traces**: `INTARSIA_SPEC_SAMPLED_ADAPT=drafts` runs (every step: a draft call at k3, then a plain step; the proposals at every position are logged) at seeds 11, 13, 17, 19 x T 1.0, 0.7, on the
   same prompts. The replay of any policy on a trace is exact up to the draft's own non-determinism on Metal (#543: 5 of 6 prompts replayed exactly; the sixth differed by 2 passes). The replay's
   time model: plain step 1 (off) or `1 + po` (policy), draft call `d`, a pass with `m` drafts `d + 1 + r_m`, an empty call `d + 1 + po` (a draft then a plain step), in units of P. Its fidelity is checked
   against the traced fixed run (passes / proposed / accepted, predicted vs measured ms; #543's table was within 3 % except `len4k`, where fixed k3 measured 12 % slower than predicted).
4. **Search, stage A (recipe-only, existing knobs)**: `g` {0.85, 0.9, 0.95, 0.97} x `margin` {0, 0.02, 0.05} x `drop` {0, 0.05, 0.1, 0.15, 0.2} x `probe` {1, 2, 3} x `hold` {8, 16, 32} x
   `hold_max` {32, 64, 128}, `explore` 8, the measured table. For every setting, over the four calibration seeds and both temperatures: per (prompt, T) the geometric mean of replayed pol / off, per (class, T) of
   pol / off and fix / off. **Score = min( min over (prompt, T) of pol/off / 0.98 , min over (class, T) of (pol/off) / (0.98 x fix/off) )**. The setting with the highest score wins; among settings within 0.003 of the best score,
   the one with the highest mean pol / off over all cells. Leave-one-seed-out (pick on 3 seeds, evaluate on the 4th) is printed as a diagnostic of overfitting; it does not change the pick.
5. **Stage B (only if stage A's best in-sample score is < 1.0, i.e. it predicts a gate failure on its own fit)**: this is the allowed one retry. Rules to try in the Python replay first, as new knobs
   whose default reproduces today's behaviour exactly: `stop_n` (stop only after the rate was below `1 - drop` in `stop_n` consecutive calls; default 1), `hold_grow` (the factor the hold grows by after a failed
   probe; default 2), and a minimum number of draft calls after a resume before the next stop. If one is chosen, `adapt.rs` gets it with a cpu unit test that the HIP recipe string
   (`[tuning.hip] sampled_adapt`) and the built-in tables give bit-identical decisions with and without the knob on random traces, and the binary is rebuilt (`intarsia-b2`); the arms of the gate then use `b2` for `pol` and `b1`
   (main) for `off` and `fix`. If a change could alter HIP behaviour, stop and report instead.
6. After the pick there is **no second search**. Whatever string is chosen goes to the gate; the gate verdict is reported as measured.

### 1.4 Metal budget plan (<= 60 min total; each slot <= 20 min)

| slot | content | est. |
|---|---|---|
| C | calibration (plain-step overhead A/B, traced fixed + policy, 8 `drafts` traces) | ~11 min |
| G1 | gate T 1.0: 3 seeds x 3 arms | ~11 min |
| G2 | gate T 0.7: 3 seeds x 3 arms | ~11 min |
| S | seed equality 100 x 2 temperatures x 2 arms, greedy md5 | ~12 min |

No slot starts after 07:00 EST.

## 2. Log
