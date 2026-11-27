# Inkling-2 sampled DSpark: adaptive draft length per request (lane ink-sampled-adapt)

Branch `ink/sampled-adapt`, stacked on `ink/sampled-dspark` (PR #535, head `f472a25ac`).
Opt-in only; no default changes. Whether sampled speculation becomes default-on is the user's call.

## 0. Brief (verbatim from the coordinator, 2026-11-26 ~20:00 EST)

> You are lane **ink-sampled-adapt** in the intarsiaml repo (Rust inference engine, `/opt/code/intarsiaml`). The coordinator (me) wrote this brief on 2026-11-26 at about 20:00 EST. Write this brief, and any later message from me, into your results doc first: context compaction loses messages.
>
> ## Goal
> Make sampled DSpark speculation (PR #535, opt-in via `INTARSIA_SPEC_SAMPLED_DFLASH=1`) never slower than plain sampling on any prompt, while keeping its average gain. Do it by choosing the draft length per request from that request's recent acceptance. Measure on HIP (R9700, tatooine) and Metal (endor), with a pre-registered rule. Open a PR stacked on #535 and do not merge it. No default changes: whether sampled speculation becomes default-on is the user's decision, made after your result.
>
> ## Why
> At main's DSpark k5, the sampled speed vs plain sampling (on/off, 95 % CI, from #535's results doc) is:
>
> | Backend, T | short | 1k | 4k |
> |---|---|---|---|
> | HIP, T 1.0 | 1.105 | 1.066 | 1.470 |
> | HIP, T 0.7 | 1.155 | 1.051 | 1.461 |
> | Metal (k per Metal's default), T 1.0 | 1.126 | 1.095 | 1.291 |
> | Metal, T 0.7 | 1.147 | 1.088 | 1.418 |
>
> - The story prompt `short_b` loses: HIP 0.900 at T 1.0 (27.5 % acceptance, 1.53 tokens per pass) and 0.989 at T 0.7; Metal 0.862 / 0.971.
> - At k3 the HIP story prompt was 0.935, so longer drafts hurt low-acceptance prompts.
> - Rough cost model, which you should verify: since #536 a verify row costs about 2.3 ms against a ~13.4 ms 1-row step on HIP. So a 2-row verify (k=1) at about 30 % acceptance may still beat a plain step, while a 6-row verify (k=5) loses.
> - So the lever is probably adaptive k, not just on/off.
>
> ## Read first
> - `/opt/code/intarsiaml/.worktrees/ink-sampled/docs/results/2026-11-26-inkling2-sampled-dspark.md`: section 3.1 (HIP protocol and the "k5 refresh" subsection); section 3.2 (Metal protocol, with 45 s GPU-idle gaps for endor's 140 W power budget); the seed-equality method; the `raw/` scripts (`speed.py`, `md5.py`, `window.sh`, `raw/k5/`, `raw/metal/slotB.sh`).
> - The sampled path: `crates/intarsia-core/src/spec.rs` (`sampled_dflash_*`, `sampled_ctx_k`, `k_min_for`, `mtp_default_k_min`); `crates/intarsia-core/src/engine.rs` and `engine/keyed.rs` (keyed verify, Match rule); `crates/intarsia-core/src/model/dspark.rs` (cut rule, draft state); `crates/intarsia-core/src/model/batch.rs`.
> - Shared tatooine rules: `/opt/code/intarsiaml/.worktrees/hip-swarm/COMMON-1126.md`. Follow them: windows with `--takes-down radiance` on `--host tatooine-r9700`; `HIP_VISIBLE_DEVICES` per card; no `--mem` on tatooine; restore radiance and check it is up after each window; pushes through fleetq on endor.
> - In window scripts, use `env VAR=x cmd` (never `VAR=x func` in sh), and trap and kill your own process tree.
> - Metal timing needs `fleetq run --host endor --exclusive`, with real-model tests under `INTARSIA_TEST_REAL_MODEL=1`.
>
> ## Key constraint: exactness
> - #535's Match rule is seed-exact: for a fixed seed, the output ids with drafting on equal those with drafting off, whatever the drafts were. So any per-request k policy keeps the output distribution exact.
> - Prove it: with your policy on, seeded runs must give ids identical to drafting off, on both backends (>=100 shared seeds each at T 1.0 and 0.6, like #535's seed test).
> - **Subtle part:** if the policy ever drafts 0 tokens (a plain step) mid-request, the DSpark draft state (its taps / KV of target hidden states) must stay in sync for later passes. Find out whether a k=0 step keeps it in sync today. If it does not, either keep k >= 1, or update the draft state on plain steps. Say which you chose and why. Prefer k_min = 1 if the cost model says a 2-row verify pays at about 30 % acceptance.
>
> ## Design (yours to refine; record it in the doc before measuring)
> - Track recent acceptance per request: an EMA or a window of the last N passes, of accepted drafts per pass.
> - Choose k for the next pass from it, in [k_min, k_max], with k_max = today's default.
> - The simplest good rule picks k to maximize expected tokens per ms. It uses the measured per-row verify cost and the per-row acceptance curve.
> - Make the policy and its parameters knobs (env + `[tuning.<backend>]`), documented in `docs/tuning.md`, each with a revert line.
> - Default for this PR: the policy is active only when sampled DSpark is on. Greedy DSpark is unchanged: bitwise, same k, same cut rule. Check that greedy md5 equals main.
>
> ## Work folders
> - Mac worktree: `/opt/code/intarsiaml/.worktrees/ink-sampled-adapt`, branch `ink/sampled-adapt` from `origin/ink/sampled-dspark` (head `f472a25ac`). I am merging main into `ink/sampled-dspark` and merging #535 soon. When I tell you it has landed, merge `origin/main` into your branch; until then, do not touch `ink/sampled-dspark`.
> - tatooine: `~/intarsiaml-ink-sampled-adapt` and docker volume `intarsia-cargo-target-ink-sampled-adapt`.
> - endor: build in your worktree with `CARGO_TARGET_DIR` inside it.
> - Nothing else, and do not delete any folder.
>
> ## Files you may not edit
> - Lane ink-hip-bpf owns `kernels/hip/ink_mix.hip`, `ink_attn.hip`, `ink_route_fused.hip` and the serve scheduler.
> - Lane ink-dflash2 owns `dflash*.rs` and spec routing in `load.rs`.
> - Lane ink-hip-i8 owns new int8 files.
> - Shared files are add-only. If you need to change spec routing, tell me first.
>
> ## Measure
> 1. **HIP:** at most 2 windows of 5:00 or less each, 10:00 total. Sampled on/off ABBA, 4 + 4 processes across both cards (each on/off pair on one card, as in the k5 refresh). Cells: short (per prompt) / 1k / 4k, T 1.0 and 0.7. Arms: off, on at fixed k5 (today), on with your policy. Also: greedy md5 vs main, spec default and `--spec off`, short + 4k, 64 tokens; and seeded equality.
> 2. **Metal:** one endor `--exclusive` slot of at most 30 min. Same arms and cells, 45 s GPU-idle gaps, macmon grouping. Seeded equality.
>
> ## Pre-registered rule (write it in the doc before measuring)
> **PASS** (the user can then decide default-on) needs all four:
> - with the policy on, no prompt is slower than off on either backend (the per-prompt geomean ratio is >= 0.98 at both temperatures, `short_b` included);
> - every class (short / 1k / 4k) keeps at least 98 % of the fixed-k5 on/off ratio, or beats it;
> - seeded ids are identical to off;
> - greedy md5 equals main.
>
> Otherwise, report which part failed and the best setting found.
>
> ## Report
> One final answer: the table first (per backend, T and class, ratios for off / fixed-k5 / policy, with CI); the per-prompt ratios, `short_b` first; the verdict against the rule; the policy as built, with its knobs and a revert line; the draft-state sync finding for k=0; checks run (command + result); radiance down per window; the endor slot length; fleetq ids; PR link and head; a LOG.md row (add-only); the unanimis capture id; leftovers.

## Later coordinator messages

**Coordinator, 2026-11-26 ~20:35 EST** (verbatim):

> 1. #535 is merged into main as `aee06201c`. That merge commit includes `ink/sampled-dspark` at `15bca842c`, which is main `a4c8f3b19` merged in: #538 packed prefill default-on with `--max-batch`, and #540 cpu-only int8 / fp8 study.
>    - Merge `origin/main` into `ink/sampled-adapt`, using a merge, not a rebase.
>    - Open your PR against `main`.
>    - Your fixed-k5 arm is main itself now.
> 2. Cleanup: tell me in your next report whether you still use any of these:
>    - the Mac worktree `.worktrees/ink-sampled` (including its `.scratch/`);
>    - the remote branch `ink/sampled-dspark`;
>    - the tatooine folder `~/intarsiaml-ink-sampled`, with binaries `intarsia-br5` / `intarsia-main5`, and its docker volume.
>    I will not touch them until you answer.

Consequences here: the greedy-md5 baseline and the fixed-k arm move to the new main (built as `main6` from `origin/main` after the merge);
the PR goes against `main`. Of the ink-sampled folders this lane read `raw/` scripts and the two prompt files (`speed.json`, `md5.json`,
copied into this lane's `.scratch/prompts/`) and uses nothing else.

## 1. Findings before building (2026-11-26 ~20:30 EST)

### 1.1 The brief's cost premise does not hold for the sampled path: k_min = 0

Since #536 a verify pass costs, on HIP at 1k (probe numbers, `2026-11-26-inkling2-hip-spec.md` "Verify cost", `2026-11-26-inkling2-hip-vrow.md`):
decode 13.30 ms; verify 2 / 3 rows 16.75 / 19.58 ms (eager; graph 16.48 / 19.00), 4 / 5 rows 21.49 / 23.63 ms. So the first extra row costs
about +3.4 ms, the later ones +2.1-2.8 ms. A sampled DSpark cycle also pays the draft call, about 3.5 ms on HIP (block 2.11 + step 0.70 +
head 0.71 + ingest 0.20, `INTARSIA_DSPARK_PROFILE`), 3.2 ms on Metal (the `keyed draft` lines of #535's Metal slot B, device draws).

- A k = 1 cycle is then about 13.3 + 3.5 + 3.4 = 20.2 ms for 1 + a tokens (a = the first draft's acceptance): it pays only at a >= ~0.52.
  At about 30 % acceptance it loses about 12 %. k = 2 at 30 % per position loses too.
- So **k_min = 1 cannot fix `short_b`**. The policy must be able to skip the draft call (a plain step, k = 0) before `dflash_draft` runs.

### 1.2 Empty drafts are a hidden cost

Under the cut rule (0.5 on HIP and Metal) a draft call can return 0 drafts (first top-1 probability below 0.5): the request has paid the
draft and then takes a plain step. `keyed_finish`'s empty branch never calls `account_verify`, so no counter sees these cycles.
From #535's per-prompt `spec[...]` lines (tokens = 1 + passes + accepted + plain), the share of draft calls that came back empty:

| backend, k | T | short_a | short_b | short_c | 1k | 4k |
|---|---|---:|---:|---:|---:|---:|
| hip, k5 (`raw/k5/speed`) | 1.0 | 15 % | **41 %** | 8 % | 36 % | 14 % |
| hip, k5 | 0.7 | 22 % | 36 % | 5 % | 48 % | 24 % |
| Metal, k3 (`raw/metal-B`) | 1.0 | 9 % | **57 %** | 6 % | 32 % | 29 % |
| Metal, k3 | 0.7 | 22 % | 47 % | 4 % | 16 % | 32 % |

At ~3.5 ms each, `short_b`'s 40 empty calls at T 1.0 on HIP are ~140 ms of its 1864 ms: most of its 10 % loss. The policy counts an empty
call as a draft call that yielded one token (the plain step's) and cost the draft plus a plain step.

### 1.3 A plain step (k = 0) keeps the DSpark draft in sync today

Code read (no change needed):

- Every target pass ends in `Model::dflash_after_forward` (`model/dflash.rs`), from both the captured decode step (`model.rs`
  `forward_inner`, the graph branch) and the eager pass. For DSpark, a 1-row pass while `active && synced && kv.len + pending + 1 == st.len`
  copies its tap row into the sequence's stash (`dflash_stash`, `pending += 1`); a longer pass flushes the stash first and then ingests its
  own rows. The stash flushes by itself when it is full: min(512 (`PEND_MAX`), the workspace rows) rows (`dflash_pend_max`; the real model's
  chunk is 2048, so 512; the tiny fixture's 16, which the test below overflows on purpose).
- `dflash_draft_rank` flushes the stash before every block pass and then checks `kv.len == st.len` (an error, not a silent desync).
- The Markov head's `f_rnn` ring (64 positions) is written for the last 64 rows of every ingest, so the row the draft reads (position
  `p0 - 1`) is current after any flush length.
- This path is live today: empty drafts (above), the back-off gate's skips and `plain_next` all take plain steps between draft calls.

Seeded-id equality cannot detect a desynced draft (Match is exact whatever the drafts are; desync only lowers acceptance), so a test
that can fail: `tests/inkling2_dspark_sampled_adapt.rs` `plain_steps_keep_the_draft_in_sync` (tiny fixture): after prefill and `n`
plain decode steps (n = 2, 3, 17; 17 overflows the tiny stash once), the state is `synced` with `kv.len + pending == st.len`, and the next draft equals the draft of a
state that ingested the same tokens in one pass (prefill of prompt + tokens). Decision: **k_min = 0**, no change to the draft state.

## 2. The policy (pre-registered before any GPU window)

Applies only to a sampled request that drafts with a DFlash-family draft by #535's keyed rule (`Seq::spec_keyed && spec_dflash`:
`INTARSIA_SPEC_SAMPLED_DFLASH=1`, `--spec dflash` or the recipe's DSpark default, single sequence, one rank). Greedy DSpark, `--spec
mtp`, `--spec auto` and the batching engine's keyed MTP are unchanged (same k, same cut rule, same code path).

**Per request** (`spec::AdaptK`), K = the request's k (today's default: hip k5, Metal k3), per draft call with cap `c` (rows above `c`
are not updated):

- discounted counts for rows j = 1..c: `N_j = g N_j + 1`, `Q_j = g Q_j + [row j proposed]`, `S_j = g S_j + [row j accepted]`;
  estimates `q_j = (Q_j + n0 q0_j) / (N_j + n0)`, `s_j = (S_j + n0 s0_j) / (N_j + n0)` with the prior `q0_j = a0^(j-1)`, `s0_j = a0^j`
  (a0 = 0.6, n0 = 1: a request starts at cap K, as fixed k does; amended 20:15 before any measurement: the first draft had `q0_j = 1`,
  which made the prior pick k3 on the hip table, caught by the unit test). An empty call is `proposed = accepted = 0`.
- expected tokens of a call with cap k: `T(k) = 1 + sum_{j<=k} s_j`; expected cost in plain steps (P): `C(k) = d + 1 + sum_{j<=k} q_j
  (r_j - r_{j-1})` (r_0 = 0), from a static per-backend cost table: d = the draft call, r_m = the extra cost of a verify pass with m
  drafts over a plain step, all over the plain step's time P.
- next cap: `k* = argmax_k T(k) / C(k)` over 1..K. Draft with cap k* if `T(k*) / C(k*) >= 1 + margin`; else take plain steps (no
  draft call) for `hold` steps, then one probe call at cap K, then decide again. `hold` starts at `hold0` and doubles after each probe
  that does not resume drafting, up to `hold_max`; it resets when drafting resumes. Every `explore`-th draft call uses cap K (keeps the
  rows above k* current).

**Parameters** (defaults; env `INTARSIA_SPEC_SAMPLED_ADAPT`, recipe `[tuning.<backend>] sampled_adapt`, docs/tuning.md):

| parameter | hip | Metal | other backends |
|---|---|---|---|
| d (draft call / P) | 0.26 | 0.24 | 0.26 |
| r_1..r_K (verify with m drafts, extra / P) | 0.26 0.47 0.60 0.75 0.91 (+0.16 per row after, extrapolated) | 0.32 0.51 0.70 (+0.19 per row after) | the hip row |
| g, n0, a0 | 0.85, 1, 0.6 | same | same |
| margin | 0.02 | same | same |
| hold0, hold_max, explore | 8, 64, 8 | same | same |
| drop, probe (amendment 2; g became 0.9) | 0, 2 | same | same |

The hip table is the probe numbers above over P = 13.3 ms; the Metal one is d = 3.2 / 13.2 and a least-squares fit of #535's Metal slot
B per-prompt totals with the draft fixed at 3.2 ms (first row +4.2 ms, later rows +2.5 ms).

`INTARSIA_SPEC_SAMPLED_ADAPT=0` (or `off`) is fixed k (PR #535's behaviour: the "fixed k" arm and the revert line). The diagnostic
`=drafts` makes every step a draft call at cap K followed by a plain step (no verify; with `INTARSIA_SPEC_TRACE=1` it logs each
call's drafts: the offline simulator's input). With `INTARSIA_SPEC_TRACE=1` the policy logs every step's wall time and kind (`adapt:`
lines).

## 3. Measurement plan and pre-registered rule (written before W1)

**HIP window W1** (radiance down <= 5:00; card 0 / card 1 at once):

1. greedy md5: card 0 main, card 1 this branch, each `--spec off` and the recipe default (DSpark k5), short + 4k (`md5.json`), 64 tokens,
   no `INTARSIA_*` variable set.
2. calibration (branch, `INTARSIA_SPEC_SAMPLED_DFLASH=1`, `INTARSIA_SPEC_TRACE=1`, `speed.json`, 128 tokens, **seed 11**, so neither the
   cost table nor the simulator sees the measured token streams of seed 7): card 0 T 1.0, card 1 T 0.7; each one fixed-k5 run
   (`INTARSIA_SPEC_SAMPLED_ADAPT=0`) and one `=drafts` run.
3. seeded equality: `tests/inkling2_sampled_real.rs` at k5, 32 tokens, seeds 0..100 x T 1.0 / 0.6: card 0 drafting off, card 1 the
   policy on (default parameters, traced: k = 0 stretches and re-entries counted from the `adapt:` lines).

**Cost table rule (pre-registered).** From W1's fixed-k5 traces: D = median `keyed draft` ms; for each m with >= 10 passes, the cycle
time of a pass with m drafts = median `adapt: step` ms; from the `=drafts` runs P = median step ms - median draft ms. Then d = D / P,
r_m = (cycle_m - D) / P - 1. W2 runs the built-in table unless one of d, r_1..r_5 differs from it by more than 15 %; then W2's policy arm
gets the measured table by env and the final commit makes it the recipe default (`[tuning.hip] sampled_adapt`). The other parameters
change only if the offline simulation on W1's seed-11 `=drafts` traces (fixed-k5 replay checked against the measured fixed-k5 run first)
predicts, for the defaults, a prompt below 0.99 x off or a class below 0.99 x fixed k5; then W2 takes, from the grid g {0.8, 0.85, 0.9}
x margin {0, 0.02, 0.05} x hold0 {4, 8, 16} x hold_max {32, 64}, the setting with the highest predicted minimum per-prompt ratio among
those keeping every class >= 0.99 x fixed k5 (both temperatures).

**HIP window W2** (<= 5:00): sampled speed, `speed.json`, 128 tokens, `--ignore-eos --top-k 0 --top-p 1.0 --seed 7`, recipe default (no
`--spec`: DSpark k5, cut 0.5), T 1.0 then 0.7, three arms: off (`INTARSIA_SPEC_SAMPLED_DFLASH` unset), fixed k5 (`=1`,
`INTARSIA_SPEC_SAMPLED_ADAPT=0`), policy (`=1`). Both cards at once, the same arm at the same time; card 0 reps 1-2, card 1 reps 3-4;
per card and temperature the order is off fix pol | pol fix off. 4 processes per arm and temperature, every (off, fix, pol) triple of a
rep on one card.

**Metal slot** (endor `--exclusive`, <= 30 min): 45 s GPU-idle gap before every timed process, `macmon pipe -i 250`, `speed.py
--macmon` power groups. Seeded equality first (as W1 step 3, Metal default k3, 24 tokens, drafting off then the policy). Then one
traced fixed-k3 calibration process (seed 11, T 1.0) and the same 15 % rule for the Metal table, computed in the slot script. Then the
speed A/B as W2 with k3 and **3 reps per arm** (not 4: 24 timed processes with 45 s gaps do not fit 30 min), Latin-square orders
per temperature (off fix pol / fix pol off / pol off fix).

**Amendments before any window (2026-11-26 ~20:30 EST, after #535 merged as `aee06201c`; nothing measured yet):**

- Binaries: branch `ad3` = this branch with `origin/main` merged (`56ec7c465`+); main `main6` = `aee06201c`. Greedy md5 is ad3 vs main6.
- The fixed-k arm is **main6** with `INTARSIA_SPEC_SAMPLED_DFLASH=1` (coordinator: "your fixed-k5 arm is main itself now"), on both
  backends; off and policy run on ad3. (`INTARSIA_SPEC_SAMPLED_ADAPT=0` on ad3 is the same code path: `Seq::adapt` stays `None`.)
- W1 adds a third traced calibration run per card, the policy at seed 11: the offline replay must reproduce its passes / proposed /
  accepted exactly (checks the Python port of `AdaptK` in `raw/sim.py`). Seeds at 24 tokens (not 32), budget 90 s, so that 100 seeds
  x 2 temperatures fit; the summary counts shared seeds per temperature.
- W2: three arms x 4 reps = 12 processes per card do not fit a 5-min window (~25 s per process in #535's k5 refresh). W2 runs **3
  reps per arm**, card 0 all of T 1.0, card 1 all of T 0.7, a Latin square per rep (off fix pol / fix pol off / pol off fix): 9
  processes per card, every (off, fix, pol) triple of a rep on one card. Pairs per ratio: 3 (95 % t-interval with t(2) = 4.303).
- Metal slot: one traced fixed-k3 calibration process only (no drafts run; P from its empty calls); 3 reps per arm as W2, both
  temperatures inside each rep.

**Amendment 2 (2026-11-26 ~20:40 EST, before any window).** A synthetic replay (`raw/sim.py` on made-up traces) showed the first
draft's weak spot: stopping and resuming used one threshold (`1 + margin`), the estimate's noise over ~7 calls is about +-0.2 in rate,
so a prompt that pays (rate ~1.1) stops on a random dip, and one probe call moves the estimate too little to resume, while the hold
doubles. Changes, all knobs: a separate stop threshold (`drop`: stop when the rate is below `1 - drop`; resume needs `1 + margin`),
a probe of `probe` calls (the last decides), and `g` 0.9 (memory ~10 calls). Defaults now `g=0.9,margin=0.02,drop=0,probe=2,hold=8,
hold_max=64,explore=8` (cost tables unchanged). The pre-registered grid becomes g {0.85, 0.9, 0.95} x margin {0, 0.02, 0.05} x drop
{0, 0.05, 0.1} x probe {1, 2, 3} x hold0 {8, 16} x hold_max {32, 64} (324 settings), same objective and bars (every prompt >= 0.99 x off
and every class >= 0.99 x fixed predicted, both temperatures; the highest predicted minimum per-prompt ratio wins); it runs on W1's
seed-11 traces, so W2 (seed 7) and Metal test it out of sample.

**Analysis** (`raw/speed3.py`, from #535's `speed.py`): per arm pair, the ratio of pooled tok/s (sum of tokens / sum of time over the
class's prompts), geomean over reps, 95 % t-interval over the paired log ratios (pairs within a rep), per class and per prompt;
acceptance, tokens per pass and plain-step tokens per arm; ids identical across all processes of a temperature.

**PASS** (the user can then decide default-on) needs all four:

1. with the policy on, no prompt is slower than off on either backend: the per-prompt geomean ratio policy / off is >= 0.98 at both
   temperatures, `short_b` included;
2. every class (short / 1k / 4k) keeps at least 98 % of the fixed-k on / off ratio, or beats it: (policy / off) >= 0.98 x (fix / off),
   per backend and temperature;
3. seeded ids are identical to off (W1 step 3 and the Metal seeds; also all speed processes of a temperature);
4. greedy md5 equals main (W1 step 1).

Otherwise the report names the part that failed and the best setting found.
