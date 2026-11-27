# Inkling-2 sampled DSpark: adaptive draft length per request (lane ink-sampled-adapt)

Branch `ink/sampled-adapt`, stacked on `ink/sampled-dspark` (PR #535, head `f472a25ac`).
Opt-in only; no default changes. Whether sampled speculation becomes default-on is the user's call.

## Result (2026-11-26 ~21:55 EST; details in sections 4-8)

Sampled decode, on / off ratio (fixed = main's fixed k, adaptive = this PR's policy; 3 paired reps, 95 % CI in section 5 / 6):

| backend, T | short fixed | short adaptive | 1k fixed | 1k adaptive | 4k fixed | 4k adaptive | `short_b` fixed | `short_b` adaptive |
|---|---|---|---|---|---|---|---|---|
| HIP, 1.0 | 1.101 | 1.172 | 1.049 | 1.042 | 1.488 | 1.466 | 0.896 | **0.981** |
| HIP, 0.7 | 1.143 | 1.163 | 1.046 | 1.068 | 1.456 | 1.458 | 0.982 | 0.991 |
| Metal, 1.0 | 1.126 | 1.144 | 1.111 | 1.123 | 1.312 | 1.358 | 0.855 | **0.939** |
| Metal, 0.7 | 1.110 | 1.133 | 1.085 | 1.059 | 1.425 | 1.342 | 0.923 | **0.944** |

Gate (fixed in advance): HIP meets all four clauses on the point estimates (lowest prompt 0.981); **Metal fails clauses 1 and 2** (`short_b` 0.939 / 0.944;
T 0.7 1k and 4k at 98 % of fixed k3's gain: 1.059 vs 1.063, 1.342 vs 1.397). Seeded ids and greedy ids are identical on both backends. Section 7 has the table
per clause. No default changed; sampled DSpark stays opt-in.

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

## Resume brief (coordinator, 2026-11-26 ~21:00 EST, new agent after the restart; condensed, wins over earlier text where it differs)

> Gate, fixed in advance, do not relax: (1) every prompt >= 0.98 of plain sampling, `short_b` included, on HIP and Metal; (2) each class
> (short, 1k, 4k) keeps >= 98 % of the fixed-k5 gain; (3) seeded output identical to `--spec off`; (4) greedy output identical to main.
> Steps: rebuild hip ad4 (check the source did not change mid-build); W1 then `sim.py` on `logs/w1/cal` (15 % table rule, replay check,
> grid if needed); W2 with ADAPT from the sim; Metal `raw/metal/slot.sh` on endor `--exclusive --expect 30`; full cpu suite on the final
> head (endor `--load cpu`); fill this doc with the speed table first (per prompt and per class, on/off ratio with 95 % CI, fixed k5 and
> adaptive, T 1.0 and 0.7, HIP and Metal), gate verdict per clause, LOG.md row, unanimis capture; push via `fleetq run --host endor
> --load cpu`, PR to `main`, do not merge, do not turn sampled DSpark on by default. If the gate fails: name clause, prompt, backend,
> margin, and what the sim says would fix it, then stop (one extra tuning round inside the budget is fine).
> Budget: radiance <= 10 min total; endor Metal one exclusive slot <= 30 min. Windows: `--host tatooine-r9700 --takes-down radiance
> --quiet gpu,cpu`, each <= 5 min, none started after 06:00 EST. Never `fleetq release` a running window job (does not stop the Mac
> driver; radiance was down 20:38-20:42 for that reason): kill the driver's process tree, confirm `curl -s http://tatooine:8080/v1/models`
> answers; confirm radiance after every window. Waits <= 4 min per call. Merge `origin/main`, never rebase. Work only in this worktree
> and `~/intarsiaml-ink-sampled-adapt`. unanimis capture: no `project`/`scope`, `intarsiaml` in `source`.
> Final report <= 15 lines: speed table first, gate per clause, seed and greedy identity, PR number + head, radiance and Metal minutes,
> not verified, fleetq ids.

Resume log: 21:00 hip ad4 rebuilt on tatooine (#12342, source = `d23914b2d`, tree synced from this worktree at 21:00:38, unchanged during the build).

## 4. HIP window W1 (2026-11-26 21:12:07-21:15:58, radiance down 3 min 51 s, fleetq #12349, binaries ad4 `8c94480a` = `d23914b2d` source, main6 `e8f62379`)

Raw: `raw/w1/` (copied from tatooine `logs/w1/`), `raw/w1/sim.txt`.

- **Greedy md5 (clause 4, HIP): identical.** 7 prompts (6 short + 4k) x 64 tokens; the `ids[...]` lines of main6 `--spec off`, ad4 `--spec off`,
  main6 default (DSpark k5) and ad4 default hash to the same `b75db738189a28fa84f4e8b0a9c0dc7d` (441 tokens after the first; default
  62.3 % acceptance, 140 passes on both binaries: greedy DSpark is bit-for-bit unchanged).
- **Seed equality (clause 3, HIP): identical.** Real model, k5, 24 tokens, seeds 0..99 x T 1.0 / 0.6: drafting off vs the policy on (default
  table, traced): 100 / 100 shared and identical at T 1.0, 100 / 100 at T 0.6 (200 requests each, 75 s / 83 s). The policy arm made 412 phase
  changes (217 stops, 28 resumes), 521 verify passes, 859 proposed, 194 accepted, so k = 0 stretches and re-entries were exercised. Tiny
  hip suite `inkling2_dspark_sampled_adapt` 2 / 2.
- **Cost table (15 % rule, `sim.py`):** D (keyed draft) 3.10 ms, P = 13.23 ms (drafts run: step 16.25 ms - draft 3.03 ms), so d = 0.234
  (built-in 0.26, -10 %). Cycle time with m drafts: m = 1 19.93 ms (n 139), 2 22.91 (87), 3 25.45 (64), 4 29.36 (43), 5 32.34 (122), so
  r = 0.273 / 0.498 / 0.690 / 0.985 / 1.211 (built-in 0.26 / 0.47 / 0.60 / 0.75 / 0.91). r_4 (+31 %) and r_5 (+33 %) exceed 15 %: the rule
  selects the **measured table**. Rows 4 and 5 cost more in the sampled cycle than the probe-derived table (eager verify probe) said; cause not investigated.
- **Replay check: exact.** Fixed k5 replayed on the `drafts` run's proposals reproduces the fixed run's passes / proposed / accepted for all
  12 prompt x temperature runs; predicted time 0.91-1.05 x measured (len4k 0.91: the cycle table is one per-m median over all prompts; the 4k prompt's longer context is not modelled).
- **Policy replay with the measured table and the other parameters at their defaults** misses the 0.99 bars at T 0.7 `short_b` (0.987)
  and T 1.0 1k (pol / (0.99 x fix) 0.909), so the pre-registered grid ran (324 settings). Best (highest predicted minimum prompt among
  settings keeping every class >= 0.99 x fixed): `g=0.95, margin=0, drop=0.05, probe=1, hold=16, hold_max=64`.
  Predicted pol / off (seed-11 traces, in-sample): `short_b` 0.989 (T 1.0) / 0.992 (T 0.7); short 1.169 / 1.148, 1k 1.170 / 1.073, 4k
  1.573 / 1.614 against fixed k5 short 1.102 / 1.109, 1k 1.163 / 1.041, 4k 1.573 / 1.614. W2 (seed 7) is the out-of-sample test.
- **W2 policy arm:** `INTARSIA_SPEC_SAMPLED_ADAPT=d=0.234,r=0.273/0.498/0.69/0.985/1.211,g=0.95,margin=0.0,drop=0.05,probe=1,hold=16,hold_max=64`.
- cpu suite, merged head (endor `--load cpu`, #12351, `.scratch/logs/cpu-2105`): all pass: `inkling2_dspark_sampled_adapt` 2/2,
  `inkling2_dspark_sampled` 2/2 (3 ignored real-model), `_gumbel` 1/1, `inkling2_dspark` 4/4, `_cut` 1/1, `verify_softcap` 1/1, `recipes_doc` 3/3,
  `docs_env_vars` 2/2, core lib 280/280 (4 ignored). Found on the way: the merge of main left two `BackendTuning { .. }` literals in
  `crates/intarsia-models/tests/mtp.rs` without `sampled_adapt` (test target did not compile); fixed (`3d2175adb`).

## 5. HIP window W2 (2026-11-26 21:33:3x-21:37:01, radiance down about 3 min 30 s, fleetq #12356; binaries as W1)

Arms: off = ad4 (`INTARSIA_SPEC_SAMPLED_DFLASH` unset), fix = main6 (main `aee06201c`, fixed k5) with `=1`, pol = ad4 with `=1` and
`INTARSIA_SPEC_SAMPLED_ADAPT=d=0.234,r=0.273/0.498/0.69/0.985/1.211,g=0.95,margin=0.0,drop=0.05,probe=1,hold=16,hold_max=64`. `speed.json`, 128
tokens, seed 7, recipe default (DSpark k5, cut 0.5), 3 reps per arm, Latin square per rep, card 0 all of T 1.0, card 1 all of T 0.7. Ratios are
geomeans over the 3 paired reps with a 95 % t-interval (t(2) = 4.303); raw: `raw/w2/`, analysis `raw/speed3.py`.

| T | class | off tok/s | fix tok/s | pol tok/s | fix / off | pol / off | pol / fix | acc., tok/pass, plain tokens: fix / pol |
|---|---|---:|---:|---:|---|---|---|---|
| 1.0 | short | 76.10 | 83.77 | 89.17 | **1.101** [1.098-1.103] | **1.172** [1.137-1.207] | **1.064** [1.034-1.095] | 40.9 %, 2.16, 53 / 52.2 %, 2.31, 104 |
| 1.0 | 1k | 74.18 | 77.85 | 77.31 | **1.049** [1.030-1.069] | **1.042** [1.003-1.083] | **0.993** [0.961-1.026] | 44.4 %, 2.20, 26 / 46.8 %, 2.04, 27 |
| 1.0 | 4k | 72.19 | 107.46 | 105.85 | **1.488** [1.381-1.604] | **1.466** [1.360-1.580] | **0.985** [0.963-1.007] | 68.0 %, 3.36, 6 / 68.0 %, 3.36, 6 |
| 0.7 | short | 76.28 | 87.17 | 88.75 | **1.143** [1.130-1.155] | **1.163** [1.137-1.191] | **1.018** [0.985-1.052] | 44.4 %, 2.27, 47 / 51.8 %, 2.48, 143 |
| 0.7 | 1k | 73.86 | 77.25 | 78.87 | **1.046** [1.034-1.058] | **1.068** [1.032-1.104] | **1.021** [0.982-1.061] | 44.0 %, 2.17, 38 / 51.1 %, 2.10, 41 |
| 0.7 | 4k | 72.96 | 106.24 | 106.36 | **1.456** [1.397-1.517] | **1.458** [1.390-1.529] | **1.001** [0.942-1.064] | 73.2 %, 3.41, 11 / 73.2 %, 3.41, 11 |

| T | prompt | fix / off | pol / off | pol / fix | acc., tok/pass, plain tokens: fix / pol |
|---|---|---|---|---|---|
| 1.0 | short_b | **0.896** [0.886-0.906] | **0.981** [0.931-1.034] | **1.095** [1.048-1.143] | 27.5 %, 1.53, 40 / 34.1 %, 1.64, 91 |
| 1.0 | short_a | **1.172** [1.138-1.206] | **1.188** [1.166-1.209] | **1.014** [0.985-1.043] | 41.6 %, 2.31, 9 / 47.1 %, 2.19, 9 |
| 1.0 | short_c | **1.323** [1.283-1.364] | **1.431** [1.373-1.492] | **1.082** [1.060-1.104] | 49.4 %, 2.80, 4 / 63.7 %, 2.80, 4 |
| 1.0 | len1k | **1.049** [1.030-1.069] | **1.042** [1.003-1.083] | **0.993** [0.961-1.026] | 44.4 %, 2.20, 26 / 46.8 %, 2.04, 27 |
| 1.0 | len4k | **1.488** [1.381-1.604] | **1.466** [1.360-1.580] | **0.985** [0.963-1.007] | 68.0 %, 3.36, 6 / 68.0 %, 3.36, 6 |
| 0.7 | short_b | **0.982** [0.959-1.006] | **0.991** [0.970-1.014] | **1.009** [0.980-1.039] | 38.8 %, 1.71, 31 / 16.7 %, 1.33, 123 |
| 0.7 | short_a | **1.100** [1.077-1.124] | **1.095** [1.055-1.137] | **0.995** [0.946-1.048] | 37.6 %, 2.22, 14 / 39.4 %, 2.06, 18 |
| 0.7 | short_c | **1.434** [1.396-1.474] | **1.526** [1.494-1.559] | **1.064** [1.036-1.093] | 55.6 %, 3.12, 2 / 67.5 %, 3.12, 2 |
| 0.7 | len1k | **1.046** [1.034-1.058] | **1.068** [1.032-1.104] | **1.021** [0.982-1.061] | 44.0 %, 2.17, 38 / 51.1 %, 2.10, 41 |
| 0.7 | len4k | **1.456** [1.397-1.517] | **1.458** [1.390-1.529] | **1.001** [0.942-1.064] | 73.2 %, 3.41, 11 / 73.2 %, 3.41, 11 |

pre-registered rule (results doc section 3):
  1. every prompt pol / off >= 0.98: PASS (lowest 0.981 at T 1.0 short_b)
  2. T 1.0 short: pol / off 1.172 vs 0.98 x fix / off 1.079: ok
  2. T 1.0 1k: pol / off 1.042 vs 0.98 x fix / off 1.028: ok
  2. T 1.0 4k: pol / off 1.466 vs 0.98 x fix / off 1.459: ok
  2. T 0.7 short: pol / off 1.163 vs 0.98 x fix / off 1.120: ok
  2. T 0.7 1k: pol / off 1.068 vs 0.98 x fix / off 1.025: ok
  2. T 0.7 4k: pol / off 1.458 vs 0.98 x fix / off 1.427: ok
  2. every class keeps >= 98 % of fixed k's ratio: PASS
  3 (speed processes). ids identical across all arms: PASS

Caveat that applies to every row: all reps of a cell run the same seed (7), so the intervals measure timing noise only, not variation between
sampled continuations. The same prompt at seed 11 (W1 calibration) gave fixed-k5 `short_b` 0.938 at T 1.0 (replay) against 0.896 measured at seed 7.

## 6. Metal slot (endor `--exclusive`, fleetq #12360, 21:17:04-21:42:55 = 25 min 51 s; ad4 `7f51c299` = `d23914b2d` source, main6 `0894595e`)

Recipe default DSpark k3, cut 0.5, device sampler (Gumbel) for plain-temperature requests; 45 s GPU-idle gap before each timed process;
macmon at 250 ms (`raw/metal-run1/speed/speed-macmon.txt` has the per-process GPU power: off arms 22-25 W mean, fix 26-29 W, pol 27 W; no arm
sat above 65 W for more than 1.8 s, so none was power-throttled by its own history). Same arms and design as W2 (3 reps, Latin square per rep,
both temperatures inside each rep). Raw: `raw/metal-run1/`.

- **Seed equality:** 70 requests per arm fitted the 110 s seed budget (the harness is slower on Metal), so 35 seeds per temperature, not
  100: T 1.0 35 / 35 identical, T 0.6 35 / 35 identical (off vs the policy; the policy arm made 79 stops / 11 resumes, 124 device-draw verify
  passes). Also identical ids across all 9 speed processes of each temperature. The tiny `inkling2_dspark_sampled_adapt` suite on Metal (run
  after the slot, because the slot script's `timeout` does not exist on macOS, rc 127, fixed in `raw/metal/slot.sh`; fleetq #12367): 2 / 2.
- **Greedy md5 on Metal (fleetq #12369, `raw/metal-run1/md5-metal.txt`):** main6 and ad4, `--spec off` and the default (DSpark k3), 7 prompts x 64
  tokens: all four `ids[...]` hashes `617b2717999004d7100222ecf7b285f6`; the default arm 172 passes, 59.8 % acceptance on both binaries.
- **Cost table (15 % rule, one traced fixed-k3 and one `drafts` process, seed 11, T 1.0):** D 3.22 ms, P 13.62 ms, d = 0.237 (built-in 0.24);
  cycle m = 1 / 2 / 3: 20.06 / 22.54 / 24.78 ms, r = 0.236 / 0.418 / 0.583 (built-in 0.32 / 0.51 / 0.70, all > 15 % lower): measured table. Replay check on Metal:
  not exact for 1k (replay 46 / 90 / 50 passes / proposed / accepted vs measured 44 / 88 / 50; the other five runs exact): Metal's draft is not
  deterministic given the context. Defaults missed the bars (`short_b` 0.987), the grid ran and picked `g=0.85, margin=0, drop=0, probe=3,
  hold=8, hold_max=32` (predicted `short_b` 1.017 in-sample).
- **Policy arm:** `INTARSIA_SPEC_SAMPLED_ADAPT=d=0.237,r=0.236/0.418/0.583,g=0.85,margin=0.0,drop=0.0,probe=3,hold=8,hold_max=32`.

| T | class | off tok/s | fix tok/s | pol tok/s | fix / off | pol / off | pol / fix | acc., tok/pass, plain tokens: fix / pol |
|---|---|---:|---:|---:|---|---|---|---|
| 1.0 | short | 75.61 | 85.12 | 86.47 | **1.126** [1.069-1.185] | **1.144** [1.101-1.187] | **1.016** [0.997-1.035] | 52.7 %, 2.11, 69 / 57.0 %, 2.24, 146 |
| 1.0 | 1k | 73.09 | 81.18 | 82.07 | **1.111** [1.090-1.132] | **1.123** [1.083-1.164] | **1.011** [0.980-1.043] | 50.9 %, 2.17, 23 / 55.1 %, 2.10, 24 |
| 1.0 | 4k | 70.20 | 92.08 | 95.25 | **1.312** [1.266-1.360] | **1.358** [1.253-1.472] | **1.035** [0.987-1.084] | 72.6 %, 2.68, 17 / 85.1 %, 3.10, 34 |
| 0.7 | short | 75.36 | 83.71 | 85.38 | **1.110** [0.974-1.264] | **1.133** [1.111-1.155] | **1.021** [0.913-1.141] | 52.6 %, 2.13, 62 / 58.1 %, 2.34, 149 |
| 0.7 | 1k | 72.65 | 78.80 | 76.93 | **1.085** [1.076-1.094] | **1.059** [1.033-1.085] | **0.976** [0.955-0.998] | 46.8 %, 2.04, 11 / 53.6 %, 2.12, 57 |
| 0.7 | 4k | 70.90 | 101.05 | 95.17 | **1.425** [1.411-1.439] | **1.342** [1.323-1.361] | **0.942** [0.932-0.952] | 86.0 %, 3.06, 17 / 85.3 %, 3.06, 32 |

| T | prompt | fix / off | pol / off | pol / fix | acc., tok/pass, plain tokens: fix / pol |
|---|---|---|---|---|---|
| 1.0 | short_b | **0.855** [0.797-0.918] | **0.939** [0.891-0.989] | **1.098** [1.078-1.118] | 36.2 %, 1.46, 60 / 30.0 %, 1.38, 116 |
| 1.0 | short_a | **1.185** [1.116-1.260] | **1.101** [1.039-1.167] | **0.929** [0.919-0.939] | 45.7 %, 2.09, 6 / 44.0 %, 1.92, 27 |
| 1.0 | short_c | **1.532** [1.494-1.571] | **1.538** [1.494-1.583] | **1.004** [0.978-1.031] | 69.6 %, 2.82, 3 / 72.5 %, 2.76, 3 |
| 1.0 | len1k | **1.111** [1.090-1.132] | **1.123** [1.083-1.164] | **1.011** [0.980-1.043] | 50.9 %, 2.17, 23 / 55.1 %, 2.10, 24 |
| 1.0 | len4k | **1.312** [1.266-1.360] | **1.358** [1.253-1.472] | **1.035** [0.987-1.084] | 72.6 %, 2.68, 17 / 85.1 %, 3.10, 34 |
| 0.7 | short_b | **0.923** [0.771-1.105] | **0.944** [0.923-0.967] | **1.023** [0.874-1.197] | 53.1 %, 1.69, 44 / 58.8 %, 1.77, 104 |
| 0.7 | short_a | **1.034** [0.943-1.133] | **1.063** [1.035-1.092] | **1.029** [0.962-1.100] | 37.3 %, 1.91, 16 / 42.9 %, 2.00, 43 |
| 0.7 | short_c | **1.534** [1.381-1.704] | **1.543** [1.532-1.554] | **1.006** [0.906-1.117] | 71.3 %, 2.91, 2 / 71.1 %, 2.84, 2 |
| 0.7 | len1k | **1.085** [1.076-1.094] | **1.059** [1.033-1.085] | **0.976** [0.955-0.998] | 46.8 %, 2.04, 11 / 53.6 %, 2.12, 57 |
| 0.7 | len4k | **1.425** [1.411-1.439] | **1.342** [1.323-1.361] | **0.942** [0.932-0.952] | 86.0 %, 3.06, 17 / 85.3 %, 3.06, 32 |

pre-registered rule (results doc section 3):
  1. every prompt pol / off >= 0.98: FAIL (lowest 0.939 at T 1.0 short_b)
  2. T 1.0 short: pol / off 1.144 vs 0.98 x fix / off 1.103: ok
  2. T 1.0 1k: pol / off 1.123 vs 0.98 x fix / off 1.089: ok
  2. T 1.0 4k: pol / off 1.358 vs 0.98 x fix / off 1.286: ok
  2. T 0.7 short: pol / off 1.133 vs 0.98 x fix / off 1.087: ok
  2. T 0.7 1k: pol / off 1.059 vs 0.98 x fix / off 1.063: FAIL
  2. T 0.7 4k: pol / off 1.342 vs 0.98 x fix / off 1.397: FAIL
  2. every class keeps >= 98 % of fixed k's ratio: FAIL
  3 (speed processes). ids identical across all arms: PASS

## 7. Gate verdict (the four clauses, fixed in advance)

| clause | HIP (R9700) | Metal (endor) |
|---|---|---|
| 1. every prompt pol / off >= 0.98, `short_b` included, both T | **pass** by the point estimate: lowest 0.981 (`short_b`, T 1.0; CI 0.931-1.034, so not statistically separated from 0.98 or from 1.0), next 0.991; fixed k5 had 0.896 | **FAIL**: `short_b` 0.939 at T 1.0 (CI 0.891-0.989), 0.944 at T 0.7 (CI 0.923-0.967); fixed k3 had 0.855 / 0.923. `short_a` T 1.0 1.101, others >= 1.04 |
| 2. each class keeps >= 98 % of fixed-k's gain | **pass**, all six cells (closest: T 1.0 1k, pol 1.042 vs 0.98 x fix 1.028; T 1.0 4k 1.466 vs 1.459) | **FAIL**: T 0.7 1k 1.059 vs 1.063 (0.4 points short); T 0.7 4k 1.342 vs 1.397 (5.5 points short, pol / fix 0.942) |
| 3. seeded ids identical to `--spec off` | **pass**: 100 seeds x T 1.0 / 0.6 (24 tokens, k5), plus all W2 processes | **pass** with fewer seeds: 35 x T 1.0 / 0.6 (k3, time-boxed), plus all slot speed processes |
| 4. greedy ids identical to main | **pass**: 7 prompts x 64 tokens, `--spec off` and default, main6 vs ad4 | **pass**: same, Metal |

**Overall: FAIL (Metal clauses 1 and 2).** HIP meets all four on the point estimates; the HIP margin on `short_b` is 0.001 and its interval
reaches below 0.98, so "passes" means the fixed rule was met, not that every prompt is shown to be at least 0.98. Whether sampled drafting
goes default-on is the user's call; this PR changes no default (the policy acts only with `INTARSIA_SPEC_SAMPLED_DFLASH=1`).

### What the failing Metal cells show, and what would fix them

- `short_b` on Metal: the arm took 8 verify passes and 116 plain steps in 127 tokens (acceptance 30 %), yet ran 6 % slower than off. The excess
  over off is 80-150 ms of ~1.7 s per process. The speed arms were not traced, so its split is not measured; the likely source is the probes (3 draft calls
  of ~3.3 ms each per probe, plus the full-cap explore calls) and the 8 verifies. The offline replay predicted 1.017 for this string on seed 11; seed 7's story drafts accepted less (36 % under
  fixed k3 against 47 % at seed 11), so the grid's pick was fitted to a continuation that was kinder than the one measured. Fixed k3 on seed 7 lost
  14 % (0.855), the policy 6 %: the direction is right, the size is not enough.
- T 0.7 4k (pol / fix 0.942) and 1k (0.976): the policy took plain steps on requests that accept about 85 % (4k: 32 plain-step tokens vs 17 for fixed k3). Metal's
  grid pick (g 0.85, drop 0, probe 3) stops on short noisy dips; HIP's pick (g 0.95, drop 0.05, probe 1, hold 16 / 64), replayed on Metal's seed-11 trace
  with Metal's table, predicts `short_b` 0.988 pol / off and 1k 0.984 pol / fix on that one seed, i.e. no better in the replay, so the sim does not say that moving
  to HIP's parameters fixes Metal. What the data say: the stop rule needs more evidence before it stops a request that has been accepting
  well (a longer memory or a stop threshold on the low-acceptance run, not the rate alone) and a longer hold for a request that has stopped.
  These are untested here; testing them needs a new Metal slot (this lane's one slot is spent) and a calibration on more than one seed.
- Per-prompt gates on 128-token single-seed cells are noisy: fixed-k `short_b` on / off is 0.938 (seed 11, HIP replay) against 0.896 measured at seed 7, and 0.995 (seed 11, Metal replay) against 0.855 measured.
  A fair per-prompt gate needs several seeds per prompt.

## 8. The policy as built, knobs, revert

- Code: `crates/intarsia-core/src/spec/adapt.rs` (`AdaptK`, `AdaptParams`, `AdaptMode`), hooked in `spec.rs` and the keyed path of `engine.rs`; tests in
  `tests/inkling2_dspark_sampled_adapt.rs` (k = 0 sync through a stash overflow; seeded ids equal `--spec off` in table / aggressive / `drafts` modes) and 6 `spec::adapt` unit tests.
- Knobs: env `INTARSIA_SPEC_SAMPLED_ADAPT` (`0` / `off` fixed k; `1` / `on`; `key=value,...` over the built-in table: `d`, `r`, `g`, `n0`, `a0`, `margin`, `drop`,
  `probe`, `hold`, `hold_max`, `explore`; `drafts` diagnostic) and the recipe key `[tuning.<backend>] sampled_adapt` (same syntax; the variable wins); docs/tuning.md and docs/recipes.md rows.
- Recipe defaults in this PR: `[tuning.hip]` the W2 string above, `[tuning.metal]` the slot's string above (failed the gate; opt-in feature, so no default behaviour changes either way).
- Revert: `INTARSIA_SPEC_SAMPLED_ADAPT=0` (fixed k, PR #535's behaviour), or delete the recipe line (built-in table), or leave `INTARSIA_SPEC_SAMPLED_DFLASH` unset (plain sampling).
- Draft state at k = 0: a plain step keeps the DSpark draft in sync (section 1.3): no change to the draft state was needed.

## Parked 2026-11-26 20:40 (coordinator: Claude Desktop restart; superseded by sections 4-8, kept as the record of what was open then)

- **Head:** `ink/sampled-adapt` local commit after `d23914b2d` (this section and `raw/` scripts on top); not pushed, no PR yet.
- **Main merged:** yes, `origin/main` `aee06201c` (#535) merged in `56ec7c465` (one conflict in `docs/recipes.md`, resolved add-only).
- **Passed (cpu, endor):** core `spec::` unit tests 58/58 (incl. 6 `spec::adapt`) and `inkling2_dspark_sampled_adapt` 2/2 on the
  final policy (fleetq #12332, logs `.scratch/logs/cpu-2031/`): the k = 0 sync test (draft after 2 / 3 / 17 plain steps, stash
  overflow included, equals the draft over the same rows ingested by verify passes) and seeded ids == `--spec off` for the table,
  aggressive and `drafts` modes (aggressive scripted: 195 plain-step tokens vs fixed k's 13, so the policy both holds and resumes).
  Earlier full cpu suite on the pre-merge head (#12314, `.scratch/logs/cpu-2012/`): `inkling2_dspark_sampled` 2/2,
  `_gumbel` 1/1, `inkling2_dspark` 4/4, `_cut` 1/1, `verify_softcap` 1/1, `recipes_doc` 3/3, `docs_env_vars` 2/2, core lib 274/274.
- **Not verified:** no GPU window and no Metal slot ran (radiance never taken down by this lane). Nothing on the real model: no
  calibration, seeds, greedy md5, or speed. The full cpu suite has not rerun on the merged head (only the adapt suite and core `spec::`).
  `sim.py` is checked only on synthetic traces.
- **Builds:** tatooine #12317 (ad1 + main5, pre-merge, unused), #12321 (ad2, pre-merge, unused), #12328 (ad3 = a source tree changed
  mid-build: do not use; main6 = main `aee06201c`, hip, `~/intarsiaml-ink-sampled-adapt/.scratch/bin/intarsia-main6`); #12336 (ad4)
  released before it started. endor Metal: #12329 (ad3 at `6da9139a7`, main6 md5 `0894595e...`), #12335 (ad4 at `d23914b2d`,
  `.scratch/bin/intarsia-ad4`). Logs: `.scratch/logs/fleet-*.log`; tatooine `~/intarsiaml-ink-sampled-adapt/logs/build-*.log`.
- **Exact next step:** rebuild hip ad4 on tatooine (`fleetq run --host tatooine-r9700 --load cpu --expect 8 -- ssh tatooine
  'SKIP_MAIN=1 BR=ad4 bash ~/intarsiaml-ink-sampled-adapt/.scratch/tat/builds.sh'`), then W1 (`raw/tat/w1.sh` under
  `raw/tat/window.sh`, CAP 280, `--takes-down radiance --quiet gpu,cpu`), then `sim.py logs/w1/cal` (15 % table rule, replay check,
  grid if needed), then W2 (`raw/tat/w2.sh`, ADAPT from sim), then the Metal slot (`raw/metal/slot.sh`, `--exclusive --expect 30`).
- **Cleanup answer:** this lane no longer uses `.worktrees/ink-sampled` (its `.scratch/prompts/*.json` were copied here), the remote
  `ink/sampled-dspark` branch, or tatooine `~/intarsiaml-ink-sampled` / `intarsia-br5` / `intarsia-main5` / its docker volume. All can go.
