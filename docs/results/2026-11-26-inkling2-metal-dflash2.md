# Inkling-2-Mini + DFlash2 drafter on Metal (lane ink-metal-dflash2, 2026-11-26)

Status: **in progress** (speed table goes first here when measured). Branch `ink/metal-dflash2` from origin/main `3fb1b5f8d`. Opt-in only; no default changes.

## Brief (coordinator, 2026-11-26 23:36 EST; the user is asleep and said "keep working")

Goal: make the published DFlash2 drafter (`incoai/Inkling-2-Mini-DFlash2`) work as **opt-in** `--spec dflash --spec-draft <dir>` for
Inkling-2-Mini on **Metal**, then measure it on endor against Metal's current default speculative path. Ship a PR. **No default changes.**
Whether this ever becomes a default is the user's call in the morning.

What exists: HIP and cpu DFlash2 for Inkling (#541, `docs/results/2026-11-26-inkling2-dflash2.md`; taps are target visits, block rows
`rms(embed[id], embed_norm)`, the draft's logits use its own quantised copy of the unembed's real rows, vocab padded, `--spec-k`
defaults to block - 1; on HIP k4 beat DSpark k5: mean 1.112, short 1.19, 4k 0.991, 16k 1.10). The Metal refusal is in `model.rs`
("hip and cpu only"). Metal DFlash2 ops exist from lane dm (Qwen3.8-27B): `crates/intarsia-backend-metal/src/dflash.rs`,
`kernels/metal/dflash.metal` (`DynConv`), `crates/intarsia-backend-metal/tests/dflash2.rs`, `docs/results/2026-11-06-metal-dflash2.md`.
Device top-k is HIP only. Baseline: main's default Inkling spec on Metal (DSpark; k from the code and startup line). Metal slot harness
of #535: `docs/results/2026-11-26-inkling2-sampled-dspark/raw/metal/` (reuse, no new timing scripts).

Where: worktree `/opt/code/intarsiaml/.worktrees/ink-metal-dflash2`, scratch under `.scratch/`. #544 (int8 HIP, opt-in) is about to
merge; once MERGED merge `origin/main` into the branch (a merge, never a rebase). Drafter weights on tatooine
`/data/models/intarsia/Inkling-2-Mini-DFlash2` (945 MB): copy next to endor's Inkling-2-Mini via fleetq, check md5/size. No other new folders.

Steps: (1) exact on cpu first (Inkling DFlash2 cpu tests on endor, `--load cpu`). (2) Lift the Metal refusal only for what Metal can run
exactly; fill missing Metal pieces (draft head / unembed copy bitwise to the HIP lane's reference rules; anything else the draft needs);
Metal tests in the style of `crates/intarsia-backend-metal/tests/dflash2.rs`. (3) Metal exactness (Metal slot): greedy ids `--spec dflash`
== `--spec off` on the golden md5 prompts, short and 4k, 64 and 256 tokens; plus the Metal Inkling tiny suite. (4) Speed: one-process k
sweep (k 3/4/5/7), then ABBA 4 + 4 at the best k against main's Metal default, the same six cells as #541 (short, 1k, prose, chat, 4k,
16k), plus prefill 4k and 16k. endor is a 140 W MacBook Pro whose GPU speed depends on recent power draw: 45 s idle gaps between runs,
arms grouped as `slot.sh` does with macmon. (5) The pre-registered rule (below) is written before any speed number and not relaxed.
(6) PR to `main` via `fleetq run --host endor --load cpu` push; do not merge; body: what changed, speed table first, bitwise results,
what is not verified, how to turn it on; ends with the Claude Code line. (7) This doc, a LOG.md row, a unanimis capture (`unim_capture_new`,
no `project`/`scope`, `intarsiaml` in `source`).

Budget and fleet: every Metal run takes an exclusive endor slot (`fleetq run --host endor --exclusive --expect N --label ...`), each slot
<= 20 min, <= 60 min in total, none after 07:00 EST; another lane needs one <= 15-min exclusive Metal slot tonight (fleetq orders them);
queue one slot at a time and do not hold the box idle. Other fleet commands through fleetq with an honest `--expect` (builds / tests
`--load cpu`); no tatooine GPU windows (radiance is not ours tonight). Never `fleetq release` a running job (kill its process tree).
No single tool call or wait over 4 min.

Stop rules: if a Metal piece cannot be made bitwise with a small change, stop and report; do not ship a non-exact path. If the k sweep
shows no k faster than the Metal default on any cell, skip the ABBA, record NO-GO, and ship the opt-in PR only if the code is clean and
tested, else leave the branch unpushed and report.

Final report (<= 15 lines): speed table first (per-cell ratio vs the Metal default with 95 % CI, mean, verdict, best k, prefill);
bitwise results; PR number and head sha; Metal minutes used; what is not verified; fleetq ids.

## Pre-registered rule (written 2026-11-26 23:37 EST, before any speed number; not to be relaxed)

Cells: short, 1k, prose, chat, 4k, 16k (the #541 definitions, `raw/speed.json`), plus prefill 4k / 16k. Ratio = DFlash2 tok/s / Metal
default tok/s per (prompt, rep), geometric mean per cell, 95 % CI = t-interval over the rep pairs.

- **GO** (for a later default-on proposal, the user's call): mean of the six cells >= 1.03, no cell with a 95 % CI upper bound < 0.99,
  and greedy ids identical to `--spec off` (and across arms).
- **OPT-IN**: faster on some cells, not GO.
- **NO-GO**: faster on none.
- Whatever the verdict, `--spec dflash` stays opt-in.
- The k sweep (k 3 / 4 / 5 / 7, one process each) only picks the ABBA's k; no k faster than the default on any cell = skip the ABBA, NO-GO.

## Log

- 2026-11-27 00:05 EST, **k sweep done (slot 2, #12478, one process per arm, 45 s idle before each, not paired), before any ABBA number**.
  Ratio DFlash2 k / DSpark k3 (the Metal default) per cell, single process each (no CI): mean of the six cells k3 1.070, k4 1.071,
  **k5 1.076**, k7 1.029 (k7 loses 1k 0.967, 16k 0.920). Every k is faster than the default on some cells, so the stop rule (no k faster
  anywhere) does not fire. The three lowest k are within 0.006 of each other, which is inside this sweep's noise. **ABBA k = k5, the k with
  the highest six-cell mean in the sweep (decided here, before the ABBA).** k4 gets a smaller ABBA afterwards only if the Metal budget
  allows it; it is reported as a secondary result and does not change the verdict.
